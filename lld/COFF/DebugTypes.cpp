//===- DebugTypes.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DebugTypes.h"
#include "Chunks.h"
#include "Driver.h"
#include "InputFiles.h"
#include "TypeMerger.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecordHelpers.h"
#include "llvm/DebugInfo/CodeView/TypeStreamMerger.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiHashing.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace lld;
using namespace lld::coff;
namespace {
class TypeServerIpiSource;

// The TypeServerSource class represents a PDB type server, a file referenced by
// OBJ files compiled with MSVC /Zi. A single PDB can be shared by several OBJ
// files, therefore there must be only once instance per OBJ lot. The file path
// is discovered from the dependent OBJ's debug type stream. The
// TypeServerSource object is then queued and loaded by the COFF Driver. The
// debug type stream for such PDB files will be merged first in the final PDB,
// before any dependent OBJ.
class TypeServerSource : public TpiSource {
public:
  explicit TypeServerSource(PDBInputFile *f)
      : TpiSource(PDB, nullptr), pdbInputFile(f) {
    if (f->loadErr && *f->loadErr)
      return;
    pdb::PDBFile &file = f->session->getPDBFile();
    auto expectedInfo = file.getPDBInfoStream();
    if (!expectedInfo)
      return;
    auto it = mappings.emplace(expectedInfo->getGuid(), this);
    assert(it.second);
    (void)it;
    indexMap.isTypeServerMap = true;

    // If we are using ghashes, create a secondary source for IPI. This assigns
    // another tpiSrcIdx which creates another ghash index space.
    if (config->debugGHashes)
      ipiSrc = make<TypeServerIpiSource>();
  }

  void loadGHashes() override;

  void remapTpiWithGHashes(GHashState *g) override;

  Error mergeDebugT(TypeMerger *m) override;

  bool isDependency() const override { return true; }

  PDBInputFile *pdbInputFile = nullptr;

  // Source of IPI information, if ghashes are in use.
  TypeServerIpiSource *ipiSrc = nullptr;

  static std::map<codeview::GUID, TypeServerSource *> mappings;
};

// Companion to TypeServerSource. Contains the ghashes for the IPI stream of the
// type server PDB. Actual IPI processing depends on the TPI stream, so this is
// done as part of the main TypeServerSource ghash loading and type merging.
class TypeServerIpiSource : public TpiSource {
public:
  explicit TypeServerIpiSource() : TpiSource(PDB, nullptr) {}

  friend class TypeServerSource;

  // All of the TpiSource methods are no-ops. The parent TypeServerSource
  // handles both TPI and IPI.
  Error mergeDebugT(TypeMerger *m) override { return Error::success(); }
  void loadGHashes() override {}
  void remapTpiWithGHashes(GHashState *g) override {}
  bool isDependency() const override { return true; }
};

// This class represents the debug type stream of an OBJ file that depends on a
// PDB type server (see TypeServerSource).
class UseTypeServerSource : public TpiSource {
  Expected<TypeServerSource *> getTypeServerSource();

public:
  UseTypeServerSource(ObjFile *f, TypeServer2Record ts)
      : TpiSource(UsingPDB, f), typeServerDependency(ts) {}

  Error mergeDebugT(TypeMerger *m) override;

  void loadGHashes() override;

  void remapTpiWithGHashes(GHashState *g) override;

  // Information about the PDB type server dependency, that needs to be loaded
  // in before merging this OBJ.
  TypeServer2Record typeServerDependency;
};

// This class represents the debug type stream of a Microsoft precompiled
// headers OBJ (PCH OBJ). This OBJ kind needs to be merged first in the output
// PDB, before any other OBJs that depend on this. Note that only MSVC generate
// such files, clang does not.
class PrecompSource : public TpiSource {
public:
  PrecompSource(ObjFile *f) : TpiSource(PCH, f) {
    if (!f->pchSignature || !*f->pchSignature)
      fatal(toString(f) +
            " claims to be a PCH object, but does not have a valid signature");
    auto it = mappings.emplace(*f->pchSignature, this);
    if (!it.second)
      fatal("a PCH object with the same signature has already been provided (" +
            toString(it.first->second->file) + " and " + toString(file) + ")");
    indexMap.isPrecompiledTypeMap = true;
  }

  void loadGHashes() override;

  bool isDependency() const override { return true; }

  static std::map<uint32_t, PrecompSource *> mappings;
};

// This class represents the debug type stream of an OBJ file that depends on a
// Microsoft precompiled headers OBJ (see PrecompSource).
class UsePrecompSource : public TpiSource {
public:
  UsePrecompSource(ObjFile *f, PrecompRecord precomp)
      : TpiSource(UsingPCH, f), precompDependency(precomp) {}

  Error mergeDebugT(TypeMerger *m) override;

  void loadGHashes() override;

  void remapTpiWithGHashes(GHashState *g) override;

  // Information about the Precomp OBJ dependency, that needs to be loaded in
  // before merging this OBJ.
  PrecompRecord precompDependency;
};
} // namespace

std::vector<TpiSource *> TpiSource::instances;

TpiSource::TpiSource(TpiKind k, ObjFile *f)
    : kind(k), tpiSrcIdx(instances.size()), file(f) {
  instances.push_back(this);
}

// Vtable key method.
TpiSource::~TpiSource() = default;

TpiSource *lld::coff::makeTpiSource(ObjFile *file) {
  return make<TpiSource>(TpiSource::Regular, file);
}

TpiSource *lld::coff::makeTypeServerSource(PDBInputFile *pdbInputFile) {
  return make<TypeServerSource>(pdbInputFile);
}

TpiSource *lld::coff::makeUseTypeServerSource(ObjFile *file,
                                              TypeServer2Record ts) {
  return make<UseTypeServerSource>(file, ts);
}

TpiSource *lld::coff::makePrecompSource(ObjFile *file) {
  return make<PrecompSource>(file);
}

TpiSource *lld::coff::makeUsePrecompSource(ObjFile *file,
                                           PrecompRecord precomp) {
  return make<UsePrecompSource>(file, precomp);
}

std::map<codeview::GUID, TypeServerSource *> TypeServerSource::mappings;

std::map<uint32_t, PrecompSource *> PrecompSource::mappings;

// A COFF .debug$H section is currently a clang extension.  This function checks
// if a .debug$H section is in a format that we expect / understand, so that we
// can ignore any sections which are coincidentally also named .debug$H but do
// not contain a format we recognize.
static bool canUseDebugH(ArrayRef<uint8_t> debugH) {
  if (debugH.size() < sizeof(object::debug_h_header))
    return false;
  auto *header =
      reinterpret_cast<const object::debug_h_header *>(debugH.data());
  debugH = debugH.drop_front(sizeof(object::debug_h_header));
  return header->Magic == COFF::DEBUG_HASHES_SECTION_MAGIC &&
         header->Version == 0 &&
         header->HashAlgorithm == uint16_t(GlobalTypeHashAlg::SHA1_8) &&
         (debugH.size() % 8 == 0);
}

static Optional<ArrayRef<uint8_t>> getDebugH(ObjFile *file) {
  SectionChunk *sec =
      SectionChunk::findByName(file->getDebugChunks(), ".debug$H");
  if (!sec)
    return llvm::None;
  ArrayRef<uint8_t> contents = sec->getContents();
  if (!canUseDebugH(contents))
    return None;
  return contents;
}

static ArrayRef<GloballyHashedType>
getHashesFromDebugH(ArrayRef<uint8_t> debugH) {
  assert(canUseDebugH(debugH));
  debugH = debugH.drop_front(sizeof(object::debug_h_header));
  uint32_t count = debugH.size() / sizeof(GloballyHashedType);
  return {reinterpret_cast<const GloballyHashedType *>(debugH.data()), count};
}

// Copies ghashes from a vector into an array. These are long lived, so it's
// worth the time to copy these into an appropriately sized vector to reduce
// memory usage.
void TpiSource::assignGHashesFromVector(
    std::vector<GloballyHashedType> &&hashVec) {
  GloballyHashedType *hashes = new GloballyHashedType[hashVec.size()];
  memcpy(hashes, hashVec.data(), hashVec.size() * sizeof(GloballyHashedType));
  ghashes = makeArrayRef(hashes, hashVec.size());
  ownedGHashes = true;
}

// Wrapper on forEachCodeViewRecord with less error handling.
static void forEachTypeChecked(ArrayRef<uint8_t> types,
                               function_ref<void(const CVType &)> fn) {
  checkError(
      forEachCodeViewRecord<CVType>(types, [fn](const CVType &ty) -> Error {
        fn(ty);
        return Error::success();
      }));
}

void TpiSource::loadGHashes() {
  if (Optional<ArrayRef<uint8_t>> debugH = getDebugH(file)) {
    ghashes = getHashesFromDebugH(*debugH);
    ownedGHashes = false;
  } else {
    CVTypeArray types;
    BinaryStreamReader reader(file->debugTypes, support::little);
    cantFail(reader.readArray(types, reader.getLength()));
    assignGHashesFromVector(GloballyHashedType::hashTypes(types));
  }

  // TODO: Either eliminate the need for this info, or make this part of the
  // ghash format so that we don't have to iterate all of .debug$T again.
  fillIsItemIndexFromDebugT();
}

// Walk over file->debugTypes and fill in the isItemIndex bit vector.
void TpiSource::fillIsItemIndexFromDebugT() {
  uint32_t index = 0;
  isItemIndex.resize(ghashes.size());
  forEachTypeChecked(file->debugTypes, [&](const CVType &ty) {
    if (isIdRecord(ty.kind()))
      isItemIndex.set(index);
    ++index;
  });
}

bool TpiSource::remapTypeIndex(TypeIndex &ti, TiRefKind refKind,
                               CVIndexMap &indexMap) {
  // This can be an item index or a type index. Choose the appropriate map.
  MutableArrayRef<TypeIndex> tpiOrIpiMap = indexMap.tpiMap;
  if (refKind == TiRefKind::IndexRef && indexMap.isTypeServerMap)
    tpiOrIpiMap = indexMap.ipiMap;

  if (ti.isSimple())
    return true;
  assert(!shouldOmitFromPdb(ti.toArrayIndex()) && "cannot remap omitted index");
  if (ti.toArrayIndex() >= tpiOrIpiMap.size())
    return false;
  ti = tpiOrIpiMap[ti.toArrayIndex()];
  return true;
}

void TpiSource::remapRecord(MutableArrayRef<uint8_t> rec,
                            CVIndexMap &indexMap,
                            ArrayRef<TiReference> typeRefs) {
  MutableArrayRef<uint8_t> contents = rec.drop_front(sizeof(RecordPrefix));
  for (const TiReference &ref : typeRefs) {
    unsigned byteSize = ref.Count * sizeof(TypeIndex);
    if (contents.size() < ref.Offset + byteSize)
      fatal("symbol record too short");

    MutableArrayRef<TypeIndex> indices(
        reinterpret_cast<TypeIndex *>(contents.data() + ref.Offset), ref.Count);
    for (TypeIndex &ti : indices) {
      if (!remapTypeIndex(ti, ref.Kind, indexMap)) {
        if (config->verbose) {
          uint16_t kind =
              reinterpret_cast<const RecordPrefix *>(rec.data())->RecordKind;
          StringRef fname = file ? file->getName() : "<unknown PDB>";
          log("failed to remap type index in record of kind 0x" +
              utohexstr(kind) + " in " + fname + " with bad " +
              (ref.Kind == TiRefKind::IndexRef ? "item" : "type") +
              " index 0x" + utohexstr(ti.getIndex()));
        }
        ti = TypeIndex(SimpleTypeKind::NotTranslated);
        continue;
      }
    }
  }
}

void TpiSource::remapTypesInTypeRecord(MutableArrayRef<uint8_t> rec,
                                       CVIndexMap &indexMap) {
  // TODO: Handle errors similar to symbols.
  SmallVector<TiReference, 32> typeRefs;
  discoverTypeIndices(CVType(rec), typeRefs);
  remapRecord(rec, indexMap, typeRefs);
}

bool TpiSource::remapTypesInSymbolRecord(MutableArrayRef<uint8_t> rec,
                                         CVIndexMap &indexMap) {
  // Discover type index references in the record. Skip it if we don't
  // know where they are.
  SmallVector<TiReference, 32> typeRefs;
  if (!discoverTypeIndicesInSymbol(rec, typeRefs))
    return false;
  remapRecord(rec, indexMap, typeRefs);
  return true;
}

void TpiSource::mergeTypeRecord(CVType ty, CVIndexMap *indexMap) {
  // Decide if the merged type goes into TPI or IPI.
  bool isItem = isIdRecord(ty.kind());
  MergedInfo &merged = isItem ? mergedIpi : mergedTpi;

  // Copy the type into our mutable buffer.
  assert(ty.length() <= codeview::MaxRecordLength);
  size_t offset = merged.recs.size();
  size_t newSize = alignTo(ty.length(), 4);
  merged.recs.resize(offset + newSize);
  auto newRec = makeMutableArrayRef(&merged.recs[offset], newSize);
  memcpy(newRec.data(), ty.data().data(), newSize);

  // Fix up the record prefix and padding bytes if it required resizing.
  if (newSize != ty.length()) {
    reinterpret_cast<RecordPrefix *>(newRec.data())->RecordLen = newSize - 2;
    for (size_t i = ty.length(); i < newSize; ++i)
      newRec[i] = LF_PAD0 + (newSize - i);
  }

  // Remap the type indices in the new record.
  remapTypesInTypeRecord(newRec, *indexMap);
  uint32_t pdbHash = check(pdb::hashTypeRecord(CVType(newRec)));
  merged.recSizes.push_back(static_cast<uint16_t>(newSize));
  merged.recHashes.push_back(pdbHash);
}

void TpiSource::mergeUniqueTypeRecords(const CVTypeArray &types,
                                       CVIndexMap *indexMap) {
  // FIXME: Pre-sort desired types.
  if (indexMap->isTypeServerMap)
    assert(std::is_sorted(uniqueTypes.begin(), uniqueTypes.end()));
  else
    llvm::sort(uniqueTypes);

  // Accumulate all the unique types into one buffer in mergedTypes.
  TypeIndex index(TypeIndex::FirstNonSimpleIndex);
  auto nextUniqueIndex = uniqueTypes.begin();
  assert(mergedTpi.recs.empty());
  assert(mergedIpi.recs.empty());
  for (CVType ty : types) {
    if (nextUniqueIndex != uniqueTypes.end() && *nextUniqueIndex == index) {
      mergeTypeRecord(ty, indexMap);
      ++nextUniqueIndex;
    }
    if (ty.kind() == LF_FUNC_ID || ty.kind() == LF_MFUNC_ID) {
      bool success = ty.length() >= 12;
      TypeIndex funcId = index;
      TypeIndex funcType;
      if (success) {
        funcType = *reinterpret_cast<const TypeIndex *>(&ty.data()[8]);
        success &= remapTypeIndex(funcId, TiRefKind::IndexRef, *indexMap);
        success &= remapTypeIndex(funcType, TiRefKind::TypeRef, *indexMap);
      }
      if (success) {
        indexMap->funcIdToType.insert({funcId, funcType});
      } else {
        StringRef fname = file ? file->getName() : "<unknown PDB>";
        warn("corrupt LF_[M]FUNC_ID record 0x" + utohexstr(index.getIndex()) +
             " in " + fname);
      }
    }
    ++index;
  }
  assert(nextUniqueIndex == uniqueTypes.end() &&
         "failed to merge all desired records");
  assert(uniqueTypes.size() ==
             mergedTpi.recSizes.size() + mergedIpi.recSizes.size() &&
         "missing desired record");
}

void TpiSource::remapTpiWithGHashes(GHashState *g) {
  assert(config->debugGHashes && "ghashes must be enabled");
  fillMapFromGHashes(g, indexMap.tpiMap);
  CVTypeArray types;
  BinaryStreamReader reader(file->debugTypes, support::little);
  cantFail(reader.readArray(types, reader.getLength()));
  mergeUniqueTypeRecords(types, &indexMap);
  // TODO: Free all unneeded ghash resources now that we have a full index map.
}

// Merge .debug$T for a generic object file.
Error TpiSource::mergeDebugT(TypeMerger *m) {
  assert(!config->debugGHashes &&
         "use remapTpiWithGHashes when ghash is enabled");

  CVTypeArray types;
  BinaryStreamReader reader(file->debugTypes, support::little);
  cantFail(reader.readArray(types, reader.getLength()));

  if (auto err = mergeTypeAndIdRecords(
          m->idTable, m->typeTable, indexMap.tpiMap, types, file->pchSignature))
    fatal("codeview::mergeTypeAndIdRecords failed: " +
          toString(std::move(err)));

  if (config->showSummary) {
    // Count how many times we saw each type record in our input. This
    // calculation requires a second pass over the type records to classify each
    // record as a type or index. This is slow, but this code executes when
    // collecting statistics.
    m->tpiCounts.resize(m->getTypeTable().size());
    m->ipiCounts.resize(m->getIDTable().size());
    uint32_t srcIdx = 0;
    for (CVType &ty : types) {
      TypeIndex dstIdx = indexMap.tpiMap[srcIdx++];
      // Type merging may fail, so a complex source type may become the simple
      // NotTranslated type, which cannot be used as an array index.
      if (dstIdx.isSimple())
        continue;
      SmallVectorImpl<uint32_t> &counts =
          isIdRecord(ty.kind()) ? m->ipiCounts : m->tpiCounts;
      ++counts[dstIdx.toArrayIndex()];
    }
  }

  return Error::success();
}

// PDBs do not actually store global hashes, so when merging a type server
// PDB we have to synthesize global hashes.  To do this, we first synthesize
// global hashes for the TPI stream, since it is independent, then we
// synthesize hashes for the IPI stream, using the hashes for the TPI stream
// as inputs.
void TypeServerSource::loadGHashes() {
  // Don't hash twice.
  if (!ghashes.empty())
    return;
  pdb::PDBFile &pdbFile = pdbInputFile->session->getPDBFile();

  // Hash TPI stream.
  Expected<pdb::TpiStream &> expectedTpi = pdbFile.getPDBTpiStream();
  if (auto e = expectedTpi.takeError())
    fatal("Type server does not have TPI stream: " + toString(std::move(e)));
  assignGHashesFromVector(
      GloballyHashedType::hashTypes(expectedTpi->typeArray()));
  isItemIndex.resize(ghashes.size());

  // Hash IPI stream, which depends on TPI ghashes.
  if (!pdbFile.hasPDBIpiStream())
    return;
  Expected<pdb::TpiStream &> expectedIpi = pdbFile.getPDBIpiStream();
  if (auto e = expectedIpi.takeError())
    fatal("error retreiving IPI stream: " + toString(std::move(e)));
  ipiSrc->assignGHashesFromVector(
      GloballyHashedType::hashIds(expectedIpi->typeArray(), ghashes));

  // The IPI stream isItemIndex bitvector should be all ones.
  ipiSrc->isItemIndex.resize(ipiSrc->ghashes.size());
  ipiSrc->isItemIndex.set(0, ipiSrc->ghashes.size());
}

// Merge types from a type server PDB.
void TypeServerSource::remapTpiWithGHashes(GHashState *g) {
  assert(config->debugGHashes && "ghashes must be enabled");

  // IPI merging depends on TPI, so do TPI first, then do IPI.  No need to
  // propagate errors, those should've been handled during ghash loading.
  pdb::PDBFile &pdbFile = pdbInputFile->session->getPDBFile();
  pdb::TpiStream &tpi = check(pdbFile.getPDBTpiStream());
  fillMapFromGHashes(g, indexMap.tpiMap);
  mergeUniqueTypeRecords(tpi.typeArray(), &indexMap);
  if (pdbFile.hasPDBIpiStream()) {
    pdb::TpiStream &ipi = check(pdbFile.getPDBIpiStream());
    ipiSrc->fillMapFromGHashes(g, indexMap.ipiMap);
    ipiSrc->mergeUniqueTypeRecords(ipi.typeArray(), &indexMap);
  }
}

// Merge types from a type server PDB.
Error TypeServerSource::mergeDebugT(TypeMerger *m) {
  assert(!config->debugGHashes &&
         "use remapTpiWithGHashes when ghash is enabled");

  pdb::PDBFile &pdbFile = pdbInputFile->session->getPDBFile();
  Expected<pdb::TpiStream &> expectedTpi = pdbFile.getPDBTpiStream();
  if (auto e = expectedTpi.takeError())
    fatal("Type server does not have TPI stream: " + toString(std::move(e)));
  pdb::TpiStream *maybeIpi = nullptr;
  if (pdbFile.hasPDBIpiStream()) {
    Expected<pdb::TpiStream &> expectedIpi = pdbFile.getPDBIpiStream();
    if (auto e = expectedIpi.takeError())
      fatal("Error getting type server IPI stream: " +
            toString(std::move(e)));
    maybeIpi = &*expectedIpi;
  }

  // Merge TPI first, because the IPI stream will reference type indices.
  if (auto err = mergeTypeRecords(m->typeTable, indexMap.tpiMap,
                                  expectedTpi->typeArray()))
    fatal("codeview::mergeTypeRecords failed: " + toString(std::move(err)));

  // Merge IPI.
  if (maybeIpi) {
    if (auto err = mergeIdRecords(m->idTable, indexMap.tpiMap,
                                  indexMap.ipiMap, maybeIpi->typeArray()))
      fatal("codeview::mergeIdRecords failed: " + toString(std::move(err)));
  }

  if (config->showSummary) {
    // Count how many times we saw each type record in our input. If a
    // destination type index is present in the source to destination type index
    // map, that means we saw it once in the input. Add it to our histogram.
    m->tpiCounts.resize(m->getTypeTable().size());
    m->ipiCounts.resize(m->getIDTable().size());
    for (TypeIndex ti : indexMap.tpiMap)
      if (!ti.isSimple())
        ++m->tpiCounts[ti.toArrayIndex()];
    for (TypeIndex ti : indexMap.ipiMap)
      if (!ti.isSimple())
        ++m->ipiCounts[ti.toArrayIndex()];
  }

  return Error::success();
}

Expected<TypeServerSource *> UseTypeServerSource::getTypeServerSource() {
  const codeview::GUID &tsId = typeServerDependency.getGuid();
  StringRef tsPath = typeServerDependency.getName();

  TypeServerSource *tsSrc;
  auto it = TypeServerSource::mappings.find(tsId);
  if (it != TypeServerSource::mappings.end()) {
    tsSrc = it->second;
  } else {
    // The file failed to load, lookup by name
    PDBInputFile *pdb = PDBInputFile::findFromRecordPath(tsPath, file);
    if (!pdb)
      return createFileError(tsPath, errorCodeToError(std::error_code(
                                         ENOENT, std::generic_category())));
    // If an error occurred during loading, throw it now
    if (pdb->loadErr && *pdb->loadErr)
      return createFileError(tsPath, std::move(*pdb->loadErr));

    tsSrc = (TypeServerSource *)pdb->debugTypesObj;
  }
  return tsSrc;
}

void UseTypeServerSource::loadGHashes() {
  // No need to load ghashes from /Zi objects.
}

void UseTypeServerSource::remapTpiWithGHashes(GHashState *g) {
  // No remapping to do with /Zi objects. Simply use the index map from the type
  // server.
  // FIXME: Avoid copying this vector.
  TypeServerSource *tsSrc = check(getTypeServerSource());
  indexMap = tsSrc->indexMap;
}

Error UseTypeServerSource::mergeDebugT(TypeMerger *m) {
  Expected<TypeServerSource *> tsSrc = getTypeServerSource();
  if (!tsSrc)
    return tsSrc.takeError();

  pdb::PDBFile &pdbSession = (*tsSrc)->pdbInputFile->session->getPDBFile();
  auto expectedInfo = pdbSession.getPDBInfoStream();
  if (!expectedInfo)
    return expectedInfo.takeError();

  // Just because a file with a matching name was found and it was an actual
  // PDB file doesn't mean it matches.  For it to match the InfoStream's GUID
  // must match the GUID specified in the TypeServer2 record.
  if (expectedInfo->getGuid() != typeServerDependency.getGuid())
    return createFileError(
        typeServerDependency.getName(),
        make_error<pdb::PDBError>(pdb::pdb_error_code::signature_out_of_date));

  // FIXME: Copying the entire index map here could be avoided if we switch to
  // ArrayRef in CVIndexMap. This duplication is temporary to allow us to
  // refactor so we can reduce memory usage by extending the lifetime of the
  // index map to defer copying symbols.
  indexMap = (*tsSrc)->indexMap;
  return Error::success();
}

static bool equalsPath(StringRef path1, StringRef path2) {
#if defined(_WIN32)
  return path1.equals_lower(path2);
#else
  return path1.equals(path2);
#endif
}

// Find by name an OBJ provided on the command line
static PrecompSource *findObjByName(StringRef fileNameOnly) {
  SmallString<128> currentPath;
  for (auto kv : PrecompSource::mappings) {
    StringRef currentFileName = sys::path::filename(kv.second->file->getName(),
                                                    sys::path::Style::windows);

    // Compare based solely on the file name (link.exe behavior)
    if (equalsPath(currentFileName, fileNameOnly))
      return kv.second;
  }
  return nullptr;
}

static PrecompSource *findPrecompSource(ObjFile *file, PrecompRecord &pr) {
  // Cross-compile warning: given that Clang doesn't generate LF_PRECOMP
  // records, we assume the OBJ comes from a Windows build of cl.exe. Thusly,
  // the paths embedded in the OBJs are in the Windows format.
  SmallString<128> prFileName =
      sys::path::filename(pr.getPrecompFilePath(), sys::path::Style::windows);

  auto it = PrecompSource::mappings.find(pr.getSignature());
  if (it != PrecompSource::mappings.end()) {
    return it->second;
  }
  // Lookup by name
  return findObjByName(prFileName);
}

static Expected<CVIndexMap *> findPrecompMap(ObjFile *file, PrecompRecord &pr) {
  PrecompSource *precomp = findPrecompSource(file, pr);

  if (!precomp)
    return createFileError(
        pr.getPrecompFilePath(),
        make_error<pdb::PDBError>(pdb::pdb_error_code::no_matching_pch));

  if (pr.getSignature() != file->pchSignature)
    return createFileError(
        toString(file),
        make_error<pdb::PDBError>(pdb::pdb_error_code::no_matching_pch));

  if (pr.getSignature() != *precomp->file->pchSignature)
    return createFileError(
        toString(precomp->file),
        make_error<pdb::PDBError>(pdb::pdb_error_code::no_matching_pch));

  return &precomp->indexMap;
}

void PrecompSource::loadGHashes() {
  if (getDebugH(file)) {
    warn("ignoring .debug$H section; pch with ghash is not implemented");
  }

  uint32_t ghashIdx = 0;
  std::vector<GloballyHashedType> hashVec;
  forEachTypeChecked(file->debugTypes, [&](const CVType &ty) {
    // Remember the index of the LF_ENDPRECOMP record so it can be excluded from
    // the PDB. There must be an entry in the list of ghashes so that the type
    // indexes of the following records in the /Yc PCH object line up.
    if (ty.kind() == LF_ENDPRECOMP)
      endPrecompGHashIdx = ghashIdx;

    hashVec.push_back(GloballyHashedType::hashType(ty, hashVec, hashVec));
    isItemIndex.push_back(isIdRecord(ty.kind()));
    ++ghashIdx;
  });
  assignGHashesFromVector(std::move(hashVec));
}

void UsePrecompSource::loadGHashes() {
  PrecompSource *pchSrc = findPrecompSource(file, precompDependency);
  if (!pchSrc)
    return; // FIXME: Test error handling.

  // To compute ghashes of a /Yu object file, we need to build on the the
  // ghashes of the /Yc PCH object. After we are done hashing, discard the
  // ghashes from the PCH source so we don't unnecessarily try to deduplicate
  // them.
  std::vector<GloballyHashedType> hashVec =
      pchSrc->ghashes.take_front(precompDependency.getTypesCount());
  forEachTypeChecked(file->debugTypes, [&](const CVType &ty) {
    hashVec.push_back(GloballyHashedType::hashType(ty, hashVec, hashVec));
    isItemIndex.push_back(isIdRecord(ty.kind()));
  });
  hashVec.erase(hashVec.begin(),
                hashVec.begin() + precompDependency.getTypesCount());
  assignGHashesFromVector(std::move(hashVec));
}

/// Merges a precompiled headers TPI map into the current TPI map. The
/// precompiled headers object will also be loaded and remapped in the
/// process.
static Error prependPrecompIndexMap(ObjFile *file, CVIndexMap *indexMap,
                                    PrecompRecord &precomp) {
  auto e = findPrecompMap(file, precomp);
  if (!e)
    return e.takeError();

  CVIndexMap *precompIndexMap = *e;
  assert(precompIndexMap->isPrecompiledTypeMap);

  if (precompIndexMap->tpiMap.empty())
    return Error::success();

  assert(precomp.getStartTypeIndex() == TypeIndex::FirstNonSimpleIndex);
  assert(precomp.getTypesCount() <= precompIndexMap->tpiMap.size());
  // Use the previously remapped index map from the precompiled headers.
  indexMap->tpiMap.append(precompIndexMap->tpiMap.begin(),
                          precompIndexMap->tpiMap.begin() +
                              precomp.getTypesCount());
  return Error::success();
}

void UsePrecompSource::remapTpiWithGHashes(GHashState *g) {
  // This object was compiled with /Yu, so process the corresponding
  // precompiled headers object (/Yc) first. Some type indices in the current
  // object are referencing data in the precompiled headers object, so we need
  // both to be loaded.
  checkError(prependPrecompIndexMap(file, &indexMap, precompDependency));
  TpiSource::remapTpiWithGHashes(g);
}

Error UsePrecompSource::mergeDebugT(TypeMerger *m) {
  // This object was compiled with /Yu, so process the corresponding
  // precompiled headers object (/Yc) first. Some type indices in the current
  // object are referencing data in the precompiled headers object, so we need
  // both to be loaded.
  if (Error e = prependPrecompIndexMap(file, &indexMap, precompDependency))
    return e;

  return TpiSource::mergeDebugT(m);
}

uint32_t TpiSource::countTypeServerPDBs() {
  return TypeServerSource::mappings.size();
}

uint32_t TpiSource::countPrecompObjs() {
  return PrecompSource::mappings.size();
}

void TpiSource::clear() {
  // Clean up any owned ghash allocations.
  for (TpiSource *src : TpiSource::instances) {
    if (src->ownedGHashes)
      delete[] src->ghashes.data();
  }
  TpiSource::instances.clear();
  TypeServerSource::mappings.clear();
  PrecompSource::mappings.clear();
}

namespace {
/// A concurrent hash table for global type hashing. It is based on this paper:
/// Concurrent Hash Tables: Fast and General(?)!
/// https://dl.acm.org/doi/10.1145/3309206
///
/// This hash table is meant to be used in two phases:
/// 1. concurrent insertion
/// 2. concurrent lookup
/// It does not support deletion or rehashing. It uses linear probing.
///
/// The paper describes storing a key-value pair in two machine words.
/// Generally, the values stored in this map are type indices, and we can use
/// those values to recover the ghash key from a side table. This allows us to
/// shrink the table entries further at the cost of some loads, and sidesteps
/// the need for a 128 bit atomic compare-and-swap operation.
///
/// The Cell type must support the following API:
///   // Get the ghash key for this cell.
///   GloballyHashedType getGHash();
///   // Must return true if members are zero, since table is initialized with
///   // memset.
///   bool isEmpty();
///   // Used to prioritize conflicting cells.
///   bool operator<(l, r);
template <typename Cell> struct GHashTable {
  GHashState *ghashState = nullptr;
  Cell *table = nullptr;
  size_t tableSize = 0;

  GHashTable() = default;
  ~GHashTable() { delete[] table; }

  void init(GHashState *gs, size_t newTableSize) {
    ghashState = gs;
    table = new Cell[newTableSize];
    memset(table, 0, newTableSize * sizeof(Cell));
    tableSize = newTableSize;
  }

  void insert(Cell newCell);

  Optional<Cell> lookup(GloballyHashedType ghash);
};

/// A ghash table cell for deduplicating types from TpiSources.
class GHashInCell {
  uint64_t data = 0;

public:
  GHashInCell() = default;

  // Construct data most to least significant so that sorting works well:
  // - isItem
  // - tpiSrcIdx
  // - ghashIdx
  GHashInCell(bool isItem, uint32_t tpiSrcIdx, uint32_t ghashIdx)
      : data((uint64_t(isItem) << 63U) | (uint64_t(tpiSrcIdx + 1) << 32ULL) |
             ghashIdx) {
    assert(tpiSrcIdx == getTpiSrcIdx() && "round trip failure");
    assert(ghashIdx == getGHashIdx() && "round trip failure");
  }

  explicit GHashInCell(uint64_t data) : data(data) {}

  // The empty cell is all zeros.
  bool isEmpty() const { return data == 0ULL; }

  uint32_t getTpiSrcIdx() const {
    return ((uint32_t)(data >> 32U) & 0x7FFFFFFF) - 1;
  }
  bool isItem() const { return data & (1ULL << 63U); }
  uint32_t getGHashIdx() const { return (uint32_t)data; }

  GloballyHashedType getGHash(GHashState * /*g*/) const {
    // TODO: Store ghashes in GHashState? Save pointer deref?
    return TpiSource::instances[getTpiSrcIdx()]->ghashes[getGHashIdx()];
  }

  friend inline bool operator<(const GHashInCell &l, const GHashInCell &r) {
    return l.data < r.data;
  }
};

/// A ghash table cell mapping to final PDB type indices. Needed during type
/// merging to remap source type indices lazily.
struct alignas(uint32_t) TypeIndexCell {
  TypeIndex ti;

  TypeIndexCell() = default;
  explicit TypeIndexCell(TypeIndex ti) : ti(ti) {}

  bool isEmpty() const { return ti.getIndex() == 0; }

  GloballyHashedType getGHash(GHashState *g) const;

  friend inline bool operator<(const TypeIndexCell &l, const TypeIndexCell &r) {
    return l.ti < r.ti;
  }
};
} // namespace

namespace lld {
namespace coff {
struct GHashState {
  // Initial ghash table for type duplication.
  GHashTable<GHashInCell> inTable;

  /// This is the global list of ghashes used by this cell during hash table
  /// insertion.
  std::vector<GloballyHashedType> itemGHashes;
  std::vector<GloballyHashedType> typeGHashes;

  /// A map from ghash to final type index.
  GHashTable<TypeIndexCell> finalGHashMap;
};
}
}

GloballyHashedType TypeIndexCell::getGHash(GHashState *g) const {
  uint32_t idx = ti.toArrayIndex();
  return ti.isDecoratedItemId() ? g->itemGHashes[idx] : g->typeGHashes[idx];
}

template <typename Cell> inline void GHashTable<Cell>::insert(Cell newCell) {
  assert(!newCell.isEmpty() && "cannot insert empty cell value");
  GloballyHashedType ghash = newCell.getGHash(ghashState);

  // FIXME: The low bytes of SHA1 have low entropy for short records, which
  // type records are. Swap the byte order for better entropy. A better ghash
  // won't need this.
  size_t startIdx =
      ByteSwap_64(*reinterpret_cast<uint64_t *>(&ghash)) % tableSize;

  // Do a linear probe starting at startIdx.
  size_t idx = startIdx;
  while (true) {
    // Run a compare and swap loop. There are four cases:
    // - cell is empty: CAS into place and return
    // - cell has matching key, earlier priority: do nothing, return
    // - cell has matching key, later priority: CAS into place and return
    // - cell has non-matching key: hash collision, probe next cell
    auto *cellPtr = reinterpret_cast<std::atomic<Cell> *>(&table[idx]);
    Cell oldCell(cellPtr->load());
    while (oldCell.isEmpty() || oldCell.getGHash(ghashState) == ghash) {
      // Check if there is an existing ghash entry with a higher priority
      // (earlier ordering). If so, this is a duplicate, we are done.
      if (!oldCell.isEmpty() && oldCell < newCell)
        return;
      // Either the cell is empty, or our value is higher priority. Try to
      // compare and swap. If it succeeds, we are done.
      if (cellPtr->compare_exchange_weak(oldCell, newCell))
        return;
      // If the CAS failed, check this cell again.
    }

    // Advance the probe. Wrap around to the beginning if we run off the end.
    ++idx;
    idx = idx == tableSize ? 0 : idx;
    if (idx == startIdx) {
      // If this becomes an issue, we could mark failure and rehash from the
      // beginning with a bigger table. There is no difference between rehashing
      // internally and starting over.
      report_fatal_error("ghash table is full");
    }
  }
  llvm_unreachable("left infloop");
}

template <typename Cell>
inline Optional<Cell> GHashTable<Cell>::lookup(GloballyHashedType ghash) {
  size_t startIdx =
      ByteSwap_64(*reinterpret_cast<uint64_t *>(&ghash)) % tableSize;
  size_t idx = startIdx;
  while (true) {
    // We should be able to use no atomics or relaxed atomics at this point.
    Cell cell = table[idx];
    if (cell.isEmpty())
      return None; // Empty
    else if (ghash == cell.getGHash(ghashState))
      return cell;

    // Advance the probe. Wrap around to the beginning if we run off the end.
    ++idx;
    idx = idx == tableSize ? 0 : idx;
    if (idx == startIdx)
      report_fatal_error("ghash table is full");
  }
  llvm_unreachable("left infloop");
}

TypeMerger::TypeMerger(llvm::BumpPtrAllocator &alloc)
    : typeTable(alloc), idTable(alloc) {}

TypeMerger::~TypeMerger() = default;

void TypeMerger::mergeTypesWithGHash() {
  ghashState.reset(new GHashState());

  // Load ghashes. Do type servers and PCH objects first.
  parallelForEach(TpiSource::instances, [&](TpiSource *source) {
    if (source->isDependency())
      source->loadGHashes();
  });
  parallelForEach(TpiSource::instances, [&](TpiSource *source) {
    if (!source->isDependency())
      source->loadGHashes();
  });

  // Estimate the size of hash table needed to deduplicate ghashes. This *must*
  // be larger than the number of unique types, or hash table insertion may not
  // be able to find a vacant slot. Summing the input types guarantees this, but
  // it is a gross overestimate. Less memory could be used with a concurrent
  // rehashing implementation.
  size_t tableSize = 0;
  for (TpiSource *source : TpiSource::instances)
    tableSize += source->ghashes.size();

  ghashState->inTable.init(ghashState.get(), tableSize);

  // Insert ghashes in parallel. It is important to detect duplicates with the
  // least amount of work possible, so that the least amount of time can be
  // spent on them.
  parallelForEachN(0, TpiSource::instances.size(), [&](size_t tpiSrcIdx) {
    TpiSource *source = TpiSource::instances[tpiSrcIdx];
    for (uint32_t i = 0, e = source->ghashes.size(); i < e; i++) {
      if (source->shouldOmitFromPdb(i))
        continue;
      bool isItem = source->isItemIndex.test(i);
      ghashState->inTable.insert(GHashInCell(isItem, tpiSrcIdx, i));
    }
  });

  // Collect all non-empty cells and sort them. This will implicitly assign
  // destination type indices, and arrange the input types into buckets formed
  // by types from the same TpiSource.
  // TODO: Avoid vector copy? Free inTable? Add `takeTable` method or something.
  std::vector<GHashInCell> entries;
  for (const GHashInCell &cell : makeArrayRef(ghashState->inTable.table, tableSize)) {
    if (!cell.isEmpty())
      entries.push_back(cell);
  }
  parallelSort(entries, std::less<GHashInCell>());
  log(formatv("ghash table load factor: {0:p} (size {1} / capacity {2})\n",
              double(entries.size()) / tableSize, entries.size(), tableSize));

  // Find out how many type and item indices there are.
  auto mid =
      std::lower_bound(entries.begin(), entries.end(), GHashInCell(true, 0, 0));
  assert((mid == entries.end() || mid->isItem()) &&
         (mid == entries.begin() || !std::prev(mid)->isItem()) &&
         "midpoint is not midpoint");
  uint32_t numTypes = std::distance(entries.begin(), mid);
  uint32_t numItems = std::distance(mid, entries.end());
  log("Tpi record count: " + Twine(numTypes));
  log("Ipi record count: " + Twine(numItems));
  ghashState->typeGHashes.reserve(numTypes);
  ghashState->itemGHashes.reserve(numItems);

  // Put a list of all unique type indices on each tpi source. Type merging
  // will skip indices not on this list.
  // TODO: Parallelize using same technique as ICF sharding.
  for (const GHashInCell &cell : entries) {
    uint32_t tpiSrcIdx = cell.getTpiSrcIdx();
    TpiSource *source = TpiSource::instances[tpiSrcIdx];
    source->uniqueTypes.push_back(
        TypeIndex::fromArrayIndex(cell.getGHashIdx()));
    auto &dstGHashVec =
        cell.isItem() ? ghashState->itemGHashes : ghashState->typeGHashes;
    dstGHashVec.push_back(cell.getGHash(ghashState.get()));
  }

  // Build a map from ghash to final PDB type index. There are two type index
  // spaces: types and items.
  ghashState->finalGHashMap.init(ghashState.get(), entries.size() * 2);
  parallelForEachN(0, entries.size(), [&](size_t entryIdx) {
    bool isItem = entryIdx >= numTypes;
    TypeIndex ti = TypeIndex::fromDecoratedArrayIndex(
        isItem, isItem ? entryIdx - numTypes : entryIdx);
    ghashState->finalGHashMap.insert(TypeIndexCell(ti));
  });

  // In parallel, remap all types.
  // FIXME: Simplify how we handle "dependency" type sources.
  for_each(TpiSource::instances, [&](TpiSource *source) {
    if (source->isDependency())
      source->remapTpiWithGHashes(ghashState.get());
  });
  parallelForEach(TpiSource::instances, [&](TpiSource *source) {
    if (!source->isDependency())
      source->remapTpiWithGHashes(ghashState.get());
  });
  ghashState.reset();
  // TODO: Free calculated ghashes.
}

static TypeIndex lookupPdbIndexFromGHash(GHashState *g,
                                         GloballyHashedType ghash) {
  auto maybeCell = g->finalGHashMap.lookup(ghash);
  assert(maybeCell && "unmapped ghash");
  return maybeCell->ti.removeDecoration();
}

// Fill in a TPI or IPI index map using ghashes. For each source type, use its
// ghash to lookup its final type index in the PDB, and store that in the map.
void TpiSource::fillMapFromGHashes(GHashState *g,
                                   SmallVectorImpl<TypeIndex> &indexMap) {
  indexMap.resize(ghashes.size());
  for (size_t i = 0, e = ghashes.size(); i < e; ++i) {
    if (shouldOmitFromPdb(i))
      indexMap[i] = TypeIndex(SimpleTypeKind::NotTranslated);
    else
      indexMap[i] = lookupPdbIndexFromGHash(g, ghashes[i]);
  }
}
