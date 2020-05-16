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
    tsIndexMap.isTypeServerMap = true;
  }

  void loadGHashes() override;

  Expected<CVIndexMap *> mergeDebugT(TypeMerger *m,
                                           CVIndexMap *indexMap) override;
  bool isDependency() const override { return true; }

  PDBInputFile *pdbInputFile = nullptr;

  CVIndexMap tsIndexMap;

  static std::map<codeview::GUID, TypeServerSource *> mappings;
};

// This class represents the debug type stream of an OBJ file that depends on a
// PDB type server (see TypeServerSource).
class UseTypeServerSource : public TpiSource {
public:
  UseTypeServerSource(ObjFile *f, TypeServer2Record ts)
      : TpiSource(UsingPDB, f), typeServerDependency(ts) {}

  Expected<CVIndexMap *> mergeDebugT(TypeMerger *m,
                                           CVIndexMap *indexMap) override;

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
    precompIndexMap.isPrecompiledTypeMap = true;
  }

  Expected<CVIndexMap *> mergeDebugT(TypeMerger *m,
                                           CVIndexMap *indexMap) override;
  bool isDependency() const override { return true; }

  CVIndexMap precompIndexMap;

  static std::map<uint32_t, PrecompSource *> mappings;
};

// This class represents the debug type stream of an OBJ file that depends on a
// Microsoft precompiled headers OBJ (see PrecompSource).
class UsePrecompSource : public TpiSource {
public:
  UsePrecompSource(ObjFile *f, PrecompRecord precomp)
      : TpiSource(UsingPCH, f), precompDependency(precomp) {}

  void loadGHashes() override;

  Expected<CVIndexMap *> mergeDebugT(TypeMerger *m,
                                           CVIndexMap *indexMap) override;

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

static ArrayRef<uint64_t> getHashesFromDebugH(ArrayRef<uint8_t> debugH) {
  assert(canUseDebugH(debugH));
  debugH = debugH.drop_front(sizeof(object::debug_h_header));
  uint32_t count = debugH.size() / sizeof(uint64_t);
  // FIXME: Endian concerns.
  return {reinterpret_cast<const uint64_t *>(debugH.data()), count};
}

static void hashCVTypeArray(TpiSource *src, const CVTypeArray &types) {
  // BumpPtrAllocator is not thread-safe, so use `new`.
  std::vector<GloballyHashedType> hashVec =
      GloballyHashedType::hashTypes(types);
  uint64_t *hashes = new uint64_t[hashVec.size()];
  // FIXME: Endian concerns.
  memcpy(hashes, hashVec.data(), hashVec.size() * sizeof(uint64_t));
  src->tpiGHashes = makeArrayRef(hashes, hashVec.size());
  src->ownedGHashes = true;
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
    tpiGHashes = getHashesFromDebugH(*debugH);
    ownedGHashes = false;
  } else {
    CVTypeArray types;
    BinaryStreamReader reader(file->debugTypes, support::little);
    cantFail(reader.readArray(types, reader.getLength()));
    hashCVTypeArray(this, types);
  }

  // Check which type records are item records or type records.
  // TODO: Get help from compiler to make this faster.
  uint32_t index = 0;
  isItemIndex.resize(tpiGHashes.size());
  forEachTypeChecked(file->debugTypes, [&](const CVType &ty) {
    if (isIdRecord(ty.kind()))
      isItemIndex.set(index);
    ++index;
  });
}

bool TpiSource::remapTypeIndex(TypeMerger *m, TypeIndex &ti,
                               MutableArrayRef<TypeIndex> typeIndexMap) {
  if (ti.isSimple())
    return true;
  if (ti.toArrayIndex() >= typeIndexMap.size())
    return false;
  TypeIndex &dstTi = typeIndexMap[ti.toArrayIndex()];
  if (config->debugGHashes && dstTi.isNoneType()) {
    // Lazily populate the object to PDB index map with a ghash lookup.
    // FIXME: This will race when multiple sources use the same map (type
    // servers).
    // FIXME: Check ipiGHashes? Probably only for type servers.
    auto *ghashPtr = &tpiGHashes[ti.toArrayIndex()];
    uint64_t ghash = *reinterpret_cast<const uint64_t *>(ghashPtr);
    auto maybeCell = m->finalGHashMap.lookup(ghash);
    assert(maybeCell && "unmapped ghash");
    dstTi = maybeCell->ti.removeDecoration();
  }
  ti = dstTi;
  return true;
}

void TpiSource::remapRecord(TypeMerger *m, MutableArrayRef<uint8_t> rec,
                            CVIndexMap &indexMap,
                            ArrayRef<TiReference> typeRefs) {
  MutableArrayRef<uint8_t> contents = rec.drop_front(sizeof(RecordPrefix));
  for (const TiReference &ref : typeRefs) {
    unsigned byteSize = ref.Count * sizeof(TypeIndex);
    if (contents.size() < ref.Offset + byteSize)
      fatal("symbol record too short");

    // This can be an item index or a type index. Choose the appropriate map.
    MutableArrayRef<TypeIndex> typeOrItemMap = indexMap.tpiMap;
    bool isItemIndex = ref.Kind == TiRefKind::IndexRef;
    if (isItemIndex && indexMap.isTypeServerMap)
      typeOrItemMap = indexMap.ipiMap;

    MutableArrayRef<TypeIndex> indices(
        reinterpret_cast<TypeIndex *>(contents.data() + ref.Offset), ref.Count);
    for (TypeIndex &ti : indices) {
      if (!remapTypeIndex(m, ti, typeOrItemMap)) {
        uint16_t kind =
            reinterpret_cast<const RecordPrefix *>(rec.data())->RecordKind;
        log("failed to remap type index in record of kind 0x" +
            utohexstr(kind) + " in " + file->getName() + " with bad " +
            (isItemIndex ? "item" : "type") + " index 0x" +
            utohexstr(ti.getIndex()));
        ti = TypeIndex(SimpleTypeKind::NotTranslated);
        continue;
      }
    }
  }
}

void TypeMerger::remapTypesInTypeRecord(MutableArrayRef<uint8_t> rec,
                                        ObjFile *file, CVIndexMap &indexMap) {
  // TODO: Handle errors similar to symbols.
  SmallVector<TiReference, 32> typeRefs;
  discoverTypeIndices(CVType(rec), typeRefs);
  file->debugTypesObj->remapRecord(this, rec, indexMap, typeRefs);
}

bool TypeMerger::remapTypesInSymbolRecord(MutableArrayRef<uint8_t> rec,
                                          ObjFile *file, CVIndexMap &indexMap) {
  // Discover type index references in the record. Skip it if we don't
  // know where they are.
  SmallVector<TiReference, 32> typeRefs;
  if (!discoverTypeIndicesInSymbol(rec, typeRefs))
    return false;
  file->debugTypesObj->remapRecord(this, rec, indexMap, typeRefs);
  return true;
}

// Merge .debug$T for a generic object file.
Expected<CVIndexMap *> TpiSource::mergeDebugT(TypeMerger *m,
                                              CVIndexMap *indexMap) {
  CVTypeArray types;
  BinaryStreamReader reader(file->debugTypes, support::little);
  cantFail(reader.readArray(types, reader.getLength()));

  if (config->debugGHashes) {
    // Zero initialize the type index map. It will be filled in lazily.
    // FIXME: Will this race for PDBs?
    if (indexMap->tpiMap.empty())
      indexMap->tpiMap.resize(tpiGHashes.size());
    if (indexMap->ipiMap.empty())
      indexMap->ipiMap.resize(ipiGHashes.size());

    // Accumulate all the unique types into one buffer in mergedTypes.
    TypeIndex index(TypeIndex::FirstNonSimpleIndex);
    llvm::sort(uniqueTypes);
    auto nextUniqueIndex = uniqueTypes.begin();
    assert(mergedTpi.recs.empty());
    assert(mergedIpi.recs.empty());
    forEachTypeChecked(file->debugTypes, [&](const CVType &ty) {
      if (nextUniqueIndex != uniqueTypes.end() && *nextUniqueIndex == index) {
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
          reinterpret_cast<RecordPrefix *>(newRec.data())->RecordLen =
              newSize - 2;
          for (size_t i = ty.length(); i < newSize; ++i)
            newRec[i] = LF_PAD0 + (newSize - i);
        }

        // Remap the type indices in the new record.
        m->remapTypesInTypeRecord(newRec, file, *indexMap);
        uint32_t pdbHash = check(pdb::hashTypeRecord(CVType(newRec)));
        merged.recSizes.push_back(static_cast<uint16_t>(newSize));
        merged.recHashes.push_back(pdbHash);
        ++nextUniqueIndex;
      }
      ++index;
    });
    assert(nextUniqueIndex == uniqueTypes.end() &&
           "failed to merge all desired records");
  } else {
    if (auto err =
            mergeTypeAndIdRecords(m->idTable, m->typeTable, indexMap->tpiMap,
                                  types, file->pchSignature))
      fatal("codeview::mergeTypeAndIdRecords failed: " +
            toString(std::move(err)));
  }

  if (config->showSummary) {
    // Count how many times we saw each type record in our input. This
    // calculation requires a second pass over the type records to classify each
    // record as a type or index. This is slow, but this code executes when
    // collecting statistics.
    m->tpiCounts.resize(m->getTypeTable().size());
    m->ipiCounts.resize(m->getIDTable().size());
    uint32_t srcIdx = 0;
    for (CVType &ty : types) {
      TypeIndex dstIdx = indexMap->tpiMap[srcIdx++];
      // Type merging may fail, so a complex source type may become the simple
      // NotTranslated type, which cannot be used as an array index.
      if (dstIdx.isSimple())
        continue;
      SmallVectorImpl<uint32_t> &counts =
          isIdRecord(ty.kind()) ? m->ipiCounts : m->tpiCounts;
      ++counts[dstIdx.toArrayIndex()];
    }
  }

  return indexMap;
}

// PDBs do not actually store global hashes, so when merging a type server
// PDB we have to synthesize global hashes.  To do this, we first synthesize
// global hashes for the TPI stream, since it is independent, then we
// synthesize hashes for the IPI stream, using the hashes for the TPI stream
// as inputs.
void TypeServerSource::loadGHashes() {
  pdb::PDBFile &pdbFile = pdbInputFile->session->getPDBFile();
  Expected<pdb::TpiStream &> expectedTpi = pdbFile.getPDBTpiStream();
  if (auto e = expectedTpi.takeError())
    fatal("Type server does not have TPI stream: " + toString(std::move(e)));
  hashCVTypeArray(this, expectedTpi->typeArray());
  isItemIndex.resize(tpiGHashes.size());

  // FIXME: hash IPI.
}

// Merge types from a type server PDB.
Expected<CVIndexMap *> TypeServerSource::mergeDebugT(TypeMerger *m,
                                                           CVIndexMap *) {
  pdb::PDBFile &pdbFile = pdbInputFile->session->getPDBFile();
  Expected<pdb::TpiStream &> expectedTpi = pdbFile.getPDBTpiStream();
  if (auto e = expectedTpi.takeError())
    fatal("Type server does not have TPI stream: " + toString(std::move(e)));
  pdb::TpiStream *maybeIpi = nullptr;
  if (pdbFile.hasPDBIpiStream()) {
    Expected<pdb::TpiStream &> expectedIpi = pdbFile.getPDBIpiStream();
    if (auto e = expectedIpi.takeError())
      fatal("Error getting type server IPI stream: " + toString(std::move(e)));
    maybeIpi = &*expectedIpi;
  }

  if (config->debugGHashes) {
    llvm_unreachable("NYI");
  } else {
    // Merge TPI first, because the IPI stream will reference type indices.
    if (auto err = mergeTypeRecords(m->typeTable, tsIndexMap.tpiMap,
                                    expectedTpi->typeArray()))
      fatal("codeview::mergeTypeRecords failed: " + toString(std::move(err)));

    // Merge IPI.
    if (maybeIpi) {
      if (auto err = mergeIdRecords(m->idTable, tsIndexMap.tpiMap,
                                    tsIndexMap.ipiMap, maybeIpi->typeArray()))
        fatal("codeview::mergeIdRecords failed: " + toString(std::move(err)));
    }
  }

  if (config->showSummary) {
    // Count how many times we saw each type record in our input. If a
    // destination type index is present in the source to destination type index
    // map, that means we saw it once in the input. Add it to our histogram.
    m->tpiCounts.resize(m->getTypeTable().size());
    m->ipiCounts.resize(m->getIDTable().size());
    for (TypeIndex ti : tsIndexMap.tpiMap)
      if (!ti.isSimple())
        ++m->tpiCounts[ti.toArrayIndex()];
    for (TypeIndex ti : tsIndexMap.ipiMap)
      if (!ti.isSimple())
        ++m->ipiCounts[ti.toArrayIndex()];
  }

  return &tsIndexMap;
}

Expected<CVIndexMap *>
UseTypeServerSource::mergeDebugT(TypeMerger *m, CVIndexMap *indexMap) {
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

  pdb::PDBFile &pdbSession = tsSrc->pdbInputFile->session->getPDBFile();
  auto expectedInfo = pdbSession.getPDBInfoStream();
  if (!expectedInfo)
    return &tsSrc->tsIndexMap;

  // Just because a file with a matching name was found and it was an actual
  // PDB file doesn't mean it matches.  For it to match the InfoStream's GUID
  // must match the GUID specified in the TypeServer2 record.
  if (expectedInfo->getGuid() != typeServerDependency.getGuid())
    return createFileError(
        tsPath,
        make_error<pdb::PDBError>(pdb::pdb_error_code::signature_out_of_date));

  return &tsSrc->tsIndexMap;
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

Expected<CVIndexMap *> findPrecompMap(ObjFile *file, PrecompRecord &pr) {
  // Cross-compile warning: given that Clang doesn't generate LF_PRECOMP
  // records, we assume the OBJ comes from a Windows build of cl.exe. Thusly,
  // the paths embedded in the OBJs are in the Windows format.
  SmallString<128> prFileName =
      sys::path::filename(pr.getPrecompFilePath(), sys::path::Style::windows);

  PrecompSource *precomp;
  auto it = PrecompSource::mappings.find(pr.getSignature());
  if (it != PrecompSource::mappings.end()) {
    precomp = it->second;
  } else {
    // Lookup by name
    precomp = findObjByName(prFileName);
  }

  if (!precomp)
    return createFileError(
        prFileName,
        make_error<pdb::PDBError>(pdb::pdb_error_code::no_matching_pch));

  if (pr.getSignature() != file->pchSignature)
    return createFileError(
        toString(file),
        make_error<pdb::PDBError>(pdb::pdb_error_code::no_matching_pch));

  if (pr.getSignature() != *precomp->file->pchSignature)
    return createFileError(
        toString(precomp->file),
        make_error<pdb::PDBError>(pdb::pdb_error_code::no_matching_pch));

  return &precomp->precompIndexMap;
}

/// Merges a precompiled headers TPI map into the current TPI map. The
/// precompiled headers object will also be loaded and remapped in the
/// process.
static Expected<CVIndexMap *>
mergeInPrecompHeaderObj(ObjFile *file, CVIndexMap *indexMap,
                        PrecompRecord &precomp) {
  auto e = findPrecompMap(file, precomp);
  if (!e)
    return e.takeError();

  CVIndexMap *precompIndexMap = *e;
  assert(precompIndexMap->isPrecompiledTypeMap);

  if (precompIndexMap->tpiMap.empty())
    return precompIndexMap;

  assert(precomp.getStartTypeIndex() == TypeIndex::FirstNonSimpleIndex);
  assert(precomp.getTypesCount() <= precompIndexMap->tpiMap.size());
  // Use the previously remapped index map from the precompiled headers.
  indexMap->tpiMap.append(precompIndexMap->tpiMap.begin(),
                          precompIndexMap->tpiMap.begin() +
                              precomp.getTypesCount());
  return indexMap;
}

void UsePrecompSource::loadGHashes() {
  report_fatal_error("NYI, can base impl be used directly?");
}

Expected<CVIndexMap *>
UsePrecompSource::mergeDebugT(TypeMerger *m, CVIndexMap *indexMap) {
  // This object was compiled with /Yu, so process the corresponding
  // precompiled headers object (/Yc) first. Some type indices in the current
  // object are referencing data in the precompiled headers object, so we need
  // both to be loaded.
  auto e = mergeInPrecompHeaderObj(file, indexMap, precompDependency);
  if (!e)
    return e.takeError();

  return TpiSource::mergeDebugT(m, indexMap);
}

Expected<CVIndexMap *> PrecompSource::mergeDebugT(TypeMerger *m,
                                                        CVIndexMap *) {
  // Note that we're not using the provided CVIndexMap. Instead, we use our
  // local one. Precompiled headers objects need to save the index map for
  // further reference by other objects which use the precompiled headers.
  return TpiSource::mergeDebugT(m, &precompIndexMap);
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
      delete[] src->tpiGHashes.data();
  }
  TpiSource::instances.clear();
  TypeServerSource::mappings.clear();
  PrecompSource::mappings.clear();
}

std::vector<uint64_t> TypeIndexCell::typeGHashes;
std::vector<uint64_t> TypeIndexCell::itemGHashes;

namespace {
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

  uint64_t getGHash() const {
    // FIXME: GloballyHashedType should expose this.
    return *reinterpret_cast<const uint64_t *>(
        &TpiSource::instances[getTpiSrcIdx()]->tpiGHashes[getGHashIdx()]);
  }

  friend inline bool operator<(const GHashInCell &l, const GHashInCell &r) {
    return l.data < r.data;
  }
};
} // namespace

template <typename Cell> inline void GHashTable<Cell>::insert(Cell newCell) {
  assert(!newCell.isEmpty() && "cannot insert empty cell value");
  uint64_t ghash = newCell.getGHash();

  // FIXME: The low bytes of SHA1 have low entropy for short records, which
  // type records are. Swap the byte order for better entropy. A better ghash
  // won't need this.
  size_t startIdx = ByteSwap_64(ghash) % tableSize;

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
    while (oldCell.isEmpty() || oldCell.getGHash() == ghash) {
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

void TypeMerger::identifyUniqueTypeIndices() {
  parallelForEach(TpiSource::instances,
                  [&](TpiSource *source) { source->loadGHashes(); });

  // Estimate the size of hash table needed to deduplicate ghashes. This *must*
  // be larger than the number of unique types, or hash table insertion may not
  // be able to find a vacant slot. Summing the input types guarantees this, but
  // it is a gross overestimate. Less memory could be used with a concurrent
  // rehashing implementation.
  size_t tableSize = 0;
  for (TpiSource *source : TpiSource::instances)
    tableSize += source->tpiGHashes.size();

  GHashTable<GHashInCell> inTable;
  inTable.init(tableSize);

  // Insert ghashes in parallel. It is important to detect duplicates with the
  // least amount of work possible, so that the least amount of time can be
  // spent on them.
  parallelForEachN(0, TpiSource::instances.size(), [&](size_t tpiSrcIdx) {
    TpiSource *source = TpiSource::instances[tpiSrcIdx];
    uint32_t ghashSize = source->tpiGHashes.size();
    for (uint32_t ghashIdx = 0; ghashIdx < ghashSize; ghashIdx++) {
      bool isItem = source->isItemIndex.test(ghashIdx);
      inTable.insert(GHashInCell(isItem, tpiSrcIdx, ghashIdx));
    }
  });

  // Collect all non-empty cells and sort them. This will implicitly assign
  // destination type indices, and arrange the input types into buckets formed
  // by types from the same TpiSource.
  std::vector<GHashInCell> entries;
  for (const GHashInCell &cell : makeArrayRef(inTable.table, tableSize)) {
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
  TypeIndexCell::typeGHashes.reserve(numTypes);
  TypeIndexCell::itemGHashes.reserve(numItems);

  // Put a list of all unique type indices on each tpi source. Type merging
  // will skip indices not on this list.
  // TODO: Parallelize using same technique as ICF sharding.
  for (const GHashInCell &cell : entries) {
    uint32_t tpiSrcIdx = cell.getTpiSrcIdx();
    TpiSource *source = TpiSource::instances[tpiSrcIdx];
    // FIXME: Carry item index-ness for type server PDBs.
    source->uniqueTypes.push_back(
        TypeIndex::fromArrayIndex(cell.getGHashIdx()));
    auto &dstGHashVec =
        cell.isItem() ? TypeIndexCell::itemGHashes : TypeIndexCell::typeGHashes;
    dstGHashVec.push_back(cell.getGHash());
  }

  // Build a map from ghash to final PDB type index. There are two type index
  // spaces: types and items.
  finalGHashMap.init(entries.size() * 2);
  parallelForEachN(0, entries.size(), [&](size_t entryIdx) {
    bool isItem = entryIdx >= numTypes;
    TypeIndex ti = TypeIndex::fromDecoratedArrayIndex(
        isItem, isItem ? entryIdx - numTypes : entryIdx);
    finalGHashMap.insert(TypeIndexCell(ti));
  });
}
