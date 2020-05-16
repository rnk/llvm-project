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
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecordHelpers.h"
#include "llvm/DebugInfo/CodeView/TypeStreamMerger.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include <atomic>

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

  Expected<const CVIndexMap *> mergeDebugT(TypeMerger *m,
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

  Expected<const CVIndexMap *> mergeDebugT(TypeMerger *m,
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

  Expected<const CVIndexMap *> mergeDebugT(TypeMerger *m,
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

  Expected<const CVIndexMap *> mergeDebugT(TypeMerger *m,
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

static ArrayRef<GloballyHashedType>
getHashesFromDebugH(ArrayRef<uint8_t> debugH) {
  assert(canUseDebugH(debugH));

  debugH = debugH.drop_front(sizeof(object::debug_h_header));
  uint32_t count = debugH.size() / sizeof(GloballyHashedType);
  return {reinterpret_cast<const GloballyHashedType *>(debugH.data()), count};
}

static void hashCVTypeArray(TpiSource *src, const CVTypeArray &types) {
  // BumpPtrAllocator is not thread-safe, so use `new`.
  std::vector<GloballyHashedType> hashVec =
      GloballyHashedType::hashTypes(types);
  GloballyHashedType *hashes = new GloballyHashedType[hashVec.size()];
  std::copy(hashVec.begin(), hashVec.end(), hashes);
  src->ghashes = makeArrayRef(hashes, hashVec.size());
  src->ownedGHashes = true;
}

void TpiSource::loadGHashes() {
  if (Optional<ArrayRef<uint8_t>> debugH = getDebugH(file)) {
    ghashes = getHashesFromDebugH(*debugH);
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
  isItemIndex.resize(ghashes.size());
  check(forEachCodeViewRecord<CVType>(file->debugTypes, [&](const CVType &ty) {
    if (isIdRecord(ty.kind()))
      isItemIndex.set(index);
    ++index;
  }));
}

// Merge .debug$T for a generic object file.
Expected<const CVIndexMap *> TpiSource::mergeDebugT(TypeMerger *m,
                                                    CVIndexMap *indexMap) {
  CVTypeArray types;
  BinaryStreamReader reader(file->debugTypes, support::little);
  cantFail(reader.readArray(types, reader.getLength()));

  if (config->debugGHashes) {
    if (auto err = mergeTypeAndIdRecords(m->globalIDTable, m->globalTypeTable,
                                         indexMap->tpiMap, types, ghashes,
                                         file->pchSignature))
      fatal("codeview::mergeTypeAndIdRecords failed: " +
            toString(std::move(err)));
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
}

// Merge types from a type server PDB.
Expected<const CVIndexMap *> TypeServerSource::mergeDebugT(TypeMerger *m,
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
    Optional<uint32_t> endPrecomp;
    // Merge TPI first, because the IPI stream will reference type indices.
    if (auto err =
            mergeTypeRecords(m->globalTypeTable, tsIndexMap.tpiMap,
                             expectedTpi->typeArray(), ghashes, endPrecomp))
      fatal("codeview::mergeTypeRecords failed: " + toString(std::move(err)));

    // Merge IPI.
    if (maybeIpi) {
      auto ipiHashes =
          GloballyHashedType::hashIds(maybeIpi->typeArray(), ghashes);
      if (auto err = mergeIdRecords(m->globalIDTable, tsIndexMap.tpiMap,
                                    tsIndexMap.ipiMap, maybeIpi->typeArray(),
                                    ipiHashes))
        fatal("codeview::mergeIdRecords failed: " + toString(std::move(err)));
    }
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

Expected<const CVIndexMap *>
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

Expected<const CVIndexMap *> findPrecompMap(ObjFile *file, PrecompRecord &pr) {
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
static Expected<const CVIndexMap *>
mergeInPrecompHeaderObj(ObjFile *file, CVIndexMap *indexMap,
                        PrecompRecord &precomp) {
  auto e = findPrecompMap(file, precomp);
  if (!e)
    return e.takeError();

  const CVIndexMap *precompIndexMap = *e;
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

Expected<const CVIndexMap *>
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

Expected<const CVIndexMap *> PrecompSource::mergeDebugT(TypeMerger *m,
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
      delete[] src->ghashes.data();
  }
  TpiSource::instances.clear();
  TypeServerSource::mappings.clear();
  PrecompSource::mappings.clear();
}

std::vector<uint64_t> TypeIndexCell::finalGHashesByIndex;

namespace {
/// A ghash table cell for deduplicating types from TpiSources.
class GHashInCell {
  uint64_t data = 0;

public:
  GHashInCell() = default;

  // Construct data most to least significant:
  // - isItem
  // - tpiSrcIdx
  // - ghashIdx
  GHashInCell(uint32_t tpiSrcIdx, bool isItem, uint32_t ghashIdx)
      : data((uint64_t(tpiSrcIdx + 1) << 33ULL) | (uint64_t(isItem) << 32U) |
             ghashIdx) {
    assert(tpiSrcIdx == getTpiSrcIdx() && "too many sources of TPI");
  }

  explicit GHashInCell(uint64_t data) : data(data) {}

  // The empty cell is all zeros.
  bool isEmpty() const { return data == 0ULL; }

  uint32_t getTpiSrcIdx() const { return (uint32_t)(data >> 33U) - 1; }
  bool isItem() const { return data & (1ULL << 32U); }
  uint32_t getGHashIdx() const { return (uint32_t)data; }

  uint64_t getGHash() const {
    // FIXME: GloballyHashedType should expose this.
    return *reinterpret_cast<const uint64_t *>(
        &TpiSource::instances[getTpiSrcIdx()]->ghashes[getGHashIdx()]);
  }

  friend inline bool operator<(const GHashInCell &l, const GHashInCell &r) {
    return l.data < r.data;
  }
};
} // namespace

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
    tableSize += source->ghashes.size();

  GHashTable<GHashInCell> inTable;
  inTable.init(tableSize);

  // Insert ghashes in parallel. It is important to detect duplicates with the
  // least amount of work possible, so that the least amount of time can be
  // spent on them.
  parallelForEachN(0, TpiSource::instances.size(), [&](size_t tpiSrcIdx) {
    TpiSource *source = TpiSource::instances[tpiSrcIdx];
    uint32_t ghashSize = source->ghashes.size();
    for (uint32_t ghashIdx = 0; ghashIdx < ghashSize; ghashIdx++)
      inTable.insert(
          GHashInCell(tpiSrcIdx, source->isItemIndex.test(ghashIdx) ghashIdx));
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
  if (config->verbose) {
    log(formatv("ghash table load factor: {0:p} (size {1} / capacity {2})\n",
                double(entries.size()) / tableSize, entries.size(), tableSize));
  }

#if 0
  // Put a list of all unique type indices on each tpi source. Type merging
  // will skip indices not on this list.
  TypeIndexCell::finalGHashesByIndex.reserve(entries.size());
  std::vector<TypeIndex> uniqueTypes;
  std::vector<TypeIndex> uniqueItems;
  for (auto i = entries.begin(), e = entries.end(); i != e;) {
    uint32_t tpiSrcIdx = i->getTpiSrcIdx();
    for (; i != e && i->getTpiSrcIdx() == tpiSrcIdx; ++i) {
      auto &dstVec = i->isItemIndex() ? uniqueTypes : uniqueItems;
      dstVec.push_back(TypeIndex::fromArrayIndex(i->getGHashIdx()));
      TypeIndexCell::finalGHashesByIndex.push_back(i->getGHash());
    }
    TpiSource *source = TpiSource::instances[tpiSrcIdx];
    TypeIndex *tiMem = bAlloc.Allocate<TypeIndex>(uniqueTypes.size());
    std::copy(uniqueTypes.begin(), uniqueTypes.end(), tiMem);
    source->typesToMerge = makeArrayRef(tiMem, uniqueTypes.size());
    uniqueTypes.clear();
  }
#endif

  // Use a load factor of 2.
  finalGHashMap.init(entries.size() * 2);
  parallelForEachN(0, entries.size(), [&](size_t dstIdx) {
    finalGHashMap.insert(TypeIndexCell(TypeIndex::fromArrayIndex(dstIdx)));
  });
}
