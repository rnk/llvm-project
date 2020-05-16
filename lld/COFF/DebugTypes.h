//===- DebugTypes.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_DEBUGTYPES_H
#define LLD_COFF_DEBUGTYPES_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace codeview {
struct GloballyHashedType;
} // namespace codeview
namespace pdb {
class NativeSession;
class TpiStream;
}
} // namespace llvm

namespace lld {
namespace coff {

using llvm::codeview::GloballyHashedType;
using llvm::codeview::TypeIndex;

class ObjFile;
class PDBInputFile;
struct CVIndexMap;
class TypeMerger;

class TpiSource {
public:
  enum TpiKind : uint8_t { Regular, PCH, UsingPCH, PDB, UsingPDB };

  TpiSource(TpiKind k, ObjFile *f);
  virtual ~TpiSource();

  /// Load global hashes, either by hashing types directly, or by loading them
  /// from LLVM's .debug$H section.
  virtual void loadGHashes();

  /// Produce a mapping from the type and item indices used in the object
  /// file to those in the destination PDB.
  ///
  /// If the object file uses a type server PDB (compiled with /Zi), merge TPI
  /// and IPI from the type server PDB and return a map for it. Each unique type
  /// server PDB is merged at most once, so this may return an existing index
  /// mapping.
  ///
  /// If the object does not use a type server PDB (compiled with /Z7), we merge
  /// all the type and item records from the .debug$S stream and fill in the
  /// caller-provided ObjectIndexMap.
  virtual Expected<CVIndexMap *> mergeDebugT(TypeMerger *m,
                                             CVIndexMap *indexMap);

  bool remapTypeIndex(TypeMerger *m, TypeIndex &ti,
                      llvm::codeview::TiRefKind refKind, CVIndexMap &indexMap);

protected:
  void remapRecord(TypeMerger *m, MutableArrayRef<uint8_t> rec,
                   CVIndexMap &indexMap,
                   ArrayRef<llvm::codeview::TiReference> typeRefs);

  void mergeTypeRecord(llvm::codeview::CVType ty, TypeMerger *m,
                       CVIndexMap *indexMap);

  void mergeUniqueTypeRecords(const llvm::codeview::CVTypeArray &types,
                              TypeMerger *m, CVIndexMap *indexMap);

  // Eagerly fill in type server index maps. They will be used concurrently, so
  // they cannot be filled lazily.
  void fillMapFromGHashes(TypeMerger *m,
                          llvm::SmallVectorImpl<TypeIndex> &indexMap);

  // Copies ghashes from a vector into an array. These are long lived, so it's
  // worth the time to copy these into an appropriately sized vector to reduce
  // memory usage.
  void assignGHashesFromVector(std::vector<GloballyHashedType> &&hashVec);

  // Walk over file->debugTypes and fill in the isItemIndex bit vector.
  void fillIsItemIndexFromDebugT();

public:
  bool remapTypesInSymbolRecord(MutableArrayRef<uint8_t> rec, TypeMerger *m,
                                CVIndexMap &indexMap);

  void remapTypesInTypeRecord(MutableArrayRef<uint8_t> rec, TypeMerger *m,
                              CVIndexMap &indexMap);

  /// Is this a dependent file that needs to be processed first, before other
  /// OBJs?
  virtual bool isDependency() const { return false; }

  /// Returns true if this type record should be omitted from the PDB, even if
  /// it is unique. This prevents a record from being added to the input ghash
  /// table.
  bool shouldOmitFromPdb(uint32_t ghashIdx) {
    return ghashIdx == endPrecompGHashIdx;
  }

  static std::vector<TpiSource *> instances;

  static uint32_t countTypeServerPDBs();
  static uint32_t countPrecompObjs();

  /// Clear global data structures for TpiSources.
  static void clear();

  const TpiKind kind;
  bool ownedGHashes = true;
  uint32_t tpiSrcIdx = 0;

protected:
  /// The ghash index (zero based, not 0x1000-based) of the LF_ENDPRECOMP record
  /// in this object, if one exists. This is the all ones value otherwise. It is
  /// recorded here so that it can be omitted from the final ghash table.
  uint32_t endPrecompGHashIdx = ~0U;

public:
  ObjFile *file;

  /// GHashes for TPI and IPI records.
  ArrayRef<GloballyHashedType> ghashes;

  /// Indicates if a type record is an item index or a type index.
  llvm::BitVector isItemIndex;

  /// A list of all "unique" type indices which must be merged into the final
  /// PDB. GHash type deduplication produces this list, and it should be
  /// considerably smaller than the input.
  std::vector<TypeIndex> uniqueTypes;

  struct MergedInfo {
    std::vector<uint8_t> recs;
    std::vector<uint16_t> recSizes;
    std::vector<uint32_t> recHashes;
  };

  MergedInfo mergedTpi;
  MergedInfo mergedIpi;
};

TpiSource *makeTpiSource(ObjFile *file);
TpiSource *makeTypeServerSource(PDBInputFile *pdbInputFile);
TpiSource *makeUseTypeServerSource(ObjFile *file,
                                   llvm::codeview::TypeServer2Record ts);
TpiSource *makePrecompSource(ObjFile *file);
TpiSource *makeUsePrecompSource(ObjFile *file,
                                llvm::codeview::PrecompRecord ts);

} // namespace coff
} // namespace lld

#endif
