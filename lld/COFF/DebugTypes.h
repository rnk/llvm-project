//===- DebugTypes.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_DEBUGTYPES_H
#define LLD_COFF_DEBUGTYPES_H

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "lld/Common/LLVM.h"

namespace llvm {
namespace codeview {
class PrecompRecord;
class TypeServer2Record;
class TypeIndex;
struct TiReference;
} // namespace codeview
namespace pdb {
class NativeSession;
}
} // namespace llvm

namespace lld {
namespace coff {

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

  bool remapTypeIndex(TypeMerger *m, llvm::codeview::TypeIndex &ti,
                      MutableArrayRef<llvm::codeview::TypeIndex> typeIndexMap);

  void remapRecord(TypeMerger *m, MutableArrayRef<uint8_t> rec,
                   CVIndexMap &indexMap,
                   ArrayRef<llvm::codeview::TiReference> typeRefs);

  /// Is this a dependent file that needs to be processed first, before other
  /// OBJs?
  virtual bool isDependency() const { return false; }

  static std::vector<TpiSource *> instances;

  static uint32_t countTypeServerPDBs();
  static uint32_t countPrecompObjs();

  /// Clear global data structures for TpiSources.
  static void clear();

  const TpiKind kind;
  bool ownedGHashes = true;
  uint32_t tpiSrcIdx = 0;
  ObjFile *file;

  /// GHashes for TPI and IPI records. ipiGHashes will be empty, except for PDB
  /// type server sources. In object files (precompiled or regular), all types
  /// are in one stream, .debug$T.
  ArrayRef<uint64_t> tpiGHashes;
  ArrayRef<uint64_t> ipiGHashes;

  /// Indicates if a type record is an item index or a type index.
  llvm::BitVector isItemIndex;

  /// A list of all "unique" type indices which must be merged into the final
  /// PDB. GHash type deduplication produces this list, and it should be
  /// considerably smaller than the input.
  std::vector<llvm::codeview::TypeIndex> uniqueTypes;

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
