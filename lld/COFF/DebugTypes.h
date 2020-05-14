//===- DebugTypes.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_DEBUGTYPES_H
#define LLD_COFF_DEBUGTYPES_H

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace codeview {
struct GloballyHashedType;
class PrecompRecord;
class TypeServer2Record;
class TypeIndex;
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
  virtual llvm::Expected<const CVIndexMap *> mergeDebugT(TypeMerger *m,
                                                         CVIndexMap *indexMap);

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
  llvm::ArrayRef<llvm::codeview::GloballyHashedType> ghashes;

  /// A list of all "unique" type indices which must be merged into the final
  /// PDB. GHash type deduplication produces this list, and it should be
  /// considerably smaller than the input.
  llvm::ArrayRef<llvm::codeview::TypeIndex> typesToMerge;
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
