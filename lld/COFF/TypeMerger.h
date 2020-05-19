//===- TypeMerger.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_TYPEMERGER_H
#define LLD_COFF_TYPEMERGER_H

#include "Config.h"
#include "llvm/DebugInfo/CodeView/MergingTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeHashing.h"
#include "llvm/Support/Allocator.h"
#include <atomic>

namespace lld {
namespace coff {

using llvm::codeview::GloballyHashedType;

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
template <typename Cell>
struct GHashTable {
  Cell *table = nullptr;
  size_t tableSize = 0;

  GHashTable() = default;
  ~GHashTable() { delete[] table; }

  void init(size_t newTableSize) {
    // TODO: Use huge pages or other fancier memory allocations.
    table = new Cell[newTableSize];
    memset(table, 0, newTableSize * sizeof(Cell));
    tableSize = newTableSize;
  }

  void insert(Cell newCell);

  Optional<Cell> lookup(GloballyHashedType ghash);
};

/// A ghash table cell mapping to final PDB type indices. Needed during type
/// merging to remap source type indices lazily.
struct alignas(uint32_t) TypeIndexCell {
  llvm::codeview::TypeIndex ti;

  TypeIndexCell() = default;
  explicit TypeIndexCell(llvm::codeview::TypeIndex ti) : ti(ti) {}

  bool isEmpty() const { return ti.getIndex() == 0; }

  GloballyHashedType getGHash() const {
    uint32_t idx = ti.toArrayIndex();
    return ti.isDecoratedItemId() ? itemGHashes[idx] : typeGHashes[idx];
  }

  friend inline bool operator<(const TypeIndexCell &l, const TypeIndexCell &r) {
    return l.ti < r.ti;
  }

  /// This is the global list of ghashes used by this cell during hash table
  /// insertion.
  static std::vector<GloballyHashedType> itemGHashes;
  static std::vector<GloballyHashedType> typeGHashes;
};

class TypeMerger {
public:
  TypeMerger(llvm::BumpPtrAllocator &alloc)
      : typeTable(alloc), idTable(alloc) {}

  /// Get the type table or the global type table if /DEBUG:GHASH is enabled.
  inline llvm::codeview::TypeCollection &getTypeTable() {
    assert(!config->debugGHashes);
    return typeTable;
  }

  /// Get the ID table or the global ID table if /DEBUG:GHASH is enabled.
  inline llvm::codeview::TypeCollection &getIDTable() {
    assert(!config->debugGHashes);
    return idTable;
  }

  /// Use global hashes to eliminate duplicate types and identify unique type
  /// indices in each TpiSource.
  void identifyUniqueTypeIndices();

  /// A map from ghash to final type index.
  GHashTable<TypeIndexCell> finalGHashMap;

  /// Type records that will go into the PDB TPI stream.
  llvm::codeview::MergingTypeTableBuilder typeTable;

  /// Item records that will go into the PDB IPI stream.
  llvm::codeview::MergingTypeTableBuilder idTable;

  // When showSummary is enabled, these are histograms of TPI and IPI records
  // keyed by type index.
  SmallVector<uint32_t, 0> tpiCounts;
  SmallVector<uint32_t, 0> ipiCounts;
};

/// Map from type index and item index in a type server PDB to the
/// corresponding index in the destination PDB.
struct CVIndexMap {
  llvm::SmallVector<llvm::codeview::TypeIndex, 0> tpiMap;
  llvm::SmallVector<llvm::codeview::TypeIndex, 0> ipiMap;
  bool isTypeServerMap = false;
  bool isPrecompiledTypeMap = false;
};

} // namespace coff
} // namespace lld

#endif
