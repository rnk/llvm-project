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
#include "llvm/DebugInfo/CodeView/GlobalTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/MergingTypeTableBuilder.h"
#include "llvm/Support/Allocator.h"

namespace lld {
namespace coff {

/// A simplified hash table customized for global type hashing.
///
/// This hash table is meant to be used in two phases:
/// 1. concurrent insertion
/// 2. concurrent lookup
/// It does not support deletion or rehashing. It uses linear probing for better
/// cache locality.
///
/// The Cell type must support the following API:
///   // Get the ghash for this cell
///   uint64_t getGHash();
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
    tableSize = newTableSize;
  }

  void insert(Cell newCell);

  Optional<Cell> lookup(uint64_t ghash);
};

/// A ghash table cell mapping to final PDB type indices. Needed during type
/// merging to remap source type indices lazily.
struct alignas(uint32_t) TypeIndexCell {
  llvm::codeview::TypeIndex ti;

  TypeIndexCell() = default;
  explicit TypeIndexCell(llvm::codeview::TypeIndex ti) : ti(ti) {}

  bool isEmpty() const { return ti.getIndex() == 0; }

  uint64_t getGHash() const { return finalGHashesByIndex[ti.toArrayIndex()]; }

  friend inline bool operator<(const TypeIndexCell &l, const TypeIndexCell &r) {
    return l.ti < r.ti;
  }

  /// This is the global list of ghashes used by this cell during hash table
  /// insertion.
  static std::vector<uint64_t> finalGHashesByIndex;
};

class TypeMerger {
public:
  TypeMerger(llvm::BumpPtrAllocator &alloc)
      : typeTable(alloc), idTable(alloc), globalTypeTable(alloc),
        globalIDTable(alloc) {}

  /// Get the type table or the global type table if /DEBUG:GHASH is enabled.
  inline llvm::codeview::TypeCollection &getTypeTable() {
    if (config->debugGHashes)
      return globalTypeTable;
    return typeTable;
  }

  /// Get the ID table or the global ID table if /DEBUG:GHASH is enabled.
  inline llvm::codeview::TypeCollection &getIDTable() {
    if (config->debugGHashes)
      return globalIDTable;
    return idTable;
  }

  /// Use global hashes to eliminate duplicate types and identify unique type
  /// indices in each TpiSource.
  void identifyUniqueTypeIndices();

  /// Lookup the final type index to use in the PDB by global hash. Must be
  /// called after identifyUniqueTypeIndices.
  llvm::codeview::TypeIndex lookupPdbTypeIndexForGHash(uint64_t ghash) {
    auto maybeCell = finalGHashMap.lookup(ghash);
    assert(maybeCell && "unmapped ghash");
    return maybeCell->ti;
  }

  /// A map from ghash to final type index.
  GHashTable<TypeIndexCell> finalGHashMap;

  /// Type records that will go into the PDB TPI stream.
  llvm::codeview::MergingTypeTableBuilder typeTable;

  /// Item records that will go into the PDB IPI stream.
  llvm::codeview::MergingTypeTableBuilder idTable;

  /// Type records that will go into the PDB TPI stream (for /DEBUG:GHASH)
  llvm::codeview::GlobalTypeTableBuilder globalTypeTable;

  /// Item records that will go into the PDB IPI stream (for /DEBUG:GHASH)
  llvm::codeview::GlobalTypeTableBuilder globalIDTable;

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

template <typename Cell> inline void GHashTable<Cell>::insert(Cell newCell) {
  uint64_t ghash = newCell.getGHash();

  // FIXME: The low bytes of SHA1 have low entropy for short records, which
  // type records are. Swap the byte order for better entropy. A better ghash
  // won't need this.
  size_t startIdx = ByteSwap_64(ghash) % tableSize;

  // Do a linear probe starting at startIdx.
  size_t idx = startIdx;
  while (true) {
    auto *cellPtr = reinterpret_cast<std::atomic<Cell> *>(&table[idx]);
    Cell oldCell(cellPtr->load());

    // If we found an empty cell or an existing cell with the same key and
    // lower priority (higher value), CAS this tell into place. If the CAS
    // fails, reload this cell and check again.
    if (oldCell.isEmpty() ||
        (oldCell.getGHash() == ghash && newCell < oldCell)) {
      if (cellPtr->compare_exchange_strong(oldCell, newCell))
        return;
      continue;
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
inline Optional<Cell> GHashTable<Cell>::lookup(uint64_t ghash) {
  size_t startIdx = ByteSwap_64(ghash) % tableSize;
  size_t idx = startIdx;
  while (true) {
    // We should be able to use no atomics or relaxed atomics at this point.
    Cell cell = table[idx];
    if (cell.isEmpty())
      return None; // Empty
    else if (ghash == cell.getGHash())
      return cell;

    // Advance the probe. Wrap around to the beginning if we run off the end.
    ++idx;
    idx = idx == tableSize ? 0 : idx;
    if (idx == startIdx)
      report_fatal_error("ghash table is full");
  }
  llvm_unreachable("left infloop");
}

} // namespace coff
} // namespace lld

#endif
