//===- TypeHashing.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeHashing.h"

#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"

#include "../../../../../BLAKE3/c/blake3.h"

using namespace llvm;
using namespace llvm::codeview;

LocallyHashedType DenseMapInfo<LocallyHashedType>::Empty{0, {}};
LocallyHashedType DenseMapInfo<LocallyHashedType>::Tombstone{hash_code(-1), {}};

static std::array<uint8_t, 8> EmptyHash = {
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}};
static std::array<uint8_t, 8> TombstoneHash = {
    {0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}};

GloballyHashedType DenseMapInfo<GloballyHashedType>::Empty{EmptyHash};
GloballyHashedType DenseMapInfo<GloballyHashedType>::Tombstone{TombstoneHash};

LocallyHashedType LocallyHashedType::hashType(ArrayRef<uint8_t> RecordData) {
  return {llvm::hash_value(RecordData), RecordData};
}

GloballyHashedType
GloballyHashedType::hashType(ArrayRef<uint8_t> RecordData,
                             ArrayRef<GloballyHashedType> PreviousTypes,
                             ArrayRef<GloballyHashedType> PreviousIds) {
  SmallVector<TiReference, 4> Refs;
  discoverTypeIndices(RecordData, Refs);

  blake3_hasher b3;
  blake3_hasher_init(&b3);

  uint32_t Off = 0;
  blake3_hasher_update(&b3, RecordData.data(), sizeof(RecordPrefix));
  RecordData = RecordData.drop_front(sizeof(RecordPrefix));
  for (const auto &Ref : Refs) {
    // Hash any data that comes before this TiRef.
    uint32_t PreLen = Ref.Offset - Off;
    ArrayRef<uint8_t> PreData = RecordData.slice(Off, PreLen);
    blake3_hasher_update(&b3, PreData.data(), PreData.size());
    auto Prev = (Ref.Kind == TiRefKind::IndexRef) ? PreviousIds : PreviousTypes;

    auto RefData = RecordData.slice(Ref.Offset, Ref.Count * sizeof(TypeIndex));
    // For each type index referenced, add in the previously computed hash
    // value of that type.
    ArrayRef<TypeIndex> Indices(
        reinterpret_cast<const TypeIndex *>(RefData.data()), Ref.Count);
    for (TypeIndex TI : Indices) {
      ArrayRef<uint8_t> BytesToHash;
      if (TI.isSimple() || TI.isNoneType()) {
        const uint8_t *IndexBytes = reinterpret_cast<const uint8_t *>(&TI);
        BytesToHash = makeArrayRef(IndexBytes, sizeof(TypeIndex));
      } else {
        if (TI.toArrayIndex() >= Prev.size() ||
            Prev[TI.toArrayIndex()].empty()) {
          // There are references to yet-unhashed records. Suspend hashing for
          // this record until all the other records are processed.
          return {};
        }
        BytesToHash = Prev[TI.toArrayIndex()].Hash;
      }
      blake3_hasher_update(&b3, BytesToHash.data(), BytesToHash.size());
    }

    Off = Ref.Offset + Ref.Count * sizeof(TypeIndex);
  }

  // Don't forget to add in any trailing bytes.
  auto TrailingBytes = RecordData.drop_front(Off);
  blake3_hasher_update(&b3, TrailingBytes.data(), TrailingBytes.size());

  uint8_t hash64[8];
  blake3_hasher_finalize(&b3, &hash64[0], 8);
  return {hash64};
}
