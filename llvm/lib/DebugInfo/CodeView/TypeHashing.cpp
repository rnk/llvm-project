//===- TypeHashing.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeHashing.h"

#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/Support/BLAKE3.h"

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
  TruncatedBLAKE3<8> S;
  S.init();
  uint32_t Off = 0;
  S.update(RecordData.take_front(sizeof(RecordPrefix)));
  ArrayRef<uint8_t> Content = RecordData.drop_front(sizeof(RecordPrefix));
  bool Success = true;
  discoverTypeIndices(RecordData, [&](TiRefKind RefKind, uint32_t RefOffset) {
    if (!Success)
      return;

    // Hash any data that comes before this TiRef.
    uint32_t PreLen = RefOffset - Off;
    ArrayRef<uint8_t> PreData = Content.slice(Off, PreLen);
    S.update(PreData);
    auto Prev = (RefKind == TiRefKind::IndexRef) ? PreviousIds : PreviousTypes;

    // For each type index referenced, add in the previously computed hash
    // value of that type.
    TypeIndex TI =
        *reinterpret_cast<const TypeIndex *>(Content.data() + RefOffset);
    ArrayRef<uint8_t> BytesToHash;
    if (TI.isSimple() || TI.isNoneType()) {
      const uint8_t *IndexBytes = reinterpret_cast<const uint8_t *>(&TI);
      BytesToHash = ArrayRef(IndexBytes, sizeof(TypeIndex));
    } else {
      if (TI.toArrayIndex() >= Prev.size() || Prev[TI.toArrayIndex()].empty()) {
        // There are references to yet-unhashed records. Suspend hashing for
        // this record until all the other records are processed.
        Success = false;
        return;
      }
      BytesToHash = Prev[TI.toArrayIndex()].Hash;
    }
    S.update(BytesToHash);

    Off = RefOffset + sizeof(TypeIndex);
  });
  if (!Success)
    return {};

  // Don't forget to add in any trailing bytes.
  auto TrailingBytes = Content.drop_front(Off);
  S.update(TrailingBytes);

  return {S.final()};
}
