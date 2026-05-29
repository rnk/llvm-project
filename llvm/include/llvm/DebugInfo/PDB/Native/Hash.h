//===- Hash.h - PDB hash functions ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_HASH_H
#define LLVM_DEBUGINFO_PDB_NATIVE_HASH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <cstdint>

namespace llvm {
namespace pdb {

namespace detail {

LLVM_CUDA_HOST_DEVICE inline uint16_t read16le(const uint8_t *Data) {
  return uint16_t(Data[0]) | (uint16_t(Data[1]) << 8);
}

LLVM_CUDA_HOST_DEVICE inline uint32_t read32le(const uint8_t *Data) {
  return uint32_t(Data[0]) | (uint32_t(Data[1]) << 8) |
         (uint32_t(Data[2]) << 16) | (uint32_t(Data[3]) << 24);
}

} // namespace detail

// Corresponds to `Hasher::lhashPbCb` in PDB/include/misc.h.
// Used for name hash table and TPI/IPI hashes.
LLVM_CUDA_HOST_DEVICE inline uint32_t hashStringV1(const uint8_t *Data,
                                                   size_t Size) {
  uint32_t Result = 0;
  size_t Offset = 0;
  for (; Offset + 4 <= Size; Offset += 4)
    Result ^= detail::read32le(Data + Offset);

  size_t RemainderSize = Size - Offset;

  // Maximum of 3 bytes left. Hash a 2 byte word if possible, then hash the
  // possibly remaining 1 byte.
  if (RemainderSize >= 2) {
    Result ^= detail::read16le(Data + Offset);
    Offset += 2;
    RemainderSize -= 2;
  }

  // Hash possible odd byte.
  if (RemainderSize == 1)
    Result ^= Data[Offset];

  const uint32_t ToLowerMask = 0x20202020U;
  Result |= ToLowerMask;
  Result ^= (Result >> 11);

  return Result ^ (Result >> 16);
}

inline uint32_t hashStringV1(StringRef Str) {
  return hashStringV1(reinterpret_cast<const uint8_t *>(Str.data()),
                      Str.size());
}

LLVM_ABI uint32_t hashStringV2(StringRef Str);

// Corresponds to `SigForPbCb` in langapi/shared/crc32.h.
LLVM_CUDA_HOST_DEVICE inline uint32_t hashBufferV8(const uint8_t *Data,
                                                   size_t Size) {
  uint32_t CRC = 0;
  for (size_t I = 0; I != Size; ++I) {
    CRC ^= Data[I];
    for (uint32_t Bit = 0; Bit != 8; ++Bit)
      CRC = (CRC >> 1) ^ (0xEDB88320U & (0U - (CRC & 1U)));
  }
  return CRC;
}

inline uint32_t hashBufferV8(ArrayRef<uint8_t> Data) {
  return hashBufferV8(Data.data(), Data.size());
}

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_NATIVE_HASH_H
