//===- Hash.cpp - PDB Hash Functions --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Endian.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::support;

// Corresponds to `HasherV2::HashULONG` in PDB/include/misc.h.
// Used for name hash table.
uint32_t pdb::hashStringV2(StringRef Str) {
  uint32_t Hash = 0xb170a1bf;

  ArrayRef<char> Buffer(Str.begin(), Str.end());

  ArrayRef<ulittle32_t> Items(
      reinterpret_cast<const ulittle32_t *>(Buffer.data()),
      Buffer.size() / sizeof(ulittle32_t));
  for (ulittle32_t Item : Items) {
    Hash += Item;
    Hash += (Hash << 10);
    Hash ^= (Hash >> 6);
  }
  Buffer = Buffer.slice(Items.size() * sizeof(ulittle32_t));
  for (uint8_t Item : Buffer) {
    Hash += Item;
    Hash += (Hash << 10);
    Hash ^= (Hash >> 6);
  }

  return Hash * 1664525U + 1013904223U;
}
