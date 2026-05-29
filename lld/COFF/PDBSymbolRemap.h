//===- PDBSymbolRemap.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_PDBSYMBOLREMAP_H
#define LLD_COFF_PDBSYMBOLREMAP_H

#include "lld/Common/LLVM.h"
#include <cstdint>

namespace lld::coff {

enum PlannedSymbolRecordFlags : uint16_t {
  PSRF_GoesInModule = 1 << 0,
  PSRF_OpensScope = 1 << 1,
  PSRF_ClosesScope = 1 << 2,
};

// Numeric-only symbol work item for the CUDA/bulk executor boundary. Scope and
// string-table fixups intentionally stay as sparse CPU-side patches for now.
struct PlannedSymbolRecordDescriptor {
  uint32_t inputOffset;
  uint32_t inputSize;
  uint32_t outputOffset;
  uint32_t alignedSize;
  uint32_t relocStartIndex;
  uint32_t relocEndIndex;
  uint16_t kind;
  uint16_t flags;
};

void executePDBSymbolRemapCUDA(
    ArrayRef<uint8_t> sectionContents,
    ArrayRef<PlannedSymbolRecordDescriptor> descriptors,
    MutableArrayRef<uint8_t> moduleSymbolStorage);

} // namespace lld::coff

#endif
