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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include <cstdint>

namespace lld::coff {

enum PlannedSymbolRecordFlags : uint16_t {
  PSRF_GoesInModule = 1 << 0,
  PSRF_OpensScope = 1 << 1,
  PSRF_ClosesScope = 1 << 2,
  PSRF_KnownTypeRefs = 1 << 3,
  PSRF_TranslateProcIdEnd = 1 << 4,
  PSRF_TranslateProcIdRecord = 1 << 5,
  PSRF_HasIdTypeIndex = 1 << 6,
  PSRF_HasIdFinalTypeIndex = 1 << 7,
  PSRF_WarnInvalidFuncId = 1 << 8,
};

enum PlannedSymbolTypeRefKind : uint8_t {
  PSTRK_TypeRef,
  PSTRK_IndexRef,
};

struct PlannedSymbolTypeRef {
  uint32_t contentOffset;
  uint8_t refKind;
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
  uint32_t typeRefStartIndex;
  uint32_t typeRefCount;
  uint32_t idTypeIndexOffset;
  uint32_t idFinalTypeIndex;
  uint16_t kind;
  uint16_t flags;
};

// Source type-index maps for the remap/translate phase. This boundary stays
// numeric: CUDA implementations should copy the raw 32-bit TypeIndex values to
// device memory before launching kernels rather than depending on CodeView
// parsing helpers.
struct PDBSymbolRemapSourceMap {
  ArrayRef<llvm::codeview::TypeIndex> tpiMap;
  ArrayRef<llvm::codeview::TypeIndex> ipiMap;
};

void executePDBSymbolRemapCUDA(
    ArrayRef<PlannedSymbolRecordDescriptor> descriptors,
    ArrayRef<PlannedSymbolTypeRef> typeRefs,
    PDBSymbolRemapSourceMap sourceMap,
    MutableArrayRef<uint8_t> moduleSymbolStorage);

} // namespace lld::coff

#endif
