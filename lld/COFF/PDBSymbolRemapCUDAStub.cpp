//===- PDBSymbolRemapCUDAStub.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PDBSymbolRemap.h"
#include "lld/Common/ErrorHandler.h"

using namespace lld;
using namespace lld::coff;

void lld::coff::executePDBSymbolRemapCUDA(
    ArrayRef<uint8_t> sectionContents,
    ArrayRef<PlannedSymbolRecordDescriptor> descriptors,
    ArrayRef<PlannedSymbolTypeRef> typeRefs,
    MutableArrayRef<uint8_t> moduleSymbolStorage) {
  (void)sectionContents;
  (void)descriptors;
  (void)typeRefs;
  (void)moduleSymbolStorage;
  fatal("CUDA PDB symbol remap requires LLD_ENABLE_COFF_GHASH_CUDA=ON");
}
