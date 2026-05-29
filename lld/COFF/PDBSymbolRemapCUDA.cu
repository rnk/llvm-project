//===- PDBSymbolRemapCUDA.cu ----------------------------------------------===//
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
    ArrayRef<PlannedSymbolRecordDescriptor> descriptors,
    ArrayRef<PlannedSymbolTypeRef> typeRefs,
    PDBSymbolRemapSourceMap sourceMap,
    MutableArrayRef<uint8_t> moduleSymbolStorage) {
  (void)descriptors;
  (void)typeRefs;
  (void)sourceMap;
  (void)moduleSymbolStorage;
  fatal("CUDA PDB symbol remap is not implemented");
}
