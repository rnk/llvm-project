//===- DebugTypesCUDAStub.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeMerger.h"
#include "lld/Common/ErrorHandler.h"

using namespace lld;
using namespace lld::coff;

void TypeMerger::mergeTypesWithCUDA() {
  Fatal(ctx) << "-lldcudaghash requires LLD_ENABLE_COFF_GHASH_CUDA=ON";
}
