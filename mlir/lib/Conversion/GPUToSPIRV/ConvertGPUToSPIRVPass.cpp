//===- ConvertGPUToSPIRVPass.cpp - GPU to SPIR-V dialect lowering passes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert a kernel function in the GPU Dialect
// into a spv.module operation
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Pass to lower GPU Dialect to SPIR-V. The pass only converts the gpu.func ops
/// inside gpu.module ops. i.e., the function that are referenced in
/// gpu.launch_func ops. For each such function
///
/// 1) Create a spirv::ModuleOp, and clone the function into spirv::ModuleOp
/// (the original function is still needed by the gpu::LaunchKernelOp, so cannot
/// replace it).
///
/// 2) Lower the body of the spirv::ModuleOp.
struct GPUToSPIRVPass : public ModulePass<GPUToSPIRVPass> {
/// Include the generated pass utilities.
#define GEN_PASS_ConvertGpuToSPIRV
#include "mlir/Conversion/Passes.h.inc"

  void runOnModule() override;
};
} // namespace

void GPUToSPIRVPass::runOnModule() {
  MLIRContext *context = &getContext();
  ModuleOp module = getModule();

  SmallVector<Operation *, 1> kernelModules;
  OpBuilder builder(context);
  module.walk([&builder, &kernelModules](gpu::GPUModuleOp moduleOp) {
    // For each kernel module (should be only 1 for now, but that is not a
    // requirement here), clone the module for conversion because the
    // gpu.launch function still needs the kernel module.
    builder.setInsertionPoint(moduleOp.getOperation());
    kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
  });

  auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
  std::unique_ptr<ConversionTarget> target =
      spirv::SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);
  OwningRewritePatternList patterns;
  populateGPUToSPIRVPatterns(context, typeConverter, patterns);
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);

  if (failed(applyFullConversion(kernelModules, *target, patterns,
                                 &typeConverter))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertGPUToSPIRVPass() {
  return std::make_unique<GPUToSPIRVPass>();
}
