//===- COFFGPURuntime.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_COFFGPURUNTIME_H
#define LLD_COFF_COFFGPURUNTIME_H

#if defined(LLD_COFF_GHASH_USE_HIP) && LLD_COFF_GHASH_USE_HIP

#include <hip/hip_runtime.h>

#define LLD_COFF_GPU_RUNTIME_NAME "HIP"
#define LLD_COFF_GPU_HAS_CUDA_MEMCPY_BATCH 0

// Keep the implementation source-compatible with the existing CUDA runtime
// calls. HIP uses the same execution syntax and analogous runtime entry points.
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaStream_t hipStream_t
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaGetDevice hipGetDevice
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize

#ifndef CUDART_VERSION
#define CUDART_VERSION 0
#endif

#else

#include <cuda_runtime.h>

#define LLD_COFF_GPU_RUNTIME_NAME "CUDA"
#define LLD_COFF_GPU_HAS_CUDA_MEMCPY_BATCH 1

#endif

#endif
