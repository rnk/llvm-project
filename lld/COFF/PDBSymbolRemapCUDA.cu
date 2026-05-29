//===- PDBSymbolRemapCUDA.cu ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PDBSymbolRemap.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

using namespace lld;
using namespace lld::coff;

namespace {

constexpr uint32_t firstNonSimpleIndex =
    llvm::codeview::TypeIndex::FirstNonSimpleIndex;
constexpr uint32_t decoratedItemIdMask =
    llvm::codeview::TypeIndex::DecoratedItemIdMask;
constexpr uint32_t notTranslatedIndex =
    static_cast<uint32_t>(llvm::codeview::SimpleTypeKind::NotTranslated);
constexpr uint32_t maxSymbolRemapErrors = 8;

enum class DeviceSymbolRemapErrorKind : uint8_t {
  BadRecordRange,
  BadTypeRefOffset,
  RemapMiss,
  BadIdTypeOffset,
};

enum class DeviceSymbolRefKind : uint8_t { TypeRef, IndexRef };

struct DeviceSymbolRemapErrorSummary {
  uint32_t count = 0;
  uint32_t recordIndex[maxSymbolRemapErrors] = {};
  uint16_t recordKind[maxSymbolRemapErrors] = {};
  uint32_t detail[maxSymbolRemapErrors] = {};
  DeviceSymbolRemapErrorKind errorKind[maxSymbolRemapErrors] = {};
};

class CudaSymbolRemapErrorChecker {
public:
  void fatalIfFailed(cudaError_t errorCode, const char *context) {
    if (errorCode == cudaSuccess)
      return;
    fatal(std::string(context) + ": " + cudaGetErrorString(errorCode));
  }
};

class CudaStream {
public:
  CudaStream() = default;
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  ~CudaStream() {
    if (stream)
      cudaStreamDestroy(stream);
  }

  cudaStream_t get(CudaSymbolRemapErrorChecker &cuErr) {
    if (!stream)
      cuErr.fatalIfFailed(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "CUDA PDB symbol remap failed to create stream");
    return stream;
  }

private:
  cudaStream_t stream = nullptr;
};

uint32_t checkedUInt32(size_t value, const char *what) {
  if (value > std::numeric_limits<uint32_t>::max())
    fatal(std::string("CUDA PDB symbol remap failed: ") + what +
          " exceeds 32-bit range");
  return static_cast<uint32_t>(value);
}

uint32_t getBlockCount(uint32_t count, uint32_t threads) {
  uint32_t blockCount = (count + threads - 1) / threads;
  if (blockCount == 0)
    return 0;
  return blockCount;
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (ptr)
      cudaFree(ptr);
  }

  T *data() { return ptr; }
  const T *data() const { return ptr; }

  T *data(size_t count) { return count == 0 ? nullptr : ptr; }
  const T *data(size_t count) const { return count == 0 ? nullptr : ptr; }

  void ensureCapacity(size_t count, CudaSymbolRemapErrorChecker &cuErr,
                      const char *context) {
    if (count <= capacity)
      return;
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
      fatal("CUDA PDB symbol remap failed: scratch buffer size overflow");

    T *newPtr = nullptr;
    cuErr.fatalIfFailed(
        cudaMalloc(reinterpret_cast<void **>(&newPtr), count * sizeof(T)),
        context);
    if (ptr)
      cuErr.fatalIfFailed(
          cudaFree(ptr),
          "CUDA PDB symbol remap failed to release old scratch buffer");
    ptr = newPtr;
    capacity = count;
  }

  void copyFrom(ArrayRef<T> values, CudaSymbolRemapErrorChecker &cuErr,
                const char *allocContext, const char *copyContext) {
    ensureCapacity(values.size(), cuErr, allocContext);
    if (values.empty())
      return;
    cuErr.fatalIfFailed(
        cudaMemcpy(ptr, values.data(), values.size() * sizeof(T),
                   cudaMemcpyHostToDevice),
        copyContext);
  }

  void copyFromAsync(ArrayRef<T> values, CudaSymbolRemapErrorChecker &cuErr,
                     cudaStream_t stream, const char *allocContext,
                     const char *copyContext) {
    ensureCapacity(values.size(), cuErr, allocContext);
    if (values.empty())
      return;
    cuErr.fatalIfFailed(
        cudaMemcpyAsync(ptr, values.data(), values.size() * sizeof(T),
                        cudaMemcpyHostToDevice, stream),
        copyContext);
  }

private:
  T *ptr = nullptr;
  size_t capacity = 0;
};

struct SymbolRemapDeviceScratch {
  CudaStream stream;
  DeviceBuffer<PlannedSymbolRecordDescriptor> descriptors;
  DeviceBuffer<PlannedSymbolTypeRef> typeRefs;
  DeviceBuffer<uint32_t> tpiMap;
  DeviceBuffer<uint32_t> ipiMap;
  DeviceBuffer<uint8_t> moduleSymbolStorage;
  DeviceBuffer<DeviceSymbolRemapErrorSummary> errors;
};

void copyTypeIndexMapAsync(DeviceBuffer<uint32_t> &buffer,
                           ArrayRef<llvm::codeview::TypeIndex> map,
                           CudaSymbolRemapErrorChecker &cuErr,
                           cudaStream_t stream, const char *allocContext,
                           const char *copyContext) {
  static_assert(sizeof(llvm::codeview::TypeIndex) == sizeof(uint32_t),
                "TypeIndex must be stored as a single 32-bit value");

  buffer.ensureCapacity(map.size(), cuErr, allocContext);
  if (map.empty())
    return;
  // cudaMemcpyAsync copies bytes from the TypeIndex object storage; no host
  // uint32_t lvalue is formed, so source alignment is not a correctness issue.
  cuErr.fatalIfFailed(
      cudaMemcpyAsync(buffer.data(), map.data(), map.size() * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, stream),
      copyContext);
}

const char *getSymbolRemapErrorName(DeviceSymbolRemapErrorKind kind) {
  switch (kind) {
  case DeviceSymbolRemapErrorKind::BadRecordRange:
    return "bad record range";
  case DeviceSymbolRemapErrorKind::BadTypeRefOffset:
    return "bad type reference offset";
  case DeviceSymbolRemapErrorKind::RemapMiss:
    return "type remap miss";
  case DeviceSymbolRemapErrorKind::BadIdTypeOffset:
    return "bad ID type offset";
  }
  return "unknown error";
}

__host__ __device__ bool isStructuralError(DeviceSymbolRemapErrorKind kind) {
  return kind == DeviceSymbolRemapErrorKind::BadRecordRange ||
         kind == DeviceSymbolRemapErrorKind::BadTypeRefOffset ||
         kind == DeviceSymbolRemapErrorKind::BadIdTypeOffset;
}

__device__ uint16_t read16le(const uint8_t *p) {
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

__device__ uint32_t read32le(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

__device__ void write16le(uint8_t *p, uint16_t v) {
  p[0] = uint8_t(v);
  p[1] = uint8_t(v >> 8);
}

__device__ void write32le(uint8_t *p, uint32_t v) {
  p[0] = uint8_t(v);
  p[1] = uint8_t(v >> 8);
  p[2] = uint8_t(v >> 16);
  p[3] = uint8_t(v >> 24);
}

__device__ void recordSymbolRemapError(
    DeviceSymbolRemapErrorSummary *errors, uint32_t recordIndex,
    uint16_t recordKind, DeviceSymbolRemapErrorKind errorKind,
    uint32_t detail) {
  if (!errors)
    return;

  uint32_t slot = atomicAdd(&errors->count, 1U);
  if (slot >= maxSymbolRemapErrors) {
    if (!isStructuralError(errorKind))
      return;
    slot = maxSymbolRemapErrors - 1;
  }
  errors->recordIndex[slot] = recordIndex;
  errors->recordKind[slot] = recordKind;
  errors->errorKind[slot] = errorKind;
  errors->detail[slot] = detail;
}

__device__ void replaceWithSkipRecord(uint8_t *recordBytes,
                                      uint32_t alignedSize) {
  for (uint32_t i = 0; i != alignedSize; ++i)
    recordBytes[i] = 0;
  write16le(recordBytes, uint16_t(alignedSize - 2));
  write16le(recordBytes + 2, uint16_t(llvm::codeview::SymbolKind::S_SKIP));
}

__device__ bool remapTypeIndexValue(uint32_t &typeIndex,
                                    DeviceSymbolRefKind refKind,
                                    const uint32_t *tpiMap,
                                    uint32_t tpiMapSize,
                                    const uint32_t *ipiMap,
                                    uint32_t ipiMapSize) {
  if (typeIndex < firstNonSimpleIndex)
    return true;

  uint32_t arrayIndex = (typeIndex & ~decoratedItemIdMask) -
                        firstNonSimpleIndex;
  const uint32_t *map =
      refKind == DeviceSymbolRefKind::IndexRef ? ipiMap : tpiMap;
  uint32_t mapSize =
      refKind == DeviceSymbolRefKind::IndexRef ? ipiMapSize : tpiMapSize;
  if (arrayIndex >= mapSize) {
    typeIndex = notTranslatedIndex;
    return false;
  }

  typeIndex = map[arrayIndex];
  return true;
}

__device__ bool remapTypeIndexAt(uint8_t *content, uint32_t contentSize,
                                 const PlannedSymbolTypeRef &typeRef,
                                 const uint32_t *tpiMap, uint32_t tpiMapSize,
                                 const uint32_t *ipiMap, uint32_t ipiMapSize,
                                 DeviceSymbolRemapErrorSummary *errors,
                                 uint32_t recordIndex, uint16_t recordKind) {
  if (typeRef.contentOffset > contentSize ||
      contentSize - typeRef.contentOffset < sizeof(uint32_t)) {
    recordSymbolRemapError(errors, recordIndex, recordKind,
                           DeviceSymbolRemapErrorKind::BadTypeRefOffset,
                           typeRef.contentOffset);
    return false;
  }

  uint32_t typeIndex = read32le(content + typeRef.contentOffset);
  uint32_t originalTypeIndex = typeIndex;
  DeviceSymbolRefKind refKind = typeRef.refKind == PSTRK_IndexRef
                                    ? DeviceSymbolRefKind::IndexRef
                                    : DeviceSymbolRefKind::TypeRef;
  bool ok = remapTypeIndexValue(typeIndex, refKind, tpiMap, tpiMapSize, ipiMap,
                                ipiMapSize);
  write32le(content + typeRef.contentOffset, typeIndex);
  if (!ok)
    recordSymbolRemapError(errors, recordIndex, recordKind,
                           DeviceSymbolRemapErrorKind::RemapMiss,
                           originalTypeIndex);
  return ok;
}

__device__ void translateIdSymbolRecord(
    uint8_t *recordBytes, uint32_t alignedSize,
    const PlannedSymbolRecordDescriptor &desc,
    DeviceSymbolRemapErrorSummary *errors, uint32_t recordIndex) {
  if (alignedSize < 4) {
    recordSymbolRemapError(errors, recordIndex, desc.kind,
                           DeviceSymbolRemapErrorKind::BadRecordRange,
                           alignedSize);
    return;
  }

  uint16_t kind = read16le(recordBytes + 2);
  if (kind == uint16_t(llvm::codeview::SymbolKind::S_SKIP))
    return;

  if ((desc.flags & PSRF_TranslateProcIdEnd) &&
      kind == uint16_t(llvm::codeview::SymbolKind::S_PROC_ID_END)) {
    write16le(recordBytes + 2, uint16_t(llvm::codeview::SymbolKind::S_END));
    return;
  }

  if (!(desc.flags & PSRF_TranslateProcIdRecord))
    return;
  if (kind != uint16_t(llvm::codeview::SymbolKind::S_GPROC32_ID) &&
      kind != uint16_t(llvm::codeview::SymbolKind::S_LPROC32_ID))
    return;

  uint32_t contentSize = alignedSize - 4;
  if (desc.idTypeIndexOffset > contentSize ||
      contentSize - desc.idTypeIndexOffset < sizeof(uint32_t)) {
    recordSymbolRemapError(errors, recordIndex, kind,
                           DeviceSymbolRemapErrorKind::BadIdTypeOffset,
                           desc.idTypeIndexOffset);
    return;
  }

  uint8_t *typeIndexBytes = recordBytes + 4 + desc.idTypeIndexOffset;
  uint32_t typeIndex = read32le(typeIndexBytes);
  if (typeIndex >= firstNonSimpleIndex && typeIndex != 0) {
    if (desc.flags & PSRF_HasIdFinalTypeIndex)
      write32le(typeIndexBytes, desc.idFinalTypeIndex);
    else if (desc.flags & PSRF_WarnInvalidFuncId)
      write32le(typeIndexBytes, notTranslatedIndex);
  }

  uint16_t translatedKind =
      kind == uint16_t(llvm::codeview::SymbolKind::S_GPROC32_ID)
          ? uint16_t(llvm::codeview::SymbolKind::S_GPROC32)
          : uint16_t(llvm::codeview::SymbolKind::S_LPROC32);
  write16le(recordBytes + 2, translatedKind);
}

[[maybe_unused]] __global__ void remapAndTranslateSymbolRecordsKernel(
    const PlannedSymbolRecordDescriptor *descriptors, uint32_t descriptorCount,
    const PlannedSymbolTypeRef *typeRefs, const uint32_t *tpiMap,
    uint32_t tpiMapSize, const uint32_t *ipiMap, uint32_t ipiMapSize,
    uint8_t *moduleSymbolStorage, uint32_t moduleSymbolStorageSize,
    DeviceSymbolRemapErrorSummary *errors) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= descriptorCount)
    return;

  const PlannedSymbolRecordDescriptor &desc = descriptors[i];
  if (!(desc.flags & PSRF_GoesInModule))
    return;

  if (desc.outputOffset > moduleSymbolStorageSize ||
      desc.alignedSize > moduleSymbolStorageSize - desc.outputOffset ||
      desc.alignedSize < 4) {
    recordSymbolRemapError(errors, i, desc.kind,
                           DeviceSymbolRemapErrorKind::BadRecordRange,
                           desc.outputOffset);
    return;
  }

  uint8_t *recordBytes = moduleSymbolStorage + desc.outputOffset;
  if (!(desc.flags & PSRF_KnownTypeRefs)) {
    replaceWithSkipRecord(recordBytes, desc.alignedSize);
    translateIdSymbolRecord(recordBytes, desc.alignedSize, desc, errors, i);
    return;
  }

  uint8_t *content = recordBytes + 4;
  uint32_t contentSize = desc.alignedSize - 4;
  for (uint32_t j = 0; j != desc.typeRefCount; ++j) {
    const PlannedSymbolTypeRef &typeRef = typeRefs[desc.typeRefStartIndex + j];
    remapTypeIndexAt(content, contentSize, typeRef, tpiMap, tpiMapSize, ipiMap,
                     ipiMapSize, errors, i, desc.kind);
  }

  translateIdSymbolRecord(recordBytes, desc.alignedSize, desc, errors, i);
}

} // namespace

void lld::coff::executePDBSymbolRemapCUDA(
    ArrayRef<PlannedSymbolRecordDescriptor> descriptors,
    ArrayRef<PlannedSymbolTypeRef> typeRefs,
    PDBSymbolRemapSourceMap sourceMap,
    MutableArrayRef<uint8_t> moduleSymbolStorage) {
  if (descriptors.empty() || moduleSymbolStorage.empty())
    return;

  uint32_t descriptorCount =
      checkedUInt32(descriptors.size(), "symbol descriptor count");
  uint32_t moduleSymbolStorageSize =
      checkedUInt32(moduleSymbolStorage.size(), "module symbol storage size");
  uint32_t tpiMapSize = checkedUInt32(sourceMap.tpiMap.size(),
                                      "TPI source symbol remap size");
  uint32_t ipiMapSize = checkedUInt32(sourceMap.ipiMap.size(),
                                      "IPI source symbol remap size");

  CudaSymbolRemapErrorChecker cuErr;

  thread_local SymbolRemapDeviceScratch scratch;
  cudaStream_t stream = scratch.stream.get(cuErr);
  scratch.descriptors.copyFromAsync(
      descriptors, cuErr, stream,
      "CUDA PDB symbol remap failed to allocate descriptor scratch buffer",
      "CUDA PDB symbol remap failed to copy descriptors");
  scratch.typeRefs.copyFromAsync(
      typeRefs, cuErr, stream,
      "CUDA PDB symbol remap failed to allocate type-ref scratch buffer",
      "CUDA PDB symbol remap failed to copy type refs");
  copyTypeIndexMapAsync(
      scratch.tpiMap, sourceMap.tpiMap, cuErr, stream,
      "CUDA PDB symbol remap failed to allocate TPI map scratch buffer",
      "CUDA PDB symbol remap failed to copy TPI map");
  copyTypeIndexMapAsync(
      scratch.ipiMap, sourceMap.ipiMap, cuErr, stream,
      "CUDA PDB symbol remap failed to allocate IPI map scratch buffer",
      "CUDA PDB symbol remap failed to copy IPI map");
  scratch.moduleSymbolStorage.copyFromAsync(
      ArrayRef<uint8_t>(moduleSymbolStorage.data(), moduleSymbolStorage.size()),
      cuErr, stream,
      "CUDA PDB symbol remap failed to allocate module symbol scratch buffer",
      "CUDA PDB symbol remap failed to copy module symbols");
  scratch.errors.ensureCapacity(
      1, cuErr,
      "CUDA PDB symbol remap failed to allocate error summary scratch buffer");
  cuErr.fatalIfFailed(
      cudaMemsetAsync(scratch.errors.data(), 0,
                      sizeof(DeviceSymbolRemapErrorSummary), stream),
      "CUDA PDB symbol remap failed to clear error summary");

  constexpr uint32_t threads = 256;
  uint32_t blocks = getBlockCount(descriptorCount, threads);
  remapAndTranslateSymbolRecordsKernel<<<blocks, threads, 0, stream>>>(
      scratch.descriptors.data(descriptorCount), descriptorCount,
      scratch.typeRefs.data(typeRefs.size()), scratch.tpiMap.data(tpiMapSize),
      tpiMapSize, scratch.ipiMap.data(ipiMapSize), ipiMapSize,
      scratch.moduleSymbolStorage.data(moduleSymbolStorage.size()),
      moduleSymbolStorageSize, scratch.errors.data());
  cuErr.fatalIfFailed(cudaGetLastError(),
                      "CUDA PDB symbol remap failed to launch kernel");

  DeviceSymbolRemapErrorSummary errors;
  cuErr.fatalIfFailed(
      cudaMemcpyAsync(&errors, scratch.errors.data(), sizeof(errors),
                      cudaMemcpyDeviceToHost, stream),
      "CUDA PDB symbol remap failed to copy error summary");
  cuErr.fatalIfFailed(
      cudaMemcpyAsync(moduleSymbolStorage.data(),
                      scratch.moduleSymbolStorage.data(),
                      moduleSymbolStorage.size(), cudaMemcpyDeviceToHost,
                      stream),
      "CUDA PDB symbol remap failed to copy module symbols");
  cuErr.fatalIfFailed(cudaStreamSynchronize(stream),
                      "CUDA PDB symbol remap failed to synchronize stream");

  uint32_t errorCount = std::min(errors.count, maxSymbolRemapErrors);
  for (uint32_t i = 0; i != errorCount; ++i) {
    if (!isStructuralError(errors.errorKind[i]))
      continue;
    fatal(std::string("CUDA PDB symbol remap failed: ") +
          getSymbolRemapErrorName(errors.errorKind[i]) + " in symbol record " +
          std::to_string(errors.recordIndex[i]) + " kind 0x" +
          llvm::utohexstr(errors.recordKind[i]) + " detail 0x" +
          llvm::utohexstr(errors.detail[i]));
  }
}
