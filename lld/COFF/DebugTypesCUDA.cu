//===- DebugTypesCUDA.cu --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeMerger.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Parallel.h"

#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace lld::coff {

namespace {

// Source-local type indices and final PDB type indices are still 32-bit. The
// flattened map space can exceed 2B entries when many sources are merged, so
// offsets into that combined space use a separate 64-bit type.
using FlatIndex = uint64_t;

struct UniqueGHash {
  uint64_t src = 0;
  FlatIndex group = 0;
};

struct GHashCUDAResult {
  FlatIndex uniqueCount = 0;
  FlatIndex numTypes = 0;
  FlatIndex numItems = 0;
};

constexpr uint32_t firstNonSimpleIndex = 0x1000;
constexpr FlatIndex maxPdbTypeIndexCount =
    FlatIndex(INT32_MAX) - firstNonSimpleIndex;
constexpr uint32_t bitWordBits = sizeof(uintptr_t) * CHAR_BIT;
static_assert(sizeof(GloballyHashedType) == sizeof(uint64_t),
              "GloballyHashedType must be one uint64_t ghash");

__host__ __device__ uint64_t encodeSrc(bool isItem, uint32_t tpiSrcIdx,
                                       uint32_t ghashIdx) {
  assert(tpiSrcIdx < 0x7FFFFFFFU && "source index does not fit");
  return (uint64_t(isItem) << 63U) | (uint64_t(tpiSrcIdx + 1) << 32ULL) |
         ghashIdx;
}

__host__ __device__ uint32_t getTpiSrcIdx(uint64_t src) {
  return (static_cast<uint32_t>(src >> 32U) & 0x7FFFFFFF) - 1;
}

__host__ __device__ uint32_t getGHashIdx(uint64_t src) {
  return static_cast<uint32_t>(src);
}

bool isItemIndex(ArrayRef<uintptr_t> itemBits, uint32_t idx) {
  if (itemBits.empty())
    return false;
  uintptr_t word = itemBits[idx / bitWordBits];
  return (word >> (idx % bitWordBits)) & 1;
}

void fillStagedGHashInputsForRange(ArrayRef<GloballyHashedType> ghashes,
                                   const llvm::BitVector &itemIndexes,
                                   uint32_t tpiSrcIdx, uint32_t ghashBegin,
                                   uint32_t ghashEnd, FlatIndex entryOffset,
                                   uint64_t *hashes, uint64_t *srcs) {
  if (ghashBegin == ghashEnd)
    return;

  uint32_t count = ghashEnd - ghashBegin;
  std::memcpy(hashes + entryOffset, ghashes.data() + ghashBegin,
              size_t(count) * sizeof(uint64_t));

  bool allItems = itemIndexes.all();
  bool noItems = itemIndexes.none();
  ArrayRef<uintptr_t> itemBits = itemIndexes.getData();
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t ghashIdx = ghashBegin + i;
    bool isItem = allItems || (!noItems && isItemIndex(itemBits, ghashIdx));
    srcs[entryOffset + i] = encodeSrc(isItem, tpiSrcIdx, ghashIdx);
  }
}

struct ByGHashThenSrc {
  template <typename A, typename B>
  __host__ __device__ bool operator()(const A &a, const B &b) const {
    uint64_t aHash = thrust::get<0>(a);
    uint64_t bHash = thrust::get<0>(b);
    if (aHash != bHash)
      return aHash < bHash;
    return thrust::get<1>(a) < thrust::get<1>(b);
  }
};

__global__ void finalizeGHashGroupsAndScatter(const uint64_t *hashes,
                                              const uint64_t *srcs,
                                              FlatIndex *groups,
                                              UniqueGHash *uniqueEntries,
                                              FlatIndex count) {
  FlatIndex i = FlatIndex(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  bool isUnique = i == 0 || hashes[i] != hashes[i - 1];
  FlatIndex group = groups[i] - 1;
  groups[i] = group;
  if (!isUnique)
    return;
  UniqueGHash entry;
  entry.src = srcs[i];
  entry.group = group;
  uniqueEntries[group] = entry;
}

__global__ void assignDestinationIndices(const UniqueGHash *uniqueEntries,
                                         FlatIndex uniqueCount,
                                         FlatIndex numTypes,
                                         uint32_t *groupToTypeIndex) {
  FlatIndex i = FlatIndex(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= uniqueCount)
    return;
  FlatIndex arrayIndex = i < numTypes ? i : i - numTypes;
  uint32_t pdbIndex = firstNonSimpleIndex + static_cast<uint32_t>(arrayIndex);
  groupToTypeIndex[uniqueEntries[i].group] = pdbIndex;
}

__global__ void fillFlatMap(const uint64_t *srcs, const FlatIndex *groups,
                            const uint32_t *groupToTypeIndex,
                            const FlatIndex *mapOffsets, FlatIndex count,
                            uint32_t *flatMap) {
  FlatIndex i = FlatIndex(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  uint64_t src = srcs[i];
  FlatIndex mapIndex = mapOffsets[getTpiSrcIdx(src)] + getGHashIdx(src);
  flatMap[mapIndex] = groupToTypeIndex[groups[i]];
}

class CudaErrorChecker {
public:
  explicit CudaErrorChecker(COFFLinkerContext &ctx) : ctx(ctx) {}

  bool check(cudaError_t errorCode) {
    if (errorCode == cudaSuccess)
      return true;
    std::snprintf(error.data(), error.size(), "%s",
                  cudaGetErrorString(errorCode));
    return false;
  }

  void fatal(cudaError_t errorCode, const char *context) {
    if (check(errorCode))
      return;
    Fatal(ctx) << context << ": " << message();
  }

  const char *message() const { return error.data(); }

private:
  COFFLinkerContext &ctx;
  std::array<char, 256> error = {};
};

uint32_t getBlockCount(COFFLinkerContext &ctx, FlatIndex count,
                       uint32_t threads) {
  FlatIndex blockCount = (count + threads - 1) / threads;
  if (blockCount > std::numeric_limits<uint32_t>::max())
    Fatal(ctx) << "-lldcudaghash failed: too many CUDA thread blocks";
  return static_cast<uint32_t>(blockCount);
}

void mergeGHashesWithCUDA(COFFLinkerContext &ctx, ArrayRef<uint64_t> hashes,
                          ArrayRef<uint64_t> srcs,
                          const FlatIndex *mapOffsets, FlatIndex mapOffsetCount,
                          FlatIndex mapCount, uint32_t notTranslated,
                          ArrayRef<TpiSource *> tpiSources,
                          std::vector<uint64_t> *uniqueSrcs,
                          GHashCUDAResult *result, CudaErrorChecker &cuErr) {
  static_assert(sizeof(TypeIndex) == sizeof(uint32_t),
                "TypeIndex must stay layout-compatible with a PDB index");
  static_assert(std::is_trivially_copyable<TypeIndex>::value,
                "TypeIndex must be safe to copy from CUDA output bytes");

  *result = GHashCUDAResult();
  assert(hashes.size() == srcs.size());
  FlatIndex entryCount = hashes.size();
  if (entryCount == 0)
    return;

  // The ragged source/type-index grid is represented as one flat destination
  // map. Initialize the whole flat space on the GPU so omitted records keep the
  // NotTranslated sentinel without requiring a host-side staging vector.
  thrust::device_vector<uint32_t> deviceMap(mapCount);
  thrust::fill(thrust::device, deviceMap.begin(), deviceMap.end(),
               notTranslated);

  thrust::device_vector<uint64_t> deviceHashes(hashes.begin(), hashes.end());
  thrust::device_vector<uint64_t> deviceSrcs(srcs.begin(), srcs.end());
  thrust::device_vector<FlatIndex> deviceMapOffsets(
      mapOffsets, mapOffsets + mapOffsetCount);

  // The input hash/source-position arrays are staged as two flat host arrays.
  // Thrust copies them to device vectors before the first sort. Host setup has
  // already omitted LF_ENDPRECOMP records, so the GPU path starts with dense
  // arrays and avoids per-source host-memory registration overhead.
  uint32_t threads = 256;

  // Sort the parallel ghash/source-position arrays together. The source value
  // carries the original TPI source row, original type-index column, and
  // TPI-vs-IPI partition bit, so the sort keeps enough information to recover
  // both the retained record and its flatMap slot.
  auto entriesBegin = thrust::make_zip_iterator(
      thrust::make_tuple(deviceHashes.begin(), deviceSrcs.begin()));
  auto entriesEnd = thrust::make_zip_iterator(
      thrust::make_tuple(deviceHashes.end(), deviceSrcs.end()));
  thrust::sort(thrust::device, entriesBegin, entriesEnd, ByGHashThenSrc());

  // Scan run starts directly from the sorted hashes. This avoids materializing
  // a separate flags array. The inclusive scan writes group + 1, so duplicate
  // cells already share their representative's value.
  thrust::device_vector<FlatIndex> groups(entryCount);
  uint32_t blocks = getBlockCount(ctx, entryCount, threads);
  const uint64_t *deviceHashData =
      thrust::raw_pointer_cast(deviceHashes.data());
  auto indices = thrust::make_counting_iterator<FlatIndex>(0);
  auto isUniqueGHash = [deviceHashData] __host__ __device__(FlatIndex i)
      -> FlatIndex {
        return i == 0 || deviceHashData[i] != deviceHashData[i - 1];
      };
  auto runStarts = thrust::make_transform_iterator(indices, isUniqueGHash);
  thrust::inclusive_scan(thrust::device, runStarts, runStarts + entryCount,
                         groups.begin());

  FlatIndex uniqueCount = 0;
  cuErr.fatal(
      cudaMemcpy(&uniqueCount,
                 thrust::raw_pointer_cast(groups.data()) + entryCount - 1,
                 sizeof(uniqueCount), cudaMemcpyDeviceToHost),
      "-lldcudaghash failed to copy unique ghash count");

  // Convert group + 1 to the final zero-based group id in place and scatter
  // only representative source positions into the compact unique table. Each
  // representative carries its original group so destination indices can be
  // mapped back to all duplicates later.
  thrust::device_vector<UniqueGHash> uniqueEntries(uniqueCount);
  finalizeGHashGroupsAndScatter<<<blocks, threads>>>(
      thrust::raw_pointer_cast(deviceHashes.data()),
      thrust::raw_pointer_cast(deviceSrcs.data()),
      thrust::raw_pointer_cast(groups.data()),
      thrust::raw_pointer_cast(uniqueEntries.data()), entryCount);
  cuErr.fatal(cudaGetLastError(),
              "-lldcudaghash failed to launch ghash group finalization kernel");

  // Sort unique representatives by original source position. This reproduces
  // the CPU path's final deterministic PDB order: all TPI records first, then
  // IPI records, each ordered by input source and source type index.
  auto bySrc = [] __host__ __device__(const UniqueGHash &a,
                                      const UniqueGHash &b) {
    return a.src < b.src;
  };
  thrust::sort(thrust::device, uniqueEntries.begin(), uniqueEntries.end(),
               bySrc);

  UniqueGHash firstItem;
  firstItem.src = 1ULL << 63U;
  // Find the boundary between destination TPI and IPI streams. IPI indices use
  // a separate destination index space, so numbering restarts at the boundary.
  FlatIndex numTypes =
      thrust::lower_bound(thrust::device, uniqueEntries.begin(),
                          uniqueEntries.end(), firstItem, bySrc) -
      uniqueEntries.begin();
  FlatIndex numItems = uniqueCount - numTypes;
  if (numTypes > maxPdbTypeIndexCount || numItems > maxPdbTypeIndexCount)
    Fatal(ctx) << "-lldcudaghash failed: too many unique CUDA ghash records";

  thrust::device_vector<uint32_t> groupToTypeIndex(uniqueCount);
  uint32_t uniqueBlocks = getBlockCount(ctx, uniqueCount, threads);
  // Build the compact group -> destination TypeIndex table. This is the only
  // place that assigns final PDB TPI/IPI indices.
  assignDestinationIndices<<<uniqueBlocks, threads>>>(
      thrust::raw_pointer_cast(uniqueEntries.data()), uniqueCount, numTypes,
      thrust::raw_pointer_cast(groupToTypeIndex.data()));
  cuErr.fatal(cudaGetLastError(),
              "-lldcudaghash failed to launch destination index kernel");

  // Fill every original flatMap slot. Each worker uses the sorted source
  // position to reconstruct the flattened map index, then writes the
  // destination TypeIndex for that entry's deduplicated ghash group.
  fillFlatMap<<<blocks, threads>>>(
      thrust::raw_pointer_cast(deviceSrcs.data()),
      thrust::raw_pointer_cast(groups.data()),
      thrust::raw_pointer_cast(groupToTypeIndex.data()),
      thrust::raw_pointer_cast(deviceMapOffsets.data()), entryCount,
      thrust::raw_pointer_cast(deviceMap.data()));
  cuErr.fatal(cudaGetLastError(),
              "-lldcudaghash failed to launch flat map fill kernel");
  cuErr.fatal(cudaDeviceSynchronize(),
              "-lldcudaghash failed while synchronizing kernels");

  std::vector<UniqueGHash> uniqueEntriesHost(uniqueCount);
  cuErr.fatal(cudaMemcpy(uniqueEntriesHost.data(),
                         thrust::raw_pointer_cast(uniqueEntries.data()),
                         uniqueCount * sizeof(UniqueGHash),
                         cudaMemcpyDeviceToHost),
              "-lldcudaghash failed to copy unique ghash entries");
  uniqueSrcs->resize(uniqueCount);
  for (FlatIndex i = 0; i < uniqueCount; ++i)
    (*uniqueSrcs)[i] = uniqueEntriesHost[i].src;

  const uint32_t *deviceMapData = thrust::raw_pointer_cast(deviceMap.data());
  for (TpiSource *source : tpiSources) {
    FlatIndex mapOffset = mapOffsets[source->tpiSrcIdx];
    size_t byteCount = source->indexMapStorage.size() * sizeof(TypeIndex);
    if (byteCount == 0)
      continue;
    cuErr.fatal(cudaMemcpy(source->indexMapStorage.data(),
                           deviceMapData + mapOffset, byteCount,
                           cudaMemcpyDeviceToHost),
                "-lldcudaghash failed to copy type index map");
  }

  result->uniqueCount = uniqueCount;
  result->numTypes = numTypes;
  result->numItems = numItems;
}

} // namespace

bool TypeMerger::mergeTypesWithCUDA() {
  constexpr uint32_t notTranslated =
      static_cast<uint32_t>(llvm::codeview::SimpleTypeKind::NotTranslated);

  // Flatten the ragged TpiSource x type-index grid into a flat, one-dimensional
  // index space. mapOffsets stores each source row's base offset. The hash
  // array is parallel to the source-position array, and each source position
  // carries the original row, original column, and TPI-vs-IPI bit needed after
  // sorting.
  FlatIndex totalMapSize = 0;
  FlatIndex entryCount = 0;
  llvm::SmallVector<FlatIndex, 0> mapOffsets;
  llvm::SmallVector<FlatIndex, 0> entryOffsets;
  mapOffsets.reserve(ctx.tpiSourceList.size() + 1);
  entryOffsets.reserve(ctx.tpiSourceList.size());
  mapOffsets.push_back(0);
  for (TpiSource *source : ctx.tpiSourceList) {
    entryOffsets.push_back(entryCount);
    if (source->ghashes.size() > maxPdbTypeIndexCount)
      Fatal(ctx) << "too many ghashes in source";
    if (source->ghashes.size() >
        std::numeric_limits<FlatIndex>::max() - totalMapSize)
      Fatal(ctx) << "too many ghashes to merge with CUDA";
    totalMapSize += source->ghashes.size();
    entryCount += source->ghashes.size() -
                  (source->endPrecompIdx < source->ghashes.size());
    mapOffsets.push_back(totalMapSize);
  }

  int device = 0;
  CudaErrorChecker cuErr(ctx);
  auto fallbackToCPU = [&](const char *reason) {
    Warn(ctx) << "-lldcudaghash setup failed, falling back to CPU ghash "
                 "merging: "
              << reason;
    return false;
  };
  if (!cuErr.check(cudaGetDevice(&device)))
    return fallbackToCPU(cuErr.message());

  std::vector<uint64_t> hashes(entryCount);
  std::vector<uint64_t> srcs(entryCount);
  llvm::parallelFor(0, ctx.tpiSourceList.size(), [&](size_t i) {
    TpiSource *source = ctx.tpiSourceList[i];
    source->indexMapStorage.resize(source->ghashes.size());

    uint32_t sourceSize = static_cast<uint32_t>(source->ghashes.size());
    uint32_t tpiSrcIdx = source->tpiSrcIdx;
    FlatIndex sourceEntryOffset = entryOffsets[i];
    if (source->endPrecompIdx < sourceSize) {
      fillStagedGHashInputsForRange(
          source->ghashes, source->isItemIndex, tpiSrcIdx, 0,
          source->endPrecompIdx, sourceEntryOffset, hashes.data(),
          srcs.data());
      fillStagedGHashInputsForRange(
          source->ghashes, source->isItemIndex, tpiSrcIdx,
          source->endPrecompIdx + 1, sourceSize,
          sourceEntryOffset + source->endPrecompIdx, hashes.data(),
          srcs.data());
    } else {
      fillStagedGHashInputsForRange(source->ghashes, source->isItemIndex,
                                    tpiSrcIdx, 0, sourceSize,
                                    sourceEntryOffset, hashes.data(),
                                    srcs.data());
    }
  });

  GHashCUDAResult result;
  std::vector<uint64_t> uniqueSrcs;
  mergeGHashesWithCUDA(ctx, hashes, srcs, mapOffsets.data(), mapOffsets.size(),
                       totalMapSize, notTranslated, ctx.tpiSourceList,
                       &uniqueSrcs, &result, cuErr);
  hashes.clear();
  srcs.clear();

  // The GPU returns unique representatives sorted in final merge order. Retain
  // their source type indices so mergeUniqueTypeRecords emits the same records
  // that the CPU ghash path would have emitted.
  for (uint64_t src : uniqueSrcs) {
    TpiSource *source = ctx.tpiSourceList[getTpiSrcIdx(src)];
    source->uniqueTypes.push_back(getGHashIdx(src));
  }

  Log(ctx) << "CUDA ghash record count: " << result.uniqueCount << " / input "
           << totalMapSize;
  Log(ctx) << "Tpi record count: " << result.numTypes;
  Log(ctx) << "Ipi record count: " << result.numItems;

  for (TpiSource *source : dependencySources)
    source->remapTpiWithGHashes();
  // Remap object type records in parallel now that every source TPI/IPI index
  // has a destination PDB type or item index.
  llvm::parallelForEach(
      objectSources, [&](TpiSource *source) { source->remapTpiWithGHashes(); });

  for (TpiSource *source : ctx.tpiSourceList) {
    funcIdToType.insert_range(source->funcIdToType);
    source->funcIdToType.clear();
  }

  clearGHashes();
  return true;
}

} // namespace lld::coff
