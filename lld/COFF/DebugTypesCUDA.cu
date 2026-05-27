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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <type_traits>
#include <vector>

namespace lld::coff {

namespace {

struct UniqueGHash {
  uint64_t src = 0;
  uint32_t group = 0;
};

struct GHashCUDAResult {
  uint32_t uniqueCount = 0;
  uint32_t numTypes = 0;
  uint32_t numItems = 0;
};

constexpr uint32_t firstNonSimpleIndex = 0x1000;

uint64_t encodeSrc(bool isItem, uint32_t tpiSrcIdx, uint32_t ghashIdx) {
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

uint64_t getGHash(GloballyHashedType ghash) {
  uint64_t hash = 0;
  memcpy(&hash, ghash.Hash.data(), sizeof(hash));
  return hash;
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

struct BySrc {
  __host__ __device__ bool operator()(const UniqueGHash &a,
                                      const UniqueGHash &b) const {
    // The source encoding sorts TPI records before IPI records, then by source
    // and source type index. This is the final deterministic PDB index order.
    return a.src < b.src;
  }
};

__global__ void markUniqueGHashes(const uint64_t *hashes, uint32_t count,
                                  uint32_t *flags) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  flags[i] = i == 0 || hashes[i] != hashes[i - 1];
}

__global__ void scatterUniqueGHashes(const uint64_t *srcs,
                                     const uint32_t *flags,
                                     const uint32_t *groups,
                                     UniqueGHash *uniqueEntries,
                                     uint32_t count) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || !flags[i])
    return;
  UniqueGHash entry;
  entry.src = srcs[i];
  entry.group = groups[i];
  uniqueEntries[groups[i]] = entry;
}

__global__ void adjustDuplicateGroups(const uint32_t *flags, uint32_t *groups,
                                      uint32_t count) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || flags[i])
    return;
  --groups[i];
}

__global__ void assignDestinationIndices(const UniqueGHash *uniqueEntries,
                                         uint32_t uniqueCount,
                                         uint32_t numTypes,
                                         uint32_t *groupToTypeIndex) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= uniqueCount)
    return;
  uint32_t arrayIndex = i < numTypes ? i : i - numTypes;
  uint32_t pdbIndex = firstNonSimpleIndex + arrayIndex;
  groupToTypeIndex[uniqueEntries[i].group] = pdbIndex;
}

__global__ void fillFlatMap(const uint64_t *srcs, const uint32_t *groups,
                            const uint32_t *groupToTypeIndex,
                            const uint32_t *mapOffsets, uint32_t count,
                            uint32_t *flatMap) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  uint64_t src = srcs[i];
  uint32_t mapIndex = mapOffsets[getTpiSrcIdx(src)] + getGHashIdx(src);
  flatMap[mapIndex] = groupToTypeIndex[groups[i]];
}

void setError(char *error, size_t errorLen, const char *message) {
  if (errorLen == 0)
    return;
  std::snprintf(error, errorLen, "%s", message);
}

bool checkCuda(cudaError_t errorCode, char *error, size_t errorLen) {
  if (errorCode == cudaSuccess)
    return true;
  setError(error, errorLen, cudaGetErrorString(errorCode));
  return false;
}

bool mergeGHashesWithCUDA(uint64_t *hashes, uint64_t *srcs, uint32_t entryCount,
                          const uint32_t *mapOffsets, uint32_t mapOffsetCount,
                          uint32_t mapCount, uint32_t notTranslated,
                          ArrayRef<TpiSource *> tpiSources,
                          GHashCUDAResult *result, char *error,
                          size_t errorLen) {
  static_assert(sizeof(TypeIndex) == sizeof(uint32_t),
                "TypeIndex must stay layout-compatible with a PDB index");
  static_assert(std::is_trivially_copyable<TypeIndex>::value,
                "TypeIndex must be safe to copy from CUDA output bytes");

  *result = GHashCUDAResult();

  thrust::device_vector<uint32_t> deviceMap(mapCount);
  // The ragged source/type-index grid is represented as one flat destination
  // map. Initialize the whole flat space on the GPU so omitted records keep the
  // NotTranslated sentinel without requiring a host-side staging vector.
  thrust::fill(thrust::device, deviceMap.begin(), deviceMap.end(),
               notTranslated);

  auto copyDeviceMapToSources = [&]() {
    const uint32_t *deviceMapData = thrust::raw_pointer_cast(deviceMap.data());
    for (TpiSource *source : tpiSources) {
      uint32_t mapOffset = mapOffsets[source->tpiSrcIdx];
      size_t byteCount = source->indexMapStorage.size() * sizeof(TypeIndex);
      if (byteCount == 0)
        continue;
      if (!checkCuda(cudaMemcpy(source->indexMapStorage.data(),
                                deviceMapData + mapOffset, byteCount,
                                cudaMemcpyDeviceToHost),
                     error, errorLen))
        return false;
    }
    return true;
  };

  if (entryCount == 0)
    return copyDeviceMapToSources();

  thrust::device_vector<uint64_t> deviceHashes(hashes, hashes + entryCount);
  thrust::device_vector<uint64_t> deviceSrcs(srcs, srcs + entryCount);
  thrust::device_vector<uint32_t> deviceMapOffsets(mapOffsets,
                                                   mapOffsets + mapOffsetCount);

  // Sort the parallel ghash/source-position arrays together. The source value
  // carries the original TPI source row, original type-index column, and
  // TPI-vs-IPI partition bit, so the sort keeps enough information to recover
  // both the retained record and its flatMap slot.
  auto entriesBegin = thrust::make_zip_iterator(
      thrust::make_tuple(deviceHashes.begin(), deviceSrcs.begin()));
  auto entriesEnd = thrust::make_zip_iterator(
      thrust::make_tuple(deviceHashes.end(), deviceSrcs.end()));
  thrust::sort(thrust::device, entriesBegin, entriesEnd, ByGHashThenSrc());

  thrust::device_vector<uint32_t> flags(entryCount);
  thrust::device_vector<uint32_t> groups(entryCount);
  uint32_t threads = 256;
  uint32_t blocks = (entryCount + threads - 1) / threads;
  // Mark the first cell in every sorted run of equal ghashes. That cell is the
  // deterministic representative because the zip-sort used source priority as
  // its tie breaker.
  markUniqueGHashes<<<blocks, threads>>>(
      thrust::raw_pointer_cast(deviceHashes.data()), entryCount,
      thrust::raw_pointer_cast(flags.data()));
  if (!checkCuda(cudaGetLastError(), error, errorLen))
    return false;

  // Prefix-sum the run starts. The resulting group id is the compact unique
  // ghash ordinal used by duplicates to find their destination index.
  thrust::exclusive_scan(thrust::device, flags.begin(), flags.end(),
                         groups.begin());

  uint32_t lastFlag = 0;
  uint32_t lastGroup = 0;
  if (!checkCuda(
          cudaMemcpy(&lastFlag,
                     thrust::raw_pointer_cast(flags.data()) + entryCount - 1,
                     sizeof(lastFlag), cudaMemcpyDeviceToHost),
          error, errorLen))
    return false;
  if (!checkCuda(
          cudaMemcpy(&lastGroup,
                     thrust::raw_pointer_cast(groups.data()) + entryCount - 1,
                     sizeof(lastGroup), cudaMemcpyDeviceToHost),
          error, errorLen))
    return false;
  uint32_t uniqueCount = lastGroup + lastFlag;

  // The exclusive scan already names representative cells correctly. Shift
  // duplicate cells back to the preceding representative's group before map
  // fill uses the group table.
  adjustDuplicateGroups<<<blocks, threads>>>(
      thrust::raw_pointer_cast(flags.data()),
      thrust::raw_pointer_cast(groups.data()), entryCount);
  if (!checkCuda(cudaGetLastError(), error, errorLen))
    return false;

  thrust::device_vector<UniqueGHash> uniqueEntries(uniqueCount);
  // Scatter only representative source positions into the compact unique table,
  // carrying each representative's original group so destination indices can be
  // mapped back to all duplicates later.
  scatterUniqueGHashes<<<blocks, threads>>>(
      thrust::raw_pointer_cast(deviceSrcs.data()),
      thrust::raw_pointer_cast(flags.data()),
      thrust::raw_pointer_cast(groups.data()),
      thrust::raw_pointer_cast(uniqueEntries.data()), entryCount);
  if (!checkCuda(cudaGetLastError(), error, errorLen))
    return false;

  // Sort unique representatives by original source position. This reproduces
  // the CPU path's final deterministic PDB order: all TPI records first, then
  // IPI records, each ordered by input source and source type index.
  thrust::sort(thrust::device, uniqueEntries.begin(), uniqueEntries.end(),
               BySrc());

  UniqueGHash firstItem;
  firstItem.src = 1ULL << 63U;
  // Find the boundary between destination TPI and IPI streams. IPI indices use
  // a separate destination index space, so numbering restarts at the boundary.
  uint32_t numTypes =
      thrust::lower_bound(thrust::device, uniqueEntries.begin(),
                          uniqueEntries.end(), firstItem, BySrc()) -
      uniqueEntries.begin();

  thrust::device_vector<uint32_t> groupToTypeIndex(uniqueCount);
  uint32_t uniqueBlocks = (uniqueCount + threads - 1) / threads;
  // Build the compact group -> destination TypeIndex table. This is the only
  // place that assigns final PDB TPI/IPI indices.
  assignDestinationIndices<<<uniqueBlocks, threads>>>(
      thrust::raw_pointer_cast(uniqueEntries.data()), uniqueCount, numTypes,
      thrust::raw_pointer_cast(groupToTypeIndex.data()));
  if (!checkCuda(cudaGetLastError(), error, errorLen))
    return false;

  // Fill every original flatMap slot. Each worker uses the sorted source
  // position to reconstruct the flattened map index, then writes the
  // destination TypeIndex for that entry's deduplicated ghash group.
  fillFlatMap<<<blocks, threads>>>(
      thrust::raw_pointer_cast(deviceSrcs.data()),
      thrust::raw_pointer_cast(groups.data()),
      thrust::raw_pointer_cast(groupToTypeIndex.data()),
      thrust::raw_pointer_cast(deviceMapOffsets.data()), entryCount,
      thrust::raw_pointer_cast(deviceMap.data()));
  if (!checkCuda(cudaGetLastError(), error, errorLen))
    return false;
  if (!checkCuda(cudaDeviceSynchronize(), error, errorLen))
    return false;

  std::vector<UniqueGHash> uniqueEntriesHost(uniqueCount);
  if (!checkCuda(cudaMemcpy(uniqueEntriesHost.data(),
                            thrust::raw_pointer_cast(uniqueEntries.data()),
                            uniqueCount * sizeof(UniqueGHash),
                            cudaMemcpyDeviceToHost),
                 error, errorLen))
    return false;
  for (uint32_t i = 0; i < uniqueCount; ++i) {
    hashes[i] = 0;
    srcs[i] = uniqueEntriesHost[i].src;
  }

  if (!copyDeviceMapToSources())
    return false;

  result->uniqueCount = uniqueCount;
  result->numTypes = numTypes;
  result->numItems = uniqueCount - numTypes;
  return true;
}

} // namespace

void TypeMerger::mergeTypesWithCUDA() {
  constexpr uint32_t notTranslated =
      static_cast<uint32_t>(llvm::codeview::SimpleTypeKind::NotTranslated);

  // Flatten the ragged TpiSource x type-index grid into one flat map index
  // space. mapOffsets stores each source row's base offset. The hash array is
  // parallel to the source-position array, and each source position carries the
  // original row, original column, and TPI-vs-IPI bit needed after sorting.
  size_t totalMapSize = 0;
  llvm::SmallVector<uint32_t, 0> mapOffsets;
  mapOffsets.reserve(ctx.tpiSourceList.size() + 1);
  mapOffsets.push_back(0);
  for (TpiSource *source : ctx.tpiSourceList) {
    if (source->ghashes.size() >
        size_t(UINT32_MAX) - TypeIndex::FirstNonSimpleIndex)
      Fatal(ctx) << "too many ghashes in source";
    totalMapSize += source->ghashes.size();
    if (totalMapSize > size_t(UINT32_MAX) - TypeIndex::FirstNonSimpleIndex)
      Fatal(ctx) << "too many ghashes to merge with CUDA";
    mapOffsets.push_back(static_cast<uint32_t>(totalMapSize));
  }

  std::vector<uint64_t> hashes;
  std::vector<uint64_t> srcs;
  hashes.reserve(totalMapSize);
  srcs.reserve(totalMapSize);

  for (TpiSource *source : ctx.tpiSourceList) {
    source->indexMapStorage.resize(source->ghashes.size());
    uint32_t tpiSrcIdx = source->tpiSrcIdx;
    for (uint32_t i = 0, e = source->ghashes.size(); i < e; ++i) {
      if (source->shouldOmitFromPdb(i))
        continue;

      hashes.push_back(getGHash(source->ghashes[i]));
      srcs.push_back(encodeSrc(source->isItemIndex.test(i), tpiSrcIdx, i));
    }
  }

  GHashCUDAResult result;
  std::array<char, 256> error = {};
  if (!mergeGHashesWithCUDA(
          hashes.data(), srcs.data(), static_cast<uint32_t>(srcs.size()),
          mapOffsets.data(), static_cast<uint32_t>(mapOffsets.size()),
          static_cast<uint32_t>(totalMapSize), notTranslated, ctx.tpiSourceList,
          &result, error.data(), error.size()))
    Fatal(ctx) << "-lldcudaghash failed: " << error.data();

  // The GPU returns unique representatives sorted in final merge order. Retain
  // their source type indices so mergeUniqueTypeRecords emits the same records
  // that the CPU ghash path would have emitted.
  srcs.resize(result.uniqueCount);
  for (uint64_t src : srcs) {
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
}

} // namespace lld::coff
