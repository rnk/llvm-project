//===- DebugTypesCUDA.cu --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeMerger.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/DebugInfo/PDB/Native/TpiStreamBuilder.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Parallel.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
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

__host__ __device__ bool isItemSrc(uint64_t src) { return (src >> 63U) != 0; }

enum class SourceItemMode : uint8_t { LeafKinds, AllTypes, AllItems };

SourceItemMode getSourceItemMode(const TpiSource &source) {
  if (source.kind == TpiSource::PDB)
    return SourceItemMode::AllTypes;
  if (source.kind == TpiSource::PDBIpi)
    return SourceItemMode::AllItems;
  return SourceItemMode::LeafKinds;
}

struct SourceDescriptor {
  uint32_t tpiSrcIdx = 0;
  uint32_t ghashCount = 0;
  uint32_t endPrecompIdx = ~0U;
  uint32_t leafKindOffset = 0;
  SourceItemMode itemMode = SourceItemMode::LeafKinds;
};

__host__ __device__ bool isIdLeafKind(uint16_t kind) {
  return kind == llvm::codeview::LF_FUNC_ID ||
         kind == llvm::codeview::LF_MFUNC_ID ||
         kind == llvm::codeview::LF_STRING_ID ||
         kind == llvm::codeview::LF_SUBSTR_LIST ||
         kind == llvm::codeview::LF_BUILDINFO ||
         kind == llvm::codeview::LF_UDT_SRC_LINE ||
         kind == llvm::codeview::LF_UDT_MOD_SRC_LINE;
}

__global__ void buildSourceCells(const SourceDescriptor *sources,
                                 const FlatIndex *entryOffsets,
                                 const uint16_t *leafKinds, uint64_t *srcs,
                                 uint32_t sourceCount) {
  uint32_t sourceIdx = blockIdx.x;
  if (sourceIdx >= sourceCount)
    return;

  SourceDescriptor source = sources[sourceIdx];
  bool skipEndPrecomp = source.endPrecompIdx < source.ghashCount;
  uint32_t entryCount = source.ghashCount - uint32_t(skipEndPrecomp);
  for (uint32_t entryIdx = threadIdx.x; entryIdx < entryCount;
       entryIdx += blockDim.x) {
    uint32_t ghashIdx =
        entryIdx + uint32_t(skipEndPrecomp && entryIdx >= source.endPrecompIdx);
    bool isItem = false;
    if (source.itemMode == SourceItemMode::AllItems)
      isItem = true;
    else if (source.itemMode == SourceItemMode::LeafKinds)
      isItem = isIdLeafKind(leafKinds[source.leafKindOffset + ghashIdx]);
    srcs[entryOffsets[sourceIdx] + entryIdx] =
        encodeSrc(isItem, source.tpiSrcIdx, ghashIdx);
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

__global__ void assignDestinationIndicesAndExtractSrcs(
    const UniqueGHash *uniqueEntries, FlatIndex uniqueCount, FlatIndex numTypes,
    uint32_t *groupToTypeIndex, uint64_t *uniqueSrcs) {
  FlatIndex i = FlatIndex(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= uniqueCount)
    return;
  UniqueGHash entry = uniqueEntries[i];
  FlatIndex arrayIndex = i < numTypes ? i : i - numTypes;
  uint32_t pdbIndex = firstNonSimpleIndex + static_cast<uint32_t>(arrayIndex);
  groupToTypeIndex[entry.group] = pdbIndex;
  uniqueSrcs[i] = entry.src;
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

constexpr uint32_t decoratedItemIdMask = 0x80000000U;
constexpr uint32_t notTranslatedIndex =
    static_cast<uint32_t>(llvm::codeview::SimpleTypeKind::NotTranslated);
constexpr uint32_t maxRemapErrors = 5;
constexpr uint32_t noTypeIndexOffset = ~0U;

enum class DeviceRefKind : uint8_t { TypeRef, IndexRef };

struct RemapErrorSummary {
  uint32_t count = 0;
  uint32_t recordIndex[maxRemapErrors] = {};
  uint16_t recordKind[maxRemapErrors] = {};
  uint32_t detail[maxRemapErrors] = {};
};

struct FuncIdToTypeEntry {
  uint32_t funcId = 0;
  uint32_t funcType = 0;
};

enum class RemapMapKind : uint8_t { FlatMap, ExtraMap, PrefixFlatMap };

// Device view of a source type-index map used while remapping selected records.
//
// The common case is a direct slice of the GPU-built flat map:
// - regular object sources own one flat map slice, used for both TPI and IPI
//   references;
// - type-server PDB TPI and companion IPI sources own separate flat slices, and
//   either stream can point at the other's slice for cross-stream references;
// - type-server object users do not contribute selected type records, but their
//   symbol maps are aliases of those type-server slices;
// - PCH objects own one direct flat map slice;
// - PCH object users have a composed host map: PCH prefix followed by the user's
//   own records. PrefixFlatMap represents that composition on the device without
//   copying the full composed map into extraMaps.
//
// ExtraMap is the conservative fallback for maps that are neither direct flat
// slices nor recognized PCH-prefix compositions.
struct DeviceRemapMapDescriptor {
  // FlatMap: base index into flatMap. ExtraMap: base index into extraMaps.
  // PrefixFlatMap: base index of the prefix slice in flatMap.
  uint64_t mapOffset = 0;

  // PrefixFlatMap only: base index of the suffix slice in flatMap. The suffix
  // starts at source index prefixSize. Ignored by FlatMap and ExtraMap.
  uint64_t suffixMapOffset = 0;

  // Total number of source map entries visible through this descriptor. This is
  // checked before lookup and includes both prefix and suffix for PrefixFlatMap.
  uint32_t mapSize = 0;

  // PrefixFlatMap only: number of leading entries read from mapOffset before
  // lookups switch to suffixMapOffset. Zero for the other map kinds.
  uint32_t prefixSize = 0;

  // Selects which backing storage and addressing rule readRemapMapValue uses.
  RemapMapKind mapKind = RemapMapKind::FlatMap;
};

struct RemapSourceDescriptor {
  DeviceRemapMapDescriptor tpiMap;
  DeviceRemapMapDescriptor ipiMap;
  uint32_t sourceTypeIndexBegin = 0;
};

struct SelectedRecordDescriptor {
  uint32_t byteOffset = 0;
  uint32_t size = 0;
  uint32_t ghashIndex = 0;
  uint32_t sourceIndex = 0;
};

__host__ __device__ uint16_t read16le(const uint8_t *p) {
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

__host__ __device__ uint32_t read32le(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

__host__ __device__ void write16le(uint8_t *p, uint16_t v) {
  p[0] = uint8_t(v);
  p[1] = uint8_t(v >> 8);
}

__host__ __device__ void write32le(uint8_t *p, uint32_t v) {
  p[0] = uint8_t(v);
  p[1] = uint8_t(v >> 8);
  p[2] = uint8_t(v >> 16);
  p[3] = uint8_t(v >> 24);
}

__device__ void recordRemapError(RemapErrorSummary *errors,
                                 uint32_t recordIndex, uint16_t recordKind,
                                 uint32_t detail) {
  uint32_t slot = atomicAdd(&errors->count, 1U);
  if (slot >= maxRemapErrors)
    return;
  errors->recordIndex[slot] = recordIndex;
  errors->recordKind[slot] = recordKind;
  errors->detail[slot] = detail;
}

__device__ uint32_t readRemapMapValue(const DeviceRemapMapDescriptor &mapDesc,
                                      uint32_t arrayIndex,
                                      const uint32_t *flatMap,
                                      const uint32_t *extraMaps) {
  switch (mapDesc.mapKind) {
  case RemapMapKind::FlatMap:
    return flatMap[mapDesc.mapOffset + arrayIndex];
  case RemapMapKind::ExtraMap:
    return extraMaps[mapDesc.mapOffset + arrayIndex];
  case RemapMapKind::PrefixFlatMap:
    if (arrayIndex < mapDesc.prefixSize)
      return flatMap[mapDesc.mapOffset + arrayIndex];
    return flatMap[mapDesc.suffixMapOffset + arrayIndex - mapDesc.prefixSize];
  }
  return notTranslatedIndex;
}

__device__ bool remapTypeIndexValue(uint32_t &typeIndex, DeviceRefKind refKind,
                                    const RemapSourceDescriptor &sourceDesc,
                                    const uint32_t *flatMap,
                                    const uint32_t *extraMaps) {
  if (typeIndex < firstNonSimpleIndex)
    return true;

  uint32_t undecorated = typeIndex & ~decoratedItemIdMask;
  if (undecorated < firstNonSimpleIndex) {
    typeIndex = notTranslatedIndex;
    return false;
  }

  uint32_t arrayIndex = undecorated - firstNonSimpleIndex;
  const DeviceRemapMapDescriptor &mapDesc = refKind == DeviceRefKind::IndexRef
                                                ? sourceDesc.ipiMap
                                                : sourceDesc.tpiMap;
  if (arrayIndex >= mapDesc.mapSize) {
    typeIndex = notTranslatedIndex;
    return false;
  }

  typeIndex = readRemapMapValue(mapDesc, arrayIndex, flatMap, extraMaps);
  return true;
}

__device__ bool remapTypeIndexAt(uint8_t *content, uint32_t contentSize,
                                 uint32_t offset, DeviceRefKind refKind,
                                 const RemapSourceDescriptor &sourceDesc,
                                 const uint32_t *flatMap,
                                 const uint32_t *extraMaps,
                                 RemapErrorSummary *errors,
                                 uint32_t recordIndex, uint16_t recordKind) {
  if (offset > contentSize || contentSize - offset < sizeof(uint32_t)) {
    recordRemapError(errors, recordIndex, recordKind, offset);
    return false;
  }

  uint32_t typeIndex = read32le(content + offset);
  uint32_t originalTypeIndex = typeIndex;
  bool ok =
      remapTypeIndexValue(typeIndex, refKind, sourceDesc, flatMap, extraMaps);
  write32le(content + offset, typeIndex);
  if (!ok)
    recordRemapError(errors, recordIndex, recordKind, originalTypeIndex);
  return ok;
}

__device__ bool remapTypeIndexRun(
    uint8_t *content, uint32_t contentSize, uint32_t offset, uint32_t count,
    DeviceRefKind refKind, const RemapSourceDescriptor &sourceDesc,
    const uint32_t *flatMap, const uint32_t *extraMaps,
    RemapErrorSummary *errors, uint32_t recordIndex, uint16_t recordKind) {
  if (count == 0)
    return true;
  if (offset > contentSize ||
      count > (contentSize - offset) / sizeof(uint32_t)) {
    recordRemapError(errors, recordIndex, recordKind, offset);
    return false;
  }

  bool ok = true;
  for (uint32_t i = 0; i != count; ++i)
    ok &= remapTypeIndexAt(content, contentSize, offset + i * sizeof(uint32_t),
                           refKind, sourceDesc, flatMap, extraMaps, errors,
                           recordIndex, recordKind);
  return ok;
}

__device__ bool isMemberPointer(uint32_t attrs) {
  uint32_t mode = (attrs >> llvm::codeview::PointerRecord::PointerModeShift) &
                  llvm::codeview::PointerRecord::PointerModeMask;
  return mode == uint32_t(llvm::codeview::PointerMode::PointerToDataMember) ||
         mode == uint32_t(llvm::codeview::PointerMode::PointerToMemberFunction);
}

__device__ bool isIntroVirtual(uint16_t attrs) {
  uint16_t methodKind =
      (attrs & uint16_t(llvm::codeview::MethodOptions::MethodKindMask)) >> 2;
  return methodKind ==
             uint16_t(llvm::codeview::MethodKind::IntroducingVirtual) ||
         methodKind ==
             uint16_t(llvm::codeview::MethodKind::PureIntroducingVirtual);
}

__device__ bool getEncodedIntegerLength(const uint8_t *data, uint32_t size,
                                        uint32_t &length) {
  if (size < sizeof(uint16_t))
    return false;
  uint16_t n = read16le(data);
  if (n < llvm::codeview::LF_NUMERIC) {
    length = 2;
    return true;
  }

  uint32_t payload = 0;
  switch (n) {
  case llvm::codeview::LF_CHAR:
    payload = 1;
    break;
  case llvm::codeview::LF_SHORT:
  case llvm::codeview::LF_USHORT:
  case llvm::codeview::LF_REAL16:
    payload = 2;
    break;
  case llvm::codeview::LF_LONG:
  case llvm::codeview::LF_ULONG:
  case llvm::codeview::LF_REAL32:
  case llvm::codeview::LF_COMPLEX32:
    payload = 4;
    break;
  case llvm::codeview::LF_REAL48:
    payload = 6;
    break;
  case llvm::codeview::LF_REAL64:
  case llvm::codeview::LF_QUADWORD:
  case llvm::codeview::LF_UQUADWORD:
  case llvm::codeview::LF_COMPLEX64:
  case llvm::codeview::LF_DATE:
    payload = 8;
    break;
  case llvm::codeview::LF_REAL80:
  case llvm::codeview::LF_COMPLEX80:
    payload = 10;
    break;
  case llvm::codeview::LF_REAL128:
  case llvm::codeview::LF_COMPLEX128:
  case llvm::codeview::LF_OCTWORD:
  case llvm::codeview::LF_UOCTWORD:
  case llvm::codeview::LF_DECIMAL:
    payload = 16;
    break;
  default:
    return false;
  }

  length = 2 + payload;
  return length <= size;
}

__device__ bool getCStringLength(const uint8_t *data, uint32_t size,
                                 uint32_t &length) {
  for (uint32_t i = 0; i != size; ++i) {
    if (data[i] == 0) {
      length = i + 1;
      return true;
    }
  }
  return false;
}

__device__ bool remapMethodOverloadList(uint8_t *content, uint32_t contentSize,
                                        const RemapSourceDescriptor &sourceDesc,
                                        const uint32_t *flatMap,
                                        const uint32_t *extraMaps,
                                        RemapErrorSummary *errors,
                                        uint32_t recordIndex,
                                        uint16_t recordKind) {
  uint32_t offset = 0;
  bool ok = true;
  while (offset < contentSize) {
    if (contentSize - offset < 8) {
      recordRemapError(errors, recordIndex, recordKind, offset);
      return false;
    }
    uint16_t attrs = read16le(content + offset);
    ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                           DeviceRefKind::TypeRef, sourceDesc, flatMap,
                           extraMaps, errors, recordIndex, recordKind);
    uint32_t len = isIntroVirtual(attrs) ? 12 : 8;
    if (contentSize - offset < len) {
      recordRemapError(errors, recordIndex, recordKind, offset);
      return false;
    }
    offset += len;
  }
  return ok;
}

__device__ bool remapFieldList(uint8_t *content, uint32_t contentSize,
                               const RemapSourceDescriptor &sourceDesc,
                               const uint32_t *flatMap,
                               const uint32_t *extraMaps,
                               RemapErrorSummary *errors, uint32_t recordIndex,
                               uint16_t recordKind) {
  uint32_t offset = 0;
  bool ok = true;
  while (offset < contentSize) {
    if (contentSize - offset < sizeof(uint16_t)) {
      recordRemapError(errors, recordIndex, recordKind, offset);
      return false;
    }

    uint16_t memberKind = read16le(content + offset);
    uint32_t len = 0;
    uint32_t encodedLen = 0;
    uint32_t nameLen = 0;
    switch (memberKind) {
    case llvm::codeview::LF_BCLASS:
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      if (contentSize - offset < 8 ||
          !getEncodedIntegerLength(content + offset + 8,
                                   contentSize - offset - 8, encodedLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len = 8 + encodedLen;
      break;
    case llvm::codeview::LF_ENUMERATE:
      if (contentSize - offset < 4 ||
          !getEncodedIntegerLength(content + offset + 4,
                                   contentSize - offset - 4, encodedLen) ||
          !getCStringLength(content + offset + 4 + encodedLen,
                            contentSize - offset - 4 - encodedLen, nameLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len = 4 + encodedLen + nameLen;
      break;
    case llvm::codeview::LF_MEMBER:
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      if (contentSize - offset < 8 ||
          !getEncodedIntegerLength(content + offset + 8,
                                   contentSize - offset - 8, encodedLen) ||
          !getCStringLength(content + offset + 8 + encodedLen,
                            contentSize - offset - 8 - encodedLen, nameLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len = 8 + encodedLen + nameLen;
      break;
    case llvm::codeview::LF_METHOD:
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      if (contentSize - offset < 8 ||
          !getCStringLength(content + offset + 8, contentSize - offset - 8,
                            nameLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len = 8 + nameLen;
      break;
    case llvm::codeview::LF_ONEMETHOD: {
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      if (contentSize - offset < 8) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      uint16_t attrs = read16le(content + offset + 2);
      uint32_t fixedLen = isIntroVirtual(attrs) ? 12 : 8;
      if (contentSize - offset < fixedLen ||
          !getCStringLength(content + offset + fixedLen,
                            contentSize - offset - fixedLen, nameLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len = fixedLen + nameLen;
      break;
    }
    case llvm::codeview::LF_NESTTYPE:
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      if (contentSize - offset < 8 ||
          !getCStringLength(content + offset + 8, contentSize - offset - 8,
                            nameLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len = 8 + nameLen;
      break;
    case llvm::codeview::LF_STMEMBER:
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      if (contentSize - offset < 8 ||
          !getCStringLength(content + offset + 8, contentSize - offset - 8,
                            nameLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len = 8 + nameLen;
      break;
    case llvm::codeview::LF_VBCLASS:
    case llvm::codeview::LF_IVBCLASS:
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      ok &= remapTypeIndexAt(content, contentSize, offset + 8,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      if (contentSize - offset < 12 ||
          !getEncodedIntegerLength(content + offset + 12,
                                   contentSize - offset - 12, encodedLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len = 12 + encodedLen;
      if (!getEncodedIntegerLength(content + offset + len,
                                   contentSize - offset - len, encodedLen)) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      len += encodedLen;
      break;
    case llvm::codeview::LF_VFUNCTAB:
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      len = 8;
      break;
    case llvm::codeview::LF_INDEX:
      ok &= remapTypeIndexAt(content, contentSize, offset + 4,
                             DeviceRefKind::TypeRef, sourceDesc, flatMap,
                             extraMaps, errors, recordIndex, recordKind);
      len = 8;
      break;
    default:
      return ok;
    }

    if (len > contentSize - offset) {
      recordRemapError(errors, recordIndex, recordKind, offset);
      return false;
    }
    offset += len;
    if (offset < contentSize && content[offset] >= llvm::codeview::LF_PAD0) {
      uint32_t skip = content[offset] & 0x0F;
      if (skip == 0)
        return ok;
      if (skip > contentSize - offset) {
        recordRemapError(errors, recordIndex, recordKind, offset);
        return false;
      }
      offset += skip;
    }
  }
  return ok;
}

__device__ bool remapTypeRecord(uint8_t *record, uint32_t recordSize,
                                const RemapSourceDescriptor &sourceDesc,
                                const uint32_t *flatMap,
                                const uint32_t *extraMaps,
                                RemapErrorSummary *errors,
                                uint32_t recordIndex) {
  if (recordSize < sizeof(llvm::codeview::RecordPrefix)) {
    recordRemapError(errors, recordIndex, 0, recordSize);
    return false;
  }

  uint16_t recordLen = read16le(record);
  uint16_t kind = read16le(record + 2);
  if (recordLen + sizeof(uint16_t) > recordSize) {
    recordRemapError(errors, recordIndex, kind, recordLen);
    return false;
  }

  uint8_t *content = record + sizeof(llvm::codeview::RecordPrefix);
  uint32_t contentSize = recordSize - sizeof(llvm::codeview::RecordPrefix);
  bool ok = true;
  uint32_t runCount = 0;
  DeviceRefKind kind0 = DeviceRefKind::TypeRef;
  DeviceRefKind kind1 = DeviceRefKind::TypeRef;
  DeviceRefKind runKind = DeviceRefKind::TypeRef;
  uint32_t offset0 = noTypeIndexOffset;
  uint32_t offset1 = noTypeIndexOffset;
  uint32_t runOffset = noTypeIndexOffset;

  switch (kind) {
  case llvm::codeview::LF_FUNC_ID:
    kind0 = DeviceRefKind::IndexRef;
    offset0 = 0;
    offset1 = 4;
    break;
  case llvm::codeview::LF_MFUNC_ID:
    offset0 = 0;
    offset1 = 4;
    break;
  case llvm::codeview::LF_STRING_ID:
    kind0 = DeviceRefKind::IndexRef;
    offset0 = 0;
    break;
  case llvm::codeview::LF_SUBSTR_LIST:
    if (contentSize < 4) {
      recordRemapError(errors, recordIndex, kind, contentSize);
      return false;
    }
    runCount = read32le(content);
    if (runCount > 0) {
      runKind = DeviceRefKind::IndexRef;
      runOffset = 4;
    }
    break;
  case llvm::codeview::LF_BUILDINFO:
    if (contentSize < 2) {
      recordRemapError(errors, recordIndex, kind, contentSize);
      return false;
    }
    runCount = read16le(content);
    if (runCount > 0) {
      runKind = DeviceRefKind::IndexRef;
      runOffset = 2;
    }
    break;
  case llvm::codeview::LF_UDT_SRC_LINE:
  case llvm::codeview::LF_UDT_MOD_SRC_LINE:
    offset0 = 0;
    kind1 = DeviceRefKind::IndexRef;
    offset1 = 4;
    break;
  case llvm::codeview::LF_MODIFIER:
    offset0 = 0;
    break;
  case llvm::codeview::LF_PROCEDURE:
    offset0 = 0;
    offset1 = 8;
    break;
  case llvm::codeview::LF_ARGLIST:
    if (contentSize < 4) {
      recordRemapError(errors, recordIndex, kind, contentSize);
      return false;
    }
    runCount = read32le(content);
    if (runCount > 0)
      runOffset = 4;
    break;
  case llvm::codeview::LF_ARRAY:
    offset0 = 0;
    offset1 = 4;
    break;
  case llvm::codeview::LF_CLASS:
  case llvm::codeview::LF_STRUCTURE:
  case llvm::codeview::LF_INTERFACE:
    runOffset = 4;
    runCount = 3;
    break;
  case llvm::codeview::LF_UNION:
    offset0 = 4;
    break;
  case llvm::codeview::LF_ENUM:
    offset0 = 4;
    offset1 = 8;
    break;
  case llvm::codeview::LF_BITFIELD:
    offset0 = 0;
    break;
  case llvm::codeview::LF_VFTABLE:
    offset0 = 0;
    offset1 = 4;
    break;
  case llvm::codeview::LF_POINTER:
    offset0 = 0;
    if (contentSize < 8) {
      recordRemapError(errors, recordIndex, kind, contentSize);
      return false;
    }
    if (isMemberPointer(read32le(content + 4)))
      offset1 = 8;
    break;
  case llvm::codeview::LF_MFUNCTION:
    ok &= remapTypeIndexAt(content, contentSize, 0, DeviceRefKind::TypeRef,
                           sourceDesc, flatMap, extraMaps, errors, recordIndex,
                           kind);
    ok &= remapTypeIndexAt(content, contentSize, 4, DeviceRefKind::TypeRef,
                           sourceDesc, flatMap, extraMaps, errors, recordIndex,
                           kind);
    ok &= remapTypeIndexAt(content, contentSize, 8, DeviceRefKind::TypeRef,
                           sourceDesc, flatMap, extraMaps, errors, recordIndex,
                           kind);
    ok &= remapTypeIndexAt(content, contentSize, 16, DeviceRefKind::TypeRef,
                           sourceDesc, flatMap, extraMaps, errors, recordIndex,
                           kind);
    return ok;
  case llvm::codeview::LF_METHODLIST:
    return remapMethodOverloadList(content, contentSize, sourceDesc, flatMap,
                                   extraMaps, errors, recordIndex, kind);
  case llvm::codeview::LF_FIELDLIST:
    return remapFieldList(content, contentSize, sourceDesc, flatMap, extraMaps,
                          errors, recordIndex, kind);
  default:
    return true;
  }

  if (offset0 != noTypeIndexOffset)
    ok &= remapTypeIndexAt(content, contentSize, offset0, kind0, sourceDesc,
                           flatMap, extraMaps, errors, recordIndex, kind);
  if (offset1 != noTypeIndexOffset)
    ok &= remapTypeIndexAt(content, contentSize, offset1, kind1, sourceDesc,
                           flatMap, extraMaps, errors, recordIndex, kind);
  if (runOffset != noTypeIndexOffset)
    ok &= remapTypeIndexRun(content, contentSize, runOffset, runCount, runKind,
                            sourceDesc, flatMap, extraMaps, errors, recordIndex,
                            kind);
  return ok;
}

struct DeviceStringRef {
  const uint8_t *data = nullptr;
  uint32_t size = 0;
};

__device__ uint32_t literalLength(const char *lit) {
  uint32_t len = 0;
  while (lit[len] != 0)
    ++len;
  return len;
}

__device__ bool equalsLiteral(DeviceStringRef str, const char *lit) {
  uint32_t len = literalLength(lit);
  if (str.size != len)
    return false;
  for (uint32_t i = 0; i != len; ++i)
    if (str.data[i] != uint8_t(lit[i]))
      return false;
  return true;
}

__device__ bool endsWithLiteral(DeviceStringRef str, const char *lit) {
  uint32_t len = literalLength(lit);
  if (str.size < len)
    return false;
  const uint8_t *tail = str.data + str.size - len;
  for (uint32_t i = 0; i != len; ++i)
    if (tail[i] != uint8_t(lit[i]))
      return false;
  return true;
}

__device__ bool isAnonymous(DeviceStringRef name) {
  return equalsLiteral(name, "<unnamed-tag>") ||
         equalsLiteral(name, "__unnamed") ||
         endsWithLiteral(name, "::<unnamed-tag>") ||
         endsWithLiteral(name, "::__unnamed");
}

__device__ bool readCStringRef(const uint8_t *data, uint32_t size,
                               DeviceStringRef &str, uint32_t &totalLen) {
  if (!getCStringLength(data, size, totalLen))
    return false;
  str.data = data;
  str.size = totalLen - 1;
  return true;
}

__device__ bool parseUdtNames(const uint8_t *content, uint32_t contentSize,
                              uint16_t kind, uint16_t &options,
                              DeviceStringRef &name,
                              DeviceStringRef &uniqueName) {
  uint32_t offset = 0;
  if (kind == llvm::codeview::LF_CLASS ||
      kind == llvm::codeview::LF_STRUCTURE ||
      kind == llvm::codeview::LF_INTERFACE) {
    if (contentSize < 16)
      return false;
    options = read16le(content + 2);
    offset = 16;
  } else if (kind == llvm::codeview::LF_UNION) {
    if (contentSize < 8)
      return false;
    options = read16le(content + 2);
    offset = 8;
  } else if (kind == llvm::codeview::LF_ENUM) {
    if (contentSize < 12)
      return false;
    options = read16le(content + 2);
    offset = 12;
  } else {
    return false;
  }

  uint32_t encodedLen = 0;
  if (kind != llvm::codeview::LF_ENUM) {
    if (!getEncodedIntegerLength(content + offset, contentSize - offset,
                                 encodedLen))
      return false;
    offset += encodedLen;
  }

  uint32_t nameTotalLen = 0;
  if (!readCStringRef(content + offset, contentSize - offset, name,
                      nameTotalLen))
    return false;
  offset += nameTotalLen;

  if (options & uint16_t(llvm::codeview::ClassOptions::HasUniqueName)) {
    uint32_t uniqueTotalLen = 0;
    if (!readCStringRef(content + offset, contentSize - offset, uniqueName,
                        uniqueTotalLen))
      return false;
  }
  return true;
}

__device__ uint32_t hashTypeRecordDevice(uint8_t *record, uint32_t recordSize,
                                         RemapErrorSummary *errors,
                                         uint32_t recordIndex) {
  uint16_t kind = read16le(record + 2);
  uint8_t *content = record + sizeof(llvm::codeview::RecordPrefix);
  uint32_t contentSize = recordSize - sizeof(llvm::codeview::RecordPrefix);
  switch (kind) {
  case llvm::codeview::LF_CLASS:
  case llvm::codeview::LF_STRUCTURE:
  case llvm::codeview::LF_INTERFACE:
  case llvm::codeview::LF_UNION:
  case llvm::codeview::LF_ENUM: {
    uint16_t options = 0;
    DeviceStringRef name;
    DeviceStringRef uniqueName;
    if (!parseUdtNames(content, contentSize, kind, options, name, uniqueName)) {
      recordRemapError(errors, recordIndex, kind, contentSize);
      return 0;
    }

    bool forwardRef =
        options & uint16_t(llvm::codeview::ClassOptions::ForwardReference);
    bool scoped = options & uint16_t(llvm::codeview::ClassOptions::Scoped);
    bool hasUnique =
        options & uint16_t(llvm::codeview::ClassOptions::HasUniqueName);
    bool anonymous = hasUnique && isAnonymous(name);
    if (!forwardRef && !scoped && !anonymous)
      return llvm::pdb::hashStringV1(name.data, name.size);
    if (!forwardRef && hasUnique && !anonymous)
      return llvm::pdb::hashStringV1(uniqueName.data, uniqueName.size);
    return llvm::pdb::hashBufferV8(record, recordSize);
  }
  case llvm::codeview::LF_UDT_SRC_LINE:
  case llvm::codeview::LF_UDT_MOD_SRC_LINE:
    if (contentSize < sizeof(uint32_t)) {
      recordRemapError(errors, recordIndex, kind, contentSize);
      return 0;
    }
    return llvm::pdb::hashStringV1(content, sizeof(uint32_t));
  default:
    return llvm::pdb::hashBufferV8(record, recordSize);
  }
}

__global__ void remapAndHashTypeRecords(
    uint8_t *records, const SelectedRecordDescriptor *recordDescs,
    uint32_t recordCount, const RemapSourceDescriptor *sourceDescs,
    const uint32_t *flatMap, const uint32_t *extraMaps, uint32_t *hashes,
    FuncIdToTypeEntry *funcIdToType, RemapErrorSummary *errorsBySource) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= recordCount)
    return;

  SelectedRecordDescriptor recordDesc = recordDescs[i];
  uint32_t sourceIdx = recordDesc.sourceIndex;
  RemapSourceDescriptor sourceDesc = sourceDescs[sourceIdx];
  RemapErrorSummary *errors = errorsBySource + sourceIdx;
  uint32_t recordIndex = recordDesc.ghashIndex;

  uint8_t *record = records + recordDesc.byteOffset;
  uint32_t recordSize = recordDesc.size;
  bool ok = remapTypeRecord(record, recordSize, sourceDesc, flatMap, extraMaps,
                            errors, recordIndex);
  if (!ok)
    return;

  hashes[i] = hashTypeRecordDevice(record, recordSize, errors, recordIndex);

  if (!funcIdToType || recordSize < 12)
    return;
  uint16_t kind = read16le(record + 2);
  if (kind != llvm::codeview::LF_FUNC_ID && kind != llvm::codeview::LF_MFUNC_ID)
    return;

  uint32_t sourceIndex =
      sourceDesc.sourceTypeIndexBegin + recordDesc.ghashIndex;
  uint32_t funcId = sourceIndex;
  if (!remapTypeIndexValue(funcId, DeviceRefKind::IndexRef, sourceDesc, flatMap,
                           extraMaps)) {
    recordRemapError(errors, recordIndex, kind, sourceIndex);
    return;
  }
  funcIdToType[i] = {funcId, read32le(record + 8)};
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

struct MemcpyBatch {
  std::vector<void *> dsts;
  std::vector<const void *> srcs;
  std::vector<size_t> sizes;

  void reserve(size_t count) {
    dsts.reserve(count);
    srcs.reserve(count);
    sizes.reserve(count);
  }

  void append(void *dst, const void *src, size_t size) {
    if (size == 0)
      return;
    dsts.push_back(dst);
    srcs.push_back(src);
    sizes.push_back(size);
  }

  bool empty() const { return sizes.empty(); }
  size_t size() const { return sizes.size(); }
};

void enqueueMemcpyBatchAndSync(MemcpyBatch &batch, const char *enqueueContext,
                               const char *syncContext,
                               CudaErrorChecker &cuErr) {
  if (batch.empty())
    return;

  cudaStream_t stream = nullptr;
  cuErr.fatal(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "-lldcudaghash failed to create memcpy stream");
#if CUDART_VERSION >= 13000
  cudaMemcpyAttributes attr = {};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attrIdx = 0;
  cudaError_t batchErr =
      cudaMemcpyBatchAsync(batch.dsts.data(), batch.srcs.data(),
                           batch.sizes.data(), batch.size(), &attr, &attrIdx,
                           1, stream);
  if (batchErr == cudaErrorCallRequiresNewerDriver ||
      batchErr == cudaErrorNotSupported) {
    for (size_t i = 0, e = batch.size(); i != e; ++i)
      cuErr.fatal(cudaMemcpyAsync(batch.dsts[i], batch.srcs[i],
                                  batch.sizes[i], cudaMemcpyDefault, stream),
                  enqueueContext);
  } else {
    cuErr.fatal(batchErr, enqueueContext);
  }
#else
  for (size_t i = 0, e = batch.size(); i != e; ++i)
    cuErr.fatal(cudaMemcpyAsync(batch.dsts[i], batch.srcs[i], batch.sizes[i],
                                cudaMemcpyDefault, stream),
                enqueueContext);
#endif
  cuErr.fatal(cudaStreamSynchronize(stream), syncContext);
  cuErr.fatal(cudaStreamDestroy(stream),
              "-lldcudaghash failed to destroy memcpy stream");
}

void appendGHashRangeCopy(ArrayRef<GloballyHashedType> ghashes,
                          uint32_t ghashBegin, uint32_t ghashEnd,
                          FlatIndex entryOffset,
                          thrust::device_vector<uint64_t> &deviceHashes,
                          MemcpyBatch &batch) {
  static_assert(sizeof(GloballyHashedType) == sizeof(uint64_t),
                "CUDA GHASH copies assume 64-bit hash records");
  if (ghashBegin == ghashEnd)
    return;

  uint32_t count = ghashEnd - ghashBegin;
  const uint64_t *hashes =
      reinterpret_cast<const uint64_t *>(ghashes.data() + ghashBegin);
  uint64_t *dst = thrust::raw_pointer_cast(deviceHashes.data()) + entryOffset;
  batch.append(dst, hashes, size_t(count) * sizeof(uint64_t));
}

void appendSourceGHashCopies(const TpiSource &source, FlatIndex entryOffset,
                             thrust::device_vector<uint64_t> &deviceHashes,
                             MemcpyBatch &batch) {
  uint32_t sourceSize = static_cast<uint32_t>(source.ghashes.size());
  bool skipEndPrecomp = source.endPrecompIdx < sourceSize;
  uint32_t firstEnd = skipEndPrecomp ? source.endPrecompIdx : sourceSize;
  appendGHashRangeCopy(source.ghashes, 0, firstEnd, entryOffset, deviceHashes,
                       batch);
  if (skipEndPrecomp)
    appendGHashRangeCopy(source.ghashes, source.endPrecompIdx + 1, sourceSize,
                         entryOffset + firstEnd, deviceHashes, batch);
}

struct GHashCUDAState {
  GHashCUDAResult result;
  thrust::device_vector<uint32_t> deviceMap;
  thrust::device_vector<uint64_t> deviceUniqueSrcs;
};

void mergeGHashesWithCUDA(COFFLinkerContext &ctx,
                          thrust::device_vector<uint64_t> &deviceHashes,
                          thrust::device_vector<uint64_t> &deviceSrcs,
                          const FlatIndex *mapOffsets, FlatIndex mapOffsetCount,
                          FlatIndex mapCount, uint32_t notTranslated,
                          ArrayRef<TpiSource *> tpiSources,
                          GHashCUDAState *state, CudaErrorChecker &cuErr) {
  static_assert(sizeof(TypeIndex) == sizeof(uint32_t),
                "TypeIndex must stay layout-compatible with a PDB index");
  static_assert(std::is_trivially_copyable<TypeIndex>::value,
                "TypeIndex must be safe to copy from CUDA output bytes");

  state->result = GHashCUDAResult();
  state->deviceMap.clear();
  state->deviceUniqueSrcs.clear();
  assert(deviceHashes.size() == deviceSrcs.size());
  FlatIndex entryCount = deviceHashes.size();
  if (entryCount == 0)
    return;

  // The ragged source/type-index grid is represented as one flat destination
  // map. Initialize the whole flat space on the GPU so omitted records keep the
  // NotTranslated sentinel without requiring a host-side staging vector.
  state->deviceMap.resize(mapCount);
  thrust::fill(thrust::device, state->deviceMap.begin(), state->deviceMap.end(),
               notTranslated);

  thrust::device_vector<FlatIndex> deviceMapOffsets(
      mapOffsets, mapOffsets + mapOffsetCount);

  // Host setup has already omitted LF_ENDPRECOMP records, so the GPU path
  // starts with dense hash/source-position arrays and can sort immediately.
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
  state->deviceUniqueSrcs.resize(uniqueCount);
  uint32_t uniqueBlocks = getBlockCount(ctx, uniqueCount, threads);
  // Build the compact group -> destination TypeIndex table and preserve the
  // sorted source order for host-side record staging. This is the only place
  // that assigns final PDB TPI/IPI indices.
  assignDestinationIndicesAndExtractSrcs<<<uniqueBlocks, threads>>>(
      thrust::raw_pointer_cast(uniqueEntries.data()), uniqueCount, numTypes,
      thrust::raw_pointer_cast(groupToTypeIndex.data()),
      thrust::raw_pointer_cast(state->deviceUniqueSrcs.data()));
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
      thrust::raw_pointer_cast(state->deviceMap.data()));
  cuErr.fatal(cudaGetLastError(),
              "-lldcudaghash failed to launch flat map fill kernel");
  cuErr.fatal(cudaDeviceSynchronize(),
              "-lldcudaghash failed while synchronizing kernels");

  const uint32_t *deviceMapData =
      thrust::raw_pointer_cast(state->deviceMap.data());
  MemcpyBatch mapCopies;
  mapCopies.reserve(tpiSources.size());
  for (TpiSource *source : tpiSources) {
    FlatIndex mapOffset = mapOffsets[source->tpiSrcIdx];
    size_t byteCount = source->indexMapStorage.size() * sizeof(TypeIndex);
    if (byteCount == 0)
      continue;
    mapCopies.append(source->indexMapStorage.data(), deviceMapData + mapOffset,
                     byteCount);
  }
  enqueueMemcpyBatchAndSync(mapCopies,
                            "-lldcudaghash failed to enqueue type index map "
                            "copy batch",
                            "-lldcudaghash failed while copying type index "
                            "maps",
                            cuErr);

  state->result.uniqueCount = uniqueCount;
  state->result.numTypes = numTypes;
  state->result.numItems = numItems;
}

class CudaTypeRecordProvider final : public llvm::pdb::TpiRecordProvider {
public:
  CudaTypeRecordProvider(
      std::shared_ptr<thrust::device_vector<uint8_t>> deviceBytes,
      size_t byteOffset, size_t byteCount,
      std::shared_ptr<std::vector<uint16_t>> sizes,
      std::shared_ptr<std::vector<uint32_t>> hashes, uint32_t recordOffset,
      uint32_t recordCount)
      : DeviceBytes(std::move(deviceBytes)), ByteOffset(byteOffset),
        ByteCount(byteCount), Sizes(std::move(sizes)),
        Hashes(std::move(hashes)), RecordOffset(recordOffset),
        RecordCount(recordCount) {}

  ArrayRef<uint16_t> getRecordSizes() const override {
    return ArrayRef(*Sizes).slice(RecordOffset, RecordCount);
  }
  ArrayRef<uint32_t> getRecordHashes() const override {
    return ArrayRef(*Hashes).slice(RecordOffset, RecordCount);
  }

  llvm::Error writeRecords(llvm::BinaryStreamWriter &writer) const override {
    if (ByteCount == 0)
      return llvm::Error::success();

    assert(DeviceBytes && "CUDA type record provider lost its device buffer");
    std::vector<uint8_t> hostBytes(ByteCount);
    const uint8_t *deviceBegin =
        thrust::raw_pointer_cast(DeviceBytes->data()) + ByteOffset;
    cudaError_t err = cudaMemcpy(hostBytes.data(), deviceBegin, ByteCount,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "failed to copy CUDA type records: %s",
                                     cudaGetErrorString(err));
    return writer.writeBytes(hostBytes);
  }

  std::shared_ptr<thrust::device_vector<uint8_t>> DeviceBytes;
  size_t ByteOffset = 0;
  size_t ByteCount = 0;
  std::shared_ptr<std::vector<uint16_t>> Sizes;
  std::shared_ptr<std::vector<uint32_t>> Hashes;
  uint32_t RecordOffset = 0;
  uint32_t RecordCount = 0;
};

struct SelectedRecordSet {
  SelectedRecordSet() : sizes(std::make_shared<std::vector<uint16_t>>()) {}

  std::vector<uint8_t> records;
  std::vector<SelectedRecordDescriptor> descs;
  std::shared_ptr<std::vector<uint16_t>> sizes;

  bool empty() const { return descs.empty(); }

  void resize(uint32_t recordCount, uint32_t byteCount) {
    records.resize(byteCount);
    descs.resize(recordCount);
    sizes->resize(recordCount);
  }
};

struct RecordRange {
  uint32_t byteOffset = 0;
  uint32_t byteCount = 0;
  uint32_t recordOffset = 0;
  uint32_t recordCount = 0;
};

struct SourceRecordRanges {
  RecordRange tpi;
  RecordRange ipi;
};

struct SelectedRecordStats {
  uint32_t byteCount = 0;
  uint32_t recordCount = 0;
};

struct SelectedRecordCursor {
  uint32_t byteOffset = 0;
  uint32_t recordOffset = 0;
};

template <typename T> T *deviceData(thrust::device_vector<T> &v) {
  return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
}

template <typename T> const T *deviceData(const thrust::device_vector<T> &v) {
  return v.empty() ? nullptr : thrust::raw_pointer_cast(v.data());
}

StringRef getTpiSourceName(const TpiSource &source) {
  return source.file ? source.file->getName() : StringRef("<type server>");
}

uint32_t checkedUInt32(COFFLinkerContext &ctx, size_t value, const char *what) {
  if (value > std::numeric_limits<uint32_t>::max())
    Fatal(ctx) << "-lldcudaghash failed: " << what << " exceeds 32-bit range";
  return static_cast<uint32_t>(value);
}

struct SelectedRecordInfo {
  const uint8_t *record = nullptr;
  uint32_t recordSize = 0;
  uint32_t alignedSize = 0;
};

SelectedRecordInfo getSelectedRecordInfo(COFFLinkerContext &ctx,
                                         const TpiSource &source,
                                         uint32_t ghashIdx) {
  StringRef sourceName = getTpiSourceName(source);
  if (ghashIdx >= source.typeIndexOffsets.size() ||
      ghashIdx >= source.typeLeafKinds.size())
    Fatal(ctx) << "-lldcudaghash selected record metadata is missing for "
               << sourceName << " record " << ghashIdx;

  uint32_t offset = source.typeIndexOffsets[ghashIdx];
  if (offset > source.ghashTypeRecords.size() ||
      source.ghashTypeRecords.size() - offset <
          sizeof(llvm::codeview::RecordPrefix))
    Fatal(ctx) << "-lldcudaghash selected record offset is corrupt for "
               << sourceName << " record " << ghashIdx;

  const uint8_t *record = source.ghashTypeRecords.data() + offset;
  uint32_t recordSize = read16le(record) + sizeof(uint16_t);
  if (recordSize > source.ghashTypeRecords.size() - offset)
    Fatal(ctx) << "-lldcudaghash selected record size is corrupt for "
               << sourceName << " record " << ghashIdx;

  uint32_t alignedSize = llvm::alignTo(recordSize, 4);
  if (alignedSize > llvm::codeview::MaxRecordLength)
    Fatal(ctx) << "-lldcudaghash selected record is too large for "
               << sourceName << " record " << ghashIdx;
  return {record, recordSize, alignedSize};
}

void addSelectedRecordStats(COFFLinkerContext &ctx,
                            const SelectedRecordInfo &info,
                            RecordRange &range,
                            SelectedRecordStats &stats) {
  if (range.recordCount == 0) {
    range.byteOffset = stats.byteCount;
    range.recordOffset = stats.recordCount;
  }
  range.byteCount =
      checkedUInt32(ctx, size_t(range.byteCount) + info.alignedSize,
                    "source record byte count");
  stats.byteCount =
      checkedUInt32(ctx, size_t(stats.byteCount) + info.alignedSize,
                    "selected type byte count");
  ++stats.recordCount;
  ++range.recordCount;
}

void copySelectedRecord(COFFLinkerContext &ctx, const TpiSource &source,
                        uint32_t sourceIdx, uint32_t ghashIdx,
                        SelectedRecordSet &recordSet,
                        SelectedRecordCursor &cursor) {
  SelectedRecordInfo info = getSelectedRecordInfo(ctx, source, ghashIdx);
  assert(cursor.recordOffset < recordSet.descs.size());
  assert(cursor.byteOffset + info.alignedSize <= recordSet.records.size());

  recordSet.descs[cursor.recordOffset] = {cursor.byteOffset, info.alignedSize,
                                          ghashIdx, sourceIdx};
  (*recordSet.sizes)[cursor.recordOffset] =
      static_cast<uint16_t>(info.alignedSize);
  memcpy(recordSet.records.data() + cursor.byteOffset, info.record,
         info.recordSize);
  if (info.alignedSize != info.recordSize) {
    write16le(recordSet.records.data() + cursor.byteOffset,
              info.alignedSize - 2);
    for (uint32_t i = info.recordSize; i != info.alignedSize; ++i) {
      recordSet.records[cursor.byteOffset + i] =
          llvm::codeview::LF_PAD0 + uint8_t(info.alignedSize - i);
    }
  }
  cursor.byteOffset += info.alignedSize;
  ++cursor.recordOffset;
}

void appendTypeIndexMap(COFFLinkerContext &ctx, ArrayRef<TypeIndex> map,
                        std::vector<uint32_t> &flatMap) {
  uint32_t mapSize = checkedUInt32(ctx, map.size(), "source type map size");
  size_t offset = flatMap.size();
  flatMap.resize(offset + mapSize);
  for (uint32_t i = 0; i != mapSize; ++i)
    flatMap[offset + i] = map[i].getIndex();
}

struct RemapMapResolver {
  RemapMapResolver(ArrayRef<TpiSource *> sources,
                   ArrayRef<FlatIndex> mapOffsets)
      : sources(sources), mapOffsets(mapOffsets) {
    for (TpiSource *source : sources) {
      if (source->indexMapStorage.empty() ||
          source->indexMapStorage.size() != source->ghashes.size())
        continue;
      directMapOwners.insert({source->indexMapStorage.data(),
                              source->tpiSrcIdx});
    }
  }

  ArrayRef<TpiSource *> sources;
  ArrayRef<FlatIndex> mapOffsets;
  llvm::DenseMap<const TypeIndex *, uint32_t> directMapOwners;
};

uint32_t findDirectMapOwner(const RemapMapResolver &resolver,
                            ArrayRef<TypeIndex> map) {
  if (map.empty())
    return ~0U;

  auto it = resolver.directMapOwners.find(map.data());
  if (it == resolver.directMapOwners.end())
    return ~0U;

  uint32_t ownerIdx = it->second;
  if (ownerIdx >= resolver.sources.size())
    return ~0U;
  TpiSource *owner = resolver.sources[ownerIdx];
  if (owner->ghashes.size() != map.size() ||
      owner->indexMapStorage.data() != map.data())
    return ~0U;
  return ownerIdx;
}

bool setPrefixedFlatMapDescriptor(COFFLinkerContext &ctx,
                                  const RemapMapResolver &resolver,
                                  const TpiSource &source,
                                  ArrayRef<TypeIndex> map,
                                  DeviceRemapMapDescriptor &mapDesc) {
  if (map.data() != source.indexMapStorage.data() ||
      map.size() <= source.ghashes.size())
    return false;

  uint32_t prefixSize = checkedUInt32(ctx, map.size() - source.ghashes.size(),
                                      "source type map prefix size");
  ArrayRef<TypeIndex> prefix = map.take_front(prefixSize);
  for (TpiSource *owner : resolver.sources) {
    if (owner == &source ||
        owner->indexMapStorage.size() != owner->ghashes.size() ||
        owner->indexMapStorage.size() < prefix.size())
      continue;

    ArrayRef<TypeIndex> ownerPrefix(owner->indexMapStorage.data(),
                                    prefix.size());
    if (!std::equal(prefix.begin(), prefix.end(), ownerPrefix.begin()))
      continue;

    mapDesc.mapOffset = resolver.mapOffsets[owner->tpiSrcIdx];
    mapDesc.suffixMapOffset = resolver.mapOffsets[source.tpiSrcIdx];
    mapDesc.prefixSize = prefixSize;
    mapDesc.mapKind = RemapMapKind::PrefixFlatMap;
    return true;
  }
  return false;
}

void setRemapMapDescriptor(COFFLinkerContext &ctx,
                           const RemapMapResolver &resolver,
                           const TpiSource &source, ArrayRef<TypeIndex> map,
                           std::vector<uint32_t> &extraMaps,
                           DeviceRemapMapDescriptor &mapDesc) {
  mapDesc.mapSize = checkedUInt32(ctx, map.size(), "source type map size");
  if (map.empty()) {
    mapDesc.mapOffset = 0;
    mapDesc.mapKind = RemapMapKind::FlatMap;
    return;
  }

  uint32_t owner = findDirectMapOwner(resolver, map);
  if (owner != ~0U) {
    mapDesc.mapOffset = resolver.mapOffsets[owner];
    mapDesc.mapKind = RemapMapKind::FlatMap;
    return;
  }

  if (setPrefixedFlatMapDescriptor(ctx, resolver, source, map, mapDesc)) {
    return;
  }

  mapDesc.mapOffset = extraMaps.size();
  mapDesc.mapKind = RemapMapKind::ExtraMap;
  appendTypeIndexMap(ctx, map, extraMaps);
}

void reportRemapErrors(COFFLinkerContext &ctx, ArrayRef<TpiSource *> sources,
                       ArrayRef<RemapErrorSummary> errors) {
  uint64_t totalErrors = 0;
  for (size_t sourceIdx = 0, e = sources.size(); sourceIdx != e; ++sourceIdx) {
    const RemapErrorSummary &sourceErrors = errors[sourceIdx];
    if (sourceErrors.count == 0)
      continue;

    totalErrors += sourceErrors.count;
    uint32_t shown = std::min(sourceErrors.count, maxRemapErrors);
    Warn(ctx) << "-lldcudaghash remap found " << sourceErrors.count
              << " issue(s) in " << getTpiSourceName(*sources[sourceIdx])
              << "; CUDA type remap failed";
    for (uint32_t i = 0; i != shown; ++i)
      Warn(ctx) << "  record " << sourceErrors.recordIndex[i] << " kind 0x"
                << llvm::utohexstr(sourceErrors.recordKind[i]) << " detail 0x"
                << llvm::utohexstr(sourceErrors.detail[i]);
  }
  if (totalErrors != 0)
    Fatal(ctx) << "-lldcudaghash type remapping failed with " << totalErrors
               << " issue(s)";
}

std::shared_ptr<CudaTypeRecordProvider>
makeRecordProvider(std::shared_ptr<thrust::device_vector<uint8_t>> deviceBytes,
                   const SelectedRecordSet &recordSet, const RecordRange &range,
                   std::shared_ptr<std::vector<uint32_t>> hashes) {
  if (range.recordCount == 0)
    return nullptr;

  assert(range.recordOffset + range.recordCount <= recordSet.sizes->size());
  assert(hashes && range.recordOffset + range.recordCount <= hashes->size());
  return std::make_shared<CudaTypeRecordProvider>(
      std::move(deviceBytes), range.byteOffset, range.byteCount,
      recordSet.sizes, std::move(hashes), range.recordOffset,
      range.recordCount);
}

void remapRecordSetWithCUDA(
    COFFLinkerContext &ctx, SelectedRecordSet &recordSet,
    const thrust::device_vector<RemapSourceDescriptor> &sourceDescs,
    const thrust::device_vector<uint32_t> &flatMap,
    const thrust::device_vector<uint32_t> &extraMaps, RemapErrorSummary *errors,
    std::shared_ptr<thrust::device_vector<uint8_t>> &deviceBytes,
    std::shared_ptr<std::vector<uint32_t>> &hashes,
    std::vector<FuncIdToTypeEntry> *funcPairs, CudaErrorChecker &cuErr) {
  deviceBytes = std::make_shared<thrust::device_vector<uint8_t>>();
  hashes = std::make_shared<std::vector<uint32_t>>();
  if (funcPairs)
    funcPairs->clear();
  if (recordSet.empty())
    return;

  uint32_t recordCount =
      checkedUInt32(ctx, recordSet.descs.size(), "selected record count");
  deviceBytes->resize(recordSet.records.size());
  if (!recordSet.records.empty())
    thrust::copy(recordSet.records.begin(), recordSet.records.end(),
                 deviceBytes->begin());

  thrust::device_vector<SelectedRecordDescriptor> deviceRecordDescs(
      recordSet.descs.begin(), recordSet.descs.end());
  thrust::device_vector<uint32_t> deviceHashes(recordCount);
  thrust::device_vector<FuncIdToTypeEntry> deviceFuncPairs;
  FuncIdToTypeEntry *funcPairData = nullptr;
  if (funcPairs) {
    deviceFuncPairs.resize(recordCount);
    cuErr.fatal(cudaMemset(deviceData(deviceFuncPairs), 0,
                           deviceFuncPairs.size() * sizeof(FuncIdToTypeEntry)),
                "-lldcudaghash failed to initialize function ID records");
    funcPairData = deviceData(deviceFuncPairs);
  }

  uint32_t threads = 256;
  uint32_t blocks = getBlockCount(ctx, recordCount, threads);
  remapAndHashTypeRecords<<<blocks, threads>>>(
      deviceData(*deviceBytes), deviceData(deviceRecordDescs), recordCount,
      deviceData(sourceDescs), deviceData(flatMap), deviceData(extraMaps),
      deviceData(deviceHashes), funcPairData, errors);
  cuErr.fatal(cudaGetLastError(),
              "-lldcudaghash failed to launch type remap kernel");
  cuErr.fatal(cudaDeviceSynchronize(),
              "-lldcudaghash failed while synchronizing type remap kernel");

  hashes->resize(recordCount);
  cuErr.fatal(cudaMemcpy(hashes->data(), deviceData(deviceHashes),
                         hashes->size() * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              "-lldcudaghash failed to copy type record hashes");
  if (funcPairs) {
    funcPairs->resize(deviceFuncPairs.size());
    cuErr.fatal(cudaMemcpy(funcPairs->data(),
                           thrust::raw_pointer_cast(deviceFuncPairs.data()),
                           funcPairs->size() * sizeof(FuncIdToTypeEntry),
                           cudaMemcpyDeviceToHost),
                "-lldcudaghash failed to copy function ID records");
  }
}

void remapAllSelectedTypeRecordsWithCUDA(COFFLinkerContext &ctx,
                                         ArrayRef<FlatIndex> mapOffsets,
                                         const GHashCUDAState &ghashState,
                                         CudaErrorChecker &cuErr) {
  SelectedRecordSet tpiRecords;
  SelectedRecordSet ipiRecords;
  std::vector<SourceRecordRanges> recordRanges(ctx.tpiSourceList.size());
  std::vector<RemapSourceDescriptor> sourceDescs(ctx.tpiSourceList.size());
  std::vector<uint8_t> remapPrepared(ctx.tpiSourceList.size());
  std::vector<uint32_t> extraMaps;
  std::vector<uint64_t> uniqueSrcs(ghashState.result.uniqueCount);
  if (!uniqueSrcs.empty())
    cuErr.fatal(cudaMemcpy(uniqueSrcs.data(),
                           deviceData(ghashState.deviceUniqueSrcs),
                           uniqueSrcs.size() * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost),
                "-lldcudaghash failed to copy unique source records");

  // Symbol remapping still consumes each source's tpiMap/ipiMap later. Prepare
  // those lightweight ArrayRef views for every source, but only build CUDA
  // remap descriptors and stage type bytes for sources that actually contribute
  // selected records below.
  for (TpiSource *source : ctx.tpiSourceList)
    remapPrepared[source->tpiSrcIdx] = source->prepareGHashRemap();

  RemapMapResolver mapResolver(ctx.tpiSourceList, mapOffsets);

  auto prepareSelectedSource = [&](TpiSource *source) {
    RemapSourceDescriptor &sourceDesc = sourceDescs[source->tpiSrcIdx];
    if (sourceDesc.sourceTypeIndexBegin != 0)
      return;

    if (!source->mergedTpi.recs.empty() || !source->mergedIpi.recs.empty() ||
        source->mergedTpi.deferredRecords || source->mergedIpi.deferredRecords)
      Fatal(ctx) << "-lldcudaghash found pre-existing merged type records for "
                 << getTpiSourceName(*source);

    if (!remapPrepared[source->tpiSrcIdx])
      Fatal(ctx) << "-lldcudaghash failed to prepare type remapping for "
                 << getTpiSourceName(*source);

    sourceDesc.sourceTypeIndexBegin =
        source->getGHashRecordStartIndex().getIndex();
    setRemapMapDescriptor(ctx, mapResolver, *source, source->tpiMap, extraMaps,
                          sourceDesc.tpiMap);
    setRemapMapDescriptor(ctx, mapResolver, *source, source->ipiMap, extraMaps,
                          sourceDesc.ipiMap);

    if (ctx.config.showSummary) {
      source->nbTypeRecords = source->ghashes.size();
      source->nbTypeRecordsBytes = source->ghashTypeRecords.size();
    }
  };

  SelectedRecordStats tpiStats;
  SelectedRecordStats ipiStats;
  for (uint64_t src : uniqueSrcs) {
    uint32_t sourceIdx = getTpiSrcIdx(src);
    uint32_t ghashIdx = getGHashIdx(src);
    TpiSource *source = ctx.tpiSourceList[sourceIdx];
    prepareSelectedSource(source);
    if (source->typeLeafKinds.size() != source->ghashes.size() ||
        source->typeIndexOffsets.size() != source->ghashes.size() ||
        source->ghashTypeRecords.empty())
      Fatal(ctx) << "-lldcudaghash selected record metadata is incomplete for "
                 << getTpiSourceName(*source);

    RecordRange &range = isItemSrc(src) ? recordRanges[sourceIdx].ipi
                                        : recordRanges[sourceIdx].tpi;
    SelectedRecordStats &stats = isItemSrc(src) ? ipiStats : tpiStats;
    addSelectedRecordStats(ctx, getSelectedRecordInfo(ctx, *source, ghashIdx),
                           range, stats);
  }

  tpiRecords.resize(tpiStats.recordCount, tpiStats.byteCount);
  ipiRecords.resize(ipiStats.recordCount, ipiStats.byteCount);
  SelectedRecordCursor tpiCursor;
  SelectedRecordCursor ipiCursor;
  for (uint64_t src : uniqueSrcs) {
    uint32_t sourceIdx = getTpiSrcIdx(src);
    uint32_t ghashIdx = getGHashIdx(src);
    TpiSource *source = ctx.tpiSourceList[sourceIdx];
    SelectedRecordSet &recordSet = isItemSrc(src) ? ipiRecords : tpiRecords;
    SelectedRecordCursor &cursor = isItemSrc(src) ? ipiCursor : tpiCursor;
    copySelectedRecord(ctx, *source, sourceIdx, ghashIdx, recordSet, cursor);
  }
  assert(tpiCursor.recordOffset == tpiRecords.descs.size());
  assert(tpiCursor.byteOffset == tpiRecords.records.size());
  assert(ipiCursor.recordOffset == ipiRecords.descs.size());
  assert(ipiCursor.byteOffset == ipiRecords.records.size());

  thrust::device_vector<RemapSourceDescriptor> deviceSourceDescs(
      sourceDescs.begin(), sourceDescs.end());
  thrust::device_vector<uint32_t> deviceExtraMaps(extraMaps.begin(),
                                                  extraMaps.end());
  thrust::device_vector<RemapErrorSummary> deviceErrors(
      ctx.tpiSourceList.size());
  if (!deviceErrors.empty())
    cuErr.fatal(cudaMemset(deviceData(deviceErrors), 0,
                           deviceErrors.size() * sizeof(RemapErrorSummary)),
                "-lldcudaghash failed to initialize remap error state");

  std::shared_ptr<thrust::device_vector<uint8_t>> deviceTpiBytes;
  std::shared_ptr<thrust::device_vector<uint8_t>> deviceIpiBytes;
  std::shared_ptr<std::vector<uint32_t>> tpiHashes;
  std::shared_ptr<std::vector<uint32_t>> ipiHashes;
  std::vector<FuncIdToTypeEntry> funcPairs;
  remapRecordSetWithCUDA(
      ctx, tpiRecords, deviceSourceDescs, ghashState.deviceMap, deviceExtraMaps,
      deviceData(deviceErrors), deviceTpiBytes, tpiHashes, nullptr, cuErr);
  remapRecordSetWithCUDA(
      ctx, ipiRecords, deviceSourceDescs, ghashState.deviceMap, deviceExtraMaps,
      deviceData(deviceErrors), deviceIpiBytes, ipiHashes, &funcPairs, cuErr);

  std::vector<RemapErrorSummary> errorsHost(deviceErrors.size());
  if (!errorsHost.empty())
    cuErr.fatal(cudaMemcpy(errorsHost.data(), deviceData(deviceErrors),
                           errorsHost.size() * sizeof(RemapErrorSummary),
                           cudaMemcpyDeviceToHost),
                "-lldcudaghash failed to copy remap error state");
  reportRemapErrors(ctx, ctx.tpiSourceList, errorsHost);

  for (TpiSource *source : ctx.tpiSourceList) {
    SourceRecordRanges &ranges = recordRanges[source->tpiSrcIdx];
    if (ranges.tpi.recordCount != 0)
      source->mergedTpi.deferredRecords =
          makeRecordProvider(deviceTpiBytes, tpiRecords, ranges.tpi, tpiHashes);
    if (ranges.ipi.recordCount != 0)
      source->mergedIpi.deferredRecords =
          makeRecordProvider(deviceIpiBytes, ipiRecords, ranges.ipi, ipiHashes);
  }

  for (size_t i = 0, e = funcPairs.size(); i != e; ++i) {
    FuncIdToTypeEntry pair = funcPairs[i];
    if (pair.funcId == 0)
      continue;
    TpiSource *source = ctx.tpiSourceList[ipiRecords.descs[i].sourceIndex];
    source->funcIdToType.push_back(
        {TypeIndex(pair.funcId), TypeIndex(pair.funcType)});
  }
}

} // namespace

bool TypeMerger::mergeTypesWithCUDA() {
  constexpr uint32_t notTranslated =
      static_cast<uint32_t>(llvm::codeview::SimpleTypeKind::NotTranslated);

  // Flatten the ragged 2D TpiSource x type-index grid into a flat,
  // one-dimensional index space. mapOffsets stores each source row's base
  // offset. The hash array is parallel to the source-position array, and each
  // source position carries the original row, original column, and TPI-vs-IPI
  // bit needed after sorting.
  FlatIndex totalMapSize = 0;
  FlatIndex entryCount = 0;
  FlatIndex leafKindCount = 0;
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
    if (getSourceItemMode(*source) == SourceItemMode::LeafKinds)
      leafKindCount += source->ghashes.size();
    mapOffsets.push_back(totalMapSize);
  }

  int device = 0;
  CudaErrorChecker cuErr(ctx);
  cudaError_t deviceErr = cudaGetDevice(&device);
  if (!cuErr.check(deviceErr))
    Fatal(ctx) << "-lldcudaghash requires a usable CUDA device: "
               << cuErr.message();

  // Allocate the dense device input arrays before host staging. Hashes are
  // copied directly from each source's existing ghash storage into the final
  // device layout, avoiding an extra flat host hash vector.
  thrust::device_vector<uint64_t> deviceHashes(entryCount);
  thrust::device_vector<uint64_t> deviceSrcs(entryCount);
  thrust::device_vector<uint16_t> deviceLeafKinds(
      checkedUInt32(ctx, leafKindCount, "CUDA leaf kind count"));
  uint16_t *deviceLeafKindData = deviceData(deviceLeafKinds);

  std::vector<SourceDescriptor> sourceDescriptors(ctx.tpiSourceList.size());
  MemcpyBatch setupCopies;
  setupCopies.reserve(ctx.tpiSourceList.size() * 3);
  uint32_t leafKindOffset = 0;
  for (size_t i = 0, e = ctx.tpiSourceList.size(); i != e; ++i) {
    TpiSource *source = ctx.tpiSourceList[i];
    source->indexMapStorage.assign(
        source->ghashes.size(),
        TypeIndex(llvm::codeview::SimpleTypeKind::NotTranslated));

    appendSourceGHashCopies(*source, entryOffsets[i], deviceHashes,
                            setupCopies);

    SourceDescriptor &desc = sourceDescriptors[i];
    desc.tpiSrcIdx = source->tpiSrcIdx;
    desc.ghashCount = static_cast<uint32_t>(source->ghashes.size());
    desc.endPrecompIdx = source->endPrecompIdx;
    desc.itemMode = getSourceItemMode(*source);

    if (desc.itemMode == SourceItemMode::LeafKinds) {
      if (source->typeLeafKinds.size() != source->ghashes.size())
        Fatal(ctx) << "-lldcudaghash setup failed: missing type leaf metadata "
                      "for "
                   << getTpiSourceName(*source);
      desc.leafKindOffset = leafKindOffset;
      if (!source->typeLeafKinds.empty()) {
        setupCopies.append(deviceLeafKindData + leafKindOffset,
                           source->typeLeafKinds.data(),
                           source->typeLeafKinds.size() * sizeof(uint16_t));
        leafKindOffset += static_cast<uint32_t>(source->typeLeafKinds.size());
      }
    }
  }
  assert(leafKindOffset == leafKindCount &&
         "source leaf kind copies must fill the device buffer");
  enqueueMemcpyBatchAndSync(setupCopies,
                            "-lldcudaghash failed to enqueue setup data copy "
                            "batch",
                            "-lldcudaghash failed while copying setup data to "
                            "the device",
                            cuErr);

  thrust::device_vector<SourceDescriptor> deviceSourceDescriptors(
      sourceDescriptors.begin(), sourceDescriptors.end());
  thrust::device_vector<FlatIndex> deviceEntryOffsets(entryOffsets.begin(),
                                                      entryOffsets.end());
  if (!sourceDescriptors.empty()) {
    buildSourceCells<<<static_cast<uint32_t>(sourceDescriptors.size()), 256>>>(
        thrust::raw_pointer_cast(deviceSourceDescriptors.data()),
        thrust::raw_pointer_cast(deviceEntryOffsets.data()), deviceLeafKindData,
        thrust::raw_pointer_cast(deviceSrcs.data()),
        static_cast<uint32_t>(sourceDescriptors.size()));
    cuErr.fatal(cudaGetLastError(),
                "-lldcudaghash failed to launch source-cell kernel");
    cuErr.fatal(cudaDeviceSynchronize(),
                "-lldcudaghash failed while synchronizing source-cell kernel");
  }

  GHashCUDAState ghashState;
  mergeGHashesWithCUDA(ctx, deviceHashes, deviceSrcs, mapOffsets.data(),
                       mapOffsets.size(), totalMapSize, notTranslated,
                       ctx.tpiSourceList, &ghashState, cuErr);

  Log(ctx) << "CUDA ghash record count: " << ghashState.result.uniqueCount
           << " / input " << totalMapSize;
  Log(ctx) << "Tpi record count: " << ghashState.result.numTypes;
  Log(ctx) << "Ipi record count: " << ghashState.result.numItems;

  remapAllSelectedTypeRecordsWithCUDA(ctx, mapOffsets, ghashState, cuErr);

  for (TpiSource *source : ctx.tpiSourceList) {
    funcIdToType.insert_range(source->funcIdToType);
    source->funcIdToType.clear();
  }

  clearGHashes();
  return true;
}

} // namespace lld::coff
