//===- TypeIndexDiscovery.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Endian.h"
#include <cstring>

using namespace llvm;
using namespace llvm::codeview;

static inline MethodKind getMethodKind(uint16_t Attrs) {
  Attrs &= uint16_t(MethodOptions::MethodKindMask);
  Attrs >>= 2;
  return MethodKind(Attrs);
}

static inline bool isIntroVirtual(uint16_t Attrs) {
  MethodKind MK = getMethodKind(Attrs);
  return MK == MethodKind::IntroducingVirtual ||
         MK == MethodKind::PureIntroducingVirtual;
}

static inline uint32_t getEncodedIntegerLength(ArrayRef<uint8_t> Data) {
  uint16_t N = support::endian::read16le(Data.data());
  if (N < LF_NUMERIC)
    return 2;

  assert(N <= LF_UQUADWORD);

  constexpr uint32_t Sizes[] = {
      1,  // LF_CHAR
      2,  // LF_SHORT
      2,  // LF_USHORT
      4,  // LF_LONG
      4,  // LF_ULONG
      4,  // LF_REAL32
      8,  // LF_REAL64
      10, // LF_REAL80
      16, // LF_REAL128
      8,  // LF_QUADWORD
      8,  // LF_UQUADWORD
  };

  return 2 + Sizes[N - LF_NUMERIC];
}

static inline uint32_t getCStringLength(ArrayRef<uint8_t> Data) {
  const char *S = reinterpret_cast<const char *>(Data.data());
  return strlen(S) + 1;
}

static void handleMethodOverloadList(ArrayRef<uint8_t> Content,
                                     TypeIndexRefCallback RefFn) {
  uint32_t Offset = 0;

  while (!Content.empty()) {
    // Array of:
    //   0: OneMethodRecord::Attrs
    //   2: Padding
    //   4: OneMethodRecord::Type
    //   if (isIntroVirtual())
    //     8: OneMethodRecord::VFTableOffset

    // At least 8 bytes are guaranteed.  4 extra bytes come iff function is an
    // intro virtual.
    uint32_t Len = 8;

    uint16_t Attrs = support::endian::read16le(Content.data());
    RefFn(TiRefKind::TypeRef, Offset + 4);

    if (LLVM_UNLIKELY(isIntroVirtual(Attrs)))
      Len += 4;
    Offset += Len;
    Content = Content.drop_front(Len);
  }
}

static uint32_t handleBaseClass(ArrayRef<uint8_t> Data, uint32_t Offset,
                                TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: BaseClassRecord::Attrs
  // 4: BaseClassRecord::Type
  // 8: BaseClassRecord::Offset
  RefFn(TiRefKind::TypeRef, Offset + 4);
  return 8 + getEncodedIntegerLength(Data.drop_front(8));
}

static uint32_t handleEnumerator(ArrayRef<uint8_t> Data, uint32_t Offset,
                                 TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: EnumeratorRecord::Attrs
  // 4: EnumeratorRecord::Value
  // <next>: EnumeratorRecord::Name
  uint32_t Size = 4 + getEncodedIntegerLength(Data.drop_front(4));
  return Size + getCStringLength(Data.drop_front(Size));
}

static uint32_t handleDataMember(ArrayRef<uint8_t> Data, uint32_t Offset,
                                 TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: DataMemberRecord::Attrs
  // 4: DataMemberRecord::Type
  // 8: DataMemberRecord::FieldOffset
  // <next>: DataMemberRecord::Name
  RefFn(TiRefKind::TypeRef, Offset + 4);
  uint32_t Size = 8 + getEncodedIntegerLength(Data.drop_front(8));
  return Size + getCStringLength(Data.drop_front(Size));
}

static uint32_t handleOverloadedMethod(ArrayRef<uint8_t> Data, uint32_t Offset,
                                       TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: OverloadedMethodRecord::NumOverloads
  // 4: OverloadedMethodRecord::MethodList
  // 8: OverloadedMethodRecord::Name
  RefFn(TiRefKind::TypeRef, Offset + 4);
  return 8 + getCStringLength(Data.drop_front(8));
}

static uint32_t handleOneMethod(ArrayRef<uint8_t> Data, uint32_t Offset,
                                TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: OneMethodRecord::Attrs
  // 4: OneMethodRecord::Type
  // if (isIntroVirtual)
  //   8: OneMethodRecord::VFTableOffset
  // <next>: OneMethodRecord::Name
  uint32_t Size = 8;
  RefFn(TiRefKind::TypeRef, Offset + 4);

  uint16_t Attrs = support::endian::read16le(Data.drop_front(2).data());
  if (LLVM_UNLIKELY(isIntroVirtual(Attrs)))
    Size += 4;

  return Size + getCStringLength(Data.drop_front(Size));
}

static uint32_t handleNestedType(ArrayRef<uint8_t> Data, uint32_t Offset,
                                 TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: Padding
  // 4: NestedTypeRecord::Type
  // 8: NestedTypeRecord::Name
  RefFn(TiRefKind::TypeRef, Offset + 4);
  return 8 + getCStringLength(Data.drop_front(8));
}

static uint32_t handleStaticDataMember(ArrayRef<uint8_t> Data, uint32_t Offset,
                                       TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: StaticDataMemberRecord::Attrs
  // 4: StaticDataMemberRecord::Type
  // 8: StaticDataMemberRecord::Name
  RefFn(TiRefKind::TypeRef, Offset + 4);
  return 8 + getCStringLength(Data.drop_front(8));
}

static uint32_t handleVirtualBaseClass(ArrayRef<uint8_t> Data, uint32_t Offset,
                                       TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: VirtualBaseClassRecord::Attrs
  // 4: VirtualBaseClassRecord::BaseType
  // 8: VirtualBaseClassRecord::VBPtrType
  // 12: VirtualBaseClassRecord::VBPtrOffset
  // <next>: VirtualBaseClassRecord::VTableIndex
  uint32_t Size = 12;
  RefFn(TiRefKind::TypeRef, Offset + 4);
  RefFn(TiRefKind::TypeRef, Offset + 8);
  Size += getEncodedIntegerLength(Data.drop_front(Size));
  Size += getEncodedIntegerLength(Data.drop_front(Size));
  return Size;
}

static uint32_t handleVFPtr(ArrayRef<uint8_t> Data, uint32_t Offset,
                            TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: Padding
  // 4: VFPtrRecord::Type
  RefFn(TiRefKind::TypeRef, Offset + 4);
  return 8;
}

static uint32_t handleListContinuation(ArrayRef<uint8_t> Data, uint32_t Offset,
                                       TypeIndexRefCallback RefFn) {
  // 0: Kind
  // 2: Padding
  // 4: ListContinuationRecord::ContinuationIndex
  RefFn(TiRefKind::TypeRef, Offset + 4);
  return 8;
}

static void handleFieldList(ArrayRef<uint8_t> Content,
                            TypeIndexRefCallback RefFn) {
  uint32_t Offset = 0;
  uint32_t ThisLen = 0;
  while (!Content.empty()) {
    TypeLeafKind Kind =
        static_cast<TypeLeafKind>(support::endian::read16le(Content.data()));
    switch (Kind) {
    case LF_BCLASS:
      ThisLen = handleBaseClass(Content, Offset, RefFn);
      break;
    case LF_ENUMERATE:
      ThisLen = handleEnumerator(Content, Offset, RefFn);
      break;
    case LF_MEMBER:
      ThisLen = handleDataMember(Content, Offset, RefFn);
      break;
    case LF_METHOD:
      ThisLen = handleOverloadedMethod(Content, Offset, RefFn);
      break;
    case LF_ONEMETHOD:
      ThisLen = handleOneMethod(Content, Offset, RefFn);
      break;
    case LF_NESTTYPE:
      ThisLen = handleNestedType(Content, Offset, RefFn);
      break;
    case LF_STMEMBER:
      ThisLen = handleStaticDataMember(Content, Offset, RefFn);
      break;
    case LF_VBCLASS:
    case LF_IVBCLASS:
      ThisLen = handleVirtualBaseClass(Content, Offset, RefFn);
      break;
    case LF_VFUNCTAB:
      ThisLen = handleVFPtr(Content, Offset, RefFn);
      break;
    case LF_INDEX:
      ThisLen = handleListContinuation(Content, Offset, RefFn);
      break;
    default:
      return;
    }
    Content = Content.drop_front(ThisLen);
    Offset += ThisLen;
    if (!Content.empty()) {
      uint8_t Pad = Content.front();
      if (Pad >= LF_PAD0) {
        uint32_t Skip = Pad & 0x0F;
        Content = Content.drop_front(Skip);
        Offset += Skip;
      }
    }
  }
}

static void handleMemberFunction(TypeIndexRefCallback RefFn) {
  // 0: MemberFunctionRecord::ReturnType
  // 4: MemberFunctionRecord::ClassType
  // 8: MemberFunctionRecord::ThisType
  // 12: CallingConvention, FunctionOptions, ParameterCount
  // 16: MemberFunctionRecord::ArgumentList
  RefFn(TiRefKind::TypeRef, 0);
  RefFn(TiRefKind::TypeRef, 4);
  RefFn(TiRefKind::TypeRef, 8);
  RefFn(TiRefKind::TypeRef, 16);
}

void llvm::codeview::discoverTypeIndicesInComplexType(
    ArrayRef<uint8_t> Content, TypeLeafKind Kind, TypeIndexRefCallback RefFn) {
  switch (Kind) {
  case TypeLeafKind::LF_MFUNCTION:
    handleMemberFunction(RefFn);
    break;
  case TypeLeafKind::LF_METHODLIST:
    handleMethodOverloadList(Content, RefFn);
    break;
  case TypeLeafKind::LF_FIELDLIST:
    handleFieldList(Content, RefFn);
    break;
  default:
    break;
  }
}
