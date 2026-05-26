//===- TypeIndexDiscovery.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPEINDEXDISCOVERY_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPEINDEXDISCOVERY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include <cstdint>
#include <utility>

namespace llvm {
namespace codeview {
enum class TiRefKind { TypeRef, IndexRef };

using TypeIndexRefCallback = function_ref<void(TiRefKind, uint32_t)>;

LLVM_ABI void discoverTypeIndicesInComplexType(ArrayRef<uint8_t> Content,
                                               TypeLeafKind Kind,
                                               TypeIndexRefCallback RefFn);

namespace detail {
constexpr uint32_t NoTypeIndexOffset = ~0U;

// Keep the common case branch-light and allocation-free. Most records have
// zero, one, two, or one contiguous run of type index references, so the
// discovery switch below fills two single-reference slots plus one run and this
// helper invokes the callback without materializing a TiReference vector.
template <typename Func>
inline void visitTypeIndexRefs(TiRefKind Kind0, uint32_t Offset0,
                               TiRefKind Kind1, uint32_t Offset1,
                               TiRefKind RunKind, uint32_t RunOffset,
                               uint32_t RunCount, Func &&RefFn) {
  if (Offset0 != NoTypeIndexOffset)
    RefFn(Kind0, Offset0);
  if (Offset1 != NoTypeIndexOffset)
    RefFn(Kind1, Offset1);
  if (RunOffset != NoTypeIndexOffset) {
    for (uint32_t I = 0; I != RunCount; ++I)
      RefFn(RunKind, RunOffset + I * sizeof(TypeIndex));
  }
}

inline PointerMode getPointerMode(uint32_t Attrs) {
  return static_cast<PointerMode>((Attrs >> PointerRecord::PointerModeShift) &
                                  PointerRecord::PointerModeMask);
}

inline bool isMemberPointer(uint32_t Attrs) {
  PointerMode Mode = getPointerMode(Attrs);
  return Mode == PointerMode::PointerToDataMember ||
         Mode == PointerMode::PointerToMemberFunction;
}
} // namespace detail

template <typename Func>
inline void discoverTypeIndices(ArrayRef<uint8_t> Content, TypeLeafKind Kind,
                                Func &&RefFn) {
  // This switch is deliberately shaped to describe the hot fixed-layout type
  // records as two individual offsets plus an optional contiguous run. That
  // lets callers process the common 0/1/2-reference cases without a dynamic
  // vector and handles arg lists or adjacent fields with a single simple loop.
  uint32_t RunCount = 0;
  TiRefKind Kind0 = TiRefKind::TypeRef;
  TiRefKind Kind1 = TiRefKind::TypeRef;
  TiRefKind RunKind = TiRefKind::TypeRef;
  uint32_t Offset0 = detail::NoTypeIndexOffset;
  uint32_t Offset1 = detail::NoTypeIndexOffset;
  uint32_t RunOffset = detail::NoTypeIndexOffset;

  // FIXME: In the future it would be nice if we could avoid hardcoding these
  // values.  One idea is to define some structures representing these types
  // that would allow the use of offsetof().
  switch (Kind) {
  case TypeLeafKind::LF_FUNC_ID:
    Kind0 = TiRefKind::IndexRef;
    Offset0 = 0; // FuncIdRecord::ParentScope
    Kind1 = TiRefKind::TypeRef;
    Offset1 = 4; // FuncIdRecord::FunctionType
    break;
  case TypeLeafKind::LF_MFUNC_ID:
    Offset0 = 0; // MemberFuncIdRecord::ClassType
    Offset1 = 4; // MemberFuncIdRecord::FunctionType
    break;
  case TypeLeafKind::LF_STRING_ID:
    Kind0 = TiRefKind::IndexRef;
    Offset0 = 0; // StringIdRecord::Id
    break;
  case TypeLeafKind::LF_SUBSTR_LIST:
    RunCount = support::endian::read32le(Content.data());
    if (RunCount > 0) {
      RunKind = TiRefKind::IndexRef;
      RunOffset = 4; // StringListRecord::StringIndices
    }
    break;
  case TypeLeafKind::LF_BUILDINFO:
    RunCount = support::endian::read16le(Content.data());
    if (RunCount > 0) {
      RunKind = TiRefKind::IndexRef;
      RunOffset = 2; // BuildInfoRecord::ArgIndices
    }
    break;
  case TypeLeafKind::LF_UDT_SRC_LINE:
    Offset0 = 0; // UdtSourceLineRecord::UDT
    Kind1 = TiRefKind::IndexRef;
    Offset1 = 4; // UdtSourceLineRecord::SourceFile
    break;
  case TypeLeafKind::LF_UDT_MOD_SRC_LINE:
    Offset0 = 0; // UdtModSourceLineRecord::UDT
    Kind1 = TiRefKind::IndexRef;
    Offset1 = 4; // UdtModSourceLineRecord::SourceFile
    break;
  case TypeLeafKind::LF_MODIFIER:
    Offset0 = 0; // ModifierRecord::ModifiedType
    break;
  case TypeLeafKind::LF_PROCEDURE:
    Offset0 = 0; // ProcedureRecord::ReturnType
    Offset1 = 8; // ProcedureRecord::ArgumentList
    break;
  case TypeLeafKind::LF_ARGLIST:
    RunCount = support::endian::read32le(Content.data());
    if (RunCount > 0)
      RunOffset = 4; // ArgListRecord::ArgIndices
    break;
  case TypeLeafKind::LF_ARRAY:
    Offset0 = 0; // ArrayRecord::ElementType
    Offset1 = 4; // ArrayRecord::IndexType
    break;
  case TypeLeafKind::LF_CLASS:
  case TypeLeafKind::LF_STRUCTURE:
  case TypeLeafKind::LF_INTERFACE:
    RunOffset = 4; // ClassRecord::{FieldList, DerivationList, VTableShape}
    RunCount = 3;
    break;
  case TypeLeafKind::LF_UNION:
    Offset0 = 4; // UnionRecord::FieldList
    break;
  case TypeLeafKind::LF_ENUM:
    Offset0 = 4; // EnumRecord::FieldList
    Offset1 = 8; // EnumRecord::UnderlyingType
    break;
  case TypeLeafKind::LF_BITFIELD:
    Offset0 = 0; // BitFieldRecord::Type
    break;
  case TypeLeafKind::LF_VFTABLE:
    Offset0 = 0; // VFTableRecord::CompleteClass
    Offset1 = 4; // VFTableRecord::OverriddenVFTable
    break;
  case TypeLeafKind::LF_VTSHAPE:
    break;
  case TypeLeafKind::LF_POINTER: {
    Offset0 = 0; // PointerRecord::ReferentType
    uint32_t Attrs = support::endian::read32le(Content.drop_front(4).data());
    if (detail::isMemberPointer(Attrs))
      Offset1 = 8; // MemberPointerInfo::ContainingType
    break;
  }
  case TypeLeafKind::LF_MFUNCTION:
  case TypeLeafKind::LF_METHODLIST:
  case TypeLeafKind::LF_FIELDLIST:
    discoverTypeIndicesInComplexType(Content, Kind, std::forward<Func>(RefFn));
    return;
  default:
    break;
  }

  detail::visitTypeIndexRefs(Kind0, Offset0, Kind1, Offset1, RunKind, RunOffset,
                             RunCount, std::forward<Func>(RefFn));
}

template <typename Func>
inline void discoverTypeIndices(ArrayRef<uint8_t> RecordData, Func &&RefFn) {
  const RecordPrefix *P =
      reinterpret_cast<const RecordPrefix *>(RecordData.data());
  TypeLeafKind K = static_cast<TypeLeafKind>(uint16_t(P->RecordKind));
  discoverTypeIndices(RecordData.drop_front(sizeof(RecordPrefix)), K,
                      std::forward<Func>(RefFn));
}

template <typename Func>
inline void discoverTypeIndices(const CVType &Type, Func &&RefFn) {
  discoverTypeIndices(Type.content(), Type.kind(), std::forward<Func>(RefFn));
}

/// Discover type indices in symbol records. Returns false if this is an unknown
/// record.
template <typename Func>
inline bool discoverTypeIndicesInSymbol(ArrayRef<uint8_t> Content,
                                        SymbolKind Kind, Func &&RefFn) {
  // Symbol records are especially hot while writing PDB module streams. Keep
  // their fixed-layout discovery inline and summarize each record as two
  // individual offsets plus an optional contiguous run, avoiding the old
  // pseudo-relocation vector in the overwhelmingly common cases.
  uint32_t RunCount = 0;
  TiRefKind Kind0 = TiRefKind::TypeRef;
  TiRefKind Kind1 = TiRefKind::TypeRef;
  TiRefKind RunKind = TiRefKind::TypeRef;
  uint32_t Offset0 = detail::NoTypeIndexOffset;
  uint32_t Offset1 = detail::NoTypeIndexOffset;
  uint32_t RunOffset = detail::NoTypeIndexOffset;

  // FIXME: In the future it would be nice if we could avoid hardcoding these
  // values.  One idea is to define some structures representing these types
  // that would allow the use of offsetof().
  switch (Kind) {
  case SymbolKind::S_GPROC32_ID:
  case SymbolKind::S_LPROC32_ID:
  case SymbolKind::S_LPROC32_DPC:
  case SymbolKind::S_LPROC32_DPC_ID:
    Kind0 = TiRefKind::IndexRef;
    Offset0 = 24; // ProcSym::FunctionType
    break;
  case SymbolKind::S_GPROC32:
  case SymbolKind::S_LPROC32:
    Offset0 = 24; // ProcSym::FunctionType
    break;
  case SymbolKind::S_UDT:
    Offset0 = 0; // UDTSym::Type
    break;
  case SymbolKind::S_GDATA32:
  case SymbolKind::S_LDATA32:
    Offset0 = 0; // DataSym::Type
    break;
  case SymbolKind::S_LTHREAD32:
  case SymbolKind::S_GTHREAD32:
    Offset0 = 0; // ThreadLocalDataSym::Type
    break;
  case SymbolKind::S_FILESTATIC:
    Offset0 = 0; // FileStaticSym::Index
    break;
  case SymbolKind::S_LOCAL:
    Offset0 = 0; // LocalSym::Type
    break;
  case SymbolKind::S_REGISTER:
    Offset0 = 0; // RegisterSym::Index
    break;
  case SymbolKind::S_CONSTANT:
    Offset0 = 0; // ConstantSym::Type
    break;
  case SymbolKind::S_BUILDINFO:
    Kind0 = TiRefKind::IndexRef;
    Offset0 = 0; // BuildInfoSym::BuildId
    break;
  case SymbolKind::S_BPREL32:
    Offset0 = 4; // BPRelativeSym::Type
    break;
  case SymbolKind::S_REGREL32:
    Offset0 = 4; // RegRelativeSym::Type
    break;
  case SymbolKind::S_REGREL32_INDIR:
    Offset0 = 4; // RegRelativeIndirSym::Type
    break;
  case SymbolKind::S_CALLSITEINFO:
    Offset0 = 8; // CallSiteInfoSym::Type
    break;
  case SymbolKind::S_CALLERS:
  case SymbolKind::S_CALLEES:
  case SymbolKind::S_INLINEES:
    // The record is a count followed by an array of type indices.
    RunCount = support::endian::read32le(Content.data());
    RunKind = TiRefKind::IndexRef;
    RunOffset = 4; // CallerSym::Indices
    break;
  case SymbolKind::S_INLINESITE:
    Kind0 = TiRefKind::IndexRef;
    Offset0 = 8; // InlineSiteSym::Inlinee
    break;
  case SymbolKind::S_HEAPALLOCSITE:
    Offset0 = 8; // HeapAllocationSiteSym::Type
    break;

  // Defranges don't have types, just registers and code offsets.
  case SymbolKind::S_DEFRANGE_REGISTER:
  case SymbolKind::S_DEFRANGE_REGISTER_REL:
  case SymbolKind::S_DEFRANGE_REGISTER_REL_INDIR:
  case SymbolKind::S_DEFRANGE_FRAMEPOINTER_REL:
  case SymbolKind::S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE:
  case SymbolKind::S_DEFRANGE_SUBFIELD_REGISTER:
  case SymbolKind::S_DEFRANGE_SUBFIELD:
    break;

  // No type references.
  case SymbolKind::S_LABEL32:
  case SymbolKind::S_OBJNAME:
  case SymbolKind::S_COMPILE:
  case SymbolKind::S_COMPILE2:
  case SymbolKind::S_COMPILE3:
  case SymbolKind::S_ENVBLOCK:
  case SymbolKind::S_BLOCK32:
  case SymbolKind::S_FRAMEPROC:
  case SymbolKind::S_THUNK32:
  case SymbolKind::S_FRAMECOOKIE:
  case SymbolKind::S_UNAMESPACE:
  case SymbolKind::S_ARMSWITCHTABLE:
    break;
  // Scope ending symbols.
  case SymbolKind::S_END:
  case SymbolKind::S_INLINESITE_END:
  case SymbolKind::S_PROC_ID_END:
    break;
  default:
    return false; // Unknown symbol.
  }

  detail::visitTypeIndexRefs(Kind0, Offset0, Kind1, Offset1, RunKind, RunOffset,
                             RunCount, std::forward<Func>(RefFn));
  return true;
}

template <typename Func>
inline bool discoverTypeIndicesInSymbol(ArrayRef<uint8_t> RecordData,
                                        Func &&RefFn) {
  const RecordPrefix *P =
      reinterpret_cast<const RecordPrefix *>(RecordData.data());
  SymbolKind K = static_cast<SymbolKind>(uint16_t(P->RecordKind));
  return discoverTypeIndicesInSymbol(
      RecordData.drop_front(sizeof(RecordPrefix)), K,
      std::forward<Func>(RefFn));
}

template <typename Func>
inline bool discoverTypeIndicesInSymbol(const CVSymbol &Symbol, Func &&RefFn) {
  return discoverTypeIndicesInSymbol(Symbol.content(), Symbol.kind(),
                                     std::forward<Func>(RefFn));
}
} // namespace codeview
} // namespace llvm

#endif
