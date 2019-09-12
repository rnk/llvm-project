//===-- X86DAGCallLowering.cpp - X86 DAG call and return lowering ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analogous to X86CallLowering.cpp, but for selection DAG ISel.
//
//===----------------------------------------------------------------------===//

#include "Utils/X86ShuffleDecode.h"
#include "X86CallingConv.h"
#include "X86ISelLowering.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

using llvm::X86::extract128BitVector;
using llvm::X86::extractSubVector;
using llvm::X86::getConstVector;
using llvm::X86::getExtendInVec;
using llvm::X86::getOpcode_EXTEND_VECTOR_INREG;
using llvm::X86::getPackDemandedElts;
using llvm::X86::getShuffleVectorZeroOrUndef;
using llvm::X86::getTargetConstantBitsFromNode;
using llvm::X86::getTargetShuffleMask;
using llvm::X86::getUnpackl;
using llvm::X86::getZeroVector;
using llvm::X86::insertSubVector;
using llvm::X86::isTargetShuffle;
using llvm::X86::MayFoldIntoStore;
using llvm::X86::MayFoldIntoZeroExtend;
using llvm::X86::MayFoldLoad;
using llvm::X86::getVectorMaskingNode;

#define DEBUG_TYPE "x86-isel"

extern cl::opt<bool> ExperimentalVectorWideningLegalization;

bool X86::isTargetShuffle(unsigned Opcode) {
  switch(Opcode) {
  default: return false;
  case X86ISD::BLENDI:
  case X86ISD::PSHUFB:
  case X86ISD::PSHUFD:
  case X86ISD::PSHUFHW:
  case X86ISD::PSHUFLW:
  case X86ISD::SHUFP:
  case X86ISD::INSERTPS:
  case X86ISD::EXTRQI:
  case X86ISD::INSERTQI:
  case X86ISD::PALIGNR:
  case X86ISD::VSHLDQ:
  case X86ISD::VSRLDQ:
  case X86ISD::MOVLHPS:
  case X86ISD::MOVHLPS:
  case X86ISD::MOVSHDUP:
  case X86ISD::MOVSLDUP:
  case X86ISD::MOVDDUP:
  case X86ISD::MOVSS:
  case X86ISD::MOVSD:
  case X86ISD::UNPCKL:
  case X86ISD::UNPCKH:
  case X86ISD::VBROADCAST:
  case X86ISD::VPERMILPI:
  case X86ISD::VPERMILPV:
  case X86ISD::VPERM2X128:
  case X86ISD::SHUF128:
  case X86ISD::VPERMIL2:
  case X86ISD::VPERMI:
  case X86ISD::VPPERM:
  case X86ISD::VPERMV:
  case X86ISD::VPERMV3:
  case X86ISD::VZEXT_MOVL:
    return true;
  }
}

static bool isTargetShuffleVariableMask(unsigned Opcode) {
  switch (Opcode) {
  default: return false;
  // Target Shuffles.
  case X86ISD::PSHUFB:
  case X86ISD::VPERMILPV:
  case X86ISD::VPERMIL2:
  case X86ISD::VPPERM:
  case X86ISD::VPERMV:
  case X86ISD::VPERMV3:
    return true;
  // 'Faux' Target Shuffles.
  case ISD::OR:
  case ISD::AND:
  case X86ISD::ANDNP:
    return true;
  }
}

/// Val is the undef sentinel value or equal to the specified value.
static bool isUndefOrEqual(int Val, int CmpVal) {
  return ((Val == SM_SentinelUndef) || (Val == CmpVal));
}

/// Val is either the undef or zero sentinel value.
static bool isUndefOrZero(int Val) {
  return ((Val == SM_SentinelUndef) || (Val == SM_SentinelZero));
}

/// Return true if every element in Mask, beginning from position Pos and ending
/// in Pos+Size is the undef sentinel value.
static bool isUndefInRange(ArrayRef<int> Mask, unsigned Pos, unsigned Size) {
  for (unsigned i = Pos, e = Pos + Size; i != e; ++i)
    if (Mask[i] != SM_SentinelUndef)
      return false;
  return true;
}

/// Return true if the mask creates a vector whose lower half is undefined.
static bool isUndefLowerHalf(ArrayRef<int> Mask) {
  unsigned NumElts = Mask.size();
  return isUndefInRange(Mask, 0, NumElts / 2);
}

/// Return true if the mask creates a vector whose upper half is undefined.
static bool isUndefUpperHalf(ArrayRef<int> Mask) {
  unsigned NumElts = Mask.size();
  return isUndefInRange(Mask, NumElts / 2, NumElts / 2);
}

/// Return true if Val falls within the specified range (L, H].
static bool isInRange(int Val, int Low, int Hi) {
  return (Val >= Low && Val < Hi);
}

/// Return true if the value of any element in Mask falls within the specified
/// range (L, H].
static bool isAnyInRange(ArrayRef<int> Mask, int Low, int Hi) {
  for (int M : Mask)
    if (isInRange(M, Low, Hi))
      return true;
  return false;
}

/// Return true if Val is undef or if its value falls within the
/// specified range (L, H].
static bool isUndefOrInRange(int Val, int Low, int Hi) {
  return (Val == SM_SentinelUndef) || isInRange(Val, Low, Hi);
}

/// Return true if every element in Mask is undef or if its value
/// falls within the specified range (L, H].
static bool isUndefOrInRange(ArrayRef<int> Mask,
                             int Low, int Hi) {
  for (int M : Mask)
    if (!isUndefOrInRange(M, Low, Hi))
      return false;
  return true;
}

/// Return true if Val is undef, zero or if its value falls within the
/// specified range (L, H].
static bool isUndefOrZeroOrInRange(int Val, int Low, int Hi) {
  return isUndefOrZero(Val) || isInRange(Val, Low, Hi);
}

/// Return true if every element in Mask is undef, zero or if its value
/// falls within the specified range (L, H].
static bool isUndefOrZeroOrInRange(ArrayRef<int> Mask, int Low, int Hi) {
  for (int M : Mask)
    if (!isUndefOrZeroOrInRange(M, Low, Hi))
      return false;
  return true;
}

/// Return true if every element in Mask, beginning
/// from position Pos and ending in Pos + Size, falls within the specified
/// sequence (Low, Low + Step, ..., Low + (Size - 1) * Step) or is undef.
static bool isSequentialOrUndefInRange(ArrayRef<int> Mask, unsigned Pos,
                                       unsigned Size, int Low, int Step = 1) {
  for (unsigned i = Pos, e = Pos + Size; i != e; ++i, Low += Step)
    if (!isUndefOrEqual(Mask[i], Low))
      return false;
  return true;
}

/// Return true if every element in Mask, beginning
/// from position Pos and ending in Pos+Size, falls within the specified
/// sequential range (Low, Low+Size], or is undef or is zero.
static bool isSequentialOrUndefOrZeroInRange(ArrayRef<int> Mask, unsigned Pos,
                                             unsigned Size, int Low) {
  for (unsigned i = Pos, e = Pos + Size; i != e; ++i, ++Low)
    if (!isUndefOrZero(Mask[i]) && Mask[i] != Low)
      return false;
  return true;
}

/// Return true if every element in Mask, beginning
/// from position Pos and ending in Pos+Size is undef or is zero.
static bool isUndefOrZeroInRange(ArrayRef<int> Mask, unsigned Pos,
                                 unsigned Size) {
  for (unsigned i = Pos, e = Pos + Size; i != e; ++i)
    if (!isUndefOrZero(Mask[i]))
      return false;
  return true;
}

/// Helper function to test whether a shuffle mask could be
/// simplified by widening the elements being shuffled.
///
/// Appends the mask for wider elements in WidenedMask if valid. Otherwise
/// leaves it in an unspecified state.
///
/// NOTE: This must handle normal vector shuffle masks and *target* vector
/// shuffle masks. The latter have the special property of a '-2' representing
/// a zero-ed lane of a vector.
bool X86::canWidenShuffleElements(ArrayRef<int> Mask,
                                  SmallVectorImpl<int> &WidenedMask) {
  WidenedMask.assign(Mask.size() / 2, 0);
  for (int i = 0, Size = Mask.size(); i < Size; i += 2) {
    int M0 = Mask[i];
    int M1 = Mask[i + 1];

    // If both elements are undef, its trivial.
    if (M0 == SM_SentinelUndef && M1 == SM_SentinelUndef) {
      WidenedMask[i / 2] = SM_SentinelUndef;
      continue;
    }

    // Check for an undef mask and a mask value properly aligned to fit with
    // a pair of values. If we find such a case, use the non-undef mask's value.
    if (M0 == SM_SentinelUndef && M1 >= 0 && (M1 % 2) == 1) {
      WidenedMask[i / 2] = M1 / 2;
      continue;
    }
    if (M1 == SM_SentinelUndef && M0 >= 0 && (M0 % 2) == 0) {
      WidenedMask[i / 2] = M0 / 2;
      continue;
    }

    // When zeroing, we need to spread the zeroing across both lanes to widen.
    if (M0 == SM_SentinelZero || M1 == SM_SentinelZero) {
      if ((M0 == SM_SentinelZero || M0 == SM_SentinelUndef) &&
          (M1 == SM_SentinelZero || M1 == SM_SentinelUndef)) {
        WidenedMask[i / 2] = SM_SentinelZero;
        continue;
      }
      return false;
    }

    // Finally check if the two mask values are adjacent and aligned with
    // a pair.
    if (M0 != SM_SentinelUndef && (M0 % 2) == 0 && (M0 + 1) == M1) {
      WidenedMask[i / 2] = M0 / 2;
      continue;
    }

    // Otherwise we can't safely widen the elements used in this shuffle.
    return false;
  }
  assert(WidenedMask.size() == Mask.size() / 2 &&
         "Incorrect size of mask after widening the elements!");

  return true;
}

bool X86::canWidenShuffleElements(ArrayRef<int> Mask, const APInt &Zeroable,
                                  SmallVectorImpl<int> &WidenedMask) {
  SmallVector<int, 32> TargetMask(Mask.begin(), Mask.end());
  for (int i = 0, Size = TargetMask.size(); i < Size; ++i) {
    if (TargetMask[i] == SM_SentinelUndef)
      continue;
    if (Zeroable[i])
      TargetMask[i] = SM_SentinelZero;
  }
  return canWidenShuffleElements(TargetMask, WidenedMask);
}

bool X86::canWidenShuffleElements(ArrayRef<int> Mask) {
  SmallVector<int, 32> WidenedMask;
  return canWidenShuffleElements(Mask, WidenedMask);
}

/// Returns true if Elt is a constant zero or a floating point constant +0.0.
bool X86::isZeroNode(SDValue Elt) {
  return isNullConstant(Elt) || isNullFPConstant(Elt);
}

// Build a vector of constants.
// Use an UNDEF node if MaskElt == -1.
// Split 64-bit constants in the 32-bit mode.
SDValue X86::getConstVector(ArrayRef<int> Values, MVT VT, SelectionDAG &DAG,
                            const SDLoc &dl, bool IsMask) {

  SmallVector<SDValue, 32>  Ops;
  bool Split = false;

  MVT ConstVecVT = VT;
  unsigned NumElts = VT.getVectorNumElements();
  bool In64BitMode = DAG.getTargetLoweringInfo().isTypeLegal(MVT::i64);
  if (!In64BitMode && VT.getVectorElementType() == MVT::i64) {
    ConstVecVT = MVT::getVectorVT(MVT::i32, NumElts * 2);
    Split = true;
  }

  MVT EltVT = ConstVecVT.getVectorElementType();
  for (unsigned i = 0; i < NumElts; ++i) {
    bool IsUndef = Values[i] < 0 && IsMask;
    SDValue OpNode = IsUndef ? DAG.getUNDEF(EltVT) :
      DAG.getConstant(Values[i], dl, EltVT);
    Ops.push_back(OpNode);
    if (Split)
      Ops.push_back(IsUndef ? DAG.getUNDEF(EltVT) :
                    DAG.getConstant(0, dl, EltVT));
  }
  SDValue ConstsNode = DAG.getBuildVector(ConstVecVT, dl, Ops);
  if (Split)
    ConstsNode = DAG.getBitcast(VT, ConstsNode);
  return ConstsNode;
}

SDValue X86::getConstVector(ArrayRef<APInt> Bits, APInt &Undefs, MVT VT,
                            SelectionDAG &DAG, const SDLoc &dl) {
  assert(Bits.size() == Undefs.getBitWidth() &&
         "Unequal constant and undef arrays");
  SmallVector<SDValue, 32> Ops;
  bool Split = false;

  MVT ConstVecVT = VT;
  unsigned NumElts = VT.getVectorNumElements();
  bool In64BitMode = DAG.getTargetLoweringInfo().isTypeLegal(MVT::i64);
  if (!In64BitMode && VT.getVectorElementType() == MVT::i64) {
    ConstVecVT = MVT::getVectorVT(MVT::i32, NumElts * 2);
    Split = true;
  }

  MVT EltVT = ConstVecVT.getVectorElementType();
  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Undefs[i]) {
      Ops.append(Split ? 2 : 1, DAG.getUNDEF(EltVT));
      continue;
    }
    const APInt &V = Bits[i];
    assert(V.getBitWidth() == VT.getScalarSizeInBits() && "Unexpected sizes");
    if (Split) {
      Ops.push_back(DAG.getConstant(V.trunc(32), dl, EltVT));
      Ops.push_back(DAG.getConstant(V.lshr(32).trunc(32), dl, EltVT));
    } else if (EltVT == MVT::f32) {
      APFloat FV(APFloat::IEEEsingle(), V);
      Ops.push_back(DAG.getConstantFP(FV, dl, EltVT));
    } else if (EltVT == MVT::f64) {
      APFloat FV(APFloat::IEEEdouble(), V);
      Ops.push_back(DAG.getConstantFP(FV, dl, EltVT));
    } else {
      Ops.push_back(DAG.getConstant(V, dl, EltVT));
    }
  }

  SDValue ConstsNode = DAG.getBuildVector(ConstVecVT, dl, Ops);
  return DAG.getBitcast(VT, ConstsNode);
}

/// Returns a vector of specified type with all zero elements.
SDValue X86::getZeroVector(MVT VT, const X86Subtarget &Subtarget,
                             SelectionDAG &DAG, const SDLoc &dl) {
  assert((VT.is128BitVector() || VT.is256BitVector() || VT.is512BitVector() ||
          VT.getVectorElementType() == MVT::i1) &&
         "Unexpected vector type");

  // Try to build SSE/AVX zero vectors as <N x i32> bitcasted to their dest
  // type. This ensures they get CSE'd. But if the integer type is not
  // available, use a floating-point +0.0 instead.
  SDValue Vec;
  if (!Subtarget.hasSSE2() && VT.is128BitVector()) {
    Vec = DAG.getConstantFP(+0.0, dl, MVT::v4f32);
  } else if (VT.isFloatingPoint()) {
    Vec = DAG.getConstantFP(+0.0, dl, VT);
  } else if (VT.getVectorElementType() == MVT::i1) {
    assert((Subtarget.hasBWI() || VT.getVectorNumElements() <= 16) &&
           "Unexpected vector type");
    Vec = DAG.getConstant(0, dl, VT);
  } else {
    unsigned Num32BitElts = VT.getSizeInBits() / 32;
    Vec = DAG.getConstant(0, dl, MVT::getVectorVT(MVT::i32, Num32BitElts));
  }
  return DAG.getBitcast(VT, Vec);
}

SDValue X86::extractSubVector(SDValue Vec, unsigned IdxVal, SelectionDAG &DAG,
                              const SDLoc &dl, unsigned vectorWidth) {
  EVT VT = Vec.getValueType();
  EVT ElVT = VT.getVectorElementType();
  unsigned Factor = VT.getSizeInBits()/vectorWidth;
  EVT ResultVT = EVT::getVectorVT(*DAG.getContext(), ElVT,
                                  VT.getVectorNumElements()/Factor);

  // Extract the relevant vectorWidth bits.  Generate an EXTRACT_SUBVECTOR
  unsigned ElemsPerChunk = vectorWidth / ElVT.getSizeInBits();
  assert(isPowerOf2_32(ElemsPerChunk) && "Elements per chunk not power of 2");

  // This is the index of the first element of the vectorWidth-bit chunk
  // we want. Since ElemsPerChunk is a power of 2 just need to clear bits.
  IdxVal &= ~(ElemsPerChunk - 1);

  // If the input is a buildvector just emit a smaller one.
  if (Vec.getOpcode() == ISD::BUILD_VECTOR)
    return DAG.getBuildVector(ResultVT, dl,
                              Vec->ops().slice(IdxVal, ElemsPerChunk));

  SDValue VecIdx = DAG.getIntPtrConstant(IdxVal, dl);
  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, ResultVT, Vec, VecIdx);
}

/// Generate a DAG to grab 128-bits from a vector > 128 bits.  This
/// sets things up to match to an AVX VEXTRACTF128 / VEXTRACTI128
/// or AVX-512 VEXTRACTF32x4 / VEXTRACTI32x4
/// instructions or a simple subregister reference. Idx is an index in the
/// 128 bits we want.  It need not be aligned to a 128-bit boundary.  That makes
/// lowering EXTRACT_VECTOR_ELT operations easier.
SDValue X86::extract128BitVector(SDValue Vec, unsigned IdxVal,
                                 SelectionDAG &DAG, const SDLoc &dl) {
  assert((Vec.getValueType().is256BitVector() ||
          Vec.getValueType().is512BitVector()) && "Unexpected vector size!");
  return extractSubVector(Vec, IdxVal, DAG, dl, 128);
}

/// Generate a DAG to grab 256-bits from a 512-bit vector.
SDValue X86::extract256BitVector(SDValue Vec, unsigned IdxVal,
                                 SelectionDAG &DAG, const SDLoc &dl) {
  assert(Vec.getValueType().is512BitVector() && "Unexpected vector size!");
  return extractSubVector(Vec, IdxVal, DAG, dl, 256);
}

SDValue X86::insertSubVector(SDValue Result, SDValue Vec, unsigned IdxVal,
                             SelectionDAG &DAG, const SDLoc &dl,
                             unsigned vectorWidth) {
  assert((vectorWidth == 128 || vectorWidth == 256) &&
         "Unsupported vector width");
  // Inserting UNDEF is Result
  if (Vec.isUndef())
    return Result;
  EVT VT = Vec.getValueType();
  EVT ElVT = VT.getVectorElementType();
  EVT ResultVT = Result.getValueType();

  // Insert the relevant vectorWidth bits.
  unsigned ElemsPerChunk = vectorWidth/ElVT.getSizeInBits();
  assert(isPowerOf2_32(ElemsPerChunk) && "Elements per chunk not power of 2");

  // This is the index of the first element of the vectorWidth-bit chunk
  // we want. Since ElemsPerChunk is a power of 2 just need to clear bits.
  IdxVal &= ~(ElemsPerChunk - 1);

  SDValue VecIdx = DAG.getIntPtrConstant(IdxVal, dl);
  return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, ResultVT, Result, Vec, VecIdx);
}

/// Generate a DAG to put 128-bits into a vector > 128 bits.  This
/// sets things up to match to an AVX VINSERTF128/VINSERTI128 or
/// AVX-512 VINSERTF32x4/VINSERTI32x4 instructions or a
/// simple superregister reference.  Idx is an index in the 128 bits
/// we want.  It need not be aligned to a 128-bit boundary.  That makes
/// lowering INSERT_VECTOR_ELT operations easier.
static SDValue insert128BitVector(SDValue Result, SDValue Vec, unsigned IdxVal,
                                  SelectionDAG &DAG, const SDLoc &dl) {
  assert(Vec.getValueType().is128BitVector() && "Unexpected vector size!");
  return insertSubVector(Result, Vec, IdxVal, DAG, dl, 128);
}

/// Widen a vector to a larger size with the same scalar type, with the new
/// elements either zero or undef.
static SDValue widenSubVector(MVT VT, SDValue Vec, bool ZeroNewElements,
                              const X86Subtarget &Subtarget, SelectionDAG &DAG,
                              const SDLoc &dl) {
  assert(Vec.getValueSizeInBits() < VT.getSizeInBits() &&
         Vec.getValueType().getScalarType() == VT.getScalarType() &&
         "Unsupported vector widening type");
  SDValue Res = ZeroNewElements ? getZeroVector(VT, Subtarget, DAG, dl)
                                : DAG.getUNDEF(VT);
  return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, VT, Res, Vec,
                     DAG.getIntPtrConstant(0, dl));
}

/// Widen a vector to a larger size with the same scalar type, with the new
/// elements either zero or undef.
static SDValue widenSubVector(SDValue Vec, bool ZeroNewElements,
                              const X86Subtarget &Subtarget, SelectionDAG &DAG,
                              const SDLoc &dl, unsigned WideSizeInBits) {
  assert(Vec.getValueSizeInBits() < WideSizeInBits &&
         (WideSizeInBits % Vec.getScalarValueSizeInBits()) == 0 &&
         "Unsupported vector widening type");
  unsigned WideNumElts = WideSizeInBits / Vec.getScalarValueSizeInBits();
  MVT SVT = Vec.getSimpleValueType().getScalarType();
  MVT VT = MVT::getVectorVT(SVT, WideNumElts);
  return widenSubVector(VT, Vec, ZeroNewElements, Subtarget, DAG, dl);
}

// Helper for splitting operands of an operation to legal target size and
// apply a function on each part.
// Useful for operations that are available on SSE2 in 128-bit, on AVX2 in
// 256-bit and on AVX512BW in 512-bit. The argument VT is the type used for
// deciding if/how to split Ops. Ops elements do *not* have to be of type VT.
// The argument Builder is a function that will be applied on each split part:
// SDValue Builder(SelectionDAG&G, SDLoc, ArrayRef<SDValue>)
SDValue X86::SplitOpsAndApply(
    SelectionDAG &DAG, const X86Subtarget &Subtarget, const SDLoc &DL, EVT VT,
    ArrayRef<SDValue> Ops,
    function_ref<SDValue(SelectionDAG &, const SDLoc &, ArrayRef<SDValue>)>
        Builder,
    bool CheckBWI) {
  assert(Subtarget.hasSSE2() && "Target assumed to support at least SSE2");
  unsigned NumSubs = 1;
  if ((CheckBWI && Subtarget.useBWIRegs()) ||
      (!CheckBWI && Subtarget.useAVX512Regs())) {
    if (VT.getSizeInBits() > 512) {
      NumSubs = VT.getSizeInBits() / 512;
      assert((VT.getSizeInBits() % 512) == 0 && "Illegal vector size");
    }
  } else if (Subtarget.hasAVX2()) {
    if (VT.getSizeInBits() > 256) {
      NumSubs = VT.getSizeInBits() / 256;
      assert((VT.getSizeInBits() % 256) == 0 && "Illegal vector size");
    }
  } else {
    if (VT.getSizeInBits() > 128) {
      NumSubs = VT.getSizeInBits() / 128;
      assert((VT.getSizeInBits() % 128) == 0 && "Illegal vector size");
    }
  }

  if (NumSubs == 1)
    return Builder(DAG, DL, Ops);

  SmallVector<SDValue, 4> Subs;
  for (unsigned i = 0; i != NumSubs; ++i) {
    SmallVector<SDValue, 2> SubOps;
    for (SDValue Op : Ops) {
      EVT OpVT = Op.getValueType();
      unsigned NumSubElts = OpVT.getVectorNumElements() / NumSubs;
      unsigned SizeSub = OpVT.getSizeInBits() / NumSubs;
      SubOps.push_back(extractSubVector(Op, i * NumSubElts, DAG, DL, SizeSub));
    }
    Subs.push_back(Builder(DAG, DL, SubOps));
  }
  return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Subs);
}

/// Insert i1-subvector to i1-vector.
static SDValue insert1BitVector(SDValue Op, SelectionDAG &DAG,
                                const X86Subtarget &Subtarget) {

  SDLoc dl(Op);
  SDValue Vec = Op.getOperand(0);
  SDValue SubVec = Op.getOperand(1);
  SDValue Idx = Op.getOperand(2);

  if (!isa<ConstantSDNode>(Idx))
    return SDValue();

  // Inserting undef is a nop. We can just return the original vector.
  if (SubVec.isUndef())
    return Vec;

  unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();
  if (IdxVal == 0 && Vec.isUndef()) // the operation is legal
    return Op;

  MVT OpVT = Op.getSimpleValueType();
  unsigned NumElems = OpVT.getVectorNumElements();

  SDValue ZeroIdx = DAG.getIntPtrConstant(0, dl);

  // Extend to natively supported kshift.
  MVT WideOpVT = OpVT;
  if ((!Subtarget.hasDQI() && NumElems == 8) || NumElems < 8)
    WideOpVT = Subtarget.hasDQI() ? MVT::v8i1 : MVT::v16i1;

  // Inserting into the lsbs of a zero vector is legal. ISel will insert shifts
  // if necessary.
  if (IdxVal == 0 && ISD::isBuildVectorAllZeros(Vec.getNode())) {
    // May need to promote to a legal type.
    Op = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideOpVT,
                     DAG.getConstant(0, dl, WideOpVT),
                     SubVec, Idx);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, OpVT, Op, ZeroIdx);
  }

  MVT SubVecVT = SubVec.getSimpleValueType();
  unsigned SubVecNumElems = SubVecVT.getVectorNumElements();

  assert(IdxVal + SubVecNumElems <= NumElems &&
         IdxVal % SubVecVT.getSizeInBits() == 0 &&
         "Unexpected index value in INSERT_SUBVECTOR");

  SDValue Undef = DAG.getUNDEF(WideOpVT);

  if (IdxVal == 0) {
    // Zero lower bits of the Vec
    SDValue ShiftBits = DAG.getConstant(SubVecNumElems, dl, MVT::i8);
    Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideOpVT, Undef, Vec,
                      ZeroIdx);
    Vec = DAG.getNode(X86ISD::KSHIFTR, dl, WideOpVT, Vec, ShiftBits);
    Vec = DAG.getNode(X86ISD::KSHIFTL, dl, WideOpVT, Vec, ShiftBits);
    // Merge them together, SubVec should be zero extended.
    SubVec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideOpVT,
                         DAG.getConstant(0, dl, WideOpVT),
                         SubVec, ZeroIdx);
    Op = DAG.getNode(ISD::OR, dl, WideOpVT, Vec, SubVec);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, OpVT, Op, ZeroIdx);
  }

  SubVec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideOpVT,
                       Undef, SubVec, ZeroIdx);

  if (Vec.isUndef()) {
    assert(IdxVal != 0 && "Unexpected index");
    SubVec = DAG.getNode(X86ISD::KSHIFTL, dl, WideOpVT, SubVec,
                         DAG.getConstant(IdxVal, dl, MVT::i8));
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, OpVT, SubVec, ZeroIdx);
  }

  if (ISD::isBuildVectorAllZeros(Vec.getNode())) {
    assert(IdxVal != 0 && "Unexpected index");
    NumElems = WideOpVT.getVectorNumElements();
    unsigned ShiftLeft = NumElems - SubVecNumElems;
    unsigned ShiftRight = NumElems - SubVecNumElems - IdxVal;
    SubVec = DAG.getNode(X86ISD::KSHIFTL, dl, WideOpVT, SubVec,
                         DAG.getConstant(ShiftLeft, dl, MVT::i8));
    if (ShiftRight != 0)
      SubVec = DAG.getNode(X86ISD::KSHIFTR, dl, WideOpVT, SubVec,
                           DAG.getConstant(ShiftRight, dl, MVT::i8));
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, OpVT, SubVec, ZeroIdx);
  }

  // Simple case when we put subvector in the upper part
  if (IdxVal + SubVecNumElems == NumElems) {
    SubVec = DAG.getNode(X86ISD::KSHIFTL, dl, WideOpVT, SubVec,
                         DAG.getConstant(IdxVal, dl, MVT::i8));
    if (SubVecNumElems * 2 == NumElems) {
      // Special case, use legal zero extending insert_subvector. This allows
      // isel to opimitize when bits are known zero.
      Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, SubVecVT, Vec, ZeroIdx);
      Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideOpVT,
                        DAG.getConstant(0, dl, WideOpVT),
                        Vec, ZeroIdx);
    } else {
      // Otherwise use explicit shifts to zero the bits.
      Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideOpVT,
                        Undef, Vec, ZeroIdx);
      NumElems = WideOpVT.getVectorNumElements();
      SDValue ShiftBits = DAG.getConstant(NumElems - IdxVal, dl, MVT::i8);
      Vec = DAG.getNode(X86ISD::KSHIFTL, dl, WideOpVT, Vec, ShiftBits);
      Vec = DAG.getNode(X86ISD::KSHIFTR, dl, WideOpVT, Vec, ShiftBits);
    }
    Op = DAG.getNode(ISD::OR, dl, WideOpVT, Vec, SubVec);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, OpVT, Op, ZeroIdx);
  }

  // Inserting into the middle is more complicated.

  NumElems = WideOpVT.getVectorNumElements();

  // Widen the vector if needed.
  Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideOpVT, Undef, Vec, ZeroIdx);
  // Move the current value of the bit to be replace to the lsbs.
  Op = DAG.getNode(X86ISD::KSHIFTR, dl, WideOpVT, Vec,
                   DAG.getConstant(IdxVal, dl, MVT::i8));
  // Xor with the new bit.
  Op = DAG.getNode(ISD::XOR, dl, WideOpVT, Op, SubVec);
  // Shift to MSB, filling bottom bits with 0.
  unsigned ShiftLeft = NumElems - SubVecNumElems;
  Op = DAG.getNode(X86ISD::KSHIFTL, dl, WideOpVT, Op,
                   DAG.getConstant(ShiftLeft, dl, MVT::i8));
  // Shift to the final position, filling upper bits with 0.
  unsigned ShiftRight = NumElems - SubVecNumElems - IdxVal;
  Op = DAG.getNode(X86ISD::KSHIFTR, dl, WideOpVT, Op,
                       DAG.getConstant(ShiftRight, dl, MVT::i8));
  // Xor with original vector leaving the new value.
  Op = DAG.getNode(ISD::XOR, dl, WideOpVT, Vec, Op);
  // Reduce to original width if needed.
  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, OpVT, Op, ZeroIdx);
}

SDValue X86::concatSubVectors(SDValue V1, SDValue V2, SelectionDAG &DAG,
                                const SDLoc &dl) {
  assert(V1.getValueType() == V2.getValueType() && "subvector type mismatch");
  EVT SubVT = V1.getValueType();
  EVT SubSVT = SubVT.getScalarType();
  unsigned SubNumElts = SubVT.getVectorNumElements();
  unsigned SubVectorWidth = SubVT.getSizeInBits();
  EVT VT = EVT::getVectorVT(*DAG.getContext(), SubSVT, 2 * SubNumElts);
  SDValue V = insertSubVector(DAG.getUNDEF(VT), V1, 0, DAG, dl, SubVectorWidth);
  return insertSubVector(V, V2, SubNumElts, DAG, dl, SubVectorWidth);
}

/// Returns a vector of specified type with all bits set.
/// Always build ones vectors as <4 x i32>, <8 x i32> or <16 x i32>.
/// Then bitcast to their original type, ensuring they get CSE'd.
static SDValue getOnesVector(EVT VT, SelectionDAG &DAG, const SDLoc &dl) {
  assert((VT.is128BitVector() || VT.is256BitVector() || VT.is512BitVector()) &&
         "Expected a 128/256/512-bit vector type");

  APInt Ones = APInt::getAllOnesValue(32);
  unsigned NumElts = VT.getSizeInBits() / 32;
  SDValue Vec = DAG.getConstant(Ones, dl, MVT::getVectorVT(MVT::i32, NumElts));
  return DAG.getBitcast(VT, Vec);
}

// Convert *_EXTEND to *_EXTEND_VECTOR_INREG opcode.
unsigned X86::getOpcode_EXTEND_VECTOR_INREG(unsigned Opcode) {
  switch (Opcode) {
  case ISD::ANY_EXTEND:
  case ISD::ANY_EXTEND_VECTOR_INREG:
    return ISD::ANY_EXTEND_VECTOR_INREG;
  case ISD::ZERO_EXTEND:
  case ISD::ZERO_EXTEND_VECTOR_INREG:
    return ISD::ZERO_EXTEND_VECTOR_INREG;
  case ISD::SIGN_EXTEND:
  case ISD::SIGN_EXTEND_VECTOR_INREG:
    return ISD::SIGN_EXTEND_VECTOR_INREG;
  }
  llvm_unreachable("Unknown opcode");
}

SDValue X86::getExtendInVec(unsigned Opcode, const SDLoc &DL, EVT VT,
                            SDValue In, SelectionDAG &DAG) {
  EVT InVT = In.getValueType();
  assert(VT.isVector() && InVT.isVector() && "Expected vector VTs.");
  assert((ISD::ANY_EXTEND == Opcode || ISD::SIGN_EXTEND == Opcode ||
          ISD::ZERO_EXTEND == Opcode) &&
         "Unknown extension opcode");

  // For 256-bit vectors, we only need the lower (128-bit) input half.
  // For 512-bit vectors, we only need the lower input half or quarter.
  if (InVT.getSizeInBits() > 128) {
    assert(VT.getSizeInBits() == InVT.getSizeInBits() &&
           "Expected VTs to be the same size!");
    unsigned Scale = VT.getScalarSizeInBits() / InVT.getScalarSizeInBits();
    In = extractSubVector(In, 0, DAG, DL,
                          std::max(128U, VT.getSizeInBits() / Scale));
    InVT = In.getValueType();
  }

  if (VT.getVectorNumElements() != InVT.getVectorNumElements())
    Opcode = getOpcode_EXTEND_VECTOR_INREG(Opcode);

  return DAG.getNode(Opcode, DL, VT, In);
}

/// Returns a vector_shuffle node for an unpackl operation.
SDValue X86::getUnpackl(SelectionDAG &DAG, const SDLoc &dl, MVT VT, SDValue V1,
                        SDValue V2) {
  SmallVector<int, 8> Mask;
  createUnpackShuffleMask(VT, Mask, /* Lo = */ true, /* Unary = */ false);
  return DAG.getVectorShuffle(VT, dl, V1, V2, Mask);
}

/// Returns a vector_shuffle node for an unpackh operation.
SDValue X86::getUnpackh(SelectionDAG &DAG, const SDLoc &dl, MVT VT, SDValue V1,
                        SDValue V2) {
  SmallVector<int, 8> Mask;
  createUnpackShuffleMask(VT, Mask, /* Lo = */ false, /* Unary = */ false);
  return DAG.getVectorShuffle(VT, dl, V1, V2, Mask);
}

/// Return a vector_shuffle of the specified vector of zero or undef vector.
/// This produces a shuffle where the low element of V2 is swizzled into the
/// zero/undef vector, landing at element Idx.
/// This produces a shuffle mask like 4,1,2,3 (idx=0) or  0,1,2,4 (idx=3).
SDValue X86::getShuffleVectorZeroOrUndef(SDValue V2, int Idx, bool IsZero,
                                         const X86Subtarget &Subtarget,
                                         SelectionDAG &DAG) {
  MVT VT = V2.getSimpleValueType();
  SDValue V1 = IsZero
    ? getZeroVector(VT, Subtarget, DAG, SDLoc(V2)) : DAG.getUNDEF(VT);
  int NumElems = VT.getVectorNumElements();
  SmallVector<int, 16> MaskVec(NumElems);
  for (int i = 0; i != NumElems; ++i)
    // If this is the insertion idx, put the low elt of V2 here.
    MaskVec[i] = (i == Idx) ? NumElems : i;
  return DAG.getVectorShuffle(VT, SDLoc(V2), V1, V2, MaskVec);
}

static const Constant *getTargetConstantFromNode(LoadSDNode *Load) {
  if (!Load || !ISD::isNormalLoad(Load))
    return nullptr;

  SDValue Ptr = Load->getBasePtr();
  if (Ptr->getOpcode() == X86ISD::Wrapper ||
      Ptr->getOpcode() == X86ISD::WrapperRIP)
    Ptr = Ptr->getOperand(0);

  auto *CNode = dyn_cast<ConstantPoolSDNode>(Ptr);
  if (!CNode || CNode->isMachineConstantPoolEntry() || CNode->getOffset() != 0)
    return nullptr;

  return CNode->getConstVal();
}

static const Constant *getTargetConstantFromNode(SDValue Op) {
  Op = peekThroughBitcasts(Op);
  return getTargetConstantFromNode(dyn_cast<LoadSDNode>(Op));
}

const Constant *
X86TargetLowering::getTargetConstantFromLoad(LoadSDNode *LD) const {
  assert(LD && "Unexpected null LoadSDNode");
  return getTargetConstantFromNode(LD);
}

// Extract raw constant bits from constant pools.
bool X86::getTargetConstantBitsFromNode(SDValue Op, unsigned EltSizeInBits,
                                        APInt &UndefElts,
                                        SmallVectorImpl<APInt> &EltBits,
                                        bool AllowWholeUndefs,
                                        bool AllowPartialUndefs) {
  assert(EltBits.empty() && "Expected an empty EltBits vector");

  Op = peekThroughBitcasts(Op);

  EVT VT = Op.getValueType();
  unsigned SizeInBits = VT.getSizeInBits();
  assert((SizeInBits % EltSizeInBits) == 0 && "Can't split constant!");
  unsigned NumElts = SizeInBits / EltSizeInBits;

  // Bitcast a source array of element bits to the target size.
  auto CastBitData = [&](APInt &UndefSrcElts, ArrayRef<APInt> SrcEltBits) {
    unsigned NumSrcElts = UndefSrcElts.getBitWidth();
    unsigned SrcEltSizeInBits = SrcEltBits[0].getBitWidth();
    assert((NumSrcElts * SrcEltSizeInBits) == SizeInBits &&
           "Constant bit sizes don't match");

    // Don't split if we don't allow undef bits.
    bool AllowUndefs = AllowWholeUndefs || AllowPartialUndefs;
    if (UndefSrcElts.getBoolValue() && !AllowUndefs)
      return false;

    // If we're already the right size, don't bother bitcasting.
    if (NumSrcElts == NumElts) {
      UndefElts = UndefSrcElts;
      EltBits.assign(SrcEltBits.begin(), SrcEltBits.end());
      return true;
    }

    // Extract all the undef/constant element data and pack into single bitsets.
    APInt UndefBits(SizeInBits, 0);
    APInt MaskBits(SizeInBits, 0);

    for (unsigned i = 0; i != NumSrcElts; ++i) {
      unsigned BitOffset = i * SrcEltSizeInBits;
      if (UndefSrcElts[i])
        UndefBits.setBits(BitOffset, BitOffset + SrcEltSizeInBits);
      MaskBits.insertBits(SrcEltBits[i], BitOffset);
    }

    // Split the undef/constant single bitset data into the target elements.
    UndefElts = APInt(NumElts, 0);
    EltBits.resize(NumElts, APInt(EltSizeInBits, 0));

    for (unsigned i = 0; i != NumElts; ++i) {
      unsigned BitOffset = i * EltSizeInBits;
      APInt UndefEltBits = UndefBits.extractBits(EltSizeInBits, BitOffset);

      // Only treat an element as UNDEF if all bits are UNDEF.
      if (UndefEltBits.isAllOnesValue()) {
        if (!AllowWholeUndefs)
          return false;
        UndefElts.setBit(i);
        continue;
      }

      // If only some bits are UNDEF then treat them as zero (or bail if not
      // supported).
      if (UndefEltBits.getBoolValue() && !AllowPartialUndefs)
        return false;

      EltBits[i] = MaskBits.extractBits(EltSizeInBits, BitOffset);
    }
    return true;
  };

  // Collect constant bits and insert into mask/undef bit masks.
  auto CollectConstantBits = [](const Constant *Cst, APInt &Mask, APInt &Undefs,
                                unsigned UndefBitIndex) {
    if (!Cst)
      return false;
    if (isa<UndefValue>(Cst)) {
      Undefs.setBit(UndefBitIndex);
      return true;
    }
    if (auto *CInt = dyn_cast<ConstantInt>(Cst)) {
      Mask = CInt->getValue();
      return true;
    }
    if (auto *CFP = dyn_cast<ConstantFP>(Cst)) {
      Mask = CFP->getValueAPF().bitcastToAPInt();
      return true;
    }
    return false;
  };

  // Handle UNDEFs.
  if (Op.isUndef()) {
    APInt UndefSrcElts = APInt::getAllOnesValue(NumElts);
    SmallVector<APInt, 64> SrcEltBits(NumElts, APInt(EltSizeInBits, 0));
    return CastBitData(UndefSrcElts, SrcEltBits);
  }

  // Extract scalar constant bits.
  if (auto *Cst = dyn_cast<ConstantSDNode>(Op)) {
    APInt UndefSrcElts = APInt::getNullValue(1);
    SmallVector<APInt, 64> SrcEltBits(1, Cst->getAPIntValue());
    return CastBitData(UndefSrcElts, SrcEltBits);
  }
  if (auto *Cst = dyn_cast<ConstantFPSDNode>(Op)) {
    APInt UndefSrcElts = APInt::getNullValue(1);
    APInt RawBits = Cst->getValueAPF().bitcastToAPInt();
    SmallVector<APInt, 64> SrcEltBits(1, RawBits);
    return CastBitData(UndefSrcElts, SrcEltBits);
  }

  // Extract constant bits from build vector.
  if (ISD::isBuildVectorOfConstantSDNodes(Op.getNode())) {
    unsigned SrcEltSizeInBits = VT.getScalarSizeInBits();
    unsigned NumSrcElts = SizeInBits / SrcEltSizeInBits;

    APInt UndefSrcElts(NumSrcElts, 0);
    SmallVector<APInt, 64> SrcEltBits(NumSrcElts, APInt(SrcEltSizeInBits, 0));
    for (unsigned i = 0, e = Op.getNumOperands(); i != e; ++i) {
      const SDValue &Src = Op.getOperand(i);
      if (Src.isUndef()) {
        UndefSrcElts.setBit(i);
        continue;
      }
      auto *Cst = cast<ConstantSDNode>(Src);
      SrcEltBits[i] = Cst->getAPIntValue().zextOrTrunc(SrcEltSizeInBits);
    }
    return CastBitData(UndefSrcElts, SrcEltBits);
  }
  if (ISD::isBuildVectorOfConstantFPSDNodes(Op.getNode())) {
    unsigned SrcEltSizeInBits = VT.getScalarSizeInBits();
    unsigned NumSrcElts = SizeInBits / SrcEltSizeInBits;

    APInt UndefSrcElts(NumSrcElts, 0);
    SmallVector<APInt, 64> SrcEltBits(NumSrcElts, APInt(SrcEltSizeInBits, 0));
    for (unsigned i = 0, e = Op.getNumOperands(); i != e; ++i) {
      const SDValue &Src = Op.getOperand(i);
      if (Src.isUndef()) {
        UndefSrcElts.setBit(i);
        continue;
      }
      auto *Cst = cast<ConstantFPSDNode>(Src);
      APInt RawBits = Cst->getValueAPF().bitcastToAPInt();
      SrcEltBits[i] = RawBits.zextOrTrunc(SrcEltSizeInBits);
    }
    return CastBitData(UndefSrcElts, SrcEltBits);
  }

  // Extract constant bits from constant pool vector.
  if (auto *Cst = getTargetConstantFromNode(Op)) {
    Type *CstTy = Cst->getType();
    unsigned CstSizeInBits = CstTy->getPrimitiveSizeInBits();
    if (!CstTy->isVectorTy() || (CstSizeInBits % SizeInBits) != 0)
      return false;

    unsigned SrcEltSizeInBits = CstTy->getScalarSizeInBits();
    unsigned NumSrcElts = SizeInBits / SrcEltSizeInBits;

    APInt UndefSrcElts(NumSrcElts, 0);
    SmallVector<APInt, 64> SrcEltBits(NumSrcElts, APInt(SrcEltSizeInBits, 0));
    for (unsigned i = 0; i != NumSrcElts; ++i)
      if (!CollectConstantBits(Cst->getAggregateElement(i), SrcEltBits[i],
                               UndefSrcElts, i))
        return false;

    return CastBitData(UndefSrcElts, SrcEltBits);
  }

  // Extract constant bits from a broadcasted constant pool scalar.
  if (Op.getOpcode() == X86ISD::VBROADCAST &&
      EltSizeInBits <= VT.getScalarSizeInBits()) {
    if (auto *Broadcast = getTargetConstantFromNode(Op.getOperand(0))) {
      unsigned SrcEltSizeInBits = Broadcast->getType()->getScalarSizeInBits();
      unsigned NumSrcElts = SizeInBits / SrcEltSizeInBits;

      APInt UndefSrcElts(NumSrcElts, 0);
      SmallVector<APInt, 64> SrcEltBits(1, APInt(SrcEltSizeInBits, 0));
      if (CollectConstantBits(Broadcast, SrcEltBits[0], UndefSrcElts, 0)) {
        if (UndefSrcElts[0])
          UndefSrcElts.setBits(0, NumSrcElts);
        SrcEltBits.append(NumSrcElts - 1, SrcEltBits[0]);
        return CastBitData(UndefSrcElts, SrcEltBits);
      }
    }
  }

  // Extract constant bits from a subvector broadcast.
  if (Op.getOpcode() == X86ISD::SUBV_BROADCAST) {
    SmallVector<APInt, 16> SubEltBits;
    if (getTargetConstantBitsFromNode(Op.getOperand(0), EltSizeInBits,
                                      UndefElts, SubEltBits, AllowWholeUndefs,
                                      AllowPartialUndefs)) {
      UndefElts = APInt::getSplat(NumElts, UndefElts);
      while (EltBits.size() < NumElts)
        EltBits.append(SubEltBits.begin(), SubEltBits.end());
      return true;
    }
  }

  // Extract a rematerialized scalar constant insertion.
  if (Op.getOpcode() == X86ISD::VZEXT_MOVL &&
      Op.getOperand(0).getOpcode() == ISD::SCALAR_TO_VECTOR &&
      isa<ConstantSDNode>(Op.getOperand(0).getOperand(0))) {
    unsigned SrcEltSizeInBits = VT.getScalarSizeInBits();
    unsigned NumSrcElts = SizeInBits / SrcEltSizeInBits;

    APInt UndefSrcElts(NumSrcElts, 0);
    SmallVector<APInt, 64> SrcEltBits;
    auto *CN = cast<ConstantSDNode>(Op.getOperand(0).getOperand(0));
    SrcEltBits.push_back(CN->getAPIntValue().zextOrTrunc(SrcEltSizeInBits));
    SrcEltBits.append(NumSrcElts - 1, APInt(SrcEltSizeInBits, 0));
    return CastBitData(UndefSrcElts, SrcEltBits);
  }

  // Insert constant bits from a base and sub vector sources.
  if (Op.getOpcode() == ISD::INSERT_SUBVECTOR &&
      isa<ConstantSDNode>(Op.getOperand(2))) {
    // TODO - support insert_subvector through bitcasts.
    if (EltSizeInBits != VT.getScalarSizeInBits())
      return false;

    APInt UndefSubElts;
    SmallVector<APInt, 32> EltSubBits;
    if (getTargetConstantBitsFromNode(Op.getOperand(1), EltSizeInBits,
                                      UndefSubElts, EltSubBits,
                                      AllowWholeUndefs, AllowPartialUndefs) &&
        getTargetConstantBitsFromNode(Op.getOperand(0), EltSizeInBits,
                                      UndefElts, EltBits, AllowWholeUndefs,
                                      AllowPartialUndefs)) {
      unsigned BaseIdx = Op.getConstantOperandVal(2);
      UndefElts.insertBits(UndefSubElts, BaseIdx);
      for (unsigned i = 0, e = EltSubBits.size(); i != e; ++i)
        EltBits[BaseIdx + i] = EltSubBits[i];
      return true;
    }
  }

  // Extract constant bits from a subvector's source.
  if (Op.getOpcode() == ISD::EXTRACT_SUBVECTOR &&
      isa<ConstantSDNode>(Op.getOperand(1))) {
    // TODO - support extract_subvector through bitcasts.
    if (EltSizeInBits != VT.getScalarSizeInBits())
      return false;

    if (getTargetConstantBitsFromNode(Op.getOperand(0), EltSizeInBits,
                                      UndefElts, EltBits, AllowWholeUndefs,
                                      AllowPartialUndefs)) {
      EVT SrcVT = Op.getOperand(0).getValueType();
      unsigned NumSrcElts = SrcVT.getVectorNumElements();
      unsigned NumSubElts = VT.getVectorNumElements();
      unsigned BaseIdx = Op.getConstantOperandVal(1);
      UndefElts = UndefElts.extractBits(NumSubElts, BaseIdx);
      if ((BaseIdx + NumSubElts) != NumSrcElts)
        EltBits.erase(EltBits.begin() + BaseIdx + NumSubElts, EltBits.end());
      if (BaseIdx != 0)
        EltBits.erase(EltBits.begin(), EltBits.begin() + BaseIdx);
      return true;
    }
  }

  // Extract constant bits from shuffle node sources.
  if (auto *SVN = dyn_cast<ShuffleVectorSDNode>(Op)) {
    // TODO - support shuffle through bitcasts.
    if (EltSizeInBits != VT.getScalarSizeInBits())
      return false;

    ArrayRef<int> Mask = SVN->getMask();
    if ((!AllowWholeUndefs || !AllowPartialUndefs) &&
        llvm::any_of(Mask, [](int M) { return M < 0; }))
      return false;

    APInt UndefElts0, UndefElts1;
    SmallVector<APInt, 32> EltBits0, EltBits1;
    if (isAnyInRange(Mask, 0, NumElts) &&
        !getTargetConstantBitsFromNode(Op.getOperand(0), EltSizeInBits,
                                       UndefElts0, EltBits0, AllowWholeUndefs,
                                       AllowPartialUndefs))
      return false;
    if (isAnyInRange(Mask, NumElts, 2 * NumElts) &&
        !getTargetConstantBitsFromNode(Op.getOperand(1), EltSizeInBits,
                                       UndefElts1, EltBits1, AllowWholeUndefs,
                                       AllowPartialUndefs))
      return false;

    UndefElts = APInt::getNullValue(NumElts);
    for (int i = 0; i != (int)NumElts; ++i) {
      int M = Mask[i];
      if (M < 0) {
        UndefElts.setBit(i);
        EltBits.push_back(APInt::getNullValue(EltSizeInBits));
      } else if (M < (int)NumElts) {
        if (UndefElts0[M])
          UndefElts.setBit(i);
        EltBits.push_back(EltBits0[M]);
      } else {
        if (UndefElts1[M - NumElts])
          UndefElts.setBit(i);
        EltBits.push_back(EltBits1[M - NumElts]);
      }
    }
    return true;
  }

  return false;
}

namespace llvm {
namespace X86 {
bool isConstantSplat(SDValue Op, APInt &SplatVal) {
  APInt UndefElts;
  SmallVector<APInt, 16> EltBits;
  if (getTargetConstantBitsFromNode(Op, Op.getScalarValueSizeInBits(),
                                    UndefElts, EltBits, true, false)) {
    int SplatIndex = -1;
    for (int i = 0, e = EltBits.size(); i != e; ++i) {
      if (UndefElts[i])
        continue;
      if (0 <= SplatIndex && EltBits[i] != EltBits[SplatIndex]) {
        SplatIndex = -1;
        break;
      }
      SplatIndex = i;
    }
    if (0 <= SplatIndex) {
      SplatVal = EltBits[SplatIndex];
      return true;
    }
  }

  return false;
}
} // namespace X86
} // namespace llvm

static bool getTargetShuffleMaskIndices(SDValue MaskNode,
                                        unsigned MaskEltSizeInBits,
                                        SmallVectorImpl<uint64_t> &RawMask,
                                        APInt &UndefElts) {
  // Extract the raw target constant bits.
  SmallVector<APInt, 64> EltBits;
  if (!getTargetConstantBitsFromNode(MaskNode, MaskEltSizeInBits, UndefElts,
                                     EltBits, /* AllowWholeUndefs */ true,
                                     /* AllowPartialUndefs */ false))
    return false;

  // Insert the extracted elements into the mask.
  for (APInt Elt : EltBits)
    RawMask.push_back(Elt.getZExtValue());

  return true;
}

/// Create a shuffle mask that matches the PACKSS/PACKUS truncation.
/// Note: This ignores saturation, so inputs must be checked first.
static void createPackShuffleMask(MVT VT, SmallVectorImpl<int> &Mask,
                                  bool Unary) {
  assert(Mask.empty() && "Expected an empty shuffle mask vector");
  unsigned NumElts = VT.getVectorNumElements();
  unsigned NumLanes = VT.getSizeInBits() / 128;
  unsigned NumEltsPerLane = 128 / VT.getScalarSizeInBits();
  unsigned Offset = Unary ? 0 : NumElts;

  for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
    for (unsigned Elt = 0; Elt != NumEltsPerLane; Elt += 2)
      Mask.push_back(Elt + (Lane * NumEltsPerLane));
    for (unsigned Elt = 0; Elt != NumEltsPerLane; Elt += 2)
      Mask.push_back(Elt + (Lane * NumEltsPerLane) + Offset);
  }
}

// Split the demanded elts of a PACKSS/PACKUS node between its operands.
void X86::getPackDemandedElts(EVT VT, const APInt &DemandedElts,
                              APInt &DemandedLHS, APInt &DemandedRHS) {
  int NumLanes = VT.getSizeInBits() / 128;
  int NumElts = DemandedElts.getBitWidth();
  int NumInnerElts = NumElts / 2;
  int NumEltsPerLane = NumElts / NumLanes;
  int NumInnerEltsPerLane = NumInnerElts / NumLanes;

  DemandedLHS = APInt::getNullValue(NumInnerElts);
  DemandedRHS = APInt::getNullValue(NumInnerElts);

  // Map DemandedElts to the packed operands.
  for (int Lane = 0; Lane != NumLanes; ++Lane) {
    for (int Elt = 0; Elt != NumInnerEltsPerLane; ++Elt) {
      int OuterIdx = (Lane * NumEltsPerLane) + Elt;
      int InnerIdx = (Lane * NumInnerEltsPerLane) + Elt;
      if (DemandedElts[OuterIdx])
        DemandedLHS.setBit(InnerIdx);
      if (DemandedElts[OuterIdx + NumInnerEltsPerLane])
        DemandedRHS.setBit(InnerIdx);
    }
  }
}

// Split the demanded elts of a HADD/HSUB node between its operands.
static void getHorizDemandedElts(EVT VT, const APInt &DemandedElts,
                                 APInt &DemandedLHS, APInt &DemandedRHS) {
  int NumLanes = VT.getSizeInBits() / 128;
  int NumElts = DemandedElts.getBitWidth();
  int NumEltsPerLane = NumElts / NumLanes;
  int HalfEltsPerLane = NumEltsPerLane / 2;

  DemandedLHS = APInt::getNullValue(NumElts);
  DemandedRHS = APInt::getNullValue(NumElts);

  // Map DemandedElts to the horizontal operands.
  for (int Idx = 0; Idx != NumElts; ++Idx) {
    if (!DemandedElts[Idx])
      continue;
    int LaneIdx = (Idx / NumEltsPerLane) * NumEltsPerLane;
    int LocalIdx = Idx % NumEltsPerLane;
    if (LocalIdx < HalfEltsPerLane) {
      DemandedLHS.setBit(LaneIdx + 2 * LocalIdx + 0);
      DemandedLHS.setBit(LaneIdx + 2 * LocalIdx + 1);
    } else {
      LocalIdx -= HalfEltsPerLane;
      DemandedRHS.setBit(LaneIdx + 2 * LocalIdx + 0);
      DemandedRHS.setBit(LaneIdx + 2 * LocalIdx + 1);
    }
  }
}

/// Calculates the shuffle mask corresponding to the target-specific opcode.
/// If the mask could be calculated, returns it in \p Mask, returns the shuffle
/// operands in \p Ops, and returns true.
/// Sets \p IsUnary to true if only one source is used. Note that this will set
/// IsUnary for shuffles which use a single input multiple times, and in those
/// cases it will adjust the mask to only have indices within that single input.
/// It is an error to call this with non-empty Mask/Ops vectors.
bool X86::getTargetShuffleMask(SDNode *N, MVT VT, bool AllowSentinelZero,
                               SmallVectorImpl<SDValue> &Ops,
                               SmallVectorImpl<int> &Mask, bool &IsUnary) {
  unsigned NumElems = VT.getVectorNumElements();
  unsigned MaskEltSize = VT.getScalarSizeInBits();
  SmallVector<uint64_t, 32> RawMask;
  APInt RawUndefs;
  SDValue ImmN;

  assert(Mask.empty() && "getTargetShuffleMask expects an empty Mask vector");
  assert(Ops.empty() && "getTargetShuffleMask expects an empty Ops vector");

  IsUnary = false;
  bool IsFakeUnary = false;
  switch (N->getOpcode()) {
  case X86ISD::BLENDI:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodeBLENDMask(NumElems, cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::SHUFP:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodeSHUFPMask(NumElems, MaskEltSize,
                    cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::INSERTPS:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodeINSERTPSMask(cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::EXTRQI:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    if (isa<ConstantSDNode>(N->getOperand(1)) &&
        isa<ConstantSDNode>(N->getOperand(2))) {
      int BitLen = N->getConstantOperandVal(1);
      int BitIdx = N->getConstantOperandVal(2);
      DecodeEXTRQIMask(NumElems, MaskEltSize, BitLen, BitIdx, Mask);
      IsUnary = true;
    }
    break;
  case X86ISD::INSERTQI:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    if (isa<ConstantSDNode>(N->getOperand(2)) &&
        isa<ConstantSDNode>(N->getOperand(3))) {
      int BitLen = N->getConstantOperandVal(2);
      int BitIdx = N->getConstantOperandVal(3);
      DecodeINSERTQIMask(NumElems, MaskEltSize, BitLen, BitIdx, Mask);
      IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    }
    break;
  case X86ISD::UNPCKH:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    DecodeUNPCKHMask(NumElems, MaskEltSize, Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::UNPCKL:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    DecodeUNPCKLMask(NumElems, MaskEltSize, Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::MOVHLPS:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    DecodeMOVHLPSMask(NumElems, Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::MOVLHPS:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    DecodeMOVLHPSMask(NumElems, Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::PALIGNR:
    assert(VT.getScalarType() == MVT::i8 && "Byte vector expected");
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodePALIGNRMask(NumElems, cast<ConstantSDNode>(ImmN)->getZExtValue(),
                      Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    Ops.push_back(N->getOperand(1));
    Ops.push_back(N->getOperand(0));
    break;
  case X86ISD::VSHLDQ:
    assert(VT.getScalarType() == MVT::i8 && "Byte vector expected");
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodePSLLDQMask(NumElems, cast<ConstantSDNode>(ImmN)->getZExtValue(),
                     Mask);
    IsUnary = true;
    break;
  case X86ISD::VSRLDQ:
    assert(VT.getScalarType() == MVT::i8 && "Byte vector expected");
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodePSRLDQMask(NumElems, cast<ConstantSDNode>(ImmN)->getZExtValue(),
                     Mask);
    IsUnary = true;
    break;
  case X86ISD::PSHUFD:
  case X86ISD::VPERMILPI:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodePSHUFMask(NumElems, MaskEltSize,
                    cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = true;
    break;
  case X86ISD::PSHUFHW:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodePSHUFHWMask(NumElems, cast<ConstantSDNode>(ImmN)->getZExtValue(),
                      Mask);
    IsUnary = true;
    break;
  case X86ISD::PSHUFLW:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodePSHUFLWMask(NumElems, cast<ConstantSDNode>(ImmN)->getZExtValue(),
                      Mask);
    IsUnary = true;
    break;
  case X86ISD::VZEXT_MOVL:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    DecodeZeroMoveLowMask(NumElems, Mask);
    IsUnary = true;
    break;
  case X86ISD::VBROADCAST: {
    SDValue N0 = N->getOperand(0);
    // See if we're broadcasting from index 0 of an EXTRACT_SUBVECTOR. If so,
    // add the pre-extracted value to the Ops vector.
    if (N0.getOpcode() == ISD::EXTRACT_SUBVECTOR &&
        N0.getOperand(0).getValueType() == VT &&
        N0.getConstantOperandVal(1) == 0)
      Ops.push_back(N0.getOperand(0));

    // We only decode broadcasts of same-sized vectors, unless the broadcast
    // came from an extract from the original width. If we found one, we
    // pushed it the Ops vector above.
    if (N0.getValueType() == VT || !Ops.empty()) {
      DecodeVectorBroadcast(NumElems, Mask);
      IsUnary = true;
      break;
    }
    return false;
  }
  case X86ISD::VPERMILPV: {
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    IsUnary = true;
    SDValue MaskNode = N->getOperand(1);
    if (getTargetShuffleMaskIndices(MaskNode, MaskEltSize, RawMask,
                                    RawUndefs)) {
      DecodeVPERMILPMask(NumElems, MaskEltSize, RawMask, RawUndefs, Mask);
      break;
    }
    return false;
  }
  case X86ISD::PSHUFB: {
    assert(VT.getScalarType() == MVT::i8 && "Byte vector expected");
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    IsUnary = true;
    SDValue MaskNode = N->getOperand(1);
    if (getTargetShuffleMaskIndices(MaskNode, 8, RawMask, RawUndefs)) {
      DecodePSHUFBMask(RawMask, RawUndefs, Mask);
      break;
    }
    return false;
  }
  case X86ISD::VPERMI:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodeVPERMMask(NumElems, cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = true;
    break;
  case X86ISD::MOVSS:
  case X86ISD::MOVSD:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    DecodeScalarMoveMask(NumElems, /* IsLoad */ false, Mask);
    break;
  case X86ISD::VPERM2X128:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    DecodeVPERM2X128Mask(NumElems, cast<ConstantSDNode>(ImmN)->getZExtValue(),
                         Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::SHUF128:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    ImmN = N->getOperand(N->getNumOperands() - 1);
    decodeVSHUF64x2FamilyMask(NumElems, MaskEltSize,
                              cast<ConstantSDNode>(ImmN)->getZExtValue(), Mask);
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    break;
  case X86ISD::MOVSLDUP:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    DecodeMOVSLDUPMask(NumElems, Mask);
    IsUnary = true;
    break;
  case X86ISD::MOVSHDUP:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    DecodeMOVSHDUPMask(NumElems, Mask);
    IsUnary = true;
    break;
  case X86ISD::MOVDDUP:
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    DecodeMOVDDUPMask(NumElems, Mask);
    IsUnary = true;
    break;
  case X86ISD::VPERMIL2: {
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    SDValue MaskNode = N->getOperand(2);
    SDValue CtrlNode = N->getOperand(3);
    if (ConstantSDNode *CtrlOp = dyn_cast<ConstantSDNode>(CtrlNode)) {
      unsigned CtrlImm = CtrlOp->getZExtValue();
      if (getTargetShuffleMaskIndices(MaskNode, MaskEltSize, RawMask,
                                      RawUndefs)) {
        DecodeVPERMIL2PMask(NumElems, MaskEltSize, CtrlImm, RawMask, RawUndefs,
                            Mask);
        break;
      }
    }
    return false;
  }
  case X86ISD::VPPERM: {
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(1);
    SDValue MaskNode = N->getOperand(2);
    if (getTargetShuffleMaskIndices(MaskNode, 8, RawMask, RawUndefs)) {
      DecodeVPPERMMask(RawMask, RawUndefs, Mask);
      break;
    }
    return false;
  }
  case X86ISD::VPERMV: {
    assert(N->getOperand(1).getValueType() == VT && "Unexpected value type");
    IsUnary = true;
    // Unlike most shuffle nodes, VPERMV's mask operand is operand 0.
    Ops.push_back(N->getOperand(1));
    SDValue MaskNode = N->getOperand(0);
    if (getTargetShuffleMaskIndices(MaskNode, MaskEltSize, RawMask,
                                    RawUndefs)) {
      DecodeVPERMVMask(RawMask, RawUndefs, Mask);
      break;
    }
    return false;
  }
  case X86ISD::VPERMV3: {
    assert(N->getOperand(0).getValueType() == VT && "Unexpected value type");
    assert(N->getOperand(2).getValueType() == VT && "Unexpected value type");
    IsUnary = IsFakeUnary = N->getOperand(0) == N->getOperand(2);
    // Unlike most shuffle nodes, VPERMV3's mask operand is the middle one.
    Ops.push_back(N->getOperand(0));
    Ops.push_back(N->getOperand(2));
    SDValue MaskNode = N->getOperand(1);
    if (getTargetShuffleMaskIndices(MaskNode, MaskEltSize, RawMask,
                                    RawUndefs)) {
      DecodeVPERMV3Mask(RawMask, RawUndefs, Mask);
      break;
    }
    return false;
  }
  default: llvm_unreachable("unknown target shuffle node");
  }

  // Empty mask indicates the decode failed.
  if (Mask.empty())
    return false;

  // Check if we're getting a shuffle mask with zero'd elements.
  if (!AllowSentinelZero)
    if (any_of(Mask, [](int M) { return M == SM_SentinelZero; }))
      return false;

  // If we have a fake unary shuffle, the shuffle mask is spread across two
  // inputs that are actually the same node. Re-map the mask to always point
  // into the first input.
  if (IsFakeUnary)
    for (int &M : Mask)
      if (M >= (int)Mask.size())
        M -= Mask.size();

  // If we didn't already add operands in the opcode-specific code, default to
  // adding 1 or 2 operands starting at 0.
  if (Ops.empty()) {
    Ops.push_back(N->getOperand(0));
    if (!IsUnary || IsFakeUnary)
      Ops.push_back(N->getOperand(1));
  }

  return true;
}

/// Check a target shuffle mask's inputs to see if we can set any values to
/// SM_SentinelZero - this is for elements that are known to be zero
/// (not just zeroable) from their inputs.
/// Returns true if the target shuffle mask was decoded.
static bool setTargetShuffleZeroElements(SDValue N,
                                         SmallVectorImpl<int> &Mask,
                                         SmallVectorImpl<SDValue> &Ops,
                                         bool ResolveZero = true) {
  bool IsUnary;
  if (!isTargetShuffle(N.getOpcode()))
    return false;

  MVT VT = N.getSimpleValueType();
  if (!getTargetShuffleMask(N.getNode(), VT, true, Ops, Mask, IsUnary))
    return false;

  SDValue V1 = Ops[0];
  SDValue V2 = IsUnary ? V1 : Ops[1];

  V1 = peekThroughBitcasts(V1);
  V2 = peekThroughBitcasts(V2);

  assert((VT.getSizeInBits() % Mask.size()) == 0 &&
         "Illegal split of shuffle value type");
  unsigned EltSizeInBits = VT.getSizeInBits() / Mask.size();

  // Extract known constant input data.
  APInt UndefSrcElts[2];
  SmallVector<APInt, 32> SrcEltBits[2];
  bool IsSrcConstant[2] = {
      getTargetConstantBitsFromNode(V1, EltSizeInBits, UndefSrcElts[0],
                                    SrcEltBits[0], true, false),
      getTargetConstantBitsFromNode(V2, EltSizeInBits, UndefSrcElts[1],
                                    SrcEltBits[1], true, false)};

  for (int i = 0, Size = Mask.size(); i < Size; ++i) {
    int M = Mask[i];

    // Already decoded as SM_SentinelZero / SM_SentinelUndef.
    if (M < 0)
      continue;

    // Determine shuffle input and normalize the mask.
    unsigned SrcIdx = M / Size;
    SDValue V = M < Size ? V1 : V2;
    M %= Size;

    // We are referencing an UNDEF input.
    if (V.isUndef()) {
      Mask[i] = SM_SentinelUndef;
      continue;
    }

    // SCALAR_TO_VECTOR - only the first element is defined, and the rest UNDEF.
    // TODO: We currently only set UNDEF for integer types - floats use the same
    // registers as vectors and many of the scalar folded loads rely on the
    // SCALAR_TO_VECTOR pattern.
    if (V.getOpcode() == ISD::SCALAR_TO_VECTOR &&
        (Size % V.getValueType().getVectorNumElements()) == 0) {
      int Scale = Size / V.getValueType().getVectorNumElements();
      int Idx = M / Scale;
      if (Idx != 0 && !VT.isFloatingPoint())
        Mask[i] = SM_SentinelUndef;
      else if (ResolveZero && Idx == 0 && X86::isZeroNode(V.getOperand(0)))
        Mask[i] = SM_SentinelZero;
      continue;
    }

    // Attempt to extract from the source's constant bits.
    if (IsSrcConstant[SrcIdx]) {
      if (UndefSrcElts[SrcIdx][M])
        Mask[i] = SM_SentinelUndef;
      else if (ResolveZero && SrcEltBits[SrcIdx][M] == 0)
        Mask[i] = SM_SentinelZero;
    }
  }

  assert(VT.getVectorNumElements() == Mask.size() &&
         "Different mask size from vector size!");
  return true;
}

// Attempt to decode ops that could be represented as a shuffle mask.
// The decoded shuffle mask may contain a different number of elements to the
// destination value type.
static bool getFauxShuffleMask(SDValue N, const APInt &DemandedElts,
                               SmallVectorImpl<int> &Mask,
                               SmallVectorImpl<SDValue> &Ops,
                               SelectionDAG &DAG, unsigned Depth,
                               bool ResolveZero) {
  Mask.clear();
  Ops.clear();

  MVT VT = N.getSimpleValueType();
  unsigned NumElts = VT.getVectorNumElements();
  unsigned NumSizeInBits = VT.getSizeInBits();
  unsigned NumBitsPerElt = VT.getScalarSizeInBits();
  if ((NumBitsPerElt % 8) != 0 || (NumSizeInBits % 8) != 0)
    return false;
  assert(NumElts == DemandedElts.getBitWidth() && "Unexpected vector size");

  unsigned Opcode = N.getOpcode();
  switch (Opcode) {
  case ISD::VECTOR_SHUFFLE: {
    // Don't treat ISD::VECTOR_SHUFFLE as a target shuffle so decode it here.
    ArrayRef<int> ShuffleMask = cast<ShuffleVectorSDNode>(N)->getMask();
    if (isUndefOrInRange(ShuffleMask, 0, 2 * NumElts)) {
      Mask.append(ShuffleMask.begin(), ShuffleMask.end());
      Ops.push_back(N.getOperand(0));
      Ops.push_back(N.getOperand(1));
      return true;
    }
    return false;
  }
  case ISD::AND:
  case X86ISD::ANDNP: {
    // Attempt to decode as a per-byte mask.
    APInt UndefElts;
    SmallVector<APInt, 32> EltBits;
    SDValue N0 = N.getOperand(0);
    SDValue N1 = N.getOperand(1);
    bool IsAndN = (X86ISD::ANDNP == Opcode);
    uint64_t ZeroMask = IsAndN ? 255 : 0;
    if (!getTargetConstantBitsFromNode(IsAndN ? N0 : N1, 8, UndefElts, EltBits))
      return false;
    for (int i = 0, e = (int)EltBits.size(); i != e; ++i) {
      if (UndefElts[i]) {
        Mask.push_back(SM_SentinelUndef);
        continue;
      }
      const APInt &ByteBits = EltBits[i];
      if (ByteBits != 0 && ByteBits != 255)
        return false;
      Mask.push_back(ByteBits == ZeroMask ? SM_SentinelZero : i);
    }
    Ops.push_back(IsAndN ? N1 : N0);
    return true;
  }
  case ISD::OR: {
    // Inspect each operand at the byte level. We can merge these into a
    // blend shuffle mask if for each byte at least one is masked out (zero).
    KnownBits Known0 =
        DAG.computeKnownBits(N.getOperand(0), DemandedElts, Depth + 1);
    KnownBits Known1 =
        DAG.computeKnownBits(N.getOperand(1), DemandedElts, Depth + 1);
    if (Known0.One.isNullValue() && Known1.One.isNullValue()) {
      bool IsByteMask = true;
      unsigned NumSizeInBytes = NumSizeInBits / 8;
      unsigned NumBytesPerElt = NumBitsPerElt / 8;
      APInt ZeroMask = APInt::getNullValue(NumBytesPerElt);
      APInt SelectMask = APInt::getNullValue(NumBytesPerElt);
      for (unsigned i = 0; i != NumBytesPerElt && IsByteMask; ++i) {
        unsigned LHS = Known0.Zero.extractBits(8, i * 8).getZExtValue();
        unsigned RHS = Known1.Zero.extractBits(8, i * 8).getZExtValue();
        if (LHS == 255 && RHS == 0)
          SelectMask.setBit(i);
        else if (LHS == 255 && RHS == 255)
          ZeroMask.setBit(i);
        else if (!(LHS == 0 && RHS == 255))
          IsByteMask = false;
      }
      if (IsByteMask) {
        for (unsigned i = 0; i != NumSizeInBytes; i += NumBytesPerElt) {
          for (unsigned j = 0; j != NumBytesPerElt; ++j) {
            unsigned Ofs = (SelectMask[j] ? NumSizeInBytes : 0);
            int Idx = (ZeroMask[j] ? (int)SM_SentinelZero : (i + j + Ofs));
            Mask.push_back(Idx);
          }
        }
        Ops.push_back(N.getOperand(0));
        Ops.push_back(N.getOperand(1));
        return true;
      }
    }

    // Handle OR(SHUFFLE,SHUFFLE) case where one source is zero and the other
    // is a valid shuffle index.
    SDValue N0 = peekThroughOneUseBitcasts(N.getOperand(0));
    SDValue N1 = peekThroughOneUseBitcasts(N.getOperand(1));
    if (!N0.getValueType().isVector() || !N1.getValueType().isVector())
      return false;
    SmallVector<int, 64> SrcMask0, SrcMask1;
    SmallVector<SDValue, 2> SrcInputs0, SrcInputs1;
    if (!X86::resolveTargetShuffleInputs(N0, SrcInputs0, SrcMask0, DAG, Depth + 1,
                                    ResolveZero) ||
        !X86::resolveTargetShuffleInputs(N1, SrcInputs1, SrcMask1, DAG, Depth + 1,
                                    ResolveZero))
      return false;
    int MaskSize = std::max(SrcMask0.size(), SrcMask1.size());
    SmallVector<int, 64> Mask0, Mask1;
    scaleShuffleMask<int>(MaskSize / SrcMask0.size(), SrcMask0, Mask0);
    scaleShuffleMask<int>(MaskSize / SrcMask1.size(), SrcMask1, Mask1);
    for (int i = 0; i != MaskSize; ++i) {
      if (Mask0[i] == SM_SentinelUndef && Mask1[i] == SM_SentinelUndef)
        Mask.push_back(SM_SentinelUndef);
      else if (Mask0[i] == SM_SentinelZero && Mask1[i] == SM_SentinelZero)
        Mask.push_back(SM_SentinelZero);
      else if (Mask1[i] == SM_SentinelZero)
        Mask.push_back(Mask0[i]);
      else if (Mask0[i] == SM_SentinelZero)
        Mask.push_back(Mask1[i] + (MaskSize * SrcInputs0.size()));
      else
        return false;
    }
    for (SDValue &Op : SrcInputs0)
      Ops.push_back(Op);
    for (SDValue &Op : SrcInputs1)
      Ops.push_back(Op);
    return true;
  }
  case ISD::INSERT_SUBVECTOR: {
    SDValue Src = N.getOperand(0);
    SDValue Sub = N.getOperand(1);
    EVT SubVT = Sub.getValueType();
    unsigned NumSubElts = SubVT.getVectorNumElements();
    if (!isa<ConstantSDNode>(N.getOperand(2)) ||
        !N->isOnlyUserOf(Sub.getNode()))
      return false;
    uint64_t InsertIdx = N.getConstantOperandVal(2);
    // Handle INSERT_SUBVECTOR(SRC0, EXTRACT_SUBVECTOR(SRC1)).
    if (Sub.getOpcode() == ISD::EXTRACT_SUBVECTOR &&
        Sub.getOperand(0).getValueType() == VT &&
        isa<ConstantSDNode>(Sub.getOperand(1))) {
      uint64_t ExtractIdx = Sub.getConstantOperandVal(1);
      for (int i = 0; i != (int)NumElts; ++i)
        Mask.push_back(i);
      for (int i = 0; i != (int)NumSubElts; ++i)
        Mask[InsertIdx + i] = NumElts + ExtractIdx + i;
      Ops.push_back(Src);
      Ops.push_back(Sub.getOperand(0));
      return true;
    }
    // Handle INSERT_SUBVECTOR(SRC0, SHUFFLE(SRC1)).
    SmallVector<int, 64> SubMask;
    SmallVector<SDValue, 2> SubInputs;
    if (!X86::resolveTargetShuffleInputs(peekThroughOneUseBitcasts(Sub), SubInputs,
                                    SubMask, DAG, Depth + 1, ResolveZero))
      return false;
    if (SubMask.size() != NumSubElts) {
      assert(((SubMask.size() % NumSubElts) == 0 ||
              (NumSubElts % SubMask.size()) == 0) && "Illegal submask scale");
      if ((NumSubElts % SubMask.size()) == 0) {
        int Scale = NumSubElts / SubMask.size();
        SmallVector<int,64> ScaledSubMask;
        scaleShuffleMask<int>(Scale, SubMask, ScaledSubMask);
        SubMask = ScaledSubMask;
      } else {
        int Scale = SubMask.size() / NumSubElts;
        NumSubElts = SubMask.size();
        NumElts *= Scale;
        InsertIdx *= Scale;
      }
    }
    Ops.push_back(Src);
    for (SDValue &SubInput : SubInputs) {
      EVT SubSVT = SubInput.getValueType().getScalarType();
      EVT AltVT = EVT::getVectorVT(*DAG.getContext(), SubSVT,
                                   NumSizeInBits / SubSVT.getSizeInBits());
      Ops.push_back(DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N), AltVT,
                                DAG.getUNDEF(AltVT), SubInput,
                                DAG.getIntPtrConstant(0, SDLoc(N))));
    }
    for (int i = 0; i != (int)NumElts; ++i)
      Mask.push_back(i);
    for (int i = 0; i != (int)NumSubElts; ++i) {
      int M = SubMask[i];
      if (0 <= M) {
        int InputIdx = M / NumSubElts;
        M = (NumElts * (1 + InputIdx)) + (M % NumSubElts);
      }
      Mask[i + InsertIdx] = M;
    }
    return true;
  }
  case ISD::SCALAR_TO_VECTOR: {
    // Match against a scalar_to_vector of an extract from a vector,
    // for PEXTRW/PEXTRB we must handle the implicit zext of the scalar.
    SDValue N0 = N.getOperand(0);
    SDValue SrcExtract;

    if ((N0.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
         N0.getOperand(0).getValueType() == VT) ||
        (N0.getOpcode() == X86ISD::PEXTRW &&
         N0.getOperand(0).getValueType() == MVT::v8i16) ||
        (N0.getOpcode() == X86ISD::PEXTRB &&
         N0.getOperand(0).getValueType() == MVT::v16i8)) {
      SrcExtract = N0;
    }

    if (!SrcExtract || !isa<ConstantSDNode>(SrcExtract.getOperand(1)))
      return false;

    SDValue SrcVec = SrcExtract.getOperand(0);
    EVT SrcVT = SrcVec.getValueType();
    unsigned NumSrcElts = SrcVT.getVectorNumElements();
    unsigned NumZeros = (NumBitsPerElt / SrcVT.getScalarSizeInBits()) - 1;

    unsigned SrcIdx = SrcExtract.getConstantOperandVal(1);
    if (NumSrcElts <= SrcIdx)
      return false;

    Ops.push_back(SrcVec);
    Mask.push_back(SrcIdx);
    Mask.append(NumZeros, SM_SentinelZero);
    Mask.append(NumSrcElts - Mask.size(), SM_SentinelUndef);
    return true;
  }
  case X86ISD::PINSRB:
  case X86ISD::PINSRW: {
    SDValue InVec = N.getOperand(0);
    SDValue InScl = N.getOperand(1);
    SDValue InIndex = N.getOperand(2);
    if (!isa<ConstantSDNode>(InIndex) ||
        cast<ConstantSDNode>(InIndex)->getAPIntValue().uge(NumElts))
      return false;
    uint64_t InIdx = N.getConstantOperandVal(2);

    // Attempt to recognise a PINSR*(VEC, 0, Idx) shuffle pattern.
    if (X86::isZeroNode(InScl)) {
      Ops.push_back(InVec);
      for (unsigned i = 0; i != NumElts; ++i)
        Mask.push_back(i == InIdx ? SM_SentinelZero : (int)i);
      return true;
    }

    // Attempt to recognise a PINSR*(PEXTR*) shuffle pattern.
    // TODO: Expand this to support INSERT_VECTOR_ELT/etc.
    unsigned ExOp =
        (X86ISD::PINSRB == Opcode ? X86ISD::PEXTRB : X86ISD::PEXTRW);
    if (InScl.getOpcode() != ExOp)
      return false;

    SDValue ExVec = InScl.getOperand(0);
    SDValue ExIndex = InScl.getOperand(1);
    if (!isa<ConstantSDNode>(ExIndex) ||
        cast<ConstantSDNode>(ExIndex)->getAPIntValue().uge(NumElts))
      return false;
    uint64_t ExIdx = InScl.getConstantOperandVal(1);

    Ops.push_back(InVec);
    Ops.push_back(ExVec);
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(i == InIdx ? NumElts + ExIdx : i);
    return true;
  }
  case X86ISD::PACKSS:
  case X86ISD::PACKUS: {
    SDValue N0 = N.getOperand(0);
    SDValue N1 = N.getOperand(1);
    assert(N0.getValueType().getVectorNumElements() == (NumElts / 2) &&
           N1.getValueType().getVectorNumElements() == (NumElts / 2) &&
           "Unexpected input value type");

    APInt EltsLHS, EltsRHS;
    getPackDemandedElts(VT, DemandedElts, EltsLHS, EltsRHS);

    // If we know input saturation won't happen we can treat this
    // as a truncation shuffle.
    if (Opcode == X86ISD::PACKSS) {
      if ((!N0.isUndef() &&
           DAG.ComputeNumSignBits(N0, EltsLHS, Depth + 1) <= NumBitsPerElt) ||
          (!N1.isUndef() &&
           DAG.ComputeNumSignBits(N1, EltsRHS, Depth + 1) <= NumBitsPerElt))
        return false;
    } else {
      APInt ZeroMask = APInt::getHighBitsSet(2 * NumBitsPerElt, NumBitsPerElt);
      if ((!N0.isUndef() &&
           !DAG.MaskedValueIsZero(N0, ZeroMask, EltsLHS, Depth + 1)) ||
          (!N1.isUndef() &&
           !DAG.MaskedValueIsZero(N1, ZeroMask, EltsRHS, Depth + 1)))
        return false;
    }

    bool IsUnary = (N0 == N1);

    Ops.push_back(N0);
    if (!IsUnary)
      Ops.push_back(N1);

    createPackShuffleMask(VT, Mask, IsUnary);
    return true;
  }
  case X86ISD::VSHLI:
  case X86ISD::VSRLI: {
    uint64_t ShiftVal = N.getConstantOperandVal(1);
    // Out of range bit shifts are guaranteed to be zero.
    if (NumBitsPerElt <= ShiftVal) {
      Mask.append(NumElts, SM_SentinelZero);
      return true;
    }

    // We can only decode 'whole byte' bit shifts as shuffles.
    if ((ShiftVal % 8) != 0)
      break;

    uint64_t ByteShift = ShiftVal / 8;
    unsigned NumBytes = NumSizeInBits / 8;
    unsigned NumBytesPerElt = NumBitsPerElt / 8;
    Ops.push_back(N.getOperand(0));

    // Clear mask to all zeros and insert the shifted byte indices.
    Mask.append(NumBytes, SM_SentinelZero);

    if (X86ISD::VSHLI == Opcode) {
      for (unsigned i = 0; i != NumBytes; i += NumBytesPerElt)
        for (unsigned j = ByteShift; j != NumBytesPerElt; ++j)
          Mask[i + j] = i + j - ByteShift;
    } else {
      for (unsigned i = 0; i != NumBytes; i += NumBytesPerElt)
        for (unsigned j = ByteShift; j != NumBytesPerElt; ++j)
          Mask[i + j - ByteShift] = i + j;
    }
    return true;
  }
  case X86ISD::VBROADCAST: {
    SDValue Src = N.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();
    if (!SrcVT.isVector())
      return false;

    if (NumSizeInBits != SrcVT.getSizeInBits()) {
      assert((NumSizeInBits % SrcVT.getSizeInBits()) == 0 &&
             "Illegal broadcast type");
      SrcVT = MVT::getVectorVT(SrcVT.getScalarType(),
                               NumSizeInBits / SrcVT.getScalarSizeInBits());
      Src = DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N), SrcVT,
                        DAG.getUNDEF(SrcVT), Src,
                        DAG.getIntPtrConstant(0, SDLoc(N)));
    }

    Ops.push_back(Src);
    Mask.append(NumElts, 0);
    return true;
  }
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
  case ISD::ZERO_EXTEND_VECTOR_INREG:
  case ISD::ANY_EXTEND_VECTOR_INREG: {
    SDValue Src = N.getOperand(0);
    EVT SrcVT = Src.getValueType();

    // Extended source must be a simple vector.
    if (!SrcVT.isSimple() || (SrcVT.getSizeInBits() % 128) != 0 ||
        (SrcVT.getScalarSizeInBits() % 8) != 0)
      return false;

    unsigned NumSrcBitsPerElt = SrcVT.getScalarSizeInBits();
    bool IsAnyExtend =
        (ISD::ANY_EXTEND == Opcode || ISD::ANY_EXTEND_VECTOR_INREG == Opcode);
    DecodeZeroExtendMask(NumSrcBitsPerElt, NumBitsPerElt, NumElts, IsAnyExtend,
                         Mask);

    if (NumSizeInBits != SrcVT.getSizeInBits()) {
      assert((NumSizeInBits % SrcVT.getSizeInBits()) == 0 &&
             "Illegal zero-extension type");
      SrcVT = MVT::getVectorVT(SrcVT.getSimpleVT().getScalarType(),
                               NumSizeInBits / NumSrcBitsPerElt);
      Src = DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N), SrcVT,
                        DAG.getUNDEF(SrcVT), Src,
                        DAG.getIntPtrConstant(0, SDLoc(N)));
    }

    Ops.push_back(Src);
    return true;
  }
  }

  return false;
}

/// Removes unused/repeated shuffle source inputs and adjusts the shuffle mask.
static void resolveTargetShuffleInputsAndMask(SmallVectorImpl<SDValue> &Inputs,
                                              SmallVectorImpl<int> &Mask) {
  int MaskWidth = Mask.size();
  SmallVector<SDValue, 16> UsedInputs;
  for (int i = 0, e = Inputs.size(); i < e; ++i) {
    int lo = UsedInputs.size() * MaskWidth;
    int hi = lo + MaskWidth;

    // Strip UNDEF input usage.
    if (Inputs[i].isUndef())
      for (int &M : Mask)
        if ((lo <= M) && (M < hi))
          M = SM_SentinelUndef;

    // Check for unused inputs.
    if (none_of(Mask, [lo, hi](int i) { return (lo <= i) && (i < hi); })) {
      for (int &M : Mask)
        if (lo <= M)
          M -= MaskWidth;
      continue;
    }

    // Check for repeated inputs.
    bool IsRepeat = false;
    for (int j = 0, ue = UsedInputs.size(); j != ue; ++j) {
      if (UsedInputs[j] != Inputs[i])
        continue;
      for (int &M : Mask)
        if (lo <= M)
          M = (M < hi) ? ((M - lo) + (j * MaskWidth)) : (M - MaskWidth);
      IsRepeat = true;
      break;
    }
    if (IsRepeat)
      continue;

    UsedInputs.push_back(Inputs[i]);
  }
  Inputs = UsedInputs;
}

/// Calls setTargetShuffleZeroElements to resolve a target shuffle mask's inputs
/// and set the SM_SentinelUndef and SM_SentinelZero values.
/// Returns true if the target shuffle mask was decoded.
static bool getTargetShuffleInputs(SDValue Op, const APInt &DemandedElts,
                                   SmallVectorImpl<SDValue> &Inputs,
                                   SmallVectorImpl<int> &Mask,
                                   SelectionDAG &DAG, unsigned Depth,
                                   bool ResolveZero) {
  if (!setTargetShuffleZeroElements(Op, Mask, Inputs, ResolveZero))
    if (!getFauxShuffleMask(Op, DemandedElts, Mask, Inputs, DAG, Depth,
                            ResolveZero))
      return false;
  return true;
}

/// Calls setTargetShuffleZeroElements to resolve a target shuffle mask's inputs
/// and set the SM_SentinelUndef and SM_SentinelZero values. Then check the
/// remaining input indices in case we now have a unary shuffle and adjust the
/// inputs accordingly.
/// Returns true if the target shuffle mask was decoded.
static bool resolveTargetShuffleInputs(SDValue Op, const APInt &DemandedElts,
                                       SmallVectorImpl<SDValue> &Inputs,
                                       SmallVectorImpl<int> &Mask,
                                       SelectionDAG &DAG, unsigned Depth,
                                       bool ResolveZero) {
  if (!setTargetShuffleZeroElements(Op, Mask, Inputs, ResolveZero))
    if (!getFauxShuffleMask(Op, DemandedElts, Mask, Inputs, DAG, Depth,
                            ResolveZero))
      return false;

  resolveTargetShuffleInputsAndMask(Inputs, Mask);
  return true;
}

bool X86::resolveTargetShuffleInputs(SDValue Op,
                                     SmallVectorImpl<SDValue> &Inputs,
                                     SmallVectorImpl<int> &Mask,
                                     SelectionDAG &DAG, unsigned Depth,
                                     bool ResolveZero) {
  unsigned NumElts = Op.getValueType().getVectorNumElements();
  APInt DemandedElts = APInt::getAllOnesValue(NumElts);
  return ::resolveTargetShuffleInputs(Op, DemandedElts, Inputs, Mask, DAG,
                                      Depth, ResolveZero);
}

/// Returns the scalar element that will make up the ith
/// element of the result of the vector shuffle.
static SDValue getShuffleScalarElt(SDNode *N, unsigned Index, SelectionDAG &DAG,
                                   unsigned Depth) {
  if (Depth == 6)
    return SDValue();  // Limit search depth.

  SDValue V = SDValue(N, 0);
  EVT VT = V.getValueType();
  unsigned Opcode = V.getOpcode();

  // Recurse into ISD::VECTOR_SHUFFLE node to find scalars.
  if (const ShuffleVectorSDNode *SV = dyn_cast<ShuffleVectorSDNode>(N)) {
    int Elt = SV->getMaskElt(Index);

    if (Elt < 0)
      return DAG.getUNDEF(VT.getVectorElementType());

    unsigned NumElems = VT.getVectorNumElements();
    SDValue NewV = (Elt < (int)NumElems) ? SV->getOperand(0)
                                         : SV->getOperand(1);
    return getShuffleScalarElt(NewV.getNode(), Elt % NumElems, DAG, Depth+1);
  }

  // Recurse into target specific vector shuffles to find scalars.
  if (isTargetShuffle(Opcode)) {
    MVT ShufVT = V.getSimpleValueType();
    MVT ShufSVT = ShufVT.getVectorElementType();
    int NumElems = (int)ShufVT.getVectorNumElements();
    SmallVector<int, 16> ShuffleMask;
    SmallVector<SDValue, 16> ShuffleOps;
    bool IsUnary;

    if (!getTargetShuffleMask(N, ShufVT, true, ShuffleOps, ShuffleMask, IsUnary))
      return SDValue();

    int Elt = ShuffleMask[Index];
    if (Elt == SM_SentinelZero)
      return ShufSVT.isInteger() ? DAG.getConstant(0, SDLoc(N), ShufSVT)
                                 : DAG.getConstantFP(+0.0, SDLoc(N), ShufSVT);
    if (Elt == SM_SentinelUndef)
      return DAG.getUNDEF(ShufSVT);

    assert(0 <= Elt && Elt < (2*NumElems) && "Shuffle index out of range");
    SDValue NewV = (Elt < NumElems) ? ShuffleOps[0] : ShuffleOps[1];
    return getShuffleScalarElt(NewV.getNode(), Elt % NumElems, DAG,
                               Depth+1);
  }

  // Recurse into insert_subvector base/sub vector to find scalars.
  if (Opcode == ISD::INSERT_SUBVECTOR &&
      isa<ConstantSDNode>(N->getOperand(2))) {
    SDValue Vec = N->getOperand(0);
    SDValue Sub = N->getOperand(1);
    EVT SubVT = Sub.getValueType();
    unsigned NumSubElts = SubVT.getVectorNumElements();
    uint64_t SubIdx = N->getConstantOperandVal(2);

    if (SubIdx <= Index && Index < (SubIdx + NumSubElts))
      return getShuffleScalarElt(Sub.getNode(), Index - SubIdx, DAG, Depth + 1);
    return getShuffleScalarElt(Vec.getNode(), Index, DAG, Depth + 1);
  }

  // Recurse into extract_subvector src vector to find scalars.
  if (Opcode == ISD::EXTRACT_SUBVECTOR &&
      isa<ConstantSDNode>(N->getOperand(1))) {
    SDValue Src = N->getOperand(0);
    uint64_t SrcIdx = N->getConstantOperandVal(1);
    return getShuffleScalarElt(Src.getNode(), Index + SrcIdx, DAG, Depth + 1);
  }

  // Actual nodes that may contain scalar elements
  if (Opcode == ISD::BITCAST) {
    V = V.getOperand(0);
    EVT SrcVT = V.getValueType();
    unsigned NumElems = VT.getVectorNumElements();

    if (!SrcVT.isVector() || SrcVT.getVectorNumElements() != NumElems)
      return SDValue();
  }

  if (V.getOpcode() == ISD::SCALAR_TO_VECTOR)
    return (Index == 0) ? V.getOperand(0)
                        : DAG.getUNDEF(VT.getVectorElementType());

  if (V.getOpcode() == ISD::BUILD_VECTOR)
    return V.getOperand(Index);

  return SDValue();
}

// Use PINSRB/PINSRW/PINSRD to create a build vector.
static SDValue LowerBuildVectorAsInsert(SDValue Op, unsigned NonZeros,
                                        unsigned NumNonZero, unsigned NumZero,
                                        SelectionDAG &DAG,
                                        const X86Subtarget &Subtarget) {
  MVT VT = Op.getSimpleValueType();
  unsigned NumElts = VT.getVectorNumElements();
  assert(((VT == MVT::v8i16 && Subtarget.hasSSE2()) ||
          ((VT == MVT::v16i8 || VT == MVT::v4i32) && Subtarget.hasSSE41())) &&
         "Illegal vector insertion");

  SDLoc dl(Op);
  SDValue V;
  bool First = true;

  for (unsigned i = 0; i < NumElts; ++i) {
    bool IsNonZero = (NonZeros & (1 << i)) != 0;
    if (!IsNonZero)
      continue;

    // If the build vector contains zeros or our first insertion is not the
    // first index then insert into zero vector to break any register
    // dependency else use SCALAR_TO_VECTOR.
    if (First) {
      First = false;
      if (NumZero || 0 != i)
        V = getZeroVector(VT, Subtarget, DAG, dl);
      else {
        assert(0 == i && "Expected insertion into zero-index");
        V = DAG.getAnyExtOrTrunc(Op.getOperand(i), dl, MVT::i32);
        V = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4i32, V);
        V = DAG.getBitcast(VT, V);
        continue;
      }
    }
    V = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, V, Op.getOperand(i),
                    DAG.getIntPtrConstant(i, dl));
  }

  return V;
}

/// Custom lower build_vector of v16i8.
static SDValue LowerBuildVectorv16i8(SDValue Op, unsigned NonZeros,
                                     unsigned NumNonZero, unsigned NumZero,
                                     SelectionDAG &DAG,
                                     const X86Subtarget &Subtarget) {
  if (NumNonZero > 8 && !Subtarget.hasSSE41())
    return SDValue();

  // SSE4.1 - use PINSRB to insert each byte directly.
  if (Subtarget.hasSSE41())
    return LowerBuildVectorAsInsert(Op, NonZeros, NumNonZero, NumZero, DAG,
                                    Subtarget);

  SDLoc dl(Op);
  SDValue V;

  // Pre-SSE4.1 - merge byte pairs and insert with PINSRW.
  for (unsigned i = 0; i < 16; i += 2) {
    bool ThisIsNonZero = (NonZeros & (1 << i)) != 0;
    bool NextIsNonZero = (NonZeros & (1 << (i + 1))) != 0;
    if (!ThisIsNonZero && !NextIsNonZero)
      continue;

    // FIXME: Investigate combining the first 4 bytes as a i32 instead.
    SDValue Elt;
    if (ThisIsNonZero) {
      if (NumZero || NextIsNonZero)
        Elt = DAG.getZExtOrTrunc(Op.getOperand(i), dl, MVT::i32);
      else
        Elt = DAG.getAnyExtOrTrunc(Op.getOperand(i), dl, MVT::i32);
    }

    if (NextIsNonZero) {
      SDValue NextElt = Op.getOperand(i + 1);
      if (i == 0 && NumZero)
        NextElt = DAG.getZExtOrTrunc(NextElt, dl, MVT::i32);
      else
        NextElt = DAG.getAnyExtOrTrunc(NextElt, dl, MVT::i32);
      NextElt = DAG.getNode(ISD::SHL, dl, MVT::i32, NextElt,
                            DAG.getConstant(8, dl, MVT::i8));
      if (ThisIsNonZero)
        Elt = DAG.getNode(ISD::OR, dl, MVT::i32, NextElt, Elt);
      else
        Elt = NextElt;
    }

    // If our first insertion is not the first index then insert into zero
    // vector to break any register dependency else use SCALAR_TO_VECTOR.
    if (!V) {
      if (i != 0)
        V = getZeroVector(MVT::v8i16, Subtarget, DAG, dl);
      else {
        V = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4i32, Elt);
        V = DAG.getBitcast(MVT::v8i16, V);
        continue;
      }
    }
    Elt = DAG.getNode(ISD::TRUNCATE, dl, MVT::i16, Elt);
    V = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, MVT::v8i16, V, Elt,
                    DAG.getIntPtrConstant(i / 2, dl));
  }

  return DAG.getBitcast(MVT::v16i8, V);
}

/// Custom lower build_vector of v8i16.
static SDValue LowerBuildVectorv8i16(SDValue Op, unsigned NonZeros,
                                     unsigned NumNonZero, unsigned NumZero,
                                     SelectionDAG &DAG,
                                     const X86Subtarget &Subtarget) {
  if (NumNonZero > 4 && !Subtarget.hasSSE41())
    return SDValue();

  // Use PINSRW to insert each byte directly.
  return LowerBuildVectorAsInsert(Op, NonZeros, NumNonZero, NumZero, DAG,
                                  Subtarget);
}

/// Custom lower build_vector of v4i32 or v4f32.
static SDValue LowerBuildVectorv4x32(SDValue Op, SelectionDAG &DAG,
                                     const X86Subtarget &Subtarget) {
  // If this is a splat of a pair of elements, use MOVDDUP (unless the target
  // has XOP; in that case defer lowering to potentially use VPERMIL2PS).
  // Because we're creating a less complicated build vector here, we may enable
  // further folding of the MOVDDUP via shuffle transforms.
  if (Subtarget.hasSSE3() && !Subtarget.hasXOP() &&
      Op.getOperand(0) == Op.getOperand(2) &&
      Op.getOperand(1) == Op.getOperand(3) &&
      Op.getOperand(0) != Op.getOperand(1)) {
    SDLoc DL(Op);
    MVT VT = Op.getSimpleValueType();
    MVT EltVT = VT.getVectorElementType();
    // Create a new build vector with the first 2 elements followed by undef
    // padding, bitcast to v2f64, duplicate, and bitcast back.
    SDValue Ops[4] = { Op.getOperand(0), Op.getOperand(1),
                       DAG.getUNDEF(EltVT), DAG.getUNDEF(EltVT) };
    SDValue NewBV = DAG.getBitcast(MVT::v2f64, DAG.getBuildVector(VT, DL, Ops));
    SDValue Dup = DAG.getNode(X86ISD::MOVDDUP, DL, MVT::v2f64, NewBV);
    return DAG.getBitcast(VT, Dup);
  }

  // Find all zeroable elements.
  std::bitset<4> Zeroable, Undefs;
  for (int i = 0; i < 4; ++i) {
    SDValue Elt = Op.getOperand(i);
    Undefs[i] = Elt.isUndef();
    Zeroable[i] = (Elt.isUndef() || X86::isZeroNode(Elt));
  }
  assert(Zeroable.size() - Zeroable.count() > 1 &&
         "We expect at least two non-zero elements!");

  // We only know how to deal with build_vector nodes where elements are either
  // zeroable or extract_vector_elt with constant index.
  SDValue FirstNonZero;
  unsigned FirstNonZeroIdx;
  for (unsigned i = 0; i < 4; ++i) {
    if (Zeroable[i])
      continue;
    SDValue Elt = Op.getOperand(i);
    if (Elt.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        !isa<ConstantSDNode>(Elt.getOperand(1)))
      return SDValue();
    // Make sure that this node is extracting from a 128-bit vector.
    MVT VT = Elt.getOperand(0).getSimpleValueType();
    if (!VT.is128BitVector())
      return SDValue();
    if (!FirstNonZero.getNode()) {
      FirstNonZero = Elt;
      FirstNonZeroIdx = i;
    }
  }

  assert(FirstNonZero.getNode() && "Unexpected build vector of all zeros!");
  SDValue V1 = FirstNonZero.getOperand(0);
  MVT VT = V1.getSimpleValueType();

  // See if this build_vector can be lowered as a blend with zero.
  SDValue Elt;
  unsigned EltMaskIdx, EltIdx;
  int Mask[4];
  for (EltIdx = 0; EltIdx < 4; ++EltIdx) {
    if (Zeroable[EltIdx]) {
      // The zero vector will be on the right hand side.
      Mask[EltIdx] = EltIdx+4;
      continue;
    }

    Elt = Op->getOperand(EltIdx);
    // By construction, Elt is a EXTRACT_VECTOR_ELT with constant index.
    EltMaskIdx = Elt.getConstantOperandVal(1);
    if (Elt.getOperand(0) != V1 || EltMaskIdx != EltIdx)
      break;
    Mask[EltIdx] = EltIdx;
  }

  if (EltIdx == 4) {
    // Let the shuffle legalizer deal with blend operations.
    SDValue VZeroOrUndef = (Zeroable == Undefs)
                               ? DAG.getUNDEF(VT)
                               : getZeroVector(VT, Subtarget, DAG, SDLoc(Op));
    if (V1.getSimpleValueType() != VT)
      V1 = DAG.getBitcast(VT, V1);
    return DAG.getVectorShuffle(VT, SDLoc(V1), V1, VZeroOrUndef, Mask);
  }

  // See if we can lower this build_vector to a INSERTPS.
  if (!Subtarget.hasSSE41())
    return SDValue();

  SDValue V2 = Elt.getOperand(0);
  if (Elt == FirstNonZero && EltIdx == FirstNonZeroIdx)
    V1 = SDValue();

  bool CanFold = true;
  for (unsigned i = EltIdx + 1; i < 4 && CanFold; ++i) {
    if (Zeroable[i])
      continue;

    SDValue Current = Op->getOperand(i);
    SDValue SrcVector = Current->getOperand(0);
    if (!V1.getNode())
      V1 = SrcVector;
    CanFold = (SrcVector == V1) && (Current.getConstantOperandAPInt(1) == i);
  }

  if (!CanFold)
    return SDValue();

  assert(V1.getNode() && "Expected at least two non-zero elements!");
  if (V1.getSimpleValueType() != MVT::v4f32)
    V1 = DAG.getBitcast(MVT::v4f32, V1);
  if (V2.getSimpleValueType() != MVT::v4f32)
    V2 = DAG.getBitcast(MVT::v4f32, V2);

  // Ok, we can emit an INSERTPS instruction.
  unsigned ZMask = Zeroable.to_ulong();

  unsigned InsertPSMask = EltMaskIdx << 6 | EltIdx << 4 | ZMask;
  assert((InsertPSMask & ~0xFFu) == 0 && "Invalid mask!");
  SDLoc DL(Op);
  SDValue Result = DAG.getNode(X86ISD::INSERTPS, DL, MVT::v4f32, V1, V2,
                               DAG.getIntPtrConstant(InsertPSMask, DL));
  return DAG.getBitcast(VT, Result);
}

/// Return a vector logical shift node.
static SDValue getVShift(bool isLeft, EVT VT, SDValue SrcOp, unsigned NumBits,
                         SelectionDAG &DAG, const TargetLowering &TLI,
                         const SDLoc &dl) {
  assert(VT.is128BitVector() && "Unknown type for VShift");
  MVT ShVT = MVT::v16i8;
  unsigned Opc = isLeft ? X86ISD::VSHLDQ : X86ISD::VSRLDQ;
  SrcOp = DAG.getBitcast(ShVT, SrcOp);
  assert(NumBits % 8 == 0 && "Only support byte sized shifts");
  SDValue ShiftVal = DAG.getConstant(NumBits/8, dl, MVT::i8);
  return DAG.getBitcast(VT, DAG.getNode(Opc, dl, ShVT, SrcOp, ShiftVal));
}

static SDValue LowerAsSplatVectorLoad(SDValue SrcOp, MVT VT, const SDLoc &dl,
                                      SelectionDAG &DAG) {

  // Check if the scalar load can be widened into a vector load. And if
  // the address is "base + cst" see if the cst can be "absorbed" into
  // the shuffle mask.
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(SrcOp)) {
    SDValue Ptr = LD->getBasePtr();
    if (!ISD::isNormalLoad(LD) || !LD->isSimple())
      return SDValue();
    EVT PVT = LD->getValueType(0);
    if (PVT != MVT::i32 && PVT != MVT::f32)
      return SDValue();

    int FI = -1;
    int64_t Offset = 0;
    if (FrameIndexSDNode *FINode = dyn_cast<FrameIndexSDNode>(Ptr)) {
      FI = FINode->getIndex();
      Offset = 0;
    } else if (DAG.isBaseWithConstantOffset(Ptr) &&
               isa<FrameIndexSDNode>(Ptr.getOperand(0))) {
      FI = cast<FrameIndexSDNode>(Ptr.getOperand(0))->getIndex();
      Offset = Ptr.getConstantOperandVal(1);
      Ptr = Ptr.getOperand(0);
    } else {
      return SDValue();
    }

    // FIXME: 256-bit vector instructions don't require a strict alignment,
    // improve this code to support it better.
    unsigned RequiredAlign = VT.getSizeInBits()/8;
    SDValue Chain = LD->getChain();
    // Make sure the stack object alignment is at least 16 or 32.
    MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();
    if (DAG.InferPtrAlignment(Ptr) < RequiredAlign) {
      if (MFI.isFixedObjectIndex(FI)) {
        // Can't change the alignment. FIXME: It's possible to compute
        // the exact stack offset and reference FI + adjust offset instead.
        // If someone *really* cares about this. That's the way to implement it.
        return SDValue();
      } else {
        MFI.setObjectAlignment(FI, RequiredAlign);
      }
    }

    // (Offset % 16 or 32) must be multiple of 4. Then address is then
    // Ptr + (Offset & ~15).
    if (Offset < 0)
      return SDValue();
    if ((Offset % RequiredAlign) & 3)
      return SDValue();
    int64_t StartOffset = Offset & ~int64_t(RequiredAlign - 1);
    if (StartOffset) {
      SDLoc DL(Ptr);
      Ptr = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(), Ptr,
                        DAG.getConstant(StartOffset, DL, Ptr.getValueType()));
    }

    int EltNo = (Offset - StartOffset) >> 2;
    unsigned NumElems = VT.getVectorNumElements();

    EVT NVT = EVT::getVectorVT(*DAG.getContext(), PVT, NumElems);
    SDValue V1 = DAG.getLoad(NVT, dl, Chain, Ptr,
                             LD->getPointerInfo().getWithOffset(StartOffset));

    SmallVector<int, 8> Mask(NumElems, EltNo);

    return DAG.getVectorShuffle(NVT, dl, V1, DAG.getUNDEF(NVT), Mask);
  }

  return SDValue();
}

// Recurse to find a LoadSDNode source and the accumulated ByteOffest.
static bool findEltLoadSrc(SDValue Elt, LoadSDNode *&Ld, int64_t &ByteOffset) {
  if (ISD::isNON_EXTLoad(Elt.getNode())) {
    auto *BaseLd = cast<LoadSDNode>(Elt);
    if (BaseLd->getMemOperand()->getFlags() & MachineMemOperand::MOVolatile)
      return false;
    Ld = BaseLd;
    ByteOffset = 0;
    return true;
  }

  switch (Elt.getOpcode()) {
  case ISD::BITCAST:
  case ISD::TRUNCATE:
  case ISD::SCALAR_TO_VECTOR:
    return findEltLoadSrc(Elt.getOperand(0), Ld, ByteOffset);
  case ISD::SRL:
    if (isa<ConstantSDNode>(Elt.getOperand(1))) {
      uint64_t Idx = Elt.getConstantOperandVal(1);
      if ((Idx % 8) == 0 && findEltLoadSrc(Elt.getOperand(0), Ld, ByteOffset)) {
        ByteOffset += Idx / 8;
        return true;
      }
    }
    break;
  case ISD::EXTRACT_VECTOR_ELT:
    if (isa<ConstantSDNode>(Elt.getOperand(1))) {
      SDValue Src = Elt.getOperand(0);
      unsigned SrcSizeInBits = Src.getScalarValueSizeInBits();
      unsigned DstSizeInBits = Elt.getScalarValueSizeInBits();
      if (DstSizeInBits == SrcSizeInBits && (SrcSizeInBits % 8) == 0 &&
          findEltLoadSrc(Src, Ld, ByteOffset)) {
        uint64_t Idx = Elt.getConstantOperandVal(1);
        ByteOffset += Idx * (SrcSizeInBits / 8);
        return true;
      }
    }
    break;
  }

  return false;
}

/// Given the initializing elements 'Elts' of a vector of type 'VT', see if the
/// elements can be replaced by a single large load which has the same value as
/// a build_vector or insert_subvector whose loaded operands are 'Elts'.
///
/// Example: <load i32 *a, load i32 *a+4, zero, undef> -> zextload a
static SDValue EltsFromConsecutiveLoads(EVT VT, ArrayRef<SDValue> Elts,
                                        const SDLoc &DL, SelectionDAG &DAG,
                                        const X86Subtarget &Subtarget,
                                        bool isAfterLegalize) {
  if ((VT.getScalarSizeInBits() % 8) != 0)
    return SDValue();

  unsigned NumElems = Elts.size();

  int LastLoadedElt = -1;
  APInt LoadMask = APInt::getNullValue(NumElems);
  APInt ZeroMask = APInt::getNullValue(NumElems);
  APInt UndefMask = APInt::getNullValue(NumElems);

  SmallVector<LoadSDNode*, 8> Loads(NumElems, nullptr);
  SmallVector<int64_t, 8> ByteOffsets(NumElems, 0);

  // For each element in the initializer, see if we've found a load, zero or an
  // undef.
  for (unsigned i = 0; i < NumElems; ++i) {
    SDValue Elt = peekThroughBitcasts(Elts[i]);
    if (!Elt.getNode())
      return SDValue();
    if (Elt.isUndef()) {
      UndefMask.setBit(i);
      continue;
    }
    if (X86::isZeroNode(Elt) || ISD::isBuildVectorAllZeros(Elt.getNode())) {
      ZeroMask.setBit(i);
      continue;
    }

    // Each loaded element must be the correct fractional portion of the
    // requested vector load.
    unsigned EltSizeInBits = Elt.getValueSizeInBits();
    if ((NumElems * EltSizeInBits) != VT.getSizeInBits())
      return SDValue();

    if (!findEltLoadSrc(Elt, Loads[i], ByteOffsets[i]) || ByteOffsets[i] < 0)
      return SDValue();
    unsigned LoadSizeInBits = Loads[i]->getValueSizeInBits(0);
    if (((ByteOffsets[i] * 8) + EltSizeInBits) > LoadSizeInBits)
      return SDValue();

    LoadMask.setBit(i);
    LastLoadedElt = i;
  }
  assert((ZeroMask.countPopulation() + UndefMask.countPopulation() +
          LoadMask.countPopulation()) == NumElems &&
         "Incomplete element masks");

  // Handle Special Cases - all undef or undef/zero.
  if (UndefMask.countPopulation() == NumElems)
    return DAG.getUNDEF(VT);

  // FIXME: Should we return this as a BUILD_VECTOR instead?
  if ((ZeroMask.countPopulation() + UndefMask.countPopulation()) == NumElems)
    return VT.isInteger() ? DAG.getConstant(0, DL, VT)
                          : DAG.getConstantFP(0.0, DL, VT);

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  int FirstLoadedElt = LoadMask.countTrailingZeros();
  SDValue EltBase = peekThroughBitcasts(Elts[FirstLoadedElt]);
  EVT EltBaseVT = EltBase.getValueType();
  assert(EltBaseVT.getSizeInBits() == EltBaseVT.getStoreSizeInBits() &&
         "Register/Memory size mismatch");
  LoadSDNode *LDBase = Loads[FirstLoadedElt];
  assert(LDBase && "Did not find base load for merging consecutive loads");
  unsigned BaseSizeInBits = EltBaseVT.getStoreSizeInBits();
  unsigned BaseSizeInBytes = BaseSizeInBits / 8;
  int LoadSizeInBits = (1 + LastLoadedElt - FirstLoadedElt) * BaseSizeInBits;
  assert((BaseSizeInBits % 8) == 0 && "Sub-byte element loads detected");

  // TODO: Support offsetting the base load.
  if (ByteOffsets[FirstLoadedElt] != 0)
    return SDValue();

  // Check to see if the element's load is consecutive to the base load
  // or offset from a previous (already checked) load.
  auto CheckConsecutiveLoad = [&](LoadSDNode *Base, int EltIdx) {
    LoadSDNode *Ld = Loads[EltIdx];
    int64_t ByteOffset = ByteOffsets[EltIdx];
    if (ByteOffset && (ByteOffset % BaseSizeInBytes) == 0) {
      int64_t BaseIdx = EltIdx - (ByteOffset / BaseSizeInBytes);
      return (0 <= BaseIdx && BaseIdx < (int)NumElems && LoadMask[BaseIdx] &&
              Loads[BaseIdx] == Ld && ByteOffsets[BaseIdx] == 0);
    }
    return DAG.areNonVolatileConsecutiveLoads(Ld, Base, BaseSizeInBytes,
                                              EltIdx - FirstLoadedElt);
  };

  // Consecutive loads can contain UNDEFS but not ZERO elements.
  // Consecutive loads with UNDEFs and ZEROs elements require a
  // an additional shuffle stage to clear the ZERO elements.
  bool IsConsecutiveLoad = true;
  bool IsConsecutiveLoadWithZeros = true;
  for (int i = FirstLoadedElt + 1; i <= LastLoadedElt; ++i) {
    if (LoadMask[i]) {
      if (!CheckConsecutiveLoad(LDBase, i)) {
        IsConsecutiveLoad = false;
        IsConsecutiveLoadWithZeros = false;
        break;
      }
    } else if (ZeroMask[i]) {
      IsConsecutiveLoad = false;
    }
  }

  auto CreateLoad = [&DAG, &DL, &Loads](EVT VT, LoadSDNode *LDBase) {
    auto MMOFlags = LDBase->getMemOperand()->getFlags();
    assert(!(MMOFlags & MachineMemOperand::MOVolatile) &&
           "Cannot merge volatile loads.");
    SDValue NewLd =
        DAG.getLoad(VT, DL, LDBase->getChain(), LDBase->getBasePtr(),
                    LDBase->getPointerInfo(), LDBase->getAlignment(), MMOFlags);
    for (auto *LD : Loads)
      if (LD)
        DAG.makeEquivalentMemoryOrdering(LD, NewLd);
    return NewLd;
  };

  // Check if the base load is entirely dereferenceable.
  bool IsDereferenceable = LDBase->getPointerInfo().isDereferenceable(
      VT.getSizeInBits() / 8, *DAG.getContext(), DAG.getDataLayout());

  // LOAD - all consecutive load/undefs (must start/end with a load or be
  // entirely dereferenceable). If we have found an entire vector of loads and
  // undefs, then return a large load of the entire vector width starting at the
  // base pointer. If the vector contains zeros, then attempt to shuffle those
  // elements.
  if (FirstLoadedElt == 0 &&
      (LastLoadedElt == (int)(NumElems - 1) || IsDereferenceable) &&
      (IsConsecutiveLoad || IsConsecutiveLoadWithZeros)) {
    if (isAfterLegalize && !TLI.isOperationLegal(ISD::LOAD, VT))
      return SDValue();

    // Don't create 256-bit non-temporal aligned loads without AVX2 as these
    // will lower to regular temporal loads and use the cache.
    if (LDBase->isNonTemporal() && LDBase->getAlignment() >= 32 &&
        VT.is256BitVector() && !Subtarget.hasInt256())
      return SDValue();

    if (NumElems == 1)
      return DAG.getBitcast(VT, Elts[FirstLoadedElt]);

    if (!ZeroMask)
      return CreateLoad(VT, LDBase);

    // IsConsecutiveLoadWithZeros - we need to create a shuffle of the loaded
    // vector and a zero vector to clear out the zero elements.
    if (!isAfterLegalize && VT.isVector()) {
      unsigned NumMaskElts = VT.getVectorNumElements();
      if ((NumMaskElts % NumElems) == 0) {
        unsigned Scale = NumMaskElts / NumElems;
        SmallVector<int, 4> ClearMask(NumMaskElts, -1);
        for (unsigned i = 0; i < NumElems; ++i) {
          if (UndefMask[i])
            continue;
          int Offset = ZeroMask[i] ? NumMaskElts : 0;
          for (unsigned j = 0; j != Scale; ++j)
            ClearMask[(i * Scale) + j] = (i * Scale) + j + Offset;
        }
        SDValue V = CreateLoad(VT, LDBase);
        SDValue Z = VT.isInteger() ? DAG.getConstant(0, DL, VT)
                                   : DAG.getConstantFP(0.0, DL, VT);
        return DAG.getVectorShuffle(VT, DL, V, Z, ClearMask);
      }
    }
  }

  // If the upper half of a ymm/zmm load is undef then just load the lower half.
  if (VT.is256BitVector() || VT.is512BitVector()) {
    unsigned HalfNumElems = NumElems / 2;
    if (UndefMask.extractBits(HalfNumElems, HalfNumElems).isAllOnesValue()) {
      EVT HalfVT =
          EVT::getVectorVT(*DAG.getContext(), VT.getScalarType(), HalfNumElems);
      SDValue HalfLD =
          EltsFromConsecutiveLoads(HalfVT, Elts.drop_back(HalfNumElems), DL,
                                   DAG, Subtarget, isAfterLegalize);
      if (HalfLD)
        return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, DAG.getUNDEF(VT),
                           HalfLD, DAG.getIntPtrConstant(0, DL));
    }
  }

  // VZEXT_LOAD - consecutive 32/64-bit load/undefs followed by zeros/undefs.
  if (IsConsecutiveLoad && FirstLoadedElt == 0 &&
      (LoadSizeInBits == 32 || LoadSizeInBits == 64) &&
      ((VT.is128BitVector() || VT.is256BitVector() || VT.is512BitVector()))) {
    MVT VecSVT = VT.isFloatingPoint() ? MVT::getFloatingPointVT(LoadSizeInBits)
                                      : MVT::getIntegerVT(LoadSizeInBits);
    MVT VecVT = MVT::getVectorVT(VecSVT, VT.getSizeInBits() / LoadSizeInBits);
    if (TLI.isTypeLegal(VecVT)) {
      SDVTList Tys = DAG.getVTList(VecVT, MVT::Other);
      SDValue Ops[] = { LDBase->getChain(), LDBase->getBasePtr() };
      SDValue ResNode =
          DAG.getMemIntrinsicNode(X86ISD::VZEXT_LOAD, DL, Tys, Ops, VecSVT,
                                  LDBase->getPointerInfo(),
                                  LDBase->getAlignment(),
                                  MachineMemOperand::MOLoad);
      for (auto *LD : Loads)
        if (LD)
          DAG.makeEquivalentMemoryOrdering(LD, ResNode);
      return DAG.getBitcast(VT, ResNode);
    }
  }

  // BROADCAST - match the smallest possible repetition pattern, load that
  // scalar/subvector element and then broadcast to the entire vector.
  if (ZeroMask.isNullValue() && isPowerOf2_32(NumElems) && Subtarget.hasAVX() &&
      (VT.is128BitVector() || VT.is256BitVector() || VT.is512BitVector())) {
    for (unsigned SubElems = 1; SubElems < NumElems; SubElems *= 2) {
      unsigned RepeatSize = SubElems * BaseSizeInBits;
      unsigned ScalarSize = std::min(RepeatSize, 64u);
      if (!Subtarget.hasAVX2() && ScalarSize < 32)
        continue;

      bool Match = true;
      SmallVector<SDValue, 8> RepeatedLoads(SubElems, DAG.getUNDEF(EltBaseVT));
      for (unsigned i = 0; i != NumElems && Match; ++i) {
        if (!LoadMask[i])
          continue;
        SDValue Elt = peekThroughBitcasts(Elts[i]);
        if (RepeatedLoads[i % SubElems].isUndef())
          RepeatedLoads[i % SubElems] = Elt;
        else
          Match &= (RepeatedLoads[i % SubElems] == Elt);
      }

      // We must have loads at both ends of the repetition.
      Match &= !RepeatedLoads.front().isUndef();
      Match &= !RepeatedLoads.back().isUndef();
      if (!Match)
        continue;

      EVT RepeatVT =
          VT.isInteger() && (RepeatSize != 64 || TLI.isTypeLegal(MVT::i64))
              ? EVT::getIntegerVT(*DAG.getContext(), ScalarSize)
              : EVT::getFloatingPointVT(ScalarSize);
      if (RepeatSize > ScalarSize)
        RepeatVT = EVT::getVectorVT(*DAG.getContext(), RepeatVT,
                                    RepeatSize / ScalarSize);
      EVT BroadcastVT =
          EVT::getVectorVT(*DAG.getContext(), RepeatVT.getScalarType(),
                           VT.getSizeInBits() / ScalarSize);
      if (TLI.isTypeLegal(BroadcastVT)) {
        if (SDValue RepeatLoad = EltsFromConsecutiveLoads(
                RepeatVT, RepeatedLoads, DL, DAG, Subtarget, isAfterLegalize)) {
          unsigned Opcode = RepeatSize > ScalarSize ? X86ISD::SUBV_BROADCAST
                                                    : X86ISD::VBROADCAST;
          SDValue Broadcast = DAG.getNode(Opcode, DL, BroadcastVT, RepeatLoad);
          return DAG.getBitcast(VT, Broadcast);
        }
      }
    }
  }

  return SDValue();
}

// Combine a vector ops (shuffles etc.) that is equal to build_vector load1,
// load2, load3, load4, <0, 1, 2, 3> into a vector load if the load addresses
// are consecutive, non-overlapping, and in the right order.
static SDValue combineToConsecutiveLoads(EVT VT, SDNode *N, const SDLoc &DL,
                                         SelectionDAG &DAG,
                                         const X86Subtarget &Subtarget,
                                         bool isAfterLegalize) {
  SmallVector<SDValue, 64> Elts;
  for (unsigned i = 0, e = VT.getVectorNumElements(); i != e; ++i) {
    if (SDValue Elt = getShuffleScalarElt(N, i, DAG, 0)) {
      Elts.push_back(Elt);
      continue;
    }
    return SDValue();
  }
  assert(Elts.size() == VT.getVectorNumElements());
  return EltsFromConsecutiveLoads(VT, Elts, DL, DAG, Subtarget,
                                  isAfterLegalize);
}

static Constant *getConstantVector(MVT VT, const APInt &SplatValue,
                                   unsigned SplatBitSize, LLVMContext &C) {
  unsigned ScalarSize = VT.getScalarSizeInBits();
  unsigned NumElm = SplatBitSize / ScalarSize;

  SmallVector<Constant *, 32> ConstantVec;
  for (unsigned i = 0; i < NumElm; i++) {
    APInt Val = SplatValue.extractBits(ScalarSize, ScalarSize * i);
    Constant *Const;
    if (VT.isFloatingPoint()) {
      if (ScalarSize == 32) {
        Const = ConstantFP::get(C, APFloat(APFloat::IEEEsingle(), Val));
      } else {
        assert(ScalarSize == 64 && "Unsupported floating point scalar size");
        Const = ConstantFP::get(C, APFloat(APFloat::IEEEdouble(), Val));
      }
    } else
      Const = Constant::getIntegerValue(Type::getIntNTy(C, ScalarSize), Val);
    ConstantVec.push_back(Const);
  }
  return ConstantVector::get(ArrayRef<Constant *>(ConstantVec));
}

static bool isFoldableUseOfShuffle(SDNode *N) {
  for (auto *U : N->uses()) {
    unsigned Opc = U->getOpcode();
    // VPERMV/VPERMV3 shuffles can never fold their index operands.
    if (Opc == X86ISD::VPERMV && U->getOperand(0).getNode() == N)
      return false;
    if (Opc == X86ISD::VPERMV3 && U->getOperand(1).getNode() == N)
      return false;
    if (isTargetShuffle(Opc))
      return true;
    if (Opc == ISD::BITCAST) // Ignore bitcasts
      return isFoldableUseOfShuffle(U);
    if (N->hasOneUse())
      return true;
  }
  return false;
}

// Check if the current node of build vector is a zero extended vector.
// // If so, return the value extended.
// // For example: (0,0,0,a,0,0,0,a,0,0,0,a,0,0,0,a) returns a.
// // NumElt - return the number of zero extended identical values.
// // EltType - return the type of the value include the zero extend.
static SDValue isSplatZeroExtended(const BuildVectorSDNode *Op,
                                   unsigned &NumElt, MVT &EltType) {
  SDValue ExtValue = Op->getOperand(0);
  unsigned NumElts = Op->getNumOperands();
  unsigned Delta = NumElts;

  for (unsigned i = 1; i < NumElts; i++) {
    if (Op->getOperand(i) == ExtValue) {
      Delta = i;
      break;
    }
    if (!(Op->getOperand(i).isUndef() || isNullConstant(Op->getOperand(i))))
      return SDValue();
  }
  if (!isPowerOf2_32(Delta) || Delta == 1)
    return SDValue();

  for (unsigned i = Delta; i < NumElts; i++) {
    if (i % Delta == 0) {
      if (Op->getOperand(i) != ExtValue)
        return SDValue();
    } else if (!(isNullConstant(Op->getOperand(i)) ||
                 Op->getOperand(i).isUndef()))
      return SDValue();
  }
  unsigned EltSize = Op->getSimpleValueType(0).getScalarSizeInBits();
  unsigned ExtVTSize = EltSize * Delta;
  EltType = MVT::getIntegerVT(ExtVTSize);
  NumElt = NumElts / Delta;
  return ExtValue;
}

/// Attempt to use the vbroadcast instruction to generate a splat value
/// from a splat BUILD_VECTOR which uses:
///  a. A single scalar load, or a constant.
///  b. Repeated pattern of constants (e.g. <0,1,0,1> or <0,1,2,3,0,1,2,3>).
///
/// The VBROADCAST node is returned when a pattern is found,
/// or SDValue() otherwise.
static SDValue lowerBuildVectorAsBroadcast(BuildVectorSDNode *BVOp,
                                           const X86Subtarget &Subtarget,
                                           SelectionDAG &DAG) {
  // VBROADCAST requires AVX.
  // TODO: Splats could be generated for non-AVX CPUs using SSE
  // instructions, but there's less potential gain for only 128-bit vectors.
  if (!Subtarget.hasAVX())
    return SDValue();

  MVT VT = BVOp->getSimpleValueType(0);
  SDLoc dl(BVOp);

  assert((VT.is128BitVector() || VT.is256BitVector() || VT.is512BitVector()) &&
         "Unsupported vector type for broadcast.");

  BitVector UndefElements;
  SDValue Ld = BVOp->getSplatValue(&UndefElements);

  // Attempt to use VBROADCASTM
  // From this paterrn:
  // a. t0 = (zext_i64 (bitcast_i8 v2i1 X))
  // b. t1 = (build_vector t0 t0)
  //
  // Create (VBROADCASTM v2i1 X)
  if (Subtarget.hasCDI() && (VT.is512BitVector() || Subtarget.hasVLX())) {
    MVT EltType = VT.getScalarType();
    unsigned NumElts = VT.getVectorNumElements();
    SDValue BOperand;
    SDValue ZeroExtended = isSplatZeroExtended(BVOp, NumElts, EltType);
    if ((ZeroExtended && ZeroExtended.getOpcode() == ISD::BITCAST) ||
        (Ld && Ld.getOpcode() == ISD::ZERO_EXTEND &&
         Ld.getOperand(0).getOpcode() == ISD::BITCAST)) {
      if (ZeroExtended)
        BOperand = ZeroExtended.getOperand(0);
      else
        BOperand = Ld.getOperand(0).getOperand(0);
      MVT MaskVT = BOperand.getSimpleValueType();
      if ((EltType == MVT::i64 && MaskVT == MVT::v8i1) || // for broadcastmb2q
          (EltType == MVT::i32 && MaskVT == MVT::v16i1)) { // for broadcastmw2d
        SDValue Brdcst =
            DAG.getNode(X86ISD::VBROADCASTM, dl,
                        MVT::getVectorVT(EltType, NumElts), BOperand);
        return DAG.getBitcast(VT, Brdcst);
      }
    }
  }

  unsigned NumElts = VT.getVectorNumElements();
  unsigned NumUndefElts = UndefElements.count();
  if (!Ld || (NumElts - NumUndefElts) <= 1) {
    APInt SplatValue, Undef;
    unsigned SplatBitSize;
    bool HasUndef;
    // Check if this is a repeated constant pattern suitable for broadcasting.
    if (BVOp->isConstantSplat(SplatValue, Undef, SplatBitSize, HasUndef) &&
        SplatBitSize > VT.getScalarSizeInBits() &&
        SplatBitSize < VT.getSizeInBits()) {
      // Avoid replacing with broadcast when it's a use of a shuffle
      // instruction to preserve the present custom lowering of shuffles.
      if (isFoldableUseOfShuffle(BVOp))
        return SDValue();
      // replace BUILD_VECTOR with broadcast of the repeated constants.
      const TargetLowering &TLI = DAG.getTargetLoweringInfo();
      LLVMContext *Ctx = DAG.getContext();
      MVT PVT = TLI.getPointerTy(DAG.getDataLayout());
      if (Subtarget.hasAVX()) {
        if (SplatBitSize <= 64 && Subtarget.hasAVX2() &&
            !(SplatBitSize == 64 && Subtarget.is32Bit())) {
          // Splatted value can fit in one INTEGER constant in constant pool.
          // Load the constant and broadcast it.
          MVT CVT = MVT::getIntegerVT(SplatBitSize);
          Type *ScalarTy = Type::getIntNTy(*Ctx, SplatBitSize);
          Constant *C = Constant::getIntegerValue(ScalarTy, SplatValue);
          SDValue CP = DAG.getConstantPool(C, PVT);
          unsigned Repeat = VT.getSizeInBits() / SplatBitSize;

          unsigned Alignment = cast<ConstantPoolSDNode>(CP)->getAlignment();
          Ld = DAG.getLoad(
              CVT, dl, DAG.getEntryNode(), CP,
              MachinePointerInfo::getConstantPool(DAG.getMachineFunction()),
              Alignment);
          SDValue Brdcst = DAG.getNode(X86ISD::VBROADCAST, dl,
                                       MVT::getVectorVT(CVT, Repeat), Ld);
          return DAG.getBitcast(VT, Brdcst);
        } else if (SplatBitSize == 32 || SplatBitSize == 64) {
          // Splatted value can fit in one FLOAT constant in constant pool.
          // Load the constant and broadcast it.
          // AVX have support for 32 and 64 bit broadcast for floats only.
          // No 64bit integer in 32bit subtarget.
          MVT CVT = MVT::getFloatingPointVT(SplatBitSize);
          // Lower the splat via APFloat directly, to avoid any conversion.
          Constant *C =
              SplatBitSize == 32
                  ? ConstantFP::get(*Ctx,
                                    APFloat(APFloat::IEEEsingle(), SplatValue))
                  : ConstantFP::get(*Ctx,
                                    APFloat(APFloat::IEEEdouble(), SplatValue));
          SDValue CP = DAG.getConstantPool(C, PVT);
          unsigned Repeat = VT.getSizeInBits() / SplatBitSize;

          unsigned Alignment = cast<ConstantPoolSDNode>(CP)->getAlignment();
          Ld = DAG.getLoad(
              CVT, dl, DAG.getEntryNode(), CP,
              MachinePointerInfo::getConstantPool(DAG.getMachineFunction()),
              Alignment);
          SDValue Brdcst = DAG.getNode(X86ISD::VBROADCAST, dl,
                                       MVT::getVectorVT(CVT, Repeat), Ld);
          return DAG.getBitcast(VT, Brdcst);
        } else if (SplatBitSize > 64) {
          // Load the vector of constants and broadcast it.
          MVT CVT = VT.getScalarType();
          Constant *VecC = getConstantVector(VT, SplatValue, SplatBitSize,
                                             *Ctx);
          SDValue VCP = DAG.getConstantPool(VecC, PVT);
          unsigned NumElm = SplatBitSize / VT.getScalarSizeInBits();
          unsigned Alignment = cast<ConstantPoolSDNode>(VCP)->getAlignment();
          Ld = DAG.getLoad(
              MVT::getVectorVT(CVT, NumElm), dl, DAG.getEntryNode(), VCP,
              MachinePointerInfo::getConstantPool(DAG.getMachineFunction()),
              Alignment);
          SDValue Brdcst = DAG.getNode(X86ISD::SUBV_BROADCAST, dl, VT, Ld);
          return DAG.getBitcast(VT, Brdcst);
        }
      }
    }

    // If we are moving a scalar into a vector (Ld must be set and all elements
    // but 1 are undef) and that operation is not obviously supported by
    // vmovd/vmovq/vmovss/vmovsd, then keep trying to form a broadcast.
    // That's better than general shuffling and may eliminate a load to GPR and
    // move from scalar to vector register.
    if (!Ld || NumElts - NumUndefElts != 1)
      return SDValue();
    unsigned ScalarSize = Ld.getValueSizeInBits();
    if (!(UndefElements[0] || (ScalarSize != 32 && ScalarSize != 64)))
      return SDValue();
  }

  bool ConstSplatVal =
      (Ld.getOpcode() == ISD::Constant || Ld.getOpcode() == ISD::ConstantFP);

  // Make sure that all of the users of a non-constant load are from the
  // BUILD_VECTOR node.
  if (!ConstSplatVal && !BVOp->isOnlyUserOf(Ld.getNode()))
    return SDValue();

  unsigned ScalarSize = Ld.getValueSizeInBits();
  bool IsGE256 = (VT.getSizeInBits() >= 256);

  // When optimizing for size, generate up to 5 extra bytes for a broadcast
  // instruction to save 8 or more bytes of constant pool data.
  // TODO: If multiple splats are generated to load the same constant,
  // it may be detrimental to overall size. There needs to be a way to detect
  // that condition to know if this is truly a size win.
  bool OptForSize = DAG.getMachineFunction().getFunction().hasOptSize();

  // Handle broadcasting a single constant scalar from the constant pool
  // into a vector.
  // On Sandybridge (no AVX2), it is still better to load a constant vector
  // from the constant pool and not to broadcast it from a scalar.
  // But override that restriction when optimizing for size.
  // TODO: Check if splatting is recommended for other AVX-capable CPUs.
  if (ConstSplatVal && (Subtarget.hasAVX2() || OptForSize)) {
    EVT CVT = Ld.getValueType();
    assert(!CVT.isVector() && "Must not broadcast a vector type");

    // Splat f32, i32, v4f64, v4i64 in all cases with AVX2.
    // For size optimization, also splat v2f64 and v2i64, and for size opt
    // with AVX2, also splat i8 and i16.
    // With pattern matching, the VBROADCAST node may become a VMOVDDUP.
    if (ScalarSize == 32 || (IsGE256 && ScalarSize == 64) ||
        (OptForSize && (ScalarSize == 64 || Subtarget.hasAVX2()))) {
      const Constant *C = nullptr;
      if (ConstantSDNode *CI = dyn_cast<ConstantSDNode>(Ld))
        C = CI->getConstantIntValue();
      else if (ConstantFPSDNode *CF = dyn_cast<ConstantFPSDNode>(Ld))
        C = CF->getConstantFPValue();

      assert(C && "Invalid constant type");

      const TargetLowering &TLI = DAG.getTargetLoweringInfo();
      SDValue CP =
          DAG.getConstantPool(C, TLI.getPointerTy(DAG.getDataLayout()));
      unsigned Alignment = cast<ConstantPoolSDNode>(CP)->getAlignment();
      Ld = DAG.getLoad(
          CVT, dl, DAG.getEntryNode(), CP,
          MachinePointerInfo::getConstantPool(DAG.getMachineFunction()),
          Alignment);

      return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Ld);
    }
  }

  bool IsLoad = ISD::isNormalLoad(Ld.getNode());

  // Handle AVX2 in-register broadcasts.
  if (!IsLoad && Subtarget.hasInt256() &&
      (ScalarSize == 32 || (IsGE256 && ScalarSize == 64)))
    return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Ld);

  // The scalar source must be a normal load.
  if (!IsLoad)
    return SDValue();

  if (ScalarSize == 32 || (IsGE256 && ScalarSize == 64) ||
      (Subtarget.hasVLX() && ScalarSize == 64))
    return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Ld);

  // The integer check is needed for the 64-bit into 128-bit so it doesn't match
  // double since there is no vbroadcastsd xmm
  if (Subtarget.hasInt256() && Ld.getValueType().isInteger()) {
    if (ScalarSize == 8 || ScalarSize == 16 || ScalarSize == 64)
      return DAG.getNode(X86ISD::VBROADCAST, dl, VT, Ld);
  }

  // Unsupported broadcast.
  return SDValue();
}

/// For an EXTRACT_VECTOR_ELT with a constant index return the real
/// underlying vector and index.
///
/// Modifies \p ExtractedFromVec to the real vector and returns the real
/// index.
static int getUnderlyingExtractedFromVec(SDValue &ExtractedFromVec,
                                         SDValue ExtIdx) {
  int Idx = cast<ConstantSDNode>(ExtIdx)->getZExtValue();
  if (!isa<ShuffleVectorSDNode>(ExtractedFromVec))
    return Idx;

  // For 256-bit vectors, LowerEXTRACT_VECTOR_ELT_SSE4 may have already
  // lowered this:
  //   (extract_vector_elt (v8f32 %1), Constant<6>)
  // to:
  //   (extract_vector_elt (vector_shuffle<2,u,u,u>
  //                           (extract_subvector (v8f32 %0), Constant<4>),
  //                           undef)
  //                       Constant<0>)
  // In this case the vector is the extract_subvector expression and the index
  // is 2, as specified by the shuffle.
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(ExtractedFromVec);
  SDValue ShuffleVec = SVOp->getOperand(0);
  MVT ShuffleVecVT = ShuffleVec.getSimpleValueType();
  assert(ShuffleVecVT.getVectorElementType() ==
         ExtractedFromVec.getSimpleValueType().getVectorElementType());

  int ShuffleIdx = SVOp->getMaskElt(Idx);
  if (isUndefOrInRange(ShuffleIdx, 0, ShuffleVecVT.getVectorNumElements())) {
    ExtractedFromVec = ShuffleVec;
    return ShuffleIdx;
  }
  return Idx;
}

static SDValue buildFromShuffleMostly(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getSimpleValueType();

  // Skip if insert_vec_elt is not supported.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!TLI.isOperationLegalOrCustom(ISD::INSERT_VECTOR_ELT, VT))
    return SDValue();

  SDLoc DL(Op);
  unsigned NumElems = Op.getNumOperands();

  SDValue VecIn1;
  SDValue VecIn2;
  SmallVector<unsigned, 4> InsertIndices;
  SmallVector<int, 8> Mask(NumElems, -1);

  for (unsigned i = 0; i != NumElems; ++i) {
    unsigned Opc = Op.getOperand(i).getOpcode();

    if (Opc == ISD::UNDEF)
      continue;

    if (Opc != ISD::EXTRACT_VECTOR_ELT) {
      // Quit if more than 1 elements need inserting.
      if (InsertIndices.size() > 1)
        return SDValue();

      InsertIndices.push_back(i);
      continue;
    }

    SDValue ExtractedFromVec = Op.getOperand(i).getOperand(0);
    SDValue ExtIdx = Op.getOperand(i).getOperand(1);

    // Quit if non-constant index.
    if (!isa<ConstantSDNode>(ExtIdx))
      return SDValue();
    int Idx = getUnderlyingExtractedFromVec(ExtractedFromVec, ExtIdx);

    // Quit if extracted from vector of different type.
    if (ExtractedFromVec.getValueType() != VT)
      return SDValue();

    if (!VecIn1.getNode())
      VecIn1 = ExtractedFromVec;
    else if (VecIn1 != ExtractedFromVec) {
      if (!VecIn2.getNode())
        VecIn2 = ExtractedFromVec;
      else if (VecIn2 != ExtractedFromVec)
        // Quit if more than 2 vectors to shuffle
        return SDValue();
    }

    if (ExtractedFromVec == VecIn1)
      Mask[i] = Idx;
    else if (ExtractedFromVec == VecIn2)
      Mask[i] = Idx + NumElems;
  }

  if (!VecIn1.getNode())
    return SDValue();

  VecIn2 = VecIn2.getNode() ? VecIn2 : DAG.getUNDEF(VT);
  SDValue NV = DAG.getVectorShuffle(VT, DL, VecIn1, VecIn2, Mask);

  for (unsigned Idx : InsertIndices)
    NV = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, VT, NV, Op.getOperand(Idx),
                     DAG.getIntPtrConstant(Idx, DL));

  return NV;
}

SDValue X86::ConvertI1VectorToInteger(SDValue Op, SelectionDAG &DAG) {
  assert(ISD::isBuildVectorOfConstantSDNodes(Op.getNode()) &&
         Op.getScalarValueSizeInBits() == 1 &&
         "Can not convert non-constant vector");
  uint64_t Immediate = 0;
  for (unsigned idx = 0, e = Op.getNumOperands(); idx < e; ++idx) {
    SDValue In = Op.getOperand(idx);
    if (!In.isUndef())
      Immediate |= (cast<ConstantSDNode>(In)->getZExtValue() & 0x1) << idx;
  }
  SDLoc dl(Op);
  MVT VT = MVT::getIntegerVT(std::max((int)Op.getValueSizeInBits(), 8));
  return DAG.getConstant(Immediate, dl, VT);
}

// Lower BUILD_VECTOR operation for v8i1 and v16i1 types.
static SDValue LowerBUILD_VECTORvXi1(SDValue Op, SelectionDAG &DAG,
                                     const X86Subtarget &Subtarget) {

  MVT VT = Op.getSimpleValueType();
  assert((VT.getVectorElementType() == MVT::i1) &&
         "Unexpected type in LowerBUILD_VECTORvXi1!");

  SDLoc dl(Op);
  if (ISD::isBuildVectorAllZeros(Op.getNode()))
    return Op;

  if (ISD::isBuildVectorAllOnes(Op.getNode()))
    return Op;

  if (ISD::isBuildVectorOfConstantSDNodes(Op.getNode())) {
    if (VT == MVT::v64i1 && !Subtarget.is64Bit()) {
      // Split the pieces.
      SDValue Lower =
          DAG.getBuildVector(MVT::v32i1, dl, Op.getNode()->ops().slice(0, 32));
      SDValue Upper =
          DAG.getBuildVector(MVT::v32i1, dl, Op.getNode()->ops().slice(32, 32));
      // We have to manually lower both halves so getNode doesn't try to
      // reassemble the build_vector.
      Lower = LowerBUILD_VECTORvXi1(Lower, DAG, Subtarget);
      Upper = LowerBUILD_VECTORvXi1(Upper, DAG, Subtarget);
      return DAG.getNode(ISD::CONCAT_VECTORS, dl, MVT::v64i1, Lower, Upper);
    }
    SDValue Imm = X86::ConvertI1VectorToInteger(Op, DAG);
    if (Imm.getValueSizeInBits() == VT.getSizeInBits())
      return DAG.getBitcast(VT, Imm);
    SDValue ExtVec = DAG.getBitcast(MVT::v8i1, Imm);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT, ExtVec,
                        DAG.getIntPtrConstant(0, dl));
  }

  // Vector has one or more non-const elements
  uint64_t Immediate = 0;
  SmallVector<unsigned, 16> NonConstIdx;
  bool IsSplat = true;
  bool HasConstElts = false;
  int SplatIdx = -1;
  for (unsigned idx = 0, e = Op.getNumOperands(); idx < e; ++idx) {
    SDValue In = Op.getOperand(idx);
    if (In.isUndef())
      continue;
    if (!isa<ConstantSDNode>(In))
      NonConstIdx.push_back(idx);
    else {
      Immediate |= (cast<ConstantSDNode>(In)->getZExtValue() & 0x1) << idx;
      HasConstElts = true;
    }
    if (SplatIdx < 0)
      SplatIdx = idx;
    else if (In != Op.getOperand(SplatIdx))
      IsSplat = false;
  }

  // for splat use " (select i1 splat_elt, all-ones, all-zeroes)"
  if (IsSplat)
    return DAG.getSelect(dl, VT, Op.getOperand(SplatIdx),
                         DAG.getConstant(1, dl, VT),
                         DAG.getConstant(0, dl, VT));

  // insert elements one by one
  SDValue DstVec;
  SDValue Imm;
  if (Immediate) {
    MVT ImmVT = MVT::getIntegerVT(std::max((int)VT.getSizeInBits(), 8));
    Imm = DAG.getConstant(Immediate, dl, ImmVT);
  }
  else if (HasConstElts)
    Imm = DAG.getConstant(0, dl, VT);
  else
    Imm = DAG.getUNDEF(VT);
  if (Imm.getValueSizeInBits() == VT.getSizeInBits())
    DstVec = DAG.getBitcast(VT, Imm);
  else {
    SDValue ExtVec = DAG.getBitcast(MVT::v8i1, Imm);
    DstVec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT, ExtVec,
                         DAG.getIntPtrConstant(0, dl));
  }

  for (unsigned i = 0, e = NonConstIdx.size(); i != e; ++i) {
    unsigned InsertIdx = NonConstIdx[i];
    DstVec = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, DstVec,
                         Op.getOperand(InsertIdx),
                         DAG.getIntPtrConstant(InsertIdx, dl));
  }
  return DstVec;
}

/// This is a helper function of LowerToHorizontalOp().
/// This function checks that the build_vector \p N in input implements a
/// 128-bit partial horizontal operation on a 256-bit vector, but that operation
/// may not match the layout of an x86 256-bit horizontal instruction.
/// In other words, if this returns true, then some extraction/insertion will
/// be required to produce a valid horizontal instruction.
///
/// Parameter \p Opcode defines the kind of horizontal operation to match.
/// For example, if \p Opcode is equal to ISD::ADD, then this function
/// checks if \p N implements a horizontal arithmetic add; if instead \p Opcode
/// is equal to ISD::SUB, then this function checks if this is a horizontal
/// arithmetic sub.
///
/// This function only analyzes elements of \p N whose indices are
/// in range [BaseIdx, LastIdx).
///
/// TODO: This function was originally used to match both real and fake partial
/// horizontal operations, but the index-matching logic is incorrect for that.
/// See the corrected implementation in isHopBuildVector(). Can we reduce this
/// code because it is only used for partial h-op matching now?
static bool isHorizontalBinOpPart(const BuildVectorSDNode *N, unsigned Opcode,
                                  SelectionDAG &DAG,
                                  unsigned BaseIdx, unsigned LastIdx,
                                  SDValue &V0, SDValue &V1) {
  EVT VT = N->getValueType(0);
  assert(VT.is256BitVector() && "Only use for matching partial 256-bit h-ops");
  assert(BaseIdx * 2 <= LastIdx && "Invalid Indices in input!");
  assert(VT.isVector() && VT.getVectorNumElements() >= LastIdx &&
         "Invalid Vector in input!");

  bool IsCommutable = (Opcode == ISD::ADD || Opcode == ISD::FADD);
  bool CanFold = true;
  unsigned ExpectedVExtractIdx = BaseIdx;
  unsigned NumElts = LastIdx - BaseIdx;
  V0 = DAG.getUNDEF(VT);
  V1 = DAG.getUNDEF(VT);

  // Check if N implements a horizontal binop.
  for (unsigned i = 0, e = NumElts; i != e && CanFold; ++i) {
    SDValue Op = N->getOperand(i + BaseIdx);

    // Skip UNDEFs.
    if (Op->isUndef()) {
      // Update the expected vector extract index.
      if (i * 2 == NumElts)
        ExpectedVExtractIdx = BaseIdx;
      ExpectedVExtractIdx += 2;
      continue;
    }

    CanFold = Op->getOpcode() == Opcode && Op->hasOneUse();

    if (!CanFold)
      break;

    SDValue Op0 = Op.getOperand(0);
    SDValue Op1 = Op.getOperand(1);

    // Try to match the following pattern:
    // (BINOP (extract_vector_elt A, I), (extract_vector_elt A, I+1))
    CanFold = (Op0.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
        Op1.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
        Op0.getOperand(0) == Op1.getOperand(0) &&
        isa<ConstantSDNode>(Op0.getOperand(1)) &&
        isa<ConstantSDNode>(Op1.getOperand(1)));
    if (!CanFold)
      break;

    unsigned I0 = cast<ConstantSDNode>(Op0.getOperand(1))->getZExtValue();
    unsigned I1 = cast<ConstantSDNode>(Op1.getOperand(1))->getZExtValue();

    if (i * 2 < NumElts) {
      if (V0.isUndef()) {
        V0 = Op0.getOperand(0);
        if (V0.getValueType() != VT)
          return false;
      }
    } else {
      if (V1.isUndef()) {
        V1 = Op0.getOperand(0);
        if (V1.getValueType() != VT)
          return false;
      }
      if (i * 2 == NumElts)
        ExpectedVExtractIdx = BaseIdx;
    }

    SDValue Expected = (i * 2 < NumElts) ? V0 : V1;
    if (I0 == ExpectedVExtractIdx)
      CanFold = I1 == I0 + 1 && Op0.getOperand(0) == Expected;
    else if (IsCommutable && I1 == ExpectedVExtractIdx) {
      // Try to match the following dag sequence:
      // (BINOP (extract_vector_elt A, I+1), (extract_vector_elt A, I))
      CanFold = I0 == I1 + 1 && Op1.getOperand(0) == Expected;
    } else
      CanFold = false;

    ExpectedVExtractIdx += 2;
  }

  return CanFold;
}

/// Emit a sequence of two 128-bit horizontal add/sub followed by
/// a concat_vector.
///
/// This is a helper function of LowerToHorizontalOp().
/// This function expects two 256-bit vectors called V0 and V1.
/// At first, each vector is split into two separate 128-bit vectors.
/// Then, the resulting 128-bit vectors are used to implement two
/// horizontal binary operations.
///
/// The kind of horizontal binary operation is defined by \p X86Opcode.
///
/// \p Mode specifies how the 128-bit parts of V0 and V1 are passed in input to
/// the two new horizontal binop.
/// When Mode is set, the first horizontal binop dag node would take as input
/// the lower 128-bit of V0 and the upper 128-bit of V0. The second
/// horizontal binop dag node would take as input the lower 128-bit of V1
/// and the upper 128-bit of V1.
///   Example:
///     HADD V0_LO, V0_HI
///     HADD V1_LO, V1_HI
///
/// Otherwise, the first horizontal binop dag node takes as input the lower
/// 128-bit of V0 and the lower 128-bit of V1, and the second horizontal binop
/// dag node takes the upper 128-bit of V0 and the upper 128-bit of V1.
///   Example:
///     HADD V0_LO, V1_LO
///     HADD V0_HI, V1_HI
///
/// If \p isUndefLO is set, then the algorithm propagates UNDEF to the lower
/// 128-bits of the result. If \p isUndefHI is set, then UNDEF is propagated to
/// the upper 128-bits of the result.
static SDValue ExpandHorizontalBinOp(const SDValue &V0, const SDValue &V1,
                                     const SDLoc &DL, SelectionDAG &DAG,
                                     unsigned X86Opcode, bool Mode,
                                     bool isUndefLO, bool isUndefHI) {
  MVT VT = V0.getSimpleValueType();
  assert(VT.is256BitVector() && VT == V1.getSimpleValueType() &&
         "Invalid nodes in input!");

  unsigned NumElts = VT.getVectorNumElements();
  SDValue V0_LO = extract128BitVector(V0, 0, DAG, DL);
  SDValue V0_HI = extract128BitVector(V0, NumElts/2, DAG, DL);
  SDValue V1_LO = extract128BitVector(V1, 0, DAG, DL);
  SDValue V1_HI = extract128BitVector(V1, NumElts/2, DAG, DL);
  MVT NewVT = V0_LO.getSimpleValueType();

  SDValue LO = DAG.getUNDEF(NewVT);
  SDValue HI = DAG.getUNDEF(NewVT);

  if (Mode) {
    // Don't emit a horizontal binop if the result is expected to be UNDEF.
    if (!isUndefLO && !V0->isUndef())
      LO = DAG.getNode(X86Opcode, DL, NewVT, V0_LO, V0_HI);
    if (!isUndefHI && !V1->isUndef())
      HI = DAG.getNode(X86Opcode, DL, NewVT, V1_LO, V1_HI);
  } else {
    // Don't emit a horizontal binop if the result is expected to be UNDEF.
    if (!isUndefLO && (!V0_LO->isUndef() || !V1_LO->isUndef()))
      LO = DAG.getNode(X86Opcode, DL, NewVT, V0_LO, V1_LO);

    if (!isUndefHI && (!V0_HI->isUndef() || !V1_HI->isUndef()))
      HI = DAG.getNode(X86Opcode, DL, NewVT, V0_HI, V1_HI);
  }

  return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, LO, HI);
}

/// Returns true iff \p BV builds a vector with the result equivalent to
/// the result of ADDSUB/SUBADD operation.
/// If true is returned then the operands of ADDSUB = Opnd0 +- Opnd1
/// (SUBADD = Opnd0 -+ Opnd1) operation are written to the parameters
/// \p Opnd0 and \p Opnd1.
static bool isAddSubOrSubAdd(const BuildVectorSDNode *BV,
                             const X86Subtarget &Subtarget, SelectionDAG &DAG,
                             SDValue &Opnd0, SDValue &Opnd1,
                             unsigned &NumExtracts,
                             bool &IsSubAdd) {

  MVT VT = BV->getSimpleValueType(0);
  if (!Subtarget.hasSSE3() || !VT.isFloatingPoint())
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  SDValue InVec0 = DAG.getUNDEF(VT);
  SDValue InVec1 = DAG.getUNDEF(VT);

  NumExtracts = 0;

  // Odd-numbered elements in the input build vector are obtained from
  // adding/subtracting two integer/float elements.
  // Even-numbered elements in the input build vector are obtained from
  // subtracting/adding two integer/float elements.
  unsigned Opc[2] = {0, 0};
  for (unsigned i = 0, e = NumElts; i != e; ++i) {
    SDValue Op = BV->getOperand(i);

    // Skip 'undef' values.
    unsigned Opcode = Op.getOpcode();
    if (Opcode == ISD::UNDEF)
      continue;

    // Early exit if we found an unexpected opcode.
    if (Opcode != ISD::FADD && Opcode != ISD::FSUB)
      return false;

    SDValue Op0 = Op.getOperand(0);
    SDValue Op1 = Op.getOperand(1);

    // Try to match the following pattern:
    // (BINOP (extract_vector_elt A, i), (extract_vector_elt B, i))
    // Early exit if we cannot match that sequence.
    if (Op0.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        Op1.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        !isa<ConstantSDNode>(Op0.getOperand(1)) ||
        !isa<ConstantSDNode>(Op1.getOperand(1)) ||
        Op0.getOperand(1) != Op1.getOperand(1))
      return false;

    unsigned I0 = cast<ConstantSDNode>(Op0.getOperand(1))->getZExtValue();
    if (I0 != i)
      return false;

    // We found a valid add/sub node, make sure its the same opcode as previous
    // elements for this parity.
    if (Opc[i % 2] != 0 && Opc[i % 2] != Opcode)
      return false;
    Opc[i % 2] = Opcode;

    // Update InVec0 and InVec1.
    if (InVec0.isUndef()) {
      InVec0 = Op0.getOperand(0);
      if (InVec0.getSimpleValueType() != VT)
        return false;
    }
    if (InVec1.isUndef()) {
      InVec1 = Op1.getOperand(0);
      if (InVec1.getSimpleValueType() != VT)
        return false;
    }

    // Make sure that operands in input to each add/sub node always
    // come from a same pair of vectors.
    if (InVec0 != Op0.getOperand(0)) {
      if (Opcode == ISD::FSUB)
        return false;

      // FADD is commutable. Try to commute the operands
      // and then test again.
      std::swap(Op0, Op1);
      if (InVec0 != Op0.getOperand(0))
        return false;
    }

    if (InVec1 != Op1.getOperand(0))
      return false;

    // Increment the number of extractions done.
    ++NumExtracts;
  }

  // Ensure we have found an opcode for both parities and that they are
  // different. Don't try to fold this build_vector into an ADDSUB/SUBADD if the
  // inputs are undef.
  if (!Opc[0] || !Opc[1] || Opc[0] == Opc[1] ||
      InVec0.isUndef() || InVec1.isUndef())
    return false;

  IsSubAdd = Opc[0] == ISD::FADD;

  Opnd0 = InVec0;
  Opnd1 = InVec1;
  return true;
}

/// Returns true if is possible to fold MUL and an idiom that has already been
/// recognized as ADDSUB/SUBADD(\p Opnd0, \p Opnd1) into
/// FMADDSUB/FMSUBADD(x, y, \p Opnd1). If (and only if) true is returned, the
/// operands of FMADDSUB/FMSUBADD are written to parameters \p Opnd0, \p Opnd1, \p Opnd2.
///
/// Prior to calling this function it should be known that there is some
/// SDNode that potentially can be replaced with an X86ISD::ADDSUB operation
/// using \p Opnd0 and \p Opnd1 as operands. Also, this method is called
/// before replacement of such SDNode with ADDSUB operation. Thus the number
/// of \p Opnd0 uses is expected to be equal to 2.
/// For example, this function may be called for the following IR:
///    %AB = fmul fast <2 x double> %A, %B
///    %Sub = fsub fast <2 x double> %AB, %C
///    %Add = fadd fast <2 x double> %AB, %C
///    %Addsub = shufflevector <2 x double> %Sub, <2 x double> %Add,
///                            <2 x i32> <i32 0, i32 3>
/// There is a def for %Addsub here, which potentially can be replaced by
/// X86ISD::ADDSUB operation:
///    %Addsub = X86ISD::ADDSUB %AB, %C
/// and such ADDSUB can further be replaced with FMADDSUB:
///    %Addsub = FMADDSUB %A, %B, %C.
///
/// The main reason why this method is called before the replacement of the
/// recognized ADDSUB idiom with ADDSUB operation is that such replacement
/// is illegal sometimes. E.g. 512-bit ADDSUB is not available, while 512-bit
/// FMADDSUB is.
static bool isFMAddSubOrFMSubAdd(const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG,
                                 SDValue &Opnd0, SDValue &Opnd1, SDValue &Opnd2,
                                 unsigned ExpectedUses) {
  if (Opnd0.getOpcode() != ISD::FMUL ||
      !Opnd0->hasNUsesOfValue(ExpectedUses, 0) || !Subtarget.hasAnyFMA())
    return false;

  // FIXME: These checks must match the similar ones in
  // DAGCombiner::visitFADDForFMACombine. It would be good to have one
  // function that would answer if it is Ok to fuse MUL + ADD to FMADD
  // or MUL + ADDSUB to FMADDSUB.
  const TargetOptions &Options = DAG.getTarget().Options;
  bool AllowFusion =
      (Options.AllowFPOpFusion == FPOpFusion::Fast || Options.UnsafeFPMath);
  if (!AllowFusion)
    return false;

  Opnd2 = Opnd1;
  Opnd1 = Opnd0.getOperand(1);
  Opnd0 = Opnd0.getOperand(0);

  return true;
}

/// Try to fold a build_vector that performs an 'addsub' or 'fmaddsub' or
/// 'fsubadd' operation accordingly to X86ISD::ADDSUB or X86ISD::FMADDSUB or
/// X86ISD::FMSUBADD node.
static SDValue lowerToAddSubOrFMAddSub(const BuildVectorSDNode *BV,
                                       const X86Subtarget &Subtarget,
                                       SelectionDAG &DAG) {
  SDValue Opnd0, Opnd1;
  unsigned NumExtracts;
  bool IsSubAdd;
  if (!isAddSubOrSubAdd(BV, Subtarget, DAG, Opnd0, Opnd1, NumExtracts,
                        IsSubAdd))
    return SDValue();

  MVT VT = BV->getSimpleValueType(0);
  SDLoc DL(BV);

  // Try to generate X86ISD::FMADDSUB node here.
  SDValue Opnd2;
  if (isFMAddSubOrFMSubAdd(Subtarget, DAG, Opnd0, Opnd1, Opnd2, NumExtracts)) {
    unsigned Opc = IsSubAdd ? X86ISD::FMSUBADD : X86ISD::FMADDSUB;
    return DAG.getNode(Opc, DL, VT, Opnd0, Opnd1, Opnd2);
  }

  // We only support ADDSUB.
  if (IsSubAdd)
    return SDValue();

  // Do not generate X86ISD::ADDSUB node for 512-bit types even though
  // the ADDSUB idiom has been successfully recognized. There are no known
  // X86 targets with 512-bit ADDSUB instructions!
  // 512-bit ADDSUB idiom recognition was needed only as part of FMADDSUB idiom
  // recognition.
  if (VT.is512BitVector())
    return SDValue();

  return DAG.getNode(X86ISD::ADDSUB, DL, VT, Opnd0, Opnd1);
}

static bool isHopBuildVector(const BuildVectorSDNode *BV, SelectionDAG &DAG,
                             unsigned &HOpcode, SDValue &V0, SDValue &V1) {
  // Initialize outputs to known values.
  MVT VT = BV->getSimpleValueType(0);
  HOpcode = ISD::DELETED_NODE;
  V0 = DAG.getUNDEF(VT);
  V1 = DAG.getUNDEF(VT);

  // x86 256-bit horizontal ops are defined in a non-obvious way. Each 128-bit
  // half of the result is calculated independently from the 128-bit halves of
  // the inputs, so that makes the index-checking logic below more complicated.
  unsigned NumElts = VT.getVectorNumElements();
  unsigned GenericOpcode = ISD::DELETED_NODE;
  unsigned Num128BitChunks = VT.is256BitVector() ? 2 : 1;
  unsigned NumEltsIn128Bits = NumElts / Num128BitChunks;
  unsigned NumEltsIn64Bits = NumEltsIn128Bits / 2;
  for (unsigned i = 0; i != Num128BitChunks; ++i) {
    for (unsigned j = 0; j != NumEltsIn128Bits; ++j) {
      // Ignore undef elements.
      SDValue Op = BV->getOperand(i * NumEltsIn128Bits + j);
      if (Op.isUndef())
        continue;

      // If there's an opcode mismatch, we're done.
      if (HOpcode != ISD::DELETED_NODE && Op.getOpcode() != GenericOpcode)
        return false;

      // Initialize horizontal opcode.
      if (HOpcode == ISD::DELETED_NODE) {
        GenericOpcode = Op.getOpcode();
        switch (GenericOpcode) {
        case ISD::ADD: HOpcode = X86ISD::HADD; break;
        case ISD::SUB: HOpcode = X86ISD::HSUB; break;
        case ISD::FADD: HOpcode = X86ISD::FHADD; break;
        case ISD::FSUB: HOpcode = X86ISD::FHSUB; break;
        default: return false;
        }
      }

      SDValue Op0 = Op.getOperand(0);
      SDValue Op1 = Op.getOperand(1);
      if (Op0.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
          Op1.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
          Op0.getOperand(0) != Op1.getOperand(0) ||
          !isa<ConstantSDNode>(Op0.getOperand(1)) ||
          !isa<ConstantSDNode>(Op1.getOperand(1)) || !Op.hasOneUse())
        return false;

      // The source vector is chosen based on which 64-bit half of the
      // destination vector is being calculated.
      if (j < NumEltsIn64Bits) {
        if (V0.isUndef())
          V0 = Op0.getOperand(0);
      } else {
        if (V1.isUndef())
          V1 = Op0.getOperand(0);
      }

      SDValue SourceVec = (j < NumEltsIn64Bits) ? V0 : V1;
      if (SourceVec != Op0.getOperand(0))
        return false;

      // op (extract_vector_elt A, I), (extract_vector_elt A, I+1)
      unsigned ExtIndex0 = Op0.getConstantOperandVal(1);
      unsigned ExtIndex1 = Op1.getConstantOperandVal(1);
      unsigned ExpectedIndex = i * NumEltsIn128Bits +
                               (j % NumEltsIn64Bits) * 2;
      if (ExpectedIndex == ExtIndex0 && ExtIndex1 == ExtIndex0 + 1)
        continue;

      // If this is not a commutative op, this does not match.
      if (GenericOpcode != ISD::ADD && GenericOpcode != ISD::FADD)
        return false;

      // Addition is commutative, so try swapping the extract indexes.
      // op (extract_vector_elt A, I+1), (extract_vector_elt A, I)
      if (ExpectedIndex == ExtIndex1 && ExtIndex0 == ExtIndex1 + 1)
        continue;

      // Extract indexes do not match horizontal requirement.
      return false;
    }
  }
  // We matched. Opcode and operands are returned by reference as arguments.
  return true;
}

static SDValue getHopForBuildVector(const BuildVectorSDNode *BV,
                                    SelectionDAG &DAG, unsigned HOpcode,
                                    SDValue V0, SDValue V1) {
  // If either input vector is not the same size as the build vector,
  // extract/insert the low bits to the correct size.
  // This is free (examples: zmm --> xmm, xmm --> ymm).
  MVT VT = BV->getSimpleValueType(0);
  unsigned Width = VT.getSizeInBits();
  if (V0.getValueSizeInBits() > Width)
    V0 = extractSubVector(V0, 0, DAG, SDLoc(BV), Width);
  else if (V0.getValueSizeInBits() < Width)
    V0 = insertSubVector(DAG.getUNDEF(VT), V0, 0, DAG, SDLoc(BV), Width);

  if (V1.getValueSizeInBits() > Width)
    V1 = extractSubVector(V1, 0, DAG, SDLoc(BV), Width);
  else if (V1.getValueSizeInBits() < Width)
    V1 = insertSubVector(DAG.getUNDEF(VT), V1, 0, DAG, SDLoc(BV), Width);

  unsigned NumElts = VT.getVectorNumElements();
  APInt DemandedElts = APInt::getAllOnesValue(NumElts);
  for (unsigned i = 0; i != NumElts; ++i)
    if (BV->getOperand(i).isUndef())
      DemandedElts.clearBit(i);

  // If we don't need the upper xmm, then perform as a xmm hop.
  unsigned HalfNumElts = NumElts / 2;
  if (VT.is256BitVector() && DemandedElts.lshr(HalfNumElts) == 0) {
    MVT HalfVT = VT.getHalfNumVectorElementsVT();
    V0 = extractSubVector(V0, 0, DAG, SDLoc(BV), 128);
    V1 = extractSubVector(V1, 0, DAG, SDLoc(BV), 128);
    SDValue Half = DAG.getNode(HOpcode, SDLoc(BV), HalfVT, V0, V1);
    return insertSubVector(DAG.getUNDEF(VT), Half, 0, DAG, SDLoc(BV), 256);
  }

  return DAG.getNode(HOpcode, SDLoc(BV), VT, V0, V1);
}

/// Lower BUILD_VECTOR to a horizontal add/sub operation if possible.
static SDValue LowerToHorizontalOp(const BuildVectorSDNode *BV,
                                   const X86Subtarget &Subtarget,
                                   SelectionDAG &DAG) {
  // We need at least 2 non-undef elements to make this worthwhile by default.
  unsigned NumNonUndefs =
      count_if(BV->op_values(), [](SDValue V) { return !V.isUndef(); });
  if (NumNonUndefs < 2)
    return SDValue();

  // There are 4 sets of horizontal math operations distinguished by type:
  // int/FP at 128-bit/256-bit. Each type was introduced with a different
  // subtarget feature. Try to match those "native" patterns first.
  MVT VT = BV->getSimpleValueType(0);
  if (((VT == MVT::v4f32 || VT == MVT::v2f64) && Subtarget.hasSSE3()) ||
      ((VT == MVT::v8i16 || VT == MVT::v4i32) && Subtarget.hasSSSE3()) ||
      ((VT == MVT::v8f32 || VT == MVT::v4f64) && Subtarget.hasAVX()) ||
      ((VT == MVT::v16i16 || VT == MVT::v8i32) && Subtarget.hasAVX2())) {
    unsigned HOpcode;
    SDValue V0, V1;
    if (isHopBuildVector(BV, DAG, HOpcode, V0, V1))
      return getHopForBuildVector(BV, DAG, HOpcode, V0, V1);
  }

  // Try harder to match 256-bit ops by using extract/concat.
  if (!Subtarget.hasAVX() || !VT.is256BitVector())
    return SDValue();

  // Count the number of UNDEF operands in the build_vector in input.
  unsigned NumElts = VT.getVectorNumElements();
  unsigned Half = NumElts / 2;
  unsigned NumUndefsLO = 0;
  unsigned NumUndefsHI = 0;
  for (unsigned i = 0, e = Half; i != e; ++i)
    if (BV->getOperand(i)->isUndef())
      NumUndefsLO++;

  for (unsigned i = Half, e = NumElts; i != e; ++i)
    if (BV->getOperand(i)->isUndef())
      NumUndefsHI++;

  SDLoc DL(BV);
  SDValue InVec0, InVec1;
  if (VT == MVT::v8i32 || VT == MVT::v16i16) {
    SDValue InVec2, InVec3;
    unsigned X86Opcode;
    bool CanFold = true;

    if (isHorizontalBinOpPart(BV, ISD::ADD, DAG, 0, Half, InVec0, InVec1) &&
        isHorizontalBinOpPart(BV, ISD::ADD, DAG, Half, NumElts, InVec2,
                              InVec3) &&
        ((InVec0.isUndef() || InVec2.isUndef()) || InVec0 == InVec2) &&
        ((InVec1.isUndef() || InVec3.isUndef()) || InVec1 == InVec3))
      X86Opcode = X86ISD::HADD;
    else if (isHorizontalBinOpPart(BV, ISD::SUB, DAG, 0, Half, InVec0,
                                   InVec1) &&
             isHorizontalBinOpPart(BV, ISD::SUB, DAG, Half, NumElts, InVec2,
                                   InVec3) &&
             ((InVec0.isUndef() || InVec2.isUndef()) || InVec0 == InVec2) &&
             ((InVec1.isUndef() || InVec3.isUndef()) || InVec1 == InVec3))
      X86Opcode = X86ISD::HSUB;
    else
      CanFold = false;

    if (CanFold) {
      // Do not try to expand this build_vector into a pair of horizontal
      // add/sub if we can emit a pair of scalar add/sub.
      if (NumUndefsLO + 1 == Half || NumUndefsHI + 1 == Half)
        return SDValue();

      // Convert this build_vector into a pair of horizontal binops followed by
      // a concat vector. We must adjust the outputs from the partial horizontal
      // matching calls above to account for undefined vector halves.
      SDValue V0 = InVec0.isUndef() ? InVec2 : InVec0;
      SDValue V1 = InVec1.isUndef() ? InVec3 : InVec1;
      assert((!V0.isUndef() || !V1.isUndef()) && "Horizontal-op of undefs?");
      bool isUndefLO = NumUndefsLO == Half;
      bool isUndefHI = NumUndefsHI == Half;
      return ExpandHorizontalBinOp(V0, V1, DL, DAG, X86Opcode, false, isUndefLO,
                                   isUndefHI);
    }
  }

  if (VT == MVT::v8f32 || VT == MVT::v4f64 || VT == MVT::v8i32 ||
      VT == MVT::v16i16) {
    unsigned X86Opcode;
    if (isHorizontalBinOpPart(BV, ISD::ADD, DAG, 0, NumElts, InVec0, InVec1))
      X86Opcode = X86ISD::HADD;
    else if (isHorizontalBinOpPart(BV, ISD::SUB, DAG, 0, NumElts, InVec0,
                                   InVec1))
      X86Opcode = X86ISD::HSUB;
    else if (isHorizontalBinOpPart(BV, ISD::FADD, DAG, 0, NumElts, InVec0,
                                   InVec1))
      X86Opcode = X86ISD::FHADD;
    else if (isHorizontalBinOpPart(BV, ISD::FSUB, DAG, 0, NumElts, InVec0,
                                   InVec1))
      X86Opcode = X86ISD::FHSUB;
    else
      return SDValue();

    // Don't try to expand this build_vector into a pair of horizontal add/sub
    // if we can simply emit a pair of scalar add/sub.
    if (NumUndefsLO + 1 == Half || NumUndefsHI + 1 == Half)
      return SDValue();

    // Convert this build_vector into two horizontal add/sub followed by
    // a concat vector.
    bool isUndefLO = NumUndefsLO == Half;
    bool isUndefHI = NumUndefsHI == Half;
    return ExpandHorizontalBinOp(InVec0, InVec1, DL, DAG, X86Opcode, true,
                                 isUndefLO, isUndefHI);
  }

  return SDValue();
}

/// If a BUILD_VECTOR's source elements all apply the same bit operation and
/// one of their operands is constant, lower to a pair of BUILD_VECTOR and
/// just apply the bit to the vectors.
/// NOTE: Its not in our interest to start make a general purpose vectorizer
/// from this, but enough scalar bit operations are created from the later
/// legalization + scalarization stages to need basic support.
static SDValue lowerBuildVectorToBitOp(BuildVectorSDNode *Op,
                                       SelectionDAG &DAG) {
  SDLoc DL(Op);
  MVT VT = Op->getSimpleValueType(0);
  unsigned NumElems = VT.getVectorNumElements();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Check that all elements have the same opcode.
  // TODO: Should we allow UNDEFS and if so how many?
  unsigned Opcode = Op->getOperand(0).getOpcode();
  for (unsigned i = 1; i < NumElems; ++i)
    if (Opcode != Op->getOperand(i).getOpcode())
      return SDValue();

  // TODO: We may be able to add support for other Ops (ADD/SUB + shifts).
  bool IsShift = false;
  switch (Opcode) {
  default:
    return SDValue();
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    IsShift = true;
    break;
  case ISD::AND:
  case ISD::XOR:
  case ISD::OR:
    // Don't do this if the buildvector is a splat - we'd replace one
    // constant with an entire vector.
    if (Op->getSplatValue())
      return SDValue();
    if (!TLI.isOperationLegalOrPromote(Opcode, VT))
      return SDValue();
    break;
  }

  SmallVector<SDValue, 4> LHSElts, RHSElts;
  for (SDValue Elt : Op->ops()) {
    SDValue LHS = Elt.getOperand(0);
    SDValue RHS = Elt.getOperand(1);

    // We expect the canonicalized RHS operand to be the constant.
    if (!isa<ConstantSDNode>(RHS))
      return SDValue();

    // Extend shift amounts.
    if (RHS.getValueSizeInBits() != VT.getScalarSizeInBits()) {
      if (!IsShift)
        return SDValue();
      RHS = DAG.getZExtOrTrunc(RHS, DL, VT.getScalarType());
    }

    LHSElts.push_back(LHS);
    RHSElts.push_back(RHS);
  }

  // Limit to shifts by uniform immediates.
  // TODO: Only accept vXi8/vXi64 special cases?
  // TODO: Permit non-uniform XOP/AVX2/MULLO cases?
  if (IsShift && any_of(RHSElts, [&](SDValue V) { return RHSElts[0] != V; }))
    return SDValue();

  SDValue LHS = DAG.getBuildVector(VT, DL, LHSElts);
  SDValue RHS = DAG.getBuildVector(VT, DL, RHSElts);
  return DAG.getNode(Opcode, DL, VT, LHS, RHS);
}

/// Create a vector constant without a load. SSE/AVX provide the bare minimum
/// functionality to do this, so it's all zeros, all ones, or some derivation
/// that is cheap to calculate.
static SDValue materializeVectorConstant(SDValue Op, SelectionDAG &DAG,
                                         const X86Subtarget &Subtarget) {
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();

  // Vectors containing all zeros can be matched by pxor and xorps.
  if (ISD::isBuildVectorAllZeros(Op.getNode()))
    return Op;

  // Vectors containing all ones can be matched by pcmpeqd on 128-bit width
  // vectors or broken into v4i32 operations on 256-bit vectors. AVX2 can use
  // vpcmpeqd on 256-bit vectors.
  if (Subtarget.hasSSE2() && ISD::isBuildVectorAllOnes(Op.getNode())) {
    if (VT == MVT::v4i32 || VT == MVT::v8i32 || VT == MVT::v16i32)
      return Op;

    return getOnesVector(VT, DAG, DL);
  }

  return SDValue();
}

/// Look for opportunities to create a VPERMV/VPERMILPV/PSHUFB variable permute
/// from a vector of source values and a vector of extraction indices.
/// The vectors might be manipulated to match the type of the permute op.
static SDValue createVariablePermute(MVT VT, SDValue SrcVec, SDValue IndicesVec,
                                     SDLoc &DL, SelectionDAG &DAG,
                                     const X86Subtarget &Subtarget) {
  MVT ShuffleVT = VT;
  EVT IndicesVT = EVT(VT).changeVectorElementTypeToInteger();
  unsigned NumElts = VT.getVectorNumElements();
  unsigned SizeInBits = VT.getSizeInBits();

  // Adjust IndicesVec to match VT size.
  assert(IndicesVec.getValueType().getVectorNumElements() >= NumElts &&
         "Illegal variable permute mask size");
  if (IndicesVec.getValueType().getVectorNumElements() > NumElts)
    IndicesVec = extractSubVector(IndicesVec, 0, DAG, SDLoc(IndicesVec),
                                  NumElts * VT.getScalarSizeInBits());
  IndicesVec = DAG.getZExtOrTrunc(IndicesVec, SDLoc(IndicesVec), IndicesVT);

  // Handle SrcVec that don't match VT type.
  if (SrcVec.getValueSizeInBits() != SizeInBits) {
    if ((SrcVec.getValueSizeInBits() % SizeInBits) == 0) {
      // Handle larger SrcVec by treating it as a larger permute.
      unsigned Scale = SrcVec.getValueSizeInBits() / SizeInBits;
      VT = MVT::getVectorVT(VT.getScalarType(), Scale * NumElts);
      IndicesVT = EVT(VT).changeVectorElementTypeToInteger();
      IndicesVec = widenSubVector(IndicesVT.getSimpleVT(), IndicesVec, false,
                                  Subtarget, DAG, SDLoc(IndicesVec));
      return extractSubVector(
          createVariablePermute(VT, SrcVec, IndicesVec, DL, DAG, Subtarget), 0,
          DAG, DL, SizeInBits);
    } else if (SrcVec.getValueSizeInBits() < SizeInBits) {
      // Widen smaller SrcVec to match VT.
      SrcVec = widenSubVector(VT, SrcVec, false, Subtarget, DAG, SDLoc(SrcVec));
    } else
      return SDValue();
  }

  auto ScaleIndices = [&DAG](SDValue Idx, uint64_t Scale) {
    assert(isPowerOf2_64(Scale) && "Illegal variable permute shuffle scale");
    EVT SrcVT = Idx.getValueType();
    unsigned NumDstBits = SrcVT.getScalarSizeInBits() / Scale;
    uint64_t IndexScale = 0;
    uint64_t IndexOffset = 0;

    // If we're scaling a smaller permute op, then we need to repeat the
    // indices, scaling and offsetting them as well.
    // e.g. v4i32 -> v16i8 (Scale = 4)
    // IndexScale = v4i32 Splat(4 << 24 | 4 << 16 | 4 << 8 | 4)
    // IndexOffset = v4i32 Splat(3 << 24 | 2 << 16 | 1 << 8 | 0)
    for (uint64_t i = 0; i != Scale; ++i) {
      IndexScale |= Scale << (i * NumDstBits);
      IndexOffset |= i << (i * NumDstBits);
    }

    Idx = DAG.getNode(ISD::MUL, SDLoc(Idx), SrcVT, Idx,
                      DAG.getConstant(IndexScale, SDLoc(Idx), SrcVT));
    Idx = DAG.getNode(ISD::ADD, SDLoc(Idx), SrcVT, Idx,
                      DAG.getConstant(IndexOffset, SDLoc(Idx), SrcVT));
    return Idx;
  };

  unsigned Opcode = 0;
  switch (VT.SimpleTy) {
  default:
    break;
  case MVT::v16i8:
    if (Subtarget.hasSSSE3())
      Opcode = X86ISD::PSHUFB;
    break;
  case MVT::v8i16:
    if (Subtarget.hasVLX() && Subtarget.hasBWI())
      Opcode = X86ISD::VPERMV;
    else if (Subtarget.hasSSSE3()) {
      Opcode = X86ISD::PSHUFB;
      ShuffleVT = MVT::v16i8;
    }
    break;
  case MVT::v4f32:
  case MVT::v4i32:
    if (Subtarget.hasAVX()) {
      Opcode = X86ISD::VPERMILPV;
      ShuffleVT = MVT::v4f32;
    } else if (Subtarget.hasSSSE3()) {
      Opcode = X86ISD::PSHUFB;
      ShuffleVT = MVT::v16i8;
    }
    break;
  case MVT::v2f64:
  case MVT::v2i64:
    if (Subtarget.hasAVX()) {
      // VPERMILPD selects using bit#1 of the index vector, so scale IndicesVec.
      IndicesVec = DAG.getNode(ISD::ADD, DL, IndicesVT, IndicesVec, IndicesVec);
      Opcode = X86ISD::VPERMILPV;
      ShuffleVT = MVT::v2f64;
    } else if (Subtarget.hasSSE41()) {
      // SSE41 can compare v2i64 - select between indices 0 and 1.
      return DAG.getSelectCC(
          DL, IndicesVec,
          getZeroVector(IndicesVT.getSimpleVT(), Subtarget, DAG, DL),
          DAG.getVectorShuffle(VT, DL, SrcVec, SrcVec, {0, 0}),
          DAG.getVectorShuffle(VT, DL, SrcVec, SrcVec, {1, 1}),
          ISD::CondCode::SETEQ);
    }
    break;
  case MVT::v32i8:
    if (Subtarget.hasVLX() && Subtarget.hasVBMI())
      Opcode = X86ISD::VPERMV;
    else if (Subtarget.hasXOP()) {
      SDValue LoSrc = extract128BitVector(SrcVec, 0, DAG, DL);
      SDValue HiSrc = extract128BitVector(SrcVec, 16, DAG, DL);
      SDValue LoIdx = extract128BitVector(IndicesVec, 0, DAG, DL);
      SDValue HiIdx = extract128BitVector(IndicesVec, 16, DAG, DL);
      return DAG.getNode(
          ISD::CONCAT_VECTORS, DL, VT,
          DAG.getNode(X86ISD::VPPERM, DL, MVT::v16i8, LoSrc, HiSrc, LoIdx),
          DAG.getNode(X86ISD::VPPERM, DL, MVT::v16i8, LoSrc, HiSrc, HiIdx));
    } else if (Subtarget.hasAVX()) {
      SDValue Lo = extract128BitVector(SrcVec, 0, DAG, DL);
      SDValue Hi = extract128BitVector(SrcVec, 16, DAG, DL);
      SDValue LoLo = DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Lo, Lo);
      SDValue HiHi = DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Hi, Hi);
      auto PSHUFBBuilder = [](SelectionDAG &DAG, const SDLoc &DL,
                              ArrayRef<SDValue> Ops) {
        // Permute Lo and Hi and then select based on index range.
        // This works as SHUFB uses bits[3:0] to permute elements and we don't
        // care about the bit[7] as its just an index vector.
        SDValue Idx = Ops[2];
        EVT VT = Idx.getValueType();
        return DAG.getSelectCC(DL, Idx, DAG.getConstant(15, DL, VT),
                               DAG.getNode(X86ISD::PSHUFB, DL, VT, Ops[1], Idx),
                               DAG.getNode(X86ISD::PSHUFB, DL, VT, Ops[0], Idx),
                               ISD::CondCode::SETGT);
      };
      SDValue Ops[] = {LoLo, HiHi, IndicesVec};
      return X86::SplitOpsAndApply(DAG, Subtarget, DL, MVT::v32i8, Ops,
                                   PSHUFBBuilder);
    }
    break;
  case MVT::v16i16:
    if (Subtarget.hasVLX() && Subtarget.hasBWI())
      Opcode = X86ISD::VPERMV;
    else if (Subtarget.hasAVX()) {
      // Scale to v32i8 and perform as v32i8.
      IndicesVec = ScaleIndices(IndicesVec, 2);
      return DAG.getBitcast(
          VT, createVariablePermute(
                  MVT::v32i8, DAG.getBitcast(MVT::v32i8, SrcVec),
                  DAG.getBitcast(MVT::v32i8, IndicesVec), DL, DAG, Subtarget));
    }
    break;
  case MVT::v8f32:
  case MVT::v8i32:
    if (Subtarget.hasAVX2())
      Opcode = X86ISD::VPERMV;
    else if (Subtarget.hasAVX()) {
      SrcVec = DAG.getBitcast(MVT::v8f32, SrcVec);
      SDValue LoLo = DAG.getVectorShuffle(MVT::v8f32, DL, SrcVec, SrcVec,
                                          {0, 1, 2, 3, 0, 1, 2, 3});
      SDValue HiHi = DAG.getVectorShuffle(MVT::v8f32, DL, SrcVec, SrcVec,
                                          {4, 5, 6, 7, 4, 5, 6, 7});
      if (Subtarget.hasXOP())
        return DAG.getBitcast(VT, DAG.getNode(X86ISD::VPERMIL2, DL, MVT::v8f32,
                                              LoLo, HiHi, IndicesVec,
                                              DAG.getConstant(0, DL, MVT::i8)));
      // Permute Lo and Hi and then select based on index range.
      // This works as VPERMILPS only uses index bits[0:1] to permute elements.
      SDValue Res = DAG.getSelectCC(
          DL, IndicesVec, DAG.getConstant(3, DL, MVT::v8i32),
          DAG.getNode(X86ISD::VPERMILPV, DL, MVT::v8f32, HiHi, IndicesVec),
          DAG.getNode(X86ISD::VPERMILPV, DL, MVT::v8f32, LoLo, IndicesVec),
          ISD::CondCode::SETGT);
      return DAG.getBitcast(VT, Res);
    }
    break;
  case MVT::v4i64:
  case MVT::v4f64:
    if (Subtarget.hasAVX512()) {
      if (!Subtarget.hasVLX()) {
        MVT WidenSrcVT = MVT::getVectorVT(VT.getScalarType(), 8);
        SrcVec = widenSubVector(WidenSrcVT, SrcVec, false, Subtarget, DAG,
                                SDLoc(SrcVec));
        IndicesVec = widenSubVector(MVT::v8i64, IndicesVec, false, Subtarget,
                                    DAG, SDLoc(IndicesVec));
        SDValue Res = createVariablePermute(WidenSrcVT, SrcVec, IndicesVec, DL,
                                            DAG, Subtarget);
        return X86::extract256BitVector(Res, 0, DAG, DL);
      }
      Opcode = X86ISD::VPERMV;
    } else if (Subtarget.hasAVX()) {
      SrcVec = DAG.getBitcast(MVT::v4f64, SrcVec);
      SDValue LoLo =
          DAG.getVectorShuffle(MVT::v4f64, DL, SrcVec, SrcVec, {0, 1, 0, 1});
      SDValue HiHi =
          DAG.getVectorShuffle(MVT::v4f64, DL, SrcVec, SrcVec, {2, 3, 2, 3});
      // VPERMIL2PD selects with bit#1 of the index vector, so scale IndicesVec.
      IndicesVec = DAG.getNode(ISD::ADD, DL, IndicesVT, IndicesVec, IndicesVec);
      if (Subtarget.hasXOP())
        return DAG.getBitcast(VT, DAG.getNode(X86ISD::VPERMIL2, DL, MVT::v4f64,
                                              LoLo, HiHi, IndicesVec,
                                              DAG.getConstant(0, DL, MVT::i8)));
      // Permute Lo and Hi and then select based on index range.
      // This works as VPERMILPD only uses index bit[1] to permute elements.
      SDValue Res = DAG.getSelectCC(
          DL, IndicesVec, DAG.getConstant(2, DL, MVT::v4i64),
          DAG.getNode(X86ISD::VPERMILPV, DL, MVT::v4f64, HiHi, IndicesVec),
          DAG.getNode(X86ISD::VPERMILPV, DL, MVT::v4f64, LoLo, IndicesVec),
          ISD::CondCode::SETGT);
      return DAG.getBitcast(VT, Res);
    }
    break;
  case MVT::v64i8:
    if (Subtarget.hasVBMI())
      Opcode = X86ISD::VPERMV;
    break;
  case MVT::v32i16:
    if (Subtarget.hasBWI())
      Opcode = X86ISD::VPERMV;
    break;
  case MVT::v16f32:
  case MVT::v16i32:
  case MVT::v8f64:
  case MVT::v8i64:
    if (Subtarget.hasAVX512())
      Opcode = X86ISD::VPERMV;
    break;
  }
  if (!Opcode)
    return SDValue();

  assert((VT.getSizeInBits() == ShuffleVT.getSizeInBits()) &&
         (VT.getScalarSizeInBits() % ShuffleVT.getScalarSizeInBits()) == 0 &&
         "Illegal variable permute shuffle type");

  uint64_t Scale = VT.getScalarSizeInBits() / ShuffleVT.getScalarSizeInBits();
  if (Scale > 1)
    IndicesVec = ScaleIndices(IndicesVec, Scale);

  EVT ShuffleIdxVT = EVT(ShuffleVT).changeVectorElementTypeToInteger();
  IndicesVec = DAG.getBitcast(ShuffleIdxVT, IndicesVec);

  SrcVec = DAG.getBitcast(ShuffleVT, SrcVec);
  SDValue Res = Opcode == X86ISD::VPERMV
                    ? DAG.getNode(Opcode, DL, ShuffleVT, IndicesVec, SrcVec)
                    : DAG.getNode(Opcode, DL, ShuffleVT, SrcVec, IndicesVec);
  return DAG.getBitcast(VT, Res);
}

// Tries to lower a BUILD_VECTOR composed of extract-extract chains that can be
// reasoned to be a permutation of a vector by indices in a non-constant vector.
// (build_vector (extract_elt V, (extract_elt I, 0)),
//               (extract_elt V, (extract_elt I, 1)),
//                    ...
// ->
// (vpermv I, V)
//
// TODO: Handle undefs
// TODO: Utilize pshufb and zero mask blending to support more efficient
// construction of vectors with constant-0 elements.
static SDValue
LowerBUILD_VECTORAsVariablePermute(SDValue V, SelectionDAG &DAG,
                                   const X86Subtarget &Subtarget) {
  SDValue SrcVec, IndicesVec;
  // Check for a match of the permute source vector and permute index elements.
  // This is done by checking that the i-th build_vector operand is of the form:
  // (extract_elt SrcVec, (extract_elt IndicesVec, i)).
  for (unsigned Idx = 0, E = V.getNumOperands(); Idx != E; ++Idx) {
    SDValue Op = V.getOperand(Idx);
    if (Op.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return SDValue();

    // If this is the first extract encountered in V, set the source vector,
    // otherwise verify the extract is from the previously defined source
    // vector.
    if (!SrcVec)
      SrcVec = Op.getOperand(0);
    else if (SrcVec != Op.getOperand(0))
      return SDValue();
    SDValue ExtractedIndex = Op->getOperand(1);
    // Peek through extends.
    if (ExtractedIndex.getOpcode() == ISD::ZERO_EXTEND ||
        ExtractedIndex.getOpcode() == ISD::SIGN_EXTEND)
      ExtractedIndex = ExtractedIndex.getOperand(0);
    if (ExtractedIndex.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return SDValue();

    // If this is the first extract from the index vector candidate, set the
    // indices vector, otherwise verify the extract is from the previously
    // defined indices vector.
    if (!IndicesVec)
      IndicesVec = ExtractedIndex.getOperand(0);
    else if (IndicesVec != ExtractedIndex.getOperand(0))
      return SDValue();

    auto *PermIdx = dyn_cast<ConstantSDNode>(ExtractedIndex.getOperand(1));
    if (!PermIdx || PermIdx->getAPIntValue() != Idx)
      return SDValue();
  }

  SDLoc DL(V);
  MVT VT = V.getSimpleValueType();
  return createVariablePermute(VT, SrcVec, IndicesVec, DL, DAG, Subtarget);
}

SDValue
X86TargetLowering::LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);

  MVT VT = Op.getSimpleValueType();
  MVT EltVT = VT.getVectorElementType();
  unsigned NumElems = Op.getNumOperands();

  // Generate vectors for predicate vectors.
  if (VT.getVectorElementType() == MVT::i1 && Subtarget.hasAVX512())
    return LowerBUILD_VECTORvXi1(Op, DAG, Subtarget);

  if (SDValue VectorConstant = materializeVectorConstant(Op, DAG, Subtarget))
    return VectorConstant;

  BuildVectorSDNode *BV = cast<BuildVectorSDNode>(Op.getNode());
  if (SDValue AddSub = lowerToAddSubOrFMAddSub(BV, Subtarget, DAG))
    return AddSub;
  if (SDValue HorizontalOp = LowerToHorizontalOp(BV, Subtarget, DAG))
    return HorizontalOp;
  if (SDValue Broadcast = lowerBuildVectorAsBroadcast(BV, Subtarget, DAG))
    return Broadcast;
  if (SDValue BitOp = lowerBuildVectorToBitOp(BV, DAG))
    return BitOp;

  unsigned EVTBits = EltVT.getSizeInBits();

  unsigned NumZero  = 0;
  unsigned NumNonZero = 0;
  uint64_t NonZeros = 0;
  bool IsAllConstants = true;
  SmallSet<SDValue, 8> Values;
  unsigned NumConstants = NumElems;
  for (unsigned i = 0; i < NumElems; ++i) {
    SDValue Elt = Op.getOperand(i);
    if (Elt.isUndef())
      continue;
    Values.insert(Elt);
    if (!isa<ConstantSDNode>(Elt) && !isa<ConstantFPSDNode>(Elt)) {
      IsAllConstants = false;
      NumConstants--;
    }
    if (X86::isZeroNode(Elt))
      NumZero++;
    else {
      assert(i < sizeof(NonZeros) * 8); // Make sure the shift is within range.
      NonZeros |= ((uint64_t)1 << i);
      NumNonZero++;
    }
  }

  // All undef vector. Return an UNDEF.  All zero vectors were handled above.
  if (NumNonZero == 0)
    return DAG.getUNDEF(VT);

  // If we are inserting one variable into a vector of non-zero constants, try
  // to avoid loading each constant element as a scalar. Load the constants as a
  // vector and then insert the variable scalar element. If insertion is not
  // supported, fall back to a shuffle to get the scalar blended with the
  // constants. Insertion into a zero vector is handled as a special-case
  // somewhere below here.
  if (NumConstants == NumElems - 1 && NumNonZero != 1 &&
      (isOperationLegalOrCustom(ISD::INSERT_VECTOR_ELT, VT) ||
       isOperationLegalOrCustom(ISD::VECTOR_SHUFFLE, VT))) {
    // Create an all-constant vector. The variable element in the old
    // build vector is replaced by undef in the constant vector. Save the
    // variable scalar element and its index for use in the insertelement.
    LLVMContext &Context = *DAG.getContext();
    Type *EltType = Op.getValueType().getScalarType().getTypeForEVT(Context);
    SmallVector<Constant *, 16> ConstVecOps(NumElems, UndefValue::get(EltType));
    SDValue VarElt;
    SDValue InsIndex;
    for (unsigned i = 0; i != NumElems; ++i) {
      SDValue Elt = Op.getOperand(i);
      if (auto *C = dyn_cast<ConstantSDNode>(Elt))
        ConstVecOps[i] = ConstantInt::get(Context, C->getAPIntValue());
      else if (auto *C = dyn_cast<ConstantFPSDNode>(Elt))
        ConstVecOps[i] = ConstantFP::get(Context, C->getValueAPF());
      else if (!Elt.isUndef()) {
        assert(!VarElt.getNode() && !InsIndex.getNode() &&
               "Expected one variable element in this vector");
        VarElt = Elt;
        InsIndex = DAG.getConstant(i, dl, getVectorIdxTy(DAG.getDataLayout()));
      }
    }
    Constant *CV = ConstantVector::get(ConstVecOps);
    SDValue DAGConstVec = DAG.getConstantPool(CV, VT);

    // The constants we just created may not be legal (eg, floating point). We
    // must lower the vector right here because we can not guarantee that we'll
    // legalize it before loading it. This is also why we could not just create
    // a new build vector here. If the build vector contains illegal constants,
    // it could get split back up into a series of insert elements.
    // TODO: Improve this by using shorter loads with broadcast/VZEXT_LOAD.
    SDValue LegalDAGConstVec = LowerConstantPool(DAGConstVec, DAG);
    MachineFunction &MF = DAG.getMachineFunction();
    MachinePointerInfo MPI = MachinePointerInfo::getConstantPool(MF);
    SDValue Ld = DAG.getLoad(VT, dl, DAG.getEntryNode(), LegalDAGConstVec, MPI);
    unsigned InsertC = cast<ConstantSDNode>(InsIndex)->getZExtValue();
    unsigned NumEltsInLow128Bits = 128 / VT.getScalarSizeInBits();
    if (InsertC < NumEltsInLow128Bits)
      return DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, Ld, VarElt, InsIndex);

    // There's no good way to insert into the high elements of a >128-bit
    // vector, so use shuffles to avoid an extract/insert sequence.
    assert(VT.getSizeInBits() > 128 && "Invalid insertion index?");
    assert(Subtarget.hasAVX() && "Must have AVX with >16-byte vector");
    SmallVector<int, 8> ShuffleMask;
    unsigned NumElts = VT.getVectorNumElements();
    for (unsigned i = 0; i != NumElts; ++i)
      ShuffleMask.push_back(i == InsertC ? NumElts : i);
    SDValue S2V = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, VarElt);
    return DAG.getVectorShuffle(VT, dl, Ld, S2V, ShuffleMask);
  }

  // Special case for single non-zero, non-undef, element.
  if (NumNonZero == 1) {
    unsigned Idx = countTrailingZeros(NonZeros);
    SDValue Item = Op.getOperand(Idx);

    // If we have a constant or non-constant insertion into the low element of
    // a vector, we can do this with SCALAR_TO_VECTOR + shuffle of zero into
    // the rest of the elements.  This will be matched as movd/movq/movss/movsd
    // depending on what the source datatype is.
    if (Idx == 0) {
      if (NumZero == 0)
        return DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Item);

      if (EltVT == MVT::i32 || EltVT == MVT::f32 || EltVT == MVT::f64 ||
          (EltVT == MVT::i64 && Subtarget.is64Bit())) {
        assert((VT.is128BitVector() || VT.is256BitVector() ||
                VT.is512BitVector()) &&
               "Expected an SSE value type!");
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Item);
        // Turn it into a MOVL (i.e. movss, movsd, or movd) to a zero vector.
        return getShuffleVectorZeroOrUndef(Item, 0, true, Subtarget, DAG);
      }

      // We can't directly insert an i8 or i16 into a vector, so zero extend
      // it to i32 first.
      if (EltVT == MVT::i16 || EltVT == MVT::i8) {
        Item = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Item);
        MVT ShufVT = MVT::getVectorVT(MVT::i32, VT.getSizeInBits()/32);
        Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, ShufVT, Item);
        Item = getShuffleVectorZeroOrUndef(Item, 0, true, Subtarget, DAG);
        return DAG.getBitcast(VT, Item);
      }
    }

    // Is it a vector logical left shift?
    if (NumElems == 2 && Idx == 1 &&
        X86::isZeroNode(Op.getOperand(0)) &&
        !X86::isZeroNode(Op.getOperand(1))) {
      unsigned NumBits = VT.getSizeInBits();
      return getVShift(true, VT,
                       DAG.getNode(ISD::SCALAR_TO_VECTOR, dl,
                                   VT, Op.getOperand(1)),
                       NumBits/2, DAG, *this, dl);
    }

    if (IsAllConstants) // Otherwise, it's better to do a constpool load.
      return SDValue();

    // Otherwise, if this is a vector with i32 or f32 elements, and the element
    // is a non-constant being inserted into an element other than the low one,
    // we can't use a constant pool load.  Instead, use SCALAR_TO_VECTOR (aka
    // movd/movss) to move this into the low element, then shuffle it into
    // place.
    if (EVTBits == 32) {
      Item = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Item);
      return getShuffleVectorZeroOrUndef(Item, Idx, NumZero > 0, Subtarget, DAG);
    }
  }

  // Splat is obviously ok. Let legalizer expand it to a shuffle.
  if (Values.size() == 1) {
    if (EVTBits == 32) {
      // Instead of a shuffle like this:
      // shuffle (scalar_to_vector (load (ptr + 4))), undef, <0, 0, 0, 0>
      // Check if it's possible to issue this instead.
      // shuffle (vload ptr)), undef, <1, 1, 1, 1>
      unsigned Idx = countTrailingZeros(NonZeros);
      SDValue Item = Op.getOperand(Idx);
      if (Op.getNode()->isOnlyUserOf(Item.getNode()))
        return LowerAsSplatVectorLoad(Item, VT, dl, DAG);
    }
    return SDValue();
  }

  // A vector full of immediates; various special cases are already
  // handled, so this is best done with a single constant-pool load.
  if (IsAllConstants)
    return SDValue();

  if (SDValue V = LowerBUILD_VECTORAsVariablePermute(Op, DAG, Subtarget))
      return V;

  // See if we can use a vector load to get all of the elements.
  {
    SmallVector<SDValue, 64> Ops(Op->op_begin(), Op->op_begin() + NumElems);
    if (SDValue LD =
            EltsFromConsecutiveLoads(VT, Ops, dl, DAG, Subtarget, false))
      return LD;
  }

  // If this is a splat of pairs of 32-bit elements, we can use a narrower
  // build_vector and broadcast it.
  // TODO: We could probably generalize this more.
  if (Subtarget.hasAVX2() && EVTBits == 32 && Values.size() == 2) {
    SDValue Ops[4] = { Op.getOperand(0), Op.getOperand(1),
                       DAG.getUNDEF(EltVT), DAG.getUNDEF(EltVT) };
    auto CanSplat = [](SDValue Op, unsigned NumElems, ArrayRef<SDValue> Ops) {
      // Make sure all the even/odd operands match.
      for (unsigned i = 2; i != NumElems; ++i)
        if (Ops[i % 2] != Op.getOperand(i))
          return false;
      return true;
    };
    if (CanSplat(Op, NumElems, Ops)) {
      MVT WideEltVT = VT.isFloatingPoint() ? MVT::f64 : MVT::i64;
      MVT NarrowVT = MVT::getVectorVT(EltVT, 4);
      // Create a new build vector and cast to v2i64/v2f64.
      SDValue NewBV = DAG.getBitcast(MVT::getVectorVT(WideEltVT, 2),
                                     DAG.getBuildVector(NarrowVT, dl, Ops));
      // Broadcast from v2i64/v2f64 and cast to final VT.
      MVT BcastVT = MVT::getVectorVT(WideEltVT, NumElems/2);
      return DAG.getBitcast(VT, DAG.getNode(X86ISD::VBROADCAST, dl, BcastVT,
                                            NewBV));
    }
  }

  // For AVX-length vectors, build the individual 128-bit pieces and use
  // shuffles to put them in place.
  if (VT.getSizeInBits() > 128) {
    MVT HVT = MVT::getVectorVT(EltVT, NumElems/2);

    // Build both the lower and upper subvector.
    SDValue Lower =
        DAG.getBuildVector(HVT, dl, Op->ops().slice(0, NumElems / 2));
    SDValue Upper = DAG.getBuildVector(
        HVT, dl, Op->ops().slice(NumElems / 2, NumElems /2));

    // Recreate the wider vector with the lower and upper part.
    return X86::concatSubVectors(Lower, Upper, DAG, dl);
  }

  // Let legalizer expand 2-wide build_vectors.
  if (EVTBits == 64) {
    if (NumNonZero == 1) {
      // One half is zero or undef.
      unsigned Idx = countTrailingZeros(NonZeros);
      SDValue V2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT,
                               Op.getOperand(Idx));
      return getShuffleVectorZeroOrUndef(V2, Idx, true, Subtarget, DAG);
    }
    return SDValue();
  }

  // If element VT is < 32 bits, convert it to inserts into a zero vector.
  if (EVTBits == 8 && NumElems == 16)
    if (SDValue V = LowerBuildVectorv16i8(Op, NonZeros, NumNonZero, NumZero,
                                          DAG, Subtarget))
      return V;

  if (EVTBits == 16 && NumElems == 8)
    if (SDValue V = LowerBuildVectorv8i16(Op, NonZeros, NumNonZero, NumZero,
                                          DAG, Subtarget))
      return V;

  // If element VT is == 32 bits and has 4 elems, try to generate an INSERTPS
  if (EVTBits == 32 && NumElems == 4)
    if (SDValue V = LowerBuildVectorv4x32(Op, DAG, Subtarget))
      return V;

  // If element VT is == 32 bits, turn it into a number of shuffles.
  if (NumElems == 4 && NumZero > 0) {
    SmallVector<SDValue, 8> Ops(NumElems);
    for (unsigned i = 0; i < 4; ++i) {
      bool isZero = !(NonZeros & (1ULL << i));
      if (isZero)
        Ops[i] = getZeroVector(VT, Subtarget, DAG, dl);
      else
        Ops[i] = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Op.getOperand(i));
    }

    for (unsigned i = 0; i < 2; ++i) {
      switch ((NonZeros >> (i*2)) & 0x3) {
        default: llvm_unreachable("Unexpected NonZero count");
        case 0:
          Ops[i] = Ops[i*2];  // Must be a zero vector.
          break;
        case 1:
          Ops[i] = getMOVL(DAG, dl, VT, Ops[i*2+1], Ops[i*2]);
          break;
        case 2:
          Ops[i] = getMOVL(DAG, dl, VT, Ops[i*2], Ops[i*2+1]);
          break;
        case 3:
          Ops[i] = getUnpackl(DAG, dl, VT, Ops[i*2], Ops[i*2+1]);
          break;
      }
    }

    bool Reverse1 = (NonZeros & 0x3) == 2;
    bool Reverse2 = ((NonZeros & (0x3 << 2)) >> 2) == 2;
    int MaskVec[] = {
      Reverse1 ? 1 : 0,
      Reverse1 ? 0 : 1,
      static_cast<int>(Reverse2 ? NumElems+1 : NumElems),
      static_cast<int>(Reverse2 ? NumElems   : NumElems+1)
    };
    return DAG.getVectorShuffle(VT, dl, Ops[0], Ops[1], MaskVec);
  }

  assert(Values.size() > 1 && "Expected non-undef and non-splat vector");

  // Check for a build vector from mostly shuffle plus few inserting.
  if (SDValue Sh = buildFromShuffleMostly(Op, DAG))
    return Sh;

  // For SSE 4.1, use insertps to put the high elements into the low element.
  if (Subtarget.hasSSE41()) {
    SDValue Result;
    if (!Op.getOperand(0).isUndef())
      Result = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Op.getOperand(0));
    else
      Result = DAG.getUNDEF(VT);

    for (unsigned i = 1; i < NumElems; ++i) {
      if (Op.getOperand(i).isUndef()) continue;
      Result = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, VT, Result,
                           Op.getOperand(i), DAG.getIntPtrConstant(i, dl));
    }
    return Result;
  }

  // Otherwise, expand into a number of unpckl*, start by extending each of
  // our (non-undef) elements to the full vector width with the element in the
  // bottom slot of the vector (which generates no code for SSE).
  SmallVector<SDValue, 8> Ops(NumElems);
  for (unsigned i = 0; i < NumElems; ++i) {
    if (!Op.getOperand(i).isUndef())
      Ops[i] = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, Op.getOperand(i));
    else
      Ops[i] = DAG.getUNDEF(VT);
  }

  // Next, we iteratively mix elements, e.g. for v4f32:
  //   Step 1: unpcklps 0, 1 ==> X: <?, ?, 1, 0>
  //         : unpcklps 2, 3 ==> Y: <?, ?, 3, 2>
  //   Step 2: unpcklpd X, Y ==>    <3, 2, 1, 0>
  for (unsigned Scale = 1; Scale < NumElems; Scale *= 2) {
    // Generate scaled UNPCKL shuffle mask.
    SmallVector<int, 16> Mask;
    for(unsigned i = 0; i != Scale; ++i)
      Mask.push_back(i);
    for (unsigned i = 0; i != Scale; ++i)
      Mask.push_back(NumElems+i);
    Mask.append(NumElems - Mask.size(), SM_SentinelUndef);

    for (unsigned i = 0, e = NumElems / (2 * Scale); i != e; ++i)
      Ops[i] = DAG.getVectorShuffle(VT, dl, Ops[2*i], Ops[(2*i)+1], Mask);
  }
  return Ops[0];
}

// 256-bit AVX can use the vinsertf128 instruction
// to create 256-bit vectors from two other 128-bit ones.
// TODO: Detect subvector broadcast here instead of DAG combine?
static SDValue LowerAVXCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG,
                                      const X86Subtarget &Subtarget) {
  SDLoc dl(Op);
  MVT ResVT = Op.getSimpleValueType();

  assert((ResVT.is256BitVector() ||
          ResVT.is512BitVector()) && "Value type must be 256-/512-bit wide");

  unsigned NumOperands = Op.getNumOperands();
  unsigned NumZero = 0;
  unsigned NumNonZero = 0;
  unsigned NonZeros = 0;
  for (unsigned i = 0; i != NumOperands; ++i) {
    SDValue SubVec = Op.getOperand(i);
    if (SubVec.isUndef())
      continue;
    if (ISD::isBuildVectorAllZeros(SubVec.getNode()))
      ++NumZero;
    else {
      assert(i < sizeof(NonZeros) * CHAR_BIT); // Ensure the shift is in range.
      NonZeros |= 1 << i;
      ++NumNonZero;
    }
  }

  // If we have more than 2 non-zeros, build each half separately.
  if (NumNonZero > 2) {
    MVT HalfVT = ResVT.getHalfNumVectorElementsVT();
    ArrayRef<SDUse> Ops = Op->ops();
    SDValue Lo = DAG.getNode(ISD::CONCAT_VECTORS, dl, HalfVT,
                             Ops.slice(0, NumOperands/2));
    SDValue Hi = DAG.getNode(ISD::CONCAT_VECTORS, dl, HalfVT,
                             Ops.slice(NumOperands/2));
    return DAG.getNode(ISD::CONCAT_VECTORS, dl, ResVT, Lo, Hi);
  }

  // Otherwise, build it up through insert_subvectors.
  SDValue Vec = NumZero ? getZeroVector(ResVT, Subtarget, DAG, dl)
                        : DAG.getUNDEF(ResVT);

  MVT SubVT = Op.getOperand(0).getSimpleValueType();
  unsigned NumSubElems = SubVT.getVectorNumElements();
  for (unsigned i = 0; i != NumOperands; ++i) {
    if ((NonZeros & (1 << i)) == 0)
      continue;

    Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, ResVT, Vec,
                      Op.getOperand(i),
                      DAG.getIntPtrConstant(i * NumSubElems, dl));
  }

  return Vec;
}

// Returns true if the given node is a type promotion (by concatenating i1
// zeros) of the result of a node that already zeros all upper bits of
// k-register.
// TODO: Merge this with LowerAVXCONCAT_VECTORS?
static SDValue LowerCONCAT_VECTORSvXi1(SDValue Op,
                                       const X86Subtarget &Subtarget,
                                       SelectionDAG & DAG) {
  SDLoc dl(Op);
  MVT ResVT = Op.getSimpleValueType();
  unsigned NumOperands = Op.getNumOperands();

  assert(NumOperands > 1 && isPowerOf2_32(NumOperands) &&
         "Unexpected number of operands in CONCAT_VECTORS");

  uint64_t Zeros = 0;
  uint64_t NonZeros = 0;
  for (unsigned i = 0; i != NumOperands; ++i) {
    SDValue SubVec = Op.getOperand(i);
    if (SubVec.isUndef())
      continue;
    assert(i < sizeof(NonZeros) * CHAR_BIT); // Ensure the shift is in range.
    if (ISD::isBuildVectorAllZeros(SubVec.getNode()))
      Zeros |= (uint64_t)1 << i;
    else
      NonZeros |= (uint64_t)1 << i;
  }

  unsigned NumElems = ResVT.getVectorNumElements();

  // If we are inserting non-zero vector and there are zeros in LSBs and undef
  // in the MSBs we need to emit a KSHIFTL. The generic lowering to
  // insert_subvector will give us two kshifts.
  if (isPowerOf2_64(NonZeros) && Zeros != 0 && NonZeros > Zeros &&
      Log2_64(NonZeros) != NumOperands - 1) {
    MVT ShiftVT = ResVT;
    if ((!Subtarget.hasDQI() && NumElems == 8) || NumElems < 8)
      ShiftVT = Subtarget.hasDQI() ? MVT::v8i1 : MVT::v16i1;
    unsigned Idx = Log2_64(NonZeros);
    SDValue SubVec = Op.getOperand(Idx);
    unsigned SubVecNumElts = SubVec.getSimpleValueType().getVectorNumElements();
    SubVec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, ShiftVT,
                         DAG.getUNDEF(ShiftVT), SubVec,
                         DAG.getIntPtrConstant(0, dl));
    Op = DAG.getNode(X86ISD::KSHIFTL, dl, ShiftVT, SubVec,
                     DAG.getConstant(Idx * SubVecNumElts, dl, MVT::i8));
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, ResVT, Op,
                       DAG.getIntPtrConstant(0, dl));
  }

  // If there are zero or one non-zeros we can handle this very simply.
  if (NonZeros == 0 || isPowerOf2_64(NonZeros)) {
    SDValue Vec = Zeros ? DAG.getConstant(0, dl, ResVT) : DAG.getUNDEF(ResVT);
    if (!NonZeros)
      return Vec;
    unsigned Idx = Log2_64(NonZeros);
    SDValue SubVec = Op.getOperand(Idx);
    unsigned SubVecNumElts = SubVec.getSimpleValueType().getVectorNumElements();
    return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, ResVT, Vec, SubVec,
                       DAG.getIntPtrConstant(Idx * SubVecNumElts, dl));
  }

  if (NumOperands > 2) {
    MVT HalfVT = ResVT.getHalfNumVectorElementsVT();
    ArrayRef<SDUse> Ops = Op->ops();
    SDValue Lo = DAG.getNode(ISD::CONCAT_VECTORS, dl, HalfVT,
                             Ops.slice(0, NumOperands/2));
    SDValue Hi = DAG.getNode(ISD::CONCAT_VECTORS, dl, HalfVT,
                             Ops.slice(NumOperands/2));
    return DAG.getNode(ISD::CONCAT_VECTORS, dl, ResVT, Lo, Hi);
  }

  assert(countPopulation(NonZeros) == 2 && "Simple cases not handled?");

  if (ResVT.getVectorNumElements() >= 16)
    return Op; // The operation is legal with KUNPCK

  SDValue Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, ResVT,
                            DAG.getUNDEF(ResVT), Op.getOperand(0),
                            DAG.getIntPtrConstant(0, dl));
  return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, ResVT, Vec, Op.getOperand(1),
                     DAG.getIntPtrConstant(NumElems/2, dl));
}

SDValue X86TargetLowering::LowerCONCAT_VECTORS(SDValue Op,
                                               SelectionDAG &DAG) const {
  MVT VT = Op.getSimpleValueType();
  if (VT.getVectorElementType() == MVT::i1)
    return LowerCONCAT_VECTORSvXi1(Op, Subtarget, DAG);

  assert((VT.is256BitVector() && Op.getNumOperands() == 2) ||
         (VT.is512BitVector() && (Op.getNumOperands() == 2 ||
          Op.getNumOperands() == 4)));

  // AVX can use the vinsertf128 instruction to create 256-bit vectors
  // from two other 128-bit ones.

  // 512-bit vector may contain 2 256-bit vectors or 4 128-bit vectors
  return LowerAVXCONCAT_VECTORS(Op, DAG, Subtarget);
}

//===----------------------------------------------------------------------===//
// Vector shuffle lowering
//
// This is an experimental code path for lowering vector shuffles on x86. It is
// designed to handle arbitrary vector shuffles and blends, gracefully
// degrading performance as necessary. It works hard to recognize idiomatic
// shuffles and lower them to optimal instruction patterns without leaving
// a framework that allows reasonably efficient handling of all vector shuffle
// patterns.
//===----------------------------------------------------------------------===//

/// Tiny helper function to identify a no-op mask.
///
/// This is a somewhat boring predicate function. It checks whether the mask
/// array input, which is assumed to be a single-input shuffle mask of the kind
/// used by the X86 shuffle instructions (not a fully general
/// ShuffleVectorSDNode mask) requires any shuffles to occur. Both undef and an
/// in-place shuffle are 'no-op's.
static bool isNoopShuffleMask(ArrayRef<int> Mask) {
  for (int i = 0, Size = Mask.size(); i < Size; ++i) {
    assert(Mask[i] >= -1 && "Out of bound mask element!");
    if (Mask[i] >= 0 && Mask[i] != i)
      return false;
  }
  return true;
}

/// Test whether there are elements crossing 128-bit lanes in this
/// shuffle mask.
///
/// X86 divides up its shuffles into in-lane and cross-lane shuffle operations
/// and we routinely test for these.
static bool is128BitLaneCrossingShuffleMask(MVT VT, ArrayRef<int> Mask) {
  int LaneSize = 128 / VT.getScalarSizeInBits();
  int Size = Mask.size();
  for (int i = 0; i < Size; ++i)
    if (Mask[i] >= 0 && (Mask[i] % Size) / LaneSize != i / LaneSize)
      return true;
  return false;
}

/// Test whether a shuffle mask is equivalent within each sub-lane.
///
/// This checks a shuffle mask to see if it is performing the same
/// lane-relative shuffle in each sub-lane. This trivially implies
/// that it is also not lane-crossing. It may however involve a blend from the
/// same lane of a second vector.
///
/// The specific repeated shuffle mask is populated in \p RepeatedMask, as it is
/// non-trivial to compute in the face of undef lanes. The representation is
/// suitable for use with existing 128-bit shuffles as entries from the second
/// vector have been remapped to [LaneSize, 2*LaneSize).
static bool isRepeatedShuffleMask(unsigned LaneSizeInBits, MVT VT,
                                  ArrayRef<int> Mask,
                                  SmallVectorImpl<int> &RepeatedMask) {
  auto LaneSize = LaneSizeInBits / VT.getScalarSizeInBits();
  RepeatedMask.assign(LaneSize, -1);
  int Size = Mask.size();
  for (int i = 0; i < Size; ++i) {
    assert(Mask[i] == SM_SentinelUndef || Mask[i] >= 0);
    if (Mask[i] < 0)
      continue;
    if ((Mask[i] % Size) / LaneSize != i / LaneSize)
      // This entry crosses lanes, so there is no way to model this shuffle.
      return false;

    // Ok, handle the in-lane shuffles by detecting if and when they repeat.
    // Adjust second vector indices to start at LaneSize instead of Size.
    int LocalM = Mask[i] < Size ? Mask[i] % LaneSize
                                : Mask[i] % LaneSize + LaneSize;
    if (RepeatedMask[i % LaneSize] < 0)
      // This is the first non-undef entry in this slot of a 128-bit lane.
      RepeatedMask[i % LaneSize] = LocalM;
    else if (RepeatedMask[i % LaneSize] != LocalM)
      // Found a mismatch with the repeated mask.
      return false;
  }
  return true;
}

/// Test whether a shuffle mask is equivalent within each 128-bit lane.
static bool
is128BitLaneRepeatedShuffleMask(MVT VT, ArrayRef<int> Mask,
                                SmallVectorImpl<int> &RepeatedMask) {
  return isRepeatedShuffleMask(128, VT, Mask, RepeatedMask);
}

bool X86::is128BitLaneRepeatedShuffleMask(MVT VT, ArrayRef<int> Mask) {
  SmallVector<int, 32> RepeatedMask;
  return isRepeatedShuffleMask(128, VT, Mask, RepeatedMask);
}

/// Test whether a shuffle mask is equivalent within each 256-bit lane.
static bool
is256BitLaneRepeatedShuffleMask(MVT VT, ArrayRef<int> Mask,
                                SmallVectorImpl<int> &RepeatedMask) {
  return isRepeatedShuffleMask(256, VT, Mask, RepeatedMask);
}

/// Test whether a target shuffle mask is equivalent within each sub-lane.
/// Unlike isRepeatedShuffleMask we must respect SM_SentinelZero.
static bool isRepeatedTargetShuffleMask(unsigned LaneSizeInBits, MVT VT,
                                        ArrayRef<int> Mask,
                                        SmallVectorImpl<int> &RepeatedMask) {
  int LaneSize = LaneSizeInBits / VT.getScalarSizeInBits();
  RepeatedMask.assign(LaneSize, SM_SentinelUndef);
  int Size = Mask.size();
  for (int i = 0; i < Size; ++i) {
    assert(isUndefOrZero(Mask[i]) || (Mask[i] >= 0));
    if (Mask[i] == SM_SentinelUndef)
      continue;
    if (Mask[i] == SM_SentinelZero) {
      if (!isUndefOrZero(RepeatedMask[i % LaneSize]))
        return false;
      RepeatedMask[i % LaneSize] = SM_SentinelZero;
      continue;
    }
    if ((Mask[i] % Size) / LaneSize != i / LaneSize)
      // This entry crosses lanes, so there is no way to model this shuffle.
      return false;

    // Ok, handle the in-lane shuffles by detecting if and when they repeat.
    // Adjust second vector indices to start at LaneSize instead of Size.
    int LocalM =
        Mask[i] < Size ? Mask[i] % LaneSize : Mask[i] % LaneSize + LaneSize;
    if (RepeatedMask[i % LaneSize] == SM_SentinelUndef)
      // This is the first non-undef entry in this slot of a 128-bit lane.
      RepeatedMask[i % LaneSize] = LocalM;
    else if (RepeatedMask[i % LaneSize] != LocalM)
      // Found a mismatch with the repeated mask.
      return false;
  }
  return true;
}

/// Checks whether a shuffle mask is equivalent to an explicit list of
/// arguments.
///
/// This is a fast way to test a shuffle mask against a fixed pattern:
///
///   if (isShuffleEquivalent(Mask, 3, 2, {1, 0})) { ... }
///
/// It returns true if the mask is exactly as wide as the argument list, and
/// each element of the mask is either -1 (signifying undef) or the value given
/// in the argument.
static bool isShuffleEquivalent(SDValue V1, SDValue V2, ArrayRef<int> Mask,
                                ArrayRef<int> ExpectedMask) {
  if (Mask.size() != ExpectedMask.size())
    return false;

  int Size = Mask.size();

  // If the values are build vectors, we can look through them to find
  // equivalent inputs that make the shuffles equivalent.
  auto *BV1 = dyn_cast<BuildVectorSDNode>(V1);
  auto *BV2 = dyn_cast<BuildVectorSDNode>(V2);

  for (int i = 0; i < Size; ++i) {
    assert(Mask[i] >= -1 && "Out of bound mask element!");
    if (Mask[i] >= 0 && Mask[i] != ExpectedMask[i]) {
      auto *MaskBV = Mask[i] < Size ? BV1 : BV2;
      auto *ExpectedBV = ExpectedMask[i] < Size ? BV1 : BV2;
      if (!MaskBV || !ExpectedBV ||
          MaskBV->getOperand(Mask[i] % Size) !=
              ExpectedBV->getOperand(ExpectedMask[i] % Size))
        return false;
    }
  }

  return true;
}

/// Checks whether a target shuffle mask is equivalent to an explicit pattern.
///
/// The masks must be exactly the same width.
///
/// If an element in Mask matches SM_SentinelUndef (-1) then the corresponding
/// value in ExpectedMask is always accepted. Otherwise the indices must match.
///
/// SM_SentinelZero is accepted as a valid negative index but must match in
/// both.
static bool isTargetShuffleEquivalent(ArrayRef<int> Mask,
                                      ArrayRef<int> ExpectedMask,
                                      SDValue V1 = SDValue(),
                                      SDValue V2 = SDValue()) {
  int Size = Mask.size();
  if (Size != (int)ExpectedMask.size())
    return false;
  assert(isUndefOrZeroOrInRange(ExpectedMask, 0, 2 * Size) &&
         "Illegal target shuffle mask");

  // Check for out-of-range target shuffle mask indices.
  if (!isUndefOrZeroOrInRange(Mask, 0, 2 * Size))
    return false;

  // If the values are build vectors, we can look through them to find
  // equivalent inputs that make the shuffles equivalent.
  auto *BV1 = dyn_cast_or_null<BuildVectorSDNode>(V1);
  auto *BV2 = dyn_cast_or_null<BuildVectorSDNode>(V2);
  BV1 = ((BV1 && Size != (int)BV1->getNumOperands()) ? nullptr : BV1);
  BV2 = ((BV2 && Size != (int)BV2->getNumOperands()) ? nullptr : BV2);

  for (int i = 0; i < Size; ++i) {
    if (Mask[i] == SM_SentinelUndef || Mask[i] == ExpectedMask[i])
      continue;
    if (0 <= Mask[i] && 0 <= ExpectedMask[i]) {
      auto *MaskBV = Mask[i] < Size ? BV1 : BV2;
      auto *ExpectedBV = ExpectedMask[i] < Size ? BV1 : BV2;
      if (MaskBV && ExpectedBV &&
          MaskBV->getOperand(Mask[i] % Size) ==
              ExpectedBV->getOperand(ExpectedMask[i] % Size))
        continue;
    }
    // TODO - handle SM_Sentinel equivalences.
    return false;
  }
  return true;
}

// Merges a general DAG shuffle mask and zeroable bit mask into a target shuffle
// mask.
static SmallVector<int, 64> createTargetShuffleMask(ArrayRef<int> Mask,
                                                    const APInt &Zeroable) {
  int NumElts = Mask.size();
  assert(NumElts == (int)Zeroable.getBitWidth() && "Mismatch mask sizes");

  SmallVector<int, 64> TargetMask(NumElts, SM_SentinelUndef);
  for (int i = 0; i != NumElts; ++i) {
    int M = Mask[i];
    if (M == SM_SentinelUndef)
      continue;
    assert(0 <= M && M < (2 * NumElts) && "Out of range shuffle index");
    TargetMask[i] = (Zeroable[i] ? SM_SentinelZero : M);
  }
  return TargetMask;
}

// Attempt to create a shuffle mask from a VSELECT condition mask.
bool X86::createShuffleMaskFromVSELECT(SmallVectorImpl<int> &Mask,
                                       SDValue Cond) {
  if (!ISD::isBuildVectorOfConstantSDNodes(Cond.getNode()))
    return false;

  unsigned Size = Cond.getValueType().getVectorNumElements();
  Mask.resize(Size, SM_SentinelUndef);

  for (int i = 0; i != (int)Size; ++i) {
    SDValue CondElt = Cond.getOperand(i);
    Mask[i] = i;
    // Arbitrarily choose from the 2nd operand if the select condition element
    // is undef.
    // TODO: Can we do better by matching patterns such as even/odd?
    if (CondElt.isUndef() || isNullConstant(CondElt))
      Mask[i] += Size;
  }

  return true;
}

// Check if the shuffle mask is suitable for the AVX vpunpcklwd or vpunpckhwd
// instructions.
static bool isUnpackWdShuffleMask(ArrayRef<int> Mask, MVT VT) {
  if (VT != MVT::v8i32 && VT != MVT::v8f32)
    return false;

  SmallVector<int, 8> Unpcklwd;
  createUnpackShuffleMask(MVT::v8i16, Unpcklwd, /* Lo = */ true,
                          /* Unary = */ false);
  SmallVector<int, 8> Unpckhwd;
  createUnpackShuffleMask(MVT::v8i16, Unpckhwd, /* Lo = */ false,
                          /* Unary = */ false);
  bool IsUnpackwdMask = (isTargetShuffleEquivalent(Mask, Unpcklwd) ||
                         isTargetShuffleEquivalent(Mask, Unpckhwd));
  return IsUnpackwdMask;
}

static bool is128BitUnpackShuffleMask(ArrayRef<int> Mask) {
  // Create 128-bit vector type based on mask size.
  MVT EltVT = MVT::getIntegerVT(128 / Mask.size());
  MVT VT = MVT::getVectorVT(EltVT, Mask.size());

  // We can't assume a canonical shuffle mask, so try the commuted version too.
  SmallVector<int, 4> CommutedMask(Mask.begin(), Mask.end());
  ShuffleVectorSDNode::commuteMask(CommutedMask);

  // Match any of unary/binary or low/high.
  for (unsigned i = 0; i != 4; ++i) {
    SmallVector<int, 16> UnpackMask;
    createUnpackShuffleMask(VT, UnpackMask, (i >> 1) % 2, i % 2);
    if (isTargetShuffleEquivalent(Mask, UnpackMask) ||
        isTargetShuffleEquivalent(CommutedMask, UnpackMask))
      return true;
  }
  return false;
}

/// Return true if a shuffle mask chooses elements identically in its top and
/// bottom halves. For example, any splat mask has the same top and bottom
/// halves. If an element is undefined in only one half of the mask, the halves
/// are not considered identical.
bool X86::hasIdenticalHalvesShuffleMask(ArrayRef<int> Mask) {
  assert(Mask.size() % 2 == 0 && "Expecting even number of elements in mask");
  unsigned HalfSize = Mask.size() / 2;
  for (unsigned i = 0; i != HalfSize; ++i) {
    if (Mask[i] != Mask[i + HalfSize])
      return false;
  }
  return true;
}

/// Get a 4-lane 8-bit shuffle immediate for a mask.
///
/// This helper function produces an 8-bit shuffle immediate corresponding to
/// the ubiquitous shuffle encoding scheme used in x86 instructions for
/// shuffling 4 lanes. It can be used with most of the PSHUF instructions for
/// example.
///
/// NB: We rely heavily on "undef" masks preserving the input lane.
static unsigned getV4X86ShuffleImm(ArrayRef<int> Mask) {
  assert(Mask.size() == 4 && "Only 4-lane shuffle masks");
  assert(Mask[0] >= -1 && Mask[0] < 4 && "Out of bound mask element!");
  assert(Mask[1] >= -1 && Mask[1] < 4 && "Out of bound mask element!");
  assert(Mask[2] >= -1 && Mask[2] < 4 && "Out of bound mask element!");
  assert(Mask[3] >= -1 && Mask[3] < 4 && "Out of bound mask element!");

  unsigned Imm = 0;
  Imm |= (Mask[0] < 0 ? 0 : Mask[0]) << 0;
  Imm |= (Mask[1] < 0 ? 1 : Mask[1]) << 2;
  Imm |= (Mask[2] < 0 ? 2 : Mask[2]) << 4;
  Imm |= (Mask[3] < 0 ? 3 : Mask[3]) << 6;
  return Imm;
}

static SDValue getV4X86ShuffleImm8ForMask(ArrayRef<int> Mask, const SDLoc &DL,
                                          SelectionDAG &DAG) {
  return DAG.getConstant(getV4X86ShuffleImm(Mask), DL, MVT::i8);
}

/// Compute whether each element of a shuffle is zeroable.
///
/// A "zeroable" vector shuffle element is one which can be lowered to zero.
/// Either it is an undef element in the shuffle mask, the element of the input
/// referenced is undef, or the element of the input referenced is known to be
/// zero. Many x86 shuffles can zero lanes cheaply and we often want to handle
/// as many lanes with this technique as possible to simplify the remaining
/// shuffle.
static APInt computeZeroableShuffleElements(ArrayRef<int> Mask,
                                            SDValue V1, SDValue V2) {
  APInt Zeroable(Mask.size(), 0);
  V1 = peekThroughBitcasts(V1);
  V2 = peekThroughBitcasts(V2);

  bool V1IsZero = ISD::isBuildVectorAllZeros(V1.getNode());
  bool V2IsZero = ISD::isBuildVectorAllZeros(V2.getNode());

  int VectorSizeInBits = V1.getValueSizeInBits();
  int ScalarSizeInBits = VectorSizeInBits / Mask.size();
  assert(!(VectorSizeInBits % ScalarSizeInBits) && "Illegal shuffle mask size");

  for (int i = 0, Size = Mask.size(); i < Size; ++i) {
    int M = Mask[i];
    // Handle the easy cases.
    if (M < 0 || (M >= 0 && M < Size && V1IsZero) || (M >= Size && V2IsZero)) {
      Zeroable.setBit(i);
      continue;
    }

    // Determine shuffle input and normalize the mask.
    SDValue V = M < Size ? V1 : V2;
    M %= Size;

    // Currently we can only search BUILD_VECTOR for UNDEF/ZERO elements.
    if (V.getOpcode() != ISD::BUILD_VECTOR)
      continue;

    // If the BUILD_VECTOR has fewer elements then the bitcasted portion of
    // the (larger) source element must be UNDEF/ZERO.
    if ((Size % V.getNumOperands()) == 0) {
      int Scale = Size / V->getNumOperands();
      SDValue Op = V.getOperand(M / Scale);
      if (Op.isUndef() || X86::isZeroNode(Op))
        Zeroable.setBit(i);
      else if (ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(Op)) {
        APInt Val = Cst->getAPIntValue();
        Val.lshrInPlace((M % Scale) * ScalarSizeInBits);
        Val = Val.getLoBits(ScalarSizeInBits);
        if (Val == 0)
          Zeroable.setBit(i);
      } else if (ConstantFPSDNode *Cst = dyn_cast<ConstantFPSDNode>(Op)) {
        APInt Val = Cst->getValueAPF().bitcastToAPInt();
        Val.lshrInPlace((M % Scale) * ScalarSizeInBits);
        Val = Val.getLoBits(ScalarSizeInBits);
        if (Val == 0)
          Zeroable.setBit(i);
      }
      continue;
    }

    // If the BUILD_VECTOR has more elements then all the (smaller) source
    // elements must be UNDEF or ZERO.
    if ((V.getNumOperands() % Size) == 0) {
      int Scale = V->getNumOperands() / Size;
      bool AllZeroable = true;
      for (int j = 0; j < Scale; ++j) {
        SDValue Op = V.getOperand((M * Scale) + j);
        AllZeroable &= (Op.isUndef() || X86::isZeroNode(Op));
      }
      if (AllZeroable)
        Zeroable.setBit(i);
      continue;
    }
  }

  return Zeroable;
}

// The Shuffle result is as follow:
// 0*a[0]0*a[1]...0*a[n] , n >=0 where a[] elements in a ascending order.
// Each Zeroable's element correspond to a particular Mask's element.
// As described in computeZeroableShuffleElements function.
//
// The function looks for a sub-mask that the nonzero elements are in
// increasing order. If such sub-mask exist. The function returns true.
static bool isNonZeroElementsInOrder(const APInt &Zeroable,
                                     ArrayRef<int> Mask, const EVT &VectorType,
                                     bool &IsZeroSideLeft) {
  int NextElement = -1;
  // Check if the Mask's nonzero elements are in increasing order.
  for (int i = 0, e = Mask.size(); i < e; i++) {
    // Checks if the mask's zeros elements are built from only zeros.
    assert(Mask[i] >= -1 && "Out of bound mask element!");
    if (Mask[i] < 0)
      return false;
    if (Zeroable[i])
      continue;
    // Find the lowest non zero element
    if (NextElement < 0) {
      NextElement = Mask[i] != 0 ? VectorType.getVectorNumElements() : 0;
      IsZeroSideLeft = NextElement != 0;
    }
    // Exit if the mask's non zero elements are not in increasing order.
    if (NextElement != Mask[i])
      return false;
    NextElement++;
  }
  return true;
}

/// Try to lower a shuffle with a single PSHUFB of V1 or V2.
static SDValue lowerShuffleWithPSHUFB(const SDLoc &DL, MVT VT,
                                      ArrayRef<int> Mask, SDValue V1,
                                      SDValue V2, const APInt &Zeroable,
                                      const X86Subtarget &Subtarget,
                                      SelectionDAG &DAG) {
  int Size = Mask.size();
  int LaneSize = 128 / VT.getScalarSizeInBits();
  const int NumBytes = VT.getSizeInBits() / 8;
  const int NumEltBytes = VT.getScalarSizeInBits() / 8;

  assert((Subtarget.hasSSSE3() && VT.is128BitVector()) ||
         (Subtarget.hasAVX2() && VT.is256BitVector()) ||
         (Subtarget.hasBWI() && VT.is512BitVector()));

  SmallVector<SDValue, 64> PSHUFBMask(NumBytes);
  // Sign bit set in i8 mask means zero element.
  SDValue ZeroMask = DAG.getConstant(0x80, DL, MVT::i8);

  SDValue V;
  for (int i = 0; i < NumBytes; ++i) {
    int M = Mask[i / NumEltBytes];
    if (M < 0) {
      PSHUFBMask[i] = DAG.getUNDEF(MVT::i8);
      continue;
    }
    if (Zeroable[i / NumEltBytes]) {
      PSHUFBMask[i] = ZeroMask;
      continue;
    }

    // We can only use a single input of V1 or V2.
    SDValue SrcV = (M >= Size ? V2 : V1);
    if (V && V != SrcV)
      return SDValue();
    V = SrcV;
    M %= Size;

    // PSHUFB can't cross lanes, ensure this doesn't happen.
    if ((M / LaneSize) != ((i / NumEltBytes) / LaneSize))
      return SDValue();

    M = M % LaneSize;
    M = M * NumEltBytes + (i % NumEltBytes);
    PSHUFBMask[i] = DAG.getConstant(M, DL, MVT::i8);
  }
  assert(V && "Failed to find a source input");

  MVT I8VT = MVT::getVectorVT(MVT::i8, NumBytes);
  return DAG.getBitcast(
      VT, DAG.getNode(X86ISD::PSHUFB, DL, I8VT, DAG.getBitcast(I8VT, V),
                      DAG.getBuildVector(I8VT, DL, PSHUFBMask)));
}

// X86 has dedicated shuffle that can be lowered to VEXPAND
static SDValue lowerShuffleToEXPAND(const SDLoc &DL, MVT VT,
                                    const APInt &Zeroable,
                                    ArrayRef<int> Mask, SDValue &V1,
                                    SDValue &V2, SelectionDAG &DAG,
                                    const X86Subtarget &Subtarget) {
  bool IsLeftZeroSide = true;
  if (!isNonZeroElementsInOrder(Zeroable, Mask, V1.getValueType(),
                                IsLeftZeroSide))
    return SDValue();
  unsigned VEXPANDMask = (~Zeroable).getZExtValue();
  MVT IntegerType =
      MVT::getIntegerVT(std::max((int)VT.getVectorNumElements(), 8));
  SDValue MaskNode = DAG.getConstant(VEXPANDMask, DL, IntegerType);
  unsigned NumElts = VT.getVectorNumElements();
  assert((NumElts == 4 || NumElts == 8 || NumElts == 16) &&
         "Unexpected number of vector elements");
  SDValue VMask = X86::getMaskNode(MaskNode, MVT::getVectorVT(MVT::i1, NumElts),
                                   Subtarget, DAG, DL);
  SDValue ZeroVector = getZeroVector(VT, Subtarget, DAG, DL);
  SDValue ExpandedVector = IsLeftZeroSide ? V2 : V1;
  return DAG.getNode(X86ISD::EXPAND, DL, VT, ExpandedVector, ZeroVector, VMask);
}

static bool matchVectorShuffleWithUNPCK(MVT VT, SDValue &V1, SDValue &V2,
                                        unsigned &UnpackOpcode, bool IsUnary,
                                        ArrayRef<int> TargetMask,
                                        const SDLoc &DL, SelectionDAG &DAG,
                                        const X86Subtarget &Subtarget) {
  int NumElts = VT.getVectorNumElements();

  bool Undef1 = true, Undef2 = true, Zero1 = true, Zero2 = true;
  for (int i = 0; i != NumElts; i += 2) {
    int M1 = TargetMask[i + 0];
    int M2 = TargetMask[i + 1];
    Undef1 &= (SM_SentinelUndef == M1);
    Undef2 &= (SM_SentinelUndef == M2);
    Zero1 &= isUndefOrZero(M1);
    Zero2 &= isUndefOrZero(M2);
  }
  assert(!((Undef1 || Zero1) && (Undef2 || Zero2)) &&
         "Zeroable shuffle detected");

  // Attempt to match the target mask against the unpack lo/hi mask patterns.
  SmallVector<int, 64> Unpckl, Unpckh;
  createUnpackShuffleMask(VT, Unpckl, /* Lo = */ true, IsUnary);
  if (isTargetShuffleEquivalent(TargetMask, Unpckl)) {
    UnpackOpcode = X86ISD::UNPCKL;
    V2 = (Undef2 ? DAG.getUNDEF(VT) : (IsUnary ? V1 : V2));
    V1 = (Undef1 ? DAG.getUNDEF(VT) : V1);
    return true;
  }

  createUnpackShuffleMask(VT, Unpckh, /* Lo = */ false, IsUnary);
  if (isTargetShuffleEquivalent(TargetMask, Unpckh)) {
    UnpackOpcode = X86ISD::UNPCKH;
    V2 = (Undef2 ? DAG.getUNDEF(VT) : (IsUnary ? V1 : V2));
    V1 = (Undef1 ? DAG.getUNDEF(VT) : V1);
    return true;
  }

  // If an unary shuffle, attempt to match as an unpack lo/hi with zero.
  if (IsUnary && (Zero1 || Zero2)) {
    // Don't bother if we can blend instead.
    if ((Subtarget.hasSSE41() || VT == MVT::v2i64 || VT == MVT::v2f64) &&
        isSequentialOrUndefOrZeroInRange(TargetMask, 0, NumElts, 0))
      return false;

    bool MatchLo = true, MatchHi = true;
    for (int i = 0; (i != NumElts) && (MatchLo || MatchHi); ++i) {
      int M = TargetMask[i];

      // Ignore if the input is known to be zero or the index is undef.
      if ((((i & 1) == 0) && Zero1) || (((i & 1) == 1) && Zero2) ||
          (M == SM_SentinelUndef))
        continue;

      MatchLo &= (M == Unpckl[i]);
      MatchHi &= (M == Unpckh[i]);
    }

    if (MatchLo || MatchHi) {
      UnpackOpcode = MatchLo ? X86ISD::UNPCKL : X86ISD::UNPCKH;
      V2 = Zero2 ? getZeroVector(VT, Subtarget, DAG, DL) : V1;
      V1 = Zero1 ? getZeroVector(VT, Subtarget, DAG, DL) : V1;
      return true;
    }
  }

  // If a binary shuffle, commute and try again.
  if (!IsUnary) {
    ShuffleVectorSDNode::commuteMask(Unpckl);
    if (isTargetShuffleEquivalent(TargetMask, Unpckl)) {
      UnpackOpcode = X86ISD::UNPCKL;
      std::swap(V1, V2);
      return true;
    }

    ShuffleVectorSDNode::commuteMask(Unpckh);
    if (isTargetShuffleEquivalent(TargetMask, Unpckh)) {
      UnpackOpcode = X86ISD::UNPCKH;
      std::swap(V1, V2);
      return true;
    }
  }

  return false;
}

// X86 has dedicated unpack instructions that can handle specific blend
// operations: UNPCKH and UNPCKL.
static SDValue lowerShuffleWithUNPCK(const SDLoc &DL, MVT VT,
                                     ArrayRef<int> Mask, SDValue V1, SDValue V2,
                                     SelectionDAG &DAG) {
  SmallVector<int, 8> Unpckl;
  createUnpackShuffleMask(VT, Unpckl, /* Lo = */ true, /* Unary = */ false);
  if (isShuffleEquivalent(V1, V2, Mask, Unpckl))
    return DAG.getNode(X86ISD::UNPCKL, DL, VT, V1, V2);

  SmallVector<int, 8> Unpckh;
  createUnpackShuffleMask(VT, Unpckh, /* Lo = */ false, /* Unary = */ false);
  if (isShuffleEquivalent(V1, V2, Mask, Unpckh))
    return DAG.getNode(X86ISD::UNPCKH, DL, VT, V1, V2);

  // Commute and try again.
  ShuffleVectorSDNode::commuteMask(Unpckl);
  if (isShuffleEquivalent(V1, V2, Mask, Unpckl))
    return DAG.getNode(X86ISD::UNPCKL, DL, VT, V2, V1);

  ShuffleVectorSDNode::commuteMask(Unpckh);
  if (isShuffleEquivalent(V1, V2, Mask, Unpckh))
    return DAG.getNode(X86ISD::UNPCKH, DL, VT, V2, V1);

  return SDValue();
}

static bool matchVectorShuffleAsVPMOV(ArrayRef<int> Mask, bool SwappedOps,
                                         int Delta) {
  int Size = (int)Mask.size();
  int Split = Size / Delta;
  int TruncatedVectorStart = SwappedOps ? Size : 0;

  // Match for mask starting with e.g.: <8, 10, 12, 14,... or <0, 2, 4, 6,...
  if (!isSequentialOrUndefInRange(Mask, 0, Split, TruncatedVectorStart, Delta))
    return false;

  // The rest of the mask should not refer to the truncated vector's elements.
  if (isAnyInRange(Mask.slice(Split, Size - Split), TruncatedVectorStart,
                   TruncatedVectorStart + Size))
    return false;

  return true;
}

// Try to lower trunc+vector_shuffle to a vpmovdb or a vpmovdw instruction.
//
// An example is the following:
//
// t0: ch = EntryToken
//           t2: v4i64,ch = CopyFromReg t0, Register:v4i64 %0
//         t25: v4i32 = truncate t2
//       t41: v8i16 = bitcast t25
//       t21: v8i16 = BUILD_VECTOR undef:i16, undef:i16, undef:i16, undef:i16,
//       Constant:i16<0>, Constant:i16<0>, Constant:i16<0>, Constant:i16<0>
//     t51: v8i16 = vector_shuffle<0,2,4,6,12,13,14,15> t41, t21
//   t18: v2i64 = bitcast t51
//
// Without avx512vl, this is lowered to:
//
// vpmovqd %zmm0, %ymm0
// vpshufb {{.*#+}} xmm0 =
// xmm0[0,1,4,5,8,9,12,13],zero,zero,zero,zero,zero,zero,zero,zero
//
// But when avx512vl is available, one can just use a single vpmovdw
// instruction.
static SDValue lowerShuffleWithVPMOV(const SDLoc &DL, ArrayRef<int> Mask,
                                     MVT VT, SDValue V1, SDValue V2,
                                     SelectionDAG &DAG,
                                     const X86Subtarget &Subtarget) {
  if (VT != MVT::v16i8 && VT != MVT::v8i16)
    return SDValue();

  if (Mask.size() != VT.getVectorNumElements())
    return SDValue();

  bool SwappedOps = false;

  if (!ISD::isBuildVectorAllZeros(V2.getNode())) {
    if (!ISD::isBuildVectorAllZeros(V1.getNode()))
      return SDValue();

    std::swap(V1, V2);
    SwappedOps = true;
  }

  // Look for:
  //
  // bitcast (truncate <8 x i32> %vec to <8 x i16>) to <16 x i8>
  // bitcast (truncate <4 x i64> %vec to <4 x i32>) to <8 x i16>
  //
  // and similar ones.
  if (V1.getOpcode() != ISD::BITCAST)
    return SDValue();
  if (V1.getOperand(0).getOpcode() != ISD::TRUNCATE)
    return SDValue();

  SDValue Src = V1.getOperand(0).getOperand(0);
  MVT SrcVT = Src.getSimpleValueType();

  // The vptrunc** instructions truncating 128 bit and 256 bit vectors
  // are only available with avx512vl.
  if (!SrcVT.is512BitVector() && !Subtarget.hasVLX())
    return SDValue();

  // Down Convert Word to Byte is only available with avx512bw. The case with
  // 256-bit output doesn't contain a shuffle and is therefore not handled here.
  if (SrcVT.getVectorElementType() == MVT::i16 && VT == MVT::v16i8 &&
      !Subtarget.hasBWI())
    return SDValue();

  // The first half/quarter of the mask should refer to every second/fourth
  // element of the vector truncated and bitcasted.
  if (!matchVectorShuffleAsVPMOV(Mask, SwappedOps, 2) &&
      !matchVectorShuffleAsVPMOV(Mask, SwappedOps, 4))
    return SDValue();

  return DAG.getNode(X86ISD::VTRUNC, DL, VT, Src);
}

// X86 has dedicated pack instructions that can handle specific truncation
// operations: PACKSS and PACKUS.
static bool matchVectorShuffleWithPACK(MVT VT, MVT &SrcVT, SDValue &V1,
                                       SDValue &V2, unsigned &PackOpcode,
                                       ArrayRef<int> TargetMask,
                                       SelectionDAG &DAG,
                                       const X86Subtarget &Subtarget) {
  unsigned NumElts = VT.getVectorNumElements();
  unsigned BitSize = VT.getScalarSizeInBits();
  MVT PackSVT = MVT::getIntegerVT(BitSize * 2);
  MVT PackVT = MVT::getVectorVT(PackSVT, NumElts / 2);

  auto MatchPACK = [&](SDValue N1, SDValue N2) {
    SDValue VV1 = DAG.getBitcast(PackVT, N1);
    SDValue VV2 = DAG.getBitcast(PackVT, N2);
    if (Subtarget.hasSSE41() || PackSVT == MVT::i16) {
      APInt ZeroMask = APInt::getHighBitsSet(BitSize * 2, BitSize);
      if ((N1.isUndef() || DAG.MaskedValueIsZero(VV1, ZeroMask)) &&
          (N2.isUndef() || DAG.MaskedValueIsZero(VV2, ZeroMask))) {
        V1 = VV1;
        V2 = VV2;
        SrcVT = PackVT;
        PackOpcode = X86ISD::PACKUS;
        return true;
      }
    }
    if ((N1.isUndef() || DAG.ComputeNumSignBits(VV1) > BitSize) &&
        (N2.isUndef() || DAG.ComputeNumSignBits(VV2) > BitSize)) {
      V1 = VV1;
      V2 = VV2;
      SrcVT = PackVT;
      PackOpcode = X86ISD::PACKSS;
      return true;
    }
    return false;
  };

  // Try binary shuffle.
  SmallVector<int, 32> BinaryMask;
  createPackShuffleMask(VT, BinaryMask, false);
  if (isTargetShuffleEquivalent(TargetMask, BinaryMask, V1, V2))
    if (MatchPACK(V1, V2))
      return true;

  // Try unary shuffle.
  SmallVector<int, 32> UnaryMask;
  createPackShuffleMask(VT, UnaryMask, true);
  if (isTargetShuffleEquivalent(TargetMask, UnaryMask, V1))
    if (MatchPACK(V1, V1))
      return true;

  return false;
}

static SDValue lowerShuffleWithPACK(const SDLoc &DL, MVT VT, ArrayRef<int> Mask,
                                    SDValue V1, SDValue V2, SelectionDAG &DAG,
                                    const X86Subtarget &Subtarget) {
  MVT PackVT;
  unsigned PackOpcode;
  if (matchVectorShuffleWithPACK(VT, PackVT, V1, V2, PackOpcode, Mask, DAG,
                                 Subtarget))
    return DAG.getNode(PackOpcode, DL, VT, DAG.getBitcast(PackVT, V1),
                       DAG.getBitcast(PackVT, V2));

  return SDValue();
}

/// Try to emit a bitmask instruction for a shuffle.
///
/// This handles cases where we can model a blend exactly as a bitmask due to
/// one of the inputs being zeroable.
static SDValue lowerShuffleAsBitMask(const SDLoc &DL, MVT VT, SDValue V1,
                                     SDValue V2, ArrayRef<int> Mask,
                                     const APInt &Zeroable,
                                     const X86Subtarget &Subtarget,
                                     SelectionDAG &DAG) {
  MVT MaskVT = VT;
  MVT EltVT = VT.getVectorElementType();
  SDValue Zero, AllOnes;
  // Use f64 if i64 isn't legal.
  if (EltVT == MVT::i64 && !Subtarget.is64Bit()) {
    EltVT = MVT::f64;
    MaskVT = MVT::getVectorVT(EltVT, Mask.size());
  }

  MVT LogicVT = VT;
  if (EltVT == MVT::f32 || EltVT == MVT::f64) {
    Zero = DAG.getConstantFP(0.0, DL, EltVT);
    AllOnes = DAG.getConstantFP(
        APFloat::getAllOnesValue(EltVT.getSizeInBits(), true), DL, EltVT);
    LogicVT =
        MVT::getVectorVT(EltVT == MVT::f64 ? MVT::i64 : MVT::i32, Mask.size());
  } else {
    Zero = DAG.getConstant(0, DL, EltVT);
    AllOnes = DAG.getAllOnesConstant(DL, EltVT);
  }

  SmallVector<SDValue, 16> VMaskOps(Mask.size(), Zero);
  SDValue V;
  for (int i = 0, Size = Mask.size(); i < Size; ++i) {
    if (Zeroable[i])
      continue;
    if (Mask[i] % Size != i)
      return SDValue(); // Not a blend.
    if (!V)
      V = Mask[i] < Size ? V1 : V2;
    else if (V != (Mask[i] < Size ? V1 : V2))
      return SDValue(); // Can only let one input through the mask.

    VMaskOps[i] = AllOnes;
  }
  if (!V)
    return SDValue(); // No non-zeroable elements!

  SDValue VMask = DAG.getBuildVector(MaskVT, DL, VMaskOps);
  VMask = DAG.getBitcast(LogicVT, VMask);
  V = DAG.getBitcast(LogicVT, V);
  SDValue And = DAG.getNode(ISD::AND, DL, LogicVT, V, VMask);
  return DAG.getBitcast(VT, And);
}

/// Try to emit a blend instruction for a shuffle using bit math.
///
/// This is used as a fallback approach when first class blend instructions are
/// unavailable. Currently it is only suitable for integer vectors, but could
/// be generalized for floating point vectors if desirable.
static SDValue lowerShuffleAsBitBlend(const SDLoc &DL, MVT VT, SDValue V1,
                                      SDValue V2, ArrayRef<int> Mask,
                                      SelectionDAG &DAG) {
  assert(VT.isInteger() && "Only supports integer vector types!");
  MVT EltVT = VT.getVectorElementType();
  SDValue Zero = DAG.getConstant(0, DL, EltVT);
  SDValue AllOnes = DAG.getAllOnesConstant(DL, EltVT);
  SmallVector<SDValue, 16> MaskOps;
  for (int i = 0, Size = Mask.size(); i < Size; ++i) {
    if (Mask[i] >= 0 && Mask[i] != i && Mask[i] != i + Size)
      return SDValue(); // Shuffled input!
    MaskOps.push_back(Mask[i] < Size ? AllOnes : Zero);
  }

  SDValue V1Mask = DAG.getBuildVector(VT, DL, MaskOps);
  V1 = DAG.getNode(ISD::AND, DL, VT, V1, V1Mask);
  V2 = DAG.getNode(X86ISD::ANDNP, DL, VT, V1Mask, V2);
  return DAG.getNode(ISD::OR, DL, VT, V1, V2);
}

static bool matchVectorShuffleAsBlend(SDValue V1, SDValue V2,
                                      MutableArrayRef<int> TargetMask,
                                      bool &ForceV1Zero, bool &ForceV2Zero,
                                      uint64_t &BlendMask) {
  bool V1IsZeroOrUndef =
      V1.isUndef() || ISD::isBuildVectorAllZeros(V1.getNode());
  bool V2IsZeroOrUndef =
      V2.isUndef() || ISD::isBuildVectorAllZeros(V2.getNode());

  BlendMask = 0;
  ForceV1Zero = false, ForceV2Zero = false;
  assert(TargetMask.size() <= 64 && "Shuffle mask too big for blend mask");

  // Attempt to generate the binary blend mask. If an input is zero then
  // we can use any lane.
  // TODO: generalize the zero matching to any scalar like isShuffleEquivalent.
  for (int i = 0, Size = TargetMask.size(); i < Size; ++i) {
    int M = TargetMask[i];
    if (M == SM_SentinelUndef)
      continue;
    if (M == i)
      continue;
    if (M == i + Size) {
      BlendMask |= 1ull << i;
      continue;
    }
    if (M == SM_SentinelZero) {
      if (V1IsZeroOrUndef) {
        ForceV1Zero = true;
        TargetMask[i] = i;
        continue;
      }
      if (V2IsZeroOrUndef) {
        ForceV2Zero = true;
        BlendMask |= 1ull << i;
        TargetMask[i] = i + Size;
        continue;
      }
    }
    return false;
  }
  return true;
}

static uint64_t scaleVectorShuffleBlendMask(uint64_t BlendMask, int Size,
                                            int Scale) {
  uint64_t ScaledMask = 0;
  for (int i = 0; i != Size; ++i)
    if (BlendMask & (1ull << i))
      ScaledMask |= ((1ull << Scale) - 1) << (i * Scale);
  return ScaledMask;
}

/// Try to emit a blend instruction for a shuffle.
///
/// This doesn't do any checks for the availability of instructions for blending
/// these values. It relies on the availability of the X86ISD::BLENDI pattern to
/// be matched in the backend with the type given. What it does check for is
/// that the shuffle mask is a blend, or convertible into a blend with zero.
static SDValue lowerShuffleAsBlend(const SDLoc &DL, MVT VT, SDValue V1,
                                   SDValue V2, ArrayRef<int> Original,
                                   const APInt &Zeroable,
                                   const X86Subtarget &Subtarget,
                                   SelectionDAG &DAG) {
  SmallVector<int, 64> Mask = createTargetShuffleMask(Original, Zeroable);

  uint64_t BlendMask = 0;
  bool ForceV1Zero = false, ForceV2Zero = false;
  if (!matchVectorShuffleAsBlend(V1, V2, Mask, ForceV1Zero, ForceV2Zero,
                                 BlendMask))
    return SDValue();

  // Create a REAL zero vector - ISD::isBuildVectorAllZeros allows UNDEFs.
  if (ForceV1Zero)
    V1 = getZeroVector(VT, Subtarget, DAG, DL);
  if (ForceV2Zero)
    V2 = getZeroVector(VT, Subtarget, DAG, DL);

  switch (VT.SimpleTy) {
  case MVT::v4i64:
  case MVT::v8i32:
    assert(Subtarget.hasAVX2() && "256-bit integer blends require AVX2!");
    LLVM_FALLTHROUGH;
  case MVT::v4f64:
  case MVT::v8f32:
    assert(Subtarget.hasAVX() && "256-bit float blends require AVX!");
    LLVM_FALLTHROUGH;
  case MVT::v2f64:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v4i32:
  case MVT::v8i16:
    assert(Subtarget.hasSSE41() && "128-bit blends require SSE41!");
    return DAG.getNode(X86ISD::BLENDI, DL, VT, V1, V2,
                       DAG.getConstant(BlendMask, DL, MVT::i8));
  case MVT::v16i16: {
    assert(Subtarget.hasAVX2() && "v16i16 blends require AVX2!");
    SmallVector<int, 8> RepeatedMask;
    if (is128BitLaneRepeatedShuffleMask(MVT::v16i16, Mask, RepeatedMask)) {
      // We can lower these with PBLENDW which is mirrored across 128-bit lanes.
      assert(RepeatedMask.size() == 8 && "Repeated mask size doesn't match!");
      BlendMask = 0;
      for (int i = 0; i < 8; ++i)
        if (RepeatedMask[i] >= 8)
          BlendMask |= 1ull << i;
      return DAG.getNode(X86ISD::BLENDI, DL, MVT::v16i16, V1, V2,
                         DAG.getConstant(BlendMask, DL, MVT::i8));
    }
    // Use PBLENDW for lower/upper lanes and then blend lanes.
    // TODO - we should allow 2 PBLENDW here and leave shuffle combine to
    // merge to VSELECT where useful.
    uint64_t LoMask = BlendMask & 0xFF;
    uint64_t HiMask = (BlendMask >> 8) & 0xFF;
    if (LoMask == 0 || LoMask == 255 || HiMask == 0 || HiMask == 255) {
      SDValue Lo = DAG.getNode(X86ISD::BLENDI, DL, MVT::v16i16, V1, V2,
                               DAG.getConstant(LoMask, DL, MVT::i8));
      SDValue Hi = DAG.getNode(X86ISD::BLENDI, DL, MVT::v16i16, V1, V2,
                               DAG.getConstant(HiMask, DL, MVT::i8));
      return DAG.getVectorShuffle(
          MVT::v16i16, DL, Lo, Hi,
          {0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31});
    }
    LLVM_FALLTHROUGH;
  }
  case MVT::v32i8:
    assert(Subtarget.hasAVX2() && "256-bit byte-blends require AVX2!");
    LLVM_FALLTHROUGH;
  case MVT::v16i8: {
    assert(Subtarget.hasSSE41() && "128-bit byte-blends require SSE41!");

    // Attempt to lower to a bitmask if we can. VPAND is faster than VPBLENDVB.
    if (SDValue Masked = lowerShuffleAsBitMask(DL, VT, V1, V2, Mask, Zeroable,
                                               Subtarget, DAG))
      return Masked;

    if (Subtarget.hasBWI() && Subtarget.hasVLX()) {
      MVT IntegerType =
          MVT::getIntegerVT(std::max((int)VT.getVectorNumElements(), 8));
      SDValue MaskNode = DAG.getConstant(BlendMask, DL, IntegerType);
      return getVectorMaskingNode(V2, MaskNode, V1, Subtarget, DAG);
    }

    // Scale the blend by the number of bytes per element.
    int Scale = VT.getScalarSizeInBits() / 8;

    // This form of blend is always done on bytes. Compute the byte vector
    // type.
    MVT BlendVT = MVT::getVectorVT(MVT::i8, VT.getSizeInBits() / 8);

    // x86 allows load folding with blendvb from the 2nd source operand. But
    // we are still using LLVM select here (see comment below), so that's V1.
    // If V2 can be load-folded and V1 cannot be load-folded, then commute to
    // allow that load-folding possibility.
    if (!ISD::isNormalLoad(V1.getNode()) && ISD::isNormalLoad(V2.getNode())) {
      ShuffleVectorSDNode::commuteMask(Mask);
      std::swap(V1, V2);
    }

    // Compute the VSELECT mask. Note that VSELECT is really confusing in the
    // mix of LLVM's code generator and the x86 backend. We tell the code
    // generator that boolean values in the elements of an x86 vector register
    // are -1 for true and 0 for false. We then use the LLVM semantics of 'true'
    // mapping a select to operand #1, and 'false' mapping to operand #2. The
    // reality in x86 is that vector masks (pre-AVX-512) use only the high bit
    // of the element (the remaining are ignored) and 0 in that high bit would
    // mean operand #1 while 1 in the high bit would mean operand #2. So while
    // the LLVM model for boolean values in vector elements gets the relevant
    // bit set, it is set backwards and over constrained relative to x86's
    // actual model.
    SmallVector<SDValue, 32> VSELECTMask;
    for (int i = 0, Size = Mask.size(); i < Size; ++i)
      for (int j = 0; j < Scale; ++j)
        VSELECTMask.push_back(
            Mask[i] < 0 ? DAG.getUNDEF(MVT::i8)
                        : DAG.getConstant(Mask[i] < Size ? -1 : 0, DL,
                                          MVT::i8));

    V1 = DAG.getBitcast(BlendVT, V1);
    V2 = DAG.getBitcast(BlendVT, V2);
    return DAG.getBitcast(
        VT,
        DAG.getSelect(DL, BlendVT, DAG.getBuildVector(BlendVT, DL, VSELECTMask),
                      V1, V2));
  }
  case MVT::v16f32:
  case MVT::v8f64:
  case MVT::v8i64:
  case MVT::v16i32:
  case MVT::v32i16:
  case MVT::v64i8: {
    // Attempt to lower to a bitmask if we can. Only if not optimizing for size.
    bool OptForSize = DAG.getMachineFunction().getFunction().hasOptSize();
    if (!OptForSize) {
      if (SDValue Masked = lowerShuffleAsBitMask(DL, VT, V1, V2, Mask, Zeroable,
                                                 Subtarget, DAG))
        return Masked;
    }

    // Otherwise load an immediate into a GPR, cast to k-register, and use a
    // masked move.
    MVT IntegerType =
        MVT::getIntegerVT(std::max((int)VT.getVectorNumElements(), 8));
    SDValue MaskNode = DAG.getConstant(BlendMask, DL, IntegerType);
    return getVectorMaskingNode(V2, MaskNode, V1, Subtarget, DAG);
  }
  default:
    llvm_unreachable("Not a supported integer vector type!");
  }
}

/// Try to lower as a blend of elements from two inputs followed by
/// a single-input permutation.
///
/// This matches the pattern where we can blend elements from two inputs and
/// then reduce the shuffle to a single-input permutation.
static SDValue lowerShuffleAsBlendAndPermute(const SDLoc &DL, MVT VT,
                                             SDValue V1, SDValue V2,
                                             ArrayRef<int> Mask,
                                             SelectionDAG &DAG,
                                             bool ImmBlends = false) {
  // We build up the blend mask while checking whether a blend is a viable way
  // to reduce the shuffle.
  SmallVector<int, 32> BlendMask(Mask.size(), -1);
  SmallVector<int, 32> PermuteMask(Mask.size(), -1);

  for (int i = 0, Size = Mask.size(); i < Size; ++i) {
    if (Mask[i] < 0)
      continue;

    assert(Mask[i] < Size * 2 && "Shuffle input is out of bounds.");

    if (BlendMask[Mask[i] % Size] < 0)
      BlendMask[Mask[i] % Size] = Mask[i];
    else if (BlendMask[Mask[i] % Size] != Mask[i])
      return SDValue(); // Can't blend in the needed input!

    PermuteMask[i] = Mask[i] % Size;
  }

  // If only immediate blends, then bail if the blend mask can't be widened to
  // i16.
  unsigned EltSize = VT.getScalarSizeInBits();
  if (ImmBlends && EltSize == 8 && !X86::canWidenShuffleElements(BlendMask))
    return SDValue();

  SDValue V = DAG.getVectorShuffle(VT, DL, V1, V2, BlendMask);
  return DAG.getVectorShuffle(VT, DL, V, DAG.getUNDEF(VT), PermuteMask);
}

/// Try to lower as an unpack of elements from two inputs followed by
/// a single-input permutation.
///
/// This matches the pattern where we can unpack elements from two inputs and
/// then reduce the shuffle to a single-input (wider) permutation.
static SDValue lowerShuffleAsUNPCKAndPermute(const SDLoc &DL, MVT VT,
                                             SDValue V1, SDValue V2,
                                             ArrayRef<int> Mask,
                                             SelectionDAG &DAG) {
  int NumElts = Mask.size();
  int NumLanes = VT.getSizeInBits() / 128;
  int NumLaneElts = NumElts / NumLanes;
  int NumHalfLaneElts = NumLaneElts / 2;

  bool MatchLo = true, MatchHi = true;
  SDValue Ops[2] = {DAG.getUNDEF(VT), DAG.getUNDEF(VT)};

  // Determine UNPCKL/UNPCKH type and operand order.
  for (int Lane = 0; Lane != NumElts; Lane += NumLaneElts) {
    for (int Elt = 0; Elt != NumLaneElts; ++Elt) {
      int M = Mask[Lane + Elt];
      if (M < 0)
        continue;

      SDValue &Op = Ops[Elt & 1];
      if (M < NumElts && (Op.isUndef() || Op == V1))
        Op = V1;
      else if (NumElts <= M && (Op.isUndef() || Op == V2))
        Op = V2;
      else
        return SDValue();

      int Lo = Lane, Mid = Lane + NumHalfLaneElts, Hi = Lane + NumLaneElts;
      MatchLo &= isUndefOrInRange(M, Lo, Mid) ||
                 isUndefOrInRange(M, NumElts + Lo, NumElts + Mid);
      MatchHi &= isUndefOrInRange(M, Mid, Hi) ||
                 isUndefOrInRange(M, NumElts + Mid, NumElts + Hi);
      if (!MatchLo && !MatchHi)
        return SDValue();
    }
  }
  assert((MatchLo ^ MatchHi) && "Failed to match UNPCKLO/UNPCKHI");

  // Now check that each pair of elts come from the same unpack pair
  // and set the permute mask based on each pair.
  // TODO - Investigate cases where we permute individual elements.
  SmallVector<int, 32> PermuteMask(NumElts, -1);
  for (int Lane = 0; Lane != NumElts; Lane += NumLaneElts) {
    for (int Elt = 0; Elt != NumLaneElts; Elt += 2) {
      int M0 = Mask[Lane + Elt + 0];
      int M1 = Mask[Lane + Elt + 1];
      if (0 <= M0 && 0 <= M1 &&
          (M0 % NumHalfLaneElts) != (M1 % NumHalfLaneElts))
        return SDValue();
      if (0 <= M0)
        PermuteMask[Lane + Elt + 0] = Lane + (2 * (M0 % NumHalfLaneElts));
      if (0 <= M1)
        PermuteMask[Lane + Elt + 1] = Lane + (2 * (M1 % NumHalfLaneElts)) + 1;
    }
  }

  unsigned UnpckOp = MatchLo ? X86ISD::UNPCKL : X86ISD::UNPCKH;
  SDValue Unpck = DAG.getNode(UnpckOp, DL, VT, Ops);
  return DAG.getVectorShuffle(VT, DL, Unpck, DAG.getUNDEF(VT), PermuteMask);
}

/// Helper to form a PALIGNR-based rotate+permute, merging 2 inputs and then
/// permuting the elements of the result in place.
static SDValue lowerShuffleAsByteRotateAndPermute(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const X86Subtarget &Subtarget, SelectionDAG &DAG) {
  if ((VT.is128BitVector() && !Subtarget.hasSSSE3()) ||
      (VT.is256BitVector() && !Subtarget.hasAVX2()) ||
      (VT.is512BitVector() && !Subtarget.hasBWI()))
    return SDValue();

  // We don't currently support lane crossing permutes.
  if (is128BitLaneCrossingShuffleMask(VT, Mask))
    return SDValue();

  int Scale = VT.getScalarSizeInBits() / 8;
  int NumLanes = VT.getSizeInBits() / 128;
  int NumElts = VT.getVectorNumElements();
  int NumEltsPerLane = NumElts / NumLanes;

  // Determine range of mask elts.
  bool Blend1 = true;
  bool Blend2 = true;
  std::pair<int, int> Range1 = std::make_pair(INT_MAX, INT_MIN);
  std::pair<int, int> Range2 = std::make_pair(INT_MAX, INT_MIN);
  for (int Lane = 0; Lane != NumElts; Lane += NumEltsPerLane) {
    for (int Elt = 0; Elt != NumEltsPerLane; ++Elt) {
      int M = Mask[Lane + Elt];
      if (M < 0)
        continue;
      if (M < NumElts) {
        Blend1 &= (M == (Lane + Elt));
        assert(Lane <= M && M < (Lane + NumEltsPerLane) && "Out of range mask");
        M = M % NumEltsPerLane;
        Range1.first = std::min(Range1.first, M);
        Range1.second = std::max(Range1.second, M);
      } else {
        M -= NumElts;
        Blend2 &= (M == (Lane + Elt));
        assert(Lane <= M && M < (Lane + NumEltsPerLane) && "Out of range mask");
        M = M % NumEltsPerLane;
        Range2.first = std::min(Range2.first, M);
        Range2.second = std::max(Range2.second, M);
      }
    }
  }

  // Bail if we don't need both elements.
  // TODO - it might be worth doing this for unary shuffles if the permute
  // can be widened.
  if (!(0 <= Range1.first && Range1.second < NumEltsPerLane) ||
      !(0 <= Range2.first && Range2.second < NumEltsPerLane))
    return SDValue();

  if (VT.getSizeInBits() > 128 && (Blend1 || Blend2))
    return SDValue();

  // Rotate the 2 ops so we can access both ranges, then permute the result.
  auto RotateAndPermute = [&](SDValue Lo, SDValue Hi, int RotAmt, int Ofs) {
    MVT ByteVT = MVT::getVectorVT(MVT::i8, VT.getSizeInBits() / 8);
    SDValue Rotate = DAG.getBitcast(
        VT, DAG.getNode(X86ISD::PALIGNR, DL, ByteVT, DAG.getBitcast(ByteVT, Hi),
                        DAG.getBitcast(ByteVT, Lo),
                        DAG.getConstant(Scale * RotAmt, DL, MVT::i8)));
    SmallVector<int, 64> PermMask(NumElts, SM_SentinelUndef);
    for (int Lane = 0; Lane != NumElts; Lane += NumEltsPerLane) {
      for (int Elt = 0; Elt != NumEltsPerLane; ++Elt) {
        int M = Mask[Lane + Elt];
        if (M < 0)
          continue;
        if (M < NumElts)
          PermMask[Lane + Elt] = Lane + ((M + Ofs - RotAmt) % NumEltsPerLane);
        else
          PermMask[Lane + Elt] = Lane + ((M - Ofs - RotAmt) % NumEltsPerLane);
      }
    }
    return DAG.getVectorShuffle(VT, DL, Rotate, DAG.getUNDEF(VT), PermMask);
  };

  // Check if the ranges are small enough to rotate from either direction.
  if (Range2.second < Range1.first)
    return RotateAndPermute(V1, V2, Range1.first, 0);
  if (Range1.second < Range2.first)
    return RotateAndPermute(V2, V1, Range2.first, NumElts);
  return SDValue();
}

/// Generic routine to decompose a shuffle and blend into independent
/// blends and permutes.
///
/// This matches the extremely common pattern for handling combined
/// shuffle+blend operations on newer X86 ISAs where we have very fast blend
/// operations. It will try to pick the best arrangement of shuffles and
/// blends.
static SDValue lowerShuffleAsDecomposedShuffleBlend(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const X86Subtarget &Subtarget, SelectionDAG &DAG) {
  // Shuffle the input elements into the desired positions in V1 and V2 and
  // blend them together.
  SmallVector<int, 32> V1Mask(Mask.size(), -1);
  SmallVector<int, 32> V2Mask(Mask.size(), -1);
  SmallVector<int, 32> BlendMask(Mask.size(), -1);
  for (int i = 0, Size = Mask.size(); i < Size; ++i)
    if (Mask[i] >= 0 && Mask[i] < Size) {
      V1Mask[i] = Mask[i];
      BlendMask[i] = i;
    } else if (Mask[i] >= Size) {
      V2Mask[i] = Mask[i] - Size;
      BlendMask[i] = i + Size;
    }

  // Try to lower with the simpler initial blend/unpack/rotate strategies unless
  // one of the input shuffles would be a no-op. We prefer to shuffle inputs as
  // the shuffle may be able to fold with a load or other benefit. However, when
  // we'll have to do 2x as many shuffles in order to achieve this, a 2-input
  // pre-shuffle first is a better strategy.
  if (!isNoopShuffleMask(V1Mask) && !isNoopShuffleMask(V2Mask)) {
    // Only prefer immediate blends to unpack/rotate.
    if (SDValue BlendPerm = lowerShuffleAsBlendAndPermute(DL, VT, V1, V2, Mask,
                                                          DAG, true))
      return BlendPerm;
    if (SDValue UnpackPerm = lowerShuffleAsUNPCKAndPermute(DL, VT, V1, V2, Mask,
                                                           DAG))
      return UnpackPerm;
    if (SDValue RotatePerm = lowerShuffleAsByteRotateAndPermute(
            DL, VT, V1, V2, Mask, Subtarget, DAG))
      return RotatePerm;
    // Unpack/rotate failed - try again with variable blends.
    if (SDValue BlendPerm = lowerShuffleAsBlendAndPermute(DL, VT, V1, V2, Mask,
                                                          DAG))
      return BlendPerm;
  }

  V1 = DAG.getVectorShuffle(VT, DL, V1, DAG.getUNDEF(VT), V1Mask);
  V2 = DAG.getVectorShuffle(VT, DL, V2, DAG.getUNDEF(VT), V2Mask);
  return DAG.getVectorShuffle(VT, DL, V1, V2, BlendMask);
}

/// Try to lower a vector shuffle as a rotation.
///
/// This is used for support PALIGNR for SSSE3 or VALIGND/Q for AVX512.
static int matchShuffleAsRotate(SDValue &V1, SDValue &V2, ArrayRef<int> Mask) {
  int NumElts = Mask.size();

  // We need to detect various ways of spelling a rotation:
  //   [11, 12, 13, 14, 15,  0,  1,  2]
  //   [-1, 12, 13, 14, -1, -1,  1, -1]
  //   [-1, -1, -1, -1, -1, -1,  1,  2]
  //   [ 3,  4,  5,  6,  7,  8,  9, 10]
  //   [-1,  4,  5,  6, -1, -1,  9, -1]
  //   [-1,  4,  5,  6, -1, -1, -1, -1]
  int Rotation = 0;
  SDValue Lo, Hi;
  for (int i = 0; i < NumElts; ++i) {
    int M = Mask[i];
    assert((M == SM_SentinelUndef || (0 <= M && M < (2*NumElts))) &&
           "Unexpected mask index.");
    if (M < 0)
      continue;

    // Determine where a rotated vector would have started.
    int StartIdx = i - (M % NumElts);
    if (StartIdx == 0)
      // The identity rotation isn't interesting, stop.
      return -1;

    // If we found the tail of a vector the rotation must be the missing
    // front. If we found the head of a vector, it must be how much of the
    // head.
    int CandidateRotation = StartIdx < 0 ? -StartIdx : NumElts - StartIdx;

    if (Rotation == 0)
      Rotation = CandidateRotation;
    else if (Rotation != CandidateRotation)
      // The rotations don't match, so we can't match this mask.
      return -1;

    // Compute which value this mask is pointing at.
    SDValue MaskV = M < NumElts ? V1 : V2;

    // Compute which of the two target values this index should be assigned
    // to. This reflects whether the high elements are remaining or the low
    // elements are remaining.
    SDValue &TargetV = StartIdx < 0 ? Hi : Lo;

    // Either set up this value if we've not encountered it before, or check
    // that it remains consistent.
    if (!TargetV)
      TargetV = MaskV;
    else if (TargetV != MaskV)
      // This may be a rotation, but it pulls from the inputs in some
      // unsupported interleaving.
      return -1;
  }

  // Check that we successfully analyzed the mask, and normalize the results.
  assert(Rotation != 0 && "Failed to locate a viable rotation!");
  assert((Lo || Hi) && "Failed to find a rotated input vector!");
  if (!Lo)
    Lo = Hi;
  else if (!Hi)
    Hi = Lo;

  V1 = Lo;
  V2 = Hi;

  return Rotation;
}

/// Try to lower a vector shuffle as a byte rotation.
///
/// SSSE3 has a generic PALIGNR instruction in x86 that will do an arbitrary
/// byte-rotation of the concatenation of two vectors; pre-SSSE3 can use
/// a PSRLDQ/PSLLDQ/POR pattern to get a similar effect. This routine will
/// try to generically lower a vector shuffle through such an pattern. It
/// does not check for the profitability of lowering either as PALIGNR or
/// PSRLDQ/PSLLDQ/POR, only whether the mask is valid to lower in that form.
/// This matches shuffle vectors that look like:
///
///   v8i16 [11, 12, 13, 14, 15, 0, 1, 2]
///
/// Essentially it concatenates V1 and V2, shifts right by some number of
/// elements, and takes the low elements as the result. Note that while this is
/// specified as a *right shift* because x86 is little-endian, it is a *left
/// rotate* of the vector lanes.
static int matchShuffleAsByteRotate(MVT VT, SDValue &V1, SDValue &V2,
                                    ArrayRef<int> Mask) {
  // Don't accept any shuffles with zero elements.
  if (any_of(Mask, [](int M) { return M == SM_SentinelZero; }))
    return -1;

  // PALIGNR works on 128-bit lanes.
  SmallVector<int, 16> RepeatedMask;
  if (!is128BitLaneRepeatedShuffleMask(VT, Mask, RepeatedMask))
    return -1;

  int Rotation = matchShuffleAsRotate(V1, V2, RepeatedMask);
  if (Rotation <= 0)
    return -1;

  // PALIGNR rotates bytes, so we need to scale the
  // rotation based on how many bytes are in the vector lane.
  int NumElts = RepeatedMask.size();
  int Scale = 16 / NumElts;
  return Rotation * Scale;
}

static SDValue lowerShuffleAsByteRotate(const SDLoc &DL, MVT VT, SDValue V1,
                                        SDValue V2, ArrayRef<int> Mask,
                                        const X86Subtarget &Subtarget,
                                        SelectionDAG &DAG) {
  assert(!isNoopShuffleMask(Mask) && "We shouldn't lower no-op shuffles!");

  SDValue Lo = V1, Hi = V2;
  int ByteRotation = matchShuffleAsByteRotate(VT, Lo, Hi, Mask);
  if (ByteRotation <= 0)
    return SDValue();

  // Cast the inputs to i8 vector of correct length to match PALIGNR or
  // PSLLDQ/PSRLDQ.
  MVT ByteVT = MVT::getVectorVT(MVT::i8, VT.getSizeInBits() / 8);
  Lo = DAG.getBitcast(ByteVT, Lo);
  Hi = DAG.getBitcast(ByteVT, Hi);

  // SSSE3 targets can use the palignr instruction.
  if (Subtarget.hasSSSE3()) {
    assert((!VT.is512BitVector() || Subtarget.hasBWI()) &&
           "512-bit PALIGNR requires BWI instructions");
    return DAG.getBitcast(
        VT, DAG.getNode(X86ISD::PALIGNR, DL, ByteVT, Lo, Hi,
                        DAG.getConstant(ByteRotation, DL, MVT::i8)));
  }

  assert(VT.is128BitVector() &&
         "Rotate-based lowering only supports 128-bit lowering!");
  assert(Mask.size() <= 16 &&
         "Can shuffle at most 16 bytes in a 128-bit vector!");
  assert(ByteVT == MVT::v16i8 &&
         "SSE2 rotate lowering only needed for v16i8!");

  // Default SSE2 implementation
  int LoByteShift = 16 - ByteRotation;
  int HiByteShift = ByteRotation;

  SDValue LoShift = DAG.getNode(X86ISD::VSHLDQ, DL, MVT::v16i8, Lo,
                                DAG.getConstant(LoByteShift, DL, MVT::i8));
  SDValue HiShift = DAG.getNode(X86ISD::VSRLDQ, DL, MVT::v16i8, Hi,
                                DAG.getConstant(HiByteShift, DL, MVT::i8));
  return DAG.getBitcast(VT,
                        DAG.getNode(ISD::OR, DL, MVT::v16i8, LoShift, HiShift));
}

/// Try to lower a vector shuffle as a dword/qword rotation.
///
/// AVX512 has a VALIGND/VALIGNQ instructions that will do an arbitrary
/// rotation of the concatenation of two vectors; This routine will
/// try to generically lower a vector shuffle through such an pattern.
///
/// Essentially it concatenates V1 and V2, shifts right by some number of
/// elements, and takes the low elements as the result. Note that while this is
/// specified as a *right shift* because x86 is little-endian, it is a *left
/// rotate* of the vector lanes.
static SDValue lowerShuffleAsRotate(const SDLoc &DL, MVT VT, SDValue V1,
                                    SDValue V2, ArrayRef<int> Mask,
                                    const X86Subtarget &Subtarget,
                                    SelectionDAG &DAG) {
  assert((VT.getScalarType() == MVT::i32 || VT.getScalarType() == MVT::i64) &&
         "Only 32-bit and 64-bit elements are supported!");

  // 128/256-bit vectors are only supported with VLX.
  assert((Subtarget.hasVLX() || (!VT.is128BitVector() && !VT.is256BitVector()))
         && "VLX required for 128/256-bit vectors");

  SDValue Lo = V1, Hi = V2;
  int Rotation = matchShuffleAsRotate(Lo, Hi, Mask);
  if (Rotation <= 0)
    return SDValue();

  return DAG.getNode(X86ISD::VALIGN, DL, VT, Lo, Hi,
                     DAG.getConstant(Rotation, DL, MVT::i8));
}

/// Try to lower a vector shuffle as a byte shift sequence.
static SDValue lowerVectorShuffleAsByteShiftMask(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const APInt &Zeroable, const X86Subtarget &Subtarget, SelectionDAG &DAG) {
  assert(!isNoopShuffleMask(Mask) && "We shouldn't lower no-op shuffles!");
  assert(VT.is128BitVector() && "Only 128-bit vectors supported");

  // We need a shuffle that has zeros at one/both ends and a sequential
  // shuffle from one source within.
  unsigned ZeroLo = Zeroable.countTrailingOnes();
  unsigned ZeroHi = Zeroable.countLeadingOnes();
  if (!ZeroLo && !ZeroHi)
    return SDValue();

  unsigned NumElts = Mask.size();
  unsigned Len = NumElts - (ZeroLo + ZeroHi);
  if (!isSequentialOrUndefInRange(Mask, ZeroLo, Len, Mask[ZeroLo]))
    return SDValue();

  unsigned Scale = VT.getScalarSizeInBits() / 8;
  ArrayRef<int> StubMask = Mask.slice(ZeroLo, Len);
  if (!isUndefOrInRange(StubMask, 0, NumElts) &&
      !isUndefOrInRange(StubMask, NumElts, 2 * NumElts))
    return SDValue();

  SDValue Res = Mask[ZeroLo] < (int)NumElts ? V1 : V2;
  Res = DAG.getBitcast(MVT::v16i8, Res);

  // Use VSHLDQ/VSRLDQ ops to zero the ends of a vector and leave an
  // inner sequential set of elements, possibly offset:
  // 01234567 --> zzzzzz01 --> 1zzzzzzz
  // 01234567 --> 4567zzzz --> zzzzz456
  // 01234567 --> z0123456 --> 3456zzzz --> zz3456zz
  if (ZeroLo == 0) {
    unsigned Shift = (NumElts - 1) - (Mask[ZeroLo + Len - 1] % NumElts);
    Res = DAG.getNode(X86ISD::VSHLDQ, DL, MVT::v16i8, Res,
                      DAG.getConstant(Scale * Shift, DL, MVT::i8));
    Res = DAG.getNode(X86ISD::VSRLDQ, DL, MVT::v16i8, Res,
                      DAG.getConstant(Scale * ZeroHi, DL, MVT::i8));
  } else if (ZeroHi == 0) {
    unsigned Shift = Mask[ZeroLo] % NumElts;
    Res = DAG.getNode(X86ISD::VSRLDQ, DL, MVT::v16i8, Res,
                      DAG.getConstant(Scale * Shift, DL, MVT::i8));
    Res = DAG.getNode(X86ISD::VSHLDQ, DL, MVT::v16i8, Res,
                      DAG.getConstant(Scale * ZeroLo, DL, MVT::i8));
  } else if (!Subtarget.hasSSSE3()) {
    // If we don't have PSHUFB then its worth avoiding an AND constant mask
    // by performing 3 byte shifts. Shuffle combining can kick in above that.
    // TODO: There may be some cases where VSH{LR}DQ+PAND is still better.
    unsigned Shift = (NumElts - 1) - (Mask[ZeroLo + Len - 1] % NumElts);
    Res = DAG.getNode(X86ISD::VSHLDQ, DL, MVT::v16i8, Res,
                      DAG.getConstant(Scale * Shift, DL, MVT::i8));
    Shift += Mask[ZeroLo] % NumElts;
    Res = DAG.getNode(X86ISD::VSRLDQ, DL, MVT::v16i8, Res,
                      DAG.getConstant(Scale * Shift, DL, MVT::i8));
    Res = DAG.getNode(X86ISD::VSHLDQ, DL, MVT::v16i8, Res,
                      DAG.getConstant(Scale * ZeroLo, DL, MVT::i8));
  } else
    return SDValue();

  return DAG.getBitcast(VT, Res);
}

/// Try to lower a vector shuffle as a bit shift (shifts in zeros).
///
/// Attempts to match a shuffle mask against the PSLL(W/D/Q/DQ) and
/// PSRL(W/D/Q/DQ) SSE2 and AVX2 logical bit-shift instructions. The function
/// matches elements from one of the input vectors shuffled to the left or
/// right with zeroable elements 'shifted in'. It handles both the strictly
/// bit-wise element shifts and the byte shift across an entire 128-bit double
/// quad word lane.
///
/// PSHL : (little-endian) left bit shift.
/// [ zz, 0, zz,  2 ]
/// [ -1, 4, zz, -1 ]
/// PSRL : (little-endian) right bit shift.
/// [  1, zz,  3, zz]
/// [ -1, -1,  7, zz]
/// PSLLDQ : (little-endian) left byte shift
/// [ zz,  0,  1,  2,  3,  4,  5,  6]
/// [ zz, zz, -1, -1,  2,  3,  4, -1]
/// [ zz, zz, zz, zz, zz, zz, -1,  1]
/// PSRLDQ : (little-endian) right byte shift
/// [  5, 6,  7, zz, zz, zz, zz, zz]
/// [ -1, 5,  6,  7, zz, zz, zz, zz]
/// [  1, 2, -1, -1, -1, -1, zz, zz]
static int matchShuffleAsShift(MVT &ShiftVT, unsigned &Opcode,
                               unsigned ScalarSizeInBits, ArrayRef<int> Mask,
                               int MaskOffset, const APInt &Zeroable,
                               const X86Subtarget &Subtarget) {
  int Size = Mask.size();
  unsigned SizeInBits = Size * ScalarSizeInBits;

  auto CheckZeros = [&](int Shift, int Scale, bool Left) {
    for (int i = 0; i < Size; i += Scale)
      for (int j = 0; j < Shift; ++j)
        if (!Zeroable[i + j + (Left ? 0 : (Scale - Shift))])
          return false;

    return true;
  };

  auto MatchShift = [&](int Shift, int Scale, bool Left) {
    for (int i = 0; i != Size; i += Scale) {
      unsigned Pos = Left ? i + Shift : i;
      unsigned Low = Left ? i : i + Shift;
      unsigned Len = Scale - Shift;
      if (!isSequentialOrUndefInRange(Mask, Pos, Len, Low + MaskOffset))
        return -1;
    }

    int ShiftEltBits = ScalarSizeInBits * Scale;
    bool ByteShift = ShiftEltBits > 64;
    Opcode = Left ? (ByteShift ? X86ISD::VSHLDQ : X86ISD::VSHLI)
                  : (ByteShift ? X86ISD::VSRLDQ : X86ISD::VSRLI);
    int ShiftAmt = Shift * ScalarSizeInBits / (ByteShift ? 8 : 1);

    // Normalize the scale for byte shifts to still produce an i64 element
    // type.
    Scale = ByteShift ? Scale / 2 : Scale;

    // We need to round trip through the appropriate type for the shift.
    MVT ShiftSVT = MVT::getIntegerVT(ScalarSizeInBits * Scale);
    ShiftVT = ByteShift ? MVT::getVectorVT(MVT::i8, SizeInBits / 8)
                        : MVT::getVectorVT(ShiftSVT, Size / Scale);
    return (int)ShiftAmt;
  };

  // SSE/AVX supports logical shifts up to 64-bit integers - so we can just
  // keep doubling the size of the integer elements up to that. We can
  // then shift the elements of the integer vector by whole multiples of
  // their width within the elements of the larger integer vector. Test each
  // multiple to see if we can find a match with the moved element indices
  // and that the shifted in elements are all zeroable.
  unsigned MaxWidth = ((SizeInBits == 512) && !Subtarget.hasBWI() ? 64 : 128);
  for (int Scale = 2; Scale * ScalarSizeInBits <= MaxWidth; Scale *= 2)
    for (int Shift = 1; Shift != Scale; ++Shift)
      for (bool Left : {true, false})
        if (CheckZeros(Shift, Scale, Left)) {
          int ShiftAmt = MatchShift(Shift, Scale, Left);
          if (0 < ShiftAmt)
            return ShiftAmt;
        }

  // no match
  return -1;
}

static SDValue lowerShuffleAsShift(const SDLoc &DL, MVT VT, SDValue V1,
                                   SDValue V2, ArrayRef<int> Mask,
                                   const APInt &Zeroable,
                                   const X86Subtarget &Subtarget,
                                   SelectionDAG &DAG) {
  int Size = Mask.size();
  assert(Size == (int)VT.getVectorNumElements() && "Unexpected mask size");

  MVT ShiftVT;
  SDValue V = V1;
  unsigned Opcode;

  // Try to match shuffle against V1 shift.
  int ShiftAmt = matchShuffleAsShift(ShiftVT, Opcode, VT.getScalarSizeInBits(),
                                     Mask, 0, Zeroable, Subtarget);

  // If V1 failed, try to match shuffle against V2 shift.
  if (ShiftAmt < 0) {
    ShiftAmt = matchShuffleAsShift(ShiftVT, Opcode, VT.getScalarSizeInBits(),
                                   Mask, Size, Zeroable, Subtarget);
    V = V2;
  }

  if (ShiftAmt < 0)
    return SDValue();

  assert(DAG.getTargetLoweringInfo().isTypeLegal(ShiftVT) &&
         "Illegal integer vector type");
  V = DAG.getBitcast(ShiftVT, V);
  V = DAG.getNode(Opcode, DL, ShiftVT, V,
                  DAG.getConstant(ShiftAmt, DL, MVT::i8));
  return DAG.getBitcast(VT, V);
}

// EXTRQ: Extract Len elements from lower half of source, starting at Idx.
// Remainder of lower half result is zero and upper half is all undef.
static bool matchShuffleAsEXTRQ(MVT VT, SDValue &V1, SDValue &V2,
                                ArrayRef<int> Mask, uint64_t &BitLen,
                                uint64_t &BitIdx, const APInt &Zeroable) {
  int Size = Mask.size();
  int HalfSize = Size / 2;
  assert(Size == (int)VT.getVectorNumElements() && "Unexpected mask size");
  assert(!Zeroable.isAllOnesValue() && "Fully zeroable shuffle mask");

  // Upper half must be undefined.
  if (!isUndefUpperHalf(Mask))
    return false;

  // Determine the extraction length from the part of the
  // lower half that isn't zeroable.
  int Len = HalfSize;
  for (; Len > 0; --Len)
    if (!Zeroable[Len - 1])
      break;
  assert(Len > 0 && "Zeroable shuffle mask");

  // Attempt to match first Len sequential elements from the lower half.
  SDValue Src;
  int Idx = -1;
  for (int i = 0; i != Len; ++i) {
    int M = Mask[i];
    if (M == SM_SentinelUndef)
      continue;
    SDValue &V = (M < Size ? V1 : V2);
    M = M % Size;

    // The extracted elements must start at a valid index and all mask
    // elements must be in the lower half.
    if (i > M || M >= HalfSize)
      return false;

    if (Idx < 0 || (Src == V && Idx == (M - i))) {
      Src = V;
      Idx = M - i;
      continue;
    }
    return false;
  }

  if (!Src || Idx < 0)
    return false;

  assert((Idx + Len) <= HalfSize && "Illegal extraction mask");
  BitLen = (Len * VT.getScalarSizeInBits()) & 0x3f;
  BitIdx = (Idx * VT.getScalarSizeInBits()) & 0x3f;
  V1 = Src;
  return true;
}

// INSERTQ: Extract lowest Len elements from lower half of second source and
// insert over first source, starting at Idx.
// { A[0], .., A[Idx-1], B[0], .., B[Len-1], A[Idx+Len], .., UNDEF, ... }
static bool matchShuffleAsINSERTQ(MVT VT, SDValue &V1, SDValue &V2,
                                  ArrayRef<int> Mask, uint64_t &BitLen,
                                  uint64_t &BitIdx) {
  int Size = Mask.size();
  int HalfSize = Size / 2;
  assert(Size == (int)VT.getVectorNumElements() && "Unexpected mask size");

  // Upper half must be undefined.
  if (!isUndefUpperHalf(Mask))
    return false;

  for (int Idx = 0; Idx != HalfSize; ++Idx) {
    SDValue Base;

    // Attempt to match first source from mask before insertion point.
    if (isUndefInRange(Mask, 0, Idx)) {
      /* EMPTY */
    } else if (isSequentialOrUndefInRange(Mask, 0, Idx, 0)) {
      Base = V1;
    } else if (isSequentialOrUndefInRange(Mask, 0, Idx, Size)) {
      Base = V2;
    } else {
      continue;
    }

    // Extend the extraction length looking to match both the insertion of
    // the second source and the remaining elements of the first.
    for (int Hi = Idx + 1; Hi <= HalfSize; ++Hi) {
      SDValue Insert;
      int Len = Hi - Idx;

      // Match insertion.
      if (isSequentialOrUndefInRange(Mask, Idx, Len, 0)) {
        Insert = V1;
      } else if (isSequentialOrUndefInRange(Mask, Idx, Len, Size)) {
        Insert = V2;
      } else {
        continue;
      }

      // Match the remaining elements of the lower half.
      if (isUndefInRange(Mask, Hi, HalfSize - Hi)) {
        /* EMPTY */
      } else if ((!Base || (Base == V1)) &&
                 isSequentialOrUndefInRange(Mask, Hi, HalfSize - Hi, Hi)) {
        Base = V1;
      } else if ((!Base || (Base == V2)) &&
                 isSequentialOrUndefInRange(Mask, Hi, HalfSize - Hi,
                                            Size + Hi)) {
        Base = V2;
      } else {
        continue;
      }

      BitLen = (Len * VT.getScalarSizeInBits()) & 0x3f;
      BitIdx = (Idx * VT.getScalarSizeInBits()) & 0x3f;
      V1 = Base;
      V2 = Insert;
      return true;
    }
  }

  return false;
}

/// Try to lower a vector shuffle using SSE4a EXTRQ/INSERTQ.
static SDValue lowerShuffleWithSSE4A(const SDLoc &DL, MVT VT, SDValue V1,
                                     SDValue V2, ArrayRef<int> Mask,
                                     const APInt &Zeroable, SelectionDAG &DAG) {
  uint64_t BitLen, BitIdx;
  if (matchShuffleAsEXTRQ(VT, V1, V2, Mask, BitLen, BitIdx, Zeroable))
    return DAG.getNode(X86ISD::EXTRQI, DL, VT, V1,
                       DAG.getConstant(BitLen, DL, MVT::i8),
                       DAG.getConstant(BitIdx, DL, MVT::i8));

  if (matchShuffleAsINSERTQ(VT, V1, V2, Mask, BitLen, BitIdx))
    return DAG.getNode(X86ISD::INSERTQI, DL, VT, V1 ? V1 : DAG.getUNDEF(VT),
                       V2 ? V2 : DAG.getUNDEF(VT),
                       DAG.getConstant(BitLen, DL, MVT::i8),
                       DAG.getConstant(BitIdx, DL, MVT::i8));

  return SDValue();
}

/// Lower a vector shuffle as a zero or any extension.
///
/// Given a specific number of elements, element bit width, and extension
/// stride, produce either a zero or any extension based on the available
/// features of the subtarget. The extended elements are consecutive and
/// begin and can start from an offsetted element index in the input; to
/// avoid excess shuffling the offset must either being in the bottom lane
/// or at the start of a higher lane. All extended elements must be from
/// the same lane.
static SDValue lowerShuffleAsSpecificZeroOrAnyExtend(
    const SDLoc &DL, MVT VT, int Scale, int Offset, bool AnyExt, SDValue InputV,
    ArrayRef<int> Mask, const X86Subtarget &Subtarget, SelectionDAG &DAG) {
  assert(Scale > 1 && "Need a scale to extend.");
  int EltBits = VT.getScalarSizeInBits();
  int NumElements = VT.getVectorNumElements();
  int NumEltsPerLane = 128 / EltBits;
  int OffsetLane = Offset / NumEltsPerLane;
  assert((EltBits == 8 || EltBits == 16 || EltBits == 32) &&
         "Only 8, 16, and 32 bit elements can be extended.");
  assert(Scale * EltBits <= 64 && "Cannot zero extend past 64 bits.");
  assert(0 <= Offset && "Extension offset must be positive.");
  assert((Offset < NumEltsPerLane || Offset % NumEltsPerLane == 0) &&
         "Extension offset must be in the first lane or start an upper lane.");

  // Check that an index is in same lane as the base offset.
  auto SafeOffset = [&](int Idx) {
    return OffsetLane == (Idx / NumEltsPerLane);
  };

  // Shift along an input so that the offset base moves to the first element.
  auto ShuffleOffset = [&](SDValue V) {
    if (!Offset)
      return V;

    SmallVector<int, 8> ShMask((unsigned)NumElements, -1);
    for (int i = 0; i * Scale < NumElements; ++i) {
      int SrcIdx = i + Offset;
      ShMask[i] = SafeOffset(SrcIdx) ? SrcIdx : -1;
    }
    return DAG.getVectorShuffle(VT, DL, V, DAG.getUNDEF(VT), ShMask);
  };

  // Found a valid a/zext mask! Try various lowering strategies based on the
  // input type and available ISA extensions.
  if (Subtarget.hasSSE41()) {
    // Not worth offsetting 128-bit vectors if scale == 2, a pattern using
    // PUNPCK will catch this in a later shuffle match.
    if (Offset && Scale == 2 && VT.is128BitVector())
      return SDValue();
    MVT ExtVT = MVT::getVectorVT(MVT::getIntegerVT(EltBits * Scale),
                                 NumElements / Scale);
    InputV = ShuffleOffset(InputV);
    InputV = getExtendInVec(AnyExt ? ISD::ANY_EXTEND : ISD::ZERO_EXTEND, DL,
                            ExtVT, InputV, DAG);
    return DAG.getBitcast(VT, InputV);
  }

  assert(VT.is128BitVector() && "Only 128-bit vectors can be extended.");

  // For any extends we can cheat for larger element sizes and use shuffle
  // instructions that can fold with a load and/or copy.
  if (AnyExt && EltBits == 32) {
    int PSHUFDMask[4] = {Offset, -1, SafeOffset(Offset + 1) ? Offset + 1 : -1,
                         -1};
    return DAG.getBitcast(
        VT, DAG.getNode(X86ISD::PSHUFD, DL, MVT::v4i32,
                        DAG.getBitcast(MVT::v4i32, InputV),
                        getV4X86ShuffleImm8ForMask(PSHUFDMask, DL, DAG)));
  }
  if (AnyExt && EltBits == 16 && Scale > 2) {
    int PSHUFDMask[4] = {Offset / 2, -1,
                         SafeOffset(Offset + 1) ? (Offset + 1) / 2 : -1, -1};
    InputV = DAG.getNode(X86ISD::PSHUFD, DL, MVT::v4i32,
                         DAG.getBitcast(MVT::v4i32, InputV),
                         getV4X86ShuffleImm8ForMask(PSHUFDMask, DL, DAG));
    int PSHUFWMask[4] = {1, -1, -1, -1};
    unsigned OddEvenOp = (Offset & 1) ? X86ISD::PSHUFLW : X86ISD::PSHUFHW;
    return DAG.getBitcast(
        VT, DAG.getNode(OddEvenOp, DL, MVT::v8i16,
                        DAG.getBitcast(MVT::v8i16, InputV),
                        getV4X86ShuffleImm8ForMask(PSHUFWMask, DL, DAG)));
  }

  // The SSE4A EXTRQ instruction can efficiently extend the first 2 lanes
  // to 64-bits.
  if ((Scale * EltBits) == 64 && EltBits < 32 && Subtarget.hasSSE4A()) {
    assert(NumElements == (int)Mask.size() && "Unexpected shuffle mask size!");
    assert(VT.is128BitVector() && "Unexpected vector width!");

    int LoIdx = Offset * EltBits;
    SDValue Lo = DAG.getBitcast(
        MVT::v2i64, DAG.getNode(X86ISD::EXTRQI, DL, VT, InputV,
                                DAG.getConstant(EltBits, DL, MVT::i8),
                                DAG.getConstant(LoIdx, DL, MVT::i8)));

    if (isUndefUpperHalf(Mask) || !SafeOffset(Offset + 1))
      return DAG.getBitcast(VT, Lo);

    int HiIdx = (Offset + 1) * EltBits;
    SDValue Hi = DAG.getBitcast(
        MVT::v2i64, DAG.getNode(X86ISD::EXTRQI, DL, VT, InputV,
                                DAG.getConstant(EltBits, DL, MVT::i8),
                                DAG.getConstant(HiIdx, DL, MVT::i8)));
    return DAG.getBitcast(VT,
                          DAG.getNode(X86ISD::UNPCKL, DL, MVT::v2i64, Lo, Hi));
  }

  // If this would require more than 2 unpack instructions to expand, use
  // pshufb when available. We can only use more than 2 unpack instructions
  // when zero extending i8 elements which also makes it easier to use pshufb.
  if (Scale > 4 && EltBits == 8 && Subtarget.hasSSSE3()) {
    assert(NumElements == 16 && "Unexpected byte vector width!");
    SDValue PSHUFBMask[16];
    for (int i = 0; i < 16; ++i) {
      int Idx = Offset + (i / Scale);
      if ((i % Scale == 0 && SafeOffset(Idx))) {
        PSHUFBMask[i] = DAG.getConstant(Idx, DL, MVT::i8);
        continue;
      }
      PSHUFBMask[i] =
          AnyExt ? DAG.getUNDEF(MVT::i8) : DAG.getConstant(0x80, DL, MVT::i8);
    }
    InputV = DAG.getBitcast(MVT::v16i8, InputV);
    return DAG.getBitcast(
        VT, DAG.getNode(X86ISD::PSHUFB, DL, MVT::v16i8, InputV,
                        DAG.getBuildVector(MVT::v16i8, DL, PSHUFBMask)));
  }

  // If we are extending from an offset, ensure we start on a boundary that
  // we can unpack from.
  int AlignToUnpack = Offset % (NumElements / Scale);
  if (AlignToUnpack) {
    SmallVector<int, 8> ShMask((unsigned)NumElements, -1);
    for (int i = AlignToUnpack; i < NumElements; ++i)
      ShMask[i - AlignToUnpack] = i;
    InputV = DAG.getVectorShuffle(VT, DL, InputV, DAG.getUNDEF(VT), ShMask);
    Offset -= AlignToUnpack;
  }

  // Otherwise emit a sequence of unpacks.
  do {
    unsigned UnpackLoHi = X86ISD::UNPCKL;
    if (Offset >= (NumElements / 2)) {
      UnpackLoHi = X86ISD::UNPCKH;
      Offset -= (NumElements / 2);
    }

    MVT InputVT = MVT::getVectorVT(MVT::getIntegerVT(EltBits), NumElements);
    SDValue Ext = AnyExt ? DAG.getUNDEF(InputVT)
                         : getZeroVector(InputVT, Subtarget, DAG, DL);
    InputV = DAG.getBitcast(InputVT, InputV);
    InputV = DAG.getNode(UnpackLoHi, DL, InputVT, InputV, Ext);
    Scale /= 2;
    EltBits *= 2;
    NumElements /= 2;
  } while (Scale > 1);
  return DAG.getBitcast(VT, InputV);
}

/// Try to lower a vector shuffle as a zero extension on any microarch.
///
/// This routine will try to do everything in its power to cleverly lower
/// a shuffle which happens to match the pattern of a zero extend. It doesn't
/// check for the profitability of this lowering,  it tries to aggressively
/// match this pattern. It will use all of the micro-architectural details it
/// can to emit an efficient lowering. It handles both blends with all-zero
/// inputs to explicitly zero-extend and undef-lanes (sometimes undef due to
/// masking out later).
///
/// The reason we have dedicated lowering for zext-style shuffles is that they
/// are both incredibly common and often quite performance sensitive.
static SDValue lowerShuffleAsZeroOrAnyExtend(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const APInt &Zeroable, const X86Subtarget &Subtarget,
    SelectionDAG &DAG) {
  int Bits = VT.getSizeInBits();
  int NumLanes = Bits / 128;
  int NumElements = VT.getVectorNumElements();
  int NumEltsPerLane = NumElements / NumLanes;
  assert(VT.getScalarSizeInBits() <= 32 &&
         "Exceeds 32-bit integer zero extension limit");
  assert((int)Mask.size() == NumElements && "Unexpected shuffle mask size");

  // Define a helper function to check a particular ext-scale and lower to it if
  // valid.
  auto Lower = [&](int Scale) -> SDValue {
    SDValue InputV;
    bool AnyExt = true;
    int Offset = 0;
    int Matches = 0;
    for (int i = 0; i < NumElements; ++i) {
      int M = Mask[i];
      if (M < 0)
        continue; // Valid anywhere but doesn't tell us anything.
      if (i % Scale != 0) {
        // Each of the extended elements need to be zeroable.
        if (!Zeroable[i])
          return SDValue();

        // We no longer are in the anyext case.
        AnyExt = false;
        continue;
      }

      // Each of the base elements needs to be consecutive indices into the
      // same input vector.
      SDValue V = M < NumElements ? V1 : V2;
      M = M % NumElements;
      if (!InputV) {
        InputV = V;
        Offset = M - (i / Scale);
      } else if (InputV != V)
        return SDValue(); // Flip-flopping inputs.

      // Offset must start in the lowest 128-bit lane or at the start of an
      // upper lane.
      // FIXME: Is it ever worth allowing a negative base offset?
      if (!((0 <= Offset && Offset < NumEltsPerLane) ||
            (Offset % NumEltsPerLane) == 0))
        return SDValue();

      // If we are offsetting, all referenced entries must come from the same
      // lane.
      if (Offset && (Offset / NumEltsPerLane) != (M / NumEltsPerLane))
        return SDValue();

      if ((M % NumElements) != (Offset + (i / Scale)))
        return SDValue(); // Non-consecutive strided elements.
      Matches++;
    }

    // If we fail to find an input, we have a zero-shuffle which should always
    // have already been handled.
    // FIXME: Maybe handle this here in case during blending we end up with one?
    if (!InputV)
      return SDValue();

    // If we are offsetting, don't extend if we only match a single input, we
    // can always do better by using a basic PSHUF or PUNPCK.
    if (Offset != 0 && Matches < 2)
      return SDValue();

    return lowerShuffleAsSpecificZeroOrAnyExtend(DL, VT, Scale, Offset, AnyExt,
                                                 InputV, Mask, Subtarget, DAG);
  };

  // The widest scale possible for extending is to a 64-bit integer.
  assert(Bits % 64 == 0 &&
         "The number of bits in a vector must be divisible by 64 on x86!");
  int NumExtElements = Bits / 64;

  // Each iteration, try extending the elements half as much, but into twice as
  // many elements.
  for (; NumExtElements < NumElements; NumExtElements *= 2) {
    assert(NumElements % NumExtElements == 0 &&
           "The input vector size must be divisible by the extended size.");
    if (SDValue V = Lower(NumElements / NumExtElements))
      return V;
  }

  // General extends failed, but 128-bit vectors may be able to use MOVQ.
  if (Bits != 128)
    return SDValue();

  // Returns one of the source operands if the shuffle can be reduced to a
  // MOVQ, copying the lower 64-bits and zero-extending to the upper 64-bits.
  auto CanZExtLowHalf = [&]() {
    for (int i = NumElements / 2; i != NumElements; ++i)
      if (!Zeroable[i])
        return SDValue();
    if (isSequentialOrUndefInRange(Mask, 0, NumElements / 2, 0))
      return V1;
    if (isSequentialOrUndefInRange(Mask, 0, NumElements / 2, NumElements))
      return V2;
    return SDValue();
  };

  if (SDValue V = CanZExtLowHalf()) {
    V = DAG.getBitcast(MVT::v2i64, V);
    V = DAG.getNode(X86ISD::VZEXT_MOVL, DL, MVT::v2i64, V);
    return DAG.getBitcast(VT, V);
  }

  // No viable ext lowering found.
  return SDValue();
}

/// Try to get a scalar value for a specific element of a vector.
///
/// Looks through BUILD_VECTOR and SCALAR_TO_VECTOR nodes to find a scalar.
static SDValue getScalarValueForVectorElement(SDValue V, int Idx,
                                              SelectionDAG &DAG) {
  MVT VT = V.getSimpleValueType();
  MVT EltVT = VT.getVectorElementType();
  V = peekThroughBitcasts(V);

  // If the bitcasts shift the element size, we can't extract an equivalent
  // element from it.
  MVT NewVT = V.getSimpleValueType();
  if (!NewVT.isVector() || NewVT.getScalarSizeInBits() != VT.getScalarSizeInBits())
    return SDValue();

  if (V.getOpcode() == ISD::BUILD_VECTOR ||
      (Idx == 0 && V.getOpcode() == ISD::SCALAR_TO_VECTOR)) {
    // Ensure the scalar operand is the same size as the destination.
    // FIXME: Add support for scalar truncation where possible.
    SDValue S = V.getOperand(Idx);
    if (EltVT.getSizeInBits() == S.getSimpleValueType().getSizeInBits())
      return DAG.getBitcast(EltVT, S);
  }

  return SDValue();
}

/// Helper to test for a load that can be folded with x86 shuffles.
///
/// This is particularly important because the set of instructions varies
/// significantly based on whether the operand is a load or not.
static bool isShuffleFoldableLoad(SDValue V) {
  V = peekThroughBitcasts(V);
  return ISD::isNON_EXTLoad(V.getNode());
}

/// Try to lower insertion of a single element into a zero vector.
///
/// This is a common pattern that we have especially efficient patterns to lower
/// across all subtarget feature sets.
static SDValue lowerShuffleAsElementInsertion(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const APInt &Zeroable, const X86Subtarget &Subtarget,
    SelectionDAG &DAG) {
  MVT ExtVT = VT;
  MVT EltVT = VT.getVectorElementType();

  int V2Index =
      find_if(Mask, [&Mask](int M) { return M >= (int)Mask.size(); }) -
      Mask.begin();
  bool IsV1Zeroable = true;
  for (int i = 0, Size = Mask.size(); i < Size; ++i)
    if (i != V2Index && !Zeroable[i]) {
      IsV1Zeroable = false;
      break;
    }

  // Check for a single input from a SCALAR_TO_VECTOR node.
  // FIXME: All of this should be canonicalized into INSERT_VECTOR_ELT and
  // all the smarts here sunk into that routine. However, the current
  // lowering of BUILD_VECTOR makes that nearly impossible until the old
  // vector shuffle lowering is dead.
  SDValue V2S = getScalarValueForVectorElement(V2, Mask[V2Index] - Mask.size(),
                                               DAG);
  if (V2S && DAG.getTargetLoweringInfo().isTypeLegal(V2S.getValueType())) {
    // We need to zext the scalar if it is smaller than an i32.
    V2S = DAG.getBitcast(EltVT, V2S);
    if (EltVT == MVT::i8 || EltVT == MVT::i16) {
      // Using zext to expand a narrow element won't work for non-zero
      // insertions.
      if (!IsV1Zeroable)
        return SDValue();

      // Zero-extend directly to i32.
      ExtVT = MVT::getVectorVT(MVT::i32, ExtVT.getSizeInBits() / 32);
      V2S = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, V2S);
    }
    V2 = DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, ExtVT, V2S);
  } else if (Mask[V2Index] != (int)Mask.size() || EltVT == MVT::i8 ||
             EltVT == MVT::i16) {
    // Either not inserting from the low element of the input or the input
    // element size is too small to use VZEXT_MOVL to clear the high bits.
    return SDValue();
  }

  if (!IsV1Zeroable) {
    // If V1 can't be treated as a zero vector we have fewer options to lower
    // this. We can't support integer vectors or non-zero targets cheaply, and
    // the V1 elements can't be permuted in any way.
    assert(VT == ExtVT && "Cannot change extended type when non-zeroable!");
    if (!VT.isFloatingPoint() || V2Index != 0)
      return SDValue();
    SmallVector<int, 8> V1Mask(Mask.begin(), Mask.end());
    V1Mask[V2Index] = -1;
    if (!isNoopShuffleMask(V1Mask))
      return SDValue();
    if (!VT.is128BitVector())
      return SDValue();

    // Otherwise, use MOVSD or MOVSS.
    assert((EltVT == MVT::f32 || EltVT == MVT::f64) &&
           "Only two types of floating point element types to handle!");
    return DAG.getNode(EltVT == MVT::f32 ? X86ISD::MOVSS : X86ISD::MOVSD, DL,
                       ExtVT, V1, V2);
  }

  // This lowering only works for the low element with floating point vectors.
  if (VT.isFloatingPoint() && V2Index != 0)
    return SDValue();

  V2 = DAG.getNode(X86ISD::VZEXT_MOVL, DL, ExtVT, V2);
  if (ExtVT != VT)
    V2 = DAG.getBitcast(VT, V2);

  if (V2Index != 0) {
    // If we have 4 or fewer lanes we can cheaply shuffle the element into
    // the desired position. Otherwise it is more efficient to do a vector
    // shift left. We know that we can do a vector shift left because all
    // the inputs are zero.
    if (VT.isFloatingPoint() || VT.getVectorNumElements() <= 4) {
      SmallVector<int, 4> V2Shuffle(Mask.size(), 1);
      V2Shuffle[V2Index] = 0;
      V2 = DAG.getVectorShuffle(VT, DL, V2, DAG.getUNDEF(VT), V2Shuffle);
    } else {
      V2 = DAG.getBitcast(MVT::v16i8, V2);
      V2 = DAG.getNode(
          X86ISD::VSHLDQ, DL, MVT::v16i8, V2,
          DAG.getConstant(V2Index * EltVT.getSizeInBits() / 8, DL, MVT::i8));
      V2 = DAG.getBitcast(VT, V2);
    }
  }
  return V2;
}

/// Try to lower broadcast of a single - truncated - integer element,
/// coming from a scalar_to_vector/build_vector node \p V0 with larger elements.
///
/// This assumes we have AVX2.
static SDValue lowerShuffleAsTruncBroadcast(const SDLoc &DL, MVT VT, SDValue V0,
                                            int BroadcastIdx,
                                            const X86Subtarget &Subtarget,
                                            SelectionDAG &DAG) {
  assert(Subtarget.hasAVX2() &&
         "We can only lower integer broadcasts with AVX2!");

  EVT EltVT = VT.getVectorElementType();
  EVT V0VT = V0.getValueType();

  assert(VT.isInteger() && "Unexpected non-integer trunc broadcast!");
  assert(V0VT.isVector() && "Unexpected non-vector vector-sized value!");

  EVT V0EltVT = V0VT.getVectorElementType();
  if (!V0EltVT.isInteger())
    return SDValue();

  const unsigned EltSize = EltVT.getSizeInBits();
  const unsigned V0EltSize = V0EltVT.getSizeInBits();

  // This is only a truncation if the original element type is larger.
  if (V0EltSize <= EltSize)
    return SDValue();

  assert(((V0EltSize % EltSize) == 0) &&
         "Scalar type sizes must all be powers of 2 on x86!");

  const unsigned V0Opc = V0.getOpcode();
  const unsigned Scale = V0EltSize / EltSize;
  const unsigned V0BroadcastIdx = BroadcastIdx / Scale;

  if ((V0Opc != ISD::SCALAR_TO_VECTOR || V0BroadcastIdx != 0) &&
      V0Opc != ISD::BUILD_VECTOR)
    return SDValue();

  SDValue Scalar = V0.getOperand(V0BroadcastIdx);

  // If we're extracting non-least-significant bits, shift so we can truncate.
  // Hopefully, we can fold away the trunc/srl/load into the broadcast.
  // Even if we can't (and !isShuffleFoldableLoad(Scalar)), prefer
  // vpbroadcast+vmovd+shr to vpshufb(m)+vmovd.
  if (const int OffsetIdx = BroadcastIdx % Scale)
    Scalar = DAG.getNode(ISD::SRL, DL, Scalar.getValueType(), Scalar,
                         DAG.getConstant(OffsetIdx * EltSize, DL, MVT::i8));

  return DAG.getNode(X86ISD::VBROADCAST, DL, VT,
                     DAG.getNode(ISD::TRUNCATE, DL, EltVT, Scalar));
}

/// Test whether this can be lowered with a single SHUFPS instruction.
///
/// This is used to disable more specialized lowerings when the shufps lowering
/// will happen to be efficient.
static bool isSingleSHUFPSMask(ArrayRef<int> Mask) {
  // This routine only handles 128-bit shufps.
  assert(Mask.size() == 4 && "Unsupported mask size!");
  assert(Mask[0] >= -1 && Mask[0] < 8 && "Out of bound mask element!");
  assert(Mask[1] >= -1 && Mask[1] < 8 && "Out of bound mask element!");
  assert(Mask[2] >= -1 && Mask[2] < 8 && "Out of bound mask element!");
  assert(Mask[3] >= -1 && Mask[3] < 8 && "Out of bound mask element!");

  // To lower with a single SHUFPS we need to have the low half and high half
  // each requiring a single input.
  if (Mask[0] >= 0 && Mask[1] >= 0 && (Mask[0] < 4) != (Mask[1] < 4))
    return false;
  if (Mask[2] >= 0 && Mask[3] >= 0 && (Mask[2] < 4) != (Mask[3] < 4))
    return false;

  return true;
}

/// If we are extracting two 128-bit halves of a vector and shuffling the
/// result, match that to a 256-bit AVX2 vperm* instruction to avoid a
/// multi-shuffle lowering.
static SDValue lowerShuffleOfExtractsAsVperm(const SDLoc &DL, SDValue N0,
                                             SDValue N1, ArrayRef<int> Mask,
                                             SelectionDAG &DAG) {
  EVT VT = N0.getValueType();
  assert((VT.is128BitVector() &&
          (VT.getScalarSizeInBits() == 32 || VT.getScalarSizeInBits() == 64)) &&
         "VPERM* family of shuffles requires 32-bit or 64-bit elements");

  // Check that both sources are extracts of the same source vector.
  if (!N0.hasOneUse() || !N1.hasOneUse() ||
      N0.getOpcode() != ISD::EXTRACT_SUBVECTOR ||
      N1.getOpcode() != ISD::EXTRACT_SUBVECTOR ||
      N0.getOperand(0) != N1.getOperand(0))
    return SDValue();

  SDValue WideVec = N0.getOperand(0);
  EVT WideVT = WideVec.getValueType();
  if (!WideVT.is256BitVector() || !isa<ConstantSDNode>(N0.getOperand(1)) ||
      !isa<ConstantSDNode>(N1.getOperand(1)))
    return SDValue();

  // Match extracts of each half of the wide source vector. Commute the shuffle
  // if the extract of the low half is N1.
  unsigned NumElts = VT.getVectorNumElements();
  SmallVector<int, 4> NewMask(Mask.begin(), Mask.end());
  const APInt &ExtIndex0 = N0.getConstantOperandAPInt(1);
  const APInt &ExtIndex1 = N1.getConstantOperandAPInt(1);
  if (ExtIndex1 == 0 && ExtIndex0 == NumElts)
    ShuffleVectorSDNode::commuteMask(NewMask);
  else if (ExtIndex0 != 0 || ExtIndex1 != NumElts)
    return SDValue();

  // Final bailout: if the mask is simple, we are better off using an extract
  // and a simple narrow shuffle. Prefer extract+unpack(h/l)ps to vpermps
  // because that avoids a constant load from memory.
  if (NumElts == 4 &&
      (isSingleSHUFPSMask(NewMask) || is128BitUnpackShuffleMask(NewMask)))
    return SDValue();

  // Extend the shuffle mask with undef elements.
  NewMask.append(NumElts, -1);

  // shuf (extract X, 0), (extract X, 4), M --> extract (shuf X, undef, M'), 0
  SDValue Shuf = DAG.getVectorShuffle(WideVT, DL, WideVec, DAG.getUNDEF(WideVT),
                                      NewMask);
  // This is free: ymm -> xmm.
  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, Shuf,
                     DAG.getIntPtrConstant(0, DL));
}

/// Try to lower broadcast of a single element.
///
/// For convenience, this code also bundles all of the subtarget feature set
/// filtering. While a little annoying to re-dispatch on type here, there isn't
/// a convenient way to factor it out.
static SDValue lowerShuffleAsBroadcast(const SDLoc &DL, MVT VT, SDValue V1,
                                       SDValue V2, ArrayRef<int> Mask,
                                       const X86Subtarget &Subtarget,
                                       SelectionDAG &DAG) {
  if (!((Subtarget.hasSSE3() && VT == MVT::v2f64) ||
        (Subtarget.hasAVX() && VT.isFloatingPoint()) ||
        (Subtarget.hasAVX2() && VT.isInteger())))
    return SDValue();

  // With MOVDDUP (v2f64) we can broadcast from a register or a load, otherwise
  // we can only broadcast from a register with AVX2.
  unsigned NumElts = Mask.size();
  unsigned NumEltBits = VT.getScalarSizeInBits();
  unsigned Opcode = (VT == MVT::v2f64 && !Subtarget.hasAVX2())
                        ? X86ISD::MOVDDUP
                        : X86ISD::VBROADCAST;
  bool BroadcastFromReg = (Opcode == X86ISD::MOVDDUP) || Subtarget.hasAVX2();

  // Check that the mask is a broadcast.
  int BroadcastIdx = -1;
  for (int i = 0; i != (int)NumElts; ++i) {
    SmallVector<int, 8> BroadcastMask(NumElts, i);
    if (isShuffleEquivalent(V1, V2, Mask, BroadcastMask)) {
      BroadcastIdx = i;
      break;
    }
  }

  if (BroadcastIdx < 0)
    return SDValue();
  assert(BroadcastIdx < (int)Mask.size() && "We only expect to be called with "
                                            "a sorted mask where the broadcast "
                                            "comes from V1.");

  // Go up the chain of (vector) values to find a scalar load that we can
  // combine with the broadcast.
  int BitOffset = BroadcastIdx * NumEltBits;
  SDValue V = V1;
  for (;;) {
    switch (V.getOpcode()) {
    case ISD::BITCAST: {
      V = V.getOperand(0);
      continue;
    }
    case ISD::CONCAT_VECTORS: {
      int OpBitWidth = V.getOperand(0).getValueSizeInBits();
      int OpIdx = BitOffset / OpBitWidth;
      V = V.getOperand(OpIdx);
      BitOffset %= OpBitWidth;
      continue;
    }
    case ISD::INSERT_SUBVECTOR: {
      SDValue VOuter = V.getOperand(0), VInner = V.getOperand(1);
      auto ConstantIdx = dyn_cast<ConstantSDNode>(V.getOperand(2));
      if (!ConstantIdx)
        break;

      int EltBitWidth = VOuter.getScalarValueSizeInBits();
      int Idx = (int)ConstantIdx->getZExtValue();
      int NumSubElts = (int)VInner.getSimpleValueType().getVectorNumElements();
      int BeginOffset = Idx * EltBitWidth;
      int EndOffset = BeginOffset + NumSubElts * EltBitWidth;
      if (BeginOffset <= BitOffset && BitOffset < EndOffset) {
        BitOffset -= BeginOffset;
        V = VInner;
      } else {
        V = VOuter;
      }
      continue;
    }
    }
    break;
  }
  assert((BitOffset % NumEltBits) == 0 && "Illegal bit-offset");
  BroadcastIdx = BitOffset / NumEltBits;

  // Do we need to bitcast the source to retrieve the original broadcast index?
  bool BitCastSrc = V.getScalarValueSizeInBits() != NumEltBits;

  // Check if this is a broadcast of a scalar. We special case lowering
  // for scalars so that we can more effectively fold with loads.
  // If the original value has a larger element type than the shuffle, the
  // broadcast element is in essence truncated. Make that explicit to ease
  // folding.
  if (BitCastSrc && VT.isInteger())
    if (SDValue TruncBroadcast = lowerShuffleAsTruncBroadcast(
            DL, VT, V, BroadcastIdx, Subtarget, DAG))
      return TruncBroadcast;

  MVT BroadcastVT = VT;

  // Also check the simpler case, where we can directly reuse the scalar.
  if (!BitCastSrc &&
      ((V.getOpcode() == ISD::BUILD_VECTOR && V.hasOneUse()) ||
       (V.getOpcode() == ISD::SCALAR_TO_VECTOR && BroadcastIdx == 0))) {
    V = V.getOperand(BroadcastIdx);

    // If we can't broadcast from a register, check that the input is a load.
    if (!BroadcastFromReg && !isShuffleFoldableLoad(V))
      return SDValue();
  } else if (MayFoldLoad(V) && cast<LoadSDNode>(V)->isSimple()) {
    // 32-bit targets need to load i64 as a f64 and then bitcast the result.
    if (!Subtarget.is64Bit() && VT.getScalarType() == MVT::i64) {
      BroadcastVT = MVT::getVectorVT(MVT::f64, VT.getVectorNumElements());
      Opcode = (BroadcastVT.is128BitVector() && !Subtarget.hasAVX2())
                   ? X86ISD::MOVDDUP
                   : Opcode;
    }

    // If we are broadcasting a load that is only used by the shuffle
    // then we can reduce the vector load to the broadcasted scalar load.
    LoadSDNode *Ld = cast<LoadSDNode>(V);
    SDValue BaseAddr = Ld->getOperand(1);
    EVT SVT = BroadcastVT.getScalarType();
    unsigned Offset = BroadcastIdx * SVT.getStoreSize();
    assert((int)(Offset * 8) == BitOffset && "Unexpected bit-offset");
    SDValue NewAddr = DAG.getMemBasePlusOffset(BaseAddr, Offset, DL);
    V = DAG.getLoad(SVT, DL, Ld->getChain(), NewAddr,
                    DAG.getMachineFunction().getMachineMemOperand(
                        Ld->getMemOperand(), Offset, SVT.getStoreSize()));
    DAG.makeEquivalentMemoryOrdering(Ld, V);
  } else if (!BroadcastFromReg) {
    // We can't broadcast from a vector register.
    return SDValue();
  } else if (BitOffset != 0) {
    // We can only broadcast from the zero-element of a vector register,
    // but it can be advantageous to broadcast from the zero-element of a
    // subvector.
    if (!VT.is256BitVector() && !VT.is512BitVector())
      return SDValue();

    // VPERMQ/VPERMPD can perform the cross-lane shuffle directly.
    if (VT == MVT::v4f64 || VT == MVT::v4i64)
      return SDValue();

    // Only broadcast the zero-element of a 128-bit subvector.
    if ((BitOffset % 128) != 0)
      return SDValue();

    assert((BitOffset % V.getScalarValueSizeInBits()) == 0 &&
           "Unexpected bit-offset");
    assert((V.getValueSizeInBits() == 256 || V.getValueSizeInBits() == 512) &&
           "Unexpected vector size");
    unsigned ExtractIdx = BitOffset / V.getScalarValueSizeInBits();
    V = extract128BitVector(V, ExtractIdx, DAG, DL);
  }

  if (Opcode == X86ISD::MOVDDUP && !V.getValueType().isVector())
    V = DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, MVT::v2f64,
                    DAG.getBitcast(MVT::f64, V));

  // Bitcast back to the same scalar type as BroadcastVT.
  if (V.getValueType().getScalarType() != BroadcastVT.getScalarType()) {
    assert(NumEltBits == BroadcastVT.getScalarSizeInBits() &&
           "Unexpected vector element size");
    MVT ExtVT;
    if (V.getValueType().isVector()) {
      unsigned NumSrcElts = V.getValueSizeInBits() / NumEltBits;
      ExtVT = MVT::getVectorVT(BroadcastVT.getScalarType(), NumSrcElts);
    } else {
      ExtVT = BroadcastVT.getScalarType();
    }
    V = DAG.getBitcast(ExtVT, V);
  }

  // 32-bit targets need to load i64 as a f64 and then bitcast the result.
  if (!Subtarget.is64Bit() && V.getValueType() == MVT::i64) {
    V = DAG.getBitcast(MVT::f64, V);
    unsigned NumBroadcastElts = BroadcastVT.getVectorNumElements();
    BroadcastVT = MVT::getVectorVT(MVT::f64, NumBroadcastElts);
  }

  // We only support broadcasting from 128-bit vectors to minimize the
  // number of patterns we need to deal with in isel. So extract down to
  // 128-bits, removing as many bitcasts as possible.
  if (V.getValueSizeInBits() > 128) {
    MVT ExtVT = V.getSimpleValueType().getScalarType();
    ExtVT = MVT::getVectorVT(ExtVT, 128 / ExtVT.getScalarSizeInBits());
    V = extract128BitVector(peekThroughBitcasts(V), 0, DAG, DL);
    V = DAG.getBitcast(ExtVT, V);
  }

  return DAG.getBitcast(VT, DAG.getNode(Opcode, DL, BroadcastVT, V));
}

// Check for whether we can use INSERTPS to perform the shuffle. We only use
// INSERTPS when the V1 elements are already in the correct locations
// because otherwise we can just always use two SHUFPS instructions which
// are much smaller to encode than a SHUFPS and an INSERTPS. We can also
// perform INSERTPS if a single V1 element is out of place and all V2
// elements are zeroable.
static bool matchShuffleAsInsertPS(SDValue &V1, SDValue &V2,
                                   unsigned &InsertPSMask,
                                   const APInt &Zeroable,
                                   ArrayRef<int> Mask, SelectionDAG &DAG) {
  assert(V1.getSimpleValueType().is128BitVector() && "Bad operand type!");
  assert(V2.getSimpleValueType().is128BitVector() && "Bad operand type!");
  assert(Mask.size() == 4 && "Unexpected mask size for v4 shuffle!");

  // Attempt to match INSERTPS with one element from VA or VB being
  // inserted into VA (or undef). If successful, V1, V2 and InsertPSMask
  // are updated.
  auto matchAsInsertPS = [&](SDValue VA, SDValue VB,
                             ArrayRef<int> CandidateMask) {
    unsigned ZMask = 0;
    int VADstIndex = -1;
    int VBDstIndex = -1;
    bool VAUsedInPlace = false;

    for (int i = 0; i < 4; ++i) {
      // Synthesize a zero mask from the zeroable elements (includes undefs).
      if (Zeroable[i]) {
        ZMask |= 1 << i;
        continue;
      }

      // Flag if we use any VA inputs in place.
      if (i == CandidateMask[i]) {
        VAUsedInPlace = true;
        continue;
      }

      // We can only insert a single non-zeroable element.
      if (VADstIndex >= 0 || VBDstIndex >= 0)
        return false;

      if (CandidateMask[i] < 4) {
        // VA input out of place for insertion.
        VADstIndex = i;
      } else {
        // VB input for insertion.
        VBDstIndex = i;
      }
    }

    // Don't bother if we have no (non-zeroable) element for insertion.
    if (VADstIndex < 0 && VBDstIndex < 0)
      return false;

    // Determine element insertion src/dst indices. The src index is from the
    // start of the inserted vector, not the start of the concatenated vector.
    unsigned VBSrcIndex = 0;
    if (VADstIndex >= 0) {
      // If we have a VA input out of place, we use VA as the V2 element
      // insertion and don't use the original V2 at all.
      VBSrcIndex = CandidateMask[VADstIndex];
      VBDstIndex = VADstIndex;
      VB = VA;
    } else {
      VBSrcIndex = CandidateMask[VBDstIndex] - 4;
    }

    // If no V1 inputs are used in place, then the result is created only from
    // the zero mask and the V2 insertion - so remove V1 dependency.
    if (!VAUsedInPlace)
      VA = DAG.getUNDEF(MVT::v4f32);

    // Update V1, V2 and InsertPSMask accordingly.
    V1 = VA;
    V2 = VB;

    // Insert the V2 element into the desired position.
    InsertPSMask = VBSrcIndex << 6 | VBDstIndex << 4 | ZMask;
    assert((InsertPSMask & ~0xFFu) == 0 && "Invalid mask!");
    return true;
  };

  if (matchAsInsertPS(V1, V2, Mask))
    return true;

  // Commute and try again.
  SmallVector<int, 4> CommutedMask(Mask.begin(), Mask.end());
  ShuffleVectorSDNode::commuteMask(CommutedMask);
  if (matchAsInsertPS(V2, V1, CommutedMask))
    return true;

  return false;
}

static SDValue lowerShuffleAsInsertPS(const SDLoc &DL, SDValue V1, SDValue V2,
                                      ArrayRef<int> Mask, const APInt &Zeroable,
                                      SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v4f32 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v4f32 && "Bad operand type!");

  // Attempt to match the insertps pattern.
  unsigned InsertPSMask;
  if (!matchShuffleAsInsertPS(V1, V2, InsertPSMask, Zeroable, Mask, DAG))
    return SDValue();

  // Insert the V2 element into the desired position.
  return DAG.getNode(X86ISD::INSERTPS, DL, MVT::v4f32, V1, V2,
                     DAG.getConstant(InsertPSMask, DL, MVT::i8));
}

/// Try to lower a shuffle as a permute of the inputs followed by an
/// UNPCK instruction.
///
/// This specifically targets cases where we end up with alternating between
/// the two inputs, and so can permute them into something that feeds a single
/// UNPCK instruction. Note that this routine only targets integer vectors
/// because for floating point vectors we have a generalized SHUFPS lowering
/// strategy that handles everything that doesn't *exactly* match an unpack,
/// making this clever lowering unnecessary.
static SDValue lowerShuffleAsPermuteAndUnpack(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const X86Subtarget &Subtarget, SelectionDAG &DAG) {
  assert(!VT.isFloatingPoint() &&
         "This routine only supports integer vectors.");
  assert(VT.is128BitVector() &&
         "This routine only works on 128-bit vectors.");
  assert(!V2.isUndef() &&
         "This routine should only be used when blending two inputs.");
  assert(Mask.size() >= 2 && "Single element masks are invalid.");

  int Size = Mask.size();

  int NumLoInputs =
      count_if(Mask, [Size](int M) { return M >= 0 && M % Size < Size / 2; });
  int NumHiInputs =
      count_if(Mask, [Size](int M) { return M % Size >= Size / 2; });

  bool UnpackLo = NumLoInputs >= NumHiInputs;

  auto TryUnpack = [&](int ScalarSize, int Scale) {
    SmallVector<int, 16> V1Mask((unsigned)Size, -1);
    SmallVector<int, 16> V2Mask((unsigned)Size, -1);

    for (int i = 0; i < Size; ++i) {
      if (Mask[i] < 0)
        continue;

      // Each element of the unpack contains Scale elements from this mask.
      int UnpackIdx = i / Scale;

      // We only handle the case where V1 feeds the first slots of the unpack.
      // We rely on canonicalization to ensure this is the case.
      if ((UnpackIdx % 2 == 0) != (Mask[i] < Size))
        return SDValue();

      // Setup the mask for this input. The indexing is tricky as we have to
      // handle the unpack stride.
      SmallVectorImpl<int> &VMask = (UnpackIdx % 2 == 0) ? V1Mask : V2Mask;
      VMask[(UnpackIdx / 2) * Scale + i % Scale + (UnpackLo ? 0 : Size / 2)] =
          Mask[i] % Size;
    }

    // If we will have to shuffle both inputs to use the unpack, check whether
    // we can just unpack first and shuffle the result. If so, skip this unpack.
    if ((NumLoInputs == 0 || NumHiInputs == 0) && !isNoopShuffleMask(V1Mask) &&
        !isNoopShuffleMask(V2Mask))
      return SDValue();

    // Shuffle the inputs into place.
    V1 = DAG.getVectorShuffle(VT, DL, V1, DAG.getUNDEF(VT), V1Mask);
    V2 = DAG.getVectorShuffle(VT, DL, V2, DAG.getUNDEF(VT), V2Mask);

    // Cast the inputs to the type we will use to unpack them.
    MVT UnpackVT = MVT::getVectorVT(MVT::getIntegerVT(ScalarSize), Size / Scale);
    V1 = DAG.getBitcast(UnpackVT, V1);
    V2 = DAG.getBitcast(UnpackVT, V2);

    // Unpack the inputs and cast the result back to the desired type.
    return DAG.getBitcast(
        VT, DAG.getNode(UnpackLo ? X86ISD::UNPCKL : X86ISD::UNPCKH, DL,
                        UnpackVT, V1, V2));
  };

  // We try each unpack from the largest to the smallest to try and find one
  // that fits this mask.
  int OrigScalarSize = VT.getScalarSizeInBits();
  for (int ScalarSize = 64; ScalarSize >= OrigScalarSize; ScalarSize /= 2)
    if (SDValue Unpack = TryUnpack(ScalarSize, ScalarSize / OrigScalarSize))
      return Unpack;

  // If we're shuffling with a zero vector then we're better off not doing
  // VECTOR_SHUFFLE(UNPCK()) as we lose track of those zero elements.
  if (ISD::isBuildVectorAllZeros(V1.getNode()) ||
      ISD::isBuildVectorAllZeros(V2.getNode()))
    return SDValue();

  // If none of the unpack-rooted lowerings worked (or were profitable) try an
  // initial unpack.
  if (NumLoInputs == 0 || NumHiInputs == 0) {
    assert((NumLoInputs > 0 || NumHiInputs > 0) &&
           "We have to have *some* inputs!");
    int HalfOffset = NumLoInputs == 0 ? Size / 2 : 0;

    // FIXME: We could consider the total complexity of the permute of each
    // possible unpacking. Or at the least we should consider how many
    // half-crossings are created.
    // FIXME: We could consider commuting the unpacks.

    SmallVector<int, 32> PermMask((unsigned)Size, -1);
    for (int i = 0; i < Size; ++i) {
      if (Mask[i] < 0)
        continue;

      assert(Mask[i] % Size >= HalfOffset && "Found input from wrong half!");

      PermMask[i] =
          2 * ((Mask[i] % Size) - HalfOffset) + (Mask[i] < Size ? 0 : 1);
    }
    return DAG.getVectorShuffle(
        VT, DL, DAG.getNode(NumLoInputs == 0 ? X86ISD::UNPCKH : X86ISD::UNPCKL,
                            DL, VT, V1, V2),
        DAG.getUNDEF(VT), PermMask);
  }

  return SDValue();
}

/// Handle lowering of 2-lane 64-bit floating point shuffles.
///
/// This is the basis function for the 2-lane 64-bit shuffles as we have full
/// support for floating point shuffles but not integer shuffles. These
/// instructions will incur a domain crossing penalty on some chips though so
/// it is better to avoid lowering through this for integer vectors where
/// possible.
static SDValue lowerV2F64Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v2f64 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v2f64 && "Bad operand type!");
  assert(Mask.size() == 2 && "Unexpected mask size for v2 shuffle!");

  if (V2.isUndef()) {
    // Check for being able to broadcast a single element.
    if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v2f64, V1, V2,
                                                    Mask, Subtarget, DAG))
      return Broadcast;

    // Straight shuffle of a single input vector. Simulate this by using the
    // single input as both of the "inputs" to this instruction..
    unsigned SHUFPDMask = (Mask[0] == 1) | ((Mask[1] == 1) << 1);

    if (Subtarget.hasAVX()) {
      // If we have AVX, we can use VPERMILPS which will allow folding a load
      // into the shuffle.
      return DAG.getNode(X86ISD::VPERMILPI, DL, MVT::v2f64, V1,
                         DAG.getConstant(SHUFPDMask, DL, MVT::i8));
    }

    return DAG.getNode(
        X86ISD::SHUFP, DL, MVT::v2f64,
        Mask[0] == SM_SentinelUndef ? DAG.getUNDEF(MVT::v2f64) : V1,
        Mask[1] == SM_SentinelUndef ? DAG.getUNDEF(MVT::v2f64) : V1,
        DAG.getConstant(SHUFPDMask, DL, MVT::i8));
  }
  assert(Mask[0] >= 0 && "No undef lanes in multi-input v2 shuffles!");
  assert(Mask[1] >= 0 && "No undef lanes in multi-input v2 shuffles!");
  assert(Mask[0] < 2 && "We sort V1 to be the first input.");
  assert(Mask[1] >= 2 && "We sort V2 to be the second input.");

  if (Subtarget.hasAVX2())
    if (SDValue Extract = lowerShuffleOfExtractsAsVperm(DL, V1, V2, Mask, DAG))
      return Extract;

  // When loading a scalar and then shuffling it into a vector we can often do
  // the insertion cheaply.
  if (SDValue Insertion = lowerShuffleAsElementInsertion(
          DL, MVT::v2f64, V1, V2, Mask, Zeroable, Subtarget, DAG))
    return Insertion;
  // Try inverting the insertion since for v2 masks it is easy to do and we
  // can't reliably sort the mask one way or the other.
  int InverseMask[2] = {Mask[0] < 0 ? -1 : (Mask[0] ^ 2),
                        Mask[1] < 0 ? -1 : (Mask[1] ^ 2)};
  if (SDValue Insertion = lowerShuffleAsElementInsertion(
          DL, MVT::v2f64, V2, V1, InverseMask, Zeroable, Subtarget, DAG))
    return Insertion;

  // Try to use one of the special instruction patterns to handle two common
  // blend patterns if a zero-blend above didn't work.
  if (isShuffleEquivalent(V1, V2, Mask, {0, 3}) ||
      isShuffleEquivalent(V1, V2, Mask, {1, 3}))
    if (SDValue V1S = getScalarValueForVectorElement(V1, Mask[0], DAG))
      // We can either use a special instruction to load over the low double or
      // to move just the low double.
      return DAG.getNode(
          X86ISD::MOVSD, DL, MVT::v2f64, V2,
          DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, MVT::v2f64, V1S));

  if (Subtarget.hasSSE41())
    if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v2f64, V1, V2, Mask,
                                            Zeroable, Subtarget, DAG))
      return Blend;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v2f64, Mask, V1, V2, DAG))
    return V;

  unsigned SHUFPDMask = (Mask[0] == 1) | (((Mask[1] - 2) == 1) << 1);
  return DAG.getNode(X86ISD::SHUFP, DL, MVT::v2f64, V1, V2,
                     DAG.getConstant(SHUFPDMask, DL, MVT::i8));
}

/// Handle lowering of 2-lane 64-bit integer shuffles.
///
/// Tries to lower a 2-lane 64-bit shuffle using shuffle operations provided by
/// the integer unit to minimize domain crossing penalties. However, for blends
/// it falls back to the floating point shuffle operation with appropriate bit
/// casting.
static SDValue lowerV2I64Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v2i64 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v2i64 && "Bad operand type!");
  assert(Mask.size() == 2 && "Unexpected mask size for v2 shuffle!");

  if (V2.isUndef()) {
    // Check for being able to broadcast a single element.
    if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v2i64, V1, V2,
                                                    Mask, Subtarget, DAG))
      return Broadcast;

    // Straight shuffle of a single input vector. For everything from SSE2
    // onward this has a single fast instruction with no scary immediates.
    // We have to map the mask as it is actually a v4i32 shuffle instruction.
    V1 = DAG.getBitcast(MVT::v4i32, V1);
    int WidenedMask[4] = {
        std::max(Mask[0], 0) * 2, std::max(Mask[0], 0) * 2 + 1,
        std::max(Mask[1], 0) * 2, std::max(Mask[1], 0) * 2 + 1};
    return DAG.getBitcast(
        MVT::v2i64,
        DAG.getNode(X86ISD::PSHUFD, DL, MVT::v4i32, V1,
                    getV4X86ShuffleImm8ForMask(WidenedMask, DL, DAG)));
  }
  assert(Mask[0] != -1 && "No undef lanes in multi-input v2 shuffles!");
  assert(Mask[1] != -1 && "No undef lanes in multi-input v2 shuffles!");
  assert(Mask[0] < 2 && "We sort V1 to be the first input.");
  assert(Mask[1] >= 2 && "We sort V2 to be the second input.");

  if (Subtarget.hasAVX2())
    if (SDValue Extract = lowerShuffleOfExtractsAsVperm(DL, V1, V2, Mask, DAG))
      return Extract;

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v2i64, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // When loading a scalar and then shuffling it into a vector we can often do
  // the insertion cheaply.
  if (SDValue Insertion = lowerShuffleAsElementInsertion(
          DL, MVT::v2i64, V1, V2, Mask, Zeroable, Subtarget, DAG))
    return Insertion;
  // Try inverting the insertion since for v2 masks it is easy to do and we
  // can't reliably sort the mask one way or the other.
  int InverseMask[2] = {Mask[0] ^ 2, Mask[1] ^ 2};
  if (SDValue Insertion = lowerShuffleAsElementInsertion(
          DL, MVT::v2i64, V2, V1, InverseMask, Zeroable, Subtarget, DAG))
    return Insertion;

  // We have different paths for blend lowering, but they all must use the
  // *exact* same predicate.
  bool IsBlendSupported = Subtarget.hasSSE41();
  if (IsBlendSupported)
    if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v2i64, V1, V2, Mask,
                                            Zeroable, Subtarget, DAG))
      return Blend;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v2i64, Mask, V1, V2, DAG))
    return V;

  // Try to use byte rotation instructions.
  // Its more profitable for pre-SSSE3 to use shuffles/unpacks.
  if (Subtarget.hasSSSE3()) {
    if (Subtarget.hasVLX())
      if (SDValue Rotate = lowerShuffleAsRotate(DL, MVT::v2i64, V1, V2, Mask,
                                                Subtarget, DAG))
        return Rotate;

    if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v2i64, V1, V2, Mask,
                                                  Subtarget, DAG))
      return Rotate;
  }

  // If we have direct support for blends, we should lower by decomposing into
  // a permute. That will be faster than the domain cross.
  if (IsBlendSupported)
    return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v2i64, V1, V2, Mask,
                                                Subtarget, DAG);

  // We implement this with SHUFPD which is pretty lame because it will likely
  // incur 2 cycles of stall for integer vectors on Nehalem and older chips.
  // However, all the alternatives are still more cycles and newer chips don't
  // have this problem. It would be really nice if x86 had better shuffles here.
  V1 = DAG.getBitcast(MVT::v2f64, V1);
  V2 = DAG.getBitcast(MVT::v2f64, V2);
  return DAG.getBitcast(MVT::v2i64,
                        DAG.getVectorShuffle(MVT::v2f64, DL, V1, V2, Mask));
}

/// Lower a vector shuffle using the SHUFPS instruction.
///
/// This is a helper routine dedicated to lowering vector shuffles using SHUFPS.
/// It makes no assumptions about whether this is the *best* lowering, it simply
/// uses it.
static SDValue lowerShuffleWithSHUFPS(const SDLoc &DL, MVT VT,
                                      ArrayRef<int> Mask, SDValue V1,
                                      SDValue V2, SelectionDAG &DAG) {
  SDValue LowV = V1, HighV = V2;
  int NewMask[4] = {Mask[0], Mask[1], Mask[2], Mask[3]};

  int NumV2Elements = count_if(Mask, [](int M) { return M >= 4; });

  if (NumV2Elements == 1) {
    int V2Index = find_if(Mask, [](int M) { return M >= 4; }) - Mask.begin();

    // Compute the index adjacent to V2Index and in the same half by toggling
    // the low bit.
    int V2AdjIndex = V2Index ^ 1;

    if (Mask[V2AdjIndex] < 0) {
      // Handles all the cases where we have a single V2 element and an undef.
      // This will only ever happen in the high lanes because we commute the
      // vector otherwise.
      if (V2Index < 2)
        std::swap(LowV, HighV);
      NewMask[V2Index] -= 4;
    } else {
      // Handle the case where the V2 element ends up adjacent to a V1 element.
      // To make this work, blend them together as the first step.
      int V1Index = V2AdjIndex;
      int BlendMask[4] = {Mask[V2Index] - 4, 0, Mask[V1Index], 0};
      V2 = DAG.getNode(X86ISD::SHUFP, DL, VT, V2, V1,
                       getV4X86ShuffleImm8ForMask(BlendMask, DL, DAG));

      // Now proceed to reconstruct the final blend as we have the necessary
      // high or low half formed.
      if (V2Index < 2) {
        LowV = V2;
        HighV = V1;
      } else {
        HighV = V2;
      }
      NewMask[V1Index] = 2; // We put the V1 element in V2[2].
      NewMask[V2Index] = 0; // We shifted the V2 element into V2[0].
    }
  } else if (NumV2Elements == 2) {
    if (Mask[0] < 4 && Mask[1] < 4) {
      // Handle the easy case where we have V1 in the low lanes and V2 in the
      // high lanes.
      NewMask[2] -= 4;
      NewMask[3] -= 4;
    } else if (Mask[2] < 4 && Mask[3] < 4) {
      // We also handle the reversed case because this utility may get called
      // when we detect a SHUFPS pattern but can't easily commute the shuffle to
      // arrange things in the right direction.
      NewMask[0] -= 4;
      NewMask[1] -= 4;
      HighV = V1;
      LowV = V2;
    } else {
      // We have a mixture of V1 and V2 in both low and high lanes. Rather than
      // trying to place elements directly, just blend them and set up the final
      // shuffle to place them.

      // The first two blend mask elements are for V1, the second two are for
      // V2.
      int BlendMask[4] = {Mask[0] < 4 ? Mask[0] : Mask[1],
                          Mask[2] < 4 ? Mask[2] : Mask[3],
                          (Mask[0] >= 4 ? Mask[0] : Mask[1]) - 4,
                          (Mask[2] >= 4 ? Mask[2] : Mask[3]) - 4};
      V1 = DAG.getNode(X86ISD::SHUFP, DL, VT, V1, V2,
                       getV4X86ShuffleImm8ForMask(BlendMask, DL, DAG));

      // Now we do a normal shuffle of V1 by giving V1 as both operands to
      // a blend.
      LowV = HighV = V1;
      NewMask[0] = Mask[0] < 4 ? 0 : 2;
      NewMask[1] = Mask[0] < 4 ? 2 : 0;
      NewMask[2] = Mask[2] < 4 ? 1 : 3;
      NewMask[3] = Mask[2] < 4 ? 3 : 1;
    }
  }
  return DAG.getNode(X86ISD::SHUFP, DL, VT, LowV, HighV,
                     getV4X86ShuffleImm8ForMask(NewMask, DL, DAG));
}

/// Lower 4-lane 32-bit floating point shuffles.
///
/// Uses instructions exclusively from the floating point unit to minimize
/// domain crossing penalties, as these are sufficient to implement all v4f32
/// shuffles.
static SDValue lowerV4F32Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v4f32 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v4f32 && "Bad operand type!");
  assert(Mask.size() == 4 && "Unexpected mask size for v4 shuffle!");

  int NumV2Elements = count_if(Mask, [](int M) { return M >= 4; });

  if (NumV2Elements == 0) {
    // Check for being able to broadcast a single element.
    if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v4f32, V1, V2,
                                                    Mask, Subtarget, DAG))
      return Broadcast;

    // Use even/odd duplicate instructions for masks that match their pattern.
    if (Subtarget.hasSSE3()) {
      if (isShuffleEquivalent(V1, V2, Mask, {0, 0, 2, 2}))
        return DAG.getNode(X86ISD::MOVSLDUP, DL, MVT::v4f32, V1);
      if (isShuffleEquivalent(V1, V2, Mask, {1, 1, 3, 3}))
        return DAG.getNode(X86ISD::MOVSHDUP, DL, MVT::v4f32, V1);
    }

    if (Subtarget.hasAVX()) {
      // If we have AVX, we can use VPERMILPS which will allow folding a load
      // into the shuffle.
      return DAG.getNode(X86ISD::VPERMILPI, DL, MVT::v4f32, V1,
                         getV4X86ShuffleImm8ForMask(Mask, DL, DAG));
    }

    // Use MOVLHPS/MOVHLPS to simulate unary shuffles. These are only valid
    // in SSE1 because otherwise they are widened to v2f64 and never get here.
    if (!Subtarget.hasSSE2()) {
      if (isShuffleEquivalent(V1, V2, Mask, {0, 1, 0, 1}))
        return DAG.getNode(X86ISD::MOVLHPS, DL, MVT::v4f32, V1, V1);
      if (isShuffleEquivalent(V1, V2, Mask, {2, 3, 2, 3}))
        return DAG.getNode(X86ISD::MOVHLPS, DL, MVT::v4f32, V1, V1);
    }

    // Otherwise, use a straight shuffle of a single input vector. We pass the
    // input vector to both operands to simulate this with a SHUFPS.
    return DAG.getNode(X86ISD::SHUFP, DL, MVT::v4f32, V1, V1,
                       getV4X86ShuffleImm8ForMask(Mask, DL, DAG));
  }

  if (Subtarget.hasAVX2())
    if (SDValue Extract = lowerShuffleOfExtractsAsVperm(DL, V1, V2, Mask, DAG))
      return Extract;

  // There are special ways we can lower some single-element blends. However, we
  // have custom ways we can lower more complex single-element blends below that
  // we defer to if both this and BLENDPS fail to match, so restrict this to
  // when the V2 input is targeting element 0 of the mask -- that is the fast
  // case here.
  if (NumV2Elements == 1 && Mask[0] >= 4)
    if (SDValue V = lowerShuffleAsElementInsertion(
            DL, MVT::v4f32, V1, V2, Mask, Zeroable, Subtarget, DAG))
      return V;

  if (Subtarget.hasSSE41()) {
    if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v4f32, V1, V2, Mask,
                                            Zeroable, Subtarget, DAG))
      return Blend;

    // Use INSERTPS if we can complete the shuffle efficiently.
    if (SDValue V = lowerShuffleAsInsertPS(DL, V1, V2, Mask, Zeroable, DAG))
      return V;

    if (!isSingleSHUFPSMask(Mask))
      if (SDValue BlendPerm = lowerShuffleAsBlendAndPermute(DL, MVT::v4f32, V1,
                                                            V2, Mask, DAG))
        return BlendPerm;
  }

  // Use low/high mov instructions. These are only valid in SSE1 because
  // otherwise they are widened to v2f64 and never get here.
  if (!Subtarget.hasSSE2()) {
    if (isShuffleEquivalent(V1, V2, Mask, {0, 1, 4, 5}))
      return DAG.getNode(X86ISD::MOVLHPS, DL, MVT::v4f32, V1, V2);
    if (isShuffleEquivalent(V1, V2, Mask, {2, 3, 6, 7}))
      return DAG.getNode(X86ISD::MOVHLPS, DL, MVT::v4f32, V2, V1);
  }

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v4f32, Mask, V1, V2, DAG))
    return V;

  // Otherwise fall back to a SHUFPS lowering strategy.
  return lowerShuffleWithSHUFPS(DL, MVT::v4f32, Mask, V1, V2, DAG);
}

/// Lower 4-lane i32 vector shuffles.
///
/// We try to handle these with integer-domain shuffles where we can, but for
/// blends we use the floating point domain blend instructions.
static SDValue lowerV4I32Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v4i32 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v4i32 && "Bad operand type!");
  assert(Mask.size() == 4 && "Unexpected mask size for v4 shuffle!");

  // Whenever we can lower this as a zext, that instruction is strictly faster
  // than any alternative. It also allows us to fold memory operands into the
  // shuffle in many cases.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(DL, MVT::v4i32, V1, V2, Mask,
                                                   Zeroable, Subtarget, DAG))
    return ZExt;

  int NumV2Elements = count_if(Mask, [](int M) { return M >= 4; });

  if (NumV2Elements == 0) {
    // Try to use broadcast unless the mask only has one non-undef element.
    if (count_if(Mask, [](int M) { return M >= 0 && M < 4; }) > 1) {
      if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v4i32, V1, V2,
                                                      Mask, Subtarget, DAG))
        return Broadcast;
    }

    // Straight shuffle of a single input vector. For everything from SSE2
    // onward this has a single fast instruction with no scary immediates.
    // We coerce the shuffle pattern to be compatible with UNPCK instructions
    // but we aren't actually going to use the UNPCK instruction because doing
    // so prevents folding a load into this instruction or making a copy.
    const int UnpackLoMask[] = {0, 0, 1, 1};
    const int UnpackHiMask[] = {2, 2, 3, 3};
    if (isShuffleEquivalent(V1, V2, Mask, {0, 0, 1, 1}))
      Mask = UnpackLoMask;
    else if (isShuffleEquivalent(V1, V2, Mask, {2, 2, 3, 3}))
      Mask = UnpackHiMask;

    return DAG.getNode(X86ISD::PSHUFD, DL, MVT::v4i32, V1,
                       getV4X86ShuffleImm8ForMask(Mask, DL, DAG));
  }

  if (Subtarget.hasAVX2())
    if (SDValue Extract = lowerShuffleOfExtractsAsVperm(DL, V1, V2, Mask, DAG))
      return Extract;

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v4i32, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // There are special ways we can lower some single-element blends.
  if (NumV2Elements == 1)
    if (SDValue V = lowerShuffleAsElementInsertion(
            DL, MVT::v4i32, V1, V2, Mask, Zeroable, Subtarget, DAG))
      return V;

  // We have different paths for blend lowering, but they all must use the
  // *exact* same predicate.
  bool IsBlendSupported = Subtarget.hasSSE41();
  if (IsBlendSupported)
    if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v4i32, V1, V2, Mask,
                                            Zeroable, Subtarget, DAG))
      return Blend;

  if (SDValue Masked = lowerShuffleAsBitMask(DL, MVT::v4i32, V1, V2, Mask,
                                             Zeroable, Subtarget, DAG))
    return Masked;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v4i32, Mask, V1, V2, DAG))
    return V;

  // Try to use byte rotation instructions.
  // Its more profitable for pre-SSSE3 to use shuffles/unpacks.
  if (Subtarget.hasSSSE3()) {
    if (Subtarget.hasVLX())
      if (SDValue Rotate = lowerShuffleAsRotate(DL, MVT::v4i32, V1, V2, Mask,
                                                Subtarget, DAG))
        return Rotate;

    if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v4i32, V1, V2, Mask,
                                                  Subtarget, DAG))
      return Rotate;
  }

  // Assume that a single SHUFPS is faster than an alternative sequence of
  // multiple instructions (even if the CPU has a domain penalty).
  // If some CPU is harmed by the domain switch, we can fix it in a later pass.
  if (!isSingleSHUFPSMask(Mask)) {
    // If we have direct support for blends, we should lower by decomposing into
    // a permute. That will be faster than the domain cross.
    if (IsBlendSupported)
      return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v4i32, V1, V2, Mask,
                                                  Subtarget, DAG);

    // Try to lower by permuting the inputs into an unpack instruction.
    if (SDValue Unpack = lowerShuffleAsPermuteAndUnpack(DL, MVT::v4i32, V1, V2,
                                                        Mask, Subtarget, DAG))
      return Unpack;
  }

  // We implement this with SHUFPS because it can blend from two vectors.
  // Because we're going to eventually use SHUFPS, we use SHUFPS even to build
  // up the inputs, bypassing domain shift penalties that we would incur if we
  // directly used PSHUFD on Nehalem and older. For newer chips, this isn't
  // relevant.
  SDValue CastV1 = DAG.getBitcast(MVT::v4f32, V1);
  SDValue CastV2 = DAG.getBitcast(MVT::v4f32, V2);
  SDValue ShufPS = DAG.getVectorShuffle(MVT::v4f32, DL, CastV1, CastV2, Mask);
  return DAG.getBitcast(MVT::v4i32, ShufPS);
}

/// Lowering of single-input v8i16 shuffles is the cornerstone of SSE2
/// shuffle lowering, and the most complex part.
///
/// The lowering strategy is to try to form pairs of input lanes which are
/// targeted at the same half of the final vector, and then use a dword shuffle
/// to place them onto the right half, and finally unpack the paired lanes into
/// their final position.
///
/// The exact breakdown of how to form these dword pairs and align them on the
/// correct sides is really tricky. See the comments within the function for
/// more of the details.
///
/// This code also handles repeated 128-bit lanes of v8i16 shuffles, but each
/// lane must shuffle the *exact* same way. In fact, you must pass a v8 Mask to
/// this routine for it to work correctly. To shuffle a 256-bit or 512-bit i16
/// vector, form the analogous 128-bit 8-element Mask.
static SDValue lowerV8I16GeneralSingleInputShuffle(
    const SDLoc &DL, MVT VT, SDValue V, MutableArrayRef<int> Mask,
    const X86Subtarget &Subtarget, SelectionDAG &DAG) {
  assert(VT.getVectorElementType() == MVT::i16 && "Bad input type!");
  MVT PSHUFDVT = MVT::getVectorVT(MVT::i32, VT.getVectorNumElements() / 2);

  assert(Mask.size() == 8 && "Shuffle mask length doesn't match!");
  MutableArrayRef<int> LoMask = Mask.slice(0, 4);
  MutableArrayRef<int> HiMask = Mask.slice(4, 4);

  // Attempt to directly match PSHUFLW or PSHUFHW.
  if (isUndefOrInRange(LoMask, 0, 4) &&
      isSequentialOrUndefInRange(HiMask, 0, 4, 4)) {
    return DAG.getNode(X86ISD::PSHUFLW, DL, VT, V,
                       getV4X86ShuffleImm8ForMask(LoMask, DL, DAG));
  }
  if (isUndefOrInRange(HiMask, 4, 8) &&
      isSequentialOrUndefInRange(LoMask, 0, 4, 0)) {
    for (int i = 0; i != 4; ++i)
      HiMask[i] = (HiMask[i] < 0 ? HiMask[i] : (HiMask[i] - 4));
    return DAG.getNode(X86ISD::PSHUFHW, DL, VT, V,
                       getV4X86ShuffleImm8ForMask(HiMask, DL, DAG));
  }

  SmallVector<int, 4> LoInputs;
  copy_if(LoMask, std::back_inserter(LoInputs), [](int M) { return M >= 0; });
  array_pod_sort(LoInputs.begin(), LoInputs.end());
  LoInputs.erase(std::unique(LoInputs.begin(), LoInputs.end()), LoInputs.end());
  SmallVector<int, 4> HiInputs;
  copy_if(HiMask, std::back_inserter(HiInputs), [](int M) { return M >= 0; });
  array_pod_sort(HiInputs.begin(), HiInputs.end());
  HiInputs.erase(std::unique(HiInputs.begin(), HiInputs.end()), HiInputs.end());
  int NumLToL = llvm::lower_bound(LoInputs, 4) - LoInputs.begin();
  int NumHToL = LoInputs.size() - NumLToL;
  int NumLToH = llvm::lower_bound(HiInputs, 4) - HiInputs.begin();
  int NumHToH = HiInputs.size() - NumLToH;
  MutableArrayRef<int> LToLInputs(LoInputs.data(), NumLToL);
  MutableArrayRef<int> LToHInputs(HiInputs.data(), NumLToH);
  MutableArrayRef<int> HToLInputs(LoInputs.data() + NumLToL, NumHToL);
  MutableArrayRef<int> HToHInputs(HiInputs.data() + NumLToH, NumHToH);

  // If we are shuffling values from one half - check how many different DWORD
  // pairs we need to create. If only 1 or 2 then we can perform this as a
  // PSHUFLW/PSHUFHW + PSHUFD instead of the PSHUFD+PSHUFLW+PSHUFHW chain below.
  auto ShuffleDWordPairs = [&](ArrayRef<int> PSHUFHalfMask,
                               ArrayRef<int> PSHUFDMask, unsigned ShufWOp) {
    V = DAG.getNode(ShufWOp, DL, VT, V,
                    getV4X86ShuffleImm8ForMask(PSHUFHalfMask, DL, DAG));
    V = DAG.getBitcast(PSHUFDVT, V);
    V = DAG.getNode(X86ISD::PSHUFD, DL, PSHUFDVT, V,
                    getV4X86ShuffleImm8ForMask(PSHUFDMask, DL, DAG));
    return DAG.getBitcast(VT, V);
  };

  if ((NumHToL + NumHToH) == 0 || (NumLToL + NumLToH) == 0) {
    int PSHUFDMask[4] = { -1, -1, -1, -1 };
    SmallVector<std::pair<int, int>, 4> DWordPairs;
    int DOffset = ((NumHToL + NumHToH) == 0 ? 0 : 2);

    // Collect the different DWORD pairs.
    for (int DWord = 0; DWord != 4; ++DWord) {
      int M0 = Mask[2 * DWord + 0];
      int M1 = Mask[2 * DWord + 1];
      M0 = (M0 >= 0 ? M0 % 4 : M0);
      M1 = (M1 >= 0 ? M1 % 4 : M1);
      if (M0 < 0 && M1 < 0)
        continue;

      bool Match = false;
      for (int j = 0, e = DWordPairs.size(); j < e; ++j) {
        auto &DWordPair = DWordPairs[j];
        if ((M0 < 0 || isUndefOrEqual(DWordPair.first, M0)) &&
            (M1 < 0 || isUndefOrEqual(DWordPair.second, M1))) {
          DWordPair.first = (M0 >= 0 ? M0 : DWordPair.first);
          DWordPair.second = (M1 >= 0 ? M1 : DWordPair.second);
          PSHUFDMask[DWord] = DOffset + j;
          Match = true;
          break;
        }
      }
      if (!Match) {
        PSHUFDMask[DWord] = DOffset + DWordPairs.size();
        DWordPairs.push_back(std::make_pair(M0, M1));
      }
    }

    if (DWordPairs.size() <= 2) {
      DWordPairs.resize(2, std::make_pair(-1, -1));
      int PSHUFHalfMask[4] = {DWordPairs[0].first, DWordPairs[0].second,
                              DWordPairs[1].first, DWordPairs[1].second};
      if ((NumHToL + NumHToH) == 0)
        return ShuffleDWordPairs(PSHUFHalfMask, PSHUFDMask, X86ISD::PSHUFLW);
      if ((NumLToL + NumLToH) == 0)
        return ShuffleDWordPairs(PSHUFHalfMask, PSHUFDMask, X86ISD::PSHUFHW);
    }
  }

  // Simplify the 1-into-3 and 3-into-1 cases with a single pshufd. For all
  // such inputs we can swap two of the dwords across the half mark and end up
  // with <=2 inputs to each half in each half. Once there, we can fall through
  // to the generic code below. For example:
  //
  // Input: [a, b, c, d, e, f, g, h] -PSHUFD[0,2,1,3]-> [a, b, e, f, c, d, g, h]
  // Mask:  [0, 1, 2, 7, 4, 5, 6, 3] -----------------> [0, 1, 4, 7, 2, 3, 6, 5]
  //
  // However in some very rare cases we have a 1-into-3 or 3-into-1 on one half
  // and an existing 2-into-2 on the other half. In this case we may have to
  // pre-shuffle the 2-into-2 half to avoid turning it into a 3-into-1 or
  // 1-into-3 which could cause us to cycle endlessly fixing each side in turn.
  // Fortunately, we don't have to handle anything but a 2-into-2 pattern
  // because any other situation (including a 3-into-1 or 1-into-3 in the other
  // half than the one we target for fixing) will be fixed when we re-enter this
  // path. We will also combine away any sequence of PSHUFD instructions that
  // result into a single instruction. Here is an example of the tricky case:
  //
  // Input: [a, b, c, d, e, f, g, h] -PSHUFD[0,2,1,3]-> [a, b, e, f, c, d, g, h]
  // Mask:  [3, 7, 1, 0, 2, 7, 3, 5] -THIS-IS-BAD!!!!-> [5, 7, 1, 0, 4, 7, 5, 3]
  //
  // This now has a 1-into-3 in the high half! Instead, we do two shuffles:
  //
  // Input: [a, b, c, d, e, f, g, h] PSHUFHW[0,2,1,3]-> [a, b, c, d, e, g, f, h]
  // Mask:  [3, 7, 1, 0, 2, 7, 3, 5] -----------------> [3, 7, 1, 0, 2, 7, 3, 6]
  //
  // Input: [a, b, c, d, e, g, f, h] -PSHUFD[0,2,1,3]-> [a, b, e, g, c, d, f, h]
  // Mask:  [3, 7, 1, 0, 2, 7, 3, 6] -----------------> [5, 7, 1, 0, 4, 7, 5, 6]
  //
  // The result is fine to be handled by the generic logic.
  auto balanceSides = [&](ArrayRef<int> AToAInputs, ArrayRef<int> BToAInputs,
                          ArrayRef<int> BToBInputs, ArrayRef<int> AToBInputs,
                          int AOffset, int BOffset) {
    assert((AToAInputs.size() == 3 || AToAInputs.size() == 1) &&
           "Must call this with A having 3 or 1 inputs from the A half.");
    assert((BToAInputs.size() == 1 || BToAInputs.size() == 3) &&
           "Must call this with B having 1 or 3 inputs from the B half.");
    assert(AToAInputs.size() + BToAInputs.size() == 4 &&
           "Must call this with either 3:1 or 1:3 inputs (summing to 4).");

    bool ThreeAInputs = AToAInputs.size() == 3;

    // Compute the index of dword with only one word among the three inputs in
    // a half by taking the sum of the half with three inputs and subtracting
    // the sum of the actual three inputs. The difference is the remaining
    // slot.
    int ADWord = 0, BDWord = 0;
    int &TripleDWord = ThreeAInputs ? ADWord : BDWord;
    int &OneInputDWord = ThreeAInputs ? BDWord : ADWord;
    int TripleInputOffset = ThreeAInputs ? AOffset : BOffset;
    ArrayRef<int> TripleInputs = ThreeAInputs ? AToAInputs : BToAInputs;
    int OneInput = ThreeAInputs ? BToAInputs[0] : AToAInputs[0];
    int TripleInputSum = 0 + 1 + 2 + 3 + (4 * TripleInputOffset);
    int TripleNonInputIdx =
        TripleInputSum - std::accumulate(TripleInputs.begin(), TripleInputs.end(), 0);
    TripleDWord = TripleNonInputIdx / 2;

    // We use xor with one to compute the adjacent DWord to whichever one the
    // OneInput is in.
    OneInputDWord = (OneInput / 2) ^ 1;

    // Check for one tricky case: We're fixing a 3<-1 or a 1<-3 shuffle for AToA
    // and BToA inputs. If there is also such a problem with the BToB and AToB
    // inputs, we don't try to fix it necessarily -- we'll recurse and see it in
    // the next pass. However, if we have a 2<-2 in the BToB and AToB inputs, it
    // is essential that we don't *create* a 3<-1 as then we might oscillate.
    if (BToBInputs.size() == 2 && AToBInputs.size() == 2) {
      // Compute how many inputs will be flipped by swapping these DWords. We
      // need
      // to balance this to ensure we don't form a 3-1 shuffle in the other
      // half.
      int NumFlippedAToBInputs =
          std::count(AToBInputs.begin(), AToBInputs.end(), 2 * ADWord) +
          std::count(AToBInputs.begin(), AToBInputs.end(), 2 * ADWord + 1);
      int NumFlippedBToBInputs =
          std::count(BToBInputs.begin(), BToBInputs.end(), 2 * BDWord) +
          std::count(BToBInputs.begin(), BToBInputs.end(), 2 * BDWord + 1);
      if ((NumFlippedAToBInputs == 1 &&
           (NumFlippedBToBInputs == 0 || NumFlippedBToBInputs == 2)) ||
          (NumFlippedBToBInputs == 1 &&
           (NumFlippedAToBInputs == 0 || NumFlippedAToBInputs == 2))) {
        // We choose whether to fix the A half or B half based on whether that
        // half has zero flipped inputs. At zero, we may not be able to fix it
        // with that half. We also bias towards fixing the B half because that
        // will more commonly be the high half, and we have to bias one way.
        auto FixFlippedInputs = [&V, &DL, &Mask, &DAG](int PinnedIdx, int DWord,
                                                       ArrayRef<int> Inputs) {
          int FixIdx = PinnedIdx ^ 1; // The adjacent slot to the pinned slot.
          bool IsFixIdxInput = is_contained(Inputs, PinnedIdx ^ 1);
          // Determine whether the free index is in the flipped dword or the
          // unflipped dword based on where the pinned index is. We use this bit
          // in an xor to conditionally select the adjacent dword.
          int FixFreeIdx = 2 * (DWord ^ (PinnedIdx / 2 == DWord));
          bool IsFixFreeIdxInput = is_contained(Inputs, FixFreeIdx);
          if (IsFixIdxInput == IsFixFreeIdxInput)
            FixFreeIdx += 1;
          IsFixFreeIdxInput = is_contained(Inputs, FixFreeIdx);
          assert(IsFixIdxInput != IsFixFreeIdxInput &&
                 "We need to be changing the number of flipped inputs!");
          int PSHUFHalfMask[] = {0, 1, 2, 3};
          std::swap(PSHUFHalfMask[FixFreeIdx % 4], PSHUFHalfMask[FixIdx % 4]);
          V = DAG.getNode(
              FixIdx < 4 ? X86ISD::PSHUFLW : X86ISD::PSHUFHW, DL,
              MVT::getVectorVT(MVT::i16, V.getValueSizeInBits() / 16), V,
              getV4X86ShuffleImm8ForMask(PSHUFHalfMask, DL, DAG));

          for (int &M : Mask)
            if (M >= 0 && M == FixIdx)
              M = FixFreeIdx;
            else if (M >= 0 && M == FixFreeIdx)
              M = FixIdx;
        };
        if (NumFlippedBToBInputs != 0) {
          int BPinnedIdx =
              BToAInputs.size() == 3 ? TripleNonInputIdx : OneInput;
          FixFlippedInputs(BPinnedIdx, BDWord, BToBInputs);
        } else {
          assert(NumFlippedAToBInputs != 0 && "Impossible given predicates!");
          int APinnedIdx = ThreeAInputs ? TripleNonInputIdx : OneInput;
          FixFlippedInputs(APinnedIdx, ADWord, AToBInputs);
        }
      }
    }

    int PSHUFDMask[] = {0, 1, 2, 3};
    PSHUFDMask[ADWord] = BDWord;
    PSHUFDMask[BDWord] = ADWord;
    V = DAG.getBitcast(
        VT,
        DAG.getNode(X86ISD::PSHUFD, DL, PSHUFDVT, DAG.getBitcast(PSHUFDVT, V),
                    getV4X86ShuffleImm8ForMask(PSHUFDMask, DL, DAG)));

    // Adjust the mask to match the new locations of A and B.
    for (int &M : Mask)
      if (M >= 0 && M/2 == ADWord)
        M = 2 * BDWord + M % 2;
      else if (M >= 0 && M/2 == BDWord)
        M = 2 * ADWord + M % 2;

    // Recurse back into this routine to re-compute state now that this isn't
    // a 3 and 1 problem.
    return lowerV8I16GeneralSingleInputShuffle(DL, VT, V, Mask, Subtarget, DAG);
  };
  if ((NumLToL == 3 && NumHToL == 1) || (NumLToL == 1 && NumHToL == 3))
    return balanceSides(LToLInputs, HToLInputs, HToHInputs, LToHInputs, 0, 4);
  if ((NumHToH == 3 && NumLToH == 1) || (NumHToH == 1 && NumLToH == 3))
    return balanceSides(HToHInputs, LToHInputs, LToLInputs, HToLInputs, 4, 0);

  // At this point there are at most two inputs to the low and high halves from
  // each half. That means the inputs can always be grouped into dwords and
  // those dwords can then be moved to the correct half with a dword shuffle.
  // We use at most one low and one high word shuffle to collect these paired
  // inputs into dwords, and finally a dword shuffle to place them.
  int PSHUFLMask[4] = {-1, -1, -1, -1};
  int PSHUFHMask[4] = {-1, -1, -1, -1};
  int PSHUFDMask[4] = {-1, -1, -1, -1};

  // First fix the masks for all the inputs that are staying in their
  // original halves. This will then dictate the targets of the cross-half
  // shuffles.
  auto fixInPlaceInputs =
      [&PSHUFDMask](ArrayRef<int> InPlaceInputs, ArrayRef<int> IncomingInputs,
                    MutableArrayRef<int> SourceHalfMask,
                    MutableArrayRef<int> HalfMask, int HalfOffset) {
    if (InPlaceInputs.empty())
      return;
    if (InPlaceInputs.size() == 1) {
      SourceHalfMask[InPlaceInputs[0] - HalfOffset] =
          InPlaceInputs[0] - HalfOffset;
      PSHUFDMask[InPlaceInputs[0] / 2] = InPlaceInputs[0] / 2;
      return;
    }
    if (IncomingInputs.empty()) {
      // Just fix all of the in place inputs.
      for (int Input : InPlaceInputs) {
        SourceHalfMask[Input - HalfOffset] = Input - HalfOffset;
        PSHUFDMask[Input / 2] = Input / 2;
      }
      return;
    }

    assert(InPlaceInputs.size() == 2 && "Cannot handle 3 or 4 inputs!");
    SourceHalfMask[InPlaceInputs[0] - HalfOffset] =
        InPlaceInputs[0] - HalfOffset;
    // Put the second input next to the first so that they are packed into
    // a dword. We find the adjacent index by toggling the low bit.
    int AdjIndex = InPlaceInputs[0] ^ 1;
    SourceHalfMask[AdjIndex - HalfOffset] = InPlaceInputs[1] - HalfOffset;
    std::replace(HalfMask.begin(), HalfMask.end(), InPlaceInputs[1], AdjIndex);
    PSHUFDMask[AdjIndex / 2] = AdjIndex / 2;
  };
  fixInPlaceInputs(LToLInputs, HToLInputs, PSHUFLMask, LoMask, 0);
  fixInPlaceInputs(HToHInputs, LToHInputs, PSHUFHMask, HiMask, 4);

  // Now gather the cross-half inputs and place them into a free dword of
  // their target half.
  // FIXME: This operation could almost certainly be simplified dramatically to
  // look more like the 3-1 fixing operation.
  auto moveInputsToRightHalf = [&PSHUFDMask](
      MutableArrayRef<int> IncomingInputs, ArrayRef<int> ExistingInputs,
      MutableArrayRef<int> SourceHalfMask, MutableArrayRef<int> HalfMask,
      MutableArrayRef<int> FinalSourceHalfMask, int SourceOffset,
      int DestOffset) {
    auto isWordClobbered = [](ArrayRef<int> SourceHalfMask, int Word) {
      return SourceHalfMask[Word] >= 0 && SourceHalfMask[Word] != Word;
    };
    auto isDWordClobbered = [&isWordClobbered](ArrayRef<int> SourceHalfMask,
                                               int Word) {
      int LowWord = Word & ~1;
      int HighWord = Word | 1;
      return isWordClobbered(SourceHalfMask, LowWord) ||
             isWordClobbered(SourceHalfMask, HighWord);
    };

    if (IncomingInputs.empty())
      return;

    if (ExistingInputs.empty()) {
      // Map any dwords with inputs from them into the right half.
      for (int Input : IncomingInputs) {
        // If the source half mask maps over the inputs, turn those into
        // swaps and use the swapped lane.
        if (isWordClobbered(SourceHalfMask, Input - SourceOffset)) {
          if (SourceHalfMask[SourceHalfMask[Input - SourceOffset]] < 0) {
            SourceHalfMask[SourceHalfMask[Input - SourceOffset]] =
                Input - SourceOffset;
            // We have to swap the uses in our half mask in one sweep.
            for (int &M : HalfMask)
              if (M == SourceHalfMask[Input - SourceOffset] + SourceOffset)
                M = Input;
              else if (M == Input)
                M = SourceHalfMask[Input - SourceOffset] + SourceOffset;
          } else {
            assert(SourceHalfMask[SourceHalfMask[Input - SourceOffset]] ==
                       Input - SourceOffset &&
                   "Previous placement doesn't match!");
          }
          // Note that this correctly re-maps both when we do a swap and when
          // we observe the other side of the swap above. We rely on that to
          // avoid swapping the members of the input list directly.
          Input = SourceHalfMask[Input - SourceOffset] + SourceOffset;
        }

        // Map the input's dword into the correct half.
        if (PSHUFDMask[(Input - SourceOffset + DestOffset) / 2] < 0)
          PSHUFDMask[(Input - SourceOffset + DestOffset) / 2] = Input / 2;
        else
          assert(PSHUFDMask[(Input - SourceOffset + DestOffset) / 2] ==
                     Input / 2 &&
                 "Previous placement doesn't match!");
      }

      // And just directly shift any other-half mask elements to be same-half
      // as we will have mirrored the dword containing the element into the
      // same position within that half.
      for (int &M : HalfMask)
        if (M >= SourceOffset && M < SourceOffset + 4) {
          M = M - SourceOffset + DestOffset;
          assert(M >= 0 && "This should never wrap below zero!");
        }
      return;
    }

    // Ensure we have the input in a viable dword of its current half. This
    // is particularly tricky because the original position may be clobbered
    // by inputs being moved and *staying* in that half.
    if (IncomingInputs.size() == 1) {
      if (isWordClobbered(SourceHalfMask, IncomingInputs[0] - SourceOffset)) {
        int InputFixed = find(SourceHalfMask, -1) - std::begin(SourceHalfMask) +
                         SourceOffset;
        SourceHalfMask[InputFixed - SourceOffset] =
            IncomingInputs[0] - SourceOffset;
        std::replace(HalfMask.begin(), HalfMask.end(), IncomingInputs[0],
                     InputFixed);
        IncomingInputs[0] = InputFixed;
      }
    } else if (IncomingInputs.size() == 2) {
      if (IncomingInputs[0] / 2 != IncomingInputs[1] / 2 ||
          isDWordClobbered(SourceHalfMask, IncomingInputs[0] - SourceOffset)) {
        // We have two non-adjacent or clobbered inputs we need to extract from
        // the source half. To do this, we need to map them into some adjacent
        // dword slot in the source mask.
        int InputsFixed[2] = {IncomingInputs[0] - SourceOffset,
                              IncomingInputs[1] - SourceOffset};

        // If there is a free slot in the source half mask adjacent to one of
        // the inputs, place the other input in it. We use (Index XOR 1) to
        // compute an adjacent index.
        if (!isWordClobbered(SourceHalfMask, InputsFixed[0]) &&
            SourceHalfMask[InputsFixed[0] ^ 1] < 0) {
          SourceHalfMask[InputsFixed[0]] = InputsFixed[0];
          SourceHalfMask[InputsFixed[0] ^ 1] = InputsFixed[1];
          InputsFixed[1] = InputsFixed[0] ^ 1;
        } else if (!isWordClobbered(SourceHalfMask, InputsFixed[1]) &&
                   SourceHalfMask[InputsFixed[1] ^ 1] < 0) {
          SourceHalfMask[InputsFixed[1]] = InputsFixed[1];
          SourceHalfMask[InputsFixed[1] ^ 1] = InputsFixed[0];
          InputsFixed[0] = InputsFixed[1] ^ 1;
        } else if (SourceHalfMask[2 * ((InputsFixed[0] / 2) ^ 1)] < 0 &&
                   SourceHalfMask[2 * ((InputsFixed[0] / 2) ^ 1) + 1] < 0) {
          // The two inputs are in the same DWord but it is clobbered and the
          // adjacent DWord isn't used at all. Move both inputs to the free
          // slot.
          SourceHalfMask[2 * ((InputsFixed[0] / 2) ^ 1)] = InputsFixed[0];
          SourceHalfMask[2 * ((InputsFixed[0] / 2) ^ 1) + 1] = InputsFixed[1];
          InputsFixed[0] = 2 * ((InputsFixed[0] / 2) ^ 1);
          InputsFixed[1] = 2 * ((InputsFixed[0] / 2) ^ 1) + 1;
        } else {
          // The only way we hit this point is if there is no clobbering
          // (because there are no off-half inputs to this half) and there is no
          // free slot adjacent to one of the inputs. In this case, we have to
          // swap an input with a non-input.
          for (int i = 0; i < 4; ++i)
            assert((SourceHalfMask[i] < 0 || SourceHalfMask[i] == i) &&
                   "We can't handle any clobbers here!");
          assert(InputsFixed[1] != (InputsFixed[0] ^ 1) &&
                 "Cannot have adjacent inputs here!");

          SourceHalfMask[InputsFixed[0] ^ 1] = InputsFixed[1];
          SourceHalfMask[InputsFixed[1]] = InputsFixed[0] ^ 1;

          // We also have to update the final source mask in this case because
          // it may need to undo the above swap.
          for (int &M : FinalSourceHalfMask)
            if (M == (InputsFixed[0] ^ 1) + SourceOffset)
              M = InputsFixed[1] + SourceOffset;
            else if (M == InputsFixed[1] + SourceOffset)
              M = (InputsFixed[0] ^ 1) + SourceOffset;

          InputsFixed[1] = InputsFixed[0] ^ 1;
        }

        // Point everything at the fixed inputs.
        for (int &M : HalfMask)
          if (M == IncomingInputs[0])
            M = InputsFixed[0] + SourceOffset;
          else if (M == IncomingInputs[1])
            M = InputsFixed[1] + SourceOffset;

        IncomingInputs[0] = InputsFixed[0] + SourceOffset;
        IncomingInputs[1] = InputsFixed[1] + SourceOffset;
      }
    } else {
      llvm_unreachable("Unhandled input size!");
    }

    // Now hoist the DWord down to the right half.
    int FreeDWord = (PSHUFDMask[DestOffset / 2] < 0 ? 0 : 1) + DestOffset / 2;
    assert(PSHUFDMask[FreeDWord] < 0 && "DWord not free");
    PSHUFDMask[FreeDWord] = IncomingInputs[0] / 2;
    for (int &M : HalfMask)
      for (int Input : IncomingInputs)
        if (M == Input)
          M = FreeDWord * 2 + Input % 2;
  };
  moveInputsToRightHalf(HToLInputs, LToLInputs, PSHUFHMask, LoMask, HiMask,
                        /*SourceOffset*/ 4, /*DestOffset*/ 0);
  moveInputsToRightHalf(LToHInputs, HToHInputs, PSHUFLMask, HiMask, LoMask,
                        /*SourceOffset*/ 0, /*DestOffset*/ 4);

  // Now enact all the shuffles we've computed to move the inputs into their
  // target half.
  if (!isNoopShuffleMask(PSHUFLMask))
    V = DAG.getNode(X86ISD::PSHUFLW, DL, VT, V,
                    getV4X86ShuffleImm8ForMask(PSHUFLMask, DL, DAG));
  if (!isNoopShuffleMask(PSHUFHMask))
    V = DAG.getNode(X86ISD::PSHUFHW, DL, VT, V,
                    getV4X86ShuffleImm8ForMask(PSHUFHMask, DL, DAG));
  if (!isNoopShuffleMask(PSHUFDMask))
    V = DAG.getBitcast(
        VT,
        DAG.getNode(X86ISD::PSHUFD, DL, PSHUFDVT, DAG.getBitcast(PSHUFDVT, V),
                    getV4X86ShuffleImm8ForMask(PSHUFDMask, DL, DAG)));

  // At this point, each half should contain all its inputs, and we can then
  // just shuffle them into their final position.
  assert(count_if(LoMask, [](int M) { return M >= 4; }) == 0 &&
         "Failed to lift all the high half inputs to the low mask!");
  assert(count_if(HiMask, [](int M) { return M >= 0 && M < 4; }) == 0 &&
         "Failed to lift all the low half inputs to the high mask!");

  // Do a half shuffle for the low mask.
  if (!isNoopShuffleMask(LoMask))
    V = DAG.getNode(X86ISD::PSHUFLW, DL, VT, V,
                    getV4X86ShuffleImm8ForMask(LoMask, DL, DAG));

  // Do a half shuffle with the high mask after shifting its values down.
  for (int &M : HiMask)
    if (M >= 0)
      M -= 4;
  if (!isNoopShuffleMask(HiMask))
    V = DAG.getNode(X86ISD::PSHUFHW, DL, VT, V,
                    getV4X86ShuffleImm8ForMask(HiMask, DL, DAG));

  return V;
}

/// Helper to form a PSHUFB-based shuffle+blend, opportunistically avoiding the
/// blend if only one input is used.
static SDValue lowerShuffleAsBlendOfPSHUFBs(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const APInt &Zeroable, SelectionDAG &DAG, bool &V1InUse, bool &V2InUse) {
  assert(!is128BitLaneCrossingShuffleMask(VT, Mask) &&
         "Lane crossing shuffle masks not supported");

  int NumBytes = VT.getSizeInBits() / 8;
  int Size = Mask.size();
  int Scale = NumBytes / Size;

  SmallVector<SDValue, 64> V1Mask(NumBytes, DAG.getUNDEF(MVT::i8));
  SmallVector<SDValue, 64> V2Mask(NumBytes, DAG.getUNDEF(MVT::i8));
  V1InUse = false;
  V2InUse = false;

  for (int i = 0; i < NumBytes; ++i) {
    int M = Mask[i / Scale];
    if (M < 0)
      continue;

    const int ZeroMask = 0x80;
    int V1Idx = M < Size ? M * Scale + i % Scale : ZeroMask;
    int V2Idx = M < Size ? ZeroMask : (M - Size) * Scale + i % Scale;
    if (Zeroable[i / Scale])
      V1Idx = V2Idx = ZeroMask;

    V1Mask[i] = DAG.getConstant(V1Idx, DL, MVT::i8);
    V2Mask[i] = DAG.getConstant(V2Idx, DL, MVT::i8);
    V1InUse |= (ZeroMask != V1Idx);
    V2InUse |= (ZeroMask != V2Idx);
  }

  MVT ShufVT = MVT::getVectorVT(MVT::i8, NumBytes);
  if (V1InUse)
    V1 = DAG.getNode(X86ISD::PSHUFB, DL, ShufVT, DAG.getBitcast(ShufVT, V1),
                     DAG.getBuildVector(ShufVT, DL, V1Mask));
  if (V2InUse)
    V2 = DAG.getNode(X86ISD::PSHUFB, DL, ShufVT, DAG.getBitcast(ShufVT, V2),
                     DAG.getBuildVector(ShufVT, DL, V2Mask));

  // If we need shuffled inputs from both, blend the two.
  SDValue V;
  if (V1InUse && V2InUse)
    V = DAG.getNode(ISD::OR, DL, ShufVT, V1, V2);
  else
    V = V1InUse ? V1 : V2;

  // Cast the result back to the correct type.
  return DAG.getBitcast(VT, V);
}

/// Generic lowering of 8-lane i16 shuffles.
///
/// This handles both single-input shuffles and combined shuffle/blends with
/// two inputs. The single input shuffles are immediately delegated to
/// a dedicated lowering routine.
///
/// The blends are lowered in one of three fundamental ways. If there are few
/// enough inputs, it delegates to a basic UNPCK-based strategy. If the shuffle
/// of the input is significantly cheaper when lowered as an interleaving of
/// the two inputs, try to interleave them. Otherwise, blend the low and high
/// halves of the inputs separately (making them have relatively few inputs)
/// and then concatenate them.
static SDValue lowerV8I16Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v8i16 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v8i16 && "Bad operand type!");
  assert(Mask.size() == 8 && "Unexpected mask size for v8 shuffle!");

  // Whenever we can lower this as a zext, that instruction is strictly faster
  // than any alternative.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(DL, MVT::v8i16, V1, V2, Mask,
                                                   Zeroable, Subtarget, DAG))
    return ZExt;

  int NumV2Inputs = count_if(Mask, [](int M) { return M >= 8; });

  if (NumV2Inputs == 0) {
    // Try to use shift instructions.
    if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v8i16, V1, V1, Mask,
                                            Zeroable, Subtarget, DAG))
      return Shift;

    // Check for being able to broadcast a single element.
    if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v8i16, V1, V2,
                                                    Mask, Subtarget, DAG))
      return Broadcast;

    // Use dedicated unpack instructions for masks that match their pattern.
    if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v8i16, Mask, V1, V2, DAG))
      return V;

    // Use dedicated pack instructions for masks that match their pattern.
    if (SDValue V = lowerShuffleWithPACK(DL, MVT::v8i16, Mask, V1, V2, DAG,
                                         Subtarget))
      return V;

    // Try to use byte rotation instructions.
    if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v8i16, V1, V1, Mask,
                                                  Subtarget, DAG))
      return Rotate;

    // Make a copy of the mask so it can be modified.
    SmallVector<int, 8> MutableMask(Mask.begin(), Mask.end());
    return lowerV8I16GeneralSingleInputShuffle(DL, MVT::v8i16, V1, MutableMask,
                                               Subtarget, DAG);
  }

  assert(llvm::any_of(Mask, [](int M) { return M >= 0 && M < 8; }) &&
         "All single-input shuffles should be canonicalized to be V1-input "
         "shuffles.");

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v8i16, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // See if we can use SSE4A Extraction / Insertion.
  if (Subtarget.hasSSE4A())
    if (SDValue V = lowerShuffleWithSSE4A(DL, MVT::v8i16, V1, V2, Mask,
                                          Zeroable, DAG))
      return V;

  // There are special ways we can lower some single-element blends.
  if (NumV2Inputs == 1)
    if (SDValue V = lowerShuffleAsElementInsertion(
            DL, MVT::v8i16, V1, V2, Mask, Zeroable, Subtarget, DAG))
      return V;

  // We have different paths for blend lowering, but they all must use the
  // *exact* same predicate.
  bool IsBlendSupported = Subtarget.hasSSE41();
  if (IsBlendSupported)
    if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v8i16, V1, V2, Mask,
                                            Zeroable, Subtarget, DAG))
      return Blend;

  if (SDValue Masked = lowerShuffleAsBitMask(DL, MVT::v8i16, V1, V2, Mask,
                                             Zeroable, Subtarget, DAG))
    return Masked;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v8i16, Mask, V1, V2, DAG))
    return V;

  // Use dedicated pack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithPACK(DL, MVT::v8i16, Mask, V1, V2, DAG,
                                       Subtarget))
    return V;

  // Try to use byte rotation instructions.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v8i16, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  if (SDValue BitBlend =
          lowerShuffleAsBitBlend(DL, MVT::v8i16, V1, V2, Mask, DAG))
    return BitBlend;

  // Try to use byte shift instructions to mask.
  if (SDValue V = lowerVectorShuffleAsByteShiftMask(
          DL, MVT::v8i16, V1, V2, Mask, Zeroable, Subtarget, DAG))
    return V;

  // Try to lower by permuting the inputs into an unpack instruction.
  if (SDValue Unpack = lowerShuffleAsPermuteAndUnpack(DL, MVT::v8i16, V1, V2,
                                                      Mask, Subtarget, DAG))
    return Unpack;

  // If we can't directly blend but can use PSHUFB, that will be better as it
  // can both shuffle and set up the inefficient blend.
  if (!IsBlendSupported && Subtarget.hasSSSE3()) {
    bool V1InUse, V2InUse;
    return lowerShuffleAsBlendOfPSHUFBs(DL, MVT::v8i16, V1, V2, Mask,
                                        Zeroable, DAG, V1InUse, V2InUse);
  }

  // We can always bit-blend if we have to so the fallback strategy is to
  // decompose into single-input permutes and blends.
  return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v8i16, V1, V2,
                                              Mask, Subtarget, DAG);
}

/// Check whether a compaction lowering can be done by dropping even
/// elements and compute how many times even elements must be dropped.
///
/// This handles shuffles which take every Nth element where N is a power of
/// two. Example shuffle masks:
///
///  N = 1:  0,  2,  4,  6,  8, 10, 12, 14,  0,  2,  4,  6,  8, 10, 12, 14
///  N = 1:  0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
///  N = 2:  0,  4,  8, 12,  0,  4,  8, 12,  0,  4,  8, 12,  0,  4,  8, 12
///  N = 2:  0,  4,  8, 12, 16, 20, 24, 28,  0,  4,  8, 12, 16, 20, 24, 28
///  N = 3:  0,  8,  0,  8,  0,  8,  0,  8,  0,  8,  0,  8,  0,  8,  0,  8
///  N = 3:  0,  8, 16, 24,  0,  8, 16, 24,  0,  8, 16, 24,  0,  8, 16, 24
///
/// Any of these lanes can of course be undef.
///
/// This routine only supports N <= 3.
/// FIXME: Evaluate whether either AVX or AVX-512 have any opportunities here
/// for larger N.
///
/// \returns N above, or the number of times even elements must be dropped if
/// there is such a number. Otherwise returns zero.
static int canLowerByDroppingEvenElements(ArrayRef<int> Mask,
                                          bool IsSingleInput) {
  // The modulus for the shuffle vector entries is based on whether this is
  // a single input or not.
  int ShuffleModulus = Mask.size() * (IsSingleInput ? 1 : 2);
  assert(isPowerOf2_32((uint32_t)ShuffleModulus) &&
         "We should only be called with masks with a power-of-2 size!");

  uint64_t ModMask = (uint64_t)ShuffleModulus - 1;

  // We track whether the input is viable for all power-of-2 strides 2^1, 2^2,
  // and 2^3 simultaneously. This is because we may have ambiguity with
  // partially undef inputs.
  bool ViableForN[3] = {true, true, true};

  for (int i = 0, e = Mask.size(); i < e; ++i) {
    // Ignore undef lanes, we'll optimistically collapse them to the pattern we
    // want.
    if (Mask[i] < 0)
      continue;

    bool IsAnyViable = false;
    for (unsigned j = 0; j != array_lengthof(ViableForN); ++j)
      if (ViableForN[j]) {
        uint64_t N = j + 1;

        // The shuffle mask must be equal to (i * 2^N) % M.
        if ((uint64_t)Mask[i] == (((uint64_t)i << N) & ModMask))
          IsAnyViable = true;
        else
          ViableForN[j] = false;
      }
    // Early exit if we exhaust the possible powers of two.
    if (!IsAnyViable)
      break;
  }

  for (unsigned j = 0; j != array_lengthof(ViableForN); ++j)
    if (ViableForN[j])
      return j + 1;

  // Return 0 as there is no viable power of two.
  return 0;
}

static SDValue lowerShuffleWithPERMV(const SDLoc &DL, MVT VT,
                                     ArrayRef<int> Mask, SDValue V1,
                                     SDValue V2, SelectionDAG &DAG) {
  MVT MaskEltVT = MVT::getIntegerVT(VT.getScalarSizeInBits());
  MVT MaskVecVT = MVT::getVectorVT(MaskEltVT, VT.getVectorNumElements());

  SDValue MaskNode = getConstVector(Mask, MaskVecVT, DAG, DL, true);
  if (V2.isUndef())
    return DAG.getNode(X86ISD::VPERMV, DL, VT, MaskNode, V1);

  return DAG.getNode(X86ISD::VPERMV3, DL, VT, V1, MaskNode, V2);
}

/// Generic lowering of v16i8 shuffles.
///
/// This is a hybrid strategy to lower v16i8 vectors. It first attempts to
/// detect any complexity reducing interleaving. If that doesn't help, it uses
/// UNPCK to spread the i8 elements across two i16-element vectors, and uses
/// the existing lowering for v8i16 blends on each half, finally PACK-ing them
/// back together.
static SDValue lowerV16I8Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v16i8 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v16i8 && "Bad operand type!");
  assert(Mask.size() == 16 && "Unexpected mask size for v16 shuffle!");

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v16i8, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // Try to use byte rotation instructions.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v16i8, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  // Use dedicated pack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithPACK(DL, MVT::v16i8, Mask, V1, V2, DAG,
                                       Subtarget))
    return V;

  // Try to use a zext lowering.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(DL, MVT::v16i8, V1, V2, Mask,
                                                   Zeroable, Subtarget, DAG))
    return ZExt;

  // See if we can use SSE4A Extraction / Insertion.
  if (Subtarget.hasSSE4A())
    if (SDValue V = lowerShuffleWithSSE4A(DL, MVT::v16i8, V1, V2, Mask,
                                          Zeroable, DAG))
      return V;

  int NumV2Elements = count_if(Mask, [](int M) { return M >= 16; });

  // For single-input shuffles, there are some nicer lowering tricks we can use.
  if (NumV2Elements == 0) {
    // Check for being able to broadcast a single element.
    if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v16i8, V1, V2,
                                                    Mask, Subtarget, DAG))
      return Broadcast;

    if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v16i8, Mask, V1, V2, DAG))
      return V;

    // Check whether we can widen this to an i16 shuffle by duplicating bytes.
    // Notably, this handles splat and partial-splat shuffles more efficiently.
    // However, it only makes sense if the pre-duplication shuffle simplifies
    // things significantly. Currently, this means we need to be able to
    // express the pre-duplication shuffle as an i16 shuffle.
    //
    // FIXME: We should check for other patterns which can be widened into an
    // i16 shuffle as well.
    auto canWidenViaDuplication = [](ArrayRef<int> Mask) {
      for (int i = 0; i < 16; i += 2)
        if (Mask[i] >= 0 && Mask[i + 1] >= 0 && Mask[i] != Mask[i + 1])
          return false;

      return true;
    };
    auto tryToWidenViaDuplication = [&]() -> SDValue {
      if (!canWidenViaDuplication(Mask))
        return SDValue();
      SmallVector<int, 4> LoInputs;
      copy_if(Mask, std::back_inserter(LoInputs),
              [](int M) { return M >= 0 && M < 8; });
      array_pod_sort(LoInputs.begin(), LoInputs.end());
      LoInputs.erase(std::unique(LoInputs.begin(), LoInputs.end()),
                     LoInputs.end());
      SmallVector<int, 4> HiInputs;
      copy_if(Mask, std::back_inserter(HiInputs), [](int M) { return M >= 8; });
      array_pod_sort(HiInputs.begin(), HiInputs.end());
      HiInputs.erase(std::unique(HiInputs.begin(), HiInputs.end()),
                     HiInputs.end());

      bool TargetLo = LoInputs.size() >= HiInputs.size();
      ArrayRef<int> InPlaceInputs = TargetLo ? LoInputs : HiInputs;
      ArrayRef<int> MovingInputs = TargetLo ? HiInputs : LoInputs;

      int PreDupI16Shuffle[] = {-1, -1, -1, -1, -1, -1, -1, -1};
      SmallDenseMap<int, int, 8> LaneMap;
      for (int I : InPlaceInputs) {
        PreDupI16Shuffle[I/2] = I/2;
        LaneMap[I] = I;
      }
      int j = TargetLo ? 0 : 4, je = j + 4;
      for (int i = 0, ie = MovingInputs.size(); i < ie; ++i) {
        // Check if j is already a shuffle of this input. This happens when
        // there are two adjacent bytes after we move the low one.
        if (PreDupI16Shuffle[j] != MovingInputs[i] / 2) {
          // If we haven't yet mapped the input, search for a slot into which
          // we can map it.
          while (j < je && PreDupI16Shuffle[j] >= 0)
            ++j;

          if (j == je)
            // We can't place the inputs into a single half with a simple i16 shuffle, so bail.
            return SDValue();

          // Map this input with the i16 shuffle.
          PreDupI16Shuffle[j] = MovingInputs[i] / 2;
        }

        // Update the lane map based on the mapping we ended up with.
        LaneMap[MovingInputs[i]] = 2 * j + MovingInputs[i] % 2;
      }
      V1 = DAG.getBitcast(
          MVT::v16i8,
          DAG.getVectorShuffle(MVT::v8i16, DL, DAG.getBitcast(MVT::v8i16, V1),
                               DAG.getUNDEF(MVT::v8i16), PreDupI16Shuffle));

      // Unpack the bytes to form the i16s that will be shuffled into place.
      V1 = DAG.getNode(TargetLo ? X86ISD::UNPCKL : X86ISD::UNPCKH, DL,
                       MVT::v16i8, V1, V1);

      int PostDupI16Shuffle[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
      for (int i = 0; i < 16; ++i)
        if (Mask[i] >= 0) {
          int MappedMask = LaneMap[Mask[i]] - (TargetLo ? 0 : 8);
          assert(MappedMask < 8 && "Invalid v8 shuffle mask!");
          if (PostDupI16Shuffle[i / 2] < 0)
            PostDupI16Shuffle[i / 2] = MappedMask;
          else
            assert(PostDupI16Shuffle[i / 2] == MappedMask &&
                   "Conflicting entries in the original shuffle!");
        }
      return DAG.getBitcast(
          MVT::v16i8,
          DAG.getVectorShuffle(MVT::v8i16, DL, DAG.getBitcast(MVT::v8i16, V1),
                               DAG.getUNDEF(MVT::v8i16), PostDupI16Shuffle));
    };
    if (SDValue V = tryToWidenViaDuplication())
      return V;
  }

  if (SDValue Masked = lowerShuffleAsBitMask(DL, MVT::v16i8, V1, V2, Mask,
                                             Zeroable, Subtarget, DAG))
    return Masked;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v16i8, Mask, V1, V2, DAG))
    return V;

  // Try to use byte shift instructions to mask.
  if (SDValue V = lowerVectorShuffleAsByteShiftMask(
          DL, MVT::v16i8, V1, V2, Mask, Zeroable, Subtarget, DAG))
    return V;

  // Check for SSSE3 which lets us lower all v16i8 shuffles much more directly
  // with PSHUFB. It is important to do this before we attempt to generate any
  // blends but after all of the single-input lowerings. If the single input
  // lowerings can find an instruction sequence that is faster than a PSHUFB, we
  // want to preserve that and we can DAG combine any longer sequences into
  // a PSHUFB in the end. But once we start blending from multiple inputs,
  // the complexity of DAG combining bad patterns back into PSHUFB is too high,
  // and there are *very* few patterns that would actually be faster than the
  // PSHUFB approach because of its ability to zero lanes.
  //
  // FIXME: The only exceptions to the above are blends which are exact
  // interleavings with direct instructions supporting them. We currently don't
  // handle those well here.
  if (Subtarget.hasSSSE3()) {
    bool V1InUse = false;
    bool V2InUse = false;

    SDValue PSHUFB = lowerShuffleAsBlendOfPSHUFBs(
        DL, MVT::v16i8, V1, V2, Mask, Zeroable, DAG, V1InUse, V2InUse);

    // If both V1 and V2 are in use and we can use a direct blend or an unpack,
    // do so. This avoids using them to handle blends-with-zero which is
    // important as a single pshufb is significantly faster for that.
    if (V1InUse && V2InUse) {
      if (Subtarget.hasSSE41())
        if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v16i8, V1, V2, Mask,
                                                Zeroable, Subtarget, DAG))
          return Blend;

      // We can use an unpack to do the blending rather than an or in some
      // cases. Even though the or may be (very minorly) more efficient, we
      // preference this lowering because there are common cases where part of
      // the complexity of the shuffles goes away when we do the final blend as
      // an unpack.
      // FIXME: It might be worth trying to detect if the unpack-feeding
      // shuffles will both be pshufb, in which case we shouldn't bother with
      // this.
      if (SDValue Unpack = lowerShuffleAsPermuteAndUnpack(
              DL, MVT::v16i8, V1, V2, Mask, Subtarget, DAG))
        return Unpack;

      // If we have VBMI we can use one VPERM instead of multiple PSHUFBs.
      if (Subtarget.hasVBMI() && Subtarget.hasVLX())
        return lowerShuffleWithPERMV(DL, MVT::v16i8, Mask, V1, V2, DAG);

      // Use PALIGNR+Permute if possible - permute might become PSHUFB but the
      // PALIGNR will be cheaper than the second PSHUFB+OR.
      if (SDValue V = lowerShuffleAsByteRotateAndPermute(
              DL, MVT::v16i8, V1, V2, Mask, Subtarget, DAG))
        return V;
    }

    return PSHUFB;
  }

  // There are special ways we can lower some single-element blends.
  if (NumV2Elements == 1)
    if (SDValue V = lowerShuffleAsElementInsertion(
            DL, MVT::v16i8, V1, V2, Mask, Zeroable, Subtarget, DAG))
      return V;

  if (SDValue Blend = lowerShuffleAsBitBlend(DL, MVT::v16i8, V1, V2, Mask, DAG))
    return Blend;

  // Check whether a compaction lowering can be done. This handles shuffles
  // which take every Nth element for some even N. See the helper function for
  // details.
  //
  // We special case these as they can be particularly efficiently handled with
  // the PACKUSB instruction on x86 and they show up in common patterns of
  // rearranging bytes to truncate wide elements.
  bool IsSingleInput = V2.isUndef();
  if (int NumEvenDrops = canLowerByDroppingEvenElements(Mask, IsSingleInput)) {
    // NumEvenDrops is the power of two stride of the elements. Another way of
    // thinking about it is that we need to drop the even elements this many
    // times to get the original input.

    // First we need to zero all the dropped bytes.
    assert(NumEvenDrops <= 3 &&
           "No support for dropping even elements more than 3 times.");
    SmallVector<SDValue, 16> ByteClearOps(16, DAG.getConstant(0, DL, MVT::i8));
    for (unsigned i = 0; i != 16; i += 1 << NumEvenDrops)
      ByteClearOps[i] = DAG.getConstant(0xFF, DL, MVT::i8);
    SDValue ByteClearMask = DAG.getBuildVector(MVT::v16i8, DL, ByteClearOps);
    V1 = DAG.getNode(ISD::AND, DL, MVT::v16i8, V1, ByteClearMask);
    if (!IsSingleInput)
      V2 = DAG.getNode(ISD::AND, DL, MVT::v16i8, V2, ByteClearMask);

    // Now pack things back together.
    V1 = DAG.getBitcast(MVT::v8i16, V1);
    V2 = IsSingleInput ? V1 : DAG.getBitcast(MVT::v8i16, V2);
    SDValue Result = DAG.getNode(X86ISD::PACKUS, DL, MVT::v16i8, V1, V2);
    for (int i = 1; i < NumEvenDrops; ++i) {
      Result = DAG.getBitcast(MVT::v8i16, Result);
      Result = DAG.getNode(X86ISD::PACKUS, DL, MVT::v16i8, Result, Result);
    }

    return Result;
  }

  // Handle multi-input cases by blending single-input shuffles.
  if (NumV2Elements > 0)
    return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v16i8, V1, V2, Mask,
                                                Subtarget, DAG);

  // The fallback path for single-input shuffles widens this into two v8i16
  // vectors with unpacks, shuffles those, and then pulls them back together
  // with a pack.
  SDValue V = V1;

  std::array<int, 8> LoBlendMask = {{-1, -1, -1, -1, -1, -1, -1, -1}};
  std::array<int, 8> HiBlendMask = {{-1, -1, -1, -1, -1, -1, -1, -1}};
  for (int i = 0; i < 16; ++i)
    if (Mask[i] >= 0)
      (i < 8 ? LoBlendMask[i] : HiBlendMask[i % 8]) = Mask[i];

  SDValue VLoHalf, VHiHalf;
  // Check if any of the odd lanes in the v16i8 are used. If not, we can mask
  // them out and avoid using UNPCK{L,H} to extract the elements of V as
  // i16s.
  if (none_of(LoBlendMask, [](int M) { return M >= 0 && M % 2 == 1; }) &&
      none_of(HiBlendMask, [](int M) { return M >= 0 && M % 2 == 1; })) {
    // Use a mask to drop the high bytes.
    VLoHalf = DAG.getBitcast(MVT::v8i16, V);
    VLoHalf = DAG.getNode(ISD::AND, DL, MVT::v8i16, VLoHalf,
                          DAG.getConstant(0x00FF, DL, MVT::v8i16));

    // This will be a single vector shuffle instead of a blend so nuke VHiHalf.
    VHiHalf = DAG.getUNDEF(MVT::v8i16);

    // Squash the masks to point directly into VLoHalf.
    for (int &M : LoBlendMask)
      if (M >= 0)
        M /= 2;
    for (int &M : HiBlendMask)
      if (M >= 0)
        M /= 2;
  } else {
    // Otherwise just unpack the low half of V into VLoHalf and the high half into
    // VHiHalf so that we can blend them as i16s.
    SDValue Zero = getZeroVector(MVT::v16i8, Subtarget, DAG, DL);

    VLoHalf = DAG.getBitcast(
        MVT::v8i16, DAG.getNode(X86ISD::UNPCKL, DL, MVT::v16i8, V, Zero));
    VHiHalf = DAG.getBitcast(
        MVT::v8i16, DAG.getNode(X86ISD::UNPCKH, DL, MVT::v16i8, V, Zero));
  }

  SDValue LoV = DAG.getVectorShuffle(MVT::v8i16, DL, VLoHalf, VHiHalf, LoBlendMask);
  SDValue HiV = DAG.getVectorShuffle(MVT::v8i16, DL, VLoHalf, VHiHalf, HiBlendMask);

  return DAG.getNode(X86ISD::PACKUS, DL, MVT::v16i8, LoV, HiV);
}

/// Dispatching routine to lower various 128-bit x86 vector shuffles.
///
/// This routine breaks down the specific type of 128-bit shuffle and
/// dispatches to the lowering routines accordingly.
static SDValue lower128BitShuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                  MVT VT, SDValue V1, SDValue V2,
                                  const APInt &Zeroable,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  switch (VT.SimpleTy) {
  case MVT::v2i64:
    return lowerV2I64Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v2f64:
    return lowerV2F64Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v4i32:
    return lowerV4I32Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v4f32:
    return lowerV4F32Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v8i16:
    return lowerV8I16Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v16i8:
    return lowerV16I8Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);

  default:
    llvm_unreachable("Unimplemented!");
  }
}

/// Generic routine to split vector shuffle into half-sized shuffles.
///
/// This routine just extracts two subvectors, shuffles them independently, and
/// then concatenates them back together. This should work effectively with all
/// AVX vector shuffle types.
static SDValue splitAndLowerShuffle(const SDLoc &DL, MVT VT, SDValue V1,
                                    SDValue V2, ArrayRef<int> Mask,
                                    SelectionDAG &DAG) {
  assert(VT.getSizeInBits() >= 256 &&
         "Only for 256-bit or wider vector shuffles!");
  assert(V1.getSimpleValueType() == VT && "Bad operand type!");
  assert(V2.getSimpleValueType() == VT && "Bad operand type!");

  ArrayRef<int> LoMask = Mask.slice(0, Mask.size() / 2);
  ArrayRef<int> HiMask = Mask.slice(Mask.size() / 2);

  int NumElements = VT.getVectorNumElements();
  int SplitNumElements = NumElements / 2;
  MVT ScalarVT = VT.getVectorElementType();
  MVT SplitVT = MVT::getVectorVT(ScalarVT, NumElements / 2);

  // Rather than splitting build-vectors, just build two narrower build
  // vectors. This helps shuffling with splats and zeros.
  auto SplitVector = [&](SDValue V) {
    V = peekThroughBitcasts(V);

    MVT OrigVT = V.getSimpleValueType();
    int OrigNumElements = OrigVT.getVectorNumElements();
    int OrigSplitNumElements = OrigNumElements / 2;
    MVT OrigScalarVT = OrigVT.getVectorElementType();
    MVT OrigSplitVT = MVT::getVectorVT(OrigScalarVT, OrigNumElements / 2);

    SDValue LoV, HiV;

    auto *BV = dyn_cast<BuildVectorSDNode>(V);
    if (!BV) {
      LoV = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, OrigSplitVT, V,
                        DAG.getIntPtrConstant(0, DL));
      HiV = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, OrigSplitVT, V,
                        DAG.getIntPtrConstant(OrigSplitNumElements, DL));
    } else {

      SmallVector<SDValue, 16> LoOps, HiOps;
      for (int i = 0; i < OrigSplitNumElements; ++i) {
        LoOps.push_back(BV->getOperand(i));
        HiOps.push_back(BV->getOperand(i + OrigSplitNumElements));
      }
      LoV = DAG.getBuildVector(OrigSplitVT, DL, LoOps);
      HiV = DAG.getBuildVector(OrigSplitVT, DL, HiOps);
    }
    return std::make_pair(DAG.getBitcast(SplitVT, LoV),
                          DAG.getBitcast(SplitVT, HiV));
  };

  SDValue LoV1, HiV1, LoV2, HiV2;
  std::tie(LoV1, HiV1) = SplitVector(V1);
  std::tie(LoV2, HiV2) = SplitVector(V2);

  // Now create two 4-way blends of these half-width vectors.
  auto HalfBlend = [&](ArrayRef<int> HalfMask) {
    bool UseLoV1 = false, UseHiV1 = false, UseLoV2 = false, UseHiV2 = false;
    SmallVector<int, 32> V1BlendMask((unsigned)SplitNumElements, -1);
    SmallVector<int, 32> V2BlendMask((unsigned)SplitNumElements, -1);
    SmallVector<int, 32> BlendMask((unsigned)SplitNumElements, -1);
    for (int i = 0; i < SplitNumElements; ++i) {
      int M = HalfMask[i];
      if (M >= NumElements) {
        if (M >= NumElements + SplitNumElements)
          UseHiV2 = true;
        else
          UseLoV2 = true;
        V2BlendMask[i] = M - NumElements;
        BlendMask[i] = SplitNumElements + i;
      } else if (M >= 0) {
        if (M >= SplitNumElements)
          UseHiV1 = true;
        else
          UseLoV1 = true;
        V1BlendMask[i] = M;
        BlendMask[i] = i;
      }
    }

    // Because the lowering happens after all combining takes place, we need to
    // manually combine these blend masks as much as possible so that we create
    // a minimal number of high-level vector shuffle nodes.

    // First try just blending the halves of V1 or V2.
    if (!UseLoV1 && !UseHiV1 && !UseLoV2 && !UseHiV2)
      return DAG.getUNDEF(SplitVT);
    if (!UseLoV2 && !UseHiV2)
      return DAG.getVectorShuffle(SplitVT, DL, LoV1, HiV1, V1BlendMask);
    if (!UseLoV1 && !UseHiV1)
      return DAG.getVectorShuffle(SplitVT, DL, LoV2, HiV2, V2BlendMask);

    SDValue V1Blend, V2Blend;
    if (UseLoV1 && UseHiV1) {
      V1Blend =
        DAG.getVectorShuffle(SplitVT, DL, LoV1, HiV1, V1BlendMask);
    } else {
      // We only use half of V1 so map the usage down into the final blend mask.
      V1Blend = UseLoV1 ? LoV1 : HiV1;
      for (int i = 0; i < SplitNumElements; ++i)
        if (BlendMask[i] >= 0 && BlendMask[i] < SplitNumElements)
          BlendMask[i] = V1BlendMask[i] - (UseLoV1 ? 0 : SplitNumElements);
    }
    if (UseLoV2 && UseHiV2) {
      V2Blend =
        DAG.getVectorShuffle(SplitVT, DL, LoV2, HiV2, V2BlendMask);
    } else {
      // We only use half of V2 so map the usage down into the final blend mask.
      V2Blend = UseLoV2 ? LoV2 : HiV2;
      for (int i = 0; i < SplitNumElements; ++i)
        if (BlendMask[i] >= SplitNumElements)
          BlendMask[i] = V2BlendMask[i] + (UseLoV2 ? SplitNumElements : 0);
    }
    return DAG.getVectorShuffle(SplitVT, DL, V1Blend, V2Blend, BlendMask);
  };
  SDValue Lo = HalfBlend(LoMask);
  SDValue Hi = HalfBlend(HiMask);
  return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Lo, Hi);
}

/// Either split a vector in halves or decompose the shuffles and the
/// blend.
///
/// This is provided as a good fallback for many lowerings of non-single-input
/// shuffles with more than one 128-bit lane. In those cases, we want to select
/// between splitting the shuffle into 128-bit components and stitching those
/// back together vs. extracting the single-input shuffles and blending those
/// results.
static SDValue lowerShuffleAsSplitOrBlend(const SDLoc &DL, MVT VT, SDValue V1,
                                          SDValue V2, ArrayRef<int> Mask,
                                          const X86Subtarget &Subtarget,
                                          SelectionDAG &DAG) {
  assert(!V2.isUndef() && "This routine must not be used to lower single-input "
         "shuffles as it could then recurse on itself.");
  int Size = Mask.size();

  // If this can be modeled as a broadcast of two elements followed by a blend,
  // prefer that lowering. This is especially important because broadcasts can
  // often fold with memory operands.
  auto DoBothBroadcast = [&] {
    int V1BroadcastIdx = -1, V2BroadcastIdx = -1;
    for (int M : Mask)
      if (M >= Size) {
        if (V2BroadcastIdx < 0)
          V2BroadcastIdx = M - Size;
        else if (M - Size != V2BroadcastIdx)
          return false;
      } else if (M >= 0) {
        if (V1BroadcastIdx < 0)
          V1BroadcastIdx = M;
        else if (M != V1BroadcastIdx)
          return false;
      }
    return true;
  };
  if (DoBothBroadcast())
    return lowerShuffleAsDecomposedShuffleBlend(DL, VT, V1, V2, Mask,
                                                Subtarget, DAG);

  // If the inputs all stem from a single 128-bit lane of each input, then we
  // split them rather than blending because the split will decompose to
  // unusually few instructions.
  int LaneCount = VT.getSizeInBits() / 128;
  int LaneSize = Size / LaneCount;
  SmallBitVector LaneInputs[2];
  LaneInputs[0].resize(LaneCount, false);
  LaneInputs[1].resize(LaneCount, false);
  for (int i = 0; i < Size; ++i)
    if (Mask[i] >= 0)
      LaneInputs[Mask[i] / Size][(Mask[i] % Size) / LaneSize] = true;
  if (LaneInputs[0].count() <= 1 && LaneInputs[1].count() <= 1)
    return splitAndLowerShuffle(DL, VT, V1, V2, Mask, DAG);

  // Otherwise, just fall back to decomposed shuffles and a blend. This requires
  // that the decomposed single-input shuffles don't end up here.
  return lowerShuffleAsDecomposedShuffleBlend(DL, VT, V1, V2, Mask, Subtarget,
                                              DAG);
}

/// Lower a vector shuffle crossing multiple 128-bit lanes as
/// a lane permutation followed by a per-lane permutation.
///
/// This is mainly for cases where we can have non-repeating permutes
/// in each lane.
///
/// TODO: This is very similar to lowerShuffleAsLanePermuteAndRepeatedMask,
/// we should investigate merging them.
static SDValue lowerShuffleAsLanePermuteAndPermute(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    SelectionDAG &DAG, const X86Subtarget &Subtarget) {
  int NumElts = VT.getVectorNumElements();
  int NumLanes = VT.getSizeInBits() / 128;
  int NumEltsPerLane = NumElts / NumLanes;

  SmallVector<int, 4> SrcLaneMask(NumLanes, SM_SentinelUndef);
  SmallVector<int, 16> PermMask(NumElts, SM_SentinelUndef);

  for (int i = 0; i != NumElts; ++i) {
    int M = Mask[i];
    if (M < 0)
      continue;

    // Ensure that each lane comes from a single source lane.
    int SrcLane = M / NumEltsPerLane;
    int DstLane = i / NumEltsPerLane;
    if (!isUndefOrEqual(SrcLaneMask[DstLane], SrcLane))
      return SDValue();
    SrcLaneMask[DstLane] = SrcLane;

    PermMask[i] = (DstLane * NumEltsPerLane) + (M % NumEltsPerLane);
  }

  // Make sure we set all elements of the lane mask, to avoid undef propagation.
  SmallVector<int, 16> LaneMask(NumElts, SM_SentinelUndef);
  for (int DstLane = 0; DstLane != NumLanes; ++DstLane) {
    int SrcLane = SrcLaneMask[DstLane];
    if (0 <= SrcLane)
      for (int j = 0; j != NumEltsPerLane; ++j) {
        LaneMask[(DstLane * NumEltsPerLane) + j] =
            (SrcLane * NumEltsPerLane) + j;
      }
  }

  // If we're only shuffling a single lowest lane and the rest are identity
  // then don't bother.
  // TODO - isShuffleMaskInputInPlace could be extended to something like this.
  int NumIdentityLanes = 0;
  bool OnlyShuffleLowestLane = true;
  for (int i = 0; i != NumLanes; ++i) {
    if (isSequentialOrUndefInRange(PermMask, i * NumEltsPerLane, NumEltsPerLane,
                                   i * NumEltsPerLane))
      NumIdentityLanes++;
    else if (SrcLaneMask[i] != 0 && SrcLaneMask[i] != NumLanes)
      OnlyShuffleLowestLane = false;
  }
  if (OnlyShuffleLowestLane && NumIdentityLanes == (NumLanes - 1))
    return SDValue();

  SDValue LanePermute = DAG.getVectorShuffle(VT, DL, V1, V2, LaneMask);
  return DAG.getVectorShuffle(VT, DL, LanePermute, DAG.getUNDEF(VT), PermMask);
}

/// Lower a vector shuffle crossing multiple 128-bit lanes by shuffling one
/// source with a lane permutation.
///
/// This lowering strategy results in four instructions in the worst case for a
/// single-input cross lane shuffle which is lower than any other fully general
/// cross-lane shuffle strategy I'm aware of. Special cases for each particular
/// shuffle pattern should be handled prior to trying this lowering.
static SDValue lowerShuffleAsLanePermuteAndShuffle(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    SelectionDAG &DAG, const X86Subtarget &Subtarget) {
  // FIXME: This should probably be generalized for 512-bit vectors as well.
  assert(VT.is256BitVector() && "Only for 256-bit vector shuffles!");
  int Size = Mask.size();
  int LaneSize = Size / 2;

  // If there are only inputs from one 128-bit lane, splitting will in fact be
  // less expensive. The flags track whether the given lane contains an element
  // that crosses to another lane.
  if (!Subtarget.hasAVX2()) {
    bool LaneCrossing[2] = {false, false};
    for (int i = 0; i < Size; ++i)
      if (Mask[i] >= 0 && (Mask[i] % Size) / LaneSize != i / LaneSize)
        LaneCrossing[(Mask[i] % Size) / LaneSize] = true;
    if (!LaneCrossing[0] || !LaneCrossing[1])
      return splitAndLowerShuffle(DL, VT, V1, V2, Mask, DAG);
  } else {
    bool LaneUsed[2] = {false, false};
    for (int i = 0; i < Size; ++i)
      if (Mask[i] >= 0)
        LaneUsed[(Mask[i] / LaneSize)] = true;
    if (!LaneUsed[0] || !LaneUsed[1])
      return splitAndLowerShuffle(DL, VT, V1, V2, Mask, DAG);
  }

  // TODO - we could support shuffling V2 in the Flipped input.
  assert(V2.isUndef() &&
         "This last part of this routine only works on single input shuffles");

  SmallVector<int, 32> InLaneMask(Mask.begin(), Mask.end());
  for (int i = 0; i < Size; ++i) {
    int &M = InLaneMask[i];
    if (M < 0)
      continue;
    if (((M % Size) / LaneSize) != (i / LaneSize))
      M = (M % LaneSize) + ((i / LaneSize) * LaneSize) + Size;
  }
  assert(!is128BitLaneCrossingShuffleMask(VT, InLaneMask) &&
         "In-lane shuffle mask expected");

  // Flip the lanes, and shuffle the results which should now be in-lane.
  MVT PVT = VT.isFloatingPoint() ? MVT::v4f64 : MVT::v4i64;
  SDValue Flipped = DAG.getBitcast(PVT, V1);
  Flipped =
      DAG.getVectorShuffle(PVT, DL, Flipped, DAG.getUNDEF(PVT), {2, 3, 0, 1});
  Flipped = DAG.getBitcast(VT, Flipped);
  return DAG.getVectorShuffle(VT, DL, V1, Flipped, InLaneMask);
}

/// Handle lowering 2-lane 128-bit shuffles.
static SDValue lowerV2X128Shuffle(const SDLoc &DL, MVT VT, SDValue V1,
                                  SDValue V2, ArrayRef<int> Mask,
                                  const APInt &Zeroable,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  // With AVX2, use VPERMQ/VPERMPD for unary shuffles to allow memory folding.
  if (Subtarget.hasAVX2() && V2.isUndef())
    return SDValue();

  SmallVector<int, 4> WidenedMask;
  if (!X86::canWidenShuffleElements(Mask, Zeroable, WidenedMask))
    return SDValue();

  bool IsLowZero = (Zeroable & 0x3) == 0x3;
  bool IsHighZero = (Zeroable & 0xc) == 0xc;

  // Try to use an insert into a zero vector.
  if (WidenedMask[0] == 0 && IsHighZero) {
    MVT SubVT = MVT::getVectorVT(VT.getVectorElementType(), 2);
    SDValue LoV = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVT, V1,
                              DAG.getIntPtrConstant(0, DL));
    return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT,
                       getZeroVector(VT, Subtarget, DAG, DL), LoV,
                       DAG.getIntPtrConstant(0, DL));
  }

  // TODO: If minimizing size and one of the inputs is a zero vector and the
  // the zero vector has only one use, we could use a VPERM2X128 to save the
  // instruction bytes needed to explicitly generate the zero vector.

  // Blends are faster and handle all the non-lane-crossing cases.
  if (SDValue Blend = lowerShuffleAsBlend(DL, VT, V1, V2, Mask, Zeroable,
                                          Subtarget, DAG))
    return Blend;

  // If either input operand is a zero vector, use VPERM2X128 because its mask
  // allows us to replace the zero input with an implicit zero.
  if (!IsLowZero && !IsHighZero) {
    // Check for patterns which can be matched with a single insert of a 128-bit
    // subvector.
    bool OnlyUsesV1 = isShuffleEquivalent(V1, V2, Mask, {0, 1, 0, 1});
    if (OnlyUsesV1 || isShuffleEquivalent(V1, V2, Mask, {0, 1, 4, 5})) {

      // With AVX1, use vperm2f128 (below) to allow load folding. Otherwise,
      // this will likely become vinsertf128 which can't fold a 256-bit memop.
      if (!isa<LoadSDNode>(peekThroughBitcasts(V1))) {
        MVT SubVT = MVT::getVectorVT(VT.getVectorElementType(), 2);
        SDValue SubVec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVT,
                                     OnlyUsesV1 ? V1 : V2,
                                     DAG.getIntPtrConstant(0, DL));
        return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, V1, SubVec,
                           DAG.getIntPtrConstant(2, DL));
      }
    }

    // Try to use SHUF128 if possible.
    if (Subtarget.hasVLX()) {
      if (WidenedMask[0] < 2 && WidenedMask[1] >= 2) {
        unsigned PermMask = ((WidenedMask[0] % 2) << 0) |
                            ((WidenedMask[1] % 2) << 1);
      return DAG.getNode(X86ISD::SHUF128, DL, VT, V1, V2,
                         DAG.getConstant(PermMask, DL, MVT::i8));
      }
    }
  }

  // Otherwise form a 128-bit permutation. After accounting for undefs,
  // convert the 64-bit shuffle mask selection values into 128-bit
  // selection bits by dividing the indexes by 2 and shifting into positions
  // defined by a vperm2*128 instruction's immediate control byte.

  // The immediate permute control byte looks like this:
  //    [1:0] - select 128 bits from sources for low half of destination
  //    [2]   - ignore
  //    [3]   - zero low half of destination
  //    [5:4] - select 128 bits from sources for high half of destination
  //    [6]   - ignore
  //    [7]   - zero high half of destination

  assert((WidenedMask[0] >= 0 || IsLowZero) &&
         (WidenedMask[1] >= 0 || IsHighZero) && "Undef half?");

  unsigned PermMask = 0;
  PermMask |= IsLowZero  ? 0x08 : (WidenedMask[0] << 0);
  PermMask |= IsHighZero ? 0x80 : (WidenedMask[1] << 4);

  // Check the immediate mask and replace unused sources with undef.
  if ((PermMask & 0x0a) != 0x00 && (PermMask & 0xa0) != 0x00)
    V1 = DAG.getUNDEF(VT);
  if ((PermMask & 0x0a) != 0x02 && (PermMask & 0xa0) != 0x20)
    V2 = DAG.getUNDEF(VT);

  return DAG.getNode(X86ISD::VPERM2X128, DL, VT, V1, V2,
                     DAG.getConstant(PermMask, DL, MVT::i8));
}

/// Lower a vector shuffle by first fixing the 128-bit lanes and then
/// shuffling each lane.
///
/// This attempts to create a repeated lane shuffle where each lane uses one
/// or two of the lanes of the inputs. The lanes of the input vectors are
/// shuffled in one or two independent shuffles to get the lanes into the
/// position needed by the final shuffle.
static SDValue lowerShuffleAsLanePermuteAndRepeatedMask(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const X86Subtarget &Subtarget, SelectionDAG &DAG) {
  assert(!V2.isUndef() && "This is only useful with multiple inputs.");

  if (X86::is128BitLaneRepeatedShuffleMask(VT, Mask))
    return SDValue();

  int Size = Mask.size();
  int NumLanes = VT.getSizeInBits() / 128;
  int LaneSize = 128 / VT.getScalarSizeInBits();
  SmallVector<int, 16> RepeatMask(LaneSize, -1);
  SmallVector<std::array<int, 2>, 2> LaneSrcs(NumLanes, {{-1, -1}});

  // First pass will try to fill in the RepeatMask from lanes that need two
  // sources.
  for (int Lane = 0; Lane != NumLanes; ++Lane) {
    int Srcs[2] = { -1, -1 };
    SmallVector<int, 16> InLaneMask(LaneSize, -1);
    for (int i = 0; i != LaneSize; ++i) {
      int M = Mask[(Lane * LaneSize) + i];
      if (M < 0)
        continue;
      // Determine which of the possible input lanes (NumLanes from each source)
      // this element comes from. Assign that as one of the sources for this
      // lane. We can assign up to 2 sources for this lane. If we run out
      // sources we can't do anything.
      int LaneSrc = M / LaneSize;
      int Src;
      if (Srcs[0] < 0 || Srcs[0] == LaneSrc)
        Src = 0;
      else if (Srcs[1] < 0 || Srcs[1] == LaneSrc)
        Src = 1;
      else
        return SDValue();

      Srcs[Src] = LaneSrc;
      InLaneMask[i] = (M % LaneSize) + Src * Size;
    }

    // If this lane has two sources, see if it fits with the repeat mask so far.
    if (Srcs[1] < 0)
      continue;

    LaneSrcs[Lane][0] = Srcs[0];
    LaneSrcs[Lane][1] = Srcs[1];

    auto MatchMasks = [](ArrayRef<int> M1, ArrayRef<int> M2) {
      assert(M1.size() == M2.size() && "Unexpected mask size");
      for (int i = 0, e = M1.size(); i != e; ++i)
        if (M1[i] >= 0 && M2[i] >= 0 && M1[i] != M2[i])
          return false;
      return true;
    };

    auto MergeMasks = [](ArrayRef<int> Mask, MutableArrayRef<int> MergedMask) {
      assert(Mask.size() == MergedMask.size() && "Unexpected mask size");
      for (int i = 0, e = MergedMask.size(); i != e; ++i) {
        int M = Mask[i];
        if (M < 0)
          continue;
        assert((MergedMask[i] < 0 || MergedMask[i] == M) &&
               "Unexpected mask element");
        MergedMask[i] = M;
      }
    };

    if (MatchMasks(InLaneMask, RepeatMask)) {
      // Merge this lane mask into the final repeat mask.
      MergeMasks(InLaneMask, RepeatMask);
      continue;
    }

    // Didn't find a match. Swap the operands and try again.
    std::swap(LaneSrcs[Lane][0], LaneSrcs[Lane][1]);
    ShuffleVectorSDNode::commuteMask(InLaneMask);

    if (MatchMasks(InLaneMask, RepeatMask)) {
      // Merge this lane mask into the final repeat mask.
      MergeMasks(InLaneMask, RepeatMask);
      continue;
    }

    // Couldn't find a match with the operands in either order.
    return SDValue();
  }

  // Now handle any lanes with only one source.
  for (int Lane = 0; Lane != NumLanes; ++Lane) {
    // If this lane has already been processed, skip it.
    if (LaneSrcs[Lane][0] >= 0)
      continue;

    for (int i = 0; i != LaneSize; ++i) {
      int M = Mask[(Lane * LaneSize) + i];
      if (M < 0)
        continue;

      // If RepeatMask isn't defined yet we can define it ourself.
      if (RepeatMask[i] < 0)
        RepeatMask[i] = M % LaneSize;

      if (RepeatMask[i] < Size) {
        if (RepeatMask[i] != M % LaneSize)
          return SDValue();
        LaneSrcs[Lane][0] = M / LaneSize;
      } else {
        if (RepeatMask[i] != ((M % LaneSize) + Size))
          return SDValue();
        LaneSrcs[Lane][1] = M / LaneSize;
      }
    }

    if (LaneSrcs[Lane][0] < 0 && LaneSrcs[Lane][1] < 0)
      return SDValue();
  }

  SmallVector<int, 16> NewMask(Size, -1);
  for (int Lane = 0; Lane != NumLanes; ++Lane) {
    int Src = LaneSrcs[Lane][0];
    for (int i = 0; i != LaneSize; ++i) {
      int M = -1;
      if (Src >= 0)
        M = Src * LaneSize + i;
      NewMask[Lane * LaneSize + i] = M;
    }
  }
  SDValue NewV1 = DAG.getVectorShuffle(VT, DL, V1, V2, NewMask);
  // Ensure we didn't get back the shuffle we started with.
  // FIXME: This is a hack to make up for some splat handling code in
  // getVectorShuffle.
  if (isa<ShuffleVectorSDNode>(NewV1) &&
      cast<ShuffleVectorSDNode>(NewV1)->getMask() == Mask)
    return SDValue();

  for (int Lane = 0; Lane != NumLanes; ++Lane) {
    int Src = LaneSrcs[Lane][1];
    for (int i = 0; i != LaneSize; ++i) {
      int M = -1;
      if (Src >= 0)
        M = Src * LaneSize + i;
      NewMask[Lane * LaneSize + i] = M;
    }
  }
  SDValue NewV2 = DAG.getVectorShuffle(VT, DL, V1, V2, NewMask);
  // Ensure we didn't get back the shuffle we started with.
  // FIXME: This is a hack to make up for some splat handling code in
  // getVectorShuffle.
  if (isa<ShuffleVectorSDNode>(NewV2) &&
      cast<ShuffleVectorSDNode>(NewV2)->getMask() == Mask)
    return SDValue();

  for (int i = 0; i != Size; ++i) {
    NewMask[i] = RepeatMask[i % LaneSize];
    if (NewMask[i] < 0)
      continue;

    NewMask[i] += (i / LaneSize) * LaneSize;
  }
  return DAG.getVectorShuffle(VT, DL, NewV1, NewV2, NewMask);
}

/// If the input shuffle mask results in a vector that is undefined in all upper
/// or lower half elements and that mask accesses only 2 halves of the
/// shuffle's operands, return true. A mask of half the width with mask indexes
/// adjusted to access the extracted halves of the original shuffle operands is
/// returned in HalfMask. HalfIdx1 and HalfIdx2 return whether the upper or
/// lower half of each input operand is accessed.
static bool
getHalfShuffleMask(ArrayRef<int> Mask, MutableArrayRef<int> HalfMask,
                   int &HalfIdx1, int &HalfIdx2) {
  assert((Mask.size() == HalfMask.size() * 2) &&
         "Expected input mask to be twice as long as output");

  // Exactly one half of the result must be undef to allow narrowing.
  bool UndefLower = isUndefLowerHalf(Mask);
  bool UndefUpper = isUndefUpperHalf(Mask);
  if (UndefLower == UndefUpper)
    return false;

  unsigned HalfNumElts = HalfMask.size();
  unsigned MaskIndexOffset = UndefLower ? HalfNumElts : 0;
  HalfIdx1 = -1;
  HalfIdx2 = -1;
  for (unsigned i = 0; i != HalfNumElts; ++i) {
    int M = Mask[i + MaskIndexOffset];
    if (M < 0) {
      HalfMask[i] = M;
      continue;
    }

    // Determine which of the 4 half vectors this element is from.
    // i.e. 0 = Lower V1, 1 = Upper V1, 2 = Lower V2, 3 = Upper V2.
    int HalfIdx = M / HalfNumElts;

    // Determine the element index into its half vector source.
    int HalfElt = M % HalfNumElts;

    // We can shuffle with up to 2 half vectors, set the new 'half'
    // shuffle mask accordingly.
    if (HalfIdx1 < 0 || HalfIdx1 == HalfIdx) {
      HalfMask[i] = HalfElt;
      HalfIdx1 = HalfIdx;
      continue;
    }
    if (HalfIdx2 < 0 || HalfIdx2 == HalfIdx) {
      HalfMask[i] = HalfElt + HalfNumElts;
      HalfIdx2 = HalfIdx;
      continue;
    }

    // Too many half vectors referenced.
    return false;
  }

  return true;
}

/// Given the output values from getHalfShuffleMask(), create a half width
/// shuffle of extracted vectors followed by an insert back to full width.
static SDValue getShuffleHalfVectors(const SDLoc &DL, SDValue V1, SDValue V2,
                                     ArrayRef<int> HalfMask, int HalfIdx1,
                                     int HalfIdx2, bool UndefLower,
                                     SelectionDAG &DAG, bool UseConcat = false) {
  assert(V1.getValueType() == V2.getValueType() && "Different sized vectors?");
  assert(V1.getValueType().isSimple() && "Expecting only simple types");

  MVT VT = V1.getSimpleValueType();
  MVT HalfVT = VT.getHalfNumVectorElementsVT();
  unsigned HalfNumElts = HalfVT.getVectorNumElements();

  auto getHalfVector = [&](int HalfIdx) {
    if (HalfIdx < 0)
      return DAG.getUNDEF(HalfVT);
    SDValue V = (HalfIdx < 2 ? V1 : V2);
    HalfIdx = (HalfIdx % 2) * HalfNumElts;
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, HalfVT, V,
                       DAG.getIntPtrConstant(HalfIdx, DL));
  };

  // ins undef, (shuf (ext V1, HalfIdx1), (ext V2, HalfIdx2), HalfMask), Offset
  SDValue Half1 = getHalfVector(HalfIdx1);
  SDValue Half2 = getHalfVector(HalfIdx2);
  SDValue V = DAG.getVectorShuffle(HalfVT, DL, Half1, Half2, HalfMask);
  if (UseConcat) {
    SDValue Op0 = V;
    SDValue Op1 = DAG.getUNDEF(HalfVT);
    if (UndefLower)
      std::swap(Op0, Op1);
    return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Op0, Op1);
  }

  unsigned Offset = UndefLower ? HalfNumElts : 0;
  return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, DAG.getUNDEF(VT), V,
                     DAG.getIntPtrConstant(Offset, DL));
}

/// Lower shuffles where an entire half of a 256 or 512-bit vector is UNDEF.
/// This allows for fast cases such as subvector extraction/insertion
/// or shuffling smaller vector types which can lower more efficiently.
static SDValue lowerShuffleWithUndefHalf(const SDLoc &DL, MVT VT, SDValue V1,
                                         SDValue V2, ArrayRef<int> Mask,
                                         const X86Subtarget &Subtarget,
                                         SelectionDAG &DAG) {
  assert((VT.is256BitVector() || VT.is512BitVector()) &&
         "Expected 256-bit or 512-bit vector");

  bool UndefLower = isUndefLowerHalf(Mask);
  if (!UndefLower && !isUndefUpperHalf(Mask))
    return SDValue();

  assert((!UndefLower || !isUndefUpperHalf(Mask)) &&
         "Completely undef shuffle mask should have been simplified already");

  // Upper half is undef and lower half is whole upper subvector.
  // e.g. vector_shuffle <4, 5, 6, 7, u, u, u, u> or <2, 3, u, u>
  MVT HalfVT = VT.getHalfNumVectorElementsVT();
  unsigned HalfNumElts = HalfVT.getVectorNumElements();
  if (!UndefLower &&
      isSequentialOrUndefInRange(Mask, 0, HalfNumElts, HalfNumElts)) {
    SDValue Hi = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, HalfVT, V1,
                             DAG.getIntPtrConstant(HalfNumElts, DL));
    return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, DAG.getUNDEF(VT), Hi,
                       DAG.getIntPtrConstant(0, DL));
  }

  // Lower half is undef and upper half is whole lower subvector.
  // e.g. vector_shuffle <u, u, u, u, 0, 1, 2, 3> or <u, u, 0, 1>
  if (UndefLower &&
      isSequentialOrUndefInRange(Mask, HalfNumElts, HalfNumElts, 0)) {
    SDValue Hi = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, HalfVT, V1,
                             DAG.getIntPtrConstant(0, DL));
    return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, DAG.getUNDEF(VT), Hi,
                       DAG.getIntPtrConstant(HalfNumElts, DL));
  }

  int HalfIdx1, HalfIdx2;
  SmallVector<int, 8> HalfMask(HalfNumElts);
  if (!getHalfShuffleMask(Mask, HalfMask, HalfIdx1, HalfIdx2))
    return SDValue();

  assert(HalfMask.size() == HalfNumElts && "Unexpected shuffle mask length");

  // Only shuffle the halves of the inputs when useful.
  unsigned NumLowerHalves =
      (HalfIdx1 == 0 || HalfIdx1 == 2) + (HalfIdx2 == 0 || HalfIdx2 == 2);
  unsigned NumUpperHalves =
      (HalfIdx1 == 1 || HalfIdx1 == 3) + (HalfIdx2 == 1 || HalfIdx2 == 3);
  assert(NumLowerHalves + NumUpperHalves <= 2 && "Only 1 or 2 halves allowed");

  // Determine the larger pattern of undef/halves, then decide if it's worth
  // splitting the shuffle based on subtarget capabilities and types.
  unsigned EltWidth = VT.getVectorElementType().getSizeInBits();
  if (!UndefLower) {
    // XXXXuuuu: no insert is needed.
    // Always extract lowers when setting lower - these are all free subreg ops.
    if (NumUpperHalves == 0)
      return getShuffleHalfVectors(DL, V1, V2, HalfMask, HalfIdx1, HalfIdx2,
                                   UndefLower, DAG);

    if (NumUpperHalves == 1) {
      // AVX2 has efficient 32/64-bit element cross-lane shuffles.
      if (Subtarget.hasAVX2()) {
        // extract128 + vunpckhps/vshufps, is better than vblend + vpermps.
        if (EltWidth == 32 && NumLowerHalves && HalfVT.is128BitVector() &&
            !is128BitUnpackShuffleMask(HalfMask) &&
            (!isSingleSHUFPSMask(HalfMask) ||
             Subtarget.hasFastVariableShuffle()))
          return SDValue();
        // If this is a unary shuffle (assume that the 2nd operand is
        // canonicalized to undef), then we can use vpermpd. Otherwise, we
        // are better off extracting the upper half of 1 operand and using a
        // narrow shuffle.
        if (EltWidth == 64 && V2.isUndef())
          return SDValue();
      }
      // AVX512 has efficient cross-lane shuffles for all legal 512-bit types.
      if (Subtarget.hasAVX512() && VT.is512BitVector())
        return SDValue();
      // Extract + narrow shuffle is better than the wide alternative.
      return getShuffleHalfVectors(DL, V1, V2, HalfMask, HalfIdx1, HalfIdx2,
                                   UndefLower, DAG);
    }

    // Don't extract both uppers, instead shuffle and then extract.
    assert(NumUpperHalves == 2 && "Half vector count went wrong");
    return SDValue();
  }

  // UndefLower - uuuuXXXX: an insert to high half is required if we split this.
  if (NumUpperHalves == 0) {
    // AVX2 has efficient 64-bit element cross-lane shuffles.
    // TODO: Refine to account for unary shuffle, splat, and other masks?
    if (Subtarget.hasAVX2() && EltWidth == 64)
      return SDValue();
    // AVX512 has efficient cross-lane shuffles for all legal 512-bit types.
    if (Subtarget.hasAVX512() && VT.is512BitVector())
      return SDValue();
    // Narrow shuffle + insert is better than the wide alternative.
    return getShuffleHalfVectors(DL, V1, V2, HalfMask, HalfIdx1, HalfIdx2,
                                 UndefLower, DAG);
  }

  // NumUpperHalves != 0: don't bother with extract, shuffle, and then insert.
  return SDValue();
}

/// Test whether the specified input (0 or 1) is in-place blended by the
/// given mask.
///
/// This returns true if the elements from a particular input are already in the
/// slot required by the given mask and require no permutation.
static bool isShuffleMaskInputInPlace(int Input, ArrayRef<int> Mask) {
  assert((Input == 0 || Input == 1) && "Only two inputs to shuffles.");
  int Size = Mask.size();
  for (int i = 0; i < Size; ++i)
    if (Mask[i] >= 0 && Mask[i] / Size == Input && Mask[i] % Size != i)
      return false;

  return true;
}

/// Handle case where shuffle sources are coming from the same 128-bit lane and
/// every lane can be represented as the same repeating mask - allowing us to
/// shuffle the sources with the repeating shuffle and then permute the result
/// to the destination lanes.
static SDValue lowerShuffleAsRepeatedMaskAndLanePermute(
    const SDLoc &DL, MVT VT, SDValue V1, SDValue V2, ArrayRef<int> Mask,
    const X86Subtarget &Subtarget, SelectionDAG &DAG) {
  int NumElts = VT.getVectorNumElements();
  int NumLanes = VT.getSizeInBits() / 128;
  int NumLaneElts = NumElts / NumLanes;

  // On AVX2 we may be able to just shuffle the lowest elements and then
  // broadcast the result.
  if (Subtarget.hasAVX2()) {
    for (unsigned BroadcastSize : {16, 32, 64}) {
      if (BroadcastSize <= VT.getScalarSizeInBits())
        continue;
      int NumBroadcastElts = BroadcastSize / VT.getScalarSizeInBits();

      // Attempt to match a repeating pattern every NumBroadcastElts,
      // accounting for UNDEFs but only references the lowest 128-bit
      // lane of the inputs.
      auto FindRepeatingBroadcastMask = [&](SmallVectorImpl<int> &RepeatMask) {
        for (int i = 0; i != NumElts; i += NumBroadcastElts)
          for (int j = 0; j != NumBroadcastElts; ++j) {
            int M = Mask[i + j];
            if (M < 0)
              continue;
            int &R = RepeatMask[j];
            if (0 != ((M % NumElts) / NumLaneElts))
              return false;
            if (0 <= R && R != M)
              return false;
            R = M;
          }
        return true;
      };

      SmallVector<int, 8> RepeatMask((unsigned)NumElts, -1);
      if (!FindRepeatingBroadcastMask(RepeatMask))
        continue;

      // Shuffle the (lowest) repeated elements in place for broadcast.
      SDValue RepeatShuf = DAG.getVectorShuffle(VT, DL, V1, V2, RepeatMask);

      // Shuffle the actual broadcast.
      SmallVector<int, 8> BroadcastMask((unsigned)NumElts, -1);
      for (int i = 0; i != NumElts; i += NumBroadcastElts)
        for (int j = 0; j != NumBroadcastElts; ++j)
          BroadcastMask[i + j] = j;
      return DAG.getVectorShuffle(VT, DL, RepeatShuf, DAG.getUNDEF(VT),
                                  BroadcastMask);
    }
  }

  // Bail if the shuffle mask doesn't cross 128-bit lanes.
  if (!is128BitLaneCrossingShuffleMask(VT, Mask))
    return SDValue();

  // Bail if we already have a repeated lane shuffle mask.
  SmallVector<int, 8> RepeatedShuffleMask;
  if (is128BitLaneRepeatedShuffleMask(VT, Mask, RepeatedShuffleMask))
    return SDValue();

  // On AVX2 targets we can permute 256-bit vectors as 64-bit sub-lanes
  // (with PERMQ/PERMPD), otherwise we can only permute whole 128-bit lanes.
  int SubLaneScale = Subtarget.hasAVX2() && VT.is256BitVector() ? 2 : 1;
  int NumSubLanes = NumLanes * SubLaneScale;
  int NumSubLaneElts = NumLaneElts / SubLaneScale;

  // Check that all the sources are coming from the same lane and see if we can
  // form a repeating shuffle mask (local to each sub-lane). At the same time,
  // determine the source sub-lane for each destination sub-lane.
  int TopSrcSubLane = -1;
  SmallVector<int, 8> Dst2SrcSubLanes((unsigned)NumSubLanes, -1);
  SmallVector<int, 8> RepeatedSubLaneMasks[2] = {
      SmallVector<int, 8>((unsigned)NumSubLaneElts, SM_SentinelUndef),
      SmallVector<int, 8>((unsigned)NumSubLaneElts, SM_SentinelUndef)};

  for (int DstSubLane = 0; DstSubLane != NumSubLanes; ++DstSubLane) {
    // Extract the sub-lane mask, check that it all comes from the same lane
    // and normalize the mask entries to come from the first lane.
    int SrcLane = -1;
    SmallVector<int, 8> SubLaneMask((unsigned)NumSubLaneElts, -1);
    for (int Elt = 0; Elt != NumSubLaneElts; ++Elt) {
      int M = Mask[(DstSubLane * NumSubLaneElts) + Elt];
      if (M < 0)
        continue;
      int Lane = (M % NumElts) / NumLaneElts;
      if ((0 <= SrcLane) && (SrcLane != Lane))
        return SDValue();
      SrcLane = Lane;
      int LocalM = (M % NumLaneElts) + (M < NumElts ? 0 : NumElts);
      SubLaneMask[Elt] = LocalM;
    }

    // Whole sub-lane is UNDEF.
    if (SrcLane < 0)
      continue;

    // Attempt to match against the candidate repeated sub-lane masks.
    for (int SubLane = 0; SubLane != SubLaneScale; ++SubLane) {
      auto MatchMasks = [NumSubLaneElts](ArrayRef<int> M1, ArrayRef<int> M2) {
        for (int i = 0; i != NumSubLaneElts; ++i) {
          if (M1[i] < 0 || M2[i] < 0)
            continue;
          if (M1[i] != M2[i])
            return false;
        }
        return true;
      };

      auto &RepeatedSubLaneMask = RepeatedSubLaneMasks[SubLane];
      if (!MatchMasks(SubLaneMask, RepeatedSubLaneMask))
        continue;

      // Merge the sub-lane mask into the matching repeated sub-lane mask.
      for (int i = 0; i != NumSubLaneElts; ++i) {
        int M = SubLaneMask[i];
        if (M < 0)
          continue;
        assert((RepeatedSubLaneMask[i] < 0 || RepeatedSubLaneMask[i] == M) &&
               "Unexpected mask element");
        RepeatedSubLaneMask[i] = M;
      }

      // Track the top most source sub-lane - by setting the remaining to UNDEF
      // we can greatly simplify shuffle matching.
      int SrcSubLane = (SrcLane * SubLaneScale) + SubLane;
      TopSrcSubLane = std::max(TopSrcSubLane, SrcSubLane);
      Dst2SrcSubLanes[DstSubLane] = SrcSubLane;
      break;
    }

    // Bail if we failed to find a matching repeated sub-lane mask.
    if (Dst2SrcSubLanes[DstSubLane] < 0)
      return SDValue();
  }
  assert(0 <= TopSrcSubLane && TopSrcSubLane < NumSubLanes &&
         "Unexpected source lane");

  // Create a repeating shuffle mask for the entire vector.
  SmallVector<int, 8> RepeatedMask((unsigned)NumElts, -1);
  for (int SubLane = 0; SubLane <= TopSrcSubLane; ++SubLane) {
    int Lane = SubLane / SubLaneScale;
    auto &RepeatedSubLaneMask = RepeatedSubLaneMasks[SubLane % SubLaneScale];
    for (int Elt = 0; Elt != NumSubLaneElts; ++Elt) {
      int M = RepeatedSubLaneMask[Elt];
      if (M < 0)
        continue;
      int Idx = (SubLane * NumSubLaneElts) + Elt;
      RepeatedMask[Idx] = M + (Lane * NumLaneElts);
    }
  }
  SDValue RepeatedShuffle = DAG.getVectorShuffle(VT, DL, V1, V2, RepeatedMask);

  // Shuffle each source sub-lane to its destination.
  SmallVector<int, 8> SubLaneMask((unsigned)NumElts, -1);
  for (int i = 0; i != NumElts; i += NumSubLaneElts) {
    int SrcSubLane = Dst2SrcSubLanes[i / NumSubLaneElts];
    if (SrcSubLane < 0)
      continue;
    for (int j = 0; j != NumSubLaneElts; ++j)
      SubLaneMask[i + j] = j + (SrcSubLane * NumSubLaneElts);
  }

  return DAG.getVectorShuffle(VT, DL, RepeatedShuffle, DAG.getUNDEF(VT),
                              SubLaneMask);
}

static bool matchShuffleWithSHUFPD(MVT VT, SDValue &V1, SDValue &V2,
                                   unsigned &ShuffleImm, ArrayRef<int> Mask) {
  int NumElts = VT.getVectorNumElements();
  assert(VT.getScalarSizeInBits() == 64 &&
         (NumElts == 2 || NumElts == 4 || NumElts == 8) &&
         "Unexpected data type for VSHUFPD");

  // Mask for V8F64: 0/1,  8/9,  2/3,  10/11, 4/5, ..
  // Mask for V4F64; 0/1,  4/5,  2/3,  6/7..
  ShuffleImm = 0;
  bool ShufpdMask = true;
  bool CommutableMask = true;
  for (int i = 0; i < NumElts; ++i) {
    if (Mask[i] == SM_SentinelUndef)
      continue;
    if (Mask[i] < 0)
      return false;
    int Val = (i & 6) + NumElts * (i & 1);
    int CommutVal = (i & 0xe) + NumElts * ((i & 1) ^ 1);
    if (Mask[i] < Val || Mask[i] > Val + 1)
      ShufpdMask = false;
    if (Mask[i] < CommutVal || Mask[i] > CommutVal + 1)
      CommutableMask = false;
    ShuffleImm |= (Mask[i] % 2) << i;
  }

  if (ShufpdMask)
    return true;
  if (CommutableMask) {
    std::swap(V1, V2);
    return true;
  }

  return false;
}

static SDValue lowerShuffleWithSHUFPD(const SDLoc &DL, MVT VT,
                                      ArrayRef<int> Mask, SDValue V1,
                                      SDValue V2, SelectionDAG &DAG) {
  assert((VT == MVT::v2f64 || VT == MVT::v4f64 || VT == MVT::v8f64)&&
         "Unexpected data type for VSHUFPD");

  unsigned Immediate = 0;
  if (!matchShuffleWithSHUFPD(VT, V1, V2, Immediate, Mask))
    return SDValue();

  return DAG.getNode(X86ISD::SHUFP, DL, VT, V1, V2,
                     DAG.getConstant(Immediate, DL, MVT::i8));
}

/// Handle lowering of 4-lane 64-bit floating point shuffles.
///
/// Also ends up handling lowering of 4-lane 64-bit integer shuffles when AVX2
/// isn't available.
static SDValue lowerV4F64Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v4f64 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v4f64 && "Bad operand type!");
  assert(Mask.size() == 4 && "Unexpected mask size for v4 shuffle!");

  if (SDValue V = lowerV2X128Shuffle(DL, MVT::v4f64, V1, V2, Mask, Zeroable,
                                     Subtarget, DAG))
    return V;

  if (V2.isUndef()) {
    // Check for being able to broadcast a single element.
    if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v4f64, V1, V2,
                                                    Mask, Subtarget, DAG))
      return Broadcast;

    // Use low duplicate instructions for masks that match their pattern.
    if (isShuffleEquivalent(V1, V2, Mask, {0, 0, 2, 2}))
      return DAG.getNode(X86ISD::MOVDDUP, DL, MVT::v4f64, V1);

    if (!is128BitLaneCrossingShuffleMask(MVT::v4f64, Mask)) {
      // Non-half-crossing single input shuffles can be lowered with an
      // interleaved permutation.
      unsigned VPERMILPMask = (Mask[0] == 1) | ((Mask[1] == 1) << 1) |
                              ((Mask[2] == 3) << 2) | ((Mask[3] == 3) << 3);
      return DAG.getNode(X86ISD::VPERMILPI, DL, MVT::v4f64, V1,
                         DAG.getConstant(VPERMILPMask, DL, MVT::i8));
    }

    // With AVX2 we have direct support for this permutation.
    if (Subtarget.hasAVX2())
      return DAG.getNode(X86ISD::VPERMI, DL, MVT::v4f64, V1,
                         getV4X86ShuffleImm8ForMask(Mask, DL, DAG));

    // Try to create an in-lane repeating shuffle mask and then shuffle the
    // results into the target lanes.
    if (SDValue V = lowerShuffleAsRepeatedMaskAndLanePermute(
            DL, MVT::v4f64, V1, V2, Mask, Subtarget, DAG))
      return V;

    // Try to permute the lanes and then use a per-lane permute.
    if (SDValue V = lowerShuffleAsLanePermuteAndPermute(DL, MVT::v4f64, V1, V2,
                                                        Mask, DAG, Subtarget))
      return V;

    // Otherwise, fall back.
    return lowerShuffleAsLanePermuteAndShuffle(DL, MVT::v4f64, V1, V2, Mask,
                                               DAG, Subtarget);
  }

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v4f64, Mask, V1, V2, DAG))
    return V;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v4f64, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  // Check if the blend happens to exactly fit that of SHUFPD.
  if (SDValue Op = lowerShuffleWithSHUFPD(DL, MVT::v4f64, Mask, V1, V2, DAG))
    return Op;

  // If we have one input in place, then we can permute the other input and
  // blend the result.
  if (isShuffleMaskInputInPlace(0, Mask) || isShuffleMaskInputInPlace(1, Mask))
    return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v4f64, V1, V2, Mask,
                                                Subtarget, DAG);

  // Try to create an in-lane repeating shuffle mask and then shuffle the
  // results into the target lanes.
  if (SDValue V = lowerShuffleAsRepeatedMaskAndLanePermute(
          DL, MVT::v4f64, V1, V2, Mask, Subtarget, DAG))
    return V;

  // Try to simplify this by merging 128-bit lanes to enable a lane-based
  // shuffle. However, if we have AVX2 and either inputs are already in place,
  // we will be able to shuffle even across lanes the other input in a single
  // instruction so skip this pattern.
  if (!(Subtarget.hasAVX2() && (isShuffleMaskInputInPlace(0, Mask) ||
                                isShuffleMaskInputInPlace(1, Mask))))
    if (SDValue V = lowerShuffleAsLanePermuteAndRepeatedMask(
            DL, MVT::v4f64, V1, V2, Mask, Subtarget, DAG))
      return V;

  // If we have VLX support, we can use VEXPAND.
  if (Subtarget.hasVLX())
    if (SDValue V = lowerShuffleToEXPAND(DL, MVT::v4f64, Zeroable, Mask, V1, V2,
                                         DAG, Subtarget))
      return V;

  // If we have AVX2 then we always want to lower with a blend because an v4 we
  // can fully permute the elements.
  if (Subtarget.hasAVX2())
    return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v4f64, V1, V2, Mask,
                                                Subtarget, DAG);

  // Otherwise fall back on generic lowering.
  return lowerShuffleAsSplitOrBlend(DL, MVT::v4f64, V1, V2, Mask,
                                    Subtarget, DAG);
}

/// Handle lowering of 4-lane 64-bit integer shuffles.
///
/// This routine is only called when we have AVX2 and thus a reasonable
/// instruction set for v4i64 shuffling..
static SDValue lowerV4I64Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v4i64 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v4i64 && "Bad operand type!");
  assert(Mask.size() == 4 && "Unexpected mask size for v4 shuffle!");
  assert(Subtarget.hasAVX2() && "We can only lower v4i64 with AVX2!");

  if (SDValue V = lowerV2X128Shuffle(DL, MVT::v4i64, V1, V2, Mask, Zeroable,
                                     Subtarget, DAG))
    return V;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v4i64, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  // Check for being able to broadcast a single element.
  if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v4i64, V1, V2, Mask,
                                                  Subtarget, DAG))
    return Broadcast;

  if (V2.isUndef()) {
    // When the shuffle is mirrored between the 128-bit lanes of the unit, we
    // can use lower latency instructions that will operate on both lanes.
    SmallVector<int, 2> RepeatedMask;
    if (is128BitLaneRepeatedShuffleMask(MVT::v4i64, Mask, RepeatedMask)) {
      SmallVector<int, 4> PSHUFDMask;
      scaleShuffleMask<int>(2, RepeatedMask, PSHUFDMask);
      return DAG.getBitcast(
          MVT::v4i64,
          DAG.getNode(X86ISD::PSHUFD, DL, MVT::v8i32,
                      DAG.getBitcast(MVT::v8i32, V1),
                      getV4X86ShuffleImm8ForMask(PSHUFDMask, DL, DAG)));
    }

    // AVX2 provides a direct instruction for permuting a single input across
    // lanes.
    return DAG.getNode(X86ISD::VPERMI, DL, MVT::v4i64, V1,
                       getV4X86ShuffleImm8ForMask(Mask, DL, DAG));
  }

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v4i64, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // If we have VLX support, we can use VALIGN or VEXPAND.
  if (Subtarget.hasVLX()) {
    if (SDValue Rotate = lowerShuffleAsRotate(DL, MVT::v4i64, V1, V2, Mask,
                                              Subtarget, DAG))
      return Rotate;

    if (SDValue V = lowerShuffleToEXPAND(DL, MVT::v4i64, Zeroable, Mask, V1, V2,
                                         DAG, Subtarget))
      return V;
  }

  // Try to use PALIGNR.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v4i64, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v4i64, Mask, V1, V2, DAG))
    return V;

  // If we have one input in place, then we can permute the other input and
  // blend the result.
  if (isShuffleMaskInputInPlace(0, Mask) || isShuffleMaskInputInPlace(1, Mask))
    return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v4i64, V1, V2, Mask,
                                                Subtarget, DAG);

  // Try to create an in-lane repeating shuffle mask and then shuffle the
  // results into the target lanes.
  if (SDValue V = lowerShuffleAsRepeatedMaskAndLanePermute(
          DL, MVT::v4i64, V1, V2, Mask, Subtarget, DAG))
    return V;

  // Try to simplify this by merging 128-bit lanes to enable a lane-based
  // shuffle. However, if we have AVX2 and either inputs are already in place,
  // we will be able to shuffle even across lanes the other input in a single
  // instruction so skip this pattern.
  if (!isShuffleMaskInputInPlace(0, Mask) &&
      !isShuffleMaskInputInPlace(1, Mask))
    if (SDValue Result = lowerShuffleAsLanePermuteAndRepeatedMask(
            DL, MVT::v4i64, V1, V2, Mask, Subtarget, DAG))
      return Result;

  // Otherwise fall back on generic blend lowering.
  return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v4i64, V1, V2, Mask,
                                              Subtarget, DAG);
}

/// Handle lowering of 8-lane 32-bit floating point shuffles.
///
/// Also ends up handling lowering of 8-lane 32-bit integer shuffles when AVX2
/// isn't available.
static SDValue lowerV8F32Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v8f32 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v8f32 && "Bad operand type!");
  assert(Mask.size() == 8 && "Unexpected mask size for v8 shuffle!");

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v8f32, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  // Check for being able to broadcast a single element.
  if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v8f32, V1, V2, Mask,
                                                  Subtarget, DAG))
    return Broadcast;

  // If the shuffle mask is repeated in each 128-bit lane, we have many more
  // options to efficiently lower the shuffle.
  SmallVector<int, 4> RepeatedMask;
  if (is128BitLaneRepeatedShuffleMask(MVT::v8f32, Mask, RepeatedMask)) {
    assert(RepeatedMask.size() == 4 &&
           "Repeated masks must be half the mask width!");

    // Use even/odd duplicate instructions for masks that match their pattern.
    if (isShuffleEquivalent(V1, V2, RepeatedMask, {0, 0, 2, 2}))
      return DAG.getNode(X86ISD::MOVSLDUP, DL, MVT::v8f32, V1);
    if (isShuffleEquivalent(V1, V2, RepeatedMask, {1, 1, 3, 3}))
      return DAG.getNode(X86ISD::MOVSHDUP, DL, MVT::v8f32, V1);

    if (V2.isUndef())
      return DAG.getNode(X86ISD::VPERMILPI, DL, MVT::v8f32, V1,
                         getV4X86ShuffleImm8ForMask(RepeatedMask, DL, DAG));

    // Use dedicated unpack instructions for masks that match their pattern.
    if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v8f32, Mask, V1, V2, DAG))
      return V;

    // Otherwise, fall back to a SHUFPS sequence. Here it is important that we
    // have already handled any direct blends.
    return lowerShuffleWithSHUFPS(DL, MVT::v8f32, RepeatedMask, V1, V2, DAG);
  }

  // Try to create an in-lane repeating shuffle mask and then shuffle the
  // results into the target lanes.
  if (SDValue V = lowerShuffleAsRepeatedMaskAndLanePermute(
          DL, MVT::v8f32, V1, V2, Mask, Subtarget, DAG))
    return V;

  // If we have a single input shuffle with different shuffle patterns in the
  // two 128-bit lanes use the variable mask to VPERMILPS.
  if (V2.isUndef()) {
    SDValue VPermMask = getConstVector(Mask, MVT::v8i32, DAG, DL, true);
    if (!is128BitLaneCrossingShuffleMask(MVT::v8f32, Mask))
      return DAG.getNode(X86ISD::VPERMILPV, DL, MVT::v8f32, V1, VPermMask);

    if (Subtarget.hasAVX2())
      return DAG.getNode(X86ISD::VPERMV, DL, MVT::v8f32, VPermMask, V1);

    // Otherwise, fall back.
    return lowerShuffleAsLanePermuteAndShuffle(DL, MVT::v8f32, V1, V2, Mask,
                                               DAG, Subtarget);
  }

  // Try to simplify this by merging 128-bit lanes to enable a lane-based
  // shuffle.
  if (SDValue Result = lowerShuffleAsLanePermuteAndRepeatedMask(
          DL, MVT::v8f32, V1, V2, Mask, Subtarget, DAG))
    return Result;

  // If we have VLX support, we can use VEXPAND.
  if (Subtarget.hasVLX())
    if (SDValue V = lowerShuffleToEXPAND(DL, MVT::v8f32, Zeroable, Mask, V1, V2,
                                         DAG, Subtarget))
      return V;

  // For non-AVX512 if the Mask is of 16bit elements in lane then try to split
  // since after split we get a more efficient code using vpunpcklwd and
  // vpunpckhwd instrs than vblend.
  if (!Subtarget.hasAVX512() && isUnpackWdShuffleMask(Mask, MVT::v8f32))
    if (SDValue V = lowerShuffleAsSplitOrBlend(DL, MVT::v8f32, V1, V2, Mask,
                                               Subtarget, DAG))
      return V;

  // If we have AVX2 then we always want to lower with a blend because at v8 we
  // can fully permute the elements.
  if (Subtarget.hasAVX2())
    return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v8f32, V1, V2, Mask,
                                                Subtarget, DAG);

  // Otherwise fall back on generic lowering.
  return lowerShuffleAsSplitOrBlend(DL, MVT::v8f32, V1, V2, Mask,
                                    Subtarget, DAG);
}

/// Handle lowering of 8-lane 32-bit integer shuffles.
///
/// This routine is only called when we have AVX2 and thus a reasonable
/// instruction set for v8i32 shuffling..
static SDValue lowerV8I32Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v8i32 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v8i32 && "Bad operand type!");
  assert(Mask.size() == 8 && "Unexpected mask size for v8 shuffle!");
  assert(Subtarget.hasAVX2() && "We can only lower v8i32 with AVX2!");

  // Whenever we can lower this as a zext, that instruction is strictly faster
  // than any alternative. It also allows us to fold memory operands into the
  // shuffle in many cases.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(DL, MVT::v8i32, V1, V2, Mask,
                                                   Zeroable, Subtarget, DAG))
    return ZExt;

  // For non-AVX512 if the Mask is of 16bit elements in lane then try to split
  // since after split we get a more efficient code than vblend by using
  // vpunpcklwd and vpunpckhwd instrs.
  if (isUnpackWdShuffleMask(Mask, MVT::v8i32) && !V2.isUndef() &&
      !Subtarget.hasAVX512())
    if (SDValue V = lowerShuffleAsSplitOrBlend(DL, MVT::v8i32, V1, V2, Mask,
                                               Subtarget, DAG))
      return V;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v8i32, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  // Check for being able to broadcast a single element.
  if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v8i32, V1, V2, Mask,
                                                  Subtarget, DAG))
    return Broadcast;

  // If the shuffle mask is repeated in each 128-bit lane we can use more
  // efficient instructions that mirror the shuffles across the two 128-bit
  // lanes.
  SmallVector<int, 4> RepeatedMask;
  bool Is128BitLaneRepeatedShuffle =
      is128BitLaneRepeatedShuffleMask(MVT::v8i32, Mask, RepeatedMask);
  if (Is128BitLaneRepeatedShuffle) {
    assert(RepeatedMask.size() == 4 && "Unexpected repeated mask size!");
    if (V2.isUndef())
      return DAG.getNode(X86ISD::PSHUFD, DL, MVT::v8i32, V1,
                         getV4X86ShuffleImm8ForMask(RepeatedMask, DL, DAG));

    // Use dedicated unpack instructions for masks that match their pattern.
    if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v8i32, Mask, V1, V2, DAG))
      return V;
  }

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v8i32, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // If we have VLX support, we can use VALIGN or EXPAND.
  if (Subtarget.hasVLX()) {
    if (SDValue Rotate = lowerShuffleAsRotate(DL, MVT::v8i32, V1, V2, Mask,
                                              Subtarget, DAG))
      return Rotate;

    if (SDValue V = lowerShuffleToEXPAND(DL, MVT::v8i32, Zeroable, Mask, V1, V2,
                                         DAG, Subtarget))
      return V;
  }

  // Try to use byte rotation instructions.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v8i32, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  // Try to create an in-lane repeating shuffle mask and then shuffle the
  // results into the target lanes.
  if (SDValue V = lowerShuffleAsRepeatedMaskAndLanePermute(
          DL, MVT::v8i32, V1, V2, Mask, Subtarget, DAG))
    return V;

  // If the shuffle patterns aren't repeated but it is a single input, directly
  // generate a cross-lane VPERMD instruction.
  if (V2.isUndef()) {
    SDValue VPermMask = getConstVector(Mask, MVT::v8i32, DAG, DL, true);
    return DAG.getNode(X86ISD::VPERMV, DL, MVT::v8i32, VPermMask, V1);
  }

  // Assume that a single SHUFPS is faster than an alternative sequence of
  // multiple instructions (even if the CPU has a domain penalty).
  // If some CPU is harmed by the domain switch, we can fix it in a later pass.
  if (Is128BitLaneRepeatedShuffle && isSingleSHUFPSMask(RepeatedMask)) {
    SDValue CastV1 = DAG.getBitcast(MVT::v8f32, V1);
    SDValue CastV2 = DAG.getBitcast(MVT::v8f32, V2);
    SDValue ShufPS = lowerShuffleWithSHUFPS(DL, MVT::v8f32, RepeatedMask,
                                            CastV1, CastV2, DAG);
    return DAG.getBitcast(MVT::v8i32, ShufPS);
  }

  // Try to simplify this by merging 128-bit lanes to enable a lane-based
  // shuffle.
  if (SDValue Result = lowerShuffleAsLanePermuteAndRepeatedMask(
          DL, MVT::v8i32, V1, V2, Mask, Subtarget, DAG))
    return Result;

  // Otherwise fall back on generic blend lowering.
  return lowerShuffleAsDecomposedShuffleBlend(DL, MVT::v8i32, V1, V2, Mask,
                                              Subtarget, DAG);
}

/// Handle lowering of 16-lane 16-bit integer shuffles.
///
/// This routine is only called when we have AVX2 and thus a reasonable
/// instruction set for v16i16 shuffling..
static SDValue lowerV16I16Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                  const APInt &Zeroable, SDValue V1, SDValue V2,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v16i16 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v16i16 && "Bad operand type!");
  assert(Mask.size() == 16 && "Unexpected mask size for v16 shuffle!");
  assert(Subtarget.hasAVX2() && "We can only lower v16i16 with AVX2!");

  // Whenever we can lower this as a zext, that instruction is strictly faster
  // than any alternative. It also allows us to fold memory operands into the
  // shuffle in many cases.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(
          DL, MVT::v16i16, V1, V2, Mask, Zeroable, Subtarget, DAG))
    return ZExt;

  // Check for being able to broadcast a single element.
  if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v16i16, V1, V2, Mask,
                                                  Subtarget, DAG))
    return Broadcast;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v16i16, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v16i16, Mask, V1, V2, DAG))
    return V;

  // Use dedicated pack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithPACK(DL, MVT::v16i16, Mask, V1, V2, DAG,
                                       Subtarget))
    return V;

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v16i16, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // Try to use byte rotation instructions.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v16i16, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  // Try to create an in-lane repeating shuffle mask and then shuffle the
  // results into the target lanes.
  if (SDValue V = lowerShuffleAsRepeatedMaskAndLanePermute(
          DL, MVT::v16i16, V1, V2, Mask, Subtarget, DAG))
    return V;

  if (V2.isUndef()) {
    // There are no generalized cross-lane shuffle operations available on i16
    // element types.
    if (is128BitLaneCrossingShuffleMask(MVT::v16i16, Mask)) {
      if (SDValue V = lowerShuffleAsLanePermuteAndPermute(
              DL, MVT::v16i16, V1, V2, Mask, DAG, Subtarget))
        return V;

      return lowerShuffleAsLanePermuteAndShuffle(DL, MVT::v16i16, V1, V2, Mask,
                                                 DAG, Subtarget);
    }

    SmallVector<int, 8> RepeatedMask;
    if (is128BitLaneRepeatedShuffleMask(MVT::v16i16, Mask, RepeatedMask)) {
      // As this is a single-input shuffle, the repeated mask should be
      // a strictly valid v8i16 mask that we can pass through to the v8i16
      // lowering to handle even the v16 case.
      return lowerV8I16GeneralSingleInputShuffle(
          DL, MVT::v16i16, V1, RepeatedMask, Subtarget, DAG);
    }
  }

  if (SDValue PSHUFB = lowerShuffleWithPSHUFB(DL, MVT::v16i16, Mask, V1, V2,
                                              Zeroable, Subtarget, DAG))
    return PSHUFB;

  // AVX512BWVL can lower to VPERMW.
  if (Subtarget.hasBWI() && Subtarget.hasVLX())
    return lowerShuffleWithPERMV(DL, MVT::v16i16, Mask, V1, V2, DAG);

  // Try to simplify this by merging 128-bit lanes to enable a lane-based
  // shuffle.
  if (SDValue Result = lowerShuffleAsLanePermuteAndRepeatedMask(
          DL, MVT::v16i16, V1, V2, Mask, Subtarget, DAG))
    return Result;

  // Try to permute the lanes and then use a per-lane permute.
  if (SDValue V = lowerShuffleAsLanePermuteAndPermute(
          DL, MVT::v16i16, V1, V2, Mask, DAG, Subtarget))
    return V;

  // Otherwise fall back on generic lowering.
  return lowerShuffleAsSplitOrBlend(DL, MVT::v16i16, V1, V2, Mask,
                                    Subtarget, DAG);
}

/// Handle lowering of 32-lane 8-bit integer shuffles.
///
/// This routine is only called when we have AVX2 and thus a reasonable
/// instruction set for v32i8 shuffling..
static SDValue lowerV32I8Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v32i8 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v32i8 && "Bad operand type!");
  assert(Mask.size() == 32 && "Unexpected mask size for v32 shuffle!");
  assert(Subtarget.hasAVX2() && "We can only lower v32i8 with AVX2!");

  // Whenever we can lower this as a zext, that instruction is strictly faster
  // than any alternative. It also allows us to fold memory operands into the
  // shuffle in many cases.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(DL, MVT::v32i8, V1, V2, Mask,
                                                   Zeroable, Subtarget, DAG))
    return ZExt;

  // Check for being able to broadcast a single element.
  if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, MVT::v32i8, V1, V2, Mask,
                                                  Subtarget, DAG))
    return Broadcast;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v32i8, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v32i8, Mask, V1, V2, DAG))
    return V;

  // Use dedicated pack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithPACK(DL, MVT::v32i8, Mask, V1, V2, DAG,
                                       Subtarget))
    return V;

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v32i8, V1, V2, Mask,
                                                Zeroable, Subtarget, DAG))
    return Shift;

  // Try to use byte rotation instructions.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v32i8, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  // Try to create an in-lane repeating shuffle mask and then shuffle the
  // results into the target lanes.
  if (SDValue V = lowerShuffleAsRepeatedMaskAndLanePermute(
          DL, MVT::v32i8, V1, V2, Mask, Subtarget, DAG))
    return V;

  // There are no generalized cross-lane shuffle operations available on i8
  // element types.
  if (V2.isUndef() && is128BitLaneCrossingShuffleMask(MVT::v32i8, Mask)) {
    if (SDValue V = lowerShuffleAsLanePermuteAndPermute(
            DL, MVT::v32i8, V1, V2, Mask, DAG, Subtarget))
      return V;

    return lowerShuffleAsLanePermuteAndShuffle(DL, MVT::v32i8, V1, V2, Mask,
                                               DAG, Subtarget);
  }

  if (SDValue PSHUFB = lowerShuffleWithPSHUFB(DL, MVT::v32i8, Mask, V1, V2,
                                              Zeroable, Subtarget, DAG))
    return PSHUFB;

  // AVX512VBMIVL can lower to VPERMB.
  if (Subtarget.hasVBMI() && Subtarget.hasVLX())
    return lowerShuffleWithPERMV(DL, MVT::v32i8, Mask, V1, V2, DAG);

  // Try to simplify this by merging 128-bit lanes to enable a lane-based
  // shuffle.
  if (SDValue Result = lowerShuffleAsLanePermuteAndRepeatedMask(
          DL, MVT::v32i8, V1, V2, Mask, Subtarget, DAG))
    return Result;

  // Try to permute the lanes and then use a per-lane permute.
  if (SDValue V = lowerShuffleAsLanePermuteAndPermute(
          DL, MVT::v32i8, V1, V2, Mask, DAG, Subtarget))
    return V;

  // Otherwise fall back on generic lowering.
  return lowerShuffleAsSplitOrBlend(DL, MVT::v32i8, V1, V2, Mask,
                                    Subtarget, DAG);
}

/// High-level routine to lower various 256-bit x86 vector shuffles.
///
/// This routine either breaks down the specific type of a 256-bit x86 vector
/// shuffle or splits it into two 128-bit shuffles and fuses the results back
/// together based on the available instructions.
static SDValue lower256BitShuffle(const SDLoc &DL, ArrayRef<int> Mask, MVT VT,
                                  SDValue V1, SDValue V2, const APInt &Zeroable,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  // If we have a single input to the zero element, insert that into V1 if we
  // can do so cheaply.
  int NumElts = VT.getVectorNumElements();
  int NumV2Elements = count_if(Mask, [NumElts](int M) { return M >= NumElts; });

  if (NumV2Elements == 1 && Mask[0] >= NumElts)
    if (SDValue Insertion = lowerShuffleAsElementInsertion(
            DL, VT, V1, V2, Mask, Zeroable, Subtarget, DAG))
      return Insertion;

  // Handle special cases where the lower or upper half is UNDEF.
  if (SDValue V =
          lowerShuffleWithUndefHalf(DL, VT, V1, V2, Mask, Subtarget, DAG))
    return V;

  // There is a really nice hard cut-over between AVX1 and AVX2 that means we
  // can check for those subtargets here and avoid much of the subtarget
  // querying in the per-vector-type lowering routines. With AVX1 we have
  // essentially *zero* ability to manipulate a 256-bit vector with integer
  // types. Since we'll use floating point types there eventually, just
  // immediately cast everything to a float and operate entirely in that domain.
  if (VT.isInteger() && !Subtarget.hasAVX2()) {
    int ElementBits = VT.getScalarSizeInBits();
    if (ElementBits < 32) {
      // No floating point type available, if we can't use the bit operations
      // for masking/blending then decompose into 128-bit vectors.
      if (SDValue V = lowerShuffleAsBitMask(DL, VT, V1, V2, Mask, Zeroable,
                                            Subtarget, DAG))
        return V;
      if (SDValue V = lowerShuffleAsBitBlend(DL, VT, V1, V2, Mask, DAG))
        return V;
      return splitAndLowerShuffle(DL, VT, V1, V2, Mask, DAG);
    }

    MVT FpVT = MVT::getVectorVT(MVT::getFloatingPointVT(ElementBits),
                                VT.getVectorNumElements());
    V1 = DAG.getBitcast(FpVT, V1);
    V2 = DAG.getBitcast(FpVT, V2);
    return DAG.getBitcast(VT, DAG.getVectorShuffle(FpVT, DL, V1, V2, Mask));
  }

  switch (VT.SimpleTy) {
  case MVT::v4f64:
    return lowerV4F64Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v4i64:
    return lowerV4I64Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v8f32:
    return lowerV8F32Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v8i32:
    return lowerV8I32Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v16i16:
    return lowerV16I16Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v32i8:
    return lowerV32I8Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);

  default:
    llvm_unreachable("Not a valid 256-bit x86 vector type!");
  }
}

/// Try to lower a vector shuffle as a 128-bit shuffles.
static SDValue lowerV4X128Shuffle(const SDLoc &DL, MVT VT, ArrayRef<int> Mask,
                                  const APInt &Zeroable, SDValue V1, SDValue V2,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  assert(VT.getScalarSizeInBits() == 64 &&
         "Unexpected element type size for 128bit shuffle.");

  // To handle 256 bit vector requires VLX and most probably
  // function lowerV2X128VectorShuffle() is better solution.
  assert(VT.is512BitVector() && "Unexpected vector size for 512bit shuffle.");

  // TODO - use Zeroable like we do for lowerV2X128VectorShuffle?
  SmallVector<int, 4> WidenedMask;
  if (!X86::canWidenShuffleElements(Mask, WidenedMask))
    return SDValue();

  // Try to use an insert into a zero vector.
  if (WidenedMask[0] == 0 && (Zeroable & 0xf0) == 0xf0 &&
      (WidenedMask[1] == 1 || (Zeroable & 0x0c) == 0x0c)) {
    unsigned NumElts = ((Zeroable & 0x0c) == 0x0c) ? 2 : 4;
    MVT SubVT = MVT::getVectorVT(VT.getVectorElementType(), NumElts);
    SDValue LoV = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVT, V1,
                              DAG.getIntPtrConstant(0, DL));
    return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT,
                       getZeroVector(VT, Subtarget, DAG, DL), LoV,
                       DAG.getIntPtrConstant(0, DL));
  }

  // Check for patterns which can be matched with a single insert of a 256-bit
  // subvector.
  bool OnlyUsesV1 = isShuffleEquivalent(V1, V2, Mask,
                                        {0, 1, 2, 3, 0, 1, 2, 3});
  if (OnlyUsesV1 || isShuffleEquivalent(V1, V2, Mask,
                                        {0, 1, 2, 3, 8, 9, 10, 11})) {
    MVT SubVT = MVT::getVectorVT(VT.getVectorElementType(), 4);
    SDValue SubVec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVT,
                                 OnlyUsesV1 ? V1 : V2,
                              DAG.getIntPtrConstant(0, DL));
    return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, V1, SubVec,
                       DAG.getIntPtrConstant(4, DL));
  }

  assert(WidenedMask.size() == 4);

  // See if this is an insertion of the lower 128-bits of V2 into V1.
  bool IsInsert = true;
  int V2Index = -1;
  for (int i = 0; i < 4; ++i) {
    assert(WidenedMask[i] >= -1);
    if (WidenedMask[i] < 0)
      continue;

    // Make sure all V1 subvectors are in place.
    if (WidenedMask[i] < 4) {
      if (WidenedMask[i] != i) {
        IsInsert = false;
        break;
      }
    } else {
      // Make sure we only have a single V2 index and its the lowest 128-bits.
      if (V2Index >= 0 || WidenedMask[i] != 4) {
        IsInsert = false;
        break;
      }
      V2Index = i;
    }
  }
  if (IsInsert && V2Index >= 0) {
    MVT SubVT = MVT::getVectorVT(VT.getVectorElementType(), 2);
    SDValue Subvec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SubVT, V2,
                                 DAG.getIntPtrConstant(0, DL));
    return insert128BitVector(V1, Subvec, V2Index * 2, DAG, DL);
  }

  // Try to lower to vshuf64x2/vshuf32x4.
  SDValue Ops[2] = {DAG.getUNDEF(VT), DAG.getUNDEF(VT)};
  unsigned PermMask = 0;
  // Insure elements came from the same Op.
  for (int i = 0; i < 4; ++i) {
    assert(WidenedMask[i] >= -1);
    if (WidenedMask[i] < 0)
      continue;

    SDValue Op = WidenedMask[i] >= 4 ? V2 : V1;
    unsigned OpIndex = i / 2;
    if (Ops[OpIndex].isUndef())
      Ops[OpIndex] = Op;
    else if (Ops[OpIndex] != Op)
      return SDValue();

    // Convert the 128-bit shuffle mask selection values into 128-bit selection
    // bits defined by a vshuf64x2 instruction's immediate control byte.
    PermMask |= (WidenedMask[i] % 4) << (i * 2);
  }

  return DAG.getNode(X86ISD::SHUF128, DL, VT, Ops[0], Ops[1],
                     DAG.getConstant(PermMask, DL, MVT::i8));
}

/// Handle lowering of 8-lane 64-bit floating point shuffles.
static SDValue lowerV8F64Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v8f64 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v8f64 && "Bad operand type!");
  assert(Mask.size() == 8 && "Unexpected mask size for v8 shuffle!");

  if (V2.isUndef()) {
    // Use low duplicate instructions for masks that match their pattern.
    if (isShuffleEquivalent(V1, V2, Mask, {0, 0, 2, 2, 4, 4, 6, 6}))
      return DAG.getNode(X86ISD::MOVDDUP, DL, MVT::v8f64, V1);

    if (!is128BitLaneCrossingShuffleMask(MVT::v8f64, Mask)) {
      // Non-half-crossing single input shuffles can be lowered with an
      // interleaved permutation.
      unsigned VPERMILPMask = (Mask[0] == 1) | ((Mask[1] == 1) << 1) |
                              ((Mask[2] == 3) << 2) | ((Mask[3] == 3) << 3) |
                              ((Mask[4] == 5) << 4) | ((Mask[5] == 5) << 5) |
                              ((Mask[6] == 7) << 6) | ((Mask[7] == 7) << 7);
      return DAG.getNode(X86ISD::VPERMILPI, DL, MVT::v8f64, V1,
                         DAG.getConstant(VPERMILPMask, DL, MVT::i8));
    }

    SmallVector<int, 4> RepeatedMask;
    if (is256BitLaneRepeatedShuffleMask(MVT::v8f64, Mask, RepeatedMask))
      return DAG.getNode(X86ISD::VPERMI, DL, MVT::v8f64, V1,
                         getV4X86ShuffleImm8ForMask(RepeatedMask, DL, DAG));
  }

  if (SDValue Shuf128 = lowerV4X128Shuffle(DL, MVT::v8f64, Mask, Zeroable, V1,
                                           V2, Subtarget, DAG))
    return Shuf128;

  if (SDValue Unpck = lowerShuffleWithUNPCK(DL, MVT::v8f64, Mask, V1, V2, DAG))
    return Unpck;

  // Check if the blend happens to exactly fit that of SHUFPD.
  if (SDValue Op = lowerShuffleWithSHUFPD(DL, MVT::v8f64, Mask, V1, V2, DAG))
    return Op;

  if (SDValue V = lowerShuffleToEXPAND(DL, MVT::v8f64, Zeroable, Mask, V1, V2,
                                       DAG, Subtarget))
    return V;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v8f64, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  return lowerShuffleWithPERMV(DL, MVT::v8f64, Mask, V1, V2, DAG);
}

/// Handle lowering of 16-lane 32-bit floating point shuffles.
static SDValue lowerV16F32Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                  const APInt &Zeroable, SDValue V1, SDValue V2,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v16f32 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v16f32 && "Bad operand type!");
  assert(Mask.size() == 16 && "Unexpected mask size for v16 shuffle!");

  // If the shuffle mask is repeated in each 128-bit lane, we have many more
  // options to efficiently lower the shuffle.
  SmallVector<int, 4> RepeatedMask;
  if (is128BitLaneRepeatedShuffleMask(MVT::v16f32, Mask, RepeatedMask)) {
    assert(RepeatedMask.size() == 4 && "Unexpected repeated mask size!");

    // Use even/odd duplicate instructions for masks that match their pattern.
    if (isShuffleEquivalent(V1, V2, RepeatedMask, {0, 0, 2, 2}))
      return DAG.getNode(X86ISD::MOVSLDUP, DL, MVT::v16f32, V1);
    if (isShuffleEquivalent(V1, V2, RepeatedMask, {1, 1, 3, 3}))
      return DAG.getNode(X86ISD::MOVSHDUP, DL, MVT::v16f32, V1);

    if (V2.isUndef())
      return DAG.getNode(X86ISD::VPERMILPI, DL, MVT::v16f32, V1,
                         getV4X86ShuffleImm8ForMask(RepeatedMask, DL, DAG));

    // Use dedicated unpack instructions for masks that match their pattern.
    if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v16f32, Mask, V1, V2, DAG))
      return V;

    if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v16f32, V1, V2, Mask,
                                            Zeroable, Subtarget, DAG))
      return Blend;

    // Otherwise, fall back to a SHUFPS sequence.
    return lowerShuffleWithSHUFPS(DL, MVT::v16f32, RepeatedMask, V1, V2, DAG);
  }

  // If we have a single input shuffle with different shuffle patterns in the
  // 128-bit lanes and don't lane cross, use variable mask VPERMILPS.
  if (V2.isUndef() &&
      !is128BitLaneCrossingShuffleMask(MVT::v16f32, Mask)) {
    SDValue VPermMask = getConstVector(Mask, MVT::v16i32, DAG, DL, true);
    return DAG.getNode(X86ISD::VPERMILPV, DL, MVT::v16f32, V1, VPermMask);
  }

  // If we have AVX512F support, we can use VEXPAND.
  if (SDValue V = lowerShuffleToEXPAND(DL, MVT::v16f32, Zeroable, Mask,
                                             V1, V2, DAG, Subtarget))
    return V;

  return lowerShuffleWithPERMV(DL, MVT::v16f32, Mask, V1, V2, DAG);
}

/// Handle lowering of 8-lane 64-bit integer shuffles.
static SDValue lowerV8I64Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v8i64 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v8i64 && "Bad operand type!");
  assert(Mask.size() == 8 && "Unexpected mask size for v8 shuffle!");

  if (V2.isUndef()) {
    // When the shuffle is mirrored between the 128-bit lanes of the unit, we
    // can use lower latency instructions that will operate on all four
    // 128-bit lanes.
    SmallVector<int, 2> Repeated128Mask;
    if (is128BitLaneRepeatedShuffleMask(MVT::v8i64, Mask, Repeated128Mask)) {
      SmallVector<int, 4> PSHUFDMask;
      scaleShuffleMask<int>(2, Repeated128Mask, PSHUFDMask);
      return DAG.getBitcast(
          MVT::v8i64,
          DAG.getNode(X86ISD::PSHUFD, DL, MVT::v16i32,
                      DAG.getBitcast(MVT::v16i32, V1),
                      getV4X86ShuffleImm8ForMask(PSHUFDMask, DL, DAG)));
    }

    SmallVector<int, 4> Repeated256Mask;
    if (is256BitLaneRepeatedShuffleMask(MVT::v8i64, Mask, Repeated256Mask))
      return DAG.getNode(X86ISD::VPERMI, DL, MVT::v8i64, V1,
                         getV4X86ShuffleImm8ForMask(Repeated256Mask, DL, DAG));
  }

  if (SDValue Shuf128 = lowerV4X128Shuffle(DL, MVT::v8i64, Mask, Zeroable, V1,
                                           V2, Subtarget, DAG))
    return Shuf128;

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v8i64, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // Try to use VALIGN.
  if (SDValue Rotate = lowerShuffleAsRotate(DL, MVT::v8i64, V1, V2, Mask,
                                            Subtarget, DAG))
    return Rotate;

  // Try to use PALIGNR.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v8i64, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  if (SDValue Unpck = lowerShuffleWithUNPCK(DL, MVT::v8i64, Mask, V1, V2, DAG))
    return Unpck;
  // If we have AVX512F support, we can use VEXPAND.
  if (SDValue V = lowerShuffleToEXPAND(DL, MVT::v8i64, Zeroable, Mask, V1, V2,
                                       DAG, Subtarget))
    return V;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v8i64, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  return lowerShuffleWithPERMV(DL, MVT::v8i64, Mask, V1, V2, DAG);
}

/// Handle lowering of 16-lane 32-bit integer shuffles.
static SDValue lowerV16I32Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                  const APInt &Zeroable, SDValue V1, SDValue V2,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v16i32 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v16i32 && "Bad operand type!");
  assert(Mask.size() == 16 && "Unexpected mask size for v16 shuffle!");

  // Whenever we can lower this as a zext, that instruction is strictly faster
  // than any alternative. It also allows us to fold memory operands into the
  // shuffle in many cases.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(
          DL, MVT::v16i32, V1, V2, Mask, Zeroable, Subtarget, DAG))
    return ZExt;

  // If the shuffle mask is repeated in each 128-bit lane we can use more
  // efficient instructions that mirror the shuffles across the four 128-bit
  // lanes.
  SmallVector<int, 4> RepeatedMask;
  bool Is128BitLaneRepeatedShuffle =
      is128BitLaneRepeatedShuffleMask(MVT::v16i32, Mask, RepeatedMask);
  if (Is128BitLaneRepeatedShuffle) {
    assert(RepeatedMask.size() == 4 && "Unexpected repeated mask size!");
    if (V2.isUndef())
      return DAG.getNode(X86ISD::PSHUFD, DL, MVT::v16i32, V1,
                         getV4X86ShuffleImm8ForMask(RepeatedMask, DL, DAG));

    // Use dedicated unpack instructions for masks that match their pattern.
    if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v16i32, Mask, V1, V2, DAG))
      return V;
  }

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v16i32, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // Try to use VALIGN.
  if (SDValue Rotate = lowerShuffleAsRotate(DL, MVT::v16i32, V1, V2, Mask,
                                            Subtarget, DAG))
    return Rotate;

  // Try to use byte rotation instructions.
  if (Subtarget.hasBWI())
    if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v16i32, V1, V2, Mask,
                                                  Subtarget, DAG))
      return Rotate;

  // Assume that a single SHUFPS is faster than using a permv shuffle.
  // If some CPU is harmed by the domain switch, we can fix it in a later pass.
  if (Is128BitLaneRepeatedShuffle && isSingleSHUFPSMask(RepeatedMask)) {
    SDValue CastV1 = DAG.getBitcast(MVT::v16f32, V1);
    SDValue CastV2 = DAG.getBitcast(MVT::v16f32, V2);
    SDValue ShufPS = lowerShuffleWithSHUFPS(DL, MVT::v16f32, RepeatedMask,
                                            CastV1, CastV2, DAG);
    return DAG.getBitcast(MVT::v16i32, ShufPS);
  }
  // If we have AVX512F support, we can use VEXPAND.
  if (SDValue V = lowerShuffleToEXPAND(DL, MVT::v16i32, Zeroable, Mask, V1, V2,
                                       DAG, Subtarget))
    return V;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v16i32, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;
  return lowerShuffleWithPERMV(DL, MVT::v16i32, Mask, V1, V2, DAG);
}

/// Handle lowering of 32-lane 16-bit integer shuffles.
static SDValue lowerV32I16Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                  const APInt &Zeroable, SDValue V1, SDValue V2,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v32i16 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v32i16 && "Bad operand type!");
  assert(Mask.size() == 32 && "Unexpected mask size for v32 shuffle!");
  assert(Subtarget.hasBWI() && "We can only lower v32i16 with AVX-512-BWI!");

  // Whenever we can lower this as a zext, that instruction is strictly faster
  // than any alternative. It also allows us to fold memory operands into the
  // shuffle in many cases.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(
          DL, MVT::v32i16, V1, V2, Mask, Zeroable, Subtarget, DAG))
    return ZExt;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v32i16, Mask, V1, V2, DAG))
    return V;

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v32i16, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // Try to use byte rotation instructions.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v32i16, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  if (V2.isUndef()) {
    SmallVector<int, 8> RepeatedMask;
    if (is128BitLaneRepeatedShuffleMask(MVT::v32i16, Mask, RepeatedMask)) {
      // As this is a single-input shuffle, the repeated mask should be
      // a strictly valid v8i16 mask that we can pass through to the v8i16
      // lowering to handle even the v32 case.
      return lowerV8I16GeneralSingleInputShuffle(
          DL, MVT::v32i16, V1, RepeatedMask, Subtarget, DAG);
    }
  }

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v32i16, V1, V2, Mask,
                                                Zeroable, Subtarget, DAG))
    return Blend;

  if (SDValue PSHUFB = lowerShuffleWithPSHUFB(DL, MVT::v32i16, Mask, V1, V2,
                                              Zeroable, Subtarget, DAG))
    return PSHUFB;

  return lowerShuffleWithPERMV(DL, MVT::v32i16, Mask, V1, V2, DAG);
}

/// Handle lowering of 64-lane 8-bit integer shuffles.
static SDValue lowerV64I8Shuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                 const APInt &Zeroable, SDValue V1, SDValue V2,
                                 const X86Subtarget &Subtarget,
                                 SelectionDAG &DAG) {
  assert(V1.getSimpleValueType() == MVT::v64i8 && "Bad operand type!");
  assert(V2.getSimpleValueType() == MVT::v64i8 && "Bad operand type!");
  assert(Mask.size() == 64 && "Unexpected mask size for v64 shuffle!");
  assert(Subtarget.hasBWI() && "We can only lower v64i8 with AVX-512-BWI!");

  // Whenever we can lower this as a zext, that instruction is strictly faster
  // than any alternative. It also allows us to fold memory operands into the
  // shuffle in many cases.
  if (SDValue ZExt = lowerShuffleAsZeroOrAnyExtend(
          DL, MVT::v64i8, V1, V2, Mask, Zeroable, Subtarget, DAG))
    return ZExt;

  // Use dedicated unpack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithUNPCK(DL, MVT::v64i8, Mask, V1, V2, DAG))
    return V;

  // Use dedicated pack instructions for masks that match their pattern.
  if (SDValue V = lowerShuffleWithPACK(DL, MVT::v64i8, Mask, V1, V2, DAG,
                                       Subtarget))
    return V;

  // Try to use shift instructions.
  if (SDValue Shift = lowerShuffleAsShift(DL, MVT::v64i8, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Shift;

  // Try to use byte rotation instructions.
  if (SDValue Rotate = lowerShuffleAsByteRotate(DL, MVT::v64i8, V1, V2, Mask,
                                                Subtarget, DAG))
    return Rotate;

  if (SDValue PSHUFB = lowerShuffleWithPSHUFB(DL, MVT::v64i8, Mask, V1, V2,
                                              Zeroable, Subtarget, DAG))
    return PSHUFB;

  // VBMI can use VPERMV/VPERMV3 byte shuffles.
  if (Subtarget.hasVBMI())
    return lowerShuffleWithPERMV(DL, MVT::v64i8, Mask, V1, V2, DAG);

  // Try to create an in-lane repeating shuffle mask and then shuffle the
  // results into the target lanes.
  if (SDValue V = lowerShuffleAsRepeatedMaskAndLanePermute(
          DL, MVT::v64i8, V1, V2, Mask, Subtarget, DAG))
    return V;

  if (SDValue Blend = lowerShuffleAsBlend(DL, MVT::v64i8, V1, V2, Mask,
                                          Zeroable, Subtarget, DAG))
    return Blend;

  // Try to simplify this by merging 128-bit lanes to enable a lane-based
  // shuffle.
  if (!V2.isUndef())
    if (SDValue Result = lowerShuffleAsLanePermuteAndRepeatedMask(
            DL, MVT::v64i8, V1, V2, Mask, Subtarget, DAG))
      return Result;

  // FIXME: Implement direct support for this type!
  return splitAndLowerShuffle(DL, MVT::v64i8, V1, V2, Mask, DAG);
}

/// High-level routine to lower various 512-bit x86 vector shuffles.
///
/// This routine either breaks down the specific type of a 512-bit x86 vector
/// shuffle or splits it into two 256-bit shuffles and fuses the results back
/// together based on the available instructions.
static SDValue lower512BitShuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                  MVT VT, SDValue V1, SDValue V2,
                                  const APInt &Zeroable,
                                  const X86Subtarget &Subtarget,
                                  SelectionDAG &DAG) {
  assert(Subtarget.hasAVX512() &&
         "Cannot lower 512-bit vectors w/ basic ISA!");

  // If we have a single input to the zero element, insert that into V1 if we
  // can do so cheaply.
  int NumElts = Mask.size();
  int NumV2Elements = count_if(Mask, [NumElts](int M) { return M >= NumElts; });

  if (NumV2Elements == 1 && Mask[0] >= NumElts)
    if (SDValue Insertion = lowerShuffleAsElementInsertion(
            DL, VT, V1, V2, Mask, Zeroable, Subtarget, DAG))
      return Insertion;

  // Handle special cases where the lower or upper half is UNDEF.
  if (SDValue V =
          lowerShuffleWithUndefHalf(DL, VT, V1, V2, Mask, Subtarget, DAG))
    return V;

  // Check for being able to broadcast a single element.
  if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, VT, V1, V2, Mask,
                                                  Subtarget, DAG))
    return Broadcast;

  // Dispatch to each element type for lowering. If we don't have support for
  // specific element type shuffles at 512 bits, immediately split them and
  // lower them. Each lowering routine of a given type is allowed to assume that
  // the requisite ISA extensions for that element type are available.
  switch (VT.SimpleTy) {
  case MVT::v8f64:
    return lowerV8F64Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v16f32:
    return lowerV16F32Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v8i64:
    return lowerV8I64Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v16i32:
    return lowerV16I32Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v32i16:
    return lowerV32I16Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);
  case MVT::v64i8:
    return lowerV64I8Shuffle(DL, Mask, Zeroable, V1, V2, Subtarget, DAG);

  default:
    llvm_unreachable("Not a valid 512-bit x86 vector type!");
  }
}

static SDValue lower1BitShuffleAsKSHIFTR(const SDLoc &DL, ArrayRef<int> Mask,
                                         MVT VT, SDValue V1, SDValue V2,
                                         const X86Subtarget &Subtarget,
                                         SelectionDAG &DAG) {
  // Shuffle should be unary.
  if (!V2.isUndef())
    return SDValue();

  int ShiftAmt = -1;
  int NumElts = Mask.size();
  for (int i = 0; i != NumElts; ++i) {
    int M = Mask[i];
    assert((M == SM_SentinelUndef || (0 <= M && M < NumElts)) &&
           "Unexpected mask index.");
    if (M < 0)
      continue;

    // The first non-undef element determines our shift amount.
    if (ShiftAmt < 0) {
      ShiftAmt = M - i;
      // Need to be shifting right.
      if (ShiftAmt <= 0)
        return SDValue();
    }
    // All non-undef elements must shift by the same amount.
    if (ShiftAmt != M - i)
      return SDValue();
  }
  assert(ShiftAmt >= 0 && "All undef?");

  // Great we found a shift right.
  MVT WideVT = VT;
  if ((!Subtarget.hasDQI() && NumElts == 8) || NumElts < 8)
    WideVT = Subtarget.hasDQI() ? MVT::v8i1 : MVT::v16i1;
  SDValue Res = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, WideVT,
                            DAG.getUNDEF(WideVT), V1,
                            DAG.getIntPtrConstant(0, DL));
  Res = DAG.getNode(X86ISD::KSHIFTR, DL, WideVT, Res,
                    DAG.getConstant(ShiftAmt, DL, MVT::i8));
  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, Res,
                     DAG.getIntPtrConstant(0, DL));
}

// Determine if this shuffle can be implemented with a KSHIFT instruction.
// Returns the shift amount if possible or -1 if not. This is a simplified
// version of matchShuffleAsShift.
static int match1BitShuffleAsKSHIFT(unsigned &Opcode, ArrayRef<int> Mask,
                                    int MaskOffset, const APInt &Zeroable) {
  int Size = Mask.size();

  auto CheckZeros = [&](int Shift, bool Left) {
    for (int j = 0; j < Shift; ++j)
      if (!Zeroable[j + (Left ? 0 : (Size - Shift))])
        return false;

    return true;
  };

  auto MatchShift = [&](int Shift, bool Left) {
    unsigned Pos = Left ? Shift : 0;
    unsigned Low = Left ? 0 : Shift;
    unsigned Len = Size - Shift;
    return isSequentialOrUndefInRange(Mask, Pos, Len, Low + MaskOffset);
  };

  for (int Shift = 1; Shift != Size; ++Shift)
    for (bool Left : {true, false})
      if (CheckZeros(Shift, Left) && MatchShift(Shift, Left)) {
        Opcode = Left ? X86ISD::KSHIFTL : X86ISD::KSHIFTR;
        return Shift;
      }

  return -1;
}


// Lower vXi1 vector shuffles.
// There is no a dedicated instruction on AVX-512 that shuffles the masks.
// The only way to shuffle bits is to sign-extend the mask vector to SIMD
// vector, shuffle and then truncate it back.
static SDValue lower1BitShuffle(const SDLoc &DL, ArrayRef<int> Mask,
                                MVT VT, SDValue V1, SDValue V2,
                                const APInt &Zeroable,
                                const X86Subtarget &Subtarget,
                                SelectionDAG &DAG) {
  assert(Subtarget.hasAVX512() &&
         "Cannot lower 512-bit vectors w/o basic ISA!");

  int NumElts = Mask.size();

  // Try to recognize shuffles that are just padding a subvector with zeros.
  int SubvecElts = 0;
  int Src = -1;
  for (int i = 0; i != NumElts; ++i) {
    if (Mask[i] >= 0) {
      // Grab the source from the first valid mask. All subsequent elements need
      // to use this same source.
      if (Src < 0)
        Src = Mask[i] / NumElts;
      if (Src != (Mask[i] / NumElts) || (Mask[i] % NumElts) != i)
        break;
    }

    ++SubvecElts;
  }
  assert(SubvecElts != NumElts && "Identity shuffle?");

  // Clip to a power 2.
  SubvecElts = PowerOf2Floor(SubvecElts);

  // Make sure the number of zeroable bits in the top at least covers the bits
  // not covered by the subvector.
  if ((int)Zeroable.countLeadingOnes() >= (NumElts - SubvecElts)) {
    assert(Src >= 0 && "Expected a source!");
    MVT ExtractVT = MVT::getVectorVT(MVT::i1, SubvecElts);
    SDValue Extract = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, ExtractVT,
                                  Src == 0 ? V1 : V2,
                                  DAG.getIntPtrConstant(0, DL));
    return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT,
                       DAG.getConstant(0, DL, VT),
                       Extract, DAG.getIntPtrConstant(0, DL));
  }

  // Try a simple shift right with undef elements. Later we'll try with zeros.
  if (SDValue Shift = lower1BitShuffleAsKSHIFTR(DL, Mask, VT, V1, V2, Subtarget,
                                                DAG))
    return Shift;

  // Try to match KSHIFTs.
  unsigned Offset = 0;
  for (SDValue V : { V1, V2 }) {
    unsigned Opcode;
    int ShiftAmt = match1BitShuffleAsKSHIFT(Opcode, Mask, Offset, Zeroable);
    if (ShiftAmt >= 0) {
      MVT WideVT = VT;
      if ((!Subtarget.hasDQI() && NumElts == 8) || NumElts < 8)
        WideVT = Subtarget.hasDQI() ? MVT::v8i1 : MVT::v16i1;
      SDValue Res = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, WideVT,
                                DAG.getUNDEF(WideVT), V,
                                DAG.getIntPtrConstant(0, DL));
      // Widened right shifts need two shifts to ensure we shift in zeroes.
      if (Opcode == X86ISD::KSHIFTR && WideVT != VT) {
        int WideElts = WideVT.getVectorNumElements();
        // Shift left to put the original vector in the MSBs of the new size.
        Res = DAG.getNode(X86ISD::KSHIFTL, DL, WideVT, Res,
                          DAG.getConstant(WideElts - NumElts, DL, MVT::i8));
        // Increase the shift amount to account for the left shift.
        ShiftAmt += WideElts - NumElts;
      }

      Res = DAG.getNode(Opcode, DL, WideVT, Res,
                        DAG.getConstant(ShiftAmt, DL, MVT::i8));
      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, Res,
                         DAG.getIntPtrConstant(0, DL));
    }
    Offset += NumElts; // Increment for next iteration.
  }



  MVT ExtVT;
  switch (VT.SimpleTy) {
  default:
    llvm_unreachable("Expected a vector of i1 elements");
  case MVT::v2i1:
    ExtVT = MVT::v2i64;
    break;
  case MVT::v4i1:
    ExtVT = MVT::v4i32;
    break;
  case MVT::v8i1:
    // Take 512-bit type, more shuffles on KNL. If we have VLX use a 256-bit
    // shuffle.
    ExtVT = Subtarget.hasVLX() ? MVT::v8i32 : MVT::v8i64;
    break;
  case MVT::v16i1:
    // Take 512-bit type, unless we are avoiding 512-bit types and have the
    // 256-bit operation available.
    ExtVT = Subtarget.canExtendTo512DQ() ? MVT::v16i32 : MVT::v16i16;
    break;
  case MVT::v32i1:
    // Take 512-bit type, unless we are avoiding 512-bit types and have the
    // 256-bit operation available.
    assert(Subtarget.hasBWI() && "Expected AVX512BW support");
    ExtVT = Subtarget.canExtendTo512BW() ? MVT::v32i16 : MVT::v32i8;
    break;
  case MVT::v64i1:
    ExtVT = MVT::v64i8;
    break;
  }

  V1 = DAG.getNode(ISD::SIGN_EXTEND, DL, ExtVT, V1);
  V2 = DAG.getNode(ISD::SIGN_EXTEND, DL, ExtVT, V2);

  SDValue Shuffle = DAG.getVectorShuffle(ExtVT, DL, V1, V2, Mask);
  // i1 was sign extended we can use X86ISD::CVT2MASK.
  int NumElems = VT.getVectorNumElements();
  if ((Subtarget.hasBWI() && (NumElems >= 32)) ||
      (Subtarget.hasDQI() && (NumElems < 32)))
    return DAG.getSetCC(DL, VT, DAG.getConstant(0, DL, ExtVT),
                       Shuffle, ISD::SETGT);

  return DAG.getNode(ISD::TRUNCATE, DL, VT, Shuffle);
}

/// Helper function that returns true if the shuffle mask should be
/// commuted to improve canonicalization.
static bool canonicalizeShuffleMaskWithCommute(ArrayRef<int> Mask) {
  int NumElements = Mask.size();

  int NumV1Elements = 0, NumV2Elements = 0;
  for (int M : Mask)
    if (M < 0)
      continue;
    else if (M < NumElements)
      ++NumV1Elements;
    else
      ++NumV2Elements;

  // Commute the shuffle as needed such that more elements come from V1 than
  // V2. This allows us to match the shuffle pattern strictly on how many
  // elements come from V1 without handling the symmetric cases.
  if (NumV2Elements > NumV1Elements)
    return true;

  assert(NumV1Elements > 0 && "No V1 indices");

  if (NumV2Elements == 0)
    return false;

  // When the number of V1 and V2 elements are the same, try to minimize the
  // number of uses of V2 in the low half of the vector. When that is tied,
  // ensure that the sum of indices for V1 is equal to or lower than the sum
  // indices for V2. When those are equal, try to ensure that the number of odd
  // indices for V1 is lower than the number of odd indices for V2.
  if (NumV1Elements == NumV2Elements) {
    int LowV1Elements = 0, LowV2Elements = 0;
    for (int M : Mask.slice(0, NumElements / 2))
      if (M >= NumElements)
        ++LowV2Elements;
      else if (M >= 0)
        ++LowV1Elements;
    if (LowV2Elements > LowV1Elements)
      return true;
    if (LowV2Elements == LowV1Elements) {
      int SumV1Indices = 0, SumV2Indices = 0;
      for (int i = 0, Size = Mask.size(); i < Size; ++i)
        if (Mask[i] >= NumElements)
          SumV2Indices += i;
        else if (Mask[i] >= 0)
          SumV1Indices += i;
      if (SumV2Indices < SumV1Indices)
        return true;
      if (SumV2Indices == SumV1Indices) {
        int NumV1OddIndices = 0, NumV2OddIndices = 0;
        for (int i = 0, Size = Mask.size(); i < Size; ++i)
          if (Mask[i] >= NumElements)
            NumV2OddIndices += i % 2;
          else if (Mask[i] >= 0)
            NumV1OddIndices += i % 2;
        if (NumV2OddIndices < NumV1OddIndices)
          return true;
      }
    }
  }

  return false;
}

/// Top-level lowering for x86 vector shuffles.
///
/// This handles decomposition, canonicalization, and lowering of all x86
/// vector shuffles. Most of the specific lowering strategies are encapsulated
/// above in helper routines. The canonicalization attempts to widen shuffles
/// to involve fewer lanes of wider elements, consolidate symmetric patterns
/// s.t. only one of the two inputs needs to be tested, etc.
SDValue X86TargetLowering::lowerVectorShuffle(SDValue Op,
                                              SelectionDAG &DAG) const {
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(Op);
  ArrayRef<int> OrigMask = SVOp->getMask();
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  MVT VT = Op.getSimpleValueType();
  int NumElements = VT.getVectorNumElements();
  SDLoc DL(Op);
  bool Is1BitVector = (VT.getVectorElementType() == MVT::i1);

  assert((VT.getSizeInBits() != 64 || Is1BitVector) &&
         "Can't lower MMX shuffles");

  bool V1IsUndef = V1.isUndef();
  bool V2IsUndef = V2.isUndef();
  if (V1IsUndef && V2IsUndef)
    return DAG.getUNDEF(VT);

  // When we create a shuffle node we put the UNDEF node to second operand,
  // but in some cases the first operand may be transformed to UNDEF.
  // In this case we should just commute the node.
  if (V1IsUndef)
    return DAG.getCommutedVectorShuffle(*SVOp);

  // Check for non-undef masks pointing at an undef vector and make the masks
  // undef as well. This makes it easier to match the shuffle based solely on
  // the mask.
  if (V2IsUndef &&
      any_of(OrigMask, [NumElements](int M) { return M >= NumElements; })) {
    SmallVector<int, 8> NewMask(OrigMask.begin(), OrigMask.end());
    for (int &M : NewMask)
      if (M >= NumElements)
        M = -1;
    return DAG.getVectorShuffle(VT, DL, V1, V2, NewMask);
  }

  // Check for illegal shuffle mask element index values.
  int MaskUpperLimit = OrigMask.size() * (V2IsUndef ? 1 : 2);
  (void)MaskUpperLimit;
  assert(llvm::all_of(OrigMask,
                      [&](int M) { return -1 <= M && M < MaskUpperLimit; }) &&
         "Out of bounds shuffle index");

  // We actually see shuffles that are entirely re-arrangements of a set of
  // zero inputs. This mostly happens while decomposing complex shuffles into
  // simple ones. Directly lower these as a buildvector of zeros.
  APInt Zeroable = computeZeroableShuffleElements(OrigMask, V1, V2);
  if (Zeroable.isAllOnesValue())
    return getZeroVector(VT, Subtarget, DAG, DL);

  bool V2IsZero = !V2IsUndef && ISD::isBuildVectorAllZeros(V2.getNode());

  // Create an alternative mask with info about zeroable elements.
  // Here we do not set undef elements as zeroable.
  SmallVector<int, 64> ZeroableMask(OrigMask.begin(), OrigMask.end());
  if (V2IsZero) {
    assert(!Zeroable.isNullValue() && "V2's non-undef elements are used?!");
    for (int i = 0; i != NumElements; ++i)
      if (OrigMask[i] != SM_SentinelUndef && Zeroable[i])
        ZeroableMask[i] = SM_SentinelZero;
  }

  // Try to collapse shuffles into using a vector type with fewer elements but
  // wider element types. We cap this to not form integers or floating point
  // elements wider than 64 bits, but it might be interesting to form i128
  // integers to handle flipping the low and high halves of AVX 256-bit vectors.
  SmallVector<int, 16> WidenedMask;
  if (VT.getScalarSizeInBits() < 64 && !Is1BitVector &&
      X86::canWidenShuffleElements(ZeroableMask, WidenedMask)) {
    // Shuffle mask widening should not interfere with a broadcast opportunity
    // by obfuscating the operands with bitcasts.
    // TODO: Avoid lowering directly from this top-level function: make this
    // a query (canLowerAsBroadcast) and defer lowering to the type-based calls.
    if (SDValue Broadcast = lowerShuffleAsBroadcast(DL, VT, V1, V2, OrigMask,
                                                    Subtarget, DAG))
      return Broadcast;

    MVT NewEltVT = VT.isFloatingPoint()
                       ? MVT::getFloatingPointVT(VT.getScalarSizeInBits() * 2)
                       : MVT::getIntegerVT(VT.getScalarSizeInBits() * 2);
    int NewNumElts = NumElements / 2;
    MVT NewVT = MVT::getVectorVT(NewEltVT, NewNumElts);
    // Make sure that the new vector type is legal. For example, v2f64 isn't
    // legal on SSE1.
    if (DAG.getTargetLoweringInfo().isTypeLegal(NewVT)) {
      if (V2IsZero) {
        // Modify the new Mask to take all zeros from the all-zero vector.
        // Choose indices that are blend-friendly.
        bool UsedZeroVector = false;
        assert(find(WidenedMask, SM_SentinelZero) != WidenedMask.end() &&
               "V2's non-undef elements are used?!");
        for (int i = 0; i != NewNumElts; ++i)
          if (WidenedMask[i] == SM_SentinelZero) {
            WidenedMask[i] = i + NewNumElts;
            UsedZeroVector = true;
          }
        // Ensure all elements of V2 are zero - isBuildVectorAllZeros permits
        // some elements to be undef.
        if (UsedZeroVector)
          V2 = getZeroVector(NewVT, Subtarget, DAG, DL);
      }
      V1 = DAG.getBitcast(NewVT, V1);
      V2 = DAG.getBitcast(NewVT, V2);
      return DAG.getBitcast(
          VT, DAG.getVectorShuffle(NewVT, DL, V1, V2, WidenedMask));
    }
  }

  // Commute the shuffle if it will improve canonicalization.
  SmallVector<int, 64> Mask(OrigMask.begin(), OrigMask.end());
  if (canonicalizeShuffleMaskWithCommute(Mask)) {
    ShuffleVectorSDNode::commuteMask(Mask);
    std::swap(V1, V2);
  }

  if (SDValue V = lowerShuffleWithVPMOV(DL, Mask, VT, V1, V2, DAG, Subtarget))
    return V;

  // For each vector width, delegate to a specialized lowering routine.
  if (VT.is128BitVector())
    return lower128BitShuffle(DL, Mask, VT, V1, V2, Zeroable, Subtarget, DAG);

  if (VT.is256BitVector())
    return lower256BitShuffle(DL, Mask, VT, V1, V2, Zeroable, Subtarget, DAG);

  if (VT.is512BitVector())
    return lower512BitShuffle(DL, Mask, VT, V1, V2, Zeroable, Subtarget, DAG);

  if (Is1BitVector)
    return lower1BitShuffle(DL, Mask, VT, V1, V2, Zeroable, Subtarget, DAG);

  llvm_unreachable("Unimplemented!");
}

/// Try to lower a VSELECT instruction to a vector shuffle.
static SDValue lowerVSELECTtoVectorShuffle(SDValue Op,
                                           const X86Subtarget &Subtarget,
                                           SelectionDAG &DAG) {
  SDValue Cond = Op.getOperand(0);
  SDValue LHS = Op.getOperand(1);
  SDValue RHS = Op.getOperand(2);
  MVT VT = Op.getSimpleValueType();

  // Only non-legal VSELECTs reach this lowering, convert those into generic
  // shuffles and re-use the shuffle lowering path for blends.
  SmallVector<int, 32> Mask;
  if (X86::createShuffleMaskFromVSELECT(Mask, Cond))
    return DAG.getVectorShuffle(VT, SDLoc(Op), LHS, RHS, Mask);

  return SDValue();
}

SDValue X86TargetLowering::LowerVSELECT(SDValue Op, SelectionDAG &DAG) const {
  SDValue Cond = Op.getOperand(0);
  SDValue LHS = Op.getOperand(1);
  SDValue RHS = Op.getOperand(2);

  // A vselect where all conditions and data are constants can be optimized into
  // a single vector load by SelectionDAGLegalize::ExpandBUILD_VECTOR().
  if (ISD::isBuildVectorOfConstantSDNodes(Cond.getNode()) &&
      ISD::isBuildVectorOfConstantSDNodes(LHS.getNode()) &&
      ISD::isBuildVectorOfConstantSDNodes(RHS.getNode()))
    return SDValue();

  // Try to lower this to a blend-style vector shuffle. This can handle all
  // constant condition cases.
  if (SDValue BlendOp = lowerVSELECTtoVectorShuffle(Op, Subtarget, DAG))
    return BlendOp;

  // If this VSELECT has a vector if i1 as a mask, it will be directly matched
  // with patterns on the mask registers on AVX-512.
  MVT CondVT = Cond.getSimpleValueType();
  unsigned CondEltSize = Cond.getScalarValueSizeInBits();
  if (CondEltSize == 1)
    return Op;

  // Variable blends are only legal from SSE4.1 onward.
  if (!Subtarget.hasSSE41())
    return SDValue();

  SDLoc dl(Op);
  MVT VT = Op.getSimpleValueType();
  unsigned EltSize = VT.getScalarSizeInBits();
  unsigned NumElts = VT.getVectorNumElements();

  // If the VSELECT is on a 512-bit type, we have to convert a non-i1 condition
  // into an i1 condition so that we can use the mask-based 512-bit blend
  // instructions.
  if (VT.getSizeInBits() == 512) {
    // Build a mask by testing the condition against zero.
    MVT MaskVT = MVT::getVectorVT(MVT::i1, NumElts);
    SDValue Mask = DAG.getSetCC(dl, MaskVT, Cond,
                                DAG.getConstant(0, dl, CondVT),
                                ISD::SETNE);
    // Now return a new VSELECT using the mask.
    return DAG.getSelect(dl, VT, Mask, LHS, RHS);
  }

  // SEXT/TRUNC cases where the mask doesn't match the destination size.
  if (CondEltSize != EltSize) {
    // If we don't have a sign splat, rely on the expansion.
    if (CondEltSize != DAG.ComputeNumSignBits(Cond))
      return SDValue();

    MVT NewCondSVT = MVT::getIntegerVT(EltSize);
    MVT NewCondVT = MVT::getVectorVT(NewCondSVT, NumElts);
    Cond = DAG.getSExtOrTrunc(Cond, dl, NewCondVT);
    return DAG.getNode(ISD::VSELECT, dl, VT, Cond, LHS, RHS);
  }

  // Only some types will be legal on some subtargets. If we can emit a legal
  // VSELECT-matching blend, return Op, and but if we need to expand, return
  // a null value.
  switch (VT.SimpleTy) {
  default:
    // Most of the vector types have blends past SSE4.1.
    return Op;

  case MVT::v32i8:
    // The byte blends for AVX vectors were introduced only in AVX2.
    if (Subtarget.hasAVX2())
      return Op;

    return SDValue();

  case MVT::v8i16:
  case MVT::v16i16: {
    // Bitcast everything to the vXi8 type and use a vXi8 vselect.
    MVT CastVT = MVT::getVectorVT(MVT::i8, NumElts * 2);
    Cond = DAG.getBitcast(CastVT, Cond);
    LHS = DAG.getBitcast(CastVT, LHS);
    RHS = DAG.getBitcast(CastVT, RHS);
    SDValue Select = DAG.getNode(ISD::VSELECT, dl, CastVT, Cond, LHS, RHS);
    return DAG.getBitcast(VT, Select);
  }
  }
}

static SDValue LowerEXTRACT_VECTOR_ELT_SSE4(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getSimpleValueType();
  SDLoc dl(Op);

  if (!Op.getOperand(0).getSimpleValueType().is128BitVector())
    return SDValue();

  if (VT.getSizeInBits() == 8) {
    SDValue Extract = DAG.getNode(X86ISD::PEXTRB, dl, MVT::i32,
                                  Op.getOperand(0), Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Extract);
  }

  if (VT == MVT::f32) {
    // EXTRACTPS outputs to a GPR32 register which will require a movd to copy
    // the result back to FR32 register. It's only worth matching if the
    // result has a single use which is a store or a bitcast to i32.  And in
    // the case of a store, it's not worth it if the index is a constant 0,
    // because a MOVSSmr can be used instead, which is smaller and faster.
    if (!Op.hasOneUse())
      return SDValue();
    SDNode *User = *Op.getNode()->use_begin();
    if ((User->getOpcode() != ISD::STORE ||
         isNullConstant(Op.getOperand(1))) &&
        (User->getOpcode() != ISD::BITCAST ||
         User->getValueType(0) != MVT::i32))
      return SDValue();
    SDValue Extract = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                                  DAG.getBitcast(MVT::v4i32, Op.getOperand(0)),
                                  Op.getOperand(1));
    return DAG.getBitcast(MVT::f32, Extract);
  }

  if (VT == MVT::i32 || VT == MVT::i64) {
    // ExtractPS/pextrq works with constant index.
    if (isa<ConstantSDNode>(Op.getOperand(1)))
      return Op;
  }

  return SDValue();
}

/// Extract one bit from mask vector, like v16i1 or v8i1.
/// AVX-512 feature.
static SDValue ExtractBitFromMaskVector(SDValue Op, SelectionDAG &DAG,
                                        const X86Subtarget &Subtarget) {
  SDValue Vec = Op.getOperand(0);
  SDLoc dl(Vec);
  MVT VecVT = Vec.getSimpleValueType();
  SDValue Idx = Op.getOperand(1);
  MVT EltVT = Op.getSimpleValueType();

  assert((VecVT.getVectorNumElements() <= 16 || Subtarget.hasBWI()) &&
         "Unexpected vector type in ExtractBitFromMaskVector");

  // variable index can't be handled in mask registers,
  // extend vector to VR512/128
  if (!isa<ConstantSDNode>(Idx)) {
    unsigned NumElts = VecVT.getVectorNumElements();
    // Extending v8i1/v16i1 to 512-bit get better performance on KNL
    // than extending to 128/256bit.
    MVT ExtEltVT = (NumElts <= 8) ? MVT::getIntegerVT(128 / NumElts) : MVT::i8;
    MVT ExtVecVT = MVT::getVectorVT(ExtEltVT, NumElts);
    SDValue Ext = DAG.getNode(ISD::SIGN_EXTEND, dl, ExtVecVT, Vec);
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, ExtEltVT, Ext, Idx);
    return DAG.getNode(ISD::TRUNCATE, dl, EltVT, Elt);
  }

  unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();
  if (IdxVal == 0) // the operation is legal
    return Op;

  // Extend to natively supported kshift.
  unsigned NumElems = VecVT.getVectorNumElements();
  MVT WideVecVT = VecVT;
  if ((!Subtarget.hasDQI() && NumElems == 8) || NumElems < 8) {
    WideVecVT = Subtarget.hasDQI() ? MVT::v8i1 : MVT::v16i1;
    Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideVecVT,
                      DAG.getUNDEF(WideVecVT), Vec,
                      DAG.getIntPtrConstant(0, dl));
  }

  // Use kshiftr instruction to move to the lower element.
  Vec = DAG.getNode(X86ISD::KSHIFTR, dl, WideVecVT, Vec,
                    DAG.getConstant(IdxVal, dl, MVT::i8));

  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, Op.getValueType(), Vec,
                     DAG.getIntPtrConstant(0, dl));
}

SDValue
X86TargetLowering::LowerEXTRACT_VECTOR_ELT(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc dl(Op);
  SDValue Vec = Op.getOperand(0);
  MVT VecVT = Vec.getSimpleValueType();
  SDValue Idx = Op.getOperand(1);

  if (VecVT.getVectorElementType() == MVT::i1)
    return ExtractBitFromMaskVector(Op, DAG, Subtarget);

  if (!isa<ConstantSDNode>(Idx)) {
    // Its more profitable to go through memory (1 cycles throughput)
    // than using VMOVD + VPERMV/PSHUFB sequence ( 2/3 cycles throughput)
    // IACA tool was used to get performance estimation
    // (https://software.intel.com/en-us/articles/intel-architecture-code-analyzer)
    //
    // example : extractelement <16 x i8> %a, i32 %i
    //
    // Block Throughput: 3.00 Cycles
    // Throughput Bottleneck: Port5
    //
    // | Num Of |   Ports pressure in cycles  |    |
    // |  Uops  |  0  - DV  |  5  |  6  |  7  |    |
    // ---------------------------------------------
    // |   1    |           | 1.0 |     |     | CP | vmovd xmm1, edi
    // |   1    |           | 1.0 |     |     | CP | vpshufb xmm0, xmm0, xmm1
    // |   2    | 1.0       | 1.0 |     |     | CP | vpextrb eax, xmm0, 0x0
    // Total Num Of Uops: 4
    //
    //
    // Block Throughput: 1.00 Cycles
    // Throughput Bottleneck: PORT2_AGU, PORT3_AGU, Port4
    //
    // |    |  Ports pressure in cycles   |  |
    // |Uops| 1 | 2 - D  |3 -  D  | 4 | 5 |  |
    // ---------------------------------------------------------
    // |2^  |   | 0.5    | 0.5    |1.0|   |CP| vmovaps xmmword ptr [rsp-0x18], xmm0
    // |1   |0.5|        |        |   |0.5|  | lea rax, ptr [rsp-0x18]
    // |1   |   |0.5, 0.5|0.5, 0.5|   |   |CP| mov al, byte ptr [rdi+rax*1]
    // Total Num Of Uops: 4

    return SDValue();
  }

  unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();

  // If this is a 256-bit vector result, first extract the 128-bit vector and
  // then extract the element from the 128-bit vector.
  if (VecVT.is256BitVector() || VecVT.is512BitVector()) {
    // Get the 128-bit vector.
    Vec = extract128BitVector(Vec, IdxVal, DAG, dl);
    MVT EltVT = VecVT.getVectorElementType();

    unsigned ElemsPerChunk = 128 / EltVT.getSizeInBits();
    assert(isPowerOf2_32(ElemsPerChunk) && "Elements per chunk not power of 2");

    // Find IdxVal modulo ElemsPerChunk. Since ElemsPerChunk is a power of 2
    // this can be done with a mask.
    IdxVal &= ElemsPerChunk - 1;
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, Op.getValueType(), Vec,
                       DAG.getIntPtrConstant(IdxVal, dl));
  }

  assert(VecVT.is128BitVector() && "Unexpected vector length");

  MVT VT = Op.getSimpleValueType();

  if (VT.getSizeInBits() == 16) {
    // If IdxVal is 0, it's cheaper to do a move instead of a pextrw, unless
    // we're going to zero extend the register or fold the store (SSE41 only).
    if (IdxVal == 0 && !MayFoldIntoZeroExtend(Op) &&
        !(Subtarget.hasSSE41() && MayFoldIntoStore(Op)))
      return DAG.getNode(ISD::TRUNCATE, dl, MVT::i16,
                         DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                                     DAG.getBitcast(MVT::v4i32, Vec), Idx));

    // Transform it so it match pextrw which produces a 32-bit result.
    SDValue Extract = DAG.getNode(X86ISD::PEXTRW, dl, MVT::i32,
                                  Op.getOperand(0), Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Extract);
  }

  if (Subtarget.hasSSE41())
    if (SDValue Res = LowerEXTRACT_VECTOR_ELT_SSE4(Op, DAG))
      return Res;

  // TODO: We only extract a single element from v16i8, we can probably afford
  // to be more aggressive here before using the default approach of spilling to
  // stack.
  if (VT.getSizeInBits() == 8 && Op->isOnlyUserOf(Vec.getNode())) {
    // Extract either the lowest i32 or any i16, and extract the sub-byte.
    int DWordIdx = IdxVal / 4;
    if (DWordIdx == 0) {
      SDValue Res = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                                DAG.getBitcast(MVT::v4i32, Vec),
                                DAG.getIntPtrConstant(DWordIdx, dl));
      int ShiftVal = (IdxVal % 4) * 8;
      if (ShiftVal != 0)
        Res = DAG.getNode(ISD::SRL, dl, MVT::i32, Res,
                          DAG.getConstant(ShiftVal, dl, MVT::i8));
      return DAG.getNode(ISD::TRUNCATE, dl, VT, Res);
    }

    int WordIdx = IdxVal / 2;
    SDValue Res = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i16,
                              DAG.getBitcast(MVT::v8i16, Vec),
                              DAG.getIntPtrConstant(WordIdx, dl));
    int ShiftVal = (IdxVal % 2) * 8;
    if (ShiftVal != 0)
      Res = DAG.getNode(ISD::SRL, dl, MVT::i16, Res,
                        DAG.getConstant(ShiftVal, dl, MVT::i8));
    return DAG.getNode(ISD::TRUNCATE, dl, VT, Res);
  }

  if (VT.getSizeInBits() == 32) {
    if (IdxVal == 0)
      return Op;

    // SHUFPS the element to the lowest double word, then movss.
    int Mask[4] = { static_cast<int>(IdxVal), -1, -1, -1 };
    Vec = DAG.getVectorShuffle(VecVT, dl, Vec, DAG.getUNDEF(VecVT), Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, VT, Vec,
                       DAG.getIntPtrConstant(0, dl));
  }

  if (VT.getSizeInBits() == 64) {
    // FIXME: .td only matches this for <2 x f64>, not <2 x i64> on 32b
    // FIXME: seems like this should be unnecessary if mov{h,l}pd were taught
    //        to match extract_elt for f64.
    if (IdxVal == 0)
      return Op;

    // UNPCKHPD the element to the lowest double word, then movsd.
    // Note if the lower 64 bits of the result of the UNPCKHPD is then stored
    // to a f64mem, the whole operation is folded into a single MOVHPDmr.
    int Mask[2] = { 1, -1 };
    Vec = DAG.getVectorShuffle(VecVT, dl, Vec, DAG.getUNDEF(VecVT), Mask);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, VT, Vec,
                       DAG.getIntPtrConstant(0, dl));
  }

  return SDValue();
}

/// Insert one bit to mask vector, like v16i1 or v8i1.
/// AVX-512 feature.
static SDValue InsertBitToMaskVector(SDValue Op, SelectionDAG &DAG,
                                     const X86Subtarget &Subtarget) {
  SDLoc dl(Op);
  SDValue Vec = Op.getOperand(0);
  SDValue Elt = Op.getOperand(1);
  SDValue Idx = Op.getOperand(2);
  MVT VecVT = Vec.getSimpleValueType();

  if (!isa<ConstantSDNode>(Idx)) {
    // Non constant index. Extend source and destination,
    // insert element and then truncate the result.
    unsigned NumElts = VecVT.getVectorNumElements();
    MVT ExtEltVT = (NumElts <= 8) ? MVT::getIntegerVT(128 / NumElts) : MVT::i8;
    MVT ExtVecVT = MVT::getVectorVT(ExtEltVT, NumElts);
    SDValue ExtOp = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, ExtVecVT,
      DAG.getNode(ISD::SIGN_EXTEND, dl, ExtVecVT, Vec),
      DAG.getNode(ISD::SIGN_EXTEND, dl, ExtEltVT, Elt), Idx);
    return DAG.getNode(ISD::TRUNCATE, dl, VecVT, ExtOp);
  }

  // Copy into a k-register, extract to v1i1 and insert_subvector.
  SDValue EltInVec = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v1i1, Elt);

  return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, VecVT, Vec, EltInVec,
                     Op.getOperand(2));
}

SDValue X86TargetLowering::LowerINSERT_VECTOR_ELT(SDValue Op,
                                                  SelectionDAG &DAG) const {
  MVT VT = Op.getSimpleValueType();
  MVT EltVT = VT.getVectorElementType();
  unsigned NumElts = VT.getVectorNumElements();

  if (EltVT == MVT::i1)
    return InsertBitToMaskVector(Op, DAG, Subtarget);

  SDLoc dl(Op);
  SDValue N0 = Op.getOperand(0);
  SDValue N1 = Op.getOperand(1);
  SDValue N2 = Op.getOperand(2);

  auto *N2C = dyn_cast<ConstantSDNode>(N2);
  if (!N2C || N2C->getAPIntValue().uge(NumElts))
    return SDValue();
  uint64_t IdxVal = N2C->getZExtValue();

  bool IsZeroElt = X86::isZeroNode(N1);
  bool IsAllOnesElt = VT.isInteger() && llvm::isAllOnesConstant(N1);

  // If we are inserting a element, see if we can do this more efficiently with
  // a blend shuffle with a rematerializable vector than a costly integer
  // insertion.
  if ((IsZeroElt || IsAllOnesElt) && Subtarget.hasSSE41() &&
      16 <= EltVT.getSizeInBits()) {
    SmallVector<int, 8> BlendMask;
    for (unsigned i = 0; i != NumElts; ++i)
      BlendMask.push_back(i == IdxVal ? i + NumElts : i);
    SDValue CstVector = IsZeroElt ? getZeroVector(VT, Subtarget, DAG, dl)
                                  : getOnesVector(VT, DAG, dl);
    return DAG.getVectorShuffle(VT, dl, N0, CstVector, BlendMask);
  }

  // If the vector is wider than 128 bits, extract the 128-bit subvector, insert
  // into that, and then insert the subvector back into the result.
  if (VT.is256BitVector() || VT.is512BitVector()) {
    // With a 256-bit vector, we can insert into the zero element efficiently
    // using a blend if we have AVX or AVX2 and the right data type.
    if (VT.is256BitVector() && IdxVal == 0) {
      // TODO: It is worthwhile to cast integer to floating point and back
      // and incur a domain crossing penalty if that's what we'll end up
      // doing anyway after extracting to a 128-bit vector.
      if ((Subtarget.hasAVX() && (EltVT == MVT::f64 || EltVT == MVT::f32)) ||
          (Subtarget.hasAVX2() && EltVT == MVT::i32)) {
        SDValue N1Vec = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, N1);
        return DAG.getNode(X86ISD::BLENDI, dl, VT, N0, N1Vec,
                           DAG.getConstant(1, dl, MVT::i8));
      }
    }

    // Get the desired 128-bit vector chunk.
    SDValue V = extract128BitVector(N0, IdxVal, DAG, dl);

    // Insert the element into the desired chunk.
    unsigned NumEltsIn128 = 128 / EltVT.getSizeInBits();
    assert(isPowerOf2_32(NumEltsIn128));
    // Since NumEltsIn128 is a power of 2 we can use mask instead of modulo.
    unsigned IdxIn128 = IdxVal & (NumEltsIn128 - 1);

    V = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, V.getValueType(), V, N1,
                    DAG.getIntPtrConstant(IdxIn128, dl));

    // Insert the changed part back into the bigger vector
    return insert128BitVector(N0, V, IdxVal, DAG, dl);
  }
  assert(VT.is128BitVector() && "Only 128-bit vector types should be left!");

  // This will be just movd/movq/movss/movsd.
  if (IdxVal == 0 && ISD::isBuildVectorAllZeros(N0.getNode()) &&
      (EltVT == MVT::i32 || EltVT == MVT::f32 || EltVT == MVT::f64 ||
       EltVT == MVT::i64)) {
    N1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT, N1);
    return getShuffleVectorZeroOrUndef(N1, 0, true, Subtarget, DAG);
  }

  // Transform it so it match pinsr{b,w} which expects a GR32 as its second
  // argument. SSE41 required for pinsrb.
  if (VT == MVT::v8i16 || (VT == MVT::v16i8 && Subtarget.hasSSE41())) {
    unsigned Opc;
    if (VT == MVT::v8i16) {
      assert(Subtarget.hasSSE2() && "SSE2 required for PINSRW");
      Opc = X86ISD::PINSRW;
    } else {
      assert(VT == MVT::v16i8 && "PINSRB requires v16i8 vector");
      assert(Subtarget.hasSSE41() && "SSE41 required for PINSRB");
      Opc = X86ISD::PINSRB;
    }

    if (N1.getValueType() != MVT::i32)
      N1 = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, N1);
    if (N2.getValueType() != MVT::i32)
      N2 = DAG.getIntPtrConstant(IdxVal, dl);
    return DAG.getNode(Opc, dl, VT, N0, N1, N2);
  }

  if (Subtarget.hasSSE41()) {
    if (EltVT == MVT::f32) {
      // Bits [7:6] of the constant are the source select. This will always be
      //   zero here. The DAG Combiner may combine an extract_elt index into
      //   these bits. For example (insert (extract, 3), 2) could be matched by
      //   putting the '3' into bits [7:6] of X86ISD::INSERTPS.
      // Bits [5:4] of the constant are the destination select. This is the
      //   value of the incoming immediate.
      // Bits [3:0] of the constant are the zero mask. The DAG Combiner may
      //   combine either bitwise AND or insert of float 0.0 to set these bits.

      bool MinSize = DAG.getMachineFunction().getFunction().hasMinSize();
      if (IdxVal == 0 && (!MinSize || !MayFoldLoad(N1))) {
        // If this is an insertion of 32-bits into the low 32-bits of
        // a vector, we prefer to generate a blend with immediate rather
        // than an insertps. Blends are simpler operations in hardware and so
        // will always have equal or better performance than insertps.
        // But if optimizing for size and there's a load folding opportunity,
        // generate insertps because blendps does not have a 32-bit memory
        // operand form.
        N1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4f32, N1);
        return DAG.getNode(X86ISD::BLENDI, dl, VT, N0, N1,
                           DAG.getConstant(1, dl, MVT::i8));
      }
      // Create this as a scalar to vector..
      N1 = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4f32, N1);
      return DAG.getNode(X86ISD::INSERTPS, dl, VT, N0, N1,
                         DAG.getConstant(IdxVal << 4, dl, MVT::i8));
    }

    // PINSR* works with constant index.
    if (EltVT == MVT::i32 || EltVT == MVT::i64)
      return Op;
  }

  return SDValue();
}

SDValue X86TargetLowering::LowerSCALAR_TO_VECTOR(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc dl(Op);
  MVT OpVT = Op.getSimpleValueType();

  // It's always cheaper to replace a xor+movd with xorps and simplifies further
  // combines.
  if (X86::isZeroNode(Op.getOperand(0)))
    return getZeroVector(OpVT, Subtarget, DAG, dl);

  // If this is a 256-bit vector result, first insert into a 128-bit
  // vector and then insert into the 256-bit vector.
  if (!OpVT.is128BitVector()) {
    // Insert into a 128-bit vector.
    unsigned SizeFactor = OpVT.getSizeInBits() / 128;
    MVT VT128 = MVT::getVectorVT(OpVT.getVectorElementType(),
                                 OpVT.getVectorNumElements() / SizeFactor);

    Op = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VT128, Op.getOperand(0));

    // Insert the 128-bit vector.
    return insert128BitVector(DAG.getUNDEF(OpVT), Op, 0, DAG, dl);
  }
  assert(OpVT.is128BitVector() && OpVT.isInteger() && OpVT != MVT::v2i64 &&
         "Expected an SSE type!");

  // Pass through a v4i32 SCALAR_TO_VECTOR as that's what we use in tblgen.
  if (OpVT == MVT::v4i32)
    return Op;

  SDValue AnyExt = DAG.getNode(ISD::ANY_EXTEND, dl, MVT::i32, Op.getOperand(0));
  return DAG.getBitcast(
      OpVT, DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v4i32, AnyExt));
}

// Lower a node with an INSERT_SUBVECTOR opcode.  This may result in a
// simple superregister reference or explicit instructions to insert
// the upper bits of a vector.
SDValue X86TargetLowering::LowerINSERT_SUBVECTOR(SDValue Op,
                                                 SelectionDAG &DAG) const {
  assert(Op.getSimpleValueType().getVectorElementType() == MVT::i1);

  return insert1BitVector(Op, DAG, Subtarget);
}

SDValue X86TargetLowering::LowerEXTRACT_SUBVECTOR(SDValue Op,
                                                  SelectionDAG &DAG) const {
  assert(Op.getSimpleValueType().getVectorElementType() == MVT::i1 &&
         "Only vXi1 extract_subvectors need custom lowering");

  SDLoc dl(Op);
  SDValue Vec = Op.getOperand(0);
  SDValue Idx = Op.getOperand(1);

  if (!isa<ConstantSDNode>(Idx))
    return SDValue();

  unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();
  if (IdxVal == 0) // the operation is legal
    return Op;

  MVT VecVT = Vec.getSimpleValueType();
  unsigned NumElems = VecVT.getVectorNumElements();

  // Extend to natively supported kshift.
  MVT WideVecVT = VecVT;
  if ((!Subtarget.hasDQI() && NumElems == 8) || NumElems < 8) {
    WideVecVT = Subtarget.hasDQI() ? MVT::v8i1 : MVT::v16i1;
    Vec = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, WideVecVT,
                      DAG.getUNDEF(WideVecVT), Vec,
                      DAG.getIntPtrConstant(0, dl));
  }

  // Shift to the LSB.
  Vec = DAG.getNode(X86ISD::KSHIFTR, dl, WideVecVT, Vec,
                    DAG.getConstant(IdxVal, dl, MVT::i8));

  return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, Op.getValueType(), Vec,
                     DAG.getIntPtrConstant(0, dl));
}

// Attempt to match a combined shuffle mask against supported unary shuffle
// instructions.
// TODO: Investigate sharing more of this with shuffle lowering.
static bool matchUnaryShuffle(MVT MaskVT, ArrayRef<int> Mask,
                              bool AllowFloatDomain, bool AllowIntDomain,
                              SDValue &V1, const SDLoc &DL, SelectionDAG &DAG,
                              const X86Subtarget &Subtarget, unsigned &Shuffle,
                              MVT &SrcVT, MVT &DstVT) {
  unsigned NumMaskElts = Mask.size();
  unsigned MaskEltSize = MaskVT.getScalarSizeInBits();

  // Match against a VZEXT_MOVL vXi32 zero-extending instruction.
  if (MaskEltSize == 32 && isUndefOrEqual(Mask[0], 0) &&
      isUndefOrZero(Mask[1]) && isUndefInRange(Mask, 2, NumMaskElts - 2)) {
    Shuffle = X86ISD::VZEXT_MOVL;
    SrcVT = DstVT = !Subtarget.hasSSE2() ? MVT::v4f32 : MaskVT;
    return true;
  }

  // Match against a ANY/ZERO_EXTEND_VECTOR_INREG instruction.
  // TODO: Add 512-bit vector support (split AVX512F and AVX512BW).
  if (AllowIntDomain && ((MaskVT.is128BitVector() && Subtarget.hasSSE41()) ||
                         (MaskVT.is256BitVector() && Subtarget.hasInt256()))) {
    unsigned MaxScale = 64 / MaskEltSize;
    for (unsigned Scale = 2; Scale <= MaxScale; Scale *= 2) {
      bool MatchAny = true;
      bool MatchZero = true;
      unsigned NumDstElts = NumMaskElts / Scale;
      for (unsigned i = 0; i != NumDstElts && (MatchAny || MatchZero); ++i) {
        if (!isUndefOrEqual(Mask[i * Scale], (int)i)) {
          MatchAny = MatchZero = false;
          break;
        }
        MatchAny &= isUndefInRange(Mask, (i * Scale) + 1, Scale - 1);
        MatchZero &= isUndefOrZeroInRange(Mask, (i * Scale) + 1, Scale - 1);
      }
      if (MatchAny || MatchZero) {
        assert(MatchZero && "Failed to match zext but matched aext?");
        unsigned SrcSize = std::max(128u, NumDstElts * MaskEltSize);
        MVT ScalarTy = MaskVT.isInteger() ? MaskVT.getScalarType() :
                                            MVT::getIntegerVT(MaskEltSize);
        SrcVT = MVT::getVectorVT(ScalarTy, SrcSize / MaskEltSize);

        if (SrcVT.getSizeInBits() != MaskVT.getSizeInBits())
          V1 = extractSubVector(V1, 0, DAG, DL, SrcSize);

        Shuffle = unsigned(MatchAny ? ISD::ANY_EXTEND : ISD::ZERO_EXTEND);
        if (SrcVT.getVectorNumElements() != NumDstElts)
          Shuffle = getOpcode_EXTEND_VECTOR_INREG(Shuffle);

        DstVT = MVT::getIntegerVT(Scale * MaskEltSize);
        DstVT = MVT::getVectorVT(DstVT, NumDstElts);
        return true;
      }
    }
  }

  // Match against a VZEXT_MOVL instruction, SSE1 only supports 32-bits (MOVSS).
  if (((MaskEltSize == 32) || (MaskEltSize == 64 && Subtarget.hasSSE2())) &&
      isUndefOrEqual(Mask[0], 0) &&
      isUndefOrZeroInRange(Mask, 1, NumMaskElts - 1)) {
    Shuffle = X86ISD::VZEXT_MOVL;
    SrcVT = DstVT = !Subtarget.hasSSE2() ? MVT::v4f32 : MaskVT;
    return true;
  }

  // Check if we have SSE3 which will let us use MOVDDUP etc. The
  // instructions are no slower than UNPCKLPD but has the option to
  // fold the input operand into even an unaligned memory load.
  if (MaskVT.is128BitVector() && Subtarget.hasSSE3() && AllowFloatDomain) {
    if (isTargetShuffleEquivalent(Mask, {0, 0})) {
      Shuffle = X86ISD::MOVDDUP;
      SrcVT = DstVT = MVT::v2f64;
      return true;
    }
    if (isTargetShuffleEquivalent(Mask, {0, 0, 2, 2})) {
      Shuffle = X86ISD::MOVSLDUP;
      SrcVT = DstVT = MVT::v4f32;
      return true;
    }
    if (isTargetShuffleEquivalent(Mask, {1, 1, 3, 3})) {
      Shuffle = X86ISD::MOVSHDUP;
      SrcVT = DstVT = MVT::v4f32;
      return true;
    }
  }

  if (MaskVT.is256BitVector() && AllowFloatDomain) {
    assert(Subtarget.hasAVX() && "AVX required for 256-bit vector shuffles");
    if (isTargetShuffleEquivalent(Mask, {0, 0, 2, 2})) {
      Shuffle = X86ISD::MOVDDUP;
      SrcVT = DstVT = MVT::v4f64;
      return true;
    }
    if (isTargetShuffleEquivalent(Mask, {0, 0, 2, 2, 4, 4, 6, 6})) {
      Shuffle = X86ISD::MOVSLDUP;
      SrcVT = DstVT = MVT::v8f32;
      return true;
    }
    if (isTargetShuffleEquivalent(Mask, {1, 1, 3, 3, 5, 5, 7, 7})) {
      Shuffle = X86ISD::MOVSHDUP;
      SrcVT = DstVT = MVT::v8f32;
      return true;
    }
  }

  if (MaskVT.is512BitVector() && AllowFloatDomain) {
    assert(Subtarget.hasAVX512() &&
           "AVX512 required for 512-bit vector shuffles");
    if (isTargetShuffleEquivalent(Mask, {0, 0, 2, 2, 4, 4, 6, 6})) {
      Shuffle = X86ISD::MOVDDUP;
      SrcVT = DstVT = MVT::v8f64;
      return true;
    }
    if (isTargetShuffleEquivalent(
            Mask, {0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14})) {
      Shuffle = X86ISD::MOVSLDUP;
      SrcVT = DstVT = MVT::v16f32;
      return true;
    }
    if (isTargetShuffleEquivalent(
            Mask, {1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15})) {
      Shuffle = X86ISD::MOVSHDUP;
      SrcVT = DstVT = MVT::v16f32;
      return true;
    }
  }

  return false;
}

// Attempt to match a combined shuffle mask against supported unary immediate
// permute instructions.
// TODO: Investigate sharing more of this with shuffle lowering.
static bool matchUnaryPermuteShuffle(MVT MaskVT, ArrayRef<int> Mask,
                                     const APInt &Zeroable,
                                     bool AllowFloatDomain, bool AllowIntDomain,
                                     const X86Subtarget &Subtarget,
                                     unsigned &Shuffle, MVT &ShuffleVT,
                                     unsigned &PermuteImm) {
  unsigned NumMaskElts = Mask.size();
  unsigned InputSizeInBits = MaskVT.getSizeInBits();
  unsigned MaskScalarSizeInBits = InputSizeInBits / NumMaskElts;
  MVT MaskEltVT = MVT::getIntegerVT(MaskScalarSizeInBits);

  bool ContainsZeros =
      llvm::any_of(Mask, [](int M) { return M == SM_SentinelZero; });

  // Handle VPERMI/VPERMILPD vXi64/vXi64 patterns.
  if (!ContainsZeros && MaskScalarSizeInBits == 64) {
    // Check for lane crossing permutes.
    if (is128BitLaneCrossingShuffleMask(MaskEltVT, Mask)) {
      // PERMPD/PERMQ permutes within a 256-bit vector (AVX2+).
      if (Subtarget.hasAVX2() && MaskVT.is256BitVector()) {
        Shuffle = X86ISD::VPERMI;
        ShuffleVT = (AllowFloatDomain ? MVT::v4f64 : MVT::v4i64);
        PermuteImm = getV4X86ShuffleImm(Mask);
        return true;
      }
      if (Subtarget.hasAVX512() && MaskVT.is512BitVector()) {
        SmallVector<int, 4> RepeatedMask;
        if (is256BitLaneRepeatedShuffleMask(MVT::v8f64, Mask, RepeatedMask)) {
          Shuffle = X86ISD::VPERMI;
          ShuffleVT = (AllowFloatDomain ? MVT::v8f64 : MVT::v8i64);
          PermuteImm = getV4X86ShuffleImm(RepeatedMask);
          return true;
        }
      }
    } else if (AllowFloatDomain && Subtarget.hasAVX()) {
      // VPERMILPD can permute with a non-repeating shuffle.
      Shuffle = X86ISD::VPERMILPI;
      ShuffleVT = MVT::getVectorVT(MVT::f64, Mask.size());
      PermuteImm = 0;
      for (int i = 0, e = Mask.size(); i != e; ++i) {
        int M = Mask[i];
        if (M == SM_SentinelUndef)
          continue;
        assert(((M / 2) == (i / 2)) && "Out of range shuffle mask index");
        PermuteImm |= (M & 1) << i;
      }
      return true;
    }
  }

  // Handle PSHUFD/VPERMILPI vXi32/vXf32 repeated patterns.
  // AVX introduced the VPERMILPD/VPERMILPS float permutes, before then we
  // had to use 2-input SHUFPD/SHUFPS shuffles (not handled here).
  if ((MaskScalarSizeInBits == 64 || MaskScalarSizeInBits == 32) &&
      !ContainsZeros && (AllowIntDomain || Subtarget.hasAVX())) {
    SmallVector<int, 4> RepeatedMask;
    if (is128BitLaneRepeatedShuffleMask(MaskEltVT, Mask, RepeatedMask)) {
      // Narrow the repeated mask to create 32-bit element permutes.
      SmallVector<int, 4> WordMask = RepeatedMask;
      if (MaskScalarSizeInBits == 64)
        scaleShuffleMask<int>(2, RepeatedMask, WordMask);

      Shuffle = (AllowIntDomain ? X86ISD::PSHUFD : X86ISD::VPERMILPI);
      ShuffleVT = (AllowIntDomain ? MVT::i32 : MVT::f32);
      ShuffleVT = MVT::getVectorVT(ShuffleVT, InputSizeInBits / 32);
      PermuteImm = getV4X86ShuffleImm(WordMask);
      return true;
    }
  }

  // Handle PSHUFLW/PSHUFHW vXi16 repeated patterns.
  if (!ContainsZeros && AllowIntDomain && MaskScalarSizeInBits == 16) {
    SmallVector<int, 4> RepeatedMask;
    if (is128BitLaneRepeatedShuffleMask(MaskEltVT, Mask, RepeatedMask)) {
      ArrayRef<int> LoMask(RepeatedMask.data() + 0, 4);
      ArrayRef<int> HiMask(RepeatedMask.data() + 4, 4);

      // PSHUFLW: permute lower 4 elements only.
      if (isUndefOrInRange(LoMask, 0, 4) &&
          isSequentialOrUndefInRange(HiMask, 0, 4, 4)) {
        Shuffle = X86ISD::PSHUFLW;
        ShuffleVT = MVT::getVectorVT(MVT::i16, InputSizeInBits / 16);
        PermuteImm = getV4X86ShuffleImm(LoMask);
        return true;
      }

      // PSHUFHW: permute upper 4 elements only.
      if (isUndefOrInRange(HiMask, 4, 8) &&
          isSequentialOrUndefInRange(LoMask, 0, 4, 0)) {
        // Offset the HiMask so that we can create the shuffle immediate.
        int OffsetHiMask[4];
        for (int i = 0; i != 4; ++i)
          OffsetHiMask[i] = (HiMask[i] < 0 ? HiMask[i] : HiMask[i] - 4);

        Shuffle = X86ISD::PSHUFHW;
        ShuffleVT = MVT::getVectorVT(MVT::i16, InputSizeInBits / 16);
        PermuteImm = getV4X86ShuffleImm(OffsetHiMask);
        return true;
      }
    }
  }

  // Attempt to match against byte/bit shifts.
  // FIXME: Add 512-bit support.
  if (AllowIntDomain && ((MaskVT.is128BitVector() && Subtarget.hasSSE2()) ||
                         (MaskVT.is256BitVector() && Subtarget.hasAVX2()))) {
    int ShiftAmt = matchShuffleAsShift(ShuffleVT, Shuffle, MaskScalarSizeInBits,
                                       Mask, 0, Zeroable, Subtarget);
    if (0 < ShiftAmt) {
      PermuteImm = (unsigned)ShiftAmt;
      return true;
    }
  }

  return false;
}

// Attempt to match a combined unary shuffle mask against supported binary
// shuffle instructions.
// TODO: Investigate sharing more of this with shuffle lowering.
static bool matchBinaryShuffle(MVT MaskVT, ArrayRef<int> Mask,
                               bool AllowFloatDomain, bool AllowIntDomain,
                               SDValue &V1, SDValue &V2, const SDLoc &DL,
                               SelectionDAG &DAG, const X86Subtarget &Subtarget,
                               unsigned &Shuffle, MVT &SrcVT, MVT &DstVT,
                               bool IsUnary) {
  unsigned EltSizeInBits = MaskVT.getScalarSizeInBits();

  if (MaskVT.is128BitVector()) {
    if (isTargetShuffleEquivalent(Mask, {0, 0}) && AllowFloatDomain) {
      V2 = V1;
      V1 = (SM_SentinelUndef == Mask[0] ? DAG.getUNDEF(MVT::v4f32) : V1);
      Shuffle = Subtarget.hasSSE2() ? X86ISD::UNPCKL : X86ISD::MOVLHPS;
      SrcVT = DstVT = Subtarget.hasSSE2() ? MVT::v2f64 : MVT::v4f32;
      return true;
    }
    if (isTargetShuffleEquivalent(Mask, {1, 1}) && AllowFloatDomain) {
      V2 = V1;
      Shuffle = Subtarget.hasSSE2() ? X86ISD::UNPCKH : X86ISD::MOVHLPS;
      SrcVT = DstVT = Subtarget.hasSSE2() ? MVT::v2f64 : MVT::v4f32;
      return true;
    }
    if (isTargetShuffleEquivalent(Mask, {0, 3}) && Subtarget.hasSSE2() &&
        (AllowFloatDomain || !Subtarget.hasSSE41())) {
      std::swap(V1, V2);
      Shuffle = X86ISD::MOVSD;
      SrcVT = DstVT = MVT::v2f64;
      return true;
    }
    if (isTargetShuffleEquivalent(Mask, {4, 1, 2, 3}) &&
        (AllowFloatDomain || !Subtarget.hasSSE41())) {
      Shuffle = X86ISD::MOVSS;
      SrcVT = DstVT = MVT::v4f32;
      return true;
    }
  }

  // Attempt to match against either an unary or binary PACKSS/PACKUS shuffle.
  if (((MaskVT == MVT::v8i16 || MaskVT == MVT::v16i8) && Subtarget.hasSSE2()) ||
      ((MaskVT == MVT::v16i16 || MaskVT == MVT::v32i8) && Subtarget.hasInt256()) ||
      ((MaskVT == MVT::v32i16 || MaskVT == MVT::v64i8) && Subtarget.hasBWI())) {
    if (matchVectorShuffleWithPACK(MaskVT, SrcVT, V1, V2, Shuffle, Mask, DAG,
                                   Subtarget)) {
      DstVT = MaskVT;
      return true;
    }
  }

  // Attempt to match against either a unary or binary UNPCKL/UNPCKH shuffle.
  if ((MaskVT == MVT::v4f32 && Subtarget.hasSSE1()) ||
      (MaskVT.is128BitVector() && Subtarget.hasSSE2()) ||
      (MaskVT.is256BitVector() && 32 <= EltSizeInBits && Subtarget.hasAVX()) ||
      (MaskVT.is256BitVector() && Subtarget.hasAVX2()) ||
      (MaskVT.is512BitVector() && Subtarget.hasAVX512())) {
    if (matchVectorShuffleWithUNPCK(MaskVT, V1, V2, Shuffle, IsUnary, Mask, DL,
                                    DAG, Subtarget)) {
      SrcVT = DstVT = MaskVT;
      if (MaskVT.is256BitVector() && !Subtarget.hasAVX2())
        SrcVT = DstVT = (32 == EltSizeInBits ? MVT::v8f32 : MVT::v4f64);
      return true;
    }
  }

  return false;
}

static bool matchBinaryPermuteShuffle(
    MVT MaskVT, ArrayRef<int> Mask, const APInt &Zeroable,
    bool AllowFloatDomain, bool AllowIntDomain, SDValue &V1, SDValue &V2,
    const SDLoc &DL, SelectionDAG &DAG, const X86Subtarget &Subtarget,
    unsigned &Shuffle, MVT &ShuffleVT, unsigned &PermuteImm) {
  unsigned NumMaskElts = Mask.size();
  unsigned EltSizeInBits = MaskVT.getScalarSizeInBits();

  // Attempt to match against PALIGNR byte rotate.
  if (AllowIntDomain && ((MaskVT.is128BitVector() && Subtarget.hasSSSE3()) ||
                         (MaskVT.is256BitVector() && Subtarget.hasAVX2()))) {
    int ByteRotation = matchShuffleAsByteRotate(MaskVT, V1, V2, Mask);
    if (0 < ByteRotation) {
      Shuffle = X86ISD::PALIGNR;
      ShuffleVT = MVT::getVectorVT(MVT::i8, MaskVT.getSizeInBits() / 8);
      PermuteImm = ByteRotation;
      return true;
    }
  }

  // Attempt to combine to X86ISD::BLENDI.
  if ((NumMaskElts <= 8 && ((Subtarget.hasSSE41() && MaskVT.is128BitVector()) ||
                            (Subtarget.hasAVX() && MaskVT.is256BitVector()))) ||
      (MaskVT == MVT::v16i16 && Subtarget.hasAVX2())) {
    uint64_t BlendMask = 0;
    bool ForceV1Zero = false, ForceV2Zero = false;
    SmallVector<int, 8> TargetMask(Mask.begin(), Mask.end());
    if (matchVectorShuffleAsBlend(V1, V2, TargetMask, ForceV1Zero, ForceV2Zero,
                                  BlendMask)) {
      if (MaskVT == MVT::v16i16) {
        // We can only use v16i16 PBLENDW if the lanes are repeated.
        SmallVector<int, 8> RepeatedMask;
        if (isRepeatedTargetShuffleMask(128, MaskVT, TargetMask,
                                        RepeatedMask)) {
          assert(RepeatedMask.size() == 8 &&
                 "Repeated mask size doesn't match!");
          PermuteImm = 0;
          for (int i = 0; i < 8; ++i)
            if (RepeatedMask[i] >= 8)
              PermuteImm |= 1 << i;
          V1 = ForceV1Zero ? getZeroVector(MaskVT, Subtarget, DAG, DL) : V1;
          V2 = ForceV2Zero ? getZeroVector(MaskVT, Subtarget, DAG, DL) : V2;
          Shuffle = X86ISD::BLENDI;
          ShuffleVT = MaskVT;
          return true;
        }
      } else {
        V1 = ForceV1Zero ? getZeroVector(MaskVT, Subtarget, DAG, DL) : V1;
        V2 = ForceV2Zero ? getZeroVector(MaskVT, Subtarget, DAG, DL) : V2;
        PermuteImm = (unsigned)BlendMask;
        Shuffle = X86ISD::BLENDI;
        ShuffleVT = MaskVT;
        return true;
      }
    }
  }

  // Attempt to combine to INSERTPS, but only if it has elements that need to
  // be set to zero.
  if (AllowFloatDomain && EltSizeInBits == 32 && Subtarget.hasSSE41() &&
      MaskVT.is128BitVector() &&
      llvm::any_of(Mask, [](int M) { return M == SM_SentinelZero; }) &&
      matchShuffleAsInsertPS(V1, V2, PermuteImm, Zeroable, Mask, DAG)) {
    Shuffle = X86ISD::INSERTPS;
    ShuffleVT = MVT::v4f32;
    return true;
  }

  // Attempt to combine to SHUFPD.
  if (AllowFloatDomain && EltSizeInBits == 64 &&
      ((MaskVT.is128BitVector() && Subtarget.hasSSE2()) ||
       (MaskVT.is256BitVector() && Subtarget.hasAVX()) ||
       (MaskVT.is512BitVector() && Subtarget.hasAVX512()))) {
    if (matchShuffleWithSHUFPD(MaskVT, V1, V2, PermuteImm, Mask)) {
      Shuffle = X86ISD::SHUFP;
      ShuffleVT = MVT::getVectorVT(MVT::f64, MaskVT.getSizeInBits() / 64);
      return true;
    }
  }

  // Attempt to combine to SHUFPS.
  if (AllowFloatDomain && EltSizeInBits == 32 &&
      ((MaskVT.is128BitVector() && Subtarget.hasSSE1()) ||
       (MaskVT.is256BitVector() && Subtarget.hasAVX()) ||
       (MaskVT.is512BitVector() && Subtarget.hasAVX512()))) {
    SmallVector<int, 4> RepeatedMask;
    if (isRepeatedTargetShuffleMask(128, MaskVT, Mask, RepeatedMask)) {
      // Match each half of the repeated mask, to determine if its just
      // referencing one of the vectors, is zeroable or entirely undef.
      auto MatchHalf = [&](unsigned Offset, int &S0, int &S1) {
        int M0 = RepeatedMask[Offset];
        int M1 = RepeatedMask[Offset + 1];

        if (isUndefInRange(RepeatedMask, Offset, 2)) {
          return DAG.getUNDEF(MaskVT);
        } else if (isUndefOrZeroInRange(RepeatedMask, Offset, 2)) {
          S0 = (SM_SentinelUndef == M0 ? -1 : 0);
          S1 = (SM_SentinelUndef == M1 ? -1 : 1);
          return getZeroVector(MaskVT, Subtarget, DAG, DL);
        } else if (isUndefOrInRange(M0, 0, 4) && isUndefOrInRange(M1, 0, 4)) {
          S0 = (SM_SentinelUndef == M0 ? -1 : M0 & 3);
          S1 = (SM_SentinelUndef == M1 ? -1 : M1 & 3);
          return V1;
        } else if (isUndefOrInRange(M0, 4, 8) && isUndefOrInRange(M1, 4, 8)) {
          S0 = (SM_SentinelUndef == M0 ? -1 : M0 & 3);
          S1 = (SM_SentinelUndef == M1 ? -1 : M1 & 3);
          return V2;
        }

        return SDValue();
      };

      int ShufMask[4] = {-1, -1, -1, -1};
      SDValue Lo = MatchHalf(0, ShufMask[0], ShufMask[1]);
      SDValue Hi = MatchHalf(2, ShufMask[2], ShufMask[3]);

      if (Lo && Hi) {
        V1 = Lo;
        V2 = Hi;
        Shuffle = X86ISD::SHUFP;
        ShuffleVT = MVT::getVectorVT(MVT::f32, MaskVT.getSizeInBits() / 32);
        PermuteImm = getV4X86ShuffleImm(ShufMask);
        return true;
      }
    }
  }

  // Attempt to combine to INSERTPS more generally if X86ISD::SHUFP failed.
  if (AllowFloatDomain && EltSizeInBits == 32 && Subtarget.hasSSE41() &&
      MaskVT.is128BitVector() &&
      matchShuffleAsInsertPS(V1, V2, PermuteImm, Zeroable, Mask, DAG)) {
    Shuffle = X86ISD::INSERTPS;
    ShuffleVT = MVT::v4f32;
    return true;
  }

  return false;
}

static SDValue combineX86ShuffleChainWithExtract(
    ArrayRef<SDValue> Inputs, SDValue Root, ArrayRef<int> BaseMask, int Depth,
    bool HasVariableMask, bool AllowVariableMask, SelectionDAG &DAG,
    const X86Subtarget &Subtarget);

/// Combine an arbitrary chain of shuffles into a single instruction if
/// possible.
///
/// This is the leaf of the recursive combine below. When we have found some
/// chain of single-use x86 shuffle instructions and accumulated the combined
/// shuffle mask represented by them, this will try to pattern match that mask
/// into either a single instruction if there is a special purpose instruction
/// for this operation, or into a PSHUFB instruction which is a fully general
/// instruction but should only be used to replace chains over a certain depth.
static SDValue combineX86ShuffleChain(ArrayRef<SDValue> Inputs, SDValue Root,
                                      ArrayRef<int> BaseMask, int Depth,
                                      bool HasVariableMask,
                                      bool AllowVariableMask, SelectionDAG &DAG,
                                      const X86Subtarget &Subtarget) {
  assert(!BaseMask.empty() && "Cannot combine an empty shuffle mask!");
  assert((Inputs.size() == 1 || Inputs.size() == 2) &&
         "Unexpected number of shuffle inputs!");

  // Find the inputs that enter the chain. Note that multiple uses are OK
  // here, we're not going to remove the operands we find.
  bool UnaryShuffle = (Inputs.size() == 1);
  SDValue V1 = peekThroughBitcasts(Inputs[0]);
  SDValue V2 = (UnaryShuffle ? DAG.getUNDEF(V1.getValueType())
                             : peekThroughBitcasts(Inputs[1]));

  MVT VT1 = V1.getSimpleValueType();
  MVT VT2 = V2.getSimpleValueType();
  MVT RootVT = Root.getSimpleValueType();
  assert(VT1.getSizeInBits() == RootVT.getSizeInBits() &&
         VT2.getSizeInBits() == RootVT.getSizeInBits() &&
         "Vector size mismatch");

  SDLoc DL(Root);
  SDValue Res;

  unsigned NumBaseMaskElts = BaseMask.size();
  if (NumBaseMaskElts == 1) {
    assert(BaseMask[0] == 0 && "Invalid shuffle index found!");
    return DAG.getBitcast(RootVT, V1);
  }

  unsigned RootSizeInBits = RootVT.getSizeInBits();
  unsigned NumRootElts = RootVT.getVectorNumElements();
  unsigned BaseMaskEltSizeInBits = RootSizeInBits / NumBaseMaskElts;
  bool FloatDomain = VT1.isFloatingPoint() || VT2.isFloatingPoint() ||
                     (RootVT.isFloatingPoint() && Depth >= 1) ||
                     (RootVT.is256BitVector() && !Subtarget.hasAVX2());

  // Don't combine if we are a AVX512/EVEX target and the mask element size
  // is different from the root element size - this would prevent writemasks
  // from being reused.
  // TODO - this currently prevents all lane shuffles from occurring.
  // TODO - check for writemasks usage instead of always preventing combining.
  // TODO - attempt to narrow Mask back to writemask size.
  bool IsEVEXShuffle =
      RootSizeInBits == 512 || (Subtarget.hasVLX() && RootSizeInBits >= 128);

  // Attempt to match a subvector broadcast.
  // shuffle(insert_subvector(undef, sub, 0), undef, 0, 0, 0, 0)
  if (UnaryShuffle &&
      (BaseMaskEltSizeInBits == 128 || BaseMaskEltSizeInBits == 256)) {
    SmallVector<int, 64> BroadcastMask(NumBaseMaskElts, 0);
    if (isTargetShuffleEquivalent(BaseMask, BroadcastMask)) {
      SDValue Src = Inputs[0];
      if (Src.getOpcode() == ISD::INSERT_SUBVECTOR &&
          Src.getOperand(0).isUndef() &&
          Src.getOperand(1).getValueSizeInBits() == BaseMaskEltSizeInBits &&
          MayFoldLoad(Src.getOperand(1)) && isNullConstant(Src.getOperand(2))) {
        return DAG.getBitcast(RootVT, DAG.getNode(X86ISD::SUBV_BROADCAST, DL,
                                                  Src.getValueType(),
                                                  Src.getOperand(1)));
      }
    }
  }

  // TODO - handle 128/256-bit lane shuffles of 512-bit vectors.

  // Handle 128-bit lane shuffles of 256-bit vectors.
  // If we have AVX2, prefer to use VPERMQ/VPERMPD for unary shuffles unless
  // we need to use the zeroing feature.
  // TODO - this should support binary shuffles.
  if (UnaryShuffle && RootVT.is256BitVector() && NumBaseMaskElts == 2 &&
      !(Subtarget.hasAVX2() && BaseMask[0] >= -1 && BaseMask[1] >= -1) &&
      !isSequentialOrUndefOrZeroInRange(BaseMask, 0, 2, 0)) {
    if (Depth == 0 && Root.getOpcode() == X86ISD::VPERM2X128)
      return SDValue(); // Nothing to do!
    MVT ShuffleVT = (FloatDomain ? MVT::v4f64 : MVT::v4i64);
    unsigned PermMask = 0;
    PermMask |= ((BaseMask[0] < 0 ? 0x8 : (BaseMask[0] & 1)) << 0);
    PermMask |= ((BaseMask[1] < 0 ? 0x8 : (BaseMask[1] & 1)) << 4);

    Res = DAG.getBitcast(ShuffleVT, V1);
    Res = DAG.getNode(X86ISD::VPERM2X128, DL, ShuffleVT, Res,
                      DAG.getUNDEF(ShuffleVT),
                      DAG.getConstant(PermMask, DL, MVT::i8));
    return DAG.getBitcast(RootVT, Res);
  }

  // For masks that have been widened to 128-bit elements or more,
  // narrow back down to 64-bit elements.
  SmallVector<int, 64> Mask;
  if (BaseMaskEltSizeInBits > 64) {
    assert((BaseMaskEltSizeInBits % 64) == 0 && "Illegal mask size");
    int MaskScale = BaseMaskEltSizeInBits / 64;
    scaleShuffleMask<int>(MaskScale, BaseMask, Mask);
  } else {
    Mask = SmallVector<int, 64>(BaseMask.begin(), BaseMask.end());
  }

  unsigned NumMaskElts = Mask.size();
  unsigned MaskEltSizeInBits = RootSizeInBits / NumMaskElts;

  // Determine the effective mask value type.
  FloatDomain &= (32 <= MaskEltSizeInBits);
  MVT MaskVT = FloatDomain ? MVT::getFloatingPointVT(MaskEltSizeInBits)
                           : MVT::getIntegerVT(MaskEltSizeInBits);
  MaskVT = MVT::getVectorVT(MaskVT, NumMaskElts);

  // Only allow legal mask types.
  if (!DAG.getTargetLoweringInfo().isTypeLegal(MaskVT))
    return SDValue();

  // Attempt to match the mask against known shuffle patterns.
  MVT ShuffleSrcVT, ShuffleVT;
  unsigned Shuffle, PermuteImm;

  // Which shuffle domains are permitted?
  // Permit domain crossing at higher combine depths.
  // TODO: Should we indicate which domain is preferred if both are allowed?
  bool AllowFloatDomain = FloatDomain || (Depth >= 3);
  bool AllowIntDomain = (!FloatDomain || (Depth >= 3)) && Subtarget.hasSSE2() &&
                        (!MaskVT.is256BitVector() || Subtarget.hasAVX2());

  // Determine zeroable mask elements.
  APInt Zeroable(NumMaskElts, 0);
  for (unsigned i = 0; i != NumMaskElts; ++i)
    if (isUndefOrZero(Mask[i]))
      Zeroable.setBit(i);

  if (UnaryShuffle) {
    // If we are shuffling a X86ISD::VZEXT_LOAD then we can use the load
    // directly if we don't shuffle the lower element and we shuffle the upper
    // (zero) elements within themselves.
    if (V1.getOpcode() == X86ISD::VZEXT_LOAD &&
        (cast<MemIntrinsicSDNode>(V1)->getMemoryVT().getScalarSizeInBits() %
         MaskEltSizeInBits) == 0) {
      unsigned Scale =
          cast<MemIntrinsicSDNode>(V1)->getMemoryVT().getScalarSizeInBits() /
          MaskEltSizeInBits;
      ArrayRef<int> HiMask(Mask.data() + Scale, NumMaskElts - Scale);
      if (isSequentialOrUndefInRange(Mask, 0, Scale, 0) &&
          isUndefOrZeroOrInRange(HiMask, Scale, NumMaskElts)) {
        return DAG.getBitcast(RootVT, V1);
      }
    }

    // Attempt to match against broadcast-from-vector.
    // Limit AVX1 to cases where we're loading+broadcasting a scalar element.
    if ((Subtarget.hasAVX2() || (Subtarget.hasAVX() && 32 <= MaskEltSizeInBits))
        && (!IsEVEXShuffle || NumRootElts == NumMaskElts)) {
      SmallVector<int, 64> BroadcastMask(NumMaskElts, 0);
      if (isTargetShuffleEquivalent(Mask, BroadcastMask)) {
        if (V1.getValueType() == MaskVT &&
            V1.getOpcode() == ISD::SCALAR_TO_VECTOR &&
            MayFoldLoad(V1.getOperand(0))) {
          if (Depth == 0 && Root.getOpcode() == X86ISD::VBROADCAST)
            return SDValue(); // Nothing to do!
          Res = V1.getOperand(0);
          Res = DAG.getNode(X86ISD::VBROADCAST, DL, MaskVT, Res);
          return DAG.getBitcast(RootVT, Res);
        }
        if (Subtarget.hasAVX2()) {
          if (Depth == 0 && Root.getOpcode() == X86ISD::VBROADCAST)
            return SDValue(); // Nothing to do!
          Res = DAG.getBitcast(MaskVT, V1);
          Res = DAG.getNode(X86ISD::VBROADCAST, DL, MaskVT, Res);
          return DAG.getBitcast(RootVT, Res);
        }
      }
    }

    SDValue NewV1 = V1; // Save operand in case early exit happens.
    if (matchUnaryShuffle(MaskVT, Mask, AllowFloatDomain, AllowIntDomain, NewV1,
                          DL, DAG, Subtarget, Shuffle, ShuffleSrcVT,
                          ShuffleVT) &&
        (!IsEVEXShuffle || (NumRootElts == ShuffleVT.getVectorNumElements()))) {
      if (Depth == 0 && Root.getOpcode() == Shuffle)
        return SDValue(); // Nothing to do!
      Res = DAG.getBitcast(ShuffleSrcVT, NewV1);
      Res = DAG.getNode(Shuffle, DL, ShuffleVT, Res);
      return DAG.getBitcast(RootVT, Res);
    }

    if (matchUnaryPermuteShuffle(MaskVT, Mask, Zeroable, AllowFloatDomain,
                                 AllowIntDomain, Subtarget, Shuffle, ShuffleVT,
                                 PermuteImm) &&
        (!IsEVEXShuffle || (NumRootElts == ShuffleVT.getVectorNumElements()))) {
      if (Depth == 0 && Root.getOpcode() == Shuffle)
        return SDValue(); // Nothing to do!
      Res = DAG.getBitcast(ShuffleVT, V1);
      Res = DAG.getNode(Shuffle, DL, ShuffleVT, Res,
                        DAG.getConstant(PermuteImm, DL, MVT::i8));
      return DAG.getBitcast(RootVT, Res);
    }
  }

  SDValue NewV1 = V1; // Save operands in case early exit happens.
  SDValue NewV2 = V2;
  if (matchBinaryShuffle(MaskVT, Mask, AllowFloatDomain, AllowIntDomain, NewV1,
                         NewV2, DL, DAG, Subtarget, Shuffle, ShuffleSrcVT,
                         ShuffleVT, UnaryShuffle) &&
      (!IsEVEXShuffle || (NumRootElts == ShuffleVT.getVectorNumElements()))) {
    if (Depth == 0 && Root.getOpcode() == Shuffle)
      return SDValue(); // Nothing to do!
    NewV1 = DAG.getBitcast(ShuffleSrcVT, NewV1);
    NewV2 = DAG.getBitcast(ShuffleSrcVT, NewV2);
    Res = DAG.getNode(Shuffle, DL, ShuffleVT, NewV1, NewV2);
    return DAG.getBitcast(RootVT, Res);
  }

  NewV1 = V1; // Save operands in case early exit happens.
  NewV2 = V2;
  if (matchBinaryPermuteShuffle(
          MaskVT, Mask, Zeroable, AllowFloatDomain, AllowIntDomain, NewV1,
          NewV2, DL, DAG, Subtarget, Shuffle, ShuffleVT, PermuteImm) &&
      (!IsEVEXShuffle || (NumRootElts == ShuffleVT.getVectorNumElements()))) {
    if (Depth == 0 && Root.getOpcode() == Shuffle)
      return SDValue(); // Nothing to do!
    NewV1 = DAG.getBitcast(ShuffleVT, NewV1);
    NewV2 = DAG.getBitcast(ShuffleVT, NewV2);
    Res = DAG.getNode(Shuffle, DL, ShuffleVT, NewV1, NewV2,
                      DAG.getConstant(PermuteImm, DL, MVT::i8));
    return DAG.getBitcast(RootVT, Res);
  }

  // Typically from here on, we need an integer version of MaskVT.
  MVT IntMaskVT = MVT::getIntegerVT(MaskEltSizeInBits);
  IntMaskVT = MVT::getVectorVT(IntMaskVT, NumMaskElts);

  // Annoyingly, SSE4A instructions don't map into the above match helpers.
  if (Subtarget.hasSSE4A() && AllowIntDomain && RootSizeInBits == 128) {
    uint64_t BitLen, BitIdx;
    if (matchShuffleAsEXTRQ(IntMaskVT, V1, V2, Mask, BitLen, BitIdx,
                            Zeroable)) {
      if (Depth == 0 && Root.getOpcode() == X86ISD::EXTRQI)
        return SDValue(); // Nothing to do!
      V1 = DAG.getBitcast(IntMaskVT, V1);
      Res = DAG.getNode(X86ISD::EXTRQI, DL, IntMaskVT, V1,
                        DAG.getConstant(BitLen, DL, MVT::i8),
                        DAG.getConstant(BitIdx, DL, MVT::i8));
      return DAG.getBitcast(RootVT, Res);
    }

    if (matchShuffleAsINSERTQ(IntMaskVT, V1, V2, Mask, BitLen, BitIdx)) {
      if (Depth == 0 && Root.getOpcode() == X86ISD::INSERTQI)
        return SDValue(); // Nothing to do!
      V1 = DAG.getBitcast(IntMaskVT, V1);
      V2 = DAG.getBitcast(IntMaskVT, V2);
      Res = DAG.getNode(X86ISD::INSERTQI, DL, IntMaskVT, V1, V2,
                        DAG.getConstant(BitLen, DL, MVT::i8),
                        DAG.getConstant(BitIdx, DL, MVT::i8));
      return DAG.getBitcast(RootVT, Res);
    }
  }

  // Don't try to re-form single instruction chains under any circumstances now
  // that we've done encoding canonicalization for them.
  if (Depth < 1)
    return SDValue();

  // Depth threshold above which we can efficiently use variable mask shuffles.
  int VariableShuffleDepth = Subtarget.hasFastVariableShuffle() ? 1 : 2;
  AllowVariableMask &= (Depth >= VariableShuffleDepth) || HasVariableMask;

  bool MaskContainsZeros =
      any_of(Mask, [](int M) { return M == SM_SentinelZero; });

  if (is128BitLaneCrossingShuffleMask(MaskVT, Mask)) {
    // If we have a single input lane-crossing shuffle then lower to VPERMV.
    if (UnaryShuffle && AllowVariableMask && !MaskContainsZeros &&
        ((Subtarget.hasAVX2() &&
          (MaskVT == MVT::v8f32 || MaskVT == MVT::v8i32)) ||
         (Subtarget.hasAVX512() &&
          (MaskVT == MVT::v8f64 || MaskVT == MVT::v8i64 ||
           MaskVT == MVT::v16f32 || MaskVT == MVT::v16i32)) ||
         (Subtarget.hasBWI() && MaskVT == MVT::v32i16) ||
         (Subtarget.hasBWI() && Subtarget.hasVLX() && MaskVT == MVT::v16i16) ||
         (Subtarget.hasVBMI() && MaskVT == MVT::v64i8) ||
         (Subtarget.hasVBMI() && Subtarget.hasVLX() && MaskVT == MVT::v32i8))) {
      SDValue VPermMask = getConstVector(Mask, IntMaskVT, DAG, DL, true);
      Res = DAG.getBitcast(MaskVT, V1);
      Res = DAG.getNode(X86ISD::VPERMV, DL, MaskVT, VPermMask, Res);
      return DAG.getBitcast(RootVT, Res);
    }

    // Lower a unary+zero lane-crossing shuffle as VPERMV3 with a zero
    // vector as the second source.
    if (UnaryShuffle && AllowVariableMask &&
        ((Subtarget.hasAVX512() &&
          (MaskVT == MVT::v8f64 || MaskVT == MVT::v8i64 ||
           MaskVT == MVT::v16f32 || MaskVT == MVT::v16i32)) ||
         (Subtarget.hasVLX() &&
          (MaskVT == MVT::v4f64 || MaskVT == MVT::v4i64 ||
           MaskVT == MVT::v8f32 || MaskVT == MVT::v8i32)) ||
         (Subtarget.hasBWI() && MaskVT == MVT::v32i16) ||
         (Subtarget.hasBWI() && Subtarget.hasVLX() && MaskVT == MVT::v16i16) ||
         (Subtarget.hasVBMI() && MaskVT == MVT::v64i8) ||
         (Subtarget.hasVBMI() && Subtarget.hasVLX() && MaskVT == MVT::v32i8))) {
      // Adjust shuffle mask - replace SM_SentinelZero with second source index.
      for (unsigned i = 0; i != NumMaskElts; ++i)
        if (Mask[i] == SM_SentinelZero)
          Mask[i] = NumMaskElts + i;

      SDValue VPermMask = getConstVector(Mask, IntMaskVT, DAG, DL, true);
      Res = DAG.getBitcast(MaskVT, V1);
      SDValue Zero = getZeroVector(MaskVT, Subtarget, DAG, DL);
      Res = DAG.getNode(X86ISD::VPERMV3, DL, MaskVT, Res, VPermMask, Zero);
      return DAG.getBitcast(RootVT, Res);
    }

    // If that failed and either input is extracted then try to combine as a
    // shuffle with the larger type.
    if (SDValue WideShuffle = combineX86ShuffleChainWithExtract(
            Inputs, Root, BaseMask, Depth, HasVariableMask, AllowVariableMask,
            DAG, Subtarget))
      return WideShuffle;

    // If we have a dual input lane-crossing shuffle then lower to VPERMV3.
    if (AllowVariableMask && !MaskContainsZeros &&
        ((Subtarget.hasAVX512() &&
          (MaskVT == MVT::v8f64 || MaskVT == MVT::v8i64 ||
           MaskVT == MVT::v16f32 || MaskVT == MVT::v16i32)) ||
         (Subtarget.hasVLX() &&
          (MaskVT == MVT::v4f64 || MaskVT == MVT::v4i64 ||
           MaskVT == MVT::v8f32 || MaskVT == MVT::v8i32)) ||
         (Subtarget.hasBWI() && MaskVT == MVT::v32i16) ||
         (Subtarget.hasBWI() && Subtarget.hasVLX() && MaskVT == MVT::v16i16) ||
         (Subtarget.hasVBMI() && MaskVT == MVT::v64i8) ||
         (Subtarget.hasVBMI() && Subtarget.hasVLX() && MaskVT == MVT::v32i8))) {
      SDValue VPermMask = getConstVector(Mask, IntMaskVT, DAG, DL, true);
      V1 = DAG.getBitcast(MaskVT, V1);
      V2 = DAG.getBitcast(MaskVT, V2);
      Res = DAG.getNode(X86ISD::VPERMV3, DL, MaskVT, V1, VPermMask, V2);
      return DAG.getBitcast(RootVT, Res);
    }
    return SDValue();
  }

  // See if we can combine a single input shuffle with zeros to a bit-mask,
  // which is much simpler than any shuffle.
  if (UnaryShuffle && MaskContainsZeros && AllowVariableMask &&
      isSequentialOrUndefOrZeroInRange(Mask, 0, NumMaskElts, 0) &&
      DAG.getTargetLoweringInfo().isTypeLegal(MaskVT)) {
    APInt Zero = APInt::getNullValue(MaskEltSizeInBits);
    APInt AllOnes = APInt::getAllOnesValue(MaskEltSizeInBits);
    APInt UndefElts(NumMaskElts, 0);
    SmallVector<APInt, 64> EltBits(NumMaskElts, Zero);
    for (unsigned i = 0; i != NumMaskElts; ++i) {
      int M = Mask[i];
      if (M == SM_SentinelUndef) {
        UndefElts.setBit(i);
        continue;
      }
      if (M == SM_SentinelZero)
        continue;
      EltBits[i] = AllOnes;
    }
    SDValue BitMask = getConstVector(EltBits, UndefElts, MaskVT, DAG, DL);
    Res = DAG.getBitcast(MaskVT, V1);
    unsigned AndOpcode =
        FloatDomain ? unsigned(X86ISD::FAND) : unsigned(ISD::AND);
    Res = DAG.getNode(AndOpcode, DL, MaskVT, Res, BitMask);
    return DAG.getBitcast(RootVT, Res);
  }

  // If we have a single input shuffle with different shuffle patterns in the
  // the 128-bit lanes use the variable mask to VPERMILPS.
  // TODO Combine other mask types at higher depths.
  if (UnaryShuffle && AllowVariableMask && !MaskContainsZeros &&
      ((MaskVT == MVT::v8f32 && Subtarget.hasAVX()) ||
       (MaskVT == MVT::v16f32 && Subtarget.hasAVX512()))) {
    SmallVector<SDValue, 16> VPermIdx;
    for (int M : Mask) {
      SDValue Idx =
          M < 0 ? DAG.getUNDEF(MVT::i32) : DAG.getConstant(M % 4, DL, MVT::i32);
      VPermIdx.push_back(Idx);
    }
    SDValue VPermMask = DAG.getBuildVector(IntMaskVT, DL, VPermIdx);
    Res = DAG.getBitcast(MaskVT, V1);
    Res = DAG.getNode(X86ISD::VPERMILPV, DL, MaskVT, Res, VPermMask);
    return DAG.getBitcast(RootVT, Res);
  }

  // With XOP, binary shuffles of 128/256-bit floating point vectors can combine
  // to VPERMIL2PD/VPERMIL2PS.
  if (AllowVariableMask && Subtarget.hasXOP() &&
      (MaskVT == MVT::v2f64 || MaskVT == MVT::v4f64 || MaskVT == MVT::v4f32 ||
       MaskVT == MVT::v8f32)) {
    // VPERMIL2 Operation.
    // Bits[3] - Match Bit.
    // Bits[2:1] - (Per Lane) PD Shuffle Mask.
    // Bits[2:0] - (Per Lane) PS Shuffle Mask.
    unsigned NumLanes = MaskVT.getSizeInBits() / 128;
    unsigned NumEltsPerLane = NumMaskElts / NumLanes;
    SmallVector<int, 8> VPerm2Idx;
    unsigned M2ZImm = 0;
    for (int M : Mask) {
      if (M == SM_SentinelUndef) {
        VPerm2Idx.push_back(-1);
        continue;
      }
      if (M == SM_SentinelZero) {
        M2ZImm = 2;
        VPerm2Idx.push_back(8);
        continue;
      }
      int Index = (M % NumEltsPerLane) + ((M / NumMaskElts) * NumEltsPerLane);
      Index = (MaskVT.getScalarSizeInBits() == 64 ? Index << 1 : Index);
      VPerm2Idx.push_back(Index);
    }
    V1 = DAG.getBitcast(MaskVT, V1);
    V2 = DAG.getBitcast(MaskVT, V2);
    SDValue VPerm2MaskOp = getConstVector(VPerm2Idx, IntMaskVT, DAG, DL, true);
    Res = DAG.getNode(X86ISD::VPERMIL2, DL, MaskVT, V1, V2, VPerm2MaskOp,
                      DAG.getConstant(M2ZImm, DL, MVT::i8));
    return DAG.getBitcast(RootVT, Res);
  }

  // If we have 3 or more shuffle instructions or a chain involving a variable
  // mask, we can replace them with a single PSHUFB instruction profitably.
  // Intel's manuals suggest only using PSHUFB if doing so replacing 5
  // instructions, but in practice PSHUFB tends to be *very* fast so we're
  // more aggressive.
  if (UnaryShuffle && AllowVariableMask &&
      ((RootVT.is128BitVector() && Subtarget.hasSSSE3()) ||
       (RootVT.is256BitVector() && Subtarget.hasAVX2()) ||
       (RootVT.is512BitVector() && Subtarget.hasBWI()))) {
    SmallVector<SDValue, 16> PSHUFBMask;
    int NumBytes = RootVT.getSizeInBits() / 8;
    int Ratio = NumBytes / NumMaskElts;
    for (int i = 0; i < NumBytes; ++i) {
      int M = Mask[i / Ratio];
      if (M == SM_SentinelUndef) {
        PSHUFBMask.push_back(DAG.getUNDEF(MVT::i8));
        continue;
      }
      if (M == SM_SentinelZero) {
        PSHUFBMask.push_back(DAG.getConstant(255, DL, MVT::i8));
        continue;
      }
      M = Ratio * M + i % Ratio;
      assert((M / 16) == (i / 16) && "Lane crossing detected");
      PSHUFBMask.push_back(DAG.getConstant(M, DL, MVT::i8));
    }
    MVT ByteVT = MVT::getVectorVT(MVT::i8, NumBytes);
    Res = DAG.getBitcast(ByteVT, V1);
    SDValue PSHUFBMaskOp = DAG.getBuildVector(ByteVT, DL, PSHUFBMask);
    Res = DAG.getNode(X86ISD::PSHUFB, DL, ByteVT, Res, PSHUFBMaskOp);
    return DAG.getBitcast(RootVT, Res);
  }

  // With XOP, if we have a 128-bit binary input shuffle we can always combine
  // to VPPERM. We match the depth requirement of PSHUFB - VPPERM is never
  // slower than PSHUFB on targets that support both.
  if (AllowVariableMask && RootVT.is128BitVector() && Subtarget.hasXOP()) {
    // VPPERM Mask Operation
    // Bits[4:0] - Byte Index (0 - 31)
    // Bits[7:5] - Permute Operation (0 - Source byte, 4 - ZERO)
    SmallVector<SDValue, 16> VPPERMMask;
    int NumBytes = 16;
    int Ratio = NumBytes / NumMaskElts;
    for (int i = 0; i < NumBytes; ++i) {
      int M = Mask[i / Ratio];
      if (M == SM_SentinelUndef) {
        VPPERMMask.push_back(DAG.getUNDEF(MVT::i8));
        continue;
      }
      if (M == SM_SentinelZero) {
        VPPERMMask.push_back(DAG.getConstant(128, DL, MVT::i8));
        continue;
      }
      M = Ratio * M + i % Ratio;
      VPPERMMask.push_back(DAG.getConstant(M, DL, MVT::i8));
    }
    MVT ByteVT = MVT::v16i8;
    V1 = DAG.getBitcast(ByteVT, V1);
    V2 = DAG.getBitcast(ByteVT, V2);
    SDValue VPPERMMaskOp = DAG.getBuildVector(ByteVT, DL, VPPERMMask);
    Res = DAG.getNode(X86ISD::VPPERM, DL, ByteVT, V1, V2, VPPERMMaskOp);
    return DAG.getBitcast(RootVT, Res);
  }

  // If that failed and either input is extracted then try to combine as a
  // shuffle with the larger type.
  if (SDValue WideShuffle = combineX86ShuffleChainWithExtract(
          Inputs, Root, BaseMask, Depth, HasVariableMask, AllowVariableMask,
          DAG, Subtarget))
    return WideShuffle;

  // If we have a dual input shuffle then lower to VPERMV3.
  if (!UnaryShuffle && AllowVariableMask && !MaskContainsZeros &&
      ((Subtarget.hasAVX512() &&
        (MaskVT == MVT::v8f64 || MaskVT == MVT::v8i64 ||
         MaskVT == MVT::v16f32 || MaskVT == MVT::v16i32)) ||
       (Subtarget.hasVLX() &&
        (MaskVT == MVT::v2f64 || MaskVT == MVT::v2i64 || MaskVT == MVT::v4f64 ||
         MaskVT == MVT::v4i64 || MaskVT == MVT::v4f32 || MaskVT == MVT::v4i32 ||
         MaskVT == MVT::v8f32 || MaskVT == MVT::v8i32)) ||
       (Subtarget.hasBWI() && MaskVT == MVT::v32i16) ||
       (Subtarget.hasBWI() && Subtarget.hasVLX() &&
        (MaskVT == MVT::v8i16 || MaskVT == MVT::v16i16)) ||
       (Subtarget.hasVBMI() && MaskVT == MVT::v64i8) ||
       (Subtarget.hasVBMI() && Subtarget.hasVLX() &&
        (MaskVT == MVT::v16i8 || MaskVT == MVT::v32i8)))) {
    SDValue VPermMask = getConstVector(Mask, IntMaskVT, DAG, DL, true);
    V1 = DAG.getBitcast(MaskVT, V1);
    V2 = DAG.getBitcast(MaskVT, V2);
    Res = DAG.getNode(X86ISD::VPERMV3, DL, MaskVT, V1, VPermMask, V2);
    return DAG.getBitcast(RootVT, Res);
  }

  // Failed to find any combines.
  return SDValue();
}

// Combine an arbitrary chain of shuffles + extract_subvectors into a single
// instruction if possible.
//
// Wrapper for combineX86ShuffleChain that extends the shuffle mask to a larger
// type size to attempt to combine:
// shuffle(extract_subvector(x,c1),extract_subvector(y,c2),m1)
// -->
// extract_subvector(shuffle(x,y,m2),0)
static SDValue combineX86ShuffleChainWithExtract(
    ArrayRef<SDValue> Inputs, SDValue Root, ArrayRef<int> BaseMask, int Depth,
    bool HasVariableMask, bool AllowVariableMask, SelectionDAG &DAG,
    const X86Subtarget &Subtarget) {
  unsigned NumMaskElts = BaseMask.size();
  unsigned NumInputs = Inputs.size();
  if (NumInputs == 0)
    return SDValue();

  SmallVector<SDValue, 4> WideInputs(Inputs.begin(), Inputs.end());
  SmallVector<unsigned, 4> Offsets(NumInputs, 0);

  // Peek through subvectors.
  // TODO: Support inter-mixed EXTRACT_SUBVECTORs + BITCASTs?
  unsigned WideSizeInBits = WideInputs[0].getValueSizeInBits();
  for (unsigned i = 0; i != NumInputs; ++i) {
    SDValue &Src = WideInputs[i];
    unsigned &Offset = Offsets[i];
    Src = peekThroughBitcasts(Src);
    EVT BaseVT = Src.getValueType();
    while (Src.getOpcode() == ISD::EXTRACT_SUBVECTOR &&
           isa<ConstantSDNode>(Src.getOperand(1))) {
      Offset += Src.getConstantOperandVal(1);
      Src = Src.getOperand(0);
    }
    WideSizeInBits = std::max(WideSizeInBits, Src.getValueSizeInBits());
    assert((Offset % BaseVT.getVectorNumElements()) == 0 &&
           "Unexpected subvector extraction");
    Offset /= BaseVT.getVectorNumElements();
    Offset *= NumMaskElts;
  }

  // Bail if we're always extracting from the lowest subvectors,
  // combineX86ShuffleChain should match this for the current width.
  if (llvm::all_of(Offsets, [](unsigned Offset) { return Offset == 0; }))
    return SDValue();

  EVT RootVT = Root.getValueType();
  unsigned RootSizeInBits = RootVT.getSizeInBits();
  unsigned Scale = WideSizeInBits / RootSizeInBits;
  assert((WideSizeInBits % RootSizeInBits) == 0 &&
         "Unexpected subvector extraction");

  // If the src vector types aren't the same, see if we can extend
  // them to match each other.
  // TODO: Support different scalar types?
  EVT WideSVT = WideInputs[0].getValueType().getScalarType();
  if (llvm::any_of(WideInputs, [&WideSVT, &DAG](SDValue Op) {
        return !DAG.getTargetLoweringInfo().isTypeLegal(Op.getValueType()) ||
               Op.getValueType().getScalarType() != WideSVT;
      }))
    return SDValue();

  for (SDValue &NewInput : WideInputs) {
    assert((WideSizeInBits % NewInput.getValueSizeInBits()) == 0 &&
           "Shuffle vector size mismatch");
    if (WideSizeInBits > NewInput.getValueSizeInBits())
      NewInput = widenSubVector(NewInput, false, Subtarget, DAG,
                                SDLoc(NewInput), WideSizeInBits);
    assert(WideSizeInBits == NewInput.getValueSizeInBits() &&
           "Unexpected subvector extraction");
  }

  // Create new mask for larger type.
  for (unsigned i = 1; i != NumInputs; ++i)
    Offsets[i] += i * Scale * NumMaskElts;

  SmallVector<int, 64> WideMask(BaseMask.begin(), BaseMask.end());
  for (int &M : WideMask) {
    if (M < 0)
      continue;
    M = (M % NumMaskElts) + Offsets[M / NumMaskElts];
  }
  WideMask.append((Scale - 1) * NumMaskElts, SM_SentinelUndef);

  // Remove unused/repeated shuffle source ops.
  resolveTargetShuffleInputsAndMask(WideInputs, WideMask);
  assert(!WideInputs.empty() && "Shuffle with no inputs detected");

  if (WideInputs.size() > 2)
    return SDValue();

  // Increase depth for every upper subvector we've peeked through.
  Depth += count_if(Offsets, [](unsigned Offset) { return Offset > 0; });

  // Attempt to combine wider chain.
  // TODO: Can we use a better Root?
  SDValue WideRoot = WideInputs[0];
  if (SDValue WideShuffle = combineX86ShuffleChain(
          WideInputs, WideRoot, WideMask, Depth, HasVariableMask,
          AllowVariableMask, DAG, Subtarget)) {
    WideShuffle =
        extractSubVector(WideShuffle, 0, DAG, SDLoc(Root), RootSizeInBits);
    return DAG.getBitcast(RootVT, WideShuffle);
  }
  return SDValue();
}

// Attempt to constant fold all of the constant source ops.
// Returns true if the entire shuffle is folded to a constant.
// TODO: Extend this to merge multiple constant Ops and update the mask.
static SDValue combineX86ShufflesConstants(ArrayRef<SDValue> Ops,
                                           ArrayRef<int> Mask, SDValue Root,
                                           bool HasVariableMask,
                                           SelectionDAG &DAG,
                                           const X86Subtarget &Subtarget) {
  MVT VT = Root.getSimpleValueType();

  unsigned SizeInBits = VT.getSizeInBits();
  unsigned NumMaskElts = Mask.size();
  unsigned MaskSizeInBits = SizeInBits / NumMaskElts;
  unsigned NumOps = Ops.size();

  // Extract constant bits from each source op.
  bool OneUseConstantOp = false;
  SmallVector<APInt, 16> UndefEltsOps(NumOps);
  SmallVector<SmallVector<APInt, 16>, 16> RawBitsOps(NumOps);
  for (unsigned i = 0; i != NumOps; ++i) {
    SDValue SrcOp = Ops[i];
    OneUseConstantOp |= SrcOp.hasOneUse();
    if (!getTargetConstantBitsFromNode(SrcOp, MaskSizeInBits, UndefEltsOps[i],
                                       RawBitsOps[i]))
      return SDValue();
  }

  // Only fold if at least one of the constants is only used once or
  // the combined shuffle has included a variable mask shuffle, this
  // is to avoid constant pool bloat.
  if (!OneUseConstantOp && !HasVariableMask)
    return SDValue();

  // Shuffle the constant bits according to the mask.
  APInt UndefElts(NumMaskElts, 0);
  APInt ZeroElts(NumMaskElts, 0);
  APInt ConstantElts(NumMaskElts, 0);
  SmallVector<APInt, 8> ConstantBitData(NumMaskElts,
                                        APInt::getNullValue(MaskSizeInBits));
  for (unsigned i = 0; i != NumMaskElts; ++i) {
    int M = Mask[i];
    if (M == SM_SentinelUndef) {
      UndefElts.setBit(i);
      continue;
    } else if (M == SM_SentinelZero) {
      ZeroElts.setBit(i);
      continue;
    }
    assert(0 <= M && M < (int)(NumMaskElts * NumOps));

    unsigned SrcOpIdx = (unsigned)M / NumMaskElts;
    unsigned SrcMaskIdx = (unsigned)M % NumMaskElts;

    auto &SrcUndefElts = UndefEltsOps[SrcOpIdx];
    if (SrcUndefElts[SrcMaskIdx]) {
      UndefElts.setBit(i);
      continue;
    }

    auto &SrcEltBits = RawBitsOps[SrcOpIdx];
    APInt &Bits = SrcEltBits[SrcMaskIdx];
    if (!Bits) {
      ZeroElts.setBit(i);
      continue;
    }

    ConstantElts.setBit(i);
    ConstantBitData[i] = Bits;
  }
  assert((UndefElts | ZeroElts | ConstantElts).isAllOnesValue());

  // Create the constant data.
  MVT MaskSVT;
  if (VT.isFloatingPoint() && (MaskSizeInBits == 32 || MaskSizeInBits == 64))
    MaskSVT = MVT::getFloatingPointVT(MaskSizeInBits);
  else
    MaskSVT = MVT::getIntegerVT(MaskSizeInBits);

  MVT MaskVT = MVT::getVectorVT(MaskSVT, NumMaskElts);

  SDLoc DL(Root);
  SDValue CstOp = getConstVector(ConstantBitData, UndefElts, MaskVT, DAG, DL);
  return DAG.getBitcast(VT, CstOp);
}

/// Fully generic combining of x86 shuffle instructions.
///
/// This should be the last combine run over the x86 shuffle instructions. Once
/// they have been fully optimized, this will recursively consider all chains
/// of single-use shuffle instructions, build a generic model of the cumulative
/// shuffle operation, and check for simpler instructions which implement this
/// operation. We use this primarily for two purposes:
///
/// 1) Collapse generic shuffles to specialized single instructions when
///    equivalent. In most cases, this is just an encoding size win, but
///    sometimes we will collapse multiple generic shuffles into a single
///    special-purpose shuffle.
/// 2) Look for sequences of shuffle instructions with 3 or more total
///    instructions, and replace them with the slightly more expensive SSSE3
///    PSHUFB instruction if available. We do this as the last combining step
///    to ensure we avoid using PSHUFB if we can implement the shuffle with
///    a suitable short sequence of other instructions. The PSHUFB will either
///    use a register or have to read from memory and so is slightly (but only
///    slightly) more expensive than the other shuffle instructions.
///
/// Because this is inherently a quadratic operation (for each shuffle in
/// a chain, we recurse up the chain), the depth is limited to 8 instructions.
/// This should never be an issue in practice as the shuffle lowering doesn't
/// produce sequences of more than 8 instructions.
///
/// FIXME: We will currently miss some cases where the redundant shuffling
/// would simplify under the threshold for PSHUFB formation because of
/// combine-ordering. To fix this, we should do the redundant instruction
/// combining in this recursive walk.
SDValue X86::combineX86ShufflesRecursively(
    ArrayRef<SDValue> SrcOps, int SrcOpIndex, SDValue Root,
    ArrayRef<int> RootMask, ArrayRef<const SDNode *> SrcNodes, unsigned Depth,
    bool HasVariableMask, bool AllowVariableMask, SelectionDAG &DAG,
    const X86Subtarget &Subtarget) {
  // Bound the depth of our recursive combine because this is ultimately
  // quadratic in nature.
  const unsigned MaxRecursionDepth = 8;
  if (Depth >= MaxRecursionDepth)
    return SDValue();

  // Directly rip through bitcasts to find the underlying operand.
  SDValue Op = SrcOps[SrcOpIndex];
  Op = peekThroughOneUseBitcasts(Op);

  MVT VT = Op.getSimpleValueType();
  if (!VT.isVector())
    return SDValue(); // Bail if we hit a non-vector.

  assert(Root.getSimpleValueType().isVector() &&
         "Shuffles operate on vector types!");
  assert(VT.getSizeInBits() == Root.getSimpleValueType().getSizeInBits() &&
         "Can only combine shuffles of the same vector register size.");

  // Extract target shuffle mask and resolve sentinels and inputs.
  SmallVector<int, 64> OpMask;
  SmallVector<SDValue, 2> OpInputs;
  if (!resolveTargetShuffleInputs(Op, OpInputs, OpMask, DAG, Depth))
    return SDValue();

  // Add the inputs to the Ops list, avoiding duplicates.
  SmallVector<SDValue, 16> Ops(SrcOps.begin(), SrcOps.end());

  auto AddOp = [&Ops](SDValue Input, int InsertionPoint) -> int {
    // Attempt to find an existing match.
    SDValue InputBC = peekThroughBitcasts(Input);
    for (int i = 0, e = Ops.size(); i < e; ++i)
      if (InputBC == peekThroughBitcasts(Ops[i]))
        return i;
    // Match failed - should we replace an existing Op?
    if (InsertionPoint >= 0) {
      Ops[InsertionPoint] = Input;
      return InsertionPoint;
    }
    // Add to the end of the Ops list.
    Ops.push_back(Input);
    return Ops.size() - 1;
  };

  SmallVector<int, 2> OpInputIdx;
  for (SDValue OpInput : OpInputs)
    OpInputIdx.push_back(AddOp(OpInput, OpInputIdx.empty() ? SrcOpIndex : -1));

  assert(((RootMask.size() > OpMask.size() &&
           RootMask.size() % OpMask.size() == 0) ||
          (OpMask.size() > RootMask.size() &&
           OpMask.size() % RootMask.size() == 0) ||
          OpMask.size() == RootMask.size()) &&
         "The smaller number of elements must divide the larger.");

  // This function can be performance-critical, so we rely on the power-of-2
  // knowledge that we have about the mask sizes to replace div/rem ops with
  // bit-masks and shifts.
  assert(isPowerOf2_32(RootMask.size()) && "Non-power-of-2 shuffle mask sizes");
  assert(isPowerOf2_32(OpMask.size()) && "Non-power-of-2 shuffle mask sizes");
  unsigned RootMaskSizeLog2 = countTrailingZeros(RootMask.size());
  unsigned OpMaskSizeLog2 = countTrailingZeros(OpMask.size());

  unsigned MaskWidth = std::max<unsigned>(OpMask.size(), RootMask.size());
  unsigned RootRatio = std::max<unsigned>(1, OpMask.size() >> RootMaskSizeLog2);
  unsigned OpRatio = std::max<unsigned>(1, RootMask.size() >> OpMaskSizeLog2);
  assert((RootRatio == 1 || OpRatio == 1) &&
         "Must not have a ratio for both incoming and op masks!");

  assert(isPowerOf2_32(MaskWidth) && "Non-power-of-2 shuffle mask sizes");
  assert(isPowerOf2_32(RootRatio) && "Non-power-of-2 shuffle mask sizes");
  assert(isPowerOf2_32(OpRatio) && "Non-power-of-2 shuffle mask sizes");
  unsigned RootRatioLog2 = countTrailingZeros(RootRatio);
  unsigned OpRatioLog2 = countTrailingZeros(OpRatio);

  SmallVector<int, 64> Mask(MaskWidth, SM_SentinelUndef);

  // Merge this shuffle operation's mask into our accumulated mask. Note that
  // this shuffle's mask will be the first applied to the input, followed by the
  // root mask to get us all the way to the root value arrangement. The reason
  // for this order is that we are recursing up the operation chain.
  for (unsigned i = 0; i < MaskWidth; ++i) {
    unsigned RootIdx = i >> RootRatioLog2;
    if (RootMask[RootIdx] < 0) {
      // This is a zero or undef lane, we're done.
      Mask[i] = RootMask[RootIdx];
      continue;
    }

    unsigned RootMaskedIdx =
        RootRatio == 1
            ? RootMask[RootIdx]
            : (RootMask[RootIdx] << RootRatioLog2) + (i & (RootRatio - 1));

    // Just insert the scaled root mask value if it references an input other
    // than the SrcOp we're currently inserting.
    if ((RootMaskedIdx < (SrcOpIndex * MaskWidth)) ||
        (((SrcOpIndex + 1) * MaskWidth) <= RootMaskedIdx)) {
      Mask[i] = RootMaskedIdx;
      continue;
    }

    RootMaskedIdx = RootMaskedIdx & (MaskWidth - 1);
    unsigned OpIdx = RootMaskedIdx >> OpRatioLog2;
    if (OpMask[OpIdx] < 0) {
      // The incoming lanes are zero or undef, it doesn't matter which ones we
      // are using.
      Mask[i] = OpMask[OpIdx];
      continue;
    }

    // Ok, we have non-zero lanes, map them through to one of the Op's inputs.
    unsigned OpMaskedIdx =
        OpRatio == 1
            ? OpMask[OpIdx]
            : (OpMask[OpIdx] << OpRatioLog2) + (RootMaskedIdx & (OpRatio - 1));

    OpMaskedIdx = OpMaskedIdx & (MaskWidth - 1);
    int InputIdx = OpMask[OpIdx] / (int)OpMask.size();
    assert(0 <= OpInputIdx[InputIdx] && "Unknown target shuffle input");
    OpMaskedIdx += OpInputIdx[InputIdx] * MaskWidth;

    Mask[i] = OpMaskedIdx;
  }

  // Handle the all undef/zero cases early.
  if (all_of(Mask, [](int Idx) { return Idx == SM_SentinelUndef; }))
    return DAG.getUNDEF(Root.getValueType());

  // TODO - should we handle the mixed zero/undef case as well? Just returning
  // a zero mask will lose information on undef elements possibly reducing
  // future combine possibilities.
  if (all_of(Mask, [](int Idx) { return Idx < 0; }))
    return getZeroVector(Root.getSimpleValueType(), Subtarget, DAG,
                         SDLoc(Root));

  // Remove unused/repeated shuffle source ops.
  resolveTargetShuffleInputsAndMask(Ops, Mask);
  assert(!Ops.empty() && "Shuffle with no inputs detected");

  HasVariableMask |= isTargetShuffleVariableMask(Op.getOpcode());

  // Update the list of shuffle nodes that have been combined so far.
  SmallVector<const SDNode *, 16> CombinedNodes(SrcNodes.begin(),
                                                SrcNodes.end());
  CombinedNodes.push_back(Op.getNode());

  // See if we can recurse into each shuffle source op (if it's a target
  // shuffle). The source op should only be generally combined if it either has
  // a single use (i.e. current Op) or all its users have already been combined,
  // if not then we can still combine but should prevent generation of variable
  // shuffles to avoid constant pool bloat.
  // Don't recurse if we already have more source ops than we can combine in
  // the remaining recursion depth.
  if (Ops.size() < (MaxRecursionDepth - Depth)) {
    for (int i = 0, e = Ops.size(); i < e; ++i) {
      bool AllowVar = false;
      if (Ops[i].getNode()->hasOneUse() ||
          SDNode::areOnlyUsersOf(CombinedNodes, Ops[i].getNode()))
        AllowVar = AllowVariableMask;
      if (SDValue Res = combineX86ShufflesRecursively(
              Ops, i, Root, Mask, CombinedNodes, Depth + 1, HasVariableMask,
              AllowVar, DAG, Subtarget))
        return Res;
    }
  }

  // Attempt to constant fold all of the constant source ops.
  if (SDValue Cst = combineX86ShufflesConstants(
          Ops, Mask, Root, HasVariableMask, DAG, Subtarget))
    return Cst;

  // We can only combine unary and binary shuffle mask cases.
  if (Ops.size() <= 2) {
    // Minor canonicalization of the accumulated shuffle mask to make it easier
    // to match below. All this does is detect masks with sequential pairs of
    // elements, and shrink them to the half-width mask. It does this in a loop
    // so it will reduce the size of the mask to the minimal width mask which
    // performs an equivalent shuffle.
    SmallVector<int, 64> WidenedMask;
    while (Mask.size() > 1 && X86::canWidenShuffleElements(Mask, WidenedMask)) {
      Mask = std::move(WidenedMask);
    }

    // Canonicalization of binary shuffle masks to improve pattern matching by
    // commuting the inputs.
    if (Ops.size() == 2 && canonicalizeShuffleMaskWithCommute(Mask)) {
      ShuffleVectorSDNode::commuteMask(Mask);
      std::swap(Ops[0], Ops[1]);
    }

    // Finally, try to combine into a single shuffle instruction.
    return combineX86ShuffleChain(Ops, Root, Mask, Depth, HasVariableMask,
                                  AllowVariableMask, DAG, Subtarget);
  }

  // If that failed and any input is extracted then try to combine as a
  // shuffle with the larger type.
  return combineX86ShuffleChainWithExtract(Ops, Root, Mask, Depth,
                                           HasVariableMask, AllowVariableMask,
                                           DAG, Subtarget);
}

/// Helper entry wrapper to combineX86ShufflesRecursively.
SDValue X86::combineX86ShufflesRecursively(SDValue Op, SelectionDAG &DAG,
                                           const X86Subtarget &Subtarget) {
  return combineX86ShufflesRecursively({Op}, 0, Op, {0}, {}, /*Depth*/ 0,
                                       /*HasVarMask*/ false,
                                       /*AllowVarMask*/ true, DAG, Subtarget);
}

/// Get the PSHUF-style mask from PSHUF node.
///
/// This is a very minor wrapper around getTargetShuffleMask to easy forming v4
/// PSHUF-style masks that can be reused with such instructions.
static SmallVector<int, 4> getPSHUFShuffleMask(SDValue N) {
  MVT VT = N.getSimpleValueType();
  SmallVector<int, 4> Mask;
  SmallVector<SDValue, 2> Ops;
  bool IsUnary;
  bool HaveMask =
      getTargetShuffleMask(N.getNode(), VT, false, Ops, Mask, IsUnary);
  (void)HaveMask;
  assert(HaveMask);

  // If we have more than 128-bits, only the low 128-bits of shuffle mask
  // matter. Check that the upper masks are repeats and remove them.
  if (VT.getSizeInBits() > 128) {
    int LaneElts = 128 / VT.getScalarSizeInBits();
#ifndef NDEBUG
    for (int i = 1, NumLanes = VT.getSizeInBits() / 128; i < NumLanes; ++i)
      for (int j = 0; j < LaneElts; ++j)
        assert(Mask[j] == Mask[i * LaneElts + j] - (LaneElts * i) &&
               "Mask doesn't repeat in high 128-bit lanes!");
#endif
    Mask.resize(LaneElts);
  }

  switch (N.getOpcode()) {
  case X86ISD::PSHUFD:
    return Mask;
  case X86ISD::PSHUFLW:
    Mask.resize(4);
    return Mask;
  case X86ISD::PSHUFHW:
    Mask.erase(Mask.begin(), Mask.begin() + 4);
    for (int &M : Mask)
      M -= 4;
    return Mask;
  default:
    llvm_unreachable("No valid shuffle instruction found!");
  }
}

/// Search for a combinable shuffle across a chain ending in pshufd.
///
/// We walk up the chain and look for a combinable shuffle, skipping over
/// shuffles that we could hoist this shuffle's transformation past without
/// altering anything.
static SDValue
combineRedundantDWordShuffle(SDValue N, MutableArrayRef<int> Mask,
                             SelectionDAG &DAG) {
  assert(N.getOpcode() == X86ISD::PSHUFD &&
         "Called with something other than an x86 128-bit half shuffle!");
  SDLoc DL(N);

  // Walk up a single-use chain looking for a combinable shuffle. Keep a stack
  // of the shuffles in the chain so that we can form a fresh chain to replace
  // this one.
  SmallVector<SDValue, 8> Chain;
  SDValue V = N.getOperand(0);
  for (; V.hasOneUse(); V = V.getOperand(0)) {
    switch (V.getOpcode()) {
    default:
      return SDValue(); // Nothing combined!

    case ISD::BITCAST:
      // Skip bitcasts as we always know the type for the target specific
      // instructions.
      continue;

    case X86ISD::PSHUFD:
      // Found another dword shuffle.
      break;

    case X86ISD::PSHUFLW:
      // Check that the low words (being shuffled) are the identity in the
      // dword shuffle, and the high words are self-contained.
      if (Mask[0] != 0 || Mask[1] != 1 ||
          !(Mask[2] >= 2 && Mask[2] < 4 && Mask[3] >= 2 && Mask[3] < 4))
        return SDValue();

      Chain.push_back(V);
      continue;

    case X86ISD::PSHUFHW:
      // Check that the high words (being shuffled) are the identity in the
      // dword shuffle, and the low words are self-contained.
      if (Mask[2] != 2 || Mask[3] != 3 ||
          !(Mask[0] >= 0 && Mask[0] < 2 && Mask[1] >= 0 && Mask[1] < 2))
        return SDValue();

      Chain.push_back(V);
      continue;

    case X86ISD::UNPCKL:
    case X86ISD::UNPCKH:
      // For either i8 -> i16 or i16 -> i32 unpacks, we can combine a dword
      // shuffle into a preceding word shuffle.
      if (V.getSimpleValueType().getVectorElementType() != MVT::i8 &&
          V.getSimpleValueType().getVectorElementType() != MVT::i16)
        return SDValue();

      // Search for a half-shuffle which we can combine with.
      unsigned CombineOp =
          V.getOpcode() == X86ISD::UNPCKL ? X86ISD::PSHUFLW : X86ISD::PSHUFHW;
      if (V.getOperand(0) != V.getOperand(1) ||
          !V->isOnlyUserOf(V.getOperand(0).getNode()))
        return SDValue();
      Chain.push_back(V);
      V = V.getOperand(0);
      do {
        switch (V.getOpcode()) {
        default:
          return SDValue(); // Nothing to combine.

        case X86ISD::PSHUFLW:
        case X86ISD::PSHUFHW:
          if (V.getOpcode() == CombineOp)
            break;

          Chain.push_back(V);

          LLVM_FALLTHROUGH;
        case ISD::BITCAST:
          V = V.getOperand(0);
          continue;
        }
        break;
      } while (V.hasOneUse());
      break;
    }
    // Break out of the loop if we break out of the switch.
    break;
  }

  if (!V.hasOneUse())
    // We fell out of the loop without finding a viable combining instruction.
    return SDValue();

  // Merge this node's mask and our incoming mask.
  SmallVector<int, 4> VMask = getPSHUFShuffleMask(V);
  for (int &M : Mask)
    M = VMask[M];
  V = DAG.getNode(V.getOpcode(), DL, V.getValueType(), V.getOperand(0),
                  getV4X86ShuffleImm8ForMask(Mask, DL, DAG));

  // Rebuild the chain around this new shuffle.
  while (!Chain.empty()) {
    SDValue W = Chain.pop_back_val();

    if (V.getValueType() != W.getOperand(0).getValueType())
      V = DAG.getBitcast(W.getOperand(0).getValueType(), V);

    switch (W.getOpcode()) {
    default:
      llvm_unreachable("Only PSHUF and UNPCK instructions get here!");

    case X86ISD::UNPCKL:
    case X86ISD::UNPCKH:
      V = DAG.getNode(W.getOpcode(), DL, W.getValueType(), V, V);
      break;

    case X86ISD::PSHUFD:
    case X86ISD::PSHUFLW:
    case X86ISD::PSHUFHW:
      V = DAG.getNode(W.getOpcode(), DL, W.getValueType(), V, W.getOperand(1));
      break;
    }
  }
  if (V.getValueType() != N.getValueType())
    V = DAG.getBitcast(N.getValueType(), V);

  // Return the new chain to replace N.
  return V;
}

/// Try to combine x86 target specific shuffles.
static SDValue combineTargetShuffle(SDValue N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const X86Subtarget &Subtarget) {
  SDLoc DL(N);
  MVT VT = N.getSimpleValueType();
  SmallVector<int, 4> Mask;
  unsigned Opcode = N.getOpcode();

  // Combine binary shuffle of 2 similar 'Horizontal' instructions into a
  // single instruction.
  if (VT.getScalarSizeInBits() == 64 &&
      (Opcode == X86ISD::MOVSD || Opcode == X86ISD::UNPCKH ||
       Opcode == X86ISD::UNPCKL)) {
    auto BC0 = peekThroughBitcasts(N.getOperand(0));
    auto BC1 = peekThroughBitcasts(N.getOperand(1));
    EVT VT0 = BC0.getValueType();
    EVT VT1 = BC1.getValueType();
    unsigned Opcode0 = BC0.getOpcode();
    unsigned Opcode1 = BC1.getOpcode();
    if (Opcode0 == Opcode1 && VT0 == VT1 &&
        (Opcode0 == X86ISD::FHADD || Opcode0 == X86ISD::HADD ||
         Opcode0 == X86ISD::FHSUB || Opcode0 == X86ISD::HSUB ||
         Opcode0 == X86ISD::PACKSS || Opcode0 == X86ISD::PACKUS)) {
      SDValue Lo, Hi;
      if (Opcode == X86ISD::MOVSD) {
        Lo = BC1.getOperand(0);
        Hi = BC0.getOperand(1);
      } else {
        Lo = BC0.getOperand(Opcode == X86ISD::UNPCKH ? 1 : 0);
        Hi = BC1.getOperand(Opcode == X86ISD::UNPCKH ? 1 : 0);
      }
      SDValue Horiz = DAG.getNode(Opcode0, DL, VT0, Lo, Hi);
      return DAG.getBitcast(VT, Horiz);
    }
  }

  switch (Opcode) {
  case X86ISD::VBROADCAST: {
    SDValue Src = N.getOperand(0);
    SDValue BC = peekThroughBitcasts(Src);
    EVT SrcVT = Src.getValueType();
    EVT BCVT = BC.getValueType();

    // If broadcasting from another shuffle, attempt to simplify it.
    // TODO - we really need a general SimplifyDemandedVectorElts mechanism.
    if (isTargetShuffle(BC.getOpcode()) &&
        VT.getScalarSizeInBits() % BCVT.getScalarSizeInBits() == 0) {
      unsigned Scale = VT.getScalarSizeInBits() / BCVT.getScalarSizeInBits();
      SmallVector<int, 16> DemandedMask(BCVT.getVectorNumElements(),
                                        SM_SentinelUndef);
      for (unsigned i = 0; i != Scale; ++i)
        DemandedMask[i] = i;
      if (SDValue Res = X86::combineX86ShufflesRecursively(
              {BC}, 0, BC, DemandedMask, {}, /*Depth*/ 0,
              /*HasVarMask*/ false, /*AllowVarMask*/ true, DAG, Subtarget))
        return DAG.getNode(X86ISD::VBROADCAST, DL, VT,
                           DAG.getBitcast(SrcVT, Res));
    }

    // broadcast(bitcast(src)) -> bitcast(broadcast(src))
    // 32-bit targets have to bitcast i64 to f64, so better to bitcast upward.
    if (Src.getOpcode() == ISD::BITCAST &&
        SrcVT.getScalarSizeInBits() == BCVT.getScalarSizeInBits()) {
      EVT NewVT = EVT::getVectorVT(*DAG.getContext(), BCVT.getScalarType(),
                                   VT.getVectorNumElements());
      return DAG.getBitcast(VT, DAG.getNode(X86ISD::VBROADCAST, DL, NewVT, BC));
    }

    // Reduce broadcast source vector to lowest 128-bits.
    if (SrcVT.getSizeInBits() > 128)
      return DAG.getNode(X86ISD::VBROADCAST, DL, VT,
                         extract128BitVector(Src, 0, DAG, DL));

    // broadcast(scalar_to_vector(x)) -> broadcast(x).
    if (Src.getOpcode() == ISD::SCALAR_TO_VECTOR)
      return DAG.getNode(X86ISD::VBROADCAST, DL, VT, Src.getOperand(0));

    // Share broadcast with the longest vector and extract low subvector (free).
    for (SDNode *User : Src->uses())
      if (User != N.getNode() && User->getOpcode() == X86ISD::VBROADCAST &&
          User->getValueSizeInBits(0) > VT.getSizeInBits()) {
        return extractSubVector(SDValue(User, 0), 0, DAG, DL,
                                VT.getSizeInBits());
      }

    return SDValue();
  }
  case X86ISD::BLENDI: {
    SDValue N0 = N.getOperand(0);
    SDValue N1 = N.getOperand(1);

    // blend(bitcast(x),bitcast(y)) -> bitcast(blend(x,y)) to narrower types.
    // TODO: Handle MVT::v16i16 repeated blend mask.
    if (N0.getOpcode() == ISD::BITCAST && N1.getOpcode() == ISD::BITCAST &&
        N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()) {
      MVT SrcVT = N0.getOperand(0).getSimpleValueType();
      if ((VT.getScalarSizeInBits() % SrcVT.getScalarSizeInBits()) == 0 &&
          SrcVT.getScalarSizeInBits() >= 32) {
        unsigned BlendMask = N.getConstantOperandVal(2);
        unsigned Size = VT.getVectorNumElements();
        unsigned Scale = VT.getScalarSizeInBits() / SrcVT.getScalarSizeInBits();
        BlendMask = scaleVectorShuffleBlendMask(BlendMask, Size, Scale);
        return DAG.getBitcast(
            VT, DAG.getNode(X86ISD::BLENDI, DL, SrcVT, N0.getOperand(0),
                            N1.getOperand(0),
                            DAG.getConstant(BlendMask, DL, MVT::i8)));
      }
    }
    return SDValue();
  }
  case X86ISD::VPERMI: {
    // vpermi(bitcast(x)) -> bitcast(vpermi(x)) for same number of elements.
    // TODO: Remove when we have preferred domains in combineX86ShuffleChain.
    SDValue N0 = N.getOperand(0);
    SDValue N1 = N.getOperand(1);
    unsigned EltSizeInBits = VT.getScalarSizeInBits();
    if (N0.getOpcode() == ISD::BITCAST &&
        N0.getOperand(0).getScalarValueSizeInBits() == EltSizeInBits) {
      SDValue Src = N0.getOperand(0);
      EVT SrcVT = Src.getValueType();
      SDValue Res = DAG.getNode(X86ISD::VPERMI, DL, SrcVT, Src, N1);
      return DAG.getBitcast(VT, Res);
    }
    return SDValue();
  }
  case X86ISD::PSHUFD:
  case X86ISD::PSHUFLW:
  case X86ISD::PSHUFHW:
    Mask = getPSHUFShuffleMask(N);
    assert(Mask.size() == 4);
    break;
  case X86ISD::MOVSD:
  case X86ISD::MOVSS: {
    SDValue N0 = N.getOperand(0);
    SDValue N1 = N.getOperand(1);

    // Canonicalize scalar FPOps:
    // MOVS*(N0, OP(N0, N1)) --> MOVS*(N0, SCALAR_TO_VECTOR(OP(N0[0], N1[0])))
    // If commutable, allow OP(N1[0], N0[0]).
    unsigned Opcode1 = N1.getOpcode();
    if (Opcode1 == ISD::FADD || Opcode1 == ISD::FMUL || Opcode1 == ISD::FSUB ||
        Opcode1 == ISD::FDIV) {
      SDValue N10 = N1.getOperand(0);
      SDValue N11 = N1.getOperand(1);
      if (N10 == N0 ||
          (N11 == N0 && (Opcode1 == ISD::FADD || Opcode1 == ISD::FMUL))) {
        if (N10 != N0)
          std::swap(N10, N11);
        MVT SVT = VT.getVectorElementType();
        SDValue ZeroIdx = DAG.getIntPtrConstant(0, DL);
        N10 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, SVT, N10, ZeroIdx);
        N11 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, SVT, N11, ZeroIdx);
        SDValue Scl = DAG.getNode(Opcode1, DL, SVT, N10, N11);
        SDValue SclVec = DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, VT, Scl);
        return DAG.getNode(Opcode, DL, VT, N0, SclVec);
      }
    }

    return SDValue();
  }
  case X86ISD::INSERTPS: {
    assert(VT == MVT::v4f32 && "INSERTPS ValueType must be MVT::v4f32");
    SDValue Op0 = N.getOperand(0);
    SDValue Op1 = N.getOperand(1);
    SDValue Op2 = N.getOperand(2);
    unsigned InsertPSMask = cast<ConstantSDNode>(Op2)->getZExtValue();
    unsigned SrcIdx = (InsertPSMask >> 6) & 0x3;
    unsigned DstIdx = (InsertPSMask >> 4) & 0x3;
    unsigned ZeroMask = InsertPSMask & 0xF;

    // If we zero out all elements from Op0 then we don't need to reference it.
    if (((ZeroMask | (1u << DstIdx)) == 0xF) && !Op0.isUndef())
      return DAG.getNode(X86ISD::INSERTPS, DL, VT, DAG.getUNDEF(VT), Op1,
                         DAG.getConstant(InsertPSMask, DL, MVT::i8));

    // If we zero out the element from Op1 then we don't need to reference it.
    if ((ZeroMask & (1u << DstIdx)) && !Op1.isUndef())
      return DAG.getNode(X86ISD::INSERTPS, DL, VT, Op0, DAG.getUNDEF(VT),
                         DAG.getConstant(InsertPSMask, DL, MVT::i8));

    // Attempt to merge insertps Op1 with an inner target shuffle node.
    SmallVector<int, 8> TargetMask1;
    SmallVector<SDValue, 2> Ops1;
    if (setTargetShuffleZeroElements(Op1, TargetMask1, Ops1)) {
      int M = TargetMask1[SrcIdx];
      if (isUndefOrZero(M)) {
        // Zero/UNDEF insertion - zero out element and remove dependency.
        InsertPSMask |= (1u << DstIdx);
        return DAG.getNode(X86ISD::INSERTPS, DL, VT, Op0, DAG.getUNDEF(VT),
                           DAG.getConstant(InsertPSMask, DL, MVT::i8));
      }
      // Update insertps mask srcidx and reference the source input directly.
      assert(0 <= M && M < 8 && "Shuffle index out of range");
      InsertPSMask = (InsertPSMask & 0x3f) | ((M & 0x3) << 6);
      Op1 = Ops1[M < 4 ? 0 : 1];
      return DAG.getNode(X86ISD::INSERTPS, DL, VT, Op0, Op1,
                         DAG.getConstant(InsertPSMask, DL, MVT::i8));
    }

    // Attempt to merge insertps Op0 with an inner target shuffle node.
    SmallVector<int, 8> TargetMask0;
    SmallVector<SDValue, 2> Ops0;
    if (setTargetShuffleZeroElements(Op0, TargetMask0, Ops0)) {
      bool Updated = false;
      bool UseInput00 = false;
      bool UseInput01 = false;
      for (int i = 0; i != 4; ++i) {
        int M = TargetMask0[i];
        if ((InsertPSMask & (1u << i)) || (i == (int)DstIdx)) {
          // No change if element is already zero or the inserted element.
          continue;
        } else if (isUndefOrZero(M)) {
          // If the target mask is undef/zero then we must zero the element.
          InsertPSMask |= (1u << i);
          Updated = true;
          continue;
        }

        // The input vector element must be inline.
        if (M != i && M != (i + 4))
          return SDValue();

        // Determine which inputs of the target shuffle we're using.
        UseInput00 |= (0 <= M && M < 4);
        UseInput01 |= (4 <= M);
      }

      // If we're not using both inputs of the target shuffle then use the
      // referenced input directly.
      if (UseInput00 && !UseInput01) {
        Updated = true;
        Op0 = Ops0[0];
      } else if (!UseInput00 && UseInput01) {
        Updated = true;
        Op0 = Ops0[1];
      }

      if (Updated)
        return DAG.getNode(X86ISD::INSERTPS, DL, VT, Op0, Op1,
                           DAG.getConstant(InsertPSMask, DL, MVT::i8));
    }

    // If we're inserting an element from a vbroadcast of a load, fold the
    // load into the X86insertps instruction. We need to convert the scalar
    // load to a vector and clear the source lane of the INSERTPS control.
    if (Op1.getOpcode() == X86ISD::VBROADCAST && Op1.hasOneUse() &&
        Op1.getOperand(0).hasOneUse() &&
        !Op1.getOperand(0).getValueType().isVector() &&
        ISD::isNormalLoad(Op1.getOperand(0).getNode()))
      return DAG.getNode(X86ISD::INSERTPS, DL, VT, Op0,
                         DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, VT,
                                     Op1.getOperand(0)),
                         DAG.getConstant(InsertPSMask & 0x3f, DL, MVT::i8));

    return SDValue();
  }
  default:
    return SDValue();
  }

  // Nuke no-op shuffles that show up after combining.
  if (isNoopShuffleMask(Mask))
    return N.getOperand(0);

  // Look for simplifications involving one or two shuffle instructions.
  SDValue V = N.getOperand(0);
  switch (N.getOpcode()) {
  default:
    break;
  case X86ISD::PSHUFLW:
  case X86ISD::PSHUFHW:
    assert(VT.getVectorElementType() == MVT::i16 && "Bad word shuffle type!");

    // See if this reduces to a PSHUFD which is no more expensive and can
    // combine with more operations. Note that it has to at least flip the
    // dwords as otherwise it would have been removed as a no-op.
    if (makeArrayRef(Mask).equals({2, 3, 0, 1})) {
      int DMask[] = {0, 1, 2, 3};
      int DOffset = N.getOpcode() == X86ISD::PSHUFLW ? 0 : 2;
      DMask[DOffset + 0] = DOffset + 1;
      DMask[DOffset + 1] = DOffset + 0;
      MVT DVT = MVT::getVectorVT(MVT::i32, VT.getVectorNumElements() / 2);
      V = DAG.getBitcast(DVT, V);
      V = DAG.getNode(X86ISD::PSHUFD, DL, DVT, V,
                      getV4X86ShuffleImm8ForMask(DMask, DL, DAG));
      return DAG.getBitcast(VT, V);
    }

    // Look for shuffle patterns which can be implemented as a single unpack.
    // FIXME: This doesn't handle the location of the PSHUFD generically, and
    // only works when we have a PSHUFD followed by two half-shuffles.
    if (Mask[0] == Mask[1] && Mask[2] == Mask[3] &&
        (V.getOpcode() == X86ISD::PSHUFLW ||
         V.getOpcode() == X86ISD::PSHUFHW) &&
        V.getOpcode() != N.getOpcode() &&
        V.hasOneUse()) {
      SDValue D = peekThroughOneUseBitcasts(V.getOperand(0));
      if (D.getOpcode() == X86ISD::PSHUFD && D.hasOneUse()) {
        SmallVector<int, 4> VMask = getPSHUFShuffleMask(V);
        SmallVector<int, 4> DMask = getPSHUFShuffleMask(D);
        int NOffset = N.getOpcode() == X86ISD::PSHUFLW ? 0 : 4;
        int VOffset = V.getOpcode() == X86ISD::PSHUFLW ? 0 : 4;
        int WordMask[8];
        for (int i = 0; i < 4; ++i) {
          WordMask[i + NOffset] = Mask[i] + NOffset;
          WordMask[i + VOffset] = VMask[i] + VOffset;
        }
        // Map the word mask through the DWord mask.
        int MappedMask[8];
        for (int i = 0; i < 8; ++i)
          MappedMask[i] = 2 * DMask[WordMask[i] / 2] + WordMask[i] % 2;
        if (makeArrayRef(MappedMask).equals({0, 0, 1, 1, 2, 2, 3, 3}) ||
            makeArrayRef(MappedMask).equals({4, 4, 5, 5, 6, 6, 7, 7})) {
          // We can replace all three shuffles with an unpack.
          V = DAG.getBitcast(VT, D.getOperand(0));
          return DAG.getNode(MappedMask[0] == 0 ? X86ISD::UNPCKL
                                                : X86ISD::UNPCKH,
                             DL, VT, V, V);
        }
      }
    }

    break;

  case X86ISD::PSHUFD:
    if (SDValue NewN = combineRedundantDWordShuffle(N, Mask, DAG))
      return NewN;

    break;
  }

  return SDValue();
}

/// Checks if the shuffle mask takes subsequent elements
/// alternately from two vectors.
/// For example <0, 5, 2, 7> or <8, 1, 10, 3, 12, 5, 14, 7> are both correct.
static bool isAddSubOrSubAddMask(ArrayRef<int> Mask, bool &Op0Even) {

  int ParitySrc[2] = {-1, -1};
  unsigned Size = Mask.size();
  for (unsigned i = 0; i != Size; ++i) {
    int M = Mask[i];
    if (M < 0)
      continue;

    // Make sure we are using the matching element from the input.
    if ((M % Size) != i)
      return false;

    // Make sure we use the same input for all elements of the same parity.
    int Src = M / Size;
    if (ParitySrc[i % 2] >= 0 && ParitySrc[i % 2] != Src)
      return false;
    ParitySrc[i % 2] = Src;
  }

  // Make sure each input is used.
  if (ParitySrc[0] < 0 || ParitySrc[1] < 0 || ParitySrc[0] == ParitySrc[1])
    return false;

  Op0Even = ParitySrc[0] == 0;
  return true;
}

/// Returns true iff the shuffle node \p N can be replaced with ADDSUB(SUBADD)
/// operation. If true is returned then the operands of ADDSUB(SUBADD) operation
/// are written to the parameters \p Opnd0 and \p Opnd1.
///
/// We combine shuffle to ADDSUB(SUBADD) directly on the abstract vector shuffle nodes
/// so it is easier to generically match. We also insert dummy vector shuffle
/// nodes for the operands which explicitly discard the lanes which are unused
/// by this operation to try to flow through the rest of the combiner
/// the fact that they're unused.
static bool isAddSubOrSubAdd(SDNode *N, const X86Subtarget &Subtarget,
                             SelectionDAG &DAG, SDValue &Opnd0, SDValue &Opnd1,
                             bool &IsSubAdd) {

  EVT VT = N->getValueType(0);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!Subtarget.hasSSE3() || !TLI.isTypeLegal(VT) ||
      !VT.getSimpleVT().isFloatingPoint())
    return false;

  // We only handle target-independent shuffles.
  // FIXME: It would be easy and harmless to use the target shuffle mask
  // extraction tool to support more.
  if (N->getOpcode() != ISD::VECTOR_SHUFFLE)
    return false;

  SDValue V1 = N->getOperand(0);
  SDValue V2 = N->getOperand(1);

  // Make sure we have an FADD and an FSUB.
  if ((V1.getOpcode() != ISD::FADD && V1.getOpcode() != ISD::FSUB) ||
      (V2.getOpcode() != ISD::FADD && V2.getOpcode() != ISD::FSUB) ||
      V1.getOpcode() == V2.getOpcode())
    return false;

  // If there are other uses of these operations we can't fold them.
  if (!V1->hasOneUse() || !V2->hasOneUse())
    return false;

  // Ensure that both operations have the same operands. Note that we can
  // commute the FADD operands.
  SDValue LHS, RHS;
  if (V1.getOpcode() == ISD::FSUB) {
    LHS = V1->getOperand(0); RHS = V1->getOperand(1);
    if ((V2->getOperand(0) != LHS || V2->getOperand(1) != RHS) &&
        (V2->getOperand(0) != RHS || V2->getOperand(1) != LHS))
      return false;
  } else {
    assert(V2.getOpcode() == ISD::FSUB && "Unexpected opcode");
    LHS = V2->getOperand(0); RHS = V2->getOperand(1);
    if ((V1->getOperand(0) != LHS || V1->getOperand(1) != RHS) &&
        (V1->getOperand(0) != RHS || V1->getOperand(1) != LHS))
      return false;
  }

  ArrayRef<int> Mask = cast<ShuffleVectorSDNode>(N)->getMask();
  bool Op0Even;
  if (!isAddSubOrSubAddMask(Mask, Op0Even))
    return false;

  // It's a subadd if the vector in the even parity is an FADD.
  IsSubAdd = Op0Even ? V1->getOpcode() == ISD::FADD
                     : V2->getOpcode() == ISD::FADD;

  Opnd0 = LHS;
  Opnd1 = RHS;
  return true;
}

/// Combine shuffle of two fma nodes into FMAddSub or FMSubAdd.
static SDValue combineShuffleToFMAddSub(SDNode *N,
                                        const X86Subtarget &Subtarget,
                                        SelectionDAG &DAG) {
  // We only handle target-independent shuffles.
  // FIXME: It would be easy and harmless to use the target shuffle mask
  // extraction tool to support more.
  if (N->getOpcode() != ISD::VECTOR_SHUFFLE)
    return SDValue();

  MVT VT = N->getSimpleValueType(0);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!Subtarget.hasAnyFMA() || !TLI.isTypeLegal(VT))
    return SDValue();

  // We're trying to match (shuffle fma(a, b, c), X86Fmsub(a, b, c).
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  SDValue FMAdd = Op0, FMSub = Op1;
  if (FMSub.getOpcode() != X86ISD::FMSUB)
    std::swap(FMAdd, FMSub);

  if (FMAdd.getOpcode() != ISD::FMA || FMSub.getOpcode() != X86ISD::FMSUB ||
      FMAdd.getOperand(0) != FMSub.getOperand(0) || !FMAdd.hasOneUse() ||
      FMAdd.getOperand(1) != FMSub.getOperand(1) || !FMSub.hasOneUse() ||
      FMAdd.getOperand(2) != FMSub.getOperand(2))
    return SDValue();

  // Check for correct shuffle mask.
  ArrayRef<int> Mask = cast<ShuffleVectorSDNode>(N)->getMask();
  bool Op0Even;
  if (!isAddSubOrSubAddMask(Mask, Op0Even))
    return SDValue();

  // FMAddSub takes zeroth operand from FMSub node.
  SDLoc DL(N);
  bool IsSubAdd = Op0Even ? Op0 == FMAdd : Op1 == FMAdd;
  unsigned Opcode = IsSubAdd ? X86ISD::FMSUBADD : X86ISD::FMADDSUB;
  return DAG.getNode(Opcode, DL, VT, FMAdd.getOperand(0), FMAdd.getOperand(1),
                     FMAdd.getOperand(2));
}

/// Try to combine a shuffle into a target-specific add-sub or
/// mul-add-sub node.
static SDValue combineShuffleToAddSubOrFMAddSub(SDNode *N,
                                                const X86Subtarget &Subtarget,
                                                SelectionDAG &DAG) {
  if (SDValue V = combineShuffleToFMAddSub(N, Subtarget, DAG))
    return V;

  SDValue Opnd0, Opnd1;
  bool IsSubAdd;
  if (!isAddSubOrSubAdd(N, Subtarget, DAG, Opnd0, Opnd1, IsSubAdd))
    return SDValue();

  MVT VT = N->getSimpleValueType(0);
  SDLoc DL(N);

  // Try to generate X86ISD::FMADDSUB node here.
  SDValue Opnd2;
  if (isFMAddSubOrFMSubAdd(Subtarget, DAG, Opnd0, Opnd1, Opnd2, 2)) {
    unsigned Opc = IsSubAdd ? X86ISD::FMSUBADD : X86ISD::FMADDSUB;
    return DAG.getNode(Opc, DL, VT, Opnd0, Opnd1, Opnd2);
  }

  if (IsSubAdd)
    return SDValue();

  // Do not generate X86ISD::ADDSUB node for 512-bit types even though
  // the ADDSUB idiom has been successfully recognized. There are no known
  // X86 targets with 512-bit ADDSUB instructions!
  if (VT.is512BitVector())
    return SDValue();

  return DAG.getNode(X86ISD::ADDSUB, DL, VT, Opnd0, Opnd1);
}

// We are looking for a shuffle where both sources are concatenated with undef
// and have a width that is half of the output's width. AVX2 has VPERMD/Q, so
// if we can express this as a single-source shuffle, that's preferable.
static SDValue combineShuffleOfConcatUndef(SDNode *N, SelectionDAG &DAG,
                                           const X86Subtarget &Subtarget) {
  if (!Subtarget.hasAVX2() || !isa<ShuffleVectorSDNode>(N))
    return SDValue();

  EVT VT = N->getValueType(0);

  // We only care about shuffles of 128/256-bit vectors of 32/64-bit values.
  if (!VT.is128BitVector() && !VT.is256BitVector())
    return SDValue();

  if (VT.getVectorElementType() != MVT::i32 &&
      VT.getVectorElementType() != MVT::i64 &&
      VT.getVectorElementType() != MVT::f32 &&
      VT.getVectorElementType() != MVT::f64)
    return SDValue();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // Check that both sources are concats with undef.
  if (N0.getOpcode() != ISD::CONCAT_VECTORS ||
      N1.getOpcode() != ISD::CONCAT_VECTORS || N0.getNumOperands() != 2 ||
      N1.getNumOperands() != 2 || !N0.getOperand(1).isUndef() ||
      !N1.getOperand(1).isUndef())
    return SDValue();

  // Construct the new shuffle mask. Elements from the first source retain their
  // index, but elements from the second source no longer need to skip an undef.
  SmallVector<int, 8> Mask;
  int NumElts = VT.getVectorNumElements();

  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  for (int Elt : SVOp->getMask())
    Mask.push_back(Elt < NumElts ? Elt : (Elt - NumElts / 2));

  SDLoc DL(N);
  SDValue Concat = DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, N0.getOperand(0),
                               N1.getOperand(0));
  return DAG.getVectorShuffle(VT, DL, Concat, DAG.getUNDEF(VT), Mask);
}

/// Eliminate a redundant shuffle of a horizontal math op.
static SDValue foldShuffleOfHorizOp(SDNode *N, SelectionDAG &DAG) {
  unsigned Opcode = N->getOpcode();
  if (Opcode != X86ISD::MOVDDUP && Opcode != X86ISD::VBROADCAST)
    if (Opcode != ISD::VECTOR_SHUFFLE || !N->getOperand(1).isUndef())
      return SDValue();

  // For a broadcast, peek through an extract element of index 0 to find the
  // horizontal op: broadcast (ext_vec_elt HOp, 0)
  EVT VT = N->getValueType(0);
  if (Opcode == X86ISD::VBROADCAST) {
    SDValue SrcOp = N->getOperand(0);
    if (SrcOp.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
        SrcOp.getValueType() == MVT::f64 &&
        SrcOp.getOperand(0).getValueType() == VT &&
        isNullConstant(SrcOp.getOperand(1)))
      N = SrcOp.getNode();
  }

  SDValue HOp = N->getOperand(0);
  if (HOp.getOpcode() != X86ISD::HADD && HOp.getOpcode() != X86ISD::FHADD &&
      HOp.getOpcode() != X86ISD::HSUB && HOp.getOpcode() != X86ISD::FHSUB)
    return SDValue();

  // 128-bit horizontal math instructions are defined to operate on adjacent
  // lanes of each operand as:
  // v4X32: A[0] + A[1] , A[2] + A[3] , B[0] + B[1] , B[2] + B[3]
  // ...similarly for v2f64 and v8i16.
  if (!HOp.getOperand(0).isUndef() && !HOp.getOperand(1).isUndef() &&
      HOp.getOperand(0) != HOp.getOperand(1))
    return SDValue();

  // The shuffle that we are eliminating may have allowed the horizontal op to
  // have an undemanded (undefined) operand. Duplicate the other (defined)
  // operand to ensure that the results are defined across all lanes without the
  // shuffle.
  auto updateHOp = [](SDValue HorizOp, SelectionDAG &DAG) {
    SDValue X;
    if (HorizOp.getOperand(0).isUndef()) {
      assert(!HorizOp.getOperand(1).isUndef() && "Not expecting foldable h-op");
      X = HorizOp.getOperand(1);
    } else if (HorizOp.getOperand(1).isUndef()) {
      assert(!HorizOp.getOperand(0).isUndef() && "Not expecting foldable h-op");
      X = HorizOp.getOperand(0);
    } else {
      return HorizOp;
    }
    return DAG.getNode(HorizOp.getOpcode(), SDLoc(HorizOp),
                       HorizOp.getValueType(), X, X);
  };

  // When the operands of a horizontal math op are identical, the low half of
  // the result is the same as the high half. If a target shuffle is also
  // replicating low and high halves, we don't need the shuffle.
  if (Opcode == X86ISD::MOVDDUP || Opcode == X86ISD::VBROADCAST) {
    if (HOp.getScalarValueSizeInBits() == 64) {
      // movddup (hadd X, X) --> hadd X, X
      // broadcast (extract_vec_elt (hadd X, X), 0) --> hadd X, X
      assert((HOp.getValueType() == MVT::v2f64 ||
        HOp.getValueType() == MVT::v4f64) && HOp.getValueType() == VT &&
        "Unexpected type for h-op");
      return updateHOp(HOp, DAG);
    }
    return SDValue();
  }

  // shuffle (hadd X, X), undef, [low half...high half] --> hadd X, X
  ArrayRef<int> Mask = cast<ShuffleVectorSDNode>(N)->getMask();
  // TODO: Other mask possibilities like {1,1} and {1,0} could be added here,
  // but this should be tied to whatever horizontal op matching and shuffle
  // canonicalization are producing.
  if (HOp.getValueSizeInBits() == 128 &&
      (isTargetShuffleEquivalent(Mask, {0, 0}) ||
       isTargetShuffleEquivalent(Mask, {0, 1, 0, 1}) ||
       isTargetShuffleEquivalent(Mask, {0, 1, 2, 3, 0, 1, 2, 3})))
    return updateHOp(HOp, DAG);

  if (HOp.getValueSizeInBits() == 256 &&
      (isTargetShuffleEquivalent(Mask, {0, 0, 2, 2}) ||
       isTargetShuffleEquivalent(Mask, {0, 1, 0, 1, 4, 5, 4, 5}) ||
       isTargetShuffleEquivalent(
           Mask, {0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11})))
    return updateHOp(HOp, DAG);

  return SDValue();
}

/// If we have a shuffle of AVX/AVX512 (256/512 bit) vectors that only uses the
/// low half of each source vector and does not set any high half elements in
/// the destination vector, narrow the shuffle to half its original size.
static SDValue narrowShuffle(ShuffleVectorSDNode *Shuf, SelectionDAG &DAG) {
  if (!Shuf->getValueType(0).isSimple())
    return SDValue();
  MVT VT = Shuf->getSimpleValueType(0);
  if (!VT.is256BitVector() && !VT.is512BitVector())
    return SDValue();

  // See if we can ignore all of the high elements of the shuffle.
  ArrayRef<int> Mask = Shuf->getMask();
  if (!isUndefUpperHalf(Mask))
    return SDValue();

  // Check if the shuffle mask accesses only the low half of each input vector
  // (half-index output is 0 or 2).
  int HalfIdx1, HalfIdx2;
  SmallVector<int, 8> HalfMask(Mask.size() / 2);
  if (!getHalfShuffleMask(Mask, HalfMask, HalfIdx1, HalfIdx2) ||
      (HalfIdx1 % 2 == 1) || (HalfIdx2 % 2 == 1))
    return SDValue();

  // Create a half-width shuffle to replace the unnecessarily wide shuffle.
  // The trick is knowing that all of the insert/extract are actually free
  // subregister (zmm<->ymm or ymm<->xmm) ops. That leaves us with a shuffle
  // of narrow inputs into a narrow output, and that is always cheaper than
  // the wide shuffle that we started with.
  return getShuffleHalfVectors(SDLoc(Shuf), Shuf->getOperand(0),
                               Shuf->getOperand(1), HalfMask, HalfIdx1,
                               HalfIdx2, false, DAG, /*UseConcat*/true);
}

SDValue X86::combineShuffle(SDNode *N, SelectionDAG &DAG,
                            TargetLowering::DAGCombinerInfo &DCI,
                            const X86Subtarget &Subtarget) {
  if (auto *Shuf = dyn_cast<ShuffleVectorSDNode>(N))
    if (SDValue V = narrowShuffle(Shuf, DAG))
      return V;

  // If we have legalized the vector types, look for blends of FADD and FSUB
  // nodes that we can fuse into an ADDSUB, FMADDSUB, or FMSUBADD node.
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (TLI.isTypeLegal(VT)) {
    if (SDValue AddSub = combineShuffleToAddSubOrFMAddSub(N, Subtarget, DAG))
      return AddSub;

    if (SDValue HAddSub = foldShuffleOfHorizOp(N, DAG))
      return HAddSub;
  }

  // During Type Legalization, when promoting illegal vector types,
  // the backend might introduce new shuffle dag nodes and bitcasts.
  //
  // This code performs the following transformation:
  // fold: (shuffle (bitcast (BINOP A, B)), Undef, <Mask>) ->
  //       (shuffle (BINOP (bitcast A), (bitcast B)), Undef, <Mask>)
  //
  // We do this only if both the bitcast and the BINOP dag nodes have
  // one use. Also, perform this transformation only if the new binary
  // operation is legal. This is to avoid introducing dag nodes that
  // potentially need to be further expanded (or custom lowered) into a
  // less optimal sequence of dag nodes.
  if (!ExperimentalVectorWideningLegalization &&
      !DCI.isBeforeLegalize() && DCI.isBeforeLegalizeOps() &&
      N->getOpcode() == ISD::VECTOR_SHUFFLE &&
      N->getOperand(0).getOpcode() == ISD::BITCAST &&
      N->getOperand(1).isUndef() && N->getOperand(0).hasOneUse()) {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);

    SDValue BC0 = N0.getOperand(0);
    EVT SVT = BC0.getValueType();
    unsigned Opcode = BC0.getOpcode();
    unsigned NumElts = VT.getVectorNumElements();

    if (BC0.hasOneUse() && SVT.isVector() &&
        SVT.getVectorNumElements() * 2 == NumElts &&
        TLI.isOperationLegal(Opcode, VT)) {
      bool CanFold = false;
      switch (Opcode) {
      default : break;
      case ISD::ADD:
      case ISD::SUB:
      case ISD::MUL:
        // isOperationLegal lies for integer ops on floating point types.
        CanFold = VT.isInteger();
        break;
      case ISD::FADD:
      case ISD::FSUB:
      case ISD::FMUL:
        // isOperationLegal lies for floating point ops on integer types.
        CanFold = VT.isFloatingPoint();
        break;
      }

      unsigned SVTNumElts = SVT.getVectorNumElements();
      ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
      for (unsigned i = 0, e = SVTNumElts; i != e && CanFold; ++i)
        CanFold = SVOp->getMaskElt(i) == (int)(i * 2);
      for (unsigned i = SVTNumElts, e = NumElts; i != e && CanFold; ++i)
        CanFold = SVOp->getMaskElt(i) < 0;

      if (CanFold) {
        SDValue BC00 = DAG.getBitcast(VT, BC0.getOperand(0));
        SDValue BC01 = DAG.getBitcast(VT, BC0.getOperand(1));
        SDValue NewBinOp = DAG.getNode(BC0.getOpcode(), dl, VT, BC00, BC01);
        return DAG.getVectorShuffle(VT, dl, NewBinOp, N1, SVOp->getMask());
      }
    }
  }

  // Attempt to combine into a vector load/broadcast.
  if (SDValue LD = combineToConsecutiveLoads(VT, N, dl, DAG, Subtarget, true))
    return LD;

  // For AVX2, we sometimes want to combine
  // (vector_shuffle <mask> (concat_vectors t1, undef)
  //                        (concat_vectors t2, undef))
  // Into:
  // (vector_shuffle <mask> (concat_vectors t1, t2), undef)
  // Since the latter can be efficiently lowered with VPERMD/VPERMQ
  if (SDValue ShufConcat = combineShuffleOfConcatUndef(N, DAG, Subtarget))
    return ShufConcat;

  if (isTargetShuffle(N->getOpcode())) {
    SDValue Op(N, 0);
    if (SDValue Shuffle = combineTargetShuffle(Op, DAG, DCI, Subtarget))
      return Shuffle;

    // Try recursively combining arbitrary sequences of x86 shuffle
    // instructions into higher-order shuffles. We do this after combining
    // specific PSHUF instruction sequences into their minimal form so that we
    // can evaluate how many specialized shuffle instructions are involved in
    // a particular chain.
    if (SDValue Res = combineX86ShufflesRecursively(Op, DAG, Subtarget))
      return Res;

    // Simplify source operands based on shuffle mask.
    // TODO - merge this into combineX86ShufflesRecursively.
    APInt KnownUndef, KnownZero;
    APInt DemandedElts = APInt::getAllOnesValue(VT.getVectorNumElements());
    if (TLI.SimplifyDemandedVectorElts(Op, DemandedElts, KnownUndef, KnownZero, DCI))
      return SDValue(N, 0);
  }

  // Look for a v2i64/v2f64 VZEXT_MOVL of a node that already produces zeros
  // in the upper 64 bits.
  // TODO: Can we generalize this using computeKnownBits.
  if (N->getOpcode() == X86ISD::VZEXT_MOVL &&
      (VT == MVT::v2f64 || VT == MVT::v2i64) &&
      N->getOperand(0).getOpcode() == ISD::BITCAST &&
      (N->getOperand(0).getOperand(0).getValueType() == MVT::v4f32 ||
       N->getOperand(0).getOperand(0).getValueType() == MVT::v4i32)) {
    SDValue In = N->getOperand(0).getOperand(0);
    switch (In.getOpcode()) {
    default:
      break;
    case X86ISD::CVTP2SI:   case X86ISD::CVTP2UI:
    case X86ISD::MCVTP2SI:  case X86ISD::MCVTP2UI:
    case X86ISD::CVTTP2SI:  case X86ISD::CVTTP2UI:
    case X86ISD::MCVTTP2SI: case X86ISD::MCVTTP2UI:
    case X86ISD::CVTSI2P:   case X86ISD::CVTUI2P:
    case X86ISD::MCVTSI2P:  case X86ISD::MCVTUI2P:
    case X86ISD::VFPROUND:  case X86ISD::VMFPROUND:
      if (In.getOperand(0).getValueType() == MVT::v2f64 ||
          In.getOperand(0).getValueType() == MVT::v2i64)
        return N->getOperand(0); // return the bitcast
      break;
    }
  }

  // Pull subvector inserts into undef through VZEXT_MOVL by making it an
  // insert into a zero vector. This helps get VZEXT_MOVL closer to
  // scalar_to_vectors where 256/512 are canonicalized to an insert and a
  // 128-bit scalar_to_vector. This reduces the number of isel patterns.
  if (N->getOpcode() == X86ISD::VZEXT_MOVL && !DCI.isBeforeLegalizeOps() &&
      N->getOperand(0).getOpcode() == ISD::INSERT_SUBVECTOR &&
      N->getOperand(0).hasOneUse() &&
      N->getOperand(0).getOperand(0).isUndef() &&
      isNullConstant(N->getOperand(0).getOperand(2))) {
    SDValue In = N->getOperand(0).getOperand(1);
    SDValue Movl = DAG.getNode(X86ISD::VZEXT_MOVL, dl, In.getValueType(), In);
    return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, VT,
                       getZeroVector(VT.getSimpleVT(), Subtarget, DAG, dl),
                       Movl, N->getOperand(0).getOperand(2));
  }

  // If this a vzmovl of a full vector load, replace it with a vzload, unless
  // the load is volatile.
  if (N->getOpcode() == X86ISD::VZEXT_MOVL && N->getOperand(0).hasOneUse() &&
      ISD::isNormalLoad(N->getOperand(0).getNode())) {
    LoadSDNode *LN = cast<LoadSDNode>(N->getOperand(0));
    if (LN->isSimple()) {
      SDVTList Tys = DAG.getVTList(VT, MVT::Other);
      SDValue Ops[] = { LN->getChain(), LN->getBasePtr() };
      SDValue VZLoad =
          DAG.getMemIntrinsicNode(X86ISD::VZEXT_LOAD, dl, Tys, Ops,
                                  VT.getVectorElementType(),
                                  LN->getPointerInfo(),
                                  LN->getAlignment(),
                                  MachineMemOperand::MOLoad);
      DAG.ReplaceAllUsesOfValueWith(SDValue(LN, 1), VZLoad.getValue(1));
      return VZLoad;
    }
  }


  // Look for a truncating shuffle to v2i32 of a PMULUDQ where one of the
  // operands is an extend from v2i32 to v2i64. Turn it into a pmulld.
  // FIXME: This can probably go away once we default to widening legalization.
  if (!ExperimentalVectorWideningLegalization &&
      Subtarget.hasSSE41() && VT == MVT::v4i32 &&
      N->getOpcode() == ISD::VECTOR_SHUFFLE &&
      N->getOperand(0).getOpcode() == ISD::BITCAST &&
      N->getOperand(0).getOperand(0).getOpcode() == X86ISD::PMULUDQ) {
    SDValue BC = N->getOperand(0);
    SDValue MULUDQ = BC.getOperand(0);
    ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
    ArrayRef<int> Mask = SVOp->getMask();
    if (BC.hasOneUse() && MULUDQ.hasOneUse() &&
        Mask[0] == 0 && Mask[1] == 2 && Mask[2] == -1 && Mask[3] == -1) {
      SDValue Op0 = MULUDQ.getOperand(0);
      SDValue Op1 = MULUDQ.getOperand(1);
      if (Op0.getOpcode() == ISD::BITCAST &&
          Op0.getOperand(0).getOpcode() == ISD::VECTOR_SHUFFLE &&
          Op0.getOperand(0).getValueType() == MVT::v4i32) {
        ShuffleVectorSDNode *SVOp0 =
          cast<ShuffleVectorSDNode>(Op0.getOperand(0));
        ArrayRef<int> Mask2 = SVOp0->getMask();
        if (Mask2[0] == 0 && Mask2[1] == -1 &&
            Mask2[2] == 1 && Mask2[3] == -1) {
          Op0 = SVOp0->getOperand(0);
          Op1 = DAG.getBitcast(MVT::v4i32, Op1);
          Op1 = DAG.getVectorShuffle(MVT::v4i32, dl, Op1, Op1, Mask);
          return DAG.getNode(ISD::MUL, dl, MVT::v4i32, Op0, Op1);
        }
      }
      if (Op1.getOpcode() == ISD::BITCAST &&
          Op1.getOperand(0).getOpcode() == ISD::VECTOR_SHUFFLE &&
          Op1.getOperand(0).getValueType() == MVT::v4i32) {
        ShuffleVectorSDNode *SVOp1 =
          cast<ShuffleVectorSDNode>(Op1.getOperand(0));
        ArrayRef<int> Mask2 = SVOp1->getMask();
        if (Mask2[0] == 0 && Mask2[1] == -1 &&
            Mask2[2] == 1 && Mask2[3] == -1) {
          Op0 = DAG.getBitcast(MVT::v4i32, Op0);
          Op0 = DAG.getVectorShuffle(MVT::v4i32, dl, Op0, Op0, Mask);
          Op1 = SVOp1->getOperand(0);
          return DAG.getNode(ISD::MUL, dl, MVT::v4i32, Op0, Op1);
        }
      }
    }
  }

  return SDValue();
}

bool X86TargetLowering::SimplifyDemandedVectorEltsForTargetNode(
    SDValue Op, const APInt &DemandedElts, APInt &KnownUndef, APInt &KnownZero,
    TargetLoweringOpt &TLO, unsigned Depth) const {
  int NumElts = DemandedElts.getBitWidth();
  unsigned Opc = Op.getOpcode();
  EVT VT = Op.getValueType();

  // Handle special case opcodes.
  switch (Opc) {
  case X86ISD::PMULDQ:
  case X86ISD::PMULUDQ: {
    APInt LHSUndef, LHSZero;
    APInt RHSUndef, RHSZero;
    SDValue LHS = Op.getOperand(0);
    SDValue RHS = Op.getOperand(1);
    if (SimplifyDemandedVectorElts(LHS, DemandedElts, LHSUndef, LHSZero, TLO,
                                   Depth + 1))
      return true;
    if (SimplifyDemandedVectorElts(RHS, DemandedElts, RHSUndef, RHSZero, TLO,
                                   Depth + 1))
      return true;
    // Multiply by zero.
    KnownZero = LHSZero | RHSZero;
    break;
  }
  case X86ISD::VSHL:
  case X86ISD::VSRL:
  case X86ISD::VSRA: {
    // We only need the bottom 64-bits of the (128-bit) shift amount.
    SDValue Amt = Op.getOperand(1);
    MVT AmtVT = Amt.getSimpleValueType();
    assert(AmtVT.is128BitVector() && "Unexpected value type");

    // If we reuse the shift amount just for sse shift amounts then we know that
    // only the bottom 64-bits are only ever used.
    bool AssumeSingleUse = llvm::all_of(Amt->uses(), [&Amt](SDNode *Use) {
      unsigned UseOpc = Use->getOpcode();
      return (UseOpc == X86ISD::VSHL || UseOpc == X86ISD::VSRL ||
              UseOpc == X86ISD::VSRA) &&
             Use->getOperand(0) != Amt;
    });

    APInt AmtUndef, AmtZero;
    unsigned NumAmtElts = AmtVT.getVectorNumElements();
    APInt AmtElts = APInt::getLowBitsSet(NumAmtElts, NumAmtElts / 2);
    if (SimplifyDemandedVectorElts(Amt, AmtElts, AmtUndef, AmtZero, TLO,
                                   Depth + 1, AssumeSingleUse))
      return true;
    LLVM_FALLTHROUGH;
  }
  case X86ISD::VSHLI:
  case X86ISD::VSRLI:
  case X86ISD::VSRAI: {
    SDValue Src = Op.getOperand(0);
    APInt SrcUndef;
    if (SimplifyDemandedVectorElts(Src, DemandedElts, SrcUndef, KnownZero, TLO,
                                   Depth + 1))
      return true;
    // TODO convert SrcUndef to KnownUndef.
    break;
  }
  case X86ISD::KSHIFTL:
  case X86ISD::KSHIFTR: {
    SDValue Src = Op.getOperand(0);
    auto *Amt = cast<ConstantSDNode>(Op.getOperand(1));
    assert(Amt->getAPIntValue().ult(NumElts) && "Out of range shift amount");
    unsigned ShiftAmt = Amt->getZExtValue();
    bool ShiftLeft = (X86ISD::KSHIFTL == Opc);

    APInt DemandedSrc =
        ShiftLeft ? DemandedElts.lshr(ShiftAmt) : DemandedElts.shl(ShiftAmt);
    if (SimplifyDemandedVectorElts(Src, DemandedSrc, KnownUndef, KnownZero, TLO,
                                   Depth + 1))
      return true;

    if (ShiftLeft) {
      KnownUndef = KnownUndef.shl(ShiftAmt);
      KnownZero = KnownZero.shl(ShiftAmt);
      KnownZero.setLowBits(ShiftAmt);
    } else {
      KnownUndef = KnownUndef.lshr(ShiftAmt);
      KnownZero = KnownZero.lshr(ShiftAmt);
      KnownZero.setHighBits(ShiftAmt);
    }
    break;
  }
  case X86ISD::CVTSI2P:
  case X86ISD::CVTUI2P: {
    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();
    APInt SrcUndef, SrcZero;
    APInt SrcElts = DemandedElts.zextOrTrunc(SrcVT.getVectorNumElements());
    if (SimplifyDemandedVectorElts(Src, SrcElts, SrcUndef, SrcZero, TLO,
                                   Depth + 1))
      return true;
    break;
  }
  case X86ISD::PACKSS:
  case X86ISD::PACKUS: {
    SDValue N0 = Op.getOperand(0);
    SDValue N1 = Op.getOperand(1);

    APInt DemandedLHS, DemandedRHS;
    getPackDemandedElts(VT, DemandedElts, DemandedLHS, DemandedRHS);

    APInt SrcUndef, SrcZero;
    if (SimplifyDemandedVectorElts(N0, DemandedLHS, SrcUndef, SrcZero, TLO,
                                   Depth + 1))
      return true;
    if (SimplifyDemandedVectorElts(N1, DemandedRHS, SrcUndef, SrcZero, TLO,
                                   Depth + 1))
      return true;

    // Aggressively peek through ops to get at the demanded elts.
    // TODO - we should do this for all target/faux shuffles ops.
    if (!DemandedElts.isAllOnesValue()) {
      APInt DemandedSrcBits =
          APInt::getAllOnesValue(N0.getScalarValueSizeInBits());
      SDValue NewN0 = SimplifyMultipleUseDemandedBits(
          N0, DemandedSrcBits, DemandedLHS, TLO.DAG, Depth + 1);
      SDValue NewN1 = SimplifyMultipleUseDemandedBits(
          N1, DemandedSrcBits, DemandedRHS, TLO.DAG, Depth + 1);
      if (NewN0 || NewN1) {
        NewN0 = NewN0 ? NewN0 : N0;
        NewN1 = NewN1 ? NewN1 : N1;
        return TLO.CombineTo(Op,
                             TLO.DAG.getNode(Opc, SDLoc(Op), VT, NewN0, NewN1));
      }
    }
    break;
  }
  case X86ISD::HADD:
  case X86ISD::HSUB:
  case X86ISD::FHADD:
  case X86ISD::FHSUB: {
    APInt DemandedLHS, DemandedRHS;
    getHorizDemandedElts(VT, DemandedElts, DemandedLHS, DemandedRHS);

    APInt LHSUndef, LHSZero;
    if (SimplifyDemandedVectorElts(Op.getOperand(0), DemandedLHS, LHSUndef,
                                   LHSZero, TLO, Depth + 1))
      return true;
    APInt RHSUndef, RHSZero;
    if (SimplifyDemandedVectorElts(Op.getOperand(1), DemandedRHS, RHSUndef,
                                   RHSZero, TLO, Depth + 1))
      return true;
    break;
  }
  case X86ISD::VTRUNC:
  case X86ISD::VTRUNCS:
  case X86ISD::VTRUNCUS: {
    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();
    APInt DemandedSrc = DemandedElts.zextOrTrunc(SrcVT.getVectorNumElements());
    APInt SrcUndef, SrcZero;
    if (SimplifyDemandedVectorElts(Src, DemandedSrc, SrcUndef, SrcZero, TLO,
                                   Depth + 1))
      return true;
    KnownZero = SrcZero.zextOrTrunc(NumElts);
    KnownUndef = SrcUndef.zextOrTrunc(NumElts);
    break;
  }
  case X86ISD::BLENDV: {
    APInt SelUndef, SelZero;
    if (SimplifyDemandedVectorElts(Op.getOperand(0), DemandedElts, SelUndef,
                                   SelZero, TLO, Depth + 1))
      return true;

    // TODO: Use SelZero to adjust LHS/RHS DemandedElts.
    APInt LHSUndef, LHSZero;
    if (SimplifyDemandedVectorElts(Op.getOperand(1), DemandedElts, LHSUndef,
                                   LHSZero, TLO, Depth + 1))
      return true;

    APInt RHSUndef, RHSZero;
    if (SimplifyDemandedVectorElts(Op.getOperand(2), DemandedElts, RHSUndef,
                                   RHSZero, TLO, Depth + 1))
      return true;

    KnownZero = LHSZero & RHSZero;
    KnownUndef = LHSUndef & RHSUndef;
    break;
  }
  case X86ISD::VBROADCAST: {
    SDValue Src = Op.getOperand(0);
    MVT SrcVT = Src.getSimpleValueType();
    if (!SrcVT.isVector())
      return false;
    // Don't bother broadcasting if we just need the 0'th element.
    if (DemandedElts == 1) {
      if (Src.getValueType() != VT)
        Src = widenSubVector(VT.getSimpleVT(), Src, false, Subtarget, TLO.DAG,
                             SDLoc(Op));
      return TLO.CombineTo(Op, Src);
    }
    APInt SrcUndef, SrcZero;
    APInt SrcElts = APInt::getOneBitSet(SrcVT.getVectorNumElements(), 0);
    if (SimplifyDemandedVectorElts(Src, SrcElts, SrcUndef, SrcZero, TLO,
                                   Depth + 1))
      return true;
    break;
  }
  case X86ISD::VPERMV: {
    SDValue Mask = Op.getOperand(0);
    APInt MaskUndef, MaskZero;
    if (SimplifyDemandedVectorElts(Mask, DemandedElts, MaskUndef, MaskZero, TLO,
                                   Depth + 1))
      return true;
    break;
  }
  case X86ISD::PSHUFB:
  case X86ISD::VPERMV3:
  case X86ISD::VPERMILPV: {
    SDValue Mask = Op.getOperand(1);
    APInt MaskUndef, MaskZero;
    if (SimplifyDemandedVectorElts(Mask, DemandedElts, MaskUndef, MaskZero, TLO,
                                   Depth + 1))
      return true;
    break;
  }
  case X86ISD::VPPERM:
  case X86ISD::VPERMIL2: {
    SDValue Mask = Op.getOperand(2);
    APInt MaskUndef, MaskZero;
    if (SimplifyDemandedVectorElts(Mask, DemandedElts, MaskUndef, MaskZero, TLO,
                                   Depth + 1))
      return true;
    break;
  }
  }

  // For 256/512-bit ops that are 128/256-bit ops glued together, if we do not
  // demand any of the high elements, then narrow the op to 128/256-bits: e.g.
  // (op ymm0, ymm1) --> insert undef, (op xmm0, xmm1), 0
  if ((VT.is256BitVector() || VT.is512BitVector()) &&
      DemandedElts.lshr(NumElts / 2) == 0) {
    unsigned SizeInBits = VT.getSizeInBits();
    unsigned ExtSizeInBits = SizeInBits / 2;

    // See if 512-bit ops only use the bottom 128-bits.
    if (VT.is512BitVector() && DemandedElts.lshr(NumElts / 4) == 0)
      ExtSizeInBits = SizeInBits / 4;

    switch (Opc) {
      // Zero upper elements.
    case X86ISD::VZEXT_MOVL: {
      SDLoc DL(Op);
      SDValue Ext0 =
          extractSubVector(Op.getOperand(0), 0, TLO.DAG, DL, ExtSizeInBits);
      SDValue ExtOp =
          TLO.DAG.getNode(Opc, DL, Ext0.getValueType(), Ext0);
      SDValue UndefVec = TLO.DAG.getUNDEF(VT);
      SDValue Insert =
          insertSubVector(UndefVec, ExtOp, 0, TLO.DAG, DL, ExtSizeInBits);
      return TLO.CombineTo(Op, Insert);
    }
      // Subvector broadcast.
    case X86ISD::SUBV_BROADCAST: {
      SDLoc DL(Op);
      SDValue Src = Op.getOperand(0);
      if (Src.getValueSizeInBits() > ExtSizeInBits)
        Src = extractSubVector(Src, 0, TLO.DAG, DL, ExtSizeInBits);
      else if (Src.getValueSizeInBits() < ExtSizeInBits) {
        MVT SrcSVT = Src.getSimpleValueType().getScalarType();
        MVT SrcVT =
            MVT::getVectorVT(SrcSVT, ExtSizeInBits / SrcSVT.getSizeInBits());
        Src = TLO.DAG.getNode(X86ISD::SUBV_BROADCAST, DL, SrcVT, Src);
      }
      return TLO.CombineTo(Op, insertSubVector(TLO.DAG.getUNDEF(VT), Src, 0,
                                               TLO.DAG, DL, ExtSizeInBits));
    }
      // Byte shifts by immediate.
    case X86ISD::VSHLDQ:
    case X86ISD::VSRLDQ:
      // Shift by uniform.
    case X86ISD::VSHL:
    case X86ISD::VSRL:
    case X86ISD::VSRA:
      // Shift by immediate.
    case X86ISD::VSHLI:
    case X86ISD::VSRLI:
    case X86ISD::VSRAI: {
      SDLoc DL(Op);
      SDValue Ext0 =
          extractSubVector(Op.getOperand(0), 0, TLO.DAG, DL, ExtSizeInBits);
      SDValue ExtOp =
          TLO.DAG.getNode(Opc, DL, Ext0.getValueType(), Ext0, Op.getOperand(1));
      SDValue UndefVec = TLO.DAG.getUNDEF(VT);
      SDValue Insert =
          insertSubVector(UndefVec, ExtOp, 0, TLO.DAG, DL, ExtSizeInBits);
      return TLO.CombineTo(Op, Insert);
    }
    case X86ISD::VPERMI: {
      // Simplify PERMPD/PERMQ to extract_subvector.
      // TODO: This should be done in shuffle combining.
      if (VT == MVT::v4f64 || VT == MVT::v4i64) {
        SmallVector<int, 4> Mask;
        DecodeVPERMMask(NumElts, Op.getConstantOperandVal(1), Mask);
        if (isUndefOrEqual(Mask[0], 2) && isUndefOrEqual(Mask[1], 3)) {
          SDLoc DL(Op);
          SDValue Ext = extractSubVector(Op.getOperand(0), 2, TLO.DAG, DL, 128);
          SDValue UndefVec = TLO.DAG.getUNDEF(VT);
          SDValue Insert = insertSubVector(UndefVec, Ext, 0, TLO.DAG, DL, 128);
          return TLO.CombineTo(Op, Insert);
        }
      }
      break;
    }
      // Target Shuffles.
    case X86ISD::PSHUFB:
    case X86ISD::UNPCKL:
    case X86ISD::UNPCKH:
      // Saturated Packs.
    case X86ISD::PACKSS:
    case X86ISD::PACKUS:
      // Horizontal Ops.
    case X86ISD::HADD:
    case X86ISD::HSUB:
    case X86ISD::FHADD:
    case X86ISD::FHSUB: {
      SDLoc DL(Op);
      MVT ExtVT = VT.getSimpleVT();
      ExtVT = MVT::getVectorVT(ExtVT.getScalarType(),
                               ExtSizeInBits / ExtVT.getScalarSizeInBits());
      SDValue Ext0 =
          extractSubVector(Op.getOperand(0), 0, TLO.DAG, DL, ExtSizeInBits);
      SDValue Ext1 =
          extractSubVector(Op.getOperand(1), 0, TLO.DAG, DL, ExtSizeInBits);
      SDValue ExtOp = TLO.DAG.getNode(Opc, DL, ExtVT, Ext0, Ext1);
      SDValue UndefVec = TLO.DAG.getUNDEF(VT);
      SDValue Insert =
          insertSubVector(UndefVec, ExtOp, 0, TLO.DAG, DL, ExtSizeInBits);
      return TLO.CombineTo(Op, Insert);
    }
    }
  }

  // Get target/faux shuffle mask.
  SmallVector<int, 64> OpMask;
  SmallVector<SDValue, 2> OpInputs;
  if (!VT.isSimple() || !getTargetShuffleInputs(Op, DemandedElts, OpInputs,
                                                OpMask, TLO.DAG, Depth, false))
    return false;

  // Shuffle inputs must be the same size as the result.
  if (OpMask.size() != (unsigned)NumElts ||
      llvm::any_of(OpInputs, [VT](SDValue V) {
        return VT.getSizeInBits() != V.getValueSizeInBits() ||
               !V.getValueType().isVector();
      }))
    return false;

  // Clear known elts that might have been set above.
  KnownZero.clearAllBits();
  KnownUndef.clearAllBits();

  // Check if shuffle mask can be simplified to undef/zero/identity.
  int NumSrcs = OpInputs.size();
  for (int i = 0; i != NumElts; ++i) {
    int &M = OpMask[i];
    if (!DemandedElts[i])
      M = SM_SentinelUndef;
    else if (0 <= M && OpInputs[M / NumElts].isUndef())
      M = SM_SentinelUndef;
  }

  if (isUndefInRange(OpMask, 0, NumElts)) {
    KnownUndef.setAllBits();
    return TLO.CombineTo(Op, TLO.DAG.getUNDEF(VT));
  }
  if (isUndefOrZeroInRange(OpMask, 0, NumElts)) {
    KnownZero.setAllBits();
    return TLO.CombineTo(
        Op, getZeroVector(VT.getSimpleVT(), Subtarget, TLO.DAG, SDLoc(Op)));
  }
  for (int Src = 0; Src != NumSrcs; ++Src)
    if (isSequentialOrUndefInRange(OpMask, 0, NumElts, Src * NumElts))
      return TLO.CombineTo(Op, TLO.DAG.getBitcast(VT, OpInputs[Src]));

  // Attempt to simplify inputs.
  for (int Src = 0; Src != NumSrcs; ++Src) {
    // TODO: Support inputs of different types.
    if (OpInputs[Src].getValueType() != VT)
      continue;

    int Lo = Src * NumElts;
    APInt SrcElts = APInt::getNullValue(NumElts);
    for (int i = 0; i != NumElts; ++i)
      if (DemandedElts[i]) {
        int M = OpMask[i] - Lo;
        if (0 <= M && M < NumElts)
          SrcElts.setBit(M);
      }

    APInt SrcUndef, SrcZero;
    if (SimplifyDemandedVectorElts(OpInputs[Src], SrcElts, SrcUndef, SrcZero,
                                   TLO, Depth + 1))
      return true;
  }

  // Extract known zero/undef elements.
  // TODO - Propagate input undef/zero elts.
  for (int i = 0; i != NumElts; ++i) {
    if (OpMask[i] == SM_SentinelUndef)
      KnownUndef.setBit(i);
    if (OpMask[i] == SM_SentinelZero)
      KnownZero.setBit(i);
  }

  return false;
}

SDValue X86::combineVectorPack(SDNode *N, SelectionDAG &DAG,
                               TargetLowering::DAGCombinerInfo &DCI,
                               const X86Subtarget &Subtarget) {
  unsigned Opcode = N->getOpcode();
  assert((X86ISD::PACKSS == Opcode || X86ISD::PACKUS == Opcode) &&
         "Unexpected shift opcode");

  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  unsigned DstBitsPerElt = VT.getScalarSizeInBits();
  unsigned SrcBitsPerElt = 2 * DstBitsPerElt;
  assert(N0.getScalarValueSizeInBits() == SrcBitsPerElt &&
         N1.getScalarValueSizeInBits() == SrcBitsPerElt &&
         "Unexpected PACKSS/PACKUS input type");

  bool IsSigned = (X86ISD::PACKSS == Opcode);

  // Constant Folding.
  APInt UndefElts0, UndefElts1;
  SmallVector<APInt, 32> EltBits0, EltBits1;
  if ((N0.isUndef() || N->isOnlyUserOf(N0.getNode())) &&
      (N1.isUndef() || N->isOnlyUserOf(N1.getNode())) &&
      getTargetConstantBitsFromNode(N0, SrcBitsPerElt, UndefElts0, EltBits0) &&
      getTargetConstantBitsFromNode(N1, SrcBitsPerElt, UndefElts1, EltBits1)) {
    unsigned NumLanes = VT.getSizeInBits() / 128;
    unsigned NumDstElts = VT.getVectorNumElements();
    unsigned NumSrcElts = NumDstElts / 2;
    unsigned NumDstEltsPerLane = NumDstElts / NumLanes;
    unsigned NumSrcEltsPerLane = NumSrcElts / NumLanes;

    APInt Undefs(NumDstElts, 0);
    SmallVector<APInt, 32> Bits(NumDstElts, APInt::getNullValue(DstBitsPerElt));
    for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
      for (unsigned Elt = 0; Elt != NumDstEltsPerLane; ++Elt) {
        unsigned SrcIdx = Lane * NumSrcEltsPerLane + Elt % NumSrcEltsPerLane;
        auto &UndefElts = (Elt >= NumSrcEltsPerLane ? UndefElts1 : UndefElts0);
        auto &EltBits = (Elt >= NumSrcEltsPerLane ? EltBits1 : EltBits0);

        if (UndefElts[SrcIdx]) {
          Undefs.setBit(Lane * NumDstEltsPerLane + Elt);
          continue;
        }

        APInt &Val = EltBits[SrcIdx];
        if (IsSigned) {
          // PACKSS: Truncate signed value with signed saturation.
          // Source values less than dst minint are saturated to minint.
          // Source values greater than dst maxint are saturated to maxint.
          if (Val.isSignedIntN(DstBitsPerElt))
            Val = Val.trunc(DstBitsPerElt);
          else if (Val.isNegative())
            Val = APInt::getSignedMinValue(DstBitsPerElt);
          else
            Val = APInt::getSignedMaxValue(DstBitsPerElt);
        } else {
          // PACKUS: Truncate signed value with unsigned saturation.
          // Source values less than zero are saturated to zero.
          // Source values greater than dst maxuint are saturated to maxuint.
          if (Val.isIntN(DstBitsPerElt))
            Val = Val.trunc(DstBitsPerElt);
          else if (Val.isNegative())
            Val = APInt::getNullValue(DstBitsPerElt);
          else
            Val = APInt::getAllOnesValue(DstBitsPerElt);
        }
        Bits[Lane * NumDstEltsPerLane + Elt] = Val;
      }
    }

    return getConstVector(Bits, Undefs, VT.getSimpleVT(), DAG, SDLoc(N));
  }

  // Try to combine a PACKUSWB/PACKSSWB implemented truncate with a regular
  // truncate to create a larger truncate.
  if (Subtarget.hasAVX512() &&
      N0.getOpcode() == ISD::TRUNCATE && N1.isUndef() && VT == MVT::v16i8 &&
      N0.getOperand(0).getValueType() == MVT::v8i32) {
    if ((IsSigned && DAG.ComputeNumSignBits(N0) > 8) ||
        (!IsSigned &&
         DAG.MaskedValueIsZero(N0, APInt::getHighBitsSet(16, 8)))) {
      if (Subtarget.hasVLX())
        return DAG.getNode(X86ISD::VTRUNC, SDLoc(N), VT, N0.getOperand(0));

      // Widen input to v16i32 so we can truncate that.
      SDLoc dl(N);
      SDValue Concat = DAG.getNode(ISD::CONCAT_VECTORS, dl, MVT::v16i32,
                                   N0.getOperand(0), DAG.getUNDEF(MVT::v8i32));
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Concat);
    }
  }

  // Attempt to combine as shuffle.
  SDValue Op(N, 0);
  if (SDValue Res = combineX86ShufflesRecursively(Op, DAG, Subtarget))
    return Res;

  return SDValue();
}

SDValue X86::combineVectorInsert(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const X86Subtarget &Subtarget) {
  EVT VT = N->getValueType(0);
  assert(((N->getOpcode() == X86ISD::PINSRB && VT == MVT::v16i8) ||
          (N->getOpcode() == X86ISD::PINSRW && VT == MVT::v8i16)) &&
         "Unexpected vector insertion");

  unsigned NumBitsPerElt = VT.getScalarSizeInBits();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (TLI.SimplifyDemandedBits(SDValue(N, 0),
                               APInt::getAllOnesValue(NumBitsPerElt), DCI))
    return SDValue(N, 0);

  // Attempt to combine PINSRB/PINSRW patterns to a shuffle.
  SDValue Op(N, 0);
  if (SDValue Res = combineX86ShufflesRecursively(Op, DAG, Subtarget))
    return Res;

  return SDValue();
}

SDValue X86::combineVectorCompare(SDNode *N, SelectionDAG &DAG,
                                  const X86Subtarget &Subtarget) {
  MVT VT = N->getSimpleValueType(0);
  SDLoc DL(N);

  if (N->getOperand(0) == N->getOperand(1)) {
    if (N->getOpcode() == X86ISD::PCMPEQ)
      return DAG.getConstant(-1, DL, VT);
    if (N->getOpcode() == X86ISD::PCMPGT)
      return DAG.getConstant(0, DL, VT);
  }

  return SDValue();
}

/// Helper that combines an array of subvector ops as if they were the operands
/// of a ISD::CONCAT_VECTORS node, but may have come from another source (e.g.
/// ISD::INSERT_SUBVECTOR). The ops are assumed to be of the same type.
static SDValue combineConcatVectorOps(const SDLoc &DL, MVT VT,
                                      ArrayRef<SDValue> Ops, SelectionDAG &DAG,
                                      TargetLowering::DAGCombinerInfo &DCI,
                                      const X86Subtarget &Subtarget) {
  assert(Subtarget.hasAVX() && "AVX assumed for concat_vectors");

  if (llvm::all_of(Ops, [](SDValue Op) { return Op.isUndef(); }))
    return DAG.getUNDEF(VT);

  if (llvm::all_of(Ops, [](SDValue Op) {
        return ISD::isBuildVectorAllZeros(Op.getNode());
      }))
    return getZeroVector(VT, Subtarget, DAG, DL);

  SDValue Op0 = Ops[0];

  // Fold subvector loads into one.
  // If needed, look through bitcasts to get to the load.
  if (auto *FirstLd = dyn_cast<LoadSDNode>(peekThroughBitcasts(Op0))) {
    bool Fast;
    const X86TargetLowering *TLI = Subtarget.getTargetLowering();
    if (TLI->allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), VT,
                                *FirstLd->getMemOperand(), &Fast) &&
        Fast) {
      if (SDValue Ld =
              EltsFromConsecutiveLoads(VT, Ops, DL, DAG, Subtarget, false))
        return Ld;
    }
  }

  // Repeated subvectors.
  if (llvm::all_of(Ops, [Op0](SDValue Op) { return Op == Op0; })) {
    // If this broadcast/subv_broadcast is inserted into both halves, use a
    // larger broadcast/subv_broadcast.
    if (Op0.getOpcode() == X86ISD::VBROADCAST ||
        Op0.getOpcode() == X86ISD::SUBV_BROADCAST)
      return DAG.getNode(Op0.getOpcode(), DL, VT, Op0.getOperand(0));

    // concat_vectors(movddup(x),movddup(x)) -> broadcast(x)
    if (Op0.getOpcode() == X86ISD::MOVDDUP && VT == MVT::v4f64 &&
        (Subtarget.hasAVX2() || MayFoldLoad(Op0.getOperand(0))))
      return DAG.getNode(X86ISD::VBROADCAST, DL, VT,
                         DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f64,
                                     Op0.getOperand(0),
                                     DAG.getIntPtrConstant(0, DL)));

    // concat_vectors(scalar_to_vector(x),scalar_to_vector(x)) -> broadcast(x)
    if (Op0.getOpcode() == ISD::SCALAR_TO_VECTOR &&
        (Subtarget.hasAVX2() ||
         (VT.getScalarSizeInBits() >= 32 && MayFoldLoad(Op0.getOperand(0)))) &&
        Op0.getOperand(0).getValueType() == VT.getScalarType())
      return DAG.getNode(X86ISD::VBROADCAST, DL, VT, Op0.getOperand(0));
  }

  bool IsSplat = llvm::all_of(Ops, [&Op0](SDValue Op) { return Op == Op0; });

  // Repeated opcode.
  // TODO - combineX86ShufflesRecursively should handle shuffle concatenation
  // but it currently struggles with different vector widths.
  if (llvm::all_of(Ops, [Op0](SDValue Op) {
        return Op.getOpcode() == Op0.getOpcode();
      })) {
    unsigned NumOps = Ops.size();
    switch (Op0.getOpcode()) {
    case X86ISD::PSHUFHW:
    case X86ISD::PSHUFLW:
    case X86ISD::PSHUFD:
      if (!IsSplat && NumOps == 2 && VT.is256BitVector() &&
          Subtarget.hasInt256() && Op0.getOperand(1) == Ops[1].getOperand(1)) {
        SmallVector<SDValue, 2> Src;
        for (unsigned i = 0; i != NumOps; ++i)
          Src.push_back(Ops[i].getOperand(0));
        return DAG.getNode(Op0.getOpcode(), DL, VT,
                           DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Src),
                           Op0.getOperand(1));
      }
      LLVM_FALLTHROUGH;
    case X86ISD::VPERMILPI:
      // TODO - add support for vXf64/vXi64 shuffles.
      if (!IsSplat && NumOps == 2 && (VT == MVT::v8f32 || VT == MVT::v8i32) &&
          Subtarget.hasAVX() && Op0.getOperand(1) == Ops[1].getOperand(1)) {
        SmallVector<SDValue, 2> Src;
        for (unsigned i = 0; i != NumOps; ++i)
          Src.push_back(DAG.getBitcast(MVT::v4f32, Ops[i].getOperand(0)));
        SDValue Res = DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v8f32, Src);
        Res = DAG.getNode(X86ISD::VPERMILPI, DL, MVT::v8f32, Res,
                          Op0.getOperand(1));
        return DAG.getBitcast(VT, Res);
      }
      break;
    case X86ISD::PACKUS:
      if (NumOps == 2 && VT.is256BitVector() && Subtarget.hasInt256()) {
        SmallVector<SDValue, 2> LHS, RHS;
        for (unsigned i = 0; i != NumOps; ++i) {
          LHS.push_back(Ops[i].getOperand(0));
          RHS.push_back(Ops[i].getOperand(1));
        }
        MVT SrcVT = Op0.getOperand(0).getSimpleValueType();
        SrcVT = MVT::getVectorVT(SrcVT.getScalarType(),
                                 NumOps * SrcVT.getVectorNumElements());
        return DAG.getNode(Op0.getOpcode(), DL, VT,
                           DAG.getNode(ISD::CONCAT_VECTORS, DL, SrcVT, LHS),
                           DAG.getNode(ISD::CONCAT_VECTORS, DL, SrcVT, RHS));
      }
      break;
    }
  }

  return SDValue();
}

SDValue X86::combineConcatVectors(SDNode *N, SelectionDAG &DAG,
                                  TargetLowering::DAGCombinerInfo &DCI,
                                  const X86Subtarget &Subtarget) {
  EVT VT = N->getValueType(0);
  EVT SrcVT = N->getOperand(0).getValueType();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Don't do anything for i1 vectors.
  if (VT.getVectorElementType() == MVT::i1)
    return SDValue();

  if (Subtarget.hasAVX() && TLI.isTypeLegal(VT) && TLI.isTypeLegal(SrcVT)) {
    SmallVector<SDValue, 4> Ops(N->op_begin(), N->op_end());
    if (SDValue R = combineConcatVectorOps(SDLoc(N), VT.getSimpleVT(), Ops, DAG,
                                           DCI, Subtarget))
      return R;
  }

  return SDValue();
}

SDValue X86::combineInsertSubvector(SDNode *N, SelectionDAG &DAG,
                                    TargetLowering::DAGCombinerInfo &DCI,
                                    const X86Subtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  MVT OpVT = N->getSimpleValueType(0);

  bool IsI1Vector = OpVT.getVectorElementType() == MVT::i1;

  SDLoc dl(N);
  SDValue Vec = N->getOperand(0);
  SDValue SubVec = N->getOperand(1);

  uint64_t IdxVal = N->getConstantOperandVal(2);
  MVT SubVecVT = SubVec.getSimpleValueType();

  if (Vec.isUndef() && SubVec.isUndef())
    return DAG.getUNDEF(OpVT);

  // Inserting undefs/zeros into zeros/undefs is a zero vector.
  if ((Vec.isUndef() || ISD::isBuildVectorAllZeros(Vec.getNode())) &&
      (SubVec.isUndef() || ISD::isBuildVectorAllZeros(SubVec.getNode())))
    return getZeroVector(OpVT, Subtarget, DAG, dl);

  if (ISD::isBuildVectorAllZeros(Vec.getNode())) {
    // If we're inserting into a zero vector and then into a larger zero vector,
    // just insert into the larger zero vector directly.
    if (SubVec.getOpcode() == ISD::INSERT_SUBVECTOR &&
        ISD::isBuildVectorAllZeros(SubVec.getOperand(0).getNode())) {
      uint64_t Idx2Val = SubVec.getConstantOperandVal(2);
      return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, OpVT,
                         getZeroVector(OpVT, Subtarget, DAG, dl),
                         SubVec.getOperand(1),
                         DAG.getIntPtrConstant(IdxVal + Idx2Val, dl));
    }

    // If we're inserting into a zero vector and our input was extracted from an
    // insert into a zero vector of the same type and the extraction was at
    // least as large as the original insertion. Just insert the original
    // subvector into a zero vector.
    if (SubVec.getOpcode() == ISD::EXTRACT_SUBVECTOR && IdxVal == 0 &&
        isNullConstant(SubVec.getOperand(1)) &&
        SubVec.getOperand(0).getOpcode() == ISD::INSERT_SUBVECTOR) {
      SDValue Ins = SubVec.getOperand(0);
      if (isNullConstant(Ins.getOperand(2)) &&
          ISD::isBuildVectorAllZeros(Ins.getOperand(0).getNode()) &&
          Ins.getOperand(1).getValueSizeInBits() <= SubVecVT.getSizeInBits())
        return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, OpVT,
                           getZeroVector(OpVT, Subtarget, DAG, dl),
                           Ins.getOperand(1), N->getOperand(2));
    }
  }

  // Stop here if this is an i1 vector.
  if (IsI1Vector)
    return SDValue();

  // If this is an insert of an extract, combine to a shuffle. Don't do this
  // if the insert or extract can be represented with a subregister operation.
  if (SubVec.getOpcode() == ISD::EXTRACT_SUBVECTOR &&
      SubVec.getOperand(0).getSimpleValueType() == OpVT &&
      (IdxVal != 0 || !Vec.isUndef())) {
    int ExtIdxVal = SubVec.getConstantOperandVal(1);
    if (ExtIdxVal != 0) {
      int VecNumElts = OpVT.getVectorNumElements();
      int SubVecNumElts = SubVecVT.getVectorNumElements();
      SmallVector<int, 64> Mask(VecNumElts);
      // First create an identity shuffle mask.
      for (int i = 0; i != VecNumElts; ++i)
        Mask[i] = i;
      // Now insert the extracted portion.
      for (int i = 0; i != SubVecNumElts; ++i)
        Mask[i + IdxVal] = i + ExtIdxVal + VecNumElts;

      return DAG.getVectorShuffle(OpVT, dl, Vec, SubVec.getOperand(0), Mask);
    }
  }

  // Match concat_vector style patterns.
  SmallVector<SDValue, 2> SubVectorOps;
  if (collectConcatOps(N, SubVectorOps)) {
    if (SDValue Fold =
            combineConcatVectorOps(dl, OpVT, SubVectorOps, DAG, DCI, Subtarget))
      return Fold;

    // If we're inserting all zeros into the upper half, change this to
    // a concat with zero. We will match this to a move
    // with implicit upper bit zeroing during isel.
    // We do this here because we don't want combineConcatVectorOps to
    // create INSERT_SUBVECTOR from CONCAT_VECTORS.
    if (SubVectorOps.size() == 2 &&
        ISD::isBuildVectorAllZeros(SubVectorOps[1].getNode()))
      return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, OpVT,
                         getZeroVector(OpVT, Subtarget, DAG, dl),
                         SubVectorOps[0], DAG.getIntPtrConstant(0, dl));
  }

  // If this is a broadcast insert into an upper undef, use a larger broadcast.
  if (Vec.isUndef() && IdxVal != 0 && SubVec.getOpcode() == X86ISD::VBROADCAST)
    return DAG.getNode(X86ISD::VBROADCAST, dl, OpVT, SubVec.getOperand(0));

  return SDValue();
}

/// If we are extracting a subvector of a vector select and the select condition
/// is composed of concatenated vectors, try to narrow the select width. This
/// is a common pattern for AVX1 integer code because 256-bit selects may be
/// legal, but there is almost no integer math/logic available for 256-bit.
/// This function should only be called with legal types (otherwise, the calls
/// to get simple value types will assert).
static SDValue narrowExtractedVectorSelect(SDNode *Ext, SelectionDAG &DAG) {
  SDValue Sel = peekThroughBitcasts(Ext->getOperand(0));
  SmallVector<SDValue, 4> CatOps;
  if (Sel.getOpcode() != ISD::VSELECT ||
      !X86::collectConcatOps(Sel.getOperand(0).getNode(), CatOps))
    return SDValue();

  // Note: We assume simple value types because this should only be called with
  //       legal operations/types.
  // TODO: This can be extended to handle extraction to 256-bits.
  MVT VT = Ext->getSimpleValueType(0);
  if (!VT.is128BitVector())
    return SDValue();

  MVT SelCondVT = Sel.getOperand(0).getSimpleValueType();
  if (!SelCondVT.is256BitVector() && !SelCondVT.is512BitVector())
    return SDValue();

  MVT WideVT = Ext->getOperand(0).getSimpleValueType();
  MVT SelVT = Sel.getSimpleValueType();
  assert((SelVT.is256BitVector() || SelVT.is512BitVector()) &&
         "Unexpected vector type with legal operations");

  unsigned SelElts = SelVT.getVectorNumElements();
  unsigned CastedElts = WideVT.getVectorNumElements();
  unsigned ExtIdx = cast<ConstantSDNode>(Ext->getOperand(1))->getZExtValue();
  if (SelElts % CastedElts == 0) {
    // The select has the same or more (narrower) elements than the extract
    // operand. The extraction index gets scaled by that factor.
    ExtIdx *= (SelElts / CastedElts);
  } else if (CastedElts % SelElts == 0) {
    // The select has less (wider) elements than the extract operand. Make sure
    // that the extraction index can be divided evenly.
    unsigned IndexDivisor = CastedElts / SelElts;
    if (ExtIdx % IndexDivisor != 0)
      return SDValue();
    ExtIdx /= IndexDivisor;
  } else {
    llvm_unreachable("Element count of simple vector types are not divisible?");
  }

  unsigned NarrowingFactor = WideVT.getSizeInBits() / VT.getSizeInBits();
  unsigned NarrowElts = SelElts / NarrowingFactor;
  MVT NarrowSelVT = MVT::getVectorVT(SelVT.getVectorElementType(), NarrowElts);
  SDLoc DL(Ext);
  SDValue ExtCond = extract128BitVector(Sel.getOperand(0), ExtIdx, DAG, DL);
  SDValue ExtT = extract128BitVector(Sel.getOperand(1), ExtIdx, DAG, DL);
  SDValue ExtF = extract128BitVector(Sel.getOperand(2), ExtIdx, DAG, DL);
  SDValue NarrowSel = DAG.getSelect(DL, NarrowSelVT, ExtCond, ExtT, ExtF);
  return DAG.getBitcast(VT, NarrowSel);
}

SDValue X86::combineExtractSubvector(SDNode *N, SelectionDAG &DAG,
                                     TargetLowering::DAGCombinerInfo &DCI,
                                     const X86Subtarget &Subtarget) {
  // For AVX1 only, if we are extracting from a 256-bit and+not (which will
  // eventually get combined/lowered into ANDNP) with a concatenated operand,
  // split the 'and' into 128-bit ops to avoid the concatenate and extract.
  // We let generic combining take over from there to simplify the
  // insert/extract and 'not'.
  // This pattern emerges during AVX1 legalization. We handle it before lowering
  // to avoid complications like splitting constant vector loads.

  // Capture the original wide type in the likely case that we need to bitcast
  // back to this type.
  if (!N->getValueType(0).isSimple())
    return SDValue();

  MVT VT = N->getSimpleValueType(0);
  EVT WideVecVT = N->getOperand(0).getValueType();
  SDValue WideVec = peekThroughBitcasts(N->getOperand(0));
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (Subtarget.hasAVX() && !Subtarget.hasAVX2() &&
      TLI.isTypeLegal(WideVecVT) &&
      WideVecVT.getSizeInBits() == 256 && WideVec.getOpcode() == ISD::AND) {
    auto isConcatenatedNot = [] (SDValue V) {
      V = peekThroughBitcasts(V);
      if (!isBitwiseNot(V))
        return false;
      SDValue NotOp = V->getOperand(0);
      return peekThroughBitcasts(NotOp).getOpcode() == ISD::CONCAT_VECTORS;
    };
    if (isConcatenatedNot(WideVec.getOperand(0)) ||
        isConcatenatedNot(WideVec.getOperand(1))) {
      // extract (and v4i64 X, (not (concat Y1, Y2))), n -> andnp v2i64 X(n), Y1
      SDValue Concat = split256IntArith(WideVec, DAG);
      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, SDLoc(N), VT,
                         DAG.getBitcast(WideVecVT, Concat), N->getOperand(1));
    }
  }

  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  if (SDValue V = narrowExtractedVectorSelect(N, DAG))
    return V;

  SDValue InVec = N->getOperand(0);
  unsigned IdxVal = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();

  if (ISD::isBuildVectorAllZeros(InVec.getNode()))
    return getZeroVector(VT, Subtarget, DAG, SDLoc(N));

  if (ISD::isBuildVectorAllOnes(InVec.getNode())) {
    if (VT.getScalarType() == MVT::i1)
      return DAG.getConstant(1, SDLoc(N), VT);
    return getOnesVector(VT, DAG, SDLoc(N));
  }

  if (InVec.getOpcode() == ISD::BUILD_VECTOR)
    return DAG.getBuildVector(
        VT, SDLoc(N),
        InVec.getNode()->ops().slice(IdxVal, VT.getVectorNumElements()));

  // Try to move vector bitcast after extract_subv by scaling extraction index:
  // extract_subv (bitcast X), Index --> bitcast (extract_subv X, Index')
  // TODO: Move this to DAGCombiner::visitEXTRACT_SUBVECTOR
  if (InVec.getOpcode() == ISD::BITCAST &&
      InVec.getOperand(0).getValueType().isVector()) {
    SDValue SrcOp = InVec.getOperand(0);
    EVT SrcVT = SrcOp.getValueType();
    unsigned SrcNumElts = SrcVT.getVectorNumElements();
    unsigned DestNumElts = InVec.getValueType().getVectorNumElements();
    if ((DestNumElts % SrcNumElts) == 0) {
      unsigned DestSrcRatio = DestNumElts / SrcNumElts;
      if ((VT.getVectorNumElements() % DestSrcRatio) == 0) {
        unsigned NewExtNumElts = VT.getVectorNumElements() / DestSrcRatio;
        EVT NewExtVT = EVT::getVectorVT(*DAG.getContext(),
                                        SrcVT.getScalarType(), NewExtNumElts);
        if ((N->getConstantOperandVal(1) % DestSrcRatio) == 0 &&
            TLI.isOperationLegalOrCustom(ISD::EXTRACT_SUBVECTOR, NewExtVT)) {
          unsigned IndexValScaled = N->getConstantOperandVal(1) / DestSrcRatio;
          SDLoc DL(N);
          SDValue NewIndex = DAG.getIntPtrConstant(IndexValScaled, DL);
          SDValue NewExtract = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, NewExtVT,
                                           SrcOp, NewIndex);
          return DAG.getBitcast(VT, NewExtract);
        }
      }
    }
  }

  // If we are extracting from an insert into a zero vector, replace with a
  // smaller insert into zero if we don't access less than the original
  // subvector. Don't do this for i1 vectors.
  if (VT.getVectorElementType() != MVT::i1 &&
      InVec.getOpcode() == ISD::INSERT_SUBVECTOR && IdxVal == 0 &&
      InVec.hasOneUse() && isNullConstant(InVec.getOperand(2)) &&
      ISD::isBuildVectorAllZeros(InVec.getOperand(0).getNode()) &&
      InVec.getOperand(1).getValueSizeInBits() <= VT.getSizeInBits()) {
    SDLoc DL(N);
    return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT,
                       getZeroVector(VT, Subtarget, DAG, DL),
                       InVec.getOperand(1), InVec.getOperand(2));
  }

  // If we're extracting from a broadcast then we're better off just
  // broadcasting to the smaller type directly, assuming this is the only use.
  // As its a broadcast we don't care about the extraction index.
  if (InVec.getOpcode() == X86ISD::VBROADCAST && InVec.hasOneUse() &&
      InVec.getOperand(0).getValueSizeInBits() <= VT.getSizeInBits())
    return DAG.getNode(X86ISD::VBROADCAST, SDLoc(N), VT, InVec.getOperand(0));

  // If we're extracting the lowest subvector and we're the only user,
  // we may be able to perform this with a smaller vector width.
  if (IdxVal == 0 && InVec.hasOneUse()) {
    unsigned InOpcode = InVec.getOpcode();
    if (VT == MVT::v2f64 && InVec.getValueType() == MVT::v4f64) {
      // v2f64 CVTDQ2PD(v4i32).
      if (InOpcode == ISD::SINT_TO_FP &&
          InVec.getOperand(0).getValueType() == MVT::v4i32) {
        return DAG.getNode(X86ISD::CVTSI2P, SDLoc(N), VT, InVec.getOperand(0));
      }
      // v2f64 CVTUDQ2PD(v4i32).
      if (InOpcode == ISD::UINT_TO_FP &&
          InVec.getOperand(0).getValueType() == MVT::v4i32) {
        return DAG.getNode(X86ISD::CVTUI2P, SDLoc(N), VT, InVec.getOperand(0));
      }
      // v2f64 CVTPS2PD(v4f32).
      if (InOpcode == ISD::FP_EXTEND &&
          InVec.getOperand(0).getValueType() == MVT::v4f32) {
        return DAG.getNode(X86ISD::VFPEXT, SDLoc(N), VT, InVec.getOperand(0));
      }
    }
    if ((InOpcode == ISD::ANY_EXTEND ||
         InOpcode == ISD::ANY_EXTEND_VECTOR_INREG ||
         InOpcode == ISD::ZERO_EXTEND ||
         InOpcode == ISD::ZERO_EXTEND_VECTOR_INREG ||
         InOpcode == ISD::SIGN_EXTEND ||
         InOpcode == ISD::SIGN_EXTEND_VECTOR_INREG) &&
        VT.is128BitVector() &&
        InVec.getOperand(0).getSimpleValueType().is128BitVector()) {
      unsigned ExtOp = getOpcode_EXTEND_VECTOR_INREG(InOpcode);
      return DAG.getNode(ExtOp, SDLoc(N), VT, InVec.getOperand(0));
    }
    if (InOpcode == ISD::VSELECT &&
        InVec.getOperand(0).getValueType().is256BitVector() &&
        InVec.getOperand(1).getValueType().is256BitVector() &&
        InVec.getOperand(2).getValueType().is256BitVector()) {
      SDLoc DL(N);
      SDValue Ext0 = extractSubVector(InVec.getOperand(0), 0, DAG, DL, 128);
      SDValue Ext1 = extractSubVector(InVec.getOperand(1), 0, DAG, DL, 128);
      SDValue Ext2 = extractSubVector(InVec.getOperand(2), 0, DAG, DL, 128);
      return DAG.getNode(InOpcode, DL, VT, Ext0, Ext1, Ext2);
    }
  }

  return SDValue();
}

SDValue X86::combineScalarToVector(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  SDValue Src = N->getOperand(0);
  SDLoc DL(N);

  // If this is a scalar to vector to v1i1 from an AND with 1, bypass the and.
  // This occurs frequently in our masked scalar intrinsic code and our
  // floating point select lowering with AVX512.
  // TODO: SimplifyDemandedBits instead?
  if (VT == MVT::v1i1 && Src.getOpcode() == ISD::AND && Src.hasOneUse())
    if (auto *C = dyn_cast<ConstantSDNode>(Src.getOperand(1)))
      if (C->getAPIntValue().isOneValue())
        return DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, MVT::v1i1,
                           Src.getOperand(0));

  // Combine scalar_to_vector of an extract_vector_elt into an extract_subvec.
  if (VT == MVT::v1i1 && Src.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
      Src.hasOneUse() && Src.getOperand(0).getValueType().isVector() &&
      Src.getOperand(0).getValueType().getVectorElementType() == MVT::i1)
    if (auto *C = dyn_cast<ConstantSDNode>(Src.getOperand(1)))
      if (C->isNullValue())
        return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, Src.getOperand(0),
                           Src.getOperand(1));

  // Reduce v2i64 to v4i32 if we don't need the upper bits.
  // TODO: Move to DAGCombine?
  if (VT == MVT::v2i64 && Src.getOpcode() == ISD::ANY_EXTEND &&
      Src.getValueType() == MVT::i64 && Src.hasOneUse() &&
      Src.getOperand(0).getScalarValueSizeInBits() <= 32)
    return DAG.getBitcast(
        VT, DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, MVT::v4i32,
                        DAG.getAnyExtOrTrunc(Src.getOperand(0), DL, MVT::i32)));

  return SDValue();
}

// Simplify PMULDQ and PMULUDQ operations.
SDValue X86::combinePMULDQ(SDNode *N, SelectionDAG &DAG,
                           TargetLowering::DAGCombinerInfo &DCI,
                           const X86Subtarget &Subtarget) {
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // Canonicalize constant to RHS.
  if (DAG.isConstantIntBuildVectorOrConstantInt(LHS) &&
      !DAG.isConstantIntBuildVectorOrConstantInt(RHS))
    return DAG.getNode(N->getOpcode(), SDLoc(N), N->getValueType(0), RHS, LHS);

  // Multiply by zero.
  // Don't return RHS as it may contain UNDEFs.
  if (ISD::isBuildVectorAllZeros(RHS.getNode()))
    return DAG.getConstant(0, SDLoc(N), N->getValueType(0));

  // PMULDQ/PMULUDQ only uses lower 32 bits from each vector element.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (TLI.SimplifyDemandedBits(SDValue(N, 0), APInt::getAllOnesValue(64), DCI))
    return SDValue(N, 0);

  // If the input is an extend_invec and the SimplifyDemandedBits call didn't
  // convert it to any_extend_invec, due to the LegalOperations check, do the
  // conversion directly to a vector shuffle manually. This exposes combine
  // opportunities missed by combineExtInVec not calling
  // combineX86ShufflesRecursively on SSE4.1 targets.
  // FIXME: This is basically a hack around several other issues related to
  // ANY_EXTEND_VECTOR_INREG.
  if (N->getValueType(0) == MVT::v2i64 && LHS.hasOneUse() &&
      (LHS.getOpcode() == ISD::ZERO_EXTEND_VECTOR_INREG ||
       LHS.getOpcode() == ISD::SIGN_EXTEND_VECTOR_INREG) &&
      LHS.getOperand(0).getValueType() == MVT::v4i32) {
    SDLoc dl(N);
    LHS = DAG.getVectorShuffle(MVT::v4i32, dl, LHS.getOperand(0),
                               LHS.getOperand(0), { 0, -1, 1, -1 });
    LHS = DAG.getBitcast(MVT::v2i64, LHS);
    return DAG.getNode(N->getOpcode(), dl, MVT::v2i64, LHS, RHS);
  }
  if (N->getValueType(0) == MVT::v2i64 && RHS.hasOneUse() &&
      (RHS.getOpcode() == ISD::ZERO_EXTEND_VECTOR_INREG ||
       RHS.getOpcode() == ISD::SIGN_EXTEND_VECTOR_INREG) &&
      RHS.getOperand(0).getValueType() == MVT::v4i32) {
    SDLoc dl(N);
    RHS = DAG.getVectorShuffle(MVT::v4i32, dl, RHS.getOperand(0),
                               RHS.getOperand(0), { 0, -1, 1, -1 });
    RHS = DAG.getBitcast(MVT::v2i64, RHS);
    return DAG.getNode(N->getOpcode(), dl, MVT::v2i64, LHS, RHS);
  }

  return SDValue();
}

SDValue X86::combineExtInVec(SDNode *N, SelectionDAG &DAG,
                             TargetLowering::DAGCombinerInfo &DCI,
                             const X86Subtarget &Subtarget) {
  EVT VT = N->getValueType(0);
  SDValue In = N->getOperand(0);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Try to merge vector loads and extend_inreg to an extload.
  if (!DCI.isBeforeLegalizeOps() && ISD::isNormalLoad(In.getNode()) &&
      In.hasOneUse()) {
    auto *Ld = cast<LoadSDNode>(In);
    if (Ld->isSimple()) {
      MVT SVT = In.getSimpleValueType().getVectorElementType();
      ISD::LoadExtType Ext = N->getOpcode() == ISD::SIGN_EXTEND_VECTOR_INREG ? ISD::SEXTLOAD : ISD::ZEXTLOAD;
      EVT MemVT = EVT::getVectorVT(*DAG.getContext(), SVT,
                                   VT.getVectorNumElements());
      if (TLI.isLoadExtLegal(Ext, VT, MemVT)) {
        SDValue Load =
            DAG.getExtLoad(Ext, SDLoc(N), VT, Ld->getChain(), Ld->getBasePtr(),
                           Ld->getPointerInfo(), MemVT, Ld->getAlignment(),
                           Ld->getMemOperand()->getFlags());
        DAG.ReplaceAllUsesOfValueWith(SDValue(Ld, 1), Load.getValue(1));
        return Load;
      }
    }
  }

  // Combine (ext_invec (ext_invec X)) -> (ext_invec X)
  // Disabling for widening legalization for now. We can enable if we find a
  // case that needs it. Otherwise it can be deleted when we switch to
  // widening legalization.
  if (!ExperimentalVectorWideningLegalization &&
      In.getOpcode() == N->getOpcode() &&
      TLI.isTypeLegal(VT) && TLI.isTypeLegal(In.getOperand(0).getValueType()))
    return DAG.getNode(N->getOpcode(), SDLoc(N), VT, In.getOperand(0));

  // Attempt to combine as a shuffle.
  // TODO: SSE41 support
  if (Subtarget.hasAVX() && N->getOpcode() != ISD::SIGN_EXTEND_VECTOR_INREG) {
    SDValue Op(N, 0);
    if (TLI.isTypeLegal(VT) && TLI.isTypeLegal(In.getValueType()))
      if (SDValue Res = combineX86ShufflesRecursively(Op, DAG, Subtarget))
        return Res;
  }

  return SDValue();
}
bool X86TargetLowering::isDesirableToCombineBuildVectorToShuffleTruncate(
    ArrayRef<int> ShuffleMask, EVT SrcVT, EVT TruncVT) const {

  assert(SrcVT.getVectorNumElements() == ShuffleMask.size() &&
         "Element count mismatch");
  assert(
      Subtarget.getTargetLowering()->isShuffleMaskLegal(ShuffleMask, SrcVT) &&
      "Shuffle Mask expected to be legal");

  // For 32-bit elements VPERMD is better than shuffle+truncate.
  // TODO: After we improve lowerBuildVector, add execption for VPERMW.
  if (SrcVT.getScalarSizeInBits() == 32 || !Subtarget.hasAVX2())
    return false;

  if (is128BitLaneCrossingShuffleMask(SrcVT.getSimpleVT(), ShuffleMask))
    return false;

  return true;
}
