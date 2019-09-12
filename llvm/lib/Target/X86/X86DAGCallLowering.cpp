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

#include "X86CallingConv.h"
#include "X86ISelLowering.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "x86-isel"

STATISTIC(NumTailCalls, "Number of tail calls");

/// Call this when the user attempts to do something unsupported, like
/// returning a double without SSE2 enabled on x86_64. This is not fatal, unlike
/// report_fatal_error, so calling code should attempt to recover without
/// crashing.
static void errorUnsupported(SelectionDAG &DAG, const SDLoc &dl,
                             const char *Msg) {
  MachineFunction &MF = DAG.getMachineFunction();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(MF.getFunction(), Msg, dl.getDebugLoc()));
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

bool X86TargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_X86);
}

const MCPhysReg *X86TargetLowering::getScratchRegisters(CallingConv::ID) const {
  static const MCPhysReg ScratchRegs[] = { X86::R11, 0 };
  return ScratchRegs;
}

/// Lowers masks values (v*i1) to the local register values
/// \returns DAG node after lowering to register type
static SDValue lowerMasksToReg(const SDValue &ValArg, const EVT &ValLoc,
                               const SDLoc &Dl, SelectionDAG &DAG) {
  EVT ValVT = ValArg.getValueType();

  if (ValVT == MVT::v1i1)
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, Dl, ValLoc, ValArg,
                       DAG.getIntPtrConstant(0, Dl));

  if ((ValVT == MVT::v8i1 && (ValLoc == MVT::i8 || ValLoc == MVT::i32)) ||
      (ValVT == MVT::v16i1 && (ValLoc == MVT::i16 || ValLoc == MVT::i32))) {
    // Two stage lowering might be required
    // bitcast:   v8i1 -> i8 / v16i1 -> i16
    // anyextend: i8   -> i32 / i16   -> i32
    EVT TempValLoc = ValVT == MVT::v8i1 ? MVT::i8 : MVT::i16;
    SDValue ValToCopy = DAG.getBitcast(TempValLoc, ValArg);
    if (ValLoc == MVT::i32)
      ValToCopy = DAG.getNode(ISD::ANY_EXTEND, Dl, ValLoc, ValToCopy);
    return ValToCopy;
  }

  if ((ValVT == MVT::v32i1 && ValLoc == MVT::i32) ||
      (ValVT == MVT::v64i1 && ValLoc == MVT::i64)) {
    // One stage lowering is required
    // bitcast:   v32i1 -> i32 / v64i1 -> i64
    return DAG.getBitcast(ValLoc, ValArg);
  }

  return DAG.getNode(ISD::ANY_EXTEND, Dl, ValLoc, ValArg);
}

/// Breaks v64i1 value into two registers and adds the new node to the DAG
static void Passv64i1ArgInRegs(
    const SDLoc &Dl, SelectionDAG &DAG, SDValue Chain, SDValue &Arg,
    SmallVector<std::pair<unsigned, SDValue>, 8> &RegsToPass, CCValAssign &VA,
    CCValAssign &NextVA, const X86Subtarget &Subtarget) {
  assert(Subtarget.hasBWI() && "Expected AVX512BW target!");
  assert(Subtarget.is32Bit() && "Expecting 32 bit target");
  assert(Arg.getValueType() == MVT::i64 && "Expecting 64 bit value");
  assert(VA.isRegLoc() && NextVA.isRegLoc() &&
         "The value should reside in two registers");

  // Before splitting the value we cast it to i64
  Arg = DAG.getBitcast(MVT::i64, Arg);

  // Splitting the value into two i32 types
  SDValue Lo, Hi;
  Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, Dl, MVT::i32, Arg,
                   DAG.getConstant(0, Dl, MVT::i32));
  Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, Dl, MVT::i32, Arg,
                   DAG.getConstant(1, Dl, MVT::i32));

  // Attach the two i32 types into corresponding registers
  RegsToPass.push_back(std::make_pair(VA.getLocReg(), Lo));
  RegsToPass.push_back(std::make_pair(NextVA.getLocReg(), Hi));
}

SDValue
X86TargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               const SDLoc &dl, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();

  // In some cases we need to disable registers from the default CSR list.
  // For example, when they are used for argument passing.
  bool ShouldDisableCalleeSavedRegister =
      CallConv == CallingConv::X86_RegCall ||
      MF.getFunction().hasFnAttribute("no_caller_saved_registers");

  if (CallConv == CallingConv::X86_INTR && !Outs.empty())
    report_fatal_error("X86 interrupts may not return any value");

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, RVLocs, *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_X86);

  SDValue Flag;
  SmallVector<SDValue, 6> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)
  // Operand #1 = Bytes To Pop
  RetOps.push_back(DAG.getTargetConstant(FuncInfo->getBytesToPopOnReturn(), dl,
                   MVT::i32));

  // Copy the result values into the output registers.
  for (unsigned I = 0, OutsIndex = 0, E = RVLocs.size(); I != E;
       ++I, ++OutsIndex) {
    CCValAssign &VA = RVLocs[I];
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Add the register to the CalleeSaveDisableRegs list.
    if (ShouldDisableCalleeSavedRegister)
      MF.getRegInfo().disableCalleeSavedRegister(VA.getLocReg());

    SDValue ValToCopy = OutVals[OutsIndex];
    EVT ValVT = ValToCopy.getValueType();

    // Promote values to the appropriate types.
    if (VA.getLocInfo() == CCValAssign::SExt)
      ValToCopy = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), ValToCopy);
    else if (VA.getLocInfo() == CCValAssign::ZExt)
      ValToCopy = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), ValToCopy);
    else if (VA.getLocInfo() == CCValAssign::AExt) {
      if (ValVT.isVector() && ValVT.getVectorElementType() == MVT::i1)
        ValToCopy = lowerMasksToReg(ValToCopy, VA.getLocVT(), dl, DAG);
      else
        ValToCopy = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), ValToCopy);
    }
    else if (VA.getLocInfo() == CCValAssign::BCvt)
      ValToCopy = DAG.getBitcast(VA.getLocVT(), ValToCopy);

    assert(VA.getLocInfo() != CCValAssign::FPExt &&
           "Unexpected FP-extend for return value.");

    // If this is x86-64, and we disabled SSE, we can't return FP values,
    // or SSE or MMX vectors.
    if ((ValVT == MVT::f32 || ValVT == MVT::f64 ||
         VA.getLocReg() == X86::XMM0 || VA.getLocReg() == X86::XMM1) &&
        (Subtarget.is64Bit() && !Subtarget.hasSSE1())) {
      errorUnsupported(DAG, dl, "SSE register return with SSE disabled");
      VA.convertToReg(X86::FP0); // Set reg to FP0, avoid hitting asserts.
    } else if (ValVT == MVT::f64 &&
               (Subtarget.is64Bit() && !Subtarget.hasSSE2())) {
      // Likewise we can't return F64 values with SSE1 only.  gcc does so, but
      // llvm-gcc has never done it right and no one has noticed, so this
      // should be OK for now.
      errorUnsupported(DAG, dl, "SSE2 register return with SSE2 disabled");
      VA.convertToReg(X86::FP0); // Set reg to FP0, avoid hitting asserts.
    }

    // Returns in ST0/ST1 are handled specially: these are pushed as operands to
    // the RET instruction and handled by the FP Stackifier.
    if (VA.getLocReg() == X86::FP0 ||
        VA.getLocReg() == X86::FP1) {
      // If this is a copy from an xmm register to ST(0), use an FPExtend to
      // change the value to the FP stack register class.
      if (isScalarFPTypeInSSEReg(VA.getValVT()))
        ValToCopy = DAG.getNode(ISD::FP_EXTEND, dl, MVT::f80, ValToCopy);
      RetOps.push_back(ValToCopy);
      // Don't emit a copytoreg.
      continue;
    }

    // 64-bit vector (MMX) values are returned in XMM0 / XMM1 except for v1i64
    // which is returned in RAX / RDX.
    if (Subtarget.is64Bit()) {
      if (ValVT == MVT::x86mmx) {
        if (VA.getLocReg() == X86::XMM0 || VA.getLocReg() == X86::XMM1) {
          ValToCopy = DAG.getBitcast(MVT::i64, ValToCopy);
          ValToCopy = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2i64,
                                  ValToCopy);
          // If we don't have SSE2 available, convert to v4f32 so the generated
          // register is legal.
          if (!Subtarget.hasSSE2())
            ValToCopy = DAG.getBitcast(MVT::v4f32, ValToCopy);
        }
      }
    }

    SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;

    if (VA.needsCustom()) {
      assert(VA.getValVT() == MVT::v64i1 &&
             "Currently the only custom case is when we split v64i1 to 2 regs");

      Passv64i1ArgInRegs(dl, DAG, Chain, ValToCopy, RegsToPass, VA, RVLocs[++I],
                         Subtarget);

      assert(2 == RegsToPass.size() &&
             "Expecting two registers after Pass64BitArgInRegs");

      // Add the second register to the CalleeSaveDisableRegs list.
      if (ShouldDisableCalleeSavedRegister)
        MF.getRegInfo().disableCalleeSavedRegister(RVLocs[I].getLocReg());
    } else {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ValToCopy));
    }

    // Add nodes to the DAG and add the values into the RetOps list
    for (auto &Reg : RegsToPass) {
      Chain = DAG.getCopyToReg(Chain, dl, Reg.first, Reg.second, Flag);
      Flag = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));
    }
  }

  // Swift calling convention does not require we copy the sret argument
  // into %rax/%eax for the return, and SRetReturnReg is not set for Swift.

  // All x86 ABIs require that for returning structs by value we copy
  // the sret argument into %rax/%eax (depending on ABI) for the return.
  // We saved the argument into a virtual register in the entry block,
  // so now we copy the value out and into %rax/%eax.
  //
  // Checking Function.hasStructRetAttr() here is insufficient because the IR
  // may not have an explicit sret argument. If FuncInfo.CanLowerReturn is
  // false, then an sret argument may be implicitly inserted in the SelDAG. In
  // either case FuncInfo->setSRetReturnReg() will have been called.
  if (unsigned SRetReg = FuncInfo->getSRetReturnReg()) {
    // When we have both sret and another return value, we should use the
    // original Chain stored in RetOps[0], instead of the current Chain updated
    // in the above loop. If we only have sret, RetOps[0] equals to Chain.

    // For the case of sret and another return value, we have
    //   Chain_0 at the function entry
    //   Chain_1 = getCopyToReg(Chain_0) in the above loop
    // If we use Chain_1 in getCopyFromReg, we will have
    //   Val = getCopyFromReg(Chain_1)
    //   Chain_2 = getCopyToReg(Chain_1, Val) from below

    // getCopyToReg(Chain_0) will be glued together with
    // getCopyToReg(Chain_1, Val) into Unit A, getCopyFromReg(Chain_1) will be
    // in Unit B, and we will have cyclic dependency between Unit A and Unit B:
    //   Data dependency from Unit B to Unit A due to usage of Val in
    //     getCopyToReg(Chain_1, Val)
    //   Chain dependency from Unit A to Unit B

    // So here, we use RetOps[0] (i.e Chain_0) for getCopyFromReg.
    SDValue Val = DAG.getCopyFromReg(RetOps[0], dl, SRetReg,
                                     getPointerTy(MF.getDataLayout()));

    unsigned RetValReg
        = (Subtarget.is64Bit() && !Subtarget.isTarget64BitILP32()) ?
          X86::RAX : X86::EAX;
    Chain = DAG.getCopyToReg(Chain, dl, RetValReg, Val, Flag);
    Flag = Chain.getValue(1);

    // RAX/EAX now acts like a return value.
    RetOps.push_back(
        DAG.getRegister(RetValReg, getPointerTy(DAG.getDataLayout())));

    // Add the returned register to the CalleeSaveDisableRegs list.
    if (ShouldDisableCalleeSavedRegister)
      MF.getRegInfo().disableCalleeSavedRegister(RetValReg);
  }

  const X86RegisterInfo *TRI = Subtarget.getRegisterInfo();
  const MCPhysReg *I =
      TRI->getCalleeSavedRegsViaCopy(&DAG.getMachineFunction());
  if (I) {
    for (; *I; ++I) {
      if (X86::GR64RegClass.contains(*I))
        RetOps.push_back(DAG.getRegister(*I, MVT::i64));
      else
        llvm_unreachable("Unexpected register class in CSRsViaCopy!");
    }
  }

  RetOps[0] = Chain;  // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  X86ISD::NodeType opcode = X86ISD::RET_FLAG;
  if (CallConv == CallingConv::X86_INTR)
    opcode = X86ISD::IRET;
  return DAG.getNode(opcode, dl, MVT::Other, RetOps);
}

bool X86TargetLowering::isUsedByReturnOnly(SDNode *N, SDValue &Chain) const {
  if (N->getNumValues() != 1 || !N->hasNUsesOfValue(1, 0))
    return false;

  SDValue TCChain = Chain;
  SDNode *Copy = *N->use_begin();
  if (Copy->getOpcode() == ISD::CopyToReg) {
    // If the copy has a glue operand, we conservatively assume it isn't safe to
    // perform a tail call.
    if (Copy->getOperand(Copy->getNumOperands()-1).getValueType() == MVT::Glue)
      return false;
    TCChain = Copy->getOperand(0);
  } else if (Copy->getOpcode() != ISD::FP_EXTEND)
    return false;

  bool HasRet = false;
  for (SDNode::use_iterator UI = Copy->use_begin(), UE = Copy->use_end();
       UI != UE; ++UI) {
    if (UI->getOpcode() != X86ISD::RET_FLAG)
      return false;
    // If we are returning more than one value, we can definitely
    // not make a tail call see PR19530
    if (UI->getNumOperands() > 4)
      return false;
    if (UI->getNumOperands() == 4 &&
        UI->getOperand(UI->getNumOperands()-1).getValueType() != MVT::Glue)
      return false;
    HasRet = true;
  }

  if (!HasRet)
    return false;

  Chain = TCChain;
  return true;
}

EVT X86TargetLowering::getTypeForExtReturn(LLVMContext &Context, EVT VT,
                                           ISD::NodeType ExtendKind) const {
  MVT ReturnMVT = MVT::i32;

  bool Darwin = Subtarget.getTargetTriple().isOSDarwin();
  if (VT == MVT::i1 || (!Darwin && (VT == MVT::i8 || VT == MVT::i16))) {
    // The ABI does not require i1, i8 or i16 to be extended.
    //
    // On Darwin, there is code in the wild relying on Clang's old behaviour of
    // always extending i8/i16 return values, so keep doing that for now.
    // (PR26665).
    ReturnMVT = MVT::i8;
  }

  EVT MinVT = getRegisterType(Context, ReturnMVT);
  return VT.bitsLT(MinVT) ? MinVT : VT;
}

/// Reads two 32 bit registers and creates a 64 bit mask value.
/// \param VA The current 32 bit value that need to be assigned.
/// \param NextVA The next 32 bit value that need to be assigned.
/// \param Root The parent DAG node.
/// \param [in,out] InFlag Represents SDvalue in the parent DAG node for
///                        glue purposes. In the case the DAG is already using
///                        physical register instead of virtual, we should glue
///                        our new SDValue to InFlag SDvalue.
/// \return a new SDvalue of size 64bit.
static SDValue getv64i1Argument(CCValAssign &VA, CCValAssign &NextVA,
                                SDValue &Root, SelectionDAG &DAG,
                                const SDLoc &Dl, const X86Subtarget &Subtarget,
                                SDValue *InFlag = nullptr) {
  assert((Subtarget.hasBWI()) && "Expected AVX512BW target!");
  assert(Subtarget.is32Bit() && "Expecting 32 bit target");
  assert(VA.getValVT() == MVT::v64i1 &&
         "Expecting first location of 64 bit width type");
  assert(NextVA.getValVT() == VA.getValVT() &&
         "The locations should have the same type");
  assert(VA.isRegLoc() && NextVA.isRegLoc() &&
         "The values should reside in two registers");

  SDValue Lo, Hi;
  SDValue ArgValueLo, ArgValueHi;

  MachineFunction &MF = DAG.getMachineFunction();
  const TargetRegisterClass *RC = &X86::GR32RegClass;

  // Read a 32 bit value from the registers.
  if (nullptr == InFlag) {
    // When no physical register is present,
    // create an intermediate virtual register.
    unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
    ArgValueLo = DAG.getCopyFromReg(Root, Dl, Reg, MVT::i32);
    Reg = MF.addLiveIn(NextVA.getLocReg(), RC);
    ArgValueHi = DAG.getCopyFromReg(Root, Dl, Reg, MVT::i32);
  } else {
    // When a physical register is available read the value from it and glue
    // the reads together.
    ArgValueLo =
      DAG.getCopyFromReg(Root, Dl, VA.getLocReg(), MVT::i32, *InFlag);
    *InFlag = ArgValueLo.getValue(2);
    ArgValueHi =
      DAG.getCopyFromReg(Root, Dl, NextVA.getLocReg(), MVT::i32, *InFlag);
    *InFlag = ArgValueHi.getValue(2);
  }

  // Convert the i32 type into v32i1 type.
  Lo = DAG.getBitcast(MVT::v32i1, ArgValueLo);

  // Convert the i32 type into v32i1 type.
  Hi = DAG.getBitcast(MVT::v32i1, ArgValueHi);

  // Concatenate the two values together.
  return DAG.getNode(ISD::CONCAT_VECTORS, Dl, MVT::v64i1, Lo, Hi);
}

/// The function will lower a register of various sizes (8/16/32/64)
/// to a mask value of the expected size (v8i1/v16i1/v32i1/v64i1)
/// \returns a DAG node contains the operand after lowering to mask type.
static SDValue lowerRegToMasks(const SDValue &ValArg, const EVT &ValVT,
                               const EVT &ValLoc, const SDLoc &Dl,
                               SelectionDAG &DAG) {
  SDValue ValReturned = ValArg;

  if (ValVT == MVT::v1i1)
    return DAG.getNode(ISD::SCALAR_TO_VECTOR, Dl, MVT::v1i1, ValReturned);

  if (ValVT == MVT::v64i1) {
    // In 32 bit machine, this case is handled by getv64i1Argument
    assert(ValLoc == MVT::i64 && "Expecting only i64 locations");
    // In 64 bit machine, There is no need to truncate the value only bitcast
  } else {
    MVT maskLen;
    switch (ValVT.getSimpleVT().SimpleTy) {
    case MVT::v8i1:
      maskLen = MVT::i8;
      break;
    case MVT::v16i1:
      maskLen = MVT::i16;
      break;
    case MVT::v32i1:
      maskLen = MVT::i32;
      break;
    default:
      llvm_unreachable("Expecting a vector of i1 types");
    }

    ValReturned = DAG.getNode(ISD::TRUNCATE, Dl, maskLen, ValReturned);
  }
  return DAG.getBitcast(ValVT, ValReturned);
}

/// Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
///
SDValue X86TargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals,
    uint32_t *RegMask) const {

  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  bool Is64Bit = Subtarget.is64Bit();
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC_X86);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned I = 0, InsIndex = 0, E = RVLocs.size(); I != E;
       ++I, ++InsIndex) {
    CCValAssign &VA = RVLocs[I];
    EVT CopyVT = VA.getLocVT();

    // In some calling conventions we need to remove the used registers
    // from the register mask.
    if (RegMask) {
      for (MCSubRegIterator SubRegs(VA.getLocReg(), TRI, /*IncludeSelf=*/true);
           SubRegs.isValid(); ++SubRegs)
        RegMask[*SubRegs / 32] &= ~(1u << (*SubRegs % 32));
    }

    // If this is x86-64, and we disabled SSE, we can't return FP values
    if ((CopyVT == MVT::f32 || CopyVT == MVT::f64 || CopyVT == MVT::f128) &&
        ((Is64Bit || Ins[InsIndex].Flags.isInReg()) && !Subtarget.hasSSE1())) {
      errorUnsupported(DAG, dl, "SSE register return with SSE disabled");
      VA.convertToReg(X86::FP0); // Set reg to FP0, avoid hitting asserts.
    }

    // If we prefer to use the value in xmm registers, copy it out as f80 and
    // use a truncate to move it from fp stack reg to xmm reg.
    bool RoundAfterCopy = false;
    if ((VA.getLocReg() == X86::FP0 || VA.getLocReg() == X86::FP1) &&
        isScalarFPTypeInSSEReg(VA.getValVT())) {
      if (!Subtarget.hasX87())
        report_fatal_error("X87 register return with X87 disabled");
      CopyVT = MVT::f80;
      RoundAfterCopy = (CopyVT != VA.getLocVT());
    }

    SDValue Val;
    if (VA.needsCustom()) {
      assert(VA.getValVT() == MVT::v64i1 &&
             "Currently the only custom case is when we split v64i1 to 2 regs");
      Val =
          getv64i1Argument(VA, RVLocs[++I], Chain, DAG, dl, Subtarget, &InFlag);
    } else {
      Chain = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), CopyVT, InFlag)
                  .getValue(1);
      Val = Chain.getValue(0);
      InFlag = Chain.getValue(2);
    }

    if (RoundAfterCopy)
      Val = DAG.getNode(ISD::FP_ROUND, dl, VA.getValVT(), Val,
                        // This truncation won't change the value.
                        DAG.getIntPtrConstant(1, dl));

    if (VA.isExtInLoc() && (VA.getValVT().getScalarType() == MVT::i1)) {
      if (VA.getValVT().isVector() &&
          ((VA.getLocVT() == MVT::i64) || (VA.getLocVT() == MVT::i32) ||
           (VA.getLocVT() == MVT::i16) || (VA.getLocVT() == MVT::i8))) {
        // promoting a mask type (v*i1) into a register of type i64/i32/i16/i8
        Val = lowerRegToMasks(Val, VA.getValVT(), VA.getLocVT(), dl, DAG);
      } else
        Val = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), Val);
    }

    InVals.push_back(Val);
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//                C & StdCall & Fast Calling Convention implementation
//===----------------------------------------------------------------------===//
//  StdCall calling convention seems to be standard for many Windows' API
//  routines and around. It differs from C calling convention just a little:
//  callee should clean up the stack, not caller. Symbols should be also
//  decorated in some fancy way :) It doesn't support any vector arguments.
//  For info on fast calling convention see Fast Calling Convention (tail call)
//  implementation LowerX86_32FastCCCallTo.

/// CallIsStructReturn - Determines whether a call uses struct return
/// semantics.
enum StructReturnType {
  NotStructReturn,
  RegStructReturn,
  StackStructReturn
};
static StructReturnType
callIsStructReturn(ArrayRef<ISD::OutputArg> Outs, bool IsMCU) {
  if (Outs.empty())
    return NotStructReturn;

  const ISD::ArgFlagsTy &Flags = Outs[0].Flags;
  if (!Flags.isSRet())
    return NotStructReturn;
  if (Flags.isInReg() || IsMCU)
    return RegStructReturn;
  return StackStructReturn;
}

/// Determines whether a function uses struct return semantics.
static StructReturnType
argsAreStructReturn(ArrayRef<ISD::InputArg> Ins, bool IsMCU) {
  if (Ins.empty())
    return NotStructReturn;

  const ISD::ArgFlagsTy &Flags = Ins[0].Flags;
  if (!Flags.isSRet())
    return NotStructReturn;
  if (Flags.isInReg() || IsMCU)
    return RegStructReturn;
  return StackStructReturn;
}

/// Make a copy of an aggregate at address specified by "Src" to address
/// "Dst" with size and alignment information specified by the specific
/// parameter attribute. The copy will be passed as a byval function parameter.
static SDValue CreateCopyOfByValArgument(SDValue Src, SDValue Dst,
                                         SDValue Chain, ISD::ArgFlagsTy Flags,
                                         SelectionDAG &DAG, const SDLoc &dl) {
  SDValue SizeNode = DAG.getConstant(Flags.getByValSize(), dl, MVT::i32);

  return DAG.getMemcpy(Chain, dl, Dst, Src, SizeNode, Flags.getByValAlign(),
                       /*isVolatile*/false, /*AlwaysInline=*/true,
                       /*isTailCall*/false,
                       MachinePointerInfo(), MachinePointerInfo());
}

/// Return true if the calling convention is one that we can guarantee TCO for.
static bool canGuaranteeTCO(CallingConv::ID CC) {
  return (CC == CallingConv::Fast || CC == CallingConv::GHC ||
          CC == CallingConv::X86_RegCall || CC == CallingConv::HiPE ||
          CC == CallingConv::HHVM);
}

/// Return true if we might ever do TCO for calls with this calling convention.
static bool mayTailCallThisCC(CallingConv::ID CC) {
  switch (CC) {
  // C calling conventions:
  case CallingConv::C:
  case CallingConv::Win64:
  case CallingConv::X86_64_SysV:
  // Callee pop conventions:
  case CallingConv::X86_ThisCall:
  case CallingConv::X86_StdCall:
  case CallingConv::X86_VectorCall:
  case CallingConv::X86_FastCall:
  // Swift:
  case CallingConv::Swift:
    return true;
  default:
    return canGuaranteeTCO(CC);
  }
}

/// Return true if the function is being made into a tailcall target by
/// changing its ABI.
static bool shouldGuaranteeTCO(CallingConv::ID CC, bool GuaranteedTailCallOpt) {
  return GuaranteedTailCallOpt && canGuaranteeTCO(CC);
}

/// Determines whether the callee is required to pop its own arguments.
/// Callee pop is necessary to support tail calls.
bool X86::isCalleePop(CallingConv::ID CallingConv,
                      bool is64Bit, bool IsVarArg, bool GuaranteeTCO) {
  // If GuaranteeTCO is true, we force some calls to be callee pop so that we
  // can guarantee TCO.
  if (!IsVarArg && shouldGuaranteeTCO(CallingConv, GuaranteeTCO))
    return true;

  switch (CallingConv) {
  default:
    return false;
  case CallingConv::X86_StdCall:
  case CallingConv::X86_FastCall:
  case CallingConv::X86_ThisCall:
  case CallingConv::X86_VectorCall:
    return !is64Bit;
  }
}

bool X86TargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  auto Attr =
      CI->getParent()->getParent()->getFnAttribute("disable-tail-calls");
  if (!CI->isTailCall() || Attr.getValueAsString() == "true")
    return false;

  ImmutableCallSite CS(CI);
  CallingConv::ID CalleeCC = CS.getCallingConv();
  if (!mayTailCallThisCC(CalleeCC))
    return false;

  return true;
}

SDValue
X86TargetLowering::LowerMemArgument(SDValue Chain, CallingConv::ID CallConv,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    const SDLoc &dl, SelectionDAG &DAG,
                                    const CCValAssign &VA,
                                    MachineFrameInfo &MFI, unsigned i) const {
  // Create the nodes corresponding to a load from this parameter slot.
  ISD::ArgFlagsTy Flags = Ins[i].Flags;
  bool AlwaysUseMutable = shouldGuaranteeTCO(
      CallConv, DAG.getTarget().Options.GuaranteedTailCallOpt);
  bool isImmutable = !AlwaysUseMutable && !Flags.isByVal();
  EVT ValVT;
  MVT PtrVT = getPointerTy(DAG.getDataLayout());

  // If value is passed by pointer we have address passed instead of the value
  // itself. No need to extend if the mask value and location share the same
  // absolute size.
  bool ExtendedInMem =
      VA.isExtInLoc() && VA.getValVT().getScalarType() == MVT::i1 &&
      VA.getValVT().getSizeInBits() != VA.getLocVT().getSizeInBits();

  if (VA.getLocInfo() == CCValAssign::Indirect || ExtendedInMem)
    ValVT = VA.getLocVT();
  else
    ValVT = VA.getValVT();

  // FIXME: For now, all byval parameter objects are marked mutable. This can be
  // changed with more analysis.
  // In case of tail call optimization mark all arguments mutable. Since they
  // could be overwritten by lowering of arguments in case of a tail call.
  if (Flags.isByVal()) {
    unsigned Bytes = Flags.getByValSize();
    if (Bytes == 0) Bytes = 1; // Don't create zero-sized stack objects.

    // FIXME: For now, all byval parameter objects are marked as aliasing. This
    // can be improved with deeper analysis.
    int FI = MFI.CreateFixedObject(Bytes, VA.getLocMemOffset(), isImmutable,
                                   /*isAliased=*/true);
    return DAG.getFrameIndex(FI, PtrVT);
  }

  // This is an argument in memory. We might be able to perform copy elision.
  // If the argument is passed directly in memory without any extension, then we
  // can perform copy elision. Large vector types, for example, may be passed
  // indirectly by pointer.
  if (Flags.isCopyElisionCandidate() &&
      VA.getLocInfo() != CCValAssign::Indirect && !ExtendedInMem) {
    EVT ArgVT = Ins[i].ArgVT;
    SDValue PartAddr;
    if (Ins[i].PartOffset == 0) {
      // If this is a one-part value or the first part of a multi-part value,
      // create a stack object for the entire argument value type and return a
      // load from our portion of it. This assumes that if the first part of an
      // argument is in memory, the rest will also be in memory.
      int FI = MFI.CreateFixedObject(ArgVT.getStoreSize(), VA.getLocMemOffset(),
                                     /*IsImmutable=*/false);
      PartAddr = DAG.getFrameIndex(FI, PtrVT);
      return DAG.getLoad(
          ValVT, dl, Chain, PartAddr,
          MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI));
    } else {
      // This is not the first piece of an argument in memory. See if there is
      // already a fixed stack object including this offset. If so, assume it
      // was created by the PartOffset == 0 branch above and create a load from
      // the appropriate offset into it.
      int64_t PartBegin = VA.getLocMemOffset();
      int64_t PartEnd = PartBegin + ValVT.getSizeInBits() / 8;
      int FI = MFI.getObjectIndexBegin();
      for (; MFI.isFixedObjectIndex(FI); ++FI) {
        int64_t ObjBegin = MFI.getObjectOffset(FI);
        int64_t ObjEnd = ObjBegin + MFI.getObjectSize(FI);
        if (ObjBegin <= PartBegin && PartEnd <= ObjEnd)
          break;
      }
      if (MFI.isFixedObjectIndex(FI)) {
        SDValue Addr =
            DAG.getNode(ISD::ADD, dl, PtrVT, DAG.getFrameIndex(FI, PtrVT),
                        DAG.getIntPtrConstant(Ins[i].PartOffset, dl));
        return DAG.getLoad(
            ValVT, dl, Chain, Addr,
            MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI,
                                              Ins[i].PartOffset));
      }
    }
  }

  int FI = MFI.CreateFixedObject(ValVT.getSizeInBits() / 8,
                                 VA.getLocMemOffset(), isImmutable);

  // Set SExt or ZExt flag.
  if (VA.getLocInfo() == CCValAssign::ZExt) {
    MFI.setObjectZExt(FI, true);
  } else if (VA.getLocInfo() == CCValAssign::SExt) {
    MFI.setObjectSExt(FI, true);
  }

  SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
  SDValue Val = DAG.getLoad(
      ValVT, dl, Chain, FIN,
      MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI));
  return ExtendedInMem
             ? (VA.getValVT().isVector()
                    ? DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, VA.getValVT(), Val)
                    : DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), Val))
             : Val;
}

// FIXME: Get this from tablegen.
static ArrayRef<MCPhysReg> get64BitArgumentGPRs(CallingConv::ID CallConv,
                                                const X86Subtarget &Subtarget) {
  assert(Subtarget.is64Bit());

  if (Subtarget.isCallingConvWin64(CallConv)) {
    static const MCPhysReg GPR64ArgRegsWin64[] = {
      X86::RCX, X86::RDX, X86::R8,  X86::R9
    };
    return makeArrayRef(std::begin(GPR64ArgRegsWin64), std::end(GPR64ArgRegsWin64));
  }

  static const MCPhysReg GPR64ArgRegs64Bit[] = {
    X86::RDI, X86::RSI, X86::RDX, X86::RCX, X86::R8, X86::R9
  };
  return makeArrayRef(std::begin(GPR64ArgRegs64Bit), std::end(GPR64ArgRegs64Bit));
}

// FIXME: Get this from tablegen.
static ArrayRef<MCPhysReg> get64BitArgumentXMMs(MachineFunction &MF,
                                                CallingConv::ID CallConv,
                                                const X86Subtarget &Subtarget) {
  assert(Subtarget.is64Bit());
  if (Subtarget.isCallingConvWin64(CallConv)) {
    // The XMM registers which might contain var arg parameters are shadowed
    // in their paired GPR.  So we only need to save the GPR to their home
    // slots.
    // TODO: __vectorcall will change this.
    return None;
  }

  const Function &F = MF.getFunction();
  bool NoImplicitFloatOps = F.hasFnAttribute(Attribute::NoImplicitFloat);
  bool isSoftFloat = Subtarget.useSoftFloat();
  assert(!(isSoftFloat && NoImplicitFloatOps) &&
         "SSE register cannot be used when SSE is disabled!");
  if (isSoftFloat || NoImplicitFloatOps || !Subtarget.hasSSE1())
    // Kernel mode asks for SSE to be disabled, so there are no XMM argument
    // registers.
    return None;

  static const MCPhysReg XMMArgRegs64Bit[] = {
    X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
    X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
  };
  return makeArrayRef(std::begin(XMMArgRegs64Bit), std::end(XMMArgRegs64Bit));
}

#ifndef NDEBUG
static bool isSortedByValueNo(ArrayRef<CCValAssign> ArgLocs) {
  return std::is_sorted(ArgLocs.begin(), ArgLocs.end(),
                        [](const CCValAssign &A, const CCValAssign &B) -> bool {
                          return A.getValNo() < B.getValNo();
                        });
}
#endif

SDValue X86TargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
  const TargetFrameLowering &TFI = *Subtarget.getFrameLowering();

  const Function &F = MF.getFunction();
  if (F.hasExternalLinkage() && Subtarget.isTargetCygMing() &&
      F.getName() == "main")
    FuncInfo->setForceFramePointer(true);

  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool Is64Bit = Subtarget.is64Bit();
  bool IsWin64 = Subtarget.isCallingConvWin64(CallConv);

  assert(
      !(isVarArg && canGuaranteeTCO(CallConv)) &&
      "Var args not supported with calling conv' regcall, fastcc, ghc or hipe");

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, MF, ArgLocs, *DAG.getContext());

  // Allocate shadow area for Win64.
  if (IsWin64)
    CCInfo.AllocateStack(32, 8);

  CCInfo.AnalyzeArguments(Ins, CC_X86);

  // In vectorcall calling convention a second pass is required for the HVA
  // types.
  if (CallingConv::X86_VectorCall == CallConv) {
    CCInfo.AnalyzeArgumentsSecondPass(Ins, CC_X86);
  }

  // The next loop assumes that the locations are in the same order of the
  // input arguments.
  assert(isSortedByValueNo(ArgLocs) &&
         "Argument Location list must be sorted before lowering");

  SDValue ArgValue;
  for (unsigned I = 0, InsIndex = 0, E = ArgLocs.size(); I != E;
       ++I, ++InsIndex) {
    assert(InsIndex < Ins.size() && "Invalid Ins index");
    CCValAssign &VA = ArgLocs[I];

    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      if (VA.needsCustom()) {
        assert(
            VA.getValVT() == MVT::v64i1 &&
            "Currently the only custom case is when we split v64i1 to 2 regs");

        // v64i1 values, in regcall calling convention, that are
        // compiled to 32 bit arch, are split up into two registers.
        ArgValue =
            getv64i1Argument(VA, ArgLocs[++I], Chain, DAG, dl, Subtarget);
      } else {
        const TargetRegisterClass *RC;
        if (RegVT == MVT::i8)
          RC = &X86::GR8RegClass;
        else if (RegVT == MVT::i16)
          RC = &X86::GR16RegClass;
        else if (RegVT == MVT::i32)
          RC = &X86::GR32RegClass;
        else if (Is64Bit && RegVT == MVT::i64)
          RC = &X86::GR64RegClass;
        else if (RegVT == MVT::f32)
          RC = Subtarget.hasAVX512() ? &X86::FR32XRegClass : &X86::FR32RegClass;
        else if (RegVT == MVT::f64)
          RC = Subtarget.hasAVX512() ? &X86::FR64XRegClass : &X86::FR64RegClass;
        else if (RegVT == MVT::f80)
          RC = &X86::RFP80RegClass;
        else if (RegVT == MVT::f128)
          RC = &X86::VR128RegClass;
        else if (RegVT.is512BitVector())
          RC = &X86::VR512RegClass;
        else if (RegVT.is256BitVector())
          RC = Subtarget.hasVLX() ? &X86::VR256XRegClass : &X86::VR256RegClass;
        else if (RegVT.is128BitVector())
          RC = Subtarget.hasVLX() ? &X86::VR128XRegClass : &X86::VR128RegClass;
        else if (RegVT == MVT::x86mmx)
          RC = &X86::VR64RegClass;
        else if (RegVT == MVT::v1i1)
          RC = &X86::VK1RegClass;
        else if (RegVT == MVT::v8i1)
          RC = &X86::VK8RegClass;
        else if (RegVT == MVT::v16i1)
          RC = &X86::VK16RegClass;
        else if (RegVT == MVT::v32i1)
          RC = &X86::VK32RegClass;
        else if (RegVT == MVT::v64i1)
          RC = &X86::VK64RegClass;
        else
          llvm_unreachable("Unknown argument type!");

        unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);
        ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);
      }

      // If this is an 8 or 16-bit value, it is really passed promoted to 32
      // bits.  Insert an assert[sz]ext to capture this, then truncate to the
      // right size.
      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::BCvt)
        ArgValue = DAG.getBitcast(VA.getValVT(), ArgValue);

      if (VA.isExtInLoc()) {
        // Handle MMX values passed in XMM regs.
        if (RegVT.isVector() && VA.getValVT().getScalarType() != MVT::i1)
          ArgValue = DAG.getNode(X86ISD::MOVDQ2Q, dl, VA.getValVT(), ArgValue);
        else if (VA.getValVT().isVector() &&
                 VA.getValVT().getScalarType() == MVT::i1 &&
                 ((VA.getLocVT() == MVT::i64) || (VA.getLocVT() == MVT::i32) ||
                  (VA.getLocVT() == MVT::i16) || (VA.getLocVT() == MVT::i8))) {
          // Promoting a mask type (v*i1) into a register of type i64/i32/i16/i8
          ArgValue = lowerRegToMasks(ArgValue, VA.getValVT(), RegVT, dl, DAG);
        } else
          ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
      }
    } else {
      assert(VA.isMemLoc());
      ArgValue =
          LowerMemArgument(Chain, CallConv, Ins, dl, DAG, VA, MFI, InsIndex);
    }

    // If value is passed via pointer - do a load.
    if (VA.getLocInfo() == CCValAssign::Indirect && !Ins[I].Flags.isByVal())
      ArgValue =
          DAG.getLoad(VA.getValVT(), dl, Chain, ArgValue, MachinePointerInfo());

    InVals.push_back(ArgValue);
  }

  for (unsigned I = 0, E = Ins.size(); I != E; ++I) {
    // Swift calling convention does not require we copy the sret argument
    // into %rax/%eax for the return. We don't set SRetReturnReg for Swift.
    if (CallConv == CallingConv::Swift)
      continue;

    // All x86 ABIs require that for returning structs by value we copy the
    // sret argument into %rax/%eax (depending on ABI) for the return. Save
    // the argument into a virtual register so that we can access it from the
    // return points.
    if (Ins[I].Flags.isSRet()) {
      unsigned Reg = FuncInfo->getSRetReturnReg();
      if (!Reg) {
        MVT PtrTy = getPointerTy(DAG.getDataLayout());
        Reg = MF.getRegInfo().createVirtualRegister(getRegClassFor(PtrTy));
        FuncInfo->setSRetReturnReg(Reg);
      }
      SDValue Copy = DAG.getCopyToReg(DAG.getEntryNode(), dl, Reg, InVals[I]);
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Copy, Chain);
      break;
    }
  }

  unsigned StackSize = CCInfo.getNextStackOffset();
  // Align stack specially for tail calls.
  if (shouldGuaranteeTCO(CallConv,
                         MF.getTarget().Options.GuaranteedTailCallOpt))
    StackSize = GetAlignedArgumentStackSize(StackSize, DAG);

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start. We
  // can skip this if there are no va_start calls.
  if (MFI.hasVAStart() &&
      (Is64Bit || (CallConv != CallingConv::X86_FastCall &&
                   CallConv != CallingConv::X86_ThisCall))) {
    FuncInfo->setVarArgsFrameIndex(MFI.CreateFixedObject(1, StackSize, true));
  }

  // Figure out if XMM registers are in use.
  assert(!(Subtarget.useSoftFloat() &&
           F.hasFnAttribute(Attribute::NoImplicitFloat)) &&
         "SSE register cannot be used when SSE is disabled!");

  // 64-bit calling conventions support varargs and register parameters, so we
  // have to do extra work to spill them in the prologue.
  if (Is64Bit && isVarArg && MFI.hasVAStart()) {
    // Find the first unallocated argument registers.
    ArrayRef<MCPhysReg> ArgGPRs = get64BitArgumentGPRs(CallConv, Subtarget);
    ArrayRef<MCPhysReg> ArgXMMs = get64BitArgumentXMMs(MF, CallConv, Subtarget);
    unsigned NumIntRegs = CCInfo.getFirstUnallocated(ArgGPRs);
    unsigned NumXMMRegs = CCInfo.getFirstUnallocated(ArgXMMs);
    assert(!(NumXMMRegs && !Subtarget.hasSSE1()) &&
           "SSE register cannot be used when SSE is disabled!");

    // Gather all the live in physical registers.
    SmallVector<SDValue, 6> LiveGPRs;
    SmallVector<SDValue, 8> LiveXMMRegs;
    SDValue ALVal;
    for (MCPhysReg Reg : ArgGPRs.slice(NumIntRegs)) {
      unsigned GPR = MF.addLiveIn(Reg, &X86::GR64RegClass);
      LiveGPRs.push_back(
          DAG.getCopyFromReg(Chain, dl, GPR, MVT::i64));
    }
    if (!ArgXMMs.empty()) {
      unsigned AL = MF.addLiveIn(X86::AL, &X86::GR8RegClass);
      ALVal = DAG.getCopyFromReg(Chain, dl, AL, MVT::i8);
      for (MCPhysReg Reg : ArgXMMs.slice(NumXMMRegs)) {
        unsigned XMMReg = MF.addLiveIn(Reg, &X86::VR128RegClass);
        LiveXMMRegs.push_back(
            DAG.getCopyFromReg(Chain, dl, XMMReg, MVT::v4f32));
      }
    }

    if (IsWin64) {
      // Get to the caller-allocated home save location.  Add 8 to account
      // for the return address.
      int HomeOffset = TFI.getOffsetOfLocalArea() + 8;
      FuncInfo->setRegSaveFrameIndex(
          MFI.CreateFixedObject(1, NumIntRegs * 8 + HomeOffset, false));
      // Fixup to set vararg frame on shadow area (4 x i64).
      if (NumIntRegs < 4)
        FuncInfo->setVarArgsFrameIndex(FuncInfo->getRegSaveFrameIndex());
    } else {
      // For X86-64, if there are vararg parameters that are passed via
      // registers, then we must store them to their spots on the stack so
      // they may be loaded by dereferencing the result of va_next.
      FuncInfo->setVarArgsGPOffset(NumIntRegs * 8);
      FuncInfo->setVarArgsFPOffset(ArgGPRs.size() * 8 + NumXMMRegs * 16);
      FuncInfo->setRegSaveFrameIndex(MFI.CreateStackObject(
          ArgGPRs.size() * 8 + ArgXMMs.size() * 16, 16, false));
    }

    // Store the integer parameter registers.
    SmallVector<SDValue, 8> MemOps;
    SDValue RSFIN = DAG.getFrameIndex(FuncInfo->getRegSaveFrameIndex(),
                                      getPointerTy(DAG.getDataLayout()));
    unsigned Offset = FuncInfo->getVarArgsGPOffset();
    for (SDValue Val : LiveGPRs) {
      SDValue FIN = DAG.getNode(ISD::ADD, dl, getPointerTy(DAG.getDataLayout()),
                                RSFIN, DAG.getIntPtrConstant(Offset, dl));
      SDValue Store =
          DAG.getStore(Val.getValue(1), dl, Val, FIN,
                       MachinePointerInfo::getFixedStack(
                           DAG.getMachineFunction(),
                           FuncInfo->getRegSaveFrameIndex(), Offset));
      MemOps.push_back(Store);
      Offset += 8;
    }

    if (!ArgXMMs.empty() && NumXMMRegs != ArgXMMs.size()) {
      // Now store the XMM (fp + vector) parameter registers.
      SmallVector<SDValue, 12> SaveXMMOps;
      SaveXMMOps.push_back(Chain);
      SaveXMMOps.push_back(ALVal);
      SaveXMMOps.push_back(DAG.getIntPtrConstant(
                             FuncInfo->getRegSaveFrameIndex(), dl));
      SaveXMMOps.push_back(DAG.getIntPtrConstant(
                             FuncInfo->getVarArgsFPOffset(), dl));
      SaveXMMOps.insert(SaveXMMOps.end(), LiveXMMRegs.begin(),
                        LiveXMMRegs.end());
      MemOps.push_back(DAG.getNode(X86ISD::VASTART_SAVE_XMM_REGS, dl,
                                   MVT::Other, SaveXMMOps));
    }

    if (!MemOps.empty())
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, MemOps);
  }

  if (isVarArg && MFI.hasMustTailInVarArgFunc()) {
    // Find the largest legal vector type.
    MVT VecVT = MVT::Other;
    // FIXME: Only some x86_32 calling conventions support AVX512.
    if (Subtarget.useAVX512Regs() &&
        (Is64Bit || (CallConv == CallingConv::X86_VectorCall ||
                     CallConv == CallingConv::Intel_OCL_BI)))
      VecVT = MVT::v16f32;
    else if (Subtarget.hasAVX())
      VecVT = MVT::v8f32;
    else if (Subtarget.hasSSE2())
      VecVT = MVT::v4f32;

    // We forward some GPRs and some vector types.
    SmallVector<MVT, 2> RegParmTypes;
    MVT IntVT = Is64Bit ? MVT::i64 : MVT::i32;
    RegParmTypes.push_back(IntVT);
    if (VecVT != MVT::Other)
      RegParmTypes.push_back(VecVT);

    // Compute the set of forwarded registers. The rest are scratch.
    SmallVectorImpl<ForwardedRegister> &Forwards =
        FuncInfo->getForwardedMustTailRegParms();
    CCInfo.analyzeMustTailForwardedRegisters(Forwards, RegParmTypes, CC_X86);

    // Conservatively forward AL on x86_64, since it might be used for varargs.
    if (Is64Bit && !CCInfo.isAllocated(X86::AL)) {
      unsigned ALVReg = MF.addLiveIn(X86::AL, &X86::GR8RegClass);
      Forwards.push_back(ForwardedRegister(ALVReg, X86::AL, MVT::i8));
    }

    // Copy all forwards from physical to virtual registers.
    for (ForwardedRegister &FR : Forwards) {
      // FIXME: Can we use a less constrained schedule?
      SDValue RegVal = DAG.getCopyFromReg(Chain, dl, FR.VReg, FR.VT);
      FR.VReg = MF.getRegInfo().createVirtualRegister(getRegClassFor(FR.VT));
      Chain = DAG.getCopyToReg(Chain, dl, FR.VReg, RegVal);
    }
  }

  // Some CCs need callee pop.
  if (X86::isCalleePop(CallConv, Is64Bit, isVarArg,
                       MF.getTarget().Options.GuaranteedTailCallOpt)) {
    FuncInfo->setBytesToPopOnReturn(StackSize); // Callee pops everything.
  } else if (CallConv == CallingConv::X86_INTR && Ins.size() == 2) {
    // X86 interrupts must pop the error code (and the alignment padding) if
    // present.
    FuncInfo->setBytesToPopOnReturn(Is64Bit ? 16 : 4);
  } else {
    FuncInfo->setBytesToPopOnReturn(0); // Callee pops nothing.
    // If this is an sret function, the return should pop the hidden pointer.
    if (!Is64Bit && !canGuaranteeTCO(CallConv) &&
        !Subtarget.getTargetTriple().isOSMSVCRT() &&
        argsAreStructReturn(Ins, Subtarget.isTargetMCU()) == StackStructReturn)
      FuncInfo->setBytesToPopOnReturn(4);
  }

  if (!Is64Bit) {
    // RegSaveFrameIndex is X86-64 only.
    FuncInfo->setRegSaveFrameIndex(0xAAAAAAA);
    if (CallConv == CallingConv::X86_FastCall ||
        CallConv == CallingConv::X86_ThisCall)
      // fastcc functions can't have varargs.
      FuncInfo->setVarArgsFrameIndex(0xAAAAAAA);
  }

  FuncInfo->setArgumentStackSize(StackSize);

  if (WinEHFuncInfo *EHInfo = MF.getWinEHFuncInfo()) {
    EHPersonality Personality = classifyEHPersonality(F.getPersonalityFn());
    if (Personality == EHPersonality::CoreCLR) {
      assert(Is64Bit);
      // TODO: Add a mechanism to frame lowering that will allow us to indicate
      // that we'd prefer this slot be allocated towards the bottom of the frame
      // (i.e. near the stack pointer after allocating the frame).  Every
      // funclet needs a copy of this slot in its (mostly empty) frame, and the
      // offset from the bottom of this and each funclet's frame must be the
      // same, so the size of funclets' (mostly empty) frames is dictated by
      // how far this slot is from the bottom (since they allocate just enough
      // space to accommodate holding this slot at the correct offset).
      int PSPSymFI = MFI.CreateStackObject(8, 8, /*isSS=*/false);
      EHInfo->PSPSymFrameIdx = PSPSymFI;
    }
  }

  if (CallConv == CallingConv::X86_RegCall ||
      F.hasFnAttribute("no_caller_saved_registers")) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    for (std::pair<unsigned, unsigned> Pair : MRI.liveins())
      MRI.disableCalleeSavedRegister(Pair.first);
  }

  return Chain;
}

SDValue X86TargetLowering::LowerMemOpCallTo(SDValue Chain, SDValue StackPtr,
                                            SDValue Arg, const SDLoc &dl,
                                            SelectionDAG &DAG,
                                            const CCValAssign &VA,
                                            ISD::ArgFlagsTy Flags) const {
  unsigned LocMemOffset = VA.getLocMemOffset();
  SDValue PtrOff = DAG.getIntPtrConstant(LocMemOffset, dl);
  PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(DAG.getDataLayout()),
                       StackPtr, PtrOff);
  if (Flags.isByVal())
    return CreateCopyOfByValArgument(Arg, PtrOff, Chain, Flags, DAG, dl);

  return DAG.getStore(
      Chain, dl, Arg, PtrOff,
      MachinePointerInfo::getStack(DAG.getMachineFunction(), LocMemOffset));
}

/// Emit a load of return address if tail call
/// optimization is performed and it is required.
SDValue X86TargetLowering::EmitTailCallLoadRetAddr(
    SelectionDAG &DAG, SDValue &OutRetAddr, SDValue Chain, bool IsTailCall,
    bool Is64Bit, int FPDiff, const SDLoc &dl) const {
  // Adjust the Return address stack slot.
  EVT VT = getPointerTy(DAG.getDataLayout());
  OutRetAddr = getReturnAddressFrameIndex(DAG);

  // Load the "old" Return address.
  OutRetAddr = DAG.getLoad(VT, dl, Chain, OutRetAddr, MachinePointerInfo());
  return SDValue(OutRetAddr.getNode(), 1);
}

/// Emit a store of the return address if tail call
/// optimization is performed and it is required (FPDiff!=0).
static SDValue EmitTailCallStoreRetAddr(SelectionDAG &DAG, MachineFunction &MF,
                                        SDValue Chain, SDValue RetAddrFrIdx,
                                        EVT PtrVT, unsigned SlotSize,
                                        int FPDiff, const SDLoc &dl) {
  // Store the return address to the appropriate stack slot.
  if (!FPDiff) return Chain;
  // Calculate the new stack slot for the return address.
  int NewReturnAddrFI =
    MF.getFrameInfo().CreateFixedObject(SlotSize, (int64_t)FPDiff - SlotSize,
                                         false);
  SDValue NewRetAddrFrIdx = DAG.getFrameIndex(NewReturnAddrFI, PtrVT);
  Chain = DAG.getStore(Chain, dl, RetAddrFrIdx, NewRetAddrFrIdx,
                       MachinePointerInfo::getFixedStack(
                           DAG.getMachineFunction(), NewReturnAddrFI));
  return Chain;
}

SDValue
X86TargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                             SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG                     = CLI.DAG;
  SDLoc &dl                             = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals     = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins   = CLI.Ins;
  SDValue Chain                         = CLI.Chain;
  SDValue Callee                        = CLI.Callee;
  CallingConv::ID CallConv              = CLI.CallConv;
  bool &isTailCall                      = CLI.IsTailCall;
  bool isVarArg                         = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  bool Is64Bit        = Subtarget.is64Bit();
  bool IsWin64        = Subtarget.isCallingConvWin64(CallConv);
  StructReturnType SR = callIsStructReturn(Outs, Subtarget.isTargetMCU());
  bool IsSibcall      = false;
  X86MachineFunctionInfo *X86Info = MF.getInfo<X86MachineFunctionInfo>();
  auto Attr = MF.getFunction().getFnAttribute("disable-tail-calls");
  const auto *CI = dyn_cast_or_null<CallInst>(CLI.CS.getInstruction());
  const Function *Fn = CI ? CI->getCalledFunction() : nullptr;
  bool HasNCSR = (CI && CI->hasFnAttr("no_caller_saved_registers")) ||
                 (Fn && Fn->hasFnAttribute("no_caller_saved_registers"));
  const auto *II = dyn_cast_or_null<InvokeInst>(CLI.CS.getInstruction());
  bool HasNoCfCheck =
      (CI && CI->doesNoCfCheck()) || (II && II->doesNoCfCheck());
  const Module *M = MF.getFunction().getParent();
  Metadata *IsCFProtectionSupported = M->getModuleFlag("cf-protection-branch");

  MachineFunction::CallSiteInfo CSInfo;

  if (CallConv == CallingConv::X86_INTR)
    report_fatal_error("X86 interrupts may not be called directly");

  if (Attr.getValueAsString() == "true")
    isTailCall = false;

  if (Subtarget.isPICStyleGOT() &&
      !MF.getTarget().Options.GuaranteedTailCallOpt) {
    // If we are using a GOT, disable tail calls to external symbols with
    // default visibility. Tail calling such a symbol requires using a GOT
    // relocation, which forces early binding of the symbol. This breaks code
    // that require lazy function symbol resolution. Using musttail or
    // GuaranteedTailCallOpt will override this.
    GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee);
    if (!G || (!G->getGlobal()->hasLocalLinkage() &&
               G->getGlobal()->hasDefaultVisibility()))
      isTailCall = false;
  }

  bool IsMustTail = CLI.CS && CLI.CS.isMustTailCall();
  if (IsMustTail) {
    // Force this to be a tail call.  The verifier rules are enough to ensure
    // that we can lower this successfully without moving the return address
    // around.
    isTailCall = true;
  } else if (isTailCall) {
    // Check if it's really possible to do a tail call.
    isTailCall = IsEligibleForTailCallOptimization(Callee, CallConv,
                    isVarArg, SR != NotStructReturn,
                    MF.getFunction().hasStructRetAttr(), CLI.RetTy,
                    Outs, OutVals, Ins, DAG);

    // Sibcalls are automatically detected tailcalls which do not require
    // ABI changes.
    if (!MF.getTarget().Options.GuaranteedTailCallOpt && isTailCall)
      IsSibcall = true;

    if (isTailCall)
      ++NumTailCalls;
  }

  assert(!(isVarArg && canGuaranteeTCO(CallConv)) &&
         "Var args not supported with calling convention fastcc, ghc or hipe");

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, MF, ArgLocs, *DAG.getContext());

  // Allocate shadow area for Win64.
  if (IsWin64)
    CCInfo.AllocateStack(32, 8);

  CCInfo.AnalyzeArguments(Outs, CC_X86);

  // In vectorcall calling convention a second pass is required for the HVA
  // types.
  if (CallingConv::X86_VectorCall == CallConv) {
    CCInfo.AnalyzeArgumentsSecondPass(Outs, CC_X86);
  }

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getAlignedCallFrameSize();
  if (IsSibcall)
    // This is a sibcall. The memory operands are available in caller's
    // own caller's stack.
    NumBytes = 0;
  else if (MF.getTarget().Options.GuaranteedTailCallOpt &&
           canGuaranteeTCO(CallConv))
    NumBytes = GetAlignedArgumentStackSize(NumBytes, DAG);

  int FPDiff = 0;
  if (isTailCall && !IsSibcall && !IsMustTail) {
    // Lower arguments at fp - stackoffset + fpdiff.
    unsigned NumBytesCallerPushed = X86Info->getBytesToPopOnReturn();

    FPDiff = NumBytesCallerPushed - NumBytes;

    // Set the delta of movement of the returnaddr stackslot.
    // But only set if delta is greater than previous delta.
    if (FPDiff < X86Info->getTCReturnAddrDelta())
      X86Info->setTCReturnAddrDelta(FPDiff);
  }

  unsigned NumBytesToPush = NumBytes;
  unsigned NumBytesToPop = NumBytes;

  // If we have an inalloca argument, all stack space has already been allocated
  // for us and be right at the top of the stack.  We don't support multiple
  // arguments passed in memory when using inalloca.
  if (!Outs.empty() && Outs.back().Flags.isInAlloca()) {
    NumBytesToPush = 0;
    if (!ArgLocs.back().isMemLoc())
      report_fatal_error("cannot use inalloca attribute on a register "
                         "parameter");
    if (ArgLocs.back().getLocMemOffset() != 0)
      report_fatal_error("any parameter with the inalloca attribute must be "
                         "the only memory argument");
  }

  if (!IsSibcall)
    Chain = DAG.getCALLSEQ_START(Chain, NumBytesToPush,
                                 NumBytes - NumBytesToPush, dl);

  SDValue RetAddrFrIdx;
  // Load return address for tail calls.
  if (isTailCall && FPDiff)
    Chain = EmitTailCallLoadRetAddr(DAG, RetAddrFrIdx, Chain, isTailCall,
                                    Is64Bit, FPDiff, dl);

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;

  // The next loop assumes that the locations are in the same order of the
  // input arguments.
  assert(isSortedByValueNo(ArgLocs) &&
         "Argument Location list must be sorted before lowering");

  // Walk the register/memloc assignments, inserting copies/loads.  In the case
  // of tail call optimization arguments are handle later.
  const X86RegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  for (unsigned I = 0, OutIndex = 0, E = ArgLocs.size(); I != E;
       ++I, ++OutIndex) {
    assert(OutIndex < Outs.size() && "Invalid Out index");
    // Skip inalloca arguments, they have already been written.
    ISD::ArgFlagsTy Flags = Outs[OutIndex].Flags;
    if (Flags.isInAlloca())
      continue;

    CCValAssign &VA = ArgLocs[I];
    EVT RegVT = VA.getLocVT();
    SDValue Arg = OutVals[OutIndex];
    bool isByVal = Flags.isByVal();

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::AExt:
      if (Arg.getValueType().isVector() &&
          Arg.getValueType().getVectorElementType() == MVT::i1)
        Arg = lowerMasksToReg(Arg, RegVT, dl, DAG);
      else if (RegVT.is128BitVector()) {
        // Special case: passing MMX values in XMM registers.
        Arg = DAG.getBitcast(MVT::i64, Arg);
        Arg = DAG.getNode(ISD::SCALAR_TO_VECTOR, dl, MVT::v2i64, Arg);
        Arg = getMOVL(DAG, dl, MVT::v2i64, DAG.getUNDEF(MVT::v2i64), Arg);
      } else
        Arg = DAG.getNode(ISD::ANY_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getBitcast(RegVT, Arg);
      break;
    case CCValAssign::Indirect: {
      if (isByVal) {
        // Memcpy the argument to a temporary stack slot to prevent
        // the caller from seeing any modifications the callee may make
        // as guaranteed by the `byval` attribute.
        int FrameIdx = MF.getFrameInfo().CreateStackObject(
            Flags.getByValSize(), std::max(16, (int)Flags.getByValAlign()),
            false);
        SDValue StackSlot =
            DAG.getFrameIndex(FrameIdx, getPointerTy(DAG.getDataLayout()));
        Chain =
            CreateCopyOfByValArgument(Arg, StackSlot, Chain, Flags, DAG, dl);
        // From now on treat this as a regular pointer
        Arg = StackSlot;
        isByVal = false;
      } else {
        // Store the argument.
        SDValue SpillSlot = DAG.CreateStackTemporary(VA.getValVT());
        int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
        Chain = DAG.getStore(
            Chain, dl, Arg, SpillSlot,
            MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI));
        Arg = SpillSlot;
      }
      break;
    }
    }

    if (VA.needsCustom()) {
      assert(VA.getValVT() == MVT::v64i1 &&
             "Currently the only custom case is when we split v64i1 to 2 regs");
      // Split v64i1 value into two registers
      Passv64i1ArgInRegs(dl, DAG, Chain, Arg, RegsToPass, VA, ArgLocs[++I],
                         Subtarget);
    } else if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      const TargetOptions &Options = DAG.getTarget().Options;
      if (Options.EnableDebugEntryValues)
        CSInfo.emplace_back(VA.getLocReg(), I);
      if (isVarArg && IsWin64) {
        // Win64 ABI requires argument XMM reg to be copied to the corresponding
        // shadow reg if callee is a varargs function.
        unsigned ShadowReg = 0;
        switch (VA.getLocReg()) {
        case X86::XMM0: ShadowReg = X86::RCX; break;
        case X86::XMM1: ShadowReg = X86::RDX; break;
        case X86::XMM2: ShadowReg = X86::R8; break;
        case X86::XMM3: ShadowReg = X86::R9; break;
        }
        if (ShadowReg)
          RegsToPass.push_back(std::make_pair(ShadowReg, Arg));
      }
    } else if (!IsSibcall && (!isTailCall || isByVal)) {
      assert(VA.isMemLoc());
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, dl, RegInfo->getStackRegister(),
                                      getPointerTy(DAG.getDataLayout()));
      MemOpChains.push_back(LowerMemOpCallTo(Chain, StackPtr, Arg,
                                             dl, DAG, VA, Flags));
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, MemOpChains);

  if (Subtarget.isPICStyleGOT()) {
    // ELF / PIC requires GOT in the EBX register before function calls via PLT
    // GOT pointer.
    if (!isTailCall) {
      RegsToPass.push_back(std::make_pair(
          unsigned(X86::EBX), DAG.getNode(X86ISD::GlobalBaseReg, SDLoc(),
                                          getPointerTy(DAG.getDataLayout()))));
    } else {
      // If we are tail calling and generating PIC/GOT style code load the
      // address of the callee into ECX. The value in ecx is used as target of
      // the tail jump. This is done to circumvent the ebx/callee-saved problem
      // for tail calls on PIC/GOT architectures. Normally we would just put the
      // address of GOT into ebx and then call target@PLT. But for tail calls
      // ebx would be restored (since ebx is callee saved) before jumping to the
      // target@PLT.

      // Note: The actual moving to ECX is done further down.
      GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee);
      if (G && !G->getGlobal()->hasLocalLinkage() &&
          G->getGlobal()->hasDefaultVisibility())
        Callee = LowerGlobalAddress(Callee, DAG);
      else if (isa<ExternalSymbolSDNode>(Callee))
        Callee = LowerExternalSymbol(Callee, DAG);
    }
  }

  if (Is64Bit && isVarArg && !IsWin64 && !IsMustTail) {
    // From AMD64 ABI document:
    // For calls that may call functions that use varargs or stdargs
    // (prototype-less calls or calls to functions containing ellipsis (...) in
    // the declaration) %al is used as hidden argument to specify the number
    // of SSE registers used. The contents of %al do not need to match exactly
    // the number of registers, but must be an ubound on the number of SSE
    // registers used and is in the range 0 - 8 inclusive.

    // Count the number of XMM registers allocated.
    static const MCPhysReg XMMArgRegs[] = {
      X86::XMM0, X86::XMM1, X86::XMM2, X86::XMM3,
      X86::XMM4, X86::XMM5, X86::XMM6, X86::XMM7
    };
    unsigned NumXMMRegs = CCInfo.getFirstUnallocated(XMMArgRegs);
    assert((Subtarget.hasSSE1() || !NumXMMRegs)
           && "SSE registers cannot be used when SSE is disabled");

    RegsToPass.push_back(std::make_pair(unsigned(X86::AL),
                                        DAG.getConstant(NumXMMRegs, dl,
                                                        MVT::i8)));
  }

  if (isVarArg && IsMustTail) {
    const auto &Forwards = X86Info->getForwardedMustTailRegParms();
    for (const auto &F : Forwards) {
      SDValue Val = DAG.getCopyFromReg(Chain, dl, F.VReg, F.VT);
      RegsToPass.push_back(std::make_pair(unsigned(F.PReg), Val));
    }
  }

  // For tail calls lower the arguments to the 'real' stack slots.  Sibcalls
  // don't need this because the eligibility check rejects calls that require
  // shuffling arguments passed in memory.
  if (!IsSibcall && isTailCall) {
    // Force all the incoming stack arguments to be loaded from the stack
    // before any new outgoing arguments are stored to the stack, because the
    // outgoing stack slots may alias the incoming argument stack slots, and
    // the alias isn't otherwise explicit. This is slightly more conservative
    // than necessary, because it means that each store effectively depends
    // on every argument instead of just those arguments it would clobber.
    SDValue ArgChain = DAG.getStackArgumentTokenFactor(Chain);

    SmallVector<SDValue, 8> MemOpChains2;
    SDValue FIN;
    int FI = 0;
    for (unsigned I = 0, OutsIndex = 0, E = ArgLocs.size(); I != E;
         ++I, ++OutsIndex) {
      CCValAssign &VA = ArgLocs[I];

      if (VA.isRegLoc()) {
        if (VA.needsCustom()) {
          assert((CallConv == CallingConv::X86_RegCall) &&
                 "Expecting custom case only in regcall calling convention");
          // This means that we are in special case where one argument was
          // passed through two register locations - Skip the next location
          ++I;
        }

        continue;
      }

      assert(VA.isMemLoc());
      SDValue Arg = OutVals[OutsIndex];
      ISD::ArgFlagsTy Flags = Outs[OutsIndex].Flags;
      // Skip inalloca arguments.  They don't require any work.
      if (Flags.isInAlloca())
        continue;
      // Create frame index.
      int32_t Offset = VA.getLocMemOffset()+FPDiff;
      uint32_t OpSize = (VA.getLocVT().getSizeInBits()+7)/8;
      FI = MF.getFrameInfo().CreateFixedObject(OpSize, Offset, true);
      FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));

      if (Flags.isByVal()) {
        // Copy relative to framepointer.
        SDValue Source = DAG.getIntPtrConstant(VA.getLocMemOffset(), dl);
        if (!StackPtr.getNode())
          StackPtr = DAG.getCopyFromReg(Chain, dl, RegInfo->getStackRegister(),
                                        getPointerTy(DAG.getDataLayout()));
        Source = DAG.getNode(ISD::ADD, dl, getPointerTy(DAG.getDataLayout()),
                             StackPtr, Source);

        MemOpChains2.push_back(CreateCopyOfByValArgument(Source, FIN,
                                                         ArgChain,
                                                         Flags, DAG, dl));
      } else {
        // Store relative to framepointer.
        MemOpChains2.push_back(DAG.getStore(
            ArgChain, dl, Arg, FIN,
            MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI)));
      }
    }

    if (!MemOpChains2.empty())
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, MemOpChains2);

    // Store the return address to the appropriate stack slot.
    Chain = EmitTailCallStoreRetAddr(DAG, MF, Chain, RetAddrFrIdx,
                                     getPointerTy(DAG.getDataLayout()),
                                     RegInfo->getSlotSize(), FPDiff, dl);
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into registers.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  if (DAG.getTarget().getCodeModel() == CodeModel::Large) {
    assert(Is64Bit && "Large code model is only legal in 64-bit mode.");
    // In the 64-bit large code model, we have to make all calls
    // through a register, since the call instruction's 32-bit
    // pc-relative offset may not be large enough to hold the whole
    // address.
  } else if (Callee->getOpcode() == ISD::GlobalAddress ||
             Callee->getOpcode() == ISD::ExternalSymbol) {
    // Lower direct calls to global addresses and external symbols. Setting
    // ForCall to true here has the effect of removing WrapperRIP when possible
    // to allow direct calls to be selected without first materializing the
    // address into a register.
    Callee = LowerGlobalOrExternal(Callee, DAG, /*ForCall=*/true);
  } else if (Subtarget.isTarget64BitILP32() &&
             Callee->getValueType(0) == MVT::i32) {
    // Zero-extend the 32-bit Callee address into a 64-bit according to x32 ABI
    Callee = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i64, Callee);
  }

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;

  if (!IsSibcall && isTailCall) {
    Chain = DAG.getCALLSEQ_END(Chain,
                               DAG.getIntPtrConstant(NumBytesToPop, dl, true),
                               DAG.getIntPtrConstant(0, dl, true), InFlag, dl);
    InFlag = Chain.getValue(1);
  }

  Ops.push_back(Chain);
  Ops.push_back(Callee);

  if (isTailCall)
    Ops.push_back(DAG.getConstant(FPDiff, dl, MVT::i32));

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  // Add a register mask operand representing the call-preserved registers.
  // If HasNCSR is asserted (attribute NoCallerSavedRegisters exists) then we
  // set X86_INTR calling convention because it has the same CSR mask
  // (same preserved registers).
  const uint32_t *Mask = RegInfo->getCallPreservedMask(
      MF, HasNCSR ? (CallingConv::ID)CallingConv::X86_INTR : CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");

  // If this is an invoke in a 32-bit function using a funclet-based
  // personality, assume the function clobbers all registers. If an exception
  // is thrown, the runtime will not restore CSRs.
  // FIXME: Model this more precisely so that we can register allocate across
  // the normal edge and spill and fill across the exceptional edge.
  if (!Is64Bit && CLI.CS && CLI.CS.isInvoke()) {
    const Function &CallerFn = MF.getFunction();
    EHPersonality Pers =
        CallerFn.hasPersonalityFn()
            ? classifyEHPersonality(CallerFn.getPersonalityFn())
            : EHPersonality::Unknown;
    if (isFuncletEHPersonality(Pers))
      Mask = RegInfo->getNoPreservedMask();
  }

  // Define a new register mask from the existing mask.
  uint32_t *RegMask = nullptr;

  // In some calling conventions we need to remove the used physical registers
  // from the reg mask.
  if (CallConv == CallingConv::X86_RegCall || HasNCSR) {
    const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();

    // Allocate a new Reg Mask and copy Mask.
    RegMask = MF.allocateRegMask();
    unsigned RegMaskSize = MachineOperand::getRegMaskSize(TRI->getNumRegs());
    memcpy(RegMask, Mask, sizeof(RegMask[0]) * RegMaskSize);

    // Make sure all sub registers of the argument registers are reset
    // in the RegMask.
    for (auto const &RegPair : RegsToPass)
      for (MCSubRegIterator SubRegs(RegPair.first, TRI, /*IncludeSelf=*/true);
           SubRegs.isValid(); ++SubRegs)
        RegMask[*SubRegs / 32] &= ~(1u << (*SubRegs % 32));

    // Create the RegMask Operand according to our updated mask.
    Ops.push_back(DAG.getRegisterMask(RegMask));
  } else {
    // Create the RegMask Operand according to the static mask.
    Ops.push_back(DAG.getRegisterMask(Mask));
  }

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  if (isTailCall) {
    // We used to do:
    //// If this is the first return lowered for this function, add the regs
    //// to the liveout set for the function.
    // This isn't right, although it's probably harmless on x86; liveouts
    // should be computed from returns not tail calls.  Consider a void
    // function making a tail call to a function returning int.
    MF.getFrameInfo().setHasTailCall();
    SDValue Ret = DAG.getNode(X86ISD::TC_RETURN, dl, NodeTys, Ops);
    DAG.addCallSiteInfo(Ret.getNode(), std::move(CSInfo));
    return Ret;
  }

  if (HasNoCfCheck && IsCFProtectionSupported) {
    Chain = DAG.getNode(X86ISD::NT_CALL, dl, NodeTys, Ops);
  } else {
    Chain = DAG.getNode(X86ISD::CALL, dl, NodeTys, Ops);
  }
  InFlag = Chain.getValue(1);
  DAG.addCallSiteInfo(Chain.getNode(), std::move(CSInfo));

  // Save heapallocsite metadata.
  if (CLI.CS)
    if (MDNode *HeapAlloc = CLI.CS->getMetadata("heapallocsite"))
      DAG.addHeapAllocSite(Chain.getNode(), HeapAlloc);

  // Create the CALLSEQ_END node.
  unsigned NumBytesForCalleeToPop;
  if (X86::isCalleePop(CallConv, Is64Bit, isVarArg,
                       DAG.getTarget().Options.GuaranteedTailCallOpt))
    NumBytesForCalleeToPop = NumBytes;    // Callee pops everything
  else if (!Is64Bit && !canGuaranteeTCO(CallConv) &&
           !Subtarget.getTargetTriple().isOSMSVCRT() &&
           SR == StackStructReturn)
    // If this is a call to a struct-return function, the callee
    // pops the hidden struct pointer, so we have to push it back.
    // This is common for Darwin/X86, Linux & Mingw32 targets.
    // For MSVC Win32 targets, the caller pops the hidden struct pointer.
    NumBytesForCalleeToPop = 4;
  else
    NumBytesForCalleeToPop = 0;  // Callee pops nothing.

  if (CLI.DoesNotReturn && !getTargetMachine().Options.TrapUnreachable) {
    // No need to reset the stack after the call if the call doesn't return. To
    // make the MI verify, we'll pretend the callee does it for us.
    NumBytesForCalleeToPop = NumBytes;
  }

  // Returns a flag for retval copy to use.
  if (!IsSibcall) {
    Chain = DAG.getCALLSEQ_END(Chain,
                               DAG.getIntPtrConstant(NumBytesToPop, dl, true),
                               DAG.getIntPtrConstant(NumBytesForCalleeToPop, dl,
                                                     true),
                               InFlag, dl);
    InFlag = Chain.getValue(1);
  }

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg, Ins, dl, DAG,
                         InVals, RegMask);
}

//===----------------------------------------------------------------------===//
//                Fast Calling Convention (tail call) implementation
//===----------------------------------------------------------------------===//

//  Like std call, callee cleans arguments, convention except that ECX is
//  reserved for storing the tail called function address. Only 2 registers are
//  free for argument passing (inreg). Tail call optimization is performed
//  provided:
//                * tailcallopt is enabled
//                * caller/callee are fastcc
//  On X86_64 architecture with GOT-style position independent code only local
//  (within module) calls are supported at the moment.
//  To keep the stack aligned according to platform abi the function
//  GetAlignedArgumentStackSize ensures that argument delta is always multiples
//  of stack alignment. (Dynamic linkers need this - darwin's dyld for example)
//  If a tail called function callee has more arguments than the caller the
//  caller needs to make sure that there is room to move the RETADDR to. This is
//  achieved by reserving an area the size of the argument delta right after the
//  original RETADDR, but before the saved framepointer or the spilled registers
//  e.g. caller(arg1, arg2) calls callee(arg1, arg2,arg3,arg4)
//  stack layout:
//    arg1
//    arg2
//    RETADDR
//    [ new RETADDR
//      move area ]
//    (possible EBP)
//    ESI
//    EDI
//    local1 ..

/// Make the stack size align e.g 16n + 12 aligned for a 16-byte align
/// requirement.
unsigned
X86TargetLowering::GetAlignedArgumentStackSize(unsigned StackSize,
                                               SelectionDAG& DAG) const {
  const X86RegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  const TargetFrameLowering &TFI = *Subtarget.getFrameLowering();
  unsigned StackAlignment = TFI.getStackAlignment();
  uint64_t AlignMask = StackAlignment - 1;
  int64_t Offset = StackSize;
  unsigned SlotSize = RegInfo->getSlotSize();
  if ( (Offset & AlignMask) <= (StackAlignment - SlotSize) ) {
    // Number smaller than 12 so just add the difference.
    Offset += ((StackAlignment - SlotSize) - (Offset & AlignMask));
  } else {
    // Mask out lower bits, add stackalignment once plus the 12 bytes.
    Offset = ((~AlignMask) & Offset) + StackAlignment +
      (StackAlignment-SlotSize);
  }
  return Offset;
}

/// Return true if the given stack call argument is already available in the
/// same position (relatively) of the caller's incoming argument stack.
static
bool MatchingStackOffset(SDValue Arg, unsigned Offset, ISD::ArgFlagsTy Flags,
                         MachineFrameInfo &MFI, const MachineRegisterInfo *MRI,
                         const X86InstrInfo *TII, const CCValAssign &VA) {
  unsigned Bytes = Arg.getValueSizeInBits() / 8;

  for (;;) {
    // Look through nodes that don't alter the bits of the incoming value.
    unsigned Op = Arg.getOpcode();
    if (Op == ISD::ZERO_EXTEND || Op == ISD::ANY_EXTEND || Op == ISD::BITCAST) {
      Arg = Arg.getOperand(0);
      continue;
    }
    if (Op == ISD::TRUNCATE) {
      const SDValue &TruncInput = Arg.getOperand(0);
      if (TruncInput.getOpcode() == ISD::AssertZext &&
          cast<VTSDNode>(TruncInput.getOperand(1))->getVT() ==
              Arg.getValueType()) {
        Arg = TruncInput.getOperand(0);
        continue;
      }
    }
    break;
  }

  int FI = INT_MAX;
  if (Arg.getOpcode() == ISD::CopyFromReg) {
    unsigned VR = cast<RegisterSDNode>(Arg.getOperand(1))->getReg();
    if (!Register::isVirtualRegister(VR))
      return false;
    MachineInstr *Def = MRI->getVRegDef(VR);
    if (!Def)
      return false;
    if (!Flags.isByVal()) {
      if (!TII->isLoadFromStackSlot(*Def, FI))
        return false;
    } else {
      unsigned Opcode = Def->getOpcode();
      if ((Opcode == X86::LEA32r || Opcode == X86::LEA64r ||
           Opcode == X86::LEA64_32r) &&
          Def->getOperand(1).isFI()) {
        FI = Def->getOperand(1).getIndex();
        Bytes = Flags.getByValSize();
      } else
        return false;
    }
  } else if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(Arg)) {
    if (Flags.isByVal())
      // ByVal argument is passed in as a pointer but it's now being
      // dereferenced. e.g.
      // define @foo(%struct.X* %A) {
      //   tail call @bar(%struct.X* byval %A)
      // }
      return false;
    SDValue Ptr = Ld->getBasePtr();
    FrameIndexSDNode *FINode = dyn_cast<FrameIndexSDNode>(Ptr);
    if (!FINode)
      return false;
    FI = FINode->getIndex();
  } else if (Arg.getOpcode() == ISD::FrameIndex && Flags.isByVal()) {
    FrameIndexSDNode *FINode = cast<FrameIndexSDNode>(Arg);
    FI = FINode->getIndex();
    Bytes = Flags.getByValSize();
  } else
    return false;

  assert(FI != INT_MAX);
  if (!MFI.isFixedObjectIndex(FI))
    return false;

  if (Offset != MFI.getObjectOffset(FI))
    return false;

  // If this is not byval, check that the argument stack object is immutable.
  // inalloca and argument copy elision can create mutable argument stack
  // objects. Byval objects can be mutated, but a byval call intends to pass the
  // mutated memory.
  if (!Flags.isByVal() && !MFI.isImmutableObjectIndex(FI))
    return false;

  if (VA.getLocVT().getSizeInBits() > Arg.getValueSizeInBits()) {
    // If the argument location is wider than the argument type, check that any
    // extension flags match.
    if (Flags.isZExt() != MFI.isObjectZExt(FI) ||
        Flags.isSExt() != MFI.isObjectSExt(FI)) {
      return false;
    }
  }

  return Bytes == MFI.getObjectSize(FI);
}

/// Check whether the call is eligible for tail call optimization. Targets
/// that want to do tail call optimization should implement this function.
bool X86TargetLowering::IsEligibleForTailCallOptimization(
    SDValue Callee, CallingConv::ID CalleeCC, bool isVarArg,
    bool isCalleeStructRet, bool isCallerStructRet, Type *RetTy,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG &DAG) const {
  if (!mayTailCallThisCC(CalleeCC))
    return false;

  // If -tailcallopt is specified, make fastcc functions tail-callable.
  MachineFunction &MF = DAG.getMachineFunction();
  const Function &CallerF = MF.getFunction();

  // If the function return type is x86_fp80 and the callee return type is not,
  // then the FP_EXTEND of the call result is not a nop. It's not safe to
  // perform a tailcall optimization here.
  if (CallerF.getReturnType()->isX86_FP80Ty() && !RetTy->isX86_FP80Ty())
    return false;

  CallingConv::ID CallerCC = CallerF.getCallingConv();
  bool CCMatch = CallerCC == CalleeCC;
  bool IsCalleeWin64 = Subtarget.isCallingConvWin64(CalleeCC);
  bool IsCallerWin64 = Subtarget.isCallingConvWin64(CallerCC);

  // Win64 functions have extra shadow space for argument homing. Don't do the
  // sibcall if the caller and callee have mismatched expectations for this
  // space.
  if (IsCalleeWin64 != IsCallerWin64)
    return false;

  if (DAG.getTarget().Options.GuaranteedTailCallOpt) {
    if (canGuaranteeTCO(CalleeCC) && CCMatch)
      return true;
    return false;
  }

  // Look for obvious safe cases to perform tail call optimization that do not
  // require ABI changes. This is what gcc calls sibcall.

  // Can't do sibcall if stack needs to be dynamically re-aligned. PEI needs to
  // emit a special epilogue.
  const X86RegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  if (RegInfo->needsStackRealignment(MF))
    return false;

  // Also avoid sibcall optimization if either caller or callee uses struct
  // return semantics.
  if (isCalleeStructRet || isCallerStructRet)
    return false;

  // Do not sibcall optimize vararg calls unless all arguments are passed via
  // registers.
  LLVMContext &C = *DAG.getContext();
  if (isVarArg && !Outs.empty()) {
    // Optimizing for varargs on Win64 is unlikely to be safe without
    // additional testing.
    if (IsCalleeWin64 || IsCallerWin64)
      return false;

    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CalleeCC, isVarArg, MF, ArgLocs, C);

    CCInfo.AnalyzeCallOperands(Outs, CC_X86);
    for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i)
      if (!ArgLocs[i].isRegLoc())
        return false;
  }

  // If the call result is in ST0 / ST1, it needs to be popped off the x87
  // stack.  Therefore, if it's not used by the call it is not safe to optimize
  // this into a sibcall.
  bool Unused = false;
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    if (!Ins[i].Used) {
      Unused = true;
      break;
    }
  }
  if (Unused) {
    SmallVector<CCValAssign, 16> RVLocs;
    CCState CCInfo(CalleeCC, false, MF, RVLocs, C);
    CCInfo.AnalyzeCallResult(Ins, RetCC_X86);
    for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
      CCValAssign &VA = RVLocs[i];
      if (VA.getLocReg() == X86::FP0 || VA.getLocReg() == X86::FP1)
        return false;
    }
  }

  // Check that the call results are passed in the same way.
  if (!CCState::resultsCompatible(CalleeCC, CallerCC, MF, C, Ins,
                                  RetCC_X86, RetCC_X86))
    return false;
  // The callee has to preserve all registers the caller needs to preserve.
  const X86RegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);
  if (!CCMatch) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }

  unsigned StackArgsSize = 0;

  // If the callee takes no arguments then go on to check the results of the
  // call.
  if (!Outs.empty()) {
    // Check if stack adjustment is needed. For now, do not do this if any
    // argument is passed on the stack.
    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CalleeCC, isVarArg, MF, ArgLocs, C);

    // Allocate shadow area for Win64
    if (IsCalleeWin64)
      CCInfo.AllocateStack(32, 8);

    CCInfo.AnalyzeCallOperands(Outs, CC_X86);
    StackArgsSize = CCInfo.getNextStackOffset();

    if (CCInfo.getNextStackOffset()) {
      // Check if the arguments are already laid out in the right way as
      // the caller's fixed stack objects.
      MachineFrameInfo &MFI = MF.getFrameInfo();
      const MachineRegisterInfo *MRI = &MF.getRegInfo();
      const X86InstrInfo *TII = Subtarget.getInstrInfo();
      for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
        CCValAssign &VA = ArgLocs[i];
        SDValue Arg = OutVals[i];
        ISD::ArgFlagsTy Flags = Outs[i].Flags;
        if (VA.getLocInfo() == CCValAssign::Indirect)
          return false;
        if (!VA.isRegLoc()) {
          if (!MatchingStackOffset(Arg, VA.getLocMemOffset(), Flags,
                                   MFI, MRI, TII, VA))
            return false;
        }
      }
    }

    bool PositionIndependent = isPositionIndependent();
    // If the tailcall address may be in a register, then make sure it's
    // possible to register allocate for it. In 32-bit, the call address can
    // only target EAX, EDX, or ECX since the tail call must be scheduled after
    // callee-saved registers are restored. These happen to be the same
    // registers used to pass 'inreg' arguments so watch out for those.
    if (!Subtarget.is64Bit() && ((!isa<GlobalAddressSDNode>(Callee) &&
                                  !isa<ExternalSymbolSDNode>(Callee)) ||
                                 PositionIndependent)) {
      unsigned NumInRegs = 0;
      // In PIC we need an extra register to formulate the address computation
      // for the callee.
      unsigned MaxInRegs = PositionIndependent ? 2 : 3;

      for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
        CCValAssign &VA = ArgLocs[i];
        if (!VA.isRegLoc())
          continue;
        Register Reg = VA.getLocReg();
        switch (Reg) {
        default: break;
        case X86::EAX: case X86::EDX: case X86::ECX:
          if (++NumInRegs == MaxInRegs)
            return false;
          break;
        }
      }
    }

    const MachineRegisterInfo &MRI = MF.getRegInfo();
    if (!parametersInCSRMatch(MRI, CallerPreserved, ArgLocs, OutVals))
      return false;
  }

  bool CalleeWillPop =
      X86::isCalleePop(CalleeCC, Subtarget.is64Bit(), isVarArg,
                       MF.getTarget().Options.GuaranteedTailCallOpt);

  if (unsigned BytesToPop =
          MF.getInfo<X86MachineFunctionInfo>()->getBytesToPopOnReturn()) {
    // If we have bytes to pop, the callee must pop them.
    bool CalleePopMatches = CalleeWillPop && BytesToPop == StackArgsSize;
    if (!CalleePopMatches)
      return false;
  } else if (CalleeWillPop && StackArgsSize > 0) {
    // If we don't have bytes to pop, make sure the callee doesn't pop any.
    return false;
  }

  return true;
}
