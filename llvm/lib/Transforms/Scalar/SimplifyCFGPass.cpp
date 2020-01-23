//===- SimplifyCFGPass.cpp - CFG Simplification Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements dead code elimination and basic block merging, along
// with a collection of other peephole control flow optimizations.  For example:
//
//   * Removes basic blocks with no predecessors.
//   * Merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor.
//   * Eliminates a basic block that only contains an unconditional branch.
//   * Changes invoke instructions to nounwind functions to be calls.
//   * Change things like "if (x) if (y)" into "if (x&y)".
//   * etc..
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Local.h"
#include <utility>
using namespace llvm;

#define DEBUG_TYPE "simplifycfg"

static cl::opt<unsigned> UserBonusInstThreshold(
    "bonus-inst-threshold", cl::Hidden, cl::init(1),
    cl::desc("Control the number of bonus instructions (default = 1)"));

static cl::opt<bool> TailMergeBBsEndingInUnreachable(
    "tail-merge-bbs-ending-in-unreachable", cl::Hidden, cl::init(false),
    cl::desc("Whether to tail merge basic blocks ending in unreachable"));

static cl::opt<bool> UserKeepLoops(
    "keep-loops", cl::Hidden, cl::init(true),
    cl::desc("Preserve canonical loop structure (default = true)"));

static cl::opt<bool> UserSwitchToLookup(
    "switch-to-lookup", cl::Hidden, cl::init(false),
    cl::desc("Convert switches to lookup tables (default = false)"));

static cl::opt<bool> UserForwardSwitchCond(
    "forward-switch-cond", cl::Hidden, cl::init(false),
    cl::desc("Forward switch condition to phi ops (default = false)"));

static cl::opt<bool> UserSinkCommonInsts(
    "sink-common-insts", cl::Hidden, cl::init(false),
    cl::desc("Sink common instructions (default = false)"));


STATISTIC(NumSimpl, "Number of blocks simplified");

/// If we have more than one empty (other than phi node) return blocks,
/// merge them together to promote recursive block merging.
static bool mergeEmptyReturnBlocks(Function &F) {
  bool Changed = false;

  BasicBlock *RetBlock = nullptr;

  // Scan all the blocks in the function, looking for empty return blocks.
  for (Function::iterator BBI = F.begin(), E = F.end(); BBI != E; ) {
    BasicBlock &BB = *BBI++;

    // Only look at return blocks.
    ReturnInst *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
    if (!Ret) continue;

    // Only look at the block if it is empty or the only other thing in it is a
    // single PHI node that is the operand to the return.
    if (Ret != &BB.front()) {
      // Check for something else in the block.
      BasicBlock::iterator I(Ret);
      --I;
      // Skip over debug info.
      while (isa<DbgInfoIntrinsic>(I) && I != BB.begin())
        --I;
      if (!isa<DbgInfoIntrinsic>(I) &&
          (!isa<PHINode>(I) || I != BB.begin() || Ret->getNumOperands() == 0 ||
           Ret->getOperand(0) != &*I))
        continue;
    }

    // If this is the first returning block, remember it and keep going.
    if (!RetBlock) {
      RetBlock = &BB;
      continue;
    }

    // Otherwise, we found a duplicate return block.  Merge the two.
    Changed = true;

    // Case when there is no input to the return or when the returned values
    // agree is trivial.  Note that they can't agree if there are phis in the
    // blocks.
    if (Ret->getNumOperands() == 0 ||
        Ret->getOperand(0) ==
          cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0)) {
      BB.replaceAllUsesWith(RetBlock);
      BB.eraseFromParent();
      continue;
    }

    // If the canonical return block has no PHI node, create one now.
    PHINode *RetBlockPHI = dyn_cast<PHINode>(RetBlock->begin());
    if (!RetBlockPHI) {
      Value *InVal = cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0);
      pred_iterator PB = pred_begin(RetBlock), PE = pred_end(RetBlock);
      RetBlockPHI = PHINode::Create(Ret->getOperand(0)->getType(),
                                    std::distance(PB, PE), "merge",
                                    &RetBlock->front());

      for (pred_iterator PI = PB; PI != PE; ++PI)
        RetBlockPHI->addIncoming(InVal, *PI);
      RetBlock->getTerminator()->setOperand(0, RetBlockPHI);
    }

    // Turn BB into a block that just unconditionally branches to the return
    // block.  This handles the case when the two return blocks have a common
    // predecessor but that return different things.
    RetBlockPHI->addIncoming(Ret->getOperand(0), &BB);
    BB.getTerminator()->eraseFromParent();
    BranchInst::Create(RetBlock, &BB);
  }

  return Changed;
}

/// Advance 'I' to the next non-debug instruction. Stop before the end.
static void iterateNextNonDbgInstr(BasicBlock::iterator &I,
                                   BasicBlock::iterator E) {
  do {
    ++I;
  } while (I != E && isa<DbgInfoIntrinsic>(&*I));
}

static bool isConstantOrPhiOfConstants(const Value *V, const BasicBlock *BB) {
  if (isa<Constant>(V))
    return true;
  if (auto *P = dyn_cast<PHINode>(V)) {
    if (P->getParent() == BB)
      return all_of(P->operands(), [](Value *O) { return isa<Constant>(O); });
  }
  return false;
}

static uint64_t computeBlockHash(BasicBlock &BB) {
  hash_code Hash(0);
  Instruction *Start = BB.getFirstNonPHIOrDbg();
  for (auto I = Start->getIterator(), E = BB.end(); I != E;
       iterateNextNonDbgInstr(I, E)) {
    // An instruction's identity is its opcode and its non-constant operands.
    // FIXME: This could be more precise if the operand is a value inside the
    // block we are considering merging.
    SmallVector<Value *, 4> ValueOperands;
    for (Value *O : I->operands()) {
      if (!isConstantOrPhiOfConstants(O, &BB))
        ValueOperands.push_back(O);
    }

    // Add constant call targets to the identity. We don't want to make indirect
    // calls by phi-ing constant call targets.
    if (auto *Call = dyn_cast<CallInst>(&*I)) {
      if (Function *F = Call->getCalledFunction())
        ValueOperands.push_back(F);
    }

    hash_code OperandHash =
        hash_combine_range(ValueOperands.begin(), ValueOperands.end());
    Hash = hash_combine(Hash, I->getOpcode(), OperandHash);
  }
  return Hash;
}

/// Merge together two noreturn blocks that are almost identical. Allow constant
/// operands and phis of constant operands to differ.
static bool
tailMergeNoReturnBlocksIfProfitable(BasicBlock *BB1, BasicBlock *BB2,
                                    SmallPtrSetImpl<BasicBlock *> &BB1Preds,
                                    unsigned PhiSizeEstimate) {
  // Do one pass to check if this transformation is possible. Skip over phi
  // nodes. They will be checked when they are used within the block, or they
  // must be dead because noreturn blocks don't have any successors.
  auto I1 = BB1->getFirstNonPHIOrDbg()->getIterator();
  auto E1 = BB1->end();
  auto I2 = BB2->getFirstNonPHIOrDbg()->getIterator();
  auto E2 = BB2->end();
  for (; I1 != E1 && I2 != E2;
       iterateNextNonDbgInstr(I1, E1), iterateNextNonDbgInstr(I2, E2)) {
    if (I1->isIdenticalTo(&*I2))
      continue;
    if (!I1->isSameOperationAs(&*I2)) {
      LLVM_DEBUG(dbgs() << "noreturn tail merge of " << BB1->getName()
                        << " and " << BB2->getName()
                        << " failed due to differing instructions:\n"
                        << *I1 << '\n'
                        << *I2 << '\n');
      return false;
    }

    // Operands must be identical, or both be constants. Op1 may be a phi of
    // constants, since we are tail merging many blocks together into it.
    auto O2I = I2->op_begin();
    for (Value *Op1 : I1->operands()) {
      Value *Op2 = *O2I;
      if (!(Op1 == Op2 ||
            (isConstantOrPhiOfConstants(Op1, BB1) &&
             isConstantOrPhiOfConstants(Op2, BB2)))) {
        LLVM_DEBUG(dbgs() << "noreturn tail merge of " << BB1->getName()
                          << " and " << BB2->getName()
                          << " failed due to differing operands:\n"
                          << *Op1 << '\n'
                          << *Op2 << '\n');
        return false;
      }
      ++O2I;
    }
  }

  // If either iterator didn't reach the end, we can't do the transform.
  if (I1 != E1 || I2 != E2) {
    LLVM_DEBUG(dbgs() << "noreturn tail merge of " << BB1->getName() << " and "
                      << BB2->getName()
                      << " failed due to differing block length\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Tail merging " << BB1->getName() << " with "
                    << BB2->getName() << '\n');

  // Delete dead phis in BB1. If left behind, they would become invalid after
  // updating the CFG. All live phis will be updated when we process their uses
  // in this block.
  for (I1 = BB1->begin(); I1 != E1;) {
    if (auto *Phi = dyn_cast<PHINode>(&*I1++)) {
      if (Phi->use_empty())
        Phi->eraseFromParent();
    } else {
      break;
    }
  }

  // If BB1 and BB2 share predecessors and a phi node is required, BB2 cannot be
  // completely replaced with BB1. Instead, BB2 will be an unconditional branch
  // to BB1.
  bool MustUseBranch = false;
  for (BasicBlock *Pred : predecessors(BB2)) {
    if (BB1Preds.count(Pred))
      MustUseBranch = true;
  }

  // In a second pass, add phis for constant operands that differ and remove
  // llvm.dbg.value intrinsics.
  I1 = BB1->getFirstNonPHI()->getIterator();
  I2 = BB2->getFirstNonPHIOrDbg()->getIterator();
  for (; I1 != E1 && I2 != E2; ++I1, iterateNextNonDbgInstr(I2, E2)) {
    // FIXME: Maybe use dbg.value undef instead?
    while (isa<DbgInfoIntrinsic>(&*I1) && I1 != E1)
      (I1++)->eraseFromParent();
    assert(I1->isSameOperationAs(&*I2));

    // Use a merged source location.
    I1->applyMergedLocation(I1->getDebugLoc(), I2->getDebugLoc());

    auto O2I = I2->op_begin();
    for (Use &U : I1->operands()) {
      Value *Op1 = U;
      Value *Op2 = *O2I;
      if (Op1 != Op2) {
        assert(isConstantOrPhiOfConstants(Op1, BB1) &&
               isConstantOrPhiOfConstants(Op2, BB2));

        // We have differing constant operands. Get or create a phi in BB1.
        PHINode *Phi = dyn_cast<PHINode>(Op1);
        if (!Phi) {
          // Make a new phi between constants.
          Constant *C1 = cast<Constant>(Op1);
          Phi = PHINode::Create(C1->getType(), std::min(12U, PhiSizeEstimate),
                                "noreturntail", &*BB1->begin());
          for (BasicBlock *Pred : predecessors(BB1))
            Phi->addIncoming(C1, Pred);
          U.set(Phi);
        }

        // Add phi entries to BB1. If we must use a branch, add one phi entry.
        // Otherwise, add a phi entry for every predecessor of BB2. If BB2 has a
        // phi, we transfer its entries directly. If not, Op2 must be a Constant
        // and it can be used in the entry.
        if (MustUseBranch) {
          Phi->addIncoming(Op2, BB2);
        } else {
          if (PHINode *Phi2 = dyn_cast<PHINode>(Op2)) {
            for (int I = 0, E = Phi2->getNumIncomingValues(); I != E; ++I)
              Phi->addIncoming(Phi2->getIncomingValue(I),
                               Phi2->getIncomingBlock(I));
          } else {
            for (BasicBlock *Pred : predecessors(BB2))
              Phi->addIncoming(Op2, Pred);
          }
        }
      }
      ++O2I;
    }
  }

  if (MustUseBranch) {
    // Clear out non-phis in BB2 and insert a branch to BB1.
    for (I2 = BB2->getFirstNonPHI()->getIterator(); I2 != E2; )
      I2 = I2->eraseFromParent();
    BranchInst::Create(BB1, BB2);
    BB1Preds.insert(BB2);
  } else {
    // BB2 can safely be replaced with BB1. RAUW it and delete it. All phi nodes
    // must be of constants, and should have been handled above.
    for (BasicBlock *Pred : predecessors(BB2))
      BB1Preds.insert(Pred);
    BB2->replaceAllUsesWith(BB1);
    BB2->dropAllReferences();
    BB2->eraseFromParent();
  }

  return true;
}

/// Merge together noreturn blocks with common code. A noreturn block is a block
/// that ends in unreachable preceded by an instruction that could transfer
/// control to a caller or end the program, such as longjmp, __assert_fail,
/// __cxa_throw, or a volatile store to null. Such blocks have no successors but
/// do not return or resume exception handling.
///
/// The goal is to canonicalize this kind of C code to the same thing:
///   if (c1)
///     abort();
///   else if (c2)
///     abort();
/// ->
///   if (c1 || c2)
///     abort();
///
/// TODO: Consider tail merging partial blocks:
///   bb1:
///     call void @foo()
///     call void @abort()
///     unreachable
///   bb2:
///     call void @abort()
///     unreachable
static bool tailMergeBBsEndingInUnreachable(Function &F) {
  // Collect all non-empty BBs ending in unreachable and hash them into buckets.
  MapVector<uint64_t, TinyPtrVector<BasicBlock*>> Buckets;
  for (BasicBlock &BB : F) {
    if (isa<UnreachableInst>(BB.getTerminator()) &&
        BB.getFirstNonPHIOrDbg() != BB.getTerminator()) {
      uint64_t Hash = computeBlockHash(BB);
      auto Insertion = Buckets.insert({Hash, TinyPtrVector<BasicBlock *>(&BB)});
      if (!Insertion.second)
        Insertion.first->second.push_back(&BB);
    }
  }

  if (Buckets.empty())
    return false;

  bool Changed = false;
  for (auto &HashBucket : Buckets) {
    // Attempt to merge the first block in this bucket with every other block in
    // the bucket. Remove the first block and all merged blocks from the bucket,
    // and repeat the process until the bucket is empty.
    TinyPtrVector<BasicBlock *> &Bucket = HashBucket.second;
    SmallPtrSet<BasicBlock *, 4> MergedBBs;
    while (Bucket.size() > 1) {
      LLVM_DEBUG({
        dbgs() << "tailMergeBBsEndingInUnreachable: considering";
        for (auto *BB : Bucket)
          dbgs() << ' ' << BB->getName();
        dbgs() << '\n';
      });
      auto I = Bucket.begin(), E = Bucket.end();
      BasicBlock *CanonicalBlock = *I++;
      MergedBBs.insert(CanonicalBlock);
      SmallPtrSet<BasicBlock *, 4> Predecessors(pred_begin(CanonicalBlock),
                                                pred_end(CanonicalBlock));
      for (; I != E; ++I) {
        if (tailMergeNoReturnBlocksIfProfitable(CanonicalBlock, *I,
                                                Predecessors, Bucket.size())) {
          MergedBBs.insert(*I);
          Changed = true;
        }
      }

      // Erase all merged BBs from the bucket.
      erase_if(Bucket,
               [&MergedBBs](BasicBlock *BB) { return MergedBBs.count(BB); });
      MergedBBs.clear();
    }
  }

  return Changed;
}

/// Call SimplifyCFG on all the blocks in the function,
/// iterating until no more changes are made.
static bool iterativelySimplifyCFG(Function &F, const TargetTransformInfo &TTI,
                                   const SimplifyCFGOptions &Options) {
  bool Changed = false;
  bool LocalChange = true;

  SmallVector<std::pair<const BasicBlock *, const BasicBlock *>, 32> Edges;
  FindFunctionBackedges(F, Edges);
  SmallPtrSet<BasicBlock *, 16> LoopHeaders;
  for (unsigned i = 0, e = Edges.size(); i != e; ++i)
    LoopHeaders.insert(const_cast<BasicBlock *>(Edges[i].second));

  while (LocalChange) {
    LocalChange = false;

    // Loop over all of the basic blocks and remove them if they are unneeded.
    for (Function::iterator BBIt = F.begin(); BBIt != F.end(); ) {
      if (simplifyCFG(&*BBIt++, TTI, Options, &LoopHeaders)) {
        LocalChange = true;
        ++NumSimpl;
      }
    }
    Changed |= LocalChange;
  }
  return Changed;
}

static bool simplifyFunctionCFG(Function &F, const TargetTransformInfo &TTI,
                                const SimplifyCFGOptions &Options) {
  bool EverChanged = removeUnreachableBlocks(F);
  EverChanged |= mergeEmptyReturnBlocks(F);
  if (TailMergeBBsEndingInUnreachable)
    EverChanged |= tailMergeBBsEndingInUnreachable(F);
  EverChanged |= iterativelySimplifyCFG(F, TTI, Options);

  // If neither pass changed anything, we're done.
  if (!EverChanged) return false;

  // iterativelySimplifyCFG can (rarely) make some loops dead.  If this happens,
  // removeUnreachableBlocks is needed to nuke them, which means we should
  // iterate between the two optimizations.  We structure the code like this to
  // avoid rerunning iterativelySimplifyCFG if the second pass of
  // removeUnreachableBlocks doesn't do anything.
  if (!removeUnreachableBlocks(F))
    return true;

  do {
    EverChanged = iterativelySimplifyCFG(F, TTI, Options);
    EverChanged |= removeUnreachableBlocks(F);
  } while (EverChanged);

  return true;
}

// Command-line settings override compile-time settings.
SimplifyCFGPass::SimplifyCFGPass(const SimplifyCFGOptions &Opts) {
  Options.BonusInstThreshold = UserBonusInstThreshold.getNumOccurrences()
                                   ? UserBonusInstThreshold
                                   : Opts.BonusInstThreshold;
  Options.ForwardSwitchCondToPhi = UserForwardSwitchCond.getNumOccurrences()
                                       ? UserForwardSwitchCond
                                       : Opts.ForwardSwitchCondToPhi;
  Options.ConvertSwitchToLookupTable = UserSwitchToLookup.getNumOccurrences()
                                           ? UserSwitchToLookup
                                           : Opts.ConvertSwitchToLookupTable;
  Options.NeedCanonicalLoop = UserKeepLoops.getNumOccurrences()
                                  ? UserKeepLoops
                                  : Opts.NeedCanonicalLoop;
  Options.SinkCommonInsts = UserSinkCommonInsts.getNumOccurrences()
                                ? UserSinkCommonInsts
                                : Opts.SinkCommonInsts;
}

PreservedAnalyses SimplifyCFGPass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  Options.AC = &AM.getResult<AssumptionAnalysis>(F);
  if (!simplifyFunctionCFG(F, TTI, Options))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  return PA;
}

namespace {
struct CFGSimplifyPass : public FunctionPass {
  static char ID;
  SimplifyCFGOptions Options;
  std::function<bool(const Function &)> PredicateFtor;

  CFGSimplifyPass(unsigned Threshold = 1, bool ForwardSwitchCond = false,
                  bool ConvertSwitch = false, bool KeepLoops = true,
                  bool SinkCommon = false,
                  std::function<bool(const Function &)> Ftor = nullptr)
      : FunctionPass(ID), PredicateFtor(std::move(Ftor)) {

    initializeCFGSimplifyPassPass(*PassRegistry::getPassRegistry());

    // Check for command-line overrides of options for debug/customization.
    Options.BonusInstThreshold = UserBonusInstThreshold.getNumOccurrences()
                                    ? UserBonusInstThreshold
                                    : Threshold;

    Options.ForwardSwitchCondToPhi = UserForwardSwitchCond.getNumOccurrences()
                                         ? UserForwardSwitchCond
                                         : ForwardSwitchCond;

    Options.ConvertSwitchToLookupTable = UserSwitchToLookup.getNumOccurrences()
                                             ? UserSwitchToLookup
                                             : ConvertSwitch;

    Options.NeedCanonicalLoop =
        UserKeepLoops.getNumOccurrences() ? UserKeepLoops : KeepLoops;

    Options.SinkCommonInsts = UserSinkCommonInsts.getNumOccurrences()
                                  ? UserSinkCommonInsts
                                  : SinkCommon;
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F) || (PredicateFtor && !PredicateFtor(F)))
      return false;

    Options.AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    return simplifyFunctionCFG(F, TTI, Options);
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char CFGSimplifyPass::ID = 0;
INITIALIZE_PASS_BEGIN(CFGSimplifyPass, "simplifycfg", "Simplify the CFG", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(CFGSimplifyPass, "simplifycfg", "Simplify the CFG", false,
                    false)

// Public interface to the CFGSimplification pass
FunctionPass *
llvm::createCFGSimplificationPass(unsigned Threshold, bool ForwardSwitchCond,
                                  bool ConvertSwitch, bool KeepLoops,
                                  bool SinkCommon,
                                  std::function<bool(const Function &)> Ftor) {
  return new CFGSimplifyPass(Threshold, ForwardSwitchCond, ConvertSwitch,
                             KeepLoops, SinkCommon, std::move(Ftor));
}
