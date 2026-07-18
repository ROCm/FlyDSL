// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Test fixture for tests/kernels/test_llvm_codegen_pass_e2e.py (compiled at test
// time, not by CMake).
//
// Legacy MachineFunctionPass plugin registering "fly-reorder": a tiny scheduler
// that, within each block, swaps adjacent instructions whenever provably safe —
// neither has memory/side effects and no def of one overlaps any register
// operand (def or use, explicit or implicit) of the other.  Semantics are
// preserved (results stay correct) but the emitted instruction order changes.

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Pass.h"
using namespace llvm;
namespace {
static bool unsafe(const MachineInstr &MI) {
  return MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects() || MI.isCall() || MI.isTerminator() ||
         MI.isBranch() || MI.isInlineAsm() || MI.isMetaInstruction();
}
static bool canSwap(const MachineInstr &A, const MachineInstr &B, const TargetRegisterInfo *TRI) {
  if (unsafe(A) || unsafe(B))
    return false;
  auto defConflicts = [&](const MachineInstr &X, const MachineInstr &Y) {
    for (const MachineOperand &dx : X.operands()) {
      if (!dx.isReg() || !dx.getReg() || !dx.isDef())
        continue;
      for (const MachineOperand &oy : Y.operands())
        if (oy.isReg() && oy.getReg() && TRI->regsOverlap(dx.getReg(), oy.getReg()))
          return true;
    }
    return false;
  };
  return !defConflicts(A, B) && !defConflicts(B, A);
}
struct FlyReorderPass : public MachineFunctionPass {
  static char ID;
  FlyReorderPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    bool changed = false;
    for (MachineBasicBlock &MBB : MF) {
      for (auto it = MBB.begin(); it != MBB.end();) {
        auto nxt = std::next(it);
        if (nxt != MBB.end() && canSwap(*it, *nxt, TRI)) {
          MBB.splice(it, &MBB, nxt); // move B before A
          changed = true;
          it = std::next(it); // it still == A; skip past the swapped pair
        } else {
          ++it;
        }
      }
    }
    return changed;
  }
  StringRef getPassName() const override { return "Fly reorder MIR pass"; }
};
char FlyReorderPass::ID = 0;
} // namespace
static RegisterPass<FlyReorderPass> X("fly-reorder", "Fly reorder", false, false);
