// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Test fixture for tests/kernels/test_llvm_codegen_pass_e2e.py (compiled at test
// time, not by CMake).
//
// Legacy MachineFunctionPass plugin registering "fly-insert-nop": inserts 8
// `s_nop` at the entry of every kernel (count must match NOP_PER_FUNC in the
// test).  The opcode is found by name so no AMDGPU target headers are needed.

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Pass.h"
using namespace llvm;
namespace {
struct FlyInsertNopPass : public MachineFunctionPass {
  static char ID;
  FlyInsertNopPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    unsigned NopOpc = ~0u;
    for (unsigned i = 0, e = TII->getNumOpcodes(); i < e; ++i)
      if (TII->getName(i) == "S_NOP") {
        NopOpc = i;
        break;
      }
    if (NopOpc == ~0u || MF.empty())
      return false;
    MachineBasicBlock &MBB = MF.front();
    auto It = MBB.begin();
    for (int k = 0; k < 8; ++k)
      BuildMI(MBB, It, DebugLoc(), TII->get(NopOpc)).addImm(0);
    return true;
  }
  StringRef getPassName() const override { return "Fly insert NOP MIR pass"; }
};
char FlyInsertNopPass::ID = 0;
} // namespace
static RegisterPass<FlyInsertNopPass> X("fly-insert-nop", "Fly insert NOP", false, false);
