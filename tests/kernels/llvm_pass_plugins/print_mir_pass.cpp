// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Test fixture for tests/kernels/test_llvm_codegen_pass_e2e.py (compiled at test
// time, not by CMake).
//
// Legacy MachineFunctionPass plugin registering "fly-mir-pass" via RegisterPass.
// Runs pre-emit during codegen and prints the machine-function name (observable
// under `pytest -s`); a no-op otherwise, so it serves as the "same codegen
// driver" baseline for the reorder test.

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
namespace {
struct FlyMirPass : public MachineFunctionPass {
  static char ID;
  FlyMirPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override {
    errs() << "fly-mir-pass: ran on " << MF.getName() << "\n";
    return false;
  }
  StringRef getPassName() const override { return "Fly demo MIR pass"; }
};
char FlyMirPass::ID = 0;
} // namespace
static RegisterPass<FlyMirPass> X("fly-mir-pass", "Fly demo MIR pass", false, false);
