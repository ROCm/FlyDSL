//===- PreferAGPRForDSRead.h --------------------------------------*- C++-*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// FlyDSL-owned AMDGPU MachineFunctionPass.  Lives entirely inside FlyDSL's
// libraries and is injected into LLVM AMDGPU codegen at runtime via
// TargetPassConfig::insertPass(...) — no LLVM source patch required.
//
//===----------------------------------------------------------------------===//

#ifndef FLYDSL_CODEGEN_AMDGPU_PREFERAGPRFORDSREAD_H
#define FLYDSL_CODEGEN_AMDGPU_PREFERAGPRFORDSREAD_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/PassRegistry.h"

namespace llvm {

class FlyAMDGPUPreferAGPRForDSReadPass
    : public PassInfoMixin<FlyAMDGPUPreferAGPRForDSReadPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

void initializeFlyAMDGPUPreferAGPRForDSReadLegacyPass(PassRegistry &);
extern char &FlyAMDGPUPreferAGPRForDSReadLegacyID;

} // namespace llvm

#endif // FLYDSL_CODEGEN_AMDGPU_PREFERAGPRFORDSREAD_H
