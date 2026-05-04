//===- MFMATieVDSTToSrc2.h ----------------------------------------*- C++-*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// FlyDSL-owned AMDGPU MachineFunctionPass.  See PreferAGPRForDSRead.h for the
// runtime injection mechanism.
//
//===----------------------------------------------------------------------===//

#ifndef FLYDSL_CODEGEN_AMDGPU_MFMATIEVDSTTOSRC2_H
#define FLYDSL_CODEGEN_AMDGPU_MFMATIEVDSTTOSRC2_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/PassRegistry.h"

namespace llvm {

class FlyAMDGPUMFMATieVDSTToSrc2Pass
    : public PassInfoMixin<FlyAMDGPUMFMATieVDSTToSrc2Pass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

void initializeFlyAMDGPUMFMATieVDSTToSrc2LegacyPass(PassRegistry &);
extern char &FlyAMDGPUMFMATieVDSTToSrc2LegacyID;

} // namespace llvm

#endif // FLYDSL_CODEGEN_AMDGPU_MFMATIEVDSTTOSRC2_H
