//===-- PassRegistration.cpp ------------------------------------------------===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
//===----------------------------------------------------------------------===//

#include "flydsl/Codegen/AMDGPU/PassRegistration.h"

#include "flydsl/Codegen/AMDGPU/MFMATieVDSTToSrc2.h"
#include "flydsl/Codegen/AMDGPU/PreferAGPRForDSRead.h"

#include "llvm/PassRegistry.h"

namespace flydsl {

void registerFlyAMDGPUCodegenPasses(llvm::PassRegistry &Registry) {
  llvm::initializeFlyAMDGPUPreferAGPRForDSReadLegacyPass(Registry);
  llvm::initializeFlyAMDGPUMFMATieVDSTToSrc2LegacyPass(Registry);
}

void registerFlyAMDGPUCodegenPasses() {
  registerFlyAMDGPUCodegenPasses(*llvm::PassRegistry::getPassRegistry());
}

} // namespace flydsl
