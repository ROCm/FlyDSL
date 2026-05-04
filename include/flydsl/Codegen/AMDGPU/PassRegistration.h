//===- PassRegistration.h -----------------------------------------*- C++-*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// One-shot registration of all FlyDSL-owned AMDGPU codegen passes with the
// global LLVM PassRegistry.  Safe to call multiple times.
//
//===----------------------------------------------------------------------===//

#ifndef FLYDSL_CODEGEN_AMDGPU_PASSREGISTRATION_H
#define FLYDSL_CODEGEN_AMDGPU_PASSREGISTRATION_H

namespace llvm {
class PassRegistry;
} // namespace llvm

namespace flydsl {

/// Register all FlyDSL-owned AMDGPU MachineFunctionPasses with `Registry`.
/// Idempotent.  Call once from FlyDSL's MLIR initialization (analogous to
/// `mlirRegisterAllPasses`).
void registerFlyAMDGPUCodegenPasses(llvm::PassRegistry &Registry);

/// Convenience overload that targets the global PassRegistry singleton.
void registerFlyAMDGPUCodegenPasses();

} // namespace flydsl

#endif // FLYDSL_CODEGEN_AMDGPU_PASSREGISTRATION_H
