//===- FlyAMDGPUTarget.h ------------------------------------------*- C++-*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Registers a FlyDSL-owned implementation of `gpu::TargetAttrInterface` on
// the upstream `#rocdl.target` attribute, REPLACING upstream's impl.  The
// FlyDSL impl is identical to upstream's except that it injects two extra
// AMDGPU MachineFunctionPasses (FlyAMDGPUPreferAGPRForDSRead and
// FlyAMDGPUMFMATieVDSTToSrc2) into the codegen pipeline at PreRegAlloc.
//
// Call order matters: this must be called *after*
// `mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(registry)`,
// because the last-attached external model wins.
//
//===----------------------------------------------------------------------===//

#ifndef FLYDSL_DIALECT_FLYROCDL_TARGET_FLYAMDGPUTARGET_H
#define FLYDSL_DIALECT_FLYROCDL_TARGET_FLYAMDGPUTARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace flydsl {

/// Registers FlyDSL's `gpu::TargetAttrInterface` external model on
/// `#rocdl.target` in the given DialectRegistry.
void registerFlyAMDGPUTargetInterfaceExternalModels(
    mlir::DialectRegistry &registry);

/// Same, but operates on the registry of an existing MLIRContext.
void registerFlyAMDGPUTargetInterfaceExternalModels(mlir::MLIRContext &context);

} // namespace flydsl

#endif // FLYDSL_DIALECT_FLYROCDL_TARGET_FLYAMDGPUTARGET_H
