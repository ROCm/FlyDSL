// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_C_FLYROCDLDIALECT_H
#define FLYDSL_C_FLYROCDLDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl);

MLIR_CAPI_EXPORTED void mlirRegisterFlyToROCDLConversionPass(void);
MLIR_CAPI_EXPORTED void mlirRegisterFlyROCDLClusterAttrPass(void);
MLIR_CAPI_EXPORTED void mlirRegisterFlyROCDLTagAMDGPUCodegenPassesPass(void);

/// Register FlyDSL-owned AMDGPU MachineFunctionPasses with the global LLVM
/// PassRegistry.  Idempotent.  See lib/Codegen/AMDGPU/.
MLIR_CAPI_EXPORTED void mlirRegisterFlyAMDGPUCodegenPasses(void);

/// Attach FlyDSL's gpu::TargetAttrInterface external model to #rocdl.target,
/// REPLACING upstream's impl.  Must be called *before* upstream's
/// mlirRegisterAllDialects (or mlirRegisterROCDLTargetInterfaceExternalModels).
/// MLIR's InterfaceMap silently drops repeated attachInterface calls for the
/// same interface id, so the FIRST registration wins.
MLIR_CAPI_EXPORTED void
mlirRegisterFlyAMDGPUTargetInterfaceExternalModels(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYROCDLDIALECT_H
