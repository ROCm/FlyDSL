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

MLIR_CAPI_EXPORTED void mlirRegisterConvertROCDLFastMathOpsPass(void);
MLIR_CAPI_EXPORTED void mlirRegisterFlyToROCDLConversionPass(void);
MLIR_CAPI_EXPORTED void mlirRegisterFlyEmitGPUBinaryPass(void);

/// True when the library was built with in-process LLD (FLYDSL_HAS_LLD_LIBRARY).
MLIR_CAPI_EXPORTED bool flydsl_has_inprocess_lld(void);

/// Link \p objectPath into the HSA code object \p hsacoPath with the in-process
/// LLD the compiler uses.  On failure returns false and writes the linker
/// diagnostics, truncated, into \p error (\p errorSize bytes).  Always fails
/// when flydsl_has_inprocess_lld() is false.
MLIR_CAPI_EXPORTED bool flydsl_link_hsaco(const char *objectPath, const char *hsacoPath,
                                          char *error, size_t errorSize);

/// Backend plugin registration: insert all ROCDL dialects into \p registry.
MLIR_CAPI_EXPORTED void flydsl_register_rocdl_dialects(MlirDialectRegistry registry);
/// Backend plugin registration: register all ROCDL passes.
MLIR_CAPI_EXPORTED void flydsl_register_rocdl_passes(void);

/// The ROCm root gpu-module-to-binary uses when its toolkit option is empty
/// (ROCM_PATH, ROCM_ROOT, ROCM_HOME, then the build's default).
MLIR_CAPI_EXPORTED MlirStringRef mlirFlyROCDLGetROCMPath(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYROCDLDIALECT_H
