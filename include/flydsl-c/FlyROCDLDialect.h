// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_C_FLYROCDLDIALECT_H
#define FLYDSL_C_FLYROCDLDIALECT_H

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// True when the library was built with in-process LLD (FLYDSL_HAS_LLD_LIBRARY).
MLIR_CAPI_EXPORTED bool mlirFlyROCDLHasInProcessLLD(void);

/// Link \p objectPath into the HSA code object \p hsacoPath with the in-process
/// LLD the compiler uses.  On failure returns false and writes the linker
/// diagnostics, truncated, into \p error (\p errorSize bytes).  Always fails
/// when mlirFlyROCDLHasInProcessLLD() is false.
MLIR_CAPI_EXPORTED bool mlirFlyROCDLLinkHsacoInProcess(const char *objectPath,
                                                       const char *hsacoPath, char *error,
                                                       size_t errorSize);

/// The ROCm root gpu-module-to-binary uses when its toolkit option is empty
/// (ROCM_PATH, ROCM_ROOT, ROCM_HOME, then the build's default).
MLIR_CAPI_EXPORTED MlirStringRef mlirFlyROCDLGetROCMPath(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYROCDLDIALECT_H
