// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_C_REGISTEREVERYTHING_H
#define FLYDSL_C_REGISTEREVERYTHING_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Add the FlyDSL dialects, the dialects of every enabled backend, and their
/// external models to `registry`.
MLIR_CAPI_EXPORTED void flydslRegisterAllDialects(MlirDialectRegistry registry);

/// Register the FlyDSL passes and the passes of every enabled backend.
MLIR_CAPI_EXPORTED void flydslRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_REGISTEREVERYTHING_H
