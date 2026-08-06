// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_C_FLYNVVMDIALECT_H
#define FLYDSL_C_FLYNVVMDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyNVVM, fly_nvvm);

MLIR_CAPI_EXPORTED void mlirRegisterFlyToNVVMConversionPass(void);

/// Backend plugin registration: insert all NVVM dialects into \p registry.
MLIR_CAPI_EXPORTED void flydsl_register_nvvm_dialects(MlirDialectRegistry registry);
/// Backend plugin registration: register all NVVM passes.
MLIR_CAPI_EXPORTED void flydsl_register_nvvm_passes(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYNVVMDIALECT_H
