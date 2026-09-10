// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_CAPI_LLVM_H
#define FLYDSL_CAPI_LLVM_H

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Set an LLVM cl::opt value at runtime and return the previous value.
/// Returns 0 on success, 1 if the option was not found, 2 on type mismatch.
MLIR_CAPI_EXPORTED int flydslSetLLVMOptionBool(const char *name, bool value, bool *oldValue);
MLIR_CAPI_EXPORTED int flydslSetLLVMOptionInt(const char *name, int value, int *oldValue);
MLIR_CAPI_EXPORTED int flydslSetLLVMOptionStr(const char *name, const char *value, char **oldValue);
MLIR_CAPI_EXPORTED void flydslFreeLLVMOptionStr(char *str);

/// Query the size and member count of a symbolic LLVM MC register class.
/// Returns false when the target or class is unavailable in this build.
MLIR_CAPI_EXPORTED bool flydslGetRegisterClassLayout(const char *target, const char *name,
                                                     unsigned *bits, unsigned *count);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_CAPI_LLVM_H
