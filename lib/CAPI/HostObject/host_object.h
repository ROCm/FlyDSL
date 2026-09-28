// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CAPI_HOST_OBJECT_H
#define FLYDSL_CAPI_HOST_OBJECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Report the normalized target triple of the host process through `callback`.
MLIR_CAPI_EXPORTED void flydslHostTargetTriple(MlirStringCallback callback, void *userData);

/// Emit a position-independent host object. Reports bytes through `onObject`
/// or an error through `onError`; `module` is not modified.
MLIR_CAPI_EXPORTED MlirLogicalResult flydslEmitHostObject(MlirOperation module, int optLevel,
                                                          MlirStringCallback onObject,
                                                          MlirStringCallback onError,
                                                          void *userData);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_CAPI_HOST_OBJECT_H
