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

/// JIT-link an object written by export_to_c into this process after loading
/// `libs` globally. Returns an opaque handle, or null after reporting an error
/// through `onError`.
MLIR_CAPI_EXPORTED void *flydslLoadHostObject(MlirStringRef object, const MlirStringRef *libs,
                                              intptr_t numLibs, MlirStringCallback onError,
                                              void *userData);

/// Address of the symbol `name` defined by a loaded object; 0 after reporting
/// an error through `onError`.
MLIR_CAPI_EXPORTED uint64_t flydslHostObjectLookup(void *handle, MlirStringRef name,
                                                   MlirStringCallback onError, void *userData);

/// Release a loaded object; its code must no longer run.
MLIR_CAPI_EXPORTED void flydslDestroyHostObject(void *handle);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_CAPI_HOST_OBJECT_H
