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

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYROCDLDIALECT_H
