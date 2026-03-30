// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Conversion/Passes.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl, mlir::fly_rocdl::FlyROCDLDialect)

void mlirRegisterFlyToROCDLConversionPass(void) { mlir::registerFlyToROCDLConversionPass(); }
