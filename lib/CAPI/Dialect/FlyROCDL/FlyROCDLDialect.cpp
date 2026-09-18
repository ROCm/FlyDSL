// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Conversion/Passes.h"
#include "flydsl/Dialect/FlyKtrace/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl, mlir::fly_rocdl::FlyROCDLDialect)

void mlirRegisterConvertROCDLFastMathOpsPass(void) { mlir::registerConvertROCDLFastMathOpsPass(); }
void mlirRegisterFlyToROCDLConversionPass(void) { mlir::registerFlyToROCDLConversionPass(); }
// ktrace is gated to CDNA3/CDNA4, so it registers with this backend.
void mlirRegisterFlyKtraceToROCDLConversionPass(void) {
  mlir::registerFlyKtraceToROCDLConversionPass();
}

void flydsl_register_rocdl_dialects(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::fly_rocdl::FlyROCDLDialect>();
  unwrap(registry)->insert<mlir::fly_ktrace::FlyKtraceDialect>();
}

void flydsl_register_rocdl_passes(void) {
  mlirRegisterConvertROCDLFastMathOpsPass();
  mlirRegisterFlyToROCDLConversionPass();
  mlir::fly_rocdl::registerFlyROCDLPasses();
  mlirRegisterFlyKtraceToROCDLConversionPass();
}
