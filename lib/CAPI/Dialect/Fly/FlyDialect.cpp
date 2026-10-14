// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyDialect.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/IR/FlyLLVMTranslation.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/FlyKtrace/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Fly, fly, mlir::fly::FlyDialect)

void mlirRegisterFlyPasses(void) { mlir::fly::registerFlyPasses(); }

void flydsl_register_ktrace_dialect(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::fly_ktrace::FlyKtraceDialect>();
}

void mlirRegisterFlyExplicitModuleOffloadingLLVMTranslation(MlirContext context) {
  mlir::fly::registerExplicitModuleOffloadingLLVMTranslation(*unwrap(context));
}
