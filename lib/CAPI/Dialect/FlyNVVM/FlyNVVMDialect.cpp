// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl-c/FlyNVVMDialect.h"

#include "flydsl/Conversion/FlyToNVVM/FlyToNVVM.h"
#include "flydsl/Dialect/FlyNVVM/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/FlyToNVVM/Passes.h.inc"
} // namespace mlir

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyNVVM, fly_nvvm, mlir::fly_nvvm::FlyNVVMDialect)

void mlirRegisterFlyToNVVMConversionPass(void) { mlir::registerFlyToNVVMConversionPass(); }

void flydsl_register_nvvm_dialects(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::fly_nvvm::FlyNVVMDialect>();
}

void flydsl_register_nvvm_passes(void) { mlirRegisterFlyToNVVMConversionPass(); }
