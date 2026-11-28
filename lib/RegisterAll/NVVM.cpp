// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// Registration hooks of the NVVM backend, called by RegisterAll.cpp.

#include "flydsl/Conversion/FlyToNVVM/FlyToNVVM.h"
#include "flydsl/Dialect/FlyNVVM/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/FlyToNVVM/Passes.h.inc"
} // namespace mlir

namespace mlir::fly {

void registerBackendDialects_nvvm(DialectRegistry &registry) {
  registry.insert<fly_nvvm::FlyNVVMDialect>();
}

void registerBackendPasses_nvvm() { registerFlyToNVVMPasses(); }

} // namespace mlir::fly
