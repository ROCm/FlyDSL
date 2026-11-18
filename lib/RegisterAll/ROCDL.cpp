// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// Registration hooks of the ROCDL backend, called by RegisterAll.cpp.

#include "flydsl/Conversion/Passes.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir::fly {

void registerBackendDialects_rocdl(DialectRegistry &registry) {
  registry.insert<fly_rocdl::FlyROCDLDialect>();
  fly_rocdl::registerTargetInfoExternalModels(registry);
}

void registerBackendPasses_rocdl() {
  registerFlyToROCDLPasses();
  fly_rocdl::registerFlyROCDLPasses();
  registerFlyKtraceToROCDLPasses();
}

} // namespace mlir::fly
