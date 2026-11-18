// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl-c/RegisterEverything.h"

#include "flydsl/InitAllDialects.h"
#include "flydsl/InitAllPasses.h"
#include "mlir/CAPI/IR.h"

void flydslRegisterAllDialects(MlirDialectRegistry registry) {
  mlir::fly::registerAllDialects(*unwrap(registry));
}

void flydslRegisterAllPasses(void) { mlir::fly::registerAllPasses(); }
