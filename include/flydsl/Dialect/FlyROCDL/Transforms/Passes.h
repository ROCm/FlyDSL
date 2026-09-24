// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYROCDL_TRANSFORMS_PASSES_H
#define FLYDSL_DIALECT_FLYROCDL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

namespace mlir {
namespace fly_rocdl {

#define GEN_PASS_DECL
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h.inc"

} // namespace fly_rocdl
} // namespace mlir

#endif // FLYDSL_DIALECT_FLYROCDL_TRANSFORMS_PASSES_H
