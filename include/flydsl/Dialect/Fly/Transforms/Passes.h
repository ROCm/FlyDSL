// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_TRANSFORMS_PASSES_H
#define FLYDSL_DIALECT_FLY_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace fly {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"

} // namespace fly
} // namespace mlir

#endif // FLYDSL_DIALECT_FLY_TRANSFORMS_PASSES_H
