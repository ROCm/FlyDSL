// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef CONVERSION_FLYTONVVM_FLYTONVVM_H
#define CONVERSION_FLYTONVVM_FLYTONVVM_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_FLYTONVVMCONVERSIONPASS
#include "flydsl/Conversion/FlyToNVVM/Passes.h.inc"
} // namespace mlir

#endif // CONVERSION_FLYTONVVM_FLYTONVVM_H
