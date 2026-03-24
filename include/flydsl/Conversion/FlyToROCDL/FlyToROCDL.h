// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef CONVERSION_FLYTOROCDL_FLYTOROCDL_H
#define CONVERSION_FLYTOROCDL_FLYTOROCDL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_FLYTOROCDLCONVERSIONPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

#endif // CONVERSION_FLYTOROCDL_FLYTOROCDL_H
