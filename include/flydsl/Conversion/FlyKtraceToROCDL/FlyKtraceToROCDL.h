// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef CONVERSION_FLYKTRACETOROCDL_FLYKTRACETOROCDL_H
#define CONVERSION_FLYKTRACETOROCDL_FLYKTRACETOROCDL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_FLYKTRACETOROCDLCONVERSIONPASS
#include "flydsl/Conversion/FlyKtraceToROCDL/Passes.h.inc"
} // namespace mlir

#endif // CONVERSION_FLYKTRACETOROCDL_FLYKTRACETOROCDL_H
