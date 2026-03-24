// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors


#ifndef FLYDSL_CONVERSION_PASSES_H
#define FLYDSL_CONVERSION_PASSES_H

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/Passes.h.inc"

} // namespace mlir

#endif // FLYDSL_CONVERSION_PASSES_H
