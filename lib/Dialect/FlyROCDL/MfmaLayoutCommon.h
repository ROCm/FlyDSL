// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Shared MFMA thread-value layout helpers for CDNA3/CDNA4 (wave64).

#ifndef FLYROCDL_MFMA_LAYOUT_COMMON_H
#define FLYROCDL_MFMA_LAYOUT_COMMON_H

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace cdna3 {

/// Thread-value layout for A/B operands of wave64 MFMA (shared by CDNA3 & CDNA4).
mlir::fly::LayoutAttr getThrValLayoutAB(mlir::MLIRContext *ctx, int32_t M, int32_t N, int32_t K,
                                        mlir::Type elemTyA, mlir::Type elemTyB,
                                        mlir::Type elemTyAcc);

} // namespace cdna3

#endif // FLYROCDL_MFMA_LAYOUT_COMMON_H
