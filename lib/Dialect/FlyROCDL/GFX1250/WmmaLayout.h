// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_LIB_DIALECT_FLYROCDL_GFX1250_WMMALAYOUT_H
#define FLYDSL_LIB_DIALECT_FLYROCDL_GFX1250_WMMALAYOUT_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

// GFX1250 WMMA (wave32) register layouts shared by the plain and MX-scaled MMA
// atoms. Definitions live in GFX1250/MmaAtom.cpp.
namespace gfx1250 {

// Shape facts both WMMA forms on this target share. gfx120x rides on the plain
// layout helpers below but does not use these; they are here so the plain and
// MX-scaled sides cannot drift apart on what a wave or a row block is.
//
// wave32: the 32 lanes split into two halves of 16, holding the same rows and
// different slices of K.
constexpr int kWaveSize = 32;
constexpr int kLaneHalfSize = 16;
constexpr int kLaneHalves = kWaveSize / kLaneHalfSize;

// Operands are built from 16-row blocks: the 16x16 forms are one block, the
// 32-row MX-scaled A is two of them stacked.
constexpr int kRowsPerBlock = 16;

// C/D hands over along M instead of K: lanes 0-15 cover M=0..7 and lanes 16-31
// cover M=8..15, so a lane half takes 8 consecutive rows before the other half
// takes over, while the columns spread across the 16 lanes within a half. Both
// accumulator sizes use this same split; the 16-bit form differs only in
// packing two values into one register slot.
constexpr int kAccChunkM = 8;

// A/B thread-value layout for a K-wide operand (reference column-major (M,K)).
::mlir::fly::LayoutAttr getThrValLayoutAB(::mlir::MLIRContext *ctx, int32_t K, ::mlir::Type elemTy);

// C/D accumulator thread-value layout for a 16x16 tile.
::mlir::fly::LayoutAttr getThrValLayoutCD(::mlir::MLIRContext *ctx, ::mlir::Type elemTyAcc);

} // namespace gfx1250

#endif // FLYDSL_LIB_DIALECT_FLYROCDL_GFX1250_WMMALAYOUT_H
