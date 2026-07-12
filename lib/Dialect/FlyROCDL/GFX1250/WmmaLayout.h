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

// A/B matrix thread-value layout for a K-wide WMMA operand of element type
// `elemTy`. Reference space is column-major (M,K) / (N,K).
::mlir::fly::LayoutAttr getThrValLayoutAB(::mlir::MLIRContext *ctx, int32_t K, ::mlir::Type elemTy);

// C/D (accumulator) thread-value layout for a 16x16 tile with accumulator
// element type `elemTyAcc`.
::mlir::fly::LayoutAttr getThrValLayoutCD(::mlir::MLIRContext *ctx, ::mlir::Type elemTyAcc);

} // namespace gfx1250

#endif // FLYDSL_LIB_DIALECT_FLYROCDL_GFX1250_WMMALAYOUT_H
