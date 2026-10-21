// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_ATOMUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_ATOMUTILS_H

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>

namespace mlir::fly {

/// Emits one copy atom call on its final operands; `pred` may be null.
using EmitCopyAtomFn = llvm::function_ref<void(Value src, Value dst, Value pred)>;

/// Emits one MMA atom call on its final operands.
using EmitMmaAtomFn = llvm::function_ref<void(Value d, ValueRange a, ValueRange b, Value c)>;

/// Expands a copy into atom calls. Every operand is imported into the Core
/// EmitOp category once; predicate alignment, grouping, slicing and
/// decomposition happen there, and only the operands handed to `emitAtom` are
/// materialized. `atomValsSrc`/`atomValsDst` are the atom's static value
/// counts, when known.
LogicalResult expandCopy(OpBuilder &builder, Location loc, Value src, Value dst, Value pred,
                         std::optional<int64_t> atomValsSrc, std::optional<int64_t> atomValsDst,
                         EmitCopyAtomFn emitAtom);

/// Expands a gemm into one atom call per (m, n, k) tile, visited in the order
/// of `traversalLayout` when given, else of `order` (NMK by default). Each
/// operand is imported into the Core EmitOp category once and sliced there.
LogicalResult expandGemm(OpBuilder &builder, Location loc, Value d, ValueRange a, ValueRange b,
                         Value c, std::optional<GemmTraversalOrder> order,
                         LayoutAttr traversalLayout, EmitMmaAtomFn emitAtom);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_ATOMUTILS_H
