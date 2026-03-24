// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_NORMALFORM_H
#define FLYDSL_DIALECT_FLY_UTILS_NORMALFORM_H

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

bool isNormalForm(TypedValue<IntTupleType> value);
bool isNormalForm(TypedValue<LayoutType> value);
bool isNormalForm(TypedValue<ComposedLayoutType> value);

// Weakly normal form: defined by a MakeViewOp with a normal form layout
bool isWeaklyNormalForm(TypedValue<MemRefType> value);
bool isWeaklyNormalForm(TypedValue<CoordTensorType> value);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_NORMALFORM_H
