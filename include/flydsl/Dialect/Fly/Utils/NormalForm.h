#ifndef FLYDSL_DIALECT_UTILS_NORMALFORM_H
#define FLYDSL_DIALECT_UTILS_NORMALFORM_H

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"

namespace mlir::fly {

bool isNormalForm(TypedValue<IntTupleType> value);
bool isNormalForm(TypedValue<BasisType> value);
bool isNormalForm(TypedValue<LayoutType> value);
bool isNormalForm(TypedValue<SwizzleType> value);
bool isNormalForm(TypedValue<ComposedLayoutType> value);
bool isNormalForm(TypedValue<TileType> value);
bool isNormalForm(TypedValue<CoordTensorType> value);

bool isNormalForm(TypedValue<PointerType> value);
bool isNormalForm(TypedValue<MemRefType> value);
bool isNormalForm(TypedValue<TiledCopyType> value);
bool isNormalForm(TypedValue<TiledMmaType> value);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_UTILS_NORMALFORM_H
