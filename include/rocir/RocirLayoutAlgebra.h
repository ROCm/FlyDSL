//===- RocirLayoutAlgebra.h - Type-level layout algebra helpers -----------===//
//
//
// These helpers are intentionally pure (no IR mutation) and operate on the
// structure/dims stored in Rocir types.
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_LAYOUT_ALGEBRA_H
#define ROCIR_LAYOUT_ALGEBRA_H

#include "rocir/RocirDialect.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::rocir {

/// Infer the result layout type for `rocir.composition`.
FailureOr<LayoutType> inferCompositionType(MLIRContext *ctx, LayoutType lhs,
                                          LayoutType rhs);

/// Infer the result layout type for `rocir.logical_product` and friends.
FailureOr<LayoutType> inferLogicalProductType(MLIRContext *ctx, LayoutType block,
                                              LayoutType tiler);

/// Infer the result layout type for `rocir.logical_divide` (base/global divide).
FailureOr<LayoutType> inferLogicalDivideType(MLIRContext *ctx, LayoutType layout,
                                             LayoutType tiler);

} // namespace mlir::rocir

#endif // ROCIR_LAYOUT_ALGEBRA_H



