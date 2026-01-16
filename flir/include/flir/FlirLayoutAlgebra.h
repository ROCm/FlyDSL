//===- FlirLayoutAlgebra.h - Type-level layout algebra helpers -----------===//
//
//
// These helpers are intentionally pure (no IR mutation) and operate on the
// pattern Attributes stored in Flir types.
//
//===----------------------------------------------------------------------===//

#ifndef FLIR_LAYOUT_ALGEBRA_H
#define FLIR_LAYOUT_ALGEBRA_H

#include "flir/FlirDialect.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::flir {

/// Infer the result layout type for `flir.composition`.
FailureOr<LayoutType> inferCompositionType(MLIRContext *ctx, LayoutType lhs,
                                          LayoutType rhs);

/// Infer the result layout type for `flir.logical_product` and friends.
FailureOr<LayoutType> inferLogicalProductType(MLIRContext *ctx, LayoutType block,
                                              LayoutType tiler);

/// Infer the result layout type for `flir.logical_divide` (base/global divide).
FailureOr<LayoutType> inferLogicalDivideType(MLIRContext *ctx, LayoutType layout,
                                             LayoutType tiler);

} // namespace mlir::flir

#endif // FLIR_LAYOUT_ALGEBRA_H



