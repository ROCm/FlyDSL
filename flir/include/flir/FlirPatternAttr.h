#ifndef FLIR_PATTERN_ATTR_H
#define FLIR_PATTERN_ATTR_H

#include "flir/FlirDialect.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::flir {

/// Count the number of leaves in a tuple-pattern Attribute tree.
/// Leaves are non-ArrayAttr nodes (e.g. IntegerAttr, #flir.dyn, #flir.underscore).
inline int64_t getPatternRank(Attribute pattern) {
  if (!pattern)
    return 0;
  if (auto arr = llvm::dyn_cast<ArrayAttr>(pattern)) {
    int64_t r = 0;
    for (Attribute e : arr.getValue())
      r += getPatternRank(e);
    return r;
  }
  return 1;
}

/// Count the number of dynamic leaves (#flir.dyn) in a tuple-pattern Attribute tree.
inline int64_t countDynLeaves(Attribute pattern) {
  if (!pattern)
    return 0;
  if (auto arr = llvm::dyn_cast<ArrayAttr>(pattern)) {
    int64_t c = 0;
    for (Attribute e : arr.getValue())
      c += countDynLeaves(e);
    return c;
  }
  return llvm::isa<DyncI32Attr>(pattern) ? 1 : 0;
}

/// Build a flat tuple pattern of rank N, where every leaf is #flir.dyn.
/// Printed form: `(?, ?, ...)` (with parentheses) via the custom printer.
inline Attribute makeAllDynTuplePattern(MLIRContext *ctx, int64_t rank) {
  llvm::SmallVector<Attribute, 8> elems;
  elems.reserve(static_cast<size_t>(std::max<int64_t>(0, rank)));
  for (int64_t i = 0; i < rank; ++i)
    elems.push_back(DyncI32Attr::get(ctx, /*dyncElemIdx=*/static_cast<int32_t>(i),
                                     /*divisibility=*/1));
  return ArrayAttr::get(ctx, elems);
}

inline ShapeType makeRankOnlyShapeType(MLIRContext *ctx, int64_t rank) {
  return ShapeType::get(ctx, makeAllDynTuplePattern(ctx, rank));
}

inline StrideType makeRankOnlyStrideType(MLIRContext *ctx, int64_t rank) {
  return StrideType::get(ctx, makeAllDynTuplePattern(ctx, rank));
}

inline CoordType makeRankOnlyCoordType(MLIRContext *ctx, int64_t rank) {
  return CoordType::get(ctx, makeAllDynTuplePattern(ctx, rank));
}

inline LayoutType makeRankOnlyLayoutType(MLIRContext *ctx, int64_t rank) {
  Attribute p = makeAllDynTuplePattern(ctx, rank);
  return LayoutType::get(ctx, p, p);
}

} // namespace mlir::flir

#endif // FLIR_PATTERN_ATTR_H

