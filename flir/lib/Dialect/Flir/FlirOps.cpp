//===- FlirOps.cpp - Flir Operations Implementation --------------------===//
//
// Implementation of Flir operation verification and methods
//
//===----------------------------------------------------------------------===//

#include "flir/FlirOps.h"
#include "flir/FlirDialect.h"
#include "flir/FlirLayoutAlgebra.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::flir;

namespace {

struct PatternStats {
  int64_t leafCount = 0;
  int64_t dynLeafCount = 0;
  int64_t maxDynIdx = -1;
  bool hasStaticOrUnderscoreLeaf = false;
  llvm::SmallSet<int32_t, 16> dynIdxSet;
};

static void collectPatternStats(Attribute pat, PatternStats &s) {
  if (auto arr = dyn_cast<ArrayAttr>(pat)) {
    for (auto e : arr.getValue())
      collectPatternStats(e, s);
    return;
  }

  // Leaf.
  ++s.leafCount;
  if (isa<IntegerAttr>(pat) || isa<UnderscoreAttr>(pat)) {
    s.hasStaticOrUnderscoreLeaf = true;
    return;
  }
  if (auto d64 = dyn_cast<DyncI64Attr>(pat)) {
    ++s.dynLeafCount;
    s.maxDynIdx = std::max<int64_t>(s.maxDynIdx, d64.getDyncElemIdx());
    s.dynIdxSet.insert(d64.getDyncElemIdx());
    return;
  }
  if (auto d32 = dyn_cast<DyncI32Attr>(pat)) {
    ++s.dynLeafCount;
    s.maxDynIdx = std::max<int64_t>(s.maxDynIdx, d32.getDyncElemIdx());
    s.dynIdxSet.insert(d32.getDyncElemIdx());
    return;
  }
  // Unknown leaf kind: treat as dynamic.
  ++s.dynLeafCount;
}

template <typename StructuredTy>
static LogicalResult verifyPatternTupleType(Operation *op, Type resultTy, ValueRange operands,
                                            llvm::StringRef prettyName,
                                            bool operandsAreDynOnly) {
  auto st = llvm::dyn_cast<StructuredTy>(resultTy);
  if (!st)
    return op->emitOpError() << "expects " << prettyName << " result type";

  Attribute pattern = st.getPattern();
  if (!pattern)
    return op->emitOpError() << "requires " << prettyName
                             << " type with a pattern (e.g. " << prettyName << "<(?,?)>)";

  PatternStats stats;
  collectPatternStats(pattern, stats);

  // Ensure dynamic placeholders are indexed contiguously 0..N-1 and unique.
  // This is the key invariant to make type-driven operand binding work.
  if (stats.dynLeafCount != static_cast<int64_t>(stats.dynIdxSet.size()))
    return op->emitOpError() << "pattern has duplicate dynamic indices";
  if (stats.dynLeafCount > 0 && stats.maxDynIdx != stats.dynLeafCount - 1)
    return op->emitOpError() << "pattern dynamic indices must be contiguous 0..N-1";

  int64_t expectedOperands = operandsAreDynOnly ? stats.dynLeafCount : stats.leafCount;
  if (static_cast<int64_t>(operands.size()) != expectedOperands) {
    return op->emitOpError()
           << "expects " << expectedOperands << (operandsAreDynOnly ? " dynamic" : "")
           << " leaf operands but got " << operands.size();
  }

  if (!operandsAreDynOnly) {
    // Coords are runtime values: all leaves must be dynamic placeholders.
    if (stats.hasStaticOrUnderscoreLeaf)
      return op->emitOpError() << "coord type pattern must be fully dynamic (no constants or '*')";
    if (stats.dynLeafCount != stats.leafCount)
      return op->emitOpError() << "coord type pattern must be fully dynamic";
  }

  return success();
}

} // namespace

LogicalResult MakeShapeOp::verify() {
  return verifyPatternTupleType<ShapeType>(getOperation(), getResult().getType(), getValues(),
                                           "!flir.shape", /*operandsAreDynOnly=*/true);
}

LogicalResult MakeStrideOp::verify() {
  return verifyPatternTupleType<StrideType>(getOperation(), getResult().getType(), getValues(),
                                            "!flir.stride", /*operandsAreDynOnly=*/true);
}

LogicalResult MakeCoordOp::verify() {
  // Coords are runtime values: operands provide all leaf coordinates.
  return verifyPatternTupleType<CoordType>(getOperation(), getResult().getType(), getValues(),
                                           "!flir.coord", /*operandsAreDynOnly=*/false);
}

//===----------------------------------------------------------------------===//
// Layout algebra ops: require successful type inference (no fallback).
//===----------------------------------------------------------------------===//

template <typename InferFn>
static LogicalResult verifyBinaryLayoutAlgebraOp(Operation *op, Type resultTy, Type aTy, Type bTy,
                                                llvm::StringRef prettyName, InferFn &&inferFn) {
  auto r = llvm::dyn_cast<LayoutType>(resultTy);
  auto a = llvm::dyn_cast<LayoutType>(aTy);
  auto b = llvm::dyn_cast<LayoutType>(bTy);
  if (!r || !a || !b)
    return op->emitOpError() << prettyName << " expects layout operands/results";

  auto inferred = inferFn(op->getContext(), a, b);
  if (failed(inferred))
    return op->emitOpError() << prettyName
                             << " type inference failed (no fallback): operands must be fully static patterns";

  if (r != *inferred)
    return op->emitOpError() << prettyName << " result type mismatch: expected " << *inferred
                             << " but got " << r;

  return success();
}

LogicalResult CompositionOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getLayoutA().getType(),
                                     getLayoutB().getType(), "flir.composition",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferCompositionType(ctx, a, b);
                                     });
}

LogicalResult LogicalProductOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.logical_product",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalProductType(ctx, a, b);
                                     });
}

LogicalResult ZippedProductOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.zipped_product",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalProductType(ctx, a, b);
                                     });
}

LogicalResult TiledProductOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.tiled_product",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalProductType(ctx, a, b);
                                     });
}

LogicalResult FlatProductOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.flat_product",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalProductType(ctx, a, b);
                                     });
}

LogicalResult RakedProductOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.raked_product",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalProductType(ctx, a, b);
                                     });
}

LogicalResult BlockedProductOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.blocked_product",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalProductType(ctx, a, b);
                                     });
}

LogicalResult LogicalDivideOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.logical_divide",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalDivideType(ctx, a, b);
                                     });
}

LogicalResult ZippedDivideOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.zipped_divide",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalDivideType(ctx, a, b);
                                     });
}

LogicalResult TiledDivideOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.tiled_divide",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalDivideType(ctx, a, b);
                                     });
}

LogicalResult FlatDivideOp::verify() {
  return verifyBinaryLayoutAlgebraOp(getOperation(), getResult().getType(), getInput().getType(),
                                     getTiler().getType(), "flir.flat_divide",
                                     [](MLIRContext *ctx, LayoutType a, LayoutType b) {
                                       return inferLogicalDivideType(ctx, a, b);
                                     });
}

//===----------------------------------------------------------------------===//
// TableGen generated code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "flir/FlirOps.cpp.inc"
