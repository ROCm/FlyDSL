#pragma once

#include "flir/Support/fltuple.h"
#include "flir/Support/fltuple_algorithms.hpp"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace mlir::flir::support::stride {

// Tags to select major order for compact (contiguous) stride generation.
struct LayoutLeft {};
struct LayoutRight {};

namespace detail {

// Materialize an index constant in the current rewriter.
inline Value cidx(Location loc, PatternRewriter &rewriter, int64_t v) {
  return rewriter.create<arith::ConstantIndexOp>(loc, v).getResult();
}

// Resolve a type-mode ShapeType's flattened leaf dims into concrete constants.
// This accepts either:
// - fully static dims in the ShapeType, or
// - dynamic `?` dims where the defining flir.make_shape provides constant operands.
//
// Returns nullopt if any leaf can't be resolved to a compile-time constant.
inline std::optional<llvm::SmallVector<int64_t>>
resolveStaticShapeLeafDims(Value shapeVal, Location loc, PatternRewriter &rewriter) {
  auto shapeTy = llvm::dyn_cast<ShapeType>(shapeVal.getType());
  if (!shapeTy) {
    emitError(loc, "Expected flir.shape type.");
    return std::nullopt;
  }

  llvm::ArrayRef<int64_t> shapeDims = shapeTy.getDims();
  llvm::SmallVector<int64_t> resolved;
  resolved.reserve(shapeDims.size());

  // If the type contains `?`, we must look at the defining make_shape operands.
  llvm::SmallVector<Value> dynLeafOperands;
  if (auto makeShape = shapeVal.getDefiningOp<MakeShapeOp>()) {
    dynLeafOperands.append(makeShape.getValues().begin(), makeShape.getValues().end());
  } else if (llvm::any_of(shapeDims, [](int64_t d) { return d == -1; })) {
    emitError(loc,
              "Expected a type-mode flir.make_shape whose leaves are compile-time "
              "constants (either encoded in the type or provided as constant operands).");
    return std::nullopt;
  }

  int64_t oi = 0;
  for (int64_t d : shapeDims) {
    if (d != -1) {
      resolved.push_back(d);
      continue;
    }
    if (oi >= (int64_t)dynLeafOperands.size()) {
      emitError(loc, "Malformed type-mode flir.make_shape: operand count mismatch.");
      return std::nullopt;
    }
    auto c = tryGetConstIndex(dynLeafOperands[oi++]);
    if (!c.has_value()) {
      emitError(loc,
                "Expected a type-mode flir.make_shape whose dynamic leaves fold to "
                "compile-time constants.");
      return std::nullopt;
    }
    resolved.push_back(*c);
  }
  if (oi != (int64_t)dynLeafOperands.size()) {
    emitError(loc, "Malformed type-mode flir.make_shape: operand count mismatch.");
    return std::nullopt;
  }

  return resolved;
}

// Tree-based compact stride generation.
// Returns (strideTuple, nextCurrent).
template <class Major>
inline std::pair<FlTuple, int64_t> compactTuple(const FlTuple &shape, int64_t current,
                                                Location loc, PatternRewriter &rewriter);

template <>
inline std::pair<FlTuple, int64_t> compactTuple<LayoutLeft>(const FlTuple &shape,
                                                            int64_t current, Location loc,
                                                            PatternRewriter &rewriter) {
  if (shape.isLeaf) {
    auto s = shape.getConstantValue();
    if (!s.has_value()) {
      emitError(loc, "compact_major requires compile-time constant shape leaves.");
      return {FlTuple(Value()), current};
    }
    if (*s == 1) {
      return {FlTuple(cidx(loc, rewriter, 0)), current};
    }
    FlTuple strideLeaf(cidx(loc, rewriter, current));
    return {strideLeaf, current * (*s)};
  }

  std::vector<FlTuple> out;
  out.reserve(shape.children.size());
  int64_t cur = current;
  for (auto const &ch : shape.children) {
    auto [st, next] = compactTuple<LayoutLeft>(ch, cur, loc, rewriter);
    out.push_back(st);
    cur = next;
  }
  return {FlTuple(std::move(out)), cur};
}

template <>
inline std::pair<FlTuple, int64_t> compactTuple<LayoutRight>(const FlTuple &shape,
                                                             int64_t current, Location loc,
                                                             PatternRewriter &rewriter) {
  if (shape.isLeaf) {
    auto s = shape.getConstantValue();
    if (!s.has_value()) {
      emitError(loc, "compact_major requires compile-time constant shape leaves.");
      return {FlTuple(Value()), current};
    }
    if (*s == 1) {
      return {FlTuple(cidx(loc, rewriter, 0)), current};
    }
    FlTuple strideLeaf(cidx(loc, rewriter, current));
    return {strideLeaf, current * (*s)};
  }

  std::vector<FlTuple> out;
  out.resize(shape.children.size());
  int64_t cur = current;
  // Right-major: rightmost extent has stride 1 => traverse children from right to left.
  for (int64_t i = (int64_t)shape.children.size() - 1; i >= 0; --i) {
    auto [st, next] = compactTuple<LayoutRight>(shape.children[(size_t)i], cur, loc, rewriter);
    out[(size_t)i] = st;
    cur = next;
  }
  return {FlTuple(std::move(out)), cur};
}

} // namespace detail

// Compute compact (contiguous) strides in the given major order.
// The result is type-mode: if successful, it returns a flir.make_stride whose
// result StrideType has fully static dims and typically has no operands.
template <class Major>
inline Value compact_major(Value shapeVal, Location loc, PatternRewriter &rewriter,
                           MLIRContext *ctx, int64_t current = 1) {
  auto shapeTy = llvm::dyn_cast<ShapeType>(shapeVal.getType());
  if (!shapeTy) {
    emitError(loc, "Expected flir.shape type.");
    return Value();
  }

  // Build a FlTuple of constant index Values matching the shape's structure.
  // This makes nested shapes work naturally, and keeps behavior aligned with
  // python-side nested tuple specifications.
  auto maybeResolved = detail::resolveStaticShapeLeafDims(shapeVal, loc, rewriter);
  if (!maybeResolved)
    return Value();

  // Reconstruct a shape tree with the same structure but constant leaf Values.
  // We intentionally preserve structure from the ShapeType, not from any runtime tuple.
  FlTuple shapeTree;
  {
    llvm::ArrayRef<int32_t> structure = shapeTy.getStructure();
    int64_t si = 0;
    int64_t di = 0;
    std::function<FlTuple()> parse = [&]() -> FlTuple {
      int32_t tag = structure[(size_t)si++];
      if (tag == -1) {
        int64_t d = (*maybeResolved)[(size_t)di++];
        return FlTuple(detail::cidx(loc, rewriter, d));
      }
      std::vector<FlTuple> kids;
      kids.reserve(tag);
      for (int32_t i = 0; i < tag; ++i)
        kids.push_back(parse());
      return FlTuple(std::move(kids));
    };
    shapeTree = parse();
  }

  auto [strideTree, _] = detail::compactTuple<Major>(shapeTree, current, loc, rewriter);
  if (!strideTree.value && strideTree.isLeaf) // error sentinel
    return Value();
  return serializeTypeModeMakeStrideOp(strideTree, loc, rewriter, ctx);
}

// Construct a layout from a shape and a stride.
inline Value make_layout(Value shapeVal, Value strideVal, Location loc,
                         PatternRewriter &rewriter, MLIRContext *ctx) {
  auto shapeTy = llvm::dyn_cast<ShapeType>(shapeVal.getType());
  auto strideTy = llvm::dyn_cast<StrideType>(strideVal.getType());
  if (!shapeTy || !strideTy) {
    emitError(loc, "make_layout expects (flir.shape, flir.stride).");
    return Value();
  }
  auto layoutTy = LayoutType::get(ctx, shapeTy, strideTy);
  return rewriter.create<MakeLayoutOp>(loc, layoutTy, shapeVal, strideVal).getResult();
}

// Construct a layout from a shape using compact strides. Default is LayoutLeft.
inline Value make_layout(Value shapeVal, Location loc, PatternRewriter &rewriter,
                         MLIRContext *ctx) {
  Value strideVal = compact_major<LayoutLeft>(shapeVal, loc, rewriter, ctx);
  if (!strideVal)
    return Value();
  return make_layout(shapeVal, strideVal, loc, rewriter, ctx);
}

template <class Major>
inline Value make_layout(Value shapeVal, Major, Location loc, PatternRewriter &rewriter,
                         MLIRContext *ctx) {
  Value strideVal = compact_major<Major>(shapeVal, loc, rewriter, ctx);
  if (!strideVal)
    return Value();
  return make_layout(shapeVal, strideVal, loc, rewriter, ctx);
}

// Transform a type-mode layout by mapping over (shapeLeaf, strideLeaf) pairs.
// The transform is performed structurally: the nested tuple structure is preserved.
//
// @pre layoutVal must be produced by flir.make_layout, and its shape/stride must
//      be produced by flir.make_shape / flir.make_stride in type-mode.
template <class LeafFn>
inline Value transform_layout(Value layoutVal, LeafFn &&fn, Location loc,
                              PatternRewriter &rewriter, MLIRContext *ctx) {
  auto makeLayout = layoutVal.getDefiningOp<MakeLayoutOp>();
  if (!makeLayout) {
    emitError(loc, "transform_layout expects a value defined by flir.make_layout.");
    return Value();
  }

  auto shapeOp = makeLayout.getShape().getDefiningOp();
  auto strideOp = makeLayout.getStride().getDefiningOp();
  auto shapeT = deserializeTypeModeFlTuple(shapeOp, rewriter, loc);
  auto strideT = deserializeTypeModeFlTuple(strideOp, rewriter, loc);
  if (!shapeT || !strideT)
    return Value();

  FlTuple newStride = zip_transform(*shapeT, *strideT, [&](Value s, Value d) {
    return static_cast<LeafFn &&>(fn)(s, d);
  });

  Value outShape = serializeTypeModeMakeShapeOp(*shapeT, loc, rewriter, ctx);
  Value outStride = serializeTypeModeMakeStrideOp(newStride, loc, rewriter, ctx);
  return make_layout(outShape, outStride, loc, rewriter, ctx);
}

} // namespace mlir::flir::support::stride


