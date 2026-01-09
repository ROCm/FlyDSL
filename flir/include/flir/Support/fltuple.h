#pragma once

#include "flir/FlirDialect.h"
#include "flir/FlirOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace mlir::flir::support {

std::optional<int64_t> tryGetConstIndex(Value v);

/// FLIR tuple-tree container for type-mode shape/stride/layout leaves.
///
/// This is intentionally not `std::tuple`:
/// - It matches FLIR's MLIR encoding: nested tuple structure is stored in
///   `!flir.shape<...>` / `!flir.stride<...>` types (structure+dims), and
///   dynamic leaves are stored as operands of `flir.make_shape` / `flir.make_stride`.
/// - Lowering needs to traverse/transform nested MLIR values in a way that
///   corresponds to python frontend inputs like `(9,(4,8)):(59,(13,1))`.
struct FlTuple {
  bool isLeaf = true;
  Value value{};                // valid iff isLeaf
  std::vector<FlTuple> children; // valid iff !isLeaf

  FlTuple() = default;
  explicit FlTuple(Value v) : isLeaf(true), value(v) {}
  explicit FlTuple(std::vector<FlTuple> c) : isLeaf(false), children(std::move(c)) {}

  bool isTuple() const { return !isLeaf; }

  /// For leaf nodes, return compile-time constant value if it is an index constant.
  std::optional<int64_t> getConstantValue() const {
    if (!isLeaf || !value)
      return std::nullopt;
    return tryGetConstIndex(value);
  }

  void flatten(llvm::SmallVectorImpl<Value> &values,
               llvm::SmallVectorImpl<int32_t> &structure) const {
    if (isLeaf) {
      values.push_back(value);
      structure.push_back(-1);
      return;
    }
    structure.push_back(static_cast<int32_t>(children.size()));
    for (auto const &ch : children)
      ch.flatten(values, structure);
  }
};

inline std::optional<int64_t> tryGetConstIndex(Value v) {
  if (!v)
    return std::nullopt;
  if (auto constIdx = v.getDefiningOp<arith::ConstantIndexOp>())
    return constIdx.value();
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (llvm::isa<IndexType>(cst.getType())) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
        return intAttr.getInt();
    }
  }
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return tryGetConstIndex(cast.getIn());
  return std::nullopt;
}

/// Deserialize type-mode `flir.make_shape` / `flir.make_stride` to `FlTuple`.
/// Returns nullopt if op is not structured type-mode.
inline std::optional<FlTuple> deserializeTypeModeFlTuple(Operation *op,
                                                         PatternRewriter &rewriter,
                                                         Location loc) {
  if (!op || op->getNumResults() == 0)
    return std::nullopt;

  llvm::SmallVector<Value> dynOperands;
  if (auto makeShape = llvm::dyn_cast<MakeShapeOp>(op)) {
    dynOperands = makeShape.getValues();
  } else if (auto makeStride = llvm::dyn_cast<MakeStrideOp>(op)) {
    dynOperands = makeStride.getValues();
  } else {
    return std::nullopt;
  }

  Type ty = op->getResult(0).getType();
  llvm::ArrayRef<int32_t> structure;
  llvm::ArrayRef<int64_t> dims;

  if (auto st = llvm::dyn_cast<ShapeType>(ty)) {
    structure = st.getStructure();
    dims = st.getDims();
  } else if (auto st = llvm::dyn_cast<StrideType>(ty)) {
    structure = st.getStructure();
    dims = st.getDims();
  } else {
    return std::nullopt;
  }

  if (structure.empty())
    return std::nullopt;

  int64_t leafCount = 0;
  for (int32_t tag : structure)
    if (tag == -1)
      ++leafCount;
  if ((int64_t)dims.size() != leafCount)
    return std::nullopt;

  int64_t dynCount = 0;
  for (int64_t d : dims)
    if (d == -1)
      ++dynCount;
  if ((int64_t)dynOperands.size() != dynCount)
    return std::nullopt;

  int64_t si = 0;
  int64_t di = 0;
  int64_t oi = 0;

  std::function<FlTuple()> parse = [&]() -> FlTuple {
    if (si >= (int64_t)structure.size())
      return FlTuple(Value());
    int32_t tag = structure[si++];
    if (tag == -1) {
      if (di >= (int64_t)dims.size())
        return FlTuple(Value());
      int64_t d = dims[di++];
      if (d == -1) {
        Value v = (oi < (int64_t)dynOperands.size()) ? dynOperands[oi] : Value();
        ++oi;
        return FlTuple(v);
      }
      Value c = rewriter.create<arith::ConstantIndexOp>(loc, d).getResult();
      return FlTuple(c);
    }
    std::vector<FlTuple> kids;
    kids.reserve(tag);
    for (int32_t i = 0; i < tag; ++i)
      kids.push_back(parse());
    return FlTuple(std::move(kids));
  };

  FlTuple root = parse();
  if (oi != dynCount)
    return std::nullopt;
  return root;
}

inline Value serializeTypeModeMakeShapeOp(const FlTuple &t, Location loc,
                                         PatternRewriter &rewriter,
                                         MLIRContext *ctx) {
  llvm::SmallVector<Value> leafValues;
  llvm::SmallVector<int32_t> structure;
  // Canonicalize rank-1 as a 1-tuple "(x)" to match python/frontend printing.
  if (t.isLeaf) {
    structure.push_back(1);
    structure.push_back(-1);
    leafValues.push_back(t.value);
  } else {
    t.flatten(leafValues, structure);
  }

  llvm::SmallVector<int64_t> dims;
  llvm::SmallVector<Value> dynOperands;
  dims.reserve(leafValues.size());
  dynOperands.reserve(leafValues.size());
  for (auto v : leafValues) {
    if (auto c = tryGetConstIndex(v)) {
      dims.push_back(*c);
    } else {
      dims.push_back(-1);
      dynOperands.push_back(v);
    }
  }

  auto shapeType = ShapeType::get(ctx, structure, dims);
  return rewriter.create<MakeShapeOp>(loc, shapeType, dynOperands).getResult();
}

inline Value serializeTypeModeMakeStrideOp(const FlTuple &t, Location loc,
                                          PatternRewriter &rewriter,
                                          MLIRContext *ctx) {
  llvm::SmallVector<Value> leafValues;
  llvm::SmallVector<int32_t> structure;
  if (t.isLeaf) {
    structure.push_back(1);
    structure.push_back(-1);
    leafValues.push_back(t.value);
  } else {
    t.flatten(leafValues, structure);
  }

  llvm::SmallVector<int64_t> dims;
  llvm::SmallVector<Value> dynOperands;
  dims.reserve(leafValues.size());
  dynOperands.reserve(leafValues.size());
  for (auto v : leafValues) {
    if (auto c = tryGetConstIndex(v)) {
      dims.push_back(*c);
    } else {
      dims.push_back(-1);
      dynOperands.push_back(v);
    }
  }

  auto strideType = StrideType::get(ctx, structure, dims);
  return rewriter.create<MakeStrideOp>(loc, strideType, dynOperands).getResult();
}

} // namespace mlir::flir::support


