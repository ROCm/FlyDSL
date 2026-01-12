//===- FlirLayoutAlgebra.cpp - Type-level layout algebra helpers ---------===//

#include "flir/FlirLayoutAlgebra.h"

#include "flir/FlirDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::flir;

namespace {

struct PatternNode {
  bool isLeaf = true;
  std::optional<int64_t> value; // nullopt => dynamic/unknown
  std::vector<PatternNode> children;

  static PatternNode leaf(std::optional<int64_t> v) {
    PatternNode n;
    n.isLeaf = true;
    n.value = v;
    return n;
  }
  static PatternNode tuple(std::vector<PatternNode> c) {
    PatternNode n;
    n.isLeaf = false;
    n.children = std::move(c);
    return n;
  }
};

static bool allStaticLeaves(const PatternNode &n) {
  if (n.isLeaf)
    return n.value.has_value();
  for (auto &c : n.children)
    if (!allStaticLeaves(c))
      return false;
  return true;
}

static void flattenLeaves(const PatternNode &n,
                          std::vector<std::optional<int64_t>> &out) {
  if (n.isLeaf) {
    out.push_back(n.value);
    return;
  }
  for (auto &c : n.children)
    flattenLeaves(c, out);
}

static PatternNode parsePatternAttr(Attribute pat) {
  if (auto arr = dyn_cast<ArrayAttr>(pat)) {
    std::vector<PatternNode> kids;
    kids.reserve(arr.size());
    for (auto e : arr.getValue())
      kids.push_back(parsePatternAttr(e));
    return PatternNode::tuple(std::move(kids));
  }
  if (auto i = dyn_cast<IntegerAttr>(pat))
    return PatternNode::leaf(i.getInt());
  // Dynamic / wildcard leaves.
  return PatternNode::leaf(std::nullopt);
}

static Attribute patternNodeToAttr(MLIRContext *ctx, const PatternNode &n, int32_t &dyncIdx) {
  Builder b(ctx);
  if (n.isLeaf) {
    if (n.value.has_value()) {
      return IntegerAttr::get(b.getIntegerType(64), APInt(64, static_cast<uint64_t>(*n.value), true));
    }
    return DyncI64Attr::get(ctx, dyncIdx++, /*divisibility=*/1);
  }
  SmallVector<Attribute, 8> elems;
  elems.reserve(n.children.size());
  for (auto &c : n.children)
    elems.push_back(patternNodeToAttr(ctx, c, dyncIdx));
  return ArrayAttr::get(ctx, elems);
}

static PatternNode makeFlatTupleFromLeaves(const std::vector<std::optional<int64_t>> &leaves) {
  std::vector<PatternNode> children;
  children.reserve(leaves.size());
  for (auto &v : leaves)
    children.push_back(PatternNode::leaf(v));
  return PatternNode::tuple(std::move(children));
}

// ---- integer helpers (static-only) ----
static std::optional<int64_t> mul(std::optional<int64_t> a,
                                  std::optional<int64_t> b) {
  if (!a || !b)
    return std::nullopt;
  return (*a) * (*b);
}
static std::optional<int64_t> divUI(std::optional<int64_t> a,
                                    std::optional<int64_t> b) {
  if (!a || !b || *b == 0)
    return std::nullopt;
  return (*a) / (*b);
}
static std::optional<int64_t> ceilDivUI(std::optional<int64_t> a,
                                        std::optional<int64_t> b) {
  if (!a || !b || *b == 0)
    return std::nullopt;
  int64_t aa = *a;
  int64_t bb = *b;
  return (aa + bb - 1) / bb;
}
static std::optional<int64_t> minUI(std::optional<int64_t> a,
                                    std::optional<int64_t> b) {
  if (!a || !b)
    return std::nullopt;
  return std::min(*a, *b);
}

// ---- composition (type-level, static-only on leaves) ----
static std::pair<PatternNode, PatternNode>
compositionImpl(const PatternNode &lhsShape, const PatternNode &lhsStride,
                const PatternNode &rhsShape, const PatternNode &rhsStride) {
  // Case 1: RHS is tuple -> distribute.
  if (!rhsShape.isLeaf) {
    std::vector<PatternNode> outShapes, outStrides;
    outShapes.reserve(rhsShape.children.size());
    outStrides.reserve(rhsShape.children.size());
    for (size_t i = 0; i < rhsShape.children.size(); ++i) {
      const PatternNode &subShape = rhsShape.children[i];
      const PatternNode &subStride =
          (!rhsStride.isLeaf && i < rhsStride.children.size()) ? rhsStride.children[i] : rhsStride;
      auto [s, d] = compositionImpl(lhsShape, lhsStride, subShape, subStride);
      outShapes.push_back(std::move(s));
      outStrides.push_back(std::move(d));
    }
    return {PatternNode::tuple(std::move(outShapes)),
            PatternNode::tuple(std::move(outStrides))};
  }

  // Case 2: RHS leaf -> fold LHS over RHS.
  if (!rhsShape.isLeaf || !rhsStride.isLeaf)
    return {rhsShape, rhsStride};

  // Flatten LHS leaves.
  std::vector<std::optional<int64_t>> lhsShapes, lhsStrides;
  flattenLeaves(lhsShape, lhsShapes);
  flattenLeaves(lhsStride, lhsStrides);

  std::optional<int64_t> restShape = rhsShape.value;
  std::optional<int64_t> restStride = rhsStride.value;

  std::vector<std::optional<int64_t>> outShapeLeaves;
  std::vector<std::optional<int64_t>> outStrideLeaves;

  for (size_t i = 0; i < lhsShapes.size(); ++i) {
    auto currShape = lhsShapes[i];
    auto currStride = (i < lhsStrides.size()) ? lhsStrides[i] : std::nullopt;

    auto nextShape = ceilDivUI(currShape, restStride);
    auto nextStride = ceilDivUI(restStride, currShape);

    if (restShape && *restShape == 1) {
      restStride = nextStride;
      break;
    }
    if (nextShape && *nextShape == 1) {
      restStride = nextStride;
      continue;
    }

    auto newShape = minUI(nextShape, restShape);
    auto newStride = mul(currStride, restStride);

    outShapeLeaves.push_back(newShape);
    outStrideLeaves.push_back(newStride);

    restShape = divUI(restShape, newShape);
    restStride = nextStride;
  }

  if (outShapeLeaves.empty()) {
    auto lastLhsStride = (!lhsStrides.empty()) ? lhsStrides.back() : std::optional<int64_t>(1);
    auto tailStride = mul(restStride, lastLhsStride);
    return {PatternNode::leaf(restShape), PatternNode::leaf(tailStride)};
  }

  if (restShape && *restShape == 1) {
    return {makeFlatTupleFromLeaves(outShapeLeaves),
            makeFlatTupleFromLeaves(outStrideLeaves)};
  }

  outShapeLeaves.push_back(restShape);
  auto lastLhsStride = (!lhsStrides.empty()) ? lhsStrides.back() : std::optional<int64_t>(1);
  auto tailStride = mul(restStride, lastLhsStride);
  outStrideLeaves.push_back(tailStride);

  return {makeFlatTupleFromLeaves(outShapeLeaves),
          makeFlatTupleFromLeaves(outStrideLeaves)};
}

// ---- complement (type-level, static-only; flat output) ----
static std::pair<PatternNode, PatternNode>
complementImpl(const PatternNode &shape, const PatternNode &stride,
               std::optional<int64_t> cosizeHi) {
  std::vector<std::optional<int64_t>> shapes, strides;
  flattenLeaves(shape, shapes);
  flattenLeaves(stride, strides);
  if (!cosizeHi)
    return {PatternNode::leaf(std::nullopt), PatternNode::leaf(std::nullopt)};

  struct Mode {
    std::optional<int64_t> shape;
    std::optional<int64_t> stride;
    int64_t constStride;
    bool hasConst;
  };
  std::vector<Mode> modes;
  modes.reserve(shapes.size());
  bool allStatic = true;
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto s = (i < strides.size()) ? strides[i] : std::nullopt;
    bool has = s.has_value();
    allStatic &= has;
    modes.push_back({shapes[i], s, has ? *s : 0, has});
  }
  if (allStatic) {
    std::sort(modes.begin(), modes.end(),
              [](const Mode &a, const Mode &b) { return a.constStride < b.constStride; });
  }

  std::optional<int64_t> currStride = 1;
  std::vector<std::optional<int64_t>> outShapeLeaves;
  std::vector<std::optional<int64_t>> outStrideLeaves;

  for (auto &m : modes) {
    auto gap = divUI(m.stride, currStride);
    outShapeLeaves.push_back(gap);
    outStrideLeaves.push_back(currStride);
    currStride = mul(m.stride, m.shape);
  }

  auto finalRest = ceilDivUI(cosizeHi, currStride);
  outShapeLeaves.push_back(finalRest);
  outStrideLeaves.push_back(currStride);

  return {makeFlatTupleFromLeaves(outShapeLeaves),
          makeFlatTupleFromLeaves(outStrideLeaves)};
}

static std::optional<int64_t> sizeOfShape(const PatternNode &shape) {
  std::vector<std::optional<int64_t>> leaves;
  flattenLeaves(shape, leaves);
  std::optional<int64_t> prod = 1;
  for (auto &v : leaves)
    prod = mul(prod, v);
  return prod;
}

} // namespace

FailureOr<LayoutType> mlir::flir::inferCompositionType(MLIRContext *ctx,
                                                       LayoutType lhs,
                                                       LayoutType rhs) {
  PatternNode lhsShape = parsePatternAttr(lhs.getShapePattern());
  PatternNode lhsStride = parsePatternAttr(lhs.getStridePattern());
  PatternNode rhsShape = parsePatternAttr(rhs.getShapePattern());
  PatternNode rhsStride = parsePatternAttr(rhs.getStridePattern());

  // Runtime-capable policy: if any operand leaf is dynamic/unknown, return a
  // rank-only dynamic layout type and let lowering compute values at runtime.
  // (We cannot soundly predict the exact output rank/structure without static
  // values because the decomposition depends on divisions and early-exit tests.)
  if (!allStaticLeaves(lhsShape) || !allStaticLeaves(lhsStride) ||
      !allStaticLeaves(rhsShape) || !allStaticLeaves(rhsStride))
    return LayoutType::get(ctx, std::max(lhs.getRank(), rhs.getRank()));

  auto [outShape, outStride] = compositionImpl(lhsShape, lhsStride, rhsShape, rhsStride);
  int32_t dyncIdx = 0;
  Attribute outShapeAttr = patternNodeToAttr(ctx, outShape, dyncIdx);
  dyncIdx = 0;
  Attribute outStrideAttr = patternNodeToAttr(ctx, outStride, dyncIdx);
  return LayoutType::get(ctx, outShapeAttr, outStrideAttr);
}

FailureOr<LayoutType> mlir::flir::inferLogicalProductType(MLIRContext *ctx,
                                                          LayoutType block,
                                                          LayoutType tiler) {
  PatternNode bShape = parsePatternAttr(block.getShapePattern());
  PatternNode bStride = parsePatternAttr(block.getStridePattern());
  PatternNode tShape = parsePatternAttr(tiler.getShapePattern());
  PatternNode tStride = parsePatternAttr(tiler.getStridePattern());

  // Match current lowering's semantics: flatten then concatenate.
  std::vector<std::optional<int64_t>> bShapeLeaves, bStrideLeaves, tShapeLeaves, tStrideLeaves;
  flattenLeaves(bShape, bShapeLeaves);
  flattenLeaves(bStride, bStrideLeaves);
  flattenLeaves(tShape, tShapeLeaves);
  flattenLeaves(tStride, tStrideLeaves);

  // Conservative: require static.
  auto blockSize = sizeOfShape(bShape);
  if (!blockSize)
    return LayoutType::get(ctx, block.getRank() + tiler.getRank());
  for (auto &v : bStrideLeaves)
    if (!v)
      return LayoutType::get(ctx, block.getRank() + tiler.getRank());
  for (auto &v : tStrideLeaves)
    if (!v)
      return LayoutType::get(ctx, block.getRank() + tiler.getRank());

  std::vector<std::optional<int64_t>> outShapeLeaves = bShapeLeaves;
  outShapeLeaves.insert(outShapeLeaves.end(), tShapeLeaves.begin(), tShapeLeaves.end());

  std::vector<std::optional<int64_t>> outStrideLeaves = bStrideLeaves;
  for (auto &s : tStrideLeaves)
    outStrideLeaves.push_back(mul(s, blockSize));

  PatternNode outShape = makeFlatTupleFromLeaves(outShapeLeaves);
  PatternNode outStride = makeFlatTupleFromLeaves(outStrideLeaves);
  int32_t dyncIdx = 0;
  Attribute outShapeAttr = patternNodeToAttr(ctx, outShape, dyncIdx);
  dyncIdx = 0;
  Attribute outStrideAttr = patternNodeToAttr(ctx, outStride, dyncIdx);
  return LayoutType::get(ctx, outShapeAttr, outStrideAttr);
}

FailureOr<LayoutType> mlir::flir::inferLogicalDivideType(MLIRContext *ctx,
                                                         LayoutType layout,
                                                         LayoutType tiler) {
  PatternNode lShape = parsePatternAttr(layout.getShapePattern());
  PatternNode lStride = parsePatternAttr(layout.getStridePattern());
  PatternNode tShape = parsePatternAttr(tiler.getShapePattern());
  PatternNode tStride = parsePatternAttr(tiler.getStridePattern());

  if (!allStaticLeaves(lShape) || !allStaticLeaves(lStride) ||
      !allStaticLeaves(tShape) || !allStaticLeaves(tStride))
    return LayoutType::get(ctx, std::max(layout.getRank(), tiler.getRank()));

  auto inputSize = sizeOfShape(lShape);
  auto [compShape, compStride] = complementImpl(tShape, tStride, inputSize);

  // RHS = (tiler, complement)
  PatternNode rhsShape = PatternNode::tuple({tShape, compShape});
  PatternNode rhsStride = PatternNode::tuple({tStride, compStride});

  auto [outShape, outStride] = compositionImpl(lShape, lStride, rhsShape, rhsStride);
  int32_t dyncIdx = 0;
  Attribute outShapeAttr = patternNodeToAttr(ctx, outShape, dyncIdx);
  dyncIdx = 0;
  Attribute outStrideAttr = patternNodeToAttr(ctx, outStride, dyncIdx);
  return LayoutType::get(ctx, outShapeAttr, outStrideAttr);
}



