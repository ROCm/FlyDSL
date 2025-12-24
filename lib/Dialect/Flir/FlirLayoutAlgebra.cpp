//===- FlirLayoutAlgebra.cpp - Type-level layout algebra helpers ---------===//

#include "flir/FlirLayoutAlgebra.h"

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

static PatternNode parsePattern(ArrayRef<int32_t> structure, ArrayRef<int64_t> dims) {
  // structure: preorder encoding (tuple -> N, leaf -> -1)
  // dims: leaf values (int64), -1 means dynamic
  size_t sIdx = 0;
  size_t dIdx = 0;

  std::function<PatternNode()> parse = [&]() -> PatternNode {
    if (sIdx >= structure.size())
      return PatternNode::leaf(std::nullopt);
    int32_t code = structure[sIdx++];
    if (code == -1) {
      if (dIdx >= dims.size())
        return PatternNode::leaf(std::nullopt);
      int64_t v = dims[dIdx++];
      if (v == -1)
        return PatternNode::leaf(std::nullopt);
      return PatternNode::leaf(v);
    }
    // tuple
    int32_t arity = code;
    std::vector<PatternNode> children;
    children.reserve(std::max<int32_t>(0, arity));
    for (int32_t i = 0; i < arity; ++i)
      children.push_back(parse());
    return PatternNode::tuple(std::move(children));
  };

  PatternNode root = parse();
  return root;
}

static void serializePattern(const PatternNode &n,
                             llvm::SmallVectorImpl<int32_t> &structure,
                             llvm::SmallVectorImpl<int64_t> &dims) {
  if (n.isLeaf) {
    structure.push_back(-1);
    dims.push_back(n.value.has_value() ? *n.value : -1);
    return;
  }
  structure.push_back(static_cast<int32_t>(n.children.size()));
  for (auto &c : n.children)
    serializePattern(c, structure, dims);
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
  // Need structure+dims to do anything meaningful.
  if (lhs.getShapeDims().empty() || lhs.getStrideDims().empty() ||
      rhs.getShapeDims().empty() || rhs.getStrideDims().empty())
    return LayoutType::get(ctx, std::max(lhs.getRank(), rhs.getRank()));

  PatternNode lhsShape = parsePattern(lhs.getShapeStructure(), lhs.getShapeDims());
  PatternNode lhsStride = parsePattern(lhs.getStrideStructure(), lhs.getStrideDims());
  PatternNode rhsShape = parsePattern(rhs.getShapeStructure(), rhs.getShapeDims());
  PatternNode rhsStride = parsePattern(rhs.getStrideStructure(), rhs.getStrideDims());

  // Conservative: require fully static leaves for now.
  if (!allStaticLeaves(lhsShape) || !allStaticLeaves(lhsStride) ||
      !allStaticLeaves(rhsShape) || !allStaticLeaves(rhsStride))
    return LayoutType::get(ctx, std::max(lhs.getRank(), rhs.getRank()));

  auto [outShape, outStride] = compositionImpl(lhsShape, lhsStride, rhsShape, rhsStride);

  llvm::SmallVector<int32_t, 16> shapeStruct, strideStruct;
  llvm::SmallVector<int64_t, 16> shapeDims, strideDims;
  serializePattern(outShape, shapeStruct, shapeDims);
  serializePattern(outStride, strideStruct, strideDims);
  return LayoutType::get(ctx, shapeStruct, shapeDims, strideStruct, strideDims);
}

FailureOr<LayoutType> mlir::flir::inferLogicalProductType(MLIRContext *ctx,
                                                          LayoutType block,
                                                          LayoutType tiler) {
  if (block.getShapeDims().empty() || block.getStrideDims().empty() ||
      tiler.getShapeDims().empty() || tiler.getStrideDims().empty())
    return LayoutType::get(ctx, block.getRank() + tiler.getRank());

  PatternNode bShape = parsePattern(block.getShapeStructure(), block.getShapeDims());
  PatternNode bStride = parsePattern(block.getStrideStructure(), block.getStrideDims());
  PatternNode tShape = parsePattern(tiler.getShapeStructure(), tiler.getShapeDims());
  PatternNode tStride = parsePattern(tiler.getStrideStructure(), tiler.getStrideDims());

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

  llvm::SmallVector<int32_t, 16> shapeStruct, strideStruct;
  llvm::SmallVector<int64_t, 16> shapeDims, strideDims;
  serializePattern(outShape, shapeStruct, shapeDims);
  serializePattern(outStride, strideStruct, strideDims);
  return LayoutType::get(ctx, shapeStruct, shapeDims, strideStruct, strideDims);
}

FailureOr<LayoutType> mlir::flir::inferLogicalDivideType(MLIRContext *ctx,
                                                         LayoutType layout,
                                                         LayoutType tiler) {
  if (layout.getShapeDims().empty() || layout.getStrideDims().empty() ||
      tiler.getShapeDims().empty() || tiler.getStrideDims().empty())
    return LayoutType::get(ctx, std::max(layout.getRank(), tiler.getRank()));

  PatternNode lShape = parsePattern(layout.getShapeStructure(), layout.getShapeDims());
  PatternNode lStride = parsePattern(layout.getStrideStructure(), layout.getStrideDims());
  PatternNode tShape = parsePattern(tiler.getShapeStructure(), tiler.getShapeDims());
  PatternNode tStride = parsePattern(tiler.getStrideStructure(), tiler.getStrideDims());

  if (!allStaticLeaves(lShape) || !allStaticLeaves(lStride) ||
      !allStaticLeaves(tShape) || !allStaticLeaves(tStride))
    return LayoutType::get(ctx, std::max(layout.getRank(), tiler.getRank()));

  auto inputSize = sizeOfShape(lShape);
  auto [compShape, compStride] = complementImpl(tShape, tStride, inputSize);

  // RHS = (tiler, complement)
  PatternNode rhsShape = PatternNode::tuple({tShape, compShape});
  PatternNode rhsStride = PatternNode::tuple({tStride, compStride});

  auto [outShape, outStride] = compositionImpl(lShape, lStride, rhsShape, rhsStride);

  llvm::SmallVector<int32_t, 16> shapeStruct, strideStruct;
  llvm::SmallVector<int64_t, 16> shapeDims, strideDims;
  serializePattern(outShape, shapeStruct, shapeDims);
  serializePattern(outStride, strideStruct, strideDims);
  return LayoutType::get(ctx, shapeStruct, shapeDims, strideStruct, strideDims);
}



