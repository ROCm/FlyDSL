// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// The expansion of copies and gemms into atom calls.

#include "flydsl/Dialect/Fly/Utils/AtomUtils.h"
#include "flydsl/Dialect/Fly/Utils/CoreBridge.h"

#include <array>
#include <variant>

namespace mlir::fly {

//===----------------------------------------------------------------------===//
// Atom expansion
//===----------------------------------------------------------------------===//

namespace {

using IntTupleTrait = core::IntTuple<core::Trait>;
using IntTupleEmit = core::IntTuple<core::EmitOp>;
using MemRefEmit = core::MemRef<core::EmitOp>;
using TensorEmit = std::variant<MemRefEmit, core::CoordTensor<core::EmitOp>>;

std::optional<int64_t> staticSize(const IntTupleTrait &shape) {
  auto size = core::product(shape);
  if (!size.isStatic())
    return std::nullopt;
  return size.staticValue();
}

/// The coordinate (*, idx...) that keeps mode 0.
IntTupleEmit sliceCoord(ArrayRef<int32_t> idx) {
  SmallVector<IntTupleEmit, 4> elems = {IntTupleEmit::getNone()};
  for (int32_t i : idx)
    elems.push_back(IntTupleEmit::getSInt(i));
  return core::make_tuple(elems);
}

/// Mode 0 of `tensor` at `coord`. Core slice keeps the selected mode in a
/// rank-1 tuple, which get unwraps.
template <class Tensor> Tensor sliceMode0(const IntTupleEmit &coord, const Tensor &tensor) {
  return core::get(core::slice(coord, tensor), 0);
}

//===----------------------------------------------------------------------===//
// Copy expansion
//===----------------------------------------------------------------------===//

/// A copy operand. It keeps its original value until a Core step rewrites it,
/// so an untouched operand reaches the atom without being rebuilt.
struct Operand {
  Value original;
  TensorEmit tensor;

  IntTupleTrait shape() const {
    return std::visit([](const auto &t) { return IntTupleTrait(core::to_trait(t.shape())); },
                      tensor);
  }
  int32_t rank() const {
    return std::visit([](const auto &t) { return int32_t(t.rank()); }, tensor);
  }
  template <class F> Operand map(F &&f) const {
    return {Value(), std::visit([&](const auto &t) -> TensorEmit { return f(t); }, tensor)};
  }
};

FailureOr<Operand> importOperand(Value value, core::EmitContext &ctx) {
  if (isa<fly::MemRefType>(value.getType())) {
    auto tensor = importValue<fly::MemRefType>(value, ctx);
    if (failed(tensor))
      return failure();
    return Operand{value, std::move(*tensor)};
  }
  auto tensor = importValue<CoordTensorType>(value, ctx);
  if (failed(tensor))
    return failure();
  return Operand{value, std::move(*tensor)};
}

FailureOr<Value> materialize(const Operand &operand, core::EmitContext &ctx) {
  if (operand.original)
    return operand.original;
  return std::visit([&](const auto &t) { return materializeValue(t, ctx); }, operand.tensor);
}

class CopyExpansion {
public:
  CopyExpansion(core::EmitContext &ctx, std::optional<int64_t> atomValsSrc,
                std::optional<int64_t> atomValsDst, EmitCopyAtomFn emitAtom)
      : ctx(ctx), atomValsSrc(atomValsSrc), atomValsDst(atomValsDst), emitAtom(emitAtom) {}

  LogicalResult expand(const Operand &src, const Operand &dst, const std::optional<Operand> &pred) {
    auto decompose = [](const auto &t) { return core::decompose(t); };
    auto srcShape = src.shape();
    // A call is issued the moment the operands hold exactly one atom's values;
    // descending to a leaf is wrong for atoms whose values are a nested tuple.
    // Composed-layout offsets are materialized before the atom consumes the
    // pointer, or the LDS swizzle offset is lost.
    if (atomValsSrc && atomValsDst && staticSize(srcShape) == atomValsSrc &&
        staticSize(dst.shape()) == atomValsDst) {
      auto decomposeMemRef = [&](const Operand &operand) {
        return std::holds_alternative<MemRefEmit>(operand.tensor) ? operand.map(decompose)
                                                                  : operand;
      };
      return call(decomposeMemRef(src), decomposeMemRef(dst), pred);
    }

    // A single atom may have different source/destination profiles (coordinate
    // bases do not coalesce like contiguous LDS addresses). Only recursive
    // decomposition requires equal outer ranks.
    int32_t rank = src.rank();
    if (rank != dst.rank())
      return failure();

    std::optional<Operand> alignedPred;
    if (pred) {
      auto aligned = alignPredicate(srcShape, *pred);
      if (failed(aligned))
        return failure();
      alignedPred = std::move(*aligned);
    }

    if (rank == 1) {
      if (srcShape.isLeaf())
        return call(src.map(decompose), dst.map(decompose), alignedPred);
      auto unwrap = [](const auto &t) { return core::get(t, 0); };
      return expand(src.map(unwrap), dst.map(unwrap),
                    alignedPred ? std::optional(alignedPred->map(unwrap)) : std::nullopt);
    }

    auto restSize = staticSize(core::take(srcShape, 1, rank));
    if (!restSize || (alignedPred && alignedPred->rank() != rank))
      return failure();

    auto group = [&](const auto &t) { return core::group(t, 1, rank); };
    auto srcGrouped = src.map(group);
    auto dstGrouped = dst.map(group);
    auto predGrouped = alignedPred ? std::optional(alignedPred->map(group)) : std::nullopt;
    for (int64_t i = 0; i < *restSize; ++i) {
      auto coord = sliceCoord({static_cast<int32_t>(i)});
      auto slice = [&](const auto &t) { return sliceMode0(coord, t); };
      if (failed(expand(srcGrouped.map(slice), dstGrouped.map(slice),
                        predGrouped ? std::optional(predGrouped->map(slice)) : std::nullopt)))
        return failure();
    }
    return success();
  }

private:
  FailureOr<Operand> alignPredicate(const IntTupleTrait &srcShape, const Operand &pred) {
    auto *memref = std::get_if<MemRefEmit>(&pred.tensor);
    if (!atomValsSrc || !memref ||
        !std::holds_alternative<core::Layout<core::EmitOp>>(memref->layout()))
      return pred;
    auto restShape = core::shape_div(srcShape, IntTupleTrait::getSInt(*atomValsSrc));
    auto predShape = pred.shape();
    if (!restShape.isStatic() || predShape == restShape)
      return pred;
    if (core::product(predShape) != core::product(restShape)) {
      if (core::product(predShape) == core::product(srcShape))
        return pred;
      mlir::emitError(ctx.location)
          << "copy predicate of shape " << core::toString(predShape) << " needs the source shape "
          << core::toString(srcShape) << " or one predicate per atom " << core::toString(restShape);
      return failure();
    }
    auto restLayout = core::bind(core::make_layout(restShape), core::EmitBindings{}, ctx);
    if (failed(restLayout))
      return failure();
    return pred.map([&](const auto &t) { return t.compose(*restLayout); });
  }

  LogicalResult call(const Operand &src, const Operand &dst, const std::optional<Operand> &pred) {
    auto srcValue = materialize(src, ctx);
    auto dstValue = materialize(dst, ctx);
    auto predValue = pred ? materialize(*pred, ctx) : FailureOr<Value>(Value());
    if (failed(srcValue) || failed(dstValue) || failed(predValue))
      return failure();
    emitAtom(*srcValue, *dstValue, *predValue);
    return success();
  }

  core::EmitContext &ctx;
  std::optional<int64_t> atomValsSrc;
  std::optional<int64_t> atomValsDst;
  EmitCopyAtomFn emitAtom;
};

//===----------------------------------------------------------------------===//
// Gemm expansion
//===----------------------------------------------------------------------===//

struct MmaTile {
  int32_t m, n, k;
};

/// The loop nest of `order` over the dims M=0, N=1, K=2, outermost first (the
/// order's name lists the dims fastest first), and whether it is serpentine.
std::pair<std::array<int32_t, 3>, bool> loopNest(GemmTraversalOrder order) {
  switch (order) {
  case GemmTraversalOrder::KMN:
    return {{1, 0, 2}, false};
  case GemmTraversalOrder::KNM:
    return {{0, 1, 2}, false};
  case GemmTraversalOrder::MKN:
    return {{1, 2, 0}, false};
  case GemmTraversalOrder::MNK:
    return {{2, 1, 0}, false};
  case GemmTraversalOrder::NKM:
    return {{0, 2, 1}, false};
  case GemmTraversalOrder::NMK:
    return {{2, 0, 1}, false};
  case GemmTraversalOrder::KMN_Serpentine:
    return {{1, 0, 2}, true};
  case GemmTraversalOrder::KNM_Serpentine:
    return {{0, 1, 2}, true};
  case GemmTraversalOrder::MKN_Serpentine:
    return {{1, 2, 0}, true};
  case GemmTraversalOrder::MNK_Serpentine:
    return {{2, 1, 0}, true};
  case GemmTraversalOrder::NKM_Serpentine:
    return {{0, 2, 1}, true};
  case GemmTraversalOrder::NMK_Serpentine:
    return {{2, 0, 1}, true};
  }
  llvm_unreachable("unknown gemm traversal order");
}

/// The (m, n, k) tiles in traversal order; a 2D gemm is the K = 1 case.
FailureOr<SmallVector<MmaTile>> gemmTiles(int32_t loopM, int32_t loopN, int32_t loopK,
                                          std::optional<GemmTraversalOrder> order,
                                          LayoutAttr traversalLayout) {
  SmallVector<MmaTile> tiles;
  if (traversalLayout) {
    const auto &tv = traversalLayout.getRef();
    int64_t total = int64_t(loopM) * loopN * loopK;
    auto cosize = core::cosize(tv);
    if (!traversalLayout.isStaticShape() || !traversalLayout.isStaticStride() ||
        !cosize.isStatic() || cosize.staticValue() != total)
      return failure();
    auto naturalShape =
        core::make_tuple(IntTupleTrait::getSInt(loopM), IntTupleTrait::getSInt(loopN),
                         IntTupleTrait::getSInt(loopK));
    auto naturalStride = core::make_tuple(IntTupleTrait::getSInt(1), IntTupleTrait::getSInt(loopM),
                                          IntTupleTrait::getSInt(int64_t(loopM) * loopN));
    for (int64_t i = 0; i < total; ++i) {
      auto linearIdx = core::crd2idx(IntTupleTrait::getSInt(i), tv.shape(), tv.stride());
      auto coord = core::idx2crd(linearIdx, naturalShape, naturalStride);
      tiles.push_back({static_cast<int32_t>(coord.at(0).getLeaf().staticValue()),
                       static_cast<int32_t>(coord.at(1).getLeaf().staticValue()),
                       static_cast<int32_t>(coord.at(2).getLeaf().staticValue())});
    }
    return tiles;
  }

  auto [nest, serpentine] = loopNest(order.value_or(GemmTraversalOrder::NMK));
  int32_t bounds[3] = {loopM, loopN, loopK};
  int32_t idx[3] = {0, 0, 0};
  for (int32_t i0 = 0; i0 < bounds[nest[0]]; ++i0) {
    idx[nest[0]] = i0;
    for (int32_t i1 = 0; i1 < bounds[nest[1]]; ++i1) {
      idx[nest[1]] = serpentine && (i0 & 1) ? bounds[nest[1]] - 1 - i1 : i1;
      for (int32_t i2 = 0; i2 < bounds[nest[2]]; ++i2) {
        idx[nest[2]] =
            serpentine && ((i0 * bounds[nest[1]] + i1) & 1) ? bounds[nest[2]] - 1 - i2 : i2;
        tiles.push_back({idx[0], idx[1], idx[2]});
      }
    }
  }
  return tiles;
}

} // namespace

LogicalResult expandCopy(OpBuilder &builder, Location loc, Value src, Value dst, Value pred,
                         std::optional<int64_t> atomValsSrc, std::optional<int64_t> atomValsDst,
                         EmitCopyAtomFn emitAtom) {
  auto ctx = makeDefaultEmitContext(builder, loc);
  auto srcOperand = importOperand(src, ctx);
  auto dstOperand = importOperand(dst, ctx);
  if (failed(srcOperand) || failed(dstOperand))
    return failure();
  std::optional<Operand> predOperand;
  if (pred) {
    auto imported = importOperand(pred, ctx);
    if (failed(imported))
      return failure();
    predOperand = std::move(*imported);
  }
  return CopyExpansion(ctx, atomValsSrc, atomValsDst, emitAtom)
      .expand(*srcOperand, *dstOperand, predOperand);
}

LogicalResult expandGemm(OpBuilder &builder, Location loc, Value d, ValueRange a, ValueRange b,
                         Value c, std::optional<GemmTraversalOrder> order,
                         LayoutAttr traversalLayout, EmitMmaAtomFn emitAtom) {
  auto layoutOf = [](Value value) -> const core::Layout<core::Trait> & {
    return cast<LayoutAttr>(cast<fly::MemRefType>(value.getType()).getLayout()).getRef();
  };
  const auto &dLayout = layoutOf(d);
  const auto &aLayout = layoutOf(a.front());
  const auto &bLayout = layoutOf(b.front());
  const auto &cLayout = layoutOf(c);

  if (dLayout.rank() == 1 && aLayout.rank() == 1 && bLayout.rank() == 1 && cLayout.rank() == 1) {
    emitAtom(d, a, b, c);
    return success();
  }

  bool hasK = aLayout.rank() == 3 && bLayout.rank() == 3;
  if (dLayout.rank() != 3 || cLayout.rank() != 3 ||
      !(hasK || (aLayout.rank() == 2 && bLayout.rank() == 2)))
    return failure();

  auto modeSize = [](const core::Layout<core::Trait> &layout, int32_t mode) {
    return staticSize(IntTupleTrait(layout.shape().at(mode)));
  };
  auto loopM = modeSize(dLayout, 1);
  auto loopN = modeSize(dLayout, 2);
  auto loopK = hasK ? modeSize(aLayout, 2) : std::optional<int64_t>(1);
  if (!loopM || !loopN || !loopK || modeSize(aLayout, 1) != loopM ||
      modeSize(bLayout, 1) != loopN || modeSize(cLayout, 1) != loopM ||
      modeSize(cLayout, 2) != loopN || (hasK && modeSize(bLayout, 2) != loopK))
    return failure();

  auto tiles = gemmTiles(*loopM, *loopN, *loopK, order, traversalLayout);
  if (failed(tiles))
    return failure();

  auto ctx = makeDefaultEmitContext(builder, loc);
  auto importGroup = [&](ValueRange values) -> FailureOr<SmallVector<MemRefEmit, 2>> {
    SmallVector<MemRefEmit, 2> tensors;
    for (Value value : values) {
      auto tensor = importValue<fly::MemRefType>(value, ctx);
      if (failed(tensor))
        return failure();
      tensors.push_back(std::move(*tensor));
    }
    return tensors;
  };
  auto dTensor = importValue<fly::MemRefType>(d, ctx);
  auto cTensor = importValue<fly::MemRefType>(c, ctx);
  auto aTensors = importGroup(a);
  auto bTensors = importGroup(b);
  if (failed(dTensor) || failed(cTensor) || failed(aTensors) || failed(bTensors))
    return failure();

  // Every tensor of a group is sliced at the same tile coordinate.
  auto sliceGroup = [&](ArrayRef<MemRefEmit> tensors,
                        const IntTupleEmit &coord) -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> slices;
    for (const auto &tensor : tensors) {
      auto slice = materializeValue(sliceMode0(coord, tensor), ctx);
      if (failed(slice))
        return failure();
      slices.push_back(*slice);
    }
    return slices;
  };

  // The accumulator source: c on the first visit of a tile, d afterwards.
  SmallVector<bool> visited(*loopM * *loopN, false);
  for (auto [m, n, k] : *tiles) {
    bool &seen = visited[m * *loopN + n];
    const MemRefEmit &accumulator = seen ? *dTensor : *cTensor;
    seen = true;
    auto mn = sliceCoord({m, n});
    auto aSlices = sliceGroup(*aTensors, hasK ? sliceCoord({m, k}) : sliceCoord({m}));
    auto bSlices = sliceGroup(*bTensors, hasK ? sliceCoord({n, k}) : sliceCoord({n}));
    auto cSlice = materializeValue(sliceMode0(mn, accumulator), ctx);
    auto dSlice = materializeValue(sliceMode0(mn, *dTensor), ctx);
    if (failed(aSlices) || failed(bSlices) || failed(cSlice) || failed(dSlice))
      return failure();
    emitAtom(*dSlice, *aSlices, *bSlices, *cSlice);
  }
  return success();
}

} // namespace mlir::fly
