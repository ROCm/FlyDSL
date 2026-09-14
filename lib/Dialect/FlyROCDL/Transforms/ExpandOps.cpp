// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Spend the geometry `fly_rocdl.make_tiled_tdm_{load,store}_atom` derived during type
// inference.
//
// The op's *type* already says everything static about the descriptor — the box, the
// mode map, the coordinate strides — because the derivation ran to produce it. What is
// left is the run-time half: the tensor's dynamic extents and strides, which have to be
// opened out of the tensor operand and handed to `fly.make_copy_atom` as the atom's
// construction arguments. So the derivation runs a second time here rather than
// being cached on the op: a side table could disagree with the type, and re-deriving is
// pure attribute algebra that costs nothing.
//
// This pass is also where an `boundary_check` state stops being written in the tensor's language
// and starts being written in the descriptor's. A caller sets one flag per *tensor mode*, because
// the descriptor's dims are not theirs to know — they permute, and they pack the tail. Translating
// that is `tensor2tdm` and nothing else, so it is done once, here, and the result is a flat
// `boundary_check_axes` tuple with one leaf per descriptor axis that every later pass can read
// without knowing what a mode was.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"
#include "flydsl/Dialect/FlyROCDL/Utils/TdmAtomBuilder.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_rocdl;

namespace mlir {
namespace fly_rocdl {
#define GEN_PASS_DEF_FLYROCDLEXPANDOPSPASS
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h.inc"
} // namespace fly_rocdl
} // namespace mlir

namespace {

/// Pulls the dynamic leaves out of the tensor's layout, once per side and on demand.
///
/// A Fly int tuple keeps its static leaves in its type and only its dynamic ones as SSA
/// values, so a mode's run-time extent is the *n*-th result of `fly.get_leaves`, where
/// *n* counts the dynamic leaves before it. A tensor with no dynamic geometry never
/// reaches this and emits nothing — not even the `fly.get_layout` that opens the tensor
/// up, which is why that one is materialized here rather than by the caller.
class LayoutLeaves {
public:
  LayoutLeaves(OpBuilder &builder, Location loc, Value tensor, LayoutAttr attr)
      : builder(builder), loc(loc), tensor(tensor), attr(attr) {}

  /// The SSA value of global mode `mode`'s extent (`fromShape`) or stride.
  Value get(int32_t mode, bool fromShape) {
    Side &side = fromShape ? shapeSide : strideSide;
    IntTupleAttr tuple = fromShape ? attr.getShape() : attr.getStride();
    if (!side.materialized) {
      if (!layout)
        layout = GetLayoutOp::create(builder, loc, tensor);
      Value tupleVal = fromShape ? Value(GetShapeOp::create(builder, loc, layout))
                                 : Value(GetStrideOp::create(builder, loc, layout));
      auto leaves = GetLeavesOp::create(builder, loc, tupleVal, /*dynamicOnly=*/true);
      side.values.assign(leaves.getResults().begin(), leaves.getResults().end());
      side.materialized = true;
    }
    // The n-th dynamic leaf, counting depth-first the way `get_leaves` orders them.
    int32_t dynIndex = 0;
    int32_t seen = 0;
    if (!countTo(tuple, mode, seen, dynIndex))
      return nullptr;
    if (dynIndex < 0 || dynIndex >= static_cast<int32_t>(side.values.size()))
      return nullptr;
    return side.values[dynIndex];
  }

private:
  struct Side {
    bool materialized = false;
    SmallVector<Value> values;
  };

  /// Walk to leaf `target`, counting the dynamic leaves passed on the way. Returns false
  /// when the target leaf is itself static (it has no SSA value to find).
  static bool countTo(IntTupleAttr t, int32_t target, int32_t &seen, int32_t &dynIndex) {
    if (t.isLeaf()) {
      if (seen == target) {
        if (t.isStatic())
          return false;
        return true;
      }
      if (!t.isStatic())
        ++dynIndex;
      ++seen;
      return false;
    }
    for (int32_t i = 0; i < t.rank(); ++i)
      if (countTo(t.at(i), target, seen, dynIndex))
        return true;
    return false;
  }

  OpBuilder &builder;
  Location loc;
  Value tensor;
  LayoutAttr attr;
  Value layout;
  Side shapeSide, strideSide;
};

/// Materialize one descriptor scalar as a value of `width` bits.
Value materializeScalar(OpBuilder &builder, Location loc, const tdm::Scalar &scalar, unsigned width,
                        LayoutLeaves &leaves) {
  Type ty = builder.getIntegerType(width);
  if (scalar.isStatic)
    return arith::ConstantIntOp::create(builder, loc, ty, scalar.value);

  Value leaf = leaves.get(scalar.mode, scalar.fromShape);
  if (!leaf)
    return nullptr;
  unsigned leafWidth = leaf.getType().getIntOrFloatBitWidth();
  if (leafWidth < width)
    leaf = arith::ExtUIOp::create(builder, loc, ty, leaf);
  else if (leafWidth > width)
    leaf = arith::TruncIOp::create(builder, loc, ty, leaf);
  if (scalar.divisor != 1) {
    // The recast's division, deferred to run time: the caller asserted the tensor is laid
    // out in whole internal units, which a value only known now cannot be checked against.
    Value divisor = arith::ConstantIntOp::create(builder, loc, ty, scalar.divisor);
    leaf = arith::DivUIOp::create(builder, loc, leaf, divisor);
  }
  return leaf;
}

/// An all-static int tuple, as a value.
Value staticTuple(OpBuilder &builder, Location loc, IntTupleAttr attr) {
  return MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr), ValueRange{});
}

/// The coordinate tensor's shape, as a value.
///
/// Unlike its base and its stride, this one is not all-static: it is the global tensor's
/// own shape — except under a recast, which splits the contiguous mode into
/// `(ratio, extent / ratio)` — so a tensor with run-time extents leaves dynamic leaves in
/// it, and those have to be resolved out of the layout operand exactly like the
/// descriptor's extents are. An operand-less `fly.make_int_tuple` would type-check and
/// then have nothing behind those leaves for a later pass to read.
Value materializeCoordShape(OpBuilder &builder, Location loc, IntTupleAttr shape,
                            const tdm::Geometry &geometry, LayoutLeaves &leaves) {
  // One scalar per leaf, in the order `tdm::coordLayout` laid them out.
  SmallVector<tdm::Scalar> scalars;
  for (int32_t mode = 0; mode < static_cast<int32_t>(geometry.modeExtent.size()); ++mode) {
    if (geometry.ratio != 1 && mode == geometry.contiguousMode)
      scalars.push_back(tdm::Scalar::getStatic(geometry.ratio));
    scalars.push_back(geometry.modeExtent[mode]);
  }

  SmallVector<IntTupleAttr> flat;
  intTupleFlattenToVector(IntTupleBuilder<IntTupleAttr>(shape.getContext()), shape, flat);
  if (flat.size() != scalars.size())
    return nullptr;

  SmallVector<Value> dynamic;
  for (auto [leaf, scalar] : llvm::zip_equal(flat, scalars)) {
    if (leaf.isStatic())
      continue;
    Value v = materializeScalar(builder, loc, scalar, leaf.extractIntFromLeaf().getWidth(), leaves);
    if (!v)
      return nullptr;
    dynamic.push_back(v);
  }
  return MakeIntTupleOp::create(builder, loc, IntTupleType::get(shape), dynamic);
}

/// Rewrite one builder into the ops that build the atom.
///
/// Driven by a plain walk rather than the greedy driver: the expansion never produces
/// another builder, so there is nothing to iterate to a fixed point, and the greedy
/// driver would additionally delete every other dead `Pure` op in the function — which
/// is not this pass's business, and would erase exactly the IR a caller wants to inspect.
template <typename OpT> LogicalResult expandOne(OpT op, IRRewriter &rewriter) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  auto emitError = [&]() -> InFlightDiagnostic { return op.emitOpError(); };

  tdm::Request request;
  Type dataType;
  FailureOr<tdm::Geometry> geometry =
      deriveTdmAtom(typename OpT::Adaptor(op), request, dataType, emitError);
  if (failed(geometry))
    return failure();

  auto atomTy = dyn_cast<CopyAtomType>(op.getAtom().getType());
  auto coordTy = dyn_cast<CoordTensorType>(op.getCoordTensor().getType());
  if (!atomTy || !coordTy)
    return failure();

  int32_t descRank = geometry->rank();
  LayoutLeaves leaves(rewriter, loc, op.getTensor(), request.gLayout);

  // Construction arguments, in tensor dim order (the reverse of descriptor order, which
  // is what the atom's `tileShape` and its state slots use): base pointer, then the
  // strides of dims 0..rank-2 (the innermost is 1 by construction and is not passed),
  // then every dim's extent. Every dim passes its extent whether or not it currently
  // clamps: `boundary_check` is per-call state, so a later call site may switch its dim on and
  // needs the bound already in the atom, and an extent no call reads dies with the rest
  // of the scalarized state struct.
  SmallVector<Value> args;
  args.push_back(GetIterOp::create(rewriter, loc, op.getTensor()));
  for (int32_t i = 0; i < descRank - 1; ++i) {
    const tdm::Dim &dim = geometry->dims[descRank - 1 - i];
    Value v = materializeScalar(rewriter, loc, dim.stride, 64, leaves);
    if (!v)
      return op.emitOpError() << "could not materialize the stride of descriptor dim "
                              << (descRank - 1 - i);
    args.push_back(v);
  }
  for (int32_t i = 0; i < descRank; ++i) {
    const tdm::Dim &dim = geometry->dims[descRank - 1 - i];
    Value v = materializeScalar(rewriter, loc, dim.tensorDim, 32, leaves);
    if (!v)
      return op.emitOpError() << "could not materialize the extent of descriptor dim "
                              << (descRank - 1 - i);
    args.push_back(v);
  }
  if (geometry->iterCount > 1) {
    Value v = materializeScalar(rewriter, loc, geometry->iterStride, 64, leaves);
    if (!v)
      return op.emitOpError() << "could not materialize the iteration stride";
    args.push_back(v);
  }

  Value atom = MakeCopyAtomOp::create(rewriter, loc, atomTy, args,
                                      static_cast<int32_t>(dataType.getIntOrFloatBitWidth()));

  // The initial `boundary_check` state, shaped like the global tensor: the builder's one
  // flag written onto every mode that has a bound to give. A mode with nothing to clamp —
  // size-1, stride-0, not spanned by the box, or sharing the rank-5 packing's dim — comes
  // out off, so the flag cannot fail here; naming such a mode is only an error when a call
  // site names it, which is the `set_value` below.
  FailureOr<IntTupleAttr> boundaryCheck = tdm::initialBoundaryCheck(
      *geometry, request.gLayout.getShape(), request.initBoundaryCheck, emitError);
  if (failed(boundaryCheck))
    return failure();
  atom = AtomSetValueOp::create(rewriter, loc, atom.getType(), atom,
                                rewriter.getStringAttr("boundary_check"),
                                staticTuple(rewriter, loc, *boundaryCheck));

  // The coordinate tensor, at the tensor origin: tiling and slicing it is what moves it,
  // and that folds into its type.
  Value origin = staticTuple(rewriter, loc, coordTy.getBase());
  auto coordLayoutAttr = cast<LayoutAttr>(coordTy.getLayout());
  Value shape = materializeCoordShape(rewriter, loc, coordLayoutAttr.getShape(), *geometry, leaves);
  if (!shape)
    return op.emitOpError() << "could not materialize the coordinate tensor's shape";
  Value stride = staticTuple(rewriter, loc, coordLayoutAttr.getStride());
  Value layout =
      MakeLayoutOp::create(rewriter, loc, LayoutType::get(coordLayoutAttr), shape, stride);
  Value coord = MakeViewOp::create(rewriter, loc, coordTy, origin, layout);

  rewriter.replaceOp(op, {atom, coord});
  return success();
}

//===----------------------------------------------------------------------===//
// "boundary_check" (tensor modes) -> "boundary_check_axes" (descriptor axes)
//===----------------------------------------------------------------------===//

/// The CDNA5 TDM atom a `fly.atom.set_value` is setting, or a null pair if it is setting
/// something else's state.
std::pair<IntTupleAttr, int32_t> tdmAtomShape(Value atom) {
  auto atomTy = dyn_cast<CopyAtomType>(atom.getType());
  if (!atomTy)
    return {};
  if (auto load = dyn_cast<CopyOpCDNA5TensorLoadType>(atomTy.getCopyOp()))
    return {load.getTensor2tdm(), static_cast<int32_t>(load.getTileShape().size())};
  if (auto store = dyn_cast<CopyOpCDNA5TensorStoreType>(atomTy.getCopyOp()))
    return {store.getTensor2tdm(), static_cast<int32_t>(store.getTileShape().size())};
  return {};
}

/// Translate one tensor-order `boundary_check` tuple into the flat axis-order tuple the lowering
/// reads, or fail with the diagnostic the caller earned.
///
/// Several modes can land on one axis — a merge or the rank-5 packing puts them there —
/// and the axis clamps if any of them asked for a bound, so this is an OR. A static flag
/// wins it outright, which is why the static leaves are swept first: an axis already
/// pinned on spends no `cmpi` on the dynamic leaves that would only OR into it. Both
/// sweeps still walk every dynamic leaf, because their SSA values are positional in the
/// tuple's `make_int_tuple` and skipping one would shift every later mode's.
FailureOr<Value> normalizeBoundaryCheckToAxes(OpBuilder &builder, Location loc, Value flags,
                                              IntTupleAttr tensor2tdm, int32_t descRank,
                                              function_ref<InFlightDiagnostic()> emitError) {
  auto tupleTy = dyn_cast<IntTupleType>(flags.getType());
  if (!tupleTy)
    return emitError()
           << "\"boundary_check\" must be an int_tuple shaped like the global tensor, got "
           << flags.getType();
  if (!intTupleIsCongruent(tensor2tdm, tupleTy.getAttr()))
    return emitError() << "\"boundary_check\" is " << tupleTy.getAttr()
                       << ", which is not congruent with the global tensor's " << tensor2tdm;

  MLIRContext *ctx = builder.getContext();
  IntTupleBuilder<IntTupleAttr> tupleBuilder(ctx);
  SmallVector<IntTupleAttr> leaves;
  intTupleFlattenToVector(tupleBuilder, tupleTy.getAttr(), leaves);
  SmallVector<int32_t> axes;
  tdm::boundaryCheckAxes(tensor2tdm, axes);

  auto tupleOp = flags.getDefiningOp<MakeIntTupleOp>();
  OperandRange dyn = tupleOp ? tupleOp.getDyncElems() : OperandRange(nullptr, 0);

  SmallVector<bool> pinnedOn(descRank, false);
  for (auto [mode, leaf] : llvm::enumerate(leaves)) {
    IntAttr value = leaf.extractIntFromLeaf();
    if (!value.isStatic() || value.getValue() == 0)
      continue;
    if (axes[mode] < 0)
      return emitError() << "\"boundary_check\" asks to clamp tensor mode " << mode
                         << ", which has no bound of its own — it is size-1 or stride-0, or it "
                            "shares a descriptor dim whose single bound would not say what a "
                            "per-mode one says";
    pinnedOn[axes[mode]] = true;
  }

  SmallVector<Value> merged(descRank);
  auto dynIt = dyn.begin();
  for (auto [mode, leaf] : llvm::enumerate(leaves)) {
    if (leaf.extractIntFromLeaf().isStatic())
      continue;
    if (!tupleOp || dynIt == dyn.end())
      return emitError() << "\"boundary_check\" leaf " << mode
                         << " is dynamic but the tuple is not normal " << "form";
    Value v = *dynIt++;
    int32_t axis = axes[mode];
    if (axis < 0)
      return emitError() << "\"boundary_check\" leaf " << mode
                         << " is dynamic but that tensor mode has no bound to switch";
    if (pinnedOn[axis])
      continue;
    Value zero = arith::ConstantIntOp::create(builder, loc, v.getType(), 0);
    Value flag = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, v, zero);
    merged[axis] = merged[axis] ? arith::OrIOp::create(builder, loc, merged[axis], flag) : flag;
  }

  // The tuple's dynamic leaves are integers, so the merged i1 goes back out as one. The
  // lowering compares it against zero again, and the pair folds.
  Type i32 = builder.getI32Type();
  SmallVector<Attribute> axisLeaves;
  SmallVector<Value> axisDyn;
  for (int32_t axis = 0; axis < descRank; ++axis) {
    if (pinnedOn[axis] || !merged[axis]) {
      axisLeaves.push_back(IntTupleAttr::get(IntAttr::getStatic(ctx, pinnedOn[axis] ? 1 : 0)));
      continue;
    }
    axisLeaves.push_back(IntTupleAttr::get(IntAttr::getDynamic(ctx, /*width=*/32)));
    axisDyn.push_back(arith::ExtUIOp::create(builder, loc, i32, merged[axis]));
  }
  IntTupleAttr result = IntTupleAttr::get(ArrayAttr::get(ctx, axisLeaves));
  return Value(MakeIntTupleOp::create(builder, loc, IntTupleType::get(result), axisDyn));
}

/// Rewrite one `set_value("boundary_check", ...)` on a CDNA5 TDM atom into
/// `set_value("boundary_check_axes", ...)`.
LogicalResult normalizeOneBoundaryCheck(AtomSetValueOp op, IRRewriter &rewriter) {
  auto [tensor2tdm, descRank] = tdmAtomShape(op.getAtom());
  if (!tensor2tdm)
    return success(); // not ours
  rewriter.setInsertionPoint(op);
  auto emitError = [&]() -> InFlightDiagnostic { return op.emitOpError(); };
  FailureOr<Value> axisFlags = normalizeBoundaryCheckToAxes(rewriter, op.getLoc(), op.getValue(),
                                                            tensor2tdm, descRank, emitError);
  if (failed(axisFlags))
    return failure();
  rewriter.replaceOpWithNewOp<AtomSetValueOp>(op, op.getResult().getType(), op.getAtom(),
                                              rewriter.getStringAttr("boundary_check_axes"),
                                              *axisFlags);
  return success();
}

class FlyROCDLExpandOpsPass
    : public mlir::fly_rocdl::impl::FlyROCDLExpandOpsPassBase<FlyROCDLExpandOpsPass> {
public:
  using mlir::fly_rocdl::impl::FlyROCDLExpandOpsPassBase<
      FlyROCDLExpandOpsPass>::FlyROCDLExpandOpsPassBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    SmallVector<Operation *> builders;
    getOperation()->walk([&](Operation *op) {
      if (isa<MakeTiledTdmLoadAtomOp, MakeTiledTdmStoreAtomOp>(op))
        builders.push_back(op);
    });
    for (Operation *op : builders) {
      LogicalResult expanded = isa<MakeTiledTdmLoadAtomOp>(op)
                                   ? expandOne(cast<MakeTiledTdmLoadAtomOp>(op), rewriter)
                                   : expandOne(cast<MakeTiledTdmStoreAtomOp>(op), rewriter);
      if (failed(expanded))
        return signalPassFailure();
    }

    // Second, in its own walk rather than inside the expansion, because the builder's own
    // initial `boundary_check` is one of these and translating it twice — once there, once here —
    // is how the two would drift apart.
    SmallVector<AtomSetValueOp> boundaryCheckSetters;
    getOperation()->walk([&](AtomSetValueOp op) {
      if (op.getField() == "boundary_check")
        boundaryCheckSetters.push_back(op);
    });
    for (AtomSetValueOp op : boundaryCheckSetters)
      if (failed(normalizeOneBoundaryCheck(op, rewriter)))
        return signalPassFailure();
  }
};

} // namespace
