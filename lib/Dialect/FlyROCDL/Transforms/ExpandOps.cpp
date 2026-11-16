// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Expand tiled TDM builders through TdmAtomBuilder.cpp, then translate their
// tensor-mode boundary flags to descriptor axes.
//
// This pass is also where an `boundary_check` state stops being written in the tensor's language
// and starts being written in the descriptor's. A caller sets one flag per *tensor mode*, because
// the descriptor's dims are not theirs to know — they permute, and they pack the tail. Translating
// that is `tensor2tdm` and nothing else, so it is done once, here, and the result is a flat
// `boundary_check_axes` tuple with one leaf per descriptor axis that every later pass can read
// without knowing what a mode was.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/Fly/Utils/CoreBridge.h"
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

//===----------------------------------------------------------------------===//
// "boundary_check" (tensor modes) -> "boundary_check_axes" (descriptor axes)
//===----------------------------------------------------------------------===//

/// The CDNA5 TDM atom a `fly.atom.set_value` is setting, or a null pair if it is setting
/// something else's state.
std::pair<IntTupleAttr, int32_t> tdmAtomShape(Value atom) {
  Type type = atom.getType();
  if (auto tiledCopyTy = dyn_cast<TiledCopyType>(type))
    type = tiledCopyTy.getCopyAtom();
  auto atomTy = dyn_cast<CopyAtomType>(type);
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
/// sweeps use the same bound Core leaf profile. Import collects dynamic SSA values
/// once in profile order, independently of the producer that supplied the tuple.
FailureOr<Value> normalizeBoundaryCheckToAxes(OpBuilder &builder, Location loc, Value flags,
                                              IntTupleAttr tensor2tdm, int32_t descRank,
                                              function_ref<InFlightDiagnostic()> emitError) {
  auto tupleTy = dyn_cast<IntTupleType>(flags.getType());
  if (!tupleTy)
    return emitError()
           << "\"boundary_check\" must be an int_tuple shaped like the global tensor, got "
           << flags.getType();
  if (!core::congruent(tensor2tdm.getRef(), tupleTy.getRef()))
    return emitError() << "\"boundary_check\" is " << tupleTy.getAttr()
                       << ", which is not congruent with the global tensor's " << tensor2tdm;

  MLIRContext *ctx = builder.getContext();
  auto emit = core::EmitContext(builder, loc);
  auto tuple = importValue<IntTupleType>(flags, emit);
  if (failed(tuple))
    return emitError() << "cannot bind boundary_check tuple scalars";
  auto leaves = tuple->leaves();
  SmallVector<int32_t> axes;
  tdm::boundaryCheckAxes(tensor2tdm, axes);

  SmallVector<bool> pinnedOn(descRank, false);
  for (auto [mode, leaf] : llvm::enumerate(leaves)) {
    auto value = leaf.scale();
    if (!value.isStatic() || value.isZero())
      continue;
    if (axes[mode] < 0)
      return emitError() << "\"boundary_check\" asks to clamp tensor mode " << mode
                         << ", which has no bound of its own — it is size-1 or stride-0, or it "
                            "shares a descriptor dim whose single bound would not say what a "
                            "per-mode one says";
    pinnedOn[axes[mode]] = true;
  }

  SmallVector<Value> merged(descRank);
  for (auto [mode, leaf] : llvm::enumerate(leaves)) {
    if (leaf.scale().isStatic())
      continue;
    int32_t axis = axes[mode];
    if (axis < 0)
      return emitError() << "\"boundary_check\" leaf " << mode
                         << " is dynamic but that tensor mode has no bound to switch";
    if (pinnedOn[axis])
      continue;
    auto scalar = leaf.scale().materialize(emit);
    if (failed(scalar))
      return emitError() << "cannot materialize boundary_check leaf " << mode;
    auto v = *scalar;
    Value zero = arith::ConstantIntOp::create(builder, loc, v.getType(), 0);
    Value flag = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, v, zero);
    merged[axis] = merged[axis] ? arith::OrIOp::create(builder, loc, merged[axis], flag) : flag;
  }

  // The tuple's dynamic leaves are integers, so the merged i1 goes back out as one. The
  // lowering compares it against zero again, and the pair folds.
  Type i32 = builder.getI32Type();
  SmallVector<core::IntTuple<core::Trait>> axisLeaves;
  SmallVector<Value> axisDyn;
  for (int32_t axis = 0; axis < descRank; ++axis) {
    if (pinnedOn[axis] || !merged[axis]) {
      axisLeaves.push_back(core::IntTuple<core::Trait>::getSInt(pinnedOn[axis] ? 1 : 0));
      continue;
    }
    axisLeaves.push_back(IntTupleAttr::getLeafDynamic(ctx, /*width=*/32).getRef());
    axisDyn.push_back(arith::ExtUIOp::create(builder, loc, i32, merged[axis]));
  }
  auto result = IntTupleAttr::get(ctx, core::make_tuple(axisLeaves));
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
    // A plain walk expands each builder once without deleting unrelated pure ops.
    // Descriptor derivation and materialization live together in TdmAtomBuilder.cpp.
    SmallVector<Operation *> builders;
    getOperation()->walk([&](Operation *op) {
      if (isa<MakeTiledTdmLoadAtomOp, MakeTiledTdmStoreAtomOp>(op))
        builders.push_back(op);
    });
    for (Operation *op : builders) {
      LogicalResult expanded = isa<MakeTiledTdmLoadAtomOp>(op)
                                   ? expandTdmAtom(cast<MakeTiledTdmLoadAtomOp>(op), rewriter)
                                   : expandTdmAtom(cast<MakeTiledTdmStoreAtomOp>(op), rewriter);
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
