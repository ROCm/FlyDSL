
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYCANONICALIZEPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

bool isStaticArg(Type ty) {
  if (auto mayStatic = dyn_cast<MayStaticTypeInterface>(ty))
    return mayStatic.isStatic();
  return false;
}

void removeStaticArgsFromFunc(FunctionOpInterface funcOp) {
  ArrayRef<Type> argTypes = funcOp.getArgumentTypes();
  if (argTypes.empty())
    return;

  bool hasBody = !funcOp.getFunctionBody().empty();
  Block *entry = hasBody ? &funcOp.getFunctionBody().front() : nullptr;
  OpBuilder builder(funcOp.getContext());
  if (hasBody)
    builder.setInsertionPointToStart(entry);
  Location loc = funcOp.getLoc();

  SmallVector<unsigned> staticArgIndices;
  SmallVector<Type> newArgTypes;
  for (unsigned i = 0; i < argTypes.size(); ++i) {
    if (isStaticArg(argTypes[i])) {
      staticArgIndices.push_back(i);
      if (hasBody) {
        BlockArgument arg = entry->getArgument(i);
        Value staticVal = StaticOp::create(builder, loc, arg.getType());
        arg.replaceAllUsesWith(staticVal);
      }
    } else {
      newArgTypes.push_back(argTypes[i]);
    }
  }
  if (staticArgIndices.empty())
    return;

  funcOp.setType(FunctionType::get(funcOp.getContext(), newArgTypes, funcOp.getResultTypes()));

  if (hasBody) {
    for (int i = staticArgIndices.size() - 1; i >= 0; --i)
      entry->eraseArgument(staticArgIndices[i]);
  }
}

void removeStaticOperandsFromLaunchFunc(gpu::LaunchFuncOp launchOp) {
  SmallVector<Value> oldOperands(launchOp.getKernelOperands().begin(),
                                 launchOp.getKernelOperands().end());
  SmallVector<Value> newOperands;
  bool changed = false;

  for (auto operand : oldOperands) {
    if (isStaticArg(operand.getType())) {
      changed = true;
      continue;
    }
    newOperands.push_back(operand);
  }

  if (changed)
    launchOp.getKernelOperandsMutable().assign(newOperands);
}

template <typename IntTupleLikeOp>
class RewriteToMakeIntTuple final : public OpRewritePattern<IntTupleLikeOp> {
  using OpRewritePattern<IntTupleLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IntTupleLikeOp op, PatternRewriter &rewriter) const override {
    auto newOp = MakeIntTupleOp::create(rewriter, op.getLoc(), op.getResult().getType(),
                                        op->getOperands(), op->getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

class StaticResultLowering : public RewritePattern {
public:
  StaticResultLowering(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // Skip ops that are already in normal form
    if (isa<MakeIntTupleOp, MakeAtomOp>(op))
      return failure();
    if (auto makeLayoutOp = dyn_cast<MakeLayoutOp>(op)) {
      if (makeLayoutOp.getShape().getDefiningOp<MakeIntTupleOp>() &&
          makeLayoutOp.getStride().getDefiningOp<MakeIntTupleOp>()) {
        return failure();
      }
    }

    if (op->getNumResults() != 1)
      return failure();
    Type resultType = op->getResult(0).getType();
    Location loc = op->getLoc();

    if (auto intTupleTy = dyn_cast<IntTupleType>(resultType)) {
      IntTupleAttr intTupleAttr = intTupleTy.getAttr();
      if (!intTupleAttr.isStatic())
        return failure();
      rewriter.replaceOpWithNewOp<MakeIntTupleOp>(op, intTupleTy, ValueRange{});
      return success();
    } else if (auto layoutTy = dyn_cast<LayoutType>(resultType)) {
      LayoutAttr layoutAttr = layoutTy.getAttr();
      if (!layoutAttr.isStatic())
        return failure();

      Value shape =
          MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(layoutAttr.getShape()), {});
      Value stride =
          MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(layoutAttr.getStride()), {});
      rewriter.replaceOpWithNewOp<MakeLayoutOp>(op, layoutTy, shape, stride);
      return success();
    } else if (isa<MmaAtomTypeInterface>(resultType)) {
      auto mayStatic = cast<MayStaticTypeInterface>(resultType);
      if (!mayStatic.isStatic())
        return failure();
      rewriter.replaceOpWithNewOp<MakeAtomOp>(op, resultType);
      return success();
    }

    return failure();
  }
};

class FlyCanonicalizePass : public mlir::fly::impl::FlyCanonicalizePassBase<FlyCanonicalizePass> {
public:
  using mlir::fly::impl::FlyCanonicalizePassBase<FlyCanonicalizePass>::FlyCanonicalizePassBase;

  void runOnOperation() override {
    getOperation()->walk(
        [&](gpu::LaunchFuncOp launchOp) { removeStaticOperandsFromLaunchFunc(launchOp); });
    getOperation()->walk([&](FunctionOpInterface funcOp) { removeStaticArgsFromFunc(funcOp); });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<RewriteToMakeIntTuple<MakeShapeOp>, RewriteToMakeIntTuple<MakeStrideOp>,
                 RewriteToMakeIntTuple<MakeCoordOp>>(context);
    patterns.add<StaticResultLowering>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace impl {

std::unique_ptr<::mlir::Pass> createFlyCanonicalizePass() {
  return std::make_unique<FlyCanonicalizePass>();
}

} // namespace impl
