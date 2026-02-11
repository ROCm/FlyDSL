
#include "mlir/IR/PatternMatch.h"
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

template <typename IntTupleLikeOp>
struct RewriteToMakeIntTuple final : public OpRewritePattern<IntTupleLikeOp> {
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
    if (isa<MakeIntTupleOp>(op))
      return failure();
    if (auto makeLayoutOp = dyn_cast<MakeLayoutOp>(op)) {
      if (makeLayoutOp.getShape().getDefiningOp<MakeIntTupleOp>() &&
          makeLayoutOp.getStride().getDefiningOp<MakeIntTupleOp>()) {
        return failure();
      }
    }

    // Must have exactly one result
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
    }

    return failure();
  }
};

class FlyCanonicalizePass : public mlir::fly::impl::FlyCanonicalizePassBase<FlyCanonicalizePass> {
public:
  using mlir::fly::impl::FlyCanonicalizePassBase<FlyCanonicalizePass>::FlyCanonicalizePassBase;

  void runOnOperation() override {
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
