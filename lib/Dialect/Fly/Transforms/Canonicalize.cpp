// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

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

class RebuildStaticResult : public RewritePattern {
public:
  RebuildStaticResult(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (isa<StaticOp>(op))
      return failure();
    bool changed = false;
    for (Value result : op->getResults()) {
      auto mayStatic = dyn_cast<MayStaticTypeInterface>(result.getType());
      // result.use_empty() keeps any side effects it has.
      if (!mayStatic || !mayStatic.isStatic() || result.use_empty())
        continue;
      rewriter.setInsertionPoint(op);
      Value rebuilt = StaticOp::create(rewriter, op->getLoc(), result.getType());
      rewriter.replaceAllUsesWith(result, rebuilt);
      changed = true;
    }
    return success(changed);
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
    patterns.add<RebuildStaticResult>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
