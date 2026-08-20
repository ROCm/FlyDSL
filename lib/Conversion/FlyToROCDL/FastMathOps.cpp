// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

#include <utility>

namespace mlir {
#define GEN_PASS_DEF_CONVERTFASTMATHOPSPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

template <typename NeutralOp, typename ROCDLOp>
struct FastMathOpMapping {
  using NeutralOpType = NeutralOp;
  using ROCDLOpType = ROCDLOp;
};

template <typename... Mappings>
struct FastMathOpMappingTable {};

// Keep target selection separate from the lowering mechanics so follow-up
// neutral math ops can be added by extending this table.
using ROCDLFastMathOpMappings =
    FastMathOpMappingTable<FastMathOpMapping<math::Exp2Op, ROCDL::ROCDLExp2>>;

static bool isSupportedFastMathType(Type type) {
  if (type.isF32())
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && !vectorType.isScalable() && vectorType.getRank() == 1 &&
         vectorType.getElementType().isF32();
}

template <typename NeutralOp, typename ROCDLOp>
class FastMathOpLowering : public OpRewritePattern<NeutralOp> {
public:
  using OpRewritePattern<NeutralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NeutralOp op, PatternRewriter &rewriter) const override {
    if (!arith::bitEnumContainsAny(op.getFastmath(), arith::FastMathFlags::afn) ||
        !isSupportedFastMathType(op.getType()))
      return failure();

    Location loc = op.getLoc();
    Type resultType = op.getType();
    if (resultType.isF32()) {
      rewriter.replaceOpWithNewOp<ROCDLOp>(op, resultType, op.getOperand());
      return success();
    }

    auto vectorType = cast<VectorType>(resultType);
    SmallVector<Value> elements;
    elements.reserve(vectorType.getNumElements());
    for (int64_t i = 0; i < vectorType.getNumElements(); ++i) {
      Value scalar = vector::ExtractOp::create(rewriter, loc, op.getOperand(), i);
      elements.push_back(ROCDLOp::create(rewriter, loc, scalar.getType(), scalar));
    }
    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, vectorType, elements);
    return success();
  }
};

template <typename... Mappings>
void populateFastMathOpPatterns(MLIRContext *context, RewritePatternSet &patterns,
                                FastMathOpMappingTable<Mappings...>) {
  (patterns.add<FastMathOpLowering<typename Mappings::NeutralOpType,
                                   typename Mappings::ROCDLOpType>>(context),
   ...);
}

class ConvertFastMathOpsPass
    : public mlir::impl::ConvertFastMathOpsPassBase<ConvertFastMathOpsPass> {
public:
  using mlir::impl::ConvertFastMathOpsPassBase<
      ConvertFastMathOpsPass>::ConvertFastMathOpsPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateFastMathOpPatterns(context, patterns, ROCDLFastMathOpMappings{});
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
