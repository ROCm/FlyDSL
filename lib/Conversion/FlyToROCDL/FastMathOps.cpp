// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToROCDL/MathToROCDL.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

#include <utility>

namespace mlir {
#define GEN_PASS_DEF_CONVERTROCDLMATHOPSPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

template <typename NeutralOp, typename ROCDLOp> struct FastMathOpMapping {
  using NeutralOpType = NeutralOp;
  using ROCDLOpType = ROCDLOp;
};

template <typename... Mappings> struct FastMathOpMappingTable {};

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

static bool allowsApproximation(arith::FastMathFlags flags) {
  return arith::bitEnumContainsAny(flags, arith::FastMathFlags::afn);
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
  (patterns
       .add<FastMathOpLowering<typename Mappings::NeutralOpType, typename Mappings::ROCDLOpType>>(
           context),
   ...);
}

// f32 log1p with afn as log(w) + c, with w = 1 + x and c = x - (w - 1) while w < 2. c is the
// rounding error of w (exact for |x| <= 1) and stands in for c / w; at tiny x, w == 1 and the
// result is x, where log(1 + x) gives 0. For w >= 2, c / w is below an ulp of log(w) and c is
// dropped, which also keeps x = +inf from giving inf - inf. w is 0 or at least 2^-24, never
// denormal, so the raw log loses nothing. The arithmetic carries no fast-math flags:
// reassociating (1 + x) - 1 to x would cancel the correction. Faster and more accurate than the
// log(1 + x) that convert-gpu-to-rocdl emits otherwise.
class Log1pLowering : public OpRewritePattern<math::Log1pOp> {
public:
  using OpRewritePattern<math::Log1pOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Log1pOp op, PatternRewriter &rewriter) const override {
    Type type = op.getType();
    if (!allowsApproximation(op.getFastmath()) || !isSupportedFastMathType(type))
      return failure();

    Location loc = op.getLoc();
    auto constant = [&](float value) -> Value {
      return createScalarOrSplatConstant(rewriter, loc, type, APFloat(value));
    };
    Value x = op.getOperand(), one = constant(1.0f);
    Value w = arith::AddFOp::create(rewriter, loc, x, one);
    Value c = arith::SubFOp::create(rewriter, loc, x, arith::SubFOp::create(rewriter, loc, w, one));
    Value belowTwo =
        arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OLT, w, constant(2.0f));
    c = arith::SelectOp::create(rewriter, loc, belowTwo, c, constant(0.0f));

    Value log2W;
    if (auto vectorType = dyn_cast<VectorType>(type)) {
      SmallVector<Value> lanes;
      lanes.reserve(vectorType.getNumElements());
      for (int64_t i = 0; i < vectorType.getNumElements(); ++i) {
        Value lane = vector::ExtractOp::create(rewriter, loc, w, i);
        lanes.push_back(ROCDL::ROCDLLog::create(rewriter, loc, lane.getType(), lane));
      }
      log2W = vector::FromElementsOp::create(rewriter, loc, vectorType, lanes);
    } else {
      log2W = ROCDL::ROCDLLog::create(rewriter, loc, type, w);
    }
    Value logW = arith::MulFOp::create(rewriter, loc, log2W,
                                       constant(static_cast<float>(llvm::numbers::ln2)));
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, logW, c);
    return success();
  }
};

// convert-gpu-to-rocdl can lower these ops both with MathToLLVM and with OCML calls, and which one
// it picks depends on the dialects loaded in the context: DialectConversion tries same-benefit
// patterns in another order once any root-less pattern is registered. The MathToLLVM forms are
// wrong on AMDGPU: the LLVM intrinsics of the first eight have no AMDGPU lowering, so the kernel
// body is dropped; log1p and expm1 become log(1 + x) and exp(x) - 1, which give 0 for tiny x; rsqrt
// becomes 1 / sqrt(x), less accurate and slower than OCML; fpowi with a runtime exponent becomes
// exp2(n * log2(x)), NaN for negative x, and vector fpowi does not lower at all. Lower them to
// OCML here, as HIP does, except the log1p already rewritten above, rsqrt with afn, and scalar
// fpowi with a constant exponent, which keep their lowering: the backend expands the latter into
// multiplications. expm1 goes to OCML even with afn, like HIP's expm1f under -ffast-math:
// exp(x) - 1 is no approximation of it at small x, and code that wants it can write it.
static LogicalResult lowerToOCML(Operation *op) {
  MLIRContext *context = op->getContext();
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  populateMathToROCDLConversionPatterns(converter, patterns, /*chipset=*/std::nullopt);

  ConversionTarget target(*context);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<math::TanOp, math::AsinOp, math::AcosOp, math::AtanOp, math::Atan2Op,
                      math::SinhOp, math::CoshOp, math::TanhOp, math::ExpM1Op, math::Log1pOp>();
  target.addDynamicallyLegalOp<math::RsqrtOp>(
      [](math::RsqrtOp rsqrt) { return allowsApproximation(rsqrt.getFastmath()); });
  target.addDynamicallyLegalOp<math::FPowIOp>([](math::FPowIOp fpowi) {
    return !isa<VectorType>(fpowi.getType()) && matchPattern(fpowi.getRhs(), m_Constant());
  });
  return applyPartialConversion(op, target, std::move(patterns));
}

class ConvertROCDLMathOpsPass
    : public mlir::impl::ConvertROCDLMathOpsPassBase<ConvertROCDLMathOpsPass> {
public:
  using mlir::impl::ConvertROCDLMathOpsPassBase<
      ConvertROCDLMathOpsPass>::ConvertROCDLMathOpsPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateFastMathOpPatterns(context, patterns, ROCDLFastMathOpMappings{});
    patterns.add<Log1pLowering>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        failed(lowerToOCML(getOperation())) ||
        failed(fly_rocdl::lowerFloatConversions(getOperation())))
      signalPassFailure();
  }
};

} // namespace
