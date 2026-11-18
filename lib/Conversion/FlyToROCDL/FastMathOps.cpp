// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

#include <utility>

namespace mlir {
#define GEN_PASS_DEF_CONVERTROCDLFASTMATHOPSPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// The element types a mapping rewrites, as scalars or 1-D vectors. Only types where the native
// instruction is not already what LLVM selects: f16 exp2 / log2 already lower to a single
// v_exp_f16 / v_log_f16, and f64 keeps the accurate lowering.
enum class NativeTypes { F32, F16F32 };

static bool hasNativeType(Type type, NativeTypes types) {
  Type elemType = type;
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    if (vectorType.isScalable() || vectorType.getRank() != 1)
      return false;
    elemType = vectorType.getElementType();
  }
  return elemType.isF32() || (types == NativeTypes::F16F32 && elemType.isF16());
}

static bool allowsApproximation(arith::FastMathFlags flags) {
  return arith::bitEnumContainsAny(flags, arith::FastMathFlags::afn);
}

// `fn` applied to `operand`, element by element for a vector.
static Value mapElements(PatternRewriter &rewriter, Location loc, Value operand,
                         function_ref<Value(Value)> fn) {
  auto vectorType = dyn_cast<VectorType>(operand.getType());
  if (!vectorType)
    return fn(operand);
  SmallVector<Value> elements;
  elements.reserve(vectorType.getNumElements());
  for (int64_t i = 0; i < vectorType.getNumElements(); ++i)
    elements.push_back(fn(vector::ExtractOp::create(rewriter, loc, operand, i)));
  return vector::FromElementsOp::create(rewriter, loc, vectorType, elements);
}

// A math function computed by one native instruction, its operand and result scaled by constants:
// f(x) = outScale * native(inScale * x). The native instructions are the raw hardware
// approximations: no range reduction, no denormal handling.
template <typename NeutralOp, typename ROCDLOp>
class NativeMathLowering : public OpRewritePattern<NeutralOp> {
public:
  NativeMathLowering(MLIRContext *context, NativeTypes types, double inScale = 1.0,
                     double outScale = 1.0)
      : OpRewritePattern<NeutralOp>(context), types(types), inScale(inScale), outScale(outScale) {}

  LogicalResult matchAndRewrite(NeutralOp op, PatternRewriter &rewriter) const override {
    Type type = op.getType();
    arith::FastMathFlags flags = op.getFastmath();
    if (!allowsApproximation(flags) || !hasNativeType(type, types))
      return failure();

    Location loc = op.getLoc();
    Type elemType = getElementTypeOrSelf(type);
    auto constant = [&](double value) -> Value {
      if (value == 1.0)
        return nullptr;
      return arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elemType, value));
    };
    Value inFactor = constant(inScale), outFactor = constant(outScale);
    auto scale = [&](Value x, Value factor) -> Value {
      return factor ? arith::MulFOp::create(rewriter, loc, x, factor, flags) : x;
    };
    rewriter.replaceOp(op, mapElements(rewriter, loc, op.getOperand(), [&](Value x) {
                         Value native =
                             ROCDLOp::create(rewriter, loc, elemType, scale(x, inFactor));
                         return scale(native, outFactor);
                       }));
    return success();
  }

private:
  NativeTypes types;
  double inScale, outScale;
};

// Which neutral ops lower to which native instruction, kept apart from the lowering mechanics.
static void populateNativeMathPatterns(MLIRContext *context, RewritePatternSet &patterns) {
  using llvm::numbers::inv_pi, llvm::numbers::ln10, llvm::numbers::ln2, llvm::numbers::log2e;
  constexpr NativeTypes kF32 = NativeTypes::F32, kF16F32 = NativeTypes::F16F32;
  constexpr double kRevolutionsPerRadian = 0.5 * inv_pi; // v_sin / v_cos take revolutions
  patterns.add<NativeMathLowering<math::Exp2Op, ROCDL::ROCDLExp2>>(context, kF32);
  patterns.add<NativeMathLowering<math::ExpOp, ROCDL::ROCDLExp2>>(context, kF16F32, log2e);
  patterns.add<NativeMathLowering<math::Log2Op, ROCDL::ROCDLLog>>(context, kF32);
  patterns.add<NativeMathLowering<math::LogOp, ROCDL::ROCDLLog>>(context, kF16F32, 1.0, ln2);
  patterns.add<NativeMathLowering<math::Log10Op, ROCDL::ROCDLLog>>(context, kF16F32, 1.0,
                                                                   ln2 / ln10);
  patterns.add<NativeMathLowering<math::SinOp, ROCDL::ROCDLSin>>(context, kF16F32,
                                                                 kRevolutionsPerRadian);
  patterns.add<NativeMathLowering<math::CosOp, ROCDL::ROCDLCos>>(context, kF16F32,
                                                                 kRevolutionsPerRadian);
}

class ConvertROCDLFastMathOpsPass
    : public mlir::impl::ConvertROCDLFastMathOpsPassBase<ConvertROCDLFastMathOpsPass> {
public:
  using mlir::impl::ConvertROCDLFastMathOpsPassBase<
      ConvertROCDLFastMathOpsPass>::ConvertROCDLFastMathOpsPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateNativeMathPatterns(context, patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
