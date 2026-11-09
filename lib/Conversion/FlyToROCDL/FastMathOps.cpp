// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

#include <string>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_CONVERTROCDLFASTMATHOPSPASS
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

// `convert-gpu-to-rocdl` lowers `math.log1p` to `log(1 + x)` and `math.expm1` to `exp(x) - 1`: its
// MathToLLVM patterns are added before, and win over, the OCML ones. Both return 0 for |x| below
// half an ulp of 1 (6e-8 in f32), so they are expanded here instead, whatever their fast-math
// flags.

static Value floatConstant(OpBuilder &builder, Location loc, Type type, double value) {
  APFloat apValue(value);
  bool losesInfo;
  apValue.convert(cast<FloatType>(getElementTypeOrSelf(type)).getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &losesInfo);
  return createScalarOrSplatConstant(builder, loc, type, apValue);
}

/// `scalarFn(x)` for a scalar `x`; lane by lane for a vector `x`, reassembled into its shape.
static Value perLane(OpBuilder &builder, Location loc, Value x,
                     function_ref<Value(OpBuilder &, Location, Value)> scalarFn) {
  auto vectorType = dyn_cast<VectorType>(x.getType());
  if (!vectorType)
    return scalarFn(builder, loc, x);
  auto flatType = VectorType::get({vectorType.getNumElements()}, vectorType.getElementType());
  Value flat =
      vectorType.getRank() == 1 ? x : vector::ShapeCastOp::create(builder, loc, flatType, x);
  SmallVector<Value> lanes;
  lanes.reserve(flatType.getNumElements());
  for (int64_t i = 0; i < flatType.getNumElements(); ++i)
    lanes.push_back(scalarFn(builder, loc, vector::ExtractOp::create(builder, loc, flat, i)));
  Value result = vector::FromElementsOp::create(builder, loc, flatType, lanes);
  return vectorType.getRank() == 1 ? result
                                   : vector::ShapeCastOp::create(builder, loc, vectorType, result);
}

/// `expand(x)` for f16, f32 and f64; other float types are computed in f32.
static Value inNativeFloat(OpBuilder &builder, Location loc, Value x,
                           function_ref<Value(OpBuilder &, Location, Value)> expand) {
  Type type = x.getType();
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isF16() || elementType.isF32() || elementType.isF64())
    return expand(builder, loc, x);
  Type f32Type = builder.getF32Type();
  if (auto vectorType = dyn_cast<VectorType>(type))
    f32Type = vectorType.clone(f32Type);
  Value wide = arith::ExtFOp::create(builder, loc, f32Type, x);
  return arith::TruncFOp::create(builder, loc, type, expand(builder, loc, wide));
}

static bool isExpandableFloatType(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  return isa<FloatType>(getElementTypeOrSelf(type)) && !(vectorType && vectorType.isScalable());
}

/// `log(x)` for a scalar `x` that is 0 or not denormal: the hardware `log2` (`v_log_f32` /
/// `v_log_f16`, which flushes denormal inputs) times ln 2; `math.log` for f64.
static Value nativeLog(OpBuilder &builder, Location loc, Value x) {
  Type type = x.getType();
  if (type.isF64())
    return math::LogOp::create(builder, loc, x);
  Value log2 = ROCDL::ROCDLLog::create(builder, loc, type, x);
  return arith::MulFOp::create(builder, loc, log2,
                               floatConstant(builder, loc, type, llvm::numbers::ln2));
}

/// `log(1 + x)` as `log(w) + c`, with `w = 1 + x` and `c = x - (w - 1)`.
///
/// For |x| <= 1, `c` is the exact rounding error of `w`, so `log(1 + x) = log(w) + c / w` to first
/// order. For `w < 2`, `c` stands in for `c / w` at a cost of about an ulp: |c| is at most half an
/// ulp of 1, and |c - c / w| is at most |c| |log w| for w >= 1 and 1.5 |c| |log w| for 0.5 <= w < 1
/// (below 0.5, `w` is exact and `c == 0`). For `w >= 2`, |c / w| is below an ulp of `log w` and `c`
/// is dropped. `w == 1` returns `x` exactly, except that -0 gives +0; +inf, -1, x < -1 and NaN give
/// +inf, -inf, NaN and NaN. `w` is 0 or at least 2^-24, never denormal, so `log w` is one
/// transcendental instruction; the result is within about 4 ulps in f32. The arith ops carry no
/// fast-math flags: reassociating `(w - 1)` would undo the correction.
static Value expandLog1p(OpBuilder &builder, Location loc, Value x) {
  Type type = x.getType();
  Value one = floatConstant(builder, loc, type, 1.0);
  Value w = arith::AddFOp::create(builder, loc, one, x);
  Value c = arith::SubFOp::create(builder, loc, x, arith::SubFOp::create(builder, loc, w, one));
  Value belowTwo = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT, w,
                                         floatConstant(builder, loc, type, 2.0));
  c = arith::SelectOp::create(builder, loc, belowTwo, c, floatConstant(builder, loc, type, 0.0));
  return arith::AddFOp::create(builder, loc, perLane(builder, loc, w, nativeLog), c);
}

/// `__ocml_expm1_<f16|f32|f64>(x)` for a scalar `x`, declared in the enclosing `gpu.module` on
/// first use.
static Value ocmlExpm1(OpBuilder &builder, Location loc, Value x) {
  Type type = x.getType();
  std::string name = "__ocml_expm1_f" + std::to_string(type.getIntOrFloatBitWidth());
  Operation *module =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<gpu::GPUModuleOp>();
  auto callee = SymbolTable::lookupSymbolIn(module, name);
  auto func = dyn_cast_or_null<LLVM::LLVMFuncOp>(callee);
  if (!func) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&module->getRegion(0).front());
    func = LLVM::LLVMFuncOp::create(builder, UnknownLoc::get(builder.getContext()), name,
                                    LLVM::LLVMFunctionType::get(type, {type}));
  }
  return LLVM::CallOp::create(builder, loc, func, ValueRange{x}).getResult();
}

struct Log1pOpLowering : public OpRewritePattern<math::Log1pOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Log1pOp op, PatternRewriter &rewriter) const override {
    if (!isExpandableFloatType(op.getType()))
      return failure();
    rewriter.replaceOp(op, inNativeFloat(rewriter, op.getLoc(), op.getOperand(), expandLog1p));
    return success();
  }
};

struct ExpM1OpLowering : public OpRewritePattern<math::ExpM1Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpM1Op op, PatternRewriter &rewriter) const override {
    if (!isExpandableFloatType(op.getType()))
      return failure();
    rewriter.replaceOp(op, inNativeFloat(rewriter, op.getLoc(), op.getOperand(),
                                         [](OpBuilder &builder, Location loc, Value x) {
                                           return perLane(builder, loc, x, ocmlExpm1);
                                         }));
    return success();
  }
};

class ConvertROCDLFastMathOpsPass
    : public mlir::impl::ConvertROCDLFastMathOpsPassBase<ConvertROCDLFastMathOpsPass> {
public:
  using mlir::impl::ConvertROCDLFastMathOpsPassBase<
      ConvertROCDLFastMathOpsPass>::ConvertROCDLFastMathOpsPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateFastMathOpPatterns(context, patterns, ROCDLFastMathOpMappings{});
    patterns.add<Log1pOpLowering, ExpM1OpLowering>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
