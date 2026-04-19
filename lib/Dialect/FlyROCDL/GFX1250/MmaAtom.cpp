#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

namespace gfx1250 {

// A/B matrix register layout for GFX1250 WMMA (wave32).
//
// The A matrix is M×K (M=16, K varies by instruction). The 32 lanes split
// into two groups of 16 (group = lane/16). Both groups hold different slices
// of the K dimension.
//
// For 32-bit elements (f32, K=4):
//   Each lane holds K/2 values. Group g covers K = g*(K/2) .. (g+1)*(K/2)-1.
//   No sub-element packing. 2 VGPRs per lane.
//   Formula: K = (l/16)*2 + v
//
// For sub-32-bit elements (f16/bf16 K=32, fp8/bf8/i8 K=64/128):
//   Each lane holds K/2 values, organized in blocks of 8. Within each
//   block, group 0 holds the lower 8 K-values, group 1 holds the upper 8.
//   Formula: K = block*16 + (l/16)*8 + within_block
//   where block = flat_val / 8, within_block = flat_val % 8.
//
// Reference space is column-major (M,K) with stride (1, M=16).
// The B matrix (N×K) uses the identical layout with N substituted for M.
LayoutAttr getThrValLayoutAB(MLIRContext *ctx, int32_t K, Type elemTy) {
  auto getContext = [&]() { return ctx; };

  int elemBits = elemTy.getIntOrFloatBitWidth();
  int valsPerLane = K / 2;

  if (elemBits == 32) {
    // f32 16×4: 2 values/lane, no sub-element packing.
    // pos = (l%16)*1 + (l/16)*(valsPerLane*16) + v*16
    return FxLayout(FxShape(FxThr(16, 2), FxVal(valsPerLane)),
                    FxStride(FxThr(1, valsPerLane * 16), FxVal(16)));
  }

  // Sub-32-bit: interleaving block of 8 values between lane groups.
  // pos = (l%16)*1 + (l/16)*128 + val_within*16 [+ block*256]
  int numBlocks = valsPerLane / 8;
  if (numBlocks == 1) {
    return FxLayout(FxShape(FxThr(16, 2), FxVal(8)), FxStride(FxThr(1, 128), FxVal(16)));
  }
  return FxLayout(FxShape(FxThr(16, 2), FxVal(8, numBlocks)),
                  FxStride(FxThr(1, 128), FxVal(16, 256)));
}

// C/D matrix register layout for GFX1250 WMMA (wave32).
//
// C/D is always 16×16 (M×N). Lane l covers N = l%16. The two lane groups
// cover M=0..7 (group 0) and M=8..15 (group 1).
//
// 32-bit accumulator (f32, i32): 8 VGPRs, one element per VGPR.
//   M = (l/16)*8 + v
//
// 16-bit accumulator (f16, bf16): 4 VGPRs, two packed sub-elements each.
//   M = (l/16)*8 + v*2 + s
//
// Reference space is column-major (M,N) with stride (1, M=16).
LayoutAttr getThrValLayoutCD(MLIRContext *ctx, Type elemTyAcc) {
  auto getContext = [&]() { return ctx; };

  int elemBits = elemTyAcc.getIntOrFloatBitWidth();
  if (elemBits >= 32) {
    return FxLayout(FxShape(FxThr(16, 2), FxVal(8)), FxStride(FxThr(16, 8), FxVal(1)));
  }
  // 16-bit: 4 VGPRs × 2 sub-elements = 8 values.
  return FxLayout(FxShape(FxThr(16, 2), FxVal(4, 2)), FxStride(FxThr(16, 8), FxVal(2, 1)));
}

} // namespace gfx1250

namespace mlir::fly_rocdl {

bool MmaOpGFX1250_WMMAType::isStatic() const { return true; }

Value MmaOpGFX1250_WMMAType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Type MmaOpGFX1250_WMMAType::getValTypeA() const { return getElemTyA(); }
Type MmaOpGFX1250_WMMAType::getValTypeB() const { return getElemTyB(); }
Type MmaOpGFX1250_WMMAType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpGFX1250_WMMAType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpGFX1250_WMMAType::getThrLayout() const { return FxLayout(FxC(32), FxC(1)); }

Attribute MmaOpGFX1250_WMMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Attribute MmaOpGFX1250_WMMAType::getThrValLayoutA() const {
  return gfx1250::getThrValLayoutAB(getContext(), getK(), getElemTyA());
}

Attribute MmaOpGFX1250_WMMAType::getThrValLayoutB() const {
  return gfx1250::getThrValLayoutAB(getContext(), getK(), getElemTyB());
}

Attribute MmaOpGFX1250_WMMAType::getThrValLayoutC() const {
  return gfx1250::getThrValLayoutCD(getContext(), getElemTyAcc());
}

LogicalResult MmaOpGFX1250_WMMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                            int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                            Type elemTyAcc) {
  if (m != 16 || n != 16)
    return emitError() << "GFX1250 WMMA requires M=N=16, got " << m << "x" << n;

  auto isF8 = [](Type ty) { return isa<Float8E4M3FNUZType>(ty) || isa<Float8E5M2FNUZType>(ty); };

  bool valid = false;

  if (k == 4 && elemTyA.isF32() && elemTyB.isF32() && elemTyAcc.isF32())
    valid = true;

  if (k == 32 && elemTyA.isF16() && elemTyB.isF16() && (elemTyAcc.isF32() || elemTyAcc.isF16()))
    valid = true;
  if (k == 32 && elemTyA.isBF16() && elemTyB.isBF16() && (elemTyAcc.isF32() || elemTyAcc.isBF16()))
    valid = true;

  if ((k == 64 || k == 128) && isF8(elemTyA) && isF8(elemTyB) &&
      (elemTyAcc.isF32() || elemTyAcc.isF16()))
    valid = true;

  if (k == 64 && elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32))
    valid = true;

  if (!valid) {
    return emitError() << "unsupported GFX1250 WMMA configuration: " << m << "x" << n << "x" << k
                       << " with A=" << elemTyA << ", B=" << elemTyB << ", Acc=" << elemTyAcc;
  }
  return success();
}

static bool isFP8(Type ty) { return isa<Float8E4M3FNUZType>(ty); }
static bool isBF8(Type ty) { return isa<Float8E5M2FNUZType>(ty); }
static bool isF8(Type ty) { return isFP8(ty) || isBF8(ty); }

static Type getWmmaABType(MLIRContext *ctx, int32_t m, int32_t k, Type elemTy) {
  if (m <= 0 || k <= 0)
    return nullptr;

  Type i32Ty = IntegerType::get(ctx, 32);

  if (isF8(elemTy)) {
    if (k == 16)
      return VectorType::get({2}, i32Ty);
    if (k == 64)
      return VectorType::get({8}, i32Ty);
    if (k == 128)
      return VectorType::get({16}, i32Ty);
    return nullptr;
  }

  if (elemTy.isInteger(8)) {
    if (k == 16 || k == 32)
      return VectorType::get({4}, i32Ty);
    if (k == 64)
      return VectorType::get({8}, i32Ty);
    return nullptr;
  }

  int64_t abElemsPerLane = static_cast<int64_t>(m) * static_cast<int64_t>(k) / 32;
  if (abElemsPerLane <= 0 || (static_cast<int64_t>(m) * static_cast<int64_t>(k)) % 32 != 0)
    return nullptr;
  return VectorType::get({abElemsPerLane}, elemTy);
}

static int64_t getWmmaAccVecSize(int32_t m, int32_t k, Type elemTyA, Type elemTyB, Type elemTyAcc) {
  if (m != 16)
    return 0;

  if (k == 4 && elemTyA.isF32() && elemTyB.isF32() && elemTyAcc.isF32())
    return 8;

  if (k == 32 && elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32())
    return 8;
  if (k == 32 && elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF16())
    return 8;
  if (k == 32 && elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32())
    return 8;
  if (k == 32 && elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isBF16())
    return 8;

  if (k == 64 && isF8(elemTyA) && isF8(elemTyB) && elemTyAcc.isF32())
    return 8;
  if (k == 64 && isF8(elemTyA) && isF8(elemTyB) && elemTyAcc.isF16())
    return 8;
  if (k == 128 && isF8(elemTyA) && isF8(elemTyB) && elemTyAcc.isF32())
    return 8;
  if (k == 128 && isF8(elemTyA) && isF8(elemTyB) && elemTyAcc.isF16())
    return 8;

  if (k == 64 && elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32))
    return 8;

  return 0;
}

enum class WmmaVariant { ModsAllReuse, ModsC, ModsABClamp };

template <typename WmmaOp, WmmaVariant Variant>
static FailureOr<Value> emitWmmaSSA(OpBuilder &builder, Location loc, VectorType accTy, Value a,
                                    Value b, Value c) {
  Value res;
  if constexpr (Variant == WmmaVariant::ModsAllReuse) {
    res = WmmaOp::create(builder, loc, accTy,
                         /*signA=*/false, a, /*signB=*/false, b,
                         /*modC=*/(uint16_t)0, c)
              .getResult();
  } else if constexpr (Variant == WmmaVariant::ModsC) {
    res = WmmaOp::create(builder, loc, accTy, a, b,
                         /*modC=*/(uint16_t)0, c,
                         /*reuseA=*/false, /*reuseB=*/false)
              .getResult();
  } else {
    static_assert(Variant == WmmaVariant::ModsABClamp);
    res = WmmaOp::create(builder, loc, accTy,
                         /*signA=*/false, a, /*signB=*/false, b, c,
                         /*reuseA=*/false, /*reuseB=*/false, /*clamp=*/false)
              .getResult();
  }
  return res;
}

FailureOr<Value> MmaOpGFX1250_WMMAType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                        Type resultTy, Type mmaAtomTyArg,
                                                        Type dTyArg, Type aTyArg, Type bTyArg,
                                                        Type cTyArg, Value atomVal, Value d,
                                                        Value a, Value b, Value c) const {
  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getWmmaABType(ctx, m, k, elemTyA);
  Type abTyB = getWmmaABType(ctx, m, k, elemTyB);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getWmmaAccVecSize(m, k, elemTyA, elemTyB, elemTyAcc);
  if (accVecSize == 0)
    return failure();

  VectorType accTy = VectorType::get({accVecSize}, elemTyAcc);

#define DISPATCH_WMMA_SSA(M_, K_, PRED, OP, VARIANT)                                               \
  if (m == M_ && n == M_ && k == K_ && (PRED)) {                                                   \
    return emitWmmaSSA<ROCDL::OP, WmmaVariant::VARIANT>(builder, loc, accTy, a, b, c);             \
  }

#define DISPATCH_WMMA_SSA_FP8(K_, ACC_PRED, ACC_PREFIX)                                            \
  DISPATCH_WMMA_SSA(16, K_, isFP8(elemTyA) && isFP8(elemTyB) && ACC_PRED,                          \
                    wmma_##ACC_PREFIX##_16x16x##K_##_fp8_fp8, ModsC)                               \
  DISPATCH_WMMA_SSA(16, K_, isFP8(elemTyA) && isBF8(elemTyB) && ACC_PRED,                          \
                    wmma_##ACC_PREFIX##_16x16x##K_##_fp8_bf8, ModsC)                               \
  DISPATCH_WMMA_SSA(16, K_, isBF8(elemTyA) && isFP8(elemTyB) && ACC_PRED,                          \
                    wmma_##ACC_PREFIX##_16x16x##K_##_bf8_fp8, ModsC)                               \
  DISPATCH_WMMA_SSA(16, K_, isBF8(elemTyA) && isBF8(elemTyB) && ACC_PRED,                          \
                    wmma_##ACC_PREFIX##_16x16x##K_##_bf8_bf8, ModsC)

  DISPATCH_WMMA_SSA(16, 4, elemTyA.isF32() && elemTyB.isF32() && elemTyAcc.isF32(),
                    wmma_f32_16x16x4_f32, ModsAllReuse)

  DISPATCH_WMMA_SSA(16, 32, elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32(),
                    wmma_f32_16x16x32_f16, ModsAllReuse)
  DISPATCH_WMMA_SSA(16, 32, elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32(),
                    wmma_f32_16x16x32_bf16, ModsAllReuse)
  DISPATCH_WMMA_SSA(16, 32, elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF16(),
                    wmma_f16_16x16x32_f16, ModsAllReuse)
  DISPATCH_WMMA_SSA(16, 32, elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isBF16(),
                    wmma_bf16_16x16x32_bf16, ModsAllReuse)

  DISPATCH_WMMA_SSA_FP8(64, elemTyAcc.isF32(), f32)
  DISPATCH_WMMA_SSA_FP8(64, elemTyAcc.isF16(), f16)
  DISPATCH_WMMA_SSA_FP8(128, elemTyAcc.isF32(), f32)
  DISPATCH_WMMA_SSA_FP8(128, elemTyAcc.isF16(), f16)

  DISPATCH_WMMA_SSA(16, 64, elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32),
                    wmma_i32_16x16x64_iu8, ModsABClamp)

#undef DISPATCH_WMMA_SSA_FP8
#undef DISPATCH_WMMA_SSA

  return failure();
}

LogicalResult MmaOpGFX1250_WMMAType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                  Type dMemTy, Type aMemTy, Type bMemTy,
                                                  Type cMemTy, Value atomVal, Value dPtr,
                                                  Value aPtr, Value bPtr, Value cPtr) const {
  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getWmmaABType(ctx, m, k, elemTyA);
  Type abTyB = getWmmaABType(ctx, n, k, elemTyB);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getWmmaAccVecSize(m, k, elemTyA, elemTyB, elemTyAcc);
  if (accVecSize == 0)
    return failure();

  VectorType accTy = VectorType::get({accVecSize}, elemTyAcc);

  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, Type{}, mmaAtomTy, accTy, abTyA, abTyB, accTy, atomVal,
                             Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

} // namespace mlir::fly_rocdl
