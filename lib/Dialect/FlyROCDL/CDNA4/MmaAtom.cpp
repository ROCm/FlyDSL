// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// MmaOpCDNA4_MFMAType — wave64 MFMA atoms for gfx950 (CDNA4).
// Superset of CDNA3: includes K=32 f16/bf16, K=128 fp8 mfma_scale,
// plus all CDNA3 (gfx942) MFMA instructions.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "../MfmaLayoutCommon.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

bool MmaOpCDNA4_MFMAType::isStatic() const { return true; }

Value MmaOpCDNA4_MFMAType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                              Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Attribute MmaOpCDNA4_MFMAType::getThrLayout() const { return FxLayout(FxC(64), FxC(1)); }

Attribute MmaOpCDNA4_MFMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpCDNA4_MFMAType::getValTypeA() const { return getElemTyA(); }
Type MmaOpCDNA4_MFMAType::getValTypeB() const { return getElemTyB(); }
Type MmaOpCDNA4_MFMAType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpCDNA4_MFMAType::getValTypeD() const { return getElemTyAcc(); }

// Thread-value layouts are identical to CDNA3 (both wave64 MFMA).
Attribute MmaOpCDNA4_MFMAType::getThrValLayoutA() const {
  return cdna3::getThrValLayoutAB(getContext(), getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                                  getElemTyAcc());
}
Attribute MmaOpCDNA4_MFMAType::getThrValLayoutB() const {
  return cdna3::getThrValLayoutAB(getContext(), getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                                  getElemTyAcc());
}
Attribute MmaOpCDNA4_MFMAType::getThrValLayoutC() const {
  int M = getM();
  int N = getN();
  auto getContext = [&]() { return this->getContext(); };

  int GroupM = 64 / N;
  int ValM0 = 4;
  int ValM1 = M / 4 / GroupM;

  return FxLayout(FxShape(FxThr(N, GroupM), FxVal(ValM0, ValM1)),
                  FxStride(FxThr(M, ValM0), FxVal(1, ValM0 * GroupM)));
}

LogicalResult MmaOpCDNA4_MFMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                          int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                          Type elemTyAcc) {
  if (m != n)
    return emitError() << "invalid MNK dimensions for CDNA4 MFMA: " << m << "x" << n << "x" << k;
  if (!elemTyAcc.isF32())
    return emitError() << "elemTyAcc must be f32, got " << elemTyAcc;

  auto isValidElemType = [](Type ty) {
    if (ty.isF16() || ty.isBF16() || ty.isF32())
      return true;
    if (isa<Float8E4M3FNUZType>(ty) || isa<Float8E5M2FNUZType>(ty) || isa<Float8E4M3FNType>(ty))
      return true;
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return intTy.getWidth() == 8;
    return false;
  };
  if (!isValidElemType(elemTyA))
    return emitError() << "elemTyA must be f16, bf16, f32, f8E4M3FNUZ, f8E5M2FNUZ, i8, got "
                       << elemTyA;
  if (!isValidElemType(elemTyB))
    return emitError() << "elemTyB must be f16, bf16, f32, f8E4M3FNUZ, f8E5M2FNUZ, i8, got "
                       << elemTyB;
  return success();
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static bool isI8(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return intTy.getWidth() == 8;
  return false;
}
static bool isFP8(Type ty) {
  return isa<Float8E4M3FNUZType>(ty) || isa<Float8E4M3FNType>(ty) || isI8(ty);
}
static bool isBF8(Type ty) { return isa<Float8E5M2FNUZType>(ty); }
static bool isF8(Type ty) { return isFP8(ty) || isBF8(ty); }

static Type getMfmaABType(MLIRContext *ctx, Type elemTy, int32_t k) {
  if (elemTy.isF32())
    return Float32Type::get(ctx);
  if (elemTy.isF16()) {
    if (k >= 32)
      return VectorType::get({8}, Float16Type::get(ctx));
    return VectorType::get({4}, Float16Type::get(ctx));
  }
  if (elemTy.isBF16()) {
    if (k >= 32)
      return VectorType::get({8}, BFloat16Type::get(ctx));
    return VectorType::get({(k >= 16) ? 4 : 2}, IntegerType::get(ctx, 16));
  }
  if (elemTy.getIntOrFloatBitWidth() == 8) {
    if (k >= 128)
      return VectorType::get({8}, IntegerType::get(ctx, 32));
    return IntegerType::get(ctx, 64);
  }
  return nullptr;
}

static int64_t getMfmaAccVecSize(int32_t m, int32_t k, Type elemTyA) {
  if (elemTyA.isF32()) {
    if (m == 32 && k == 1) return 32;
    if (m == 32 && k == 2) return 16;
    if (m == 16 && k == 1) return 16;
    if (m == 16 && k == 4) return 4;
    if (m == 4 && k == 1) return 4;
  }
  if (elemTyA.isF16()) {
    if (m == 32 && k == 4) return 32;
    if (m == 32 && k == 8) return 16;
    if (m == 16 && k == 4) return 16;
    if (m == 16 && k == 16) return 4;
    if (m == 16 && k == 32) return 4;
    if (m == 4 && k == 4) return 4;
  }
  if (elemTyA.isBF16()) {
    if (m == 32 && k == 2) return 32;
    if (m == 32 && k == 4) return 16;
    if (m == 16 && k == 2) return 16;
    if (m == 16 && k == 8) return 4;
    if (m == 16 && k == 16) return 4;
    if (m == 16 && k == 32) return 4;
    if (m == 4 && k == 2) return 4;
  }
  if (isF8(elemTyA)) {
    if (m == 16 && k == 128) return 4;
    if (m == 16 && k == 32) return 4;
    if (m == 32 && k == 16) return 16;
  }
  return 0;
}

// ── Atom emission ────────────────────────────────────────────────────────────

FailureOr<Value> MmaOpCDNA4_MFMAType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                      Type resultTy, Type mmaAtomTyArg, Type dTyArg,
                                                      Type aTyArg, Type bTyArg, Type cTyArg,
                                                      Value atomVal, Value d, Value a, Value b,
                                                      Value c) const {
  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getMfmaABType(ctx, elemTyA, k);
  Type abTyB = getMfmaABType(ctx, elemTyB, k);
  if (!abTyA || !abTyB)
    return failure();

  // Bitcast SSA operands when type doesn't match MFMA ABI
  auto bitcastToABI = [&](Value v, Type targetTy) -> Value {
    if (v.getType() == targetTy)
      return v;
    if (!isa<VectorType>(targetTy))
      return LLVM::BitcastOp::create(builder, loc, targetTy, v);
    return vector::BitCastOp::create(builder, loc, targetTy, v);
  };
  a = bitcastToABI(a, abTyA);
  b = bitcastToABI(b, abTyB);

  int64_t accVecSize = getMfmaAccVecSize(m, k, elemTyA);
  if (accVecSize == 0)
    return failure();

  Type accElemTy = getElemTyAcc();
  VectorType accTy = VectorType::get({accVecSize}, accElemTy);

#define DISPATCH_MFMA_SSA(M_, K_, PRED, OP)                                                        \
  if (m == M_ && n == M_ && k == K_ && (PRED)) {                                                   \
    auto zeroAttr = builder.getI32IntegerAttr(0);                                                  \
    return ROCDL::OP::create(builder, loc, accTy, a, b, c, zeroAttr, zeroAttr, zeroAttr)           \
        .getResult();                                                                              \
  }

  // ── CDNA4-only instructions ────────────────────────────────────────────────

  // K=32 f16/bf16 (gfx950)
  DISPATCH_MFMA_SSA(16, 32, elemTyA.isF16(), mfma_f32_16x16x32_f16)
  DISPATCH_MFMA_SSA(16, 32, elemTyA.isBF16(), mfma_f32_16x16x32_bf16)

  // K=128 fp8 mfma_scale (gfx950)
  if (m == 16 && n == 16 && k == 128 && isFP8(elemTyA) && isFP8(elemTyB)) {
    auto zeroAttr = builder.getI32IntegerAttr(0);
    Value zeroVal = arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
    return ROCDL::mfma_scale_f32_16x16x128_f8f6f4::create(
               builder, loc, accTy, a, b, c,
               /*cbsz=*/zeroAttr, /*blgp=*/zeroAttr,
               /*opselA=*/zeroAttr, /*scaleA=*/zeroVal,
               /*opselB=*/zeroAttr, /*scaleB=*/zeroVal)
        .getResult();
  }

  // ── CDNA3 instructions (also available on CDNA4) ───────────────────────────

  DISPATCH_MFMA_SSA(32, 1, elemTyA.isF32(), mfma_f32_32x32x1f32)
  DISPATCH_MFMA_SSA(16, 1, elemTyA.isF32(), mfma_f32_16x16x1f32)
  DISPATCH_MFMA_SSA(4, 1, elemTyA.isF32(), mfma_f32_4x4x1f32)
  DISPATCH_MFMA_SSA(32, 2, elemTyA.isF32(), mfma_f32_32x32x2f32)
  DISPATCH_MFMA_SSA(16, 4, elemTyA.isF32(), mfma_f32_16x16x4f32)

  DISPATCH_MFMA_SSA(32, 4, elemTyA.isF16(), mfma_f32_32x32x4f16)
  DISPATCH_MFMA_SSA(16, 4, elemTyA.isF16(), mfma_f32_16x16x4f16)
  DISPATCH_MFMA_SSA(4, 4, elemTyA.isF16(), mfma_f32_4x4x4f16)
  DISPATCH_MFMA_SSA(32, 8, elemTyA.isF16(), mfma_f32_32x32x8f16)
  DISPATCH_MFMA_SSA(16, 16, elemTyA.isF16(), mfma_f32_16x16x16f16)

  DISPATCH_MFMA_SSA(32, 2, elemTyA.isBF16(), mfma_f32_32x32x2bf16)
  DISPATCH_MFMA_SSA(16, 2, elemTyA.isBF16(), mfma_f32_16x16x2bf16)
  DISPATCH_MFMA_SSA(4, 2, elemTyA.isBF16(), mfma_f32_4x4x2bf16)
  DISPATCH_MFMA_SSA(32, 4, elemTyA.isBF16(), mfma_f32_32x32x4bf16)
  DISPATCH_MFMA_SSA(16, 8, elemTyA.isBF16(), mfma_f32_16x16x8bf16)
  DISPATCH_MFMA_SSA(16, 16, elemTyA.isBF16(), mfma_f32_16x16x16bf16_1k)

  DISPATCH_MFMA_SSA(16, 32, isFP8(elemTyA) && isFP8(elemTyB), mfma_f32_16x16x32_fp8_fp8)
  DISPATCH_MFMA_SSA(16, 32, isFP8(elemTyA) && isBF8(elemTyB), mfma_f32_16x16x32_fp8_bf8)
  DISPATCH_MFMA_SSA(16, 32, isBF8(elemTyA) && isFP8(elemTyB), mfma_f32_16x16x32_bf8_fp8)
  DISPATCH_MFMA_SSA(16, 32, isBF8(elemTyA) && isBF8(elemTyB), mfma_f32_16x16x32_bf8_bf8)
  DISPATCH_MFMA_SSA(32, 16, isFP8(elemTyA) && isFP8(elemTyB), mfma_f32_32x32x16_fp8_fp8)
  DISPATCH_MFMA_SSA(32, 16, isFP8(elemTyA) && isBF8(elemTyB), mfma_f32_32x32x16_fp8_bf8)
  DISPATCH_MFMA_SSA(32, 16, isBF8(elemTyA) && isFP8(elemTyB), mfma_f32_32x32x16_bf8_fp8)
  DISPATCH_MFMA_SSA(32, 16, isBF8(elemTyA) && isBF8(elemTyB), mfma_f32_32x32x16_bf8_bf8)

#undef DISPATCH_MFMA_SSA

  return failure();
}

LogicalResult MmaOpCDNA4_MFMAType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                Type dMemTy, Type aMemTy, Type bMemTy, Type cMemTy,
                                                Value atomVal, Value dPtr, Value aPtr, Value bPtr,
                                                Value cPtr) const {
  int32_t m = getM();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getMfmaABType(ctx, elemTyA, k);
  Type abTyB = getMfmaABType(ctx, elemTyB, k);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getMfmaAccVecSize(m, k, elemTyA);
  if (accVecSize == 0)
    return failure();

  Type accElemTy = getElemTyAcc();
  VectorType accTy = VectorType::get({accVecSize}, accElemTy);

  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, accTy, mmaAtomTy, Type{}, abTyA, abTyB, accTy, atomVal,
                             Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

} // namespace mlir::fly_rocdl
