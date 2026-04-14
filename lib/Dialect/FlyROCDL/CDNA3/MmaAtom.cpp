// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::fly;

namespace cdna3 {

LayoutAttr getThrValLayoutAB(MLIRContext *ctx, int32_t M, int32_t N, int32_t K, Type elemTyA,
                             Type elemTyB, Type elemTyAcc) {
  auto getContext = [&]() { return ctx; };

  int MN = M;
  assert(M == N && "M and N must be equal");

  int GroupK = 64 / MN;
  int KPerThread = K / GroupK;

  return FxLayout(FxShape(FxThr(MN, GroupK), FxVal(KPerThread)),
                  FxStride(FxThr(1, MN * KPerThread), FxVal(MN)));
}

} // namespace cdna3

namespace cdna4 {}

namespace mlir::fly_rocdl {

bool MmaOpCDNA3_MFMAType::isStatic() const { return true; }

Value MmaOpCDNA3_MFMAType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                              Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Attribute MmaOpCDNA3_MFMAType::getThrLayout() const { return FxLayout(FxC(64), FxC(1)); }

Attribute MmaOpCDNA3_MFMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpCDNA3_MFMAType::getValTypeA() const { return getElemTyA(); }
Type MmaOpCDNA3_MFMAType::getValTypeB() const { return getElemTyB(); }
Type MmaOpCDNA3_MFMAType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpCDNA3_MFMAType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpCDNA3_MFMAType::getThrValLayoutA() const {
  return cdna3::getThrValLayoutAB(getContext(), getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                                  getElemTyAcc());
}
Attribute MmaOpCDNA3_MFMAType::getThrValLayoutB() const {
  return cdna3::getThrValLayoutAB(getContext(), getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                                  getElemTyAcc());
}
Attribute MmaOpCDNA3_MFMAType::getThrValLayoutC() const {
  int M = getM();
  int N = getN();

  int GroupM = 64 / N;
  int ValM0 = 4;
  int ValM1 = M / 4 / GroupM;

  return FxLayout(FxShape(FxThr(N, GroupM), FxVal(ValM0, ValM1)),
                  FxStride(FxThr(M, ValM0), FxVal(1, ValM0 * GroupM)));
}

LogicalResult MmaOpCDNA3_MFMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                          int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                          Type elemTyAcc) {
  assert(m == n && "M and N must be equal");
  if (m != n) {
    return emitError() << "invalid MNK dimensions for CDNA3 MFMA: " << m << "x" << n << "x" << k;
  }
  if (!elemTyAcc.isF32())
    return emitError() << "elemTyAcc must be f32, got " << elemTyAcc;

  auto isValidElemType = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isF32() || isa<Float8E4M3FNUZType>(ty) ||
           isa<Float8E5M2FNUZType>(ty);
  };
  if (!isValidElemType(elemTyA)) {
    return emitError() << "elemTyA must be f16, bf16, f32, f8E4M3FNUZ, f8E5M2FNUZ, got " << elemTyA;
  }
  if (!isValidElemType(elemTyB)) {
    return emitError() << "elemTyB must be f16, bf16, f32, f8E4M3FNUZ, f8E5M2FNUZ, got " << elemTyB;
  }
  return success();
}

static bool isFP8(Type ty) { return isa<Float8E4M3FNUZType>(ty); }
static bool isBF8(Type ty) { return isa<Float8E5M2FNUZType>(ty); }
static bool isF8(Type ty) { return isFP8(ty) || isBF8(ty); }

static Type getMfmaABType(MLIRContext *ctx, Type elemTy, int32_t k = 0) {
  if (elemTy.isF32())
    return Float32Type::get(ctx);
  if (elemTy.isF16())
    return VectorType::get({4}, Float16Type::get(ctx));
  if (elemTy.isBF16())
    return VectorType::get({(k >= 16) ? 4 : 2}, IntegerType::get(ctx, 16));
  if (isF8(elemTy))
    return IntegerType::get(ctx, 64);
  return nullptr;
}

static int64_t getMfmaAccVecSize(int32_t m, int32_t k, Type elemTyA) {
  if (elemTyA.isF32()) {
    if (m == 32 && k == 1)
      return 32;
    if (m == 32 && k == 2)
      return 16;
    if (m == 16 && k == 1)
      return 16;
    if (m == 16 && k == 4)
      return 4;
    if (m == 4 && k == 1)
      return 4;
  }
  if (elemTyA.isF16()) {
    if (m == 32 && k == 4)
      return 32;
    if (m == 32 && k == 8)
      return 16;
    if (m == 16 && k == 4)
      return 16;
    if (m == 16 && k == 16)
      return 4;
    if (m == 4 && k == 4)
      return 4;
  }
  if (elemTyA.isBF16()) {
    if (m == 32 && k == 2)
      return 32;
    if (m == 32 && k == 4)
      return 16;
    if (m == 16 && k == 2)
      return 16;
    if (m == 16 && k == 8)
      return 4;
    if (m == 16 && k == 16)
      return 4;
    if (m == 4 && k == 2)
      return 4;
  }
  if (isF8(elemTyA)) {
    if (m == 16 && k == 32)
      return 4;
    if (m == 32 && k == 16)
      return 16;
  }
  return 0;
}

template <typename MfmaOp>
static LogicalResult emitMfma(OpBuilder &builder, Location loc, Type abTyA, Type abTyB,
                              VectorType accTy, Value aPtr, Value bPtr, Value cPtr, Value dPtr) {
  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto zeroAttr = builder.getI32IntegerAttr(0);
  Value res = MfmaOp::create(builder, loc, accTy, a, b, c, zeroAttr, zeroAttr, zeroAttr);
  LLVM::StoreOp::create(builder, loc, res, dPtr);
  return success();
}

LogicalResult MmaOpCDNA3_MFMAType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                Type dMemTy, Type aMemTy, Type bMemTy, Type cMemTy,
                                                Value atomVal, Value dPtr, Value aPtr, Value bPtr,
                                                Value cPtr) const {
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

  int64_t accVecSize = getMfmaAccVecSize(m, k, elemTyA);
  if (accVecSize == 0)
    return failure();

  Type accElemTy = getElemTyAcc();
  VectorType accTy = VectorType::get({accVecSize}, accElemTy);

#define DISPATCH_MFMA(M_, K_, PRED, OP)                                                            \
  if (m == M_ && n == M_ && k == K_ && (PRED))                                                     \
    return emitMfma<ROCDL::OP>(builder, loc, abTyA, abTyB, accTy, aPtr, bPtr, cPtr, dPtr);

  DISPATCH_MFMA(32, 1, elemTyA.isF32(), mfma_f32_32x32x1f32)
  DISPATCH_MFMA(16, 1, elemTyA.isF32(), mfma_f32_16x16x1f32)
  DISPATCH_MFMA(4, 1, elemTyA.isF32(), mfma_f32_4x4x1f32)
  DISPATCH_MFMA(32, 2, elemTyA.isF32(), mfma_f32_32x32x2f32)
  DISPATCH_MFMA(16, 4, elemTyA.isF32(), mfma_f32_16x16x4f32)

  DISPATCH_MFMA(32, 4, elemTyA.isF16(), mfma_f32_32x32x4f16)
  DISPATCH_MFMA(16, 4, elemTyA.isF16(), mfma_f32_16x16x4f16)
  DISPATCH_MFMA(4, 4, elemTyA.isF16(), mfma_f32_4x4x4f16)
  DISPATCH_MFMA(32, 8, elemTyA.isF16(), mfma_f32_32x32x8f16)
  DISPATCH_MFMA(16, 16, elemTyA.isF16(), mfma_f32_16x16x16f16)

  DISPATCH_MFMA(32, 2, elemTyA.isBF16(), mfma_f32_32x32x2bf16)
  DISPATCH_MFMA(16, 2, elemTyA.isBF16(), mfma_f32_16x16x2bf16)
  DISPATCH_MFMA(4, 2, elemTyA.isBF16(), mfma_f32_4x4x2bf16)
  DISPATCH_MFMA(32, 4, elemTyA.isBF16(), mfma_f32_32x32x4bf16)
  DISPATCH_MFMA(16, 8, elemTyA.isBF16(), mfma_f32_16x16x8bf16)
  DISPATCH_MFMA(16, 16, elemTyA.isBF16(), mfma_f32_16x16x16bf16_1k)

  DISPATCH_MFMA(16, 32, isFP8(elemTyA) && isFP8(elemTyB), mfma_f32_16x16x32_fp8_fp8)
  DISPATCH_MFMA(16, 32, isFP8(elemTyA) && isBF8(elemTyB), mfma_f32_16x16x32_fp8_bf8)
  DISPATCH_MFMA(16, 32, isBF8(elemTyA) && isFP8(elemTyB), mfma_f32_16x16x32_bf8_fp8)
  DISPATCH_MFMA(16, 32, isBF8(elemTyA) && isBF8(elemTyB), mfma_f32_16x16x32_bf8_bf8)
  DISPATCH_MFMA(32, 16, isFP8(elemTyA) && isFP8(elemTyB), mfma_f32_32x32x16_fp8_fp8)
  DISPATCH_MFMA(32, 16, isFP8(elemTyA) && isBF8(elemTyB), mfma_f32_32x32x16_fp8_bf8)
  DISPATCH_MFMA(32, 16, isBF8(elemTyA) && isFP8(elemTyB), mfma_f32_32x32x16_bf8_fp8)
  DISPATCH_MFMA(32, 16, isBF8(elemTyA) && isBF8(elemTyB), mfma_f32_32x32x16_bf8_bf8)

#undef DISPATCH_MFMA

  return failure();
}

} // namespace mlir::fly_rocdl
