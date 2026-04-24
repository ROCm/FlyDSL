// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

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
           isa<Float8E5M2FNUZType>(ty) || isa<Float8E4M3FNType>(ty);
  };
  if (!isValidElemType(elemTyA)) {
    return emitError() << "elemTyA must be f16, bf16, f32, f8E4M3FNUZ, f8E5M2FNUZ, got " << elemTyA;
  }
  if (!isValidElemType(elemTyB)) {
    return emitError() << "elemTyB must be f16, bf16, f32, f8E4M3FNUZ, f8E5M2FNUZ, got " << elemTyB;
  }
  return success();
}

static bool isFP8(Type ty) { return isa<Float8E4M3FNUZType>(ty) || isa<Float8E4M3FNType>(ty); }
static bool isBF8(Type ty) { return isa<Float8E5M2FNUZType>(ty); }

static Type getMfmaABType(MLIRContext *ctx, Type elemTy, int32_t mn, int32_t k = 0) {
  if (elemTy.isF32())
    return Float32Type::get(ctx);
  if (elemTy.isF16())
    return VectorType::get({mn * k / 64}, Float16Type::get(ctx));
  if (elemTy.isBF16()) {
    int vecSize = mn * k / 64;
    Type elemTy;
    if (vecSize == 8) {
      elemTy = BFloat16Type::get(ctx); // CDNA4 version
    } else {
      elemTy = IntegerType::get(ctx, 16);
    }
    return VectorType::get({vecSize}, elemTy);
  }
  if (elemTy.getIntOrFloatBitWidth() == 8)
    return IntegerType::get(ctx, 64);
  return nullptr;
}

static int64_t getMfmaAccVecSize(int32_t m, int32_t n, Type elemTyA) {
  if (m == 16 && n == 16)
    return 4;
  if (m == 32 && n == 32)
    return 16;
  return 0;
}

FailureOr<Value> MmaOpCDNA3_MFMAType::emitAtomCallSSA(OpBuilder &builder, Location loc,
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

  Type abTyA = getMfmaABType(ctx, elemTyA, m, k);
  Type abTyB = getMfmaABType(ctx, elemTyB, n, k);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getMfmaAccVecSize(m, n, elemTyA);
  if (accVecSize == 0)
    return failure();

  Type accElemTy = getElemTyAcc();
  VectorType accTy = VectorType::get({accVecSize}, accElemTy);

  if (a.getType() != abTyA)
    a = LLVM::BitcastOp::create(builder, loc, abTyA, a);
  if (b.getType() != abTyB)
    b = LLVM::BitcastOp::create(builder, loc, abTyB, b);
  if (c.getType() != accTy)
    c = LLVM::BitcastOp::create(builder, loc, accTy, c);

#define DISPATCH_MFMA_SSA(M_, K_, PRED, OP)                                                        \
  if (m == M_ && n == M_ && k == K_ && (PRED)) {                                                   \
    auto zeroAttr = builder.getI32IntegerAttr(0);                                                  \
    return ROCDL::OP::create(builder, loc, accTy, a, b, c, zeroAttr, zeroAttr, zeroAttr)           \
        .getResult();                                                                              \
  }

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
  DISPATCH_MFMA_SSA(16, 32, elemTyA.isF16(), mfma_f32_16x16x32_f16)
  DISPATCH_MFMA_SSA(32, 16, elemTyA.isF16(), mfma_f32_32x32x16_f16)

  DISPATCH_MFMA_SSA(32, 2, elemTyA.isBF16(), mfma_f32_32x32x2bf16)
  DISPATCH_MFMA_SSA(16, 2, elemTyA.isBF16(), mfma_f32_16x16x2bf16)
  DISPATCH_MFMA_SSA(4, 2, elemTyA.isBF16(), mfma_f32_4x4x2bf16)
  DISPATCH_MFMA_SSA(32, 4, elemTyA.isBF16(), mfma_f32_32x32x4bf16)
  DISPATCH_MFMA_SSA(16, 8, elemTyA.isBF16(), mfma_f32_16x16x8bf16)
  DISPATCH_MFMA_SSA(16, 16, elemTyA.isBF16(), mfma_f32_16x16x16bf16_1k)
  DISPATCH_MFMA_SSA(16, 32, elemTyA.isBF16(), mfma_f32_16x16x32_bf16)
  DISPATCH_MFMA_SSA(32, 16, elemTyA.isBF16(), mfma_f32_32x32x16_bf16)

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

  Type abTyA = getMfmaABType(ctx, elemTyA, m, k);
  Type abTyB = getMfmaABType(ctx, elemTyB, n, k);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getMfmaAccVecSize(m, n, elemTyA);
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
