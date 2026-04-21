// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace cdna4 {

LayoutAttr getThrValLayoutAB(MLIRContext *ctx, int32_t M, int32_t N, int32_t K, Type elemTy) {
  auto getContext = [&]() { return ctx; };

  int MN = M;
  assert(M == N && "M and N must be equal");

  int GroupK = 64 / MN;
  int KPerThread = K / GroupK;

  return FxLayout(FxShape(FxThr(MN, GroupK), FxVal(KPerThread)),
                  FxStride(FxThr(1, MN * KPerThread), FxVal(MN)));
}

LayoutAttr getThrValLayoutC(MLIRContext *ctx, int32_t M, int32_t N) {
  auto getContext = [&]() { return ctx; };

  int GroupM = 64 / N;
  int ValM0 = 4;
  int ValM1 = M / 4 / GroupM;

  return FxLayout(FxShape(FxThr(N, GroupM), FxVal(ValM0, ValM1)),
                  FxStride(FxThr(M, ValM0), FxVal(1, ValM0 * GroupM)));
}

} // namespace cdna4

namespace mlir::fly_rocdl {

//===----------------------------------------------------------------------===//
// MmaOpCDNA4_MFMAScaleType
//===----------------------------------------------------------------------===//

std::optional<unsigned> MmaOpCDNA4_MFMAScaleType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::ScaleA:
    return 0;
  case AtomStateField::ScaleB:
    return 1;
  default:
    return std::nullopt;
  }
}

Type MmaOpCDNA4_MFMAScaleType::getConvertedType(MLIRContext *ctx) const {
  auto i32Ty = IntegerType::get(ctx, 32);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty});
}

Value MmaOpCDNA4_MFMAScaleType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ScaleA)});
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ScaleB)});
  return state;
}

Value MmaOpCDNA4_MFMAScaleType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                             Attribute fieldAttr, Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto field = symbolizeAtomStateField(fieldStr.getValue());
  if (!field)
    return nullptr;
  auto idx = getFieldIndex(*field);
  if (!idx)
    return nullptr;
  Value scaleVal = fieldValue;
  Type srcTy = scaleVal.getType();
  Type i32Ty = IntegerType::get(builder.getContext(), 32);
  if (srcTy != i32Ty) {
    auto bitWidthOf = [](Type t) -> unsigned {
      if (auto vec = dyn_cast<VectorType>(t)) {
        Type elt = vec.getElementType();
        if (!elt.isIntOrFloat())
          return 0;
        return elt.getIntOrFloatBitWidth() * vec.getNumElements();
      }
      if (auto intTy = dyn_cast<IntegerType>(t))
        return intTy.getWidth();
      return 0;
    };
    if (bitWidthOf(srcTy) != 32)
      return nullptr;
    scaleVal = LLVM::BitcastOp::create(builder, loc, i32Ty, scaleVal);
  }
  return LLVM::InsertValueOp::create(builder, loc, atomStruct, scaleVal, ArrayRef<int64_t>{*idx});
}

Attribute MmaOpCDNA4_MFMAScaleType::getThrLayout() const { return FxLayout(FxC(64), FxC(1)); }

Attribute MmaOpCDNA4_MFMAScaleType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpCDNA4_MFMAScaleType::getValTypeA() const { return getElemTyA(); }
Type MmaOpCDNA4_MFMAScaleType::getValTypeB() const { return getElemTyB(); }
Type MmaOpCDNA4_MFMAScaleType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpCDNA4_MFMAScaleType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpCDNA4_MFMAScaleType::getThrValLayoutA() const {
  return cdna4::getThrValLayoutAB(getContext(), getM(), getN(), getK(), getElemTyA());
}

Attribute MmaOpCDNA4_MFMAScaleType::getThrValLayoutB() const {
  return cdna4::getThrValLayoutAB(getContext(), getM(), getN(), getK(), getElemTyB());
}

Attribute MmaOpCDNA4_MFMAScaleType::getThrValLayoutC() const {
  return cdna4::getThrValLayoutC(getContext(), getM(), getN());
}

static std::optional<uint32_t> mfmaFloatTypeEncode(Type elemTy) {
  if (isa<Float8E4M3FNType>(elemTy))
    return 0u;
  if (isa<Float8E5M2Type>(elemTy))
    return 1u;
  if (isa<Float6E2M3FNType>(elemTy))
    return 2u;
  if (isa<Float6E3M2FNType>(elemTy))
    return 3u;
  if (isa<Float4E2M1FNType>(elemTy))
    return 4u;
  return std::nullopt;
}

static bool isSupportedScaledElemTy(Type ty) {
  return isa<Float8E4M3FNType, Float8E5M2Type, Float6E2M3FNType, Float6E3M2FNType,
             Float4E2M1FNType>(ty);
}

LogicalResult MmaOpCDNA4_MFMAScaleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                               int32_t m, int32_t n, int32_t k, Type elemTyA,
                                               Type elemTyB, Type elemTyAcc, int32_t opselA,
                                               int32_t opselB) {
  if (!((m == 16 && n == 16 && k == 128) || (m == 32 && n == 32 && k == 64))) {
    return emitError() << "unsupported MNK for CDNA4 MFMA_Scale: " << m << "x" << n << "x" << k
                       << " (expected 16x16x128 or 32x32x64)";
  }
  if (!elemTyAcc.isF32())
    return emitError() << "elemTyAcc must be f32, got " << elemTyAcc;
  if (!isSupportedScaledElemTy(elemTyA)) {
    return emitError() << "elemTyA must be one of f8E4M3FN, f8E5M2, f6E2M3FN, "
                          "f6E3M2FN, f4E2M1FN, got "
                       << elemTyA;
  }
  if (!isSupportedScaledElemTy(elemTyB)) {
    return emitError() << "elemTyB must be one of f8E4M3FN, f8E5M2, f6E2M3FN, "
                          "f6E3M2FN, f4E2M1FN, got "
                       << elemTyB;
  }
  if (opselA < 0 || opselA > 3)
    return emitError() << "opselA must be in [0, 3], got " << opselA;
  if (opselB < 0 || opselB > 3)
    return emitError() << "opselB must be in [0, 3], got " << opselB;
  return success();
}

static Type getScaledMfmaABType(MLIRContext *ctx, Type elemTy) {
  Type i32Ty = IntegerType::get(ctx, 32);
  if (isa<Float8E4M3FNType, Float8E5M2Type>(elemTy))
    return VectorType::get({8}, i32Ty);
  if (isa<Float6E2M3FNType, Float6E3M2FNType>(elemTy))
    return VectorType::get({6}, i32Ty);
  if (isa<Float4E2M1FNType>(elemTy))
    return VectorType::get({4}, i32Ty);
  return nullptr;
}

static int64_t getScaledMfmaAccVecSize(int32_t m, int32_t n) {
  if (m == 16 && n == 16)
    return 4;
  if (m == 32 && n == 32)
    return 16;
  return 0;
}

FailureOr<Value> MmaOpCDNA4_MFMAScaleType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                           Type resultTy, Type mmaAtomTyArg,
                                                           Type dTyArg, Type aTyArg, Type bTyArg,
                                                           Type cTyArg, Value atomVal, Value d,
                                                           Value a, Value b, Value c) const {
  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getScaledMfmaABType(ctx, elemTyA);
  Type abTyB = getScaledMfmaABType(ctx, elemTyB);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getScaledMfmaAccVecSize(m, n);
  if (accVecSize == 0)
    return failure();

  std::optional<uint32_t> aTypeCode = mfmaFloatTypeEncode(elemTyA);
  std::optional<uint32_t> bTypeCode = mfmaFloatTypeEncode(elemTyB);
  if (!aTypeCode || !bTypeCode)
    return failure();

  Type accElemTy = getElemTyAcc();
  VectorType accTy = VectorType::get({accVecSize}, accElemTy);

  if (a.getType() != abTyA)
    a = LLVM::BitcastOp::create(builder, loc, abTyA, a);
  if (b.getType() != abTyB)
    b = LLVM::BitcastOp::create(builder, loc, abTyB, b);
  if (c.getType() != accTy)
    c = LLVM::BitcastOp::create(builder, loc, accTy, c);

  Value scaleA = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ScaleA)});
  Value scaleB = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ScaleB)});

  auto cbszAttr = builder.getI32IntegerAttr(*aTypeCode);
  auto blgpAttr = builder.getI32IntegerAttr(*bTypeCode);
  auto opselAAttr = builder.getI32IntegerAttr(getOpselA());
  auto opselBAttr = builder.getI32IntegerAttr(getOpselB());

  if (m == 16 && n == 16 && k == 128) {
    return ROCDL::mfma_scale_f32_16x16x128_f8f6f4::create(builder, loc, accTy, a, b, c, cbszAttr,
                                                          blgpAttr, opselAAttr, scaleA, opselBAttr,
                                                          scaleB)
        .getResult();
  }
  if (m == 32 && n == 32 && k == 64) {
    return ROCDL::mfma_scale_f32_32x32x64_f8f6f4::create(builder, loc, accTy, a, b, c, cbszAttr,
                                                         blgpAttr, opselAAttr, scaleA, opselBAttr,
                                                         scaleB)
        .getResult();
  }

  return failure();
}

LogicalResult MmaOpCDNA4_MFMAScaleType::emitAtomCall(OpBuilder &builder, Location loc,
                                                     Type mmaAtomTy, Type dMemTy, Type aMemTy,
                                                     Type bMemTy, Type cMemTy, Value atomVal,
                                                     Value dPtr, Value aPtr, Value bPtr,
                                                     Value cPtr) const {
  int32_t m = getM();
  int32_t n = getN();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getScaledMfmaABType(ctx, elemTyA);
  Type abTyB = getScaledMfmaABType(ctx, elemTyB);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getScaledMfmaAccVecSize(m, n);
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
