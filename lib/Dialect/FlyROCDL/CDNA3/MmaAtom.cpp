#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

namespace cdna3 {

LayoutAttr getThrValLayoutAB(MLIRContext *ctx, int32_t M, int32_t N, int32_t K, Type elemTyA,
                             Type elemTyB, Type elemTyAcc) {
  auto getContext = [&]() { return ctx; };

  int MN = M;
  int TyBit = elemTyA.getIntOrFloatBitWidth();
  assert(TyBit == int(elemTyB.getIntOrFloatBitWidth()) &&
         "Element types must have the same bit width");
  assert(M == N && "M and N must be equal");

  int GroupK = 64 / MN;
  int KPerThread = K / GroupK;

  return FxLayout(
      FxShape(FxThr(MN, GroupK), FxVal(TyBit, KPerThread)),
      FxStride(FxThr(TyBit, TyBit * MN * KPerThread), FxVal(1, TyBit * MN * KPerThread)));
}

} // namespace cdna3

namespace cdna4 {}

namespace mlir::fly_rocdl {

bool MmaAtomCDNA3_MFMAType::isStatic() const { return true; }

Attribute MmaAtomCDNA3_MFMAType::getThrLayout() const { return FxLayout(FxC(64), FxC(1)); }

Attribute MmaAtomCDNA3_MFMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Attribute MmaAtomCDNA3_MFMAType::getThrValLayoutA() const {
  return cdna3::getThrValLayoutAB(getContext(), getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                                  getElemTyAcc());
}
Attribute MmaAtomCDNA3_MFMAType::getThrValLayoutB() const {
  return cdna3::getThrValLayoutAB(getContext(), getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                                  getElemTyAcc());
}
Attribute MmaAtomCDNA3_MFMAType::getThrValLayoutC() const {
  int M = getM();
  int N = getN();

  int GroupM = 64 / N;
  int ValM0 = 4;
  int ValM1 = M / 4 / GroupM;
  int TyBitAcc = 32;

  return FxLayout(
      FxShape(FxThr(N, GroupM), FxVal(TyBitAcc * ValM0, ValM1)),
      FxStride(FxThr(M * TyBitAcc, TyBitAcc * ValM0), FxVal(1, TyBitAcc * ValM0 * GroupM)));
}

LogicalResult MmaAtomCDNA3_MFMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
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

} // namespace mlir::fly_rocdl
