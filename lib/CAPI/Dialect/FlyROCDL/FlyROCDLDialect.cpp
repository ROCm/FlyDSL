#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

// Pull in the conversion pass registration (inline functions).
namespace mlir {
#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly_rocdl;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl, mlir::fly_rocdl::FlyROCDLDialect)

//===----------------------------------------------------------------------===//
// MmaAtomCDNA3_MFMAType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyROCDLMmaAtomCDNA3_MFMAType(MlirType type) {
  return isa<MmaAtomCDNA3_MFMAType>(unwrap(type));
}

MlirTypeID mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetTypeID(void) {
  return wrap(MmaAtomCDNA3_MFMAType::getTypeID());
}

MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGet(int32_t m, int32_t n, int32_t k, MlirType elemTyA,
                                              MlirType elemTyB, MlirType elemTyAcc) {
  return wrap(
      MmaAtomCDNA3_MFMAType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB), unwrap(elemTyAcc)));
}

int32_t mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetM(MlirType type) {
  return cast<MmaAtomCDNA3_MFMAType>(unwrap(type)).getM();
}

int32_t mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetN(MlirType type) {
  return cast<MmaAtomCDNA3_MFMAType>(unwrap(type)).getN();
}

int32_t mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetK(MlirType type) {
  return cast<MmaAtomCDNA3_MFMAType>(unwrap(type)).getK();
}

MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyA(MlirType type) {
  return wrap(cast<MmaAtomCDNA3_MFMAType>(unwrap(type)).getElemTyA());
}

MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyB(MlirType type) {
  return wrap(cast<MmaAtomCDNA3_MFMAType>(unwrap(type)).getElemTyB());
}

MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyAcc(MlirType type) {
  return wrap(cast<MmaAtomCDNA3_MFMAType>(unwrap(type)).getElemTyAcc());
}

//===----------------------------------------------------------------------===//
// CopyOp_CDNA3_BufferLSAType
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void mlirRegisterFlyToROCDLConversionPass(void) {
  mlir::registerFlyToROCDLConversionPass();
}
