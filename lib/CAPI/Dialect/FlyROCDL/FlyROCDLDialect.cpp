#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

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
