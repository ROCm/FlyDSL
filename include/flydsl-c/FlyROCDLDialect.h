#ifndef FLYDSL_C_FLYROCDLDIALECT_H
#define FLYDSL_C_FLYROCDLDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl);

//===----------------------------------------------------------------------===//
// MmaAtomCDNA3_MFMAType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLMmaAtomCDNA3_MFMAType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYROCDLDIALECT_H
