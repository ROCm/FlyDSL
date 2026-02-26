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

// Constructor
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGet(int32_t m, int32_t n, int32_t k,
                                                                 MlirType elemTyA, MlirType elemTyB,
                                                                 MlirType elemTyAcc);

// Accessors
MLIR_CAPI_EXPORTED int32_t mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetM(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetN(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetK(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyB(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyAcc(MlirType type);

//===----------------------------------------------------------------------===//
// CopyOp_CDNA3_BufferLSAType
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register the FlyToROCDL conversion pass.
MLIR_CAPI_EXPORTED void mlirRegisterFlyToROCDLConversionPass(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYROCDLDIALECT_H
