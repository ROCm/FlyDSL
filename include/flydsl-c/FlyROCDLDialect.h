// Copyright (c) 2025 FlyDSL Project Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
// CopyOpCDNA3BufferLDSTType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLCopyOpCDNA3BufferLDSTType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyROCDLCopyOpCDNA3BufferLDSTTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLCopyOpCDNA3BufferLDSTTypeGet(MlirContext ctx,
                                                                     int32_t bitSize);
MLIR_CAPI_EXPORTED int32_t mlirFlyROCDLCopyOpCDNA3BufferLDSTTypeGetBitSize(MlirType type);

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void mlirRegisterFlyToROCDLConversionPass(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYROCDLDIALECT_H
