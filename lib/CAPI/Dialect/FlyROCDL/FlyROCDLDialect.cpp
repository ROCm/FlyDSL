// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Conversion/Passes.h"
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
// MmaAtomGFX1250_WMMAType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyROCDLMmaAtomGFX1250_WMMAType(MlirType type) {
  return isa<MmaAtomGFX1250_WMMAType>(unwrap(type));
}

MlirTypeID mlirFlyROCDLMmaAtomGFX1250_WMMATypeGetTypeID(void) {
  return wrap(MmaAtomGFX1250_WMMAType::getTypeID());
}

MlirType mlirFlyROCDLMmaAtomGFX1250_WMMATypeGet(int32_t m, int32_t n, int32_t k,
                                                 MlirType elemTyA,
                                                 MlirType elemTyB,
                                                 MlirType elemTyAcc) {
  return wrap(MmaAtomGFX1250_WMMAType::get(m, n, k, unwrap(elemTyA),
                                           unwrap(elemTyB), unwrap(elemTyAcc)));
}

int32_t mlirFlyROCDLMmaAtomGFX1250_WMMATypeGetM(MlirType type) {
  return cast<MmaAtomGFX1250_WMMAType>(unwrap(type)).getM();
}

int32_t mlirFlyROCDLMmaAtomGFX1250_WMMATypeGetN(MlirType type) {
  return cast<MmaAtomGFX1250_WMMAType>(unwrap(type)).getN();
}

int32_t mlirFlyROCDLMmaAtomGFX1250_WMMATypeGetK(MlirType type) {
  return cast<MmaAtomGFX1250_WMMAType>(unwrap(type)).getK();
}

MlirType mlirFlyROCDLMmaAtomGFX1250_WMMATypeGetElemTyA(MlirType type) {
  return wrap(cast<MmaAtomGFX1250_WMMAType>(unwrap(type)).getElemTyA());
}

MlirType mlirFlyROCDLMmaAtomGFX1250_WMMATypeGetElemTyB(MlirType type) {
  return wrap(cast<MmaAtomGFX1250_WMMAType>(unwrap(type)).getElemTyB());
}

MlirType mlirFlyROCDLMmaAtomGFX1250_WMMATypeGetElemTyAcc(MlirType type) {
  return wrap(cast<MmaAtomGFX1250_WMMAType>(unwrap(type)).getElemTyAcc());
}

//===----------------------------------------------------------------------===//
// CopyOpCDNA3BufferCopyType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopyType(MlirType type) {
  return isa<CopyOpCDNA3BufferCopyType>(unwrap(type));
}

MlirTypeID mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGetTypeID(void) {
  return wrap(CopyOpCDNA3BufferCopyType::getTypeID());
}

MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGet(MlirContext ctx, int32_t bitSize) {
  return wrap(CopyOpCDNA3BufferCopyType::get(unwrap(ctx), bitSize));
}

int32_t mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGetBitSize(MlirType type) {
  return cast<CopyOpCDNA3BufferCopyType>(unwrap(type)).getBitSize();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void mlirRegisterFlyToROCDLConversionPass(void) { mlir::registerFlyToROCDLConversionPass(); }
void mlirRegisterFlyROCDLClusterAttrPass(void) { mlir::registerFlyROCDLClusterAttrPass(); }
