// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Conversion/Passes.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"

#include <cstdio>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl, mlir::fly_rocdl::FlyROCDLDialect)

void mlirRegisterConvertROCDLFastMathOpsPass(void) { mlir::registerConvertROCDLFastMathOpsPass(); }
void mlirRegisterFlyToROCDLConversionPass(void) { mlir::registerFlyToROCDLConversionPass(); }
void mlirRegisterFlyKtraceToROCDLConversionPass(void) {
  mlir::registerFlyKtraceToROCDLConversionPass();
}
void mlirRegisterFlyEmitGPUBinaryPass(void) { mlir::registerFlyEmitGPUBinaryPass(); }

bool flydsl_has_inprocess_lld(void) {
#ifdef FLYDSL_HAS_LLD_LIBRARY
  return true;
#else
  return false;
#endif
}

bool flydsl_link_hsaco(const char *objectPath, const char *hsacoPath, char *error,
                       size_t errorSize) {
#ifdef FLYDSL_HAS_LLD_LIBRARY
  std::optional<std::string> diag = mlir::fly::linkHsacoInProcess(objectPath, hsacoPath);
  if (!diag)
    return true;
#else
  std::optional<std::string> diag = "FlyDSL was built without in-process LLD";
#endif
  if (error && errorSize)
    snprintf(error, errorSize, "%s", diag->c_str());
  return false;
}

void flydsl_register_rocdl_dialects(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::fly_rocdl::FlyROCDLDialect>();
}

void flydsl_register_rocdl_passes(void) {
  mlirRegisterConvertROCDLFastMathOpsPass();
  mlirRegisterFlyToROCDLConversionPass();
  mlirRegisterFlyEmitGPUBinaryPass();
  mlir::fly_rocdl::registerFlyROCDLPasses();
  mlirRegisterFlyKtraceToROCDLConversionPass();
}

MlirStringRef mlirFlyROCDLGetROCMPath(void) { return wrap(mlir::ROCDL::getROCMPath()); }
