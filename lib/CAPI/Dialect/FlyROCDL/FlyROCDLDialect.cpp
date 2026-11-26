// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"

#include <cstdio>

bool mlirFlyROCDLHasInProcessLLD(void) {
#ifdef FLYDSL_HAS_LLD_LIBRARY
  return true;
#else
  return false;
#endif
}

bool mlirFlyROCDLLinkHsacoInProcess(const char *objectPath, const char *hsacoPath, char *error,
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

MlirStringRef mlirFlyROCDLGetROCMPath(void) { return wrap(mlir::ROCDL::getROCMPath()); }
