// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Codegen/AMDGPU/PassRegistration.h"
#include "flydsl/Conversion/Passes.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Target/FlyAMDGPUTarget.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl, mlir::fly_rocdl::FlyROCDLDialect)

void mlirRegisterFlyToROCDLConversionPass(void) { mlir::registerFlyToROCDLConversionPass(); }
void mlirRegisterFlyROCDLClusterAttrPass(void) { mlir::registerFlyROCDLClusterAttrPass(); }

void mlirRegisterFlyAMDGPUCodegenPasses(void) {
  flydsl::registerFlyAMDGPUCodegenPasses();
}

void mlirRegisterFlyAMDGPUTargetInterfaceExternalModels(
    MlirDialectRegistry registry) {
  flydsl::registerFlyAMDGPUTargetInterfaceExternalModels(*unwrap(registry));
}
