// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"

#include "flydsl/Backend/Backend.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"

// Forward-declare per-backend CAPI registration functions.
// FLYDSL_BACKEND_COUNT and FLYDSL_BACKEND_0..N-1 are set by CMake.
#define DECLARE_BACKEND(name)                                                                      \
  extern "C" void flydsl_register_##name##_dialects(MlirDialectRegistry);                          \
  extern "C" void flydsl_register_##name##_passes(void);
FLYDSL_FOR_EACH_BACKEND(DECLARE_BACKEND)

#define REGISTER_BACKEND_DIALECTS(name) flydsl_register_##name##_dialects(wrap(&registry));
#define REGISTER_BACKEND_PASSES(name) flydsl_register_##name##_passes();

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::fly::registerFlyPasses();
  FLYDSL_FOR_EACH_BACKEND(REGISTER_BACKEND_PASSES)

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  registry.insert<mlir::fly::FlyDialect>();
  FLYDSL_FOR_EACH_BACKEND(REGISTER_BACKEND_DIALECTS)

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Fly Optimizer Driver\n", registry));
}
