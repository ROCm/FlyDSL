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

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Backend/ForEachBackend.h"

// Forward-declare per-backend CAPI registration functions.
// FLYDSL_BACKENDS_TUPLE is set by CMake (e.g. (rocdl)).
#define DECLARE_BACKEND(name)                                                  \
  extern "C" void flydsl_register_##name##_dialects(MlirDialectRegistry);      \
  extern "C" void flydsl_register_##name##_passes(void);
FOR_EACH_BACKEND(DECLARE_BACKEND, FLYDSL_BACKENDS_TUPLE)

#define REGISTER_BACKEND_DIALECTS(name)                                        \
  flydsl_register_##name##_dialects(wrap(&registry));
#define REGISTER_BACKEND_PASSES(name) flydsl_register_##name##_passes();

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::fly::registerFlyPasses();
  FOR_EACH_BACKEND(REGISTER_BACKEND_PASSES, FLYDSL_BACKENDS_TUPLE)

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  registry.insert<mlir::fly::FlyDialect>();
  FOR_EACH_BACKEND(REGISTER_BACKEND_DIALECTS, FLYDSL_BACKENDS_TUPLE)

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Fly Optimizer Driver\n", registry));
}
