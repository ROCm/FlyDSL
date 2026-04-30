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
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "flydsl/Conversion/Passes.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#ifdef FLYDSL_HAS_ROCDL_TARGET_STACK
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#endif

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::fly::registerFlyPasses();
#ifdef FLYDSL_HAS_ROCDL_TARGET_STACK
  mlir::registerFlyToROCDLConversionPass();
  mlir::registerFlyROCDLClusterAttrPass();
#endif

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::fly::FlyDialect>();
#ifdef FLYDSL_HAS_ROCDL_TARGET_STACK
  registry.insert<mlir::fly_rocdl::FlyROCDLDialect>();
#endif

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Fly Optimizer Driver\n", registry));
}
