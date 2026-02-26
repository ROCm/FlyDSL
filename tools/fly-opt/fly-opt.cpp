#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Conversion/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::fly::registerFlyPasses();
  mlir::registerFlyToROCDLConversionPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::fly::FlyDialect>();
  registry.insert<mlir::fly_rocdl::FlyROCDLDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Fly Optimizer Driver\n", registry));
}
