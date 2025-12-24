#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "flir/FlirDialect.h"
#include "flir/FlirPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  
  // Register implemented flir passes
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::flir::createFlirToStandardPass();
  });
  
  mlir::DialectRegistry registry;
  registry.insert<mlir::flir::FlirDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::ROCDL::ROCDLDialect>();
  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Flir Optimizer Driver", registry));
}
