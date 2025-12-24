//===- FlirPasses.td - Layout IR Lowering Passes ----------*- tablegen -*-===//
//
// Lowering passes for Flir IR dialects
// Transformation chain: flir → gpu/scf → rocdl → LLVM IR → AMDGCN
//
//===----------------------------------------------------------------------===//

#ifndef FLIR_PASSES
#define FLIR_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// flir → Standard Dialects (scf, arith, memref)
//===----------------------------------------------------------------------===//

// Run at module scope so we can lower Flir ops inside GPU kernels (`gpu.func`)
// as well as regular `func.func`.
def FlirToStandardPass : Pass<"flir-to-standard", "::mlir::ModuleOp"> {
  let summary = "Lower Flir IR to standard MLIR dialects";
  let description = [{
    Converts flir operations to combinations of scf, arith, and memref.
  }];
  
  let constructor = "mlir::flir::createFlirToStandardPass()";
  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Dead Code Elimination (standalone / general)
//===----------------------------------------------------------------------===//

// Upstream MLIR's generic 'dce/adce' passes are not registered in our embedded
// Python environment. Provide a standalone, general DCE pass with a distinct
// name to avoid conflicting with upstream if it becomes available later.
def FlirTrivialDCEPass : Pass<"trivial-dce", "::mlir::ModuleOp"> {
  let summary = "Erase trivially-dead operations (general DCE)";
  let description = [{
    Iteratively erases operations that are unused and have no side effects, as
    determined by MLIR's side-effect interfaces.
  }];
  let constructor = "mlir::flir::createFlirTrivialDCEPass()";
}

//===----------------------------------------------------------------------===//

#endif // FLIR_PASSES
