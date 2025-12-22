//===- RocirPasses.td - Layout IR Lowering Passes ----------*- tablegen -*-===//
//
// Lowering passes for Rocir IR dialects
// Transformation chain: rocdsl → gpu/scf → rocdl → LLVM IR → AMDGCN
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_PASSES
#define ROCIR_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// rocdsl → Standard Dialects (scf, arith, memref)
//===----------------------------------------------------------------------===//

// Run at module scope so we can lower Rocir ops inside GPU kernels (`gpu.func`)
// as well as regular `func.func`.
def RocirToStandardPass : Pass<"rocir-to-standard", "::mlir::ModuleOp"> {
  let summary = "Lower Rocir IR to standard MLIR dialects";
  let description = [{
    Converts rocdsl operations to combinations of scf, arith, and memref.
  }];
  
  let constructor = "mlir::rocir::createRocirToStandardPass()";
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
def RocirTrivialDCEPass : Pass<"trivial-dce", "::mlir::ModuleOp"> {
  let summary = "Erase trivially-dead operations (general DCE)";
  let description = [{
    Iteratively erases operations that are unused and have no side effects, as
    determined by MLIR's side-effect interfaces.
  }];
  let constructor = "mlir::rocir::createRocirTrivialDCEPass()";
}

//===----------------------------------------------------------------------===//

#endif // ROCIR_PASSES
