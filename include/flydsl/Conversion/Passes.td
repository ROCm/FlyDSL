#ifndef FLYDSL_CONVERSION_PASSES
#define FLYDSL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FlyToROCDLConversionPass : Pass<"convert-fly-to-rocdl"> {
  let summary = "Lower Fly to MLIR upstream and rocdl dialects ";  
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
    "ROCDL::ROCDLDialect"
  ];
}

def FlyGpuStreamInjectPass : Pass<"fly-gpu-stream-inject", "::mlir::ModuleOp"> {
  let summary = "Inject external stream parameter into host functions for GPU kernel launches";
  let description = [{
    Adds an `!llvm.ptr` stream argument to each host `func.func` that contains
    `gpu.launch_func`, and wires it as the `asyncObject` operand.  The downstream
    translation phase (SelectObjectAttr) will then use this stream directly
    instead of calling mgpuStreamCreate / mgpuStreamSynchronize / mgpuStreamDestroy.
  }];
  let dependentDialects = [
    "gpu::GPUDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect"
  ];
}

#endif // FLYDSL_CONVERSION_PASSES
