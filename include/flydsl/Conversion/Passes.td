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

def FlyGpuStreamMarkPass : Pass<"fly-gpu-stream-mark", "::mlir::ModuleOp"> {
  let summary = "Mark stream argument indices before gpu-to-llvm drops asyncObject";
  let description = [{
    Scans host functions for `gpu.launch_func` ops that carry an `asyncObject`
    operand.  Records the corresponding function-argument index as a
    `fly.stream_arg_index` attribute on the enclosing function so that the
    downstream `fly-gpu-stream-inject` pass can re-wire it precisely after
    `gpu-to-llvm` has dropped the `asyncObject`.
  }];
  let dependentDialects = [
    "gpu::GPUDialect",
    "func::FuncDialect"
  ];
}

def FlyGpuStreamInjectPass : Pass<"fly-gpu-stream-inject", "::mlir::ModuleOp"> {
  let summary = "Inject external stream parameter into host functions for GPU kernel launches";
  let description = [{
    Re-wires the stream argument (identified by `fly.stream_arg_index` attribute
    set by the earlier `fly-gpu-stream-mark` pass) as the `asyncObject` operand
    on each `gpu.launch_func`.  If no attribute is present (i.e. the function had
    no stream parameter), a new `!llvm.ptr` argument is appended instead.

    The downstream translation phase (SelectObjectAttr) will then use this
    stream directly instead of calling mgpuStreamCreate / mgpuStreamSynchronize
    / mgpuStreamDestroy.
  }];
  let dependentDialects = [
    "gpu::GPUDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect"
  ];
}

#endif // FLYDSL_CONVERSION_PASSES
