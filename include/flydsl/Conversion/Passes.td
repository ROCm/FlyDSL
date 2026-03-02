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

def FlyGpuToLLVMPass : Pass<"fly-gpu-to-llvm", "::mlir::ModuleOp"> {
  let summary = "GPU to LLVM conversion that preserves asyncObject on gpu.launch_func";
  let description = [{
    Drop-in replacement for the upstream `gpu-to-llvm` pass.  Registers a
    higher-priority pattern for `gpu.launch_func` that preserves the
    `asyncObject` operand through the conversion.  The upstream pattern
    ignores `asyncObject` and always creates/destroys its own stream,
    which breaks CUDA/HIP graph capture.

    All other conversion patterns are identical to the upstream pass.
  }];
  let options = [
    Option<"hostBarePtrCallConv", "use-bare-pointers-for-host", "bool",
           /*default=*/"false",
           "Use bare pointers to pass memref arguments to host functions.">,
    Option<"kernelBarePtrCallConv", "use-bare-pointers-for-kernels", "bool",
           /*default=*/"false",
           "Use bare pointers to pass memref arguments to kernels.">,
    Option<"kernelIntersperseSizeCallConv", "intersperse-sizes-for-kernels",
           "bool", /*default=*/"false",
           "Insert size argument after each memref argument.">
  ];
  let dependentDialects = [
    "gpu::GPUDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "memref::MemRefDialect"
  ];
}

#endif // FLYDSL_CONVERSION_PASSES
