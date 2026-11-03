// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def ConvertROCDLFastMathOpsPass : Pass<"convert-rocdl-fastmath-ops", "gpu::GPUModuleOp"> {
  let summary = "Lower approximate math operations to native ROCDL operations";
  let description = [{
    Rewrites supported neutral math operations carrying the `afn` fast-math
    flag to their native ROCDL counterparts. This pass runs immediately before
    `convert-gpu-to-rocdl`, which would otherwise route those operations
    through device-library calls.
  }];
  let dependentDialects = [
    "ROCDL::ROCDLDialect",
    "vector::VectorDialect"
  ];
}

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

def ROCDLSerializeRegisterKernelsPass : Pass<"rocdl-serialize-register-kernels"> {
  let summary = "Serialize explicit register kernels with LLVM assembler validation";
  let description = [{
    Serializes AMDGPU modules containing FlyDSL register placement carriers.
    Uses the upstream ROCDL serializer and checks LLVM MC assembler errors
    before returning ISA or linking an object. Other GPU modules are left for
    the ordinary gpu-module-to-binary pass.
  }];
  let options = [
    Option<"format", "format", "std::string", "\"fatbin\"", "Output format: isa, bin, or fatbin">,
    Option<"toolkit", "toolkit", "std::string", "\"\"", "ROCm toolkit path">,
    Option<"cmdOptions", "opts", "std::string", "\"\"", "Target compilation options">,
    Option<"section", "section", "std::string", "\"\"", "ELF section name">,
    ListOption<"linkFiles", "link-files", "std::string", "Files to link">
  ];
  let dependentDialects = ["gpu::GPUDialect", "ROCDL::ROCDLDialect"];
}
