// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def FlyRocmModuleToBinaryPass : Pass<"fly-rocm-module-to-binary"> {
  let summary = "Serialize ROCm GPU modules with an explicitly selected linker";
  let dependentDialects = ["gpu::GPUDialect", "ROCDL::ROCDLDialect"];
  let options = [
    Option<"toolkitPath", "toolkit", "std::string", [{""}], "Root containing amdgcn/bitcode">,
    Option<"linkerPath", "linker", "std::string", [{""}], "Path to the ld.lld executable">,
    Option<"compilationTarget", "format", "std::string", [{"fatbin"}], "Output representation">,
    ListOption<"linkFiles", "l", "std::string", "Extra device bitcode files to link">,
    Option<"cmdOptions", "opts", "std::string", [{""}], "Target command options">,
    Option<"elfSection", "section", "std::string", [{""}], "ELF section for the binary">,
    ListOption<"objectFiles", "object-files", "std::string",
               "External relocatable GPU objects, one per ROCm target in module order">
  ];
}

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
