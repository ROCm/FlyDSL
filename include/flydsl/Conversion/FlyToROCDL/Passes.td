// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def ConvertROCDLMathOpsPass : Pass<"convert-rocdl-math-ops", "gpu::GPUModuleOp"> {
  let summary = "Lower math operations that convert-gpu-to-rocdl would get wrong or slow";
  let description = [{
    Runs immediately before `convert-gpu-to-rocdl`.

    Rewrites supported neutral math operations carrying the `afn` fast-math
    flag to their native ROCDL counterparts, which `convert-gpu-to-rocdl`
    would otherwise route through device-library calls.

    Lowers to OCML calls the math operations that `convert-gpu-to-rocdl` could
    also lower with MathToLLVM, whose forms are wrong on AMDGPU: `tan`, `asin`,
    `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `expm1` and the remaining
    `log1p` always, `rsqrt` without `afn`, `fpowi` unless it is scalar with a
    constant exponent. Which lowering `convert-gpu-to-rocdl` picks for them
    depends on the dialects loaded in the context.
  }];
  let dependentDialects = ["arith::ArithDialect", "LLVM::LLVMDialect", "ROCDL::ROCDLDialect",
                           "vector::VectorDialect"];
}

def FlyToROCDLConversionPass : Pass<"convert-fly-to-rocdl"> {
  let summary = "Lower Fly to MLIR upstream and rocdl dialects ";
  let dependentDialects = ["arith::ArithDialect", "scf::SCFDialect", "vector::VectorDialect",
                           "LLVM::LLVMDialect", "ROCDL::ROCDLDialect"];
}

def FlyEmitGPUBinaryPass : Pass<"fly-emit-gpu-binary", ""> {
  let summary = "Transform a GPU module into a GPU binary without an external linker";
  let description = [{
    Wrapper around the upstream `gpu-module-to-binary` pass that stops it at
    assembly (`format=isa`) and performs the remaining assemble and link steps
    in process: `mlir::ROCDL::assembleIsa` for the AMDGPU MC assembler, and the
    LLD ELF driver library for the link.

    Upstream instead spawns `<toolkit>/llvm/bin/ld.lld`, where `<toolkit>` comes
    from `ROCM_PATH` or from a path baked into the LLVM build.  That lookup
    fails on any container that installs ROCm somewhere else, and reports only
    `lld invocation failed`.  Linking against the LLD libraries removes the
    lookup entirely and pins the linker to the LLVM revision that produced the
    ISA.

    Options mirror `gpu-module-to-binary`, minus `format`.  Objects that are
    not ROCDL targets are left untouched, and builds whose LLVM has no LLD
    libraries fall back to running `gpu-module-to-binary` unmodified.
  }];
  let options = [Option<"toolkitPath", "toolkit", "std::string", [{""}], "Toolkit path.">,
                 ListOption<"linkFiles", "l", "std::string", "Extra files to link to.">,
                 Option<"cmdOptions", "opts", "std::string", [{""}],
                        "Command line options to pass to the tools.">,
                 Option<"elfSection", "section", "std::string", [{""}],
                        "ELF section where binary is to be located.">];
  let dependentDialects = ["gpu::GPUDialect"];
}
