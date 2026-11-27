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

    Lowers the float conversions (`arith.truncf`, `extf`, `sitofp`, `uitofp`,
    `fptosi`, `fptoui`) for the chip of the module's single `#rocdl.target`,
    and leaves modules without one alone. Conversions the chip does with one
    native instruction stay for `convert-gpu-to-rocdl`; the others become the
    chip's instructions or integer sequences, which honour the `roundingmode`
    of `truncf` and the `nnan` / `afn` fast-math flags, also implied by the
    target flags `fast`, `finite_only` and `unsafe_math`:
    - to bf16 without `v_cvt_pk_bf16_f32`, or in a mode other than to nearest
      even: integer rounding, a NaN giving 0x7fc0 unless `nnan`; values bf16
      holds exactly (integers and floats of at most 8 bits) are truncated;
    - to f16 toward zero: `v_cvt_pkrtz_f16_f32`; to f16 or f32 in the other
      modes: the native rounding to nearest even, moved by one step where the
      mode rounds the other way;
    - f64 to f16 / bf16 / fp8: round to odd in f32 first, so rounding twice
      stays exact; with `afn`, plain rounding through f32;
    - fp8 / bf8 to and from the formats the chip converts natively (FNUZ on
      gfx94x, OCP on gfx950 and gfx12+): the packed conversion instructions,
      through f32 for other types; only to nearest even.
    Other narrow float types and fp8 / bf8 in other modes are errors.
  }];
  let dependentDialects = ["arith::ArithDialect", "LLVM::LLVMDialect", "math::MathDialect",
                           "ROCDL::ROCDLDialect", "vector::VectorDialect"];
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
