// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

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

def FlyROCDLClusterAttrPass : Pass<"fly-rocdl-cluster-attr"> {
  let summary = "Inject amdgpu-cluster-dims into llvm.func passthrough";
  let description = [{
    Converts the `rocdl.cluster_dims` discardable attribute on `llvm.func`
    into an LLVM IR `passthrough` entry containing `amdgpu-cluster-dims`.

    This is a hack to work around the upstream ROCDL dialect not supporting
    cluster_dims translation.  Run this pass inside `gpu.module(...)` after
    `convert-gpu-to-rocdl` so that `llvm.func` ops already exist.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
}

def FlyROCDLTagAMDGPUCodegenPassesPass
    : Pass<"fly-rocdl-tag-amdgpu-codegen-passes"> {
  let summary = "Mark gpu.module to opt into FlyDSL AMDGPU codegen passes";
  let description = [{
    Attaches a `fly.amdgpu_codegen_passes` DictionaryAttr on each
    `gpu.module` with a UnitAttr entry per enabled FlyDSL-owned AMDGPU
    MachineFunctionPass.  FlyDSL's `gpu::TargetAttrInterface` impl on
    `#rocdl.target` reads this attribute and only injects the named
    passes into the AMDGPU codegen pipeline; modules without the
    attribute fall back to the stock upstream pipeline.

    Run inside `gpu.module(...)` so `getOperation()` is the gpu.module.
    The pass is a no-op when no option is set, so it's safe to add to
    the pipeline unconditionally and toggle via options.
  }];
  let options = [
    Option<"preferAGPRForDSRead", "prefer-agpr-for-ds-read", "bool",
           /*default=*/"false",
           "Inject FlyAMDGPUPreferAGPRForDSRead at PreRegAlloc.">,
    Option<"mfmaTieVDSTToSrc2", "mfma-tie-vdst-to-src2", "bool",
           /*default=*/"false",
           "Inject FlyAMDGPUMFMATieVDSTToSrc2 at PreRegAlloc.">,
  ];
}
