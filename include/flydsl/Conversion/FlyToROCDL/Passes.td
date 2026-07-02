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

def FlyROCDLArgInRegPass : Pass<"fly-rocdl-arg-inreg"> {
  let summary = "Mark kernel arguments as inreg for SGPR preloading";
  let description = [{
    Adds the `llvm.inreg` attribute to all arguments of `llvm.func` ops
    that carry the `fly.arg_inreg` marker attribute, hinting the firmware
    to preload kernel arguments into user SGPRs.

    Kernels opt in by attaching `fly.arg_inreg` as a unit attribute on
    the `gpu.func` (which `convert-gpu-to-rocdl` copies to `llvm.func`).
    The marker is consumed (removed) by this pass.

    Run inside `gpu.module(...)` after `convert-gpu-to-rocdl`.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
}
