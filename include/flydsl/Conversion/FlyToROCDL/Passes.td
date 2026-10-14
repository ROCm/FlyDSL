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

def FlyPinRegisterReservePass : Pass<"fly-pin-register-reserve"> {
  let summary = "Emit amdgpu-reserved-{v,s,a}gprs passthrough from pinned register markers";
  let description = [{
    Reads the `fly.pinned_vgprs` / `fly.pinned_sgprs` / `fly.pinned_agprs`
    discardable attributes set by `PinRegisterLowering` and converts them
    into `amdgpu-reserved-{v,s,a}gprs` LLVM passthrough entries on `llvm.func`.
    Run inside `gpu.module(...)` after `convert-gpu-to-rocdl`.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
}
