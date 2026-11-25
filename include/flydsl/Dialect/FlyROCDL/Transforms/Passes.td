// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYROCDL_PASSES
#define FLYROCDL_PASSES

include "mlir/Pass/PassBase.td"

def FlyROCDLExpandOpsPass : Pass<"fly-rocdl-expand-ops"> {
  let summary = "Expand fly_rocdl target specific ops into the ops";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "mlir::fly::FlyDialect"
  ];
}

def ROCDLInvariantCodeMotionPass : Pass<"rocdl-invariant-code-motion"> {
  let summary = "Loop-invariant code motion that also moves AMDGPU lane intrinsics and M0 reads";
  let description = [{
    Moves loop-invariant ops out of loops, as `loop-invariant-code-motion`
    does, and in addition moves the convergent AMDGPU intrinsics without
    memory effects, as ROCDL ops or `llvm.call_intrinsic`, by what the
    hardware does with them. Generic code motion leaves those in place:

    - `readlane`, `writelane`, `inverse.ballot`, `s.bitreplicate`,
      `s.quadmask` and `s.wqm` read or write the lane they name, or work on
      scalars, whichever lanes are active. They move out of any loop when
      their scalar operands are the same in every lane; those are otherwise
      read from the first active lane.
    - The others (`readfirstlane`, `ballot`, `icmp`, `fcmp`, the DPP,
      permlane, swizzle, permute and wave reduction intrinsics,
      `set.inactive` and the whole-wave intrinsics) read the active lanes.
      They move out of an `scf.for` whose bounds and step are the same in
      every lane: such a loop runs the same iterations in every lane, so the
      top level of its body runs with the lanes active before the loop.

    An LDS DMA (`load.to.lds`, `global.load.lds`, the buffer `load.lds`
    intrinsics and their async forms) reads its LDS address from the first
    active lane into M0. In such a loop, a loop-invariant LDS address at the
    top level of the body is read once with `rocdl.readfirstlane` before the
    loop instead.

    Run it on the gpu.module before scf.for is converted to control flow.
  }];
}

def FlyROCDLClusterAttrPass : Pass<"fly-rocdl-cluster-attr"> {
  let summary = "Inject amdgpu-cluster-dims into llvm.func passthrough";
  let description = [{
    Converts the `rocdl.cluster_dims` discardable attribute on `llvm.func`
    into an LLVM IR `passthrough` entry containing `amdgpu-cluster-dims`.

    This is a hack to work around the upstream ROCDL dialect not supporting
    cluster_dims translation.  Run this pass inside `gpu.module(...)` after
    `convert-gpu-to-rocdl` so that `llvm.func` ops already exist -- it rewrites
    an attribute on those, and so is not part of the Fly -> ROCDL conversion.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
}

#endif // FLYROCDL_PASSES
