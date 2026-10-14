// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def FlyKtraceToROCDLConversionPass : Pass<"convert-fly-ktrace-to-rocdl"> {
  let summary = "Expand fly_ktrace annotations into their record-writing form";
  let description = [{
    Rewrites each `fly_ktrace.*` op into the slot claim, leader guard, timestamp
    read and record stores that actually write a trace record, leaving behind only
    `llvm.*`, `scf.*` and `arith.*` ops.

    Runs before `convert-fly-to-rocdl` so it never has to reason about Fly's atom
    forms, and after the kernel signature is fixed -- the trailing trace-buffer
    parameter is added during tracing (`kernel_function.py`), not here, because a
    pass cannot add a kernel parameter without also rewriting every
    `gpu.launch_func` and the host-side ABI packing that matches it.

    A kernel with no `fly_ktrace` ops is left untouched.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "LLVM::LLVMDialect",
    "ROCDL::ROCDLDialect",
    "gpu::GPUDialect"
  ];
}
