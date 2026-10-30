// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def FlyKtraceToROCDLConversionPass : Pass<"convert-fly-ktrace-to-rocdl"> {
  let summary = "Expand fly_ktrace annotations into their record-writing form";
  let description = [{
    Rewrites each `fly_ktrace.*` op into the timestamp read and the record stores
    that write a trace record, behind a prologue that claims the wave's slots from
    its partition, leaving behind only `llvm.*`, `rocdl.*`, `scf.*`, `arith.*` and
    `ub.poison` ops.
    Events carry no branch and no atomic: a wave's event count lives in a register,
    and stores the wave may not make are dropped by its buffer descriptor's bound.
    After every region that records, the wave agrees on its count, and an `scf.if`
    whose branches both record becomes two `scf.if`s on its condition, so the lanes
    of a divergent wave never share a slot.

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
    "gpu::GPUDialect",
    "ub::UBDialect"
  ];
}
