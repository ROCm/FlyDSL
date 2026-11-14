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

    It also expands `math.log1p` and `math.expm1`, whatever their fast-math
    flags, which `convert-gpu-to-rocdl` lowers to `log(1 + x)` and
    `exp(x) - 1`, both 0 for tiny `x`: `log1p` becomes a compensated
    `log(1 + x)` on the native `log`, and `expm1` a call to OCML's
    `__ocml_expm1_*`.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
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
