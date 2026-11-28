// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def FlyToNVVMConversionPass : Pass<"convert-fly-to-nvvm"> {
  let summary = "Lower Fly to MLIR upstream and nvvm dialects ";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
    "NVVM::NVVMDialect"
  ];
}
