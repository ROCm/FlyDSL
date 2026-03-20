// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_PASSES
#define FLY_PASSES

include "mlir/Pass/PassBase.td"

def FlyCanonicalizePass : Pass<"fly-canonicalize"> {
  let summary = "Canonicalize Pattern";
  let description = [{
    Canonicalize Fly operations.
  }];
}

def FlyLayoutLoweringPass : Pass<"fly-layout-lowering"> {
  let summary = "Lower layout algebra operations";
  let description = [{ 
    Lowers layout algebra operations to simpler forms.
  }];
  
  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "LLVM::LLVMDialect",
    "vector::VectorDialect"
  ];
}

#endif // FLY_PASSES
