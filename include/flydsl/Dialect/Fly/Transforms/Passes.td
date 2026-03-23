// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_PASSES
#define FLY_PASSES

include "mlir/Pass/PassBase.td"

def FlyRewriteFuncSignaturePass : Pass<"fly-rewrite-func-signature"> {
  let summary = "Rewrite function signatures to use LLVM struct types";
  let description = [{
    This pass does the following:
    1. DSLTypes (IntTuple, Layout, ComposedLayout, CoordTensor, MemRef) are always lowered to
       packed LLVM struct types that contain only their dynamic components (static subcomponents
       are omitted). They are reconstructed in the function body using constructor ops
       (e.g. `MakeIntTupleOp`, `MakeLayoutOp`).
    2. Fully static DSL arguments (and static subcomponents of otherwise dynamic DSL arguments)
       are removed from function signatures and reconstructed via `fly.static` inside the
       function body.
    3. On the caller side (`gpu.launch_func`), DSL operands are packed into the corresponding
       struct values using only their dynamic components, and static operands are omitted.
  }];
  
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "gpu::GPUDialect",
    "LLVM::LLVMDialect"
  ];
}

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
    "vector::VectorDialect"
  ];
}

#endif // FLY_PASSES
