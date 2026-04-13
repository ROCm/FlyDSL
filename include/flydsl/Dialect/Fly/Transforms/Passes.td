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

  let options = [
    Option<"enableRegAllocaForwarding", "enable-reg-alloca-forwarding",
           "bool", /*default=*/"true",
           "Enable register-alloca store forwarding and loop-carried promotion">
  ];
}

def FlyForwardLLVMAllocasPass : Pass<"fly-forward-llvm-allocas"> {
  let summary = "Forward LLVM AS5 alloca stores to loads and promote to loop-carried";
  let description = [{
    After convert-fly-to-rocdl, CopyAtomCall and MmaAtomCallValue ops have been
    expanded into explicit LLVM::StoreOp and LLVM::LoadOp sequences on AS5
    (private scratch) allocas.  This pass:
    1. Forwards LLVM::StoreOp values to subsequent LLVM::LoadOp from the same
       alloca+offset within each basic block (intra-iteration forwarding).
    2. Promotes cross-iteration alloca load/store patterns in scf.for loops
       to loop-carried SSA values (phi nodes).
    3. Eliminates dead AS5 allocas that have no remaining loads.

    This must run AFTER convert-fly-to-rocdl and BEFORE convert-scf-to-cf,
    since it relies on scf.for structure for loop-carried promotion.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "scf::SCFDialect"
  ];

  let options = [
    Option<"enabled", "enabled",
           "bool", /*default=*/"true",
           "Enable the pass (set to false to skip all transformations)">
  ];
}

#endif // FLY_PASSES
