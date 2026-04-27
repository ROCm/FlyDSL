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

def FlyConvertAtomCallToSSAFormPass : Pass<"fly-convert-atom-call-to-ssa-form"> {
  let summary = "Convert copy_atom_call/mma_atom_call to SSA form";
  let description = [{
    Try to convert copy_atom_call and mma_atom_call to their SSA counterparts 
    (copy_atom_call_ssa / mma_atom_call_ssa) if possible.
    Promote register tensor to vector ssa value. Keep others memref unchanged.
  }];

  let dependentDialects = [
    "vector::VectorDialect"
  ];
}

def FlyIntSwizzleSimplifyPass : Pass<"fly-int-swizzle-simplify"> {
  let summary = "Algebraically simplify swizzle-shaped arith sequences";
  let description = [{
      Recognize the canonical swizzle sequence emitted by `applySwizzle`,

        %m  = arith.constant ((1<<M)-1) << (base+shift) : iN     // ring of 1s
        %s  = arith.constant shift                       : iN
        %a  = arith.andi %x, %m : iN
        %h  = arith.shrui %a, %s : iN
        %r  = arith.xori %x, %h : iN

      and apply the algebraic identity

        swizzle(a + d) = swizzle(a) + d   when d % period == 0,

      The peeled-off addends survive as plain `arith.addi`, exposing them to CSE and constant
      folding that the bit-level form would hide.
    }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect"
  ];
}

def FlyPromoteRegMemToVectorSSAPass : Pass<"fly-promote-regmem-to-vectorssa"> {
  let summary = "Promote register memory to vector SSA values";
  let description = [{
    Promotes fly.make_ptr(register) memory semantics to vector SSA values.
    Requires fly-convert-atom-call-to-ssa-form to have run first. 
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "ub::UBDialect"
  ];
}

#endif // FLY_PASSES
