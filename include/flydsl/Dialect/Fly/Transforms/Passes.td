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

def FlyCheckResourceLimitsPass : Pass<"fly-check-resource-limits", "gpu::GPUModuleOp"> {
  let summary = "Reject kernels whose static resource usage exceeds the target's limits";
  let description = [{
    For every kernel in the `gpu.module`, finds per resource the most that the
    allocations made in the kernel body hold at once, from the allocations and frees its
    ops report through `ResourceUsageOpInterface`, and fails when that exceeds the limit
    one of the module's targets reports through `GPUTargetInfoAttrInterface`. Ops are
    taken in the order they appear, as in a linear scan: an allocation is held until the
    op freeing it or the end of the kernel, so allocations never freed add up wherever
    they are. A resource with a free the pass cannot match to an earlier allocation is
    not checked.

    The pass names no resource. As with address spaces, a resource is an attribute:
    generic ones such as `#fly.kernel_resource<shared_memory>` come from Fly,
    target-specific ones from the backend dialect, and a resource a target does not know
    is not checked. Shared memory is allocated by shared `fly.make_ptr` (`allocBytes`)
    and never freed.

    Usage that is only known at run time, such as the dynamic shared memory of a launch,
    is not counted.
  }];
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
