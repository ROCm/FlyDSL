// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --split-input-file | FileCheck %s

// FLYDSL_KTRACE_BLOCKS reaches the pass as a module attribute, published beside
// capacity_slots by the frontend. It gates TWO things, and the first is a
// correctness requirement rather than an optimisation: with a filter over a large
// grid, every wave would still claim its slot range and the cursor would run past
// the buffer even though only a handful ever record -- the claim itself overflows,
// which is an illegal access rather than a truncated trace.

// CHECK-LABEL: gpu.func @no_attr
// An absent attribute records every workgroup, so the claim is unguarded: the
// atomic is at the top level, not inside an scf.if yielding the base.
// CHECK-NOT: gpu.block_id
// CHECK: llvm.atomicrmw add
// CHECK-NOT: scf.if {{.*}} -> (i32)
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @no_attr(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// "all" is spelled differently but means the same as absent.
// CHECK-LABEL: gpu.func @all
// CHECK-NOT: gpu.block_id
// CHECK: llvm.atomicrmw add
// CHECK-NOT: scf.if {{.*}} -> (i32)
module attributes {fly_ktrace.blocks = "all", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @all(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// CHECK-LABEL: gpu.func @xcc
// The predicate is built in the prologue from the cached XCC_ID read...
// CHECK: llvm.mlir.constant(6164 : i32)
// CHECK: %[[XCC:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.getreg"
// CHECK: %[[WANT:.*]] = llvm.mlir.constant(3 : i32)
// CHECK: %[[REC:.*]] = arith.cmpi eq, %[[XCC]], %[[WANT]]
//
// ...and gates the claim, so a filtered-out wave performs no atomic at all.
// CHECK: scf.if %[[REC]] -> (i32) {
// CHECK:   llvm.atomicrmw add
//
// The same predicate is AND-ed into the leader guard: a filtered wave took the
// zero base above, so reaching a store would land it on a recording wave's range.
// CHECK: arith.andi {{.*}}, %[[REC]]
module attributes {fly_ktrace.blocks = "xcc:3", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @xcc(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// CHECK-LABEL: gpu.func @xyz
// Three workgroup coordinates, AND-ed. gpu.block_id is index-typed and lowered
// later by convert-gpu-to-rocdl, which runs after this pass.
// CHECK: %[[X:.*]] = gpu.block_id  x
// CHECK: arith.index_cast %[[X]]
// CHECK: %[[Y:.*]] = gpu.block_id  y
// CHECK: arith.index_cast %[[Y]]
// CHECK: %[[Z:.*]] = gpu.block_id  z
// CHECK: arith.index_cast %[[Z]]
// CHECK: scf.if {{.*}} -> (i32) {
// CHECK:   llvm.atomicrmw add
module attributes {fly_ktrace.blocks = "1,2,3", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @xyz(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
