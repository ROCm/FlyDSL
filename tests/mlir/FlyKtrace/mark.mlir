// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s

// The trailing argument is a !fly.ptr, which is what the frontend actually
// appends (a PointerJitArg); the LLVM atomics and stores below need it converted
// first. Hand-writing !llvm.ptr here would test a signature the frontend never
// produces.

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @probe
    gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
      // CHECK: fly.to_llvm_ptr

      // Prologue, once per wave, at the entry block so its values dominate uses
      // inside loops. 63492 = HW_ID (id 4, size 32); 6164 = XCC_ID (id 20, size 4).
      // CHECK: llvm.mlir.constant(63492 : i32)
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.getreg"
      // CHECK: llvm.mlir.constant(6164 : i32)
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.getreg"
      // The claim is confined to ONE lane and broadcast. Unguarded, it runs on all
      // 64 lanes, so a wave claims 64 ranges instead of one: the cursor advances 64x
      // too fast, the buffer is exhausted 64x early, and waves claiming past the end
      // store out of bounds -- an illegal access on device, not a truncated trace.
      // CHECK: rocdl.ballot
      // CHECK: scf.if
      // CHECK: llvm.atomicrmw add
      // CHECK: rocdl.readfirstlane

      // The record is written by the first ACTIVE lane, not lane 0.
      // CHECK: rocdl.ballot
      // CHECK: llvm.intr.cttz
      // CHECK: arith.cmpi eq
      // CHECK: scf.if

      // Both bounds compares are UNSIGNED: the counters come from an unsaturated
      // i32 atomic add, and a signed compare inverts past 2^31.
      // CHECK: arith.cmpi ult
      // CHECK: arith.cmpi ult
      // CHECK: scf.if

      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK-COUNT-6: llvm.store
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
