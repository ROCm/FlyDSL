// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// Conversion-level guard for the add_offset canonicalization (issue #898).
//
// A runtime offset fused with a compile-time constant makes every access
// compute its own address, so the backend materializes one base register per
// access.  Keeping `add_offset(add_offset(ptr, dyn), static)` split lets a
// single runtime GEP feed many constant GEPs.
//
// Structural assertions only: no ISA register numbers are pinned here.

// === LDS: one runtime GEP shared by two constant tails ===
//
// allocBytes/allocAlign make this lower to a real relocatable @__shared_alloc_*
// LDS symbol, which is the shape the issue reproduces with.
// CHECK-LABEL: gpu.func @lds_shared_runtime_base
gpu.module @m_lds {
  gpu.func @lds_shared_runtime_base(%x: i32) kernel {
    %smem = fly.make_ptr() {dictAttrs = {allocBytes = 4096 : i64, allocAlign = 128 : i64}} : () -> !fly.ptr<bf16, shared>
    %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
    %c1 = fly.make_int_tuple() : () -> !fly.int_tuple<64>
    %c2 = fly.make_int_tuple() : () -> !fly.int_tuple<128>
    // The runtime index must reach the GEP unmodified: no `addi %x, const`.
    // CHECK: %[[SYM:.*]] = llvm.mlir.addressof @__shared_alloc_{{[0-9]+}} : !llvm.ptr<3>
    // CHECK-NOT: arith.addi
    // CHECK: %[[BASE:.*]] = llvm.getelementptr %[[SYM]][%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    // CHECK: %[[C1:.*]] = arith.constant 64 : i32
    // CHECK: %[[PA:.*]] = llvm.getelementptr %[[BASE]][%[[C1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    // CHECK: %[[C2:.*]] = arith.constant 128 : i32
    // CHECK: %[[PB:.*]] = llvm.getelementptr %[[BASE]][%[[C2]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    // CHECK: llvm.load %[[PA]]
    // CHECK: llvm.load %[[PB]]
    %p0 = fly.add_offset(%smem, %d) : (!fly.ptr<bf16, shared>, !fly.int_tuple<?>) -> !fly.ptr<bf16, shared>
    %pa = fly.add_offset(%p0, %c1) : (!fly.ptr<bf16, shared>, !fly.int_tuple<64>) -> !fly.ptr<bf16, shared>
    %pb = fly.add_offset(%p0, %c2) : (!fly.ptr<bf16, shared>, !fly.int_tuple<128>) -> !fly.ptr<bf16, shared>
    %a = fly.ptr.load(%pa) : (!fly.ptr<bf16, shared>) -> bf16
    %b = fly.ptr.load(%pb) : (!fly.ptr<bf16, shared>) -> bf16
    gpu.return
  }
}

// -----

// === Swizzled LDS: applicability boundary ===
//
// A non-trivial swizzle is applied at the load, to the FINAL address, so the
// static tail is folded into the XOR and cannot become a ds_read immediate.
// The canonicalization is neutral here, not harmful: the runtime base is still
// a single GEP and no runtime+constant fusion happens.  This test pins that
// "neutral, still correct" behaviour so the boundary stays visible.
// CHECK-LABEL: gpu.func @lds_swizzled_neutral
gpu.module @m_swz {
  gpu.func @lds_swizzled_neutral(%x: i32) kernel {
    %smem = fly.make_ptr() {dictAttrs = {allocBytes = 4096 : i64, allocAlign = 128 : i64}} : () -> !fly.ptr<bf16, shared, S<3,3,3>>
    %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
    %c1 = fly.make_int_tuple() : () -> !fly.int_tuple<64>
    // CHECK: %[[SYM:.*]] = llvm.mlir.addressof @__shared_alloc_{{[0-9]+}} : !llvm.ptr<3>
    // CHECK: %[[BASE:.*]] = llvm.getelementptr %[[SYM]][%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    // CHECK: %[[C1:.*]] = arith.constant 64 : i32
    // CHECK: %[[PA:.*]] = llvm.getelementptr %[[BASE]][%[[C1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
    // The swizzle consumes the whole address, constant tail included.
    // CHECK: %[[I:.*]] = llvm.ptrtoint %[[PA]] : !llvm.ptr<3> to i64
    // CHECK: arith.xori %[[I]]
    %p0 = fly.add_offset(%smem, %d) : (!fly.ptr<bf16, shared, S<3,3,3>>, !fly.int_tuple<?>) -> !fly.ptr<bf16, shared, S<3,3,3>>
    %pa = fly.add_offset(%p0, %c1) : (!fly.ptr<bf16, shared, S<3,3,3>>, !fly.int_tuple<64>) -> !fly.ptr<bf16, shared, S<3,3,3>>
    %a = fly.ptr.load(%pa) : (!fly.ptr<bf16, shared, S<3,3,3>>) -> bf16
    gpu.return
  }
}

// -----

// === Buffer descriptor: separate offset adds, no fusion ===
//
// Buffer pointers add into the fat-pointer offset field instead of emitting a
// GEP.  The runtime and the constant must stay two separate adds so the backend
// can still fold the constant into the buffer instruction's immediate offset.
// CHECK-LABEL: @buffer_runtime_then_static
func.func @buffer_runtime_then_static(%ptr: !fly.ptr<f32, #fly_rocdl.buffer_desc>, %x: i32) -> f32 {
  %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  %c1 = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  // Runtime offset added on its own...
  // CHECK: %[[O0:.*]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: arith.addi %[[O0]], %{{.*}} : i32
  // ...then the constant added separately, never merged into the runtime add.
  // CHECK: %[[C1:.*]] = arith.constant 64 : i32
  // CHECK: %[[O1:.*]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: arith.addi %[[O1]], %[[C1]] : i32
  %p0 = fly.add_offset(%ptr, %d) : (!fly.ptr<f32, #fly_rocdl.buffer_desc>, !fly.int_tuple<?>) -> !fly.ptr<f32, #fly_rocdl.buffer_desc>
  %pa = fly.add_offset(%p0, %c1) : (!fly.ptr<f32, #fly_rocdl.buffer_desc>, !fly.int_tuple<64>) -> !fly.ptr<f32, #fly_rocdl.buffer_desc>
  %a = fly.ptr.load(%pa) : (!fly.ptr<f32, #fly_rocdl.buffer_desc>) -> f32
  return %a : f32
}
