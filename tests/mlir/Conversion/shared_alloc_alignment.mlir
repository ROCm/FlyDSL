// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// `fly.make_ptr` in the shared address space carries the LDS sub-allocation's size and
// alignment in dictAttrs; FlyToROCDL turns each one into its own `@__shared_alloc_*`
// global and must propagate `allocAlign` verbatim onto that global.
//
// This is the contract SharedAllocator depends on for `allocate(..., alignment=N)`:
// the alignment on the global is what lets the backend's LoadStoreVectorizer prove a
// run of element loads is wide enough to merge into ds_read_b128. Under-reporting it
// here silently degrades those reads into narrower pairs, so pin it.

// === alignment is carried through verbatim, not derived from the element type ===

// CHECK-LABEL: gpu.module @m_align16
// CHECK: llvm.mlir.global external @__shared_alloc_{{[0-9]+}}() {addr_space = 3 : i32, alignment = 16 : i64, dso_local}
gpu.module @m_align16 {
  gpu.func @lds_align16() kernel {
    // CHECK-LABEL: gpu.func @lds_align16
    // CHECK: llvm.mlir.addressof @__shared_alloc_{{[0-9]+}} : !llvm.ptr<3>
    %smem = fly.make_ptr() {dictAttrs = {allocBytes = 65536 : i64, allocAlign = 16 : i64}} : () -> !fly.ptr<f32, shared>
    %c0 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
    %p = fly.add_offset(%smem, %c0) : (!fly.ptr<f32, shared>, !fly.int_tuple<0>) -> !fly.ptr<f32, shared>
    %v = fly.ptr.load(%p) : (!fly.ptr<f32, shared>) -> f32
    gpu.return
  }
}

// -----

// === a larger request is not clamped to the element alignment ===

// CHECK-LABEL: gpu.module @m_align128
// CHECK: llvm.mlir.global external @__shared_alloc_{{[0-9]+}}() {addr_space = 3 : i32, alignment = 128 : i64, dso_local}
gpu.module @m_align128 {
  gpu.func @lds_align128() kernel {
    // f16 has a 2-byte natural alignment; the 128 in dictAttrs must win.
    %smem = fly.make_ptr() {dictAttrs = {allocBytes = 4096 : i64, allocAlign = 128 : i64}} : () -> !fly.ptr<f16, shared>
    %c0 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
    %p = fly.add_offset(%smem, %c0) : (!fly.ptr<f16, shared>, !fly.int_tuple<0>) -> !fly.ptr<f16, shared>
    %v = fly.ptr.load(%p) : (!fly.ptr<f16, shared>) -> f16
    gpu.return
  }
}

// -----

// === each make_ptr gets its own global, each keeping its own alignment ===

// CHECK-LABEL: gpu.module @m_align_mixed
// CHECK-DAG: llvm.mlir.global external @__shared_alloc_{{[0-9]+}}() {addr_space = 3 : i32, alignment = 2 : i64, dso_local}
// CHECK-DAG: llvm.mlir.global external @__shared_alloc_{{[0-9]+}}() {addr_space = 3 : i32, alignment = 32 : i64, dso_local}
gpu.module @m_align_mixed {
  gpu.func @lds_align_mixed() kernel {
    %a = fly.make_ptr() {dictAttrs = {allocBytes = 256 : i64, allocAlign = 2 : i64}} : () -> !fly.ptr<f16, shared>
    %b = fly.make_ptr() {dictAttrs = {allocBytes = 256 : i64, allocAlign = 32 : i64}} : () -> !fly.ptr<f16, shared>
    %c0 = fly.make_int_tuple() : () -> !fly.int_tuple<0>
    %pa = fly.add_offset(%a, %c0) : (!fly.ptr<f16, shared>, !fly.int_tuple<0>) -> !fly.ptr<f16, shared>
    %pb = fly.add_offset(%b, %c0) : (!fly.ptr<f16, shared>, !fly.int_tuple<0>) -> !fly.ptr<f16, shared>
    %va = fly.ptr.load(%pa) : (!fly.ptr<f16, shared>) -> f16
    %vb = fly.ptr.load(%pb) : (!fly.ptr<f16, shared>) -> f16
    gpu.return
  }
}
