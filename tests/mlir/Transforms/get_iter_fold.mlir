// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --canonicalize | FileCheck %s
// RUN: %fly-opt %s --canonicalize --fly-convert-atom-call-to-ssa-form --fly-promote-regmem-to-vectorssa | FileCheck %s --check-prefix=PROMOTE

// A register fragment stored under a dynamic scf.if and yielded out of it.
// Canonicalize drops the scf.if, which yields the same view on every branch,
// and leaves get_iter(make_view(%ptr)); get_iter must fold to %ptr so the
// fragment still promotes to vector SSA.

// CHECK-LABEL: gpu.func @fragment_yielded_from_if
// CHECK: %[[PTR:.*]] = fly.make_ptr() {{.*}} -> !fly.ptr<i32, register>
// CHECK-NOT: fly.get_iter
// CHECK: %[[VIEW:.*]] = fly.make_view(%[[PTR]],
// CHECK-NOT: fly.get_iter
// CHECK: fly.copy_atom_call(%{{.*}}, %[[VIEW]], %{{.*}})

// PROMOTE-LABEL: gpu.func @fragment_yielded_from_if
// PROMOTE-NOT: register
// PROMOTE: %[[V:.*]] = scf.if %{{.*}} -> (vector<4xi32>)
// PROMOTE: %[[SRC:.*]] = vector.extract_strided_slice %[[V]]
// PROMOTE: fly.copy_atom_call_ssa(%{{.*}}, %[[SRC]], %{{.*}})
gpu.module @get_iter_fold {
  gpu.func @fragment_yielded_from_if(%out: !fly.ptr<i32, global>, %cond: i1) kernel {
    %a = arith.constant dense<1> : vector<4xi32>
    %b = arith.constant dense<-1> : vector<4xi32>
    %atom = fly.static : !fly.copy_atom<!fly.universal_copy<128>, 32>
    %layout = fly.static : !fly.layout<4:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 4 : i64}} : () -> !fly.ptr<i32, register>
    %frag = fly.make_view(%ptr, %layout) : (!fly.ptr<i32, register>, !fly.layout<4:1>) -> !fly.memref<i32, register, 4:1>
    %res = scf.if %cond -> (!fly.memref<i32, register, 4:1>) {
      fly.ptr.store(%a, %ptr) : (vector<4xi32>, !fly.ptr<i32, register>) -> ()
      scf.yield %frag : !fly.memref<i32, register, 4:1>
    } else {
      fly.ptr.store(%b, %ptr) : (vector<4xi32>, !fly.ptr<i32, register>) -> ()
      scf.yield %frag : !fly.memref<i32, register, 4:1>
    }
    %iter = fly.get_iter(%res) : (!fly.memref<i32, register, 4:1>) -> !fly.ptr<i32, register>
    %src = fly.make_view(%iter, %layout) : (!fly.ptr<i32, register>, !fly.layout<4:1>) -> !fly.memref<i32, register, 4:1>
    %dst = fly.make_view(%out, %layout) : (!fly.ptr<i32, global>, !fly.layout<4:1>) -> !fly.memref<i32, global, 4:1>
    fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly.universal_copy<128>, 32>, !fly.memref<i32, register, 4:1>, !fly.memref<i32, global, 4:1>) -> ()
    gpu.return
  }
}
