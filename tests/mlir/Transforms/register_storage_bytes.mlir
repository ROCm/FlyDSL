// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-promote-regmem-to-vectorssa --canonicalize | FileCheck %s

// Byte-backed storage preserves aliasing through recast and offsets measured
// in the current pointer's element type. A word load spans both half stores.
// CHECK-LABEL: gpu.func @packed_alias
// CHECK-NOT: fly.make_ptr
// CHECK-NOT: fly.recast_iter
// CHECK: vector.shuffle
// CHECK: vector.shuffle
// CHECK: fly.register_value {{.*}}bitOffset = 32 : i64{{.*}}storageBits = 64 : i64
// CHECK: fly.ptr.store
gpu.module @bytes {
  gpu.func @packed_alias(%lo: i16, %hi: i16, %out: !fly.ptr<i32, global>) kernel {
    %p = fly.make_ptr() {dictAttrs = {allocSize = 8 : i64}} : () -> !fly.ptr<i8, register>
    fly.set_register %p {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 68 : i64} : !fly.ptr<i8, register>
    %four = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %q = fly.add_offset(%p, %four) : (!fly.ptr<i8, register>, !fly.int_tuple<4>) -> !fly.ptr<i8, register>
    %halves = fly.recast_iter(%q) : (!fly.ptr<i8, register>) -> !fly.ptr<i16, register>
    fly.ptr.store(%lo, %halves) : (i16, !fly.ptr<i16, register>) -> ()
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %upper = fly.add_offset(%halves, %one) : (!fly.ptr<i16, register>, !fly.int_tuple<1>) -> !fly.ptr<i16, register>
    fly.ptr.store(%hi, %upper) : (i16, !fly.ptr<i16, register>) -> ()
    %word = fly.recast_iter(%q) : (!fly.ptr<i8, register>) -> !fly.ptr<i32, register>
    %value = fly.ptr.load(%word) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%value, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }
}
