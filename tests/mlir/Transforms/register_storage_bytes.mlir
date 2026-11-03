// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-promote-regmem-to-vectorssa --canonicalize --verify-diagnostics | FileCheck %s

// Byte-backed storage preserves aliasing through recast and offsets measured
// in the current pointer's element type. A word load spans both half stores.
// CHECK-LABEL: gpu.func @packed_alias
// CHECK-NOT: fly.make_ptr
// CHECK-NOT: fly.recast_iter
// CHECK: vector.shuffle
// CHECK: vector.shuffle
// CHECK: fly.register_value({{.*}}bitOffset = 32 : i32{{.*}}storageBits = 64 : i32
// CHECK: fly.ptr.store
gpu.module @bytes {
  gpu.func @packed_alias(%lo: i16, %hi: i16, %out: !fly.ptr<i32, global>) kernel {
    %p = fly.make_ptr() {dictAttrs = {allocSize = 8 : i64}} : () -> !fly.ptr<i8, register>
    fly.register_place(%p) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>, startIdx = 68 : i32} : (!fly.ptr<i8, register>) -> ()
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

// -----

// Unplaced byte storage with a sub-byte access cannot be modeled in byte
// slots. It stays in memory form, as recast-reachable storage always did.
// CHECK-LABEL: gpu.func @nibble
// CHECK: fly.make_ptr
// CHECK: fly.recast_iter
// CHECK: fly.ptr.store
gpu.module @m {
  gpu.func @nibble(%x: i4, %out: !fly.ptr<i8, global>) kernel {
    %p = fly.make_ptr() {dictAttrs = {allocSize = 4 : i64}} : () -> !fly.ptr<i8, register>
    %n = fly.recast_iter(%p) : (!fly.ptr<i8, register>) -> !fly.ptr<i4, register>
    fly.ptr.store(%x, %n) : (i4, !fly.ptr<i4, register>) -> ()
    %v = fly.ptr.load(%p) : (!fly.ptr<i8, register>) -> i8
    fly.ptr.store(%v, %out) : (i8, !fly.ptr<i8, global>) -> ()
    gpu.return
  }
}

// -----

// Placed storage must promote. Without placement the same access stays in
// memory form (@nibble above).
gpu.module @bounds {
  gpu.func @out_of_bounds(%input: i32) kernel {
    %p = fly.make_ptr() {dictAttrs = {allocSize = 4 : i64}} : () -> !fly.ptr<i8, register>
    fly.register_place(%p) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>, startIdx = 68 : i32} : (!fly.ptr<i8, register>) -> ()
    %two = fly.make_int_tuple() : () -> !fly.int_tuple<2>
    %q = fly.add_offset(%p, %two) : (!fly.ptr<i8, register>, !fly.int_tuple<2>) -> !fly.ptr<i8, register>
    %word = fly.recast_iter(%q) : (!fly.ptr<i8, register>) -> !fly.ptr<i32, register>
    // expected-error @+1 {{register access must be static, in bounds, and use whole storage elements}}
    fly.ptr.store(%input, %word) : (i32, !fly.ptr<i32, register>) -> ()
    gpu.return
  }
}

// -----

gpu.module @dynamic {
  gpu.func @dynamic_offset(%index: i32, %value: i16) kernel {
    // expected-error @+1 {{explicit register storage requires static offsets}}
    %p = fly.make_ptr() {dictAttrs = {allocSize = 8 : i64}} : () -> !fly.ptr<i8, register>
    fly.register_place(%p) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>, startIdx = 68 : i32} : (!fly.ptr<i8, register>) -> ()
    %halves = fly.recast_iter(%p) : (!fly.ptr<i8, register>) -> !fly.ptr<i16, register>
    %off = fly.make_int_tuple(%index) : (i32) -> !fly.int_tuple<?>
    %q = fly.add_offset(%halves, %off) : (!fly.ptr<i16, register>, !fly.int_tuple<?>) -> !fly.ptr<i16, register>
    fly.ptr.store(%value, %q) : (i16, !fly.ptr<i16, register>) -> ()
    gpu.return
  }
}
