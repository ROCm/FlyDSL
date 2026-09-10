// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-promote-regmem-to-vectorssa --verify-diagnostics

gpu.module @bounds {
  gpu.func @out_of_bounds(%input: i32) kernel {
    %p = fly.make_ptr() {dictAttrs = {allocSize = 4 : i64}} : () -> !fly.ptr<i8, register>
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
    fly.set_register %p {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 68 : i64} : !fly.ptr<i8, register>
    %halves = fly.recast_iter(%p) : (!fly.ptr<i8, register>) -> !fly.ptr<i16, register>
    %off = fly.make_int_tuple(%index) : (i32) -> !fly.int_tuple<?>
    %q = fly.add_offset(%halves, %off) : (!fly.ptr<i16, register>, !fly.int_tuple<?>) -> !fly.ptr<i16, register>
    fly.ptr.store(%value, %q) : (i16, !fly.ptr<i16, register>) -> ()
    gpu.return
  }
}
