// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-promote-regmem-to-vectorssa --split-input-file --verify-diagnostics

// Errors: a static access outside a promoted allocation, an access outside the
// region that allocates the register memory, and a register address that is
// still in use after promotion.

gpu.module @out_of_bounds {
  gpu.func @out_of_bounds(%out: !fly.ptr<f32, global>) kernel {
    %c2 = fly.static : !fly.int_tuple<2>
    %reg = fly.make_ptr() {dictAttrs = {allocSize = 4 : i64}} : () -> !fly.ptr<f32, register>
    %slot = fly.add_offset(%reg, %c2) : (!fly.ptr<f32, register>, !fly.int_tuple<2>) -> !fly.ptr<f32, register>
    // expected-error@+1 {{out-of-bounds register access: bits [64, 192) of a 128-bit allocation}}
    %val = fly.ptr.load(%slot) : (!fly.ptr<f32, register>) -> vector<4xf32>
    fly.ptr.store(%val, %out) : (vector<4xf32>, !fly.ptr<f32, global>) -> ()
    gpu.return
  }
}

// -----

gpu.module @outside_allocating_region {
  gpu.func @outside_allocating_region(%out: !fly.ptr<i32, global>, %x: i32) kernel {
    %false = arith.constant false
    %r = scf.while : () -> !fly.ptr<i32, register> {
      %p = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i32, register>
      fly.ptr.store(%x, %p) : (i32, !fly.ptr<i32, register>) -> ()
      scf.condition(%false) %p : !fly.ptr<i32, register>
    } do {
    ^bb0(%p: !fly.ptr<i32, register>):
      scf.yield
    }
    // expected-error@+1 {{register memory accessed outside the region that allocates it}}
    %v = fly.ptr.load(%r) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%v, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }
}

// -----

gpu.module @register_poison_in_input {
  gpu.func @register_poison_in_input(%out: !fly.ptr<i32, global>, %x: i32) kernel {
    %r = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i32, register>
    fly.ptr.store(%x, %r) : (i32, !fly.ptr<i32, register>) -> ()
    %p = ub.poison : !fly.ptr<i32, register>
    // expected-error@+1 {{register operand/result remain after rmem SSA promotion}}
    %v = fly.ptr.load(%p) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%v, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }
}
