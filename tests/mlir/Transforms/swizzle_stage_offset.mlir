// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering --fly-int-swizzle-simplify --canonicalize | FileCheck %s

// A composed layout whose offset comes from algebra: after layout lowering the
// offset is its arithmetic, not a get_leaves of a constructor, so the swizzle
// simplification can move the stage offset outside the xor.

// CHECK-LABEL: gpu.func @stage_offset_outside_swizzle
// CHECK-NOT: fly.get_leaves
// CHECK: %[[STAGE:.*]] = arith.muli %arg0, %c4096_i32
// CHECK: %[[SW:.*]] = arith.xori
// CHECK: arith.addi %[[SW]], %[[STAGE]]
gpu.module @m {
  gpu.func @stage_offset_outside_swizzle(%stage: i32, %row: i32) kernel {
    %sw = fly.static : !fly.swizzle<S<3,3,3>>
    %st = fly.make_int_tuple(%stage) : (i32) -> !fly.int_tuple<?>
    %c = fly.make_int_tuple() : () -> !fly.int_tuple<4096>
    %off = fly.int_tuple_mul(%st, %c) : (!fly.int_tuple<?>, !fly.int_tuple<4096>) -> !fly.int_tuple<?{div=4096}>
    %outer = fly.static : !fly.layout<(64,64):(64,1)>
    %cl = fly.make_composed_layout(%sw, %off, %outer) : (!fly.swizzle<S<3,3,3>>, !fly.int_tuple<?{div=4096}>, !fly.layout<(64,64):(64,1)>) -> !fly.composed_layout<S<3,3,3> o ?{div=4096} o (64,64):(64,1)>
    %crd = fly.make_int_tuple(%row) : (i32) -> !fly.int_tuple<(?,8)>
    %idx = fly.crd2idx(%crd, %cl) : (!fly.int_tuple<(?,8)>, !fly.composed_layout<S<3,3,3> o ?{div=4096} o (64,64):(64,1)>) -> !fly.int_tuple<?{div=8}>
    %s = fly.get_scalar(%idx) : (!fly.int_tuple<?{div=8}>) -> i32
    %t = fly.make_int_tuple(%s) : (i32) -> !fly.int_tuple<?>
    gpu.printf "%d", %s : i32
    gpu.return
  }
}
