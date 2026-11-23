// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// add_offset canonicalization in fly-layout-lowering (issue #898).
//
// Normal form is `add_offset(add_offset(ptr, dyn), static)`: at most one dynamic
// layer plus one static tail.  Fusing a runtime offset with a compile-time
// constant would give every access its own base register instead of sharing one.

// dynamic -> static is already canonical: the two levels must stay split so
// many accesses share one runtime base instead of each computing its own.
// CHECK-LABEL: @test_add_offset_dyn_static_kept
func.func @test_add_offset_dyn_static_kept(%ptr: !fly.ptr<f32, shared>, %x: i32) -> f32 {
  %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  %c = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  // The runtime offset must never be merged into an arith.addi with the constant.
  // CHECK-NOT: arith.addi
  // CHECK: %[[BASE:.*]] = fly.add_offset(%{{.*}}, %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  // CHECK: fly.add_offset(%[[BASE]], %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %p1 = fly.add_offset(%ptr, %d) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %c) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %v = fly.ptr.load(%p2) : (!fly.ptr<f32, shared>) -> f32
  return %v : f32
}

// static -> dynamic is reordered into the canonical dynamic -> static form.
// CHECK-LABEL: @test_add_offset_static_dyn_swapped
func.func @test_add_offset_static_dyn_swapped(%ptr: !fly.ptr<f32, shared>, %x: i32) -> f32 {
  %c = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  // CHECK-NOT: arith.addi
  // CHECK: %[[BASE:.*]] = fly.add_offset(%{{.*}}, %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  // CHECK: fly.add_offset(%[[BASE]], %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %p1 = fly.add_offset(%ptr, %c) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %d) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %v = fly.ptr.load(%p2) : (!fly.ptr<f32, shared>) -> f32
  return %v : f32
}

// static -> static still fuses: no runtime value is merged with a constant.
// CHECK-LABEL: @test_add_offset_static_static_fused
func.func @test_add_offset_static_static_fused(%ptr: !fly.ptr<f32, shared>) -> f32 {
  %a = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  %b = fly.make_int_tuple() : () -> !fly.int_tuple<32>
  // CHECK: %[[C:.*]] = fly.static : !fly.int_tuple<96>
  // CHECK: fly.add_offset(%{{.*}}, %[[C]]) : (!fly.ptr<f32, shared>, !fly.int_tuple<96>) -> !fly.ptr<f32, shared>
  // CHECK-NOT: fly.add_offset
  %p1 = fly.add_offset(%ptr, %a) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %b) : (!fly.ptr<f32, shared>, !fly.int_tuple<32>) -> !fly.ptr<f32, shared>
  %v = fly.ptr.load(%p2) : (!fly.ptr<f32, shared>) -> f32
  return %v : f32
}

// dynamic -> dynamic still fuses: no constant is lost into a runtime value.
// CHECK-LABEL: @test_add_offset_dyn_dyn_fused
func.func @test_add_offset_dyn_dyn_fused(%ptr: !fly.ptr<f32, shared>, %x: i32, %y: i32) -> f32 {
  %a = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  %b = fly.make_int_tuple(%y) : (i32) -> !fly.int_tuple<?>
  // CHECK: %[[SUM:.*]] = arith.addi
  // CHECK: %[[T:.*]] = fly.make_int_tuple(%[[SUM]])
  // CHECK: fly.add_offset(%{{.*}}, %[[T]])
  // CHECK-NOT: fly.add_offset
  %p1 = fly.add_offset(%ptr, %a) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %b) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %v = fly.ptr.load(%p2) : (!fly.ptr<f32, shared>) -> f32
  return %v : f32
}

// Two constants behind one runtime offset collapse into a single static tail.
// CHECK-LABEL: @test_add_offset_dyn_c1_c2
func.func @test_add_offset_dyn_c1_c2(%ptr: !fly.ptr<f32, shared>, %x: i32) -> f32 {
  %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  %c1 = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  %c2 = fly.make_int_tuple() : () -> !fly.int_tuple<32>
  // CHECK-NOT: arith.addi
  // CHECK: %[[BASE:.*]] = fly.add_offset(%{{.*}}, %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  // CHECK: %[[C:.*]] = fly.static : !fly.int_tuple<96>
  // CHECK: fly.add_offset(%[[BASE]], %[[C]]) : (!fly.ptr<f32, shared>, !fly.int_tuple<96>) -> !fly.ptr<f32, shared>
  %p1 = fly.add_offset(%ptr, %d) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %c1) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %p3 = fly.add_offset(%p2, %c2) : (!fly.ptr<f32, shared>, !fly.int_tuple<32>) -> !fly.ptr<f32, shared>
  %v = fly.ptr.load(%p3) : (!fly.ptr<f32, shared>) -> f32
  return %v : f32
}

// A three-level static -> dynamic -> static chain converges to the normal form
// with both constants merged; the pattern must not ping-pong (pass must not fail).
// CHECK-LABEL: @test_add_offset_static_dyn_static
func.func @test_add_offset_static_dyn_static(%ptr: !fly.ptr<f32, shared>, %x: i32) -> f32 {
  %c1 = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  %c2 = fly.make_int_tuple() : () -> !fly.int_tuple<32>
  // CHECK-NOT: arith.addi
  // CHECK: %[[BASE:.*]] = fly.add_offset(%{{.*}}, %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  // CHECK: %[[C:.*]] = fly.static : !fly.int_tuple<96>
  // CHECK: fly.add_offset(%[[BASE]], %[[C]]) : (!fly.ptr<f32, shared>, !fly.int_tuple<96>) -> !fly.ptr<f32, shared>
  %p1 = fly.add_offset(%ptr, %c1) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %d) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %p3 = fly.add_offset(%p2, %c2) : (!fly.ptr<f32, shared>, !fly.int_tuple<32>) -> !fly.ptr<f32, shared>
  %v = fly.ptr.load(%p3) : (!fly.ptr<f32, shared>) -> f32
  return %v : f32
}

// One runtime base shared by two different static tails: the base must stay a
// single add_offset, which is what collapses the ds_read base registers.
// CHECK-LABEL: @test_add_offset_shared_runtime_base
func.func @test_add_offset_shared_runtime_base(%ptr: !fly.ptr<f32, shared>, %x: i32) -> f32 {
  %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  %c1 = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  %c2 = fly.make_int_tuple() : () -> !fly.int_tuple<128>
  // CHECK-NOT: arith.addi
  // CHECK: %[[BASE:.*]] = fly.add_offset(%{{.*}}, %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  // CHECK: fly.add_offset(%[[BASE]], %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  // CHECK: fly.add_offset(%[[BASE]], %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<128>) -> !fly.ptr<f32, shared>
  %p0 = fly.add_offset(%ptr, %d) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %pa = fly.add_offset(%p0, %c1) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %pb = fly.add_offset(%p0, %c2) : (!fly.ptr<f32, shared>, !fly.int_tuple<128>) -> !fly.ptr<f32, shared>
  %a = fly.ptr.load(%pa) : (!fly.ptr<f32, shared>) -> f32
  %b = fly.ptr.load(%pb) : (!fly.ptr<f32, shared>) -> f32
  %s = arith.addf %a, %b : f32
  return %s : f32
}

// A multi-use inner add_offset is left alone: swapping it would keep the old op
// alive and add a second add_offset instead of replacing one.  The form stays
// non-canonical but must still never fuse a runtime offset with a constant.
// CHECK-LABEL: @test_add_offset_multi_use_inner_kept
func.func @test_add_offset_multi_use_inner_kept(%ptr: !fly.ptr<f32, shared>, %x: i32) -> f32 {
  %c1 = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  // CHECK-NOT: arith.addi
  // CHECK: %[[P1:.*]] = fly.add_offset(%{{.*}}, %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  // CHECK: %[[P2:.*]] = fly.add_offset(%[[P1]], %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  // CHECK: fly.ptr.load(%[[P1]])
  // CHECK: fly.ptr.load(%[[P2]])
  %p1 = fly.add_offset(%ptr, %c1) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %d) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %a = fly.ptr.load(%p1) : (!fly.ptr<f32, shared>) -> f32
  %b = fly.ptr.load(%p2) : (!fly.ptr<f32, shared>) -> f32
  %s = arith.addf %a, %b : f32
  return %s : f32
}

// Zero and negative static offsets: 0 + (-32) merges to a single -32 tail.
// Guards the offsetDiv == 0 branch in AddOffsetOp type inference.
// CHECK-LABEL: @test_add_offset_zero_and_negative
func.func @test_add_offset_zero_and_negative(%ptr: !fly.ptr<f32, shared>, %x: i32) -> f32 {
  %d = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  %z = fly.make_int_tuple() : () -> !fly.int_tuple<0>
  %n = fly.make_int_tuple() : () -> !fly.int_tuple<-32>
  // CHECK-NOT: arith.addi
  // CHECK: %[[BASE:.*]] = fly.add_offset(%{{.*}}, %{{.*}}) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  // CHECK: %[[C:.*]] = fly.static : !fly.int_tuple<-32>
  // CHECK: fly.add_offset(%[[BASE]], %[[C]]) : (!fly.ptr<f32, shared>, !fly.int_tuple<-32>) -> !fly.ptr<f32, shared>
  %p1 = fly.add_offset(%ptr, %d) : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %z) : (!fly.ptr<f32, shared>, !fly.int_tuple<0>) -> !fly.ptr<f32, shared>
  %p3 = fly.add_offset(%p2, %n) : (!fly.ptr<f32, shared>, !fly.int_tuple<-32>) -> !fly.ptr<f32, shared>
  %v = fly.ptr.load(%p3) : (!fly.ptr<f32, shared>) -> f32
  return %v : f32
}

// A multi-use inner op still fuses when neither offset is dynamic: fusion
// rewrites only the outer op, so the inner one stays valid for its other
// users.  Only the swap needs a single-use inner op.
// CHECK-LABEL: @test_add_offset_multi_use_inner_still_fuses
func.func @test_add_offset_multi_use_inner_still_fuses(%ptr: !fly.ptr<f32, shared>) -> f32 {
  %a = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  %b = fly.make_int_tuple() : () -> !fly.int_tuple<32>
  // The outer offset is folded into a single static tail off the base pointer,
  // even though %p1 is also loaded from directly.
  // CHECK: %[[C96:.*]] = fly.static : !fly.int_tuple<96>
  // CHECK: fly.add_offset(%{{.*}}, %[[C96]]) : (!fly.ptr<f32, shared>, !fly.int_tuple<96>) -> !fly.ptr<f32, shared>
  %p1 = fly.add_offset(%ptr, %a) : (!fly.ptr<f32, shared>, !fly.int_tuple<64>) -> !fly.ptr<f32, shared>
  %p2 = fly.add_offset(%p1, %b) : (!fly.ptr<f32, shared>, !fly.int_tuple<32>) -> !fly.ptr<f32, shared>
  %x = fly.ptr.load(%p1) : (!fly.ptr<f32, shared>) -> f32
  %y = fly.ptr.load(%p2) : (!fly.ptr<f32, shared>) -> f32
  %s = arith.addf %x, %y : f32
  return %s : f32
}
