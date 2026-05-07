// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s | FileCheck %s

// Tests for CoordSwizzle attribute/type and its use as the inner slot of
// ComposedLayoutAttr. Covers:
//   * Standalone CoordSwizzle materialized via fly.static.
//   * ComposedLayout whose inner is a CoordSwizzle (single level).
//   * Nested ComposedLayout: outer ComposedLayout wrapping an inner
//     ComposedLayout that bottoms out in a CoordSwizzle.
//   * Extractors: composed_get_inner / composed_get_offset / composed_get_outer
//     against a CoordSwizzle-inner composed layout.
//   * Trivial CoordSwizzle (mask=0) round-trip.

// -----

// CHECK-LABEL: @test_coord_swizzle_static
func.func @test_coord_swizzle_static() -> !fly.coord_swizzle<CS<3,0,[1],0,[2]>> {
  // CHECK: fly.static : !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
  %cs = fly.static : !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
  return %cs : !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
}

// CHECK-LABEL: @test_coord_swizzle_trivial
func.func @test_coord_swizzle_trivial() -> !fly.coord_swizzle<CS<0,0,[],0,[]>> {
  // CHECK: fly.static : !fly.coord_swizzle<CS<0,0,[],0,[]>>
  %cs = fly.static : !fly.coord_swizzle<CS<0,0,[],0,[]>>
  return %cs : !fly.coord_swizzle<CS<0,0,[],0,[]>>
}

// CHECK-LABEL: @test_composed_layout_with_coord_swizzle_inner
func.func @test_composed_layout_with_coord_swizzle_inner()
    -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)> {
  %cs = fly.static : !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
  %off = fly.static : !fly.int_tuple<(0, 0, 0)>
  %s = fly.static : !fly.int_tuple<(4, 8, 8)>
  %d = fly.static : !fly.int_tuple<(1E0, 1E1, 1E2)>
  %outer = fly.make_layout(%s, %d)
      : (!fly.int_tuple<(4, 8, 8)>, !fly.int_tuple<(1E0, 1E1, 1E2)>)
      -> !fly.layout<(4, 8, 8) : (1E0, 1E1, 1E2)>
  // CHECK: fly.make_composed_layout
  // CHECK-SAME: !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
  // CHECK-SAME: -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>
  %cl = fly.make_composed_layout(%cs, %off, %outer)
      : (!fly.coord_swizzle<CS<3,0,[1],0,[2]>>,
         !fly.int_tuple<(0, 0, 0)>,
         !fly.layout<(4, 8, 8) : (1E0, 1E1, 1E2)>)
      -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>
  return %cl : !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>
}

// CHECK-LABEL: @test_composed_layout_nested_coord_swizzle
func.func @test_composed_layout_nested_coord_swizzle()
    -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2) o (1,2,3) o (2,4,8):(1E0,1E1,1E2)> {
  %cs = fly.static : !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
  %off1 = fly.static : !fly.int_tuple<(0, 0, 0)>
  %s1 = fly.static : !fly.int_tuple<(4, 8, 8)>
  %d1 = fly.static : !fly.int_tuple<(1E0, 1E1, 1E2)>
  %outer1 = fly.make_layout(%s1, %d1)
      : (!fly.int_tuple<(4, 8, 8)>, !fly.int_tuple<(1E0, 1E1, 1E2)>)
      -> !fly.layout<(4, 8, 8) : (1E0, 1E1, 1E2)>
  // CHECK: %[[CL1:.+]] = fly.make_composed_layout
  // CHECK-SAME: -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>
  %cl1 = fly.make_composed_layout(%cs, %off1, %outer1)
      : (!fly.coord_swizzle<CS<3,0,[1],0,[2]>>,
         !fly.int_tuple<(0, 0, 0)>,
         !fly.layout<(4, 8, 8) : (1E0, 1E1, 1E2)>)
      -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>
  %off2 = fly.static : !fly.int_tuple<(1, 2, 3)>
  %s2 = fly.static : !fly.int_tuple<(2, 4, 8)>
  %d2 = fly.static : !fly.int_tuple<(1E0, 1E1, 1E2)>
  %outer2 = fly.make_layout(%s2, %d2)
      : (!fly.int_tuple<(2, 4, 8)>, !fly.int_tuple<(1E0, 1E1, 1E2)>)
      -> !fly.layout<(2, 4, 8) : (1E0, 1E1, 1E2)>
  // CHECK: fly.make_composed_layout(%[[CL1]],
  // CHECK-SAME: -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2) o (1,2,3) o (2,4,8):(1E0,1E1,1E2)>
  %cl2 = fly.make_composed_layout(%cl1, %off2, %outer2)
      : (!fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>,
         !fly.int_tuple<(1, 2, 3)>,
         !fly.layout<(2, 4, 8) : (1E0, 1E1, 1E2)>)
      -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2) o (1, 2, 3) o (2, 4, 8) : (1E0, 1E1, 1E2)>
  return %cl2 : !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2) o (1, 2, 3) o (2, 4, 8) : (1E0, 1E1, 1E2)>
}

// CHECK-LABEL: @test_composed_layout_with_composed_outer
func.func @test_composed_layout_with_composed_outer()
    -> !fly.composed_layout<32:2 o 0 o [32:1 o 0 o (4,8):(1,4)]> {
  %off = fly.static : !fly.int_tuple<0>
  %sa = fly.static : !fly.int_tuple<32>
  %da = fly.static : !fly.int_tuple<2>
  %inner = fly.make_layout(%sa, %da) : (!fly.int_tuple<32>, !fly.int_tuple<2>) -> !fly.layout<32:2>
  %sb = fly.static : !fly.int_tuple<32>
  %db = fly.static : !fly.int_tuple<1>
  %mid = fly.make_layout(%sb, %db) : (!fly.int_tuple<32>, !fly.int_tuple<1>) -> !fly.layout<32:1>
  %sc = fly.static : !fly.int_tuple<(4, 8)>
  %dc = fly.static : !fly.int_tuple<(1, 4)>
  %outer = fly.make_layout(%sc, %dc) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %cl1 = fly.make_composed_layout(%mid, %off, %outer)
      : (!fly.layout<32:1>, !fly.int_tuple<0>, !fly.layout<(4, 8) : (1, 4)>)
      -> !fly.composed_layout<32:1 o 0 o (4, 8) : (1, 4)>
  // CHECK: fly.make_composed_layout{{.*}} -> !fly.composed_layout<32:2 o 0 o [32:1 o 0 o (4,8):(1,4)]>
  %cl2 = fly.make_composed_layout(%inner, %off, %cl1)
      : (!fly.layout<32:2>, !fly.int_tuple<0>,
         !fly.composed_layout<32:1 o 0 o (4, 8) : (1, 4)>)
      -> !fly.composed_layout<32:2 o 0 o [32:1 o 0 o (4, 8) : (1, 4)]>
  return %cl2 : !fly.composed_layout<32:2 o 0 o [32:1 o 0 o (4, 8) : (1, 4)]>
}

// CHECK-LABEL: @test_composed_get_outer_with_composed_outer
func.func @test_composed_get_outer_with_composed_outer(
    %cl: !fly.composed_layout<32:2 o 0 o [32:1 o 0 o (4,8):(1,4)]>)
    -> !fly.composed_layout<32:1 o 0 o (4,8):(1,4)> {
  // CHECK: fly.composed_get_outer
  // CHECK-SAME: -> !fly.composed_layout<32:1 o 0 o (4,8):(1,4)>
  %outer = fly.composed_get_outer(%cl)
      : (!fly.composed_layout<32:2 o 0 o [32:1 o 0 o (4, 8) : (1, 4)]>)
      -> !fly.composed_layout<32:1 o 0 o (4, 8) : (1, 4)>
  return %outer : !fly.composed_layout<32:1 o 0 o (4, 8) : (1, 4)>
}

// CHECK-LABEL: @test_composed_get_inner_coord_swizzle
// CHECK-SAME: %[[CL:.+]]: !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>
func.func @test_composed_get_inner_coord_swizzle(
    %cl: !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>)
    -> !fly.coord_swizzle<CS<3,0,[1],0,[2]>> {
  // CHECK: fly.composed_get_inner(%[[CL]])
  // CHECK-SAME: -> !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
  %inner = fly.composed_get_inner(%cl)
      : (!fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>)
      -> !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
  return %inner : !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
}

// CHECK-LABEL: @test_composed_get_offset_with_coord_swizzle_inner
func.func @test_composed_get_offset_with_coord_swizzle_inner(
    %cl: !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>)
    -> !fly.int_tuple<(0,0,0)> {
  // CHECK: fly.composed_get_offset
  // CHECK-SAME: -> !fly.int_tuple<(0,0,0)>
  %off = fly.composed_get_offset(%cl)
      : (!fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>)
      -> !fly.int_tuple<(0, 0, 0)>
  return %off : !fly.int_tuple<(0, 0, 0)>
}

// CHECK-LABEL: @test_composed_get_outer_with_coord_swizzle_inner
func.func @test_composed_get_outer_with_coord_swizzle_inner(
    %cl: !fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>)
    -> !fly.layout<(4,8,8):(1E0,1E1,1E2)> {
  // CHECK: fly.composed_get_outer
  // CHECK-SAME: -> !fly.layout<(4,8,8):(1E0,1E1,1E2)>
  %outer = fly.composed_get_outer(%cl)
      : (!fly.composed_layout<CS<3,0,[1],0,[2]> o (0, 0, 0) o (4, 8, 8) : (1E0, 1E1, 1E2)>)
      -> !fly.layout<(4, 8, 8) : (1E0, 1E1, 1E2)>
  return %outer : !fly.layout<(4, 8, 8) : (1E0, 1E1, 1E2)>
}
