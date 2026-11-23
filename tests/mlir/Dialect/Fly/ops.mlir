// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s | %fly-opt | FileCheck %s
// RUN: %fly-opt %s --fly-layout-lowering -o /dev/null

// One use of each layout operation. The written result type must equal the inferred one, which
// is the Core function of the operation evaluated on the operand types; the algebra itself is
// tested in tests/core. Every result is returned, so layout lowering, which evaluates the same
// function on the operand values, fails unless it produces the same type.

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

// The dynamic leaves of the written result type are the operands of make_int_tuple, make_shape,
// make_stride, make_coord and make_tile.

// CHECK-LABEL: func.func @static(
// CHECK: fly.static : !fly.int_tuple<(4,8)>
func.func @static() -> !fly.int_tuple<(4,8)> {
  %0 = fly.static : !fly.int_tuple<(4,8)>
  return %0 : !fly.int_tuple<(4,8)>
}

// CHECK-LABEL: func.func @make_int_tuple(
// CHECK: fly.make_int_tuple{{.*}} -> !fly.int_tuple<(?,?{i64})>
func.func @make_int_tuple(%m: i32, %n: i64) -> !fly.int_tuple<(?,?{i64})> {
  %0 = fly.make_int_tuple(%m, %n) : (i32, i64) -> !fly.int_tuple<(?,?{i64})>
  return %0 : !fly.int_tuple<(?,?{i64})>
}

// CHECK-LABEL: func.func @make_shape(
// CHECK: fly.make_shape{{.*}} -> !fly.int_tuple<(?,8)>
func.func @make_shape(%m: i32) -> !fly.int_tuple<(?,8)> {
  %0 = fly.make_shape(%m) : (i32) -> !fly.int_tuple<(?,8)>
  return %0 : !fly.int_tuple<(?,8)>
}

// CHECK-LABEL: func.func @make_stride(
// CHECK: fly.make_stride{{.*}} -> !fly.int_tuple<(1,?{i64})>
func.func @make_stride(%n: i64) -> !fly.int_tuple<(1,?{i64})> {
  %0 = fly.make_stride(%n) : (i64) -> !fly.int_tuple<(1,?{i64})>
  return %0 : !fly.int_tuple<(1,?{i64})>
}

// CHECK-LABEL: func.func @make_coord(
// CHECK: fly.make_coord{{.*}} -> !fly.int_tuple<(?,0)>
func.func @make_coord(%m: i32) -> !fly.int_tuple<(?,0)> {
  %0 = fly.make_coord(%m) : (i32) -> !fly.int_tuple<(?,0)>
  return %0 : !fly.int_tuple<(?,0)>
}

// CHECK-LABEL: func.func @make_tile(
// CHECK: fly.make_tile{{.*}} -> !fly.tile<[?:1|(4,?{i64}):(1,4)|*]>
func.func @make_tile(%m: i32, %n: i64) -> !fly.tile<[?:1|(4,?{i64}):(1,4)|*]> {
  %0 = fly.make_tile(%m, %n) : (i32, i64) -> !fly.tile<[?:1|(4,?{i64}):(1,4)|*]>
  return %0 : !fly.tile<[?:1|(4,?{i64}):(1,4)|*]>
}

// CHECK-LABEL: func.func @make_layout(
// CHECK: fly.make_layout{{.*}} -> !fly.layout<(4,8):(1,4)>
func.func @make_layout(%shape: !fly.int_tuple<(4,8)>, %stride: !fly.int_tuple<(1,4)>) -> !fly.layout<(4,8):(1,4)> {
  %0 = fly.make_layout(%shape, %stride) : (!fly.int_tuple<(4,8)>, !fly.int_tuple<(1,4)>) -> !fly.layout<(4,8):(1,4)>
  return %0 : !fly.layout<(4,8):(1,4)>
}

// CHECK-LABEL: func.func @make_layout_like(
// CHECK: fly.make_layout_like{{.*}} -> !fly.layout<(4,8):(1,4)>
func.func @make_layout_like(%layout: !fly.layout<(4,8):(1,4)>) -> !fly.layout<(4,8):(1,4)> {
  %0 = fly.make_layout_like(%layout) : (!fly.layout<(4,8):(1,4)>) -> !fly.layout<(4,8):(1,4)>
  return %0 : !fly.layout<(4,8):(1,4)>
}

// CHECK-LABEL: func.func @make_ordered_layout(
// CHECK: fly.make_ordered_layout{{.*}} -> !fly.layout<(4,8):(8,1)>
func.func @make_ordered_layout(%shape: !fly.int_tuple<(4,8)>, %order: !fly.int_tuple<(1,0)>) -> !fly.layout<(4,8):(8,1)> {
  %0 = fly.make_ordered_layout(%shape, %order) : (!fly.int_tuple<(4,8)>, !fly.int_tuple<(1,0)>) -> !fly.layout<(4,8):(8,1)>
  return %0 : !fly.layout<(4,8):(8,1)>
}

// CHECK-LABEL: func.func @make_identity_layout(
// CHECK: fly.make_identity_layout{{.*}} -> !fly.layout<(4,8):(1E0,1E1)>
func.func @make_identity_layout(%shape: !fly.int_tuple<(4,8)>) -> !fly.layout<(4,8):(1E0,1E1)> {
  %0 = fly.make_identity_layout(%shape) : (!fly.int_tuple<(4,8)>) -> !fly.layout<(4,8):(1E0,1E1)>
  return %0 : !fly.layout<(4,8):(1E0,1E1)>
}

// CHECK-LABEL: func.func @make_composed_layout(
// CHECK: fly.make_composed_layout{{.*}} -> !fly.composed_layout<S<3,3,3> o 0 o (8,8):(8,1)>
func.func @make_composed_layout(%swizzle: !fly.swizzle<S<3,3,3>>, %offset: !fly.int_tuple<0>, %outer: !fly.layout<(8,8):(8,1)>) -> !fly.composed_layout<S<3,3,3> o 0 o (8,8):(8,1)> {
  %0 = fly.make_composed_layout(%swizzle, %offset, %outer) : (!fly.swizzle<S<3,3,3>>, !fly.int_tuple<0>, !fly.layout<(8,8):(8,1)>) -> !fly.composed_layout<S<3,3,3> o 0 o (8,8):(8,1)>
  return %0 : !fly.composed_layout<S<3,3,3> o 0 o (8,8):(8,1)>
}

// CHECK-LABEL: func.func @make_composed_layout_coord_swizzle(
// CHECK: fly.make_composed_layout{{.*}} -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>
func.func @make_composed_layout_coord_swizzle(%swizzle: !fly.coord_swizzle<CS<3,0,[1],0,[2]>>, %offset: !fly.int_tuple<(0,0,0)>, %outer: !fly.layout<(4,8,8):(1E0,1E1,1E2)>) -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)> {
  %0 = fly.make_composed_layout(%swizzle, %offset, %outer) : (!fly.coord_swizzle<CS<3,0,[1],0,[2]>>, !fly.int_tuple<(0,0,0)>, !fly.layout<(4,8,8):(1E0,1E1,1E2)>) -> !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>
  return %0 : !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>
}

// CHECK-LABEL: func.func @make_composed_layout_chain(
// CHECK: fly.make_composed_layout{{.*}} -> !fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1) o 1 o (2,2):(1,2)>
func.func @make_composed_layout_chain(%inner: !fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1)>, %offset: !fly.int_tuple<1>, %outer: !fly.layout<(2,2):(1,2)>) -> !fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1) o 1 o (2,2):(1,2)> {
  %0 = fly.make_composed_layout(%inner, %offset, %outer) : (!fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1)>, !fly.int_tuple<1>, !fly.layout<(2,2):(1,2)>) -> !fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1) o 1 o (2,2):(1,2)>
  return %0 : !fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1) o 1 o (2,2):(1,2)>
}

// CHECK-LABEL: func.func @make_view(
// CHECK: fly.make_view{{.*}} -> !fly.memref<f32, global, (4,8):(1,4)>
func.func @make_view(%ptr: !fly.ptr<f32, global>, %layout: !fly.layout<(4,8):(1,4)>) -> !fly.memref<f32, global, (4,8):(1,4)> {
  %0 = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, global>, !fly.layout<(4,8):(1,4)>) -> !fly.memref<f32, global, (4,8):(1,4)>
  return %0 : !fly.memref<f32, global, (4,8):(1,4)>
}

// CHECK-LABEL: func.func @make_view_coord_tensor(
// CHECK: fly.make_view{{.*}} -> !fly.coord_tensor<(0,0), (4,8):(1E0,1E1)>
func.func @make_view_coord_tensor(%base: !fly.int_tuple<(0,0)>, %layout: !fly.layout<(4,8):(1E0,1E1)>) -> !fly.coord_tensor<(0,0), (4,8):(1E0,1E1)> {
  %0 = fly.make_view(%base, %layout) : (!fly.int_tuple<(0,0)>, !fly.layout<(4,8):(1E0,1E1)>) -> !fly.coord_tensor<(0,0), (4,8):(1E0,1E1)>
  return %0 : !fly.coord_tensor<(0,0), (4,8):(1E0,1E1)>
}

// CHECK-LABEL: func.func @make_fragment_like(
// CHECK: fly.make_fragment_like{{.*}} -> !fly.memref<f16, register, (4,8):(1,4)>
func.func @make_fragment_like(%layout: !fly.layout<(4,8):(1,4)>) -> !fly.memref<f16, register, (4,8):(1,4)> {
  %0 = fly.make_fragment_like(%layout, f16) : (!fly.layout<(4,8):(1,4)>) -> !fly.memref<f16, register, (4,8):(1,4)>
  return %0 : !fly.memref<f16, register, (4,8):(1,4)>
}

// CHECK-LABEL: func.func @make_fragment_layout_like(
// CHECK: fly.make_fragment_layout_like{{.*}} -> !fly.layout<(4,8):(1,4)>
func.func @make_fragment_layout_like(%layout: !fly.layout<(4,8):(8,1)>) -> !fly.layout<(4,8):(1,4)> {
  %0 = fly.make_fragment_layout_like(%layout) : (!fly.layout<(4,8):(8,1)>) -> !fly.layout<(4,8):(1,4)>
  return %0 : !fly.layout<(4,8):(1,4)>
}

//===----------------------------------------------------------------------===//
// Extractors
//===----------------------------------------------------------------------===//

// get_scalar looks through nested singleton tuples.

// CHECK-LABEL: func.func @get_scalar(
// CHECK: fly.get_scalar{{.*}} -> i64
func.func @get_scalar(%leaf: !fly.int_tuple<((?{i64}))>) -> i64 {
  %0 = fly.get_scalar(%leaf) : (!fly.int_tuple<((?{i64}))>) -> i64
  return %0 : i64
}

// CHECK-LABEL: func.func @get_leaves(
// CHECK: fly.get_leaves{{.*}} -> (i32, i64)
func.func @get_leaves(%layout: !fly.layout<(4,?):(1,?{i64})>) -> (i32, i64) {
  %0:2 = fly.get_leaves(%layout) : (!fly.layout<(4,?):(1,?{i64})>) -> (i32, i64)
  return %0#0, %0#1 : i32, i64
}

// CHECK-LABEL: func.func @get_shape(
// CHECK: fly.get_shape{{.*}} -> !fly.int_tuple<(4,?)>
func.func @get_shape(%layout: !fly.layout<(4,?):(1,?{i64})>) -> !fly.int_tuple<(4,?)> {
  %0 = fly.get_shape(%layout) : (!fly.layout<(4,?):(1,?{i64})>) -> !fly.int_tuple<(4,?)>
  return %0 : !fly.int_tuple<(4,?)>
}

// CHECK-LABEL: func.func @get_stride(
// CHECK: fly.get_stride{{.*}} -> !fly.int_tuple<(1,?{i64})>
func.func @get_stride(%layout: !fly.layout<(4,?):(1,?{i64})>) -> !fly.int_tuple<(1,?{i64})> {
  %0 = fly.get_stride(%layout) : (!fly.layout<(4,?):(1,?{i64})>) -> !fly.int_tuple<(1,?{i64})>
  return %0 : !fly.int_tuple<(1,?{i64})>
}

// CHECK-LABEL: func.func @get_layout(
// CHECK: fly.get_layout{{.*}} -> !fly.layout<(4,8):(1,4)>
func.func @get_layout(%memref: !fly.memref<f32, global, (4,8):(1,4)>) -> !fly.layout<(4,8):(1,4)> {
  %0 = fly.get_layout(%memref) : (!fly.memref<f32, global, (4,8):(1,4)>) -> !fly.layout<(4,8):(1,4)>
  return %0 : !fly.layout<(4,8):(1,4)>
}

// CHECK-LABEL: func.func @get_iter(
// CHECK: fly.get_iter{{.*}} -> !fly.ptr<f32, global>
func.func @get_iter(%memref: !fly.memref<f32, global, (4,8):(1,4)>) -> !fly.ptr<f32, global> {
  %0 = fly.get_iter(%memref) : (!fly.memref<f32, global, (4,8):(1,4)>) -> !fly.ptr<f32, global>
  return %0 : !fly.ptr<f32, global>
}

// CHECK-LABEL: func.func @composed_get_inner(
// CHECK: fly.composed_get_inner{{.*}} -> !fly.swizzle<S<3,3,3>>
func.func @composed_get_inner(%layout: !fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1)>) -> !fly.swizzle<S<3,3,3>> {
  %0 = fly.composed_get_inner(%layout) : (!fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1)>) -> !fly.swizzle<S<3,3,3>>
  return %0 : !fly.swizzle<S<3,3,3>>
}

// CHECK-LABEL: func.func @composed_get_offset(
// CHECK: fly.composed_get_offset{{.*}} -> !fly.int_tuple<2>
func.func @composed_get_offset(%layout: !fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1)>) -> !fly.int_tuple<2> {
  %0 = fly.composed_get_offset(%layout) : (!fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1)>) -> !fly.int_tuple<2>
  return %0 : !fly.int_tuple<2>
}

// CHECK-LABEL: func.func @composed_get_outer(
// CHECK: fly.composed_get_outer{{.*}} -> !fly.layout<(8,8):(8,1)>
func.func @composed_get_outer(%layout: !fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1)>) -> !fly.layout<(8,8):(8,1)> {
  %0 = fly.composed_get_outer(%layout) : (!fly.composed_layout<S<3,3,3> o 2 o (8,8):(8,1)>) -> !fly.layout<(8,8):(8,1)>
  return %0 : !fly.layout<(8,8):(8,1)>
}

// CHECK-LABEL: func.func @composed_get_inner_coord_swizzle(
// CHECK: fly.composed_get_inner{{.*}} -> !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
func.func @composed_get_inner_coord_swizzle(%layout: !fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>) -> !fly.coord_swizzle<CS<3,0,[1],0,[2]>> {
  %0 = fly.composed_get_inner(%layout) : (!fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>) -> !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
  return %0 : !fly.coord_swizzle<CS<3,0,[1],0,[2]>>
}

//===----------------------------------------------------------------------===//
// Int tuple arithmetic
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @int_tuple_add(
// CHECK: fly.int_tuple_add{{.*}} -> !fly.int_tuple<(11,10)>
func.func @int_tuple_add(%a: !fly.int_tuple<(8,6)>, %b: !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(11,10)> {
  %0 = fly.int_tuple_add(%a, %b) : (!fly.int_tuple<(8,6)>, !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(11,10)>
  return %0 : !fly.int_tuple<(11,10)>
}

// CHECK-LABEL: func.func @int_tuple_sub(
// CHECK: fly.int_tuple_sub{{.*}} -> !fly.int_tuple<(5,2)>
func.func @int_tuple_sub(%a: !fly.int_tuple<(8,6)>, %b: !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(5,2)> {
  %0 = fly.int_tuple_sub(%a, %b) : (!fly.int_tuple<(8,6)>, !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(5,2)>
  return %0 : !fly.int_tuple<(5,2)>
}

// CHECK-LABEL: func.func @int_tuple_mul(
// CHECK: fly.int_tuple_mul{{.*}} -> !fly.int_tuple<(24,24)>
func.func @int_tuple_mul(%a: !fly.int_tuple<(8,6)>, %b: !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(24,24)> {
  %0 = fly.int_tuple_mul(%a, %b) : (!fly.int_tuple<(8,6)>, !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(24,24)>
  return %0 : !fly.int_tuple<(24,24)>
}

// CHECK-LABEL: func.func @int_tuple_div(
// CHECK: fly.int_tuple_div{{.*}} -> !fly.int_tuple<(2,1)>
func.func @int_tuple_div(%a: !fly.int_tuple<(8,6)>, %b: !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(2,1)> {
  %0 = fly.int_tuple_div(%a, %b) : (!fly.int_tuple<(8,6)>, !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(2,1)>
  return %0 : !fly.int_tuple<(2,1)>
}

// CHECK-LABEL: func.func @int_tuple_mod(
// CHECK: fly.int_tuple_mod{{.*}} -> !fly.int_tuple<(2,2)>
func.func @int_tuple_mod(%a: !fly.int_tuple<(8,6)>, %b: !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(2,2)> {
  %0 = fly.int_tuple_mod(%a, %b) : (!fly.int_tuple<(8,6)>, !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(2,2)>
  return %0 : !fly.int_tuple<(2,2)>
}

// CHECK-LABEL: func.func @int_tuple_product(
// CHECK: fly.int_tuple_product{{.*}} -> !fly.int_tuple<24>
func.func @int_tuple_product(%a: !fly.int_tuple<((2,4),3)>) -> !fly.int_tuple<24> {
  %0 = fly.int_tuple_product(%a) : (!fly.int_tuple<((2,4),3)>) -> !fly.int_tuple<24>
  return %0 : !fly.int_tuple<24>
}

// CHECK-LABEL: func.func @int_tuple_product_each(
// CHECK: fly.int_tuple_product_each{{.*}} -> !fly.int_tuple<(8,3)>
func.func @int_tuple_product_each(%a: !fly.int_tuple<((2,4),3)>) -> !fly.int_tuple<(8,3)> {
  %0 = fly.int_tuple_product_each(%a) : (!fly.int_tuple<((2,4),3)>) -> !fly.int_tuple<(8,3)>
  return %0 : !fly.int_tuple<(8,3)>
}

// CHECK-LABEL: func.func @int_tuple_product_like(
// CHECK: fly.int_tuple_product_like{{.*}} -> !fly.int_tuple<(8,3)>
func.func @int_tuple_product_like(%a: !fly.int_tuple<((2,4),3)>, %b: !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(8,3)> {
  %0 = fly.int_tuple_product_like(%a, %b) : (!fly.int_tuple<((2,4),3)>, !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(8,3)>
  return %0 : !fly.int_tuple<(8,3)>
}

// CHECK-LABEL: func.func @shape_div(
// CHECK: fly.shape_div{{.*}} -> !fly.int_tuple<(4,2)>
func.func @shape_div(%a: !fly.int_tuple<(8,6)>, %b: !fly.int_tuple<(2,3)>) -> !fly.int_tuple<(4,2)> {
  %0 = fly.shape_div(%a, %b) : (!fly.int_tuple<(8,6)>, !fly.int_tuple<(2,3)>) -> !fly.int_tuple<(4,2)>
  return %0 : !fly.int_tuple<(4,2)>
}

// CHECK-LABEL: func.func @ceil_div(
// CHECK: fly.ceil_div{{.*}} -> !fly.int_tuple<(3,2)>
func.func @ceil_div(%a: !fly.int_tuple<(8,6)>, %b: !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(3,2)> {
  %0 = fly.ceil_div(%a, %b) : (!fly.int_tuple<(8,6)>, !fly.int_tuple<(3,4)>) -> !fly.int_tuple<(3,2)>
  return %0 : !fly.int_tuple<(3,2)>
}

// CHECK-LABEL: func.func @elem_less(
// CHECK: fly.elem_less{{.*}} -> i1
func.func @elem_less(%a: !fly.int_tuple<(3,4)>, %b: !fly.int_tuple<(8,6)>) -> i1 {
  %0 = fly.elem_less(%a, %b) : (!fly.int_tuple<(3,4)>, !fly.int_tuple<(8,6)>) -> i1
  return %0 : i1
}

// CHECK-LABEL: func.func @equal(
// CHECK: fly.equal{{.*}} -> i1
func.func @equal(%a: !fly.int_tuple<(8,6)>, %b: !fly.int_tuple<(3,4)>) -> i1 {
  %0 = fly.equal(%a, %b) : (!fly.int_tuple<(8,6)>, !fly.int_tuple<(3,4)>) -> i1
  return %0 : i1
}

//===----------------------------------------------------------------------===//
// Modes
//===----------------------------------------------------------------------===//

// A single mode taken from a layout is the mode itself, not a rank-1 layout around it.

// CHECK-LABEL: func.func @get(
// CHECK: fly.get{{.*}} -> !fly.int_tuple<(3,4)>
func.func @get(%tuple: !fly.int_tuple<(2,(3,4),5)>) -> !fly.int_tuple<(3,4)> {
  %0 = fly.get(%tuple) {mode = array<i32: 1>} : (!fly.int_tuple<(2,(3,4),5)>) -> !fly.int_tuple<(3,4)>
  return %0 : !fly.int_tuple<(3,4)>
}

// CHECK-LABEL: func.func @take(
// CHECK: fly.take{{.*}} -> !fly.int_tuple<((3,4),5)>
func.func @take(%tuple: !fly.int_tuple<(2,(3,4),5)>) -> !fly.int_tuple<((3,4),5)> {
  %0 = fly.take(%tuple) {begin = 1 : i32, end = 3 : i32} : (!fly.int_tuple<(2,(3,4),5)>) -> !fly.int_tuple<((3,4),5)>
  return %0 : !fly.int_tuple<((3,4),5)>
}

// CHECK-LABEL: func.func @take_one_mode(
// CHECK: fly.take{{.*}} -> !fly.layout<(3,4):(2,6)>
func.func @take_one_mode(%layout: !fly.layout<(2,(3,4)):(1,(2,6))>) -> !fly.layout<(3,4):(2,6)> {
  %0 = fly.take(%layout) {begin = 1 : i32, end = 2 : i32} : (!fly.layout<(2,(3,4)):(1,(2,6))>) -> !fly.layout<(3,4):(2,6)>
  return %0 : !fly.layout<(3,4):(2,6)>
}

// CHECK-LABEL: func.func @select(
// CHECK: fly.select{{.*}} -> !fly.int_tuple<(5,2)>
func.func @select(%tuple: !fly.int_tuple<(2,(3,4),5)>) -> !fly.int_tuple<(5,2)> {
  %0 = fly.select(%tuple) {indices = array<i32: 2, 0>} : (!fly.int_tuple<(2,(3,4),5)>) -> !fly.int_tuple<(5,2)>
  return %0 : !fly.int_tuple<(5,2)>
}

// CHECK-LABEL: func.func @group(
// CHECK: fly.group{{.*}} -> !fly.int_tuple<((2,(3,4)),5)>
func.func @group(%tuple: !fly.int_tuple<(2,(3,4),5)>) -> !fly.int_tuple<((2,(3,4)),5)> {
  %0 = fly.group(%tuple) {begin = 0 : i32, end = 2 : i32} : (!fly.int_tuple<(2,(3,4),5)>) -> !fly.int_tuple<((2,(3,4)),5)>
  return %0 : !fly.int_tuple<((2,(3,4)),5)>
}

// CHECK-LABEL: func.func @append(
// CHECK: fly.append{{.*}} -> !fly.int_tuple<(2,(3,4),5,7)>
func.func @append(%tuple: !fly.int_tuple<(2,(3,4),5)>, %elem: !fly.int_tuple<7>) -> !fly.int_tuple<(2,(3,4),5,7)> {
  %0 = fly.append (%tuple, %elem) : (!fly.int_tuple<(2,(3,4),5)>, !fly.int_tuple<7>) -> !fly.int_tuple<(2,(3,4),5,7)>
  return %0 : !fly.int_tuple<(2,(3,4),5,7)>
}

// CHECK-LABEL: func.func @prepend(
// CHECK: fly.prepend{{.*}} -> !fly.int_tuple<(7,2,(3,4),5)>
func.func @prepend(%tuple: !fly.int_tuple<(2,(3,4),5)>, %elem: !fly.int_tuple<7>) -> !fly.int_tuple<(7,2,(3,4),5)> {
  %0 = fly.prepend (%tuple, %elem) : (!fly.int_tuple<(2,(3,4),5)>, !fly.int_tuple<7>) -> !fly.int_tuple<(7,2,(3,4),5)>
  return %0 : !fly.int_tuple<(7,2,(3,4),5)>
}

// CHECK-LABEL: func.func @slice(
// CHECK: fly.slice{{.*}} -> !fly.layout<(3,4):(2,6)>
func.func @slice(%layout: !fly.layout<(2,(3,4)):(1,(2,6))>, %coord: !fly.int_tuple<(1,*)>) -> !fly.layout<(3,4):(2,6)> {
  %0 = fly.slice(%layout, %coord) : (!fly.layout<(2,(3,4)):(1,(2,6))>, !fly.int_tuple<(1,*)>) -> !fly.layout<(3,4):(2,6)>
  return %0 : !fly.layout<(3,4):(2,6)>
}

// CHECK-LABEL: func.func @dice(
// CHECK: fly.dice{{.*}} -> !fly.layout<2:1>
func.func @dice(%layout: !fly.layout<(2,(3,4)):(1,(2,6))>, %coord: !fly.int_tuple<(1,*)>) -> !fly.layout<2:1> {
  %0 = fly.dice(%layout, %coord) : (!fly.layout<(2,(3,4)):(1,(2,6))>, !fly.int_tuple<(1,*)>) -> !fly.layout<2:1>
  return %0 : !fly.layout<2:1>
}

//===----------------------------------------------------------------------===//
// Layout functions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @size(
// CHECK: fly.size{{.*}} -> !fly.int_tuple<32>
func.func @size(%layout: !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<32> {
  %0 = fly.size(%layout) : (!fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<32>
  return %0 : !fly.int_tuple<32>
}

// CHECK-LABEL: func.func @coprofile(
// CHECK: fly.coprofile{{.*}} -> !fly.int_tuple<(0,0)>
func.func @coprofile(%layout: !fly.layout<(4,8):(1E0,1E1)>) -> !fly.int_tuple<(0,0)> {
  %0 = fly.coprofile(%layout) : (!fly.layout<(4,8):(1E0,1E1)>) -> !fly.int_tuple<(0,0)>
  return %0 : !fly.int_tuple<(0,0)>
}

// CHECK-LABEL: func.func @coshape(
// CHECK: fly.coshape{{.*}} -> !fly.int_tuple<(4,8)>
func.func @coshape(%layout: !fly.layout<(4,8):(1E0,1E1)>) -> !fly.int_tuple<(4,8)> {
  %0 = fly.coshape(%layout) : (!fly.layout<(4,8):(1E0,1E1)>) -> !fly.int_tuple<(4,8)>
  return %0 : !fly.int_tuple<(4,8)>
}

// CHECK-LABEL: func.func @cosize(
// CHECK: fly.cosize{{.*}} -> !fly.int_tuple<7>
func.func @cosize(%layout: !fly.layout<4:2>) -> !fly.int_tuple<7> {
  %0 = fly.cosize(%layout) : (!fly.layout<4:2>) -> !fly.int_tuple<7>
  return %0 : !fly.int_tuple<7>
}

// CHECK-LABEL: func.func @crd2idx(
// CHECK: fly.crd2idx{{.*}} -> !fly.int_tuple<13>
func.func @crd2idx(%coord: !fly.int_tuple<(1,3)>, %layout: !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<13> {
  %0 = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(1,3)>, !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<13>
  return %0 : !fly.int_tuple<13>
}

// CHECK-LABEL: func.func @idx2crd(
// CHECK: fly.idx2crd{{.*}} -> !fly.int_tuple<((1,2),1)>
func.func @idx2crd(%index: !fly.int_tuple<13>, %layout: !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<((1,2),1)> {
  %0 = fly.idx2crd(%index, %layout) : (!fly.int_tuple<13>, !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<((1,2),1)>
  return %0 : !fly.int_tuple<((1,2),1)>
}

// CHECK-LABEL: func.func @get_flat_coord(
// CHECK: fly.get_flat_coord{{.*}} -> !fly.int_tuple<(5,1)>
func.func @get_flat_coord(%index: !fly.int_tuple<13>, %layout: !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<(5,1)> {
  %0 = fly.get_flat_coord(%index, %layout) : (!fly.int_tuple<13>, !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<(5,1)>
  return %0 : !fly.int_tuple<(5,1)>
}

// CHECK-LABEL: func.func @get_1d_coord(
// CHECK: fly.get_1d_coord{{.*}} -> !fly.int_tuple<13>
func.func @get_1d_coord(%index: !fly.int_tuple<13>, %layout: !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<13> {
  %0 = fly.get_1d_coord(%index, %layout) : (!fly.int_tuple<13>, !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<13>
  return %0 : !fly.int_tuple<13>
}

// CHECK-LABEL: func.func @coalesce(
// CHECK: fly.coalesce{{.*}} -> !fly.layout<32:1>
func.func @coalesce(%layout: !fly.layout<(4,8):(1,4)>) -> !fly.layout<32:1> {
  %0 = fly.coalesce(%layout) : (!fly.layout<(4,8):(1,4)>) -> !fly.layout<32:1>
  return %0 : !fly.layout<32:1>
}

// CHECK-LABEL: func.func @composition(
// CHECK: fly.composition{{.*}} -> !fly.layout<(2,2):(8,16)>
func.func @composition(%layout: !fly.layout<(4,8):(8,1)>, %tiler: !fly.layout<(2,2):(1,2)>) -> !fly.layout<(2,2):(8,16)> {
  %0 = fly.composition(%layout, %tiler) : (!fly.layout<(4,8):(8,1)>, !fly.layout<(2,2):(1,2)>) -> !fly.layout<(2,2):(8,16)>
  return %0 : !fly.layout<(2,2):(8,16)>
}

// CHECK-LABEL: func.func @complement(
// CHECK: fly.complement{{.*}} -> !fly.layout<(2,2):(1,8)>
func.func @complement(%layout: !fly.layout<4:2>, %codomain: !fly.int_tuple<16>) -> !fly.layout<(2,2):(1,8)> {
  %0 = fly.complement(%layout, %codomain) : (!fly.layout<4:2>, !fly.int_tuple<16>) -> !fly.layout<(2,2):(1,8)>
  return %0 : !fly.layout<(2,2):(1,8)>
}

// CHECK-LABEL: func.func @complement_free(
// CHECK: fly.complement{{.*}} -> !fly.layout<(2,1):(1,8)>
func.func @complement_free(%layout: !fly.layout<4:2>) -> !fly.layout<(2,1):(1,8)> {
  %0 = fly.complement(%layout) : (!fly.layout<4:2>) -> !fly.layout<(2,1):(1,8)>
  return %0 : !fly.layout<(2,1):(1,8)>
}

// CHECK-LABEL: func.func @right_inverse(
// CHECK: fly.right_inverse{{.*}} -> !fly.layout<1:0>
func.func @right_inverse(%layout: !fly.layout<4:2>) -> !fly.layout<1:0> {
  %0 = fly.right_inverse(%layout) : (!fly.layout<4:2>) -> !fly.layout<1:0>
  return %0 : !fly.layout<1:0>
}

// CHECK-LABEL: func.func @left_inverse(
// CHECK: fly.left_inverse{{.*}} -> !fly.layout<(2,4):(0,1)>
func.func @left_inverse(%layout: !fly.layout<4:2>) -> !fly.layout<(2,4):(0,1)> {
  %0 = fly.left_inverse(%layout) : (!fly.layout<4:2>) -> !fly.layout<(2,4):(0,1)>
  return %0 : !fly.layout<(2,4):(0,1)>
}

// CHECK-LABEL: func.func @recast_layout(
// CHECK: fly.recast_layout{{.*}} -> !fly.layout<(8,8):(1,8)>
func.func @recast_layout(%layout: !fly.layout<(4,8):(1,4)>) -> !fly.layout<(8,8):(1,8)> {
  %0 = fly.recast_layout(%layout) {new_type_bits = 16 : i32, old_type_bits = 32 : i32} : (!fly.layout<(4,8):(1,4)>) -> !fly.layout<(8,8):(1,8)>
  return %0 : !fly.layout<(8,8):(1,8)>
}

// CHECK-LABEL: func.func @tile_to_shape(
// CHECK: fly.tile_to_shape{{.*}} -> !fly.layout<((2,2),(2,4)):((1,16),(2,4))>
func.func @tile_to_shape(%block: !fly.layout<(2,2):(1,2)>, %shape: !fly.int_tuple<(4,8)>, %order: !fly.int_tuple<(1,0)>) -> !fly.layout<((2,2),(2,4)):((1,16),(2,4))> {
  %0 = fly.tile_to_shape(%block, %shape, %order) : (!fly.layout<(2,2):(1,2)>, !fly.int_tuple<(4,8)>, !fly.int_tuple<(1,0)>) -> !fly.layout<((2,2),(2,4)):((1,16),(2,4))>
  return %0 : !fly.layout<((2,2),(2,4)):((1,16),(2,4))>
}

//===----------------------------------------------------------------------===//
// Divides and products
//===----------------------------------------------------------------------===//

// The four divides differ only in how they group the tile and rest modes. A layout tiler has a
// single tile mode, so its logical and zipped products agree.

// CHECK-LABEL: func.func @logical_divide(
// CHECK: fly.logical_divide{{.*}} -> !fly.layout<((2,2),(4,2)):((1,2),(4,16))>
func.func @logical_divide(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.tile<[2:1|4:1]>) -> !fly.layout<((2,2),(4,2)):((1,2),(4,16))> {
  %0 = fly.logical_divide(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.tile<[2:1|4:1]>) -> !fly.layout<((2,2),(4,2)):((1,2),(4,16))>
  return %0 : !fly.layout<((2,2),(4,2)):((1,2),(4,16))>
}

// CHECK-LABEL: func.func @zipped_divide(
// CHECK: fly.zipped_divide{{.*}} -> !fly.layout<((2,4),(2,2)):((1,4),(2,16))>
func.func @zipped_divide(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.tile<[2:1|4:1]>) -> !fly.layout<((2,4),(2,2)):((1,4),(2,16))> {
  %0 = fly.zipped_divide(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.tile<[2:1|4:1]>) -> !fly.layout<((2,4),(2,2)):((1,4),(2,16))>
  return %0 : !fly.layout<((2,4),(2,2)):((1,4),(2,16))>
}

// CHECK-LABEL: func.func @tiled_divide(
// CHECK: fly.tiled_divide{{.*}} -> !fly.layout<((2,4),2,2):((1,4),2,16)>
func.func @tiled_divide(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.tile<[2:1|4:1]>) -> !fly.layout<((2,4),2,2):((1,4),2,16)> {
  %0 = fly.tiled_divide(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.tile<[2:1|4:1]>) -> !fly.layout<((2,4),2,2):((1,4),2,16)>
  return %0 : !fly.layout<((2,4),2,2):((1,4),2,16)>
}

// CHECK-LABEL: func.func @flat_divide(
// CHECK: fly.flat_divide{{.*}} -> !fly.layout<(2,4,2,2):(1,4,2,16)>
func.func @flat_divide(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.tile<[2:1|4:1]>) -> !fly.layout<(2,4,2,2):(1,4,2,16)> {
  %0 = fly.flat_divide(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.tile<[2:1|4:1]>) -> !fly.layout<(2,4,2,2):(1,4,2,16)>
  return %0 : !fly.layout<(2,4,2,2):(1,4,2,16)>
}

// CHECK-LABEL: func.func @logical_product(
// CHECK: fly.logical_product{{.*}} -> !fly.layout<((4,8),(2,2)):((1,4),(32,64))>
func.func @logical_product(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,8),(2,2)):((1,4),(32,64))> {
  %0 = fly.logical_product(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,8),(2,2)):((1,4),(32,64))>
  return %0 : !fly.layout<((4,8),(2,2)):((1,4),(32,64))>
}

// CHECK-LABEL: func.func @zipped_product(
// CHECK: fly.zipped_product{{.*}} -> !fly.layout<((4,8),(2,2)):((1,4),(32,64))>
func.func @zipped_product(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,8),(2,2)):((1,4),(32,64))> {
  %0 = fly.zipped_product(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,8),(2,2)):((1,4),(32,64))>
  return %0 : !fly.layout<((4,8),(2,2)):((1,4),(32,64))>
}

// CHECK-LABEL: func.func @tiled_product(
// CHECK: fly.tiled_product{{.*}} -> !fly.layout<((4,8),2,2):((1,4),32,64)>
func.func @tiled_product(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,8),2,2):((1,4),32,64)> {
  %0 = fly.tiled_product(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,8),2,2):((1,4),32,64)>
  return %0 : !fly.layout<((4,8),2,2):((1,4),32,64)>
}

// CHECK-LABEL: func.func @flat_product(
// CHECK: fly.flat_product{{.*}} -> !fly.layout<(4,8,2,2):(1,4,32,64)>
func.func @flat_product(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.layout<(2,2):(1,2)>) -> !fly.layout<(4,8,2,2):(1,4,32,64)> {
  %0 = fly.flat_product(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.layout<(2,2):(1,2)>) -> !fly.layout<(4,8,2,2):(1,4,32,64)>
  return %0 : !fly.layout<(4,8,2,2):(1,4,32,64)>
}

// CHECK-LABEL: func.func @blocked_product(
// CHECK: fly.blocked_product{{.*}} -> !fly.layout<((4,2),(8,2)):((1,32),(4,64))>
func.func @blocked_product(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,2),(8,2)):((1,32),(4,64))> {
  %0 = fly.blocked_product(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,2),(8,2)):((1,32),(4,64))>
  return %0 : !fly.layout<((4,2),(8,2)):((1,32),(4,64))>
}

// CHECK-LABEL: func.func @raked_product(
// CHECK: fly.raked_product{{.*}} -> !fly.layout<((2,4),(2,8)):((32,1),(64,4))>
func.func @raked_product(%layout: !fly.layout<(4,8):(1,4)>, %tiler: !fly.layout<(2,2):(1,2)>) -> !fly.layout<((2,4),(2,8)):((32,1),(64,4))> {
  %0 = fly.raked_product(%layout, %tiler) : (!fly.layout<(4,8):(1,4)>, !fly.layout<(2,2):(1,2)>) -> !fly.layout<((2,4),(2,8)):((32,1),(64,4))>
  return %0 : !fly.layout<((2,4),(2,8)):((32,1),(64,4))>
}
