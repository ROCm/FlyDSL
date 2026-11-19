// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// fly-layout-lowering replaces each layout operation by its Core value. A static value becomes
// fly.static, or an arith.constant for a scalar; a dynamic value is rebuilt from its leaves, which
// bind to the operands of their constructor or to fly.get_leaves of an opaque value. The algebra
// is tested in tests/core, and Dialect/Fly/ops.mlir lowers one use of every operation.

// -----

// === Static values ===

// A fully static get_shape folds to fly.static.
// CHECK-LABEL: @test_get_shape
func.func @test_get_shape() -> !fly.int_tuple<(4, 8)> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK-NOT: fly.get_shape
  // CHECK-NOT: fly.make_layout
  // CHECK: %[[S:.*]] = fly.static : !fly.int_tuple<(4,8)>
  // CHECK: return %[[S]]
  %shape = fly.get_shape(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(4, 8)>
  return %shape : !fly.int_tuple<(4, 8)>
}

// -----

// === Tensors ===

// A fully static get_layout folds to fly.static; make_view is DCE'd.
// CHECK-LABEL: @test_get_layout
func.func @test_get_layout(%ptr: !fly.ptr<f32, global>) -> !fly.layout<(4,8):(1,4)> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %view = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, global>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.memref<f32, global, (4, 8) : (1, 4)>
  // CHECK-NOT: fly.get_layout
  // CHECK-NOT: fly.make_view
  // CHECK: %[[LAYOUT:.*]] = fly.static : !fly.layout<(4,8):(1,4)>
  // CHECK: return %[[LAYOUT]]
  %result = fly.get_layout(%view) : (!fly.memref<f32, global, (4, 8) : (1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  return %result : !fly.layout<(4,8):(1,4)>
}

// get_iter forwards the iterator (ptr) from make_view; all Fly ops are eliminated.
// CHECK-LABEL: @test_get_iter
// CHECK-SAME: (%[[PTR:.*]]: !fly.ptr<f32, global>)
func.func @test_get_iter(%ptr: !fly.ptr<f32, global>) -> !fly.ptr<f32, global> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %view = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, global>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.memref<f32, global, (4, 8) : (1, 4)>
  // CHECK-NOT: fly.get_iter
  // CHECK-NOT: fly.make_view
  // CHECK: return %[[PTR]]
  %result = fly.get_iter(%view) : (!fly.memref<f32, global, (4, 8) : (1, 4)>) -> !fly.ptr<f32, global>
  return %result : !fly.ptr<f32, global>
}

// A coordinate tensor's iterator is its base coordinate, read from the leaves
// of any producer; a static base folds to fly.static.
// CHECK-LABEL: @test_coord_tensor_get_iter
// CHECK-SAME: (%[[T:.*]]: !fly.coord_tensor
// CHECK-NOT: fly.get_iter
// CHECK: %[[LEAVES:.*]]:3 = fly.get_leaves(%[[T]])
// CHECK: %[[BASE:.*]] = fly.make_int_tuple(%[[LEAVES]]#0) : (i64) -> !fly.int_tuple<?{i64}>
// CHECK: return %[[BASE]]
func.func @test_coord_tensor_get_iter(%t: !fly.coord_tensor<?{i64},(?,4):(1,?{i64})>) -> !fly.int_tuple<?{i64}> {
  %result = fly.get_iter(%t) : (!fly.coord_tensor<?{i64},(?,4):(1,?{i64})>) -> !fly.int_tuple<?{i64}>
  return %result : !fly.int_tuple<?{i64}>
}

// CHECK-LABEL: @test_coord_tensor_static_base_get_iter
// CHECK-NOT: fly.get_iter
// CHECK: %[[BASE:.*]] = fly.static : !fly.int_tuple<0>
// CHECK: return %[[BASE]]
func.func @test_coord_tensor_static_base_get_iter(%t: !fly.coord_tensor<0,(?,4):(1,?{i64})>) -> !fly.int_tuple<0> {
  %result = fly.get_iter(%t) : (!fly.coord_tensor<0,(?,4):(1,?{i64})>) -> !fly.int_tuple<0>
  return %result : !fly.int_tuple<0>
}

// Decomposition recursively peels composed outers until only a linear LayoutAttr
// remains in the resulting tensor type.
// CHECK-LABEL: @test_decomposition_composed_outer
func.func @test_decomposition_composed_outer(%ptr: !fly.ptr<f32, global>)
    -> !fly.memref<f32, global, (4,8):(1,4)> {
  %off1 = fly.make_int_tuple() : () -> !fly.int_tuple<5>
  %off2 = fly.make_int_tuple() : () -> !fly.int_tuple<7>

  %sa = fly.make_int_tuple() : () -> !fly.int_tuple<128>
  %da = fly.make_int_tuple() : () -> !fly.int_tuple<2>
  %inner = fly.make_layout(%sa, %da) : (!fly.int_tuple<128>, !fly.int_tuple<2>) -> !fly.layout<128:2>

  %sb = fly.make_int_tuple() : () -> !fly.int_tuple<128>
  %db = fly.make_int_tuple() : () -> !fly.int_tuple<3>
  %mid = fly.make_layout(%sb, %db) : (!fly.int_tuple<128>, !fly.int_tuple<3>) -> !fly.layout<128:3>

  %sc = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %dc = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %outer = fly.make_layout(%sc, %dc) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>

  %cl1 = fly.make_composed_layout(%mid, %off2, %outer)
      : (!fly.layout<128:3>, !fly.int_tuple<7>, !fly.layout<(4, 8) : (1, 4)>)
      -> !fly.composed_layout<128:3 o 7 o (4, 8) : (1, 4)>
  %cl2 = fly.make_composed_layout(%inner, %off1, %cl1)
      : (!fly.layout<128:2>, !fly.int_tuple<5>,
         !fly.composed_layout<128:3 o 7 o (4, 8) : (1, 4)>)
      -> !fly.composed_layout<128:2 o 5 o [128:3 o 7 o (4, 8) : (1, 4)]>
  %view = fly.make_view(%ptr, %cl2)
      : (!fly.ptr<f32, global>,
         !fly.composed_layout<128:2 o 5 o [128:3 o 7 o (4, 8) : (1, 4)]>)
      -> !fly.memref<f32, global, 128:2 o 5 o [128:3 o 7 o (4, 8) : (1, 4)]>
  // CHECK-NOT: fly.decomposition
  // CHECK: fly.static : !fly.int_tuple<52>
  // CHECK: fly.make_view{{.*}} -> !fly.memref<f32, global, (4,8):(1,4)>
  %decomp = fly.decomposition(%view)
      : (!fly.memref<f32, global, 128:2 o 5 o [128:3 o 7 o (4, 8) : (1, 4)]>)
      -> !fly.memref<f32, global, (4, 8) : (1, 4)>
  return %decomp : !fly.memref<f32, global, (4, 8) : (1, 4)>
}

// -----

// === Scalars ===

// Static get_scalar lowers to arith.constant, through nested singleton tuples.
// CHECK-LABEL: @test_get_scalar_static
func.func @test_get_scalar_static() -> i32 {
  %t = fly.make_int_tuple() : () -> !fly.int_tuple<((42))>
  // CHECK-NOT: fly.get_scalar
  // CHECK-NOT: fly.make_int_tuple
  // CHECK: %[[C:.*]] = arith.constant 42 : i32
  // CHECK: return %[[C]]
  %s = fly.get_scalar(%t) : (!fly.int_tuple<((42))>) -> i32
  return %s : i32
}

// Dynamic get_scalar forwards the original dynamic operand.
// CHECK-LABEL: @test_get_scalar_dynamic
// CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @test_get_scalar_dynamic(%x: i32) -> i32 {
  %t = fly.make_int_tuple(%x) : (i32) -> !fly.int_tuple<?>
  // CHECK-NOT: fly.get_scalar
  // CHECK: return %[[ARG]]
  %s = fly.get_scalar(%t) : (!fly.int_tuple<?>) -> i32
  return %s : i32
}

// -----

// === Dynamic values ===

// Dynamic crd2idx: c0*1 + c1*4 lowered to arith.muli + arith.addi
// CHECK-LABEL: @test_crd2idx_dynamic
// CHECK-SAME: (%[[C0:.*]]: i32, %[[C1:.*]]: i32)
func.func @test_crd2idx_dynamic(%c0: i32, %c1: i32) -> i32 {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %coord = fly.make_int_tuple(%c0, %c1) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(?, ?)>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<?>
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[MUL:.*]] = arith.muli %[[C1]], %[[C4]] : i32
  // CHECK: %[[ADD:.*]] = arith.addi %[[C0]], %[[MUL]] overflow<nsw> : i32
  // CHECK: return %[[ADD]]
  %scalar = fly.get_scalar(%idx) : (!fly.int_tuple<?>) -> i32
  return %scalar : i32
}

// -----

// === Leaf bindings ===

// Only dynamic leaves are returned; static leaves stay in the type.
// Mixed i32/i64 dynamic leaves.
// CHECK-LABEL: @test_get_leaves_dynamic
// CHECK-SAME: (%[[X:.*]]: i32, %[[Y:.*]]: i64)
func.func @test_get_leaves_dynamic(%x: i32, %y: i64) -> (i32, i64) {
  %t = fly.make_int_tuple(%x, %y) : (i32, i64) -> !fly.int_tuple<(4, ?, ?{i64})>
  // CHECK-NOT: fly.get_leaves
  // CHECK-NOT: arith.constant
  // CHECK: return %[[X]], %[[Y]]
  %0:2 = fly.get_leaves(%t) : (!fly.int_tuple<(4, ?, ?{i64})>) -> (i32, i64)
  return %0#0, %0#1 : i32, i64
}

// -----

// A tile constructor's operands are the dynamic leaves of its layouts.
// CHECK-LABEL: @test_get_leaves_make_tile
// CHECK-SAME: (%[[X:.*]]: i32, %[[Y:.*]]: i64)
func.func @test_get_leaves_make_tile(%x: i32, %y: i64) -> (i32, i64) {
  %t = fly.make_tile(%x, %y) : (i32, i64) -> !fly.tile<[?:1|(4,?{i64}):(1,4)|*]>
  // CHECK-NOT: fly.get_leaves
  // CHECK: return %[[X]], %[[Y]]
  %0:2 = fly.get_leaves(%t) : (!fly.tile<[?:1|(4,?{i64}):(1,4)|*]>) -> (i32, i64)
  return %0#0, %0#1 : i32, i64
}

// -----

// A dynamic tiler binds the operands of its constructor.
// CHECK-LABEL: @test_composition_dynamic_tile
// CHECK-SAME: (%[[X:.*]]: i32,
func.func @test_composition_dynamic_tile(%x: i32, %l: !fly.layout<(8,16):(1,8)>) -> !fly.layout<(?,16):(1,8)> {
  %t = fly.make_tile(%x) : (i32) -> !fly.tile<[?:1|*]>
  // CHECK-NOT: fly.composition
  // CHECK: %[[S:.*]] = fly.make_int_tuple(%[[X]]) : (i32) -> !fly.int_tuple<(?,16)>
  // CHECK: %[[D:.*]] = fly.static : !fly.int_tuple<(1,8)>
  // CHECK: %[[R:.*]] = fly.make_layout(%[[S]], %[[D]])
  // CHECK: return %[[R]]
  %r = fly.composition(%l, %t) : (!fly.layout<(8,16):(1,8)>, !fly.tile<[?:1|*]>) -> !fly.layout<(?,16):(1,8)>
  return %r : !fly.layout<(?,16):(1,8)>
}

// -----

// A dynamic tiler without a constructor binds the leaves fly.get_leaves extracts.
// CHECK-LABEL: @test_logical_divide_dynamic_tile_arg
// CHECK-SAME: (%[[T:.*]]: !fly.tile<[?:1|*]>,
func.func @test_logical_divide_dynamic_tile_arg(%t: !fly.tile<[?:1|*]>, %l: !fly.layout<(8,16):(1,8)>) -> !fly.layout<((?,?),16):((1,?),8)> {
  // CHECK-NOT: fly.logical_divide
  // CHECK: %[[N:.*]] = fly.get_leaves(%[[T]]) : (!fly.tile<[?:1|*]>) -> i32
  // CHECK: %[[Q:.*]] = arith.ceildivsi %{{.*}}, %[[N]] : i32
  // CHECK: fly.make_int_tuple(%[[N]], %[[Q]]) : (i32, i32) -> !fly.int_tuple<((?,?),16)>
  // CHECK: fly.make_int_tuple(%[[N]]) : (i32) -> !fly.int_tuple<((1,?),8)>
  %r = fly.logical_divide(%l, %t) : (!fly.layout<(8,16):(1,8)>, !fly.tile<[?:1|*]>) -> !fly.layout<((?,?),16):((1,?),8)>
  return %r : !fly.layout<((?,?),16):((1,?),8)>
}
