// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s --check-prefix=WIDTH

// fly-layout-lowering replaces each layout operation by its Core value. A static value becomes
// fly.static, or an arith.constant for a scalar; a dynamic value is rebuilt from its leaves, which
// bind to the operands of their constructor or to fly.get_leaves of an opaque value. The algebra
// is tested in tests/core, and Dialect/Fly/ops.mlir lowers one use of every operation.
//
// The WIDTH prefix holds the checks that an operation is absent from a whole function: a
// WIDTH-NOT reaches from the WIDTH-LABEL of its function to the next WIDTH-LABEL.

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

// -----

// Dynamic extents whose product can leave i32. The rest of a divide holds the tile counts of the
// modes, here the extents themselves, so the lowering forms no product and the coordinate is split
// by the first extent: nothing in the function is i64.
// CHECK-LABEL: @test_logical_divide_dynamic_extent_product
// CHECK-SAME: (%[[M:.*]]: i32, %[[N:.*]]: i32, %[[I:.*]]: i32)
// CHECK: arith.remsi %[[I]], %[[M]] : i32
// CHECK: arith.divsi %[[I]], %[[M]] : i32
// CHECK: return
// WIDTH-LABEL: @test_logical_divide_dynamic_extent_product
// WIDTH-NOT: i64
func.func @test_logical_divide_dynamic_extent_product(%m: i32, %n: i32, %i: i32) -> i32 {
  %s = fly.make_int_tuple(%m, %n) : (i32, i32) -> !fly.int_tuple<(?,?)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(16,1)>
  %l = fly.make_layout(%s, %d) : (!fly.int_tuple<(?,?)>, !fly.int_tuple<(16,1)>) -> !fly.layout<(?,?):(16,1)>
  %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %t = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
  %div = fly.logical_divide(%l, %t) : (!fly.layout<(?,?):(16,1)>, !fly.layout<1:1>) -> !fly.layout<(1,(?,?)):(1,(16,1))>
  %c = fly.make_int_tuple(%i) : (i32) -> !fly.int_tuple<(0,?)>
  %idx = fly.crd2idx(%c, %div) : (!fly.int_tuple<(0,?)>, !fly.layout<(1,(?,?)):(1,(16,1))>) -> !fly.int_tuple<?>
  %r = fly.get_scalar(%idx) : (!fly.int_tuple<?>) -> i32
  return %r : i32
}

// -----

// The size of two dynamic extents is i64; with a static extent it keeps the dynamic one's type.
// CHECK-LABEL: @test_size_dynamic_extents
// CHECK-SAME: (%[[M:.*]]: i32, %[[N:.*]]: i32)
// WIDTH-LABEL: @test_size_dynamic_extents
func.func @test_size_dynamic_extents(%m: i32, %n: i32) -> (i64, i32) {
  %s = fly.make_int_tuple(%m, %n) : (i32, i32) -> !fly.int_tuple<(?,?)>
  // CHECK: %[[M64:.*]] = arith.extsi %[[M]] : i32 to i64
  // CHECK: %[[N64:.*]] = arith.extsi %[[N]] : i32 to i64
  // CHECK: %[[P:.*]] = arith.muli %[[M64]], %[[N64]] : i64
  %z = fly.size(%s) : (!fly.int_tuple<(?,?)>) -> !fly.int_tuple<?{i64}>
  %zs = fly.get_scalar(%z) : (!fly.int_tuple<?{i64}>) -> i64
  %h = fly.make_int_tuple(%m) : (i32) -> !fly.int_tuple<(4,?)>
  // CHECK: %[[Q:.*]] = arith.muli %[[M]], %{{.*}} : i32
  // CHECK: return %[[P]], %[[Q]]
  %w = fly.size(%h) : (!fly.int_tuple<(4,?)>) -> !fly.int_tuple<?{div=4}>
  %ws = fly.get_scalar(%w) : (!fly.int_tuple<?{div=4}>) -> i32
  return %zs, %ws : i64, i32
}

// -----

// A static extent between or before two dynamic ones. Each dynamic extent is cast to i64 first,
// so the product, taken in leaf order, is i64 from its first dynamic factor on: no factor is
// multiplied in at i32.
// CHECK-LABEL: @test_size_mixed_extents
// CHECK-SAME: (%[[M:.*]]: i32, %[[N:.*]]: i32)
// CHECK: %[[M64:.*]] = arith.extsi %[[M]] : i32 to i64
// CHECK: %[[P:.*]] = arith.muli %[[M64]], %{{.*}} : i64
// CHECK: %[[N64:.*]] = arith.extsi %[[N]] : i32 to i64
// CHECK: %[[Z:.*]] = arith.muli %[[P]], %[[N64]] : i64
// CHECK: return %[[Z]], %[[Z]]
// WIDTH-LABEL: @test_size_mixed_extents
// WIDTH-NOT: arith.muli {{.*}} : i32
func.func @test_size_mixed_extents(%m: i32, %n: i32) -> (i64, i64) {
  %a = fly.make_int_tuple(%m, %n) : (i32, i32) -> !fly.int_tuple<(?,8192,?)>
  %za = fly.size(%a) : (!fly.int_tuple<(?,8192,?)>) -> !fly.int_tuple<?{i64 div=8192}>
  %sa = fly.get_scalar(%za) : (!fly.int_tuple<?{i64 div=8192}>) -> i64
  %b = fly.make_int_tuple(%m, %n) : (i32, i32) -> !fly.int_tuple<(8192,?,?)>
  %zb = fly.size(%b) : (!fly.int_tuple<(8192,?,?)>) -> !fly.int_tuple<?{i64 div=8192}>
  %sb = fly.get_scalar(%zb) : (!fly.int_tuple<?{i64 div=8192}>) -> i64
  return %sa, %sb : i64, i64
}

// -----

// A divide by a 64-element tile of a tensor with two dynamic extents: every tile count is dynamic.
// The rest of the divide holds them as they are, each an i32 ceildivsi, and a coordinate is split
// by them: the function has no 64-bit division or remainder, and no product of the counts.
// CHECK-LABEL: @test_logical_divide_tile_counts
// CHECK-SAME: (%[[B:.*]]: i32, %[[S:.*]]: i32, %{{.*}}: i64, %{{.*}}: i64, %[[L:.*]]: i32, %[[J:.*]]: i32)
// CHECK: %[[C0:.*]] = arith.ceildivsi %[[B]], %{{.*}} : i32
// CHECK: %[[C1:.*]] = arith.ceildivsi %[[S]], %{{.*}} : i32
// CHECK: arith.remsi %[[J]], %[[C0]] : i32
// CHECK: %[[J1:.*]] = arith.divsi %[[J]], %[[C0]] : i32
// CHECK: arith.remsi %[[J1]], %[[C1]] : i32
// CHECK: return
// WIDTH-LABEL: @test_logical_divide_tile_counts
// WIDTH-NOT: arith.{{[a-z]*}}div{{[a-z]*}} {{.*}} : i64
// WIDTH-NOT: arith.rem{{[a-z]*}} {{.*}} : i64
func.func @test_logical_divide_tile_counts(%b: i32, %s: i32, %d0: i64, %d1: i64, %l: i32, %j: i32) -> i64 {
  %sh = fly.make_int_tuple(%b, %s) : (i32, i32) -> !fly.int_tuple<(?,?,8,16)>
  %st = fly.make_int_tuple(%d0, %d1) : (i64, i64) -> !fly.int_tuple<(?{i64},?{i64},16,1)>
  %lay = fly.make_layout(%sh, %st) : (!fly.int_tuple<(?,?,8,16)>, !fly.int_tuple<(?{i64},?{i64},16,1)>) -> !fly.layout<(?,?,8,16):(?{i64},?{i64},16,1)>
  %t64 = fly.make_int_tuple() : () -> !fly.int_tuple<64>
  %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %t = fly.make_layout(%t64, %one) : (!fly.int_tuple<64>, !fly.int_tuple<1>) -> !fly.layout<64:1>
  %div = fly.logical_divide(%lay, %t) : (!fly.layout<(?,?,8,16):(?{i64},?{i64},16,1)>, !fly.layout<64:1>) -> !fly.layout<((?,?,?,?),(?,?,?,?)):((?{i64},?{i64},16,1),(?{i64 div=64},?{i64},?{div=16},?))>
  %c = fly.make_int_tuple(%l, %j) : (i32, i32) -> !fly.int_tuple<(?,?)>
  %idx = fly.crd2idx(%c, %div) : (!fly.int_tuple<(?,?)>, !fly.layout<((?,?,?,?),(?,?,?,?)):((?{i64},?{i64},16,1),(?{i64 div=64},?{i64},?{div=16},?))>) -> !fly.int_tuple<?{i64}>
  %r = fly.get_scalar(%idx) : (!fly.int_tuple<?{i64}>) -> i64
  return %r : i64
}

// -----

// A divide by 1:1 of a tensor with static extents after the dynamic ones. The first group of the
// rest holds the dynamic extents and splits the coordinate by them. The strides of the second group
// are multiples of the product of those extents, an i64, and its extents divide it at i32 after a
// clamp to INT32_MAX. Dividing that product by an extent is the one 64-bit division left, a
// ceildivsi per mode; no divsi or remsi is i64.
// CHECK-LABEL: @test_logical_divide_static_extents_after_dynamic
// CHECK-SAME: (%[[B:.*]]: i32, %[[S:.*]]: i32, %[[SB:.*]]: i64, %[[I:.*]]: i32)
// CHECK: %[[P:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
// CHECK: %[[CLAMP:.*]] = arith.minsi %{{.*}}, %{{.*}} : i64
// CHECK: %[[D:.*]] = arith.trunci %[[CLAMP]] : i64 to i32
// CHECK: arith.ceildivsi %[[B]], %[[D]] : i32
// CHECK: arith.ceildivsi %{{.*}}, %{{.*}} : i64
// CHECK: arith.ceildivsi %{{.*}}, %{{.*}} : i64
// CHECK: arith.ceildivsi %{{.*}}, %{{.*}} : i64
// CHECK-NOT: arith.ceildivsi %{{.*}}, %{{.*}} : i64
// CHECK: arith.remsi %[[I]], %[[B]] : i32
// CHECK: %[[I1:.*]] = arith.divsi %[[I]], %[[B]] : i32
// CHECK: arith.remsi %[[I1]], %[[S]] : i32
// CHECK-NOT: arith.ceildivsi %{{.*}}, %{{.*}} : i64
// CHECK: return
// WIDTH-LABEL: @test_logical_divide_static_extents_after_dynamic
// WIDTH-NOT: arith.{{(div|rem)}}si {{.*}} : i64
func.func @test_logical_divide_static_extents_after_dynamic(%b: i32, %s: i32, %sb: i64, %i: i32) -> i64 {
  %sh = fly.make_int_tuple(%b, %s) : (i32, i32) -> !fly.int_tuple<(?,?,32,128)>
  %st = fly.make_int_tuple(%sb) : (i64) -> !fly.int_tuple<(?{i64},4096,128,1)>
  %l = fly.make_layout(%sh, %st) : (!fly.int_tuple<(?,?,32,128)>, !fly.int_tuple<(?{i64},4096,128,1)>) -> !fly.layout<(?,?,32,128):(?{i64},4096,128,1)>
  %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %t = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
  %div = fly.logical_divide(%l, %t) : (!fly.layout<(?,?,32,128):(?{i64},4096,128,1)>, !fly.layout<1:1>) -> !fly.layout<(1,((?,?,?,?),(?,?,?,?))):(1,((?{i64},4096,128,1),(?{i64},?{i64 div=4096},?{i64 div=128},?{i64})))>
  %c = fly.make_int_tuple(%i) : (i32) -> !fly.int_tuple<(0,?)>
  %idx = fly.crd2idx(%c, %div) : (!fly.int_tuple<(0,?)>, !fly.layout<(1,((?,?,?,?),(?,?,?,?))):(1,((?{i64},4096,128,1),(?{i64},?{i64 div=4096},?{i64 div=128},?{i64})))>) -> !fly.int_tuple<?{i64}>
  %r = fly.get_scalar(%idx) : (!fly.int_tuple<?{i64}>) -> i64
  return %r : i64
}

// -----

// An i64 extent inside a mode that a coordinate divides through. The extents make the size of
// such a mode i64 by themselves, so the coordinate is divided by it at i64 and not split by each
// extent: its coordinates in that mode and after it are i64, and so are their products with the
// strides. The offset is exact beyond 2^31.
// CHECK-LABEL: @test_crd2idx_wide_extent_inside_mode
// CHECK-SAME: (%[[I:.*]]: i32, %[[A:.*]]: i64, %[[B:.*]]: i32, %[[C:.*]]: i32, %{{.*}}: i32)
// CHECK: %[[B64:.*]] = arith.extsi %[[B]] : i32 to i64
// CHECK: %[[SIZE:.*]] = arith.muli %[[A]], %[[B64]] : i64
// CHECK: %[[I64:.*]] = arith.extsi %[[I]] : i32 to i64
// CHECK: %[[R:.*]] = arith.remsi %[[I64]], %[[SIZE]] : i64
// CHECK: %[[Q:.*]] = arith.divsi %[[I64]], %[[SIZE]] : i64
// CHECK: %[[C0:.*]] = arith.remsi %[[R]], %[[A]] : i64
// CHECK: %[[C64:.*]] = arith.extsi %[[C]] : i32 to i64
// CHECK: %[[C2:.*]] = arith.remsi %[[Q]], %[[C64]] : i64
// CHECK: arith.muli %[[C2]], %{{.*}} : i64
// CHECK: return
// WIDTH-LABEL: @test_crd2idx_wide_extent_inside_mode
func.func @test_crd2idx_wide_extent_inside_mode(%i: i32, %a: i64, %b: i32, %c: i32, %d: i32) -> i64 {
  %idx = fly.make_int_tuple(%i) : (i32) -> !fly.int_tuple<?>
  %s = fly.make_int_tuple(%a, %b, %c, %d) : (i64, i32, i32, i32) -> !fly.int_tuple<((?{i64},?),(?,?))>
  %st = fly.make_int_tuple() : () -> !fly.int_tuple<((1,4),(64,4096))>
  %lay = fly.make_layout(%s, %st) : (!fly.int_tuple<((?{i64},?),(?,?))>, !fly.int_tuple<((1,4),(64,4096))>) -> !fly.layout<((?{i64},?),(?,?)):((1,4),(64,4096))>
  %r = fly.crd2idx(%idx, %lay) : (!fly.int_tuple<?>, !fly.layout<((?{i64},?),(?,?)):((1,4),(64,4096))>) -> !fly.int_tuple<?{i64}>
  %x = fly.get_scalar(%r) : (!fly.int_tuple<?{i64}>) -> i64
  return %x : i64
}

// -----

// A static index through a mode with an i64 extent behind a static one. It is divided by the size
// of the mode, at i64, so its coordinates are run-time i64 values, the one of the static extent
// too: its product with an i32 stride is i64 and exact beyond 2^31. A static coordinate times the
// stride would be an i32 product.
// CHECK-LABEL: @test_crd2idx_static_index_wide_mode
// CHECK-SAME: (%[[A:.*]]: i64, %[[S:.*]]: i32)
// CHECK: %[[SIZE:.*]] = arith.muli %[[A]], %{{.*}} : i64
// CHECK: %[[R:.*]] = arith.remsi %{{.*}}, %[[SIZE]] : i64
// CHECK: %[[C0:.*]] = arith.remsi %[[R]], %{{.*}} : i64
// CHECK: %[[W:.*]] = arith.extsi %[[S]] : i32 to i64
// CHECK: arith.muli %[[C0]], %[[W]] : i64
// CHECK: return
// WIDTH-LABEL: @test_crd2idx_static_index_wide_mode
func.func @test_crd2idx_static_index_wide_mode(%a: i64, %s: i32) -> i64 {
  %idx = fly.make_int_tuple() : () -> !fly.int_tuple<37>
  %sh = fly.make_int_tuple(%a) : (i64) -> !fly.int_tuple<((16,?{i64}),4)>
  %st = fly.make_int_tuple(%s) : (i32) -> !fly.int_tuple<((?,1),1)>
  %lay = fly.make_layout(%sh, %st) : (!fly.int_tuple<((16,?{i64}),4)>, !fly.int_tuple<((?,1),1)>) -> !fly.layout<((16,?{i64}),4):((?,1),1)>
  %r = fly.crd2idx(%idx, %lay) : (!fly.int_tuple<37>, !fly.layout<((16,?{i64}),4):((?,1),1)>) -> !fly.int_tuple<?{i64}>
  %x = fly.get_scalar(%r) : (!fly.int_tuple<?{i64}>) -> i64
  return %x : i64
}

// -----

// A mode that an index is split through and that ends in an extent of 1. The coordinate of that
// extent is 0, but a run-time one, as it is when the index is divided by the size of the mode: its
// i64 stride makes the offset i64, so the i32 terms are added at i64. Nothing is computed for it.
// CHECK-LABEL: @test_crd2idx_unit_extent_ends_mode
// CHECK-SAME: (%[[I:.*]]: i32, %[[M:.*]]: i32, %[[N:.*]]: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i64, %{{.*}}: i32)
// CHECK: %[[C0:.*]] = arith.remsi %[[I]], %[[M]] : i32
// CHECK: %[[R0:.*]] = arith.divsi %[[I]], %[[M]] : i32
// CHECK: %[[C1:.*]] = arith.remsi %[[R0]], %[[N]] : i32
// CHECK: %[[R1:.*]] = arith.divsi %[[R0]], %[[N]] : i32
// CHECK: %[[SUM:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK: %[[WIDE:.*]] = arith.extsi %[[SUM]] : i32 to i64
// CHECK: arith.addi %[[WIDE]], %{{.*}} : i64
// CHECK: return
// WIDTH-LABEL: @test_crd2idx_unit_extent_ends_mode
// WIDTH-NOT: arith.{{(div|rem|mul|sub)}}{{[a-z]*}} {{.*}} : i64
func.func @test_crd2idx_unit_extent_ends_mode(%i: i32, %m: i32, %n: i32, %s0: i32, %s1: i32, %s2: i64, %s3: i32) -> i64 {
  %idx = fly.make_int_tuple(%i) : (i32) -> !fly.int_tuple<?>
  %sh = fly.make_int_tuple(%m, %n) : (i32, i32) -> !fly.int_tuple<((?,?,1),4)>
  %st = fly.make_int_tuple(%s0, %s1, %s2, %s3) : (i32, i32, i64, i32) -> !fly.int_tuple<((?,?,?{i64}),?)>
  %lay = fly.make_layout(%sh, %st) : (!fly.int_tuple<((?,?,1),4)>, !fly.int_tuple<((?,?,?{i64}),?)>) -> !fly.layout<((?,?,1),4):((?,?,?{i64}),?)>
  %r = fly.crd2idx(%idx, %lay) : (!fly.int_tuple<?>, !fly.layout<((?,?,1),4):((?,?,?{i64}),?)>) -> !fly.int_tuple<?{i64}>
  %x = fly.get_scalar(%r) : (!fly.int_tuple<?{i64}>) -> i64
  return %x : i64
}
