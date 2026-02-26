// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// Tests for fly-layout-lowering pass:
//   - Lowers layout algebra operations to simpler make_int_tuple/make_layout/arith forms
//   - Static operations fully resolved at compile time

// -----

// === Extractor Lowering: get_shape, get_stride ===

// get_shape forwards the shape operand from make_layout.
// CHECK-LABEL: @test_get_shape
func.func @test_get_shape() -> !fly.int_tuple<(4, 8)> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // get_shape is eliminated, returns the shape int_tuple directly.
  // CHECK-NOT: fly.get_shape
  // CHECK-NOT: fly.make_layout
  // CHECK: %[[S:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<(4,8)>
  // CHECK: return %[[S]]
  %shape = fly.get_shape(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(4, 8)>
  return %shape : !fly.int_tuple<(4, 8)>
}

// get_stride forwards the stride operand from make_layout.
// CHECK-LABEL: @test_get_stride
func.func @test_get_stride() -> !fly.int_tuple<(1, 4)> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK-NOT: fly.get_stride
  // CHECK-NOT: fly.make_layout
  // CHECK: %[[D:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<(1,4)>
  // CHECK: return %[[D]]
  %stride = fly.get_stride(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(1, 4)>
  return %stride : !fly.int_tuple<(1, 4)>
}

// -----

// === Extractor Lowering: get_layout, get_iter (from make_view) ===

// get_layout forwards the layout from make_view; make_view and get_iter are DCE'd.
// CHECK-LABEL: @test_get_layout
func.func @test_get_layout(%ptr: !fly.ptr<f32, global>) -> !fly.layout<(4,8):(1,4)> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %view = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, global>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.memref<f32, global, (4, 8) : (1, 4)>
  // CHECK-NOT: fly.get_layout
  // CHECK-NOT: fly.make_view
  // CHECK: %[[LAYOUT:.*]] = fly.make_layout
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

// -----

// === get_leaf Lowering ===

// get_leaf on IntTuple extracts the element at index.
// CHECK-LABEL: @test_get_leaf_int_tuple
func.func @test_get_leaf_int_tuple() -> !fly.int_tuple<4> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  // CHECK-NOT: fly.get_leaf
  // CHECK: %[[LEAF:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<4>
  // CHECK: return %[[LEAF]]
  %leaf = fly.get_leaf(%s, 0) : (!fly.int_tuple<(4, 8)>) -> !fly.int_tuple<4>
  return %leaf : !fly.int_tuple<4>
}

// get_leaf on Layout extracts a sub-layout (shape[i]:stride[i]).
// CHECK-LABEL: @test_get_leaf_layout
func.func @test_get_leaf_layout() -> !fly.layout<4:1> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK-NOT: fly.get_leaf
  // CHECK: %[[LS:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<4>
  // CHECK: %[[LD:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  // CHECK: %[[LEAF:.*]] = fly.make_layout(%[[LS]], %[[LD]])
  // CHECK: return %[[LEAF]]
  %leaf = fly.get_leaf(%layout, 0) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<4:1>
  return %leaf : !fly.layout<4:1>
}

// -----

// === get_scalar Lowering ===

// Static get_scalar lowers to arith.constant.
// CHECK-LABEL: @test_get_scalar_static
func.func @test_get_scalar_static() -> i32 {
  %t = fly.make_int_tuple() : () -> !fly.int_tuple<42>
  // CHECK-NOT: fly.get_scalar
  // CHECK-NOT: fly.make_int_tuple
  // CHECK: %[[C:.*]] = arith.constant 42 : i32
  // CHECK: return %[[C]]
  %s = fly.get_scalar(%t) : (!fly.int_tuple<42>) -> i32
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

// === SizeOp Lowering ===

// Size of IntTuple computes product of all elements.
// CHECK-LABEL: @test_size_int_tuple
func.func @test_size_int_tuple() -> !fly.int_tuple<32> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  // 4 * 8 = 32
  // CHECK-NOT: fly.size
  // CHECK: %[[R:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<32>
  // CHECK: return %[[R]]
  %size = fly.size(%s) : (!fly.int_tuple<(4, 8)>) -> !fly.int_tuple<32>
  return %size : !fly.int_tuple<32>
}

// Size of Layout computes product of shape elements.
// CHECK-LABEL: @test_size_layout
func.func @test_size_layout() -> !fly.int_tuple<32> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK-NOT: fly.size
  // CHECK: %[[R:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<32>
  // CHECK: return %[[R]]
  %size = fly.size(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<32>
  return %size : !fly.int_tuple<32>
}

// -----

// === Crd2Idx Lowering ===

// Static crd2idx: coord dot stride = 2*1 + 3*4 = 14
// CHECK-LABEL: @test_crd2idx_static
func.func @test_crd2idx_static() -> !fly.int_tuple<14> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %coord = fly.make_int_tuple() : () -> !fly.int_tuple<(2, 3)>
  // CHECK-NOT: fly.crd2idx
  // CHECK: %[[R:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<14>
  // CHECK: return %[[R]]
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(2, 3)>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<14>
  return %idx : !fly.int_tuple<14>
}

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
  // CHECK: %[[ADD:.*]] = arith.addi %[[C0]], %[[MUL]] : i32
  // CHECK: return %[[ADD]]
  %scalar = fly.get_scalar(%idx) : (!fly.int_tuple<?>) -> i32
  return %scalar : i32
}

// -----

// === IntTuple Binary Ops Lowering ===

// Static int_tuple_add: (2,3) + (3,8) = (5,11)
// CHECK-LABEL: @test_int_tuple_add
func.func @test_int_tuple_add() -> !fly.int_tuple<(5, 11)> {
  %a = fly.make_int_tuple() : () -> !fly.int_tuple<(2, 3)>
  %b = fly.make_int_tuple() : () -> !fly.int_tuple<(3, 8)>
  // CHECK-NOT: fly.int_tuple_add
  // CHECK: %[[R:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<(5,11)>
  // CHECK: return %[[R]]
  %result = fly.int_tuple_add(%a, %b) : (!fly.int_tuple<(2, 3)>, !fly.int_tuple<(3, 8)>) -> !fly.int_tuple<(5, 11)>
  return %result : !fly.int_tuple<(5, 11)>
}

// Static int_tuple_mul: (2,3) * (3,8) = (6,24)
// CHECK-LABEL: @test_int_tuple_mul
func.func @test_int_tuple_mul() -> !fly.int_tuple<(6, 24)> {
  %a = fly.make_int_tuple() : () -> !fly.int_tuple<(2, 3)>
  %b = fly.make_int_tuple() : () -> !fly.int_tuple<(3, 8)>
  // CHECK-NOT: fly.int_tuple_mul
  // CHECK: %[[R:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<(6,24)>
  // CHECK: return %[[R]]
  %result = fly.int_tuple_mul(%a, %b) : (!fly.int_tuple<(2, 3)>, !fly.int_tuple<(3, 8)>) -> !fly.int_tuple<(6, 24)>
  return %result : !fly.int_tuple<(6, 24)>
}

// -----

// === Layout Algebra: logical_divide ===

// logical_divide expands a layout by a divisor layout.
// CHECK-LABEL: @test_logical_divide
func.func @test_logical_divide() -> !fly.layout<((2,4),1):((1,2),0)> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<8>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<1>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<8>, !fly.int_tuple<1>) -> !fly.layout<8:1>
  %ds = fly.make_int_tuple() : () -> !fly.int_tuple<(2,4)>
  %dd = fly.make_int_tuple() : () -> !fly.int_tuple<(1,2)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(2,4)>, !fly.int_tuple<(1,2)>) -> !fly.layout<(2,4):(1,2)>
  // CHECK-NOT: fly.logical_divide
  // CHECK: %[[RS:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<((2,4),1)>
  // CHECK: %[[RD:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<((1,2),0)>
  // CHECK: %[[R:.*]] = fly.make_layout(%[[RS]], %[[RD]])
  // CHECK: return %[[R]]
  %result = fly.logical_divide(%layout, %divisor) : (!fly.layout<8:1>, !fly.layout<(2,4):(1,2)>) -> !fly.layout<((2,4),1):((1,2),0)>
  return %result : !fly.layout<((2,4),1):((1,2),0)>
}

// -----

// === Layout Algebra: logical_product ===

// logical_product combines layout with a tile.
// CHECK-LABEL: @test_logical_product
func.func @test_logical_product() -> !fly.layout<((4,8),(2,2)):((1,4),(32,64))> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4,8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1,4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4,8)>, !fly.int_tuple<(1,4)>) -> !fly.layout<(4,8):(1,4)>
  %ts = fly.make_int_tuple() : () -> !fly.int_tuple<(2,2)>
  %td = fly.make_int_tuple() : () -> !fly.int_tuple<(1,2)>
  %tile = fly.make_layout(%ts, %td) : (!fly.int_tuple<(2,2)>, !fly.int_tuple<(1,2)>) -> !fly.layout<(2,2):(1,2)>
  // CHECK-NOT: fly.logical_product
  // CHECK-DAG: fly.make_int_tuple() : () -> !fly.int_tuple<((4,8),(2,2))>
  // CHECK-DAG: fly.make_int_tuple() : () -> !fly.int_tuple<((1,4),(32,64))>
  // CHECK: %[[R:.*]] = fly.make_layout
  // CHECK: return %[[R]]
  %result = fly.logical_product(%layout, %tile) : (!fly.layout<(4,8):(1,4)>, !fly.layout<(2,2):(1,2)>) -> !fly.layout<((4,8),(2,2)):((1,4),(32,64))>
  return %result : !fly.layout<((4,8),(2,2)):((1,4),(32,64))>
}

// -----

// === Layout Algebra: right_inverse ===

// right_inverse computes the inverse mapping of a layout.
// CHECK-LABEL: @test_right_inverse
func.func @test_right_inverse() -> !fly.layout<(4,2):(2,1)> {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(2,4)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(4,1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(2,4)>, !fly.int_tuple<(4,1)>) -> !fly.layout<(2,4):(4,1)>
  // CHECK-NOT: fly.right_inverse
  // CHECK: %[[RS:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<(4,2)>
  // CHECK: %[[RD:.*]] = fly.make_int_tuple() : () -> !fly.int_tuple<(2,1)>
  // CHECK: %[[R:.*]] = fly.make_layout(%[[RS]], %[[RD]])
  // CHECK: return %[[R]]
  %result = fly.right_inverse(%layout) : (!fly.layout<(2,4):(4,1)>) -> !fly.layout<(4,2):(2,1)>
  return %result : !fly.layout<(4,2):(2,1)>
}

