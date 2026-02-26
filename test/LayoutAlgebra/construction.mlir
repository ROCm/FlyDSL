// RUN: %fly-opt %s | FileCheck %s

// Tests for layout construction operations:
//   fly.static, fly.make_int_tuple, fly.make_shape, fly.make_stride,
//   fly.make_layout, fly.make_ordered_layout, fly.make_identity_layout,
//   fly.get_shape, fly.get_stride, fly.get_leaves

// -----

// CHECK-LABEL: @test_static_int_tuple
func.func @test_static_int_tuple() -> !fly.int_tuple<(4, 8)> {
  // CHECK: fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4,8)>
  %0 = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  return %0 : !fly.int_tuple<(4, 8)>
}

// CHECK-LABEL: @test_static_scalar
func.func @test_static_scalar() -> !fly.int_tuple<42> {
  // CHECK: fly.static {elems = [42 : i32]} : () -> !fly.int_tuple<42>
  %0 = fly.static {elems = [42 : i32]} : () -> !fly.int_tuple<42>
  return %0 : !fly.int_tuple<42>
}

// CHECK-LABEL: @test_static_nested
func.func @test_static_nested() -> !fly.int_tuple<((2, 4), 8)> {
  // CHECK: fly.static
  %0 = fly.static {elems = [2 : i32, 4 : i32, 8 : i32], shape = [2 : i32, 1 : i32]} : () -> !fly.int_tuple<((2, 4), 8)>
  return %0 : !fly.int_tuple<((2, 4), 8)>
}

// CHECK-LABEL: @test_dynamic_make_int_tuple
func.func @test_dynamic_make_int_tuple(%a: i32, %b: i32) -> !fly.int_tuple<(?, ?)> {
  // CHECK: fly.make_int_tuple(%{{.*}}, %{{.*}}) : (i32, i32) -> !fly.int_tuple<(?,?)>
  %0 = fly.make_int_tuple(%a, %b) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  return %0 : !fly.int_tuple<(?, ?)>
}

// CHECK-LABEL: @test_dynamic_make_shape
func.func @test_dynamic_make_shape(%m: i32, %n: i32) -> !fly.int_tuple<(?, ?)> {
  // CHECK: fly.make_shape(%{{.*}}, %{{.*}}) : (i32, i32) -> !fly.int_tuple<(?,?)>
  %0 = fly.make_shape(%m, %n) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  return %0 : !fly.int_tuple<(?, ?)>
}

// CHECK-LABEL: @test_dynamic_make_stride
func.func @test_dynamic_make_stride(%s0: i32, %s1: i32) -> !fly.int_tuple<(?, ?)> {
  // CHECK: fly.make_stride(%{{.*}}, %{{.*}}) : (i32, i32) -> !fly.int_tuple<(?,?)>
  %0 = fly.make_stride(%s0, %s1) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  return %0 : !fly.int_tuple<(?, ?)>
}

// CHECK-LABEL: @test_make_layout_static
func.func @test_make_layout_static() -> !fly.layout<(4, 8) : (1, 4)> {
  %shape = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %stride = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  // CHECK: fly.make_layout(%{{.*}}, %{{.*}}) : (!fly.int_tuple<(4,8)>, !fly.int_tuple<(1,4)>) -> !fly.layout<(4,8):(1,4)>
  %layout = fly.make_layout(%shape, %stride) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  return %layout : !fly.layout<(4, 8) : (1, 4)>
}

// CHECK-LABEL: @test_make_layout_shape_only
func.func @test_make_layout_shape_only(%m: i32, %n: i32) -> !fly.layout<(?, ?) : (1, ?)> {
  %shape = fly.make_shape(%m, %n) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  // CHECK: fly.make_layout(%{{.*}}) : (!fly.int_tuple<(?,?)>) -> !fly.layout<(?,?):(1,?)>
  %layout = fly.make_layout(%shape) : (!fly.int_tuple<(?, ?)>) -> !fly.layout<(?, ?) : (1, ?)>
  return %layout : !fly.layout<(?, ?) : (1, ?)>
}

// CHECK-LABEL: @test_make_layout_3d
func.func @test_make_layout_3d() -> !fly.layout<(2, 4, 8) : (1, 2, 8)> {
  %shape = fly.static {elems = [2 : i32, 4 : i32, 8 : i32]} : () -> !fly.int_tuple<(2, 4, 8)>
  %stride = fly.static {elems = [1 : i32, 2 : i32, 8 : i32]} : () -> !fly.int_tuple<(1, 2, 8)>
  %layout = fly.make_layout(%shape, %stride) : (!fly.int_tuple<(2, 4, 8)>, !fly.int_tuple<(1, 2, 8)>) -> !fly.layout<(2, 4, 8) : (1, 2, 8)>
  return %layout : !fly.layout<(2, 4, 8) : (1, 2, 8)>
}

// CHECK-LABEL: @test_make_ordered_layout_row_major
func.func @test_make_ordered_layout_row_major() -> !fly.layout<(4, 8) : (8, 1)> {
  %shape = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %order = fly.static {elems = [1 : i32, 0 : i32]} : () -> !fly.int_tuple<(1, 0)>
  // CHECK: fly.make_ordered_layout
  %layout = fly.make_ordered_layout(%shape, %order) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 0)>) -> !fly.layout<(4, 8) : (8, 1)>
  return %layout : !fly.layout<(4, 8) : (8, 1)>
}

// CHECK-LABEL: @test_make_ordered_layout_col_major
func.func @test_make_ordered_layout_col_major() -> !fly.layout<(4, 8) : (1, 4)> {
  %shape = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %order = fly.static {elems = [0 : i32, 1 : i32]} : () -> !fly.int_tuple<(0, 1)>
  %layout = fly.make_ordered_layout(%shape, %order) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(0, 1)>) -> !fly.layout<(4, 8) : (1, 4)>
  return %layout : !fly.layout<(4, 8) : (1, 4)>
}

// CHECK-LABEL: @test_make_identity_layout
func.func @test_make_identity_layout() -> !fly.layout<(4, 8) : (1E0, 1E1)> {
  %shape = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  // CHECK: fly.make_identity_layout
  %layout = fly.make_identity_layout(%shape) : (!fly.int_tuple<(4, 8)>) -> !fly.layout<(4, 8) : (1E0, 1E1)>
  return %layout : !fly.layout<(4, 8) : (1E0, 1E1)>
}

// CHECK-LABEL: @test_get_shape
func.func @test_get_shape(%l: !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(4, 8)> {
  // CHECK: fly.get_shape(%{{.*}}) : (!fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<(4,8)>
  %shape = fly.get_shape(%l) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(4, 8)>
  return %shape : !fly.int_tuple<(4, 8)>
}

// CHECK-LABEL: @test_get_stride
func.func @test_get_stride(%l: !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(1, 4)> {
  // CHECK: fly.get_stride(%{{.*}}) : (!fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<(1,4)>
  %stride = fly.get_stride(%l) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(1, 4)>
  return %stride : !fly.int_tuple<(1, 4)>
}

// CHECK-LABEL: @test_get_leaves
func.func @test_get_leaves(%l: !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(4, 8)> {
  // CHECK: fly.get_leaves(%{{.*}}) : (!fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<(4,8)>
  %leaves = fly.get_leaves(%l) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(4, 8)>
  return %leaves : !fly.int_tuple<(4, 8)>
}

// CHECK-LABEL: @test_make_layout_like
func.func @test_make_layout_like(%src: !fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<(4, 8) : (1, 4)> {
  // CHECK: fly.make_layout_like(%{{.*}}) : (!fly.layout<(4,8):(1,4)>) -> !fly.layout<(4,8):(1,4)>
  %copy = fly.make_layout_like(%src) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  return %copy : !fly.layout<(4, 8) : (1, 4)>
}
