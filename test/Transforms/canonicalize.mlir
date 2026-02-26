// RUN: %fly-opt %s --fly-canonicalize | FileCheck %s

// Tests for fly-canonicalize pass:
//   - Static argument folding (fly.static -> fly.make_int_tuple)
//   - Constant propagation through layout construction

// -----

// CHECK-LABEL: @test_static_to_make_int_tuple
func.func @test_static_to_make_int_tuple() -> !fly.int_tuple<(4, 8)> {
  // fly-canonicalize folds fly.static into fly.make_int_tuple with no dynamic args
  %0 = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  // CHECK: fly.make_int_tuple() : () -> !fly.int_tuple<(4,8)>
  return %0 : !fly.int_tuple<(4, 8)>
}

// CHECK-LABEL: @test_make_layout_canonicalized
func.func @test_make_layout_canonicalized() -> !fly.layout<(4, 8) : (1, 4)> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.make_int_tuple
  // CHECK: fly.make_layout
  // CHECK: return
  return %layout : !fly.layout<(4, 8) : (1, 4)>
}

// CHECK-LABEL: @test_3d_static_canonicalized
func.func @test_3d_static_canonicalized() -> !fly.int_tuple<(2, 4, 8)> {
  %0 = fly.static {elems = [2 : i32, 4 : i32, 8 : i32]} : () -> !fly.int_tuple<(2, 4, 8)>
  // CHECK: fly.make_int_tuple() : () -> !fly.int_tuple<(2,4,8)>
  return %0 : !fly.int_tuple<(2, 4, 8)>
}

// CHECK-LABEL: @test_size_static
func.func @test_size_static() -> !fly.int_tuple<32> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %size = fly.size(%s) : (!fly.int_tuple<(4, 8)>) -> !fly.int_tuple<32>
  // CHECK: return
  return %size : !fly.int_tuple<32>
}

// CHECK-LABEL: @test_composition_static
func.func @test_composition_static() -> !fly.layout<(2, 4) : (1, 2)> {
  %s1 = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d1 = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %outer = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %s2 = fly.static {elems = [2 : i32, 4 : i32]} : () -> !fly.int_tuple<(2, 4)>
  %d2 = fly.static {elems = [1 : i32, 2 : i32]} : () -> !fly.int_tuple<(1, 2)>
  %inner = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  %result = fly.composition(%outer, %inner) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  // CHECK: return
  return %result : !fly.layout<(2, 4) : (1, 2)>
}
