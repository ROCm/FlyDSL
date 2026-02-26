// RUN: %fly-opt %s | FileCheck %s

// Tests for layout composition, complement, coalesce, inverse operations

// -----

// CHECK-LABEL: @test_composition
func.func @test_composition() -> !fly.layout<(2, 4) : (1, 2)> {
  // composition(Layout<(4,8):(1,4)>, Layout<(2,4):(1,2)>)
  // Compose outer layout with inner layout
  %s1 = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d1 = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %outer = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %s2 = fly.static {elems = [2 : i32, 4 : i32]} : () -> !fly.int_tuple<(2, 4)>
  %d2 = fly.static {elems = [1 : i32, 2 : i32]} : () -> !fly.int_tuple<(1, 2)>
  %inner = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  // CHECK: fly.composition
  %result = fly.composition(%outer, %inner) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  return %result : !fly.layout<(2, 4) : (1, 2)>
}

// CHECK-LABEL: @test_complement
func.func @test_complement() -> !fly.layout<8 : 4> {
  // complement(Layout<(4):(1)>, 32) = Layout<8:4>
  // The complement fills in the "gaps" in the codomain
  %s = fly.static {elems = [4 : i32]} : () -> !fly.int_tuple<(4)>
  %d = fly.static {elems = [1 : i32]} : () -> !fly.int_tuple<(1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %codom = fly.static {elems = [32 : i32]} : () -> !fly.int_tuple<32>
  // CHECK: fly.complement(%{{.*}}, %{{.*}})
  %result = fly.complement(%layout, %codom) : (!fly.layout<(4) : (1)>, !fly.int_tuple<32>) -> !fly.layout<8 : 4>
  return %result : !fly.layout<8 : 4>
}

// CHECK-LABEL: @test_complement_no_codom
func.func @test_complement_no_codom(%l: !fly.layout<(4) : (1)>) -> !fly.layout<1 : 0> {
  // CHECK: fly.complement(%{{.*}})
  %result = fly.complement(%l) : (!fly.layout<(4) : (1)>) -> !fly.layout<1 : 0>
  return %result : !fly.layout<1 : 0>
}

// CHECK-LABEL: @test_coalesce
func.func @test_coalesce() -> !fly.layout<32 : 1> {
  // coalesce merges contiguous modes: Layout<(4,8):(1,4)> -> Layout<32:1>
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.coalesce(%{{.*}})
  %result = fly.coalesce(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<32 : 1>
  return %result : !fly.layout<32 : 1>
}

// CHECK-LABEL: @test_coalesce_non_contiguous
func.func @test_coalesce_non_contiguous() -> !fly.layout<(4, 8) : (1, 8)> {
  // Non-contiguous layout cannot be fully coalesced
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 8 : i32]} : () -> !fly.int_tuple<(1, 8)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 8)>) -> !fly.layout<(4, 8) : (1, 8)>
  %result = fly.coalesce(%layout) : (!fly.layout<(4, 8) : (1, 8)>) -> !fly.layout<(4, 8) : (1, 8)>
  return %result : !fly.layout<(4, 8) : (1, 8)>
}

// CHECK-LABEL: @test_right_inverse
func.func @test_right_inverse() -> !fly.layout<32 : 1> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.right_inverse(%{{.*}})
  %result = fly.right_inverse(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<32 : 1>
  return %result : !fly.layout<32 : 1>
}

// CHECK-LABEL: @test_left_inverse
func.func @test_left_inverse() -> !fly.layout<(4, 8) : (1, 4)> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.left_inverse(%{{.*}})
  %result = fly.left_inverse(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  return %result : !fly.layout<(4, 8) : (1, 4)>
}
