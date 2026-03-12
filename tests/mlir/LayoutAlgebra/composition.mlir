// RUN: %fly-opt %s | FileCheck %s

// Tests for layout composition, complement, coalesce, inverse operations

// -----

// CHECK-LABEL: @test_composition
func.func @test_composition() -> !fly.layout<(2, 4) : (1, 2)> {
  // composition(Layout<(4,8):(1,4)>, Layout<(2,4):(1,2)>)
  // Compose outer layout with inner layout
  %s1 = fly.static : !fly.int_tuple<(4, 8)>
  %d1 = fly.static : !fly.int_tuple<(1, 4)>
  %outer = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %s2 = fly.static : !fly.int_tuple<(2, 4)>
  %d2 = fly.static : !fly.int_tuple<(1, 2)>
  %inner = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  // CHECK: fly.composition
  %result = fly.composition(%outer, %inner) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  return %result : !fly.layout<(2, 4) : (1, 2)>
}

// CHECK-LABEL: @test_complement
func.func @test_complement() -> !fly.layout<8 : 4> {
  // complement(Layout<(4):(1)>, 32) = Layout<8:4>
  // The complement fills in the "gaps" in the codomain
  %s = fly.static : !fly.int_tuple<(4)>
  %d = fly.static : !fly.int_tuple<(1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %codom = fly.static : !fly.int_tuple<32>
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
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.coalesce(%{{.*}})
  %result = fly.coalesce(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<32 : 1>
  return %result : !fly.layout<32 : 1>
}

// CHECK-LABEL: @test_coalesce_non_contiguous
func.func @test_coalesce_non_contiguous() -> !fly.layout<(4, 8) : (1, 8)> {
  // Non-contiguous layout cannot be fully coalesced
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 8)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 8)>) -> !fly.layout<(4, 8) : (1, 8)>
  %result = fly.coalesce(%layout) : (!fly.layout<(4, 8) : (1, 8)>) -> !fly.layout<(4, 8) : (1, 8)>
  return %result : !fly.layout<(4, 8) : (1, 8)>
}

// CHECK-LABEL: @test_right_inverse
func.func @test_right_inverse() -> !fly.layout<32 : 1> {
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.right_inverse(%{{.*}})
  %result = fly.right_inverse(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<32 : 1>
  return %result : !fly.layout<32 : 1>
}

// CHECK-LABEL: @test_left_inverse
func.func @test_left_inverse() -> !fly.layout<32 : 1> {
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.left_inverse(%{{.*}})
  %result = fly.left_inverse(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<32 : 1>
  return %result : !fly.layout<32 : 1>
}

// -----
// PyIR-aligned composition/complement tests from tests/pyir/test_layout_algebra.py

// CHECK-LABEL: @pyir_composition_basic
func.func @pyir_composition_basic() -> !fly.layout<((2, 3), 3) : ((57, 69), 19)> {
  %as = fly.static : !fly.int_tuple<(6, 9)>
  %ad = fly.static : !fly.int_tuple<(19, 69)>
  %a = fly.make_layout(%as, %ad) : (!fly.int_tuple<(6, 9)>, !fly.int_tuple<(19, 69)>) -> !fly.layout<(6, 9) : (19, 69)>
  %bs = fly.static : !fly.int_tuple<(6, 3)>
  %bd = fly.static : !fly.int_tuple<(3, 1)>
  %b = fly.make_layout(%bs, %bd) : (!fly.int_tuple<(6, 3)>, !fly.int_tuple<(3, 1)>) -> !fly.layout<(6, 3) : (3, 1)>
  // CHECK: fly.composition
  %result = fly.composition(%a, %b) : (!fly.layout<(6, 9) : (19, 69)>, !fly.layout<(6, 3) : (3, 1)>) -> !fly.layout<((2, 3), 3) : ((57, 69), 19)>
  return %result : !fly.layout<((2, 3), 3) : ((57, 69), 19)>
}

// CHECK-LABEL: @pyir_composition_static
func.func @pyir_composition_static() -> !fly.layout<(3, 5) : (19, 51)> {
  %as = fly.static : !fly.int_tuple<(5, 15)>
  %ad = fly.static : !fly.int_tuple<(19, 51)>
  %a = fly.make_layout(%as, %ad) : (!fly.int_tuple<(5, 15)>, !fly.int_tuple<(19, 51)>) -> !fly.layout<(5, 15) : (19, 51)>
  %bs = fly.static : !fly.int_tuple<(3, 5)>
  %bd = fly.static : !fly.int_tuple<(1, 5)>
  %b = fly.make_layout(%bs, %bd) : (!fly.int_tuple<(3, 5)>, !fly.int_tuple<(1, 5)>) -> !fly.layout<(3, 5) : (1, 5)>
  %result = fly.composition(%a, %b) : (!fly.layout<(5, 15) : (19, 51)>, !fly.layout<(3, 5) : (1, 5)>) -> !fly.layout<(3, 5) : (19, 51)>
  return %result : !fly.layout<(3, 5) : (19, 51)>
}

// CHECK-LABEL: @pyir_composition_with_tuple
func.func @pyir_composition_with_tuple() -> !fly.layout<2 : 1> {
  %as = fly.static : !fly.int_tuple<(4)>
  %ad = fly.static : !fly.int_tuple<(1)>
  %a = fly.make_layout(%as, %ad) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %bs = fly.static : !fly.int_tuple<(2)>
  %bd = fly.static : !fly.int_tuple<(1)>
  %b = fly.make_layout(%bs, %bd) : (!fly.int_tuple<(2)>, !fly.int_tuple<(1)>) -> !fly.layout<(2) : (1)>
  %result = fly.composition(%a, %b) : (!fly.layout<(4) : (1)>, !fly.layout<(2) : (1)>) -> !fly.layout<2 : 1>
  return %result : !fly.layout<2 : 1>
}

// CHECK-LABEL: @pyir_complement_rank1
func.func @pyir_complement_rank1() -> !fly.layout<4 : 3> {
  %s = fly.static : !fly.int_tuple<(3)>
  %d = fly.static : !fly.int_tuple<(1)>
  %tiler = fly.make_layout(%s, %d) : (!fly.int_tuple<(3)>, !fly.int_tuple<(1)>) -> !fly.layout<(3) : (1)>
  %codom = fly.static : !fly.int_tuple<12>
  %result = fly.complement(%tiler, %codom) : (!fly.layout<(3) : (1)>, !fly.int_tuple<12>) -> !fly.layout<4 : 3>
  return %result : !fly.layout<4 : 3>
}

// CHECK-LABEL: @pyir_complement_rank2
func.func @pyir_complement_rank2() -> !fly.layout<2 : 6> {
  %s = fly.static : !fly.int_tuple<(3, 2)>
  %d = fly.static : !fly.int_tuple<(2, 1)>
  %tiler = fly.make_layout(%s, %d) : (!fly.int_tuple<(3, 2)>, !fly.int_tuple<(2, 1)>) -> !fly.layout<(3, 2) : (2, 1)>
  %codom = fly.static : !fly.int_tuple<12>
  %result = fly.complement(%tiler, %codom) : (!fly.layout<(3, 2) : (2, 1)>, !fly.int_tuple<12>) -> !fly.layout<2 : 6>
  return %result : !fly.layout<2 : 6>
}
