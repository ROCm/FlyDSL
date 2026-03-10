// RUN: %fly-opt %s | FileCheck %s

// Tests for fly.recast_layout
// Cover:
// 1) upcast      (old=16, new=32)
// 2) downcast    (old=32, new=16)
// 3) up+down     (old=24, new=40, gcd=8 => num=5, den=3)

// CHECK-LABEL: @test_recast_layout_upcast
func.func @test_recast_layout_upcast() -> !fly.layout<(1, 3, 4) : (1, 1, 2)> {
  %s = fly.static : () -> !fly.int_tuple<(2, 3, 4)>
  %d = fly.static : () -> !fly.int_tuple<(1, 2, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(2, 3, 4)>, !fly.int_tuple<(1, 2, 4)>) -> !fly.layout<(2, 3, 4) : (1, 2, 4)>
  // CHECK: fly.recast_layout
  %result = fly.recast_layout(32, 16, %layout) : (!fly.layout<(2, 3, 4) : (1, 2, 4)>) -> !fly.layout<(1, 3, 4) : (1, 1, 2)>
  return %result : !fly.layout<(1, 3, 4) : (1, 1, 2)>
}

// CHECK-LABEL: @test_recast_layout_downcast
func.func @test_recast_layout_downcast() -> !fly.layout<(4, 3, 4) : (1, 4, 8)> {
  %s = fly.static : () -> !fly.int_tuple<(2, 3, 4)>
  %d = fly.static : () -> !fly.int_tuple<(1, 2, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(2, 3, 4)>, !fly.int_tuple<(1, 2, 4)>) -> !fly.layout<(2, 3, 4) : (1, 2, 4)>
  // CHECK: fly.recast_layout
  %result = fly.recast_layout(16, 32, %layout) : (!fly.layout<(2, 3, 4) : (1, 2, 4)>) -> !fly.layout<(4, 3, 4) : (1, 4, 8)>
  return %result : !fly.layout<(4, 3, 4) : (1, 4, 8)>
}

// CHECK-LABEL: @test_recast_layout_upcast_then_downcast_gcd_not_1
func.func @test_recast_layout_upcast_then_downcast_gcd_not_1() -> !fly.layout<12 : 1> {
  %s = fly.static : () -> !fly.int_tuple<4>
  %d = fly.static : () -> !fly.int_tuple<5>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<4>, !fly.int_tuple<5>) -> !fly.layout<4 : 5>
  // old=24, new=40 => gcd=8, num=5, den=3 (upcast then downcast)
  // CHECK: fly.recast_layout
  %result = fly.recast_layout(40, 24, %layout) : (!fly.layout<4 : 5>) -> !fly.layout<12 : 1>
  return %result : !fly.layout<12 : 1>
}
