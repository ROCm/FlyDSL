// RUN: %fly-opt %s | FileCheck %s

// Tests for fly.right_inverse

// CHECK-LABEL: @test_right_inverse_coalesced
func.func @test_right_inverse_coalesced() -> !fly.layout<32 : 1> {
  %s = fly.static : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.right_inverse
  %result = fly.right_inverse(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.layout<32 : 1>
  return %result : !fly.layout<32 : 1>
}

// CHECK-LABEL: @test_right_inverse_row_major
func.func @test_right_inverse_row_major() -> !fly.layout<(4, 2) : (2, 1)> {
  %s = fly.static : () -> !fly.int_tuple<(2, 4)>
  %d = fly.static : () -> !fly.int_tuple<(4, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(4, 1)>) -> !fly.layout<(2, 4) : (4, 1)>
  // CHECK: fly.right_inverse
  %result = fly.right_inverse(%layout) : (!fly.layout<(2, 4) : (4, 1)>) -> !fly.layout<(4, 2) : (2, 1)>
  return %result : !fly.layout<(4, 2) : (2, 1)>
}
