// RUN: %fly-opt %s | FileCheck %s

// Tests for layout divide operations:
//   fly.logical_divide, fly.zipped_divide, fly.tiled_divide, fly.flat_divide

// -----

// CHECK-LABEL: @test_logical_divide
func.func @test_logical_divide() -> !fly.layout<((2, 4), 4) : ((1, 2), 8)> {
  // logical_divide partitions the layout by a divisor tile
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %ds = fly.static {elems = [2 : i32, 4 : i32]} : () -> !fly.int_tuple<(2, 4)>
  %dd = fly.static {elems = [1 : i32, 2 : i32]} : () -> !fly.int_tuple<(1, 2)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %divisor) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<((2, 4), 4) : ((1, 2), 8)>
  return %result : !fly.layout<((2, 4), 4) : ((1, 2), 8)>
}

// CHECK-LABEL: @test_zipped_divide
func.func @test_zipped_divide() -> !fly.layout<((2, 4), 4) : ((1, 2), 8)> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %ds = fly.static {elems = [2 : i32, 4 : i32]} : () -> !fly.int_tuple<(2, 4)>
  %dd = fly.static {elems = [1 : i32, 2 : i32]} : () -> !fly.int_tuple<(1, 2)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %divisor) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<((2, 4), 4) : ((1, 2), 8)>
  return %result : !fly.layout<((2, 4), 4) : ((1, 2), 8)>
}

// CHECK-LABEL: @test_tiled_divide
func.func @test_tiled_divide(%layout: !fly.layout<(4, 8) : (1, 4)>,
                              %divisor: !fly.layout<(2, 4) : (1, 2)>) {
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %divisor) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<((2, 4), 4) : ((1, 2), 8)>
  return
}

// CHECK-LABEL: @test_flat_divide
func.func @test_flat_divide(%layout: !fly.layout<(4, 8) : (1, 4)>,
                             %divisor: !fly.layout<(2, 4) : (1, 2)>) {
  // flat_divide flattens the result (no nesting)
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %divisor) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<(2, 4, 4) : (1, 2, 8)>
  return
}

// CHECK-LABEL: @test_logical_divide_1d
func.func @test_logical_divide_1d() -> !fly.layout<((4), 4) : ((1), 4)> {
  // Divide a 1D contiguous layout: (16):(1) / (4):(1) -> ((4),4):((1),4)
  %s = fly.static {elems = [16 : i32]} : () -> !fly.int_tuple<(16)>
  %d = fly.static {elems = [1 : i32]} : () -> !fly.int_tuple<(1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(16)>, !fly.int_tuple<(1)>) -> !fly.layout<(16) : (1)>
  %ds = fly.static {elems = [4 : i32]} : () -> !fly.int_tuple<(4)>
  %dd = fly.static {elems = [1 : i32]} : () -> !fly.int_tuple<(1)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %result = fly.logical_divide(%layout, %divisor) : (!fly.layout<(16) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<((4), 4) : ((1), 4)>
  return %result : !fly.layout<((4), 4) : ((1), 4)>
}
