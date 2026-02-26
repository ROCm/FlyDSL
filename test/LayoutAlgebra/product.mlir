// RUN: %fly-opt %s | FileCheck %s

// Tests for layout product operations:
//   fly.logical_product, fly.zipped_product, fly.tiled_product,
//   fly.flat_product, fly.blocked_product, fly.raked_product

// -----

// CHECK-LABEL: @test_logical_product
func.func @test_logical_product() -> !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))> {
  // logical_product appends tile modes as new outer dimensions
  %s1 = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d1 = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %s2 = fly.static {elems = [2 : i32, 2 : i32]} : () -> !fly.int_tuple<(2, 2)>
  %d2 = fly.static {elems = [1 : i32, 2 : i32]} : () -> !fly.int_tuple<(1, 2)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(2, 2)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 2) : (1, 2)>
  // CHECK: fly.logical_product
  %result = fly.logical_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))>
  return %result : !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))>
}

// CHECK-LABEL: @test_zipped_product
func.func @test_zipped_product() -> !fly.layout<((4, 2), (8, 2)) : ((1, 32), (4, 64))> {
  // zipped_product zips base and tile modes together
  %s1 = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d1 = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %s2 = fly.static {elems = [2 : i32, 2 : i32]} : () -> !fly.int_tuple<(2, 2)>
  %d2 = fly.static {elems = [1 : i32, 2 : i32]} : () -> !fly.int_tuple<(1, 2)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(2, 2)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 2) : (1, 2)>
  // CHECK: fly.zipped_product
  %result = fly.zipped_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((4, 2), (8, 2)) : ((1, 32), (4, 64))>
  return %result : !fly.layout<((4, 2), (8, 2)) : ((1, 32), (4, 64))>
}

// CHECK-LABEL: @test_tiled_product
func.func @test_tiled_product(%base: !fly.layout<(4, 8) : (1, 4)>,
                               %tile: !fly.layout<(2, 2) : (1, 2)>) {
  // CHECK: fly.tiled_product
  %result = fly.tiled_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((4, 2), (8, 2)) : ((1, 32), (4, 64))>
  return
}

// CHECK-LABEL: @test_flat_product
func.func @test_flat_product(%base: !fly.layout<(4, 8) : (1, 4)>,
                              %tile: !fly.layout<(2, 2) : (1, 2)>) {
  // CHECK: fly.flat_product
  %result = fly.flat_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((4, 2), (8, 2)) : ((1, 32), (4, 64))>
  return
}

// CHECK-LABEL: @test_blocked_product
func.func @test_blocked_product(%base: !fly.layout<(4, 8) : (1, 4)>,
                                 %tile: !fly.layout<(2, 2) : (1, 2)>) {
  // CHECK: fly.blocked_product
  %result = fly.blocked_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((4, 2), (8, 2)) : ((1, 32), (4, 64))>
  return
}

// CHECK-LABEL: @test_raked_product
func.func @test_raked_product(%base: !fly.layout<(4, 8) : (1, 4)>,
                               %tile: !fly.layout<(2, 2) : (1, 2)>) {
  // raked_product interleaves tile modes at inner positions
  // CHECK: fly.raked_product
  %result = fly.raked_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((2, 4), (2, 8)) : ((32, 1), (64, 4))>
  return
}

// CHECK-LABEL: @test_logical_product_1d
func.func @test_logical_product_1d() -> !fly.layout<((8), (4)) : ((1), (8))> {
  // 1D base with 1D tile preserves nesting structure
  %s1 = fly.static {elems = [8 : i32]} : () -> !fly.int_tuple<(8)>
  %d1 = fly.static {elems = [1 : i32]} : () -> !fly.int_tuple<(1)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(8)>, !fly.int_tuple<(1)>) -> !fly.layout<(8) : (1)>
  %s2 = fly.static {elems = [4 : i32]} : () -> !fly.int_tuple<(4)>
  %d2 = fly.static {elems = [1 : i32]} : () -> !fly.int_tuple<(1)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %result = fly.logical_product(%base, %tile) : (!fly.layout<(8) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<((8), (4)) : ((1), (8))>
  return %result : !fly.layout<((8), (4)) : ((1), (8))>
}
