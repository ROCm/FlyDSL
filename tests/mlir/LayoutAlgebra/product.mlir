// RUN: %fly-opt %s | FileCheck %s

// Tests for layout product operations:
//   fly.logical_product, fly.zipped_product, fly.tiled_product,
//   fly.flat_product, fly.blocked_product, fly.raked_product

// -----

// CHECK-LABEL: @test_logical_product
func.func @test_logical_product() -> !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))> {
  // logical_product appends tile modes as new outer dimensions
  %s1 = fly.static : !fly.int_tuple<(4, 8)>
  %d1 = fly.static : !fly.int_tuple<(1, 4)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %s2 = fly.static : !fly.int_tuple<(2, 2)>
  %d2 = fly.static : !fly.int_tuple<(1, 2)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(2, 2)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 2) : (1, 2)>
  // CHECK: fly.logical_product
  %result = fly.logical_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))>
  return %result : !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))>
}

// CHECK-LABEL: @test_zipped_product
func.func @test_zipped_product() -> !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))> {
  %s1 = fly.static : !fly.int_tuple<(4, 8)>
  %d1 = fly.static : !fly.int_tuple<(1, 4)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %s2 = fly.static : !fly.int_tuple<(2, 2)>
  %d2 = fly.static : !fly.int_tuple<(1, 2)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(2, 2)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 2) : (1, 2)>
  // CHECK: fly.zipped_product
  %result = fly.zipped_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))>
  return %result : !fly.layout<((4, 8), (2, 2)) : ((1, 4), (32, 64))>
}

// CHECK-LABEL: @test_tiled_product
func.func @test_tiled_product(%base: !fly.layout<(4, 8) : (1, 4)>,
                               %tile: !fly.layout<(2, 2) : (1, 2)>) {
  // CHECK: fly.tiled_product
  %result = fly.tiled_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<((4, 8), 2, 2) : ((1, 4), 32, 64)>
  return
}

// CHECK-LABEL: @test_flat_product
func.func @test_flat_product(%base: !fly.layout<(4, 8) : (1, 4)>,
                              %tile: !fly.layout<(2, 2) : (1, 2)>) {
  // CHECK: fly.flat_product
  %result = fly.flat_product(%base, %tile) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 2) : (1, 2)>) -> !fly.layout<(4, 8, 2, 2) : (1, 4, 32, 64)>
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
func.func @test_logical_product_1d() -> !fly.layout<(8, 4) : (1, 8)> {
  // 1D base with 1D tile canonicalizes singleton tuple wrappers
  %s1 = fly.static : !fly.int_tuple<(8)>
  %d1 = fly.static : !fly.int_tuple<(1)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(8)>, !fly.int_tuple<(1)>) -> !fly.layout<(8) : (1)>
  %s2 = fly.static : !fly.int_tuple<(4)>
  %d2 = fly.static : !fly.int_tuple<(1)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %result = fly.logical_product(%base, %tile) : (!fly.layout<(8) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(8, 4) : (1, 8)>
  return %result : !fly.layout<(8, 4) : (1, 8)>
}

// CHECK-LABEL: @test_zipped_product_1d
func.func @test_zipped_product_1d() -> !fly.layout<(8, 4) : (1, 8)> {
  %s1 = fly.static : !fly.int_tuple<(8)>
  %d1 = fly.static : !fly.int_tuple<(1)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(8)>, !fly.int_tuple<(1)>) -> !fly.layout<(8) : (1)>
  %s2 = fly.static : !fly.int_tuple<(4)>
  %d2 = fly.static : !fly.int_tuple<(1)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %result = fly.zipped_product(%base, %tile) : (!fly.layout<(8) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(8, 4) : (1, 8)>
  return %result : !fly.layout<(8, 4) : (1, 8)>
}

// CHECK-LABEL: @test_tiled_product_1d
func.func @test_tiled_product_1d() -> !fly.layout<(8, 4) : (1, 8)> {
  %s1 = fly.static : !fly.int_tuple<(8)>
  %d1 = fly.static : !fly.int_tuple<(1)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(8)>, !fly.int_tuple<(1)>) -> !fly.layout<(8) : (1)>
  %s2 = fly.static : !fly.int_tuple<(4)>
  %d2 = fly.static : !fly.int_tuple<(1)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %result = fly.tiled_product(%base, %tile) : (!fly.layout<(8) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(8, 4) : (1, 8)>
  return %result : !fly.layout<(8, 4) : (1, 8)>
}

// CHECK-LABEL: @test_flat_product_1d
func.func @test_flat_product_1d() -> !fly.layout<(8, 4) : (1, 8)> {
  %s1 = fly.static : !fly.int_tuple<(8)>
  %d1 = fly.static : !fly.int_tuple<(1)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(8)>, !fly.int_tuple<(1)>) -> !fly.layout<(8) : (1)>
  %s2 = fly.static : !fly.int_tuple<(4)>
  %d2 = fly.static : !fly.int_tuple<(1)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %result = fly.flat_product(%base, %tile) : (!fly.layout<(8) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(8, 4) : (1, 8)>
  return %result : !fly.layout<(8, 4) : (1, 8)>
}

// CHECK-LABEL: @test_blocked_product_1d
func.func @test_blocked_product_1d() -> !fly.layout<(8, 4) : (1, 8)> {
  %s1 = fly.static : !fly.int_tuple<(8)>
  %d1 = fly.static : !fly.int_tuple<(1)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(8)>, !fly.int_tuple<(1)>) -> !fly.layout<(8) : (1)>
  %s2 = fly.static : !fly.int_tuple<(4)>
  %d2 = fly.static : !fly.int_tuple<(1)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %result = fly.blocked_product(%base, %tile) : (!fly.layout<(8) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(8, 4) : (1, 8)>
  return %result : !fly.layout<(8, 4) : (1, 8)>
}

// CHECK-LABEL: @test_raked_product_1d
func.func @test_raked_product_1d() -> !fly.layout<(4, 8) : (8, 1)> {
  %s1 = fly.static : !fly.int_tuple<(8)>
  %d1 = fly.static : !fly.int_tuple<(1)>
  %base = fly.make_layout(%s1, %d1) : (!fly.int_tuple<(8)>, !fly.int_tuple<(1)>) -> !fly.layout<(8) : (1)>
  %s2 = fly.static : !fly.int_tuple<(4)>
  %d2 = fly.static : !fly.int_tuple<(1)>
  %tile = fly.make_layout(%s2, %d2) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  %result = fly.raked_product(%base, %tile) : (!fly.layout<(8) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(4, 8) : (8, 1)>
  return %result : !fly.layout<(4, 8) : (8, 1)>
}

// CHECK-LABEL: @test_logical_product_wrapped_tuple_1d
func.func @test_logical_product_wrapped_tuple_1d(
    %base: !fly.layout<((8, 1)) : ((1, 8))>,
    %tile: !fly.layout<((4, 1)) : ((1, 4))>) -> !fly.layout<((8, 1), (4, 1)) : ((1, 0), (8, 0))> {
  // Outer singleton wrappers are accepted and handled in inference.
  %result = fly.logical_product(%base, %tile)
      : (!fly.layout<((8, 1)) : ((1, 8))>, !fly.layout<((4, 1)) : ((1, 4))>)
      -> !fly.layout<((8, 1), (4, 1)) : ((1, 0), (8, 0))>
  return %result : !fly.layout<((8, 1), (4, 1)) : ((1, 0), (8, 0))>
}

// -----
// PyIR-aligned product tests from tests/pyir/test_layout_algebra.py

// CHECK-LABEL: @pyir_logical_product_1d
func.func @pyir_logical_product_1d() -> !fly.layout<((6, 9), 9) : ((9, 1), 54)> {
  %s = fly.static : !fly.int_tuple<(6, 9)>
  %d = fly.static : !fly.int_tuple<(9, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(6, 9)>, !fly.int_tuple<(9, 1)>) -> !fly.layout<(6, 9) : (9, 1)>
  %ts = fly.static : !fly.int_tuple<(9)>
  %td = fly.static : !fly.int_tuple<(1)>
  %tiler = fly.make_layout(%ts, %td) : (!fly.int_tuple<(9)>, !fly.int_tuple<(1)>) -> !fly.layout<(9) : (1)>
  // CHECK: fly.logical_product
  %result = fly.logical_product(%layout, %tiler) : (!fly.layout<(6, 9) : (9, 1)>, !fly.layout<(9) : (1)>) -> !fly.layout<((6, 9), 9) : ((9, 1), 54)>
  return %result : !fly.layout<((6, 9), 9) : ((9, 1), 54)>
}

// CHECK-LABEL: @pyir_blocked_product
func.func @pyir_blocked_product() -> !fly.layout<((3, 4), (6, 5)) : ((6, 18), (1, 72))> {
  %s = fly.static : !fly.int_tuple<(3, 6)>
  %d = fly.static : !fly.int_tuple<(6, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(3, 6)>, !fly.int_tuple<(6, 1)>) -> !fly.layout<(3, 6) : (6, 1)>
  %ts = fly.static : !fly.int_tuple<(4, 5)>
  %td = fly.static : !fly.int_tuple<(1, 4)>
  %tiler = fly.make_layout(%ts, %td) : (!fly.int_tuple<(4, 5)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 5) : (1, 4)>
  // CHECK: fly.blocked_product
  %result = fly.blocked_product(%layout, %tiler) : (!fly.layout<(3, 6) : (6, 1)>, !fly.layout<(4, 5) : (1, 4)>) -> !fly.layout<((3, 4), (6, 5)) : ((6, 18), (1, 72))>
  return %result : !fly.layout<((3, 4), (6, 5)) : ((6, 18), (1, 72))>
}

// CHECK-LABEL: @pyir_flat_product
func.func @pyir_flat_product() -> !fly.layout<(3, 6, 4, 5) : (6, 1, 18, 72)> {
  %s = fly.static : !fly.int_tuple<(3, 6)>
  %d = fly.static : !fly.int_tuple<(6, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(3, 6)>, !fly.int_tuple<(6, 1)>) -> !fly.layout<(3, 6) : (6, 1)>
  %ts = fly.static : !fly.int_tuple<(4, 5)>
  %td = fly.static : !fly.int_tuple<(1, 4)>
  %tiler = fly.make_layout(%ts, %td) : (!fly.int_tuple<(4, 5)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 5) : (1, 4)>
  // CHECK: fly.flat_product
  %result = fly.flat_product(%layout, %tiler) : (!fly.layout<(3, 6) : (6, 1)>, !fly.layout<(4, 5) : (1, 4)>) -> !fly.layout<(3, 6, 4, 5) : (6, 1, 18, 72)>
  return %result : !fly.layout<(3, 6, 4, 5) : (6, 1, 18, 72)>
}
