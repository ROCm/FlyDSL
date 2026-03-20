// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s | FileCheck %s

// Tests for layout divide operations:
//   fly.logical_divide, fly.zipped_divide, fly.tiled_divide, fly.flat_divide

// -----

// CHECK-LABEL: @test_logical_divide
func.func @test_logical_divide() -> !fly.layout<((2, 4), 4) : ((1, 2), 8)> {
  // logical_divide partitions the layout by a divisor tile
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %ds = fly.static : !fly.int_tuple<(2, 4)>
  %dd = fly.static : !fly.int_tuple<(1, 2)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(2, 4)>, !fly.int_tuple<(1, 2)>) -> !fly.layout<(2, 4) : (1, 2)>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %divisor) : (!fly.layout<(4, 8) : (1, 4)>, !fly.layout<(2, 4) : (1, 2)>) -> !fly.layout<((2, 4), 4) : ((1, 2), 8)>
  return %result : !fly.layout<((2, 4), 4) : ((1, 2), 8)>
}

// CHECK-LABEL: @test_zipped_divide
func.func @test_zipped_divide() -> !fly.layout<((2, 4), 4) : ((1, 2), 8)> {
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %ds = fly.static : !fly.int_tuple<(2, 4)>
  %dd = fly.static : !fly.int_tuple<(1, 2)>
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
func.func @test_logical_divide_1d() -> !fly.layout<(4, 4) : (1, 4)> {
  // Divide a 1D contiguous layout: (16):(1) / (4):(1) -> (4,4):(1,4)
  %s = fly.static : !fly.int_tuple<(16)>
  %d = fly.static : !fly.int_tuple<(1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(16)>, !fly.int_tuple<(1)>) -> !fly.layout<(16) : (1)>
  %ds = fly.static : !fly.int_tuple<(4)>
  %dd = fly.static : !fly.int_tuple<(1)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %divisor) : (!fly.layout<(16) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(4, 4) : (1, 4)>
  return %result : !fly.layout<(4, 4) : (1, 4)>
}

// CHECK-LABEL: @test_zipped_divide_1d
func.func @test_zipped_divide_1d() -> !fly.layout<(4, 4) : (1, 4)> {
  %s = fly.static : !fly.int_tuple<(16)>
  %d = fly.static : !fly.int_tuple<(1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(16)>, !fly.int_tuple<(1)>) -> !fly.layout<(16) : (1)>
  %ds = fly.static : !fly.int_tuple<(4)>
  %dd = fly.static : !fly.int_tuple<(1)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %divisor) : (!fly.layout<(16) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(4, 4) : (1, 4)>
  return %result : !fly.layout<(4, 4) : (1, 4)>
}

// CHECK-LABEL: @test_tiled_divide_1d
func.func @test_tiled_divide_1d() -> !fly.layout<(4, 4) : (1, 4)> {
  %s = fly.static : !fly.int_tuple<(16)>
  %d = fly.static : !fly.int_tuple<(1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(16)>, !fly.int_tuple<(1)>) -> !fly.layout<(16) : (1)>
  %ds = fly.static : !fly.int_tuple<(4)>
  %dd = fly.static : !fly.int_tuple<(1)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %divisor) : (!fly.layout<(16) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(4, 4) : (1, 4)>
  return %result : !fly.layout<(4, 4) : (1, 4)>
}

// CHECK-LABEL: @test_flat_divide_1d
func.func @test_flat_divide_1d() -> !fly.layout<(4, 4) : (1, 4)> {
  %s = fly.static : !fly.int_tuple<(16)>
  %d = fly.static : !fly.int_tuple<(1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(16)>, !fly.int_tuple<(1)>) -> !fly.layout<(16) : (1)>
  %ds = fly.static : !fly.int_tuple<(4)>
  %dd = fly.static : !fly.int_tuple<(1)>
  %divisor = fly.make_layout(%ds, %dd) : (!fly.int_tuple<(4)>, !fly.int_tuple<(1)>) -> !fly.layout<(4) : (1)>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %divisor) : (!fly.layout<(16) : (1)>, !fly.layout<(4) : (1)>) -> !fly.layout<(4, 4) : (1, 4)>
  return %result : !fly.layout<(4, 4) : (1, 4)>
}

// CHECK-LABEL: @test_logical_divide_wrapped_tuple_1d
func.func @test_logical_divide_wrapped_tuple_1d(
    %layout: !fly.layout<((16, 1)) : ((1, 16))>,
    %divisor: !fly.layout<((4, 1)) : ((1, 4))>) -> !fly.layout<((4, 1), 4) : ((1, 0), 4)> {
  // Outer singleton wrappers are accepted and handled in inference.
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %divisor)
      : (!fly.layout<((16, 1)) : ((1, 16))>, !fly.layout<((4, 1)) : ((1, 4))>)
      -> !fly.layout<((4, 1), 4) : ((1, 0), 4)>
  return %result : !fly.layout<((4, 1), 4) : ((1, 0), 4)>
}

// -----
// PyIR-aligned divide tests from tests/pyir/test_layout_algebra.py

// CHECK-LABEL: @pyir_logical_divide_with_complement
func.func @pyir_logical_divide_with_complement() -> !fly.layout<(3, 4) : (1, 3)> {
  %s = fly.static : !fly.int_tuple<(12)>
  %d = fly.static : !fly.int_tuple<(1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(12)>, !fly.int_tuple<(1)>) -> !fly.layout<(12) : (1)>
  %ts = fly.static : !fly.int_tuple<(3)>
  %td = fly.static : !fly.int_tuple<(1)>
  %tiler = fly.make_layout(%ts, %td) : (!fly.int_tuple<(3)>, !fly.int_tuple<(1)>) -> !fly.layout<(3) : (1)>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(12) : (1)>, !fly.layout<(3) : (1)>) -> !fly.layout<(3, 4) : (1, 3)>
  return %result : !fly.layout<(3, 4) : (1, 3)>
}

// CHECK-LABEL: @pyir_logical_divide_1d
func.func @pyir_logical_divide_1d() -> !fly.layout<((2, 3, 6), (2, 9)) : ((133, 69, 19), (207, 1))> {
  %s = fly.static : !fly.int_tuple<(14, 6, 9)>
  %d = fly.static : !fly.int_tuple<(19, 69, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, 6, 9)>, !fly.int_tuple<(19, 69, 1)>) -> !fly.layout<(14, 6, 9) : (19, 69, 1)>
  %ts = fly.static : !fly.int_tuple<(2, 3, 6)>
  %td = fly.static : !fly.int_tuple<(7, 14, 1)>
  %tiler = fly.make_layout(%ts, %td) : (!fly.int_tuple<(2, 3, 6)>, !fly.int_tuple<(7, 14, 1)>) -> !fly.layout<(2, 3, 6) : (7, 14, 1)>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(14, 6, 9) : (19, 69, 1)>, !fly.layout<(2, 3, 6) : (7, 14, 1)>) -> !fly.layout<((2, 3, 6), (2, 9)) : ((133, 69, 19), (207, 1))>
  return %result : !fly.layout<((2, 3, 6), (2, 9)) : ((133, 69, 19), (207, 1))>
}

// CHECK-LABEL: @pyir_zipped_divide_1d
func.func @pyir_zipped_divide_1d() -> !fly.layout<((2, 3, 6), (2, 9)) : ((133, 69, 19), (207, 1))> {
  %s = fly.static : !fly.int_tuple<(14, 6, 9)>
  %d = fly.static : !fly.int_tuple<(19, 69, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, 6, 9)>, !fly.int_tuple<(19, 69, 1)>) -> !fly.layout<(14, 6, 9) : (19, 69, 1)>
  %ts = fly.static : !fly.int_tuple<(2, 3, 6)>
  %td = fly.static : !fly.int_tuple<(7, 14, 1)>
  %tiler = fly.make_layout(%ts, %td) : (!fly.int_tuple<(2, 3, 6)>, !fly.int_tuple<(7, 14, 1)>) -> !fly.layout<(2, 3, 6) : (7, 14, 1)>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %tiler) : (!fly.layout<(14, 6, 9) : (19, 69, 1)>, !fly.layout<(2, 3, 6) : (7, 14, 1)>) -> !fly.layout<((2, 3, 6), (2, 9)) : ((133, 69, 19), (207, 1))>
  return %result : !fly.layout<((2, 3, 6), (2, 9)) : ((133, 69, 19), (207, 1))>
}

// CHECK-LABEL: @pyir_tiled_divide_1d
func.func @pyir_tiled_divide_1d() -> !fly.layout<((2, 3, 6), 2, 9) : ((133, 69, 19), 207, 1)> {
  %s = fly.static : !fly.int_tuple<(14, 6, 9)>
  %d = fly.static : !fly.int_tuple<(19, 69, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, 6, 9)>, !fly.int_tuple<(19, 69, 1)>) -> !fly.layout<(14, 6, 9) : (19, 69, 1)>
  %ts = fly.static : !fly.int_tuple<(2, 3, 6)>
  %td = fly.static : !fly.int_tuple<(7, 14, 1)>
  %tiler = fly.make_layout(%ts, %td) : (!fly.int_tuple<(2, 3, 6)>, !fly.int_tuple<(7, 14, 1)>) -> !fly.layout<(2, 3, 6) : (7, 14, 1)>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %tiler) : (!fly.layout<(14, 6, 9) : (19, 69, 1)>, !fly.layout<(2, 3, 6) : (7, 14, 1)>) -> !fly.layout<((2, 3, 6), 2, 9) : ((133, 69, 19), 207, 1)>
  return %result : !fly.layout<((2, 3, 6), 2, 9) : ((133, 69, 19), 207, 1)>
}

// CHECK-LABEL: @pyir_flat_divide_1d
func.func @pyir_flat_divide_1d() -> !fly.layout<(2, 3, 6, 2, 9) : (133, 69, 19, 207, 1)> {
  %s = fly.static : !fly.int_tuple<(14, 6, 9)>
  %d = fly.static : !fly.int_tuple<(19, 69, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, 6, 9)>, !fly.int_tuple<(19, 69, 1)>) -> !fly.layout<(14, 6, 9) : (19, 69, 1)>
  %ts = fly.static : !fly.int_tuple<(2, 3, 6)>
  %td = fly.static : !fly.int_tuple<(7, 14, 1)>
  %tiler = fly.make_layout(%ts, %td) : (!fly.int_tuple<(2, 3, 6)>, !fly.int_tuple<(7, 14, 1)>) -> !fly.layout<(2, 3, 6) : (7, 14, 1)>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %tiler) : (!fly.layout<(14, 6, 9) : (19, 69, 1)>, !fly.layout<(2, 3, 6) : (7, 14, 1)>) -> !fly.layout<(2, 3, 6, 2, 9) : (133, 69, 19, 207, 1)>
  return %result : !fly.layout<(2, 3, 6, 2, 9) : (133, 69, 19, 207, 1)>
}

// -----
// PyIR-aligned by-mode 2d divide tests

// CHECK-LABEL: @pyir_logical_divide_2d_bymode
func.func @pyir_logical_divide_2d_bymode() -> !fly.layout<((2, 7), ((3, (2, 3)), 3)) : ((133, 19), ((69, (207, 1)), 3))> {
  %s = fly.static : !fly.int_tuple<(14, (6, 9))>
  %d = fly.static : !fly.int_tuple<(19, (69, 1))>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, (6, 9))>, !fly.int_tuple<(19, (69, 1))>) -> !fly.layout<(14, (6, 9)) : (19, (69, 1))>
  %m0s = fly.static : !fly.int_tuple<2>
  %m0d = fly.static : !fly.int_tuple<7>
  %m0 = fly.make_layout(%m0s, %m0d) : (!fly.int_tuple<2>, !fly.int_tuple<7>) -> !fly.layout<2 : 7>
  %m1s = fly.static : !fly.int_tuple<(3, 6)>
  %m1d = fly.static : !fly.int_tuple<(1, 3)>
  %m1 = fly.make_layout(%m1s, %m1d) : (!fly.int_tuple<(3, 6)>, !fly.int_tuple<(1, 3)>) -> !fly.layout<(3, 6) : (1, 3)>
  %tiler = fly.make_tile(%m0, %m1) : (!fly.layout<2 : 7>, !fly.layout<(3, 6) : (1, 3)>) -> !fly.tile<[2:7|(3, 6):(1, 3)]>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(14, (6, 9)) : (19, (69, 1))>, !fly.tile<[2:7|(3, 6):(1, 3)]>) -> !fly.layout<((2, 7), ((3, (2, 3)), 3)) : ((133, 19), ((69, (207, 1)), 3))>
  return %result : !fly.layout<((2, 7), ((3, (2, 3)), 3)) : ((133, 19), ((69, (207, 1)), 3))>
}

// CHECK-LABEL: @pyir_zipped_divide_2d_bymode
func.func @pyir_zipped_divide_2d_bymode() -> !fly.layout<((2, (3, (2, 3))), (7, 3)) : ((133, (69, (207, 1))), (19, 3))> {
  %s = fly.static : !fly.int_tuple<(14, (6, 9))>
  %d = fly.static : !fly.int_tuple<(19, (69, 1))>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, (6, 9))>, !fly.int_tuple<(19, (69, 1))>) -> !fly.layout<(14, (6, 9)) : (19, (69, 1))>
  %m0s = fly.static : !fly.int_tuple<2>
  %m0d = fly.static : !fly.int_tuple<7>
  %m0 = fly.make_layout(%m0s, %m0d) : (!fly.int_tuple<2>, !fly.int_tuple<7>) -> !fly.layout<2 : 7>
  %m1s = fly.static : !fly.int_tuple<(3, 6)>
  %m1d = fly.static : !fly.int_tuple<(1, 3)>
  %m1 = fly.make_layout(%m1s, %m1d) : (!fly.int_tuple<(3, 6)>, !fly.int_tuple<(1, 3)>) -> !fly.layout<(3, 6) : (1, 3)>
  %tiler = fly.make_tile(%m0, %m1) : (!fly.layout<2 : 7>, !fly.layout<(3, 6) : (1, 3)>) -> !fly.tile<[2:7|(3, 6):(1, 3)]>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %tiler) : (!fly.layout<(14, (6, 9)) : (19, (69, 1))>, !fly.tile<[2:7|(3, 6):(1, 3)]>) -> !fly.layout<((2, (3, (2, 3))), (7, 3)) : ((133, (69, (207, 1))), (19, 3))>
  return %result : !fly.layout<((2, (3, (2, 3))), (7, 3)) : ((133, (69, (207, 1))), (19, 3))>
}

// CHECK-LABEL: @pyir_tiled_divide_2d_bymode
func.func @pyir_tiled_divide_2d_bymode() -> !fly.layout<((2, (3, (2, 3))), 7, 3) : ((133, (69, (207, 1))), 19, 3)> {
  %s = fly.static : !fly.int_tuple<(14, (6, 9))>
  %d = fly.static : !fly.int_tuple<(19, (69, 1))>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, (6, 9))>, !fly.int_tuple<(19, (69, 1))>) -> !fly.layout<(14, (6, 9)) : (19, (69, 1))>
  %m0s = fly.static : !fly.int_tuple<2>
  %m0d = fly.static : !fly.int_tuple<7>
  %m0 = fly.make_layout(%m0s, %m0d) : (!fly.int_tuple<2>, !fly.int_tuple<7>) -> !fly.layout<2 : 7>
  %m1s = fly.static : !fly.int_tuple<(3, 6)>
  %m1d = fly.static : !fly.int_tuple<(1, 3)>
  %m1 = fly.make_layout(%m1s, %m1d) : (!fly.int_tuple<(3, 6)>, !fly.int_tuple<(1, 3)>) -> !fly.layout<(3, 6) : (1, 3)>
  %tiler = fly.make_tile(%m0, %m1) : (!fly.layout<2 : 7>, !fly.layout<(3, 6) : (1, 3)>) -> !fly.tile<[2:7|(3, 6):(1, 3)]>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %tiler) : (!fly.layout<(14, (6, 9)) : (19, (69, 1))>, !fly.tile<[2:7|(3, 6):(1, 3)]>) -> !fly.layout<((2, (3, (2, 3))), 7, 3) : ((133, (69, (207, 1))), 19, 3)>
  return %result : !fly.layout<((2, (3, (2, 3))), 7, 3) : ((133, (69, (207, 1))), 19, 3)>
}

// CHECK-LABEL: @pyir_flat_divide_2d_bymode
func.func @pyir_flat_divide_2d_bymode() -> !fly.layout<(2, (3, (2, 3)), 7, 3) : (133, (69, (207, 1)), 19, 3)> {
  %s = fly.static : !fly.int_tuple<(14, (6, 9))>
  %d = fly.static : !fly.int_tuple<(19, (69, 1))>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, (6, 9))>, !fly.int_tuple<(19, (69, 1))>) -> !fly.layout<(14, (6, 9)) : (19, (69, 1))>
  %m0s = fly.static : !fly.int_tuple<2>
  %m0d = fly.static : !fly.int_tuple<7>
  %m0 = fly.make_layout(%m0s, %m0d) : (!fly.int_tuple<2>, !fly.int_tuple<7>) -> !fly.layout<2 : 7>
  %m1s = fly.static : !fly.int_tuple<(3, 6)>
  %m1d = fly.static : !fly.int_tuple<(1, 3)>
  %m1 = fly.make_layout(%m1s, %m1d) : (!fly.int_tuple<(3, 6)>, !fly.int_tuple<(1, 3)>) -> !fly.layout<(3, 6) : (1, 3)>
  %tiler = fly.make_tile(%m0, %m1) : (!fly.layout<2 : 7>, !fly.layout<(3, 6) : (1, 3)>) -> !fly.tile<[2:7|(3, 6):(1, 3)]>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %tiler) : (!fly.layout<(14, (6, 9)) : (19, (69, 1))>, !fly.tile<[2:7|(3, 6):(1, 3)]>) -> !fly.layout<(2, (3, (2, 3)), 7, 3) : (133, (69, (207, 1)), 19, 3)>
  return %result : !fly.layout<(2, (3, (2, 3)), 7, 3) : (133, (69, (207, 1)), 19, 3)>
}
