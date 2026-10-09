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
  %tiler = fly.static : !fly.tile<[2:7|(3, 6):(1, 3)]>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(14, (6, 9)) : (19, (69, 1))>, !fly.tile<[2:7|(3, 6):(1, 3)]>) -> !fly.layout<((2, 7), ((3, (2, 3)), 3)) : ((133, 19), ((69, (207, 1)), 3))>
  return %result : !fly.layout<((2, 7), ((3, (2, 3)), 3)) : ((133, 19), ((69, (207, 1)), 3))>
}

// CHECK-LABEL: @pyir_zipped_divide_2d_bymode
func.func @pyir_zipped_divide_2d_bymode() -> !fly.layout<((2, (3, (2, 3))), (7, 3)) : ((133, (69, (207, 1))), (19, 3))> {
  %s = fly.static : !fly.int_tuple<(14, (6, 9))>
  %d = fly.static : !fly.int_tuple<(19, (69, 1))>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, (6, 9))>, !fly.int_tuple<(19, (69, 1))>) -> !fly.layout<(14, (6, 9)) : (19, (69, 1))>
  %tiler = fly.static : !fly.tile<[2:7|(3, 6):(1, 3)]>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %tiler) : (!fly.layout<(14, (6, 9)) : (19, (69, 1))>, !fly.tile<[2:7|(3, 6):(1, 3)]>) -> !fly.layout<((2, (3, (2, 3))), (7, 3)) : ((133, (69, (207, 1))), (19, 3))>
  return %result : !fly.layout<((2, (3, (2, 3))), (7, 3)) : ((133, (69, (207, 1))), (19, 3))>
}

// CHECK-LABEL: @pyir_tiled_divide_2d_bymode
func.func @pyir_tiled_divide_2d_bymode() -> !fly.layout<((2, (3, (2, 3))), 7, 3) : ((133, (69, (207, 1))), 19, 3)> {
  %s = fly.static : !fly.int_tuple<(14, (6, 9))>
  %d = fly.static : !fly.int_tuple<(19, (69, 1))>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, (6, 9))>, !fly.int_tuple<(19, (69, 1))>) -> !fly.layout<(14, (6, 9)) : (19, (69, 1))>
  %tiler = fly.static : !fly.tile<[2:7|(3, 6):(1, 3)]>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %tiler) : (!fly.layout<(14, (6, 9)) : (19, (69, 1))>, !fly.tile<[2:7|(3, 6):(1, 3)]>) -> !fly.layout<((2, (3, (2, 3))), 7, 3) : ((133, (69, (207, 1))), 19, 3)>
  return %result : !fly.layout<((2, (3, (2, 3))), 7, 3) : ((133, (69, (207, 1))), 19, 3)>
}

// CHECK-LABEL: @pyir_flat_divide_2d_bymode
func.func @pyir_flat_divide_2d_bymode() -> !fly.layout<(2, (3, (2, 3)), 7, 3) : (133, (69, (207, 1)), 19, 3)> {
  %s = fly.static : !fly.int_tuple<(14, (6, 9))>
  %d = fly.static : !fly.int_tuple<(19, (69, 1))>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(14, (6, 9))>, !fly.int_tuple<(19, (69, 1))>) -> !fly.layout<(14, (6, 9)) : (19, (69, 1))>
  %tiler = fly.static : !fly.tile<[2:7|(3, 6):(1, 3)]>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %tiler) : (!fly.layout<(14, (6, 9)) : (19, (69, 1))>, !fly.tile<[2:7|(3, 6):(1, 3)]>) -> !fly.layout<(2, (3, (2, 3)), 7, 3) : (133, (69, (207, 1)), 19, 3)>
  return %result : !fly.layout<(2, (3, (2, 3)), 7, 3) : (133, (69, (207, 1)), 19, 3)>
}

// Tiler shape drives the result shape. All cases use (96, 5) : (1, 256).

// A leaf tiler keeps zipped_divide identical to logical_divide.

// CHECK-LABEL: @divide_tiler_leaf_logical
func.func @divide_tiler_leaf_logical(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(16, (6, 5)) : (1, (16, 256))> {
  %tiler = fly.static : !fly.tile<16>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<16>) -> !fly.layout<(16, (6, 5)) : (1, (16, 256))>
  return %result : !fly.layout<(16, (6, 5)) : (1, (16, 256))>
}

// CHECK-LABEL: @divide_tiler_leaf_zipped
func.func @divide_tiler_leaf_zipped(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(16, (6, 5)) : (1, (16, 256))> {
  %tiler = fly.static : !fly.tile<16>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<16>) -> !fly.layout<(16, (6, 5)) : (1, (16, 256))>
  return %result : !fly.layout<(16, (6, 5)) : (1, (16, 256))>
}

// CHECK-LABEL: @divide_tiler_leaf_tiled
func.func @divide_tiler_leaf_tiled(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(16, 6, 5) : (1, 16, 256)> {
  %tiler = fly.static : !fly.tile<16>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<16>) -> !fly.layout<(16, 6, 5) : (1, 16, 256)>
  return %result : !fly.layout<(16, 6, 5) : (1, 16, 256)>
}

// A leaf tiler leaves the tile group a scalar, so flat matches tiled.

// CHECK-LABEL: @divide_tiler_leaf_flat
func.func @divide_tiler_leaf_flat(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(16, 6, 5) : (1, 16, 256)> {
  %tiler = fly.static : !fly.tile<16>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<16>) -> !fly.layout<(16, 6, 5) : (1, 16, 256)>
  return %result : !fly.layout<(16, 6, 5) : (1, 16, 256)>
}

// A singleton tuple tiler must NOT collapse into logical_divide: same layout,
// same tiler, result types have to differ.

// CHECK-LABEL: @divide_tiler_singleton_logical
func.func @divide_tiler_singleton_logical(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<((16, 6), 5) : ((1, 16), 256)> {
  %tiler = fly.static : !fly.tile<[16]>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16]>) -> !fly.layout<((16, 6), 5) : ((1, 16), 256)>
  return %result : !fly.layout<((16, 6), 5) : ((1, 16), 256)>
}

// CHECK-LABEL: @divide_tiler_singleton_zipped
func.func @divide_tiler_singleton_zipped(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<((16), (6, 5)) : ((1), (16, 256))> {
  %tiler = fly.static : !fly.tile<[16]>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16]>) -> !fly.layout<((16), (6, 5)) : ((1), (16, 256))>
  return %result : !fly.layout<((16), (6, 5)) : ((1), (16, 256))>
}

// tiled keeps the tile group, flat flattens both groups.

// CHECK-LABEL: @divide_tiler_singleton_tiled
func.func @divide_tiler_singleton_tiled(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<((16), 6, 5) : ((1), 16, 256)> {
  %tiler = fly.static : !fly.tile<[16]>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16]>) -> !fly.layout<((16), 6, 5) : ((1), 16, 256)>
  return %result : !fly.layout<((16), 6, 5) : ((1), 16, 256)>
}

// CHECK-LABEL: @divide_tiler_singleton_flat
func.func @divide_tiler_singleton_flat(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(16, 6, 5) : (1, 16, 256)> {
  %tiler = fly.static : !fly.tile<[16]>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16]>) -> !fly.layout<(16, 6, 5) : (1, 16, 256)>
  return %result : !fly.layout<(16, 6, 5) : (1, 16, 256)>
}

// Both modes covered: zip regroups across modes.

// CHECK-LABEL: @divide_tiler_two_modes_logical
func.func @divide_tiler_two_modes_logical(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<((16, 6), (5, 1)) : ((1, 16), (256, 0))> {
  %tiler = fly.static : !fly.tile<[16|5]>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16|5]>) -> !fly.layout<((16, 6), (5, 1)) : ((1, 16), (256, 0))>
  return %result : !fly.layout<((16, 6), (5, 1)) : ((1, 16), (256, 0))>
}

// CHECK-LABEL: @divide_tiler_two_modes_zipped
func.func @divide_tiler_two_modes_zipped(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<((16, 5), (6, 1)) : ((1, 256), (16, 0))> {
  %tiler = fly.static : !fly.tile<[16|5]>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16|5]>) -> !fly.layout<((16, 5), (6, 1)) : ((1, 256), (16, 0))>
  return %result : !fly.layout<((16, 5), (6, 1)) : ((1, 256), (16, 0))>
}

// The tile group is a real tuple here, so tiled keeps it and flat splits it.

// CHECK-LABEL: @divide_tiler_two_modes_tiled
func.func @divide_tiler_two_modes_tiled(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<((16, 5), 6, 1) : ((1, 256), 16, 0)> {
  %tiler = fly.static : !fly.tile<[16|5]>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16|5]>) -> !fly.layout<((16, 5), 6, 1) : ((1, 256), 16, 0)>
  return %result : !fly.layout<((16, 5), 6, 1) : ((1, 256), 16, 0)>
}

// CHECK-LABEL: @divide_tiler_two_modes_flat
func.func @divide_tiler_two_modes_flat(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(16, 5, 6, 1) : (1, 256, 16, 0)> {
  %tiler = fly.static : !fly.tile<[16|5]>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16|5]>) -> !fly.layout<(16, 5, 6, 1) : (1, 256, 16, 0)>
  return %result : !fly.layout<(16, 5, 6, 1) : (1, 256, 16, 0)>
}

// A nested tiler must recurse: flattening the guide to one level would put a
// rest mode in the tile group, giving ((2,3),4) instead of ((2,2),4).

// CHECK-LABEL: @divide_tiler_nested_logical
func.func @divide_tiler_nested_logical(%layout: !fly.layout<((6, 4), 8) : ((1, 6), 24)>)
    -> !fly.layout<(((2, 3), (2, 2)), (4, 2)) : (((1, 2), (6, 12)), (24, 96))> {
  %tiler = fly.static : !fly.tile<[[2|2]|4]>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<((6, 4), 8) : ((1, 6), 24)>, !fly.tile<[[2|2]|4]>) -> !fly.layout<(((2, 3), (2, 2)), (4, 2)) : (((1, 2), (6, 12)), (24, 96))>
  return %result : !fly.layout<(((2, 3), (2, 2)), (4, 2)) : (((1, 2), (6, 12)), (24, 96))>
}

// CHECK-LABEL: @divide_tiler_nested_zipped
func.func @divide_tiler_nested_zipped(%layout: !fly.layout<((6, 4), 8) : ((1, 6), 24)>)
    -> !fly.layout<(((2, 2), 4), ((3, 2), 2)) : (((1, 6), 24), ((2, 12), 96))> {
  %tiler = fly.static : !fly.tile<[[2|2]|4]>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %tiler) : (!fly.layout<((6, 4), 8) : ((1, 6), 24)>, !fly.tile<[[2|2]|4]>) -> !fly.layout<(((2, 2), 4), ((3, 2), 2)) : (((1, 6), 24), ((2, 12), 96))>
  return %result : !fly.layout<(((2, 2), 4), ((3, 2), 2)) : (((1, 6), 24), ((2, 12), 96))>
}

// CHECK-LABEL: @divide_tiler_nested_tiled
func.func @divide_tiler_nested_tiled(%layout: !fly.layout<((6, 4), 8) : ((1, 6), 24)>)
    -> !fly.layout<(((2, 2), 4), (3, 2), 2) : (((1, 6), 24), (2, 12), 96)> {
  %tiler = fly.static : !fly.tile<[[2|2]|4]>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %tiler) : (!fly.layout<((6, 4), 8) : ((1, 6), 24)>, !fly.tile<[[2|2]|4]>) -> !fly.layout<(((2, 2), 4), (3, 2), 2) : (((1, 6), 24), (2, 12), 96)>
  return %result : !fly.layout<(((2, 2), 4), (3, 2), 2) : (((1, 6), 24), (2, 12), 96)>
}

// CHECK-LABEL: @divide_tiler_nested_flat
func.func @divide_tiler_nested_flat(%layout: !fly.layout<((6, 4), 8) : ((1, 6), 24)>)
    -> !fly.layout<((2, 2), 4, (3, 2), 2) : ((1, 6), 24, (2, 12), 96)> {
  %tiler = fly.static : !fly.tile<[[2|2]|4]>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %tiler) : (!fly.layout<((6, 4), 8) : ((1, 6), 24)>, !fly.tile<[[2|2]|4]>) -> !fly.layout<((2, 2), 4, (3, 2), 2) : ((1, 6), 24, (2, 12), 96)>
  return %result : !fly.layout<((2, 2), 4, (3, 2), 2) : ((1, 6), 24, (2, 12), 96)>
}

// Mixing numbers and `*` in one tuple tiler. No zipped/tiled/flat counterparts
// on purpose: a `*` inside a tuple tiler still aborts on the terminal rank
// check. Add them once that becomes a diagnostic.

// CHECK-LABEL: @divide_tiler_num_then_skip_logical
func.func @divide_tiler_num_then_skip_logical(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<((16, 6), 5) : ((1, 16), 256)> {
  %tiler = fly.static : !fly.tile<[16|*]>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[16|*]>) -> !fly.layout<((16, 6), 5) : ((1, 16), 256)>
  return %result : !fly.layout<((16, 6), 5) : ((1, 16), 256)>
}

// A leading `*` keeps mode 0 whole and still divides mode 1.

// CHECK-LABEL: @divide_tiler_skip_then_num_logical
func.func @divide_tiler_skip_then_num_logical(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(96, (5, 1)) : (1, (256, 0))> {
  %tiler = fly.static : !fly.tile<[*|5]>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[*|5]>) -> !fly.layout<(96, (5, 1)) : (1, (256, 0))>
  return %result : !fly.layout<(96, (5, 1)) : (1, (256, 0))>
}

// A `*` wrapped in a tuple behaves like a leaf `*` for logical_divide.

// CHECK-LABEL: @divide_tiler_skip_in_tuple_logical
func.func @divide_tiler_skip_in_tuple_logical(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(96, 5) : (1, 256)> {
  %tiler = fly.static : !fly.tile<[*]>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<[*]>) -> !fly.layout<(96, 5) : (1, 256)>
  return %result : !fly.layout<(96, 5) : (1, 256)>
}

// A leaf `*` divides nothing: the layout passes through all four ops.

// CHECK-LABEL: @divide_tiler_skip_all_logical
func.func @divide_tiler_skip_all_logical(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(96, 5) : (1, 256)> {
  %tiler = fly.static : !fly.tile<*>
  // CHECK: fly.logical_divide
  %result = fly.logical_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<*>) -> !fly.layout<(96, 5) : (1, 256)>
  return %result : !fly.layout<(96, 5) : (1, 256)>
}

// CHECK-LABEL: @divide_tiler_skip_all_zipped
func.func @divide_tiler_skip_all_zipped(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(96, 5) : (1, 256)> {
  %tiler = fly.static : !fly.tile<*>
  // CHECK: fly.zipped_divide
  %result = fly.zipped_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<*>) -> !fly.layout<(96, 5) : (1, 256)>
  return %result : !fly.layout<(96, 5) : (1, 256)>
}

// CHECK-LABEL: @divide_tiler_skip_all_tiled
func.func @divide_tiler_skip_all_tiled(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(96, 5) : (1, 256)> {
  %tiler = fly.static : !fly.tile<*>
  // CHECK: fly.tiled_divide
  %result = fly.tiled_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<*>) -> !fly.layout<(96, 5) : (1, 256)>
  return %result : !fly.layout<(96, 5) : (1, 256)>
}

// CHECK-LABEL: @divide_tiler_skip_all_flat
func.func @divide_tiler_skip_all_flat(%layout: !fly.layout<(96, 5) : (1, 256)>)
    -> !fly.layout<(96, 5) : (1, 256)> {
  %tiler = fly.static : !fly.tile<*>
  // CHECK: fly.flat_divide
  %result = fly.flat_divide(%layout, %tiler) : (!fly.layout<(96, 5) : (1, 256)>, !fly.tile<*>) -> !fly.layout<(96, 5) : (1, 256)>
  return %result : !fly.layout<(96, 5) : (1, 256)>
}

// Product family baseline, block (12, 3) : (1, 16) over tiler (3, 2) : (1, 4).
// zipped_product equals logical_product; blocked and raked interleave the block
// and tiler modes in opposite orders.

// CHECK-LABEL: @product_baseline_logical
func.func @product_baseline_logical(%block: !fly.layout<(12, 3) : (1, 16)>, %tiler: !fly.layout<(3, 2) : (1, 4)>)
    -> !fly.layout<((12, 3), (3, 2)) : ((1, 16), (48, 192))> {
  // CHECK: fly.logical_product
  %result = fly.logical_product(%block, %tiler) : (!fly.layout<(12, 3) : (1, 16)>, !fly.layout<(3, 2) : (1, 4)>) -> !fly.layout<((12, 3), (3, 2)) : ((1, 16), (48, 192))>
  return %result : !fly.layout<((12, 3), (3, 2)) : ((1, 16), (48, 192))>
}

// CHECK-LABEL: @product_baseline_zipped
func.func @product_baseline_zipped(%block: !fly.layout<(12, 3) : (1, 16)>, %tiler: !fly.layout<(3, 2) : (1, 4)>)
    -> !fly.layout<((12, 3), (3, 2)) : ((1, 16), (48, 192))> {
  // CHECK: fly.zipped_product
  %result = fly.zipped_product(%block, %tiler) : (!fly.layout<(12, 3) : (1, 16)>, !fly.layout<(3, 2) : (1, 4)>) -> !fly.layout<((12, 3), (3, 2)) : ((1, 16), (48, 192))>
  return %result : !fly.layout<((12, 3), (3, 2)) : ((1, 16), (48, 192))>
}

// CHECK-LABEL: @product_baseline_tiled
func.func @product_baseline_tiled(%block: !fly.layout<(12, 3) : (1, 16)>, %tiler: !fly.layout<(3, 2) : (1, 4)>)
    -> !fly.layout<((12, 3), 3, 2) : ((1, 16), 48, 192)> {
  // CHECK: fly.tiled_product
  %result = fly.tiled_product(%block, %tiler) : (!fly.layout<(12, 3) : (1, 16)>, !fly.layout<(3, 2) : (1, 4)>) -> !fly.layout<((12, 3), 3, 2) : ((1, 16), 48, 192)>
  return %result : !fly.layout<((12, 3), 3, 2) : ((1, 16), 48, 192)>
}

// CHECK-LABEL: @product_baseline_flat
func.func @product_baseline_flat(%block: !fly.layout<(12, 3) : (1, 16)>, %tiler: !fly.layout<(3, 2) : (1, 4)>)
    -> !fly.layout<(12, 3, 3, 2) : (1, 16, 48, 192)> {
  // CHECK: fly.flat_product
  %result = fly.flat_product(%block, %tiler) : (!fly.layout<(12, 3) : (1, 16)>, !fly.layout<(3, 2) : (1, 4)>) -> !fly.layout<(12, 3, 3, 2) : (1, 16, 48, 192)>
  return %result : !fly.layout<(12, 3, 3, 2) : (1, 16, 48, 192)>
}

// CHECK-LABEL: @product_baseline_blocked
func.func @product_baseline_blocked(%block: !fly.layout<(12, 3) : (1, 16)>, %tiler: !fly.layout<(3, 2) : (1, 4)>)
    -> !fly.layout<((12, 3), (3, 2)) : ((1, 48), (16, 192))> {
  // CHECK: fly.blocked_product
  %result = fly.blocked_product(%block, %tiler) : (!fly.layout<(12, 3) : (1, 16)>, !fly.layout<(3, 2) : (1, 4)>) -> !fly.layout<((12, 3), (3, 2)) : ((1, 48), (16, 192))>
  return %result : !fly.layout<((12, 3), (3, 2)) : ((1, 48), (16, 192))>
}

// CHECK-LABEL: @product_baseline_raked
func.func @product_baseline_raked(%block: !fly.layout<(12, 3) : (1, 16)>, %tiler: !fly.layout<(3, 2) : (1, 4)>)
    -> !fly.layout<((3, 12), (2, 3)) : ((48, 1), (192, 16))> {
  // CHECK: fly.raked_product
  %result = fly.raked_product(%block, %tiler) : (!fly.layout<(12, 3) : (1, 16)>, !fly.layout<(3, 2) : (1, 4)>) -> !fly.layout<((3, 12), (2, 3)) : ((48, 1), (192, 16))>
  return %result : !fly.layout<((3, 12), (2, 3)) : ((48, 1), (192, 16))>
}
