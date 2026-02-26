// RUN: %fly-opt %s | FileCheck %s

// Tests for size and cosize operations on IntTuples and Layouts

// -----

// CHECK-LABEL: @test_size_int_tuple
func.func @test_size_int_tuple() -> !fly.int_tuple<32> {
  %0 = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  // CHECK: fly.size(%{{.*}}) : (!fly.int_tuple<(4,8)>) -> !fly.int_tuple<32>
  %1 = fly.size(%0) : (!fly.int_tuple<(4, 8)>) -> !fly.int_tuple<32>
  return %1 : !fly.int_tuple<32>
}

// CHECK-LABEL: @test_size_3d
func.func @test_size_3d() -> !fly.int_tuple<64> {
  // size((2, 4, 8)) = 2 * 4 * 8 = 64
  %0 = fly.static {elems = [2 : i32, 4 : i32, 8 : i32]} : () -> !fly.int_tuple<(2, 4, 8)>
  %1 = fly.size(%0) : (!fly.int_tuple<(2, 4, 8)>) -> !fly.int_tuple<64>
  return %1 : !fly.int_tuple<64>
}

// CHECK-LABEL: @test_size_scalar
func.func @test_size_scalar() -> !fly.int_tuple<16> {
  %0 = fly.static {elems = [16 : i32]} : () -> !fly.int_tuple<16>
  %1 = fly.size(%0) : (!fly.int_tuple<16>) -> !fly.int_tuple<16>
  return %1 : !fly.int_tuple<16>
}

// CHECK-LABEL: @test_size_layout
func.func @test_size_layout() -> !fly.int_tuple<32> {
  // size(Layout<(4,8):(1,4)>) = 4 * 8 = 32
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.size(%{{.*}}) : (!fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<32>
  %1 = fly.size(%layout) : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<32>
  return %1 : !fly.int_tuple<32>
}

// CHECK-LABEL: @test_cosize_contiguous
func.func @test_cosize_contiguous() -> !fly.int_tuple<32> {
  // cosize(Layout<(4,8):(1,4)>) = (4-1)*1 + (8-1)*4 + 1 = 3 + 28 + 1 = 32
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK: fly.cosize %{{.*}} : (!fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<32>
  %1 = fly.cosize %layout : (!fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<32>
  return %1 : !fly.int_tuple<32>
}

// CHECK-LABEL: @test_cosize_strided
func.func @test_cosize_strided() -> !fly.int_tuple<2040> {
  // cosize(Layout<(8,128):(1,16)>) = (8-1)*1 + (128-1)*16 + 1 = 7 + 2032 + 1 = 2040
  %s = fly.static {elems = [8 : i32, 128 : i32]} : () -> !fly.int_tuple<(8, 128)>
  %d = fly.static {elems = [1 : i32, 16 : i32]} : () -> !fly.int_tuple<(1, 16)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(8, 128)>, !fly.int_tuple<(1, 16)>) -> !fly.layout<(8, 128) : (1, 16)>
  %1 = fly.cosize %layout : (!fly.layout<(8, 128) : (1, 16)>) -> !fly.int_tuple<2040>
  return %1 : !fly.int_tuple<2040>
}
