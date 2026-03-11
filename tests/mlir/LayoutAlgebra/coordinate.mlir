// RUN: %fly-opt %s | FileCheck %s

// Tests for coordinate mapping operations:
//   fly.crd2idx, fly.idx2crd, fly.get_flat_coord, fly.get_1d_coord

// -----

// CHECK-LABEL: @test_crd2idx_static
func.func @test_crd2idx_static() -> !fly.int_tuple<14> {
  // crd2idx((2,3), Layout<(4,8):(1,4)>) = 2*1 + 3*4 = 14
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %coord = fly.static : !fly.int_tuple<(2, 3)>
  // CHECK: fly.crd2idx(%{{.*}}, %{{.*}}) : (!fly.int_tuple<(2,3)>, !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<14>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(2, 3)>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<14>
  return %idx : !fly.int_tuple<14>
}

// CHECK-LABEL: @test_crd2idx_origin
func.func @test_crd2idx_origin() -> !fly.int_tuple<0> {
  // crd2idx((0,0), Layout<(4,8):(1,4)>) = 0*1 + 0*4 = 0
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %coord = fly.static : !fly.int_tuple<(0, 0)>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(0, 0)>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<0>
  return %idx : !fly.int_tuple<0>
}

// CHECK-LABEL: @test_crd2idx_row_major
func.func @test_crd2idx_row_major() -> !fly.int_tuple<19> {
  // Row-major (4,8):(8,1): crd2idx((2,3)) = 2*8 + 3*1 = 19
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(8, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(8, 1)>) -> !fly.layout<(4, 8) : (8, 1)>
  %coord = fly.static : !fly.int_tuple<(2, 3)>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(2, 3)>, !fly.layout<(4, 8) : (8, 1)>) -> !fly.int_tuple<19>
  return %idx : !fly.int_tuple<19>
}

// CHECK-LABEL: @test_idx2crd
func.func @test_idx2crd() -> !fly.int_tuple<(2, 3)> {
  // idx2crd(14, Layout<(4,8):(1,4)>) = (14 % 4, 14 / 4 % 8) = (2, 3)
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %idx = fly.static : !fly.int_tuple<14>
  // CHECK: fly.idx2crd(%{{.*}}, %{{.*}}) : (!fly.int_tuple<14>, !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<(2,3)>
  %crd = fly.idx2crd(%idx, %layout) : (!fly.int_tuple<14>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(2, 3)>
  return %crd : !fly.int_tuple<(2, 3)>
}

// CHECK-LABEL: @test_crd2idx_dynamic
func.func @test_crd2idx_dynamic(%c0: i32, %c1: i32) -> !fly.int_tuple<?> {
  %s = fly.static : !fly.int_tuple<(4, 8)>
  %d = fly.static : !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %coord = fly.make_coord(%c0, %c1) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  // CHECK: fly.crd2idx(%{{.*}}, %{{.*}}) : (!fly.int_tuple<(?,?)>, !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<?>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(?, ?)>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<?>
  return %idx : !fly.int_tuple<?>
}

// CHECK-LABEL: @test_get_flat_coord
func.func @test_get_flat_coord() -> !fly.int_tuple<(5, 0)> {
  // get_flat_coord with a depth>1 layout: ((2,4),8):((1,2),8)
  // Flattens the hierarchical index 5 into per-mode flat coordinates
  %s1 = fly.static : !fly.int_tuple<((2, 4), 8)>
  %d1 = fly.static : !fly.int_tuple<((1, 2), 8)>
  %layout = fly.make_layout(%s1, %d1) : (!fly.int_tuple<((2, 4), 8)>, !fly.int_tuple<((1, 2), 8)>) -> !fly.layout<((2, 4), 8) : ((1, 2), 8)>
  %idx = fly.static : !fly.int_tuple<5>
  // CHECK: fly.get_flat_coord(%{{.*}}, %{{.*}}) : (!fly.int_tuple<5>, !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<(5,0)>
  %crd = fly.get_flat_coord(%idx, %layout) : (!fly.int_tuple<5>, !fly.layout<((2, 4), 8) : ((1, 2), 8)>) -> !fly.int_tuple<(5, 0)>
  return %crd : !fly.int_tuple<(5, 0)>
}

// CHECK-LABEL: @test_get_flat_coord_dynamic
func.func @test_get_flat_coord_dynamic(%c: i32) -> !fly.int_tuple<(?, ?)> {
  %s1 = fly.static : !fly.int_tuple<((2, 4), 8)>
  %d1 = fly.static : !fly.int_tuple<((1, 2), 8)>
  %layout = fly.make_layout(%s1, %d1) : (!fly.int_tuple<((2, 4), 8)>, !fly.int_tuple<((1, 2), 8)>) -> !fly.layout<((2, 4), 8) : ((1, 2), 8)>
  %idx = fly.make_int_tuple(%c) : (i32) -> !fly.int_tuple<?>
  // CHECK: fly.get_flat_coord(%{{.*}}, %{{.*}}) : (!fly.int_tuple<?>, !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<(?,?)>
  %crd = fly.get_flat_coord(%idx, %layout) : (!fly.int_tuple<?>, !fly.layout<((2, 4), 8) : ((1, 2), 8)>) -> !fly.int_tuple<(?, ?)>
  return %crd : !fly.int_tuple<(?, ?)>
}

// CHECK-LABEL: @test_get_1d_coord
func.func @test_get_1d_coord() -> !fly.int_tuple<5> {
  // get_1d_coord maps a flat index through a depth>1 layout to a 1D index
  %s1 = fly.static : !fly.int_tuple<((2, 4), 8)>
  %d1 = fly.static : !fly.int_tuple<((1, 2), 8)>
  %layout = fly.make_layout(%s1, %d1) : (!fly.int_tuple<((2, 4), 8)>, !fly.int_tuple<((1, 2), 8)>) -> !fly.layout<((2, 4), 8) : ((1, 2), 8)>
  %idx = fly.static : !fly.int_tuple<5>
  // CHECK: fly.get_1d_coord(%{{.*}}, %{{.*}}) : (!fly.int_tuple<5>, !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<5>
  %crd = fly.get_1d_coord(%idx, %layout) : (!fly.int_tuple<5>, !fly.layout<((2, 4), 8) : ((1, 2), 8)>) -> !fly.int_tuple<5>
  return %crd : !fly.int_tuple<5>
}

// CHECK-LABEL: @test_get_1d_coord_dynamic
func.func @test_get_1d_coord_dynamic(%c: i32) -> !fly.int_tuple<?> {
  %s1 = fly.static : !fly.int_tuple<((2, 4), 8)>
  %d1 = fly.static : !fly.int_tuple<((1, 2), 8)>
  %layout = fly.make_layout(%s1, %d1) : (!fly.int_tuple<((2, 4), 8)>, !fly.int_tuple<((1, 2), 8)>) -> !fly.layout<((2, 4), 8) : ((1, 2), 8)>
  %idx = fly.make_int_tuple(%c) : (i32) -> !fly.int_tuple<?>
  // CHECK: fly.get_1d_coord(%{{.*}}, %{{.*}}) : (!fly.int_tuple<?>, !fly.layout<((2,4),8):((1,2),8)>) -> !fly.int_tuple<?>
  %crd = fly.get_1d_coord(%idx, %layout) : (!fly.int_tuple<?>, !fly.layout<((2, 4), 8) : ((1, 2), 8)>) -> !fly.int_tuple<?>
  return %crd : !fly.int_tuple<?>
}
