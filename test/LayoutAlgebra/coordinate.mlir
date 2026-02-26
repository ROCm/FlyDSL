// RUN: %fly-opt %s | FileCheck %s

// Tests for coordinate mapping operations:
//   fly.crd2idx, fly.idx2crd, fly.get_flat_coord, fly.get_hier_coord

// -----

// CHECK-LABEL: @test_crd2idx_static
func.func @test_crd2idx_static() -> !fly.int_tuple<14> {
  // crd2idx((2,3), Layout<(4,8):(1,4)>) = 2*1 + 3*4 = 14
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %coord = fly.static {elems = [2 : i32, 3 : i32]} : () -> !fly.int_tuple<(2, 3)>
  // CHECK: fly.crd2idx(%{{.*}}, %{{.*}}) : (!fly.int_tuple<(2,3)>, !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<14>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(2, 3)>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<14>
  return %idx : !fly.int_tuple<14>
}

// CHECK-LABEL: @test_crd2idx_origin
func.func @test_crd2idx_origin() -> !fly.int_tuple<0> {
  // crd2idx((0,0), Layout<(4,8):(1,4)>) = 0*1 + 0*4 = 0
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %coord = fly.static {elems = [0 : i32, 0 : i32]} : () -> !fly.int_tuple<(0, 0)>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(0, 0)>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<0>
  return %idx : !fly.int_tuple<0>
}

// CHECK-LABEL: @test_crd2idx_row_major
func.func @test_crd2idx_row_major() -> !fly.int_tuple<19> {
  // Row-major (4,8):(8,1): crd2idx((2,3)) = 2*8 + 3*1 = 19
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [8 : i32, 1 : i32]} : () -> !fly.int_tuple<(8, 1)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(8, 1)>) -> !fly.layout<(4, 8) : (8, 1)>
  %coord = fly.static {elems = [2 : i32, 3 : i32]} : () -> !fly.int_tuple<(2, 3)>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(2, 3)>, !fly.layout<(4, 8) : (8, 1)>) -> !fly.int_tuple<19>
  return %idx : !fly.int_tuple<19>
}

// CHECK-LABEL: @test_idx2crd
func.func @test_idx2crd() -> !fly.int_tuple<(4, 8)> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %idx = fly.static {elems = [14 : i32]} : () -> !fly.int_tuple<14>
  // CHECK: fly.idx2crd(%{{.*}}, %{{.*}}) : (!fly.int_tuple<14>, !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<(4,8)>
  %crd = fly.idx2crd(%idx, %layout) : (!fly.int_tuple<14>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<(4, 8)>
  return %crd : !fly.int_tuple<(4, 8)>
}

// CHECK-LABEL: @test_crd2idx_dynamic
func.func @test_crd2idx_dynamic(%c0: i32, %c1: i32) -> !fly.int_tuple<?> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %d = fly.static {elems = [1 : i32, 4 : i32]} : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  %coord = fly.make_coord(%c0, %c1) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  // CHECK: fly.crd2idx(%{{.*}}, %{{.*}}) : (!fly.int_tuple<(?,?)>, !fly.layout<(4,8):(1,4)>) -> !fly.int_tuple<?>
  %idx = fly.crd2idx(%coord, %layout) : (!fly.int_tuple<(?, ?)>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.int_tuple<?>
  return %idx : !fly.int_tuple<?>
}

// CHECK-LABEL: @test_get_flat_coord
func.func @test_get_flat_coord() -> !fly.int_tuple<(4, 8)> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %idx = fly.static {elems = [5 : i32]} : () -> !fly.int_tuple<5>
  // CHECK: fly.get_flat_coord(%{{.*}}, %{{.*}})
  %crd = fly.get_flat_coord(%idx, %s) : (!fly.int_tuple<5>, !fly.int_tuple<(4, 8)>) -> !fly.int_tuple<(4, 8)>
  return %crd : !fly.int_tuple<(4, 8)>
}

// CHECK-LABEL: @test_get_hier_coord
func.func @test_get_hier_coord() -> !fly.int_tuple<(4, 8)> {
  %s = fly.static {elems = [4 : i32, 8 : i32]} : () -> !fly.int_tuple<(4, 8)>
  %idx = fly.static {elems = [5 : i32]} : () -> !fly.int_tuple<5>
  // CHECK: fly.get_hier_coord(%{{.*}}, %{{.*}})
  %crd = fly.get_hier_coord(%idx, %s) : (!fly.int_tuple<5>, !fly.int_tuple<(4, 8)>) -> !fly.int_tuple<(4, 8)>
  return %crd : !fly.int_tuple<(4, 8)>
}
