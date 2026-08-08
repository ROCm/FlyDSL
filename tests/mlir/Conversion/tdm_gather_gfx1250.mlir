// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// gfx1250 TDM row gather/scatter (index_width 16/32 selects gather; rank 2). The
// row-index buffer rides as the `index_ptr` atom state (slot 11, a global pointer);
// its element width is the atom's index_width and the row count comes from the tile
// layout's outer extent. The global base pointer comes from the copy_atom_call
// operand (as in the tiled path). Gather adds one struct slot vs tiled:
//   {mask, extent_0..4, stride_0..3, imm_offset, index_ptr}.

// -----

// The gather atom's converted state struct carries the extra index_ptr slot.
// CHECK-LABEL: @test_tdm_gather_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, ptr<1>)>)
func.func @test_tdm_gather_type(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false, iwidth = 32>, 0>) {
  return
}

// -----

// i32 gather load (8 rows): the index buffer pointer is read from slot 11, each
// row index is loaded (i32, one per descriptor word) and the tile is gathered
// Global -> Shared via rocdl.tensor.load.to.lds.
// CHECK-LABEL: @test_tdm_gather_load_i32
// CHECK: %[[IDXP:.*]] = llvm.extractvalue %{{.*}}[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, ptr<1>)>
// CHECK: llvm.getelementptr %[[IDXP]]{{.*}} -> !llvm.ptr<1>, i32
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> i32
// CHECK: rocdl.tensor.load.to.lds {{.*}} : vector<4xi32>, vector<8xi32>
func.func @test_tdm_gather_load_i32(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false, iwidth = 32>, 0>,
    %src: !fly.memref<f16, global, (8,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (8,64):(64,1)>) {
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false, iwidth = 32>, 0>, !fly.memref<f16, global, (8,64):(64,1)>, !fly.memref<f16, shared, (8,64):(64,1)>) -> ()
  return
}

// -----

// i16 gather store (8 rows): 16-bit indices are packed two per descriptor word
// (lo | hi<<16) and the tile is scattered Shared -> Global via
// rocdl.tensor.store.from.lds.
// CHECK-LABEL: @test_tdm_gather_store_i16
// CHECK: %[[IDXP:.*]] = llvm.extractvalue %{{.*}}[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, ptr<1>)>
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> i16
// CHECK: llvm.zext {{.*}} : i16 to i32
// CHECK: rocdl.tensor.store.from.lds {{.*}} : vector<4xi32>, vector<8xi32>
func.func @test_tdm_gather_store_i16(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false, iwidth = 16>, 0>,
    %src: !fly.memref<f16, shared, (8,64):(64,1)>,
    %dst: !fly.memref<f16, global, (8,64):(64,1)>) {
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false, iwidth = 16>, 0>, !fly.memref<f16, shared, (8,64):(64,1)>, !fly.memref<f16, global, (8,64):(64,1)>) -> ()
  return
}
