// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// gfx1250 TDM gather copy atom lowering. Rows selected by explicit indices
// (carried as a packed vector<8xi32> atom state field) are moved Global<->LDS.
// Descriptor groups 2/3 carry the indices; groups 0/1 carry addr + config.
//   Global -> Shared  =>  rocdl.tensor.load.to.lds
//   Shared -> Global  =>  rocdl.tensor.store.from.lds

// -----

// Stateful gather atom type converts to the descriptor struct
// {mask, base, tensor_dim1, tensor_dim0, stride, row_indices, count, imm_offset}.
// CHECK-LABEL: @test_gather_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, ptr<1>, i32, i32, i32, vector<8xi32>, i32, i64)>)
func.func @test_gather_type(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>) {
  return
}

// -----

// 32-bit gather load: indices ride in row_indices (slot 5), split into groups
// 2/3 via vector.shuffle. GROUP0 pred = 1 | (1<<30 gather-index) | (1<<31 type)
// = 0xC0000001 = -1073741823. row_width (tile inner = 64) packs into GROUP1 s3
// at bit 16 (64<<16 = 4194304). gather_count (slot 6) falls back to the tile row
// count (8) when left unset.

// CHECK-LABEL: @test_gather_load
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, ptr<1>, i32, i32, i32, vector<8xi32>, i32, i64)>, %[[IDX:.*]]: vector<8xi32>, %[[NR:.*]]: i32,
func.func @test_gather_load(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>,
    %idx: vector<8xi32>, %nrows: i32,
    %src: !fly.memref<f16, global, (8,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (8,64):(64,1)>) {
  // CHECK: llvm.insertvalue %[[IDX]], %{{.*}}[5]
  %a1 = fly.atom.set_value(%atom, "row_indices", %idx) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>, vector<8xi32>) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>
  // outer_extent (tensor_dim1) = runtime row count for OOB on indices.
  %a2 = fly.atom.set_value(%a1, "extent_0", %nrows) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>
  // gather_count (slot 6) unset -> select to tile row count 8.
  // CHECK-DAG: %[[CNT:.*]] = llvm.extractvalue %{{.*}}[6]
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK-DAG: arith.select %{{.*}}, %[[C8]], %[[CNT]]
  // pred word with the gather-index + type bits set.
  // CHECK-DAG: arith.constant -1073741823 : i32
  // row_width (64) packed at bit 16.
  // CHECK-DAG: arith.constant 4194304 : i32
  // GROUP2/GROUP3: the row-index vector split 0..3 / 4..7.
  // CHECK-DAG: %[[RI:.*]] = llvm.extractvalue %{{.*}}[5]
  // CHECK-DAG: vector.shuffle %[[RI]], %[[RI]] [0, 1, 2, 3]
  // CHECK-DAG: vector.shuffle %[[RI]], %[[RI]] [4, 5, 6, 7]
  // CHECK: rocdl.tensor.load.to.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} cachepolicy 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%a2, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>, !fly.memref<f16, global, (8,64):(64,1)>, !fly.memref<f16, shared, (8,64):(64,1)>) -> ()
  return
}

// -----

// Store direction (Shared -> Global) -> tensor.store.from.lds.

// CHECK-LABEL: @test_gather_store
func.func @test_gather_store(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>,
    %idx: vector<8xi32>,
    %src: !fly.memref<f16, shared, (8,64):(64,1)>,
    %dst: !fly.memref<f16, global, (8,64):(64,1)>) {
  %a1 = fly.atom.set_value(%atom, "row_indices", %idx) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>, vector<8xi32>) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>
  // CHECK: rocdl.tensor.store.from.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} cachepolicy 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%a1, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 32, pad = 0, 0, cache = 0>, 0>, !fly.memref<f16, shared, (8,64):(64,1)>, !fly.memref<f16, global, (8,64):(64,1)>) -> ()
  return
}

// -----

// 16-bit gather: the gather-index bit [30] is 0 (pred = 1 | (1<<31) = 0x80000001
// = -2147483647). The index vector is already packed 2-per-lane by the builder.

// CHECK-LABEL: @test_gather_load_i16
func.func @test_gather_load_i16(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 16, pad = 0, 0, cache = 0>, 0>,
    %idx: vector<8xi32>,
    %src: !fly.memref<f16, global, (16,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (16,64):(64,1)>) {
  %a1 = fly.atom.set_value(%atom, "row_indices", %idx) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 16, pad = 0, 0, cache = 0>, 0>, vector<8xi32>) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 16, pad = 0, 0, cache = 0>, 0>
  // CHECK-DAG: arith.constant -2147483647 : i32
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%a1, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<index = 16, pad = 0, 0, cache = 0>, 0>, !fly.memref<f16, global, (16,64):(64,1)>, !fly.memref<f16, shared, (16,64):(64,1)>) -> ()
  return
}
