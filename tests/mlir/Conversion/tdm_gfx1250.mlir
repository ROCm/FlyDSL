// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// gfx1250 N-D TDM copy atom lowering. The atom carries the tile descriptor as
// runtime state (base pointer, per-dim extent_i, per-dim stride_i, imm_offset);
// the global operand only marks the copy direction and supplies the compile-time
// tile shape. Struct: {mask, base, extent_0..4 (i32), stride_0..3 (i64), imm_offset
// (i64)}.
//   Global -> Shared  =>  rocdl.tensor.load.to.lds
//   Shared -> Global  =>  rocdl.tensor.store.from.lds

// -----

// CHECK-LABEL: @test_tdm_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, ptr<1>, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64)>)
func.func @test_tdm_type(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>) {
  return
}

// -----

// Load, single warp, rank 2: base / extents / stride come from atom state. The
// global operand's pointer is unused (only its layout gives the tile shape).
// extent_0 = outer (slot 2), extent_1 = inner (slot 3), stride_0 = i64 (slot 7).

// CHECK-LABEL: @test_tdm_load
func.func @test_tdm_load(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>,
    %oe: i32, %ie: i32, %os: i64,
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %a1 = fly.atom.set_value(%atom, "extent_0", %oe) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>
  %a2 = fly.atom.set_value(%a1, "extent_1", %ie) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>
  %a3 = fly.atom.set_value(%a2, "stride_0", %os) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, i64) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>
  // stride_0 fallback: select(stride == unset-sentinel (i64), static_layout_stride, stride)
  // CHECK-DAG: %[[STRIDE:.*]] = llvm.extractvalue %{{.*}}[7] : !llvm.struct<(i32, ptr<1>, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64)>
  // CHECK-DAG: %[[SENT:.*]] = arith.constant -2147483648 : i64
  // CHECK: arith.cmpi eq, %[[STRIDE]], %[[SENT]]
  // CHECK: arith.select
  // base (slot 1) -> global address via ptrtoint.
  // CHECK: %[[BASE:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i32, ptr<1>, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64)>
  // CHECK: llvm.ptrtoint %[[BASE]] : !llvm.ptr<1> to i64
  // OOB clamp from the extent state fields.
  // CHECK: arith.subi
  // CHECK: arith.maxsi
  // CHECK: rocdl.tensor.load.to.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} cachepolicy 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%a3, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// Store direction (Shared -> Global) -> tensor.store.from.lds.

// CHECK-LABEL: @test_tdm_store
func.func @test_tdm_store(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>,
    %src: !fly.memref<f16, shared, (128,64):(64,1)>,
    %dst: !fly.memref<f16, global, (128,64):(64,1)>) {
  // CHECK: rocdl.tensor.store.from.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} cachepolicy 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.memref<f16, global, (128,64):(64,1)>) -> ()
  return
}

// -----

// Multi-warp load distributes the tile: wave.id is read and the per-wave offsets
// scale the descriptor addresses. warps=4 over a 128x64 tile.

// CHECK-LABEL: @test_tdm_load_warps
func.func @test_tdm_load_warps(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 4, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>,
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  // CHECK: %[[WID:.*]] = rocdl.wave.id : i32
  // CHECK-DAG: arith.remui %[[WID]]
  // CHECK-DAG: arith.divui %[[WID]]
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 4, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// 3D load (rank 3): the descriptor fills group2 as well; tensor.load consumes 4
// descriptor groups. tile (8,16,32); strides via state (i64).

// CHECK-LABEL: @test_tdm_load_3d
func.func @test_tdm_load_3d(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 3, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>,
    %s0: i64, %s1: i64,
    %src: !fly.memref<f16, global, (8,16,32):(512,32,1)>,
    %dst: !fly.memref<f16, shared, (8,16,32):(512,32,1)>) {
  %a1 = fly.atom.set_value(%atom, "stride_0", %s0) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 3, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, i64) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 3, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>
  %a2 = fly.atom.set_value(%a1, "stride_1", %s1) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 3, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, i64) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 3, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>
  // CHECK: llvm.extractvalue %{{.*}}[8] : !llvm.struct<(i32, ptr<1>, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64)>
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%a2, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 3, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, !fly.memref<f16, global, (8,16,32):(512,32,1)>, !fly.memref<f16, shared, (8,16,32):(512,32,1)>) -> ()
  return
}

// -----

// Load with LDS padding (interval=64 elems, amount=8 elems, f16 -> 16-bit):
//   interval_dw = 64*16/32 = 32 -> enc_interval = log2(32)-1 = 4
//   amount_dw   = 8*16/32  = 4  -> enc_amount   = 4-1 = 3
//   g1_s0 = (1<<16) | (1<<20) | (4<<22) | (3<<25) = 118554624

// CHECK-LABEL: @test_tdm_load_pad
func.func @test_tdm_load_pad(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 64, 8, cache = 0, barrier = false, timeout = false>, 0>,
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  // CHECK-DAG: arith.constant 118554624 : i32
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 64, 8, cache = 0, barrier = false, timeout = false>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// MCAST: a runtime workgroup_mask (atom state slot 0) is ORed into GROUP1 config
// [15:0].

// CHECK-LABEL: @test_tdm_load_mcast
func.func @test_tdm_load_mcast(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>,
    %mask: i32,
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  // CHECK: %[[A1:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[0]
  %a1 = fly.atom.set_value(%atom, "workgroup_mask", %mask) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>
  // CHECK: %[[M:.*]] = llvm.extractvalue %[[A1]][0]
  // CHECK-DAG: %[[MLOW:.*]] = arith.andi %[[M]], %{{.*}}
  // CHECK-DAG: %[[UPPER:.*]] = arith.constant 65536 : i32
  // CHECK: arith.ori %[[UPPER]], %[[MLOW]]
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%a1, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// barrier=true (bit18) and timeout=true (bit21) set the descriptor config word
// GROUP1 sgpr0. f16 => data_size = log2(2) = 1 (bit16). Upper config =
// (1<<16) | (1<<18) | (1<<21) = 2424832.

// CHECK-LABEL: @test_tdm_load_barrier_timeout
func.func @test_tdm_load_barrier_timeout(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = true, timeout = true>, 0>,
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  // CHECK-DAG: arith.constant 2424832 : i32
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = true, timeout = true>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// imm_offset (state slot 11, i64): a K-loop advances the tile by bumping one
// scalar, added to the base in i64 after ptrtoint (carry into glb_hi is automatic).

// CHECK-LABEL: @test_tdm_load_imm_offset
func.func @test_tdm_load_imm_offset(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>,
    %off: i64,
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  // CHECK: %[[A1:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[11] : !llvm.struct<(i32, ptr<1>, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64)>
  %a1 = fly.atom.set_value(%atom, "imm_offset", %off) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, i64) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>
  // CHECK-DAG: %[[BASE:.*]] = llvm.extractvalue %[[A1]][1]
  // CHECK-DAG: %[[IMM:.*]] = llvm.extractvalue %[[A1]][11]
  // CHECK-DAG: %[[BI:.*]] = llvm.ptrtoint %[[BASE]] : !llvm.ptr<1> to i64
  // CHECK: arith.addi %[[BI]], %[[IMM]] : i64
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%a1, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}
