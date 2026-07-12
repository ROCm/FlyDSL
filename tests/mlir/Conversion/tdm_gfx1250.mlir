// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// gfx1250 2D TDM copy atom lowering. The Global-side memref layout supplies the
// tile geometry; direction (load vs store) is inferred from the address spaces.
//   Global -> Shared  =>  rocdl.tensor.load.to.lds
//   Shared -> Global  =>  rocdl.tensor.store.from.lds

// -----

// Stateful atom type converts to a 2-i32 struct (workgroup_mask, oob_outer).
// CHECK-LABEL: @test_tdm_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, i32)>)
func.func @test_tdm_type(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>) {
  return
}

// -----

// Load, single warp: descriptor addresses derive from the memref base pointers.
// tile = 128x64 f16, stride (64, 1). tdim0/tile dims/stride are compile-time;
// tensor_dim1 is runtime = max(0, oob_outer - startOuter) (default oob_outer =
// INT32_MAX => whole tile in-bounds).
//
// Compile-time GROUP1 parts: s0 = 1<<16 = 65536, s1 = (64&0xffff)<<16 = 4194304,
// tileD0<<16 = 64<<16 = 4194304, s4 = tile_dim1 = 128, s5 = outer stride = 64.

// CHECK-LABEL: @test_tdm_load
// CHECK-SAME: (%[[SRC:.*]]: !llvm.ptr<1>, %[[DST:.*]]: !llvm.ptr<3>)
func.func @test_tdm_load(
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom {valBits = 0 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>
  // Default state: workgroup_mask=0, oob_outer=INT32_MAX (no clamp).
  // CHECK: arith.constant 2147483647 : i32
  // Global base and LDS base come from the pointers.
  // CHECK-DAG: %[[GBASE:.*]] = llvm.ptrtoint %[[SRC]] : !llvm.ptr<1> to i64
  // CHECK-DAG: %[[LBASE:.*]] = llvm.ptrtoint %[[DST]] : !llvm.ptr<3> to i32
  // GROUP0 packs (pred=1, lds_addr, glb_lo, glb_hi | type):
  // CHECK: %[[GLO:.*]] = llvm.trunc %[[GBASE]] : i64 to i32
  // CHECK: %[[GHI0:.*]] = llvm.lshr %[[GBASE]]
  // CHECK: %[[GHI:.*]] = llvm.trunc %[[GHI0]] : i64 to i32
  // CHECK: %[[GHITY:.*]] = arith.ori %[[GHI]], %{{.*}}
  // CHECK: %[[DG0:.*]] = vector.from_elements %{{.*}}, %[[LBASE]], %[[GLO]], %[[GHITY]] : vector<4xi32>
  // GROUP1: tensor_dim1 runtime (OOB clamp) from the oob_outer state field.
  // CHECK: %[[OOB:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i32, i32)>
  // CHECK: arith.subi %[[OOB]], %{{.*}}
  // CHECK: arith.maxsi
  // CHECK: %[[DG1:.*]] = vector.from_elements
  // CHECK: rocdl.tensor.load.to.lds %[[DG0]], %[[DG1]], %{{.*}}, %{{.*}}, %{{.*}} cachepolicy 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// Store direction (Shared -> Global) -> tensor.store.from.lds.

// CHECK-LABEL: @test_tdm_store
func.func @test_tdm_store(
    %src: !fly.memref<f16, shared, (128,64):(64,1)>,
    %dst: !fly.memref<f16, global, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom {valBits = 0 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>
  // CHECK: rocdl.tensor.store.from.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} cachepolicy 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.memref<f16, global, (128,64):(64,1)>) -> ()
  return
}

// -----

// Multi-warp load distributes the tile: wave.id is read and the per-wave
// offsets scale the descriptor addresses. warps=4 over a 128x64 tile splits the
// outer dim 4-ways (bpw_outer=32, bpw_inner=64).

// CHECK-LABEL: @test_tdm_load_warps
func.func @test_tdm_load_warps(
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom {valBits = 0 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 4, pad = 0, 0, cache = 0>, 0>
  // CHECK: %[[WID:.*]] = rocdl.wave.id : i32
  // CHECK-DAG: %[[W4:.*]] = arith.constant 4 : i32
  // CHECK-DAG: arith.remui %[[WID]], %[[W4]]
  // CHECK-DAG: arith.divui %[[WID]], %[[W4]]
  // GROUP1 per-warp: tdim0=inner=64, tile_dim1=outer per-warp=32; tensor_dim1
  // runtime = max(0, oob_outer - warp_off_outer).
  // CHECK: arith.subi
  // CHECK: arith.maxsi
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 4, pad = 0, 0, cache = 0>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// Load with LDS padding (interval=64 elems, amount=8 elems, f16 -> 16-bit):
//   interval_dw = 64*16/32 = 32 -> enc_interval = log2(32)-1 = 4
//   amount_dw   = 8*16/32  = 4  -> enc_amount   = 4-1 = 3
//   g1_s0 = (1<<16) | (1<<20) | (4<<22) | (3<<25) = 118554624

// CHECK-LABEL: @test_tdm_load_pad
func.func @test_tdm_load_pad(
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom {valBits = 0 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 64, 8, cache = 0>, 0>
  // CHECK-DAG: arith.constant 118554624 : i32
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 64, 8, cache = 0>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// MCAST: a runtime workgroup_mask (atom state) is ORed into GROUP1 config [15:0].

// CHECK-LABEL: @test_tdm_load_mcast
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>, %[[MASK:.*]]: i32,
func.func @test_tdm_load_mcast(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>,
    %mask: i32,
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  // CHECK: %[[A1:.*]] = llvm.insertvalue %[[MASK]], %[[ATOM]][0]
  %a1 = fly.atom.set_value(%atom, "workgroup_mask", %mask) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>
  // CHECK: %[[M:.*]] = llvm.extractvalue %[[A1]][0]
  // CHECK-DAG: %[[MLOW:.*]] = arith.andi %[[M]], %{{.*}}
  // CHECK-DAG: %[[UPPER:.*]] = arith.constant 65536 : i32
  // CHECK: %[[S0:.*]] = arith.ori %[[UPPER]], %[[MLOW]]
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%a1, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// OOB: a runtime oob_outer (atom state) sets tensor_dim1 = max(0, oob_outer -
// startOuter) while tile_dim1 stays static, so ragged / partial tiles are
// HW-guarded (load: zero-fill; store: drop).

// CHECK-LABEL: @test_tdm_load_oob
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>, %[[ROWS:.*]]: i32,
func.func @test_tdm_load_oob(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>,
    %rows: i32,
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  // CHECK: %[[A1:.*]] = llvm.insertvalue %[[ROWS]], %[[ATOM]][1]
  %a1 = fly.atom.set_value(%atom, "oob_outer", %rows) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>
  // CHECK: %[[O:.*]] = llvm.extractvalue %[[A1]][1]
  // CHECK: arith.subi %[[O]], %{{.*}}
  // CHECK: arith.maxsi
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%a1, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 0, 0, cache = 0>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}
