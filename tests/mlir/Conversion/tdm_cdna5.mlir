// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-canonicalize --fly-rocdl-expand-ops --fly-layout-lowering --canonicalize --fly-convert-atom-call-to-ssa-form --convert-fly-to-rocdl --canonicalize | FileCheck %s
//
// `--fly-convert-atom-call-to-ssa-form` is in the pipeline only to keep it
// honest: it walks every copy_atom_call to decide register promotion, so a
// coordinate-tensor operand must survive it. It has nothing to promote here.
//
// `--fly-rocdl-expand-ops` is here for the `boundary_check` state: a caller writes one flag per
// *tensor mode* and that pass rewrites it through `tensor2tdm` into the flat
// `boundary_check_axes` tuple the lowering reads. These tensors map one mode to one axis, so the
// translation is the identity and what is checked below is unchanged by it -- but the
// lowering now rejects an untranslated `boundary_check`, so the pass has to be in the pipeline.
//
// The trailing `--canonicalize` is load-bearing rather than cosmetic: per-dim
// clamping is atom *state*, so the lowering always emits the select between the
// clamped bound and the untouched extent, and it is the folder walking
// extractvalue back through insertvalue that turns a constant `boundary_check` leaf into no
// arithmetic at all. Checking that here is checking the claim the design rests on.

// CDNA5 TDM atom: a whole-tile DMA addressed by a coordinate tensor.
//
// The atom bakes the tensor (base pointer, per-dim stride, per-dim extent) as
// construction arguments of `fly.make_copy_atom`; the tile's position arrives as the
// `!fly.coord_tensor` operand, whose runtime value *is* that coordinate -- exactly as a
// memref operand's is its pointer. Which dims clamp is the `boundary_check` state -- one int_tuple
// leaf per descriptor dim, all clamping by default.
//   load  struct: {mask, early_timeout, atomic_barrier_addr (shared ptr), base (ptr),
//                  stride_0..3 (i64), extent_0..4 (i32), boundary_check_0..4 (i1),
//                  iter_stride (i64)}
//   store struct: the same minus the two MCAST fields.

// -----

// CHECK-LABEL: @test_cdna5_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, i32, ptr<3>, ptr<1>, i64, i64, i64, i64, i32, i32, i32, i32, i32, i1, i1, i1, i1, i1, i64)>)
func.func @test_cdna5_type(
    %atom: !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 0>) {
  return
}

// -----

// The store type has no MCAST slots, so its state struct is two i32 fields shorter.

// CHECK-LABEL: @test_cdna5_store_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(ptr<3>, ptr<1>, i64, i64, i64, i64, i32, i32, i32, i32, i32, i1, i1, i1, i1, i1, i64)>)
func.func @test_cdna5_store_type(
    %atom: !fly.copy_atom<!fly_rocdl.cdna5.tensor_store<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 0>) {
  return
}

// -----

// A static tile coordinate. `local_tile` has already folded it into the coord
// tensor's type — origin (384, 128) — so the copy site carries no arithmetic at all;
// layout lowering turns that origin into coord_0 / coord_1 and the descriptor math
// falls out as constants folded against the runtime strides.
//
// Both dims clamp (the default `boundary_check`), so the same coordinate that advances the
// address also shrinks the window: tensor_dim_i = max(extent_i - coord_i, 0).

// CHECK-LABEL: @test_cdna5_load_static_coord
func.func @test_cdna5_load_static_coord(
    %base: !fly.ptr<f16, global>, %s0: i64, %e0: i32, %e1: i32,
    %lds: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom(%base, %s0, %e0, %e1 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %org = fly.make_coord() : () -> !fly.int_tuple<(384,128)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(128,64)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(128,64)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(128,64):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(384,128)>, !fly.layout<(128,64):(1E0,1E1)>) -> !fly.coord_tensor<(384,128), (128,64):(1E0,1E1)>
  // A fully static origin lives in the operand's *type*, so it materializes as two
  // constants and nothing was computed to get them.
  // CHECK-DAG: %[[C0:.*]] = arith.constant 384 : i32
  // CHECK-DAG: %[[C1:.*]] = arith.constant 128 : i32
  // The address half: sum_i coord_i * stride_i, scaled to bytes, added to the baked
  // base pointer (%arg0).
  // CHECK-DAG: %[[BI:.*]] = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
  // CHECK-DAG: arith.addi %[[BI]]
  // The bounds half: the same two coordinates shrink the in-bounds window. Both
  // `boundary_check` leaves are the constructed default, so the outer state selects fold away.
  // `maxsi` against zero, not the equivalent compare-and-select: AMDGPU has the max on
  // the SALU but only a VALU saturating subtract, and a descriptor field is uniform.
  // CHECK-DAG: %[[R0:.*]] = arith.subi %arg2, %[[C0]] : i32
  // CHECK-DAG: arith.maxsi %[[R0]]
  // CHECK-DAG: %[[R1:.*]] = arith.subi %arg3, %[[C1]] : i32
  // CHECK-DAG: arith.maxsi %[[R1]]
  // CHECK: rocdl.tensor.load.to.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%atom, %gt, %lds) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.coord_tensor<(384,128), (128,64):(1E0,1E1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// No dim clamps: the tile is guaranteed in range by the tiling, so every `boundary_check`
// leaf is zero and `tensor_dim` is the extent itself, passed through untouched. Not a
// saturated sentinel: the caller has asserted `coord_i + tile_i <= extent_i`, so the
// un-shifted extent is already a bound the tile cannot reach, and it is an SGPR that is
// live anyway. So the dim costs no subtract, no clamp, and no constant to materialize --
// checking for the *absence* of that arithmetic is the only way to keep it honest.
// Descriptor dim 0 is the tensor's innermost mode, hence %arg3 before %arg2.

// CHECK-LABEL: @test_cdna5_load_no_boundary_check
// CHECK-NOT: arith.maxsi
// CHECK-NOT: arith.subi
// CHECK-NOT: arith.select
// CHECK-DAG: arith.andi %arg3, %{{.*}} : i32
// CHECK-DAG: arith.andi %arg2, %{{.*}} : i32
// CHECK: rocdl.tensor.load.to.lds
func.func @test_cdna5_load_no_boundary_check(
    %base: !fly.ptr<f16, global>, %s0: i64, %e0: i32, %e1: i32,
    %lds: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %off = fly.make_int_tuple() : () -> !fly.int_tuple<(0,0)>
  %a0 = fly.make_copy_atom(%base, %s0, %e0, %e1 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %atom = fly.atom.set_value(%a0, "boundary_check", %off) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.int_tuple<(0,0)>) -> !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %org = fly.make_coord() : () -> !fly.int_tuple<(384,128)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(128,64)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(128,64)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(128,64):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(384,128)>, !fly.layout<(128,64):(1E0,1E1)>) -> !fly.coord_tensor<(384,128), (128,64):(1E0,1E1)>
  fly.copy_atom_call(%atom, %gt, %lds) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.coord_tensor<(384,128), (128,64):(1E0,1E1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// Mixed: dim 0 clamps, dim 1 does not. The clamp is priced per dim -- a subtract-and-clamp
// on the extent at *every* call -- so it is bought per dim. Dim 1 passes its extent
// through unmodified, which is why exactly one subtract-and-clamp reaches the
// instruction while both dims still read an extent. Because this is state and not a type
// parameter, the *same* atom type serves both this function and the two above. The
// tuple names every dim, including the one it leaves clamping: that is the price of
// checking the rank rather than accepting a leaf at a time.

// CHECK-LABEL: @test_cdna5_load_mixed_boundary_check
// CHECK-COUNT-1: arith.maxsi
// CHECK-NOT: arith.maxsi
// CHECK-NOT: arith.select
// CHECK: rocdl.tensor.load.to.lds
func.func @test_cdna5_load_mixed_boundary_check(
    %base: !fly.ptr<f16, global>, %s0: i64, %e0: i32, %e1: i32,
    %lds: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %off = fly.make_int_tuple() : () -> !fly.int_tuple<(1,0)>
  %a0 = fly.make_copy_atom(%base, %s0, %e0, %e1 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %atom = fly.atom.set_value(%a0, "boundary_check", %off) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.int_tuple<(1,0)>) -> !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %org = fly.make_coord() : () -> !fly.int_tuple<(384,128)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(128,64)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(128,64)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(128,64):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(384,128)>, !fly.layout<(128,64):(1E0,1E1)>) -> !fly.coord_tensor<(384,128), (128,64):(1E0,1E1)>
  fly.copy_atom_call(%atom, %gt, %lds) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.coord_tensor<(384,128), (128,64):(1E0,1E1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// A genuinely dynamic `boundary_check` leaf is the case that does not fold: the select survives,
// and so does the arithmetic on both of its sides. This is the honest upper bound on
// what the knob costs -- one v_cndmask -- and it is only paid by a caller that actually
// varies the flag at runtime. An int_tuple's dynamic leaves are i32, so reaching the i1
// the slot holds also costs the compare; a static leaf pays neither.

// CHECK-LABEL: @test_cdna5_load_dynamic_boundary_check
// CHECK-DAG: arith.cmpi ne
// CHECK-DAG: arith.select
// CHECK: rocdl.tensor.load.to.lds
func.func @test_cdna5_load_dynamic_boundary_check(
    %base: !fly.ptr<f16, global>, %s0: i64, %e0: i32, %e1: i32, %flag: i32,
    %lds: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %off = fly.make_int_tuple(%flag) : (i32) -> !fly.int_tuple<(1,?)>
  %a0 = fly.make_copy_atom(%base, %s0, %e0, %e1 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %atom = fly.atom.set_value(%a0, "boundary_check", %off) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.int_tuple<(1,?)>) -> !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %org = fly.make_coord() : () -> !fly.int_tuple<(384,128)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(128,64)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(128,64)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(128,64):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(384,128)>, !fly.layout<(128,64):(1E0,1E1)>) -> !fly.coord_tensor<(384,128), (128,64):(1E0,1E1)>
  fly.copy_atom_call(%atom, %gt, %lds) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.coord_tensor<(384,128), (128,64):(1E0,1E1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// A dynamic tile coordinate (a block index) is the one leaf that survives into the
// IR; everything else about the position stayed in the type.

// CHECK-LABEL: @test_cdna5_load_dynamic_coord
func.func @test_cdna5_load_dynamic_coord(
    %base: !fly.ptr<f16, global>, %s0: i64, %e0: i32, %e1: i32, %m: i32,
    %lds: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom(%base, %s0, %e0, %e1 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %org = fly.make_coord(%m) : (i32) -> !fly.int_tuple<(?,0)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(128,64)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(128,64)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(128,64):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(?,0)>, !fly.layout<(128,64):(1E0,1E1)>) -> !fly.coord_tensor<(?,0), (128,64):(1E0,1E1)>
  // The dynamic leaf is the block index itself, taken straight off the operand; the
  // static one is still a constant from the type.
  // CHECK: arith.subi %{{.*}}, %arg4 : i32
  // CHECK: rocdl.tensor.load.to.lds
  fly.copy_atom_call(%atom, %gt, %lds) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.coord_tensor<(?,0), (128,64):(1E0,1E1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// The store direction: LDS -> global, with the coordinate tensor on the dst side.

// CHECK-LABEL: @test_cdna5_store
func.func @test_cdna5_store(
    %base: !fly.ptr<f16, global>, %s0: i64, %e0: i32, %e1: i32,
    %lds: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom(%base, %s0, %e0, %e1 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_store<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>
  %org = fly.make_coord() : () -> !fly.int_tuple<(0,0)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(128,64)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(128,64)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(128,64):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(0,0)>, !fly.layout<(128,64):(1E0,1E1)>) -> !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>
  // CHECK: rocdl.tensor.store.from.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%atom, %lds, %gt) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_store<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>) -> ()
  return
}

// -----

// A recast: `elem` is the *descriptor's* unit, and it
// may be wider than the tensor's own element. Here an FP4 tile is moved as bytes --
// `data_size` is 1/2/4/8 bytes and 4 bits is none of them, so this is the only way to
// describe such a tensor at all. The LDS operand keeps its FP4 element type; what has to
// agree is the bit count, 128*32*8 == 128*64*4.
// CHECK-LABEL: @test_cdna5_load_recast_subbyte
func.func @test_cdna5_load_recast_subbyte(
    %base: !fly.ptr<f4E2M1FN, global>, %s0: i64, %e0: i32, %e1: i32,
    %lds: !fly.memref<f4E2M1FN, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom(%base, %s0, %e0, %e1 : !fly.ptr<f4E2M1FN, global>, i64, i32, i32) {valBits = 8 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 32], elem = i8, tensor2tdm = (1E0,1E1)>, 8>
  %org = fly.make_coord() : () -> !fly.int_tuple<(384,64)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(128,32)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(128,32)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(128,32):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(384,64)>, !fly.layout<(128,32):(1E0,1E1)>) -> !fly.coord_tensor<(384,64), (128,32):(1E0,1E1)>
  // data_size 0 == 1 byte, and the address arithmetic scales the coordinate by that
  // byte rather than by the tensor's 4-bit element.
  // CHECK: rocdl.tensor.load.to.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%atom, %gt, %lds) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 32], elem = i8, tensor2tdm = (1E0,1E1)>, 8>, !fly.coord_tensor<(384,64), (128,32):(1E0,1E1)>, !fly.memref<f4E2M1FN, shared, (128,64):(64,1)>) -> ()
  return
}

// -----

// Descriptor iteration: one instruction replays the same D# `iterCount` times, stepping
// global by the trailing construction argument and LDS by one whole box. It is the TDM
// spelling of a residual axis the box cannot travel on, and it is paid for out of GROUP2
// (lds_addr_increment, global_addr_increment, iterate_count), which is why the descriptor
// is left with two dims. GROUP1 bit 19 turns it on.
// CHECK-LABEL: @test_cdna5_load_iterate
func.func @test_cdna5_load_iterate(
    %base: !fly.ptr<f16, global>, %s0: i64, %e0: i32, %e1: i32, %istride: i64,
    %lds: !fly.memref<f16, shared, (8,64):(64,1)>) {
  %atom = fly.make_copy_atom(%base, %s0, %e0, %e1, %istride : !fly.ptr<f16, global>, i64, i32, i32, i64) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [1, 64], elem = f16, tensor2tdm = (0,1E1), iterCount = 8>, 16>
  %org = fly.make_coord() : () -> !fly.int_tuple<(0,0)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(1,64)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(1,64)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(1,64):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(0,0)>, !fly.layout<(1,64):(1E0,1E1)>) -> !fly.coord_tensor<(0,0), (1,64):(1E0,1E1)>
  // GROUP1 word 0 carries data_size (1 << 16 for 2-byte elements) together with
  // iterate_enable (1 << 19): 0x80000 | 0x10000 == 589824.
  // CHECK-DAG: arith.constant 589824 : i32
  // GROUP2 word 1 is the LDS increment, one box of 64 elements.
  // CHECK-DAG: arith.constant 64 : i32
  // GROUP2 word 3's upper half is iterate_count encoded as value-minus-one: 7 << 16.
  // CHECK-DAG: arith.constant 458752 : i32
  // CHECK: rocdl.tensor.load.to.lds %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0 : vector<4xi32>, vector<8xi32>
  fly.copy_atom_call(%atom, %gt, %lds) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [1, 64], elem = f16, tensor2tdm = (0,1E1), iterCount = 8>, 16>, !fly.coord_tensor<(0,0), (1,64):(1E0,1E1)>, !fly.memref<f16, shared, (8,64):(64,1)>) -> ()
  return
}

// -----

// The HW auto-barrier: *whether* the copy arrives on one is a type parameter and *which*
// one is atom state, and the two are kept genuinely separate. The state is the barrier
// itself -- a shared pointer, as the kernel holds it -- and only the descriptor's byte
// address is taken from it, here rather than at the call site. Nothing reads it as an
// enable, so no pointer value is spent on meaning "none" and LDS offset 0 is a barrier
// like any other; the price is that an atom whose type asks for a barrier and whose
// state was never given one arrives on offset 0, which is the caller's to get right, as
// the base pointer is. On an atom whose type does not enable it neither the bit nor the
// address appears, and the `set_value` has no field to write at all, which
// `tdm_cdna5_neg.mlir` checks.

// CHECK-LABEL: @test_cdna5_atomic_barrier
// The enable is the type, so bit 18 is not conditional on anything and folds into the
// GROUP1 config constant -- 327680, that bit alongside the tile's own bit 16. A constant
// there is the whole claim: it could not fold if the pointer were also read as an enable.
// CHECK-DAG: %[[CFG:.*]] = arith.constant 327680 : i32
// The barrier arrives as `!llvm.ptr<3>` and is flattened once; an LDS pointer is 32-bit,
// so this is a bitcast and not a truncation.
// CHECK-DAG: %[[B:.*]] = llvm.ptrtoint %arg4 : !llvm.ptr<3> to i32
// CHECK-DAG: arith.shrui %[[B]], %{{.*}} : i32
// CHECK: vector.from_elements %[[CFG]],
// CHECK: rocdl.tensor.load.to.lds
func.func @test_cdna5_atomic_barrier(
    %base: !fly.ptr<f16, global>, %s0: i64, %e0: i32, %e1: i32,
    %bar: !fly.ptr<i64, shared>,
    %lds: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %a0 = fly.make_copy_atom(%base, %s0, %e0, %e1 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1), atomicBarrier = true>, 16>
  %atom = fly.atom.set_value(%a0, "atomic_barrier_addr", %bar) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1), atomicBarrier = true>, 16>, !fly.ptr<i64, shared>) -> !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1), atomicBarrier = true>, 16>
  %org = fly.make_coord() : () -> !fly.int_tuple<(0,0)>
  %shp = fly.make_int_tuple() : () -> !fly.int_tuple<(128,64)>
  %str = fly.make_int_tuple() : () -> !fly.int_tuple<(1E0,1E1)>
  %lay = fly.make_layout(%shp, %str) : (!fly.int_tuple<(128,64)>, !fly.int_tuple<(1E0,1E1)>) -> !fly.layout<(128,64):(1E0,1E1)>
  %gt = fly.make_view(%org, %lay) : (!fly.int_tuple<(0,0)>, !fly.layout<(128,64):(1E0,1E1)>) -> !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>
  fly.copy_atom_call(%atom, %gt, %lds) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1), atomicBarrier = true>, 16>, !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}
