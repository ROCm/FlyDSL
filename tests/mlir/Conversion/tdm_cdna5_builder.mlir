// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-rocdl-expand-ops --canonicalize | FileCheck %s --check-prefix=EXPAND
// RUN: %fly-opt %s --split-input-file --fly-canonicalize --fly-rocdl-expand-ops --fly-layout-lowering --canonicalize --fly-convert-atom-call-to-ssa-form --convert-fly-to-rocdl --canonicalize | FileCheck %s --check-prefix=LOWER

// The tiled TDM builder used the way a kernel uses it: build the atom and its coordinate
// tensor, cut the CTA tile out of the coordinate tensor, partition both sides with
// `tdm_partition`, and `fly.copy`. The IR below is what the Python `make_tiled_tdm_atom`,
// `zipped_divide`, `tdm_partition` and `fx.copy` emit.
//
// EXPAND checks the descriptor the builder derives: the atom type, the construction
// arguments (base, then the strides of tensor dims 0..rank-2, then every extent, in
// tensor dim order, i.e. descriptor axes reversed), the initial `boundary_check_axes` and the
// coordinate tensor. LOWER checks every issue the copy expands to. The expected values are
// derived by hand from the global layout, not read back from the implementation.
//
// GROUP1 words, as `emitTdmAtomCall` packs them:
//   w0 = data_size << 16 | pad_enable << 20 | pad_interval << 22 | pad_amount << 25
//        (data_size 0/1/2/3 = 1/2/4/8 bytes; interval = log2(dwords) - 1; amount = dwords - 1)
//   w1 = tensor_dim0 << 16,  w2 = tensor_dim1 << 16,  w3 = tile_dim0 << 16,
//   w4 = tile_dim1 | tile_dim2 << 16
// A tensor_dim is the extent left after the tile origin on an axis that clamps.

// -----

// A rank-6 tensor whose strides leave gaps, so no two modes coalesce. Descriptor axes
// 0..3 take global modes 5..2 and the two outermost modes share axis 4: its box extent and
// descriptor extent are 1 and its stride is gcd(65536, 16384) = 16384, so mode 0 moves 4
// steps along it and mode 1 one step. The 2048-element tile is four 512-element issues in
// ITER, whose axis-4 coordinate is 4*c0 + c1. The CTA tile is the second one along mode 5,
// so every issue starts at column 64.

// EXPAND-LABEL: @rank6
// EXPAND: fly.make_copy_atom(%{{.*}}, %c16384_i64, %c4096_i64, %c1024_i64, %c256_i64, %c1_i32, %c2_i32, %c2_i32, %c2_i32, %c128_i32 : {{.*}}) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [1, 2, 2, 2, 64], elem = f16, tensor2tdm = (0,0,1E1,1E2,1E3,1E4)>, 16>
// The packed axis (state slot 0) has no bound to clamp.
// EXPAND: fly.atom.set_value(%{{.*}}, "boundary_check_axes", %{{.*}}) : ({{.*}}, !fly.int_tuple<(0,1,1,1,1)>)
// EXPAND: !fly.coord_tensor<(0,0,0,0,0), (2,2,2,2,2,128):(4E4,1E4,1E3,1E2,1E1,1E0)>
// ATOM walks modes 5..2, ITER modes 1 and 0.
// EXPAND: fly.copy({{.*}}, !fly.coord_tensor<(64,0,0,0,0), ((64,2,2,2),(2,2)):((1E0,1E1,1E2,1E3),(1E4,4E4))>, !fly.memref<f16, shared, (512,4):(1,512)>) -> ()

// LOWER-LABEL: @rank6
// Issue (c0, c1) = (0, 0): column 64, 128 bytes in.
// LOWER: arith.addi %{{.*}}, %c128_i64 : i64
// LOWER: %[[S1:.*]] = llvm.trunc %c256_i64 : i64 to i32
// f16 (data_size 1); tensor_dim0 = 128 - 64 = 64, tensor_dim1 = 2; tile = 64 x 2 x 2.
// LOWER: vector.from_elements %c65536_i32, %c4194304_i32, %c131072_i32, %c4194304_i32, %c131074_i32, %[[S1]], %{{.*}}, %{{.*}} : vector<8xi32>
// tensor_dim2 = tensor_dim3 = 2, axis 3's stride 4096, tile_dim3 = 2.
// LOWER: %[[S3:.*]] = llvm.trunc %c4096_i64 : i64 to i32
// LOWER: vector.from_elements %c2_i32, %c2_i32, %[[S3]], %{{.*}} : vector<4xi32>
// The packed axis: stride 16384, tensor_dim4 = tile_dim4 = 1.
// LOWER: %[[S4:.*]] = llvm.trunc %c16384_i64 : i64 to i32
// LOWER: %[[TD4:.*]] = arith.ori %{{.*}}, %c65536_i32 : i32
// LOWER: vector.from_elements %[[S4]], %[[TD4]], %c65536_i32, %c0_i32 : vector<4xi32>
// LOWER: rocdl.tensor.load.to.lds
// (0, 1): 64 + 16384 elements; LDS one issue (512 elements) further.
// LOWER: llvm.getelementptr %arg1[512]
// LOWER: arith.addi %{{.*}}, %c32896_i64 : i64
// LOWER: rocdl.tensor.load.to.lds
// (1, 0): 64 + 65536 elements.
// LOWER: llvm.getelementptr %arg1[1024]
// LOWER: arith.addi %{{.*}}, %c131200_i64 : i64
// LOWER: rocdl.tensor.load.to.lds
// (1, 1): 64 + 65536 + 16384 elements.
// LOWER: llvm.getelementptr %arg1[1536]
// LOWER: arith.addi %{{.*}}, %c163968_i64 : i64
// LOWER: rocdl.tensor.load.to.lds
// LOWER-NOT: rocdl.tensor.load.to.lds
// LOWER: return
func.func @rank6(%arg0: !fly.memref<f16, global, (2,2,2,2,2,128):(65536,16384,4096,1024,256,1)>, %arg1: !fly.ptr<f16, shared>) {
  %0 = fly.make_int_tuple() : () -> !fly.int_tuple<(2,2,2,2,2,64)>
  %1 = fly.make_int_tuple() : () -> !fly.int_tuple<(1024,512,256,128,64,1)>
  %2 = fly.make_layout(%0, %1) : (!fly.int_tuple<(2,2,2,2,2,64)>, !fly.int_tuple<(1024,512,256,128,64,1)>) -> !fly.layout<(2,2,2,2,2,64):(1024,512,256,128,64,1)>
  %3 = fly.make_int_tuple() : () -> !fly.int_tuple<((2,2,2,2,2,64),1)>
  %4 = fly.make_int_tuple() : () -> !fly.int_tuple<((1024,512,256,128,64,1),2048)>
  %5 = fly.make_layout(%3, %4) : (!fly.int_tuple<((2,2,2,2,2,64),1)>, !fly.int_tuple<((1024,512,256,128,64,1),2048)>) -> !fly.layout<((2,2,2,2,2,64),1):((1024,512,256,128,64,1),2048)>
  %6 = fly.make_view(%arg1, %5) : (!fly.ptr<f16, shared>, !fly.layout<((2,2,2,2,2,64),1):((1024,512,256,128,64,1),2048)>) -> !fly.memref<f16, shared, ((2,2,2,2,2,64),1):((1024,512,256,128,64,1),2048)>
  %7 = fly.static : !fly.tile<[2|2|2|2|2|64]>
  %atom, %coordTensor = fly_rocdl.make_tiled_tdm_load_atom %arg0, %2, %7 : !fly.memref<f16, global, (2,2,2,2,2,128):(65536,16384,4096,1024,256,1)>, !fly.layout<(2,2,2,2,2,64):(1024,512,256,128,64,1)>, !fly.tile<[2|2|2|2|2|64]>
  %8 = fly.static : !fly.tile<[2|2|2|2|2|64]>
  %9 = fly.zipped_divide(%coordTensor, %8) : (!fly.coord_tensor<(0,0,0,0,0), (2,2,2,2,2,128):(4E4,1E4,1E3,1E2,1E1,1E0)>, !fly.tile<[2|2|2|2|2|64]>) -> !fly.coord_tensor<(0,0,0,0,0), ((2,2,2,2,2,64),(1,1,1,1,1,2)):((4E4,1E4,1E3,1E2,1E1,1E0),(0,0,0,0,0,64E0))>
  %10 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,(0,0,0,0,0,*))>
  %11 = fly.slice(%9, %10) : (!fly.coord_tensor<(0,0,0,0,0), ((2,2,2,2,2,64),(1,1,1,1,1,2)):((4E4,1E4,1E3,1E2,1E1,1E0),(0,0,0,0,0,64E0))>, !fly.int_tuple<(*,(0,0,0,0,0,*))>) -> !fly.coord_tensor<(0,0,0,0,0), ((2,2,2,2,2,64),2):((4E4,1E4,1E3,1E2,1E1,1E0),64E0)>
  %12 = fly.static : !fly.tile<[(((64,2,2,2),1),(2,2)):(((32,16,8,4),0),(2,1))|*]>
  %13 = fly.composition(%6, %12) : (!fly.memref<f16, shared, ((2,2,2,2,2,64),1):((1024,512,256,128,64,1),2048)>, !fly.tile<[(((64,2,2,2),1),(2,2)):(((32,16,8,4),0),(2,1))|*]>) -> !fly.memref<f16, shared, ((((64,2,2,2),1),(2,2)),1):((((1,64,128,256),0),(512,1024)),2048)>
  %14 = fly.make_int_tuple() : () -> !fly.int_tuple<(((*,0),*),*)>
  %15 = fly.slice(%13, %14) : (!fly.memref<f16, shared, ((((64,2,2,2),1),(2,2)),1):((((1,64,128,256),0),(512,1024)),2048)>, !fly.int_tuple<(((*,0),*),*)>) -> !fly.memref<f16, shared, ((64,2,2,2),(2,2),1):((1,64,128,256),(512,1024),2048)>
  %16 = fly.group(%15) {begin = 0 : i32, end = 2 : i32} : (!fly.memref<f16, shared, ((64,2,2,2),(2,2),1):((1,64,128,256),(512,1024),2048)>) -> !fly.memref<f16, shared, (((64,2,2,2),(2,2)),1):(((1,64,128,256),(512,1024)),2048)>
  %17 = fly.make_int_tuple() : () -> !fly.int_tuple<((1,1))>
  %18 = fly.coalesce(%16, %17) : (!fly.memref<f16, shared, (((64,2,2,2),(2,2)),1):(((1,64,128,256),(512,1024)),2048)>, !fly.int_tuple<((1,1))>) -> !fly.memref<f16, shared, ((512,4),1):((1,512),2048)>
  %19 = fly.static : !fly.tile<[(((64,2,2,2),1),(2,2)):(((32,16,8,4),0),(2,1))|*]>
  %20 = fly.composition(%11, %19) : (!fly.coord_tensor<(0,0,0,0,0), ((2,2,2,2,2,64),2):((4E4,1E4,1E3,1E2,1E1,1E0),64E0)>, !fly.tile<[(((64,2,2,2),1),(2,2)):(((32,16,8,4),0),(2,1))|*]>) -> !fly.coord_tensor<(0,0,0,0,0), ((((64,2,2,2),1),(2,2)),2):((((1E0,1E1,1E2,1E3),0),(1E4,4E4)),64E0)>
  %21 = fly.make_int_tuple() : () -> !fly.int_tuple<(((*,0),*),*)>
  %22 = fly.slice(%20, %21) : (!fly.coord_tensor<(0,0,0,0,0), ((((64,2,2,2),1),(2,2)),2):((((1E0,1E1,1E2,1E3),0),(1E4,4E4)),64E0)>, !fly.int_tuple<(((*,0),*),*)>) -> !fly.coord_tensor<(0,0,0,0,0), ((64,2,2,2),(2,2),2):((1E0,1E1,1E2,1E3),(1E4,4E4),64E0)>
  %23 = fly.group(%22) {begin = 0 : i32, end = 2 : i32} : (!fly.coord_tensor<(0,0,0,0,0), ((64,2,2,2),(2,2),2):((1E0,1E1,1E2,1E3),(1E4,4E4),64E0)>) -> !fly.coord_tensor<(0,0,0,0,0), (((64,2,2,2),(2,2)),2):(((1E0,1E1,1E2,1E3),(1E4,4E4)),64E0)>
  %24 = fly.make_int_tuple() : () -> !fly.int_tuple<((1,1))>
  %25 = fly.coalesce(%23, %24) : (!fly.coord_tensor<(0,0,0,0,0), (((64,2,2,2),(2,2)),2):(((1E0,1E1,1E2,1E3),(1E4,4E4)),64E0)>, !fly.int_tuple<((1,1))>) -> !fly.coord_tensor<(0,0,0,0,0), (((64,2,2,2),(2,2)),2):(((1E0,1E1,1E2,1E3),(1E4,4E4)),64E0)>
  %26 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,1)>
  %27 = fly.slice(%25, %26) : (!fly.coord_tensor<(0,0,0,0,0), (((64,2,2,2),(2,2)),2):(((1E0,1E1,1E2,1E3),(1E4,4E4)),64E0)>, !fly.int_tuple<(*,1)>) -> !fly.coord_tensor<(64,0,0,0,0), ((64,2,2,2),(2,2)):((1E0,1E1,1E2,1E3),(1E4,4E4))>
  %28 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,0)>
  %29 = fly.slice(%18, %28) : (!fly.memref<f16, shared, ((512,4),1):((1,512),2048)>, !fly.int_tuple<(*,0)>) -> !fly.memref<f16, shared, (512,4):(1,512)>
  fly.copy(%atom, %27, %29) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [1, 2, 2, 2, 64], elem = f16, tensor2tdm = (0,0,1E1,1E2,1E3,1E4)>, 16>, !fly.coord_tensor<(64,0,0,0,0), ((64,2,2,2),(2,2)):((1E0,1E1,1E2,1E3),(1E4,4E4))>, !fly.memref<f16, shared, (512,4):(1,512)>) -> ()
  return
}

// -----

// The same copy with the two packed strides dynamic. The packed axis's stride is their
// gcd, computed at run time, and each packed mode's coordinate scale is its stride over
// that gcd, so an issue's axis-4 coordinate times the axis stride is again c0*s0 + c1*s1.

// EXPAND-LABEL: @rank6_dynamic
// EXPAND: %[[S:.*]]:2 = fly.get_leaves(%arg0)
// EXPAND: %[[GCD:.*]]:2 = scf.while
// EXPAND: arith.remui
// EXPAND-DAG: arith.divsi %[[S]]#0, %{{.*}} : i32
// EXPAND-DAG: arith.divsi %[[S]]#1, %{{.*}} : i32
// EXPAND: %[[G64:.*]] = arith.extui %[[GCD]]#0 : i32 to i64
// EXPAND: fly.make_copy_atom(%{{.*}}, %[[G64]], %c4096_i64, %c1024_i64, %c256_i64, %c1_i32, %c2_i32, %c2_i32, %c2_i32, %c128_i32 : {{.*}}) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [1, 2, 2, 2, 64], elem = f16, tensor2tdm = (0,0,1E1,1E2,1E3,1E4)>, 16>
// EXPAND: !fly.coord_tensor<(0,0,0,0,0), (2,2,2,2,2,128):(?E4,?E4,1E3,1E2,1E1,1E0)>

// LOWER-LABEL: @rank6_dynamic
// LOWER-DAG: %[[S0:.*]] = llvm.extractvalue %arg1[0, 0]
// LOWER-DAG: %[[S1:.*]] = llvm.extractvalue %arg1[0, 1]
// LOWER: %[[GCD:.*]]:2 = scf.while
// LOWER: %[[G:.*]] = arith.maxsi %[[GCD]]#0, %c1_i32 : i32
// LOWER-DAG: %[[Q0:.*]] = arith.divsi %[[S0]], %[[G]] : i32
// LOWER-DAG: %[[Q1:.*]] = arith.divsi %[[S1]], %[[G]] : i32
// LOWER: %[[G64:.*]] = arith.extui %[[GCD]]#0 : i32 to i64
// (0, 0): column 64 only.
// LOWER: arith.addi %{{.*}}, %c128_i64 : i64
// The packed axis's descriptor stride is the gcd.
// LOWER: %[[S4:.*]] = llvm.trunc %[[G64]] : i64 to i32
// LOWER: vector.from_elements %[[S4]], %{{.*}}, %c65536_i32, %c0_i32 : vector<4xi32>
// LOWER: rocdl.tensor.load.to.lds
// (0, 1): axis-4 coordinate s1 / gcd.
// LOWER: %[[C1:.*]] = arith.extui %[[Q1]] : i32 to i64
// LOWER: %[[O1:.*]] = arith.muli %[[C1]], %[[G64]] : i64
// LOWER: %[[E1:.*]] = arith.addi %[[O1]], %c64_i64 : i64
// LOWER: arith.muli %[[E1]], %c2_i64 : i64
// LOWER: rocdl.tensor.load.to.lds
// (1, 0): s0 / gcd.
// LOWER: %[[C2:.*]] = arith.extui %[[Q0]] : i32 to i64
// LOWER: arith.muli %[[C2]], %[[G64]] : i64
// LOWER: rocdl.tensor.load.to.lds
// (1, 1): s1 / gcd + s0 / gcd.
// LOWER: %[[Q01:.*]] = arith.addi %[[Q1]], %[[Q0]]
// LOWER: %[[C3:.*]] = arith.extui %[[Q01]] : i32 to i64
// LOWER: arith.muli %[[C3]], %[[G64]] : i64
// LOWER: rocdl.tensor.load.to.lds
// LOWER-NOT: rocdl.tensor.load.to.lds
// LOWER: return
func.func @rank6_dynamic(%arg0: !fly.memref<f16, global, (2,2,2,2,2,128):(?,?,4096,1024,256,1)>, %arg1: !fly.ptr<f16, shared>) {
  %0 = fly.make_int_tuple() : () -> !fly.int_tuple<(2,2,2,2,2,64)>
  %1 = fly.make_int_tuple() : () -> !fly.int_tuple<(1024,512,256,128,64,1)>
  %2 = fly.make_layout(%0, %1) : (!fly.int_tuple<(2,2,2,2,2,64)>, !fly.int_tuple<(1024,512,256,128,64,1)>) -> !fly.layout<(2,2,2,2,2,64):(1024,512,256,128,64,1)>
  %3 = fly.make_int_tuple() : () -> !fly.int_tuple<((2,2,2,2,2,64),1)>
  %4 = fly.make_int_tuple() : () -> !fly.int_tuple<((1024,512,256,128,64,1),2048)>
  %5 = fly.make_layout(%3, %4) : (!fly.int_tuple<((2,2,2,2,2,64),1)>, !fly.int_tuple<((1024,512,256,128,64,1),2048)>) -> !fly.layout<((2,2,2,2,2,64),1):((1024,512,256,128,64,1),2048)>
  %6 = fly.make_view(%arg1, %5) : (!fly.ptr<f16, shared>, !fly.layout<((2,2,2,2,2,64),1):((1024,512,256,128,64,1),2048)>) -> !fly.memref<f16, shared, ((2,2,2,2,2,64),1):((1024,512,256,128,64,1),2048)>
  %7 = fly.static : !fly.tile<[2|2|2|2|2|64]>
  %atom, %coordTensor = fly_rocdl.make_tiled_tdm_load_atom %arg0, %2, %7 : !fly.memref<f16, global, (2,2,2,2,2,128):(?,?,4096,1024,256,1)>, !fly.layout<(2,2,2,2,2,64):(1024,512,256,128,64,1)>, !fly.tile<[2|2|2|2|2|64]>
  %8 = fly.static : !fly.tile<[2|2|2|2|2|64]>
  %9 = fly.zipped_divide(%coordTensor, %8) : (!fly.coord_tensor<(0,0,0,0,0), (2,2,2,2,2,128):(?E4,?E4,1E3,1E2,1E1,1E0)>, !fly.tile<[2|2|2|2|2|64]>) -> !fly.coord_tensor<(0,0,0,0,0), ((2,2,2,2,2,64),(1,1,1,1,1,2)):((?E4,?E4,1E3,1E2,1E1,1E0),(0,0,0,0,0,64E0))>
  %10 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,(0,0,0,0,0,*))>
  %11 = fly.slice(%9, %10) : (!fly.coord_tensor<(0,0,0,0,0), ((2,2,2,2,2,64),(1,1,1,1,1,2)):((?E4,?E4,1E3,1E2,1E1,1E0),(0,0,0,0,0,64E0))>, !fly.int_tuple<(*,(0,0,0,0,0,*))>) -> !fly.coord_tensor<(0,0,0,0,0), ((2,2,2,2,2,64),2):((?E4,?E4,1E3,1E2,1E1,1E0),64E0)>
  %12 = fly.static : !fly.tile<[(((64,2,2,2),1),(2,2)):(((32,16,8,4),0),(2,1))|*]>
  %13 = fly.composition(%6, %12) : (!fly.memref<f16, shared, ((2,2,2,2,2,64),1):((1024,512,256,128,64,1),2048)>, !fly.tile<[(((64,2,2,2),1),(2,2)):(((32,16,8,4),0),(2,1))|*]>) -> !fly.memref<f16, shared, ((((64,2,2,2),1),(2,2)),1):((((1,64,128,256),0),(512,1024)),2048)>
  %14 = fly.make_int_tuple() : () -> !fly.int_tuple<(((*,0),*),*)>
  %15 = fly.slice(%13, %14) : (!fly.memref<f16, shared, ((((64,2,2,2),1),(2,2)),1):((((1,64,128,256),0),(512,1024)),2048)>, !fly.int_tuple<(((*,0),*),*)>) -> !fly.memref<f16, shared, ((64,2,2,2),(2,2),1):((1,64,128,256),(512,1024),2048)>
  %16 = fly.group(%15) {begin = 0 : i32, end = 2 : i32} : (!fly.memref<f16, shared, ((64,2,2,2),(2,2),1):((1,64,128,256),(512,1024),2048)>) -> !fly.memref<f16, shared, (((64,2,2,2),(2,2)),1):(((1,64,128,256),(512,1024)),2048)>
  %17 = fly.make_int_tuple() : () -> !fly.int_tuple<((1,1))>
  %18 = fly.coalesce(%16, %17) : (!fly.memref<f16, shared, (((64,2,2,2),(2,2)),1):(((1,64,128,256),(512,1024)),2048)>, !fly.int_tuple<((1,1))>) -> !fly.memref<f16, shared, ((512,4),1):((1,512),2048)>
  %19 = fly.static : !fly.tile<[(((64,2,2,2),1),(2,2)):(((32,16,8,4),0),(2,1))|*]>
  %20 = fly.composition(%11, %19) : (!fly.coord_tensor<(0,0,0,0,0), ((2,2,2,2,2,64),2):((?E4,?E4,1E3,1E2,1E1,1E0),64E0)>, !fly.tile<[(((64,2,2,2),1),(2,2)):(((32,16,8,4),0),(2,1))|*]>) -> !fly.coord_tensor<(0,0,0,0,0), ((((64,2,2,2),1),(2,2)),2):((((1E0,1E1,1E2,1E3),0),(?E4,?E4)),64E0)>
  %21 = fly.make_int_tuple() : () -> !fly.int_tuple<(((*,0),*),*)>
  %22 = fly.slice(%20, %21) : (!fly.coord_tensor<(0,0,0,0,0), ((((64,2,2,2),1),(2,2)),2):((((1E0,1E1,1E2,1E3),0),(?E4,?E4)),64E0)>, !fly.int_tuple<(((*,0),*),*)>) -> !fly.coord_tensor<(0,0,0,0,0), ((64,2,2,2),(2,2),2):((1E0,1E1,1E2,1E3),(?E4,?E4),64E0)>
  %23 = fly.group(%22) {begin = 0 : i32, end = 2 : i32} : (!fly.coord_tensor<(0,0,0,0,0), ((64,2,2,2),(2,2),2):((1E0,1E1,1E2,1E3),(?E4,?E4),64E0)>) -> !fly.coord_tensor<(0,0,0,0,0), (((64,2,2,2),(2,2)),2):(((1E0,1E1,1E2,1E3),(?E4,?E4)),64E0)>
  %24 = fly.make_int_tuple() : () -> !fly.int_tuple<((1,1))>
  %25 = fly.coalesce(%23, %24) : (!fly.coord_tensor<(0,0,0,0,0), (((64,2,2,2),(2,2)),2):(((1E0,1E1,1E2,1E3),(?E4,?E4)),64E0)>, !fly.int_tuple<((1,1))>) -> !fly.coord_tensor<(0,0,0,0,0), (((64,2,2,2),(2,2)),2):(((1E0,1E1,1E2,1E3),(?E4,?E4)),64E0)>
  %26 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,1)>
  %27 = fly.slice(%25, %26) : (!fly.coord_tensor<(0,0,0,0,0), (((64,2,2,2),(2,2)),2):(((1E0,1E1,1E2,1E3),(?E4,?E4)),64E0)>, !fly.int_tuple<(*,1)>) -> !fly.coord_tensor<(64,0,0,0,0), ((64,2,2,2),(2,2)):((1E0,1E1,1E2,1E3),(?E4,?E4))>
  %28 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,0)>
  %29 = fly.slice(%18, %28) : (!fly.memref<f16, shared, ((512,4),1):((1,512),2048)>, !fly.int_tuple<(*,0)>) -> !fly.memref<f16, shared, (512,4):(1,512)>
  fly.copy(%atom, %27, %29) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [1, 2, 2, 2, 64], elem = f16, tensor2tdm = (0,0,1E1,1E2,1E3,1E4)>, 16>, !fly.coord_tensor<(64,0,0,0,0), ((64,2,2,2),(2,2)):((1E0,1E1,1E2,1E3),(?E4,?E4))>, !fly.memref<f16, shared, (512,4):(1,512)>) -> ()
  return
}

// -----

// An f16 tensor moved as i32, into a padded LDS tile (8 f16 of slack per 32-f16 row),
// split between two warps. The descriptor counts i32: a 32 x 16 box, pad 4 dwords every
// 16, and each warp issues 16 rows. The atom's values stay f16 (valBits = 16), so
// `fly.copy` and `tdm_partition` count the same 512 elements per issue as the tensors.
// The CTA tile is rows 32..63, columns 32..63, and warp 1 takes rows 48..63.

// EXPAND-LABEL: @internal_pad_warps
// EXPAND: fly.make_copy_atom(%{{.*}}, %c32_i64, %c128_i32, %c32_i32 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [16, 16], elem = i32, tensor2tdm = (1E0,1/2E1), padInterval = 16, padAmount = 4>, 16>
// EXPAND: !fly.coord_tensor<(0,0), (128,64):(1E1,1/2E0)>
// Column 32 is i32 coordinate 16; row 48.
// EXPAND: fly.copy({{.*}}, !fly.coord_tensor<(16,48), ((32,16),1):((1/2E0,1E1),0)>, !fly.memref<f16, shared, ((32,16),1):((1,40),0)>) -> ()

// LOWER-LABEL: @internal_pad_warps
// Warp 1's share starts 16 padded rows into LDS.
// LOWER: llvm.getelementptr %arg1[640] : (!llvm.ptr<3>) -> !llvm.ptr<3>, f16
// (48 * 32 + 16) i32 = 6208 bytes.
// LOWER: arith.addi %{{.*}}, %c6208_i64 : i64
// LOWER: %[[S1:.*]] = llvm.trunc %c32_i64 : i64 to i32
// w0 = 0x06D20000: data_size 2 (i32), pad on, interval 16 dwords, amount 4 dwords.
// tensor_dim0 = 32 - 16, tensor_dim1 = 128 - 48 = 80; tile 16 x 16; row stride 32 i32.
// LOWER: vector.from_elements %c114425856_i32, %c1048576_i32, %c5242880_i32, %c1048576_i32, %c16_i32, %[[S1]], %{{.*}}, %c0_i32 : vector<8xi32>
// LOWER: rocdl.tensor.load.to.lds
// LOWER-NOT: rocdl.tensor.load.to.lds
// LOWER: return
func.func @internal_pad_warps(%arg0: !fly.memref<f16, global, (128,64):(64,1)>, %arg1: !fly.ptr<f16, shared>) {
  %0 = fly.make_int_tuple() : () -> !fly.int_tuple<(32,32)>
  %1 = fly.make_int_tuple() : () -> !fly.int_tuple<(40,1)>
  %2 = fly.make_layout(%0, %1) : (!fly.int_tuple<(32,32)>, !fly.int_tuple<(40,1)>) -> !fly.layout<(32,32):(40,1)>
  %3 = fly.make_int_tuple() : () -> !fly.int_tuple<((32,32),1)>
  %4 = fly.make_int_tuple() : () -> !fly.int_tuple<((40,1),1280)>
  %5 = fly.make_layout(%3, %4) : (!fly.int_tuple<((32,32),1)>, !fly.int_tuple<((40,1),1280)>) -> !fly.layout<((32,32),1):((40,1),1280)>
  %6 = fly.make_view(%arg1, %5) : (!fly.ptr<f16, shared>, !fly.layout<((32,32),1):((40,1),1280)>) -> !fly.memref<f16, shared, ((32,32),1):((40,1),1280)>
  %7 = fly.static : !fly.tile<[32|32]>
  %atom, %coordTensor = fly_rocdl.make_tiled_tdm_load_atom %arg0, %2, %7 {internalType = i32, numWarps = 2 : i32} : !fly.memref<f16, global, (128,64):(64,1)>, !fly.layout<(32,32):(40,1)>, !fly.tile<[32|32]>
  %8 = fly.static : !fly.tile<[32|32]>
  %9 = fly.zipped_divide(%coordTensor, %8) : (!fly.coord_tensor<(0,0), (128,64):(1E1,1/2E0)>, !fly.tile<[32|32]>) -> !fly.coord_tensor<(0,0), ((32,32),(4,2)):((1E1,1/2E0),(32E1,16E0))>
  %10 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,(1,*))>
  %11 = fly.slice(%9, %10) : (!fly.coord_tensor<(0,0), ((32,32),(4,2)):((1E1,1/2E0),(32E1,16E0))>, !fly.int_tuple<(*,(1,*))>) -> !fly.coord_tensor<(0,32), ((32,32),2):((1E1,1/2E0),16E0)>
  %12 = fly.static : !fly.tile<[(((32,16),2),1):(((32,1),16),0)|*]>
  %13 = fly.composition(%6, %12) : (!fly.memref<f16, shared, ((32,32),1):((40,1),1280)>, !fly.tile<[(((32,16),2),1):(((32,1),16),0)|*]>) -> !fly.memref<f16, shared, ((((32,16),2),1),1):((((1,40),640),0),1280)>
  %14 = fly.make_int_tuple() : () -> !fly.int_tuple<(((*,1),*),*)>
  %15 = fly.slice(%13, %14) : (!fly.memref<f16, shared, ((((32,16),2),1),1):((((1,40),640),0),1280)>, !fly.int_tuple<(((*,1),*),*)>) -> !fly.memref<f16, shared, ((32,16),1,1):((1,40),0,1280)>
  %16 = fly.group(%15) {begin = 0 : i32, end = 2 : i32} : (!fly.memref<f16, shared, ((32,16),1,1):((1,40),0,1280)>) -> !fly.memref<f16, shared, (((32,16),1),1):(((1,40),0),1280)>
  %17 = fly.make_int_tuple() : () -> !fly.int_tuple<((1,1))>
  %18 = fly.coalesce(%16, %17) : (!fly.memref<f16, shared, (((32,16),1),1):(((1,40),0),1280)>, !fly.int_tuple<((1,1))>) -> !fly.memref<f16, shared, (((32,16),1),1):(((1,40),0),1280)>
  %19 = fly.static : !fly.tile<[(((32,16),2),1):(((32,1),16),0)|*]>
  %20 = fly.composition(%11, %19) : (!fly.coord_tensor<(0,32), ((32,32),2):((1E1,1/2E0),16E0)>, !fly.tile<[(((32,16),2),1):(((32,1),16),0)|*]>) -> !fly.coord_tensor<(0,32), ((((32,16),2),1),2):((((1/2E0,1E1),16E1),0),16E0)>
  %21 = fly.make_int_tuple() : () -> !fly.int_tuple<(((*,1),*),*)>
  %22 = fly.slice(%20, %21) : (!fly.coord_tensor<(0,32), ((((32,16),2),1),2):((((1/2E0,1E1),16E1),0),16E0)>, !fly.int_tuple<(((*,1),*),*)>) -> !fly.coord_tensor<(0,48), ((32,16),1,2):((1/2E0,1E1),0,16E0)>
  %23 = fly.group(%22) {begin = 0 : i32, end = 2 : i32} : (!fly.coord_tensor<(0,48), ((32,16),1,2):((1/2E0,1E1),0,16E0)>) -> !fly.coord_tensor<(0,48), (((32,16),1),2):(((1/2E0,1E1),0),16E0)>
  %24 = fly.make_int_tuple() : () -> !fly.int_tuple<((1,1))>
  %25 = fly.coalesce(%23, %24) : (!fly.coord_tensor<(0,48), (((32,16),1),2):(((1/2E0,1E1),0),16E0)>, !fly.int_tuple<((1,1))>) -> !fly.coord_tensor<(0,48), (((32,16),1),2):(((1/2E0,1E1),0),16E0)>
  %26 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,1)>
  %27 = fly.slice(%25, %26) : (!fly.coord_tensor<(0,48), (((32,16),1),2):(((1/2E0,1E1),0),16E0)>, !fly.int_tuple<(*,1)>) -> !fly.coord_tensor<(16,48), ((32,16),1):((1/2E0,1E1),0)>
  %28 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,0)>
  %29 = fly.slice(%18, %28) : (!fly.memref<f16, shared, (((32,16),1),1):(((1,40),0),1280)>, !fly.int_tuple<(*,0)>) -> !fly.memref<f16, shared, ((32,16),1):((1,40),0)>
  fly.copy(%atom, %27, %29) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [16, 16], elem = i32, tensor2tdm = (1E0,1/2E1), padInterval = 16, padAmount = 4>, 16>, !fly.coord_tensor<(16,48), ((32,16),1):((1/2E0,1E1),0)>, !fly.memref<f16, shared, ((32,16),1):((1,40),0)>) -> ()
  return
}

// -----

// An extent-1 mode never reaches an address, so its stride does not have to survive the
// recast to i32: stride 3 is not a whole number of i32, and it is ignored rather than
// refused. The tile is rows 32..63, columns 32..63 of the only batch.

// EXPAND-LABEL: @unit_odd_stride
// EXPAND: fly.make_copy_atom(%{{.*}}, %c32_i64, %c128_i32, %c32_i32 : !fly.ptr<f16, global>, i64, i32, i32) {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [32, 16], elem = i32, tensor2tdm = (0,1E0,1/2E1)>, 16>
// EXPAND: !fly.coord_tensor<(0,0), (1,128,64):(0,1E1,1/2E0)>
// EXPAND: fly.copy({{.*}}, !fly.coord_tensor<(16,32), ((32,32),1):((1/2E0,1E1),0)>, !fly.memref<f16, shared, (1024,1):(1,0)>) -> ()

// LOWER-LABEL: @unit_odd_stride
// (32 * 64 + 32) f16 = 4160 bytes.
// LOWER: arith.addi %{{.*}}, %c4160_i64 : i64
// data_size 2, no pad; tensor_dim0 = 32 - 16, tensor_dim1 = 128 - 32 = 96; tile 16 x 32.
// LOWER: vector.from_elements %c131072_i32, %c1048576_i32, %c6291456_i32, %c1048576_i32, %c32_i32, %{{.*}}, %{{.*}}, %c0_i32 : vector<8xi32>
// LOWER: rocdl.tensor.load.to.lds
// LOWER-NOT: rocdl.tensor.load.to.lds
// LOWER: return
func.func @unit_odd_stride(%arg0: !fly.memref<f16, global, (1,128,64):(3,64,1)>, %arg1: !fly.ptr<f16, shared>) {
  %0 = fly.make_int_tuple() : () -> !fly.int_tuple<(32,32)>
  %1 = fly.make_int_tuple() : () -> !fly.int_tuple<(32,1)>
  %2 = fly.make_layout(%0, %1) : (!fly.int_tuple<(32,32)>, !fly.int_tuple<(32,1)>) -> !fly.layout<(32,32):(32,1)>
  %3 = fly.make_int_tuple() : () -> !fly.int_tuple<((32,32),1)>
  %4 = fly.make_int_tuple() : () -> !fly.int_tuple<((32,1),1024)>
  %5 = fly.make_layout(%3, %4) : (!fly.int_tuple<((32,32),1)>, !fly.int_tuple<((32,1),1024)>) -> !fly.layout<((32,32),1):((32,1),1024)>
  %6 = fly.make_view(%arg1, %5) : (!fly.ptr<f16, shared>, !fly.layout<((32,32),1):((32,1),1024)>) -> !fly.memref<f16, shared, ((32,32),1):((32,1),1024)>
  %7 = fly.static : !fly.tile<[1|32|32]>
  %atom, %coordTensor = fly_rocdl.make_tiled_tdm_load_atom %arg0, %2, %7 {internalType = i32} : !fly.memref<f16, global, (1,128,64):(3,64,1)>, !fly.layout<(32,32):(32,1)>, !fly.tile<[1|32|32]>
  %8 = fly.static : !fly.tile<[1|32|32]>
  %9 = fly.zipped_divide(%coordTensor, %8) : (!fly.coord_tensor<(0,0), (1,128,64):(0,1E1,1/2E0)>, !fly.tile<[1|32|32]>) -> !fly.coord_tensor<(0,0), ((1,32,32),(1,4,2)):((0,1E1,1/2E0),(0,32E1,16E0))>
  %10 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,(0,1,*))>
  %11 = fly.slice(%9, %10) : (!fly.coord_tensor<(0,0), ((1,32,32),(1,4,2)):((0,1E1,1/2E0),(0,32E1,16E0))>, !fly.int_tuple<(*,(0,1,*))>) -> !fly.coord_tensor<(0,32), ((1,32,32),2):((0,1E1,1/2E0),16E0)>
  %12 = fly.static : !fly.tile<[(((32,32),1),1):(((32,1),0),0)|*]>
  %13 = fly.composition(%6, %12) : (!fly.memref<f16, shared, ((32,32),1):((32,1),1024)>, !fly.tile<[(((32,32),1),1):(((32,1),0),0)|*]>) -> !fly.memref<f16, shared, ((((32,32),1),1),1):((((1,32),0),0),1024)>
  %14 = fly.make_int_tuple() : () -> !fly.int_tuple<(((*,0),*),*)>
  %15 = fly.slice(%13, %14) : (!fly.memref<f16, shared, ((((32,32),1),1),1):((((1,32),0),0),1024)>, !fly.int_tuple<(((*,0),*),*)>) -> !fly.memref<f16, shared, ((32,32),1,1):((1,32),0,1024)>
  %16 = fly.group(%15) {begin = 0 : i32, end = 2 : i32} : (!fly.memref<f16, shared, ((32,32),1,1):((1,32),0,1024)>) -> !fly.memref<f16, shared, (((32,32),1),1):(((1,32),0),1024)>
  %17 = fly.make_int_tuple() : () -> !fly.int_tuple<((1,1))>
  %18 = fly.coalesce(%16, %17) : (!fly.memref<f16, shared, (((32,32),1),1):(((1,32),0),1024)>, !fly.int_tuple<((1,1))>) -> !fly.memref<f16, shared, ((1024,1),1):((1,0),1024)>
  %19 = fly.static : !fly.tile<[(((32,32),1),1):(((32,1),0),0)|*]>
  %20 = fly.composition(%11, %19) : (!fly.coord_tensor<(0,32), ((1,32,32),2):((0,1E1,1/2E0),16E0)>, !fly.tile<[(((32,32),1),1):(((32,1),0),0)|*]>) -> !fly.coord_tensor<(0,32), ((((32,32),1),1),2):((((1/2E0,1E1),0),0),16E0)>
  %21 = fly.make_int_tuple() : () -> !fly.int_tuple<(((*,0),*),*)>
  %22 = fly.slice(%20, %21) : (!fly.coord_tensor<(0,32), ((((32,32),1),1),2):((((1/2E0,1E1),0),0),16E0)>, !fly.int_tuple<(((*,0),*),*)>) -> !fly.coord_tensor<(0,32), ((32,32),1,2):((1/2E0,1E1),0,16E0)>
  %23 = fly.group(%22) {begin = 0 : i32, end = 2 : i32} : (!fly.coord_tensor<(0,32), ((32,32),1,2):((1/2E0,1E1),0,16E0)>) -> !fly.coord_tensor<(0,32), (((32,32),1),2):(((1/2E0,1E1),0),16E0)>
  %24 = fly.make_int_tuple() : () -> !fly.int_tuple<((1,1))>
  %25 = fly.coalesce(%23, %24) : (!fly.coord_tensor<(0,32), (((32,32),1),2):(((1/2E0,1E1),0),16E0)>, !fly.int_tuple<((1,1))>) -> !fly.coord_tensor<(0,32), (((32,32),1),2):(((1/2E0,1E1),0),16E0)>
  %26 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,1)>
  %27 = fly.slice(%25, %26) : (!fly.coord_tensor<(0,32), (((32,32),1),2):(((1/2E0,1E1),0),16E0)>, !fly.int_tuple<(*,1)>) -> !fly.coord_tensor<(16,32), ((32,32),1):((1/2E0,1E1),0)>
  %28 = fly.make_int_tuple() : () -> !fly.int_tuple<(*,0)>
  %29 = fly.slice(%18, %28) : (!fly.memref<f16, shared, ((1024,1),1):((1,0),1024)>, !fly.int_tuple<(*,0)>) -> !fly.memref<f16, shared, (1024,1):(1,0)>
  fly.copy(%atom, %27, %29) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [32, 16], elem = i32, tensor2tdm = (0,1E0,1/2E1)>, 16>, !fly.coord_tensor<(16,32), ((32,32),1):((1/2E0,1E1),0)>, !fly.memref<f16, shared, (1024,1):(1,0)>) -> ()
  return
}
