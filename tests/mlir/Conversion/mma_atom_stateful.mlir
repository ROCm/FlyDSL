// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// Stateful MmaAtom lowering tests (CDNA4 MFMA_Scale).
//   fly.make_mma_atom   -> default (0, 0) state
//   fly.atom.set_value  -> llvm.insertvalue at scale_a / scale_b field
//   fly.mma_atom_call   -> rocdl.mfma.scale.f32.*.f8f6f4 intrinsic
// Register memrefs are materialised inside the function body via
// fly.memref.alloca (register memrefs are not valid function arguments).

// -----

// Stateful MMA atom type converts to !llvm.struct<(i32, i32)>
// CHECK-LABEL: @test_stateful_mma_scale_type
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>)
func.func @test_stateful_mma_scale_type(
    %atom: !fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>) {
  return
}

// -----

// make_mma_atom produces default state via getDefaultState (scaleA/scaleB = 0)
// and the atom feeds into mma_atom_call so it is not DCE'd.

// CHECK-LABEL: @test_make_mma_atom_default_scales
func.func @test_make_mma_atom_default_scales() {
  %lay_ab = fly.static : !fly.layout<32:1>
  %lay_cd = fly.static : !fly.layout<4:1>

  // CHECK: llvm.alloca %{{.*}} x f32 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca %{{.*}} x i8 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca %{{.*}} x i8 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca %{{.*}} x f32 : (i64) -> !llvm.ptr<5>
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<4:1>) -> !fly.memref<f32, register, 4:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E4M3FN, register, 32:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E4M3FN, register, 32:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<4:1>) -> !fly.memref<f32, register, 4:1>

  // CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[C0]], %[[UNDEF]][0]
  // CHECK: llvm.insertvalue %[[C0]], %[[S1]][1]
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>, !fly.memref<f32, register, 4:1>, !fly.memref<f8E4M3FN, register, 32:1>, !fly.memref<f8E4M3FN, register, 32:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// -----

// End-to-end: set scales then mma_atom_call lowers to rocdl.mfma.scale.f32.16x16x128.f8f6f4

// CHECK-LABEL: @test_mma_scale_atom_call_16x16x128
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>, %[[SA:.*]]: i32, %[[SB:.*]]: i32)
func.func @test_mma_scale_atom_call_16x16x128(
    %atom: !fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>,
    %scale_a: i32,
    %scale_b: i32) {
  %lay_ab = fly.static : !fly.layout<32:1>
  %lay_cd = fly.static : !fly.layout<4:1>

  // CHECK: %[[D:.*]] = llvm.alloca %{{.*}} x f32 : (i64) -> !llvm.ptr<5>
  // CHECK: %[[A:.*]] = llvm.alloca %{{.*}} x i8 : (i64) -> !llvm.ptr<5>
  // CHECK: %[[B:.*]] = llvm.alloca %{{.*}} x i8 : (i64) -> !llvm.ptr<5>
  // CHECK: %[[C:.*]] = llvm.alloca %{{.*}} x f32 : (i64) -> !llvm.ptr<5>
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<4:1>) -> !fly.memref<f32, register, 4:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E4M3FN, register, 32:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E4M3FN, register, 32:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<4:1>) -> !fly.memref<f32, register, 4:1>

  // CHECK: %[[A1:.*]] = llvm.insertvalue %[[SA]], %[[ATOM]][0]
  %atom_a = fly.atom.set_value(%atom, "scale_a", %scale_a) : (!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>, i32) -> !fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>
  // CHECK: %[[A2:.*]] = llvm.insertvalue %[[SB]], %[[A1]][1]
  %atom_ab = fly.atom.set_value(%atom_a, "scale_b", %scale_b) : (!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>, i32) -> !fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<8xi32>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<8xi32>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<4xf32>
  // CHECK-DAG: %[[SA_VAL:.*]] = llvm.extractvalue %[[A2]][0]
  // CHECK-DAG: %[[SB_VAL:.*]] = llvm.extractvalue %[[A2]][1]
  // CHECK: %[[RES:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]], 0, 0, 0, %[[SA_VAL]], 0, %[[SB_VAL]]
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<4xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom_ab, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>, !fly.memref<f32, register, 4:1>, !fly.memref<f8E4M3FN, register, 32:1>, !fly.memref<f8E4M3FN, register, 32:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// -----

// 32x32x64 shape with f4E2M1FN / f8E5M2 mixed operands -> cbsz=4 (fp4), blgp=1 (bf8/E5M2)

// CHECK-LABEL: @test_mma_scale_atom_call_32x32x64_mixed
func.func @test_mma_scale_atom_call_32x32x64_mixed(
    %atom: !fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<32x32x64, (f4E2M1FN, f8E5M2) -> f32, opselA = 0, opselB = 0>>) {
  %lay_ab = fly.static : !fly.layout<32:1>
  %lay_cd = fly.static : !fly.layout<16:1>

  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<16:1>) -> !fly.memref<f32, register, 16:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f4E2M1FN, register, 32:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E5M2, register, 32:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<16:1>) -> !fly.memref<f32, register, 16:1>

  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 %{{.*}}, %{{.*}}, %{{.*}}, 4, 1, 0, %{{.*}}, 0, %{{.*}}
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<32x32x64, (f4E2M1FN, f8E5M2) -> f32, opselA = 0, opselB = 0>>, !fly.memref<f32, register, 16:1>, !fly.memref<f4E2M1FN, register, 32:1>, !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// -----

// User-specified opselA / opselB are forwarded to the intrinsic as the two
// I32 attrs attached to scaleA / scaleB. opsel is part of the atom type — to
// change it at runtime, construct a new mma_atom with the desired opsel.

// CHECK-LABEL: @test_mma_scale_atom_call_with_opsel
func.func @test_mma_scale_atom_call_with_opsel(
    %atom: !fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 2, opselB = 3>>) {
  %lay_ab = fly.static : !fly.layout<32:1>
  %lay_cd = fly.static : !fly.layout<4:1>

  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<4:1>) -> !fly.memref<f32, register, 4:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E4M3FN, register, 32:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E4M3FN, register, 32:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<4:1>) -> !fly.memref<f32, register, 4:1>

  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 %{{.*}}, %{{.*}}, %{{.*}}, 0, 0, 2, %{{.*}}, 3, %{{.*}}
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 2, opselB = 3>>, !fly.memref<f32, register, 4:1>, !fly.memref<f8E4M3FN, register, 32:1>, !fly.memref<f8E4M3FN, register, 32:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}
