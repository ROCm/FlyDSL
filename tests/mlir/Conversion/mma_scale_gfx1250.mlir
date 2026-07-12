// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// gfx1250 MX-scaled WMMA atom lowering (E8M0 block scale, V_WMMA_SCALE).
//   fly.make_mma_atom  -> default (0, 0) scale state (!llvm.struct<(i32, i32)>)
//   fly.atom.set_value -> llvm.insertvalue at scale_a / scale_b field
//   fly.mma_atom_call  -> rocdl.wmma.scale.f32.16x16x128.f8f6f4

// -----

// Stateful scaled MMA atom type converts to !llvm.struct<(i32, i32)>
// CHECK-LABEL: @test_wmma_scale_type
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>)
func.func @test_wmma_scale_type(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>) {
  return
}

// -----

// make_mma_atom produces default scale state (scale_a = scale_b = 0).
// CHECK-LABEL: @test_make_wmma_scale_default
func.func @test_make_wmma_scale_default() {
  %lay_ab = fly.static : !fly.layout<64:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<64:1>) -> !fly.memref<f8E4M3FN, register, 64:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<64:1>) -> !fly.memref<f8E4M3FN, register, 64:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>

  // CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[C0]], %[[UNDEF]][0]
  // CHECK: llvm.insertvalue %[[C0]], %[[S1]][1]
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>, !fly.memref<f32, register, 8:1>, !fly.memref<f8E4M3FN, register, 64:1>, !fly.memref<f8E4M3FN, register, 64:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// -----

// End-to-end fp8: set scales then mma_atom_call -> rocdl.wmma.scale.f32.16x16x128.f8f6f4
// fp8 A/B operands are vector<16xi32>; f32 accumulator is vector<8xf32>.

// CHECK-LABEL: @test_wmma_scale_call_fp8
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>, %[[SA:.*]]: i32, %[[SB:.*]]: i32)
func.func @test_wmma_scale_call_fp8(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>,
    %scale_a: i32,
    %scale_b: i32) {
  %lay_ab = fly.static : !fly.layout<64:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<64:1>) -> !fly.memref<f8E4M3FN, register, 64:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<64:1>) -> !fly.memref<f8E4M3FN, register, 64:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>

  // CHECK: %[[A1:.*]] = llvm.insertvalue %[[SA]], %[[ATOM]][0]
  %atom_a = fly.atom.set_value(%atom, "scale_a", %scale_a) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>, i32) -> !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>
  // CHECK: %[[A2:.*]] = llvm.insertvalue %[[SB]], %[[A1]][1]
  %atom_ab = fly.atom.set_value(%atom_a, "scale_b", %scale_b) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>, i32) -> !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf32>
  // CHECK-DAG: %[[SA_VAL:.*]] = llvm.extractvalue %[[A2]][0]
  // CHECK-DAG: %[[SB_VAL:.*]] = llvm.extractvalue %[[A2]][1]
  // CHECK: %[[RES:.*]] = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]], %[[SA_VAL]], %[[SB_VAL]] : (vector<16xi32>, vector<16xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>
  // CHECK: llvm.store %[[RES]], %{{.*}} : vector<8xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom_ab, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>>, !fly.memref<f32, register, 8:1>, !fly.memref<f8E4M3FN, register, 64:1>, !fly.memref<f8E4M3FN, register, 64:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// -----

// fp4 operands are vector<8xi32>; opsel is forwarded via scaleAType / scaleBType.

// CHECK-LABEL: @test_wmma_scale_call_fp4_opsel
func.func @test_wmma_scale_call_fp4_opsel(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f4E2M1FN, f4E2M1FN) -> f32, opselA = 1, opselB = 2>>) {
  %lay_ab = fly.static : !fly.layout<32:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f4E2M1FN, register, 32:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f4E2M1FN, register, 32:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>

  // CHECK: rocdl.wmma.scale.f32.16x16x128.f8f6f4 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {fmtA = 4 : i32, fmtB = 4 : i32, scaleAType = 1 : i32, scaleBType = 2 : i32} : (vector<8xi32>, vector<8xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f4E2M1FN, f4E2M1FN) -> f32, opselA = 1, opselB = 2>>, !fly.memref<f32, register, 8:1>, !fly.memref<f4E2M1FN, register, 32:1>, !fly.memref<f4E2M1FN, register, 32:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}
