// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// MMA atom call lowering tests:
//   fly.mma_atom_call -> rocdl.mfma intrinsic
//   Loads A/B as scalars, C as accumulator vector,
//   calls MFMA, stores result back to D

// CHECK-LABEL: @test_mma_atom_call
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_mma_atom_call(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f32, register, 1:1>,
    %b: !fly.memref<f32, register, 1:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> f32
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> f32
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: %[[RES:.*]] = rocdl.mfma.f32.16x16x4f32 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]]
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<4xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>, !fly.memref<f32, register, 4:1>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// CHECK-LABEL: @test_gemm_from_tiled_mma_arg
// CHECK: rocdl.mfma.f32.16x16x4f32
func.func @test_gemm_from_tiled_mma_arg(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>, <(1,4,1):(0,1,0)>>,
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f32, register, 1:1>,
    %b: !fly.memref<f32, register, 1:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  fly.gemm(%tiled_mma, %d, %a, %b, %c) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>, <(1,4,1):(0,1,0)>>, !fly.memref<f32, register, 4:1>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// CHECK-LABEL: @test_mma_atom_call_ssa_fp8
// CHECK-SAME: (%[[A:.*]]: vector<8xi8>, %[[B:.*]]: vector<8xi8>, %[[C:.*]]: vector<4xf32>)
func.func @test_mma_atom_call_ssa_fp8(
    %a: vector<8xi8>,
    %b: vector<8xi8>,
    %c: vector<4xf32>) -> vector<4xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (f8E4M3FNUZ, f8E4M3FNUZ) -> f32>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<8xi8> to i64
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<8xi8> to i64
  // CHECK: %[[RES:.*]] = rocdl.mfma.f32.16x16x32.fp8.fp8 %[[A_CAST]], %[[B_CAST]], %[[C]]
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (f8E4M3FNUZ, f8E4M3FNUZ) -> f32>>, vector<8xi8>, vector<8xi8>, vector<4xf32>) -> vector<4xf32>
  return %res : vector<4xf32>
}
