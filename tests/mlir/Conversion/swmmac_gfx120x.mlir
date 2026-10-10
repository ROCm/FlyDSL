// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// GFX120X SWMMAC wave32 atom lowering.
// Sparse index is the second A-group operand. Same-type f16/bf16 accumulators
// are not emitted.

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_f16
// CHECK-SAME: (%[[A:.*]]: vector<8xf16>, %[[IDX:.*]]: i32, %[[B:.*]]: vector<16xf16>, %[[C:.*]]: vector<8xf32>)
func.func @test_gfx120x_swmmac_ssa_f16(
    %a: vector<8xf16>,
    %idx: i32,
    %b: vector<16xf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.f32.16x16x32.f16 %[[A]], %[[B]], %[[C]], %[[IDX]]
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf16>, i32, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_bf16
// CHECK-SAME: (%[[A:.*]]: vector<8xbf16>, %[[IDX:.*]]: i32, %[[B:.*]]: vector<16xbf16>, %[[C:.*]]: vector<8xf32>)
func.func @test_gfx120x_swmmac_ssa_bf16(
    %a: vector<8xbf16>,
    %idx: i32,
    %b: vector<16xbf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<8xbf16> to vector<8xi16>
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<16xbf16> to vector<16xi16>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.f32.16x16x32.bf16 %[[A_CAST]], %[[B_CAST]], %[[C]], %[[IDX]]
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, vector<8xbf16>, i32, vector<16xbf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_fp8_fp8
// CHECK-SAME: (%[[A:.*]]: vector<8xi8>, %[[IDX:.*]]: i32, %[[B:.*]]: vector<16xi8>, %[[C:.*]]: vector<8xf32>)
func.func @test_gfx120x_swmmac_ssa_fp8_fp8(
    %a: vector<8xf8E4M3FN>,
    %idx: i32,
    %b: vector<16xf8E4M3FN>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<8xi8> to vector<2xi32>
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<16xi8> to vector<4xi32>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.f32.16x16x32.fp8.fp8 %[[A_CAST]], %[[B_CAST]], %[[C]], %[[IDX]]
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E4M3FN>, i32, vector<16xf8E4M3FN>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_fp8_bf8
func.func @test_gfx120x_swmmac_ssa_fp8_bf8(
    %a: vector<8xf8E4M3FN>, %idx: i32, %b: vector<16xf8E5M2>, %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E4M3FN, f8E5M2) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.bf8
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E4M3FN, f8E5M2) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E4M3FN>, i32, vector<16xf8E5M2>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_bf8_fp8
func.func @test_gfx120x_swmmac_ssa_bf8_fp8(
    %a: vector<8xf8E5M2>, %idx: i32, %b: vector<16xf8E4M3FN>, %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E5M2, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.fp8
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E5M2, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E5M2>, i32, vector<16xf8E4M3FN>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_bf8_bf8
func.func @test_gfx120x_swmmac_ssa_bf8_bf8(
    %a: vector<8xf8E5M2>, %idx: i32, %b: vector<16xf8E5M2>, %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E5M2, f8E5M2) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.bf8
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E5M2, f8E5M2) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E5M2>, i32, vector<16xf8E5M2>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_iu8
// CHECK-SAME: (%[[A:.*]]: vector<8xi8>, %[[IDX:.*]]: i32, %[[B:.*]]: vector<16xi8>, %[[C:.*]]: vector<8xi32>)
func.func @test_gfx120x_swmmac_ssa_iu8(
    %a: vector<8xi8>, %idx: i32, %b: vector<16xi8>, %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i8, i8) -> i32, signA = true, signB = true, clamp = false>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<8xi8> to vector<2xi32>
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<16xi8> to vector<4xi32>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.i32.16x16x32.iu8 %[[A_CAST]], %[[B_CAST]], %[[C]], %[[IDX]] {{{.*}}signA = true{{.*}}signB = true
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i8, i8) -> i32, signA = true, signB = true, clamp = false>>, vector<8xi8>, i32, vector<16xi8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_iu4_k32
// CHECK-SAME: (%[[A:.*]]: vector<8xi4>, %[[IDX:.*]]: i32, %[[B:.*]]: vector<16xi4>, %[[C:.*]]: vector<8xi32>)
func.func @test_gfx120x_swmmac_ssa_iu4_k32(
    %a: vector<8xi4>, %idx: i32, %b: vector<16xi4>, %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i4, i4) -> i32, signA = true, signB = true, clamp = false>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<8xi4> to i32
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<16xi4> to vector<2xi32>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.i32.16x16x32.iu4 %[[A_CAST]], %[[B_CAST]], %[[C]], %[[IDX]] {{{.*}}signA = true{{.*}}signB = true
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i4, i4) -> i32, signA = true, signB = true, clamp = false>>, vector<8xi4>, i32, vector<16xi4>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_ssa_iu4_k64
// CHECK-SAME: (%[[A:.*]]: vector<16xi4>, %[[IDX:.*]]: i32, %[[B:.*]]: vector<32xi4>, %[[C:.*]]: vector<8xi32>)
func.func @test_gfx120x_swmmac_ssa_iu4_k64(
    %a: vector<16xi4>, %idx: i32, %b: vector<32xi4>, %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x64, (i4, i4) -> i32, signA = true, signB = true, clamp = false>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<16xi4> to vector<2xi32>
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<32xi4> to vector<4xi32>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.i32.16x16x64.iu4 %[[A_CAST]], %[[B_CAST]], %[[C]], %[[IDX]] {{{.*}}signA = true{{.*}}signB = true
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x64, (i4, i4) -> i32, signA = true, signB = true, clamp = false>>, vector<16xi4>, i32, vector<32xi4>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// Memref form: loads A/index/B/C then stores the result.
// CHECK-LABEL: @test_gfx120x_swmmac_atom_call_f16
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[IDX:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_gfx120x_swmmac_atom_call_f16(
    %d: !fly.memref<f32, register, 8:1>,
    %a: !fly.memref<f16, register, 8:1>,
    %idx: !fly.memref<i32, register, 1:1>,
    %b: !fly.memref<f16, register, 16:1>,
    %c: !fly.memref<f32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<8xf16>
  // CHECK: %[[IDX_VAL:.*]] = llvm.load %[[IDX]] : !llvm.ptr<5> -> i32
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<16xf16>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xf32>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.f32.16x16x32.f16 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]], %[[IDX_VAL]]
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<8xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, 8:1>, !fly.memref<f16, register, 8:1>, !fly.memref<i32, register, 1:1>, !fly.memref<f16, register, 16:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}
