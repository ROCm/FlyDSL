// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// GFX120X (RDNA4: gfx1200 / gfx1201) WMMA wave32 atom lowering tests:
//   fly.mma_atom_call -> rocdl.wmma.f32.16x16x16.{f16,bf16,fp8_fp8} intrinsic
//
// RDNA4 keeps the gfx11 16x16x16 instruction shapes but uses the gfx1250
// "v8" register ABI, so the per-lane fragment shapes are half the gfx11 ones:
//   A, B : 8 elements   (gfx11 has 16, broadcast across the two lane halves)
//   C, D : 8 f32 slots  (vector<8xf32>, same as gfx11)
//
// The bf16 intrinsic takes integer operands, so bf16 A/B lower to
// vector<8xi16>.

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_fp8
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_gfx120x_wmma_atom_call_fp8(
    %d: !fly.memref<f32, register, 8:1>,
    %a: !fly.memref<f8E4M3FN, register, 8:1>,
    %b: !fly.memref<f8E4M3FN, register, 8:1>,
    %c: !fly.memref<f32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<2xi32>
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<2xi32>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xf32>
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x16.fp8_fp8 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]]
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<8xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, 8:1>, !fly.memref<f8E4M3FN, register, 8:1>, !fly.memref<f8E4M3FN, register, 8:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_fp8
// CHECK-SAME: (%[[A:.*]]: vector<8xi8>, %[[B:.*]]: vector<8xi8>, %[[C:.*]]: vector<8xf32>)
func.func @test_gfx120x_wmma_atom_call_ssa_fp8(
    %a: vector<8xf8E4M3FN>,
    %b: vector<8xf8E4M3FN>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<8xi8> to vector<2xi32>
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<8xi8> to vector<2xi32>
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x16.fp8_fp8 %[[A_CAST]], %[[B_CAST]], %[[C]]
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E4M3FN>, vector<8xf8E4M3FN>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_bf16
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_gfx120x_wmma_atom_call_bf16(
    %d: !fly.memref<f32, register, 8:1>,
    %a: !fly.memref<bf16, register, 8:1>,
    %b: !fly.memref<bf16, register, 8:1>,
    %c: !fly.memref<f32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<8xi16>
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<8xi16>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xf32>
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x16.bf16 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]]
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<8xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// A 2x2 wave layout over the 32-lane atom covers 128 threads, which is the
// launch shape the RDNA4 GEMM kernel uses.
//
// CHECK-LABEL: @test_gfx120x_wmma_gemm_from_tiled_mma_arg
// CHECK: rocdl.wmma.f32.16x16x16.bf16
func.func @test_gfx120x_wmma_gemm_from_tiled_mma_arg(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, <(2,2,1):(2,1,0)>>,
    %d: !fly.memref<f32, register, 8:1>,
    %a: !fly.memref<bf16, register, 8:1>,
    %b: !fly.memref<bf16, register, 8:1>,
    %c: !fly.memref<f32, register, 8:1>) {
  fly.gemm(%tiled_mma, %d, %a, %b, %c) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, <(2,2,1):(2,1,0)>>, !fly.memref<f32, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_bf16
// CHECK-SAME: (%[[A:.*]]: vector<8xbf16>, %[[B:.*]]: vector<8xbf16>, %[[C:.*]]: vector<8xf32>)
func.func @test_gfx120x_wmma_atom_call_ssa_bf16(
    %a: vector<8xbf16>,
    %b: vector<8xbf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<8xbf16> to vector<8xi16>
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<8xbf16> to vector<8xi16>
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x16.bf16 %[[A_CAST]], %[[B_CAST]], %[[C]]
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, vector<8xbf16>, vector<8xbf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_f16
// CHECK-SAME: (%[[A:.*]]: vector<8xf16>, %[[B:.*]]: vector<8xf16>, %[[C:.*]]: vector<8xf32>)
func.func @test_gfx120x_wmma_atom_call_ssa_f16(
    %a: vector<8xf16>,
    %b: vector<8xf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x16.f16 %[[A]], %[[B]], %[[C]]
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf16>, vector<8xf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}
