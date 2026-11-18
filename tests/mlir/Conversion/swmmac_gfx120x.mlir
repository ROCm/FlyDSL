// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// GFX120X 2:4 structured-sparse SWMMAC. Packed widths are RDNA4 wave32 ISel classes:
//   f16     16x16x32 : A = vector<8xf16>,  B = vector<16xf16>, C = vector<8xf32> | vector<8xf16>, idx = i32
//   bf16    16x16x32 : A = vector<8xi16>,  B = vector<16xi16>, C = vector<8xf32> | vector<8xi16>, idx = i32
//   fp8/bf8 16x16x32 : A = vector<2xi32>,  B = vector<4xi32>,  C = vector<8xf32>, idx = i32
//   iu8     16x16x32 : A = vector<2xi32>,  B = vector<4xi32>,  C = vector<8xi32>, idx = i32
//   iu4     16x16x32 : A = i32,            B = vector<2xi32>,  C = vector<8xi32>, idx = i32
//   iu4     16x16x64 : A = vector<2xi32>,  B = vector<4xi32>,  C = vector<8xi32>, idx = i32

// CHECK-LABEL: @test_gfx120x_swmmac_f16
func.func @test_gfx120x_swmmac_f16(
    %a: vector<8xf16>,
    %idx: i32,
    %b: vector<16xf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: rocdl.swmmac.f32.16x16x32.f16 {{.*}}, {{.*}}, {{.*}}, {{.*}} : (vector<8xf16>, vector<16xf16>, vector<8xf32>, i32) -> vector<8xf32>
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf16>, i32, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_f16_f16
func.func @test_gfx120x_swmmac_f16_f16(
    %a: vector<8xf16>,
    %idx: i32,
    %b: vector<16xf16>,
    %c: vector<8xf16>) -> vector<8xf16> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f16, signA = false, signB = false, clamp = false>>
  // CHECK: rocdl.swmmac.f16.16x16x32.f16 {{.*}}, {{.*}}, {{.*}}, {{.*}} : (vector<8xf16>, vector<16xf16>, vector<8xf16>, i32) -> vector<8xf16>
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f16, signA = false, signB = false, clamp = false>>, vector<8xf16>, i32, vector<16xf16>, vector<8xf16>) -> vector<8xf16>
  return %res : vector<8xf16>
}

// CHECK-LABEL: @test_gfx120x_swmmac_bf16
func.func @test_gfx120x_swmmac_bf16(
    %a: vector<8xbf16>,
    %idx: i32,
    %b: vector<16xbf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<8xbf16> to vector<8xi16>
  // CHECK: llvm.bitcast {{.*}} : vector<16xbf16> to vector<16xi16>
  // CHECK: rocdl.swmmac.f32.16x16x32.bf16
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, vector<8xbf16>, i32, vector<16xbf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_bf16_bf16
func.func @test_gfx120x_swmmac_bf16_bf16(
    %a: vector<8xbf16>,
    %idx: i32,
    %b: vector<16xbf16>,
    %c: vector<8xbf16>) -> vector<8xbf16> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> bf16, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<8xbf16> to vector<8xi16>
  // CHECK: llvm.bitcast {{.*}} : vector<16xbf16> to vector<16xi16>
  // CHECK: llvm.bitcast {{.*}} : vector<8xbf16> to vector<8xi16>
  // CHECK: %[[RAW:.*]] = rocdl.swmmac.bf16.16x16x32.bf16 {{.*}}, {{.*}}, {{.*}}, {{.*}} : (vector<8xi16>, vector<16xi16>, vector<8xi16>, i32) -> vector<8xi16>
  // CHECK: llvm.bitcast %[[RAW]] : vector<8xi16> to vector<8xbf16>
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> bf16, signA = false, signB = false, clamp = false>>, vector<8xbf16>, i32, vector<16xbf16>, vector<8xbf16>) -> vector<8xbf16>
  return %res : vector<8xbf16>
}

// CHECK-LABEL: @test_gfx120x_swmmac_fp8_fp8
func.func @test_gfx120x_swmmac_fp8_fp8(
    %a: vector<8xf8E4M3FN>,
    %idx: i32,
    %b: vector<16xf8E4M3FN>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<8xi8> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.fp8
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E4M3FN>, i32, vector<16xf8E4M3FN>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_fp8_bf8
func.func @test_gfx120x_swmmac_fp8_bf8(
    %a: vector<8xf8E4M3FN>,
    %idx: i32,
    %b: vector<16xf8E5M2>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E4M3FN, f8E5M2) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.bf8
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E4M3FN, f8E5M2) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E4M3FN>, i32, vector<16xf8E5M2>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_bf8_fp8
func.func @test_gfx120x_swmmac_bf8_fp8(
    %a: vector<8xf8E5M2>,
    %idx: i32,
    %b: vector<16xf8E4M3FN>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E5M2, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.fp8
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E5M2, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E5M2>, i32, vector<16xf8E4M3FN>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_bf8_bf8
func.func @test_gfx120x_swmmac_bf8_bf8(
    %a: vector<8xf8E5M2>,
    %idx: i32,
    %b: vector<16xf8E5M2>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E5M2, f8E5M2) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.bf8
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f8E5M2, f8E5M2) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf8E5M2>, i32, vector<16xf8E5M2>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_iu8_unsigned
func.func @test_gfx120x_swmmac_iu8_unsigned(
    %a: vector<8xui8>,
    %idx: i32,
    %b: vector<16xui8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (ui8, ui8) -> i32, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<8xui8> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<16xui8> to vector<4xi32>
  // CHECK: rocdl.swmmac.i32.16x16x32.iu8
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (ui8, ui8) -> i32, signA = false, signB = false, clamp = false>>, vector<8xui8>, i32, vector<16xui8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_iu8_signed_clamp
func.func @test_gfx120x_swmmac_iu8_signed_clamp(
    %a: vector<8xsi8>,
    %idx: i32,
    %b: vector<16xsi8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (si8, si8) -> i32, signA = true, signB = true, clamp = true>>
  // CHECK: rocdl.swmmac.i32.16x16x32.iu8
  // CHECK-SAME: clamp = true
  // CHECK-SAME: signA = true
  // CHECK-SAME: signB = true
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (si8, si8) -> i32, signA = true, signB = true, clamp = true>>, vector<8xsi8>, i32, vector<16xsi8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_iu4_k32_unsigned
func.func @test_gfx120x_swmmac_iu4_k32_unsigned(
    %a: vector<8xi4>,
    %idx: i32,
    %b: vector<16xi4>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<8xi4> to i32
  // CHECK: llvm.bitcast {{.*}} : vector<16xi4> to vector<2xi32>
  // CHECK: rocdl.swmmac.i32.16x16x32.iu4
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>, vector<8xi4>, i32, vector<16xi4>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_iu4_k64_signed_clamp
func.func @test_gfx120x_swmmac_iu4_k64_signed_clamp(
    %a: vector<16xi4>,
    %idx: i32,
    %b: vector<32xi4>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x64, (i4, i4) -> i32, signA = true, signB = true, clamp = true>>
  // CHECK: llvm.bitcast {{.*}} : vector<16xi4> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<32xi4> to vector<4xi32>
  // CHECK: rocdl.swmmac.i32.16x16x64.iu4
  // CHECK-SAME: clamp = true
  // CHECK-SAME: signA = true
  // CHECK-SAME: signB = true
  %res = fly.mma_atom_call_ssa(%atom, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x64, (i4, i4) -> i32, signA = true, signB = true, clamp = true>>, vector<16xi4>, i32, vector<32xi4>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_swmmac_atom_call_memref_iu4_k32
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[IDX:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_gfx120x_swmmac_atom_call_memref_iu4_k32(
    %d: !fly.memref<i32, register, 8:1>,
    %a: !fly.memref<i4, register, 8:1>,
    %idx: !fly.memref<i32, register, 1:1>,
    %b: !fly.memref<i4, register, 16:1>,
    %c: !fly.memref<i32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i4, i4) -> i32, signA = true, signB = false, clamp = false>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> i32
  // CHECK: %[[IDX_VAL:.*]] = llvm.load %[[IDX]] : !llvm.ptr<5> -> i32
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<2xi32>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xi32>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.i32.16x16x32.iu4 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]], %[[IDX_VAL]] {signA = true}
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<8xi32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i4, i4) -> i32, signA = true, signB = false, clamp = false>>, !fly.memref<i32, register, 8:1>, !fly.memref<i4, register, 8:1>, !fly.memref<i32, register, 1:1>, !fly.memref<i4, register, 16:1>, !fly.memref<i32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_gfx120x_swmmac_atom_call_memref_iu4_k64
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[IDX:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_gfx120x_swmmac_atom_call_memref_iu4_k64(
    %d: !fly.memref<i32, register, 8:1>,
    %a: !fly.memref<i4, register, 16:1>,
    %idx: !fly.memref<i32, register, 1:1>,
    %b: !fly.memref<i4, register, 32:1>,
    %c: !fly.memref<i32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x64, (i4, i4) -> i32, signA = false, signB = true, clamp = false>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<2xi32>
  // CHECK: %[[IDX_VAL:.*]] = llvm.load %[[IDX]] : !llvm.ptr<5> -> i32
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<4xi32>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xi32>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.i32.16x16x64.iu4 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]], %[[IDX_VAL]] {signB = true}
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<8xi32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x64, (i4, i4) -> i32, signA = false, signB = true, clamp = false>>, !fly.memref<i32, register, 8:1>, !fly.memref<i4, register, 16:1>, !fly.memref<i32, register, 1:1>, !fly.memref<i4, register, 32:1>, !fly.memref<i32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_gfx120x_swmmac_atom_call_memref_bf16_bf16
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[IDX:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_gfx120x_swmmac_atom_call_memref_bf16_bf16(
    %d: !fly.memref<bf16, register, 8:1>,
    %a: !fly.memref<bf16, register, 8:1>,
    %idx: !fly.memref<i32, register, 1:1>,
    %b: !fly.memref<bf16, register, 16:1>,
    %c: !fly.memref<bf16, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> bf16, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<8xi16>
  // CHECK: %[[IDX_VAL:.*]] = llvm.load %[[IDX]] : !llvm.ptr<5> -> i32
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<16xi16>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xi16>
  // CHECK: %[[RES:.*]] = rocdl.swmmac.bf16.16x16x32.bf16 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]], %[[IDX_VAL]] : (vector<8xi16>, vector<16xi16>, vector<8xi16>, i32) -> vector<8xi16>
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<8xi16>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, [%a, %idx], %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> bf16, signA = false, signB = false, clamp = false>>, !fly.memref<bf16, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<i32, register, 1:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<bf16, register, 8:1>) -> ()
  return
}
