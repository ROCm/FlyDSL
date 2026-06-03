// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// GFX11 (RDNA3 / RDNA3.5) WMMA wave32 atom lowering tests:
//   fly.mma_atom_call -> rocdl.wmma.f32.16x16x16.bf16 intrinsic
//
// Wave32 fragment shapes per lane (16x16x16 bf16 -> f32):
//   A, B : 16 bf16 elements        (lowered to vector<16xi16> for the intrinsic)
//   C, D : 8 f32 accumulator slots (vector<8xf32>)
//
// The atom type carries three always-printed bool parameters
// (signA, signB, clamp) — forwarded to the ROCDL iu8/iu4 intrinsic on
// integer paths, and required to be false on fp16/bf16 paths.

// CHECK-LABEL: @test_gfx11_wmma_atom_call_bf16
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_gfx11_wmma_atom_call_bf16(
    %d: !fly.memref<f32, register, 8:1>,
    %a: !fly.memref<bf16, register, 16:1>,
    %b: !fly.memref<bf16, register, 16:1>,
    %c: !fly.memref<f32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>
  // Loads land directly in the i16 representation expected by the WMMA intrinsic
  // (the bf16->i16 reinterpretation happens at type-conversion time, not via a
  // separate llvm.bitcast like the SSA path below).
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<16xi16>
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<16xi16>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xf32>
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x16.bf16 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]]
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<8xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, 8:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_gfx11_wmma_gemm_from_tiled_mma_arg
// CHECK: rocdl.wmma.f32.16x16x16.bf16
func.func @test_gfx11_wmma_gemm_from_tiled_mma_arg(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, <(2,4,1):(4,1,0)>>,
    %d: !fly.memref<f32, register, 8:1>,
    %a: !fly.memref<bf16, register, 16:1>,
    %b: !fly.memref<bf16, register, 16:1>,
    %c: !fly.memref<f32, register, 8:1>) {
  fly.gemm(%tiled_mma, %d, %a, %b, %c) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, <(2,4,1):(4,1,0)>>, !fly.memref<f32, register, 8:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_gfx11_wmma_atom_call_ssa_bf16
// CHECK-SAME: (%[[A:.*]]: vector<16xbf16>, %[[B:.*]]: vector<16xbf16>, %[[C:.*]]: vector<8xf32>)
func.func @test_gfx11_wmma_atom_call_ssa_bf16(
    %a: vector<16xbf16>,
    %b: vector<16xbf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_CAST:.*]] = llvm.bitcast %[[A]] : vector<16xbf16> to vector<16xi16>
  // CHECK: %[[B_CAST:.*]] = llvm.bitcast %[[B]] : vector<16xbf16> to vector<16xi16>
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x16.bf16 %[[A_CAST]], %[[B_CAST]], %[[C]]
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, vector<16xbf16>, vector<16xbf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// CHECK-LABEL: @test_gfx11_wmma_atom_call_ssa_f16
// CHECK-SAME: (%[[A:.*]]: vector<16xf16>, %[[B:.*]]: vector<16xf16>, %[[C:.*]]: vector<8xf32>)
func.func @test_gfx11_wmma_atom_call_ssa_f16(
    %a: vector<16xf16>,
    %b: vector<16xf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x16.f16 %[[A]], %[[B]], %[[C]]
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>, vector<16xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}

// Unsigned i8 inputs (signA=signB=false, clamp=false) lower to
// rocdl.wmma.i32.16x16x16.iu8 with the corresponding attrs. A/B operands
// (vector<16xui8>) are bitcast to the packed representation (vector<4xi32>)
// expected by the intrinsic.
//
// CHECK-LABEL: @test_gfx11_wmma_atom_call_ssa_iu8_unsigned
func.func @test_gfx11_wmma_atom_call_ssa_iu8_unsigned(
    %a: vector<16xui8>,
    %b: vector<16xui8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (ui8, ui8) -> i32, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<16xui8> to vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<16xui8> to vector<4xi32>
  // CHECK: rocdl.wmma.i32.16x16x16.iu8
  // signA=signB=clamp=false attrs are elided by the printer when at default.
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (ui8, ui8) -> i32, signA = false, signB = false, clamp = false>>, vector<16xui8>, vector<16xui8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// Mixed-sign i8 (signed A x unsigned B, no clamp) — the three knobs are
// independent type params, so any combination must round-trip and forward to
// the iu8 intrinsic. The printer elides false-valued attrs, so we only assert
// the non-default signA = true here.
//
// CHECK-LABEL: @test_gfx11_wmma_atom_call_ssa_iu8_mixed_sign
func.func @test_gfx11_wmma_atom_call_ssa_iu8_mixed_sign(
    %a: vector<16xi8>,
    %b: vector<16xi8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (i8, i8) -> i32, signA = true, signB = false, clamp = false>>
  // CHECK: rocdl.wmma.i32.16x16x16.iu8 {{.*}} {signA = true}
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (i8, i8) -> i32, signA = true, signB = false, clamp = false>>, vector<16xi8>, vector<16xi8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// Signed i8 inputs use signA=signB=true so the intrinsic treats the packed
// bytes as signed. clamp=true requests saturation on the i32 accumulator
// path. This exercises the dev-suggested type-parameter knobs end-to-end.
//
// CHECK-LABEL: @test_gfx11_wmma_atom_call_ssa_iu8_signed_clamp
func.func @test_gfx11_wmma_atom_call_ssa_iu8_signed_clamp(
    %a: vector<16xsi8>,
    %b: vector<16xsi8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (si8, si8) -> i32, signA = true, signB = true, clamp = true>>
  // CHECK: llvm.bitcast {{.*}} : vector<16xsi8> to vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<16xsi8> to vector<4xi32>
  // CHECK: rocdl.wmma.i32.16x16x16.iu8 {{.*}} {clamp = true, signA = true, signB = true}
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (si8, si8) -> i32, signA = true, signB = true, clamp = true>>, vector<16xsi8>, vector<16xsi8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}
