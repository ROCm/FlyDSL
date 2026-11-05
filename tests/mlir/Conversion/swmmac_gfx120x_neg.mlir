// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: { %fly-opt --split-input-file --convert-fly-to-rocdl %s 2>&1 || true; } | FileCheck %s

// Verifier and lowering diagnostics for the gfx120x (RDNA4) SWMMAC atom type.

// -----

// CHECK: GFX120X SWMMAC requires M=N=16, got 32x16
func.func @bad_shape_mn(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<32x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// Floating-point RDNA4 SWMMAC forms require K=32 (16x16x16 is dense WMMA, and
// 16x16x64 FP16/BF16 SWMMAC is gfx1250-only).
// CHECK: GFX120X SWMMAC floating-point forms require M=N=16, K=32, got 16x16x16
func.func @bad_fp_k16(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// CHECK: GFX120X SWMMAC floating-point forms require M=N=16, K=32, got 16x16x64
func.func @bad_fp_k64(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x64, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// RDNA4 has no mixed F16/BF16 SWMMAC instruction.
// CHECK: unsupported GFX120X SWMMAC configuration: 16x16x32 with A='f16', B='bf16'
func.func @bad_mixed_16bit(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, bf16) -> f32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// Same-type 16-bit accumulators are not exposed by GFX120X SWMMAC.
// CHECK: unsupported GFX120X SWMMAC configuration: 16x16x32 with A='f16', B='f16', Acc='f16'
func.func @bad_f16_accumulator(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f16, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// Floating-point SWMMAC intrinsics take no sign/clamp operands.
// CHECK: GFX120X SWMMAC floating-point path does not accept signA/signB/clamp
func.func @bad_fp_sign_clamp(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> f32, signA = true, signB = false, clamp = false>>) {
  return
}

// -----

// CHECK: GFX120X SWMMAC IU8 requires K=32, got 16x16x16
func.func @bad_iu8_k16(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x16, (i8, i8) -> i32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// CHECK: GFX120X SWMMAC IU8 requires K=32, got 16x16x64
func.func @bad_iu8_k64(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x64, (i8, i8) -> i32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// CHECK: GFX120X SWMMAC IU4 requires K=32 or K=64, got 16x16x16
func.func @bad_iu4_k16(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x16, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// Integer intrinsics require a signless i32 accumulator.
// CHECK: unsupported GFX120X SWMMAC configuration
func.func @bad_iu4_acc_si32(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i4, i4) -> si32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// Integer intrinsics require a signless i32 accumulator.
// CHECK: unsupported GFX120X SWMMAC configuration
func.func @bad_iu4_acc_ui32(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (i4, i4) -> ui32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// Missing sparse_index in A operand group must fail lowering cleanly.
// CHECK: GFX120X SWMMAC expects [a, sparse_index] for A and [b] for B
func.func @bad_missing_sparse_index(
    %a: vector<8xf16>,
    %b: vector<16xf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>, vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}
