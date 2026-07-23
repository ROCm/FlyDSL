// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: { %fly-opt --split-input-file %s 2>&1 || true; } | FileCheck %s

// Verifier diagnostics for the gfx1250 MX-scale WMMA and 2D TDM atom types.

// -----

// CHECK: unsupported MNK for GFX1250 WMMA_Scale: 32x32x64 (expected 16x16x128 or 32x16x128)
func.func @bad_mma_shape(
    %a: !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<32x32x64, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, modC = 0, reuseA = false, reuseB = false, blockSize = 32>>) {
  return
}

// -----

// The 32x16x128 form is fp4-only.
// CHECK: GFX1250 WMMA_Scale 32x16x128 requires f4E2M1FN A and B
func.func @bad_mma_32x16_fp8(
    %a: !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<32x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, modC = 0, reuseA = false, reuseB = false, blockSize = 32>>) {
  return
}

// -----

// blockSize must be 16 or 32 (elements per shared E8M0 scale).
// CHECK: blockSize must be 16 or 32, got 8
func.func @bad_mma_blocksize(
    %a: !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, modC = 0, reuseA = false, reuseB = false, blockSize = 8>>) {
  return
}

// -----

// CHECK: numWarps must be a positive power of two, got 3
func.func @bad_tdm_warps(
    %a: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 3, pad = 0, 0, cache = 0, barrier = false, timeout = false, iwidth = 0>, 0>) {
  return
}

// -----

// CHECK: padInterval and padAmount must both be zero or both non-zero
func.func @bad_tdm_pad(
    %a: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 64, 0, cache = 0, barrier = false, timeout = false, iwidth = 0>, 0>) {
  return
}

// -----

// padInterval must be a power of two in elements (48 -> non-power-of-two dword
// interval -> a wrong encoded bitfield). Caught statically by the verifier.
// CHECK: padInterval must be a power of two (in elements), got 48
func.func @bad_tdm_pad_pow2(
    %a: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 48, 8, cache = 0, barrier = false, timeout = false, iwidth = 0>, 0>) {
  return
}

// -----

// signA/signB/clamp are integer-only (iu4/iu8) controls; the verifier rejects
// them on the float WMMA paths.
// CHECK: signA/signB/clamp are only valid for integer (iu4/iu8) WMMA
func.func @bad_wmma_sign_on_fp(
    %a: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (f16, f16) -> f32, signA = true, signB = false, clamp = false>>) {
  return
}
