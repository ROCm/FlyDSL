// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: { %fly-opt --split-input-file %s 2>&1 || true; } | FileCheck %s

// Verifier diagnostics for the gfx1250 MX-scale WMMA and 2D TDM atom types.

// -----

// CHECK: unsupported MNK for GFX1250 WMMA_Scale: 32x32x64 (expected 16x16x128 or 32x16x128)
func.func @bad_mma_shape(
    %a: !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<32x32x64, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, modC = 0, reuseA = false, reuseB = false>>) {
  return
}

// -----

// The 32x16x128 form is fp4-only.
// CHECK: GFX1250 WMMA_Scale 32x16x128 requires f4E2M1FN A and B
func.func @bad_mma_32x16_fp8(
    %a: !fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<32x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, modC = 0, reuseA = false, reuseB = false>>) {
  return
}

// -----

// CHECK: numWarps must be a positive power of two, got 3
func.func @bad_tdm_warps(
    %a: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 3, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>) {
  return
}

// -----

// CHECK: padInterval and padAmount must both be zero or both non-zero
func.func @bad_tdm_pad(
    %a: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 64, 0, cache = 0, barrier = false, timeout = false>, 0>) {
  return
}

// -----

// padInterval must be a power of two in elements (48 -> non-power-of-two dword
// interval -> a wrong encoded bitfield). Caught statically by the verifier.
// CHECK: padInterval must be a power of two (in elements), got 48
func.func @bad_tdm_pad_pow2(
    %a: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 48, 8, cache = 0, barrier = false, timeout = false>, 0>) {
  return
}
