// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: { %fly-opt --split-input-file %s 2>&1 || true; } | FileCheck %s

// Verifier diagnostics for the gfx120x (RDNA4) WMMA atom type.

// -----

// The supported floating-point RDNA4 WMMA forms have K=16. In particular,
// 16x16x32 BF16 is a gfx1250 instruction and must not be accepted here.
// CHECK: GFX120X WMMA floating-point forms require M=N=K=16, got 16x16x32
func.func @bad_shape_k32(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// RDNA4 has no mixed F16/BF16 floating-point WMMA instruction.
// CHECK: unsupported GFX120X WMMA configuration: 16x16x16 with A='f16', B='bf16'
func.func @bad_elem_ty_mixed_16bit(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (f16, bf16) -> f32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// The fp WMMA intrinsics take no sign/clamp operands, so an atom that promises
// them would silently drop the request at codegen time.
// CHECK: GFX120X WMMA floating-point path does not accept signA/signB/clamp
func.func @bad_sign_clamp(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = true, signB = true, clamp = true>>) {
  return
}
