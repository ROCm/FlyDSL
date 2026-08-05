// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: { %fly-opt --split-input-file %s 2>&1 || true; } | FileCheck %s

// Verifier diagnostics for the gfx120x (RDNA4) WMMA atom type.

// -----

// RDNA4 only has the 16x16x16 WMMA shapes; 16x16x32 is a gfx1250 instruction
// and must not be silently accepted here, or codegen would emit an instruction
// the target cannot select.
// CHECK: GFX120X WMMA requires M=N=K=16, got 16x16x32
func.func @bad_shape_k32(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// Only the fp16/bf16 -> f32 forms are wired up so far.
// CHECK: unsupported GFX120X WMMA configuration: 16x16x16 with A='f8E4M3FN'
func.func @bad_elem_ty_fp8(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>) {
  return
}

// -----

// The fp WMMA intrinsics take no sign/clamp operands, so an atom that promises
// them would silently drop the request at codegen time.
// CHECK: GFX120X WMMA fp16/bf16 path does not accept signA/signB/clamp
func.func @bad_sign_clamp(
    %a: !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = true, signB = true, clamp = true>>) {
  return
}
