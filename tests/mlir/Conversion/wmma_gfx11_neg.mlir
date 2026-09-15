// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: { %fly-opt %s 2>&1 || true; } | FileCheck %s

// The fp16/bf16 ROCDL WMMA intrinsics on RDNA3 do not have signA/signB/clamp
// operands. The atom type must refuse to be constructed with any of them set
// to true on an fp path, since the codegen could not honor it.

// CHECK: GFX11 WMMA fp16/bf16 path does not accept signA/signB/clamp

func.func @test_gfx11_wmma_bf16_signA_rejected(
    %a: vector<16xbf16>,
    %b: vector<16xbf16>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (bf16, bf16) -> f32, signA = true, signB = false, clamp = false>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (bf16, bf16) -> f32, signA = true, signB = false, clamp = false>>, vector<16xbf16>, vector<16xbf16>, vector<8xf32>) -> vector<8xf32>
  return %res : vector<8xf32>
}
