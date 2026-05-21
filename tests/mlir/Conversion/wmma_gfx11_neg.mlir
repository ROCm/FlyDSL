// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: { %fly-opt %s 2>&1 || true; } | FileCheck %s

// Verify that signless i8/i4 inputs are rejected with a clear diagnostic that
// explains the unsigned-only contract and points at signA=signB=false.

// CHECK: GFX11 WMMA integer inputs must be unsigned
// CHECK-SAME: ui8/ui4
// CHECK-SAME: signA=signB=false

func.func @test_gfx11_wmma_signless_i8_rejected(
    %a: vector<16xi8>,
    %b: vector<16xi8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (i8, i8) -> i32>>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx11.wmma<16x16x16, (i8, i8) -> i32>>, vector<16xi8>, vector<16xi8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}
