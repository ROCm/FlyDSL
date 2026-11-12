// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

// A fragment covers whole tiles of the tiled MMA (32x16x16 here): an extent that is not a multiple
// of the tile would make it, and the copies through it, reach past the view. Two atoms along M
// make the M tile 32, so each check is against its own mode's tile.

func.func @k_tail(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(2,1,1):(1,0,0)>>,
    %ptr: !fly.ptr<f16, shared>) {
  %layout = fly.static : !fly.layout<(16,24):(1,16)>
  %input = fly.make_view(%ptr, %layout) : (!fly.ptr<f16, shared>, !fly.layout<(16,24):(1,16)>) -> !fly.memref<f16, shared, (16,24):(1,16)>
  // expected-error @+2 {{MmaMakeFragmentOp: the K extent 24 is not a multiple of the tiled MMA's K tile 16}}
  // expected-error @+1 {{failed to infer returned types}}
  %frag = fly.mma.make_fragment(b, %tiled_mma, %input) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(2,1,1):(1,0,0)>>, !fly.memref<f16, shared, (16,24):(1,16)>) -> !fly.memref<f16, register, (4,1,2):(1,0,4)>
  return
}

// -----

func.func @m_tail(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(2,1,1):(1,0,0)>>,
    %ptr: !fly.ptr<f16, shared>) {
  %layout = fly.static : !fly.layout<(48,16):(1,48)>
  %input = fly.make_view(%ptr, %layout) : (!fly.ptr<f16, shared>, !fly.layout<(48,16):(1,48)>) -> !fly.memref<f16, shared, (48,16):(1,48)>
  // expected-error @+2 {{MmaMakeFragmentOp: the M extent 48 is not a multiple of the tiled MMA's M tile 32}}
  // expected-error @+1 {{failed to infer returned types}}
  %frag = fly.mma.make_fragment(a, %tiled_mma, %input) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(2,1,1):(1,0,0)>>, !fly.memref<f16, shared, (48,16):(1,48)>) -> !fly.memref<f16, register, (4,2,1):(1,4,0)>
  return
}

// -----

func.func @n_tail(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(2,1,1):(1,0,0)>>,
    %ptr: !fly.ptr<f32, shared>) {
  %layout = fly.static : !fly.layout<(32,8):(1,32)>
  %input = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, shared>, !fly.layout<(32,8):(1,32)>) -> !fly.memref<f32, shared, (32,8):(1,32)>
  // expected-error @+2 {{MmaMakeFragmentOp: the N extent 8 is not a multiple of the tiled MMA's N tile 16}}
  // expected-error @+1 {{failed to infer returned types}}
  %frag = fly.mma.make_fragment(c, %tiled_mma, %input) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(2,1,1):(1,0,0)>>, !fly.memref<f32, shared, (32,8):(1,32)>) -> !fly.memref<f32, register, (4,1,1):(1,0,0)>
  return
}

// -----

// A permutation sets the tile: M 64 here, so 32 rows are a partial tile.
func.func @permuted_m_tail(
    %tiled_mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(1,1,1):(0,1,2)>, !fly.tile<[64|16|16]>>,
    %ptr: !fly.ptr<f16, shared>) {
  %layout = fly.static : !fly.layout<(32,16):(1,32)>
  %input = fly.make_view(%ptr, %layout) : (!fly.ptr<f16, shared>, !fly.layout<(32,16):(1,32)>) -> !fly.memref<f16, shared, (32,16):(1,32)>
  // expected-error @+2 {{MmaMakeFragmentOp: the M extent 32 is not a multiple of the tiled MMA's M tile 64}}
  // expected-error @+1 {{failed to infer returned types}}
  %frag = fly.mma.make_fragment(a, %tiled_mma, %input) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, <(1,1,1):(0,1,2)>, !fly.tile<[64|16|16]>>, !fly.memref<f16, shared, (32,16):(1,32)>) -> !fly.memref<f16, register, (4,2,1):(1,4,0)>
  return
}
