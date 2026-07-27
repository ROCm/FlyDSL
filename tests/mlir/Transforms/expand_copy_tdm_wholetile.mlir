// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering | FileCheck %s

// A whole-tile copy atom (gfx1250 TDM 2D DMA) must NOT be decomposed per element
// by fly-layout-lowering: the tile geometry lives in the rank-2 global memref
// layout that the TDM lowering reads. Expect a single copy_atom_call whose
// global memref stays rank-2 (128,64):(64,1), not a coalesced/rank-1 form.

// CHECK-LABEL: @tdm_wholetile
// CHECK: fly.copy_atom_call
// CHECK-SAME: !fly.memref<f16, global, (128,64):(64,1)>
// CHECK-SAME: !fly.memref<f16, shared, (128,64):(64,1)
// CHECK-NOT: fly.copy_atom_call
func.func @tdm_wholetile(
    %g: !fly.memref<f16, global, (128,64):(64,1)>,
    %s: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom {valBits = 0 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>
  fly.copy(%atom, %g, %s) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}
