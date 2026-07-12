// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: { %fly-opt --split-input-file --convert-fly-to-rocdl %s 2>&1 || true; } | FileCheck %s

// Lowering-time TDM padding validation. padInterval passes the (element-space)
// power-of-two verifier, but the dword interval / bitfield range can only be
// checked once the element type is known (from the memref). These must fail
// loudly instead of silently emitting a wrong TDM descriptor.

// -----

// pad = 2 elements at 16-bit => interval_dw = 1 => encoded_interval = log2(1)-1
// = -1, out of the [24:22] field. (padInterval=2 is a power of two, so it passes
// the type verifier and is only rejected here.)
// CHECK: gfx1250 TDM: padding (interval=2, amount=8 elements at 16-bit) is not encodable
func.func @bad_tdm_pad_interval_dw(
    %src: !fly.memref<f16, global, (128,64):(64,1)>,
    %dst: !fly.memref<f16, shared, (128,64):(64,1)>) {
  %atom = fly.make_copy_atom {valBits = 0 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 2, 8, cache = 0>, 0>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_2d<warps = 1, pad = 2, 8, cache = 0>, 0>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>) -> ()
  return
}
