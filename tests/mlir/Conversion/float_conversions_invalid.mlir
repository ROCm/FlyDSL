// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --pass-pipeline="builtin.module(gpu.module(convert-rocdl-math-ops))" --verify-diagnostics

gpu.module @gfx942 [#rocdl.target<chip = "gfx942">] {
  gpu.func @ocp_fp8(%x: f32) -> f8E4M3FN {
    // expected-error @+1 {{'arith.truncf' op from 'f32' to 'f8E4M3FN' is not supported on gfx942}}
    %0 = arith.truncf %x : f32 to f8E4M3FN
    gpu.return %0 : f8E4M3FN
  }
}

// -----

gpu.module @gfx942 [#rocdl.target<chip = "gfx942">] {
  gpu.func @fp8_toward_zero(%x: f32) -> f8E4M3FNUZ {
    // expected-error @+1 {{'arith.truncf' op from 'f32' to 'f8E4M3FNUZ' rounding toward_zero is not supported on gfx942}}
    %0 = arith.truncf %x toward_zero : f32 to f8E4M3FNUZ
    gpu.return %0 : f8E4M3FNUZ
  }
}

// -----

gpu.module @gfx1151 [#rocdl.target<chip = "gfx1151">] {
  gpu.func @fp8(%x: f8E4M3FN) -> f32 {
    // expected-error @+1 {{'arith.extf' op from 'f8E4M3FN' to 'f32' is not supported on gfx1151}}
    %0 = arith.extf %x : f8E4M3FN to f32
    gpu.return %0 : f32
  }
}
