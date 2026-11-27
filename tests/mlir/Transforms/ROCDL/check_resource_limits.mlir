// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --split-input-file --pass-pipeline="builtin.module(gpu.module(fly-check-resource-limits))" --verify-diagnostics

// Shared memory is never freed, so its allocations add up wherever they are and exceed
// gfx942's 64 KiB; padding between them is not counted.
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // expected-error @below {{kernel 'too_big' uses 65537 bytes of shared memory, exceeding the limit of 65536 of target #rocdl.target<chip = "gfx942">}}
    gpu.func @too_big(%cond: i1) kernel {
      scf.if %cond {
        %a = fly.make_ptr() {dictAttrs = {allocBytes = 1 : i64, allocAlign = 1 : i64}} : () -> !fly.ptr<i8, shared>
      }
      %b = fly.make_ptr() {dictAttrs = {allocBytes = 65536 : i64, allocAlign = 16 : i64}} : () -> !fly.ptr<i8, shared>
      gpu.return
    }
  }
}

// -----

// Each target is checked against its own limit; a target whose LDS size LLVM does not state
// exactly (unknown chip, or no size feature) is not checked.
module attributes {gpu.container_module} {
  gpu.module @fits_gfx950 [#rocdl.target<chip = "gfx950">, #rocdl.target<chip = "gfx9999">,
                           #rocdl.target<chip = "gfx942", features = "-addressablelocalmemorysize65536">] {
    gpu.func @k() kernel {
      %p = fly.make_ptr() {dictAttrs = {allocBytes = 163840 : i64, allocAlign = 16 : i64}} : () -> !fly.ptr<i8, shared>
      gpu.return
    }
  }
  gpu.module @fat [#rocdl.target<chip = "gfx950">, #rocdl.target<chip = "gfx942">] {
    // expected-error @below {{exceeding the limit of 65536 of target #rocdl.target<chip = "gfx942">}}
    gpu.func @k() kernel {
      %p = fly.make_ptr() {dictAttrs = {allocBytes = 100000 : i64, allocAlign = 16 : i64}} : () -> !fly.ptr<i8, shared>
      gpu.return
    }
  }
}
