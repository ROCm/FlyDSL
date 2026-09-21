// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --reconcile-unrealized-casts | FileCheck %s

// Choosing between two open ranges is legal control flow, and canonicalize rewrites
// the `scf.if` that expresses it into an arith.select. The SCF structural patterns do
// not cover select, so the token reached the expansion still typed and tripped an
// assertion -- a compiler abort on a kernel the frontend is happy to emit.

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @select_between_tokens
    gpu.func @select_between_tokens(%c: i1, %buf: !fly.ptr<i8, global>) kernel {
      // CHECK-NOT: fly_ktrace.token
      // CHECK-NOT: unrealized_conversion_cast
      %a = fly_ktrace.range_start "a" -> !fly_ktrace.token
      %b = fly_ktrace.range_start "b" -> !fly_ktrace.token

      // The select carries the slot index, not the token type.
      // CHECK: arith.select {{.*}} : i32
      %sel = arith.select %c, %a, %b : !fly_ktrace.token

      // And the chosen slot is what the end pairs on, stored at byte 28.
      // CHECK: llvm.mlir.constant(28 : i32)
      // CHECK: llvm.store
      fly_ktrace.range_end %sel, "a" : !fly_ktrace.token
      gpu.return
    }
  }
}
