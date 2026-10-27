// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --reconcile-unrealized-casts | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @select_between_tokens
    gpu.func @select_between_tokens(%c: i1, %buf: !fly.ptr<i8, global>) kernel {
      // CHECK-NOT: fly_ktrace.token
      // CHECK-NOT: unrealized_conversion_cast
      %a = fly_ktrace.range_start "a" -> !fly_ktrace.token
      %b = fly_ktrace.range_start "b" -> !fly_ktrace.token

      // CHECK: arith.select {{.*}} : i32
      %sel = arith.select %c, %a, %b : !fly_ktrace.token

      // CHECK: llvm.mlir.constant(28 : i32)
      // CHECK: llvm.store
      fly_ktrace.range_stop %sel, "a" : !fly_ktrace.token
      gpu.return
    }
  }
}
