// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --reconcile-unrealized-casts | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @loop_carried
    gpu.func @loop_carried(%buf: !fly.ptr<i8, global>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // CHECK-NOT: fly_ktrace.token
      // CHECK-NOT: unrealized_conversion_cast

      // CHECK: %[[SEED:.*]] = llvm.mlir.constant(-1 : i32)
      %init = fly_ktrace.inactive_token -> !fly_ktrace.token

      // CHECK: scf.for {{.*}} iter_args(%[[TOK:.*]] = %[[SEED]]) -> (i32)
      %out = scf.for %i = %c0 to %c4 step %c1
             iter_args(%tok = %init) -> (!fly_ktrace.token) {
        // CHECK: llvm.mlir.constant(28 : i32)
        // CHECK: llvm.store %[[TOK]]
        fly_ktrace.range_stop %tok, "phase" : !fly_ktrace.token
        %next = fly_ktrace.range_start "phase" -> !fly_ktrace.token
        scf.yield %next : !fly_ktrace.token
      }
      fly_ktrace.range_stop %out, "phase" : !fly_ktrace.token
      gpu.return
    }
  }
}
