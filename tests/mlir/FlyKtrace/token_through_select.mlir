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

      // A lane-varying select still records one start slot: the latest any active lane holds.
      // CHECK: %[[SEL:.*]] = arith.select %arg0, %{{.*}}, %{{.*}} : i32
      %sel = arith.select %c, %a, %b : !fly_ktrace.token

      // CHECK: %[[LATEST:.*]] = llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"(%[[SEL]]
      // CHECK-NEXT: rocdl.readfirstlane %[[LATEST]] : i32
      // CHECK: "s_buffer_store_dwordx4 $0, $1, $2"
      fly_ktrace.range_stop %sel, "a" : !fly_ktrace.token
      gpu.return
    }
  }
}
