// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @probe
    gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
      // CHECK-NOT: !fly_ktrace.token

      // CHECK: rocdl.readfirstlane
      // CHECK: %[[SLOT:.*]] = scf.if {{.*}} -> (i32)

      // CHECK: llvm.mlir.constant(50331649 : i32)

      // CHECK: llvm.mlir.constant(-1 : i32)
      // CHECK: llvm.mlir.constant(-1 : i32)
      %t = fly_ktrace.range_start "phase" -> !fly_ktrace.token

      // CHECK: llvm.mlir.constant(67108865 : i32)

      // CHECK: %[[OFF:.*]] = llvm.mlir.constant(28 : i32)
      // CHECK: %[[GEP:.*]] = llvm.getelementptr {{.*}}[%[[OFF]]]
      // CHECK: llvm.store %[[SLOT]], %[[GEP]] : i32, !llvm.ptr<1>
      fly_ktrace.range_end %t, "phase" : !fly_ktrace.token
      gpu.return
    }
  }
}
