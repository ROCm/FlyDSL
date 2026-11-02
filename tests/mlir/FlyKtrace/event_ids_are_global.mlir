// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s

// CHECK: fly_ktrace.event_names = {epilogue = 2 : i32, mainloop = 1 : i32}

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @first
    gpu.func @first(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "mainloop"
      gpu.return
    }

    // CHECK-LABEL: gpu.func @second
    // CHECK: llvm.mlir.constant(2 : i32)
    gpu.func @second(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "epilogue"
      gpu.return
    }
  }
}
