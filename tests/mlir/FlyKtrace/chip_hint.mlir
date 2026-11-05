// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --split-input-file | FileCheck %s

// CHECK-LABEL: gpu.func @no_target_but_hinted
// CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
module attributes {fly_ktrace.chip = "gfx950", gpu.container_module} {
  gpu.module @kernels {
    gpu.func @no_target_but_hinted(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "m"
      gpu.return
    }
  }
}

// -----

// CHECK-LABEL: gpu.func @target_wins
module attributes {fly_ktrace.chip = "gfx950", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @target_wins(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "m"
      gpu.return
    }
  }
}
