// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-emit-gpu-binary | FileCheck %s

// fly-emit-gpu-binary wraps the upstream gpu-module-to-binary pass.  When
// built with in-process LLD (FLYDSL_HAS_LLD_LIBRARY), it links HSA code
// objects without spawning ld.lld; otherwise it falls back to the upstream
// fatbin path which requires a reachable ROCm toolkit.

// CHECK: gpu.binary @kernels
// CHECK-SAME: #rocdl.target<chip = "gfx942">
// CHECK-SAME: "\7FELF
// CHECK-NOT: gpu.module @kernels
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    llvm.func @empty_kernel() attributes {rocdl.kernel} {
      llvm.return
    }
  }
}
