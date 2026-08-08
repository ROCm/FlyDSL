// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: env ROCM_PATH=/nonexistent-rocm ROCM_ROOT=/nonexistent-rocm ROCM_HOME=/nonexistent-rocm %fly-opt %s --fly-emit-gpu-binary | FileCheck %s

// fly-emit-gpu-binary links the HSA code object through the in-process LLD
// library, so it must succeed even when no ROCm toolkit is reachable.  The
// upstream gpu-module-to-binary pass fails here with "lld invocation failed"
// because it spawns <toolkit>/llvm/bin/ld.lld.

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
