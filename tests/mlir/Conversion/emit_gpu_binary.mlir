// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// When fly-opt was built with in-process LLD, prove the pass still emits a
// fatbin even if every ROCm toolkit env var points at a missing path.  Builds
// without FLYDSL_HAS_LLD_LIBRARY fall back to upstream format=fatbin and need a
// reachable toolkit, so the env override is applied only when the binary
// contains the in-process error string (compiled in only with LLD).
// RUN: bash -c 'F="%fly-opt"; S="%s"; if grep -aql "in-process lld failed" "$F"; then env ROCM_PATH=/nonexistent-rocm ROCM_ROOT=/nonexistent-rocm ROCM_HOME=/nonexistent-rocm "$F" "$S" --fly-emit-gpu-binary; else "$F" "$S" --fly-emit-gpu-binary; fi' | %FileCheck %s

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
