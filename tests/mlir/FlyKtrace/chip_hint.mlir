// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --split-input-file | FileCheck %s

// An extern-linked kernel has its gpu.module targets stripped before compile --
// rocdl-attach-target is then the sole source of them, and it runs after this pass.
// The arch gate would have no target to read, so the frontend publishes the chip it
// already knows; without this, enabling tracing made such a kernel fail to compile.

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

// A target on the module still wins: the hint is only consulted when there is none,
// so a gfx1201 target is rejected even though the hint says otherwise.
// CHECK-LABEL: gpu.func @target_wins
module attributes {fly_ktrace.chip = "gfx950", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @target_wins(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "m"
      gpu.return
    }
  }
}
