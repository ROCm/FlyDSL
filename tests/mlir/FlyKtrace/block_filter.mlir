// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --split-input-file | FileCheck %s

// CHECK-LABEL: gpu.func @no_attr
// CHECK: llvm.atomicrmw add
// CHECK: rocdl.readfirstlane
// CHECK-NOT: scf.if {{.*}} -> (i32)
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @no_attr(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// CHECK-LABEL: gpu.func @all
// CHECK: llvm.atomicrmw add
// CHECK: rocdl.readfirstlane
// CHECK-NOT: scf.if {{.*}} -> (i32)
module attributes {fly_ktrace.blocks = "all", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @all(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// CHECK-LABEL: gpu.func @xcc
// CHECK: llvm.mlir.constant(6164 : i32)
// CHECK: %[[XCC:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.getreg"
// CHECK: %[[WANT:.*]] = llvm.mlir.constant(3 : i32)
// CHECK: %[[REC:.*]] = arith.cmpi eq, %[[XCC]], %[[WANT]]
// CHECK: scf.if %[[REC]] -> (i32) {
// CHECK:   llvm.atomicrmw add
// CHECK: arith.andi {{.*}}, %[[REC]]
module attributes {fly_ktrace.blocks = "xcc:3", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @xcc(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// CHECK-LABEL: gpu.func @xyz
// CHECK: %[[X:.*]] = gpu.block_id  x
// CHECK: arith.index_cast %[[X]]
// CHECK: %[[Y:.*]] = gpu.block_id  y
// CHECK: arith.index_cast %[[Y]]
// CHECK: %[[Z:.*]] = gpu.block_id  z
// CHECK: arith.index_cast %[[Z]]
// CHECK: scf.if {{.*}} -> (i32) {
// CHECK:   llvm.atomicrmw add
module attributes {fly_ktrace.blocks = "1,2,3", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @xyz(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
