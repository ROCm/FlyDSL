// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --split-input-file | FileCheck %s

// Without a filter every wave claims and owns its whole region.
// CHECK-LABEL: gpu.func @no_attr
// CHECK: %[[FIRST:.*]] = llvm.trunc %{{.*}} : i64 to i32
// CHECK-NEXT: %[[LEADER:.*]] = arith.cmpi eq, %{{.*}}, %[[FIRST]] : i32
// CHECK-NEXT: scf.if %[[LEADER]] -> (i32)
// CHECK: %[[SLOTS:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// CHECK: %[[BYTES:.*]] = arith.muli %[[SLOTS]], %{{.*}} : i32
// CHECK-NOT: arith.select
// CHECK: %[[UNIFORM:.*]] = rocdl.readfirstlane %[[BYTES]] : i32
// CHECK: %[[RECORDS:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: llvm.insertelement %[[UNIFORM]], %{{.*}}[%[[RECORDS]] : i32] : vector<4xi32>
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
// CHECK: %[[FIRST:.*]] = llvm.trunc %{{.*}} : i64 to i32
// CHECK-NEXT: %[[LEADER:.*]] = arith.cmpi eq, %{{.*}}, %[[FIRST]] : i32
// CHECK-NEXT: scf.if %[[LEADER]] -> (i32)
// CHECK: %[[SLOTS:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// CHECK: %[[BYTES:.*]] = arith.muli %[[SLOTS]], %{{.*}} : i32
// CHECK-NOT: arith.select
// CHECK: %[[UNIFORM:.*]] = rocdl.readfirstlane %[[BYTES]] : i32
// CHECK: llvm.insertelement %[[UNIFORM]], %{{.*}} : vector<4xi32>
module attributes {fly_ktrace.blocks = "all", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @all(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}

// -----

// A filtered-out wave neither claims nor owns any bytes, so its stores are dropped.
// CHECK-LABEL: gpu.func @xcc
// CHECK: llvm.mlir.constant(6164 : i32)
// CHECK: %[[XCC:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.getreg"
// CHECK: %[[WANT:.*]] = llvm.mlir.constant(3 : i32)
// CHECK: %[[REC:.*]] = arith.cmpi eq, %[[XCC]], %[[WANT]]
// CHECK: %[[FIRST:.*]] = llvm.trunc %{{.*}} : i64 to i32
// CHECK-NEXT: %[[LEADER:.*]] = arith.cmpi eq, %{{.*}}, %[[FIRST]] : i32
// CHECK-NEXT: %[[CLAIMS:.*]] = arith.andi %[[LEADER]], %[[REC]]
// CHECK-NEXT: scf.if %[[CLAIMS]] -> (i32)
// CHECK: llvm.atomicrmw add
// CHECK: %[[BYTES:.*]] = arith.select %[[REC]], %{{.*}}, %{{.*}} : i32
// CHECK: %[[UNIFORM:.*]] = rocdl.readfirstlane %[[BYTES]] : i32
// CHECK: %[[RECORDS:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: llvm.insertelement %[[UNIFORM]], %{{.*}}[%[[RECORDS]] : i32] : vector<4xi32>
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
// CHECK: %[[REC:.*]] = arith.andi
// CHECK: %[[FIRST:.*]] = llvm.trunc %{{.*}} : i64 to i32
// CHECK-NEXT: %[[LEADER:.*]] = arith.cmpi eq, %{{.*}}, %[[FIRST]] : i32
// CHECK-NEXT: %[[CLAIMS:.*]] = arith.andi %[[LEADER]], %[[REC]]
// CHECK-NEXT: scf.if %[[CLAIMS]] -> (i32)
// CHECK: llvm.atomicrmw add
// CHECK: %[[BYTES:.*]] = arith.select %[[REC]], %{{.*}}, %{{.*}} : i32
// CHECK: %[[UNIFORM:.*]] = rocdl.readfirstlane %[[BYTES]] : i32
// CHECK: llvm.insertelement %[[UNIFORM]], %{{.*}} : vector<4xi32>
module attributes {fly_ktrace.blocks = "1,2,3", gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    gpu.func @xyz(%buf: !fly.ptr<i8, global>) kernel {
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
