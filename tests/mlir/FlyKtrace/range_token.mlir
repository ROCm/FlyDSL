// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl --reconcile-unrealized-casts | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @probe
    gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
      // CHECK-NOT: !fly_ktrace.token
      // CHECK: arith.extui %[[REGION:[0-9]+]] : i32 to i64
      // CHECK: "s_buffer_store_dwordx4 $0, $1, $2", "s,s,s" %{{[0-9]+}}, %[[RSRC:[0-9]+]], %{{[0-9]+}}

      // The token is the slot the start record is stored at: its region slot plus one past
      // the wave's count, the same index that sets the store's offset.
      // CHECK: %[[CLAMPED:.*]] = arith.minui
      // CHECK: %[[IN_REGION:.*]] = arith.addi %[[CLAMPED]], %{{.*}} : i32
      // CHECK: %[[OFFSET:.*]] = arith.muli %[[IN_REGION]], %{{.*}} : i32
      // CHECK: "s_buffer_store_dwordx4 $0, $1, $2", "s,s,s" %{{[0-9]+}}, %[[RSRC]], %[[OFFSET]]
      // CHECK: %[[UPPER:.*]] = arith.addi %[[OFFSET]], %{{.*}} : i32
      // CHECK: llvm.mlir.constant(50331649 : i32)
      // CHECK: "s_buffer_store_dwordx4 $0, $1, $2", "s,s,s" %{{[0-9]+}}, %[[RSRC]], %[[UPPER]]
      // CHECK: %[[TOKEN:.*]] = arith.addi %[[REGION]], %[[IN_REGION]] : i32
      %t = fly_ktrace.range_start "phase" -> !fly_ktrace.token

      // CHECK: %[[LATEST:.*]] = llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"(%[[TOKEN]]
      // CHECK-NEXT: %[[START:.*]] = rocdl.readfirstlane %[[LATEST]] : i32
      // CHECK: llvm.mlir.constant(67108865 : i32)
      // CHECK: %[[LANE3:.*]] = llvm.mlir.constant(3 : i32)
      // CHECK: llvm.insertelement %[[START]], %{{.*}}[%[[LANE3]] : i32] : vector<4xi32>
      // CHECK: "s_buffer_store_dwordx4 $0, $1, $2"
      fly_ktrace.range_stop %t, "phase" : !fly_ktrace.token
      gpu.return
    }
  }
}
