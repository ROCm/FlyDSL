// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s
// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s --check-prefix=COUNTS

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @probe
    gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
      // CHECK: fly.to_llvm_ptr
      // CHECK: %[[COUNT:.*]] = llvm.alloca {{.*}} x i32 {{.*}} -> !llvm.ptr<5>

      // CHECK: llvm.mlir.constant(63492 : i32)
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.getreg"
      // CHECK: llvm.mlir.constant(6164 : i32)
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.getreg"

      // CHECK: rocdl.ballot
      // CHECK: scf.if {{.*}} -> (i32)
      // CHECK: llvm.atomicrmw add
      // CHECK: rocdl.readfirstlane
      // CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64
      // CHECK: "s_buffer_store_dwordx4 $0, $1, $2", "s,s,s" %{{[0-9]+}}, %[[RSRC:[0-9]+]], %{{[0-9]+}}
      // CHECK: "s_buffer_store_dwordx4 $0, $1, $2", "s,s,s" %{{[0-9]+}}, %[[RSRC]], %{{[0-9]+}}

      // CHECK: llvm.load %[[COUNT]]
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"
      // CHECK: llvm.store {{.*}}, %[[COUNT]]
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK: %[[LIMIT:.*]] = llvm.mlir.constant(255 : i32)
      // CHECK: arith.minui {{.*}}, %[[LIMIT]]
      // CHECK-COUNT-2: "s_buffer_store_dwordx4 $0, $1, $2", "s,s,s" %{{[0-9]+}}, %[[RSRC]], %{{[0-9]+}}

      // The count goes to the header, and the scalar cache is written back before the wave ends.
      // CHECK: llvm.load %[[COUNT]]
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"
      // CHECK: "s_buffer_store_dword $0, $1, $2", "s,s,s" %{{[0-9]+}}, %[[RSRC]], %{{[0-9]+}} : (i32,
      // CHECK-NEXT: llvm.call_intrinsic "llvm.amdgcn.s.dcache.wb"()
      // CHECK-NEXT: gpu.return

      // Only the prologue branches or touches an atomic: the launch count and the wave's claim.
      // COUNTS: scf.if
      // COUNTS: llvm.atomicrmw
      // COUNTS: scf.if
      // COUNTS: llvm.atomicrmw
      // COUNTS-NOT: scf.if
      // COUNTS-NOT: llvm.atomicrmw
      // COUNTS: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // COUNTS-NOT: scf.if
      // COUNTS-NOT: llvm.atomicrmw
      // COUNTS: gpu.return
      fly_ktrace.mark "tick"
      gpu.return
    }
  }
}
