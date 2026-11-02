// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

// RUN: %fly-opt %s --convert-fly-ktrace-to-rocdl | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    // CHECK-LABEL: gpu.func @both_sides
    // CHECK-SAME: (%[[COND:[^:]+]]: i1, %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32
    gpu.func @both_sides(%cond: i1, %a: i32, %b: i32, %buf: !fly.ptr<i8, global>) kernel {
      // CHECK: %[[COUNT:.*]] = llvm.alloca {{.*}} x i32

      // The then branch runs first; the lanes that take the else branch get poison they never read.
      // CHECK: %[[FIRST:.*]] = scf.if %[[COND]] -> (i32) {
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK: scf.yield %[[A]] : i32
      // CHECK-NEXT: } else {
      // CHECK-NEXT: %[[UNREAD:.*]] = ub.poison : i32
      // CHECK-NEXT: scf.yield %[[UNREAD]] : i32
      // CHECK-NEXT: }

      // The wave agrees on its count before the else branch numbers its events.
      // CHECK-NEXT: %[[SEEN:.*]] = llvm.load %[[COUNT]]
      // CHECK-NEXT: llvm.mlir.constant(0 : i32)
      // CHECK-NEXT: %[[MAX:.*]] = llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"(%[[SEEN]]
      // CHECK-NEXT: %[[AGREED:.*]] = rocdl.readfirstlane %[[MAX]]
      // CHECK-NEXT: llvm.store %[[AGREED]], %[[COUNT]]

      // CHECK-NEXT: %[[SECOND:.*]] = scf.if %[[COND]] -> (i32) {
      // CHECK-NEXT: scf.yield %[[FIRST]] : i32
      // CHECK-NEXT: } else {
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK: scf.yield %[[B]] : i32
      // CHECK-NEXT: }
      // CHECK-NEXT: llvm.load %[[COUNT]]
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"
      // CHECK: llvm.store {{.*}}, %[[COUNT]]
      // CHECK-NEXT: arith.addi %[[SECOND]], %[[SECOND]]
      %r = scf.if %cond -> (i32) {
        fly_ktrace.mark "upper"
        scf.yield %a : i32
      } else {
        fly_ktrace.mark "lower"
        scf.yield %b : i32
      }
      %sum = arith.addi %r, %r : i32
      %wide = arith.extui %sum : i32 to i64
      fly_ktrace.mark "after" payload %wide : i64
      gpu.return
    }

    // Results without a constant, such as the pointer each branch of a tensor-valued if
    // computes, are split the same way.
    // CHECK-LABEL: gpu.func @pointer_results
    // CHECK-SAME: (%[[COND:[^:]+]]: i1, %[[A:[^:]+]]: !fly.ptr<f32, global>, %[[B:[^:]+]]: !fly.ptr<f32, global>
    gpu.func @pointer_results(%cond: i1, %a: !fly.ptr<f32, global>, %b: !fly.ptr<f32, global>,
                              %buf: !fly.ptr<i8, global>) kernel {
      // CHECK: %[[FIRST:.*]] = scf.if %[[COND]] -> (!fly.ptr<f32, global>) {
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK: scf.yield %[[A]] : !fly.ptr<f32, global>
      // CHECK-NEXT: } else {
      // CHECK-NEXT: %[[UNREAD:.*]] = ub.poison : !fly.ptr<f32, global>
      // CHECK-NEXT: scf.yield %[[UNREAD]] : !fly.ptr<f32, global>
      // CHECK-NEXT: }
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"
      // CHECK: %[[SECOND:.*]] = scf.if %[[COND]] -> (!fly.ptr<f32, global>) {
      // CHECK-NEXT: scf.yield %[[FIRST]] : !fly.ptr<f32, global>
      // CHECK-NEXT: } else {
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK: scf.yield %[[B]] : !fly.ptr<f32, global>
      // CHECK: fly.ptr.load(%[[SECOND]])
      %p = scf.if %cond -> (!fly.ptr<f32, global>) {
        fly_ktrace.mark "upper"
        scf.yield %a : !fly.ptr<f32, global>
      } else {
        fly_ktrace.mark "lower"
        scf.yield %b : !fly.ptr<f32, global>
      }
      %v = fly.ptr.load(%p) : (!fly.ptr<f32, global>) -> f32
      gpu.return
    }

    // A branch that records on one side stays one if, and the wave agrees on its count after it.
    // CHECK-LABEL: gpu.func @one_side
    // CHECK-SAME: (%[[COND:[^:]+]]: i1
    gpu.func @one_side(%cond: i1, %buf: !fly.ptr<i8, global>) kernel {
      // CHECK: %[[COUNT:.*]] = llvm.alloca {{.*}} x i32
      // CHECK: scf.if %[[COND]] {
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK: {{^ *}}}{{$}}
      // CHECK-NEXT: %[[SEEN:.*]] = llvm.load %[[COUNT]]
      // CHECK-NEXT: llvm.mlir.constant(0 : i32)
      // CHECK-NEXT: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"(%[[SEEN]]
      // CHECK-NOT: scf.if
      // CHECK: gpu.return
      scf.if %cond {
        fly_ktrace.mark "upper"
      }
      fly_ktrace.mark "after"
      gpu.return
    }

    // The same holds after a loop that records, whose lanes may leave at different iterations.
    // CHECK-LABEL: gpu.func @loop
    gpu.func @loop(%n: index, %buf: !fly.ptr<i8, global>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // CHECK: %[[COUNT:.*]] = llvm.alloca {{.*}} x i32
      // CHECK: scf.for
      // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"
      // CHECK: {{^ *}}}{{$}}
      // CHECK-NEXT: %[[SEEN:.*]] = llvm.load %[[COUNT]]
      // CHECK-NEXT: llvm.mlir.constant(0 : i32)
      // CHECK-NEXT: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umax"(%[[SEEN]]
      scf.for %i = %c0 to %n step %c1 {
        fly_ktrace.mark "iter"
      }
      gpu.return
    }
  }
}
