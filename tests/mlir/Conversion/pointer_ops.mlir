// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// Pointer operation lowering tests:
//   - fly.add_offset -> llvm.getelementptr
//   - fly.make_view -> identity/bitcast

// -----

// === AddOffset ===

// CHECK-LABEL: @test_add_offset_static
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<1>)
func.func @test_add_offset_static(%ptr: !fly.ptr<f32, global>) {
  %offset = fly.make_int_tuple() : () -> !fly.int_tuple<4>
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: llvm.getelementptr %[[PTR]][%[[C4]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  %result = fly.add_offset(%ptr, %offset) : (!fly.ptr<f32, global>, !fly.int_tuple<4>) -> !fly.ptr<f32, global>
  return
}

// CHECK-LABEL: @test_add_offset_dynamic
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<1>, %[[OFF:.*]]: i32)
func.func @test_add_offset_dynamic(%ptr: !fly.ptr<f32, global>, %off: i32) {
  %offset = fly.make_int_tuple(%off) : (i32) -> !fly.int_tuple<?>
  // CHECK: llvm.getelementptr %[[PTR]][%[[OFF]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  %result = fly.add_offset(%ptr, %offset) : (!fly.ptr<f32, global>, !fly.int_tuple<?>) -> !fly.ptr<f32, global>
  return
}

// -----

// === GetDynShared ===

// get_dyn_shared returns a pointer to dynamic shared memory.
// After lowering, it creates a global [0 x i8] addrspace(3) and returns its address.

// CHECK: llvm.mlir.global external @__dynamic_shared__
// CHECK-SAME: {addr_space = 3 : i32, alignment = 1024 : i64, dso_local} : !llvm.array<0 x i8>
// CHECK-LABEL: gpu.func @test_get_dyn_shared
gpu.module @dyn_shared_module {
  gpu.func @test_get_dyn_shared() kernel {
    // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @__dynamic_shared__
    // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[ADDR]][0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    // CHECK-NOT: fly.get_dyn_shared
    %ptr = fly.get_dyn_shared() : !fly.ptr<i8, shared, align<1024>>
    gpu.return
  }
}
