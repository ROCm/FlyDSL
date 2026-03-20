// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s

// Pointer operation lowering tests:
//   - fly.get_iter -> identity (memref ptr passthrough)
//   - fly.add_offset -> llvm.getelementptr
//   - fly.make_view -> identity/bitcast

// -----

// === GetIter (identity) ===

// get_iter extracts a raw pointer from a memref, then add_offset advances it.
// After lowering, get_iter becomes identity and add_offset becomes GEP.

// CHECK-LABEL: @test_get_iter_global
// CHECK-SAME: (%[[MEM:.*]]: !llvm.ptr<1>)
func.func @test_get_iter_global(%mem: !fly.memref<f32, global, 32:1>) {
  // CHECK-NOT: fly.get_iter
  %iter = fly.get_iter(%mem) : (!fly.memref<f32, global, 32:1>) -> !fly.ptr<f32, global>
  %offset = fly.make_int_tuple() : () -> !fly.int_tuple<8>
  // CHECK: llvm.getelementptr %[[MEM]][{{.*}}] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  %result = fly.add_offset(%iter, %offset) : (!fly.ptr<f32, global>, !fly.int_tuple<8>) -> !fly.ptr<f32, global>
  return
}

// CHECK-LABEL: @test_get_iter_shared
// CHECK-SAME: (%[[MEM:.*]]: !llvm.ptr<3>)
func.func @test_get_iter_shared(%mem: !fly.memref<f32, shared, 16:1>) {
  // CHECK-NOT: fly.get_iter
  %iter = fly.get_iter(%mem) : (!fly.memref<f32, shared, 16:1>) -> !fly.ptr<f32, shared>
  %offset = fly.make_int_tuple() : () -> !fly.int_tuple<4>
  // CHECK: llvm.getelementptr %[[MEM]][{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
  %result = fly.add_offset(%iter, %offset) : (!fly.ptr<f32, shared>, !fly.int_tuple<4>) -> !fly.ptr<f32, shared>
  return
}

// CHECK-LABEL: @test_get_iter_register
// CHECK-SAME: (%[[MEM:.*]]: !llvm.ptr<5>)
func.func @test_get_iter_register(%mem: !fly.memref<f32, register, 4:1>) {
  // CHECK-NOT: fly.get_iter
  %iter = fly.get_iter(%mem) : (!fly.memref<f32, register, 4:1>) -> !fly.ptr<f32, register>
  %offset = fly.make_int_tuple() : () -> !fly.int_tuple<2>
  // CHECK: llvm.getelementptr %[[MEM]][{{.*}}] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
  %result = fly.add_offset(%iter, %offset) : (!fly.ptr<f32, register>, !fly.int_tuple<2>) -> !fly.ptr<f32, register>
  return
}

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

// -----

// === MakeView (identity when address spaces match) ===

// CHECK-LABEL: @test_make_view
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<1>)
func.func @test_make_view(%ptr: !fly.ptr<f32, global>) -> f32 {
  %s = fly.make_int_tuple() : () -> !fly.int_tuple<(4, 8)>
  %d = fly.make_int_tuple() : () -> !fly.int_tuple<(1, 4)>
  %layout = fly.make_layout(%s, %d) : (!fly.int_tuple<(4, 8)>, !fly.int_tuple<(1, 4)>) -> !fly.layout<(4, 8) : (1, 4)>
  // CHECK-NOT: fly.make_view
  %view = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, global>, !fly.layout<(4, 8) : (1, 4)>) -> !fly.memref<f32, global, (4, 8) : (1, 4)>
  %iter = fly.get_iter(%view) : (!fly.memref<f32, global, (4, 8) : (1, 4)>) -> !fly.ptr<f32, global>
  %offset = fly.make_int_tuple() : () -> !fly.int_tuple<7>
  // CHECK: llvm.getelementptr %[[PTR]][{{.*}}] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  %gep = fly.add_offset(%iter, %offset) : (!fly.ptr<f32, global>, !fly.int_tuple<7>) -> !fly.ptr<f32, global>
  // CHECK: %[[VAL:.*]] = llvm.load
  %val = fly.ptr.load(%gep) : (!fly.ptr<f32, global>) -> f32
  // CHECK: return %[[VAL]]
  return %val : f32
}
