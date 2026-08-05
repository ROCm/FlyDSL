// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// Lowered address arithmetic of a nested add_offset chain, i.e. the shape the AMDGPU
// backend gets to see.
//
// An LDS access keeps its `ds_read ... offset:` immediate only while the compile-time part
// of the address is still a constant getelementptr index: `gep(gep(@lds, %runtime), C)`
// folds C into the instruction and lets every access share one base register, whereas
// `gep(@lds, add(%runtime, C))` hides C inside a dynamic value and makes each access
// materialize its own base. Layout lowering therefore leaves a mixed offset chain on a
// shared pointer nested, and keeps merging offsets everywhere else.
//
// These check the address arithmetic only. tests/kernels/test_lds_offset_chain.py runs both
// mixed orderings on device and checks the values they read against a reference.

// === Shared: the constant stays outside the runtime index ===

// CHECK-LABEL: @test_shared_const_outside_runtime_index
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<3>, %[[OFF:.*]]: i32)
func.func @test_shared_const_outside_runtime_index(%ptr: !fly.ptr<bf16, shared>, %off: i32) -> bf16 {
  %o1 = fly.make_int_tuple(%off) : (i32) -> !fly.int_tuple<?>
  %o2 = fly.make_int_tuple() : () -> !fly.int_tuple<1024>
  // The runtime offset indexes the base once, the constant is a separate gep index.
  // CHECK-NOT: arith.addi
  // CHECK: %[[BASE:.*]] = llvm.getelementptr %[[PTR]][%[[OFF]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
  // CHECK: %[[C1024:.*]] = arith.constant 1024 : i32
  // CHECK: %[[ADDR:.*]] = llvm.getelementptr %[[BASE]][%[[C1024]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
  // CHECK: llvm.load %[[ADDR]]
  // CHECK-NOT: arith.addi
  %p1 = fly.add_offset(%ptr, %o1) : (!fly.ptr<bf16, shared>, !fly.int_tuple<?>) -> !fly.ptr<bf16, shared>
  %p2 = fly.add_offset(%p1, %o2) : (!fly.ptr<bf16, shared>, !fly.int_tuple<1024>) -> !fly.ptr<bf16, shared>
  %v = fly.ptr.load(%p2) : (!fly.ptr<bf16, shared>) -> bf16
  return %v : bf16
}

// The mirrored chain keeps the constant as its own index too, in the order the author wrote
// it. LLVM folds that inner gep into a constant expression on the LDS symbol, so the reads
// still share one base register and carry their constants in the instruction offset field.
// CHECK-LABEL: @test_shared_const_first
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<3>, %[[OFF:.*]]: i32)
func.func @test_shared_const_first(%ptr: !fly.ptr<bf16, shared>, %off: i32) -> bf16 {
  %o1 = fly.make_int_tuple() : () -> !fly.int_tuple<1024>
  %o2 = fly.make_int_tuple(%off) : (i32) -> !fly.int_tuple<?>
  // CHECK-NOT: arith.addi
  // CHECK: %[[C1024:.*]] = arith.constant 1024 : i32
  // CHECK: %[[BASE:.*]] = llvm.getelementptr %[[PTR]][%[[C1024]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
  // CHECK: %[[ADDR:.*]] = llvm.getelementptr %[[BASE]][%[[OFF]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
  // CHECK: llvm.load %[[ADDR]]
  // CHECK-NOT: arith.addi
  %p1 = fly.add_offset(%ptr, %o1) : (!fly.ptr<bf16, shared>, !fly.int_tuple<1024>) -> !fly.ptr<bf16, shared>
  %p2 = fly.add_offset(%p1, %o2) : (!fly.ptr<bf16, shared>, !fly.int_tuple<?>) -> !fly.ptr<bf16, shared>
  %v = fly.ptr.load(%p2) : (!fly.ptr<bf16, shared>) -> bf16
  return %v : bf16
}

// Two compile-time offsets are still folded into one constant index: nothing runtime is
// involved, so there is no constant to lose.
// CHECK-LABEL: @test_shared_all_const
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<3>)
func.func @test_shared_all_const(%ptr: !fly.ptr<bf16, shared>) -> bf16 {
  %o1 = fly.make_int_tuple() : () -> !fly.int_tuple<16>
  %o2 = fly.make_int_tuple() : () -> !fly.int_tuple<32>
  // CHECK: %[[C48:.*]] = arith.constant 48 : i32
  // CHECK: %[[ADDR:.*]] = llvm.getelementptr %[[PTR]][%[[C48]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
  // CHECK: llvm.load %[[ADDR]]
  // CHECK-NOT: llvm.getelementptr
  %p1 = fly.add_offset(%ptr, %o1) : (!fly.ptr<bf16, shared>, !fly.int_tuple<16>) -> !fly.ptr<bf16, shared>
  %p2 = fly.add_offset(%p1, %o2) : (!fly.ptr<bf16, shared>, !fly.int_tuple<32>) -> !fly.ptr<bf16, shared>
  %v = fly.ptr.load(%p2) : (!fly.ptr<bf16, shared>) -> bf16
  return %v : bf16
}

// Two runtime offsets are still merged: one add, one index, nothing constant is buried.
// CHECK-LABEL: @test_shared_all_runtime
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<3>, %[[A:.*]]: i32, %[[B:.*]]: i32)
func.func @test_shared_all_runtime(%ptr: !fly.ptr<bf16, shared>, %a: i32, %b: i32) -> bf16 {
  %o1 = fly.make_int_tuple(%a) : (i32) -> !fly.int_tuple<?>
  %o2 = fly.make_int_tuple(%b) : (i32) -> !fly.int_tuple<?>
  // CHECK: %[[SUM:.*]] = arith.addi %[[A]], %[[B]]
  // CHECK: %[[ADDR:.*]] = llvm.getelementptr %[[PTR]][%[[SUM]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, bf16
  // CHECK: llvm.load %[[ADDR]]
  // CHECK-NOT: llvm.getelementptr
  %p1 = fly.add_offset(%ptr, %o1) : (!fly.ptr<bf16, shared>, !fly.int_tuple<?>) -> !fly.ptr<bf16, shared>
  %p2 = fly.add_offset(%p1, %o2) : (!fly.ptr<bf16, shared>, !fly.int_tuple<?>) -> !fly.ptr<bf16, shared>
  %v = fly.ptr.load(%p2) : (!fly.ptr<bf16, shared>) -> bf16
  return %v : bf16
}

// === Global: address formation is unchanged ===

// A global pointer keeps the merged form: the two offsets become one index expression and
// one getelementptr, exactly as before the shared-memory rule was added.
// CHECK-LABEL: @test_global_offsets_stay_merged
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<1>, %[[OFF:.*]]: i32)
func.func @test_global_offsets_stay_merged(%ptr: !fly.ptr<f32, global>, %off: i32) -> f32 {
  %o1 = fly.make_int_tuple(%off) : (i32) -> !fly.int_tuple<?>
  %o2 = fly.make_int_tuple() : () -> !fly.int_tuple<8>
  // CHECK: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: %[[SUM:.*]] = arith.addi %[[OFF]], %[[C8]]
  // CHECK: %[[ADDR:.*]] = llvm.getelementptr %[[PTR]][%[[SUM]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  // CHECK: llvm.load %[[ADDR]]
  // CHECK-NOT: llvm.getelementptr
  %p1 = fly.add_offset(%ptr, %o1) : (!fly.ptr<f32, global>, !fly.int_tuple<?>) -> !fly.ptr<f32, global>
  %p2 = fly.add_offset(%p1, %o2) : (!fly.ptr<f32, global>, !fly.int_tuple<8>) -> !fly.ptr<f32, global>
  %v = fly.ptr.load(%p2) : (!fly.ptr<f32, global>) -> f32
  return %v : f32
}
