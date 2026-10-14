// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl --canonicalize | FileCheck %s

// Test the SSA lowering boundary with SSA data and a register memref pred.
// These are direct copy_atom_call_ssa inputs: atom-call conversion is not run.
// Loads preserve the old destination on the false branch; stores/atomics issue
// only in the true branch. No non-register predicate storage is supported.

// CHECK-LABEL: func.func @universal_store(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: llvm.store
// CHECK-NEXT: }
func.func @universal_store(%atom: !fly.copy_atom<!fly.universal_copy<32>, 32>, %dst: !fly.memref<f32, global, 1:1>, %pred: !fly.memref<i1, register, 1:1>, %src: f32) {
  fly.copy_atom_call_ssa(%atom, %src, %dst, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly.universal_copy<32>, 32>, f32, !fly.memref<f32, global, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @buffer_store(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.raw.ptr.buffer.store
// CHECK-NEXT: }
func.func @buffer_store(%atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, %dst: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %pred: !fly.memref<i1, register, 1:1>, %src: f32) {
  fly.copy_atom_call_ssa(%atom, %src, %dst, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, f32, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @universal_atomic(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: llvm.atomicrmw fadd
// CHECK-NEXT: }
func.func @universal_atomic(%atom: !fly.copy_atom<!fly.universal_atomic<#fly<atomic_op add>, f32>, 32>, %dst: !fly.memref<f32, global, 1:1>, %pred: !fly.memref<i1, register, 1:1>, %src: f32) {
  fly.copy_atom_call_ssa(%atom, %src, %dst, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly.universal_atomic<#fly<atomic_op add>, f32>, 32>, f32, !fly.memref<f32, global, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @buffer_atomic(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.raw.ptr.buffer.atomic.fadd
// CHECK-NEXT: }
func.func @buffer_atomic(%atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_atomic<#fly<atomic_op add>, f32>, 32>, %dst: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %pred: !fly.memref<i1, register, 1:1>, %src: f32) {
  fly.copy_atom_call_ssa(%atom, %src, %dst, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_atomic<#fly<atomic_op add>, f32>, 32>, f32, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @universal_load(
// CHECK-SAME: %[[OLD:[a-zA-Z0-9_]+]]: f32
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: %[[RESULT:.*]] = scf.if %[[PRED]] -> (f32) {
// CHECK: llvm.load
// CHECK: scf.yield
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[OLD]] : f32
// CHECK: llvm.store %[[RESULT]],
func.func @universal_load(%atom: !fly.copy_atom<!fly.universal_copy<32>, 32>, %src: !fly.memref<f32, global, 1:1>, %pred: !fly.memref<i1, register, 1:1>, %out: !fly.ptr<f32, global>, %old: f32) {
  %result = fly.copy_atom_call_ssa(%atom, %src, %old, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly.universal_copy<32>, 32>, !fly.memref<f32, global, 1:1>, f32, !fly.memref<i1, register, 1:1>) -> f32
  fly.ptr.store(%result, %out) : (f32, !fly.ptr<f32, global>) -> ()
  return
}

// CHECK-LABEL: func.func @buffer_load(
// CHECK-SAME: %[[OLD:[a-zA-Z0-9_]+]]: f32
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: %[[RESULT:.*]] = scf.if %[[PRED]] -> (f32) {
// CHECK: rocdl.raw.ptr.buffer.load
// CHECK: scf.yield
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[OLD]] : f32
// CHECK: llvm.store %[[RESULT]],
func.func @buffer_load(%atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %pred: !fly.memref<i1, register, 1:1>, %out: !fly.ptr<f32, global>, %old: f32) {
  %result = fly.copy_atom_call_ssa(%atom, %src, %old, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, f32, !fly.memref<i1, register, 1:1>) -> f32
  fly.ptr.store(%result, %out) : (f32, !fly.ptr<f32, global>) -> ()
  return
}

// CHECK-LABEL: func.func @lds_transpose(
// CHECK-SAME: %[[OLD:[a-zA-Z0-9_]+]]: vector<2xi32>
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: %[[RESULT:.*]] = scf.if %[[PRED]] -> (vector<2xi32>) {
// CHECK: rocdl.ds.read.tr4.b64
// CHECK: scf.yield
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[OLD]] : vector<2xi32>
// CHECK: llvm.store %[[RESULT]],
func.func @lds_transpose(%atom: !fly.copy_atom<!fly_rocdl.cdna4.lds_read_trans<trans = 4b, 64>, 32>, %src: !fly.memref<i32, shared, 2:1>, %pred: !fly.memref<i1, register, 1:1>, %out: !fly.ptr<i32, global>, %old: vector<2xi32>) {
  %result = fly.copy_atom_call_ssa(%atom, %src, %old, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly_rocdl.cdna4.lds_read_trans<trans = 4b, 64>, 32>, !fly.memref<i32, shared, 2:1>, vector<2xi32>, !fly.memref<i1, register, 1:1>) -> vector<2xi32>
  fly.ptr.store(%result, %out) : (vector<2xi32>, !fly.ptr<i32, global>) -> ()
  return
}
