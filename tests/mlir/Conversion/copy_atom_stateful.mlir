// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// Stateful CopyAtom lowering tests

// -----

// CHECK-LABEL: @test_stateful_buffer_copy_type
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32)>)
func.func @test_stateful_buffer_copy_type(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>) {
  return
}

// -----

// make_copy_atom produces default state via getDefaultState (soffset = 0)

// CHECK-LABEL: @test_make_copy_atom_default_soffset
func.func @test_make_copy_atom_default_soffset(
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, register, 1:1>) {
  // CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[ATOM:.*]] = llvm.insertvalue %[[C0]], %[[UNDEF]][0]
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>
  // CHECK: %[[SOFF_RAW:.*]] = llvm.extractvalue %[[ATOM]][0]
  // CHECK: %[[SOFF:.*]] = arith.muli %[[SOFF_RAW]], %{{.*}}
  // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %[[SOFF]]
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, register, 1:1>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_atom_set_soffset
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32)>, %[[SOFF:.*]]: i32,
func.func @test_atom_set_soffset(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>,
    %soff: i32,
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, register, 1:1>) {
  // CHECK: %[[RES:.*]] = llvm.insertvalue %[[SOFF]], %[[ATOM]][0]
  %new_atom = fly.atom.set_value(%atom, "soffset", %soff) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>
  fly.copy_atom_call(%new_atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, register, 1:1>) -> ()
  return
}

// -----

// === copy_atom_call: soffset flows into rocdl.raw.ptr.buffer.load ===

// CHECK-LABEL: @test_copy_atom_call_with_soffset
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32)>, %[[SRC:.*]]: !llvm.struct<(ptr<8>, i32)>, %[[DST:.*]]: !llvm.ptr<5>)
func.func @test_copy_atom_call_with_soffset(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>,
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, register, 1:1>) {
  // CHECK: %[[SOFF_RAW:.*]] = llvm.extractvalue %[[ATOM]][0]
  // CHECK: %[[SOFF:.*]] = arith.muli %[[SOFF_RAW]], %{{.*}}
  // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %[[SOFF]]
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, register, 1:1>) -> ()
  return
}

// -----

// === copy_atom_call: soffset flows into rocdl.raw.ptr.buffer.store ===

// CHECK-LABEL: @test_copy_atom_call_store_with_soffset
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32)>, %[[SRC:.*]]: !llvm.ptr<5>, %[[DST:.*]]: !llvm.struct<(ptr<8>, i32)>)
func.func @test_copy_atom_call_store_with_soffset(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>,
    %src: !fly.memref<f32, register, 1:1>,
    %dst: !fly.memref<f32, buffer_desc, 1:1>) {
  // CHECK: %[[SOFF_RAW:.*]] = llvm.extractvalue %[[ATOM]][0]
  // CHECK: %[[SOFF:.*]] = arith.muli %[[SOFF_RAW]], %{{.*}}
  // CHECK: llvm.load %[[SRC]]
  // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %[[SOFF]]
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, buffer_desc, 1:1>) -> ()
  return
}

// -----

// === End-to-end: set soffset then copy ===

// CHECK-LABEL: @test_set_soffset_then_copy
func.func @test_set_soffset_then_copy(
    %atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>,
    %src: !fly.memref<f32, buffer_desc, 1:1>,
    %dst: !fly.memref<f32, register, 1:1>,
    %soff: i32) {
  // CHECK: %[[NEW_ATOM:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[0]
  %new_atom = fly.atom.set_value(%atom, "soffset", %soff) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>
  // CHECK: %[[SOFF_RAW:.*]] = llvm.extractvalue %[[NEW_ATOM]][0]
  // CHECK: %[[SOFF:.*]] = arith.muli %[[SOFF_RAW]], %{{.*}}
  // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %[[SOFF]]
  fly.copy_atom_call(%new_atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, !fly.memref<f32, buffer_desc, 1:1>, !fly.memref<f32, register, 1:1>) -> ()
  return
}
