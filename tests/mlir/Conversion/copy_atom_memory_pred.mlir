// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl --canonicalize | FileCheck %s

// Exercise the non-SSA emitter directly with register memref predicates.
// Deliberately omit atom-call conversion: a scalar register predicate would
// otherwise promote and route these calls to the SSA emitter.

// CHECK-LABEL: func.func @buffer_copy_lds_memory_pred(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.raw.ptr.buffer.load.lds
// CHECK-NEXT: }
func.func @buffer_copy_lds_memory_pred(%atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %dst: !fly.memref<f32, shared, 1:1>, %pred: !fly.memref<i1, register, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @buffer_load_async_lds_memory_pred(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.raw.ptr.buffer.load.async.lds
// CHECK-NEXT: }
func.func @buffer_load_async_lds_memory_pred(%atom: !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>, %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %dst: !fly.memref<f32, shared, 1:1>, %pred: !fly.memref<i1, register, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @global_load_async_lds_memory_pred(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.global.load.async.lds
// CHECK-NEXT: }
func.func @global_load_async_lds_memory_pred(%atom: !fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<32>, 32>, %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>, %pred: !fly.memref<i1, register, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<32>, 32>, !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @gfx1250_tdm_load_memory_pred(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.tensor.load.to.lds
// CHECK-NEXT: }
func.func @gfx1250_tdm_load_memory_pred(%atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 16>, %src: !fly.memref<f16, global, (128,64):(64,1)>, %dst: !fly.memref<f16, shared, (128,64):(64,1)>, %pred: !fly.memref<i1, register, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 16>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @gfx1250_tdm_store_memory_pred(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.tensor.store.from.lds
// CHECK-NEXT: }
func.func @gfx1250_tdm_store_memory_pred(%atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 16>, %src: !fly.memref<f16, shared, (128,64):(64,1)>, %dst: !fly.memref<f16, global, (128,64):(64,1)>, %pred: !fly.memref<i1, register, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 16>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @cdna5_tdm_load_memory_pred(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.tensor.load.to.lds
// CHECK-NEXT: }
func.func @cdna5_tdm_load_memory_pred(%atom: !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, %src: !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, %dst: !fly.memref<f16, shared, (128,64):(64,1)>, %pred: !fly.memref<i1, register, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.memref<i1, register, 1:1>) -> ()
  return
}

// CHECK-LABEL: func.func @cdna5_tdm_store_memory_pred(
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CHECK: scf.if %[[PRED]] {
// CHECK: rocdl.tensor.store.from.lds
// CHECK-NEXT: }
func.func @cdna5_tdm_store_memory_pred(%atom: !fly.copy_atom<!fly_rocdl.cdna5.tensor_store<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, %src: !fly.memref<f16, shared, (128,64):(64,1)>, %dst: !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, %pred: !fly.memref<i1, register, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_store<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, !fly.memref<i1, register, 1:1>) -> ()
  return
}
