// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering --canonicalize --fly-convert-atom-call-to-ssa-form --fly-promote-regmem-to-vectorssa --convert-fly-to-rocdl --canonicalize | FileCheck %s

// These atoms have no register src/dst: only pred triggers SSA call conversion.
// A runtime flag must guard the actual instruction, with no predicate load left
// after register promotion. Both TDM load and store directions are covered.
gpu.module @copy_pred_promotion {

  // CHECK-LABEL: gpu.func @buffer_copy_lds(
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: i1
  // CHECK-NOT: llvm.load
  // CHECK: scf.if %[[FLAG]] {
  // CHECK: rocdl.raw.ptr.buffer.load.lds
  // CHECK-NEXT: }
  gpu.func @buffer_copy_lds(%atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %dst: !fly.memref<f32, shared, 1:1>, %flag: i1) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%flag, %ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred = fly.make_view(%ptr, %layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy_lds<32>, 32>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @buffer_load_async_lds(
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: i1
  // CHECK-NOT: llvm.load
  // CHECK: scf.if %[[FLAG]] {
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  // CHECK-NEXT: }
  gpu.func @buffer_load_async_lds(%atom: !fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>, %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %dst: !fly.memref<f32, shared, 1:1>, %flag: i1) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%flag, %ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred = fly.make_view(%ptr, %layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna4.buffer_load_async_lds<32>, 32>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<f32, shared, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @global_load_async_lds(
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: i1
  // CHECK-NOT: llvm.load
  // CHECK: scf.if %[[FLAG]] {
  // CHECK: rocdl.global.load.async.lds
  // CHECK-NEXT: }
  gpu.func @global_load_async_lds(%atom: !fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<32>, 32>, %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>, %flag: i1) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%flag, %ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred = fly.make_view(%ptr, %layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<32>, 32>, !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @gfx1250_tdm_load(
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: i1
  // CHECK-NOT: llvm.load
  // CHECK: scf.if %[[FLAG]] {
  // CHECK: rocdl.tensor.load.to.lds
  // CHECK-NEXT: }
  gpu.func @gfx1250_tdm_load(%atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 16>, %src: !fly.memref<f16, global, (128,64):(64,1)>, %dst: !fly.memref<f16, shared, (128,64):(64,1)>, %flag: i1) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%flag, %ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred = fly.make_view(%ptr, %layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 16>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @gfx1250_tdm_store(
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: i1
  // CHECK-NOT: llvm.load
  // CHECK: scf.if %[[FLAG]] {
  // CHECK: rocdl.tensor.store.from.lds
  // CHECK-NEXT: }
  gpu.func @gfx1250_tdm_store(%atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 16>, %src: !fly.memref<f16, shared, (128,64):(64,1)>, %dst: !fly.memref<f16, global, (128,64):(64,1)>, %flag: i1) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%flag, %ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred = fly.make_view(%ptr, %layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm<rank = 2, warps = 1, pad = 0, 0, cache = 0, barrier = false, timeout = false>, 16>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.memref<f16, global, (128,64):(64,1)>, !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @cdna5_tdm_load(
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: i1
  // CHECK-NOT: llvm.load
  // CHECK: scf.if %[[FLAG]] {
  // CHECK: rocdl.tensor.load.to.lds
  // CHECK-NEXT: }
  gpu.func @cdna5_tdm_load(%atom: !fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, %src: !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, %dst: !fly.memref<f16, shared, (128,64):(64,1)>, %flag: i1) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%flag, %ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred = fly.make_view(%ptr, %layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_load<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @cdna5_tdm_store(
  // CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: i1
  // CHECK-NOT: llvm.load
  // CHECK: scf.if %[[FLAG]] {
  // CHECK: rocdl.tensor.store.from.lds
  // CHECK-NEXT: }
  gpu.func @cdna5_tdm_store(%atom: !fly.copy_atom<!fly_rocdl.cdna5.tensor_store<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, %src: !fly.memref<f16, shared, (128,64):(64,1)>, %dst: !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, %flag: i1) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%flag, %ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred = fly.make_view(%ptr, %layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna5.tensor_store<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>, !fly.memref<f16, shared, (128,64):(64,1)>, !fly.coord_tensor<(0,0), (128,64):(1E0,1E1)>, !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }

  // The dynamic slot excludes the whole allocation from regmem promotion, but
  // the scalar pred view is still eligible for SSA call conversion.
  // CHECK-LABEL: gpu.func @dynamic_pred_slot(
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
  // CHECK: scf.if %[[PRED]] {
  // CHECK: rocdl.global.load.async.lds
  // CHECK-NEXT: }
  gpu.func @dynamic_pred_slot(%atom: !fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<32>, 32>, %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>, %index: i32, %flag: i1) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 4 : i64}} : () -> !fly.ptr<i1, register>
    %offset = fly.make_int_tuple(%index) : (i32) -> !fly.int_tuple<?>
    %slot = fly.add_offset(%ptr, %offset) : (!fly.ptr<i1, register>, !fly.int_tuple<?>) -> !fly.ptr<i1, register>
    fly.ptr.store(%flag, %slot) : (i1, !fly.ptr<i1, register>) -> ()
    %pred = fly.make_view(%slot, %layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna4.global_load_async_lds<32>, 32>, !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }

}
