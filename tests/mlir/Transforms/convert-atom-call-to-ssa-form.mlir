// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-convert-atom-call-to-ssa-form | FileCheck %s

gpu.module @convert_atom_call_to_ssa_form {

  // Test 1: copy_atom_call with register dst (rank=1, stride=1) should be promoted
  // CHECK-LABEL: gpu.func @copy_dst_register
  // CHECK-NOT: fly.copy_atom_call(
  // CHECK: %[[REG_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK: %[[SSA:.*]] = fly.copy_atom_call_ssa(%{{.*}}, %{{.*}}) {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
  // CHECK-SAME: : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, global, 4:1>) -> vector<4xf16>
  // CHECK: fly.ptr.store(%[[SSA]], %[[REG_PTR]]) : (vector<4xf16>, !fly.ptr<f16, register>) -> ()
  gpu.func @copy_dst_register(%src: !fly.ptr<f16, global>) kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %src_view = fly.make_view(%src, %vec4) : (!fly.ptr<f16, global>, !fly.layout<4:1>) -> !fly.memref<f16, global, 4:1>
    %copy = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly.universal_copy<64>, 16>

    %reg_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %reg_view = fly.make_view(%reg_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>

    fly.copy_atom_call(%copy, %src_view, %reg_view) : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, global, 4:1>, !fly.memref<f16, register, 4:1>) -> ()
    gpu.return
  }

  // Test 1b: copy_atom_call with register src (rank=1, stride=1) should be promoted
  // src is pre-loaded via ptr.load, then passed to copy_atom_call_ssa as vector operand
  // CHECK-LABEL: gpu.func @copy_src_register
  // CHECK-NOT: fly.copy_atom_call(
  // CHECK: %[[SRC_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK: %[[VEC:.*]] = fly.ptr.load(%[[SRC_PTR]]) : (!fly.ptr<f16, register>) -> vector<4xf16>
  // CHECK: fly.copy_atom_call_ssa(%{{.*}}, %[[VEC]], %{{.*}}) {operandSegmentSizes = array<i32: 1, 1, 1, 0>}
  // CHECK-SAME: : (!fly.copy_atom<!fly.universal_copy<64>, 16>, vector<4xf16>, !fly.memref<f16, global, 4:1>) -> ()
  gpu.func @copy_src_register(%dst: !fly.ptr<f16, global>) kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %dst_view = fly.make_view(%dst, %vec4) : (!fly.ptr<f16, global>, !fly.layout<4:1>) -> !fly.memref<f16, global, 4:1>
    %copy = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly.universal_copy<64>, 16>

    %reg_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %reg_view = fly.make_view(%reg_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>

    fly.copy_atom_call(%copy, %reg_view, %dst_view) : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, global, 4:1>) -> ()
    gpu.return
  }

  // Test 1c: copy_atom_call with both src and dst register should be promoted
  // src is pre-loaded via ptr.load, result stored back to dst register
  // CHECK-LABEL: gpu.func @copy_both_register
  // CHECK-NOT: fly.copy_atom_call(
  // CHECK: %[[SRC_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK: %[[DST_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK: %[[VEC:.*]] = fly.ptr.load(%[[SRC_PTR]]) : (!fly.ptr<f16, register>) -> vector<4xf16>
  // CHECK: %[[SSA:.*]] = fly.copy_atom_call_ssa(%{{.*}}, %[[VEC]]) {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
  // CHECK-SAME: : (!fly.copy_atom<!fly.universal_copy<64>, 16>, vector<4xf16>) -> vector<4xf16>
  // CHECK: fly.ptr.store(%[[SSA]], %[[DST_PTR]]) : (vector<4xf16>, !fly.ptr<f16, register>) -> ()
  gpu.func @copy_both_register() kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %copy = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly.universal_copy<64>, 16>

    %src_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %src_view = fly.make_view(%src_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %dst_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %dst_view = fly.make_view(%dst_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>

    fly.copy_atom_call(%copy, %src_view, %dst_view) : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, register, 4:1>) -> ()
    gpu.return
  }

  // Test 2: copy_atom_call with non-register src and dst should NOT be promoted
  // CHECK-LABEL: gpu.func @copy_global_unchanged
  // CHECK: fly.copy_atom_call(
  // CHECK-NOT: fly.copy_atom_call_ssa
  gpu.func @copy_global_unchanged(%src: !fly.ptr<f16, global>, %dst: !fly.ptr<f16, global>) kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %src_view = fly.make_view(%src, %vec4) : (!fly.ptr<f16, global>, !fly.layout<4:1>) -> !fly.memref<f16, global, 4:1>
    %dst_view = fly.make_view(%dst, %vec4) : (!fly.ptr<f16, global>, !fly.layout<4:1>) -> !fly.memref<f16, global, 4:1>
    %copy = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly.universal_copy<64>, 16>

    fly.copy_atom_call(%copy, %src_view, %dst_view) : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, global, 4:1>, !fly.memref<f16, global, 4:1>) -> ()
    gpu.return
  }

  // Test 3: copy_atom_call with register dst, non-leaf layout that coalesces to rank=1 stride=1
  // (4,1):(1,0) coalesces to 4:1, so should be promoted
  // CHECK-LABEL: gpu.func @copy_dst_register_coalescable
  // CHECK-NOT: fly.copy_atom_call(
  // CHECK: %[[REG_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK: %[[SSA:.*]] = fly.copy_atom_call_ssa(%{{.*}}, %{{.*}}) {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
  // CHECK-SAME: : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, global, 4:1>) -> vector<4xf16>
  // CHECK: fly.ptr.store(%[[SSA]], %[[REG_PTR]]) : (vector<4xf16>, !fly.ptr<f16, register>) -> ()
  gpu.func @copy_dst_register_coalescable(%src: !fly.ptr<f16, global>) kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %src_view = fly.make_view(%src, %vec4) : (!fly.ptr<f16, global>, !fly.layout<4:1>) -> !fly.memref<f16, global, 4:1>

    %acc_shape = fly.make_int_tuple() : () -> !fly.int_tuple<(4,1)>
    %acc_stride = fly.make_int_tuple() : () -> !fly.int_tuple<(1,0)>
    %acc_layout = fly.make_layout(%acc_shape, %acc_stride) : (!fly.int_tuple<(4,1)>, !fly.int_tuple<(1,0)>) -> !fly.layout<(4,1):(1,0)>

    %copy = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly.universal_copy<64>, 16>

    %reg_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %reg_view = fly.make_view(%reg_ptr, %acc_layout) : (!fly.ptr<f16, register>, !fly.layout<(4,1):(1,0)>) -> !fly.memref<f16, register, (4,1):(1,0)>

    fly.copy_atom_call(%copy, %src_view, %reg_view) : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, global, 4:1>, !fly.memref<f16, register, (4,1):(1,0)>) -> ()
    gpu.return
  }

  // Test 3b: copy_atom_call with register dst, non-coalescable layout should NOT be promoted
  // (4,2):(1,8) cannot coalesce to rank=1 stride=1
  // CHECK-LABEL: gpu.func @copy_dst_register_non_coalescable
  // CHECK: fly.copy_atom_call(
  // CHECK-NOT: fly.copy_atom_call_ssa
  gpu.func @copy_dst_register_non_coalescable(%src: !fly.ptr<f16, global>) kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %src_view = fly.make_view(%src, %vec4) : (!fly.ptr<f16, global>, !fly.layout<4:1>) -> !fly.memref<f16, global, 4:1>

    %nc_shape = fly.make_int_tuple() : () -> !fly.int_tuple<(4,2)>
    %nc_stride = fly.make_int_tuple() : () -> !fly.int_tuple<(1,8)>
    %nc_layout = fly.make_layout(%nc_shape, %nc_stride) : (!fly.int_tuple<(4,2)>, !fly.int_tuple<(1,8)>) -> !fly.layout<(4,2):(1,8)>

    %copy = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly.universal_copy<64>, 16>

    %reg_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f16, register>
    %reg_view = fly.make_view(%reg_ptr, %nc_layout) : (!fly.ptr<f16, register>, !fly.layout<(4,2):(1,8)>) -> !fly.memref<f16, register, (4,2):(1,8)>

    fly.copy_atom_call(%copy, %src_view, %reg_view) : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, global, 4:1>, !fly.memref<f16, register, (4,2):(1,8)>) -> ()
    gpu.return
  }

  // Test 4: mma_atom_call with register d (rank=1, stride=1) should be promoted
  // a, b, c are also register eligible, so they get pre-loaded as vectors
  // CHECK-LABEL: gpu.func @mma_d_register
  // CHECK-NOT: fly.mma_atom_call(
  // CHECK-DAG: %[[A_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK-DAG: %[[B_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK-DAG: %[[D_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f32, register>
  // CHECK-DAG: %[[C_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f32, register>
  // CHECK: %[[A:.*]] = fly.ptr.load(%[[A_PTR]]) : (!fly.ptr<f16, register>) -> vector<4xf16>
  // CHECK: %[[B:.*]] = fly.ptr.load(%[[B_PTR]]) : (!fly.ptr<f16, register>) -> vector<4xf16>
  // CHECK: %[[C:.*]] = fly.ptr.load(%[[C_PTR]]) : (!fly.ptr<f32, register>) -> vector<4xf32>
  // CHECK: %[[SSA:.*]] = fly.mma_atom_call_ssa(%{{.*}}, %[[A]], %[[B]], %[[C]])
  // CHECK-SAME: -> vector<4xf32>
  // CHECK: fly.ptr.store(%[[SSA]], %[[D_PTR]]) : (vector<4xf32>, !fly.ptr<f32, register>) -> ()
  gpu.func @mma_d_register(%out: !fly.ptr<f32, global>) kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4_f16 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>
    %vec4_f32 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %a_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %b_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %d_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f32, register>
    %c_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f32, register>

    %a_view = fly.make_view(%a_ptr, %vec4_f16) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %b_view = fly.make_view(%b_ptr, %vec4_f16) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %d_view = fly.make_view(%d_ptr, %vec4_f32) : (!fly.ptr<f32, register>, !fly.layout<4:1>) -> !fly.memref<f32, register, 4:1>
    %c_view = fly.make_view(%c_ptr, %vec4_f32) : (!fly.ptr<f32, register>, !fly.layout<4:1>) -> !fly.memref<f32, register, 4:1>

    %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>

    fly.mma_atom_call(%atom, %d_view, %a_view, %b_view, %c_view) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, !fly.memref<f32, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
    gpu.return
  }

  // Test 5: mma_atom_call with non-leaf register d layout that coalesces to rank=1 stride=1
  // (4,1):(1,0) coalesces to 4:1, so should be promoted. a, b, c also pre-loaded
  // CHECK-LABEL: gpu.func @mma_d_register_coalescable
  // CHECK-NOT: fly.mma_atom_call(
  // CHECK-DAG: %[[A_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK-DAG: %[[B_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK-DAG: %[[D_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f32, register>
  // CHECK-DAG: %[[C_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f32, register>
  // CHECK: %[[A:.*]] = fly.ptr.load(%[[A_PTR]]) : (!fly.ptr<f16, register>) -> vector<4xf16>
  // CHECK: %[[B:.*]] = fly.ptr.load(%[[B_PTR]]) : (!fly.ptr<f16, register>) -> vector<4xf16>
  // CHECK: %[[C:.*]] = fly.ptr.load(%[[C_PTR]]) : (!fly.ptr<f32, register>) -> vector<4xf32>
  // CHECK: %[[SSA:.*]] = fly.mma_atom_call_ssa(%{{.*}}, %[[A]], %[[B]], %[[C]])
  // CHECK-SAME: -> vector<4xf32>
  // CHECK: fly.ptr.store(%[[SSA]], %[[D_PTR]]) : (vector<4xf32>, !fly.ptr<f32, register>) -> ()
  gpu.func @mma_d_register_coalescable(%out: !fly.ptr<f32, global>) kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4_f16 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %acc_shape = fly.make_int_tuple() : () -> !fly.int_tuple<(4,1)>
    %acc_stride = fly.make_int_tuple() : () -> !fly.int_tuple<(1,0)>
    %acc_layout = fly.make_layout(%acc_shape, %acc_stride) : (!fly.int_tuple<(4,1)>, !fly.int_tuple<(1,0)>) -> !fly.layout<(4,1):(1,0)>

    %a_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %b_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %d_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f32, register>
    %c_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f32, register>

    %a_view = fly.make_view(%a_ptr, %vec4_f16) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %b_view = fly.make_view(%b_ptr, %vec4_f16) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %d_view = fly.make_view(%d_ptr, %acc_layout) : (!fly.ptr<f32, register>, !fly.layout<(4,1):(1,0)>) -> !fly.memref<f32, register, (4,1):(1,0)>
    %c_view = fly.make_view(%c_ptr, %acc_layout) : (!fly.ptr<f32, register>, !fly.layout<(4,1):(1,0)>) -> !fly.memref<f32, register, (4,1):(1,0)>

    %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>

    fly.mma_atom_call(%atom, %d_view, %a_view, %b_view, %c_view) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, !fly.memref<f32, register, (4,1):(1,0)>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, (4,1):(1,0)>) -> ()
    gpu.return
  }

  // Test 5b: mma_atom_call with register d non-coalescable, but a/b are register eligible
  // d and c have (4,2):(1,8) which cannot coalesce, but a/b have 4:1 which is eligible
  // a and b should be pre-loaded as vectors, mma_atom_call_ssa is used
  // d and c remain as memref (not promoted to SSA)
  // CHECK-LABEL: gpu.func @mma_d_register_non_coalescable
  // CHECK-DAG: %[[A_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK-DAG: %[[B_PTR:.*]] = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
  // CHECK: %[[A:.*]] = fly.ptr.load(%[[A_PTR]]) : (!fly.ptr<f16, register>) -> vector<4xf16>
  // CHECK: %[[B:.*]] = fly.ptr.load(%[[B_PTR]]) : (!fly.ptr<f16, register>) -> vector<4xf16>
  // CHECK: fly.mma_atom_call_ssa(%{{.*}}, %{{.*}}, %[[A]], %[[B]], %{{.*}}) :
  // CHECK-SAME: (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>,
  // CHECK-SAME: !fly.memref<f32, register, (4,2):(1,8)>, vector<4xf16>, vector<4xf16>,
  // CHECK-SAME: !fly.memref<f32, register, (4,2):(1,8)>) -> ()
  // CHECK-NOT: fly.ptr.store
  gpu.func @mma_d_register_non_coalescable(%out: !fly.ptr<f32, global>) kernel {
    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4_f16 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %nc_shape = fly.make_int_tuple() : () -> !fly.int_tuple<(4,2)>
    %nc_stride = fly.make_int_tuple() : () -> !fly.int_tuple<(1,8)>
    %nc_layout = fly.make_layout(%nc_shape, %nc_stride) : (!fly.int_tuple<(4,2)>, !fly.int_tuple<(1,8)>) -> !fly.layout<(4,2):(1,8)>

    %a_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %b_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %d_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f32, register>
    %c_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f32, register>

    %a_view = fly.make_view(%a_ptr, %vec4_f16) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %b_view = fly.make_view(%b_ptr, %vec4_f16) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %d_view = fly.make_view(%d_ptr, %nc_layout) : (!fly.ptr<f32, register>, !fly.layout<(4,2):(1,8)>) -> !fly.memref<f32, register, (4,2):(1,8)>
    %c_view = fly.make_view(%c_ptr, %nc_layout) : (!fly.ptr<f32, register>, !fly.layout<(4,2):(1,8)>) -> !fly.memref<f32, register, (4,2):(1,8)>

    %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>

    fly.mma_atom_call(%atom, %d_view, %a_view, %b_view, %c_view) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, !fly.memref<f32, register, (4,2):(1,8)>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, (4,2):(1,8)>) -> ()
    gpu.return
  }
}
