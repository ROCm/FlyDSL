// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-convert-atom-call-to-ssa-form --fly-promote-regmem-to-vectorssa | FileCheck %s


// CHECK-LABEL: gpu.func @promote_copy_atoms_to_ssa
// CHECK-NOT: register
// CHECK: %[[POISON:.*]] = ub.poison : vector<4xf16>
// CHECK: %[[LOAD:.*]] = fly.copy_atom_call_ssa(%{{.*}}, %{{.*}}) {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
// CHECK-SAME: : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>) -> vector<4xf16>
// CHECK: %[[STATE:.*]] = vector.insert_strided_slice %[[LOAD]], %[[POISON]] {offsets = [0], strides = [1]} : vector<4xf16> into vector<4xf16>
// CHECK: %[[READ:.*]] = vector.extract_strided_slice %[[STATE]] {offsets = [0], sizes = [4], strides = [1]} : vector<4xf16> to vector<4xf16>
// CHECK: fly.copy_atom_call_ssa(%{{.*}}, %[[READ]], %{{.*}}) {operandSegmentSizes = array<i32: 1, 1, 1, 0>}
// CHECK-SAME: : (!fly.copy_atom<!fly.universal_copy<64>, 16>, vector<4xf16>, !fly.memref<f16, global, 4:1>) -> ()

gpu.module @promote_rmem_to_vector_ssa_copy {
  gpu.func @promote_copy_atoms_to_ssa(%src: !fly.ptr<f16, global>, %dst: !fly.ptr<f16, global>) kernel {
    %c0_i16 = arith.constant 0 : i16
    %c4_i32 = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c4294967295_i64 = arith.constant 4294967295 : i64

    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %src_desc = fly.make_ptr(%src, %c0_i16, %c4294967295_i64, %c1024_i32) : (!fly.ptr<f16, global>, i16, i64, i32) -> !fly.ptr<f16, #fly_rocdl.buffer_desc>
    %src_view = fly.make_view(%src_desc, %vec4) : (!fly.ptr<f16, #fly_rocdl.buffer_desc>, !fly.layout<4:1>) -> !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>
    %dst_view = fly.make_view(%dst, %vec4) : (!fly.ptr<f16, global>, !fly.layout<4:1>) -> !fly.memref<f16, global, 4:1>

    %copy_in = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>
    %copy_in_soff = fly.atom.set_value(%copy_in, "soffset", %c4_i32) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>
    %copy_out = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly.universal_copy<64>, 16>

    %reg_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %reg_view = fly.make_view(%reg_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>

    fly.copy_atom_call(%copy_in_soff, %src_view, %reg_view) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>, !fly.memref<f16, register, 4:1>) -> ()
    fly.copy_atom_call(%copy_out, %reg_view, %dst_view) : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, global, 4:1>) -> ()
    gpu.return
  }

// CHECK-LABEL: gpu.func @promote_loop_local_copy_to_ssa
// CHECK-NOT: register
// CHECK: %[[LOOP_POISON:.*]] = ub.poison : vector<4xf16>
// CHECK: %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_STATE:.*]] = %[[LOOP_POISON]]) -> (vector<4xf16>) {
// CHECK:   %[[ITER_SSA:.*]] = fly.copy_atom_call_ssa(%{{.*}}, %{{.*}}) {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
// CHECK-SAME: : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>) -> vector<4xf16>
// CHECK:   %[[NEW_STATE:.*]] = vector.insert_strided_slice %[[ITER_SSA]], %[[ITER_STATE]] {offsets = [0], strides = [1]} : vector<4xf16> into vector<4xf16>
// CHECK:   %[[SLICE:.*]] = vector.extract_strided_slice %[[NEW_STATE]] {offsets = [0], sizes = [4], strides = [1]} : vector<4xf16> to vector<4xf16>
// CHECK:   %[[ELEM:.*]] = vector.extract %[[SLICE]][%{{.*}}] : f16 from vector<4xf16>
  gpu.func @promote_loop_local_copy_to_ssa(%src: !fly.ptr<f16, global>, %dst: !fly.ptr<f16, global>) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i16 = arith.constant 0 : i16
    %c4_i32 = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c4294967295_i64 = arith.constant 4294967295 : i64

    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %src_desc = fly.make_ptr(%src, %c0_i16, %c4294967295_i64, %c1024_i32) : (!fly.ptr<f16, global>, i16, i64, i32) -> !fly.ptr<f16, #fly_rocdl.buffer_desc>
    %src_view = fly.make_view(%src_desc, %vec4) : (!fly.ptr<f16, #fly_rocdl.buffer_desc>, !fly.layout<4:1>) -> !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>

    %copy_in = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>
    %reg_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %reg_view = fly.make_view(%reg_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>

    scf.for %iv = %c0 to %c2 step %c1 {
      %iv_i32 = arith.index_cast %iv : index to i32
      %soff = arith.muli %iv_i32, %c4_i32 : i32
      %copy_iter = fly.atom.set_value(%copy_in, "soffset", %soff) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>
      fly.copy_atom_call(%copy_iter, %src_view, %reg_view) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>, !fly.memref<f16, register, 4:1>) -> ()
      %vec = fly.ptr.load(%reg_ptr) : (!fly.ptr<f16, register>) -> vector<4xf16>
      %elem = vector.extract %vec[%c0] : f16 from vector<4xf16>
      fly.ptr.store(%elem, %dst) : (f16, !fly.ptr<f16, global>) -> ()
    }
    gpu.return
  }
}


// Verify that when a predicated copy_atom_call has a register-memref pred
// and a register-memref dst, after both passes:
// 1. pred is promoted to i1 SSA via vector<1xi1> state
// 2. old dst value is extracted from vector state before the SSA call
// 3. copy_atom_call_ssa receives old dst and pred as SSA operands
// 4. result is inserted back into vector state
// 5. all register types are eliminated

// CHECK-LABEL: gpu.func @promote_pred_copy_to_ssa
// CHECK-NOT: register
// CHECK: %[[PRED_POISON:.*]] = ub.poison : vector<1xi1>
// CHECK: %[[PRED_STATE:.*]] = vector.insert %arg2, %[[PRED_POISON]] [0] : i1 into vector<1xi1>
// CHECK: %[[DST_POISON:.*]] = ub.poison : vector<4xf16>
// CHECK: %[[PRED_VAL:.*]] = vector.extract %[[PRED_STATE]][0] : i1 from vector<1xi1>
// CHECK: %[[OLD_DST:.*]] = vector.extract_strided_slice %[[DST_POISON]] {offsets = [0], sizes = [4], strides = [1]} : vector<4xf16> to vector<4xf16>
// CHECK: %[[SSA:.*]] = fly.copy_atom_call_ssa(%{{.*}}, %{{.*}}, %[[OLD_DST]], %[[PRED_VAL]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1>}
// CHECK-SAME: : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>, vector<4xf16>, i1) -> vector<4xf16>
// CHECK: %[[UPDATED:.*]] = vector.insert_strided_slice %[[SSA]], %[[DST_POISON]] {offsets = [0], strides = [1]} : vector<4xf16> into vector<4xf16>
// CHECK: %[[OUT_VEC:.*]] = vector.extract_strided_slice %[[UPDATED]] {offsets = [0], sizes = [4], strides = [1]} : vector<4xf16> to vector<4xf16>
// CHECK: fly.copy_atom_call_ssa(%{{.*}}, %[[OUT_VEC]], %{{.*}}) {operandSegmentSizes = array<i32: 1, 1, 1, 0>}
// CHECK-SAME: : (!fly.copy_atom<!fly.universal_copy<64>, 16>, vector<4xf16>, !fly.memref<f16, global, 4:1>) -> ()
gpu.module @promote_rmem_to_vector_ssa_copy_pred {
  gpu.func @promote_pred_copy_to_ssa(%src: !fly.ptr<f16, global>, %dst: !fly.ptr<f16, global>, %pred_val: i1) kernel {
    %c0_i16 = arith.constant 0 : i16
    %c4_i32 = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c4294967295_i64 = arith.constant 4294967295 : i64

    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>

    %shape1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %pred_layout = fly.make_layout(%shape1, %stride1) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>

    %src_desc = fly.make_ptr(%src, %c0_i16, %c4294967295_i64, %c1024_i32) : (!fly.ptr<f16, global>, i16, i64, i32) -> !fly.ptr<f16, #fly_rocdl.buffer_desc>
    %src_view = fly.make_view(%src_desc, %vec4) : (!fly.ptr<f16, #fly_rocdl.buffer_desc>, !fly.layout<4:1>) -> !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>
    %dst_view = fly.make_view(%dst, %vec4) : (!fly.ptr<f16, global>, !fly.layout<4:1>) -> !fly.memref<f16, global, 4:1>

    %copy_in = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>
    %copy_in_soff = fly.atom.set_value(%copy_in, "soffset", %c4_i32) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, i32) -> !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>

    %pred_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%pred_val, %pred_ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred_view = fly.make_view(%pred_ptr, %pred_layout) : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>

    %reg_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %reg_view = fly.make_view(%reg_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>

    fly.copy_atom_call(%copy_in_soff, %src_view, %reg_view, %pred_view) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<64>, 16>, !fly.memref<f16, #fly_rocdl.buffer_desc, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<i1, register, 1:1>) -> ()

    %copy_out = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly.universal_copy<64>, 16>
    fly.copy_atom_call(%copy_out, %reg_view, %dst_view) : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, global, 4:1>) -> ()
    gpu.return
  }
}
