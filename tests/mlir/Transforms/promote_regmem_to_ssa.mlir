// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-convert-atom-call-to-ssa-form --fly-promote-regmem-to-vectorssa | FileCheck %s --check-prefix=CHECK

// Tests for fly-promote-regmem-to-vectorssa pass:
//   - register state is threaded across scf.for / scf.if / scf.while
//   - tail mma_atom_call after the loop is rewritten to SSA form
//   - final register ptr.load is replaced by vector extract ops

// CHECK-LABEL: gpu.func @promote_accumulator_to_vector_ssa
// CHECK-SAME: (%[[OUT:.*]]: !fly.ptr<f32, global>)
// CHECK-NOT: register
// CHECK-NOT: fly.mma_atom_call(
// CHECK: %[[A_STATE:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0], strides = [1]} : vector<4xf16> into vector<4xf16>
// CHECK: %[[B_STATE:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0], strides = [1]} : vector<4xf16> into vector<4xf16>
// CHECK: %[[ACC_INIT:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
gpu.module @promote_rmem_to_vector_ssa {
  gpu.func @promote_accumulator_to_vector_ssa(%out: !fly.ptr<f32, global>) kernel {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %a_init = arith.constant dense<1.000000e+00> : vector<4xf16>
    %b_init = arith.constant dense<2.000000e+00> : vector<4xf16>
    %zero = arith.constant dense<0.000000e+00> : vector<4xf32>

    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec4 = fly.make_layout(%shape4, %stride1) : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>
    %acc_shape = fly.make_int_tuple() : () -> !fly.int_tuple<(4,1)>
    %acc_stride = fly.make_int_tuple() : () -> !fly.int_tuple<(1,0)>
    %acc_layout = fly.make_layout(%acc_shape, %acc_stride) : (!fly.int_tuple<(4,1)>, !fly.int_tuple<(1,0)>) -> !fly.layout<(4,1):(1,0)>

    %a_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %b_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f16, register>
    %acc_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f32, register>

    fly.ptr.store(%a_init, %a_ptr) : (vector<4xf16>, !fly.ptr<f16, register>) -> ()
    fly.ptr.store(%b_init, %b_ptr) : (vector<4xf16>, !fly.ptr<f16, register>) -> ()

    %acc_off = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %acc_slot = fly.add_offset(%acc_ptr, %acc_off) : (!fly.ptr<f32, register>, !fly.int_tuple<4>) -> !fly.ptr<f32, register>
    fly.ptr.store(%zero, %acc_slot) : (vector<4xf32>, !fly.ptr<f32, register>) -> ()

    %a_view = fly.make_view(%a_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %b_view = fly.make_view(%b_ptr, %vec4) : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %acc_view = fly.make_view(%acc_slot, %acc_layout) : (!fly.ptr<f32, register>, !fly.layout<(4,1):(1,0)>) -> !fly.memref<f32, register, (4,1):(1,0)>
    %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>

    // CHECK: %{{.*}}:3 = scf.for {{.*}} iter_args(%[[A_ITER:.*]] = %[[A_STATE]], %[[B_ITER:.*]] = %[[B_STATE]], %[[ACC:.*]] = %[[ACC_INIT]]) -> (vector<4xf16>, vector<4xf16>, vector<8xf32>) {
    // CHECK: %[[LOOP_A:.*]] = vector.extract_strided_slice %[[A_ITER]] {offsets = [0], sizes = [4], strides = [1]} : vector<4xf16> to vector<4xf16>
    // CHECK: %[[LOOP_B:.*]] = vector.extract_strided_slice %[[B_ITER]] {offsets = [0], sizes = [4], strides = [1]} : vector<4xf16> to vector<4xf16>
    // CHECK: %[[LOOP_C:.*]] = vector.extract_strided_slice %[[ACC]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[LOOP_RES:.*]] = fly.mma_atom_call_ssa
    // CHECK: %[[LOOP_ACC_NEXT:.*]] = vector.insert_strided_slice %[[LOOP_RES]], %[[ACC]] {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
    // CHECK: scf.yield %[[A_ITER]], %[[B_ITER]], %[[LOOP_ACC_NEXT]] : vector<4xf16>, vector<4xf16>, vector<8xf32>
    scf.for %iv = %c0 to %c2 step %c1 {
      fly.mma_atom_call(%atom, %acc_view, %a_view, %b_view, %acc_view) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, !fly.memref<f32, register, (4,1):(1,0)>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, (4,1):(1,0)>) -> ()
    }

    // CHECK: %[[TAIL_A:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [4], strides = [1]} : vector<4xf16> to vector<4xf16>
    // CHECK: %[[TAIL_B:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [4], strides = [1]} : vector<4xf16> to vector<4xf16>
    // CHECK: %[[TAIL_C:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[TAIL_RES:.*]] = fly.mma_atom_call_ssa
    // CHECK: %[[TAIL_ACC:.*]] = vector.insert_strided_slice %[[TAIL_RES]], %{{.*}} {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
    fly.mma_atom_call(%atom, %acc_view, %a_view, %b_view, %acc_view) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>, !fly.memref<f32, register, (4,1):(1,0)>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, (4,1):(1,0)>) -> ()

    // CHECK: %[[FINAL:.*]] = vector.extract_strided_slice %[[TAIL_ACC]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
    // CHECK: %[[ELEM:.*]] = vector.extract %[[FINAL]][%{{.*}}] : f32 from vector<4xf32>
    // CHECK: fly.ptr.store(%[[ELEM]], %[[OUT]]) : (f32, !fly.ptr<f32, global>) -> ()
    %final = fly.ptr.load(%acc_slot) : (!fly.ptr<f32, register>) -> vector<4xf32>
    %elem0 = vector.extract %final[%c0] : f32 from vector<4xf32>
    fly.ptr.store(%elem0, %out) : (f32, !fly.ptr<f32, global>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @promote_fp8_mma_to_vector_ssa
  // CHECK-NOT: register
  // CHECK-NOT: fly.mma_atom_call(
  // CHECK: %[[A_STATE:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0], strides = [1]} : vector<8xf8E4M3FNUZ> into vector<8xf8E4M3FNUZ>
  // CHECK: %[[B_STATE:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0], strides = [1]} : vector<8xf8E4M3FNUZ> into vector<8xf8E4M3FNUZ>
  // CHECK: %[[ACC_INIT:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
  // CHECK: %[[A:.*]] = vector.extract_strided_slice %[[A_STATE]] {offsets = [0], sizes = [8], strides = [1]} : vector<8xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
  // CHECK: %[[B:.*]] = vector.extract_strided_slice %[[B_STATE]] {offsets = [0], sizes = [8], strides = [1]} : vector<8xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
  // CHECK: %[[C:.*]] = vector.extract_strided_slice %[[ACC_INIT]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf32> to vector<4xf32>
  // CHECK: %[[RES:.*]] = fly.mma_atom_call_ssa(%{{.*}}, %[[A]], %[[B]], %[[C]])
  // CHECK-SAME: -> vector<4xf32>
  gpu.func @promote_fp8_mma_to_vector_ssa(%out: !fly.ptr<f32, global>) kernel {
    %c0 = arith.constant 0 : index
    %a_init = arith.constant dense<1.000000e+00> : vector<8xf8E4M3FNUZ>
    %b_init = arith.constant dense<2.000000e+00> : vector<8xf8E4M3FNUZ>
    %zero = arith.constant dense<0.000000e+00> : vector<4xf32>

    %shape8 = fly.make_int_tuple() : () -> !fly.int_tuple<8>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %vec8 = fly.make_layout(%shape8, %stride1) : (!fly.int_tuple<8>, !fly.int_tuple<1>) -> !fly.layout<8:1>
    %acc_shape = fly.make_int_tuple() : () -> !fly.int_tuple<(4,1)>
    %acc_stride = fly.make_int_tuple() : () -> !fly.int_tuple<(1,0)>
    %acc_layout = fly.make_layout(%acc_shape, %acc_stride) : (!fly.int_tuple<(4,1)>, !fly.int_tuple<(1,0)>) -> !fly.layout<(4,1):(1,0)>

    %a_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f8E4M3FNUZ, register>
    %b_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f8E4M3FNUZ, register>
    %acc_ptr = fly.make_ptr() {dictAttrs = {allocaSize = 8 : i64}} : () -> !fly.ptr<f32, register>

    fly.ptr.store(%a_init, %a_ptr) : (vector<8xf8E4M3FNUZ>, !fly.ptr<f8E4M3FNUZ, register>) -> ()
    fly.ptr.store(%b_init, %b_ptr) : (vector<8xf8E4M3FNUZ>, !fly.ptr<f8E4M3FNUZ, register>) -> ()

    %acc_off = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %acc_slot = fly.add_offset(%acc_ptr, %acc_off) : (!fly.ptr<f32, register>, !fly.int_tuple<4>) -> !fly.ptr<f32, register>
    fly.ptr.store(%zero, %acc_slot) : (vector<4xf32>, !fly.ptr<f32, register>) -> ()

    %a_view = fly.make_view(%a_ptr, %vec8) : (!fly.ptr<f8E4M3FNUZ, register>, !fly.layout<8:1>) -> !fly.memref<f8E4M3FNUZ, register, 8:1>
    %b_view = fly.make_view(%b_ptr, %vec8) : (!fly.ptr<f8E4M3FNUZ, register>, !fly.layout<8:1>) -> !fly.memref<f8E4M3FNUZ, register, 8:1>
    %acc_view = fly.make_view(%acc_slot, %acc_layout) : (!fly.ptr<f32, register>, !fly.layout<(4,1):(1,0)>) -> !fly.memref<f32, register, (4,1):(1,0)>
    %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (f8E4M3FNUZ, f8E4M3FNUZ) -> f32>>

    fly.mma_atom_call(%atom, %acc_view, %a_view, %b_view, %acc_view) : (!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (f8E4M3FNUZ, f8E4M3FNUZ) -> f32>>, !fly.memref<f32, register, (4,1):(1,0)>, !fly.memref<f8E4M3FNUZ, register, 8:1>, !fly.memref<f8E4M3FNUZ, register, 8:1>, !fly.memref<f32, register, (4,1):(1,0)>) -> ()

    %final = fly.ptr.load(%acc_slot) : (!fly.ptr<f32, register>) -> vector<4xf32>
    %elem0 = vector.extract %final[%c0] : f32 from vector<4xf32>
    fly.ptr.store(%elem0, %out) : (f32, !fly.ptr<f32, global>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @promote_if_register_state_to_vector_ssa
  // CHECK-NOT: register
  // CHECK: %[[INIT:.*]] = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0], strides = [1]} : vector<4xf32> into vector<4xf32>
  // CHECK: %[[IF_STATE:.*]] = scf.if %arg1 -> (vector<4xf32>) {
  // CHECK:   %[[THEN_STATE:.*]] = vector.insert_strided_slice %{{.*}}, %[[INIT]] {offsets = [0], strides = [1]} : vector<4xf32> into vector<4xf32>
  // CHECK:   scf.yield %[[THEN_STATE]] : vector<4xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[INIT]] : vector<4xf32>
  // CHECK: }
  // CHECK: %[[FINAL:.*]] = vector.extract_strided_slice %[[IF_STATE]] {offsets = [0], sizes = [4], strides = [1]} : vector<4xf32> to vector<4xf32>
  gpu.func @promote_if_register_state_to_vector_ssa(%out: !fly.ptr<f32, global>, %pred: i1) kernel {
    %c0 = arith.constant 0 : index
    %zero = arith.constant dense<0.000000e+00> : vector<4xf32>
    %one = arith.constant dense<1.000000e+00> : vector<4xf32>
    %reg = fly.make_ptr() {dictAttrs = {allocaSize = 4 : i64}} : () -> !fly.ptr<f32, register>

    fly.ptr.store(%zero, %reg) : (vector<4xf32>, !fly.ptr<f32, register>) -> ()
    scf.if %pred {
      fly.ptr.store(%one, %reg) : (vector<4xf32>, !fly.ptr<f32, register>) -> ()
    }

    %final = fly.ptr.load(%reg) : (!fly.ptr<f32, register>) -> vector<4xf32>
    %elem0 = vector.extract %final[%c0] : f32 from vector<4xf32>
    fly.ptr.store(%elem0, %out) : (f32, !fly.ptr<f32, global>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @promote_while_register_state_to_vector_ssa
  // CHECK-NOT: register
  // CHECK: %[[INIT:.*]] = vector.insert %{{.*}}, %{{.*}}[0] : i32 into vector<1xi32>
  // CHECK: %[[WHILE:.*]] = scf.while (%[[STATE:.*]] = %[[INIT]]) : (vector<1xi32>) -> vector<1xi32> {
  // CHECK:   %[[CUR:.*]] = vector.extract %[[STATE]][0] : i32 from vector<1xi32>
  // CHECK:   %[[COND:.*]] = arith.cmpi slt, %[[CUR]], %{{.*}} : i32
  // CHECK:   scf.condition(%[[COND]]) %[[STATE]] : vector<1xi32>
  // CHECK: } do {
  // CHECK: ^bb0(%[[LOOP_STATE:.*]]: vector<1xi32>):
  // CHECK:   %[[CUR2:.*]] = vector.extract %[[LOOP_STATE]][0] : i32 from vector<1xi32>
  // CHECK:   %[[NEXT:.*]] = arith.addi %[[CUR2]], %{{.*}} : i32
  // CHECK:   %[[NEXT_STATE:.*]] = vector.insert %[[NEXT]], %[[LOOP_STATE]] [0] : i32 into vector<1xi32>
  // CHECK:   scf.yield %[[NEXT_STATE]] : vector<1xi32>
  // CHECK: }
  // CHECK: %[[FINAL:.*]] = vector.extract %[[WHILE]][0] : i32 from vector<1xi32>
  gpu.func @promote_while_register_state_to_vector_ssa(%out: !fly.ptr<i32, global>) kernel {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %reg = fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<i32, register>

    fly.ptr.store(%c0_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    scf.while : () -> () {
      %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
      %cond = arith.cmpi slt, %cur, %c2_i32 : i32
      scf.condition(%cond)
    } do {
      %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
      %next = arith.addi %cur, %c1_i32 : i32
      fly.ptr.store(%next, %reg) : (i32, !fly.ptr<i32, register>) -> ()
      scf.yield
    }

    %final = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%final, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @promote_if_with_nested_while_preserves_results_and_state
  // CHECK-NOT: register
  // CHECK: %[[INIT:.*]] = vector.insert %{{.*}}, %{{.*}} [0] : i32 into vector<1xi32>
  // CHECK: %[[IF:.*]]:2 = scf.if %arg1 -> (i32, vector<1xi32>) {
  // CHECK:   %[[WHILE:.*]]:2 = scf.while (%[[ITER:.*]] = %c0_i32, %[[STATE:.*]] = %[[INIT]]) : (i32, vector<1xi32>) -> (i32, vector<1xi32>) {
  // CHECK:     %[[COND:.*]] = arith.cmpi slt, %[[ITER]], %{{.*}} : i32
  // CHECK:     scf.condition(%[[COND]]) %[[ITER]], %[[STATE]] : i32, vector<1xi32>
  // CHECK:   } do {
  // CHECK:   ^bb0(%[[ITER_IN:.*]]: i32, %[[STATE_IN:.*]]: vector<1xi32>):
  // CHECK:     %[[CUR:.*]] = vector.extract %[[STATE_IN]][0] : i32 from vector<1xi32>
  // CHECK:     %[[NEXT_VAL:.*]] = arith.addi %[[CUR]], %{{.*}} : i32
  // CHECK:     %[[STATE_NEXT:.*]] = vector.insert %[[NEXT_VAL]], %[[STATE_IN]] [0] : i32 into vector<1xi32>
  // CHECK:     %[[ITER_NEXT:.*]] = arith.addi %[[ITER_IN]], %{{.*}} : i32
  // CHECK:     scf.yield %[[ITER_NEXT]], %[[STATE_NEXT]] : i32, vector<1xi32>
  // CHECK:   }
  // CHECK:   scf.yield %[[WHILE]]#0, %[[WHILE]]#1 : i32, vector<1xi32>
  // CHECK: } else {
  // CHECK:   %[[ELSE_STATE:.*]] = vector.insert %{{.*}}, %[[INIT]] [0] : i32 into vector<1xi32>
  // CHECK:   scf.yield %{{.*}}, %[[ELSE_STATE]] : i32, vector<1xi32>
  // CHECK: }
  // CHECK: %[[FINAL:.*]] = vector.extract %[[IF]]#1[0] : i32 from vector<1xi32>
  // CHECK: %[[SUM:.*]] = arith.addi %[[IF]]#0, %[[FINAL]] : i32
  // CHECK: fly.ptr.store(%[[SUM]], %arg0) : (i32, !fly.ptr<i32, global>) -> ()
  gpu.func @promote_if_with_nested_while_preserves_results_and_state(%out: !fly.ptr<i32, global>, %pred: i1) kernel {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %reg = fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<i32, register>

    fly.ptr.store(%c0_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    %if_res = scf.if %pred -> (i32) {
      %while_res = scf.while (%iter = %c0_i32) : (i32) -> (i32) {
        %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
        %cond = arith.cmpi slt, %iter, %c2_i32 : i32
        scf.condition(%cond) %iter : i32
      } do {
      ^bb0(%iter_in: i32):
        %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
        %next = arith.addi %cur, %c1_i32 : i32
        fly.ptr.store(%next, %reg) : (i32, !fly.ptr<i32, register>) -> ()
        %iter_next = arith.addi %iter_in, %c1_i32 : i32
        scf.yield %iter_next : i32
      }
      scf.yield %while_res : i32
    } else {
      fly.ptr.store(%c2_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
      scf.yield %c2_i32 : i32
    }

    %final = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
    %sum = arith.addi %if_res, %final : i32
    fly.ptr.store(%sum, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @promote_for_with_nested_if_while_preserves_results_and_state
  // CHECK-NOT: register
  // CHECK: %[[INIT:.*]] = vector.insert %{{.*}}, %{{.*}} [0] : i32 into vector<1xi32>
  // CHECK: %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[SUM:.*]] = %c0_i32, %[[STATE:.*]] = %[[INIT]]) -> (i32, vector<1xi32>) {
  // CHECK:   %[[IF:.*]] = scf.if %arg1 -> (vector<1xi32>) {
  // CHECK:     %[[WHILE:.*]]:2 = scf.while (%[[INNER:.*]] = %c0_i32, %[[WHILE_STATE:.*]] = %[[STATE]]) : (i32, vector<1xi32>) -> (i32, vector<1xi32>) {
  // CHECK:       %[[INNER_COND:.*]] = arith.cmpi slt, %[[INNER]], %{{.*}} : i32
  // CHECK:       scf.condition(%[[INNER_COND]]) %[[INNER]], %[[WHILE_STATE]] : i32, vector<1xi32>
  // CHECK:     } do {
  // CHECK:     ^bb0(%[[INNER_IN:.*]]: i32, %[[BODY_STATE:.*]]: vector<1xi32>):
  // CHECK:       %[[CUR:.*]] = vector.extract %[[BODY_STATE]][0] : i32 from vector<1xi32>
  // CHECK:       %[[NEXT_VAL:.*]] = arith.addi %[[CUR]], %{{.*}} : i32
  // CHECK:       %[[NEXT_STATE:.*]] = vector.insert %[[NEXT_VAL]], %[[BODY_STATE]] [0] : i32 into vector<1xi32>
  // CHECK:       %[[INNER_NEXT:.*]] = arith.addi %[[INNER_IN]], %{{.*}} : i32
  // CHECK:       scf.yield %[[INNER_NEXT]], %[[NEXT_STATE]] : i32, vector<1xi32>
  // CHECK:     }
  // CHECK:     scf.yield %[[WHILE]]#1 : vector<1xi32>
  // CHECK:   } else {
  // CHECK:     scf.yield %[[STATE]] : vector<1xi32>
  // CHECK:   }
  // CHECK:   %[[SUM_NEXT:.*]] = arith.addi %[[SUM]], %{{.*}} : i32
  // CHECK:   scf.yield %[[SUM_NEXT]], %[[IF]] : i32, vector<1xi32>
  // CHECK: }
  // CHECK: %[[FINAL:.*]] = vector.extract %[[FOR]]#1[0] : i32 from vector<1xi32>
  // CHECK: %[[OUTVAL:.*]] = arith.addi %[[FOR]]#0, %[[FINAL]] : i32
  // CHECK: fly.ptr.store(%[[OUTVAL]], %arg0) : (i32, !fly.ptr<i32, global>) -> ()
  gpu.func @promote_for_with_nested_if_while_preserves_results_and_state(%out: !fly.ptr<i32, global>, %pred: i1) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %reg = fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<i32, register>

    fly.ptr.store(%c0_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    %sum = scf.for %iv = %c0 to %c2 step %c1 iter_args(%sum_iter = %c0_i32) -> (i32) {
      scf.if %pred {
        %while_res = scf.while (%inner = %c0_i32) : (i32) -> (i32) {
          %cond = arith.cmpi slt, %inner, %c1_i32 : i32
          scf.condition(%cond) %inner : i32
        } do {
        ^bb0(%inner_in: i32):
          %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
          %next = arith.addi %cur, %c1_i32 : i32
          fly.ptr.store(%next, %reg) : (i32, !fly.ptr<i32, register>) -> ()
          %inner_next = arith.addi %inner_in, %c1_i32 : i32
          scf.yield %inner_next : i32
        }
        scf.yield
      }
      %sum_next = arith.addi %sum_iter, %c1_i32 : i32
      scf.yield %sum_next : i32
    }

    %final = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
    %out_val = arith.addi %sum, %final : i32
    fly.ptr.store(%out_val, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @promote_region_local_nested_register_state_to_vector_ssa
  // CHECK-NOT: register
  // CHECK: %[[IFRES:.*]] = scf.if %arg1 -> (i32) {
  // CHECK:   %[[LOCAL_INIT:.*]] = vector.insert %{{.*}}, %{{.*}} [0] : i32 into vector<1xi32>
  // CHECK:   %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[STATE:.*]] = %[[LOCAL_INIT]]) -> (vector<1xi32>) {
  // CHECK:     %[[WHILE:.*]] = scf.while (%[[WHILE_STATE:.*]] = %[[STATE]]) : (vector<1xi32>) -> vector<1xi32> {
  // CHECK:       %[[CUR:.*]] = vector.extract %[[WHILE_STATE]][0] : i32 from vector<1xi32>
  // CHECK:       %[[COND:.*]] = arith.cmpi slt, %[[CUR]], %{{.*}} : i32
  // CHECK:       scf.condition(%[[COND]]) %[[WHILE_STATE]] : vector<1xi32>
  // CHECK:     } do {
  // CHECK:     ^bb0(%[[BODY_STATE:.*]]: vector<1xi32>):
  // CHECK:       %[[CUR2:.*]] = vector.extract %[[BODY_STATE]][0] : i32 from vector<1xi32>
  // CHECK:       %[[NEXT_VAL:.*]] = arith.addi %[[CUR2]], %{{.*}} : i32
  // CHECK:       %[[NEXT_STATE:.*]] = vector.insert %[[NEXT_VAL]], %[[BODY_STATE]] [0] : i32 into vector<1xi32>
  // CHECK:       scf.yield %[[NEXT_STATE]] : vector<1xi32>
  // CHECK:     }
  // CHECK:     scf.yield %[[WHILE]] : vector<1xi32>
  // CHECK:   }
  // CHECK:   %[[FINAL:.*]] = vector.extract %[[FOR]][0] : i32 from vector<1xi32>
  // CHECK:   fly.ptr.store(%[[FINAL]], %arg0) : (i32, !fly.ptr<i32, global>) -> ()
  // CHECK:   scf.yield %{{.*}} : i32
  // CHECK: } else {
  // CHECK:   scf.yield %{{.*}} : i32
  // CHECK: }
  gpu.func @promote_region_local_nested_register_state_to_vector_ssa(%out: !fly.ptr<i32, global>, %pred: i1) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %if_res = scf.if %pred -> (i32) {
      %local = fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<i32, register>
      fly.ptr.store(%c0_i32, %local) : (i32, !fly.ptr<i32, register>) -> ()

      scf.for %iv = %c0 to %c2 step %c1 {
        scf.while : () -> () {
          %cur = fly.ptr.load(%local) : (!fly.ptr<i32, register>) -> i32
          %cond = arith.cmpi slt, %cur, %c1_i32 : i32
          scf.condition(%cond)
        } do {
          %cur = fly.ptr.load(%local) : (!fly.ptr<i32, register>) -> i32
          %next = arith.addi %cur, %c1_i32 : i32
          fly.ptr.store(%next, %local) : (i32, !fly.ptr<i32, register>) -> ()
          scf.yield
        }
        scf.yield
      }

      %final = fly.ptr.load(%local) : (!fly.ptr<i32, register>) -> i32
      fly.ptr.store(%final, %out) : (i32, !fly.ptr<i32, global>) -> ()
      scf.yield %c0_i32 : i32
    } else {
      scf.yield %c1_i32 : i32
    }
    %use_if_res = arith.addi %if_res, %c0_i32 : i32
    gpu.return
  }
}
