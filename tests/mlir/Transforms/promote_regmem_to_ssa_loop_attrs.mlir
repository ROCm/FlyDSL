// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-promote-regmem-to-vectorssa | FileCheck %s

// The pass rebuilds every scf.for / scf.if / scf.while in a function it promotes, whether or not the
// op touches register state, and must keep their discardable attributes: `loop_annotation` (from
// range(..., unroll=)) is read when the loop is lowered to LLVM.

// CHECK-DAG: #[[UNROLL4:loop_unroll[0-9]*]] = #llvm.loop_unroll<count = 4 : i32>
// CHECK-DAG: #[[UNROLL_OFF:loop_unroll[0-9]*]] = #llvm.loop_unroll<disable = true>
// CHECK-DAG: #[[UNROLL2:loop_unroll[0-9]*]] = #llvm.loop_unroll<count = 2 : i32>
// CHECK-DAG: #[[UNROLL8:loop_unroll[0-9]*]] = #llvm.loop_unroll<count = 8 : i32>
// CHECK-DAG: #[[ANN4:loop_annotation[0-9]*]] = #llvm.loop_annotation<unroll = #[[UNROLL4]]>
// CHECK-DAG: #[[ANN_OFF:loop_annotation[0-9]*]] = #llvm.loop_annotation<unroll = #[[UNROLL_OFF]]>
// CHECK-DAG: #[[ANN2:loop_annotation[0-9]*]] = #llvm.loop_annotation<unroll = #[[UNROLL2]]>
// CHECK-DAG: #[[ANN8:loop_annotation[0-9]*]] = #llvm.loop_annotation<unroll = #[[UNROLL8]]>

// CHECK-LABEL: gpu.func @for_attrs
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<1xi32>) {
// CHECK: } {loop_annotation = #[[ANN4]]}
gpu.module @promote_rmem_loop_attrs {
  gpu.func @for_attrs(%out: !fly.ptr<i32, global>, %n: index) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %reg = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i32, register>
    fly.ptr.store(%c0_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    scf.for %i = %c0 to %n step %c1 {
      %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
      %next = arith.addi %cur, %c1_i32 : i32
      fly.ptr.store(%next, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    } {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    %final = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%final, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }

// CHECK-LABEL: gpu.func @nested_for_in_if_attrs
// CHECK: scf.if {{.*}} -> (vector<1xi32>) {
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<1xi32>) {
// CHECK: } {loop_annotation = #[[ANN_OFF]]}
// CHECK: } else {
// CHECK: } {if_marker}
  gpu.func @nested_for_in_if_attrs(%out: !fly.ptr<i32, global>, %n: index, %cond: i1) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %reg = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i32, register>
    fly.ptr.store(%c0_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    scf.if %cond {
      scf.for %i = %c0 to %n step %c1 {
        %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
        %next = arith.addi %cur, %c1_i32 : i32
        fly.ptr.store(%next, %reg) : (i32, !fly.ptr<i32, register>) -> ()
      } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    } {if_marker}
    %final = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%final, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }

// CHECK-LABEL: gpu.func @while_attrs
// CHECK: scf.while {{.*}} -> (i32, vector<1xi32>) {
// CHECK: } attributes {loop_annotation = #[[ANN2]]}
  gpu.func @while_attrs(%out: !fly.ptr<i32, global>, %n: i32) kernel {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %reg = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i32, register>
    fly.ptr.store(%c0_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    %r = scf.while (%arg = %c0_i32) : (i32) -> i32 {
      %cmp = arith.cmpi slt, %arg, %n : i32
      scf.condition(%cmp) %arg : i32
    } do {
    ^bb0(%arg1: i32):
      %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
      %next = arith.addi %cur, %c1_i32 : i32
      fly.ptr.store(%next, %reg) : (i32, !fly.ptr<i32, register>) -> ()
      %inc = arith.addi %arg1, %c1_i32 : i32
      scf.yield %inc : i32
    } attributes {loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    %final = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%final, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }

// CHECK-LABEL: gpu.func @stateless_for_attrs
// CHECK-NOT: register
// CHECK: scf.for {{.*}} {
// CHECK: } {loop_annotation = #[[ANN8]]}
  gpu.func @stateless_for_attrs(%out: !fly.ptr<i32, global>, %n: index) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %reg = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i32, register>
    fly.ptr.store(%c0_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    scf.for %i = %c0 to %n step %c1 {
      fly.ptr.store(%c0_i32, %out) : (i32, !fly.ptr<i32, global>) -> ()
    } {loop_annotation = #llvm.loop_annotation<unroll = <count = 8 : i32>>}
    %final = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%final, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }
}
