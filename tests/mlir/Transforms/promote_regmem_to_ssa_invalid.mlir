// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-promote-regmem-to-vectorssa | FileCheck %s

// CHECK-LABEL: gpu.func @skip_register_recast_iter
// CHECK: fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<f32, register>
// CHECK: fly.recast_iter
// CHECK: fly.ptr.load
// CHECK: fly.ptr.store
gpu.module @promote_regmem_to_ssa_recast_skip {
  gpu.func @skip_register_recast_iter(%out: !fly.ptr<i32, global>) kernel {
    %reg = fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<f32, register>
    %recast = "fly.recast_iter"(%reg) : (!fly.ptr<f32, register>) -> !fly.ptr<i32, register>
    %val = fly.ptr.load(%recast) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%val, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }

// CHECK-LABEL: gpu.func @mixed_recast_and_normal
// CHECK: fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<f32, register>
// CHECK: fly.recast_iter
// CHECK: fly.ptr.load(%{{.*}}) : (!fly.ptr<i32, register>) -> i32
// CHECK-NOT: !fly.ptr<i32, register, 1 : i64>
// CHECK: ub.poison : vector<1xi32>
// CHECK: %[[STATE:.*]] = vector.insert %{{.*}}, %{{.*}} [0] : i32 into vector<1xi32>
// CHECK: %[[FINAL:.*]] = vector.extract %[[STATE]][0] : i32 from vector<1xi32>
  gpu.func @mixed_recast_and_normal(%out: !fly.ptr<i32, global>) kernel {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %recast_reg = fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<f32, register>
    %recast = "fly.recast_iter"(%recast_reg) : (!fly.ptr<f32, register>) -> !fly.ptr<i32, register>
    %recast_val = fly.ptr.load(%recast) : (!fly.ptr<i32, register>) -> i32

    %normal_reg = fly.make_ptr() {dictAttrs = {allocaSize = 1 : i64}} : () -> !fly.ptr<i32, register>
    fly.ptr.store(%c1_i32, %normal_reg) : (i32, !fly.ptr<i32, register>) -> ()
    %normal_val = fly.ptr.load(%normal_reg) : (!fly.ptr<i32, register>) -> i32

    %sum = arith.addi %recast_val, %normal_val : i32
    fly.ptr.store(%sum, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }
}
