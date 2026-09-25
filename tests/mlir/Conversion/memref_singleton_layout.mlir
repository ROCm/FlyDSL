// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// Tests for singleton (size-one) dimensions in memref load_vec/store_vec
// contiguous-segment detection: a size-one stride-1 dim must be ignored when
// locating the contiguous segment, so layouts like (1,4):(1,1) resolve as a
// single vector of 4 instead of being rejected as ambiguous.

// ---

// Size-one dim first: (1,4):(1,1) → contiguous segment is leaf dim 1 (width 4),
// a single vector load, no chunking or permutation.
// CHECK-LABEL: @test_load_vec_singleton_first
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>)
func.func @test_load_vec_singleton_first(%mem: !fly.memref<f32, register, (1, 4):(1, 1)>) -> vector<4xf32> {
  // CHECK: %[[VEC:.*]] = llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<5> -> vector<4xf32>
  %vec = fly.memref.load_vec(%mem) : (!fly.memref<f32, register, (1, 4):(1, 1)>) -> vector<4xf32>
  // CHECK: return %[[VEC]]
  return %vec : vector<4xf32>
}

// CHECK-LABEL: @test_store_vec_singleton_first
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>, %[[VEC:.*]]: vector<4xf32>)
func.func @test_store_vec_singleton_first(%mem: !fly.memref<f32, register, (1, 4):(1, 1)>, %vec: vector<4xf32>) {
  // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
  fly.memref.store_vec(%vec, %mem) : (vector<4xf32>, !fly.memref<f32, register, (1, 4):(1, 1)>) -> ()
  return
}

// ---

// Size-one dim last: (4,1):(1,1) → contiguous segment is leaf dim 0 (width 4).
// CHECK-LABEL: @test_load_vec_singleton_last
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>)
func.func @test_load_vec_singleton_last(%mem: !fly.memref<f32, register, (4, 1):(1, 1)>) -> vector<4xf32> {
  // CHECK: %[[VEC:.*]] = llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<5> -> vector<4xf32>
  %vec = fly.memref.load_vec(%mem) : (!fly.memref<f32, register, (4, 1):(1, 1)>) -> vector<4xf32>
  // CHECK: return %[[VEC]]
  return %vec : vector<4xf32>
}

// CHECK-LABEL: @test_store_vec_singleton_last
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<5>, %[[VEC:.*]]: vector<4xf32>)
func.func @test_store_vec_singleton_last(%mem: !fly.memref<f32, register, (4, 1):(1, 1)>, %vec: vector<4xf32>) {
  // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
  fly.memref.store_vec(%vec, %mem) : (vector<4xf32>, !fly.memref<f32, register, (4, 1):(1, 1)>) -> ()
  return
}
