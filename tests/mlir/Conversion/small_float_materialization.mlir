// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-rocdl | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// Small floats are converted to same-width integers, while the ops around them
// stay legal and keep the float type; arith.bitcast bridges the two (issue #1211).

// CHECK-LABEL: func.func @fp8_loop
// CHECK: %[[INIT:.*]] = arith.bitcast %{{.*}} : vector<64xf8E4M3FNUZ> to vector<64xi8>
// CHECK: %[[OUT:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (vector<64xi8>)
// CHECK: arith.bitcast %[[OUT]] : vector<64xi8> to vector<64xf8E4M3FNUZ>
func.func @fp8_loop(%input: vector<64xi8>, %count: index) -> vector<64xi8> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %fp8 = vector.bitcast %input : vector<64xi8> to vector<64xf8E4M3FNUZ>
  %out = scf.for %i = %c0 to %count step %c1 iter_args(%state = %fp8) -> (vector<64xf8E4M3FNUZ>) {
    scf.yield %state : vector<64xf8E4M3FNUZ>
  }
  %result = vector.bitcast %out : vector<64xf8E4M3FNUZ> to vector<64xi8>
  return %result : vector<64xi8>
}

// CHECK-LABEL: func.func @fp8_ptr_load
// CHECK: %[[LOADED:.*]] = llvm.load {{.*}} -> vector<4xi8>
// CHECK: arith.bitcast %[[LOADED]] : vector<4xi8> to vector<4xf8E4M3FN>
func.func @fp8_ptr_load(%p: !fly.ptr<f8E4M3FN, global>) -> vector<4xi8> {
  %v = fly.ptr.load(%p) : (!fly.ptr<f8E4M3FN, global>) -> vector<4xf8E4M3FN>
  %r = vector.bitcast %v : vector<4xf8E4M3FN> to vector<4xi8>
  return %r : vector<4xi8>
}
