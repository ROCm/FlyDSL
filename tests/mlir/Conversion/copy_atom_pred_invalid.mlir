// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --convert-fly-to-rocdl --verify-diagnostics

func.func @vector_pred(%src: f32, %dst: !fly.memref<f32, global, 1:1>, %pred: vector<2xi1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<32>, 32>
  // expected-error @+2 {{expected a scalar i1 predicate after SSA lowering}}
  // expected-error @+1 {{failed to legalize operation}}
  fly.copy_atom_call_ssa(%atom, %src, %dst, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly.universal_copy<32>, 32>, f32, !fly.memref<f32, global, 1:1>, vector<2xi1>) -> ()
  return
}

// -----

func.func @non_boolean_memref_pred(%src: f32, %dst: !fly.memref<f32, global, 1:1>, %pred: !fly.memref<i32, register, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<32>, 32>
  // expected-error @+2 {{expected an i1 register memref predicate with a lowered LLVM pointer}}
  // expected-error @+1 {{failed to legalize operation}}
  fly.copy_atom_call_ssa(%atom, %src, %dst, %pred) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (!fly.copy_atom<!fly.universal_copy<32>, 32>, f32, !fly.memref<f32, global, 1:1>, !fly.memref<i32, register, 1:1>) -> ()
  return
}
