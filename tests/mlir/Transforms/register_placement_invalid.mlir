// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

func.func @negative_start(%p: !fly.ptr<i32, register>) {
  // expected-error @+1 {{start must be a nonnegative register class index}}
  fly.set_register %p {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = -1 : i64} : !fly.ptr<i32, register>
  return
}

// -----

func.func @wrong_storage(%p: !fly.ptr<i32, global>) {
  // expected-error @+1 {{requires a register-memory pointer or tensor}}
  fly.set_register %p {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 0 : i64} : !fly.ptr<i32, global>
  return
}

// -----

func.func @slice_outside(%v: vector<4xi32>) {
  // expected-error @+1 {{slice exceeds register storage size}}
  %result = fly.register_value %v {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 0 : i64, bitOffset = 64 : i64, storageBits = 128 : i64} : vector<4xi32>
  return
}

// -----

// expected-error @+1 {{register class target and name must be nonempty}}
module attributes {test = #fly.register_class<"", "VGPR_32">} {}
