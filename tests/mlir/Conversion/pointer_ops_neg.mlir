// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --verify-diagnostics

// === ToLLVMPtr verifier ===

func.func @test_to_llvm_ptr_requires_llvm_pointer(%ptr: !fly.ptr<f32, global>) -> i32 {
  // expected-error @+1 {{LLVM pointer type}}
  %r = fly.to_llvm_ptr(%ptr) {llvm_address_space = 1 : i32} : (!fly.ptr<f32, global>) -> i32
  return %r : i32
}
