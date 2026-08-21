// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// Alignment carried into llvm.load/llvm.store by ptr.load/ptr.store lowering.

// -----

// A trivially swizzled pointer keeps the alignment its type promises.

// CHECK-LABEL: @test_align_no_swizzle
func.func @test_align_no_swizzle(%p: !fly.ptr<f16, shared, align<16>>) -> vector<4xf16> {
  // CHECK: llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf16>
  %v = fly.ptr.load(%p) : (!fly.ptr<f16, shared, align<16>>) -> vector<4xf16>
  return %v : vector<4xf16>
}

// -----

// A swizzle XORs address bits at and above `base`, so only the low `base` bits
// survive: align<16> under S<_,3,_> may only be reported as 8 bytes.

// CHECK-LABEL: @test_align_swizzle_weakens
func.func @test_align_swizzle_weakens(%p: !fly.ptr<f16, shared, align<16>, S<3,3,3>>) -> vector<4xf16> {
  // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<3> -> vector<4xf16>
  %v = fly.ptr.load(%p) : (!fly.ptr<f16, shared, align<16>, S<3,3,3>>) -> vector<4xf16>
  return %v : vector<4xf16>
}

// -----

// CHECK-LABEL: @test_align_swizzle_weakens_store
func.func @test_align_swizzle_weakens_store(%p: !fly.ptr<f16, shared, align<16>, S<3,3,3>>, %v: vector<4xf16>) {
  // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<4xf16>, !llvm.ptr<3>
  fly.ptr.store(%v, %p) : (vector<4xf16>, !fly.ptr<f16, shared, align<16>, S<3,3,3>>) -> ()
  return
}

// -----

// AlignAttr arithmetic can produce a non-power-of-two byte count, which
// llvm::Align rejects; the lowering must weaken it to its greatest
// power-of-two divisor.

// CHECK-LABEL: @test_align_non_power_of_two
func.func @test_align_non_power_of_two(%mem: !fly.memref<f32, global, 32:1, align<12>>) -> f32 {
  %idx = fly.make_int_tuple() : () -> !fly.int_tuple<3>
  // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> f32
  %val = fly.memref.load(%mem, %idx) : (!fly.memref<f32, global, 32:1, align<12>>, !fly.int_tuple<3>) -> f32
  return %val : f32
}
