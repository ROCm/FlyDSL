// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// RDNA 3 / 3.5 WMMA atom-call lowering tests.
// The wave32, M=N=K=16-only WMMA atom from `lib/Dialect/FlyROCDL/RDNA3/MmaAtom.cpp`
// must lower to the corresponding ROCDL `wmma.*.16x16x16.*` op. Each test
// covers one of the six supported (A,B,Acc) dtype combinations.

// CHECK-LABEL: @test_rdna3_wmma_f32_f16
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_rdna3_wmma_f32_f16(
    %d: !fly.memref<f32, register, 8:1>,
    %a: !fly.memref<f16, register, 8:1>,
    %b: !fly.memref<f16, register, 8:1>,
    %c: !fly.memref<f32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (f16, f16) -> f32>>
  // The atom loads <8 x f16> per-lane operands (FlyDSL invariant) and
  // shufflevector-duplicates them to <16 x f16> for the gfx11 WMMA256b
  // intrinsic. The accumulator is <8 x f32>.
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<8xf16>
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<8xf16>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xf32>
  // CHECK: llvm.shufflevector {{.*}} : vector<8xf16>
// CHECK: rocdl.wmma.f32.16x16x16.f16 {{.*}} : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
// CHECK: llvm.store {{.*}}, %[[D]] : vector<8xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (f16, f16) -> f32>>, !fly.memref<f32, register, 8:1>, !fly.memref<f16, register, 8:1>, !fly.memref<f16, register, 8:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_rdna3_wmma_f32_bf16
// CHECK: rocdl.wmma.f32.16x16x16.bf16
func.func @test_rdna3_wmma_f32_bf16(
    %d: !fly.memref<f32, register, 8:1>,
    %a: !fly.memref<bf16, register, 8:1>,
    %b: !fly.memref<bf16, register, 8:1>,
    %c: !fly.memref<f32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (bf16, bf16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (bf16, bf16) -> f32>>, !fly.memref<f32, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<f32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_rdna3_wmma_f16_f16
// For stable multi-K accumulation semantics, f16-acc lowers through the f32
// WMMA op and truncates lane-local results back to f16.
// CHECK: rocdl.wmma.f32.16x16x16.f16
// CHECK: llvm.fptrunc {{.*}} : vector<8xf32> to vector<8xf16>
func.func @test_rdna3_wmma_f16_f16(
    %d: !fly.memref<f16, register, 8:1>,
    %a: !fly.memref<f16, register, 8:1>,
    %b: !fly.memref<f16, register, 8:1>,
    %c: !fly.memref<f16, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (f16, f16) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (f16, f16) -> f16>>, !fly.memref<f16, register, 8:1>, !fly.memref<f16, register, 8:1>, !fly.memref<f16, register, 8:1>, !fly.memref<f16, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_rdna3_wmma_i32_iu8
// CHECK: rocdl.wmma.i32.16x16x16.iu8
func.func @test_rdna3_wmma_i32_iu8(
    %d: !fly.memref<i32, register, 8:1>,
    %a: !fly.memref<i8, register, 8:1>,
    %b: !fly.memref<i8, register, 8:1>,
    %c: !fly.memref<i32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (i8, i8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (i8, i8) -> i32>>, !fly.memref<i32, register, 8:1>, !fly.memref<i8, register, 8:1>, !fly.memref<i8, register, 8:1>, !fly.memref<i32, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_rdna3_wmma_bf16_bf16
// For stable multi-K accumulation semantics, bf16-acc lowers through the f32
// WMMA op (BF16 has the same exponent range as f32, no overflow risk) and
// truncates lane-local results back to bf16.
// CHECK: rocdl.wmma.f32.16x16x16.bf16
// CHECK: llvm.fptrunc {{.*}} : vector<8xf32> to vector<8xbf16>
func.func @test_rdna3_wmma_bf16_bf16(
    %d: !fly.memref<bf16, register, 8:1>,
    %a: !fly.memref<bf16, register, 8:1>,
    %b: !fly.memref<bf16, register, 8:1>,
    %c: !fly.memref<bf16, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (bf16, bf16) -> bf16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (bf16, bf16) -> bf16>>, !fly.memref<bf16, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<bf16, register, 8:1>, !fly.memref<bf16, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_rdna3_wmma_i32_iu4
// CHECK: rocdl.wmma.i32.16x16x16.iu4
func.func @test_rdna3_wmma_i32_iu4(
    %d: !fly.memref<i32, register, 8:1>,
    %a: !fly.memref<i4, register, 8:1>,
    %b: !fly.memref<i4, register, 8:1>,
    %c: !fly.memref<i32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (i4, i4) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.rdna3.wmma<16x16x16, (i4, i4) -> i32>>, !fly.memref<i32, register, 8:1>, !fly.memref<i4, register, 8:1>, !fly.memref<i4, register, 8:1>, !fly.memref<i32, register, 8:1>) -> ()
  return
}
