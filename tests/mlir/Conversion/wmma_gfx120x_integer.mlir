// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// GFX120X integer WMMA. Packed widths are the RDNA4 wave32 ISel classes:
//   IU4 16x16x16 A/B : i32            (NOT vector<1xi32>, NOT gfx11 vector<2xi32>)
//   IU8 16x16x16 A/B : vector<2xi32>
//   IU4 16x16x32 A/B : vector<2xi32>
//   C                 : vector<8xi32>
// Printer elides false sign/clamp attrs.

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_iu8_unsigned
func.func @test_gfx120x_wmma_atom_call_ssa_iu8_unsigned(
    %a: vector<8xui8>,
    %b: vector<8xui8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (ui8, ui8) -> i32, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<8xui8> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<8xui8> to vector<2xi32>
  // CHECK: rocdl.wmma.i32.16x16x16.iu8
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (ui8, ui8) -> i32, signA = false, signB = false, clamp = false>>, vector<8xui8>, vector<8xui8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_iu8_mixed_sign
func.func @test_gfx120x_wmma_atom_call_ssa_iu8_mixed_sign(
    %a: vector<8xi8>,
    %b: vector<8xi8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (i8, i8) -> i32, signA = true, signB = false, clamp = false>>
  // CHECK: rocdl.wmma.i32.16x16x16.iu8
  // CHECK-SAME: signA = true
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (i8, i8) -> i32, signA = true, signB = false, clamp = false>>, vector<8xi8>, vector<8xi8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_iu8_signed_clamp
func.func @test_gfx120x_wmma_atom_call_ssa_iu8_signed_clamp(
    %a: vector<8xsi8>,
    %b: vector<8xsi8>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (si8, si8) -> i32, signA = true, signB = true, clamp = true>>
  // CHECK: llvm.bitcast {{.*}} : vector<8xsi8> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<8xsi8> to vector<2xi32>
  // CHECK: rocdl.wmma.i32.16x16x16.iu8
  // CHECK-SAME: clamp = true
  // CHECK-SAME: signA = true
  // CHECK-SAME: signB = true
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (si8, si8) -> i32, signA = true, signB = true, clamp = true>>, vector<8xsi8>, vector<8xsi8>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_iu4_unsigned
func.func @test_gfx120x_wmma_atom_call_ssa_iu4_unsigned(
    %a: vector<8xi4>,
    %b: vector<8xi4>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<8xi4> to i32
  // CHECK: llvm.bitcast {{.*}} : vector<8xi4> to i32
  // CHECK: rocdl.wmma.i32.16x16x16.iu4
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>, vector<8xi4>, vector<8xi4>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_iu4_signed_clamp
func.func @test_gfx120x_wmma_atom_call_ssa_iu4_signed_clamp(
    %a: vector<8xi4>,
    %b: vector<8xi4>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (i4, i4) -> i32, signA = true, signB = true, clamp = true>>
  // CHECK: rocdl.wmma.i32.16x16x16.iu4
  // CHECK-SAME: clamp = true
  // CHECK-SAME: signA = true
  // CHECK-SAME: signB = true
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (i4, i4) -> i32, signA = true, signB = true, clamp = true>>, vector<8xi4>, vector<8xi4>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_iu4_k32_unsigned
func.func @test_gfx120x_wmma_atom_call_ssa_iu4_k32_unsigned(
    %a: vector<16xi4>,
    %b: vector<16xi4>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x32, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>
  // CHECK: llvm.bitcast {{.*}} : vector<16xi4> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<16xi4> to vector<2xi32>
  // CHECK: rocdl.wmma.i32.16x16x32.iu4
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x32, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>, vector<16xi4>, vector<16xi4>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_ssa_iu4_k32_signed_clamp
func.func @test_gfx120x_wmma_atom_call_ssa_iu4_k32_signed_clamp(
    %a: vector<16xi4>,
    %b: vector<16xi4>,
    %c: vector<8xi32>) -> vector<8xi32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x32, (i4, i4) -> i32, signA = true, signB = true, clamp = true>>
  // CHECK: llvm.bitcast {{.*}} : vector<16xi4> to vector<2xi32>
  // CHECK: rocdl.wmma.i32.16x16x32.iu4
  // CHECK-SAME: clamp = true
  // CHECK-SAME: signA = true
  // CHECK-SAME: signB = true
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x32, (i4, i4) -> i32, signA = true, signB = true, clamp = true>>, vector<16xi4>, vector<16xi4>, vector<8xi32>) -> vector<8xi32>
  return %res : vector<8xi32>
}

// CHECK-LABEL: @test_gfx120x_wmma_atom_call_iu4
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_gfx120x_wmma_atom_call_iu4(
    %d: !fly.memref<i32, register, 8:1>,
    %a: !fly.memref<i4, register, 8:1>,
    %b: !fly.memref<i4, register, 8:1>,
    %c: !fly.memref<i32, register, 8:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> i32
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> i32
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<8xi32>
  // CHECK: %[[RES:.*]] = rocdl.wmma.i32.16x16x16.iu4 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]]
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<8xi32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx120x.wmma<16x16x16, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>, !fly.memref<i32, register, 8:1>, !fly.memref<i4, register, 8:1>, !fly.memref<i4, register, 8:1>, !fly.memref<i32, register, 8:1>) -> ()
  return
}
