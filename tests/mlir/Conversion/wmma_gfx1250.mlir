// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// gfx1250 plain WMMA atom lowering for the int4 (iu4) config:
//   16x16x32, (i4, i4) -> i32  =>  rocdl.wmma.i32.16x16x32.iu4
// A/B fragments are vector<2xi32> (16 i4 per lane), acc is vector<8xi32>.
// The iu4 intrinsic form carries signA/signB/clamp but no reuseA/reuseB.

// CHECK-LABEL: @test_wmma_iu4
func.func @test_wmma_iu4(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>) {
  %lay_ab = fly.static : !fly.layout<16:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<i32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<i4, register, 16:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<i4, register, 16:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<i32, register, 8:1>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<2xi32>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<2xi32>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi32>
  // Unsigned (default): the printer elides the false sign/clamp attrs.
  // CHECK: %[[RES:.*]] = rocdl.wmma.i32.16x16x32.iu4 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]] : (vector<2xi32>, vector<2xi32>, vector<8xi32>) -> vector<8xi32>
  // CHECK: llvm.store %[[RES]], %{{.*}} : vector<8xi32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (i4, i4) -> i32, signA = false, signB = false, clamp = false>>, !fly.memref<i32, register, 8:1>, !fly.memref<i4, register, 16:1>, !fly.memref<i4, register, 16:1>, !fly.memref<i32, register, 8:1>) -> ()
  return
}

// -----

// Signed int4 with clamp: signA/signB/clamp are forwarded to the intrinsic
// (the rocdl op printer only shows the non-false attrs).

// CHECK-LABEL: @test_wmma_iu4_signed_clamp
func.func @test_wmma_iu4_signed_clamp(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (i4, i4) -> i32, signA = true, signB = true, clamp = true>>) {
  %lay_ab = fly.static : !fly.layout<16:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<i32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<i4, register, 16:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<i4, register, 16:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<i32, register, 8:1>

  // CHECK: rocdl.wmma.i32.16x16x32.iu4
  // CHECK-SAME: clamp = true
  // CHECK-SAME: signA = true
  // CHECK-SAME: signB = true
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (i4, i4) -> i32, signA = true, signB = true, clamp = true>>, !fly.memref<i32, register, 8:1>, !fly.memref<i4, register, 16:1>, !fly.memref<i4, register, 16:1>, !fly.memref<i32, register, 8:1>) -> ()
  return
}

// -----

// gfx1250 plain WMMA atom lowering for the f16 config:
// 16x16x32, (f16, f16) -> f32  =>  rocdl.wmma.f32.16x16x32.f16
// A/B fragments are vector<16xf16>, acc is vector<8xf32>.

// CHECK-LABEL: @test_wmma_f16
func.func @test_wmma_f16(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>) {
  %lay_ab = fly.static : !fly.layout<16:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<f16, register, 16:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<f16, register, 16:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf32>
  
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x32.f16 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]] : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
  // CHECK: llvm.store %[[RES]], %{{.*}} : vector<8xf32>, !llvm.ptr<5>
  
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, 8:1>, !fly.memref<f16, register, 16:1>, !fly.memref<f16, register, 16:1>, !fly.memref<f32, register, 8:1>) -> ()
  
  return
}

// -----

// gfx1250 plain WMMA atom lowering for the bf16 config:
// 16x16x32, (bf16, bf16) -> f32  =>  rocdl.wmma.f32.16x16x32.bf16
// A/B fragments are vector<16xbf16>, acc is vector<8xf32>.

// CHECK-LABEL: @test_wmma_bf16
func.func @test_wmma_bf16(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>) {
  %lay_ab = fly.static : !fly.layout<16:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<bf16, register, 16:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<bf16, register, 16:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xbf16>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xbf16>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf32>
  
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x32.bf16 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]] : (vector<16xbf16>, vector<16xbf16>, vector<8xf32>) -> vector<8xf32>
  // CHECK: llvm.store %[[RES]], %{{.*}} : vector<8xf32>, !llvm.ptr<5>
  
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, 8:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<f32, register, 8:1>) -> ()
  
  return
}

// -----

// gfx1250 plain WMMA atom lowering for the f32 config:
// 16x16x4, (f32, f32) -> f32  =>  rocdl.wmma.f32.16x16x4.f32
// A/B fragments are vector<2xf32>, acc is vector<8xf32>.

// CHECK-LABEL: @test_wmma_f32
func.func @test_wmma_f32(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x4, (f32, f32) -> f32, signA = false, signB = false, clamp = false>>) {
  %lay_ab = fly.static : !fly.layout<2:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<2:1>) -> !fly.memref<f32, register, 2:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<2:1>) -> !fly.memref<f32, register, 2:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<2xf32>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<2xf32>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf32>
  
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x4.f32 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]] : (vector<2xf32>, vector<2xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK: llvm.store %[[RES]], %{{.*}} : vector<8xf32>, !llvm.ptr<5>
  
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x4, (f32, f32) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, 8:1>, !fly.memref<f32, register, 2:1>, !fly.memref<f32, register, 2:1>, !fly.memref<f32, register, 8:1>) -> ()
  
  return
}

// -----

// gfx1250 plain WMMA atom lowering for the i8 (iu8) config:
// 16x16x64, (i8, i8) -> i32  =>  rocdl.wmma.i32.16x16x64.iu8
// A/B fragments are vector<8xi32> (4 i8 per lane), acc is vector<8xi32>.

// CHECK-LABEL: @test_wmma_i8
func.func @test_wmma_i8(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x64, (i8, i8) -> i32, signA = false, signB = false, clamp = false>>) {
  %lay_ab = fly.static : !fly.layout<32:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<i32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<i8, register, 32:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<i8, register, 32:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<i32, register, 8:1>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi32>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi32>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi32>
  
  // CHECK: %[[RES:.*]] = rocdl.wmma.i32.16x16x64.iu8 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]] : (vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  // CHECK: llvm.store %[[RES]], %{{.*}} : vector<8xi32>, !llvm.ptr<5>
  
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x64, (i8, i8) -> i32, signA = false, signB = false, clamp = false>>, !fly.memref<i32, register, 8:1>, !fly.memref<i8, register, 32:1>, !fly.memref<i8, register, 32:1>, !fly.memref<i32, register, 8:1>) -> ()
  
  return
}

// -----

// gfx1250 plain WMMA atom lowering for the fp8 config:
// 16x16x64, (f8E4M3FN, f8E4M3FN) -> f32  =>  rocdl.wmma.f32.16x16x64.fp8_fp8
// A/B fragments are vector<8xi32> (4 fp8 per lane), acc is vector<8xf32>.

// CHECK-LABEL: @test_wmma_fp8
func.func @test_wmma_fp8(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x64, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>) {
  %lay_ab = fly.static : !fly.layout<32:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E4M3FN, register, 32:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<32:1>) -> !fly.memref<f8E4M3FN, register, 32:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<f32, register, 8:1>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi32>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi32>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf32>
  
  // CHECK: %[[RES:.*]] = rocdl.wmma.f32.16x16x64.fp8_fp8 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]] : (vector<8xi32>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  // CHECK: llvm.store %[[RES]], %{{.*}} : vector<8xf32>, !llvm.ptr<5>
  
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x64, (f8E4M3FN, f8E4M3FN) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, 8:1>, !fly.memref<f8E4M3FN, register, 32:1>, !fly.memref<f8E4M3FN, register, 32:1>, !fly.memref<f32, register, 8:1>) -> ()
  
  return
}
