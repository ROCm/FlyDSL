// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// gfx1250 plain WMMA atom lowering for the int4 (iu4) config:
//   16x16x32, (i4, i4) -> i32  =>  rocdl.wmma.i32.16x16x32.iu4
// A/B fragments are vector<2xi32> (16 i4 per lane), acc is vector<8xi32>.
// The iu4 intrinsic form carries signA/signB/clamp but no reuseA/reuseB.

// CHECK-LABEL: @test_wmma_iu4
func.func @test_wmma_iu4(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (i4, i4) -> i32>>) {
  %lay_ab = fly.static : !fly.layout<16:1>
  %lay_cd = fly.static : !fly.layout<8:1>
  %d = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<i32, register, 8:1>
  %a = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<i4, register, 16:1>
  %b = fly.memref.alloca(%lay_ab) : (!fly.layout<16:1>) -> !fly.memref<i4, register, 16:1>
  %c = fly.memref.alloca(%lay_cd) : (!fly.layout<8:1>) -> !fly.memref<i32, register, 8:1>

  // CHECK-DAG: %[[A_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<2xi32>
  // CHECK-DAG: %[[B_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<2xi32>
  // CHECK-DAG: %[[C_VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi32>
  // CHECK: %[[RES:.*]] = rocdl.wmma.i32.16x16x32.iu4 %[[A_VAL]], %[[B_VAL]], %[[C_VAL]] : (vector<2xi32>, vector<2xi32>, vector<8xi32>) -> vector<8xi32>
  // CHECK: llvm.store %[[RES]], %{{.*}} : vector<8xi32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (i4, i4) -> i32>>, !fly.memref<i32, register, 8:1>, !fly.memref<i4, register, 16:1>, !fly.memref<i4, register, 16:1>, !fly.memref<i32, register, 8:1>) -> ()
  return
}
