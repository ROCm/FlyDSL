// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-nvvm | FileCheck %s

// NVVM SM80 mma.sync.aligned atom call lowering:
//   fly.mma_atom_call_ssa -> nvvm.mma.sync (m16n8k16, f16 -> f32)
// A is unpacked into 4 x vector<2xf16>, B into 2 x vector<2xf16>, C into 4 x f32,
// and the result struct is repacked into vector<4xf32>.

// CHECK-LABEL: @test_mma_sync_aligned_ssa
// CHECK-SAME: (%[[A:.*]]: vector<8xf16>, %[[B:.*]]: vector<4xf16>, %[[C:.*]]: vector<4xf32>)
func.func @test_mma_sync_aligned_ssa(
    %a: vector<8xf16>,
    %b: vector<4xf16>,
    %c: vector<4xf32>) -> vector<4xf32> {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_nvvm.sm80.mma.sync<16x8x16, (f16, f16) -> f32>>
  // CHECK: llvm.shufflevector %[[A]], %[[A]] [0, 1]
  // CHECK: llvm.shufflevector %[[A]], %[[A]] [2, 3]
  // CHECK: llvm.shufflevector %[[A]], %[[A]] [4, 5]
  // CHECK: llvm.shufflevector %[[A]], %[[A]] [6, 7]
  // CHECK: llvm.shufflevector %[[B]], %[[B]] [0, 1]
  // CHECK: llvm.shufflevector %[[B]], %[[B]] [2, 3]
  // CHECK: nvvm.mma.sync
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
  // CHECK-SAME: -> !llvm.struct<(f32, f32, f32, f32)>
  %res = fly.mma_atom_call_ssa(%atom, %a, %b, %c) : (!fly.mma_atom<!fly_nvvm.sm80.mma.sync<16x8x16, (f16, f16) -> f32>>, vector<8xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  return %res : vector<4xf32>
}
