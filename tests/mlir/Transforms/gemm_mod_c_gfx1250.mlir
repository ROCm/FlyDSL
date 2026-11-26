// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --fly-rocdl-expand-ops --verify-diagnostics

// A gfx1250 WMMA C modifier applies to every WMMA, and fly.gemm passes a tile's partial sum as C
// after its first K step, so fly.gemm rejects a C modifier with more than one K step per (m, n) tile.

func.func @wmma_two_k_steps(
    %atom: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false, modC = 1>>,
    %acc: !fly.memref<f32, register, (8,1,1):(1,0,0)>, %a: !fly.memref<bf16, register, (16,1,2):(1,0,16)>,
    %b: !fly.memref<bf16, register, (16,1,2):(1,0,16)>) {
  // expected-error@+1 {{a gfx1250 WMMA C modifier (mod_c = 1) applies to every WMMA, and this GEMM has 2 K steps per (m, n) tile}}
  fly.gemm(%atom, %acc, %a, %b, %acc) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false, modC = 1>>, !fly.memref<f32, register, (8,1,1):(1,0,0)>, !fly.memref<bf16, register, (16,1,2):(1,0,16)>, !fly.memref<bf16, register, (16,1,2):(1,0,16)>, !fly.memref<f32, register, (8,1,1):(1,0,0)>) -> ()
  return
}

// -----

func.func @tiled_wmma_scale_two_k_steps(
    %mma: !fly.tiled_mma<!fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, modC = 3, reuseA = false, reuseB = false, blockSize = 32>>, <(1,1,1):(0,0,0)>>,
    %acc: !fly.memref<f32, register, (8,2,1):(1,8,0)>, %a: !fly.memref<f8E4M3FN, register, (64,2,2):(1,64,128)>,
    %b: !fly.memref<f8E4M3FN, register, (64,1,2):(1,0,64)>) {
  // expected-error@+1 {{a gfx1250 WMMA C modifier (mod_c = 3) applies to every WMMA, and this GEMM has 2 K steps per (m, n) tile}}
  fly.gemm(%mma, %acc, %a, %b, %acc) : (!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, modC = 3, reuseA = false, reuseB = false, blockSize = 32>>, <(1,1,1):(0,0,0)>>, !fly.memref<f32, register, (8,2,1):(1,8,0)>, !fly.memref<f8E4M3FN, register, (64,2,2):(1,64,128)>, !fly.memref<f8E4M3FN, register, (64,1,2):(1,0,64)>, !fly.memref<f32, register, (8,2,1):(1,8,0)>) -> ()
  return
}

// -----

// Accepted: one K step per tile (rank-3 with K = 1, rank-2 A and B, rank-1), and no C modifier with two K steps.
func.func @accepted(
    %neg: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false, modC = 1>>,
    %plain: !fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>,
    %acc: !fly.memref<f32, register, (8,2,1):(1,8,0)>, %a1: !fly.memref<bf16, register, (16,2,1):(1,16,0)>,
    %b1: !fly.memref<bf16, register, (16,1,1):(1,0,0)>, %a2: !fly.memref<bf16, register, (16,2,2):(1,16,32)>,
    %b2: !fly.memref<bf16, register, (16,1,2):(1,0,16)>, %acc0: !fly.memref<f32, register, 8:1>,
    %a0: !fly.memref<bf16, register, 16:1>, %b0: !fly.memref<bf16, register, 16:1>,
    %am: !fly.memref<bf16, register, (16,2):(1,16)>, %bn: !fly.memref<bf16, register, (16,1):(1,0)>) {
  fly.gemm(%neg, %acc, %a1, %b1, %acc) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false, modC = 1>>, !fly.memref<f32, register, (8,2,1):(1,8,0)>, !fly.memref<bf16, register, (16,2,1):(1,16,0)>, !fly.memref<bf16, register, (16,1,1):(1,0,0)>, !fly.memref<f32, register, (8,2,1):(1,8,0)>) -> ()
  fly.gemm(%neg, %acc, %am, %bn, %acc) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false, modC = 1>>, !fly.memref<f32, register, (8,2,1):(1,8,0)>, !fly.memref<bf16, register, (16,2):(1,16)>, !fly.memref<bf16, register, (16,1):(1,0)>, !fly.memref<f32, register, (8,2,1):(1,8,0)>) -> ()
  fly.gemm(%neg, %acc0, %a0, %b0, %acc0) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false, modC = 1>>, !fly.memref<f32, register, 8:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<bf16, register, 16:1>, !fly.memref<f32, register, 8:1>) -> ()
  fly.gemm(%plain, %acc, %a2, %b2, %acc) : (!fly.mma_atom<!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>>, !fly.memref<f32, register, (8,2,1):(1,8,0)>, !fly.memref<bf16, register, (16,2,2):(1,16,32)>, !fly.memref<bf16, register, (16,1,2):(1,0,16)>, !fly.memref<f32, register, (8,2,1):(1,8,0)>) -> ()
  return
}
