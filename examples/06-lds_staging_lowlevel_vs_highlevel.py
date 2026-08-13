# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Two ways to stage a tile through LDS (global -> LDS -> registers -> global):
#
#   stage_lowlevel  — raw fx.ptr_load / fx.ptr_store with hand-computed offsets
#   stage_highlevel — a layouted LDS view driven by fx.copy over a TiledCopy
#
# Both kernels compute the identical result (B == A). The high-level version
# encodes the per-thread addressing in layout objects instead of arithmetic, so
# it is portable and lets the compiler reason about the access pattern. Companion
# to docs/flydsl_for_dummies/09_loads_stores_intrinsics.md.

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BM, BN = 8, 8  # one 8x8 = 64-element tile per block, one thread per element


# ---------- LOW-LEVEL: manual pointers + ptr_load / ptr_store ----------
@flyc.kernel
def stage_lowlevel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr):
    tid = fx.thread_idx.x  # 0..63 within the block
    bid = fx.block_idx.x  # which 8-column block
    r = tid // fx.Int32(BN)  # row inside the tile
    c = tid % fx.Int32(BN)  # column inside the tile
    # global (row-major MxN) index of this thread's element:
    g = r * fx.Int32(N) + (bid * fx.Int32(BN) + c)

    x = fx.ptr_load(fx.get_iter(A) + g)  # global -> reg  (global_load)
    smem = fx.get_dyn_shared(fx.Float32)  # raw LDS base pointer
    fx.ptr_store(x, smem + tid)  # reg -> LDS     (ds_write_b32)
    fx.gpu.barrier()
    y = fx.ptr_load(smem + tid)  # LDS -> reg     (ds_read_b32)
    fx.ptr_store(y, fx.get_iter(B) + g)  # reg -> global  (global_store)


# ---------- HIGH-LEVEL: layouted LDS view + fx.copy ----------
@flyc.kernel
def stage_highlevel(A: fx.Tensor, B: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    bA = fx.slice(fx.zipped_divide(A, (BM, BN)), (None, bid))  # this block's tile
    bB = fx.slice(fx.zipped_divide(B, (BM, BN)), (None, bid))

    # LDS staging buffer, given an explicit (BM, BN) row-major layout
    smem = fx.get_dyn_shared(fx.Float32)
    sT = fx.make_view(smem, fx.make_layout((BM, BN), (BN, 1)))

    # one thread per element; the TV layout encodes the tid -> (row, col) map
    thr_layout = fx.make_layout((BM, BN), (BN, 1))
    val_layout = fx.make_layout((1, 1), (1, 1))
    tile_mn, tv = fx.make_layout_tv(thr_layout, val_layout)

    gcopy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)  # global path
    scopy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)  # LDS path
    tc_g = fx.make_tiled_copy(gcopy, tv, tile_mn).get_slice(tid)
    tc_s = fx.make_tiled_copy(scopy, tv, tile_mn).get_slice(tid)

    frag = fx.make_fragment_like(tc_g.partition_S(bA))
    fx.copy(gcopy, tc_g.partition_S(bA), frag)  # global -> reg
    fx.copy(scopy, frag, tc_s.partition_D(sT))  # reg -> LDS
    fx.gpu.barrier()
    fx.copy(scopy, tc_s.partition_S(sT), frag)  # LDS -> reg
    fx.copy(gcopy, frag, tc_g.partition_D(bB))  # reg -> global


@flyc.jit
def run_low(A: fx.Tensor, B: fx.Tensor, nb: fx.Constexpr, N: fx.Constexpr, stream: fx.Stream = fx.Stream(None)):
    stage_lowlevel(A, B, N).launch(grid=(nb, 1, 1), block=(BM * BN, 1, 1), smem=BM * BN * 4, stream=stream)


@flyc.jit
def run_high(A: fx.Tensor, B: fx.Tensor, nb: fx.Constexpr, stream: fx.Stream = fx.Stream(None)):
    stage_highlevel(A, B).launch(grid=(nb, 1, 1), block=(BM * BN, 1, 1), smem=BM * BN * 4, stream=stream)


if __name__ == "__main__":
    M, N = BM, BN * 4  # 4 column-blocks
    A = torch.arange(M * N, dtype=torch.float32).reshape(M, N).cuda()

    B_low = torch.zeros(M, N, dtype=torch.float32).cuda()
    run_low(A, B_low, N // BN, N, stream=torch.cuda.Stream())

    B_high = torch.zeros(M, N, dtype=torch.float32).cuda()
    run_high(A, B_high, N // BN, stream=torch.cuda.Stream())

    torch.cuda.synchronize()
    print("low-level  correct:", torch.allclose(B_low, A))
    print("high-level correct:", torch.allclose(B_high, A))
    print("low == high        :", torch.allclose(B_low, B_high))
