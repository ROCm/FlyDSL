# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tensor copy with the CDNA5 TDM engine (gfx1250).

Each block copies its N-slab tile-by-tile down the M axis, reusing a single LDS box. It
shows that ``tdm_partition`` only relates mode-0 (the TDM box): here the coordinate tile
carries a ``K_TILES`` rest while the LDS tile carries a size-1 rest, so their *total* sizes
differ and only ``size<0>`` has to agree.

``TENSOR_LOAD_TO_LDS`` / ``TENSOR_STORE_FROM_LDS`` move a whole box on their own (no VGPRs,
no per-lane addressing, EXEC ignored), so a copy is the two of them back to back with an
``s_wait_tensorcnt`` between.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

TM, TN = 128, 64
K_TILES = 2  # tiles each block streams down M -- the coordinate-side "rest"


@flyc.kernel
def tdm_copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    tm: fx.Constexpr[int] = TM,
    tn: fx.Constexpr[int] = TN,
    k_tiles: fx.Constexpr[int] = K_TILES,
):
    # One packed LDS box, reused across the k tiles. Mode-0 is the box; a size-1 rest keeps
    # the view rank-2 like the coordinate tile, but its rest (1) differs from the coordinate's
    # (K) -- so the two total sizes differ while mode-0 matches.
    lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, tm * tn]).peek()
    box_layout = fx.make_layout((tm, tn), (tn, 1))
    smem_tensor = fx.make_view(lds.ptr, fx.make_layout(((tm, tn), 1), ((tn, 1), tm * tn)))

    tdm_load_atom, tdmA = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorLoad(), A, box_layout, (tm, tn))
    tdm_store_atom, tdmB = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorStore(), B, box_layout, (tm, tn))

    # ((tm, tn), K): keep all K tiles down M (None), pick this block's N-slab.
    blk_tdmA = fx.zipped_divide(tdmA, (tm, tn))[None, (None, fx.block_idx.x)]
    blk_tdmB = fx.zipped_divide(tdmB, (tm, tn))[None, (None, fx.block_idx.x)]

    warp_crd, warp_layout = 0, fx.make_layout(1, 1)

    # LDS rest = 1, coordinate rest = K_TILES -- different total sizes; only size<0> agrees.
    tAsA, tAgA = fx.rocdl.cdna5.tdm_partition(tdm_load_atom, warp_crd, warp_layout, smem_tensor, blk_tdmA)
    tBsB, tBgB = fx.rocdl.cdna5.tdm_partition(tdm_store_atom, warp_crd, warp_layout, smem_tensor, blk_tdmB)

    for k in fx.range_constexpr(k_tiles):
        fx.copy(tdm_load_atom, tAgA[None, k], tAsA[None, 0])  # global tile k -> LDS box
        fx.rocdl.s_wait_tensorcnt(0)
        fx.barrier()
        fx.copy(tdm_store_atom, tBsB[None, 0], tBgB[None, k])  # LDS box -> global tile k
        fx.rocdl.s_wait_tensorcnt(0)
        fx.barrier()


@flyc.jit
def tdm_tensor_copy(
    A: fx.Tensor,
    B: fx.Tensor,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    # One block per N-slab; each streams K_TILES tiles down M (so M must be K_TILES * TM).
    grid = ((n + TN - 1) // TN, 1, 1)
    tdm_copy_kernel(A, B).launch(grid=grid, block=(32, 1, 1), stream=stream)


NBLOCKS = 3
M, N = K_TILES * TM, NBLOCKS * TN  # 256 x 192

A = torch.arange(M * N, dtype=torch.int32).reshape(M, N).to(torch.float16).cuda()
B = torch.zeros(M, N, dtype=torch.float16).cuda()

tdm_tensor_copy(A, B, N)

torch.cuda.synchronize()
ok = torch.equal(A.cpu(), B.cpu())
print(f"Result correct: {ok}")
