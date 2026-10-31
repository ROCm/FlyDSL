# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Double-buffered streaming copy with the CDNA5 TDM engine (gfx1250).

Each block streams ``K_TILES`` (tm, tn) tiles down the M axis of its N-slab, ping-ponging
through ``PIPE`` LDS buffers. It exercises the mode-0 partition convention:
``tdm_partition`` relates only mode-0 (the TDM box) and lets the two operands keep
*different* trailing "rest" modes -- here the LDS side carries a ``PIPE`` rest while the
coordinate side carries a ``k-tiles`` rest, and only ``size<0>`` has to agree.

``TENSOR_LOAD_TO_LDS`` / ``TENSOR_STORE_FROM_LDS`` move a whole box on their own (no VGPRs,
no per-lane addressing, EXEC ignored), so a copy is the two of them back to back with an
``s_wait_tensorcnt`` between.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

TM, TN = 128, 64
K_TILES = 4  # (tm, tn) tiles each block streams down M -- the coordinate-side "rest"
PIPE = 2  # LDS double-buffer stages -- the smem-side "rest"


@flyc.kernel
def tdm_stream_copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    tm: fx.Constexpr[int] = TM,
    tn: fx.Constexpr[int] = TN,
    k_tiles: fx.Constexpr[int] = K_TILES,
    pipe: fx.Constexpr[int] = PIPE,
):
    # LDS double-buffer: PIPE packed (tm, tn) boxes back to back. The box is packed (no row
    # padding), which the store needs -- TDM drains LDS with the tile stride, no de-padding.
    lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, pipe * tm * tn]).peek()
    box_layout = fx.make_layout((tm, tn), (tn, 1))  # one packed box
    # smem view: mode-0 = the box (grouped), mode-1 = PIPE -- the LDS-side "rest".
    smem_tensor = fx.make_view(lds.ptr, fx.make_layout(((tm, tn), pipe), ((tn, 1), tm * tn)))

    # One atom per direction; each carries its own tensor's pointer/stride/extent.
    tdm_load_atom, tdmA = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorLoad(), A, box_layout, (tm, tn))
    tdm_store_atom, tdmB = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorStore(), B, box_layout, (tm, tn))

    # This block's column of tiles: zipped_divide -> ((tm,tn), (K, n_blocks)); keep all K
    # tiles down M (None), pick this block's N-slab. Mode-1 = K is the coordinate-side "rest".
    blk_tdmA = fx.zipped_divide(tdmA, (tm, tn))[None, (None, fx.block_idx.x)]  # ((tm,tn), K)
    blk_tdmB = fx.zipped_divide(tdmB, (tm, tn))[None, (None, fx.block_idx.x)]

    # One warp does the copy; no thread index in the partition.
    warp_crd, warp_layout = 0, fx.make_layout(1, 1)

    # LDS rest = PIPE, coordinate rest = K_TILES -- different sizes; only size<0> is checked.
    tAsA, tAgA = fx.rocdl.cdna5.tdm_partition(tdm_load_atom, warp_crd, warp_layout, smem_tensor, blk_tdmA)
    tBsB, tBgB = fx.rocdl.cdna5.tdm_partition(tdm_store_atom, warp_crd, warp_layout, smem_tensor, blk_tdmB)

    for k in fx.range_constexpr(k_tiles):
        stage = k % pipe
        fx.copy(tdm_load_atom, tAgA[None, k], tAsA[None, stage])  # global tile k -> LDS stage
        fx.rocdl.s_wait_tensorcnt(0)
        fx.barrier()
        fx.copy(tdm_store_atom, tBsB[None, stage], tBgB[None, k])  # LDS stage -> global tile k
        fx.rocdl.s_wait_tensorcnt(0)
        fx.barrier()


@flyc.jit
def tdm_stream_copy(
    A: fx.Tensor,
    B: fx.Tensor,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    # One block per N-slab; each streams K_TILES tiles down M (so M must be K_TILES * TM).
    grid = ((n + TN - 1) // TN, 1, 1)
    tdm_stream_copy_kernel(A, B).launch(grid=grid, block=(32, 1, 1), stream=stream)


# Exact multiples: M is K_TILES tiles down, N is a few TN-slabs across.
NBLOCKS = 3
M, N = K_TILES * TM, NBLOCKS * TN  # 512 x 192

A = torch.arange(M * N, dtype=torch.int32).reshape(M, N).to(torch.float16).cuda()
B = torch.zeros(M, N, dtype=torch.float16).cuda()

tdm_stream_copy(A, B, N)

torch.cuda.synchronize()
ok = torch.equal(A.cpu(), B.cpu())
print(f"Result correct: {ok}")
