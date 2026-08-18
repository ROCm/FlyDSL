# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Point-to-point tensor copy with the CDNA5 TDM engine (gfx1250).

``TENSOR_LOAD_TO_LDS`` and ``TENSOR_STORE_FROM_LDS`` move a whole tile between global
memory and LDS on their own: no VGPRs, no per-lane addressing, EXEC ignored. A copy is
therefore just the two of them back to back, with an ``s_wait_tensorcnt`` in between.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

TM, TN = 128, 64


@flyc.kernel
def tdm_copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    tm: fx.Constexpr[int] = TM,
    tn: fx.Constexpr[int] = TN,
):
    # The LDS tile is packed (no row padding), which is what the store needs: TDM
    # drains LDS with the tile stride and has no de-padding of its own.
    lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, tm * tn]).peek()
    smem_layout = fx.make_layout((tm, tn), (tn, 1))
    smem_tensor = fx.make_view(lds.ptr, smem_layout)

    # One atom per direction -- the direction is a property of the instruction.
    # Each carries its own tensor's pointer/stride/extent, so a caller cannot
    # pair an atom with a coordinate from a different tensor.
    tdm_load_atom, tdmA = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorLoad(), A, smem_layout, (tm, tn))
    tdm_store_atom, tdmB = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorStore(), B, smem_layout, (tm, tn))

    # Taking this block's tile is zipped_divide + slice.
    blk_tdmA = fx.zipped_divide(tdmA, (tm, tn))[None, (fx.block_idx.x, fx.block_idx.y)]
    blk_tdmB = fx.zipped_divide(tdmB, (tm, tn))[None, (fx.block_idx.x, fx.block_idx.y)]

    # One layout cuts both sides, so they keep describing the same elements.
    # There is no thread index in it -- TDM is issued by a single wave, so every
    # lane sees one partition.
    #
    #  The warp coordinate says which warp's share this is: one warp does the
    # whole tile here, so it is the trivial `0` over a size-1 layout. Splitting
    # the tile between N warps is `num_warps=N` on the atom and this warp's
    # index over `make_layout(N)` here.
    warp_crd, warp_layout = 0, fx.make_layout(1, 1)

    tAsA, tAgA = fx.rocdl.cdna5.tdm_partition(tdm_load_atom, warp_crd, warp_layout, smem_tensor, blk_tdmA)
    tBsB, tBgB = fx.rocdl.cdna5.tdm_partition(tdm_store_atom, warp_crd, warp_layout, smem_tensor, blk_tdmB)

    fx.copy(tdm_load_atom, tAgA, tAsA)
    fx.rocdl.s_wait_tensorcnt(0)
    fx.barrier()

    fx.copy(tdm_store_atom, tBsB, tBgB)
    fx.rocdl.s_wait_tensorcnt(0)


@flyc.jit
def tdm_tensor_copy(
    A: fx.Tensor,
    B: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    grid = ((m + TM - 1) // TM, (n + TN - 1) // TN, 1)
    tdm_copy_kernel(A, B).launch(grid=grid, block=(32, 1, 1), stream=stream)


# Deliberately not a multiple of the tile in either dim: the atom is built with
# `init_boundary_check=True` by default, so the descriptor clamps the ragged edge tiles and the
# lowering derives the clamp from the same coordinate that moved the base address.
M, N = 128 * 2 + 40, 64 * 3 + 24

A = torch.arange(M * N, dtype=torch.int32).reshape(M, N).to(torch.float16).cuda()
B = torch.zeros(M, N, dtype=torch.float16).cuda()

tdm_tensor_copy(A, B, M, N)

torch.cuda.synchronize()
ok = torch.equal(A.cpu(), B.cpu())
print(f"Result correct: {ok}")
