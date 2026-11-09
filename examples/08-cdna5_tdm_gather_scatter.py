# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CDNA5 TDM indexed addressing: gather rows in, scatter the same rows back out.

A plain TDM tile walks its outer dimension with a coordinate. An indexed tile does
not walk it at all -- it is handed a list of row numbers, one per row of the tile.
A load gathers those rows into LDS; a store scatters LDS back to them. That makes
it the right tool whenever the rows a block needs are scattered through the tensor,
such as a token table or a paged cache.

The copy op selects the mode -- ``TensorLoadGather`` / ``TensorStoreScatter`` instead
of ``TensorLoad`` / ``TensorStore`` -- and ``index_tensor`` supplies the row list. The
list then travels beside the coordinate: ``tdm_partition`` hands back a third tensor,
and ``copy`` takes the pair as ``[coord, index]``.

Both forms issue the same TDM instruction; only the descriptor differs. One issue
carries 1..8 rows at 32-bit indices, 1..16 at 16-bit.

Run with COMPILE_ONLY=1 ARCH=gfx1250 to build it without the hardware.
"""

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

ROWS = 8  # rows per issue; 32-bit indices carry at most 8
WIDTH = 128  # elements per row, must span whole 4-byte units
N = 256  # row width of the tensors
M_ROWS = 64  # rows of the tensors


@flyc.kernel
def gather_scatter(src: fx.Tensor, dst: fx.Tensor, rows: fx.Tensor):
    lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, ROWS * WIDTH]).peek()
    tile = fx.make_layout((ROWS, WIDTH), (WIDTH, 1))
    smem = fx.make_view(lds.ptr, fx.make_layout(((ROWS, WIDTH), 1), ((WIDTH, 1), ROWS * WIDTH)))

    # The row list, shaped like the data it indexes: one index per row, the same index
    # for every column. The data's tiler then cuts the list too.
    gI = fx.Tensor(fx.make_view(rows.iter, fx.make_layout((M_ROWS, N), (1, 0))))

    # Gather: the atom reads `gI` to learn which source rows to collect.
    load_atom, load_coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
        fx.rocdl.TensorLoadGather(), src, tile, (ROWS, WIDTH), index_tensor=gI
    )
    load_blk = fx.zipped_divide(load_coord, (ROWS, WIDTH))[None, (0, None)]
    load_blkI = fx.zipped_divide(gI, (ROWS, WIDTH))[None, (0, None)]
    tAs, tAg, tAi = fx.rocdl.cdna5.tdm_partition(load_atom, 0, fx.make_layout(1, 1), smem, [load_blk, load_blkI])
    # One coordinate addresses the tile and its row list together.
    fx.copy(load_atom, [tAg[None, 0], tAi[None, 0]], tAs[None, 0])
    fx.rocdl.s_wait_tensorcnt(0)

    # Scatter: the same row list, now naming where the tile lands.
    store_atom, store_coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
        fx.rocdl.TensorStoreScatter(), dst, tile, (ROWS, WIDTH), index_tensor=gI
    )
    store_blk = fx.zipped_divide(store_coord, (ROWS, WIDTH))[None, (0, None)]
    store_blkI = fx.zipped_divide(gI, (ROWS, WIDTH))[None, (0, None)]
    tBs, tBg, tBi = fx.rocdl.cdna5.tdm_partition(store_atom, 0, fx.make_layout(1, 1), smem, [store_blk, store_blkI])
    fx.copy(store_atom, tBs[None, 0], [tBg[None, 0], tBi[None, 0]])
    fx.rocdl.s_wait_tensorcnt(0)


@flyc.jit
def run(src: fx.Tensor, dst: fx.Tensor, rows: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    gather_scatter(src, dst, rows).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


if __name__ == "__main__":
    src = torch.arange(64 * N, dtype=torch.float16, device="cuda").reshape(64, N)
    dst = torch.zeros_like(src)
    # Any rows, in any order: that is the point of an index list.
    picked = [3, 1, 17, 5, 40, 2, 9, 33]
    rows = torch.tensor(picked, dtype=torch.int32, device="cuda")
    run(src, dst, rows)
    torch.cuda.synchronize()

    if os.environ.get("COMPILE_ONLY", "") not in ("", "0"):
        # The kernel was built but never ran, so dst still holds its initial value and
        # comparing it would report a failure the build did not have.
        print("compiled; results not checked under COMPILE_ONLY")
        raise SystemExit(0)

    # The gathered rows land in LDS in list order and the scatter puts each one back
    # where it came from, so every picked row survives the round trip unchanged.
    gathered_ok = torch.equal(dst[picked], src[picked])
    untouched = torch.ones(src.shape[0], dtype=torch.bool, device="cuda")
    untouched[picked] = False
    untouched_ok = bool(dst[untouched].eq(0).all())
    print(f"gathered rows match: {gathered_ok}")
    print(f"other rows untouched: {untouched_ok}")
    print(f"Result correct: {gathered_ok and untouched_ok}")
