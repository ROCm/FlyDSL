# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CDNA5 TDM indexed addressing: what it accepts and what it refuses.

The limits come from where the row list lives. It occupies the two descriptor
groups a linear tile would spend on dims 2..4, which is why an indexed tile is 2D
only and why one issue carries a fixed number of rows: sixteen at 16-bit indices,
eight at 32-bit.
"""

import os

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

# TDM is gfx1250-only. Everything here is decided while tracing, so building without
# a device is enough; on any other card the kernels still have to be compiled for
# gfx1250, which is what COMPILE_ONLY allows.
_arch = get_rocm_arch() or ""
pytestmark.append(
    pytest.mark.skipif(
        not _arch.startswith("gfx1250") and os.environ.get("COMPILE_ONLY", "") in ("", "0"),
        reason=f"requires gfx1250 or COMPILE_ONLY, got {_arch}",
    )
)

WIDTH = 128
N = 256


def _build(rows, width=WIDTH, index_bits=32, tiler=None, index_dtype=torch.int32):
    """Trace a one-issue indexed load; return nothing, raise whatever it raises."""
    tile = tiler if tiler is not None else (rows, width)

    @flyc.kernel
    def k(src: fx.Tensor, idx: fx.Tensor):
        lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, rows * width]).peek()
        box = fx.make_layout((rows, width), (width, 1))
        smem = fx.make_view(lds.ptr, fx.make_layout(((rows, width), 1), ((width, 1), rows * width)))
        atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(), src, box, tile, index_tensor=idx, index_bits=index_bits
        )
        blk = fx.zipped_divide(coord, (rows, width))[None, (0, None)]
        tAs, tAg, tAi = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, [blk, idx])
        fx.copy(atom, [tAg[None, 0], tAi], tAs[None, 0])
        fx.rocdl.s_wait_tensorcnt(0)

    @flyc.jit
    def j(src: fx.Tensor, idx: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        k(src, idx).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    j(
        torch.zeros(64, N, dtype=torch.float16, device="cuda"),
        torch.zeros(rows if rows > 0 else 1, dtype=index_dtype, device="cuda"),
    )


def test_gather_builds():
    """Eight rows at 32-bit indices is the widest a single issue carries."""
    _build(rows=8)


def test_gather_builds_16bit():
    """Halving the index width doubles the rows one issue can name."""
    _build(rows=16, index_bits=16)


def test_too_many_rows_for_32bit():
    with pytest.raises(Exception, match="at most 8 rows"):
        _build(rows=16, index_bits=32)


def test_too_many_rows_for_16bit():
    with pytest.raises(Exception, match="at most 16 rows"):
        _build(rows=32, index_bits=16)


def test_rank_must_be_two():
    with pytest.raises(Exception, match="2D only"):
        _build(rows=8, tiler=(2, 4, WIDTH))


def test_index_width_must_be_16_or_32():
    with pytest.raises(Exception, match="index_bits must be 16 or 32"):
        _build(rows=8, index_bits=8)


def test_index_tensor_must_be_i32():
    with pytest.raises(Exception, match="Int32 row indices"):
        _build(rows=8, index_dtype=torch.int16)


def test_row_must_span_whole_dwords():
    """A 2-element f16 row is 4 bytes; a 1-element row is 2 and cannot be encoded."""
    _build(rows=8, width=2)
    with pytest.raises(Exception, match="whole multiple of 4 bytes"):
        _build(rows=8, width=1)


def test_partition_rejects_a_malformed_pair():
    with pytest.raises(Exception, match=r"\[coord, index\]"):

        @flyc.kernel
        def k(src: fx.Tensor, idx: fx.Tensor):
            lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, 8 * WIDTH]).peek()
            box = fx.make_layout((8, WIDTH), (WIDTH, 1))
            smem = fx.make_view(lds.ptr, fx.make_layout(((8, WIDTH), 1), ((WIDTH, 1), 8 * WIDTH)))
            atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoad(), src, box, (8, WIDTH), index_tensor=idx
            )
            blk = fx.zipped_divide(coord, (8, WIDTH))[None, (0, None)]
            fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, [blk, idx, idx])

        @flyc.jit
        def j(src: fx.Tensor, idx: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            k(src, idx).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

        j(
            torch.zeros(64, N, dtype=torch.float16, device="cuda"),
            torch.zeros(8, dtype=torch.int32, device="cuda"),
        )
