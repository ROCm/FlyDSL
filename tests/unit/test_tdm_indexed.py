# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CDNA5 TDM indexed addressing: what the gather / scatter ops accept and refuse.

The mode is chosen by the copy op -- ``TensorLoadGather`` / ``TensorStoreScatter``
rather than ``TensorLoad`` / ``TensorStore``. Both forms issue the same TDM
instruction; only the descriptor differs. The index width is not an op parameter: it
is the element width of the row list, so a list and the width it is read at can never
disagree. The atom sets the list up and a ``copy`` may hand over a different one.

The limits come from where the row list lives. It occupies the two descriptor groups
a linear tile would spend on dims 2..4, which is why an indexed tile is 2D only and
why one issue carries a bounded number of rows: up to sixteen at 16-bit indices, up
to eight at 32-bit. Unlike a fixed-width gather, any count in ``[1, max]`` is legal.
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
# A rejected configuration is rejected while tracing, before anything is compiled, so
# those cases run anywhere. A configuration that is accepted goes on to be compiled for
# gfx1250, which needs either that card or COMPILE_ONLY -- on another card the TDM
# instructions reach a backend that has none and it aborts rather than diagnosing.
_arch = get_rocm_arch() or ""
_needs_target = pytest.mark.skipif(
    not _arch.startswith("gfx1250") and os.environ.get("COMPILE_ONLY", "") in ("", "0"),
    reason=f"compiling an accepted indexed atom needs gfx1250 or COMPILE_ONLY, got {_arch}",
)

WIDTH = 128
N = 256


def _build(rows, width=WIDTH, op=None, tiler=None, index_dtype=torch.int32, index_tensor=True, copy_dtype=None):
    """Trace a one-issue indexed load; return nothing, raise whatever it raises."""
    tile = tiler if tiler is not None else (rows, width)
    make_op = op if op is not None else fx.rocdl.TensorLoadGather

    @flyc.kernel
    def k(src: fx.Tensor, idx: fx.Tensor, idx2: fx.Tensor):
        lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, rows * width]).peek()
        box = fx.make_layout((rows, width), (width, 1))
        smem = fx.make_view(lds.ptr, fx.make_layout(((rows, width), 1), ((width, 1), rows * width)))
        atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
            make_op(), src, box, tile, index_tensor=idx if index_tensor else None
        )
        blk = fx.zipped_divide(coord, (rows, width))[None, (0, None)]
        tAs, tAg, tAi = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, [blk, idx])
        # The atom set the list up; the call site may replace it.
        fx.copy(atom, [tAg[None, 0], idx2 if copy_dtype is not None else tAi], tAs[None, 0])
        fx.rocdl.s_wait_tensorcnt(0)

    @flyc.jit
    def j(src: fx.Tensor, idx: fx.Tensor, idx2: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        k(src, idx, idx2).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    n = rows if rows > 0 else 1
    j(
        torch.zeros(64, N, dtype=torch.float16, device="cuda"),
        torch.zeros(n, dtype=index_dtype, device="cuda"),
        torch.zeros(n, dtype=copy_dtype or index_dtype, device="cuda"),
    )


def _idx(bits):
    """Index dtype of a given width, as ``_build`` wants it."""
    return torch.int16 if bits == 16 else torch.int32


# -- the op struct is what selects the mode -------------------------------------


def test_linear_op_refuses_an_index_tensor():
    """A plain TensorLoad has no indexed descriptor to put the list in."""
    with pytest.raises(Exception, match="index_tensor needs an indexed op"):
        _build(rows=8, op=fx.rocdl.TensorLoad)


@_needs_target
def test_atom_without_a_list_defaults_to_16_bit():
    """The atom may leave the list to the call site; the default width is 16-bit,
    which is why 16 rows are allowed here with no list to read it from."""
    _build(rows=16, index_tensor=False, index_dtype=_idx(16))


def test_index_width_comes_from_the_element_type():
    """A 16-bit list doubles the row budget a 32-bit one allows, with no op change."""
    assert fx.rocdl.cdna5.index_rows_limit(16) == 16
    assert fx.rocdl.cdna5.index_rows_limit(32) == 8


def test_indexed_ops_keep_their_direction():
    """Dispatch to the load or store builder still goes through the base class."""
    assert isinstance(fx.rocdl.TensorLoadGather(), fx.rocdl.TensorLoad)
    assert isinstance(fx.rocdl.TensorStoreScatter(), fx.rocdl.TensorStore)
    assert not isinstance(fx.rocdl.TensorLoadGather(), fx.rocdl.TensorStore)


# -- row counts ------------------------------------------------------------------


@_needs_target
def test_gather_builds():
    """Eight rows at 32-bit indices is the widest a single issue carries."""
    _build(rows=8)


@_needs_target
def test_gather_builds_16bit():
    """Halving the index width doubles the rows one issue can name."""
    _build(rows=16, index_dtype=_idx(16))


@_needs_target
def test_copy_may_replace_the_row_list():
    """The atom initialises the list; a call site hands over a different one."""
    _build(rows=8, copy_dtype=torch.int32)


@_needs_target
def test_copy_may_narrow_the_index_width():
    """The replacement carries its own width, so the descriptor follows the operand."""
    _build(rows=8, index_dtype=_idx(32), copy_dtype=_idx(16))


@_needs_target
@pytest.mark.parametrize("rows", [1, 2, 3, 5, 8])
def test_any_row_count_up_to_the_budget_builds(rows):
    """The row count is not fixed: the descriptor says how many indices to consume,
    so a short tile needs no padding entries."""
    _build(rows=rows)


@_needs_target
@pytest.mark.parametrize("rows", [1, 7, 16])
def test_any_row_count_up_to_the_budget_builds_16bit(rows):
    """The same at 16-bit indices, where two share a descriptor word."""
    _build(rows=rows, index_dtype=_idx(16))


def test_too_many_rows_for_32bit():
    with pytest.raises(Exception, match="1 to 8 rows"):
        _build(rows=16, index_dtype=_idx(32))


def test_too_many_rows_for_16bit():
    with pytest.raises(Exception, match="1 to 16 rows"):
        _build(rows=32, index_dtype=_idx(16))


# -- descriptor shape ------------------------------------------------------------


def test_rank_must_be_two():
    """The row list occupies the groups dims 2..4 would use, so 2D is the only shape.

    The tiler rank is refused while tracing; a tiler that is rank 2 but whose derived
    descriptor is not reaches the same refusal from the geometry, which is the
    authoritative one -- hence the regex spans both messages.
    """
    with pytest.raises(Exception, match="2D only|must be 2D"):
        _build(rows=8, tiler=(2, 4, WIDTH))


def test_index_element_type_must_be_16_or_32_bit():
    """The width is a descriptor bit, so anything else has no encoding."""
    with pytest.raises(Exception, match="16- or 32-bit integer row indices"):
        _build(rows=8, index_dtype=torch.int8)


def test_row_shorter_than_a_dword_is_rejected():
    """A 1-element f16 row is 2 bytes, which the descriptor cannot encode."""
    with pytest.raises(Exception, match="whole multiple of 4 bytes"):
        _build(rows=8, width=1)


@_needs_target
def test_row_of_exactly_one_dword_builds():
    """Two f16 elements are 4 bytes, the narrowest row there is."""
    _build(rows=8, width=2)


# -- the scatter direction --------------------------------------------------------


@_needs_target
def test_scatter_builds():
    """The store side takes the same row list, naming where each row lands."""

    @flyc.kernel
    def k(dst: fx.Tensor, idx: fx.Tensor):
        lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, 8 * WIDTH]).peek()
        box = fx.make_layout((8, WIDTH), (WIDTH, 1))
        smem = fx.make_view(lds.ptr, fx.make_layout(((8, WIDTH), 1), ((WIDTH, 1), 8 * WIDTH)))
        atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorStoreScatter(), dst, box, (8, WIDTH), index_tensor=idx
        )
        blk = fx.zipped_divide(coord, (8, WIDTH))[None, (0, None)]
        tBs, tBg, tBi = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, [blk, idx])
        fx.copy(atom, tBs[None, 0], [tBg[None, 0], tBi])
        fx.rocdl.s_wait_tensorcnt(0)

    @flyc.jit
    def j(dst: fx.Tensor, idx: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        k(dst, idx).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    j(
        torch.zeros(64, N, dtype=torch.float16, device="cuda"),
        torch.zeros(8, dtype=torch.int32, device="cuda"),
    )


def test_partition_rejects_a_malformed_pair():
    with pytest.raises(Exception, match=r"\[coord, index\]"):

        @flyc.kernel
        def k(src: fx.Tensor, idx: fx.Tensor):
            lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, 8 * WIDTH]).peek()
            box = fx.make_layout((8, WIDTH), (WIDTH, 1))
            smem = fx.make_view(lds.ptr, fx.make_layout(((8, WIDTH), 1), ((WIDTH, 1), 8 * WIDTH)))
            atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoadGather(), src, box, (8, WIDTH), index_tensor=idx
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
