# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CDNA5 TDM indexed addressing, run on the hardware.

``tests/unit/test_tdm_indexed.py`` compiles these kernels and reads the emitted
ISA; it never launches one. This file launches them and compares the data, which
is the only way to settle the questions the ISA cannot answer:

* does the row list actually select those rows, in list order;
* does a row index past the end of the tensor read as zero on a load and get
  dropped on a store, so a short tile needs no padding entries;
* may the list repeat an index or run out of order;
* do two issues from one atom really move two different sets of rows, given that
  the backend rewrites descriptor registers in place between them.

The tile is deliberately narrower than the tensor's row. A tile that spans the
whole row is contiguous, so the geometry folds the two dims into one and the
descriptor is no longer 2D -- which is what indexed addressing needs.
"""

import pytest

try:
    import torch
except ImportError:
    torch = None

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx1250"):
    pytest.skip(f"CDNA5 TDM indexed addressing requires gfx1250, got {_ARCH}", allow_module_level=True)

M, N = 64, 256  # the tensors
ROWS, WIDTH = 8, 128  # one issue: 8 rows of 128 f16 = 256 bytes each


def _src():
    """Every element distinct, so a row that lands in the wrong place is visible."""
    return (torch.arange(M * N, dtype=torch.float32, device="cuda") % 4096).to(torch.float16).reshape(M, N)


def _views(src, dst):
    """The global views both kernels share: data row-major, row list broadcast."""
    g = fx.Tensor(fx.make_view(src.iter, fx.make_layout((M, N), (N, 1))))
    d = fx.Tensor(fx.make_view(dst.iter, fx.make_layout((M, N), (N, 1))))
    return g, d


def _lds():
    lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, ROWS * WIDTH]).peek()
    box = fx.make_layout((ROWS, WIDTH), (WIDTH, 1))
    smem = fx.make_view(lds.ptr, fx.make_layout(((ROWS, WIDTH), 1), ((WIDTH, 1), ROWS * WIDTH)))
    return box, smem


def _store_rows_0_to_7(dst_view, box, smem):
    """Drain the LDS tile to dst[0:ROWS, 0:WIDTH] with a plain linear store.

    Keeping the write side linear is what makes the gather observable: whatever
    lands in LDS row i shows up in dst row i, so dst reports the list order.
    """
    atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorStore(), dst_view, box, (ROWS, WIDTH))
    blk = fx.zipped_divide(coord, (ROWS, WIDTH))[None, (0, None)]
    tBs, tBg = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, blk)
    fx.copy(atom, tBs[None, 0], tBg[None, 0])
    fx.rocdl.s_wait_tensorcnt(0)


# -- runtime row list: the indices live in memory ---------------------------------


@flyc.kernel
def _gather_runtime(src: fx.Tensor, dst: fx.Tensor, idx: fx.Tensor):
    box, smem = _lds()
    g, d = _views(src, dst)
    gI = fx.Tensor(fx.make_view(idx.iter, fx.make_layout((M, N), (1, 0))))
    atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
        fx.rocdl.TensorLoadGather(), g, box, (ROWS, WIDTH), index_tensor=gI
    )
    blk = fx.zipped_divide(coord, (ROWS, WIDTH))[None, (0, None)]
    blkI = fx.zipped_divide(gI, (ROWS, WIDTH))[None, (0, None)]
    tAs, tAg, tAi = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, [blk, blkI])
    fx.copy(atom, [tAg[None, 0], tAi[None, 0]], tAs[None, 0])
    fx.rocdl.s_wait_tensorcnt(0)
    _store_rows_0_to_7(d, box, smem)


@flyc.jit
def _run_gather_runtime(src: fx.Tensor, dst: fx.Tensor, idx: fx.Tensor):
    _gather_runtime(src, dst, idx).launch(grid=1, block=32)


def _gather_with(rows, dtype=torch.int32):
    """Gather ``rows`` through a row list in memory and return dst."""
    src, dst = _src(), torch.zeros(M, N, dtype=torch.float16, device="cuda")
    idx = torch.zeros(M, dtype=dtype, device="cuda")
    idx[: len(rows)] = torch.tensor(rows, dtype=dtype, device="cuda")
    _run_gather_runtime(src, dst, idx)
    torch.cuda.synchronize()
    return src, dst


def test_runtime_rows_land_in_list_order():
    rows = [3, 1, 17, 5, 40, 2, 9, 33]
    src, dst = _gather_with(rows)
    torch.testing.assert_close(dst[:ROWS, :WIDTH], src[rows, :WIDTH])


def test_runtime_rows_may_repeat_and_run_out_of_order():
    """The hardware neither sorts nor de-duplicates the list."""
    rows = [7, 7, 0, 63, 2, 2, 2, 1]
    src, dst = _gather_with(rows)
    torch.testing.assert_close(dst[:ROWS, :WIDTH], src[rows, :WIDTH])


def test_a_row_past_the_end_reads_zero():
    """An out-of-range index is how a short tile masks the rows it does not want:
    the load returns zeros rather than faulting or reading neighbouring memory."""
    rows = [0, 1, 2, M, M + 100, 5, 6, 7]
    src, dst = _gather_with(rows)
    live = [0, 1, 2, 5, 6, 7]
    torch.testing.assert_close(dst[live, :WIDTH], src[[rows[i] for i in live], :WIDTH])
    assert bool(dst[[3, 4], :WIDTH].eq(0).all()), f"out-of-range rows did not read zero:\n{dst[[3, 4], :8]}"


def test_a_16_bit_row_list_selects_the_same_rows():
    """The width is read off the list's element type; narrowing it changes only
    how the indices are packed into the descriptor."""
    rows = [3, 1, 17, 5, 40, 2, 9, 33]
    src, dst = _gather_with(rows, dtype=torch.int16)
    torch.testing.assert_close(dst[:ROWS, :WIDTH], src[rows, :WIDTH])


# -- compile-time row list: the indices live in the type --------------------------

STATIC_ROWS = (3, 1, 17, 5, 40, 2, 9, 33)


@flyc.kernel
def _gather_static(src: fx.Tensor, dst: fx.Tensor):
    box, smem = _lds()
    g, d = _views(src, dst)
    atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorLoadGather(), g, box, (ROWS, WIDTH))
    blk = fx.zipped_divide(coord, (ROWS, WIDTH))[None, (0, None)]
    tAs, tAg = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, blk)
    fx.copy(atom, [tAg[None, 0], fx.make_int_tuple(STATIC_ROWS)], tAs[None, 0])
    fx.rocdl.s_wait_tensorcnt(0)
    _store_rows_0_to_7(d, box, smem)


@flyc.jit
def _run_gather_static(src: fx.Tensor, dst: fx.Tensor):
    _gather_static(src, dst).launch(grid=1, block=32)


def test_a_compile_time_row_list_selects_the_same_rows():
    """Same rows as the runtime case, but the indices are descriptor immediates --
    the kernel takes no index argument and issues no load for them."""
    src, dst = _src(), torch.zeros(M, N, dtype=torch.float16, device="cuda")
    _run_gather_static(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst[:ROWS, :WIDTH], src[list(STATIC_ROWS), :WIDTH])


# -- a tuple decided per entry ----------------------------------------------------


@flyc.kernel
def _gather_mixed(src: fx.Tensor, dst: fx.Tensor, last: fx.Int32):
    box, smem = _lds()
    g, d = _views(src, dst)
    atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorLoadGather(), g, box, (ROWS, WIDTH))
    blk = fx.zipped_divide(coord, (ROWS, WIDTH))[None, (0, None)]
    tAs, tAg = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, blk)
    rows = fx.make_int_tuple((3, 1, 17, 5, 40, 2, 9, last))
    fx.copy(atom, [tAg[None, 0], rows], tAs[None, 0])
    fx.rocdl.s_wait_tensorcnt(0)
    _store_rows_0_to_7(d, box, smem)


@flyc.jit
def _run_gather_mixed(src: fx.Tensor, dst: fx.Tensor, last: fx.Int32):
    _gather_mixed(src, dst, last).launch(grid=1, block=32)


def test_a_tuple_may_mix_known_and_runtime_entries():
    """Seven entries are folded into the descriptor, the eighth arrives as an
    operand; both halves have to reach the same descriptor groups."""
    src, dst = _src(), torch.zeros(M, N, dtype=torch.float16, device="cuda")
    _run_gather_mixed(src, dst, 44)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst[:ROWS, :WIDTH], src[[3, 1, 17, 5, 40, 2, 9, 44], :WIDTH])


# -- two issues from one atom -----------------------------------------------------

SECOND_ROWS = (11, 12, 13, 14, 15, 16, 18, 19)


@flyc.kernel
def _two_gathers(src: fx.Tensor, dst: fx.Tensor, idx: fx.Tensor):
    lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, ROWS * WIDTH * 2]).peek()
    box = fx.make_layout((ROWS, WIDTH), (WIDTH, 1))
    smem = fx.make_view(lds.ptr, fx.make_layout(((ROWS, WIDTH), 2), ((WIDTH, 1), ROWS * WIDTH)))
    g, d = _views(src, dst)
    gI = fx.Tensor(fx.make_view(idx.iter, fx.make_layout((M, N), (1, 0))))
    atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
        fx.rocdl.TensorLoadGather(), g, box, (ROWS, WIDTH), index_tensor=gI
    )
    blk = fx.zipped_divide(coord, (ROWS, WIDTH))[None, (0, None)]
    blkI = fx.zipped_divide(gI, (ROWS, WIDTH))[None, (0, None)]
    tAs, tAg, tAi = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, [blk, blkI])
    # Two issues back to back: the first on the atom's list in memory, the second on
    # a list the call site knows. The backend rewrites the shared descriptor groups
    # between them, so this is also what shows the engine reads a descriptor at issue.
    fx.copy(atom, [tAg[None, 0], tAi[None, 0]], tAs[None, 0])
    fx.copy(atom, [tAg[None, 0], fx.make_int_tuple(SECOND_ROWS)], tAs[None, 1])
    fx.rocdl.s_wait_tensorcnt(0)

    # Drain both LDS stages: stage 0 to dst rows 0..7, stage 1 to dst rows 8..15.
    store, scoord = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorStore(), d, box, (ROWS, WIDTH))
    sblk = fx.zipped_divide(scoord, (ROWS, WIDTH))[None, (None, None)]
    tBs, tBg = fx.rocdl.cdna5.tdm_partition(store, 0, fx.make_layout(1, 1), smem, sblk)
    fx.copy(store, tBs[None, 0], tBg[(None, 0, 0)])
    fx.copy(store, tBs[None, 1], tBg[(None, 1, 0)])
    fx.rocdl.s_wait_tensorcnt(0)


@flyc.jit
def _run_two_gathers(src: fx.Tensor, dst: fx.Tensor, idx: fx.Tensor):
    _two_gathers(src, dst, idx).launch(grid=1, block=32)


def test_two_issues_from_one_atom_move_two_different_row_sets():
    """The ISA test shows the two lists sit in different registers. This shows the
    data: each issue moved exactly the rows its own list named."""
    first = [3, 1, 17, 5, 40, 2, 9, 33]
    src, dst = _src(), torch.zeros(M, N, dtype=torch.float16, device="cuda")
    idx = torch.zeros(M, dtype=torch.int32, device="cuda")
    idx[: len(first)] = torch.tensor(first, dtype=torch.int32, device="cuda")
    _run_two_gathers(src, dst, idx)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst[0:ROWS, :WIDTH], src[first, :WIDTH])
    torch.testing.assert_close(dst[ROWS : 2 * ROWS, :WIDTH], src[list(SECOND_ROWS), :WIDTH])


# -- the scatter direction --------------------------------------------------------


@flyc.kernel
def _scatter_runtime(src: fx.Tensor, dst: fx.Tensor, idx: fx.Tensor):
    box, smem = _lds()
    g, d = _views(src, dst)
    gI = fx.Tensor(fx.make_view(idx.iter, fx.make_layout((M, N), (1, 0))))

    # Read rows 0..ROWS-1 of src linearly, then scatter them to the listed rows.
    load, lcoord = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorLoad(), g, box, (ROWS, WIDTH))
    lblk = fx.zipped_divide(lcoord, (ROWS, WIDTH))[None, (0, None)]
    tAs, tAg = fx.rocdl.cdna5.tdm_partition(load, 0, fx.make_layout(1, 1), smem, lblk)
    fx.copy(load, tAg[None, 0], tAs[None, 0])
    fx.rocdl.s_wait_tensorcnt(0)

    store, scoord = fx.rocdl.cdna5.make_tiled_tdm_atom(
        fx.rocdl.TensorStoreScatter(), d, box, (ROWS, WIDTH), index_tensor=gI
    )
    sblk = fx.zipped_divide(scoord, (ROWS, WIDTH))[None, (0, None)]
    sblkI = fx.zipped_divide(gI, (ROWS, WIDTH))[None, (0, None)]
    tBs, tBg, tBi = fx.rocdl.cdna5.tdm_partition(store, 0, fx.make_layout(1, 1), smem, [sblk, sblkI])
    fx.copy(store, tBs[None, 0], [tBg[None, 0], tBi[None, 0]])
    fx.rocdl.s_wait_tensorcnt(0)


@flyc.jit
def _run_scatter_runtime(src: fx.Tensor, dst: fx.Tensor, idx: fx.Tensor):
    _scatter_runtime(src, dst, idx).launch(grid=1, block=32)


def _scatter_with(rows):
    src, dst = _src(), torch.zeros(M, N, dtype=torch.float16, device="cuda")
    idx = torch.zeros(M, dtype=torch.int32, device="cuda")
    idx[: len(rows)] = torch.tensor(rows, dtype=torch.int32, device="cuda")
    _run_scatter_runtime(src, dst, idx)
    torch.cuda.synchronize()
    return src, dst


def test_scatter_puts_row_i_where_the_list_says():
    rows = [3, 1, 17, 5, 40, 2, 9, 33]
    src, dst = _scatter_with(rows)
    for i, r in enumerate(rows):
        torch.testing.assert_close(dst[r, :WIDTH], src[i, :WIDTH], msg=f"row {i} did not land at {r}")
    touched = torch.zeros(M, dtype=torch.bool, device="cuda")
    touched[rows] = True
    assert bool(dst[~touched].eq(0).all()), "the scatter wrote rows its list never named"


def test_a_scatter_row_past_the_end_is_dropped():
    """The store mirror of the masked load: an out-of-range index writes nowhere
    rather than wrapping or corrupting the tail of the tensor."""
    rows = [0, 1, 2, M, M + 100, 5, 6, 7]
    src, dst = _scatter_with(rows)
    for i in (0, 1, 2, 5, 6, 7):
        torch.testing.assert_close(dst[rows[i], :WIDTH], src[i, :WIDTH])
    touched = torch.zeros(M, dtype=torch.bool, device="cuda")
    touched[[rows[i] for i in (0, 1, 2, 5, 6, 7)]] = True
    assert bool(dst[~touched].eq(0).all()), "an out-of-range scatter index wrote somewhere"
