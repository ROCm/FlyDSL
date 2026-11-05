# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CDNA5 TDM indexed addressing: what the gather / scatter ops accept and refuse.

The mode is chosen by the copy op -- ``TensorLoadGather`` / ``TensorStoreScatter``
rather than ``TensorLoad`` / ``TensorStore``. Both forms issue the same TDM
instruction; only the descriptor differs.

The row list comes in two forms, and which one is used is visible in the generated
code. A tuple carries the indices themselves, so entries known while tracing become
descriptor immediates and cost no load. A tensor is data in memory, read at run time,
and its element width also picks the index width.

The limits come from where the row list lives. It occupies the two descriptor groups
a linear tile would spend on dims 2..4, which is why an indexed tile is 2D only and
why one issue carries a bounded number of rows: up to sixteen at 16-bit indices, up
to eight at 32-bit. Any count in ``[1, limit]`` is legal.

Nothing here needs a GPU: the kernels are compiled for gfx1250 under COMPILE_ONLY and
the assertions read the emitted ISA.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

WIDTH = 128
M, N = 64, 256
# Picked so the packed 16-bit words are unmistakable in the ISA: (1 << 16) | 3 and so on.
ROW_LIST = (3, 1, 17, 5, 40, 2, 9, 33)
PACKED_16 = ("0x10003", "0x50011", "0x20028", "0x210009")


def _target_env(monkeypatch, tmp_path):
    """Compile for gfx1250 without running, and keep every IR stage on disk."""
    for name, value in (
        ("FLYDSL_COMPILE_BACKEND", "rocm"),
        ("FLYDSL_RUNTIME_KIND", "rocm"),
        ("ARCH", "gfx1250"),
        ("FLYDSL_GPU_ARCH", "gfx1250"),
        ("COMPILE_ONLY", "1"),
        ("FLYDSL_RUNTIME_ENABLE_CACHE", "0"),
        ("FLYDSL_DUMP_IR", "1"),
        ("FLYDSL_DUMP_DIR", str(tmp_path)),
    ):
        monkeypatch.setenv(name, value)


def _isa(tmp_path):
    """The one kernel's final assembly."""
    files = list(tmp_path.glob("*/*_final_isa.s"))
    assert len(files) == 1, f"expected one kernel's ISA, got {files}"
    return files[0].read_text()


def _build(
    *,
    rows=8,
    width=WIDTH,
    op=None,
    tiler=None,
    index_dtype=torch.int32,
    atom_list=False,
    copy_list="tuple",
    entries=None,
):
    """Trace and compile one indexed copy.

    ``copy_list`` picks what the copy is handed: ``"tuple"`` for the indices
    themselves, ``"tensor"`` for data in memory. ``atom_list`` additionally gives the
    atom a tensor, which is what fixes the index width.
    """
    tile = tiler if tiler is not None else (rows, width)
    make_op = op if op is not None else fx.rocdl.TensorLoadGather
    storing = issubclass(make_op, fx.rocdl.TensorStore)
    row_entries = tuple(range(rows)) if entries is None else entries

    @flyc.kernel
    def k(src: fx.Tensor, idx: fx.Tensor):
        lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, rows * width]).peek()
        box = fx.make_layout((rows, width), (width, 1))
        smem = fx.make_view(lds.ptr, fx.make_layout(((rows, width), 1), ((width, 1), rows * width)))
        # A static shape keeps the geometry out of the kernel arguments, so the only
        # loads left are the ones the row list costs.
        g = fx.Tensor(fx.make_view(src.iter, fx.make_layout((M, N), (N, 1))))
        atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
            make_op(), g, box, tile, index_tensor=idx if atom_list else None
        )
        blk = fx.zipped_divide(coord, (rows, width))[None, (0, None)]
        tAs, tAg = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, blk)
        row_list = idx if fx.const_expr(copy_list == "tensor") else fx.make_int_tuple(row_entries)
        # The global side is the source of a gather and the destination of a scatter,
        # and the row list travels with it either way.
        if fx.const_expr(storing):
            fx.copy(atom, tAs[None, 0], [tAg[None, 0], row_list])
        else:
            fx.copy(atom, [tAg[None, 0], row_list], tAs[None, 0])
        fx.rocdl.s_wait_tensorcnt(0)

    @flyc.jit
    def j(src: fx.Tensor, idx: fx.Tensor):
        k(src, idx).launch(grid=1, block=32)

    # CPU tensors suffice for target compilation; this does not launch on a GPU.
    j(
        torch.empty(M * N, dtype=torch.float16, device="cpu"),
        torch.empty(max(rows, 1), dtype=index_dtype, device="cpu"),
    )


# -- the row list decides when the descriptor is built ----------------------------


def test_a_static_tuple_becomes_descriptor_immediates(monkeypatch, tmp_path):
    """Every entry is known while tracing, so the descriptor is built at compile time:
    the indices appear as immediates and no load reads them back."""
    _target_env(monkeypatch, tmp_path)
    _build(entries=ROW_LIST)
    isa = _isa(tmp_path)
    for word in PACKED_16:
        assert word in isa, f"expected the packed row pair {word} as an immediate, got:\n{isa}"


def test_a_tensor_row_list_is_read_at_run_time(monkeypatch, tmp_path):
    """The same indices live in memory, so the kernel never sees their values and has
    to read them instead."""
    _target_env(monkeypatch, tmp_path)
    _build(atom_list=True, copy_list="tensor")
    isa = _isa(tmp_path)
    for word in PACKED_16:
        assert word not in isa, f"a row list in memory must not fold to {word}, got:\n{isa}"
    assert "s_load" in isa


def test_a_tuple_may_mix_known_and_runtime_entries(monkeypatch, tmp_path):
    """A tuple is decided per entry: what is known folds, the rest rides in as an
    operand, so a partly known list still costs no load."""
    _target_env(monkeypatch, tmp_path)
    _build(rows=4, entries=(3, 1, 17, fx.Int32(5)))
    isa = _isa(tmp_path)
    assert "0x10003" in isa


def test_the_mode_bit_is_set(monkeypatch, tmp_path):
    """Bit 31 of descriptor group 0 is what turns indexed addressing on; bit 30 picks
    the width, and a tuple leaves it at the 16-bit default."""
    _target_env(monkeypatch, tmp_path)
    _build(entries=ROW_LIST)
    assert "0x80000001" in _isa(tmp_path)


def test_a_tensor_row_list_picks_the_width(monkeypatch, tmp_path):
    """A 32-bit list sets bit 30 as well, which is the same descriptor word with the
    width bit flipped."""
    _target_env(monkeypatch, tmp_path)
    _build(atom_list=True, copy_list="tensor", index_dtype=torch.int32)
    assert "0xc0000001" in _isa(tmp_path)


def test_a_narrow_tensor_row_list_clears_the_width_bit(monkeypatch, tmp_path):
    _target_env(monkeypatch, tmp_path)
    _build(rows=16, atom_list=True, copy_list="tensor", index_dtype=torch.int16)
    assert "0x80000001" in _isa(tmp_path)


# -- the op struct is what selects the mode ---------------------------------------


def test_linear_op_refuses_a_row_list():
    """A plain TensorLoad has no indexed descriptor to put the list in."""
    with pytest.raises(Exception, match="index_tensor needs an indexed op"):
        _build(op=fx.rocdl.TensorLoad, atom_list=True)


def test_indexed_ops_keep_their_direction():
    """Dispatch to the load or store builder still goes through the base class."""
    assert isinstance(fx.rocdl.TensorLoadGather(), fx.rocdl.TensorLoad)
    assert isinstance(fx.rocdl.TensorStoreScatter(), fx.rocdl.TensorStore)
    assert not isinstance(fx.rocdl.TensorLoadGather(), fx.rocdl.TensorStore)


def test_the_row_budget_follows_the_width():
    assert fx.rocdl.cdna5.index_rows_limit(16) == 16
    assert fx.rocdl.cdna5.index_rows_limit(32) == 8


# -- row counts -------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 16])
def test_any_row_count_up_to_the_budget_builds(monkeypatch, tmp_path, rows):
    """The count is not fixed: the descriptor says how many indices to consume, so a
    short tile needs no padding entries."""
    _target_env(monkeypatch, tmp_path)
    _build(rows=rows)


def test_too_many_rows_for_16bit():
    """A tuple keeps the 16-bit default, so seventeen rows have nowhere to go."""
    with pytest.raises(Exception, match="1 to 16 rows|rows exceed the 16"):
        _build(rows=32)


def test_too_many_rows_for_32bit():
    with pytest.raises(Exception, match="1 to 8 rows|rows exceed the 8"):
        _build(rows=16, atom_list=True, copy_list="tensor", index_dtype=torch.int32)


def test_a_tuple_must_match_the_row_count():
    with pytest.raises(Exception, match="entries but the tile has"):
        _build(rows=8, entries=(3, 1, 17))


# -- descriptor shape -------------------------------------------------------------


def test_rank_must_be_two():
    """The row list occupies the groups dims 2..4 would use, so 2D is the only shape.

    The tiler rank is refused while tracing; a tiler that is rank 2 but whose derived
    descriptor is not reaches the same refusal from the geometry, which is the
    authoritative one -- hence the regex spans both messages.
    """
    with pytest.raises(Exception, match="2D only|must be 2D"):
        _build(tiler=(2, 4, WIDTH))


def test_row_list_element_type_must_be_16_or_32_bit():
    """The width is a descriptor bit, so anything else has no encoding."""
    with pytest.raises(Exception, match="16- or 32-bit integer row indices"):
        _build(atom_list=True, index_dtype=torch.int8)


def test_row_shorter_than_a_dword_is_rejected():
    """A 1-element f16 row is 2 bytes, which the descriptor cannot encode."""
    with pytest.raises(Exception, match="whole multiple of 4 bytes"):
        _build(width=1)


def test_row_of_exactly_one_dword_builds(monkeypatch, tmp_path):
    """Two f16 elements are 4 bytes, the narrowest row there is."""
    _target_env(monkeypatch, tmp_path)
    _build(width=2)


# -- the scatter direction --------------------------------------------------------


def test_scatter_builds(monkeypatch, tmp_path):
    """The store side takes the same row list, naming where each row lands."""
    _target_env(monkeypatch, tmp_path)
    _build(op=fx.rocdl.TensorStoreScatter, entries=ROW_LIST)
    isa = _isa(tmp_path)
    assert "tensor_store_from_lds" in isa
    assert "0x10003" in isa


def test_partition_rejects_a_malformed_pair():
    with pytest.raises(Exception, match=r"\[coord, index\]"):

        @flyc.kernel
        def k(src: fx.Tensor, idx: fx.Tensor):
            lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, 8 * WIDTH]).peek()
            box = fx.make_layout((8, WIDTH), (WIDTH, 1))
            smem = fx.make_view(lds.ptr, fx.make_layout(((8, WIDTH), 1), ((WIDTH, 1), 8 * WIDTH)))
            g = fx.Tensor(fx.make_view(src.iter, fx.make_layout((M, N), (N, 1))))
            atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoadGather(), g, box, (8, WIDTH), index_tensor=idx
            )
            blk = fx.zipped_divide(coord, (8, WIDTH))[None, (0, None)]
            fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, [blk, idx, idx])

        @flyc.jit
        def j(src: fx.Tensor, idx: fx.Tensor):
            k(src, idx).launch(grid=1, block=32)

        j(
            torch.empty(M * N, dtype=torch.float16, device="cpu"),
            torch.empty(8, dtype=torch.int32, device="cpu"),
        )
