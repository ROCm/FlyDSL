#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A TDM tile one element wide along the contiguous global mode.

The LDS run drops an extent-one axis, so a tile that takes a single element
along the global tensor's contiguous mode loses it and leaves a strided mode to
lead the descriptor. The filler puts that mode back; this checks it goes to the
front, where dim 0's hardwired stride of 1 belongs.

aiter's mx128 B-scale tile is (tile_n / 128, tile_k / 128) over a row-major
scale tensor, so every tile_k = 128 config lands here. Its K extent is a kernel
argument, which reaches the filler's guards as a dynamic `modeExtent` rather
than through their static arm.

The same extent-one axis 0 is how TDM transposes: whenever the LDS run does not
start on the contiguous mode, the box stops before that mode and the mode's
traversal stays in ITER, one column per issue.
"""

import math

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

N_BLOCKS = 2
KT_ALL = 64


def _compile_static():
    @flyc.kernel(name="tdm_contiguous_filler_static", known_block_size=[32, 1, 1])
    def load_scales(arg: fx.Pointer):
        g = fx.Tensor(fx.make_view(arg, fx.make_layout((N_BLOCKS, KT_ALL), (KT_ALL, 1))))
        return fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(),
            g,
            fx.make_layout((N_BLOCKS, 1), (1, 1)),
            (N_BLOCKS, 1),
            num_warps=1,
        )

    @flyc.jit
    def run(p: fx.Pointer, stream: fx.Stream = fx.Stream(None)):
        load_scales(p).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    flyc.compile(run, flyc.from_c_void_p(fx.Uint8, 0))


def _compile_dynamic():
    @flyc.kernel(name="tdm_contiguous_filler_dynamic", known_block_size=[32, 1, 1])
    def load_scales(arg: fx.Pointer, i32_kt: fx.Int32):
        kt = fx.Int64(i32_kt)
        g = fx.Tensor(fx.make_view(arg, fx.make_layout((N_BLOCKS, kt), (kt, 1))))
        return fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(),
            g,
            fx.make_layout((N_BLOCKS, 1), (1, 1)),
            (N_BLOCKS, 1),
            num_warps=1,
        )

    @flyc.jit
    def run(p: fx.Pointer, kt: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        load_scales(p, kt).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    flyc.compile(run, flyc.from_c_void_p(fx.Uint8, 0), KT_ALL)


@pytest.mark.parametrize("extent", ["static", "dynamic"])
def test_tile_one_wide_along_contiguous_mode(monkeypatch, extent):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1250")
    (_compile_static if extent == "static" else _compile_dynamic)()


def _compile_degenerate_mode():
    """The same tile behind a size-one global mode that also reads as stride one."""

    @flyc.kernel(name="tdm_contiguous_filler_degenerate", known_block_size=[32, 1, 1])
    def load_scales(arg: fx.Pointer):
        g = fx.Tensor(fx.make_view(arg, fx.make_layout((N_BLOCKS, 1, KT_ALL), (KT_ALL, 1, 1))))
        return fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(),
            g,
            fx.make_layout((N_BLOCKS, 1, 1), (1, 1, 1)),
            (N_BLOCKS, 1, 1),
            num_warps=1,
        )

    @flyc.jit
    def run(p: fx.Pointer, stream: fx.Stream = fx.Stream(None)):
        load_scales(p).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    flyc.compile(run, flyc.from_c_void_p(fx.Uint8, 0))


def test_degenerate_mode_does_not_hide_the_contiguous_one(monkeypatch):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1250")
    _compile_degenerate_mode()


def _compile_tile_copy(g_layout, lds_layout, tiler, box, smem_part, gmem_part):
    """Copy the first tile and check the atom's box and both partitions.

    The partitions fix the issue order: ITER walks the LDS in address order, and
    the global side must name the same element for every step.
    """
    size = math.prod(tiler)
    rest = (0,) * (len(tiler) - 1) + (None,)

    @flyc.kernel(name="tdm_contiguous_filler_copy", known_block_size=[32, 1, 1])
    def copy_tile(arg: fx.Pointer):
        lds = fx.SharedAllocator().allocate(fx.Array[fx.Uint8, size]).peek()
        smem = fx.make_view(lds.ptr, fx.make_layout((lds_layout[0], 1), (lds_layout[1], size)))
        g = fx.Tensor(fx.make_view(arg, fx.make_layout(*g_layout)))
        atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(), g, fx.make_layout(*lds_layout), tiler, num_warps=1
        )
        t_s, t_g = fx.rocdl.cdna5.tdm_partition(
            atom, 0, fx.make_layout(1, 1), smem, fx.zipped_divide(coord, tiler)[None, rest]
        )
        assert f"shape = {box}," in str(atom.type), str(atom.type)
        assert str(t_s.layout.type) == f"!fly.layout<{smem_part}>"
        assert str(t_g.layout.type) == f"!fly.layout<{gmem_part}>"
        fx.copy(atom, t_g[None, 0], t_s[None, 0])
        fx.rocdl.s_wait_tensorcnt(0)

    @flyc.jit
    def run(p: fx.Pointer, stream: fx.Stream = fx.Stream(None)):
        copy_tile(p).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    flyc.compile(run, flyc.from_c_void_p(fx.Uint8, 0))


@pytest.mark.parametrize(
    "g_layout, lds_layout, tiler, box, smem_part, gmem_part",
    [
        # The LDS is column-major over a row-major tensor: one column per issue.
        (
            ((4, 64), (64, 1)),
            ((4, 4), (1, 4)),
            (4, 4),
            [4, 1],
            "((4,4),1):((1,4),16)",
            "((4,4),16):((1E1,1E0),4E0)",
        ),
        # The same transpose, with the columns stored out of order: a non-unit
        # step stops the LDS run before it reaches the contiguous mode.
        (
            ((4, 64), (64, 1)),
            ((4, (2, 2)), (1, (8, 4))),
            (4, 4),
            [4, 1],
            "((4,4),1):((1,4),16)",
            "((4,(2,2)),16):((1E1,(2E0,1E0)),4E0)",
        ),
        # The contiguous mode sits in the middle of the run; the box keeps the
        # part before it and both later modes stay in ITER.
        (
            ((2, 4, 64), (256, 64, 1)),
            ((2, 4, 4), (16, 1, 4)),
            (2, 4, 4),
            [1, 4, 1],
            "((4,8),1):((1,4),32)",
            "((4,(4,2)),16):((1E1,(1E0,1E2)),4E0)",
        ),
        # The LDS splits mode 1 so that its halves sit apart. The run is
        # (64:1E2, 2:1E0, 2:2E1, 2:1E1), and `2:1E0` must not join `2:2E1`.
        (
            ((2, 4, 64), (256, 64, 1)),
            ((2, (2, 2), 64), (64, (256, 128), 1)),
            (2, 4, 64),
            [1, 2, 64],
            "((128,4),1):((1,128),512)",
            "(((64,2),(2,2)),1):(((1E0,1E1),(2E2,1E2)),0)",
        ),
        # The same split with the tile one element wide along the contiguous
        # mode, which the filler leads with.
        (
            ((2, 4, 64), (256, 64, 1)),
            ((2, (2, 2), 1), (1, (4, 2), 1)),
            (2, 4, 1),
            [1, 2, 1],
            "((2,4),1):((1,2),8)",
            "((2,(2,2)),64):((1E1,(2E2,1E2)),1E0)",
        ),
    ],
    ids=["column_major", "column_major_split", "contiguous_mid_run", "split_mode", "split_mode_filled"],
)
def test_contiguous_mode_leads_the_box(monkeypatch, g_layout, lds_layout, tiler, box, smem_part, gmem_part):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1250")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    _compile_tile_copy(g_layout, lds_layout, tiler, box, smem_part, gmem_part)


def test_no_contiguous_mode_is_refused(monkeypatch):
    """Every mode is strided, so nothing can take axis 0's implicit stride of 1."""
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1250")

    @flyc.kernel(name="tdm_contiguous_filler_strided", known_block_size=[32, 1, 1])
    def load(arg: fx.Pointer):
        g = fx.Tensor(fx.make_view(arg, fx.make_layout((2, 64), (128, 2))))
        return fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(), g, fx.make_layout((2, 4), (4, 1)), (2, 4), num_warps=1
        )

    @flyc.jit
    def run(p: fx.Pointer, stream: fx.Stream = fx.Stream(None)):
        load(p).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    with pytest.raises(ir.MLIRError, match="no unique static stride-1 mode"):
        flyc.compile(run, flyc.from_c_void_p(fx.Uint8, 0))
