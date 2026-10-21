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
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

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
        g = fx.Tensor(
            fx.make_view(arg, fx.make_layout((N_BLOCKS, 1, KT_ALL), (KT_ALL, 1, 1)))
        )
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
