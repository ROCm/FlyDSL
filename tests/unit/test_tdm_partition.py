#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""tdm_partition on an LDS tensor with and without rest modes, traced for
gfx1250 without running it.
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

torch = pytest.importorskip("torch")

TM, TN = 128, 64


@pytest.fixture
def trace_only_gfx1250(monkeypatch):
    monkeypatch.setenv("ARCH", "gfx1250")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(lambda cls, module, **_kwargs: module))


@pytest.mark.parametrize(
    "smem_layout, expected",
    [
        # (TDM_Tile, rest): the rest mode is kept.
        ((((TM, TN), 1), ((TN, 1), TM * TN)), "((8192,1),1):((1,0),8192)"),
        # (TDM_Tile,): no rest mode.
        ((((TM, TN),), ((TN, 1),)), "((8192,1)):((1,0))"),
    ],
)
def test_tdm_partition_rest_modes(trace_only_gfx1250, smem_layout, expected):
    shape, stride = smem_layout

    @flyc.kernel
    def kernel(A: fx.Tensor):
        lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, TM * TN]).peek()
        smem = fx.make_view(lds.ptr, fx.make_layout(shape, stride))
        atom, tdm = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(), A, fx.make_layout((TM, TN), (TN, 1)), (TM, TN)
        )
        gmem = fx.zipped_divide(tdm, (TM, TN))[None, (None, 0)]
        partitioned, _ = fx.rocdl.cdna5.tdm_partition(atom, 0, fx.make_layout(1, 1), smem, gmem)
        assert str(partitioned.layout.type) == f"!fly.layout<{expected}>"

    @flyc.jit
    def launch(A: fx.Tensor):
        kernel(A).launch(grid=(1, 1, 1), block=(32, 1, 1))

    launch(torch.zeros(2 * TM, TN, dtype=torch.float16))
