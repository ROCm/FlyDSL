# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""An element offset into a 4- or 6-bit view steps in packed bytes.

Compiled for gfx942 without a launch: the final ISA's global loads and stores must cover
exactly the packed buffer. A 64-element fp4 view copied in two halves used to touch bytes
[0, 16) and [32, 48) of its 32-byte buffer, and fp6 [0, 24) and [32, 56) of 48.
"""

import re

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.diagnostics import DSLCompileError

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

N = 64
DTYPES = {
    "fp4": fx.Float4E2M1FN,
    "fp6": fx.Float6E2M3FN,
    "bf6": fx.Float6E3M2FN,
    "fp8": fx.Float8E4M3FN,
}
BITS = {"fp4": 4, "fp6": 6, "bf6": 6, "fp8": 8}
ACCESS_BYTES = {"ubyte": 1, "byte": 1, "short": 2, "dword": 4, "dwordx2": 8, "dwordx3": 12, "dwordx4": 16}


@pytest.fixture(autouse=True)
def compile_for_gfx942(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "1")
    monkeypatch.delenv("FLYDSL_DUMP_IR", raising=False)


def _byte_ranges(isa, kind):
    """Merged [lo, hi) byte ranges of the immediate-offset global loads or stores."""
    ranges = sorted(
        (int(off or 0), int(off or 0) + ACCESS_BYTES[width])
        for width, off in re.findall(rf"global_{kind}_(\w+?)\s[^\n]*?(?:offset:(\d+))?\n", isa)
    )
    merged = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _compile_halves(name, use):
    """Copy a recast N-element view in two halves of N/2; return the final ISA."""
    dtype = DTYPES[name]

    @flyc.kernel(name=f"subbyte_{name}_{use}")
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        src = fx.recast_iter(dtype, fx.get_iter(A))
        dst = fx.recast_iter(dtype, fx.get_iter(Out))
        if fx.const_expr(use == "row"):
            rows = fx.make_layout((2, N // 2), (N // 2, 1))
            s, d = fx.make_view(src, rows), fx.make_view(dst, rows)
            for i in range(2):
                fx.slice(d, (i, None)).store(fx.slice(s, (i, None)).load())
        if fx.const_expr(use == "iter"):
            half = fx.make_layout((N // 2,), (1,))
            for i in range(2):
                fx.make_view(dst + i * (N // 2), half).store(fx.make_view(src + i * (N // 2), half).load())

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    nbytes = N * BITS[name] // 8
    launch(torch.zeros(nbytes, dtype=torch.uint8), torch.zeros(nbytes, dtype=torch.uint8))
    return nbytes


@pytest.mark.parametrize("use", ["row", "iter"])
@pytest.mark.parametrize("name", list(DTYPES))
def test_halves_cover_the_packed_buffer(name, use, tmp_path):
    nbytes = _compile_halves(name, use)
    (isa_file,) = tmp_path.rglob("*_final_isa.s")
    isa = isa_file.read_text()
    assert _byte_ranges(isa, "load") == [(0, nbytes)]
    assert _byte_ranges(isa, "store") == [(0, nbytes)]


def test_static_offset_off_a_byte_is_refused():
    @flyc.kernel
    def kernel(A: fx.Tensor):
        p = fx.recast_iter(fx.Float4E2M1FN, fx.get_iter(A)) + 1
        fx.make_view(p, fx.make_layout((2,), (1,))).store(fx.Vector.filled(2, 0.0, fx.Float4E2M1FN))

    @flyc.jit
    def launch(A: fx.Tensor):
        kernel(A).launch(grid=(1, 1, 1), block=(64, 1, 1))

    with pytest.raises(DSLCompileError, match="offset 1 into 4-bit elements does not land on a byte"):
        launch(torch.zeros(8, dtype=torch.uint8))
