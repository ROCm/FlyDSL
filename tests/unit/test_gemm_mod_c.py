# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A gfx1250 WMMA C modifier applies to every WMMA, so ``fx.gemm`` rejects it on a tile's later K steps,
where C is the partial sum in D; an ``atom_callback`` may still use it on each tile's first K step."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.diagnostics import DSLCompileError

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@pytest.fixture(autouse=True)
def _gfx1250_compile_only(monkeypatch):
    for name, value in {"FLYDSL_GPU_ARCH": "gfx1250", "COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0"}.items():
        monkeypatch.setenv(name, value)


def _wmma(mod_c):
    return fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32, mod_c=mod_c))


def _compile_gemm(mnk, *, mod_c=1, scaled=False, rank=3, tiled=False, traversal=None, atom_callback=None):
    """Compiles one gfx1250 ``fx.gemm`` on register fragments; rank 2 drops K from A and B."""
    m, n, k = mnk
    v, dtype = (64, fx.Float8E4M3FN) if scaled else (16, fx.BFloat16)

    def fragment(rows):
        return fx.make_rmem_tensor(v if rank == 1 else ((v, rows) if rank == 2 else (v, rows, k)), dtype)

    def make_atom():
        if scaled:
            return fx.make_mma_atom(fx.rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, mod_c=mod_c))
        atom = _wmma(mod_c)
        return fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (1, 1, 1))) if tiled else atom

    kwargs = {"scale_a": 127, "scale_b": 127} if scaled else {}
    if traversal is not None:
        kwargs["traversal_order"] = getattr(fx.GemmTraversalOrder, traversal)
    if atom_callback is not None:
        kwargs["atom_callback"] = atom_callback

    @flyc.kernel
    def gemm_kernel(x: fx.Tensor):
        acc = fx.make_rmem_tensor(8 if rank == 1 else (8, m, n), fx.Float32)
        acc.fill(2.0)
        fx.gemm(make_atom(), acc, fragment(m), fragment(n), acc, **kwargs)

    @flyc.jit
    def launch(x: fx.Tensor):
        gemm_kernel(x).launch(grid=(1, 1, 1), block=(32, 1, 1))

    launch(torch.empty(1, 32))


def _same(atom, mnk):
    return atom


def _rejected(mod_c, callback):
    # Without a callback the C++ expansion's pass rejects the fx.gemm; with one, the Python expansion does.
    if callback:
        return pytest.raises(
            ValueError,
            match=rf"atom_callback returned an atom with a C modifier \(mod_c = {mod_c}\) for a later K step.*"
            "return an atom without mod_c for this step",
        )
    return pytest.raises(
        DSLCompileError,
        match=rf"a gfx1250 WMMA C modifier \(mod_c = {mod_c}\) applies to every WMMA, and this GEMM has \d+ K "
        r"steps per \(m, n\) tile.*pass an atom_callback that returns an atom without mod_c",
    )


@pytest.mark.parametrize("callback", [False, True], ids=["gemm", "callback"])
@pytest.mark.parametrize("traversal", [None, "MNK", "KMN_Serpentine"])
@pytest.mark.parametrize("mod_c", [1, 2, 3, -1], ids=["neg", "abs", "neg_abs", "negative"])
def test_multi_k_gemm_rejects_mod_c(mod_c, traversal, callback):
    with _rejected(mod_c, callback):
        _compile_gemm((2, 1, 2), mod_c=mod_c, traversal=traversal, atom_callback=_same if callback else None)


@pytest.mark.parametrize("callback", [False, True], ids=["gemm", "callback"])
@pytest.mark.parametrize("variant", ["scaled", "tiled"])
def test_multi_k_scaled_or_tiled_gemm_rejects_mod_c(variant, callback):
    with _rejected(1, callback):
        _compile_gemm((1, 2, 3), **{variant: True}, atom_callback=_same if callback else None)


@pytest.mark.parametrize("callback", [False, True], ids=["gemm", "callback"])
@pytest.mark.parametrize(
    "mnk,mod_c,rank,scaled",
    [
        ((2, 1, 2), 0, 3, False),
        ((2, 1, 2), 0, 3, True),
        ((2, 2, 1), 1, 3, False),
        ((2, 2, 1), 3, 3, True),
        ((2, 2, 1), 1, 2, False),
        ((1, 1, 1), 1, 1, False),
        ((1, 1, 1), 1, 1, True),
    ],
    ids=["no_mod_c", "scaled_no_mod_c", "one_k_step", "scaled_one_k_step", "rank2", "rank1", "scaled_rank1"],
)
def test_gemm_with_one_k_step_or_no_mod_c_compiles(mnk, mod_c, rank, scaled, callback):
    _compile_gemm(mnk, mod_c=mod_c, rank=rank, scaled=scaled, atom_callback=_same if callback else None)


def test_callback_atom_decides_per_call():
    """The callback path checks the atom the callback returns for each later K step of a tile."""
    seen = set()

    def mod_c_on_first_visit(atom, mnk):
        first = mnk[:2] not in seen
        seen.add(mnk[:2])
        return atom if first else _wmma(0)

    # KMN_Serpentine visits tile (1, 0) at k = 1 first.
    _compile_gemm((2, 1, 2), mod_c=1, traversal="KMN_Serpentine", atom_callback=mod_c_on_first_visit)
    with _rejected(1, callback=True):
        _compile_gemm((1, 1, 2), mod_c=0, atom_callback=lambda atom, mnk: _wmma(1))
