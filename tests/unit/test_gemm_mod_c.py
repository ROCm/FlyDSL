# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""An MMA atom that modifies C on every call (``MmaAtomType.modifies_c``, e.g. a gfx1250 WMMA with
``mod_c``) is rejected on a tile's later K steps, where C is the partial sum in D; an ``atom_callback`` may
still use it on each tile's first K step."""

import os
import subprocess
import sys

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.diagnostics import DSLCompileError

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@pytest.fixture(autouse=True)
def _gfx1250_compile_only(monkeypatch):
    for name, value in {"FLYDSL_GPU_ARCH": "gfx1250", "COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0"}.items():
        monkeypatch.setenv(name, value)


def _wmma(mod_c):
    return fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32, mod_c=mod_c))


def _compile_gemm(mnk, *, mod_c=1, scaled=False, fma=False, rank=3, tiled=False, traversal=None, atom_callback=None):
    """Compiles one gfx1250 ``fx.gemm`` on register fragments; rank 2 drops K from A and B."""
    m, n, k = mnk
    v, dtype = (64, fx.Float8E4M3FN) if scaled else ((1, fx.Float32) if fma else (16, fx.BFloat16))
    acc_rows = 1 if fma else 8

    def fragment(rows):
        return fx.make_rmem_tensor(v if rank == 1 else ((v, rows) if rank == 2 else (v, rows, k)), dtype)

    def make_atom():
        if fma:
            return fx.make_mma_atom(fx.UniversalFMA(fx.Float32))
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
        acc = fx.make_rmem_tensor(acc_rows if rank == 1 else (acc_rows, m, n), fx.Float32)
        acc.fill(2.0)
        fx.gemm(make_atom(), acc, fragment(m), fragment(n), acc, **kwargs)

    @flyc.jit
    def launch(x: fx.Tensor):
        gemm_kernel(x).launch(grid=(1, 1, 1), block=(32, 1, 1))

    launch(torch.empty(1, 32))


def _same(atom, mnk):
    return atom


def _rejected(callback):
    # Without a callback the C++ expansion's pass rejects the fx.gemm; with one, the Python expansion does.
    if callback:
        return pytest.raises(
            ValueError,
            match=r"atom_callback returned an atom with a C modifier \(mod_c\) for a later K step.*"
            "return an atom without mod_c for this step",
        )
    return pytest.raises(
        DSLCompileError,
        match=r"MMA atom applies a C modifier \(mod_c\) on every call; this GEMM has \d+ K steps per \(m, n\) "
        "tile.*pass an atom_callback that returns an atom without mod_c",
    )


@pytest.mark.parametrize("callback", [False, True], ids=["gemm", "callback"])
@pytest.mark.parametrize("traversal", [None, "MNK", "KMN_Serpentine"])
@pytest.mark.parametrize("mod_c", [1, 2, 3, -1], ids=["neg", "abs", "neg_abs", "negative"])
def test_multi_k_gemm_rejects_mod_c(mod_c, traversal, callback):
    with _rejected(callback):
        _compile_gemm((2, 1, 2), mod_c=mod_c, traversal=traversal, atom_callback=_same if callback else None)


@pytest.mark.parametrize("callback", [False, True], ids=["gemm", "callback"])
@pytest.mark.parametrize("variant", ["scaled", "tiled"])
def test_multi_k_scaled_or_tiled_gemm_rejects_mod_c(variant, callback):
    with _rejected(callback):
        _compile_gemm((1, 2, 3), **{variant: True}, atom_callback=_same if callback else None)


@pytest.mark.parametrize("callback", [False, True], ids=["gemm", "callback"])
@pytest.mark.parametrize(
    "mnk,mod_c,rank,atom",
    [
        ((2, 1, 2), 0, 3, "wmma"),
        ((2, 1, 2), 0, 3, "scaled"),
        ((2, 1, 2), 0, 3, "fma"),
        ((2, 2, 1), 1, 3, "wmma"),
        ((2, 2, 1), 3, 3, "scaled"),
        ((2, 2, 1), 1, 2, "wmma"),
        ((1, 1, 1), 1, 1, "wmma"),
        ((1, 1, 1), 1, 1, "scaled"),
    ],
    ids=["no_mod_c", "scaled_no_mod_c", "fma", "one_k_step", "scaled_one_k_step", "rank2", "rank1", "scaled_rank1"],
)
def test_gemm_with_one_k_step_or_no_mod_c_compiles(mnk, mod_c, rank, atom, callback):
    _compile_gemm(
        mnk,
        mod_c=mod_c,
        rank=rank,
        scaled=atom == "scaled",
        fma=atom == "fma",
        atom_callback=_same if callback else None,
    )


def test_callback_atom_decides_per_call():
    """The callback path checks the atom the callback returns for each later K step of a tile."""
    seen = set()

    def mod_c_on_first_visit(atom, mnk):
        first = mnk[:2] not in seen
        seen.add(mnk[:2])
        return atom if first else _wmma(0)

    # KMN_Serpentine visits tile (1, 0) at k = 1 first.
    _compile_gemm((2, 1, 2), mod_c=1, traversal="KMN_Serpentine", atom_callback=mod_c_on_first_visit)
    with _rejected(callback=True):
        _compile_gemm((1, 1, 2), mod_c=0, atom_callback=lambda atom, mnk: _wmma(1))


WMMA = "!fly_rocdl.gfx1250.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false{}>"
SCALE = (
    "!fly_rocdl.gfx1250.wmma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0, "
    "modC = {}, reuseA = false, reuseB = false, blockSize = 32>"
)


@pytest.mark.parametrize(
    "op,modifies",
    [
        (WMMA.format(""), False),
        (WMMA.format(", modC = 1"), True),
        (WMMA.format(", modC = 3, reuseA = true"), True),
        (WMMA.format(", modC = -1"), True),
        (SCALE.format(0), False),
        (SCALE.format(2), True),
        ("!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>", False),
        ("!fly_rocdl.cdna4.mfma_scale<16x16x128, (f8E4M3FN, f8E4M3FN) -> f32, opselA = 0, opselB = 0>", False),
        ("!fly_rocdl.gfx11.wmma<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>", False),
        ("!fly_rocdl.gfx120x.wmma<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>", False),
    ],
    ids=["wmma", "wmma_neg", "wmma_neg_abs_reuse", "wmma_negative", "scale", "scale_abs", "cdna3", "cdna4_scale"]
    + ["gfx11", "gfx120x"],
)
def test_modifies_c_query(op, modifies):
    """Only a gfx1250 WMMA or WMMAScale with a nonzero mod_c modifies C; other atoms keep the default."""
    with ir.Context(), ir.Location.unknown():
        assert fx.MmaAtomType.get(mma_op=ir.Type.parse(op)).modifies_c is modifies


_NO_TARGET_MODULES = """
import importlib.abc
import sys

BLOCKED = {"flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL", "flydsl._mlir.dialects.fly_rocdl", "flydsl._mlir.dialects.rocdl"}


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in BLOCKED:
            raise ModuleNotFoundError("blocked target module", name=fullname)
        return None


sys.meta_path.insert(0, _Block())
import flydsl.expr as fx
from flydsl._mlir import ir

with ir.Context(), ir.Location.unknown():
    assert fx.MmaAtomType.get(mma_op=fx.UniversalFMA(fx.Float32)).modifies_c is False
"""


def test_query_needs_no_target_module():
    """The query is on the target-neutral MmaAtomType, so expr can ask any atom without a target module."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", _NO_TARGET_MODULES], check=True, env=env, timeout=300)
