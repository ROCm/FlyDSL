# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Pytest harness for the FlyDSL puzzles.

By default it imports the **reference solutions** (``puzzles/solutions/``) and
validates each against its torch reference from ``common.py``. To grade your own
answers instead, change ``SOLUTION_PKG`` below to ``"puzzles"`` so it imports the
skeletons you filled in.

Every solution module exposes a uniform ``run(*inputs) -> torch.Tensor`` entry
that allocates outputs, launches the kernel, synchronizes, and returns the
result. The tests here only construct inputs and compare.

These tests execute on the GPU and require an AMD CDNA device (gfx942/gfx950).
"""

from __future__ import annotations

import importlib

import pytest
import torch

from puzzles import common

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

# Flip to "puzzles" to grade the skeletons you edited instead of the solutions.
SOLUTION_PKG = "puzzles.solutions"


def _load(mod: str):
    return importlib.import_module(f"{SOLUTION_PKG}.{mod}")


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CDNA GPU visible to torch")


def _rand(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype, device="cuda")


# --------------------------------------------------------------------------- #
# A. Warmup
# --------------------------------------------------------------------------- #


def test_puzzle01_copy():
    _skip_if_no_gpu()
    a = _rand(8 * 3, 24 * 5)  # (24, 120): tiles evenly by (8, 24)
    out = _load("puzzle01_copy").run(a)
    assert common.check(out, common.ref_copy(a), name="puzzle01 copy")


def test_puzzle02_vector_add():
    _skip_if_no_gpu()
    a, b = _rand(100, 1000), _rand(100, 1000)
    out = _load("puzzle02_vector_add").run(a, b)
    assert common.check(out, common.ref_vector_add(a, b), name="puzzle02 vector_add")


def test_puzzle03_scale_bias():
    _skip_if_no_gpu()
    a = _rand(64, 256)
    alpha, beta = 2.5, -1.25
    out = _load("puzzle03_scale_bias").run(a, alpha, beta)
    assert common.check(out, common.ref_scale_bias(a, alpha, beta), name="puzzle03 scale_bias")


def test_puzzle04_tiled_copy():
    _skip_if_no_gpu()
    a = _rand(8 * 3, 24 * 5)
    out = _load("puzzle04_tiled_copy").run(a)
    assert common.check(out, common.ref_copy(a), name="puzzle04 tiled_copy")


def test_puzzle05_transpose():
    _skip_if_no_gpu()
    a = _rand(64, 48)  # non-square, both multiples of 16
    out = _load("puzzle05_transpose").run(a)
    assert common.check(out, common.ref_transpose(a), name="puzzle05 transpose")


# --------------------------------------------------------------------------- #
# B. Layout & reductions
# --------------------------------------------------------------------------- #


def test_puzzle06_row_sum():
    _skip_if_no_gpu()
    a = _rand(128, 512)
    out = _load("puzzle06_row_sum").run(a)
    assert common.check(out, common.ref_row_sum(a), name="puzzle06 row_sum", atol=1e-2, rtol=1e-2)


def test_puzzle07_softmax():
    _skip_if_no_gpu()
    a = _rand(128, 512)
    out = _load("puzzle07_softmax").run(a)
    assert common.check(out, common.ref_softmax(a), name="puzzle07 softmax")


def test_puzzle08_rmsnorm():
    _skip_if_no_gpu()
    a = _rand(128, 512)
    gamma = _rand(512)
    out = _load("puzzle08_rmsnorm").run(a, gamma)
    assert common.check(out, common.ref_rmsnorm(a, gamma), name="puzzle08 rmsnorm")


# --------------------------------------------------------------------------- #
# C. Efficient GEMM pipeline
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mod", ["puzzle09_gemm_single_tile"])
def test_puzzle09_gemm_single_tile(mod):
    _skip_if_no_gpu()
    m, n, k = 64, 64, 8
    a, b = _rand(m, k), _rand(n, k)
    out = _load(mod).run(a, b)
    assert common.check(out, common.ref_gemm(a, b), name=mod, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "mod",
    [
        "puzzle10_gemm_kloop",
        "puzzle11_gemm_lds",
        "puzzle12_gemm_double_buffer",
        "puzzle13_gemm_swizzled",
    ],
)
def test_puzzle_gemm_full(mod):
    _skip_if_no_gpu()
    m, n, k = 256, 256, 256
    a, b = _rand(m, k), _rand(n, k)
    out = _load(mod).run(a, b)
    assert common.check(out, common.ref_gemm(a, b), name=mod, atol=5e-2, rtol=5e-2)


# --------------------------------------------------------------------------- #
# D. Applications
# --------------------------------------------------------------------------- #


def test_puzzle14_flash_attention():
    _skip_if_no_gpu()
    s, d = 128, 64
    q, k, v = _rand(s, d), _rand(s, d), _rand(s, d)
    out = _load("puzzle14_flash_attention").run(q, k, v)
    assert common.check(out, common.ref_flash_attention(q, k, v), name="puzzle14 flash_attn", atol=5e-2, rtol=5e-2)


def test_puzzle15_conv2d():
    _skip_if_no_gpu()
    x = _rand(1, 8, 16, 16)
    w = _rand(16, 8, 3, 3)
    out = _load("puzzle15_conv2d").run(x, w)
    assert common.check(out, common.ref_conv2d(x, w), name="puzzle15 conv2d", atol=5e-2, rtol=5e-2)
