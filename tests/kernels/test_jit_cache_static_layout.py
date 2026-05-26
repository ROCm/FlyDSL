# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regression coverage for JIT cache keys with static tensor layouts."""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@flyc.kernel
def _vec_add_kernel(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width

    ta = fx.logical_divide(a, fx.make_layout(tile_elems, 1))
    tb = fx.logical_divide(b, fx.make_layout(tile_elems, 1))
    tc = fx.logical_divide(c, fx.make_layout(tile_elems, 1))

    ta = fx.slice(ta, (None, bid))
    tb = fx.slice(tb, (None, bid))
    tc = fx.slice(tc, (None, bid))

    ta = fx.logical_divide(ta, fx.make_layout(vec_width, 1))
    tb = fx.logical_divide(tb, fx.make_layout(vec_width, 1))
    tc = fx.logical_divide(tc, fx.make_layout(vec_width, 1))

    copy_atom = fx.make_copy_atom(fx.UniversalCopy(vec_width * 32), fx.Float32)
    ra = fx.make_rmem_tensor(vec_width, fx.Float32)
    rb = fx.make_rmem_tensor(vec_width, fx.Float32)
    rc = fx.make_rmem_tensor(vec_width, fx.Float32)

    fx.copy_atom_call(copy_atom, fx.slice(ta, (None, tid)), ra)
    fx.copy_atom_call(copy_atom, fx.slice(tb, (None, tid)), rb)
    fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(ra), fx.memref_load_vec(rb)), rc)
    fx.copy_atom_call(copy_atom, rc, fx.slice(tc, (None, tid)))


@flyc.jit
def _vec_add(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    _vec_add_kernel(a, b, c, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


def _run_vec_add(n: int):
    a = torch.ones(n, 1, dtype=torch.float32, device="cuda")
    b = torch.ones(n, 1, dtype=torch.float32, device="cuda")
    c = torch.zeros(n, 1, dtype=torch.float32, device="cuda")

    _vec_add(a, b, c, n, 256, 8, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return c, a + b


def test_static_layout_tensor_shapes_do_not_reuse_stale_jit_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))

    first, first_expected = _run_vec_add(512)
    second, second_expected = _run_vec_add(4096)

    assert torch.allclose(first, first_expected)
    assert torch.allclose(second, second_expected)
