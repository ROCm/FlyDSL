# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""SimpleNamespace can be built inside @flyc.jit, but not passed in from host."""

from types import SimpleNamespace

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

BLOCK_DIM = 256
VEC_WIDTH = 4
TILE_ELEMS = BLOCK_DIM * VEC_WIDTH
SIZE = TILE_ELEMS * 100


@flyc.kernel
def _vecadd_kernel_ns(
    bundle: SimpleNamespace,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    A = bundle.A
    B = bundle.B
    C = bundle.C

    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile = block_dim * vec_width

    tA = fx.slice(fx.logical_divide(A, fx.make_layout(tile, 1)), (None, bid))
    tB = fx.slice(fx.logical_divide(B, fx.make_layout(tile, 1)), (None, bid))
    tC = fx.slice(fx.logical_divide(C, fx.make_layout(tile, 1)), (None, bid))

    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

    atom = fx.make_copy_atom(fx.UniversalCopy(vec_width * fx.Float32.width), fx.Float32)

    rA = fx.make_fragment_like(fx.make_layout(vec_width, 1), fx.Float32)
    rB = fx.make_fragment_like(fx.make_layout(vec_width, 1), fx.Float32)
    rC = fx.make_fragment_like(fx.make_layout(vec_width, 1), fx.Float32)

    fx.copy_atom_call(atom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(atom, fx.slice(tB, (None, tid)), rB)
    fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB)), rC)
    fx.copy_atom_call(atom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def _vecadd_ns(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    size: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    bundle = SimpleNamespace(A=A, B=B, C=C, SIZE=size)
    tile = block_dim * vec_width
    grid_x = (bundle.SIZE + tile - 1) // tile
    _vecadd_kernel_ns(bundle, block_dim, vec_width).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


@pytest.fixture
def abc_tensors():
    a = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    b = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    return a, b, c


@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_simple_namespace_built_inside_jit_passes_to_kernel(abc_tensors):
    a, b, c = abc_tensors
    _vecadd_ns(a, b, c, SIZE, BLOCK_DIM, VEC_WIDTH)
    torch.cuda.synchronize()
    assert torch.allclose(c.cpu(), a.cpu() + b.cpu(), atol=1e-5)
