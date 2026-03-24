# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Integration tests: GPU kernels with raw int stream and torch.nn.Parameter.

Validates end-to-end that:
  - Raw int cuda_stream (as passed by aiter) works correctly
  - torch.nn.Parameter (Tensor subclass) is accepted as kernel input
  - Stream value is preserved through JIT compilation and kernel launch
"""
import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx


# ──────────────────────────────────────────────────────────────
# Minimal vecadd kernel for testing
# ──────────────────────────────────────────────────────────────

BLOCK_DIM = 256
VEC_WIDTH = 4
TILE_ELEMS = BLOCK_DIM * VEC_WIDTH


@flyc.kernel
def _vecadd_kernel(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile = block_dim * vec_width

    tA = fx.slice(fx.logical_divide(A, fx.make_layout(tile, 1)), (None, bid))
    tB = fx.slice(fx.logical_divide(B, fx.make_layout(tile, 1)), (None, bid))
    tC = fx.slice(fx.logical_divide(C, fx.make_layout(tile, 1)), (None, bid))

    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

    copy_bits = vec_width * 32
    reg_ty = fx.MemRefType.get(
        fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register)
    atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

    rA = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rB = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rC = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))

    fx.copy_atom_call(atom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(atom, fx.slice(tB, (None, tid)), rB)
    fx.memref_store_vec(
        fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB)), rC)
    fx.copy_atom_call(atom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def _vecadd(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    n: fx.Int32,
    block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile = block_dim * vec_width
    grid_x = (n + tile - 1) // tile
    _vecadd_kernel(A, B, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

SIZE = TILE_ELEMS * 100


@pytest.fixture
def abc_tensors():
    a = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    b = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    return a, b, c


# ──────────────────────────────────────────────────────────────
# Stream tests
# ──────────────────────────────────────────────────────────────

class TestKernelStream:
    """Kernel launch with different stream passing styles."""

    def test_default_stream(self, abc_tensors):
        a, b, c = abc_tensors
        _vecadd(a, b, c, SIZE, BLOCK_DIM, VEC_WIDTH)
        torch.cuda.synchronize()
        assert torch.allclose(c, a + b, atol=1e-5)

    def test_torch_stream_object(self, abc_tensors):
        a, b, c = abc_tensors
        stream = torch.cuda.Stream()
        _vecadd(a, b, c, SIZE, BLOCK_DIM, VEC_WIDTH, stream=stream)
        torch.cuda.synchronize()
        assert torch.allclose(c, a + b, atol=1e-5)

    def test_raw_int_stream(self, abc_tensors):
        """aiter passes cuda_stream as raw int."""
        a, b, c = abc_tensors
        stream = torch.cuda.Stream()
        _vecadd(a, b, c, SIZE, BLOCK_DIM, VEC_WIDTH, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        assert torch.allclose(c, a + b, atol=1e-5)

    def test_raw_int_stream_correctness(self, abc_tensors):
        """Raw int stream must produce same result as Stream object."""
        a, b, c1 = abc_tensors
        c2 = torch.empty_like(a)
        stream = torch.cuda.Stream()

        _vecadd(a, b, c1, SIZE, BLOCK_DIM, VEC_WIDTH, stream=stream)
        _vecadd(a, b, c2, SIZE, BLOCK_DIM, VEC_WIDTH, stream=stream.cuda_stream)
        torch.cuda.synchronize()

        assert torch.equal(c1, c2), "Raw int stream should produce identical results"

    def test_current_stream_as_int(self, abc_tensors):
        """Passing current stream's raw pointer (common in vllm/aiter)."""
        a, b, c = abc_tensors
        raw = torch.cuda.current_stream().cuda_stream
        _vecadd(a, b, c, SIZE, BLOCK_DIM, VEC_WIDTH, stream=raw)
        torch.cuda.synchronize()
        assert torch.allclose(c, a + b, atol=1e-5)


# ──────────────────────────────────────────────────────────────
# torch.nn.Parameter tests
# ──────────────────────────────────────────────────────────────

class TestKernelParameter:
    """Kernel launch with torch.nn.Parameter inputs."""

    def test_parameter_as_input(self):
        """torch.nn.Parameter (Tensor subclass) should be accepted."""
        w = torch.nn.Parameter(
            torch.randn(SIZE, device="cuda", dtype=torch.float32),
            requires_grad=False,
        )
        b = torch.randn(SIZE, device="cuda", dtype=torch.float32)
        c = torch.empty_like(b)
        _vecadd(w, b, c, SIZE, BLOCK_DIM, VEC_WIDTH)
        torch.cuda.synchronize()
        assert torch.allclose(c, w.data + b, atol=1e-5)

    def test_parameter_with_raw_int_stream(self):
        """Parameter + raw int stream together (aiter's exact pattern)."""
        w = torch.nn.Parameter(
            torch.randn(SIZE, device="cuda", dtype=torch.float32),
            requires_grad=False,
        )
        b = torch.randn(SIZE, device="cuda", dtype=torch.float32)
        c = torch.empty_like(b)
        stream = torch.cuda.Stream()
        _vecadd(w, b, c, SIZE, BLOCK_DIM, VEC_WIDTH, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        assert torch.allclose(c, w.data + b, atol=1e-5)

    def test_multiple_parameters(self):
        """Multiple Parameters as inputs."""
        a = torch.nn.Parameter(
            torch.randn(SIZE, device="cuda", dtype=torch.float32),
            requires_grad=False,
        )
        b = torch.nn.Parameter(
            torch.randn(SIZE, device="cuda", dtype=torch.float32),
            requires_grad=False,
        )
        c = torch.empty(SIZE, device="cuda", dtype=torch.float32)
        _vecadd(a, b, c, SIZE, BLOCK_DIM, VEC_WIDTH)
        torch.cuda.synchronize()
        assert torch.allclose(c, a.data + b.data, atol=1e-5)
