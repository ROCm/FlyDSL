#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for flydsl.expr.arith DSL wrappers.

Currently focused on ``maxnumf`` (libm ``fmax`` semantics), which lowers to the
MLIR ``arith.maxnumf`` op and preserves the DSL type of its first operand.
"""

import sys

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _raw_arith
from flydsl._mlir.dialects import func
from flydsl.expr import arith as fly_arith
from flydsl.expr.numeric import Float32


def _build_module(build_fn, arg_types=None):
    """Build an MLIR module with a function that calls build_fn(args...) and return its IR text."""
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            if arg_types is None:
                types = [ir.F32Type.get()]
            else:
                types = [t() if callable(t) else t for t in arg_types]
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                ftype = ir.FunctionType.get(types, [])
                f = func.FuncOp("test", ftype)
                with ir.InsertionPoint(f.add_entry_block()):
                    args = list(f.entry_block.arguments)
                    build_fn(*args)
                    func.ReturnOp([])
            module.operation.verify()
            return str(module)


# ---------------------------------------------------------------------------
# maxnumf — compile-tier tests
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_maxnumf_op():
    """maxnumf emits the arith.maxnumf op."""

    def build(x):
        fly_arith.maxnumf(x, x)

    ir_text = _build_module(build)
    assert "arith.maxnumf" in ir_text


@pytest.mark.l0_backend_agnostic
def test_maxnumf_wrapper_overrides_raw():
    """fly_arith.maxnumf must be our wrapper, not the raw MLIR binding."""
    assert fly_arith.maxnumf is not _raw_arith.maxnumf, "fly_arith.maxnumf is still the raw MLIR function"
    assert fly_arith.maxnumf.__closure__ is not None, "fly_arith.maxnumf has no closure (not wrapped)"


@pytest.mark.l0_backend_agnostic
def test_maxnumf_exported_via_fx():
    """fx.maxnumf should resolve to the arith wrapper (exported through expr.__init__)."""
    import flydsl.expr as fx

    assert fx.maxnumf is fly_arith.maxnumf


@pytest.mark.l0_backend_agnostic
def test_maxnumf_numeric_unwrap():
    """maxnumf should accept Float32 DSL inputs and auto-unwrap them."""

    def build(x_raw):
        x = Float32(x_raw)
        fly_arith.maxnumf(x, x)

    ir_text = _build_module(build)
    assert "arith.maxnumf" in ir_text


@pytest.mark.l0_backend_agnostic
def test_maxnumf_class_invariance():
    """Float32 in → Float32 out, so results can be chained with DSL ops."""

    def build(x_raw):
        x = Float32(x_raw)
        y = fly_arith.maxnumf(x, x)
        assert isinstance(y, Float32), f"maxnumf: expected Float32, got {type(y).__name__}"

    _build_module(build)


@pytest.mark.l0_backend_agnostic
def test_maxnumf_vector():
    """maxnumf works elementwise on vector<4xf32> inputs."""

    def build(x):
        vtype = ir.VectorType.get([4], ir.F32Type.get())
        splat = _raw_arith.ConstantOp(
            vtype,
            ir.DenseElementsAttr.get_splat(vtype, ir.FloatAttr.get(ir.F32Type.get(), 1.0)),
        ).result
        fly_arith.maxnumf(splat, splat)

    ir_text = _build_module(build)
    assert "vector<4xf32>" in ir_text
    assert "arith.maxnumf" in ir_text


@pytest.mark.l0_backend_agnostic
def test_maxnumf_raw_value_passthrough():
    """Raw ir.Value input should NOT be wrapped in a Numeric."""

    def build(x_raw):
        y = fly_arith.maxnumf(x_raw, x_raw)
        assert not isinstance(y, Float32), f"raw input should not produce Float32, got {type(y).__name__}"

    _build_module(build)


# ---------------------------------------------------------------------------
# maxnumf — end-to-end GPU correctness
# ---------------------------------------------------------------------------

try:
    import torch as _torch

    _HAS_GPU = _torch.cuda.is_available()
except ImportError:
    _torch = None
    _HAS_GPU = False

_gpu_skip = pytest.mark.skipif(not _HAS_GPU, reason="CUDA/ROCm not available")


@_gpu_skip
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_maxnumf_gpu():
    """C = maxnumf(A, B) — elementwise max with libm fmax NaN semantics."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    VEC_WIDTH = 4
    BLOCK_DIM = 256
    TILE_ELEMS = BLOCK_DIM * VEC_WIDTH
    N = TILE_ELEMS * 64

    @flyc.kernel
    def maxnumf_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile_elems = block_dim * vec_width

        tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
        tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
        tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
        tA = fx.slice(tA, (None, bid))
        tB = fx.slice(tB, (None, bid))
        tC = fx.slice(tC, (None, bid))
        tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
        tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
        tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

        copy_bits = vec_width * 32
        copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

        rA = fx.make_rmem_tensor(vec_width, fx.Float32)
        rB = fx.make_rmem_tensor(vec_width, fx.Float32)
        rC = fx.make_rmem_tensor(vec_width, fx.Float32)

        fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
        fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

        vC = fx.maxnumf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
        fx.memref_store_vec(vC, rC)

        fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C,
        n: fx.Int32,
        const_n: fx.Constexpr[int],
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        tile = block_dim * vec_width
        grid_x = (n + tile - 1) // tile
        maxnumf_kernel(A, B, C, block_dim, vec_width).launch(
            grid=(grid_x, 1, 1),
            block=(block_dim, 1, 1),
            stream=stream,
        )

    a_host = _torch.empty(N, dtype=_torch.float32).uniform_(-5.0, 5.0)
    b_host = _torch.empty(N, dtype=_torch.float32).uniform_(-5.0, 5.0)
    # Sprinkle NaNs into B to exercise the non-NaN-propagating (fmax) semantics.
    b_host[::97] = float("nan")
    a_dev = a_host.cuda()
    b_dev = b_host.cuda()
    c_dev = _torch.empty_like(a_dev)

    tA = flyc.from_torch_tensor(a_dev).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)
    tB = flyc.from_torch_tensor(b_dev).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)

    stream = _torch.cuda.Stream()
    launch(tA, tB, c_dev, N, N, BLOCK_DIM, VEC_WIDTH, stream=stream)
    _torch.cuda.synchronize()

    # torch.fmax matches arith.maxnumf: returns the non-NaN operand on NaN.
    c_ref = _torch.fmax(a_host, b_host).cuda()
    assert _torch.equal(
        c_dev, c_ref
    ), f"maxnumf GPU mismatch: max_diff={(_torch.abs(c_dev - c_ref).nan_to_num()).max().item():.6e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
