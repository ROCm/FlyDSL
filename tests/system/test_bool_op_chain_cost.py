#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""E2E: a predicate chain mixing compile-time traits with dynamic comparisons costs one op per
dynamic operand, and none for the static ones.

The shape is the one real kernels use: an unrolled pipeline where every stage rebuilds a mask
from a few `traits.*` switches and a few coordinate comparisons.
"""

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.compiler.jit_function as jit_function  # noqa: E402
import flydsl.expr as fx  # noqa: E402

STAGES = 8
STATIC_TERMS = 4  # traits.* operands in the chain
DYNAMIC_TERMS = 4  # comparison operands in the chain


class _Traits:
    def __init__(self, check_m=True, check_k=True):
        self.CHECK_M = check_m
        self.CHECK_N = True
        self.GUARD_TAIL = True
        self.CHECK_K = check_k
        self.STAGES = STAGES


@flyc.kernel
def _masked_axpby_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    alpha: fx.Float32,
    beta: fx.Float32,
    traits: fx.Constexpr,
    tiled_copy: fx.TiledCopy,
):
    tid = fx.thread_idx.x
    bid_x, bid_y = fx.block_idx.x, fx.block_idx.y

    M, N = A.shape.unpack()
    idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))
    TileMN = tiled_copy.tile_mn

    gA = fx.flat_divide(A, TileMN)[None, None, bid_x, bid_y]
    gB = fx.flat_divide(B, TileMN)[None, None, bid_x, bid_y]
    gC = fx.flat_divide(C, TileMN)[None, None, bid_x, bid_y]
    cC = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]

    thr_copy = tiled_copy.get_slice(tid)
    thr_gA = thr_copy.partition_S(gA)
    thr_gB = thr_copy.partition_S(gB)
    thr_gC = thr_copy.partition_D(gC)
    thr_cC = thr_copy.partition_S(cC)[(0, None), None, None]

    thr_rA = fx.make_fragment_like(thr_gA)
    thr_rB = fx.make_fragment_like(thr_gB)
    thr_rC = fx.make_fragment_like(thr_gC)
    thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)

    n_atoms = fx.size(thr_pC.shape).unpack()

    # One unrolled pipeline stage per prefetch buffer; each rebuilds the mask.
    for _stage in fx.range_constexpr(traits.STAGES):
        for a in fx.range_constexpr(n_atoms):
            in_bounds = fx.elem_less(thr_cC[a], (M, N))
            thr_pC[a] = (
                traits.CHECK_M
                and in_bounds
                and traits.CHECK_N
                and in_bounds
                and traits.GUARD_TAIL
                and in_bounds
                and traits.CHECK_K
                and in_bounds
            )

    # A chain decided entirely at trace time is still a DSL value: a plain Python bool has no
    # `select`, and the chain's type would otherwise depend on whether its operands happened to
    # be compile-time.
    gate = traits.CHECK_N and traits.GUARD_TAIL
    assert isinstance(gate, fx.Boolean), f"a compile-time chain must stay a DSL value, got {type(gate)}"
    scale = gate.select(fx.Float32(1.0), fx.Float32(0.0))

    # A chain carries the truth of its operands: an int folds by its truth rather than being
    # returned, and a dynamic operand that is not a Boolean comes back as an i1.
    stages = traits.STAGES and traits.CHECK_N
    assert isinstance(stages, fx.Boolean), f"an int operand must fold by its truth, got {type(stages)}"
    row = tid and traits.CHECK_N
    assert isinstance(row, fx.Boolean), f"a dynamic operand must fold to a Boolean, got {type(row)}"
    assert str(row.ir_value().type) == "i1", f"the chain must be an i1, got {row.ir_value().type}"

    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    fx.copy(copy_atom, thr_gA, thr_rA, pred=thr_pC)
    fx.copy(copy_atom, thr_gB, thr_rB, pred=thr_pC)
    thr_rC.store((thr_rA.load() * alpha + thr_rB.load() * beta) * scale)
    fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)


@flyc.jit
def _masked_axpby(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    alpha: fx.Float32,
    beta: fx.Float32,
    traits: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    tiled_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_ordered_layout((8, 16), order=(1, 0)),
        fx.make_ordered_layout((1, 4), order=(0, 1)),
    )
    tile_m, tile_n = tiled_copy.tile_mn.unpack()
    M, N = A.shape.unpack()
    _masked_axpby_kernel(A, B, C, alpha, beta, traits, tiled_copy).launch(
        grid=((M + tile_m - 1) // tile_m, (N + tile_n - 1) // tile_n, 1),
        block=(8 * 16, 1, 1),
        stream=stream,
    )


def _run(traits, monkeypatch):
    """Launch once and return (andi count in the traced IR, result tensor)."""
    counts = []
    original = jit_function.JitFunction._trace

    def counting_trace(self, *args, **kwargs):
        traced = original(self, *args, **kwargs)
        if getattr(traced, "module", None) is not None:
            counts.append(traced.module.operation.get_asm(enable_debug_info=False).count("arith.andi"))
        return traced

    monkeypatch.setattr(jit_function.JitFunction, "_trace", counting_trace)

    M, N = 128, 256
    a = torch.randn(M, N, dtype=torch.float32, device="cuda")
    b = torch.randn(M, N, dtype=torch.float32, device="cuda")
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _masked_axpby(a, b, c, 2.0, 3.0, traits, stream=stream)
    torch.cuda.synchronize()
    return sum(counts), a, b, c


def test_masked_axpby_is_correct(monkeypatch):
    _, a, b, c = _run(_Traits(), monkeypatch)
    torch.testing.assert_close(c, a * 2.0 + b * 3.0, rtol=1e-5, atol=1e-5)


def test_a_chain_is_a_dsl_value_whatever_its_operands(monkeypatch):
    """The kernel asserts the chain's type while tracing, and scales the result through it.

    Without the normalisation the compile-time chain is a plain Python bool, whose `select` call
    raises, and the chain over a dynamic integer keeps that integer's width instead of i1.
    """
    _, a, b, c = _run(_Traits(), monkeypatch)
    torch.testing.assert_close(c, a * 2.0 + b * 3.0, rtol=1e-5, atol=1e-5)


def test_static_traits_in_the_chain_cost_no_ops(monkeypatch):
    """Only the dynamic operands fold, so the chain is linear in those alone."""
    andi, _, _, _ = _run(_Traits(), monkeypatch)
    assert andi == STAGES * (DYNAMIC_TERMS - 1)


def test_a_false_trait_stops_the_chain_where_it_sits(monkeypatch):
    """CHECK_K is the seventh operand, so the three before it still fold, and the one after
    it is never even evaluated."""
    andi, _, _, _ = _run(_Traits(check_k=False), monkeypatch)
    assert andi == STAGES * (DYNAMIC_TERMS - 2)
    assert andi < STAGES * (DYNAMIC_TERMS - 1)


def test_a_leading_false_trait_removes_the_chain(monkeypatch):
    """CHECK_M comes first, so a False there decides the chain before any comparison runs."""
    andi, _, _, _ = _run(_Traits(check_m=False), monkeypatch)
    assert andi == 0
