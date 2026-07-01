#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for the ambient ``fx.fastmath(...)`` context manager.

Verifies that:
1. Float operators (+, -, *, /, %) inside the block emit fastmath flags.
2. ``math`` functions inherit the ambient flags.
3. An explicit ``fastmath=`` argument overrides the ambient context.
4. Flags are restored on block exit (including nested blocks).
5. Integer operators are unaffected.
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func
from flydsl.expr.numeric import Float32, Int32

try:
    import torch
except ImportError:
    torch = None

# GPU gating is applied per-test via ``requires_gpu``; the ``l0_backend_agnostic``
# IR-string tests below build IR without a device and must run on non-GPU runners.
requires_gpu = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="CUDA/ROCm not available",
)


def _build(build_fn, arg_types):
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                ftype = ir.FunctionType.get([t() for t in arg_types], [])
                f = func.FuncOp("test", ftype)
                with ir.InsertionPoint(f.add_entry_block()):
                    build_fn(*f.entry_block.arguments)
                    func.ReturnOp([])
            return str(module)


@pytest.mark.l0_backend_agnostic
def test_operators_pick_up_ambient_fastmath():
    def build(a, b):
        fa, fb = Float32(a), Float32(b)
        with fx.fastmath(fx.FastMathFlags.fast):
            _ = fa + fb
            _ = fa - fb
            _ = fa * fb
            _ = fa / fb

    ir_text = _build(build, [ir.F32Type.get, ir.F32Type.get])
    assert "arith.addf" in ir_text and "fastmath<fast>" in ir_text
    assert ir_text.count("fastmath<fast>") >= 4


@pytest.mark.l0_backend_agnostic
def test_no_flag_outside_block():
    def build(a, b):
        fa, fb = Float32(a), Float32(b)
        _ = fa + fb

    ir_text = _build(build, [ir.F32Type.get, ir.F32Type.get])
    assert "fastmath" not in ir_text


@pytest.mark.l0_backend_agnostic
def test_math_function_inherits_ambient():
    def build(a):
        fa = Float32(a)
        with fx.fastmath(fx.FastMathFlags.fast):
            fx.exp(fa)

    ir_text = _build(build, [ir.F32Type.get])
    assert "math.exp" in ir_text and "fastmath<fast>" in ir_text


@pytest.mark.l0_backend_agnostic
def test_explicit_arg_overrides_ambient():
    def build(a):
        fa = Float32(a)
        with fx.fastmath(fx.FastMathFlags.fast):
            fx.exp(fa, fastmath="contract")

    ir_text = _build(build, [ir.F32Type.get])
    assert "fastmath<contract>" in ir_text
    assert "fastmath<fast>" not in ir_text


@pytest.mark.l0_backend_agnostic
def test_combined_flags():
    def build(a, b):
        fa, fb = Float32(a), Float32(b)
        with fx.fastmath(fx.FastMathFlags.reassoc | fx.FastMathFlags.contract):
            _ = fa + fb

    ir_text = _build(build, [ir.F32Type.get, ir.F32Type.get])
    assert "fastmath<reassoc,contract>" in ir_text


@pytest.mark.l0_backend_agnostic
def test_nested_blocks_restore():
    def build(a, b):
        fa, fb = Float32(a), Float32(b)
        with fx.fastmath(fx.FastMathFlags.fast):
            with fx.fastmath(fx.FastMathFlags.contract):
                _ = fa * fb  # contract
            _ = fa + fb  # back to fast
        _ = fa - fb  # no flag

    ir_text = _build(build, [ir.F32Type.get, ir.F32Type.get])
    lines = [ln for ln in ir_text.splitlines() if "arith." in ln]
    mul = next(ln for ln in lines if "mulf" in ln)
    add = next(ln for ln in lines if "addf" in ln)
    sub = next(ln for ln in lines if "subf" in ln)
    assert "fastmath<contract>" in mul
    assert "fastmath<fast>" in add
    assert "fastmath" not in sub


@pytest.mark.l0_backend_agnostic
def test_neg_and_pow_operators_pick_up_ambient():
    """Unary negation (negf) and power (**, math.powf) also inherit ambient."""

    def build(a, b):
        fa, fb = Float32(a), Float32(b)
        _ = -fa  # outside → no flag
        _ = fa**fb  # outside → no flag
        with fx.fastmath(fx.FastMathFlags.fast):
            _ = -fa  # negf inherits
            _ = fa**fb  # powf inherits

    ir_text = _build(build, [ir.F32Type.get, ir.F32Type.get])
    negs = [ln for ln in ir_text.splitlines() if "arith.negf" in ln]
    pows = [ln for ln in ir_text.splitlines() if "math.powf" in ln]
    assert "fastmath" not in negs[0] and "fastmath<fast>" in negs[1]
    assert "fastmath" not in pows[0] and "fastmath<fast>" in pows[1]


@pytest.mark.l0_backend_agnostic
def test_named_methods_inherit_ambient():
    """ArithValue/Numeric named fastmath methods (addf/exp2) inherit ambient."""

    def build(a, b):
        fa, fb = Float32(a), Float32(b)
        _ = fa + fb  # outside → no flag
        with fx.fastmath(fx.FastMathFlags.fast):
            _ = fa + fb  # inherits ambient
            _ = fa.exp2()  # inherits ambient
            _ = fa.addf(fb, fastmath="none")  # explicit overrides

    ir_text = _build(build, [ir.F32Type.get, ir.F32Type.get])
    addf_lines = [ln for ln in ir_text.splitlines() if "arith.addf" in ln]
    assert "fastmath" not in addf_lines[0]  # outside block
    assert "fastmath<fast>" in addf_lines[1]  # inherited
    assert "fastmath" not in addf_lines[2]  # explicit "none" overrides → default (omitted)
    assert any("math.exp2" in ln and "fastmath<fast>" in ln for ln in ir_text.splitlines())


@pytest.mark.l0_backend_agnostic
def test_integer_ops_unaffected():
    def build(a, b):
        ia, ib = Int32(a), Int32(b)
        with fx.fastmath(fx.FastMathFlags.fast):
            _ = ia + ib
            _ = ia * ib

    def i32():
        return ir.IntegerType.get_signless(32)

    ir_text = _build(build, [i32, i32])
    assert "arith.addi" in ir_text and "arith.muli" in ir_text
    assert "fastmath" not in ir_text


@flyc.kernel
def _fm_kernel():
    tid = fx.thread_idx.x
    x = fx.Float32(tid)
    z = x * x + x  # operators → kernel-level flag
    _ = fx.exp(z)  # math fn → inherits kernel-level flag

    with fx.fastmath(fx.FastMathFlags.contract):
        _ = x * z  # block overrides kernel |
        _ = fx.exp(x, fastmath="none")  # explicit op-level override


@flyc.jit
def _fm_launch_plain(stream: fx.Stream = fx.Stream(None)):
    _fm_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@flyc.jit
def _fm_launch_hinted(stream: fx.Stream = fx.Stream(None)):
    _fm_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def _source_ir(launch_fn):
    launch_fn(stream=torch.cuda.current_stream())
    assert launch_fn._mem_cache, "expected at least one cached compilation"
    return next(iter(launch_fn._mem_cache.values())).source_ir


def _arith_lines(ir_text, needle):
    return [ln.strip() for ln in ir_text.splitlines() if needle in ln and ln.strip().startswith("%")]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
def test_no_hint_only_block_scope_applies():
    """Without a kernel hint, kernel-level ops carry no flag, but an inner
    ``with fx.fastmath`` block still applies (block is independent of kernel)."""
    ir_text = _source_ir(_fm_launch_plain)
    muls = _arith_lines(ir_text, "arith.mulf")  # [0]=x*x, [1]=x*z(block)
    assert "fastmath" not in muls[0]  # kernel-level op, no hint → plain
    assert "fastmath<contract>" in muls[1]  # inner block still applies
    assert all("fastmath" not in ln for ln in _arith_lines(ir_text, "arith.addf"))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
def test_kernel_level_fastmath_and_scope_overrides():
    hinted = flyc.compile[{"fastmath": "fast"}](_fm_launch_hinted)
    ir_text = _source_ir(hinted)

    muls = _arith_lines(ir_text, "arith.mulf")  # [0]=x*x, [1]=x*z(block)
    assert "fastmath<fast>" in muls[0]  # kernel-level fast
    assert "fastmath<contract>" in muls[1]  # block overrides kernel
    # addf x*x+x → kernel-level fast
    assert all("fastmath<fast>" in ln for ln in _arith_lines(ir_text, "arith.addf"))
    # math.exp(z) inherits fast; math.exp(x, "none") explicitly opts out
    exps = _arith_lines(ir_text, "math.exp")  # [0]=exp(z), [1]=exp(x,none)
    assert "fastmath<fast>" in exps[0]  # inherits kernel-level
    assert "fastmath" not in exps[1]  # explicit op-level "none" override


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
def test_hint_changes_cache_key():
    """The fastmath hint must be part of the cache key (rides _hints_)."""
    _fm_launch_hinted._ensure_sig()
    sig = _fm_launch_hinted._sig
    bound = sig.bind()
    bound.apply_defaults()

    _fm_launch_hinted.compile_hints = {}
    key_none = _fm_launch_hinted._resolve_and_make_cache_key(bound.arguments)
    _fm_launch_hinted.compile_hints = {"fastmath": "fast"}
    key_fast = _fm_launch_hinted._resolve_and_make_cache_key(bound.arguments)
    _fm_launch_hinted.compile_hints = {"fastmath": "contract"}
    key_contract = _fm_launch_hinted._resolve_and_make_cache_key(bound.arguments)

    assert key_none != key_fast
    assert key_fast != key_contract
