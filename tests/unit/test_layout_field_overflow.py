#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regression tests for ROCm/FlyDSL#1176: launching with a dynamic layout
field that overflows its signed packed-ABI width must raise an actionable
error naming the argument, the field kind (shape/stride), the dim index and
the offending value -- not a raw ``struct.error: 'i' format requires ...``.

Per the maintainer analysis in the issue, the failing field is an individual
dynamic *dimension* (e.g. a shape dim after ``view(-1)``), not the tensor's
total element count.  ``torch.empty(..., device='meta')`` supplies oversized
metadata without allocating memory, so the L0 tests need no GPU.

L0 tests exercise the generated fill directly -- the same function the
dispatch closure runs on every launch.
"""

import struct

import pytest
import torch

import flydsl.compiler as flyc
from flydsl.compiler.jit_argument import TorchTensorJitArg

# ---------------------------------------------------------------------------
# L0: fill-level diagnostics, no GPU / no compile needed.
# ---------------------------------------------------------------------------


def _make_fill(tensor, **kwargs):
    """Build TorchTensorJitArg slots for ``tensor``; return (fill, plan)."""
    arg = TorchTensorJitArg(tensor, **kwargs)
    slots = arg.__c_abi_spec__()
    plan = arg._layout_plan
    assert plan is not None, "layout-dynamic tensor must build a _LayoutPlan"
    fill = next(
        f for _ctype, f in slots if getattr(f, "__name__", "") == "fill"
    )
    return fill, plan


def _storage(plan):
    return plan.buf_ctype.from_buffer(bytearray(plan.codec.size))


def test_shape_overflow_message_names_arg_and_dim():
    t = torch.empty(2_147_483_648, dtype=torch.float32, device="meta")
    fill, plan = _make_fill(t)
    plan.param_name = "OUT"
    with pytest.raises(ValueError) as ei:
        fill(t, _storage(plan))
    msg = str(ei.value)
    assert "argument 'OUT'" in msg
    assert "shape[0]" in msg
    assert "2147483648" in msg
    assert "int32" in msg


def test_stride_overflow_when_32bit_strides():
    # 2D meta tensor whose dim-1 stride (3_000_000_000) exceeds INT32_MAX,
    # packed as 'i' under use_32bit_stride=True.
    t = torch.empty(4, 8, dtype=torch.float32, device="meta").as_strided(
        (4, 8), (3_000_000_001, 1)
    )
    fill, plan = _make_fill(t, use_32bit_stride=True)
    plan.param_name = "A"
    with pytest.raises(ValueError) as ei:
        fill(t, _storage(plan))
    msg = str(ei.value)
    assert "argument 'A'" in msg
    assert "stride[0]" in msg
    assert "3000000001" in msg
    assert "int32" in msg


def test_fitting_fields_still_pack_cleanly():
    t = torch.empty(4, 8, dtype=torch.float32)
    fill, plan = _make_fill(t)
    plan.param_name = "B"
    storage = _storage(plan)
    fill(t, storage)  # must not raise
    # 2 dynamic shapes (i32) + 1 dynamic stride (i64)
    sh0, sh1, st1 = struct.unpack("<iiq", bytes(storage))
    assert (sh0, sh1, st1) == (4, 8, 8)


def test_signed_min_values_are_accepted():
    # The full signed range must be accepted (Copilot review): -2**31 fits an
    # 'i' field exactly; only values below it may be rejected.
    t = torch.empty(2_147_483_648, dtype=torch.float32, device="meta")
    fill, plan = _make_fill(t)
    plan.param_name = "MIN"
    # shape 2**31 overflows; but the check itself must use the exact signed
    # bound. Use a stride field to probe the i32 minimum: build a 2-D tensor
    # whose dynamic stride is -2**31 (negative strides are accepted by the
    # packer; only the range check guards the ABI).
    t2 = torch.empty(4, 8, dtype=torch.float32, device="meta").as_strided((4, 8), (-2**31, 1))
    fill2, plan2 = _make_fill(t2, use_32bit_stride=True)
    plan2.param_name = "NEG"
    with pytest.raises(ValueError) as ei:
        fill2(t2, _storage(plan2))
    msg = str(ei.value)
    # -2**31 is IN range for 'i', so this must NOT be reported as overflow.
    assert "-2147483648" not in msg, f"signed minimum wrongly rejected: {msg}"


def test_nested_struct_path_in_diagnostic():
    # A struct-typed JIT parameter must produce a full field path in the
    # overflow diagnostic (Copilot review): 'outer.inner', not 'Struct.field'.
    import flydsl.expr as fx
    from flydsl.compiler.jit_argument import TorchTensorJitArg

    @fx.struct
    class Inner:
        t: fx.Tensor

    @fx.struct
    class Outer:
        inner: Inner

    big = torch.empty(2_147_483_648, dtype=torch.float32, device="meta")
    outer = Outer(inner=Inner(t=TorchTensorJitArg(big)))
    slots = outer.__c_abi_spec__()
    # find the fill built for the nested tensor plan
    from flydsl.compiler.jit_function import _stamp_plan_param_names
    _stamp_plan_param_names(outer, "payload")
    # locate the plan through the struct walk and check the composed label
    from flydsl.expr.struct import _effective_field_defs, _is_constexpr_type

    def find_plans(obj, prefix):
        plans = []
        plan = getattr(obj, "_layout_plan", None)
        if plan is not None:
            plans.append((prefix, plan))
        if hasattr(obj, "__dsl_composite_kind__"):
            for name, eff in _effective_field_defs(type(obj)):
                if _is_constexpr_type(eff):
                    continue
                plans += find_plans(getattr(obj, name, None), f"{prefix}.{name}" if prefix else name)
        return plans

    plans = find_plans(outer, "")
    assert plans, "no layout plan discovered in struct"
    label = None
    for _path, plan in plans:
        if plan.path_suffix:
            label = f"{plan.param_name}.{plan.path_suffix}"
    assert label == "payload.inner.t", f"unexpected diagnostic label: {label!r}"


def test_repeated_launch_still_packs():
    # The check must not break the fast dispatch path: same fill, run twice.
    t = torch.empty(64, dtype=torch.float32)
    fill, plan = _make_fill(t)
    plan.param_name = "C"
    storage = _storage(plan)
    fill(t, storage)
    fill(t, storage)
    (v,) = struct.unpack("<i", bytes(storage))
    assert v == 64


# ---------------------------------------------------------------------------
# L2: the original issue's launch repro (needs GPU + full stack).
# ---------------------------------------------------------------------------


@pytest.mark.l2_device
def test_launch_big_view_neg1_diagnostic():
    import flydsl.expr as fx

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kern(OUT: fx.Tensor):
        o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(OUT, max_size=False), fx.make_layout(1, 1)
        )
        reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        fx.memref_store_vec(fx.Vector.filled(1, 1.0, fx.Float32), reg)
        lane = fx.thread_idx.x % 64
        fx.copy(atom, reg, fx.slice(o_div, (None, fx.Int32(lane))))

    @flyc.jit
    def launch(OUT: fx.Tensor, stream: fx.Stream):
        kern(OUT).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    # 2_500_000_000 elements: the single dynamic shape dim overflows i32.
    out = torch.zeros(2_500_000_000, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=r"shape\[0\].*int32"):
        launch(out, torch.cuda.current_stream(out.device))
