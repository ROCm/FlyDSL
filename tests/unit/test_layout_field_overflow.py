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
    # The full signed range must be accepted (Copilot review): -2**31 packs
    # fine as 'i' and must not be reported; only values beyond the signed
    # minimum may be. Probe overflow_report directly with synthetic fields.
    from flydsl.compiler.jit_argument import _LayoutPlan

    plan32 = _LayoutPlan((), (0,), use_32bit_stride=True)
    plan32.param_name = "STR"
    assert plan32.overflow_report(None, (-2**31,)) == [], "signed i32 minimum must be accepted"
    bad = plan32.overflow_report(None, (-2**31 - 1,))
    assert bad and bad[0][2] == -(2**31) - 1, bad

    plan64 = _LayoutPlan((), (0,), use_32bit_stride=False)
    assert plan64.overflow_report(None, (-2**63,)) == [], "signed i64 minimum must be accepted"
    bad64 = plan64.overflow_report(None, (-2**63 - 1,))
    assert bad64 and bad64[0][2] == -(2**63) - 1, bad64


def test_nested_struct_path_in_diagnostic():
    # A struct-typed JIT parameter must compose a full field path in the
    # diagnostic (Copilot review): "payload.inner.t", not an ambiguous
    # "Struct.field" shared by every field of the same struct type.
    #
    # Struct fields coerce to DSL wrappers at construction (host tensors are
    # rejected — see test_raw_framework_tensor_is_not_an_fx_tensor_field), so
    # the JIT arg is swapped in post-construction; the ABI recursion only
    # reads field *values*, not annotations. object.__setattr__ is the same
    # escape hatch the specializer itself uses for frozen fields.
    from flydsl.compiler.jit_argument import _check_layout_fields
    from flydsl.compiler.jit_function import _stamp_plan_param_names

    @fx.struct
    class InnerS:
        t: fx.Int32

    @fx.struct
    class OuterS:
        inner: InnerS

    big = torch.empty(2_147_483_648, dtype=torch.float32, device="meta")
    arg = TorchTensorJitArg(big)
    arg.__c_abi_spec__()  # build the plan so the struct recursion can stamp it

    holder = OuterS(inner=InnerS(t=1))
    object.__setattr__(holder.inner, "t", arg)

    holder.__c_abi_spec__()
    _stamp_plan_param_names(holder, "payload")

    plan = getattr(arg, "_layout_plan", None)
    assert plan is not None, "nested tensor plan was not built/stamped"
    label = plan.param_name + (f".{plan.path_suffix}" if plan.path_suffix else "")
    assert label == "payload.inner.t", f"unexpected diagnostic label: {label!r}"

    with pytest.raises(ValueError, match=r"argument 'payload\.inner\.t'"):
        _check_layout_fields(plan, (2_147_483_648,), None)


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
