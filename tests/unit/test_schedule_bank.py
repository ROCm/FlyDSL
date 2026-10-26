#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for the rocdl.schedule_bank DSL wrapper.

Validates IR emission for strict and soft VGPR bank scheduling hints.
Requires the custom LLVM build with the rocdl.schedule.bank op; tests
skip when the ODS binding is unavailable.
"""

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import func, rocdl

_has_schedule_bank = hasattr(rocdl, "schedule_bank") or hasattr(rocdl, "ScheduleBank")

pytestmark = [
    pytest.mark.l0_backend_agnostic,
    pytest.mark.skipif(not _has_schedule_bank, reason="rocdl.schedule.bank op not available (needs custom LLVM)"),
]


def _build_module(build_fn, arg_types=None):
    """Build an MLIR module containing a function that calls *build_fn* and return the IR text."""
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            if arg_types is None:
                types = [ir.F32Type.get()]
            else:
                types = [t() if callable(t) else t for t in arg_types]
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                ftype = ir.FunctionType.get(types, types)
                f = func.FuncOp("test", ftype)
                with ir.InsertionPoint(f.add_entry_block()):
                    args = list(f.entry_block.arguments)
                    results = build_fn(*args)
                    if not isinstance(results, (list, tuple)):
                        results = [results]
                    func.ReturnOp(results)
            module.operation.verify()
            return str(module)


def test_schedule_bank_strict():
    """Default (strict) schedule_bank emits the op without the soft keyword."""
    from flydsl.expr.rocdl import schedule_bank

    def build(x):
        return schedule_bank(x, bank=0)

    ir_text = _build_module(build)
    assert "rocdl.schedule.bank" in ir_text
    assert "soft" not in ir_text


def test_schedule_bank_soft():
    """soft=True emits the soft keyword in the printed op."""
    from flydsl.expr.rocdl import schedule_bank

    def build(x):
        return schedule_bank(x, bank=2, soft=True)

    ir_text = _build_module(build)
    assert "rocdl.schedule.bank" in ir_text
    assert "soft" in ir_text


def test_schedule_bank_identity_type():
    """Result type must match the input value type (identity semantics)."""
    from flydsl.expr.rocdl import schedule_bank

    def build(x):
        return schedule_bank(x, bank=1)

    ir_text = _build_module(build)
    assert "-> f32" in ir_text


def test_schedule_bank_all_banks():
    """Banks 0-3 are all accepted."""
    from flydsl.expr.rocdl import schedule_bank

    for bank_id in range(4):

        def build(x, b=bank_id):
            return schedule_bank(x, bank=b)

        ir_text = _build_module(build)
        # IR prints as: rocdl.schedule.bank %arg0, <bank_id> : f32
        assert f"rocdl.schedule.bank %arg0, {bank_id}" in ir_text


def test_schedule_bank_integer_input():
    """schedule_bank on an integer-typed value preserves the type."""
    from flydsl.expr.rocdl import schedule_bank

    def build(x):
        return schedule_bank(x, bank=3)

    ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
    assert "rocdl.schedule.bank" in ir_text
    assert "i32" in ir_text
