#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for the rocdl.sched_bank DSL wrapper.

Validates IR emission for strict and soft VGPR bank scheduling hints.
Requires the custom LLVM build with the rocdl.sched.bank op; tests
skip when the ODS binding is unavailable.
"""

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import func, rocdl

_has_sched_bank = hasattr(rocdl, "sched_bank") or hasattr(rocdl, "SchedBank")

pytestmark = [
    pytest.mark.l0_backend_agnostic,
    pytest.mark.skipif(not _has_sched_bank, reason="rocdl.sched.bank op not available (needs custom LLVM)"),
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


def test_sched_bank_strict():
    """Default (strict) sched_bank emits the op without the soft keyword."""
    from flydsl.expr.rocdl import sched_bank

    def build(x):
        return sched_bank(x, bank=0)

    ir_text = _build_module(build)
    assert "rocdl.sched.bank" in ir_text
    assert "soft" not in ir_text


def test_sched_bank_soft():
    """soft=True emits the soft keyword in the printed op."""
    from flydsl.expr.rocdl import sched_bank

    def build(x):
        return sched_bank(x, bank=2, soft=True)

    ir_text = _build_module(build)
    assert "rocdl.sched.bank" in ir_text
    assert "soft" in ir_text


def test_sched_bank_identity_type():
    """Result type must match the input value type (identity semantics)."""
    from flydsl.expr.rocdl import sched_bank

    def build(x):
        return sched_bank(x, bank=1)

    ir_text = _build_module(build)
    assert "-> f32" in ir_text


def test_sched_bank_all_banks():
    """Banks 0-3 are all accepted."""
    from flydsl.expr.rocdl import sched_bank

    for bank_id in range(4):

        def build(x, b=bank_id):
            return sched_bank(x, bank=b)

        ir_text = _build_module(build)
        # IR prints as: rocdl.sched.bank %arg0, <bank_id> : f32
        assert f"rocdl.sched.bank %arg0, {bank_id}" in ir_text


def test_sched_bank_integer_input():
    """sched_bank on an integer-typed value preserves the type."""
    from flydsl.expr.rocdl import sched_bank

    def build(x):
        return sched_bank(x, bank=3)

    ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
    assert "rocdl.sched.bank" in ir_text
    assert "i32" in ir_text
