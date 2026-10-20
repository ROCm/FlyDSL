#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for rocdl.schedule_{vgpr,sgpr,agpr} DSL wrappers.

Validates IR emission for exact-register scheduling hints (strict and soft).
Requires the custom LLVM build with the rocdl.schedule.{vgpr,sgpr,agpr} ops;
tests skip when the ODS bindings are unavailable.
"""

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import func, rocdl

_has_schedule_vgpr = hasattr(rocdl, "schedule_vgpr") or hasattr(rocdl, "ScheduleVGPR")
_has_schedule_sgpr = hasattr(rocdl, "schedule_sgpr") or hasattr(rocdl, "ScheduleSGPR")
_has_schedule_agpr = hasattr(rocdl, "schedule_agpr") or hasattr(rocdl, "ScheduleAGPR")

pytestmark = [
    pytest.mark.l0_backend_agnostic,
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


# ── schedule_vgpr ────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_schedule_vgpr, reason="rocdl.schedule.vgpr op not available (needs custom LLVM)")
class TestScheduleVGPR:
    def test_strict(self):
        """Default (strict) schedule_vgpr emits the op without the soft keyword."""
        from flydsl.expr.rocdl import schedule_vgpr

        def build(x):
            return schedule_vgpr(x, regnum=42)

        ir_text = _build_module(build)
        assert "rocdl.schedule.vgpr" in ir_text
        assert "soft" not in ir_text

    def test_soft(self):
        """soft=True emits the soft keyword in the printed op."""
        from flydsl.expr.rocdl import schedule_vgpr

        def build(x):
            return schedule_vgpr(x, regnum=42, soft=True)

        ir_text = _build_module(build)
        assert "rocdl.schedule.vgpr" in ir_text
        assert "soft" in ir_text

    def test_identity_type(self):
        """Result type must match the input value type (identity semantics)."""
        from flydsl.expr.rocdl import schedule_vgpr

        def build(x):
            return schedule_vgpr(x, regnum=100)

        ir_text = _build_module(build)
        assert "-> f32" in ir_text

    def test_regnum_values(self):
        """Various register numbers are accepted."""
        from flydsl.expr.rocdl import schedule_vgpr

        for regnum in [0, 42, 100, 512]:

            def build(x, r=regnum):
                return schedule_vgpr(x, regnum=r)

            ir_text = _build_module(build)
            assert f"rocdl.schedule.vgpr %arg0, {regnum}" in ir_text

    def test_integer_input(self):
        """schedule_vgpr on an integer-typed value preserves the type."""
        from flydsl.expr.rocdl import schedule_vgpr

        def build(x):
            return schedule_vgpr(x, regnum=0)

        ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
        assert "rocdl.schedule.vgpr" in ir_text
        assert "i32" in ir_text


# ── schedule_sgpr ────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_schedule_sgpr, reason="rocdl.schedule.sgpr op not available (needs custom LLVM)")
class TestScheduleSGPR:
    def test_strict(self):
        from flydsl.expr.rocdl import schedule_sgpr

        def build(x):
            return schedule_sgpr(x, regnum=10)

        ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
        assert "rocdl.schedule.sgpr" in ir_text
        assert "soft" not in ir_text

    def test_soft(self):
        from flydsl.expr.rocdl import schedule_sgpr

        def build(x):
            return schedule_sgpr(x, regnum=10, soft=True)

        ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
        assert "rocdl.schedule.sgpr" in ir_text
        assert "soft" in ir_text

    def test_identity_type(self):
        from flydsl.expr.rocdl import schedule_sgpr

        def build(x):
            return schedule_sgpr(x, regnum=4)

        ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
        assert "-> i32" in ir_text

    def test_regnum_values(self):
        from flydsl.expr.rocdl import schedule_sgpr

        for regnum in [0, 10, 50, 105]:

            def build(x, r=regnum):
                return schedule_sgpr(x, regnum=r)

            ir_text = _build_module(build, arg_types=[lambda: ir.IntegerType.get_signless(32)])
            assert f"rocdl.schedule.sgpr %arg0, {regnum}" in ir_text


# ── schedule_agpr ────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_schedule_agpr, reason="rocdl.schedule.agpr op not available (needs custom LLVM)")
class TestScheduleAGPR:
    def test_strict(self):
        from flydsl.expr.rocdl import schedule_agpr

        def build(x):
            return schedule_agpr(x, regnum=8)

        ir_text = _build_module(build)
        assert "rocdl.schedule.agpr" in ir_text
        assert "soft" not in ir_text

    def test_soft(self):
        from flydsl.expr.rocdl import schedule_agpr

        def build(x):
            return schedule_agpr(x, regnum=8, soft=True)

        ir_text = _build_module(build)
        assert "rocdl.schedule.agpr" in ir_text
        assert "soft" in ir_text

    def test_identity_type(self):
        from flydsl.expr.rocdl import schedule_agpr

        def build(x):
            return schedule_agpr(x, regnum=0)

        ir_text = _build_module(build)
        assert "-> f32" in ir_text

    def test_regnum_values(self):
        from flydsl.expr.rocdl import schedule_agpr

        for regnum in [0, 8, 128, 255]:

            def build(x, r=regnum):
                return schedule_agpr(x, regnum=r)

            ir_text = _build_module(build)
            assert f"rocdl.schedule.agpr %arg0, {regnum}" in ir_text
