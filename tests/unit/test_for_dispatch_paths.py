#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MLIR-level unit tests for scf_for_dispatch (no GPU required)."""

import pytest

from flydsl._mlir.dialects import arith, func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor
from flydsl.expr.numeric import Int32


def test_scf_for_dispatch_single_iter_arg():
    """for i in range(4): acc = acc + 1  →  acc should be scf.for result."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_single_iter_arg", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = Int32(arith.ConstantOp(i32, 0).result)

                def body_fn(iv, names, acc):
                    one = Int32(arith.ConstantOp(i32, 1).result)
                    return {"acc": acc + one}

                result = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0,
                    4,
                    1,
                    body_fn,
                    result_names=("acc",),
                    result_values=(acc,),
                )
                assert isinstance(result, Int32)
                func.ReturnOp([result.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.for" in ir_text
        assert "-> (i32)" in ir_text


def test_scf_for_dispatch_multi_iter_args():
    """for i in range(3): a += 1; b -= 1  →  two iter_args."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_multi_iter_args", FunctionType.get([], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                a = Int32(arith.ConstantOp(i32, 0).result)
                b = Int32(arith.ConstantOp(i32, 100).result)

                def body_fn(iv, names, a, b):
                    one = Int32(arith.ConstantOp(i32, 1).result)
                    return {"a": a + one, "b": b - one}

                result = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0,
                    3,
                    1,
                    body_fn,
                    result_names=("a", "b"),
                    result_values=(a, b),
                )
                assert isinstance(result, tuple)
                assert len(result) == 2
                func.ReturnOp([result[0].ir_value(), result[1].ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.for" in ir_text
        assert "-> (i32, i32)" in ir_text


def test_scf_for_dispatch_no_iter_args():
    """Side-effect only loop: no iter_args, no yield values."""
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f = func.FuncOp("test_no_iter_args", FunctionType.get([], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):

                def body_fn(iv, names):
                    pass

                InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0,
                    4,
                    1,
                    body_fn,
                    result_names=(),
                    result_values=(),
                )
                func.ReturnOp([])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.for" in ir_text
        assert "-> (" not in ir_text


def test_scf_for_dispatch_none_value_raises_error():
    """result_values containing None should raise TypeError."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_none_error", FunctionType.get([], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):

                def body_fn(iv, names, x):
                    return {"x": Int32(arith.ConstantOp(i32, 1).result)}

                with pytest.raises(TypeError, match="None"):
                    InsertEmptyYieldForSCFFor.scf_for_dispatch(
                        0,
                        4,
                        1,
                        body_fn,
                        result_names=("x",),
                        result_values=(None,),
                    )


def test_scf_for_dispatch_type_mismatch_raises_error():
    """Yielded type differs from init type → TypeError."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_type_mismatch", FunctionType.get([], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                x = Int32(arith.ConstantOp(i32, 0).result)

                def body_fn(iv, names, x):
                    return {"x": arith.ConstantOp(i64, 99).result}

                with pytest.raises(TypeError, match="does not match the region entry"):
                    InsertEmptyYieldForSCFFor.scf_for_dispatch(
                        0,
                        4,
                        1,
                        body_fn,
                        result_names=("x",),
                        result_values=(x,),
                    )


def test_scf_for_dispatch_range_with_step():
    """range(0, 8, 2) → 4 iterations, verify IR structure."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("test_range_step", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = Int32(arith.ConstantOp(i32, 0).result)

                def body_fn(iv, names, acc):
                    return {"acc": acc + Int32(arith.ConstantOp(i32, 1).result)}

                result = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0,
                    8,
                    2,
                    body_fn,
                    result_names=("acc",),
                    result_values=(acc,),
                )
                func.ReturnOp([result.ir_value()])

        assert module.operation.verify()
        ir_text = str(module)
        assert "scf.for" in ir_text
        assert "c8" in ir_text or "8" in ir_text


def test_ast_rewrite_for_generates_dispatch_call():
    """AST rewrite of for loop: verify scf_for_dispatch call is injected."""
    from flydsl.compiler.ast_rewriter import ASTRewriter

    def sample(n):
        acc = 0
        for i in range(n):
            acc = acc + 1
        return acc

    ASTRewriter.transform(sample)
    assert "scf_for_dispatch" in sample.__globals__, "scf_for_dispatch not injected into globals"
    assert "scf_for_collect_results" in sample.__globals__, "scf_for_collect_results not injected"


def test_ast_rewrite_for_multi_var_generates_dispatch():
    """AST rewrite of for loop with multiple vars: dispatch is injected."""
    from flydsl.compiler.ast_rewriter import ASTRewriter

    def sample(n):
        a = 0
        b = 100
        for i in range(n):
            a = a + 1
            b = b - 1
        return a, b

    ASTRewriter.transform(sample)
    assert "scf_for_dispatch" in sample.__globals__


# ---------------------------------------------------------------------------
# unroll / unroll_full loop-annotation directives
# ---------------------------------------------------------------------------


def _for_dispatch_ir(*, unroll=-1, unroll_full=False):
    """Build a single-iter-arg scf.for via dispatch and return its module text."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("t", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = Int32(arith.ConstantOp(i32, 0).result)

                def body_fn(iv, names, acc):
                    return {"acc": acc + Int32(arith.ConstantOp(i32, 1).result)}

                r = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0,
                    8,
                    1,
                    body_fn,
                    result_names=("acc",),
                    result_values=(acc,),
                    unroll=unroll,
                    unroll_full=unroll_full,
                )
                func.ReturnOp([r.ir_value()])
        assert module.operation.verify()
        return str(module)


def test_for_dispatch_unroll_full_attr():
    ir_text = _for_dispatch_ir(unroll_full=True)
    assert "loop_annotation" in ir_text
    assert "llvm.loop_unroll<full = true>" in ir_text


def test_for_dispatch_unroll_count_attr():
    ir_text = _for_dispatch_ir(unroll=4)
    assert "llvm.loop_unroll<count = 4 : i32>" in ir_text


def test_for_dispatch_unroll_one_disables():
    ir_text = _for_dispatch_ir(unroll=1)
    assert "disable = true" in ir_text


def test_for_dispatch_no_unroll_by_default():
    assert "loop_annotation" not in _for_dispatch_ir()


def test_for_dispatch_unroll_conflict_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _for_dispatch_ir(unroll=4, unroll_full=True)


def test_for_dispatch_unroll_non_constant_rejected():
    with pytest.raises(TypeError, match="unroll"):
        _for_dispatch_ir(unroll="4")


def test_scf_range_unroll_annotation():
    """The init-carrying generator path also honors unroll."""
    from flydsl._mlir.dialects import scf

    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("t", FunctionType.get([], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                zero = Int32(arith.ConstantOp(i32, 0).result)
                for _iv, _args in InsertEmptyYieldForSCFFor.scf_range(
                    0, 8, 1, init=[zero.ir_value()], unroll_full=True
                ):
                    scf.YieldOp(list(_args))
                func.ReturnOp([])
        assert module.operation.verify()
        assert "llvm.loop_unroll<full = true>" in str(module)


def test_ast_rewrite_for_threads_unroll_kwargs():
    """unroll / unroll_full on range(...) reach the injected dispatch call and
    stay separate from the init routing."""
    import ast
    import types
    from textwrap import dedent

    def rewrite(fsrc):
        module = ast.parse(dedent(fsrc))
        ctx = types.SimpleNamespace(python_globals={}, filename="<t>")
        rw = InsertEmptyYieldForSCFFor(context=ctx, first_lineno=0)
        node = rw.visit(module.body[0])
        return ast.unparse(ast.fix_missing_locations(node))

    auto = rewrite("""
        def s(n):
            acc = 0
            for i in range(0, n, 1, unroll=4):
                acc = acc + 1
            return acc
        """)
    assert "scf_for_dispatch(" in auto
    assert "unroll=4" in auto

    init_path = rewrite("""
        def s(n):
            for i, (acc,) in range(0, n, 1, init=[0], unroll_full=True):
                acc = acc + 1
            return acc
        """)
    # init routes to scf_range; unroll rides alongside without disturbing it.
    assert "scf_range(" in init_path
    assert "init=[0]" in init_path
    assert "unroll_full=True" in init_path


# ---------------------------------------------------------------------------
# IR-level proof: the unroll hint survives lowering onto the llvm.br back-edge,
# which is exactly where MLIR->LLVM translation turns it into !llvm.loop
# metadata for the LLVM backend's loop-unroll pass. Running the same
# scf->cf->llvm conversions the ROCm pipeline uses shows the directive is not
# silently dropped -- and that each variant (count / disable / full) lands
# distinctly. No target/GPU needed.
# ---------------------------------------------------------------------------


def _for_dispatch_lowered_to_llvm(*, unroll=-1, unroll_full=False):
    from flydsl._mlir.passmanager import PassManager

    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("t", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = Int32(arith.ConstantOp(i32, 0).result)

                def body_fn(iv, names, acc):
                    return {"acc": acc + Int32(arith.ConstantOp(i32, 1).result)}

                r = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0,
                    8,
                    1,
                    body_fn,
                    result_names=("acc",),
                    result_values=(acc,),
                    unroll=unroll,
                    unroll_full=unroll_full,
                )
                func.ReturnOp([r.ir_value()])
        PassManager.parse("builtin.module(convert-scf-to-cf,convert-cf-to-llvm)").run(module.operation)
        return str(module)


def _annotates_backedge(ir_text):
    """The loop annotation must ride a loop back-edge branch, not sit as a dead attr."""
    return any("llvm.br" in line and "loop_annotation" in line for line in ir_text.splitlines())


@pytest.mark.l0_backend_agnostic
def test_unroll_count_survives_lowering_to_llvm():
    ir_text = _for_dispatch_lowered_to_llvm(unroll=4)
    assert "llvm.loop_unroll<count = 4 : i32>" in ir_text
    assert _annotates_backedge(ir_text)


@pytest.mark.l0_backend_agnostic
def test_unroll_one_disables_unrolling_after_lowering():
    ir_text = _for_dispatch_lowered_to_llvm(unroll=1)
    assert "disable = true" in ir_text
    assert _annotates_backedge(ir_text)


@pytest.mark.l0_backend_agnostic
def test_unroll_full_survives_lowering_to_llvm():
    ir_text = _for_dispatch_lowered_to_llvm(unroll_full=True)
    assert "llvm.loop_unroll<full = true>" in ir_text
    assert _annotates_backedge(ir_text)


@pytest.mark.l0_backend_agnostic
def test_no_unroll_hint_after_lowering_by_default():
    ir_text = _for_dispatch_lowered_to_llvm()
    assert "loop_annotation" not in ir_text
    assert "llvm.loop_unroll" not in ir_text
