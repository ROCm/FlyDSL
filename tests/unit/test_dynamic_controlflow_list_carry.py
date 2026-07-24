#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MLIR-level unit tests (no GPU) for carrying Python containers (list / tuple /
dict / nested) through dynamic scf.if / scf.for / scf.while.

Before this feature a reassigned container local tripped the dispatchers with
``TypeError: ... is list, not an MLIR Value``; the region rewriter could only
carry a single ir.Value per name. These tests lock in the container round-trip
(explode to per-element iter_args/results, assemble on exit) and the guard rails
when a branch/body changes a container's shape or an element's dtype.
"""

from types import SimpleNamespace

import pytest

from flydsl._mlir.dialects import arith, func
from flydsl._mlir.ir import Context, F32Type, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl.compiler.ast_rewriter import (
    CanonicalizeWhile,
    InsertEmptyYieldForSCFFor,
    ReplaceIfWithDispatch,
)
from flydsl.expr.numeric import Float32, Int32, Int64


def _i32(v):
    return Int32(arith.ConstantOp(IntegerType.get_signless(32), v).result)


def _i64(v):
    return Int64(arith.ConstantOp(IntegerType.get_signless(64), v).result)


def _f32(v):
    return Float32(arith.ConstantOp(F32Type.get(), float(v)).result)


# ─────────────────────────────── if ────────────────────────────────────────


def test_if_carries_list():
    """A reassigned list local carried across a dynamic scf.if. Result must be a
    Python list of the same length, and the scf.if must carry one result per list
    element."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("if_list", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                lst = [_i32(0), _i32(1)]

                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    cond,
                    lambda names, lst: {"lst": [_i32(10), _i32(11)]},
                    lambda names, lst: {"lst": [_i32(20), _i32(21)]},
                    result_names=("lst",),
                    result_values=(lst,),
                )
                assert isinstance(out, list) and len(out) == 2
                assert all(isinstance(x, Int32) for x in out)
                func.ReturnOp([])

        text = str(module)
        assert module.operation.verify()
        assert "scf.if" in text
        # one scf.if carrying two i32 results (one per list element)
        assert "-> (i32, i32)" in text


def test_if_carries_nested_list():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("if_nested", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                nested = [[_i32(0), _i32(1)], [_i32(2)]]  # 3 elements

                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    cond,
                    lambda names, nested: {"nested": [[_i32(10), _i32(11)], [_i32(12)]]},
                    lambda names, nested: {"nested": [[_i32(20), _i32(21)], [_i32(22)]]},
                    result_names=("nested",),
                    result_values=(nested,),
                )
                assert isinstance(out, list) and len(out) == 2
                assert len(out[0]) == 2 and len(out[1]) == 1
                func.ReturnOp([])

        assert module.operation.verify()
        assert "-> (i32, i32, i32)" in str(module)


def test_if_carries_tuple_and_dict():
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("if_tuple_dict", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                tup = (_i32(0), _i32(1))
                dct = {"x": _i32(2), "y": _i32(3)}

                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    cond,
                    lambda names, tup, dct: {"tup": (_i32(10), _i32(11)), "dct": {"x": _i32(12), "y": _i32(13)}},
                    lambda names, tup, dct: {"tup": (_i32(20), _i32(21)), "dct": {"x": _i32(22), "y": _i32(23)}},
                    result_names=("tup", "dct"),
                    result_values=(tup, dct),
                )
                rtup, rdct = out
                assert isinstance(rtup, tuple) and len(rtup) == 2
                assert isinstance(rdct, dict) and set(rdct) == {"x", "y"}
                func.ReturnOp([])

        assert module.operation.verify()


def test_if_mixes_list_and_scalar():
    """A list carry and a plain scalar carry in the same if must both work
    (scalar is the degenerate single-element case)."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("if_mixed", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                lst = [_i32(0), _i32(1)]
                s = _i32(7)

                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    cond,
                    lambda names, lst, s: {"lst": [_i32(10), _i32(11)], "s": _i32(70)},
                    lambda names, lst, s: {"lst": [_i32(20), _i32(21)], "s": _i32(80)},
                    result_names=("lst", "s"),
                    result_values=(lst, s),
                )
                rlst, rs = out
                assert isinstance(rlst, list) and len(rlst) == 2
                assert isinstance(rs, Int32)
                func.ReturnOp([])

        assert module.operation.verify()
        assert "-> (i32, i32, i32)" in str(module)


def test_if_static_cond_with_list_no_ifop():
    """A const_expr (python-bool) condition must resolve at trace time: pick a
    branch, build no scf.if, still return a list."""
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f = func.FuncOp("if_static_list", FunctionType.get([], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                lst = [_i32(0), _i32(1)]
                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    True,
                    lambda names, lst: {"lst": [_i32(10), _i32(11)]},
                    lambda names, lst: {"lst": [_i32(20), _i32(21)]},
                    result_names=("lst",),
                    result_values=(lst,),
                )
                assert isinstance(out, list) and len(out) == 2
                func.ReturnOp([])

        assert module.operation.verify()
        assert "scf.if" not in str(module)


def test_if_list_length_change_errors():
    """A branch that changes the carried list's length must fail with a clear
    error (scf.if result arity would otherwise mismatch)."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("if_badlen", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                lst = [_i32(0), _i32(1)]
                with pytest.raises(TypeError, match="does not match the region entry"):
                    ReplaceIfWithDispatch.scf_if_dispatch(
                        cond,
                        lambda names, lst: {"lst": [_i32(10)]},  # length 1 != 2
                        lambda names, lst: {"lst": [_i32(20), _i32(21)]},
                        result_names=("lst",),
                        result_values=(lst,),
                    )


def test_if_list_dtype_change_errors():
    """A branch that changes an element's dtype must fail with a clear error."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("if_baddtype", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                lst = [_i32(0)]
                with pytest.raises(TypeError, match="does not match the region entry"):
                    ReplaceIfWithDispatch.scf_if_dispatch(
                        cond,
                        lambda names, lst: {"lst": [_i64(10)]},  # i64 != i32
                        lambda names, lst: {"lst": [_i32(20)]},
                        result_names=("lst",),
                        result_values=(lst,),
                    )


def test_if_dict_key_reorder_ok():
    """A branch may list dict keys in a different order (a dict's key order is not
    semantic); the values are aligned by key to the entry, not silently swapped."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("if_reorder", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                d = {"a": _i32(1), "b": _i32(2)}
                out = ReplaceIfWithDispatch.scf_if_dispatch(
                    cond,
                    lambda names, d: {"d": {"a": _i32(10), "b": _i32(20)}},
                    lambda names, d: {"d": {"b": _i32(200), "a": _i32(100)}},  # keys reordered
                    result_names=("d",),
                    result_values=(d,),
                )
                # rebuilt in the entry's key order regardless of the branch's order
                assert isinstance(out, dict) and list(out) == ["a", "b"]
                func.ReturnOp([])
        assert module.operation.verify()


def test_if_dict_key_set_mismatch_errors():
    """A branch with a different key set (not just reordered) must still error."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("if_badkeys", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                d = {"a": _i32(1), "b": _i32(2)}
                with pytest.raises(TypeError, match="does not match the region entry"):
                    ReplaceIfWithDispatch.scf_if_dispatch(
                        cond,
                        lambda names, d: {"d": {"a": _i32(10), "b": _i32(20)}},
                        lambda names, d: {"d": {"a": _i32(100), "c": _i32(200)}},  # 'c' not in entry
                        result_names=("d",),
                        result_values=(d,),
                    )


def test_for_carries_bare_scalar_literal():
    """A bare python int initializer (``acc = 0``) must carry through the loop and
    come back as a DSL numeric, not a raw ir.Value (guards the acc = 0 pattern)."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("for_bare_scalar", FunctionType.get([], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = 0  # bare python int, not fx.Int32

                def body_fn(iv, names, acc):
                    assert isinstance(acc, Int32)  # packed back as a DSL numeric
                    return {"acc": acc + _i32(1)}

                out = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0, 5, 1, body_fn, result_names=("acc",), result_values=(acc,)
                )
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])
        assert module.operation.verify()
        assert "scf.for" in str(module)


def test_for_carries_bare_float_literal():
    """Same as above for a bare python float initializer (``acc = 0.0``)."""
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f = func.FuncOp("for_bare_float", FunctionType.get([], [F32Type.get()]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                acc = 0.0  # bare python float

                def body_fn(iv, names, acc):
                    assert isinstance(acc, Float32)
                    return {"acc": acc + _f32(1.0)}

                out = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0, 3, 1, body_fn, result_names=("acc",), result_values=(acc,)
                )
                assert isinstance(out, Float32)
                func.ReturnOp([out.ir_value()])
        assert module.operation.verify()
        assert "scf.for" in str(module)


# ── container variety: dict / tuple / SimpleNamespace + nesting/mixed ────────


def _carry_through_if(make_exemplar, make_then, make_else, fname="if_carry"):
    """Carry ``make_exemplar()``'s value through a dynamic scf.if whose branches
    reassign it to ``make_then()`` / ``make_else()``; return ``(out, ir_text)``.

    Must be called inside ``with Context(), Location.unknown()``. Reused by the
    container-variety tests below to avoid repeating the MLIR scaffolding.
    """
    module = Module.create()
    i1 = IntegerType.get_signless(1)
    with InsertionPoint(module.body):
        f = func.FuncOp(fname, FunctionType.get([i1], []))
        entry = f.add_entry_block()
        with InsertionPoint(entry):
            cond = entry.arguments[0]
            out = ReplaceIfWithDispatch.scf_if_dispatch(
                cond,
                lambda names, v: {"v": make_then()},
                lambda names, v: {"v": make_else()},
                result_names=("v",),
                result_values=(make_exemplar(),),
            )
            func.ReturnOp([])
    assert module.operation.verify()
    return out, str(module)


def test_if_carries_pure_dict():
    with Context(), Location.unknown():
        out, ir = _carry_through_if(
            lambda: {"a": _i32(1), "b": _i32(2)},
            lambda: {"a": _i32(10), "b": _i32(20)},
            lambda: {"a": _i32(30), "b": _i32(40)},
            fname="if_dict",
        )
        assert isinstance(out, dict) and list(out) == ["a", "b"]
        assert all(isinstance(v, Int32) for v in out.values())
        assert "-> (i32, i32)" in ir


def test_if_carries_pure_tuple():
    with Context(), Location.unknown():
        out, ir = _carry_through_if(
            lambda: (_i32(1), _i32(2), _i32(3)),
            lambda: (_i32(10), _i32(20), _i32(30)),
            lambda: (_i32(40), _i32(50), _i32(60)),
            fname="if_tuple",
        )
        assert isinstance(out, tuple) and len(out) == 3
        assert all(isinstance(v, Int32) for v in out)
        assert "-> (i32, i32, i32)" in ir


def test_if_carries_simplenamespace():
    with Context(), Location.unknown():
        out, ir = _carry_through_if(
            lambda: SimpleNamespace(x=_i32(1), y=_i32(2)),
            lambda: SimpleNamespace(x=_i32(10), y=_i32(20)),
            lambda: SimpleNamespace(x=_i32(30), y=_i32(40)),
            fname="if_ns",
        )
        assert isinstance(out, SimpleNamespace)
        assert isinstance(out.x, Int32) and isinstance(out.y, Int32)
        assert "-> (i32, i32)" in ir


def test_if_carries_tuple_of_dict():
    """The requested mixed case: a tuple containing a dict."""
    with Context(), Location.unknown():
        out, ir = _carry_through_if(
            lambda: (_i32(1), {"k": _i32(2), "m": _i32(3)}),
            lambda: (_i32(10), {"k": _i32(20), "m": _i32(30)}),
            lambda: (_i32(40), {"k": _i32(50), "m": _i32(60)}),
            fname="if_tuple_of_dict",
        )
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], Int32)
        assert isinstance(out[1], dict) and list(out[1]) == ["k", "m"]
        assert all(isinstance(v, Int32) for v in out[1].values())
        assert "-> (i32, i32, i32)" in ir  # 1 + 2 leaves


def test_if_carries_dict_of_list():
    """A dict whose value is a list (and a scalar sibling)."""
    with Context(), Location.unknown():
        out, ir = _carry_through_if(
            lambda: {"xs": [_i32(1), _i32(2)], "n": _i32(3)},
            lambda: {"xs": [_i32(10), _i32(20)], "n": _i32(30)},
            lambda: {"xs": [_i32(40), _i32(50)], "n": _i32(60)},
            fname="if_dict_of_list",
        )
        assert isinstance(out, dict) and list(out) == ["xs", "n"]
        assert isinstance(out["xs"], list) and len(out["xs"]) == 2
        assert isinstance(out["n"], Int32)
        assert "-> (i32, i32, i32)" in ir


def test_if_carries_ns_of_mixed():
    """SimpleNamespace holding a list, a tuple and a scalar."""
    with Context(), Location.unknown():
        out, ir = _carry_through_if(
            lambda: SimpleNamespace(vec=[_i32(1), _i32(2)], tup=(_i32(3), _i32(4)), scal=_i32(5)),
            lambda: SimpleNamespace(vec=[_i32(10), _i32(20)], tup=(_i32(30), _i32(40)), scal=_i32(50)),
            lambda: SimpleNamespace(vec=[_i32(60), _i32(70)], tup=(_i32(80), _i32(90)), scal=_i32(99)),
            fname="if_ns_mixed",
        )
        assert isinstance(out, SimpleNamespace)
        assert isinstance(out.vec, list) and len(out.vec) == 2
        assert isinstance(out.tup, tuple) and len(out.tup) == 2
        assert isinstance(out.scal, Int32)
        assert "-> (i32, i32, i32, i32, i32)" in ir  # 2 + 2 + 1 leaves


def test_if_carries_deeply_nested_mixed():
    """list -> [ dict{a: tuple(i32,i32)}, list[i32] ]: containers nested 3 deep."""
    with Context(), Location.unknown():
        out, ir = _carry_through_if(
            lambda: [{"a": (_i32(1), _i32(2))}, [_i32(3)]],
            lambda: [{"a": (_i32(10), _i32(20))}, [_i32(30)]],
            lambda: [{"a": (_i32(40), _i32(50))}, [_i32(60)]],
            fname="if_deep",
        )
        assert isinstance(out, list) and len(out) == 2
        assert isinstance(out[0], dict) and isinstance(out[0]["a"], tuple)
        assert isinstance(out[1], list) and len(out[1]) == 1
        assert "-> (i32, i32, i32)" in ir  # 2 + 1 leaves


# ─────────────────────────────── ifexp ─────────────────────────────────────


def test_ifexp_carries_list():
    """A ternary whose value is a list: `[a, b] if cond else [c, d]`."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("ifexp_list", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                out = ReplaceIfWithDispatch.scf_ifexp_dispatch(
                    cond, lambda: [_i32(1), _i32(2)], lambda: [_i32(3), _i32(4)]
                )
                assert isinstance(out, list) and len(out) == 2
                assert all(isinstance(x, Int32) for x in out)
                func.ReturnOp([])

        text = str(module)
        assert module.operation.verify()
        assert "scf.if" in text
        assert "-> (i32, i32)" in text


def test_ifexp_scalar_still_scalar():
    """Scalar ternary keeps returning a scalar (backward compatible)."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("ifexp_scalar", FunctionType.get([i1], [i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                out = ReplaceIfWithDispatch.scf_ifexp_dispatch(cond, lambda: _i32(1), lambda: _i32(2))
                assert isinstance(out, Int32)
                func.ReturnOp([out.ir_value()])

        assert module.operation.verify()
        assert "-> (i32)" in str(module)


def test_ifexp_shape_mismatch_errors():
    """then/else producing different container shapes must fail clearly."""
    with Context(), Location.unknown():
        module = Module.create()
        i1 = IntegerType.get_signless(1)
        with InsertionPoint(module.body):
            f = func.FuncOp("ifexp_bad", FunctionType.get([i1], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                cond = entry.arguments[0]
                with pytest.raises(TypeError, match="does not match the region entry"):
                    ReplaceIfWithDispatch.scf_ifexp_dispatch(
                        cond, lambda: [_i32(1), _i32(2)], lambda: [_i32(3)]  # len 1 != 2
                    )


# ─────────────────────────────── for ───────────────────────────────────────


def test_for_carries_list():
    """for i in range(4): lst = [lst[0]+1, lst[1]+2]  →  list survives as iter_args."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("for_list", FunctionType.get([], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                lst = [_i32(0), _i32(1)]

                def body_fn(iv, names, lst):
                    # body receives the reassembled list
                    assert isinstance(lst, list) and len(lst) == 2
                    return {"lst": [lst[0] + _i32(1), lst[1] + _i32(2)]}

                out = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0, 4, 1, body_fn, result_names=("lst",), result_values=(lst,)
                )
                assert isinstance(out, list) and len(out) == 2
                func.ReturnOp([out[0].ir_value(), out[1].ir_value()])

        text = str(module)
        assert module.operation.verify()
        assert "scf.for" in text
        assert "-> (i32, i32)" in text


def test_for_carries_dict():
    """for i in range(3): carry a dict whose value is a nested list."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("for_dict", FunctionType.get([], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                d = {"acc": _i32(0), "vec": [_i32(5)]}

                def body_fn(iv, names, d):
                    assert isinstance(d, dict) and list(d) == ["acc", "vec"]
                    assert isinstance(d["vec"], list)
                    return {"d": {"acc": d["acc"] + _i32(1), "vec": [d["vec"][0] + _i32(1)]}}

                out = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                    0, 3, 1, body_fn, result_names=("d",), result_values=(d,)
                )
                assert isinstance(out, dict) and isinstance(out["vec"], list)
                func.ReturnOp([out["acc"].ir_value(), out["vec"][0].ir_value()])

        text = str(module)
        assert module.operation.verify()
        assert "scf.for" in text
        assert "-> (i32, i32)" in text


# ─────────────────────────────── while ─────────────────────────────────────


def test_while_carries_list():
    """while lst[0] > 0: lst = [lst[0]-1, lst[1]+1]  →  list survives before/after."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("while_list", FunctionType.get([], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                lst = [_i32(4), _i32(0)]

                def before_fn(names, lst):
                    assert isinstance(lst, list) and len(lst) == 2
                    return lst[0] > _i32(0)

                def after_fn(names, lst):
                    return {"lst": [lst[0] - _i32(1), lst[1] + _i32(1)]}

                out = CanonicalizeWhile.scf_while_dispatch(
                    before_fn, after_fn, result_names=("lst",), result_values=(lst,)
                )
                assert isinstance(out, list) and len(out) == 2
                func.ReturnOp([out[0].ir_value(), out[1].ir_value()])

        text = str(module)
        assert module.operation.verify()
        assert "scf.while" in text


def test_while_carries_tuple_of_dict():
    """while t[0] > 0: carry a tuple containing a dict (mixed nesting)."""
    with Context(), Location.unknown():
        module = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(module.body):
            f = func.FuncOp("while_tuple_of_dict", FunctionType.get([], [i32, i32]))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                t = (_i32(4), {"c": _i32(0)})

                def before_fn(names, t):
                    assert isinstance(t, tuple) and isinstance(t[1], dict)
                    return t[0] > _i32(0)

                def after_fn(names, t):
                    return {"t": (t[0] - _i32(1), {"c": t[1]["c"] + _i32(1)})}

                out = CanonicalizeWhile.scf_while_dispatch(before_fn, after_fn, result_names=("t",), result_values=(t,))
                assert isinstance(out, tuple) and isinstance(out[1], dict)
                func.ReturnOp([out[0].ir_value(), out[1]["c"].ir_value()])

        text = str(module)
        assert module.operation.verify()
        assert "scf.while" in text
