#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Regression tests for Vector iteration / list() conversion (ROCm/FlyDSL#793).

Previously ``list(Vec(buffer_load(vec_width>1)))`` hung during tracing: Vector
defined ``__getitem__`` but neither ``__iter__`` nor ``__len__``, so CPython fell
back to the legacy sequence protocol (``v[0], v[1], ...``) which only stops on
``IndexError`` -- and the integer branch never raised one, emitting an unbounded
stream of ``vector.extract`` ops.
"""

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import func
from flydsl.expr.typing import Vector


def _in_vector_func(numel, body):
    """Run ``body(v)`` on a Vector over a ``vector<numel x f32>`` func argument."""
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                vec_ty = ir.VectorType.get([numel], ir.F32Type.get())
                f = func.FuncOp("t", ir.FunctionType.get([vec_ty], []))
                with ir.InsertionPoint(f.add_entry_block()):
                    (arg,) = list(f.entry_block.arguments)
                    result = body(Vector(arg))
                    func.ReturnOp([])
                    return result


@pytest.mark.l0_backend_agnostic
def test_list_of_vector_terminates():
    """list(Vec) yields exactly numel scalar values instead of hanging (#793)."""
    vals = _in_vector_func(4, list)
    assert len(vals) == 4


@pytest.mark.l0_backend_agnostic
def test_len_of_vector():
    assert _in_vector_func(4, len) == 4


@pytest.mark.l0_backend_agnostic
def test_vector_iteration():
    n = _in_vector_func(3, lambda v: len([x for x in v]))
    assert n == 3


@pytest.mark.l0_backend_agnostic
def test_vector_unpacking():
    def body(v):
        a, b, c = v
        return len((a, b, c))

    assert _in_vector_func(3, body) == 3


@pytest.mark.l0_backend_agnostic
def test_int_index_out_of_range_raises():
    def body(v):
        with pytest.raises(IndexError):
            v[v.numel]
        return True

    assert _in_vector_func(4, body)


@pytest.mark.l0_backend_agnostic
def test_negative_index():
    """v[-1] resolves to the last lane; too-negative raises IndexError."""

    def body(v):
        assert v[-1] is not None
        with pytest.raises(IndexError):
            v[-(v.numel + 1)]
        return True

    assert _in_vector_func(4, body)
