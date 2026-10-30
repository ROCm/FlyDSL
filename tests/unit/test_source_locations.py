# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Source coordinates and full user call chains survive the fast frame lookup."""

import gc
import inspect
import subprocess
import sys
import types
import weakref

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith
from flydsl.expr import meta

pytestmark = [pytest.mark.l0_backend_agnostic]


def _inspect_position(frame):
    info = inspect.getframeinfo(frame, context=0)
    pos = getattr(info, "positions", None)
    line = pos.lineno if pos is not None and pos.lineno is not None else frame.f_lineno
    col = pos.col_offset if pos is not None and pos.col_offset is not None else 0
    return info.filename, line, col


def _collect_position():
    frame = inspect.currentframe().f_back
    try:
        actual = meta._frame_source_position(frame)
        assert actual == _inspect_position(frame)
        return actual
    finally:
        del frame


def _same_line_calls():
    return _collect_position(), _collect_position()


def test_distinguishes_columns_on_the_same_line():
    left, right = _same_line_calls()
    assert left[:2] == right[:2]
    if hasattr(_same_line_calls.__code__, "co_positions") and any(
        col is not None for _line, _end_line, col, _end_col in _same_line_calls.__code__.co_positions()
    ):
        assert left[2] != right[2]


def test_equal_code_objects_keep_their_own_filenames(tmp_path):
    original = _same_line_calls.__code__
    relocated = original.replace(co_filename=str(tmp_path / "relocated.py"))
    assert original == relocated and original is not relocated
    first = _same_line_calls()
    second = types.FunctionType(relocated, globals())()
    assert first[0][0] != second[0][0]
    assert first[0][1:] == second[0][1:]


def test_replaced_code_uses_new_source_positions(tmp_path):
    namespace = {"collect": _collect_position}
    filename = str(tmp_path / "changing.py")
    exec(compile("def probe(): return collect()\n", filename, "exec"), namespace)
    probe = namespace["probe"]
    before = probe()
    exec(compile("\n\ndef probe():\n    return collect()\n", filename, "exec"), namespace)
    probe.__code__ = namespace["probe"].__code__
    after = probe()
    assert before[0] == after[0]
    assert before[1] != after[1]


def test_bytecode_filename_tracks_source_availability(tmp_path):
    source = tmp_path / "changing.py"
    bytecode = str(source) + "c"
    namespace = {"collect": _collect_position}
    exec(compile("def probe(): return collect()\n", bytecode, "exec"), namespace)
    probe = namespace["probe"]
    assert probe()[0] == bytecode
    source.write_text("def probe(): return collect()\n")
    assert probe()[0] == str(source)
    source.unlink()
    assert probe()[0] == bytecode


def test_generator_positions_before_and_between_yields():
    def generate():
        yield 1
        yield 2

    generator = generate()
    try:
        for _ in range(3):
            frame = generator.gi_frame
            assert meta._frame_source_position(frame) == _inspect_position(frame)
            next(generator, None)
    finally:
        generator.close()
        del frame


def test_legacy_frame_without_column_information():
    class LegacyCode:
        co_filename = "legacy.py"

    frame = types.SimpleNamespace(f_code=LegacyCode(), f_lasti=0, f_lineno=42)
    assert meta._frame_source_position(frame) == ("legacy.py", 42, 0)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="co_positions requires Python 3.11")
def test_missing_debug_ranges_preserve_line_and_zero_column():
    program = """
import inspect
from flydsl.expr import meta
def check():
    frame = inspect.currentframe().f_back
    try:
        expected = inspect.getframeinfo(frame, context=0)
        assert expected.positions.col_offset is None
        assert meta._frame_source_position(frame) == (expected.filename, expected.lineno, 0)
    finally:
        del frame
def caller():
    check()
caller()
"""
    subprocess.run([sys.executable, "-X", "no_debug_ranges", "-c", program], check=True)


def test_position_cache_does_not_retain_frame_locals():
    class Sentinel:
        pass

    def query():
        sentinel = Sentinel()
        frame = inspect.currentframe()
        try:
            meta._frame_source_position(frame)
            return weakref.ref(sentinel)
        finally:
            del frame

    reference = query()
    gc.collect()
    assert reference() is None


@meta.dsl_loc_tracing
def _emit_constant():
    return arith.ConstantOp(ir.IntegerType.get_signless(32), 1)


def _nested_emit():
    return _emit_constant(), _emit_constant()


def _trace_boundary():
    _nested_emit()


def _traced_module():
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body), meta.tracing_context(_trace_boundary):
            _trace_boundary()
        module.operation.verify()
        return module.operation.get_asm(enable_debug_info=True)


@pytest.mark.parametrize("max_depth", [1, 2, 8])
def test_full_mlir_locations_match_inspect(monkeypatch, max_depth):
    monkeypatch.setenv("FLYDSL_DEBUG_MAX_LOC_DEPTH", str(max_depth))
    actual = _traced_module()
    monkeypatch.setattr(meta, "_frame_source_position", _inspect_position)
    assert actual == _traced_module()
