# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``capture_user_location`` records the file, line and column of each user frame an op is built under."""

import importlib.util
import sys

import pytest

import flydsl.expr.meta as meta
from flydsl._mlir import ir
from flydsl.expr.meta import capture_user_location, tracing_context

pytestmark = [pytest.mark.l0_backend_agnostic]


def _source_position(marker, call="capture_user_location"):
    """Line and column of *call* on the line ending in ``# <marker>`` (column 0 before 3.11)."""
    with open(_SOURCE) as f:
        for line, text in enumerate(f, 1):
            if text.rstrip().endswith(f"# {marker}"):
                col = text.index(call) if sys.version_info >= (3, 11) else 0
                return line, col
    raise AssertionError(f"no line ends in # {marker}")


def _printed(path):
    """*path* as MLIR prints it in a location, which escapes the backslashes of Windows paths."""
    return str(path).replace("\\", "\\\\")


def _capture_two():
    first = capture_user_location()  # first
    second = [capture_user_location()][0]  # second
    return first, second


def _boundary():
    return _inner_capture()  # boundary


def _inner_capture():
    return capture_user_location()  # inner


# The file the captures are reported in; it can differ from ``__file__`` when the checkout is reached
# through another path and the cached bytecode remembers the first one.
_SOURCE = _capture_two.__code__.co_filename


@pytest.fixture(autouse=True)
def _default_loc_depth(monkeypatch):
    # A depth of 1 drops the call-site chain these tests match on.
    monkeypatch.delenv("FLYDSL_DEBUG_MAX_LOC_DEPTH", raising=False)


def test_each_capture_records_its_own_line_and_column():
    with ir.Context(), ir.Location.unknown():
        locs = _capture_two()

    for loc, marker in zip(locs, ("first", "second")):
        line, col = _source_position(marker)
        assert str(loc).startswith(f'loc(callsite("{_printed(_SOURCE)}":{line}:{col} at ')


def test_identical_code_in_two_files_keeps_each_file(tmp_path):
    # Code objects with the same body and line numbers compare equal across files.
    helpers = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.py"
        path.write_text("def helper(fn):\n    return fn()\n")
        spec = importlib.util.spec_from_file_location(f"_capture_location_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helpers.append((module.helper, path))
    assert helpers[0][0].__code__ == helpers[1][0].__code__

    # One tracing scope, so both captures go through the same position cache.
    with ir.Context(), ir.Location.unknown(), tracing_context(_boundary):
        for helper, path in helpers:
            assert str(helper(capture_user_location)).startswith(f'loc(callsite("{_printed(path)}":2:')


def test_kernel_frame_is_kept_at_the_depth_limit(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_MAX_LOC_DEPTH", "1")
    with ir.Context(), ir.Location.unknown(), tracing_context(_boundary):
        loc = str(_boundary())

    for marker, call in (("inner", "capture_user_location"), ("boundary", "_inner_capture")):
        line, col = _source_position(marker, call)
        assert f'"{_printed(_SOURCE)}":{line}:{col}' in loc, loc


def test_position_cache_lives_for_one_trace():
    with ir.Context(), ir.Location.unknown():
        with tracing_context(_boundary):
            _boundary()
            assert meta._tls.source_position_cache
        assert meta._tls.source_position_cache is None


def test_position_cache_is_per_trace_not_per_fastmath_scope():
    with ir.Context(), ir.Location.unknown(), tracing_context(fastmath=None):
        with tracing_context(_boundary):
            _boundary()
            cache = meta._tls.source_position_cache
            assert cache
            with tracing_context(_capture_two):
                assert meta._tls.source_position_cache is cache
        assert meta._tls.source_position_cache is None


def test_position_cache_is_dropped_when_a_trace_raises():
    with pytest.raises(RuntimeError):
        with tracing_context(_boundary):
            raise RuntimeError
    assert meta._tls.source_position_cache is None
