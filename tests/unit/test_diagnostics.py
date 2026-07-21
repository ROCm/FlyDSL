# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for the DSL compile-error formatting and excepthook filtering.

These cover the pure-Python diagnostics layer (message + caret rendering, the
``sys.excepthook`` frame filtering, and the ``FLYDSL_DEBUG_SHOW_STACKTRACE``
escape hatch) with no MLIR pass or GPU execution involved.
"""

import sys

import pytest

import flydsl.compiler.diagnostics as diagnostics
from flydsl.compiler.diagnostics import DiagRecord, DSLCompileError, SourceFrame, install_excepthook

pytestmark = [pytest.mark.l0_backend_agnostic]


# --------------------------------------------------------------------------- #
# DSLCompileError message / caret formatting
# --------------------------------------------------------------------------- #
def test_plain_message_without_diagnostics():
    err = DSLCompileError("pipeline failed")
    assert str(err) == "pipeline failed"


def test_diagnostics_without_message_fall_back_to_plain():
    # A record carrying no message must not produce an empty "DSL Traceback" block.
    err = DSLCompileError("verification failed", diagnostics=[DiagRecord(message="", chain=[])])
    assert str(err) == "verification failed"


def test_message_renders_source_snippet_and_caret(tmp_path):
    src_file = tmp_path / "user_kernel.py"
    # Indented by 4 spaces; the offending span is ``compute(a, b)``.
    src_file.write_text("def k():\n    x = compute(a, b)\n")

    # col=8 points at the start of ``compute`` (0-based within the raw line),
    # end_col=21 at the closing paren -> a 13-char span.
    frame = SourceFrame(filename=str(src_file), line=2, col=8, end_col=21)
    err = DSLCompileError("op failed", diagnostics=[DiagRecord(message="bad op", chain=[frame])])
    text = str(err)

    assert "bad op" in text
    assert "DSL Traceback (most recent operation last):" in text
    assert f'File "{src_file}", line 2' in text
    # The snippet is stripped of its leading indentation.
    assert "    x = compute(a, b)" in text
    # Caret column is offset by the stripped indentation (8 - 4 = 4) and the
    # caret width matches the span (21 - 8 = 13).
    assert "    " + " " * 4 + "^" * 13 in text


def test_chain_is_printed_outermost_first(tmp_path):
    src_file = tmp_path / "chain.py"
    src_file.write_text("outer_call()\ninner_call()\n")

    # chain is innermost-first; the rendered traceback must end with the innermost.
    inner = SourceFrame(filename=str(src_file), line=2, col=0, end_col=1)
    outer = SourceFrame(filename=str(src_file), line=1, col=0, end_col=1)
    err = DSLCompileError("x", diagnostics=[DiagRecord(message="m", chain=[inner, outer])])
    text = str(err)

    assert text.index("line 1") < text.index("line 2")


# --------------------------------------------------------------------------- #
# install_excepthook: frame filtering + escape hatch
# --------------------------------------------------------------------------- #
@pytest.fixture
def fresh_excepthook(monkeypatch):
    """Reset the idempotency guard and restore ``sys.excepthook`` afterwards."""
    saved = sys.excepthook
    monkeypatch.setattr(diagnostics, "_dsl_excepthook_installed", False)
    yield
    sys.excepthook = saved


def _make_traceback(err):
    """Build a real traceback that passes through a 'framework' then a 'user' file."""
    g = {"err": err}
    exec(compile("def framework_call():\n    raise err\n", "/fake/framework/lib.py", "exec"), g)
    exec(compile("def user_entry():\n    framework_call()\n", "/fake/user/app.py", "exec"), g)
    try:
        g["user_entry"]()
    except DSLCompileError as caught:
        return caught.__traceback__


def test_excepthook_filters_framework_frames(monkeypatch, capsys, fresh_excepthook):
    # Treat only the synthetic framework path as DSL-internal.
    monkeypatch.setattr(diagnostics, "_is_framework_file", lambda fn: "/fake/framework/" in fn)
    monkeypatch.setenv("FLYDSL_DEBUG_SHOW_STACKTRACE", "0")

    previous_called = []
    sys.excepthook = lambda *a: previous_called.append(a)
    install_excepthook()
    hook = sys.excepthook

    err = DSLCompileError("boom")
    tb = _make_traceback(err)
    hook(DSLCompileError, err, tb)

    out = capsys.readouterr().err
    assert "/fake/user/app.py" in out  # user frame kept
    assert "/fake/framework/lib.py" not in out  # framework frame filtered out
    assert "DSLCompileError: boom" in out
    assert "-" * 40 in out  # rule separating the user stack from the DSL error
    assert not previous_called  # custom rendering, not the raw traceback


def test_show_stacktrace_uses_raw_traceback(monkeypatch, capsys, fresh_excepthook):
    monkeypatch.setenv("FLYDSL_DEBUG_SHOW_STACKTRACE", "1")

    previous_called = []
    sys.excepthook = lambda *a: previous_called.append(a)
    install_excepthook()
    hook = sys.excepthook

    err = DSLCompileError("boom")
    tb = _make_traceback(err)
    hook(DSLCompileError, err, tb)

    assert previous_called  # escape hatch delegates to the original hook
    assert capsys.readouterr().err == ""  # no custom rendering emitted


def test_excepthook_passes_through_non_dsl_errors(monkeypatch, capsys, fresh_excepthook):
    monkeypatch.setenv("FLYDSL_DEBUG_SHOW_STACKTRACE", "0")

    previous_called = []
    sys.excepthook = lambda *a: previous_called.append(a)
    install_excepthook()
    hook = sys.excepthook

    err = ValueError("not a dsl error")
    hook(ValueError, err, None)

    assert previous_called  # non-DSL errors delegate to the original hook
    assert capsys.readouterr().err == ""
