# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import linecache
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

from .._mlir import ir
from ..expr.meta import _is_framework_file
from ..utils import env

__all__ = [
    "DSLCompileError",
    "diag_records_from_mlir_error",
    "dsl_ir_diagnostics",
    "install_excepthook",
]


@dataclass
class SourceFrame:
    filename: str
    line: int
    col: int = 0
    end_col: Optional[int] = None


@dataclass
class DiagRecord:
    message: str
    chain: List[SourceFrame] = field(default_factory=list)


def location_chain(loc) -> List[SourceFrame]:
    """Flatten an MLIR ``Location`` into ``[innermost, ..., outermost]`` frames.

    Handles call-site chains (``callee`` first, then ``caller`` recursively),
    name locations (unwrap ``child_loc``), and fused locations (each child in
    order).  Frames pointing at synthetic ``<...>`` sources are skipped.
    """
    if loc is None:
        return []
    try:
        if loc.is_a_callsite():
            return location_chain(loc.callee) + location_chain(loc.caller)
        if loc.is_a_name():
            return location_chain(loc.child_loc)
        if loc.is_a_fused():
            out: List[SourceFrame] = []
            for child in loc.locations:
                out.extend(location_chain(child))
            return out
        if loc.is_a_file():
            filename, line = loc.filename, loc.start_line
            if not filename or filename.startswith("<") or not line:
                return []  # synthetic source we cannot point a user at
            end_col = getattr(loc, "end_col", 0) or 0
            return [SourceFrame(filename, line, getattr(loc, "start_col", 0) or 0, end_col or None)]
    except Exception:
        return []
    return []  # unknown / opaque location: nothing locatable


def diag_record_from_diagnostic(d) -> DiagRecord:
    message = str(getattr(d, "message", "") or d)
    chain = []
    loc = getattr(d, "location", None)
    if loc is not None:
        try:
            chain = location_chain(loc)
        except Exception:
            chain = []
    return DiagRecord(message=message, chain=chain)


def diag_records_from_mlir_error(err) -> List[DiagRecord]:
    records: List[DiagRecord] = []
    for d in getattr(err, "error_diagnostics", None) or []:
        records.append(diag_record_from_diagnostic(d))
    return records


class DSLCompileError(RuntimeError):
    """Raised when MLIR verification or an MLIR pass pipeline fails."""

    def __init__(self, message: str, *, diagnostics: Optional[list] = None):
        self.diagnostics = diagnostics or []
        usable = [r for r in self.diagnostics if r and r.message]
        if not usable:
            super().__init__(message)
            return

        blocks = []
        for rec in usable:
            parts = [rec.message]
            if rec.chain:
                parts.append("")
                parts.append("DSL Traceback (most recent operation last):")
                # chain is innermost-first; print outermost-first so the offending op is last.
                for frame in reversed(rec.chain):
                    parts.append(f'  File "{frame.filename}", line {frame.line}')
                    src = linecache.getline(frame.filename, frame.line)
                    if src:
                        stripped = src.rstrip("\n")
                        parts.append(f"    {stripped.strip()}")
                        # caret aligned under the column within the stripped line
                        indent = len(stripped) - len(stripped.lstrip())
                        caret_col = max(frame.col - indent, 0)
                        width = frame.end_col - frame.col if (frame.end_col and frame.end_col > frame.col) else 1
                        parts.append("    " + " " * caret_col + "^" * width)
            blocks.append("\n".join(parts))
        super().__init__("\n\n".join(blocks))


_dsl_excepthook_installed = False


def install_excepthook() -> None:
    """Make an uncaught :class:`DSLCompileError` print as a clean Python-native error.

    The output keeps the user's own Python call stack (where they invoked the
    ``@flyc.jit``) -- with DSL-internal frames filtered out -- followed by
    ``DSLCompileError: <message>`` whose message already carries the kernel
    call-site chain and source snippet. Installed lazily and idempotently from
    the ``@flyc.jit`` / ``@flyc.kernel`` decorators.
    """
    global _dsl_excepthook_installed
    if _dsl_excepthook_installed:
        return
    _dsl_excepthook_installed = True
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if not isinstance(exc, DSLCompileError) or env.debug.show_stacktrace:
            # Not dsl error, or the escape hatch is on: show the full raw traceback
            # (DSL-internal frames + the chained MLIRError).
            previous(exc_type, exc, tb)
            return
        # Keep only the user's frames from the Python call stack (drop the DSL
        # library frames between the launcher call and where the error is raised).
        user_frames = [fs for fs in traceback.extract_tb(tb) if not _is_framework_file(fs.filename)]
        out = ""
        if user_frames:
            # User's Python call stack, then a rule separating it from the DSL error.
            out += "Traceback (most recent call last):\n" + "".join(traceback.format_list(user_frames))
            out += "-" * 40 + "\n"
        out += f"{exc_type.__name__}: {exc}\n"
        sys.stderr.write(out)

    sys.excepthook = hook


@contextmanager
def dsl_ir_diagnostics(ctx):
    """Collect MLIR error diagnostics emitted during a ``with`` block.

    Yields a list of :class:`DiagRecord`.
    Only ``ERROR`` severity messages are captured.
    """
    records: list = []

    def _handler(d):
        if d.severity == ir.DiagnosticSeverity.ERROR:
            records.append(diag_record_from_diagnostic(d))
            return True
        return False

    handler = ctx.attach_diagnostic_handler(_handler)
    try:
        yield records
    finally:
        if handler.attached:
            handler.detach()
