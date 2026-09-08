#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Regression tests for the agent-docs gate.

Each case here is a bug the gate actually had, so a regression is a silent loss
of coverage (a miss) or a blocked PR (a false positive).
"""

import pytest

from scripts.check_docs_api import (
    _check_paths,
    _check_symbols,
    _split_frontmatter,
)


class _P:
    """Stand-in for a Path whose relative_to() is a no-op, so the checks are unit-testable."""

    def __init__(self, name="doc.md"):
        self._name = name

    def relative_to(self, _other):
        return self._name


KNOWN = {"copy", "gemm", "Int32", "rocdl", "make_buffer_tensor"}


# --- frontmatter delimiting -------------------------------------------------


def test_frontmatter_value_may_contain_a_triple_dash():
    """A `---` inside a value must not truncate the block (was read as invalid YAML)."""
    text = "---\nname: demo\ndescription: pass --- to separate args\n---\n\n# Body\n"
    assert _split_frontmatter(text) == "name: demo\ndescription: pass --- to separate args"


def test_frontmatter_absent_is_none():
    assert _split_frontmatter("# Just a heading\n") is None


def test_frontmatter_unterminated_is_none():
    assert _split_frontmatter("---\nname: demo\n") is None


# --- symbol scanning --------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "See https://example.com/fx.html for details.",
        "Trace written to `traces/rocdl.json`.",
        "The file kernels/fx.cfg holds the config.",
    ],
)
def test_urls_and_filenames_are_not_symbols(line):
    """`fx.`/`rocdl.` inside a URL or filename must not become a symbol lookup."""
    assert _check_symbols(_P(), line, KNOWN) == []


def test_python_fence_with_attributes_is_still_scanned():
    """A ```python fence carrying attributes was skipped wholesale, hiding drift."""
    text = '```python title="example.py"\nfx.no_such_symbol()\n```\n'
    assert len(_check_symbols(_P(), text, KNOWN)) == 1


def test_python3_fence_is_still_scanned():
    text = "```python3\nfx.no_such_symbol()\n```\n"
    assert len(_check_symbols(_P(), text, KNOWN)) == 1


def test_mlir_fence_is_skipped():
    """MLIR spells ops as rocdl.mfma.f32...; that is not a Python attribute."""
    text = "```mlir\nrocdl.mfma.f32.16x16x32f16 %a, %b\n```\n"
    assert _check_symbols(_P(), text, KNOWN) == []


def test_known_symbol_passes_and_unknown_fails():
    assert _check_symbols(_P(), "Use `fx.copy` here.", KNOWN) == []
    assert len(_check_symbols(_P(), "Use `fx.definitely_missing` here.", KNOWN)) == 1


def test_ignore_marker_suppresses_a_line():
    line = "fx.deliberate_placeholder <!-- api-check: ignore -->"
    assert _check_symbols(_P(), line, KNOWN) == []


# --- path scanning ----------------------------------------------------------


def test_dot_slash_prefixed_path_is_checked(tmp_path):
    """`./scripts/x` skipped the existence check entirely."""
    assert len(_check_paths(_P(), "Run `./scripts/definitely_missing.sh` first.")) == 1


def test_file_line_citation_is_not_a_path():
    """`path.py:419` is a citation, not a file that must exist."""
    assert _check_paths(_P(), "See `python/flydsl/expr/typing.py:419`.") == []


def test_existing_path_passes_and_missing_path_fails():
    assert _check_paths(_P(), "See `scripts/check_docs_api.py`.") == []
    assert len(_check_paths(_P(), "See `scripts/no_such_file_here.py`.")) == 1
