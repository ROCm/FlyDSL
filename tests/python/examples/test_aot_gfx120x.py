# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Pytest collection for the gfx120x AOT exports.

``aot_gfx120x_example.py`` is also a script, so ``python_files = test_*.py``
does not collect it. These wrappers are the functions that scan does collect.
"""

from pathlib import Path

import pytest

from tests.python.examples.aot_gfx120x_example import (
    test_aot_gfx120x_broaden_export as _broaden,
)
from tests.python.examples.aot_gfx120x_example import (
    test_aot_gfx120x_flash_attn_export as _fa,
)
from tests.python.examples.aot_gfx120x_example import (
    test_aot_gfx120x_rope_and_w8a16_export as _rope,
)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_aot_gfx120x_rope_and_w8a16_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile and export gfx120x RoPE and W8A16. No live GPU."""
    return _rope(tmp_path, monkeypatch)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_aot_gfx120x_flash_attn_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile and export gfx120x bf16 FlashAttention. No live GPU."""
    return _fa(tmp_path, monkeypatch)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_aot_gfx120x_broaden_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile and export the rest of the shipped gfx120x AOT families."""
    return _broaden(tmp_path, monkeypatch)
