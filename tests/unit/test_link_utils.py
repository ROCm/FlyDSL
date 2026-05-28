# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Backend attach-target option helpers."""

import importlib
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPILER_DIR = _REPO_ROOT / "python" / "flydsl" / "compiler"


def _load_link_utils(monkeypatch):
    """Import compiler link helpers without importing JIT-only compiler exports."""
    for name in list(sys.modules):
        if name == "flydsl.compiler" or name.startswith("flydsl.compiler.link_utils"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    compiler_pkg = types.ModuleType("flydsl.compiler")
    compiler_pkg.__path__ = [str(_COMPILER_DIR)]
    monkeypatch.setitem(sys.modules, "flydsl.compiler", compiler_pkg)
    return importlib.import_module("flydsl.compiler.link_utils")


def test_link_lib_options_reject_pipeline_syntax_chars(monkeypatch):
    link_utils = _load_link_utils(monkeypatch)

    assert link_utils._format_link_lib_options(["/tmp/mori_shmem.bc"]) == "l=/tmp/mori_shmem.bc"

    for path in ["/tmp/with space.bc", "/tmp/with,comma.bc", "/tmp/with}brace.bc"]:
        with pytest.raises(ValueError, match="Cannot pass external bitcode path"):
            link_utils._format_link_lib_options([path])


def test_link_lib_options_attach_to_backend_target_passes(monkeypatch):
    link_utils = _load_link_utils(monkeypatch)

    fragments, found = link_utils._append_link_lib_options_to_attach_targets(
        [
            "gpu.module(convert-gpu-to-widget)",
            "widget-attach-target{chip=wgt100}",
            "gpu-module-to-binary{format=fatbin}",
        ],
        "l=/tmp/mori_shmem.bc",
    )

    assert found
    assert fragments[1] == "widget-attach-target{chip=wgt100 l=/tmp/mori_shmem.bc}"


def test_link_lib_options_still_attach_to_rocdl_target_passes(monkeypatch):
    link_utils = _load_link_utils(monkeypatch)

    fragments, found = link_utils._append_link_lib_options_to_attach_targets(
        [
            "gpu.module(convert-gpu-to-rocdl)",
            "rocdl-attach-target{chip=gfx942}",
            "gpu-module-to-binary{format=fatbin}",
        ],
        "l=/tmp/mori_shmem.bc",
    )

    assert found
    assert fragments[1] == "rocdl-attach-target{chip=gfx942 l=/tmp/mori_shmem.bc}"
