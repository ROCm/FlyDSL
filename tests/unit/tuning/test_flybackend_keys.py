# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GPU-free tests: how ``FlyBackend`` reads config keys (``llvm:``, ``env:``, hints)."""

import os

import pytest

from flydsl.tuning.artifacts import export_winner
from flydsl.tuning.flybackend import env_overrides, merge_hints, split_config, split_env

pytestmark = pytest.mark.l0_backend_agnostic


def test_split_config_routes_each_key_kind():
    kwargs, hints = split_config(
        {
            "tile": 64,
            "waves_per_eu": 2,
            "llvm:amdgpu-sched-strategy": "max-ilp",
            "env:FLYDSL_FOO": True,
        }
    )
    assert kwargs == {"tile": 64}
    assert hints == {"waves_per_eu": 2, "llvm_options": {"amdgpu-sched-strategy": "max-ilp"}}


def test_llvm_none_means_default():
    _, hints = split_config({"llvm:a": None, "llvm:b": False})
    assert hints == {"llvm_options": {"b": False}}
    _, hints = split_config({"llvm:a": None})
    assert "llvm_options" not in hints


def test_split_env_values():
    got = split_env({"env:A": True, "env:B": False, "env:C": 8, "env:D": "xor", "env:E": None, "tile": 1})
    assert got == {"A": "1", "B": "0", "C": "8", "D": "xor", "E": None}


def test_env_overrides_set_unset_and_restore(monkeypatch):
    monkeypatch.setenv("FSTEST_KEEP", "old")
    monkeypatch.delenv("FSTEST_NEW", raising=False)
    monkeypatch.setenv("FSTEST_GONE", "x")
    with env_overrides({"FSTEST_KEEP": "new", "FSTEST_NEW": "1", "FSTEST_GONE": None}):
        assert os.environ["FSTEST_KEEP"] == "new"
        assert os.environ["FSTEST_NEW"] == "1"
        assert "FSTEST_GONE" not in os.environ
    assert os.environ["FSTEST_KEEP"] == "old"
    assert "FSTEST_NEW" not in os.environ
    assert os.environ["FSTEST_GONE"] == "x"


def test_env_overrides_restore_on_error(monkeypatch):
    monkeypatch.delenv("FSTEST_NEW", raising=False)
    with pytest.raises(RuntimeError):
        with env_overrides({"FSTEST_NEW": "1"}):
            raise RuntimeError("boom")
    assert "FSTEST_NEW" not in os.environ


def test_merge_hints_merges_llvm_options_key_by_key():
    base = {"fast_fp_math": True, "llvm_options": {"a": False, "b": True}}
    over = {"waves_per_eu": 2, "llvm_options": {"b": False, "c": 1}}
    assert merge_hints(base, over) == {
        "fast_fp_math": True,
        "waves_per_eu": 2,
        "llvm_options": {"a": False, "b": False, "c": 1},
    }
    assert merge_hints(base, {}) == base
    assert merge_hints({}, over) == over


@pytest.mark.parametrize("key", ["llvm:x", "env:X"])
def test_export_winner_refuses_keys_outside_the_launcher(key):
    with pytest.raises(ValueError, match="offline artifacts"):
        export_winner(None, {"tile": 64, key: 1})
