# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import pytest

from flydsl.utils.file import atomic_write


def test_atomic_write_replaces_destination_only_after_success(tmp_path):
    destination = tmp_path / "artifact.json"
    destination.write_text("old", encoding="utf-8")

    with atomic_write(destination, mode="w", encoding="utf-8") as output:
        output.write("new")
        assert destination.read_text(encoding="utf-8") == "old"

    assert destination.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_write_preserves_destination_and_removes_temp_file_on_failure(tmp_path):
    destination = tmp_path / "artifact.json"
    destination.write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="serialize failed"):
        with atomic_write(destination, mode="w", encoding="utf-8") as output:
            output.write("partial")
            raise RuntimeError("serialize failed")

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_write_supports_binary_output(tmp_path):
    destination = tmp_path / "cache.pkl"

    with atomic_write(destination, mode="wb") as output:
        output.write(b"\x00cache")

    assert destination.read_bytes() == b"\x00cache"
