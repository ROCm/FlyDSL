# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU-free regression guard for committed offline tuned-config artifacts.

Analogous to aiter's ``check_tuned_op_regression.sh``: any config JSON checked
into ``configs/autotune/`` must be well-formed, loadable, and self-consistent
(its filename must match its declared name/spec/device). This catches a
hand-edited or corrupted committed config before it silently mis-serves at
runtime — without needing a GPU.

If no configs are committed yet, the test is a no-op (it only guards what
exists).
"""

import json
from pathlib import Path

import pytest

from flydsl.autotune import Config, offline_config_filename

# Committed offline configs live here (relative to repo root). Kept in sync with
# the FLYDSL_AUTOTUNE_CONFIG_DIR a project would point at.
_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "autotune"


def _committed_config_files():
    if not _CONFIG_ROOT.exists():
        return []
    return sorted(_CONFIG_ROOT.rglob("*.json"))


_FILES = _committed_config_files()


@pytest.mark.skipif(not _FILES, reason="no committed offline configs to guard")
@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_committed_config_wellformed(path):
    payload = json.loads(path.read_text())

    # Required, self-describing fields (matches _emit_offline_config).
    for field in ("name", "spec", "device_name", "config"):
        assert field in payload, f"{path.name}: missing '{field}'"
    assert isinstance(payload["name"], str) and payload["name"], f"{path.name}: name must be a non-empty string"
    assert (
        isinstance(payload["device_name"], str) and payload["device_name"]
    ), f"{path.name}: device_name must be a non-empty string"
    assert isinstance(payload["spec"], dict) and payload["spec"], f"{path.name}: spec must be a non-empty object"
    assert isinstance(payload["config"], dict) and payload["config"], f"{path.name}: config must be a non-empty object"

    # spec values must be JSON scalars (the offline lookup compares them by
    # value; a nested object/None would never match a live call).
    for k, v in payload["spec"].items():
        assert isinstance(v, (str, int, float, bool)), f"{path.name}: spec[{k!r}]={v!r} is not a scalar"

    # config values must be JSON scalars too — Config.from_dict accepts any dict
    # (it swallows unknown keys into kwargs), so a round-trip check alone proves
    # nothing; assert the values are sane knob types, not nested garbage.
    cfg = Config.from_dict(payload["config"])
    assert cfg.to_dict() == payload["config"], f"{path.name}: config not round-trip stable"
    for k, v in payload["config"].items():
        assert isinstance(v, (str, int, float, bool)), f"{path.name}: config[{k!r}]={v!r} is not a scalar"

    # Filename must equal the key the emit/lookup path builds. _emit writes JSON
    # with sort_keys and the tuner's key_fn sorts the spec, so the on-disk axis
    # order is alphabetical — reconstruct the expected name the same way.
    spec = sorted(payload["spec"].items())
    expected = offline_config_filename(payload["name"], spec, device_name=payload["device_name"])
    assert path.name == expected, f"{path.name}: filename does not match content ({expected})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
