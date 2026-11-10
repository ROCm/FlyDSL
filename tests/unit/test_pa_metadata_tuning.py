# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import importlib

import kernels.attention.pa_metadata_tuning as tuning


def _lookup(**overrides):
    kwargs = {
        "num_cu": 80,
        "batch_size": 81,
        "num_blocks": 648,
        "query_length": 4,
        "per_token_kv": True,
        "num_query_heads": 16,
        "num_kv_heads": 1,
        "head_dim": 128,
        "block_size": 1024,
        "device_tensor": None,
    }
    kwargs.update(overrides)
    return tuning.lookup_pa_metadata_grid_multiplier(**kwargs)


def test_pa_metadata_tuner_persists_and_runtime_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    importlib.reload(tuning)
    active_grid = {"value": None}

    def runner(grid_multiplier):
        active_grid["value"] = grid_multiplier
        return grid_multiplier

    def bench(call, warmup, rep):
        call()
        return {1: 2.0, 2: 1.0}[active_grid["value"]]

    tuner = tuning.make_pa_metadata_grid_autotuner(
        candidates=[1, 2],
        warmup=0,
        rep=1,
        do_bench=bench,
    )
    tuner_args = (81, 648, 4, True, 80, 16, 1, 128, 1024, None, runner)
    tuner(*tuner_args)

    assert tuning.get_cached_config(tuner, *tuner_args).kwargs["grid_multiplier"] == 2
    config_path = tuning.persistent_config_path(tuner, *tuner_args)
    assert config_path == tmp_path / "run_pa_metadata_grid_config.json"
    assert config_path.is_file()

    importlib.reload(tuning)
    assert _lookup() == 2
    assert _lookup(batch_size=32) is None


def test_pa_metadata_tuner_key_excludes_context_length():
    assert "context_length" not in tuning.PA_METADATA_GRID_KEY
