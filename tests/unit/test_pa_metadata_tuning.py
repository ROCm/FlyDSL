# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import csv

import pytest

from kernels.attention.pa_metadata_tuning import (
    PA_METADATA_TUNING_CSV,
    lookup_pa_metadata_grid_multiplier,
    read_pa_metadata_tuning_rows,
    write_pa_metadata_tuning_rows,
)


def _row(**overrides):
    row = {
        "arch": "gfx942",
        "num_cu": 80,
        "batch_size": 81,
        "num_blocks": 648,
        "context_length": 8192,
        "query_length": 4,
        "per_token_kv": True,
        "num_query_heads": 16,
        "num_kv_heads": 1,
        "head_dim": 128,
        "block_size": 1024,
        "grid_multiplier": 1,
        "latency_us": 263.4,
    }
    row.update(overrides)
    return row


def test_default_pa_metadata_tuning_table():
    rows = read_pa_metadata_tuning_rows()
    assert rows
    assert (
        lookup_pa_metadata_grid_multiplier(
            arch="gfx942",
            num_cu=80,
            batch_size=81,
            num_blocks=648,
            query_length=4,
            per_token_kv=True,
            num_query_heads=16,
            num_kv_heads=1,
            head_dim=128,
            block_size=1024,
        )
        == 1
    )
    assert PA_METADATA_TUNING_CSV.is_file()


def test_pa_metadata_tuning_csv_roundtrip_and_upsert(tmp_path):
    path = tmp_path / "pa_tuning.csv"
    write_pa_metadata_tuning_rows([_row()], path)
    write_pa_metadata_tuning_rows(
        [_row(grid_multiplier=2, latency_us=250.0), _row(batch_size=32, num_blocks=256)], path
    )

    rows = read_pa_metadata_tuning_rows(path)
    assert len(rows) == 2
    assert (
        lookup_pa_metadata_grid_multiplier(
            arch="gfx942",
            num_cu=80,
            batch_size=81,
            num_blocks=648,
            query_length=4,
            per_token_kv=True,
            num_query_heads=16,
            num_kv_heads=1,
            head_dim=128,
            block_size=1024,
            path=path,
        )
        == 2
    )
    assert (
        lookup_pa_metadata_grid_multiplier(
            arch="gfx950",
            num_cu=80,
            batch_size=81,
            num_blocks=648,
            query_length=4,
            per_token_kv=True,
            num_query_heads=16,
            num_kv_heads=1,
            head_dim=128,
            block_size=1024,
            path=path,
        )
        is None
    )


def test_pa_metadata_tuning_rejects_invalid_boolean(tmp_path):
    path = tmp_path / "invalid.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_row())
        writer.writeheader()
        writer.writerow(_row(per_token_kv="true"))

    with pytest.raises(ValueError, match="per_token_kv"):
        read_pa_metadata_tuning_rows(path)
