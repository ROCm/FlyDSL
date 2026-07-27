# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import csv
import functools
from pathlib import Path
from typing import Iterable, Mapping

PA_METADATA_TUNING_CSV = Path(__file__).with_name("pa_metadata_grid_tuning.csv")

PA_METADATA_TUNING_FIELDS = (
    "arch",
    "num_cu",
    "batch_size",
    "num_blocks",
    "context_length",
    "query_length",
    "per_token_kv",
    "num_query_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "grid_multiplier",
    "latency_us",
)

_KEY_FIELDS = (
    "arch",
    "num_cu",
    "batch_size",
    "num_blocks",
    "query_length",
    "per_token_kv",
    "num_query_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
)

_INT_FIELDS = set(_KEY_FIELDS) - {"arch", "per_token_kv"} | {
    "context_length",
    "grid_multiplier",
}


def _normalize_row(row: Mapping) -> dict:
    normalized = {field: row[field] for field in PA_METADATA_TUNING_FIELDS}
    normalized["arch"] = str(normalized["arch"])
    for field in _INT_FIELDS:
        normalized[field] = int(normalized[field])
    per_token_kv = normalized["per_token_kv"]
    if isinstance(per_token_kv, str):
        if per_token_kv not in {"0", "1"}:
            raise ValueError(f"per_token_kv must be 0 or 1, got {per_token_kv!r}")
        per_token_kv = per_token_kv == "1"
    normalized["per_token_kv"] = bool(per_token_kv)
    normalized["latency_us"] = float(normalized["latency_us"])
    if normalized["grid_multiplier"] < 1:
        raise ValueError("grid_multiplier must be positive")
    return normalized


def _row_key(row: Mapping) -> tuple:
    return tuple(row[field] for field in _KEY_FIELDS)


def read_pa_metadata_tuning_rows(path: Path = PA_METADATA_TUNING_CSV) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(PA_METADATA_TUNING_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [_normalize_row(row) for row in reader]
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate tuning keys")
    return rows


@functools.lru_cache(maxsize=None)
def _load_pa_metadata_tuning(path: Path) -> dict[tuple, int]:
    return {_row_key(row): row["grid_multiplier"] for row in read_pa_metadata_tuning_rows(path)}


def lookup_pa_metadata_grid_multiplier(
    *,
    arch: str,
    num_cu: int,
    batch_size: int,
    num_blocks: int,
    query_length: int,
    per_token_kv: bool,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    path: Path = PA_METADATA_TUNING_CSV,
) -> int | None:
    key = (
        str(arch),
        int(num_cu),
        int(batch_size),
        int(num_blocks),
        int(query_length),
        bool(per_token_kv),
        int(num_query_heads),
        int(num_kv_heads),
        int(head_dim),
        int(block_size),
    )
    return _load_pa_metadata_tuning(Path(path)).get(key)


def write_pa_metadata_tuning_rows(
    rows: Iterable[Mapping],
    path: Path = PA_METADATA_TUNING_CSV,
) -> None:
    path = Path(path)
    merged = {_row_key(row): row for row in read_pa_metadata_tuning_rows(path)}
    for row in rows:
        normalized = _normalize_row(row)
        merged[_row_key(normalized)] = normalized

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PA_METADATA_TUNING_FIELDS)
        writer.writeheader()
        for key in sorted(merged):
            row = dict(merged[key])
            row["per_token_kv"] = int(row["per_token_kv"])
            writer.writerow(row)
    temporary.replace(path)
    _load_pa_metadata_tuning.cache_clear()
