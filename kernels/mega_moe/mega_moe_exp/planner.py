# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Host-side sizing contract for the MegaMoE v2 stage-1 dispatch planner."""

from dataclasses import dataclass
from enum import IntEnum

BUFFER_OFFSET_ABI_BYTES = 1 << 32


class DispatchSlot(IntEnum):
    PAIR_BASE = 4
    P2P_TOKEN = 8
    P2P_SCALE = 9
    P2P_WEIGHT = 11
    P2P_SRCMAP = 12
    SORTED_EXPERT = 15
    TILE_ROW_BASE = 16
    NUM_VALID = 17
    SRCMAP = 19
    LOCAL_HIST = 29
    COUNT_MATRIX = 30
    P2P_COUNT_MATRIX = 31
    COUNT_DONE = 32
    P2P_COUNT_DONE = 33
    TASK_ROW_BASE = 34
    LOCAL_CURSOR = 35
    P2P_PAYLOAD_READY = 43
    PAIR_ORDER = 44
    P2P_TASK_ROW_BASE = 45
    P2P_PLAN_READY = 46
    PLAN_READY = 47
    PAIR_READY = 48


DISPATCH_TABLE_SIZE = max(DispatchSlot) + 1


class SmallFixedSlot(IntEnum):
    RUNNING = 0
    P2P_RUNNING = 1
    P2P_TOKEN = 2
    P2P_SCALE = 3
    P2P_WEIGHT = 5
    P2P_SRCMAP = 6
    EXPERT_COUNT = 7
    SORTED_EXPERT = 8
    TILE_ROW_BASE = 9
    NUM_VALID = 10
    ROUTE_DONE = 12
    LEADER_CLAIM = 13
    META_READY = 14
    SOURCE_DONE = 15
    P2P_SOURCE_DONE = 16
    ENTRY_DONE = 17
    P2P_ENTRY_DONE = 18


SMALL_FIXED_TABLE_SIZE = max(SmallFixedSlot) + 1


@dataclass(frozen=True)
class Stage1DispatchPlan:
    max_rows: int
    payload_bytes: int
    epoch_increment: int


def make_stage1_dispatch_plan(
    *,
    batch_size,
    npes,
    experts_per_rank,
    topk,
    tile_m,
    row_bytes,
    use_per_tile_payload_resource=False,
):
    """Build the compact dispatch capacity contract and validate its buffer ABI.

    Compact payload descriptors use an i64 tile base and i32 tile-local offsets.
    Payloads at or above 4 GiB therefore require the per-tile resource path.
    """
    batch_size = int(batch_size)
    npes = int(npes)
    experts_per_rank = int(experts_per_rank)
    topk = int(topk)
    tile_m = int(tile_m)
    row_bytes = int(row_bytes)
    max_rows = npes * batch_size * topk + experts_per_rank * tile_m
    payload_bytes = max_rows * row_bytes
    if payload_bytes >= BUFFER_OFFSET_ABI_BYTES and not use_per_tile_payload_resource:
        raise ValueError(
            "MegaMoE v2 compact stage1 exceeds the 32-bit buffer-resource ABI: "
            f"batch_size={batch_size}, max_rows={max_rows}, row_bytes={row_bytes}, "
            f"payload_bytes={payload_bytes} >= {BUFFER_OFFSET_ABI_BYTES}. "
            "Enable the per-tile payload resource path for allocations at or above 4 GiB."
        )
    return Stage1DispatchPlan(
        max_rows=max_rows,
        payload_bytes=payload_bytes,
        epoch_increment=npes,
    )
