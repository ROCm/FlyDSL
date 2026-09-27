# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-time storage layout and CTA schedule for the GLM-5 MonoKernel."""

from kernels.common.fused_layer_config import (
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    QKV_A_ROWS,
    TOP_K,
    V_DIM,
)
from kernels.common.fused_layer_layout import (
    BLOCKS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
)

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
INDEX_KEYS_PER_TASK = 64

N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
N_UG_PER_SLOT = INTER // UG_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert XQ_WAVES * 4 <= WAVES


def dn_tile(samples: int, expert_mxfp4: bool = False) -> int:
    """Return rows per expert-down/FFN-reduce task for the tuned schedule."""

    return 32 if samples == 1 or (samples > 4 and expert_mxfp4) else HIDDEN // BLOCKS


def sparse_keys_per_task(samples: int) -> int:
    """Use narrower sparse-attention tiles when batch eight fills the LDS arena."""

    return 32 if samples > 4 else 64


def ug_split(samples: int):
    """Return the balanced up/gate leftover split for batches two and four."""

    full_tiles, remainder = divmod((samples * TOP_K + 1) * N_UG_PER_SLOT, BLOCKS)
    if samples not in (2, 4) or remainder == 0 or BLOCKS % remainder:
        return None
    segments = BLOCKS // remainder
    if (HIDDEN // 128) % segments or (HIDDEN // 128) // segments > WAVES // 2:
        return None
    return full_tiles, segments


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
):
    """Return byte offsets for per-rank scratch and symmetric peer buffers."""

    split_count = sparse_attention_topk // sparse_keys_per_task(samples)
    pair_bytes = 8
    items = [
        ("q_a", samples * Q_LORA * pair_bytes),
        ("q_an", samples * Q_LORA // 2 * pair_bytes),
        ("kv_a", samples * (KV_LORA + PE_DIM) * pair_bytes),
        ("kvnew", samples * KV_LORA * pair_bytes),
        ("penew", samples * PE_DIM * pair_bytes),
        ("q_nope", samples * heads * NOPE_DIM * pair_bytes),
        ("q_pe", samples * heads * PE_DIM * pair_bytes),
        ("q_lat", samples * heads * KV_LORA * pair_bytes),
        ("sp_acc", samples * split_count * heads * KV_LORA * pair_bytes),
        ("sp_m", samples * split_count * heads * pair_bytes),
        ("sp_l", samples * split_count * heads * pair_bytes),
        ("o", samples * heads * V_DIM * pair_bytes),
        ("a", samples * HIDDEN * pair_bytes),
        ("scores", samples * N_EXPERTS * pair_bytes),
        ("xq", samples * HIDDEN // 4 * pair_bytes),
        ("xqs", samples * XQ_BLOCKS * pair_bytes),
        ("sel", samples * MOE_SLOTS * pair_bytes),
        ("prob", samples * MOE_SLOTS * pair_bytes),
        ("mid", samples * MOE_SLOTS * INTER * pair_bytes),
        ("ugp", BLOCKS * samples * 2 * UG_TILE * pair_bytes),
        ("xqd", samples * HIDDEN * 4),
    ]
    if with_indexer:
        items += [
            ("index_k", samples * INDEX_DIM * pair_bytes),
            ("index_k_new", samples * INDEX_DIM // 2 * pair_bytes),
            ("index_ready", samples * pair_bytes),
            ("index_w", samples * INDEX_HEADS * pair_bytes),
            ("index_q", samples * INDEX_Q_ROWS // 2 * pair_bytes),
            ("index_scores", samples * index_max_seq * pair_bytes),
            ("indices", samples * sparse_attention_topk * 4),
            ("indices_ready", samples * pair_bytes),
        ]

    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    symmetric = {
        "attn": 0,
        "ffn": region,
        "_part_stride": part,
        "_bytes": 2 * region,
    }
    return scratch, symmetric


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
):
    """Return ``(stage name, task count)`` pairs in execution order."""

    tasks = [
        ("qkv_a", N_QKV_A),
        ("q_norm", samples),
        ("cache", 1),
        ("q_b", heads * (NOPE_DIM + PE_DIM) // Q_B_TILE),
    ]
    if with_indexer:
        tasks += [("index_q", INDEX_Q_ROWS // INDEX_TILE)]
    tasks += [("uk", heads * KV_LORA // UK_TILE)]
    if with_indexer:
        tasks += [
            ("index_score", samples * ((index_max_seq + INDEX_KEYS_PER_TASK - 1) // INDEX_KEYS_PER_TASK)),
            ("index_select", samples),
        ]
    tasks += [
        ("split", samples * (sparse_attention_topk // sparse_keys_per_task(samples))),
        ("uv", samples * (heads * V_DIM // UV_TILE)),
        ("o", N_ROW_TILES),
        ("router", samples * N_ROUTER),
        ("ug", BLOCKS if samples == 1 else samples * BLOCKS),
        ("down", HIDDEN // dn_tile(samples, expert_mxfp4)),
    ]
    return tasks
