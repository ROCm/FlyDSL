# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Explicit specialization and per-tile state for inlined stage emitters.

KernelPlan is a tuple so FlyDSL includes every option in closure cache identity.
Execution and operation contexts exist only during IR construction, never on the GPU.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


class KernelPlan(NamedTuple):
    """Validated numeric formats, stage geometry, ownership and LDS offsets.

    Uppercase fields retain the formulas used by the device tile mappings.
    Dictionaries contain only compile-time integer offsets or task bases.
    """

    WORK_WORDS: int
    index_grid_scale: int
    MISC_OFF: int
    RED_OFF: int
    OUT_OFF: int
    CACHE_READERS_JOINED: bool
    CACHE_TILE: int
    COHORT_ROW_TILES: bool
    CP_PARTS: int
    CP_RANKS: int
    D: int
    DEFER_OFF_CACHE: bool
    DNW_OFF: int
    DN_COHORT_PAIR: bool
    DN_CONTIG_CONTRIBUTIONS: bool
    DN_FULL_TASKS: int
    DN_INPUT_REUSE: bool
    DN_PAIR_FULL: int
    DN_PAIR_G: int
    DN_PAIR_ROWS: int
    DN_PAIR_SPLIT: bool
    DN_PAIR_WORK: int
    DN_SCALE_ROW: int
    DN_SPLIT_UNION_TAIL: bool
    DN_TILE: int
    DN_UNION_ALIGNED: bool
    DN_UNION_CANDIDATE: bool
    DN_UNION_INPUT_REUSE: bool
    DN_UNION_ROW10: bool
    DN_WORK_ITEMS: int
    DN_X_ROW: int
    DP_CROFF: int
    DP_CWORDS: int
    DP_DWOFF: int
    DP_KEYOFF: int
    DP_MSOFF: int
    DP_ROUTE_WORDS: int
    DP_SCROW: int
    DP_SLOTS: int
    DP_UMAX: int
    DP_XROW: int
    DU_CROFF: int
    DU_CWORDS: int
    DU_DWOFF: int
    DU_KEYOFF: int
    DU_KEY_WORDS: int
    DU_MSOFF: int
    DU_ROUTE_WORDS: int
    DU_SCROW: int
    DU_SLOTS: int
    DU_UMAX: int
    DU_XROW: int
    FRONT_PAYLOAD_RETIRE: bool
    G: int
    GLOBAL_QKV_A: bool
    GLOBAL_Q_B: bool
    GLOBAL_UG: bool
    GLOBAL_UG_WIDTH: int
    GLOBAL_UV: bool
    GROUPED_DN: bool
    GROUPED_UG: bool
    H: int
    INDEX_BF16_OFF: int
    INDEX_CAPACITY: int
    INDEX_CP_TILE: int
    INDEX_FP4_BYTES: int
    INDEX_PRE_ROPE: bool
    INDEX_QUERY_REUSE: int
    INDEX_Q_BF16_OFF: int
    INDEX_RADIX_BINS: int
    INDEX_RADIX_DIGITS: int
    INDEX_ROT_OFF: int
    INDEX_SCORE_REUSE: int
    INPUT_NORM_OFF: int
    INPUT_NORM_ONCE: bool
    INPUT_QA_BATCH: int
    I_PER_SLOT: int
    KEYS_OFF: int
    KS: int
    KT_OFF: int
    L: int
    LOCAL_INDEX_CAPACITY: int
    MIDQ_WORDS: int
    MID_SCALE_WORDS: int
    N_DN_TILES: int
    N_QB: int
    N_SPLIT: int
    N_SPLIT_TASKS: int
    N_UG: int
    N_UK: int
    N_UV: int
    O_K: int
    O_PAD4: bool
    PS: int
    PT_OFF: int
    QB_PER_HEAD: int
    QK_DIM: int
    QS: int
    RED_WORDS: int
    RESIDENT_INDEX_SELECT: bool
    ROW_TILES: int
    S: int
    SAMPLE_TILE: int
    SC: dict[str, int]
    SHARD_INDEX_Q: bool
    SPLIT_ALL_HEADS: bool
    SPLIT_GATHER_KEYS: int
    SPLIT_KEYS: int
    SPLIT_TASK_PAIR: int
    SY: dict[str, int]
    TILED_INDEXER: bool
    TILE_PEER_BYTES: int
    TILE_SCRATCH_BYTES: int
    UG_SCALE_ROW: int
    UG_TASK_ROUNDS: int
    UG_UNION_MAX: int
    UG_X_ROW: int
    UK_PER_HEAD: int
    UV_MERGE_OFF: int
    UV_REUSE: int
    W: int
    partition: str
    activation_fp4: bool
    atom_experts: bool
    attention_bf16: bool
    attention_k_chunks_per_unit: int
    attention_ptpc: bool
    base: dict[str, int]
    cache_fp8: bool
    dcp_size: int
    expert_inter: int
    expert_mxfp4: bool
    index_block_table_stride: int
    index_page_size: int
    index_request_width: int
    indexer_cp: bool
    indexer_quant: bool
    native_fp4_mfma: bool
    npes: int
    paired_ug_rows: bool
    rope_dtype: str
    scale: float
    topk: int
    total_samples: int
    use_atom_kv_cache: bool
    uv_scale_rows: int
    with_indexer: bool


@dataclass(frozen=True)
class ThreadContext:
    block_id: object
    global_block_id: object
    thread_id: object
    lane_id: object
    wave_id: object
    tile_id: object
    row_offset: object
    epoch: object


@dataclass(frozen=True)
class Workspace:
    scratch: object
    all_scratch: object
    lds: object
    activations: object
    reduction: object
    output: object
    routing_ids: object
    routing_weights: object
    temporary: object
    kv_tile: object
    rope_tile: object
    probabilities: object
    vector_type: object


@dataclass(frozen=True)
class ParallelContext:
    rank: object
    peer_table: object
    peer_destination: object
    peer_offset: object
    peer_slot: object
    buffer: object
    all_buffer: object


@dataclass(frozen=True)
class TokenMetadata:
    block_tables: object
    positions: object
    slot_mapping: object
    kv_indptr: object
    all_positions: object
    all_slots: object
    all_indptr: object
    first_position: object


@dataclass(frozen=True)
class ExecutionContext:
    """Scheduling, scratch ownership and communication; no model activations/weights."""

    thread: ThreadContext
    workspace: Workspace
    parallel: ParallelContext
    tokens: TokenMetadata


@dataclass(frozen=True)
class Operations:
    """Shared primitives grouped by operation instead of a flat list of callbacks."""

    memory: object
    reductions: object
    gemm: object
    normalization: object
    quantization: object
    communication: object


@dataclass(frozen=True)
class StageContext:
    """Compile-time implementation context, separate from each op's tensor contract."""

    plan: KernelPlan
    execution: ExecutionContext
    ops: Operations


def stage_source_key(source_dir: Path | None = None) -> str:
    """Include extracted stages/tools even on runtimes with same-directory key walks.

    Read at factory creation so edits in a long-lived development process also
    produce fresh compile identities. The digest is a host scalar in the kernel
    closure; it emits no device instructions.
    """
    root = Path(__file__).parent if source_dir is None else source_dir
    sources = sorted([*root.glob("*.py"), *(root / "tools").glob("*.py")])
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
