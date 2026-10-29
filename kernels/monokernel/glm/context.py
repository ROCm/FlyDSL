# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Explicit specialization and per-tile state for inlined stage emitters.

KernelPlan is a tuple so FlyDSL includes every option in closure cache identity.
TileState and DeviceTools exist only during IR construction, never on the GPU.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple


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
class TileState:
    """Per-tile address/view expressions and thread/rank/epoch state.

    Every value is created inside the kernel trace; stages never mutate this
    bundle or carry it across GPU launches. LDS contents may change in place.
    """

    attn_keys: object
    bias: object
    bid: object
    cur_pos: object
    dnw: object
    full_bid: object
    full_indptr: object
    full_out: object
    full_positions: object
    full_scratch: object
    full_slots: object
    full_sym: object
    g_in: object
    g_kv: object
    g_post: object
    g_q: object
    indices: object
    keys: object
    ktile: object
    kv_cache: object
    lane: object
    lds: object
    misc: object
    outs: object
    pe_cache: object
    peer_dst: object
    peer_offset: object
    peer_slot: object
    petile: object
    pl: object
    pos0: object
    positions: object
    r_h: object
    r_peers: object
    rank: object
    red: object
    rope_cos: object
    rope_sin: object
    row_offset: object
    row_tile: object
    s_dn: object
    s_o: object
    s_q_b: object
    s_qkv_a: object
    s_ug: object
    s_uk: object
    s_uv: object
    scratch: object
    slot_mapping: object
    sparse_kv_indptr: object
    sym: object
    tag: object
    tid: object
    index_params: object
    v4f: object
    w_dn: object
    w_o: object
    w_q_b: object
    w_qkv_a: object
    w_r: object
    w_ug: object
    w_uk: object
    w_uv: object
    wave: object
    x_out: object
    xs: object


@dataclass(frozen=True)
class DeviceTools:
    """Inlined expression emitters shared by every computation stage."""

    quant_moe: Callable
    ld_f32: Callable
    ld_bf16: Callable
    ld_rope: Callable
    quant_index_pair: Callable
    row_index_bounds: Callable
    row_active: Callable
    row_position: Callable
    row_slot: Callable
    row_writes_cache: Callable
    index_request_start: Callable
    index_page_slot: Callable
    lds_ld: Callable
    lds_st: Callable
    bf16_pair: Callable
    bf16_round: Callable
    index_arg: Callable
    mb: Callable
    cp_local_rank: Callable
    cp_peer: Callable
    cp_buffer: Callable
    put: Callable
    put2: Callable
    put_bf: Callable
    bf2_f32: Callable
    poll: Callable
    hint_wait: Callable
    pre_poll: Callable
    get: Callable
    getf: Callable
    getf_many: Callable
    get2_many: Callable
    get2: Callable
    get_bf2_many: Callable
    wave_sum: Callable
    wave_max: Callable
    block_sums: Callable
    block_sums_f64: Callable
    block_maxs: Callable
    block_sum: Callable
    unit_fp8: Callable
    unit_fp8x2: Callable
    unit_f8f8: Callable
    unit_ptpc: Callable
    unit_ptpc64: Callable
    unit_mxfp4: Callable
    unit_mxfp4_bf16: Callable
    unit_native_mxfp4: Callable
    lds_mxfp8: Callable
    unit_bf16: Callable
    unit_attention: Callable
    mma_units: Callable
    run_units: Callable
    reduce_rows: Callable
    emit_out: Callable
    pick: Callable
    attention_b_word: Callable
    emit_ptpc: Callable
    emit_ptpc_scalar: Callable
    stage_x_rmsnorm_ptpc: Callable
    stage_x_pairs_ptpc: Callable
    stage_x_rmsnorm: Callable
    load_x_rmsnorm: Callable
    stage_x_pairs: Callable
    quant_scaled: Callable
    quant_mxfp8: Callable
    pack_moe4: Callable
    publish_moe_quant: Callable
    moe_roundtrip: Callable
    stage_xq: Callable
    st_f8: Callable
    st_mxfp8: Callable
    st_ptpc: Callable
    load_bias: Callable
    route_top8: Callable
    ffn_owner_reduce: Callable
    o_owner_reduce: Callable
    peer_reduce: Callable
    dcp_merge_latent: Callable
    start: Callable
    n_sel: Callable
    dn_route: Callable


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
