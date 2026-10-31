# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile-time dispatch, task placement and overlaid LDS layout.

All live specialization predicates are evaluated here; stage emitters consume
the resulting immutable tuple. No GPU operations run during planning.
"""

from kernels.monokernel.config import (
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    SOFTMAX_SCALE,
    TOP_K,
    V_DIM,
    AttentionWeight,
    KvCacheLayout,
    as_kv_cache_layout,
)
from kernels.monokernel.glm.context import KernelPlan
from kernels.monokernel.glm.layout import (
    BLOCKS,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_KEYS_PER_TASK,
    INDEX_LOCAL_LDS_CAPACITY,
    INDEX_Q_ROWS,
    INDEX_TILE,
    N_QKV_A,
    Q_B_TILE,
    QKV_A_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
    XQ_BLOCKS,
    XQ_GROUPS,
    cache_samples_per_task,
    cohort_eligible,
    dn_tile,
    down_x_words,
    global_ug_chunks,
    grouped_ug_enabled,
    index_cp_capacity,
    index_cp_partitions,
    index_cp_tile,
    index_cp_world,
    index_local_capacity,
    index_radix_geometry,
    indexer_uses_tiles,
    layout,
    row_tile_size,
    sparse_heads_per_task,
    sparse_keys_per_task,
    stage_tasks,
)
from kernels.monokernel.layout import LAYER_SLOTS, THREADS


def build_plan(
    S: int = 1,
    heads: int = 8,
    npes: int = 8,
    topk: int = 2048,
    launches_per_step: int = 1,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    index_request_width: int = 1,
    index_page_size: int = 16,
    index_block_table_stride: int | None = None,
    expert_mxfp4: bool = False,
    atom_experts: bool = False,
    attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
    kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
    kv_cache_dtype: str = "bf16",
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    scale: float = SOFTMAX_SCALE,
    uv_scale_rows: int = 128,
    native_fp4_mfma: bool = False,
    moe_activation_dtype: str = "fp8",
    rope_dtype: str = "fp32",
    indexer_cp: bool = False,
    indexer_quant: bool = False,
    _partition: str = "full",
):
    INDEX_CP_TILE = index_cp_tile(indexer_quant)
    assert not indexer_quant or with_indexer
    INDEX_FP4_BYTES = INDEX_DIM // 2 + INDEX_DIM // 32
    total_samples = S
    assert 1 <= total_samples <= (40 if native_fp4_mfma else 12)
    S = row_tile_size(total_samples, native_fp4_mfma)
    ROW_TILES = total_samples // S
    assert uv_scale_rows in (64, 128)
    assert heads % WAVES == 0, "split attention maps one local head to each wave"
    assert kv_cache_dtype in ("bf16", "fp8")
    assert rope_dtype in ("fp32", "bf16")
    attention_weight = AttentionWeight(attention_weight)
    cache_layout = as_kv_cache_layout(kv_cache_layout)
    attention_bf16 = attention_weight is AttentionWeight.BF16
    attention_ptpc = attention_weight is AttentionWeight.FP8_PTPC
    use_atom_kv_cache = cache_layout is KvCacheLayout.ATOM
    cache_fp8 = kv_cache_dtype == "fp8"
    assert moe_activation_dtype in ("fp8", "fp4")
    activation_fp4 = moe_activation_dtype == "fp4"
    assert not activation_fp4 or native_fp4_mfma
    # Extend cohorts by geometry for TP4; retain the released TP8 dispatch.
    # Cohort scheduling and deferred cache publication require the ownership
    # and layout contracts below. Bounded row unrolling is selected separately
    # at dispatch and also applies to the other supported tiled paths.
    STATIC_ROW_TILES = (
        (total_samples == 40 or (npes == 4 and cohort_eligible(total_samples, npes, with_indexer, native_fp4_mfma)))
        and npes in (4, 8)
        and dcp_size == 1
        and (not indexer_cp or npes == 4)
        and attention_weight is AttentionWeight.FP8_BLOCK128
        and use_atom_kv_cache
        and (not cache_fp8 or npes == 4)
        and rope_dtype == "fp32"
    )
    assert not cache_fp8 or use_atom_kv_cache
    assert not atom_experts or expert_mxfp4
    assert not native_fp4_mfma or atom_experts
    attention_k_chunks_per_unit = 1 if attention_bf16 else 2
    SPLIT_KEYS = sparse_keys_per_task(S, heads)
    assert topk % SPLIT_KEYS == 0 and 1 <= S <= 12
    assert 1 <= launches_per_step <= LAYER_SLOTS
    assert not with_indexer or (topk == 2048 and index_max_seq % INDEX_KEYS_PER_TASK == 0)
    assert not with_indexer or (total_samples % index_request_width == 0 and index_max_seq % index_page_size == 0)
    if index_block_table_stride is None:
        index_block_table_stride = index_max_seq // index_page_size
    assert not with_indexer or index_block_table_stride >= index_max_seq // index_page_size
    assert not (attention_bf16 and with_indexer), "BF16 attention uses the external indexer"
    assert not indexer_cp or with_indexer
    H = heads
    L = H if output_heads is None else output_heads
    W = npes
    D = dcp_size
    expert_inter = inter
    assert expert_inter in (
        256,
        512,
    ), "GLM TP8/TP4 requires complete 256-CTA expert rounds"
    paired_ug_rows = expert_mxfp4 and not atom_experts
    assert D == 1 or (D == W and H == L * D)
    I_PER_SLOT = expert_inter // UG_TILE
    # Indexed queries permit early cache publication. External indices require
    # deferred stores and the cache version at each original tile boundary.
    DEFER_OFF_CACHE = STATIC_ROW_TILES and not with_indexer
    # TP4 uses bounded radix tasks at long capacities. The cache join and
    # per-tile scratch/peer epochs do not depend on context capacity.
    COHORT_ROW_TILES = STATIC_ROW_TILES and (not with_indexer or npes == 4 or index_max_seq == 4096)
    GLOBAL_UG = COHORT_ROW_TILES and npes == 4 and grouped_ug_enabled(S, npes, native_fp4_mfma, True)
    GLOBAL_UG_WIDTH = (
        32
        if indexer_quant and (_partition == "back" or (_partition == "full" and not indexer_cp))
        else (64 if with_indexer else 128)
    )
    INDEX_PRE_ROPE = (
        COHORT_ROW_TILES
        and with_indexer
        and (not indexer_cp or indexer_uses_tiles(index_max_seq, indexer_cp, indexer_quant))
    )
    INDEX_RADIX_DIGITS, INDEX_RADIX_BINS = index_radix_geometry(indexer_quant)
    INDEX_SCORE_REUSE = 8 if indexer_quant else (2 if COHORT_ROW_TILES else 1)
    # Reuse groups must fit both the resident tile and one request.
    INDEX_QUERY_REUSE = next(
        (q for q in (8, 6, 4, 2) if indexer_quant and S % q == 0 and index_request_width % q == 0), 1
    )
    SPLIT_TASK_PAIR = 2 if COHORT_ROW_TILES and npes == 8 and heads == 8 else 1
    SPLIT_GATHER_KEYS = SPLIT_KEYS * SPLIT_TASK_PAIR
    G = BLOCKS // ROW_TILES if COHORT_ROW_TILES else BLOCKS
    UG_TASK_ROUNDS = (expert_inter + G - 1) // G
    SC, SY = layout(
        S,
        H,
        W,
        topk,
        with_indexer,
        index_max_seq,
        indexer_cp=indexer_cp,
        inter=expert_inter,
        output_heads=L,
        dcp_size=D,
        native_fp4_mfma=native_fp4_mfma,
        indexer_quant=indexer_quant,
    )
    TILE_SCRATCH_BYTES = SC["_bytes"]
    TILE_PEER_BYTES = SY["_bytes"]
    stages = stage_tasks(
        S,
        H,
        topk,
        with_indexer,
        index_max_seq,
        expert_mxfp4,
        inter=expert_inter,
        indexer_cp=indexer_cp,
        npes=W,
        native_fp4_mfma=native_fp4_mfma,
        cohort_rows=COHORT_ROW_TILES,
        uv_reuse=COHORT_ROW_TILES,
        split_task_pair=SPLIT_TASK_PAIR,
        cohort_samples=total_samples,
        indexer_quant=indexer_quant,
    )
    CP_RANKS = index_cp_world(W, S, index_max_seq) if indexer_cp else 1
    INDEX_CAPACITY = index_cp_capacity(index_max_seq, CP_RANKS) if indexer_cp else index_max_seq
    TILED_INDEXER = with_indexer and indexer_uses_tiles(index_max_seq, indexer_cp, indexer_quant)
    LOCAL_INDEX_CAPACITY = index_local_capacity(index_max_seq, indexer_quant)
    SHARD_INDEX_Q = indexer_cp and not TILED_INDEXER and W > 1
    CP_PARTS = index_cp_partitions(index_max_seq, CP_RANKS, indexer_quant) if TILED_INDEXER else 0
    # Preserve v102 reuse when score work fills the resident grid. For smaller
    # workloads expose independent query/key groups before increasing reuse.
    # Four64-key tiles is the minimum: each of the four wave pairs owns one.
    if indexer_quant:
        score_tiles = INDEX_CAPACITY // INDEX_KEYS_PER_TASK
        score_workers = G * (2 if ROW_TILES == 1 or COHORT_ROW_TILES else 1)
        score_queries = [q for q in (8, 6, 4, 2, 1) if S % q == 0 and index_request_width % q == 0]
        INDEX_SCORE_REUSE, INDEX_QUERY_REUSE = next(
            ((k, q) for k in (8, 4) for q in score_queries if (S // q) * ((score_tiles + k - 1) // k) >= score_workers),
            (4, 1),
        )
    N_SPLIT = topk // SPLIT_KEYS
    N_SPLIT_TASKS = N_SPLIT // SPLIT_TASK_PAIR
    SPLIT_ALL_HEADS = (
        sparse_heads_per_task(S, H, W, native_fp4_mfma, COHORT_ROW_TILES and (with_indexer or npes == 4)) == 16
    )
    if DEFER_OFF_CACHE:
        assert ROW_TILES >= 2
    GLOBAL_QKV_A = (
        COHORT_ROW_TILES
        and total_samples % 4 == 0
        and npes == 4
        and native_fp4_mfma
        and H == 16
        and (not with_indexer or (indexer_quant and _partition in ("full", "front")))
        and not attention_ptpc
        and not attention_bf16
    )
    GLOBAL_UV = GLOBAL_QKV_A and GLOBAL_UG and D == 1 and L == H and not with_indexer
    O_PAD4 = GLOBAL_UV
    GLOBAL_Q_B = COHORT_ROW_TILES and total_samples % 4 == 0 and npes == 4 and native_fp4_mfma and H == 16
    QB_ROWS = H * (NOPE_DIM + PE_DIM)
    N_QB = QB_ROWS // Q_B_TILE
    # The index-Q row space is fixed at 256 tiles.  TP8 produces half from
    # q_b CTAs; TP4 produces all of them, one per CTA.
    assert not with_indexer or (INDEX_Q_ROWS // INDEX_TILE == BLOCKS and N_QB in (BLOCKS // 2, BLOCKS))
    QB_PER_HEAD = (NOPE_DIM + PE_DIM) // Q_B_TILE
    N_UK = H * KV_LORA // UK_TILE
    UK_PER_HEAD = KV_LORA // UK_TILE
    N_UV = H * V_DIM // UV_TILE
    UV_REUSE = 4 if COHORT_ROW_TILES else 1
    assert UV_REUSE == 1 or V_DIM // UV_TILE == UV_REUSE
    O_K = L * V_DIM
    N_UG = S * MOE_SLOTS * I_PER_SLOT
    QK_DIM = KV_LORA + PE_DIM
    # split LDS: bf16 q of all heads, then the KV latent / k_pe tiles (bf16 pairs); row
    # strides are padded by 4 words so the MFMA operand rows spread over the banks
    QS = QK_DIM // 2 + 4
    KS = KV_LORA // 2 + 4
    PS = PE_DIM // 2 + 4
    KT_OFF = H * QS
    PT_OFF = KT_OFF + SPLIT_GATHER_KEYS * KS
    # Avoid a second QKV weight pass for S5/S6/S8 and a third for S10/S12.
    # Other shapes retain four-query tiles to bound live activation state.
    widen_qkv = native_fp4_mfma and (not with_indexer or npes == 8 or S > 6)
    SAMPLE_TILE = (8 if S == 8 else min(S, 6 if S in (5, 6, 10, 12) else 4)) if widen_qkv else min(S, 4)
    INDEX_BF16_OFF = SAMPLE_TILE * HIDDEN // 4
    INDEX_Q_BF16_OFF = S * Q_LORA // 4
    CACHE_TILE = cache_samples_per_task(S, npes, with_indexer, native_fp4_mfma)
    GROUPED_UG = grouped_ug_enabled(S, npes, native_fp4_mfma, COHORT_ROW_TILES)
    UG_X_ROW = HIDDEN // 4 + (4 if GROUPED_UG else 0)
    UG_SCALE_ROW = XQ_GROUPS + (4 if GROUPED_UG else 0)
    UG_UNION_MAX = S * TOP_K + 1
    UG_UNION_WORDS = 2 * UG_UNION_MAX + N_EXPERTS // 64 + 1 if GROUPED_UG else 0
    DN_TILE = dn_tile(S, expert_mxfp4)
    N_DN_TILES = HIDDEN // DN_TILE
    DN_UNION_ROW10 = (
        COHORT_ROW_TILES
        and not with_indexer
        and native_fp4_mfma
        and atom_experts
        and npes == 4
        and S == 10
        and expert_inter == 512
    )
    GROUPED_DN = GROUPED_UG and (S in (8, 12) or DN_UNION_ROW10)
    DN_UNION_INPUT_REUSE = DN_UNION_ROW10 and H == 16
    DN_INPUT_REUSE = COHORT_ROW_TILES and npes == 4 and with_indexer and native_fp4_mfma and H == 16
    DN_X_ROW = expert_inter // 4 + (4 if GROUPED_DN else 0)
    DN_SCALE_ROW = expert_inter // 32 + (4 if GROUPED_DN else 0)

    RED_WORDS = WAVES * 64 * 4 * (2 if GROUPED_UG and (S == 12 or COHORT_ROW_TILES) else 1)
    LDS_KEYS = max(264 if with_indexer else 0, SPLIT_KEYS, S * MOE_SLOTS + UG_UNION_WORDS)

    # TileRT lineage: use one phase-overlaid arena instead of summing every
    # stage's LDS requirement.  The Kimi kernel reuses this same fusion pattern.
    # The largest X users are the four-sample input projection, all-sample MoE FP8
    # activations, and sparse attention.  Metadata, reductions, and outputs live
    # after that common X region because they are simultaneously live in GEMVs.
    SPLIT_X_WORDS = PT_OFF + SPLIT_GATHER_KEYS * PS
    index_lds_words = LOCAL_INDEX_CAPACITY if with_indexer and (not TILED_INDEXER or indexer_cp) else 0
    INPUT_NORM_ONCE = (
        native_fp4_mfma and ((npes == 4 and S == 12) or COHORT_ROW_TILES) and not attention_ptpc and not attention_bf16
    )
    INPUT_QA_BATCH = 2 if COHORT_ROW_TILES and not with_indexer else 4
    X_WORDS = max(
        S * (INPUT_QA_BATCH * WAVES * 128 // 2 + 4) if INPUT_NORM_ONCE else 0,
        SAMPLE_TILE * HIDDEN // 2,
        S * UG_X_ROW,
        SPLIT_X_WORDS,
        index_lds_words,
        ((INDEX_BF16_OFF + SAMPLE_TILE * HIDDEN // 2) if with_indexer and attention_ptpc else 0),
        (INDEX_Q_BF16_OFF + S * Q_LORA // 2) if with_indexer and attention_ptpc else 0,
        down_x_words(S, expert_inter, expert_mxfp4, native_fp4_mfma) + (S * MOE_SLOTS * 4 if GROUPED_DN else 0),
    )
    assert not (with_indexer and indexer_quant) or X_WORDS >= max(
        INDEX_RADIX_BINS, INDEX_HEADS * 20 * INDEX_QUERY_REUSE
    )
    MISC_OFF = X_WORDS
    MISC_WORDS = max(
        8 + S * (UG_SCALE_ROW if native_fp4_mfma else XQ_BLOCKS),
        S * MOE_SLOTS * (DN_SCALE_ROW if GROUPED_DN else expert_inter // (32 if native_fp4_mfma else 128)),
        N_SPLIT + 2 + (4 if attention_ptpc else 0),
    )
    KEYS_OFF = MISC_OFF + MISC_WORDS
    global_ug_keys = (
        total_samples * MOE_SLOTS
        + 3 * (N_EXPERTS + 1)
        + N_EXPERTS // 64
        + 1
        + total_samples
        + total_samples * MOE_SLOTS
    )
    global_ug_words = (
        KEYS_OFF + global_ug_keys + total_samples * MOE_SLOTS + RED_WORDS + max(S * ROW_TILE, S * 2 * UG_TILE, UV_TILE)
    )
    # Global route metadata shares the same finite LDS allocation as staging.
    # Retain the cohort-local UG body whenever that live arena would exceed it.
    GLOBAL_UG = GLOBAL_UG and global_ug_words <= 32768
    if GLOBAL_UG:
        LDS_KEYS = global_ug_keys
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + (total_samples if GLOBAL_UG else S) * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    # UK writes its 128-row result straight from the reduction tile to q_lat;
    # all remaining stages need at most these compact output tiles.
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE, UV_TILE)
    # Each globally projected batch must fit its actual staging/output arena.
    # Cohorts without this capacity keep the per-cohort projection bodies.
    GLOBAL_QKV_A = GLOBAL_QKV_A and 16 * 1028 <= X_WORDS and 16 * QKV_A_TILE <= OUT_WORDS
    GLOBAL_Q_B = GLOBAL_Q_B and 16 * (Q_LORA // 2 + 4) <= X_WORDS and 16 * Q_B_TILE <= OUT_WORDS
    GLOBAL_UV = GLOBAL_UV and GLOBAL_UG and GLOBAL_QKV_A and S * (O_K // 2 + 4) <= X_WORDS
    # Apply the existing O-input row padding independently of UV weight reuse.
    O_PAD4 = GLOBAL_UV or (
        indexer_quant
        and _partition == "back"
        and GLOBAL_Q_B
        and not attention_ptpc
        and not attention_bf16
        and S * (O_K // 2 + 4) <= X_WORDS
    )
    # The all-row global route gather already joins every persistent
    # cache reader through split -> UV -> O -> router data dependencies.
    CACHE_READERS_JOINED = (
        DEFER_OFF_CACHE
        and npes == 4
        and GLOBAL_UG
        and GROUPED_UG
        and S > 1
        and D == 1
        and L == H
        and total_samples * MOE_SLOTS <= THREADS
        and 0 < N_SPLIT <= 64
        and N_SPLIT % 4 == 0
    )
    # Explicit QKV completion polls retire every normalized-input reader
    # before global UV reuses that storage. Other paths retain the grid join.
    FRONT_PAYLOAD_RETIRE = (
        COHORT_ROW_TILES
        and DEFER_OFF_CACHE
        and GLOBAL_QKV_A
        and GLOBAL_UV
        and INPUT_NORM_ONCE
        and not with_indexer
        and S <= 16
        and N_QKV_A <= THREADS
        and Q_LORA % QKV_A_TILE == 0
        and (KV_LORA + PE_DIM) % QKV_A_TILE == 0
        and N_QKV_A * QKV_A_TILE == Q_LORA + KV_LORA + PE_DIM
    )
    MIDQ_WORDS = S * MOE_SLOTS * expert_inter // 4
    MID_SCALE_WORDS = S * MOE_SLOTS * expert_inter // 32
    INPUT_NORM_OFF = BLOCKS * S * 2 * UG_TILE - S * HIDDEN // 2
    INDEX_ROT_OFF = MIDQ_WORDS + MID_SCALE_WORDS
    UV_MERGE_OFF = MIDQ_WORDS + MID_SCALE_WORDS
    if GLOBAL_UV:
        assert GLOBAL_UG and not with_indexer
        assert UV_MERGE_OFF + S * H * KV_LORA // 2 <= BLOCKS * S * 2 * UG_TILE
        assert 16 * 260 <= X_WORDS and UV_TILE == 64
    if INDEX_PRE_ROPE:
        assert INDEX_ROT_OFF + S * INDEX_Q_ROWS // 2 <= INPUT_NORM_OFF
    if INPUT_NORM_ONCE:
        assert MIDQ_WORDS + MID_SCALE_WORDS <= INPUT_NORM_OFF
    if GROUPED_UG:
        assert (MIDQ_WORDS + MID_SCALE_WORDS) * 8 <= BLOCKS * S * 2 * UG_TILE * 8

    WORK_WORDS = OUT_OFF + OUT_WORDS
    if O_PAD4:
        assert S * (O_K // 2 + 4) <= X_WORDS
    if GLOBAL_QKV_A:
        assert INPUT_NORM_ONCE and 16 * 1028 <= X_WORDS
        assert 16 * QKV_A_TILE <= OUT_WORDS
        assert HIDDEN // 128 // WAVES == 6 and THREADS == 512
    if GLOBAL_Q_B:
        assert 16 * (Q_LORA // 2 + 4) <= X_WORDS
        assert 16 * Q_B_TILE <= OUT_WORDS
        assert N_QB == BLOCKS
    assert WORK_WORDS <= 32768, "keep static LDS below the MI355X per-workgroup budget"

    DN_UNION_CANDIDATE = (
        native_fp4_mfma and atom_experts and npes == 4 and S == 12 and expert_inter == 512 and not with_indexer
    )
    # The original down partition gives each wave complete token columns when
    # its row tile divides the four K-partition waves. The existing union
    # contribution arena can then replay each token's slot/K additions exactly.
    aligned_union_prefix = S * MOE_SLOTS * (132 + 20 + 1 + 1) + 2 * (S * TOP_K + 1) + 5
    aligned_union_words = ((aligned_union_prefix + 3) // 4 * 4) + S * MOE_SLOTS * 4 * 32 + (257 * S + 3) // 4
    DN_UNION_ALIGNED = (
        COHORT_ROW_TILES
        and GROUPED_DN
        and (not with_indexer or _partition == "back")
        and native_fp4_mfma
        and atom_experts
        and npes == 4
        and H == 16
        and expert_inter == 512
        and DN_TILE == 32
        and S % (WAVES // 2) == 0
        and aligned_union_words <= WORK_WORDS
    )
    # Redistribute a partial final round using complete 16-row MFMA groups.
    # Both subtiles fit one worker round; full rounds retain their 32-row tile.
    DN_TAIL_TASKS = N_DN_TILES % G
    DN_FULL_TASKS = N_DN_TILES - DN_TAIL_TASKS
    DN_SPLIT_UNION_TAIL = (
        DN_UNION_ALIGNED and N_DN_TILES > G and DN_TILE == 32 and 0 < DN_TAIL_TASKS and 2 * DN_TAIL_TASKS <= G
    )
    DN_WORK_ITEMS = N_DN_TILES + (DN_TAIL_TASKS if DN_SPLIT_UNION_TAIL else 0)
    DU_ROWS = S if DN_UNION_ALIGNED else (10 if DN_UNION_ROW10 else 12)
    DU_SLOTS, DU_XROW, DU_SCROW, DU_UMAX = 9, 132, 20, DU_ROWS * 8 + 1
    DU_ROUTE_WORDS = DU_ROWS * DU_SLOTS
    DU_KEY_WORDS = DU_ROUTE_WORDS + 2 * DU_UMAX + 5
    DU_MSOFF = DU_ROUTE_WORDS * DU_XROW
    DU_KEYOFF = DU_MSOFF + DU_ROUTE_WORDS * DU_SCROW
    DU_DWOFF = DU_KEYOFF + DU_KEY_WORDS
    DU_CROFF = DU_DWOFF + DU_ROUTE_WORDS
    DU_CWORDS = DU_ROUTE_WORDS * 4 * 32
    # Contribution storage does not change slot/K ownership or task geometry.
    # Vectorize any selected union body whose padded arena and LUT fit.
    DU_FRAGMENT_WORDS = (DU_CROFF + 3) // 4 * 4 + DU_CWORDS + (257 * DU_ROWS + 3) // 4
    DN_CONTIG_CONTRIBUTIONS = (
        DN_UNION_CANDIDATE or DN_UNION_ROW10 or DN_UNION_ALIGNED
    ) and DU_FRAGMENT_WORDS <= WORK_WORDS
    if DN_CONTIG_CONTRIBUTIONS:
        # The work array is 16-byte aligned; each fragment starts at a float4.
        DU_CROFF = (DU_CROFF + 3) // 4 * 4
    if DN_UNION_CANDIDATE or DN_UNION_ROW10 or DN_UNION_ALIGNED:
        assert GROUPED_DN
        assert WORK_WORDS * 4 >= (DU_CROFF + DU_CWORDS) * 4 + 257 * DU_ROWS

    # A pair of existing cohorts shares expert weights across up to 16 tokens.
    # Keep all partials in LDS and preserve each original token's slot/K/wave
    # accumulation. The compact FP4 input makes this live set fit 128 KiB.
    DN_PAIR_ROWS = 2 * S
    DN_PAIR_G = 2 * G
    DP_ROUTE_WORDS = DN_PAIR_ROWS * MOE_SLOTS
    DP_UMAX = DN_PAIR_ROWS * TOP_K + 1
    DP_XROW, DP_SCROW, DP_SLOTS = 68, 20, 9
    DP_KEY_WORDS = DP_ROUTE_WORDS + 2 * DP_UMAX + 5
    DP_MSOFF = DP_ROUTE_WORDS * DP_XROW
    DP_KEYOFF = DP_MSOFF + DP_ROUTE_WORDS * DP_SCROW
    DP_DWOFF = DP_KEYOFF + DP_KEY_WORDS
    DP_CROFF = (DP_DWOFF + DP_ROUTE_WORDS + 3) // 4 * 4
    DP_CWORDS = DP_ROUTE_WORDS * 4 * 32
    DP_WORDS = DP_CROFF + DP_CWORDS + ((N_EXPERTS + 1) * DN_PAIR_ROWS + 3) // 4
    DN_COHORT_PAIR = (
        DN_UNION_ALIGNED
        and activation_fp4
        and GLOBAL_UG
        and (CACHE_READERS_JOINED or _partition == "back")
        and ROW_TILES % 2 == 0
        and DN_PAIR_ROWS <= 16
        and DP_WORDS <= 32768
        and OUT_OFF + DN_PAIR_ROWS * DN_TILE <= 32768
    )
    DN_PAIR_TAIL = N_DN_TILES % DN_PAIR_G
    DN_PAIR_FULL = N_DN_TILES - DN_PAIR_TAIL
    DN_PAIR_SPLIT = N_DN_TILES > DN_PAIR_G and 0 < DN_PAIR_TAIL and 2 * DN_PAIR_TAIL <= DN_PAIR_G
    DN_PAIR_WORK = N_DN_TILES + (DN_PAIR_TAIL if DN_PAIR_SPLIT else 0)
    if DN_COHORT_PAIR:
        WORK_WORDS = max(WORK_WORDS, DP_WORDS, OUT_OFF + DN_PAIR_ROWS * DN_TILE)
        assert WORK_WORDS <= 32768

    base = {}
    # CTA placement: split before uk, so every split tile lands on a CTA freed by
    # qkv_a (uk shares the q_b CTAs it waits on anyway)
    tasks = dict(stages)
    if GLOBAL_UG:
        # Each routed expert contributes its first chunk, then at most one extra
        # chunk per S additional routes. Include every shared-expert row chunk.
        global_chunks_max = global_ug_chunks(total_samples, S)
        global_rounds_max = (global_chunks_max * (expert_inter // GLOBAL_UG_WIDTH) + BLOCKS - 1) // BLOCKS
        assert global_chunks_max <= total_samples * MOE_SLOTS
        assert global_rounds_max * G * (GLOBAL_UG_WIDTH // 8) <= tasks["ug"]
    acc = 0
    for name in (
        "qkv_a",
        "q_norm",
        "cache",
        "q_b",
        "split",
        "uk",
        "uv",
        "o",
        "router",
        "ug",
        "down",
    ):
        base[name] = acc % G
        # Keep later stages at their tuned CTA placements when cache tasks split.
        # Execution counts reflect the 16-head body, but its first experiment
        # reserves the original split span to keep every later CTA placement.
        acc += 1 if name == "cache" else (S * (H // WAVES) * N_SPLIT if name == "split" else tasks[name])
    # Qualified placement bucket: native ATOM TP4, C2/S6, local indexer at 4K capacity.
    if (
        SPLIT_ALL_HEADS
        and with_indexer
        and not indexer_cp
        and index_request_width == 6
        and index_max_seq == 4096
        and expert_inter == 512
        and atom_experts
        and use_atom_kv_cache
    ):
        base["uv"] = (base["uv"] + G // 2) % G
        base["o"] = (base["o"] + G // 2) % G
        base["router"] = (base["router"] + G // 2) % G
        base["ug"] = (base["ug"] + G // 2) % G
        base["down"] = (base["down"] + G // 2) % G
    if with_indexer:
        # TP8 has index-Q tiles beyond q_b's row space.  TP4 produces the full
        # index-Q row space alongside its q_b tile on every CTA.
        base["index_q"] = (base["q_b"] + N_QB) % G
        base["index_score"] = 101
        base["index_select"] = 100
        if TILED_INDEXER:
            for digit in range(INDEX_RADIX_DIGITS):
                base[f"index_hist_{digit}"] = 101
                base[f"index_prefix_{digit}"] = 100
            base["index_count"] = 101
            base["index_offsets"] = 100
            base["index_emit"] = 101

    index_grid_scale = 2 if ROW_TILES == 1 or COHORT_ROW_TILES else 1
    # Use one CTA per CU when the complete resident selector role set fits.
    # Keep the doubled cohort grid for larger role sets. This bounds competing
    # score producers and polling consumers without changing the CP work.
    if (
        ROW_TILES == 1
        and indexer_quant
        and indexer_cp
        and CP_RANKS == 4
        and TILED_INDEXER
        and S * (CP_PARTS + 1) <= G
        and INDEX_RADIX_BINS + INDEX_CP_TILE
        <= max(INDEX_RADIX_BINS, INDEX_HEADS * 20 * INDEX_QUERY_REUSE, LOCAL_INDEX_CAPACITY)
    ):
        index_grid_scale = 1
    # The widened short selector needs more than half a CU's LDS.
    # Keep its peer-waiting grid resident; long-path v102 geometry is unchanged.
    if indexer_quant and not TILED_INDEXER and LOCAL_INDEX_CAPACITY > INDEX_LOCAL_LDS_CAPACITY:
        index_grid_scale = 1
    if _partition == "index":
        G *= index_grid_scale
        X_WORDS = max(INDEX_RADIX_BINS, INDEX_HEADS * 20 * INDEX_QUERY_REUSE, LOCAL_INDEX_CAPACITY)
        KEYS_OFF = X_WORDS
        LDS_KEYS = max(264, INDEX_HEADS * INDEX_QUERY_REUSE)
        RED_OFF = KEYS_OFF + LDS_KEYS
        WORK_WORDS = RED_OFF + max(topk, 256)
        # Unused phase views have no accesses in this specialization.
        MISC_OFF = DNW_OFF = OUT_OFF = KT_OFF = PT_OFF = 0

    # Finite, per-invocation selector roles. Keep the original selector for
    # geometries which cannot assign one producer per part plus row coordinators.
    RESIDENT_INDEX_SELECT = (
        _partition == "index"
        and indexer_quant
        and indexer_cp
        and CP_RANKS == 4
        and TILED_INDEXER
        and S * CP_PARTS + S <= G
        and INDEX_RADIX_BINS + INDEX_CP_TILE <= X_WORDS
    )

    return KernelPlan(
        WORK_WORDS=WORK_WORDS,
        index_grid_scale=index_grid_scale,
        MISC_OFF=MISC_OFF,
        RED_OFF=RED_OFF,
        OUT_OFF=OUT_OFF,
        CACHE_READERS_JOINED=CACHE_READERS_JOINED,
        CACHE_TILE=CACHE_TILE,
        COHORT_ROW_TILES=COHORT_ROW_TILES,
        CP_PARTS=CP_PARTS,
        CP_RANKS=CP_RANKS,
        D=D,
        DEFER_OFF_CACHE=DEFER_OFF_CACHE,
        DNW_OFF=DNW_OFF,
        DN_COHORT_PAIR=DN_COHORT_PAIR,
        DN_CONTIG_CONTRIBUTIONS=DN_CONTIG_CONTRIBUTIONS,
        DN_FULL_TASKS=DN_FULL_TASKS,
        DN_INPUT_REUSE=DN_INPUT_REUSE,
        DN_PAIR_FULL=DN_PAIR_FULL,
        DN_PAIR_G=DN_PAIR_G,
        DN_PAIR_ROWS=DN_PAIR_ROWS,
        DN_PAIR_SPLIT=DN_PAIR_SPLIT,
        DN_PAIR_WORK=DN_PAIR_WORK,
        DN_SCALE_ROW=DN_SCALE_ROW,
        DN_SPLIT_UNION_TAIL=DN_SPLIT_UNION_TAIL,
        DN_TILE=DN_TILE,
        DN_UNION_ALIGNED=DN_UNION_ALIGNED,
        DN_UNION_CANDIDATE=DN_UNION_CANDIDATE,
        DN_UNION_INPUT_REUSE=DN_UNION_INPUT_REUSE,
        DN_UNION_ROW10=DN_UNION_ROW10,
        DN_WORK_ITEMS=DN_WORK_ITEMS,
        DN_X_ROW=DN_X_ROW,
        DP_CROFF=DP_CROFF,
        DP_CWORDS=DP_CWORDS,
        DP_DWOFF=DP_DWOFF,
        DP_KEYOFF=DP_KEYOFF,
        DP_MSOFF=DP_MSOFF,
        DP_ROUTE_WORDS=DP_ROUTE_WORDS,
        DP_SCROW=DP_SCROW,
        DP_SLOTS=DP_SLOTS,
        DP_UMAX=DP_UMAX,
        DP_XROW=DP_XROW,
        DU_CROFF=DU_CROFF,
        DU_CWORDS=DU_CWORDS,
        DU_DWOFF=DU_DWOFF,
        DU_KEYOFF=DU_KEYOFF,
        DU_KEY_WORDS=DU_KEY_WORDS,
        DU_MSOFF=DU_MSOFF,
        DU_ROUTE_WORDS=DU_ROUTE_WORDS,
        DU_SCROW=DU_SCROW,
        DU_SLOTS=DU_SLOTS,
        DU_UMAX=DU_UMAX,
        DU_XROW=DU_XROW,
        FRONT_PAYLOAD_RETIRE=FRONT_PAYLOAD_RETIRE,
        G=G,
        GLOBAL_QKV_A=GLOBAL_QKV_A,
        GLOBAL_Q_B=GLOBAL_Q_B,
        GLOBAL_UG=GLOBAL_UG,
        GLOBAL_UG_WIDTH=GLOBAL_UG_WIDTH,
        GLOBAL_UV=GLOBAL_UV,
        GROUPED_DN=GROUPED_DN,
        GROUPED_UG=GROUPED_UG,
        H=H,
        INDEX_BF16_OFF=INDEX_BF16_OFF,
        INDEX_CAPACITY=INDEX_CAPACITY,
        INDEX_CP_TILE=INDEX_CP_TILE,
        INDEX_FP4_BYTES=INDEX_FP4_BYTES,
        INDEX_PRE_ROPE=INDEX_PRE_ROPE,
        INDEX_QUERY_REUSE=INDEX_QUERY_REUSE,
        INDEX_Q_BF16_OFF=INDEX_Q_BF16_OFF,
        INDEX_RADIX_BINS=INDEX_RADIX_BINS,
        INDEX_RADIX_DIGITS=INDEX_RADIX_DIGITS,
        INDEX_ROT_OFF=INDEX_ROT_OFF,
        INDEX_SCORE_REUSE=INDEX_SCORE_REUSE,
        INPUT_NORM_OFF=INPUT_NORM_OFF,
        INPUT_NORM_ONCE=INPUT_NORM_ONCE,
        INPUT_QA_BATCH=INPUT_QA_BATCH,
        I_PER_SLOT=I_PER_SLOT,
        KEYS_OFF=KEYS_OFF,
        KS=KS,
        KT_OFF=KT_OFF,
        L=L,
        LOCAL_INDEX_CAPACITY=LOCAL_INDEX_CAPACITY,
        MIDQ_WORDS=MIDQ_WORDS,
        MID_SCALE_WORDS=MID_SCALE_WORDS,
        N_DN_TILES=N_DN_TILES,
        N_QB=N_QB,
        N_SPLIT=N_SPLIT,
        N_SPLIT_TASKS=N_SPLIT_TASKS,
        N_UG=N_UG,
        N_UK=N_UK,
        N_UV=N_UV,
        O_K=O_K,
        O_PAD4=O_PAD4,
        PS=PS,
        PT_OFF=PT_OFF,
        QB_PER_HEAD=QB_PER_HEAD,
        QK_DIM=QK_DIM,
        QS=QS,
        RED_WORDS=RED_WORDS,
        RESIDENT_INDEX_SELECT=RESIDENT_INDEX_SELECT,
        ROW_TILES=ROW_TILES,
        S=S,
        SAMPLE_TILE=SAMPLE_TILE,
        SC=SC,
        SHARD_INDEX_Q=SHARD_INDEX_Q,
        SPLIT_ALL_HEADS=SPLIT_ALL_HEADS,
        SPLIT_GATHER_KEYS=SPLIT_GATHER_KEYS,
        SPLIT_KEYS=SPLIT_KEYS,
        SPLIT_TASK_PAIR=SPLIT_TASK_PAIR,
        SY=SY,
        TILED_INDEXER=TILED_INDEXER,
        TILE_PEER_BYTES=TILE_PEER_BYTES,
        TILE_SCRATCH_BYTES=TILE_SCRATCH_BYTES,
        UG_SCALE_ROW=UG_SCALE_ROW,
        UG_TASK_ROUNDS=UG_TASK_ROUNDS,
        UG_UNION_MAX=UG_UNION_MAX,
        UG_X_ROW=UG_X_ROW,
        UK_PER_HEAD=UK_PER_HEAD,
        UV_MERGE_OFF=UV_MERGE_OFF,
        UV_REUSE=UV_REUSE,
        W=W,
        partition=_partition,
        activation_fp4=activation_fp4,
        atom_experts=atom_experts,
        attention_bf16=attention_bf16,
        attention_k_chunks_per_unit=attention_k_chunks_per_unit,
        attention_ptpc=attention_ptpc,
        base=base,
        cache_fp8=cache_fp8,
        dcp_size=dcp_size,
        expert_inter=expert_inter,
        expert_mxfp4=expert_mxfp4,
        index_block_table_stride=index_block_table_stride,
        index_page_size=index_page_size,
        index_request_width=index_request_width,
        indexer_cp=indexer_cp,
        indexer_quant=indexer_quant,
        native_fp4_mfma=native_fp4_mfma,
        npes=npes,
        paired_ug_rows=paired_ug_rows,
        rope_dtype=rope_dtype,
        scale=scale,
        topk=topk,
        total_samples=total_samples,
        use_atom_kv_cache=use_atom_kv_cache,
        uv_scale_rows=uv_scale_rows,
        with_indexer=with_indexer,
    )
