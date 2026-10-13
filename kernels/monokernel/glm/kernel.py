# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GLM-5 indexed decode MonoKernel: one persistent launch per TP rank.

One launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs the
whole decoder-layer body for this rank's TP shard.  The indexed path adds the
selection refresh to the same resident grid instead of consuming externally
prepared sparse indices::

    input RMSNorm -> q_a / kv_a projection -> q_a RMSNorm -> q_b (+RoPE)
      -> KV RMSNorm / k_pe RoPE -> KV/PE cache publish
      -> index K/Q/W projection -> index K norm/RoPE/cache
      -> index score -> exact radix top-2048
      -> absorbed q (W_UK) -> sparse MLA split softmax -> merge -> W_UV -> W_o
      -> attention TP peer reduce + residual                      (sym_attn)
      -> post-attention RMSNorm -> router sigmoid + activation FP8 quant
      -> top-8 -> 1 shared + 8 routed expert up/gate/SiLU
      -> mid FP8 quant -> expert down + route weighting
      -> MoE TP peer reduce + residual -> x_out                   (sym_ffn)

Scheduling: every stage is a list of tasks, and each resident CTA walks its
assigned tasks in stage order. Native launches above twelve rows use separate
scratch and peer arenas per tile. The general path executes tiles sequentially;
a tagged grid barrier completes each tile's cache writes before the next tile.

The native forty-row TP4/TP8 path assigns one ten-row tile to each of four
cohorts of 64 CTAs. Task ``t`` belongs to cohort-local CTA ``(stage_base + t) % 64``.
All 256 CTAs finish their front-stage tasks and join one tagged grid barrier.
The indexed path at 4K capacity publishes current cache rows at this boundary;
causal index selection then permits every cohort to read the cache.

External sparse indices can reference slots belonging to later tiles. Their
path preserves the persistent cache until every reader finishes, selecting the
latest incoming cache row visible at each original ten-row tile boundary from
tagged scratch. A second grid barrier precedes final cache stores; the last
incoming writer owns each physical slot, including duplicate slots. This keeps
the sequential cache semantics while the four tiles compute concurrently.

Indexed forty-row cohorts normalize input once per row. TP8 also uses the
existing expert-union gate/up body and publishes quantized mid values once.
TP4 forty-row gate/up compacts selected tokens across all cohorts and
distributes expert/token-chunk/output tasks over the resident grid. Each task
reuses one activation tile for64 indexed or128 external-index outputs, preserving every original
eight-output MFMA and reduction. Other tile shapes retain their schedules. Both
schedules require all
256 CTAs to be co-resident and preserve each task's original arithmetic.

Mailboxes are *tagged pairs*: every 32-bit value a task hands to another CTA
(or GPU) is stored next to this launch's epoch tag, ``(value, tag)``, with
device- (``sc1``) or system-coherent (``sc0 sc1``) 8 / 16-byte stores.  A
consumer polls the payload itself until the tags match, so a hand-off costs
one memory round trip: no store drain, no separate flag, no second load.

GEMVs run on the matrix cores: weights are host-packed (``pack_fp8`` /
``pack_bf16``) so one wave loads 16 rows x 64 k as one contiguous 1 KB, FP8 is
widened exactly to bf16 and fed to ``mfma_f32_16x16x32_bf16`` with the samples
as the N dimension.  Each 64-k chunk's partial is scaled by its f32 block
scale (times any activation scale / route weight) into the accumulator, so the
math is exact block-scaled FP8 on bf16 activations.  Weight loads that do not
depend on upstream results are issued before the task waits for its inputs.

Cross-GPU: the general path pushes tagged partials to every peer, which sums
in rank order. TP4/T12 native ATOM FFN without an indexer uses a tile owner:
BF16 partials go to that owner, whose original-order FP32 sum is broadcast.
Each rank adds its local residual before BF16 output; both paths preserve the
same numerical order and epoch protocol.

TileRT shared/reuse MonoKernel reference (fusion-boundary comparison):
https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import (
    EPS,
    FP8_MAX,
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    ROUTE_SCALE,
    SCALE_BM,
    SHARED_EXPERT,
    SOFTMAX_SCALE,
    TOP_K,
    V_DIM,
    AttentionWeight,
    KvCacheLayout,
    as_kv_cache_layout,
)
from kernels.monokernel.glm.layout import (
    BLOCKS,
    DCP_SUMMARY_PAIRS,
    INDEX_CP_TILE,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_KEYS_PER_TASK,
    INDEX_LOCAL_LDS_CAPACITY,
    INDEX_Q_ROWS,
    INDEX_RADIX_BINS,
    INDEX_RADIX_DIGITS,
    INDEX_TILE,
    N_QKV_A,
    N_ROUTER,
    N_ROW_TILES,
    NATIVE_UG_TILE,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
    XQ_BLOCKS,
    XQ_GROUPS,
    XQ_WAVES,
    cache_samples_per_task,
    dcp_local_uv_tile,
    dcp_summary_index,
    dcp_uv_owner,
    dn_tile,
    down_prefetch_batch,
    down_x_words,
    fp8_kv_upper_pair_lane,
    fp8_pe_upper_pair_lane,
    grouped_ug_enabled,
    index_cp_capacity,
    index_cp_partitions,
    index_cp_world,
    indexer_uses_tiles,
    layout,
    row_tile_size,
    sample_wave_batches,
    sparse_heads_per_task,
    sparse_keys_per_task,
    split_acc_head,
    split_score_column,
    stage_tasks,
    ug_split,
)
from kernels.monokernel.layout import (
    CM_DEV,
    CM_SYS,
    LAYER_SLOTS,
    NEG,
    POLL_MAX,
    THREADS,
    TL_COLS,
    atom_mxfp4_scale_index,
)
from kernels.monokernel.ops import (
    bpermute_i32,
    div_rn,
    f8_word,
    fp8_pack4,
    mem_realtime,
    read_lane_i32,
    spin_pause,
    wave_umax_dpp,
    write_lane_i32,
)
from kernels.monokernel.ops import exp as _exp
from kernels.monokernel.ops import fp8_roundtrip as _fp8_roundtrip
from kernels.monokernel.ops import fp8_to_bf16x8 as _fp8_to_bf16x8
from kernels.monokernel.ops import mxfp4_to_bf16x8 as _mxfp4_to_bf16x8
from kernels.monokernel.ops import rcp as _rcp
from kernels.monokernel.ops import rsq as _rsq
from kernels.monokernel.ops import rsrc as _rsrc
from kernels.monokernel.ops import uniform as _uniform
from kernels.monokernel.ops import uniform_f32 as _uniform_f32
from kernels.monokernel.ops import xred as _xred
from kernels.monokernel.ops import xshfl as _xshfl


def build_glm5_monokernel(
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
    rope_dtype: str = "fp32",
    timeline: bool = False,
    indexer_cp: bool = False,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer.

    ``timeline=True`` records ``s_memrealtime`` (100 MHz) at the start and end of
    every task, and once its inputs have arrived, into the ``timeline`` buffer:
    int64 ``[sum(task counts), TL_COLS]`` (start, hint seen, inputs staged, compute
    done, end, then free debug marks) in ``stage_tasks`` order.
    """
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
    # The measured forty-row native path benefits from static address and
    # lifetime specialization. Keep other precision and parallel paths tiled
    # with the original dynamic loop.
    STATIC_ROW_TILES = (
        total_samples == 40
        and npes in (4, 8)
        and dcp_size == 1
        and not indexer_cp
        and attention_weight is AttentionWeight.FP8_BLOCK128
        and use_atom_kv_cache
        and not cache_fp8
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
    COHORT_ROW_TILES = STATIC_ROW_TILES and (not with_indexer or index_max_seq == 4096)
    GLOBAL_UG40 = COHORT_ROW_TILES and npes == 4 and native_fp4_mfma
    GLOBAL_UG_WIDTH = 64 if with_indexer else 128
    INDEX_PRE_ROPE = COHORT_ROW_TILES and with_indexer
    INDEX_SCORE_REUSE = 2 if COHORT_ROW_TILES else 1
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
    )
    TILE_SCRATCH_BYTES = SC["_bytes"]
    TILE_PEER_BYTES = SY["_bytes"]
    TILE_TIMELINE_TASKS = sum(
        count
        for _, count in stage_tasks(
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
        )
    )
    CP_RANKS = index_cp_world(W, S, index_max_seq) if indexer_cp else 1
    INDEX_CAPACITY = index_cp_capacity(index_max_seq, CP_RANKS) if indexer_cp else index_max_seq
    TILED_INDEXER = with_indexer and indexer_uses_tiles(index_max_seq, indexer_cp)
    LOCAL_INDEX_CAPACITY = min(index_max_seq, INDEX_LOCAL_LDS_CAPACITY)
    SHARD_INDEX_Q = indexer_cp and not TILED_INDEXER and W > 1
    CP_PARTS = index_cp_partitions(index_max_seq, CP_RANKS) if TILED_INDEXER else 0
    N_SPLIT = topk // SPLIT_KEYS
    N_SPLIT_TASKS = N_SPLIT // SPLIT_TASK_PAIR
    SPLIT_ALL_HEADS = (
        sparse_heads_per_task(S, H, W, native_fp4_mfma, COHORT_ROW_TILES and (with_indexer or npes == 4)) == 16
    )
    if DEFER_OFF_CACHE:
        assert ROW_TILES >= 3
    GLOBAL_QA40 = (
        COHORT_ROW_TILES
        and npes == 4
        and native_fp4_mfma
        and H == 16
        and not with_indexer
        and not attention_ptpc
        and not attention_bf16
    )
    GLOBAL_UV40 = GLOBAL_QA40 and D == 1 and L == H
    GLOBAL_QB40 = COHORT_ROW_TILES and npes == 4 and native_fp4_mfma and H == 16
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
    DN_UNION_ROWS40 = (
        COHORT_ROW_TILES
        and not with_indexer
        and native_fp4_mfma
        and atom_experts
        and npes == 4
        and S == 10
        and expert_inter == 512
    )
    GROUPED_DN = GROUPED_UG and (S in (8, 12) or DN_UNION_ROWS40)
    DN_INPUT_REUSE40 = COHORT_ROW_TILES and npes == 4 and with_indexer and native_fp4_mfma and H == 16
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
    MISC_OFF = X_WORDS
    MISC_WORDS = max(
        8 + S * (UG_SCALE_ROW if native_fp4_mfma else XQ_BLOCKS),
        S * MOE_SLOTS * (DN_SCALE_ROW if GROUPED_DN else expert_inter // (32 if native_fp4_mfma else 128)),
        N_SPLIT + 2 + (4 if attention_ptpc else 0),
    )
    KEYS_OFF = MISC_OFF + MISC_WORDS
    if GLOBAL_UG40:
        LDS_KEYS = (
            total_samples * MOE_SLOTS
            + 3 * (N_EXPERTS + 1)
            + N_EXPERTS // 64
            + 1
            + total_samples
            + total_samples * MOE_SLOTS
        )
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + (total_samples if GLOBAL_UG40 else S) * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    # UK writes its 128-row result straight from the reduction tile to q_lat;
    # all remaining stages need at most these compact output tiles.
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE, UV_TILE)
    MIDQ_WORDS = S * MOE_SLOTS * expert_inter // 4
    MID_SCALE_WORDS = S * MOE_SLOTS * expert_inter // 32
    INPUT_NORM_OFF = BLOCKS * S * 2 * UG_TILE - S * HIDDEN // 2
    INDEX_ROT_OFF = MIDQ_WORDS + MID_SCALE_WORDS
    UV_MERGE_OFF = MIDQ_WORDS + MID_SCALE_WORDS
    if GLOBAL_UV40:
        assert GLOBAL_UG40 and not with_indexer
        assert UV_MERGE_OFF + S * H * KV_LORA // 2 <= BLOCKS * S * 2 * UG_TILE
        assert 16 * 260 <= X_WORDS and UV_TILE == 64
    if INDEX_PRE_ROPE:
        assert INDEX_ROT_OFF + S * INDEX_Q_ROWS // 2 <= INPUT_NORM_OFF
    if INPUT_NORM_ONCE:
        assert MIDQ_WORDS + MID_SCALE_WORDS <= INPUT_NORM_OFF
    if GROUPED_UG:
        assert (MIDQ_WORDS + MID_SCALE_WORDS) * 8 <= BLOCKS * S * 2 * UG_TILE * 8

    WORK_WORDS = OUT_OFF + OUT_WORDS
    if GLOBAL_QA40:
        assert INPUT_NORM_ONCE and 16 * 1028 <= X_WORDS
        assert 16 * QKV_A_TILE <= OUT_WORDS
        assert HIDDEN // 128 // WAVES == 6 and THREADS == 512
    if GLOBAL_QB40:
        assert 16 * (Q_LORA // 2 + 4) <= X_WORDS
        assert 16 * Q_B_TILE <= OUT_WORDS
        assert N_QB == BLOCKS
    assert WORK_WORDS <= 32768, "keep static LDS below the MI355X per-workgroup budget"

    DN_UNION_CANDIDATE = (
        native_fp4_mfma and atom_experts and npes == 4 and S == 12 and expert_inter == 512 and not with_indexer
    )
    DU_ROWS = 10 if DN_UNION_ROWS40 else 12
    DU_SLOTS, DU_XROW, DU_SCROW, DU_UMAX = 9, 132, 20, DU_ROWS * 8 + 1
    DU_ROUTE_WORDS = DU_ROWS * DU_SLOTS
    DU_KEY_WORDS = DU_ROUTE_WORDS + 2 * DU_UMAX + 5
    DU_MSOFF = DU_ROUTE_WORDS * DU_XROW
    DU_KEYOFF = DU_MSOFF + DU_ROUTE_WORDS * DU_SCROW
    DU_DWOFF = DU_KEYOFF + DU_KEY_WORDS
    DU_CROFF = DU_DWOFF + DU_ROUTE_WORDS
    DU_CWORDS = DU_ROUTE_WORDS * 4 * 32
    if DN_UNION_CANDIDATE or DN_UNION_ROWS40:
        assert GROUPED_DN
        assert WORK_WORDS * 4 >= (DU_CROFF + DU_CWORDS) * 4 + 257 * DU_ROWS

    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(
        S,
        H,
        topk,
        with_indexer,
        index_max_seq,
        expert_mxfp4,
        indexer_cp=indexer_cp,
        npes=W,
        inter=expert_inter,
        native_fp4_mfma=native_fp4_mfma,
        cohort_rows=COHORT_ROW_TILES,
        uv_reuse=COHORT_ROW_TILES,
        split_task_pair=SPLIT_TASK_PAIR,
    ):
        first[name] = acc
        acc += n
    # CTA placement: split before uk, so every split tile lands on a CTA freed by
    # qkv_a (uk shares the q_b CTAs it waits on anyway)
    tasks = dict(
        stage_tasks(
            S,
            H,
            topk,
            with_indexer,
            index_max_seq,
            expert_mxfp4,
            indexer_cp=indexer_cp,
            npes=W,
            inter=expert_inter,
            native_fp4_mfma=native_fp4_mfma,
            cohort_rows=COHORT_ROW_TILES,
            uv_reuse=COHORT_ROW_TILES,
            split_task_pair=SPLIT_TASK_PAIR,
        )
    )
    if GLOBAL_UG40:
        # Each routed expert contributes its first chunk, then at most one extra
        # chunk per S additional routes. Include the shared expert's four chunks.
        global_chunks_max = N_EXPERTS + (total_samples * TOP_K - N_EXPERTS) // S + ROW_TILES
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

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, WORK_WORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def glm5_monokernel_v19_router_pair2(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        full_bid = fx.block_idx.x
        bid = full_bid % G if const_expr(COHORT_ROW_TILES) else full_bid
        lane = tid % 64
        wave = tid // 64
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Smem).peek()
        xs = lds.work.ptr
        misc = xs + MISC_OFF
        keys = fx.recast_iter(fx.Int32, xs + KEYS_OFF)
        dnw = xs + DNW_OFF
        red = xs + RED_OFF
        outs = xs + OUT_OFF
        pl = xs  # score Q is dead before split probabilities are written
        attn_keys = keys
        ktile = xs + KT_OFF  # f32-typed views holding raw bf16 pairs
        petile = xs + PT_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        full_h, full_out = h_in, x_out
        full_positions, full_slots, full_indptr = (
            positions,
            slot_mapping,
            sparse_kv_indptr,
        )
        full_scratch, full_sym, full_indices = scratch, sym, indices

        def run_row_tile(row_tile):
            row_offset = fx.Int32(row_tile) * S
            peer_offset = fx.Int64(row_tile) * TILE_PEER_BYTES
            h_in = full_h + fx.Int64(row_offset) * (HIDDEN * 2)
            x_out = full_out + fx.Int64(row_offset) * (HIDDEN * 2)
            positions = full_positions + fx.Int64(row_offset) * 8
            slot_mapping = full_slots + fx.Int64(row_offset) * 8
            sparse_kv_indptr = full_indptr + fx.Int64(row_offset) * 4
            scratch = full_scratch + fx.Int64(row_tile) * TILE_SCRATCH_BYTES
            sym = full_sym + peer_offset
            indices = full_indices
            if const_expr(not with_indexer and not use_atom_kv_cache):
                indices = indices + fx.Int64(row_offset) * (topk * 4)
            r_h = _rsrc(h_in)
            # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
            # device counter bumped once per decode step (graph friendly); ``layer``
            # makes it unique per layer within the step.
            step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
            tag = step_value * LAYER_SLOTS + layer + 1
            peer_slot = (step_value * launches_per_step + layer) & 1
            # In fused mode cur_pos carries the per-request page table.  ATOM cache
            # rows and selected attention keys both use physical page slots.
            if const_expr(with_indexer and use_atom_kv_cache):
                pos0 = fx.Int32(0)
            else:
                pos0 = _uniform(bo.buffer_load(_rsrc(cur_pos), 0, vec_width=1, dtype=T.i32)) + row_offset
            r_peers = _rsrc(peers)
            # One wave sends to one peer, so retain only that wave's destination.
            pv = fx.Vector(bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32))
            peer_dst = ((fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))) + peer_offset

            # ------------------------------------------------------------ helpers
            def ld_f32(r, i):
                return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

            def ld_bf16(r, i):
                return fx.Float32(fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16)))

            def ld_rope(r, i):
                if const_expr(rope_dtype == "bf16"):
                    return ld_bf16(r, i)
                return ld_f32(r, i)

            def row_index_bounds(s):
                begin = fx.Int32(bo.buffer_load(_rsrc(sparse_kv_indptr), s, vec_width=1, dtype=T.i32))
                end = fx.Int32(bo.buffer_load(_rsrc(sparse_kv_indptr), s + 1, vec_width=1, dtype=T.i32))
                return begin, end

            def row_active(s):
                if const_expr(use_atom_kv_cache):
                    begin, end = row_index_bounds(s)
                    return end > begin
                return True

            def row_position(s):
                if const_expr(use_atom_kv_cache):
                    position = fx.Int32(bo.buffer_load(_rsrc(positions), s * 2, vec_width=1, dtype=T.i32))
                    present = row_active(s) | (row_slot(s) >= 0)
                    return present.select(position, fx.Int32(0))
                return pos0 + s

            def row_slot(s):
                if const_expr(use_atom_kv_cache):
                    return fx.Int32(bo.buffer_load(_rsrc(slot_mapping), s * 2, vec_width=1, dtype=T.i32))
                return pos0 + s

            def row_writes_cache(s):
                return row_slot(s) >= 0

            def index_request_start(s):
                return fx.max(
                    fx.Int32(0),
                    ((row_offset + s) // index_request_width) * index_request_width - row_offset,
                )

            def index_page_slot(s, logical_pos):
                if const_expr(not use_atom_kv_cache):
                    return logical_pos
                request = (row_offset + s) // index_request_width
                page = fx.Int32(
                    bo.buffer_load(
                        _rsrc(cur_pos),
                        request * index_block_table_stride + logical_pos // index_page_size,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                return page * index_page_size + logical_pos % index_page_size

            def lds_ld(ptr, i):
                return fx.ptr_load(ptr + i)

            def lds_st(ptr, i, v):
                fx.ptr_store(v, ptr + i)

            def bf16_pair(a, b):
                """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
                return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]

            def bf16_round(a):
                return fx.Float32(fx.Float32(a).to(fx.BFloat16))

            def index_arg(i):
                """Load one uniform pointer from the compact indexer parameter table.

                Fused-indexer launches carry this table in the otherwise independent
                timeline argument, keeping the no-indexer kernel ABI identical to the
                original layer.  The indices argument similarly carries index_cache.
                """
                pv = fx.Vector(bo.buffer_load(_rsrc(timeline_buf), i * 2, vec_width=2, dtype=T.i32))
                return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))

            # ---- tagged-pair mailboxes
            # TileRT lineage: payload + launch epoch is the progress protocol for
            # resident CTAs; the helpers below are the FlyDSL/ROCm adaptation.
            def mb(name):
                return scratch + fx.Int64(SC[name])

            def cp_local_rank():
                if const_expr(indexer_cp and CP_RANKS < W):
                    return rank % CP_RANKS
                return rank

            def cp_peer(peer):
                if const_expr(not indexer_cp):
                    return sym
                if const_expr(CP_RANKS < W):
                    peer = (rank // CP_RANKS) * CP_RANKS + peer
                words = fx.Vector(bo.buffer_load(r_peers, peer * 2, vec_width=2, dtype=T.i32))
                return ((fx.Int64(_uniform(words[1])) << 32) | fx.Int64(fx.Uint32(_uniform(words[0])))) + peer_offset

            def cp_buffer(name, address):
                return address + fx.Int64(SY[name]) + fx.Int64(peer_slot) * SY[name + "_stride"]

            def put(base_addr, i, v, cm=CM_DEV):
                """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
                bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
                bo.buffer_store(
                    fx.Vector.from_elements([bits, tag], fx.Int32),
                    _rsrc(base_addr),
                    i * 2,
                    cache_modifier=cm,
                )

            def put2(base_addr, i, v0, v1, cm=CM_DEV):
                """Pairs i, i+1 (i even) in one 16-byte store."""
                vec = fx.Vector.from_elements(
                    [
                        fx.Float32(v0).bitcast(fx.Int32),
                        tag,
                        fx.Float32(v1).bitcast(fx.Int32),
                        tag,
                    ],
                    fx.Int32,
                )
                bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

            def put_bf(base_addr, i, vs, cm=CM_DEV):
                """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
                i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store.
                """
                words = []
                for j in range_constexpr(len(vs) // 2):
                    words += [
                        bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32),
                        tag,
                    ]
                bo.buffer_store(
                    fx.Vector.from_elements(words, fx.Int32),
                    _rsrc(base_addr),
                    i,
                    cache_modifier=cm,
                )

            def bf2_f32(w):
                """Packed bf16 pair word -> (f32 low, f32 high)."""
                return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)

            def _qptr(addr):
                return fx.inttoptr(
                    fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                    fx.Int64(addr),
                )

            def _ld_pair(addr, scope):
                """One (value, tag) pair as a single 64-bit relaxed atomic load: never hoisted,
                coherent at ``scope`` (agent -> sc1, system -> sc0 sc1)."""
                return fx.generic_load(
                    _qptr(addr),
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope=scope,
                )

            def poll(specs, scope="agent", batch=POLL_MAX):
                """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

                All pairs are loaded together with plain 8 / 16-byte coherent buffer loads
                (sc1 locally, sc0 sc1 for peer memory); while any tag is not this launch's
                the whole batch is re-loaded, so a batch costs one round trip after its
                last producer lands.  A side-effecting (compiler-opaque) asm statement in
                the retry loop keeps the loads from being hoisted.  Returns one list of
                Int32 value bits per spec."""
                if const_expr(len(specs) == 0):
                    return []
                if const_expr(len(specs) > batch):  # bound live registers
                    return poll(specs[:batch], scope, batch) + poll(specs[batch:], scope, batch)
                cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

                def load_all():
                    words = []
                    for b, i, n in specs:
                        w = fx.Vector(
                            bo.buffer_load(
                                _rsrc(b),
                                fx.Int32(i) * 2,
                                vec_width=2 * n,
                                dtype=T.i32,
                                cache_modifier=cm,
                            )
                        )
                        words += [w[e] for e in range(2 * n)]
                    return fx.Vector.from_elements(words, fx.Int32)

                nw = sum(2 * n for _, _, n in specs)

                def pending(v):
                    bad = v[1] != tag
                    for e in range_constexpr(3, nw, 2):
                        bad = bad | (v[e] != tag)
                    return bad

                v = load_all()
                while pending(v):
                    spin_pause()
                    v = load_all()
                outs_, e = [], 0
                for _, _, n in specs:
                    outs_.append([v[e + 2 * q] for q in range(n)])
                    e += 2 * n
                return outs_

            def hint_wait(n, addr_of, mark=None):
                """Consumers poll their payload directly (tight per-wave spins); a wave-0
                pre-poll of each producer's last pair only added a hop of latency."""
                if const_expr(mark is not None):
                    stamp(mark[0], mark[1], 1)
                gpu.barrier()

            def pre_poll(n, addr_of):
                """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
                before a large payload poll, so waiting CTAs do not flood memory."""
                if wave == 0:
                    b, i = addr_of(fx.min(lane, n - 1))
                    poll([(b, i, 1)])
                gpu.barrier()

            def get(base_addr, i):
                return poll([(base_addr, i, 1)])[0][0]

            def getf(base_addr, i):
                return get(base_addr, i).bitcast(fx.Float32)

            def getf_many(specs):
                """[(base, i)] single pairs -> list of f32."""
                return [v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])]

            def get2_many(specs):
                """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
                return [
                    (v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in poll([(b, i, 2) for b, i in specs])
                ]

            def get2(base_addr, i):
                return get2_many([(base_addr, i)])[0]

            def get_bf2_many(specs):
                """[(base, i)] packed bf16 elements i, i + 1 (i even) -> list of (f32, f32)."""
                return [bf2_f32(v[0]) for v in poll([(b, i // 2, 1) for b, i in specs])]

            # ---- wave reductions
            def wave_sum(v):
                for sh in range_constexpr(6):
                    v = _xred(v, 32 >> sh, lambda a, b: a + b)
                return v

            def wave_max(v):
                for sh in range_constexpr(6):
                    v = _xred(v, 32 >> sh, fx.max)
                return v

            def block_sums(vs):
                """Block-wide sums of several per-thread values with one LDS exchange."""
                ws = [wave_sum(v) for v in vs]
                if lane == 0:
                    for i in range_constexpr(len(vs)):
                        lds_st(red, i * WAVES + wave, ws[i])
                gpu.barrier()
                tots = []
                for i in range_constexpr(len(vs)):
                    t = lds_ld(red, i * WAVES)
                    for w in range_constexpr(1, WAVES):
                        t = t + lds_ld(red, i * WAVES + w)
                    tots.append(t)
                gpu.barrier()
                return tots

            def block_maxs(vs):
                """Block-wide maxima of several per-thread values with one LDS exchange."""
                ws = [wave_max(v) for v in vs]
                if lane == 0:
                    for i in range_constexpr(len(vs)):
                        lds_st(red, i * WAVES + wave, ws[i])
                gpu.barrier()
                tots = []
                for i in range_constexpr(len(vs)):
                    t = lds_ld(red, i * WAVES)
                    for w in range_constexpr(1, WAVES):
                        t = fx.max(t, lds_ld(red, i * WAVES + w))
                    tots.append(t)
                gpu.barrier()
                return tots

            def block_sum(v):
                w = wave_sum(v)
                if lane == 0:
                    lds_st(red, wave, w)
                gpu.barrier()
                t = lds_ld(red, 0)
                for i in range_constexpr(1, WAVES):
                    t = t + lds_ld(red, i)
                gpu.barrier()
                return t

            # ------------------------------------------------ MFMA GEMV machinery
            def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None):
                """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
                bf16 activation chunk starts at LDS word ``b_word``."""
                wv = fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc) * 64 + lane) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
                if const_expr(callable(coef)):  # factor known only after a later wait
                    return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
                if const_expr(coef is not None):
                    s = s * coef
                return ("fp8", [wv], s, b_word + (lane // 16) * 4)

            def unit_fp8x2(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef=None, scale_rows=SCALE_BM):
                """Issue both 64-k halves of one 128-k FP8 weight-scale block."""
                wv = [
                    fx.Vector(
                        bo.buffer_load(
                            w_rsrc,
                            ((rg * NKC + kc + h) * 64 + lane) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for h in range(2)
                ]
                s = ld_f32(s_rsrc, (rg * 16 // scale_rows) * (K // 128) + kc // 2)
                if const_expr(callable(coef)):
                    return ("fp8x2", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
                if const_expr(coef is not None):
                    s = s * coef
                return ("fp8x2", wv, s, b_word + (lane // 16) * 4)

            def unit_f8f8(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
                """Issue one 128-k chunk (packed 64-k chunks kc, kc + 1; kc even) of row group
                ``rg`` against the FP8 activation of LDS words ``b_word`` + [0, 32) (``f8_word``
                order); ``coef()`` = activation block scale (times route weight).  ``ln``
                = the lane whose weights are loaded (default: own lane)."""
                ln = lane if ln is None else ln
                wv = [
                    fx.Vector(
                        bo.buffer_load(
                            w_rsrc,
                            ((rg * NKC + kc + h) * 64 + ln) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for h in range(2)
                ]
                s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
                return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

            def unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
                """Issue one packed 128-K PTPC FP8 tile; output-channel scale is applied after reduction."""
                ln = lane if ln is None else ln
                wv = [
                    fx.Vector(
                        bo.buffer_load(
                            w_rsrc,
                            ((rg * NKC + kc + h) * 64 + ln) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for h in range(2)
                ]
                return ("ptpc", wv, coef, b_word + (lane // 16) * 4)

            def unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
                """Issue one packed 64-K PTPC FP8 tile."""
                ln = lane if ln is None else ln
                weight = fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                return ("ptpc64", [weight], coef, b_word + (lane // 16) * 4)

            def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
                """Issue one native packed 128-K MXFP4 tile and four E8M0 row scales."""
                ln = lane if ln is None else ln
                row = rg * 16 + ln % 16
                if const_expr(atom_experts):
                    g4 = ln // 16
                    raw = fx.Vector.from_elements(
                        [
                            fx.Int32(
                                bo.buffer_load(
                                    w_rsrc,
                                    (rg * (K // 64) + kc * 2 + sp // 2) * 128 + (sp % 2) * 64 + (row % 16) * 4 + g4,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            for sp in range_constexpr(4)
                        ],
                        fx.Int32,
                    )
                    scales = [
                        (
                            fx.Int32(
                                bo.buffer_load(
                                    s_rsrc,
                                    atom_mxfp4_scale_index(row, kc * 4 + sp, K // 32),
                                    vec_width=1,
                                    dtype=T.i8,
                                )
                            )
                            << fx.Int32(23)
                        ).bitcast(fx.Float32)
                        for sp in range_constexpr(4)
                    ]
                else:
                    raw = fx.Vector(
                        bo.buffer_load(
                            w_rsrc,
                            ((rg * (K // 128) + kc) * 64 + ln) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    packed_scale = fx.Int32(
                        bo.buffer_load(
                            s_rsrc,
                            (rg * (K // 128) + kc) * 16 + ln % 16,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    scales = [
                        ((packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
                        for sp in range_constexpr(4)
                    ]
                return ("mxfp4", (raw, scales), coef, b_word + (lane // 16) * 4)

            def unit_mxfp4_bf16(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
                fmt, weights, factor, _ = unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln)
                return ("mxfp4_bf16", weights, factor, b_word + (lane // 16) * 4)

            def unit_native_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, b_scale_word, coef=None, ln=None):
                """Issue ATOM's 16-byte/lane W4 tile directly on the scaled MFMA."""
                ln = lane if ln is None else ln
                raw = fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (rg * (K // 64) + kc * 2) * 128 + ln * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                scale = fx.Int32(
                    bo.buffer_load(
                        s_rsrc,
                        atom_mxfp4_scale_index(rg * 16 + ln % 16, kc * 4 + ln // 16, K // 32),
                        vec_width=1,
                        dtype=T.i8,
                    )
                )
                return ("native_mxfp4", (raw, scale, b_scale_word), coef, b_word)

            def lds_mxfp8(chunk_word):
                """One scaled-MFMA B operand from a row-major 128-value MXFP8 tile."""
                words = [
                    fx.Vector(
                        fx.ptr_load(
                            xs + chunk_word + 16 * half + 4 * (lane // 16),
                            result_type=v4f,
                        )
                    ).bitcast(fx.Int32)
                    for half in range(2)
                ]
                return fx.Vector.from_elements([words[half][e] for half in range(2) for e in range(4)], fx.Int32)

            def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
                ln = lane if ln is None else ln
                wv = [
                    fx.Vector(
                        bo.buffer_load(
                            w_rsrc,
                            (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for sp in range(2)
                ]
                return ("bf16", wv, None, b_word + (lane // 16) * 4)

            def unit_attention(
                w_rsrc,
                s_rsrc,
                rg,
                kc,
                NKC,
                K,
                BK,
                b_word,
                ln=None,
                scale_rows=SCALE_BM,
                coef=None,
            ):
                if const_expr(attention_bf16):
                    return unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln)
                if const_expr(attention_ptpc):
                    if const_expr(BK == 64):
                        return unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
                    return unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
                if const_expr(BK == 64):
                    return unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word)
                return unit_fp8x2(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, scale_rows=scale_rows)

            def mma_units(acc, units):
                """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
                for fmt, wv, coef, bw in units:
                    if const_expr(callable(coef)):
                        coef = coef()
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    if const_expr(fmt in ("mxfp4", "mxfp4_bf16")):
                        raw, scales = wv
                        for sp in range_constexpr(4):
                            a = _mxfp4_to_bf16x8(raw[sp], scales[sp])
                            if const_expr(fmt == "mxfp4"):
                                wh, ws = sp // 2, sp % 2
                                bv = fx.Vector(fx.ptr_load(xs + (bw + wh * 16), result_type=v4f)).bitcast(fx.Int32)
                                b = _fp8_to_bf16x8(bv[ws * 2], bv[ws * 2 + 1])
                            else:
                                b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                            c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    if const_expr(fmt == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                        a = fx.Vector.from_elements([wv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                        bv = [
                            fx.Vector(fx.ptr_load(xs + (bw + h * 16), result_type=v4f)).bitcast(fx.Int32)
                            for h in range(2)
                        ]
                        b = fx.Vector.from_elements([bv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                        one = fx.Int32(127)
                        c = fx.Vector(
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one])
                        )
                    if const_expr(fmt == "native_mxfp4"):
                        raw, weight_scale, activation_scale_word = wv
                        activation_scale = lds_ld(misc, activation_scale_word).bitcast(fx.Int32)
                        c = fx.Vector(
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.vec(4, T.f32),
                                [
                                    raw,
                                    lds_mxfp8(bw),
                                    c,
                                    4,
                                    0,
                                    0,
                                    weight_scale,
                                    0,
                                    activation_scale,
                                ],
                            )
                        )
                    if const_expr(fmt in ("ptpc", "ptpc64")):
                        halves64 = 2 if fmt == "ptpc" else 1
                        for half64 in range_constexpr(halves64):
                            a = wv[half64].bitcast(fx.Int64)
                            b = fx.Vector(fx.ptr_load(xs + (bw + half64 * 16), result_type=v4f)).bitcast(fx.Int64)
                            for half32 in range_constexpr(2):
                                c = fx.Vector(
                                    rocdl.mfma_f32_16x16x32_fp8_fp8(
                                        T.vec(4, T.f32),
                                        [a[half32], b[half32], c, 0, 0, 0],
                                    )
                                )
                    nsp = (
                        4
                        if fmt == "fp8x2"
                        else (
                            2
                            if fmt
                            not in (
                                "f8f8",
                                "ptpc",
                                "ptpc64",
                                "mxfp4",
                                "mxfp4_bf16",
                                "native_mxfp4",
                            )
                            else 0
                        )
                    )
                    for sp in range_constexpr(nsp):
                        if const_expr(fmt in ("fp8", "fp8x2")):
                            wh = sp // 2 if fmt == "fp8x2" else 0
                            ws = sp % 2 if fmt == "fp8x2" else sp
                            a = _fp8_to_bf16x8(wv[wh][ws * 2], wv[wh][ws * 2 + 1])
                        else:
                            a = wv[sp].bitcast(fx.BFloat16)
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    if const_expr(coef is None):
                        acc = [acc[e] + c[e] for e in range(4)]
                    else:
                        acc = [acc[e] + c[e] * coef for e in range(4)]
                return acc

            def run_units(make_unit, cpw, batch, pre=None):
                """Software pipelined: issue batch b+1's loads before computing batch b.
                ``pre`` = the already-issued first batch (prefetched before a wait)."""
                acc = [fx.Float32(0.0) for _ in range(4)]
                starts = list(range(0, cpw, batch))
                cur = pre if pre is not None else [make_unit(c) for c in range(0, min(batch, cpw))]
                for bi in range_constexpr(len(starts)):
                    nxt = None
                    if const_expr(bi + 1 < len(starts)):
                        n0 = starts[bi + 1]
                        nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
                    acc = mma_units(acc, cur)
                    cur = nxt
                return acc

            def reduce_rows(R, acc, emit, count=S):
                """Sum per-wave MFMA tiles; emit(row_local, local sample column, value)."""
                wpr = WAVES // R
                fx.ptr_store(
                    fx.Vector.from_elements(acc, fx.Float32),
                    red + (wave * 64 + lane) * 4,
                )
                gpu.barrier()
                n_out = R * 16 * count
                for i in range_constexpr((n_out + THREADS - 1) // THREADS):
                    t = tid + i * THREADS
                    if t < n_out:
                        rl = t % (R * 16)
                        n = t // (R * 16)
                        r = rl % 16
                        tot = fx.Float32(0.0)
                        for j in range_constexpr(wpr):
                            ww = (rl // 16) * wpr + j
                            tot = tot + lds_ld(red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4)
                        emit(rl, n, tot)

            def emit_out(stride):
                def f(rl, n, v):
                    lds_st(outs, n * stride + rl, v)

                return f

            def pick(values, index):
                value = values[0]
                for i in range_constexpr(1, len(values)):
                    value = (index == i).select(values[i], value)
                return value

            def attention_b_word(element):
                return element // 4 if const_expr(attention_ptpc) else element // 2

            def emit_ptpc(stride, row_base, scale_rsrc, x_scales):
                def f(rl, n, v):
                    lds_st(
                        outs,
                        n * stride + rl,
                        bf16_round(v * pick(x_scales, n) * ld_f32(scale_rsrc, row_base + rl)),
                    )

                return f

            def emit_ptpc_weight(stride, row_base, scale_rsrc):
                def f(rl, n, v):
                    lds_st(
                        outs,
                        n * stride + rl,
                        bf16_round(v * ld_f32(scale_rsrc, row_base + rl)),
                    )

                return f

            def emit_ptpc_scalar(stride, scale):
                def f(rl, n, v):
                    lds_st(outs, n * stride + rl, bf16_round(v * scale))

                return f

            def stage_x_rmsnorm_ptpc(ld4s, n, gamma, loaded=None, count=S):
                """RMSNorm followed by production per-token FP8 quantization."""
                per = n // (4 * THREADS)
                ks = [(tid + i * THREADS) * 4 for i in range(per)]
                gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
                sums = []
                for s in range_constexpr(count):
                    ss = fx.Float32(0.0)
                    for i in range_constexpr(per):
                        for value in vals[s * per + i]:
                            ss = ss + value * value
                    sums.append(ss)
                totals = block_sums(sums)
                normed = []
                for s in range_constexpr(count):
                    rstd = _rsq(totals[s] * (1.0 / n) + EPS)
                    for i in range_constexpr(per):
                        normed.append([vals[s * per + i][j] * rstd * gs[i][j] for j in range_constexpr(4)])
                local_max = []
                for s in range_constexpr(count):
                    amax = fx.Float32(0.0)
                    for i in range_constexpr(per):
                        for value in normed[s * per + i]:
                            amax = fx.max(amax, fmath.absf(value))
                    local_max.append(amax)
                maxima = block_maxs(local_max)
                scales = []
                for s in range_constexpr(count):
                    scale = (maxima[s] == 0.0).select(fx.Float32(1.0), maxima[s] * (1.0 / FP8_MAX))
                    scales.append(scale)
                    reciprocal = 1.0 / scale
                    for i in range_constexpr(per):
                        if const_expr(with_indexer):
                            for pair in range_constexpr(2):
                                lds_st(
                                    xs,
                                    INDEX_BF16_OFF + (s * n + ks[i]) // 2 + pair,
                                    bf16_pair(
                                        normed[s * per + i][pair * 2],
                                        normed[s * per + i][pair * 2 + 1],
                                    ),
                                )
                        values = [div_rn(value, scale, reciprocal) for value in normed[s * per + i]]
                        lds_st(
                            xs,
                            (s * n + ks[i]) // 4,
                            fp8_pack4(values[0], values[1], values[2], values[3]).bitcast(fx.Float32),
                        )
                return scales

            def stage_x_pairs_ptpc(name, count, n, src_of, group=None, retain_bf16=False):
                """Quantize packed-BF16 mailbox rows with one scale per activation group."""
                group = n if group is None else group
                groups = (n + group - 1) // group
                scales = []
                for s in range_constexpr(count):
                    for g in range_constexpr(groups):
                        width = min(group, n - g * group)
                        words = width // 4
                        per = (words + THREADS - 1) // THREADS
                        local_max = fx.Float32(0.0)
                        row_values = []
                        valids = []
                        for i in range_constexpr(per):
                            word = tid + i * THREADS
                            source_word = fx.min(word, words - 1)
                            element = s * n + g * group + source_word * 4
                            got = poll([(mb(name), src_of(element) // 2, 2)])[0]
                            first = bf2_f32(got[0])
                            second = bf2_f32(got[1])
                            values = [first[0], first[1], second[0], second[1]]
                            valid = word < words
                            row_values.append(values)
                            valids.append(valid)
                            for value in values:
                                local_max = fx.max(
                                    local_max,
                                    valid.select(fmath.absf(value), fx.Float32(0.0)),
                                )
                        maximum = block_maxs([local_max])[0]
                        scale = (maximum == 0.0).select(fx.Float32(1.0), maximum * (1.0 / FP8_MAX))
                        scales.append(scale)
                        reciprocal = 1.0 / scale
                        for i in range_constexpr(per):
                            word = tid + i * THREADS
                            if valids[i]:
                                values = row_values[i]
                                if const_expr(retain_bf16):
                                    pair_word = (s * n + g * group + word * 4) // 2
                                    lds_st(
                                        xs,
                                        INDEX_Q_BF16_OFF + pair_word,
                                        bf16_pair(values[0], values[1]),
                                    )
                                    lds_st(
                                        xs,
                                        INDEX_Q_BF16_OFF + pair_word + 1,
                                        bf16_pair(values[2], values[3]),
                                    )
                                quantized = [div_rn(value, scale, reciprocal) for value in values]
                                lds_st(
                                    xs,
                                    (s * n + g * group + word * 4) // 4,
                                    fp8_pack4(
                                        quantized[0],
                                        quantized[1],
                                        quantized[2],
                                        quantized[3],
                                    ).bitcast(fx.Float32),
                                )
                return scales

            def stage_x_rmsnorm(ld4s, n, gamma, mark=None, loaded=None, count=S):
                """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
                ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
                ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
                per = n // (4 * THREADS)
                ks = [(tid + i * THREADS) * 4 for i in range(per)]
                gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
                sss = []
                for s in range_constexpr(count):
                    ss = fx.Float32(0.0)
                    for i in range_constexpr(per):
                        for a in vals[s * per + i]:
                            ss = ss + a * a
                    sss.append(ss)
                if const_expr(mark is not None):
                    stamp(mark[0], mark[1], 6)
                rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
                if const_expr(mark is not None):
                    stamp(mark[0], mark[1], 7)
                for s in range_constexpr(count):
                    for i in range_constexpr(per):
                        a = vals[s * per + i]
                        for j in range_constexpr(2):
                            lds_st(
                                xs,
                                (s * n + ks[i]) // 2 + j,
                                bf16_pair(
                                    a[2 * j] * rstds[s] * gs[i][2 * j],
                                    a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1],
                                ),
                            )
                return rstds

            def load_x_rmsnorm(ld4s, n, gamma, count=S):
                """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
                rg_ = _rsrc(gamma)
                ks = [(tid + i * THREADS) * 4 for i in range(n // (4 * THREADS))]
                gs = []
                for k in ks:
                    g = (
                        fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32))
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    gs.append([g[j] for j in range(4)])
                return gs, ld4s([(s, k) for s in range(count) for k in ks])

            def stage_x_pairs(name, n_total, src_of, padded_row=0):
                """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
                (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements.
                """
                nq = n_total // 4
                full = nq // THREADS
                vals = poll([(mb(name), src_of((tid + i * THREADS) * 4) // 2, 2) for i in range(full)])
                for i in range_constexpr(full):
                    for j in range_constexpr(2):
                        xp_word = (tid + i * THREADS) * 2 + j
                        if const_expr(padded_row > 0):
                            xp_word = (xp_word // (padded_row // 2)) * (padded_row // 2 + 4) + xp_word % (
                                padded_row // 2
                            )
                        lds_st(xs, xp_word, vals[i][j].bitcast(fx.Float32))
                if const_expr(nq % THREADS):
                    w = tid + full * THREADS
                    if w < nq:
                        v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
                        for j in range_constexpr(2):
                            xp_word = w * 2 + j
                            if const_expr(padded_row > 0):
                                xp_word = (xp_word // (padded_row // 2)) * (padded_row // 2 + 4) + xp_word % (
                                    padded_row // 2
                                )
                            lds_st(xs, xp_word, v[j].bitcast(fx.Float32))

            def quant_scaled(a0, a1):
                """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
                amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
                nz = amax > 0.0
                qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
                inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))  # hardware rcp, no IEEE divide
                q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
                q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
                return q0, q1, qs

            def quant_mxfp8(a0, a1):
                """Per-32 E4M3 values and E8M0 scale used by ATOM/AITER W4A8."""
                amax = fx.max(fmath.absf(a0), fmath.absf(a1))
                for offset in (1, 2, 4, 8):
                    amax = _xred(amax, offset, fx.max)
                bits = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
                exponent = (bits >> 23) & fx.Int32(0xFF)
                exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) != 0).select(fx.Int32(1), fx.Int32(0))
                exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
                inverse = _rcp((exponent << 23).bitcast(fx.Float32))
                q0 = fx.min(fx.max(a0 * inverse, -FP8_MAX), FP8_MAX)
                q1 = fx.min(fx.max(a1 * inverse, -FP8_MAX), FP8_MAX)
                return q0, q1, exponent

            def quant_block(a0, a1):
                """quant_scaled, values returned as the FP8-rounded f32s."""
                q0, q1, qs = quant_scaled(a0, a1)
                d0, d1 = _fp8_roundtrip(q0, q1)
                return d0, d1, qs

            def stage_xq(samples):
                """Poll the router's packed FP8 activation + block scales of ``samples``
                (sample list, or one runtime sample) into LDS, slot 0 for one runtime sample.
                """
                nxw = HIDDEN // 4 // THREADS
                nscale = XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS
                got = poll(
                    [(mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1) for sx in samples for i in range(nxw)]
                    + [(mb("xqs"), sx * nscale + fx.min(tid, nscale - 1), 1) for sx in samples]
                )
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(nxw):
                        wd = tid + i * THREADS if const_expr(native_fp4_mfma) else f8_word((tid + i * THREADS) * 4)
                        lds_st(
                            xs,
                            j * UG_X_ROW + wd,
                            got[j * nxw + i][0].bitcast(fx.Float32),
                        )
                    if tid < nscale:
                        lds_st(
                            misc,
                            8 + j * (UG_SCALE_ROW if native_fp4_mfma else nscale) + tid,
                            got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                        )

            def st_f8(k, q0, q1):
                """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
                k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
                w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                nb = _xshfl(w, 1)
                if lane % 2 == 0:
                    lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

            def st_mxfp8(k, q0, q1):
                """Store one row-major MXFP8 pair for the scaled MFMA."""
                w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                nb = _xshfl(w, 1)
                if lane % 2 == 0:
                    lds_st(xs, k // 4, (w | (nb << 16)).bitcast(fx.Float32))

            def st_ptpc(k, q0, q1):
                """Store one wave's PTPC FP8 pairs in the AITER GEMM's linear K order."""
                word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                neighbor = _xshfl(word, 1)
                if lane % 2 == 0:
                    lds_st(xs, k // 4, (word | (neighbor << 16)).bitcast(fx.Float32))

            def load_bias():
                """This lane's 4 expert biases (issue before the scores wait)."""
                return [ld_f32(_rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

            def route_top8(s, raws=None, bs=None):
                """Top-8 of sample s (call from one whole wave, after the router scores landed).

                Preserve all FP32 bits of (sigmoid + bias). Each round reduces the score
                first, then the expert ID only among exactly equal winners. Packing the
                ID into the score's low byte can change GLM's top-k on real checkpoints.
                Candidate i of this lane is expert lane + 64 i.
                Returns (expert id, route weight = raw score / sum of
                the 8 raw scores * ROUTE_SCALE) of pick ``lane`` in score order, valid in
                lanes < TOP_K."""
                if const_expr(bs is None):
                    bs = load_bias()
                if const_expr(raws is None):
                    raws = getf_many([(mb("scores"), s * N_EXPERTS + lane + i * 64) for i in range(N_EXPERTS // 64)])
                    stamp("ug", bid, 7)
                ks = []
                for i in range_constexpr(N_EXPERTS // 64):
                    kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                    ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                    ks.append(fx.Uint32(ok))
                ids = [lane + i * 64 for i in range(N_EXPERTS // 64)]
                # sort this lane's 4 keys descending; each round then takes the wave max of
                # the lane heads and shifts the winning lane's list (0 is below every key)
                for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
                    first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (ids[a] < ids[b]))
                    ka, kb, ia, ib = ks[a], ks[b], ids[a], ids[b]
                    ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
                    ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
                ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
                ids = ids + [fx.Int32(N_EXPERTS)]
                mv = fx.Int32(0)  # lane k: the expert ID of pick k
                for k in range_constexpr(TOP_K):
                    m = wave_umax_dpp(ks[0])
                    winner = wave_umax_dpp((ks[0] == m).select(255 - ids[0], fx.Int32(0)))
                    expert = 255 - winner
                    hit = (ks[0] == m) & (ids[0] == expert)
                    ks = [hit.select(ks[i + 1], ks[i]) for i in range(4)] + [ks[4]]
                    ids = [hit.select(ids[i + 1], ids[i]) for i in range(4)] + [ids[4]]
                    mv = write_lane_i32(expert, k, mv)
                e = mv
                src = (e % 64) * 4
                got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
                raw = got[0]
                for i in range_constexpr(1, N_EXPERTS // 64):
                    raw = (e // 64 == i).select(got[i], raw)
                raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
                tot = raw
                for off in (1, 2, 4):
                    tot = _xred(tot, off, lambda a, b: a + b)
                return e, raw * (_rcp(tot) * ROUTE_SCALE)

            def ffn_owner_reduce(t, residual, out_fn, tile=DN_TILE):
                """Gather rounded partials at a tile owner, then broadcast the raw FP32 sum."""
                region_base = fx.Int64(SY["ffn"]) + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
                owner = t % W
                owner_words = fx.Vector(bo.buffer_load(r_peers, owner * 2, vec_width=2, dtype=T.i32))
                owner_addr = (
                    (fx.Int64(_uniform(owner_words[1])) << 32) | fx.Int64(fx.Uint32(_uniform(owner_words[0])))
                ) + peer_offset
                if tid < S * tile // 2:
                    s = tid // (tile // 2)
                    r = (tid % (tile // 2)) * 2
                    put_bf(
                        owner_addr + region_base,
                        (rank * S + s) * HIDDEN + t * tile + r,
                        [lds_ld(outs, s * tile + r), lds_ld(outs, s * tile + r + 1)],
                        CM_SYS,
                    )
                gpu.barrier()
                if rank == owner:
                    if tid < S * tile // 2:
                        s = tid // (tile // 2)
                        r = (tid % (tile // 2)) * 2
                        row = t * tile + r
                        specs = [(sym + region_base, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
                        got = poll(specs, "one-as")
                        parts = [bf2_f32(v[0]) for v in got]
                        t0, t1 = fx.Float32(0.0), fx.Float32(0.0)
                        for src in range_constexpr(W):
                            t0 = t0 + parts[src][0]
                            t1 = t1 + parts[src][1]
                        lds_st(outs, s * tile + r, t0)
                        lds_st(outs, s * tile + r + 1, t1)
                    gpu.barrier()
                    if wave < W:
                        pair_count = S * tile // 2
                        for batch in range_constexpr((pair_count + 63) // 64):
                            pair = lane + batch * 64
                            if pair < pair_count:
                                s = pair // (tile // 2)
                                r = (pair % (tile // 2)) * 2
                                result_base = peer_dst + region_base + fx.Int64(W * S * HIDDEN * 4)
                                put2(
                                    result_base,
                                    s * HIDDEN + t * tile + r,
                                    lds_ld(outs, s * tile + r),
                                    lds_ld(outs, s * tile + r + 1),
                                    CM_SYS,
                                )
                gpu.barrier()
                if tid < S * tile // 2:
                    s = tid // (tile // 2)
                    r = (tid % (tile // 2)) * 2
                    row = t * tile + r
                    result_base = sym + region_base + fx.Int64(W * S * HIDDEN * 4)
                    got = poll([(result_base, s * HIDDEN + row, 2)], "one-as")
                    r0, r1 = residual(s, row)
                    out_fn(
                        s,
                        row,
                        r0 + got[0][0].bitcast(fx.Float32),
                        r1 + got[0][1].bitcast(fx.Float32),
                    )

            def o_owner_reduce(t, residual, out_fn, tile=ROW_TILE):
                """Gather rounded partials at a tile owner, then broadcast the raw FP32 sum."""
                region_base = fx.Int64(SY["attn"]) + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
                owner = t % W
                owner_words = fx.Vector(bo.buffer_load(r_peers, owner * 2, vec_width=2, dtype=T.i32))
                owner_addr = (
                    (fx.Int64(_uniform(owner_words[1])) << 32) | fx.Int64(fx.Uint32(_uniform(owner_words[0])))
                ) + peer_offset
                if tid < S * tile // 2:
                    s = tid // (tile // 2)
                    r = (tid % (tile // 2)) * 2
                    put_bf(
                        owner_addr + region_base,
                        (rank * S + s) * HIDDEN + t * tile + r,
                        [lds_ld(outs, s * tile + r), lds_ld(outs, s * tile + r + 1)],
                        CM_SYS,
                    )
                gpu.barrier()
                if rank == owner:
                    if tid < S * tile // 2:
                        s = tid // (tile // 2)
                        r = (tid % (tile // 2)) * 2
                        row = t * tile + r
                        specs = [(sym + region_base, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
                        got = poll(specs, "one-as")
                        parts = [bf2_f32(v[0]) for v in got]
                        t0, t1 = fx.Float32(0.0), fx.Float32(0.0)
                        for src in range_constexpr(W):
                            t0 = t0 + parts[src][0]
                            t1 = t1 + parts[src][1]
                        lds_st(outs, s * tile + r, t0)
                        lds_st(outs, s * tile + r + 1, t1)
                    gpu.barrier()
                    if wave < W:
                        pair_count = S * tile // 2
                        for batch in range_constexpr((pair_count + 63) // 64):
                            pair = lane + batch * 64
                            if pair < pair_count:
                                s = pair // (tile // 2)
                                r = (pair % (tile // 2)) * 2
                                result_base = peer_dst + region_base + fx.Int64(W * S * HIDDEN * 4)
                                put2(
                                    result_base,
                                    s * HIDDEN + t * tile + r,
                                    lds_ld(outs, s * tile + r),
                                    lds_ld(outs, s * tile + r + 1),
                                    CM_SYS,
                                )
                gpu.barrier()
                if tid < S * tile // 2:
                    s = tid // (tile // 2)
                    r = (tid % (tile // 2)) * 2
                    row = t * tile + r
                    result_base = sym + region_base + fx.Int64(W * S * HIDDEN * 4)
                    got = poll([(result_base, s * HIDDEN + row, 2)], "one-as")
                    r0, r1 = residual(s, row)
                    out_fn(
                        s,
                        row,
                        r0 + got[0][0].bitcast(fx.Float32),
                        r1 + got[0][1].bitcast(fx.Float32),
                    )

            def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
                """Push BF16 partials to every peer, then sum them in rank order.

                One wave owns each destination, allowing the peer stores to progress
                concurrently without retaining all peer pointers in every wave."""
                region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
                if const_expr(W > 1):
                    if wave < W:
                        pair_count = S * tile // 2
                        for batch in range_constexpr((pair_count + 63) // 64):
                            pair = lane + batch * 64
                            if pair < pair_count:
                                si = pair // (tile // 2)
                                ri = (pair % (tile // 2)) * 2
                                put_bf(
                                    peer_dst + region_base,
                                    (rank * S + si) * HIDDEN + t * tile + ri,
                                    [
                                        lds_ld(outs, si * tile + ri),
                                        lds_ld(outs, si * tile + ri + 1),
                                    ],
                                    CM_SYS,
                                )
                    gpu.barrier()
                if tid < S * tile // 2:
                    s = tid // (tile // 2)
                    r = (tid % (tile // 2)) * 2
                    row = t * tile + r
                    if const_expr(callable(residual)):
                        r0, r1 = residual(s, row)
                    v0 = lds_ld(outs, s * tile + r)
                    v1 = lds_ld(outs, s * tile + r + 1)
                    if const_expr(W == 1):  # no TP peers: the sum is the local value
                        parts = [(v0, v1)]
                        got = []
                        if const_expr(not callable(residual)):
                            got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
                    else:
                        own = sym + region_base
                        specs = [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
                        if const_expr(not callable(residual)):  # packed bf16 pair
                            specs.append((residual, (s * HIDDEN + row) // 2, 1))
                        got = poll(specs, "one-as")
                        parts = [bf2_f32(v[0]) for v in got[:W]]
                        got = got[W:]
                    if const_expr(not callable(residual)):
                        r0, r1 = bf2_f32(got[0][0])
                    t0 = fx.Float32(0.0)
                    t1 = fx.Float32(0.0)
                    for src in range_constexpr(W):
                        t0 = t0 + parts[src][0]
                        t1 = t1 + parts[src][1]
                    out_fn(s, row, r0 + t0, r1 + t1)

            def dcp_merge_latent(s, t):
                if const_expr(D == 1):
                    return True
                region_base = fx.Int64(SY["dcp"]) + fx.Int64(peer_slot) * fx.Int64(SY["_dcp_part_stride"])
                summary_base = dcp_summary_index(rank, s, t, 0, S, N_UV)
                if wave < D:
                    for batch in range_constexpr((DCP_SUMMARY_PAIRS + 63) // 64):
                        item = lane + batch * 64
                        if item < DCP_SUMMARY_PAIRS:
                            value = fx.Float32(0.0)
                            if item < KV_LORA // 2:
                                value = lds_ld(xs, item)
                            elif item == KV_LORA // 2:
                                value = lds_ld(misc, N_SPLIT)
                            else:
                                value = lds_ld(misc, N_SPLIT + 1)
                            put(
                                peer_dst + region_base,
                                summary_base + item,
                                value,
                                CM_SYS,
                            )
                gpu.barrier()
                owner = dcp_uv_owner(t, L, rank)
                if owner:
                    own = sym + region_base
                    if wave == 0:
                        src = fx.min(lane, D - 1)
                        src_base = dcp_summary_index(src, s, t, 0, S, N_UV)
                        values = poll(
                            [
                                (own, src_base + KV_LORA // 2, 1),
                                (own, src_base + KV_LORA // 2 + 1, 1),
                            ],
                            "one-as",
                        )
                        valid = lane < D
                        m = valid.select(values[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                        local_sum = valid.select(values[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                        m_all = wave_max(m)
                        z = local_sum * _exp(m - m_all)
                        den = wave_sum(z)
                        if valid:
                            lds_st(
                                misc,
                                lane,
                                z * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                            )
                    gpu.barrier()
                    if tid < KV_LORA // 2:
                        parts = poll(
                            [(own, dcp_summary_index(src, s, t, tid, S, N_UV), 1) for src in range(D)],
                            "one-as",
                        )
                        o0, o1 = fx.Float32(0.0), fx.Float32(0.0)
                        for src in range_constexpr(D):
                            a0, a1 = bf2_f32(parts[src][0])
                            weight = lds_ld(misc, src)
                            o0, o1 = o0 + a0 * weight, o1 + a1 * weight
                        lds_st(xs, tid, bf16_pair(o0, o1))
                    gpu.barrier()
                return owner

            def start(name):
                return (bid + (G - base[name])) & (G - 1)

            def stamp(name, t, which, lead=0):
                if const_expr(timeline):
                    if tid == lead:
                        now = mem_realtime()
                        tl_addr = index_arg(7) if const_expr(with_indexer) else timeline_buf
                        fx.generic_store(
                            fx.inttoptr(
                                fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                                tl_addr
                                + fx.Int64((row_tile * TILE_TIMELINE_TASKS + first[name] + t) * TL_COLS + which) * 8,
                            ),
                            now,
                        )

            def n_sel(count=S):
                """This lane's local MFMA B column; inactive columns duplicate the last one."""
                return fx.min(lane % 16, count - 1)

            if const_expr(INPUT_NORM_ONCE):
                # One resident CTA per token publishes the exact normalized BF16 input.
                # The suffix is retired at the front-stage barrier; global UV may then reuse it.
                if bid < S:

                    def input_norm_load(sks):
                        result = []
                        for sn, k in sks:
                            packed = fx.Vector(
                                bo.buffer_load(
                                    r_h,
                                    (bid * HIDDEN + k) // 2,
                                    vec_width=2,
                                    dtype=T.i32,
                                )
                            )
                            values = packed.bitcast(fx.BFloat16).to(fx.Float32)
                            result.append([values[j] for j in range(4)])
                        return result

                    stage_x_rmsnorm(input_norm_load, HIDDEN, g_in, count=1)
                    gpu.barrier()
                    for input_norm_round in range_constexpr(HIDDEN // (2 * THREADS)):
                        input_norm_word = tid + input_norm_round * THREADS
                        input_norm_value = lds_ld(xs, input_norm_word).bitcast(fx.Int32)
                        put(
                            mb("ugp"),
                            INPUT_NORM_OFF + bid * (HIDDEN // 2) + input_norm_word,
                            input_norm_value,
                        )
                    gpu.barrier()

            # ================================================= 1. q_a / kv_a GEMV
            # 1 row group x 96 chunks: 8 waves split K, 12 chunks each (all prefetched)
            r_wqa, r_sqa = _rsrc(w_qkv_a), _rsrc(s_qkv_a)
            QA_NKC = HIDDEN // 64
            QA_UNITS = QA_NKC // (attention_k_chunks_per_unit * WAVES)
            if const_expr(with_indexer):
                r_wik, r_sik = _rsrc(index_arg(0)), _rsrc(index_arg(1))
                r_wiw = _rsrc(index_arg(2))
            if const_expr(GLOBAL_QA40):
                # Reuse weights across 16 global tokens while preserving each
                # original per-wave K accumulator and the eight-wave reduction.
                for qa_task in range(full_bid, ((total_samples + 15) // 16) * N_QKV_A, BLOCKS):
                    qa_task = fx.Int32(qa_task)
                    qa_first = qa_task // N_QKV_A * 16
                    qa_count = fx.min(fx.Int32(16), total_samples - qa_first)
                    t = qa_task % N_QKV_A
                    # Two task rounds fit in the existing per-cohort timeline slots.
                    qa_stamp = bid + qa_task // BLOCKS * G
                    stamp("qkv_a", qa_stamp, 0)
                    qa_acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for qa_base in range_constexpr(0, QA_UNITS, 2):

                        def qa_unit(c):
                            kc = (wave * QA_UNITS + qa_base + c) * 2
                            return unit_fp8x2(
                                r_wqa,
                                r_sqa,
                                t,
                                kc,
                                QA_NKC,
                                HIDDEN,
                                fx.min(lane % 16, qa_count - 1) * 1028 + (wave * 2 + c) * 64,
                            )

                        qa_pre = [qa_unit(c) for c in range(2)]
                        for qa_batch in range_constexpr(0, 16, 4):
                            if qa_batch < qa_count:
                                qa_specs = []
                                for qa_part in range_constexpr(4):
                                    qa_row = qa_first + qa_batch + qa_part
                                    qa_src = full_scratch + fx.Int64(qa_row // S) * TILE_SCRATCH_BYTES
                                    qa_k = (tid // 64) * QA_UNITS * 128 + qa_base * 128 + (tid % 64) * 4
                                    qa_specs.append(
                                        (qa_src + SC["ugp"], INPUT_NORM_OFF + ((qa_row % S) * HIDDEN + qa_k) // 2, 2)
                                    )
                                qa_values = poll(qa_specs)
                                for qa_part in range_constexpr(4):
                                    for qa_pair in range_constexpr(2):
                                        lds_st(
                                            xs,
                                            (qa_batch + qa_part) * 1028 + tid * 2 + qa_pair,
                                            qa_values[qa_part][qa_pair].bitcast(fx.Float32),
                                        )
                        gpu.barrier()
                        if const_expr(qa_base == 0):
                            stamp("qkv_a", qa_stamp, 2)
                        qa_acc = fx.Vector.from_elements(mma_units([qa_acc[e] for e in range(4)], qa_pre), fx.Float32)
                        gpu.barrier()
                    reduce_rows(1, [qa_acc[e] for e in range(4)], emit_out(QKV_A_TILE), count=16)
                    stamp("qkv_a", qa_stamp, 3)
                    gpu.barrier()
                    if tid < qa_count * QKV_A_TILE:
                        qa_local = tid // QKV_A_TILE
                        qa_row = qa_first + qa_local
                        qa_dst = full_scratch + fx.Int64(qa_row // S) * TILE_SCRATCH_BYTES
                        qa_output_row = t * QKV_A_TILE + tid % QKV_A_TILE
                        qa_value = lds_ld(outs, tid)
                        if qa_output_row < Q_LORA:
                            put(qa_dst + SC["q_a"], (qa_row % S) * Q_LORA + qa_output_row, qa_value)
                        else:
                            put(
                                qa_dst + SC["kv_a"],
                                (qa_row % S) * (KV_LORA + PE_DIM) + qa_output_row - Q_LORA,
                                qa_value,
                            )
                    stamp("qkv_a", qa_stamp, 4)
            elif const_expr(INPUT_NORM_ONCE):
                for t in range(start("qkv_a"), N_QKV_A, G):
                    t = fx.Int32(t)
                    stamp("qkv_a", t, 0)
                    qin_acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    qin_k_acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    qin_w_acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for qin_base in range_constexpr(0, QA_UNITS, INPUT_QA_BATCH):
                        QIN_PART = min(INPUT_QA_BATCH, QA_UNITS - qin_base)
                        QIN_K = WAVES * QIN_PART * 128

                        def qin_source(element):
                            qin_row = element // QIN_K
                            qin_wave = (element % QIN_K) // (QIN_PART * 128)
                            qin_local = element % (QIN_PART * 128)
                            return (
                                INPUT_NORM_OFF * 2
                                + qin_row * HIDDEN
                                + qin_wave * QA_UNITS * 128
                                + qin_base * 128
                                + qin_local
                            )

                        def qin_unit(c):
                            qin_kc = (wave * QA_UNITS + qin_base + c) * 2
                            return unit_fp8x2(
                                r_wqa,
                                r_sqa,
                                t,
                                qin_kc,
                                QA_NKC,
                                HIDDEN,
                                n_sel() * (QIN_K // 2 + 4) + (wave * QIN_PART + c) * 64,
                            )

                        qin_pre = [qin_unit(c) for c in range(QIN_PART)]
                        stage_x_pairs("ugp", S * QIN_K, qin_source, padded_row=QIN_K)
                        gpu.barrier()
                        if const_expr(qin_base == 0):
                            stamp("qkv_a", t, 2)
                        qin_acc = fx.Vector.from_elements(
                            mma_units([qin_acc[e] for e in range(4)], qin_pre),
                            fx.Float32,
                        )
                        if const_expr(with_indexer):
                            if t < INDEX_DIM // QKV_A_TILE:
                                qin_k_units = [
                                    unit_fp8x2(
                                        r_wik,
                                        r_sik,
                                        t,
                                        (wave * QA_UNITS + qin_base + c) * 2,
                                        QA_NKC,
                                        HIDDEN,
                                        n_sel() * (QIN_K // 2 + 4) + (wave * QIN_PART + c) * 64,
                                    )
                                    for c in range(QIN_PART)
                                ]
                                qin_k_acc = fx.Vector.from_elements(
                                    mma_units([qin_k_acc[e] for e in range(4)], qin_k_units),
                                    fx.Float32,
                                )
                                if t < INDEX_HEADS // QKV_A_TILE:
                                    qin_w_units = [
                                        unit_bf16(
                                            r_wiw,
                                            t,
                                            (wave * QA_UNITS + qin_base) * 2 + c,
                                            QA_NKC,
                                            n_sel() * (QIN_K // 2 + 4) + wave * QIN_PART * 64 + c * 32,
                                        )
                                        for c in range(QIN_PART * 2)
                                    ]
                                    qin_w_acc = fx.Vector.from_elements(
                                        mma_units(
                                            [qin_w_acc[e] for e in range(4)],
                                            qin_w_units,
                                        ),
                                        fx.Float32,
                                    )
                        # All MFMA consumers finish before the next K slice reuses xs.
                        gpu.barrier()
                    reduce_rows(1, [qin_acc[e] for e in range(4)], emit_out(QKV_A_TILE))
                    stamp("qkv_a", t, 3)
                    gpu.barrier()
                    if tid < S * QKV_A_TILE:
                        qin_sample = tid // QKV_A_TILE
                        qin_row = t * QKV_A_TILE + tid % QKV_A_TILE
                        qin_value = lds_ld(outs, tid)
                        if qin_row < Q_LORA:
                            put(mb("q_a"), qin_sample * Q_LORA + qin_row, qin_value)
                        else:
                            put(
                                mb("kv_a"),
                                qin_sample * (KV_LORA + PE_DIM) + qin_row - Q_LORA,
                                qin_value,
                            )
                    if const_expr(with_indexer):
                        if t < INDEX_DIM // QKV_A_TILE:
                            gpu.barrier()
                            reduce_rows(
                                1,
                                [qin_k_acc[e] for e in range(4)],
                                emit_out(INDEX_TILE),
                            )
                            gpu.barrier()
                            if tid < S * INDEX_TILE:
                                qin_sample = tid // INDEX_TILE
                                qin_row = t * INDEX_TILE + tid % INDEX_TILE
                                put(
                                    mb("index_k"),
                                    qin_sample * INDEX_DIM + qin_row,
                                    lds_ld(outs, tid),
                                )
                            if t < INDEX_HEADS // QKV_A_TILE:
                                gpu.barrier()
                                reduce_rows(
                                    1,
                                    [qin_w_acc[e] for e in range(4)],
                                    emit_out(INDEX_TILE),
                                )
                                gpu.barrier()
                                if tid < S * INDEX_TILE:
                                    qin_sample = tid // INDEX_TILE
                                    qin_row = t * INDEX_TILE + tid % INDEX_TILE
                                    put(
                                        mb("index_w"),
                                        qin_sample * INDEX_HEADS + qin_row,
                                        lds_ld(outs, tid),
                                    )
                    stamp("qkv_a", t, 4)
            else:
                for t in range(start("qkv_a"), N_QKV_A, G):
                    t = fx.Int32(t)
                    stamp("qkv_a", t, 0)
                    for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                        group_count = min(SAMPLE_TILE, S - sample_base)

                        def u_qa(c):
                            kc = (wave * QA_UNITS + c) * attention_k_chunks_per_unit
                            return unit_attention(
                                r_wqa,
                                r_sqa,
                                t,
                                kc,
                                QA_NKC,
                                HIDDEN,
                                128,
                                attention_b_word(n_sel(group_count) * HIDDEN + kc * 64),
                            )

                        def ld_h(sks):
                            res = []
                            for s, k in sks:
                                w = fx.Vector(
                                    bo.buffer_load(
                                        r_h,
                                        ((sample_base + s) * HIDDEN + k) // 2,
                                        vec_width=2,
                                        dtype=T.i32,
                                    )
                                )
                                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                                res.append([v[j] for j in range(4)])
                            return res

                        if const_expr(S <= SAMPLE_TILE):
                            h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in, group_count)
                            pre = [u_qa(c) for c in range(QA_UNITS)]
                            if const_expr(attention_ptpc):
                                qa_scales = stage_x_rmsnorm_ptpc(ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count)
                            else:
                                stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count)
                        else:
                            if const_expr(attention_ptpc):
                                qa_scales = stage_x_rmsnorm_ptpc(ld_h, HIDDEN, g_in, count=group_count)
                            else:
                                stage_x_rmsnorm(ld_h, HIDDEN, g_in, count=group_count)
                            pre = [u_qa(c) for c in range(QA_UNITS)]
                        gpu.barrier()
                        stamp("qkv_a", t, 2)
                        acc = run_units(u_qa, QA_UNITS, QA_UNITS, pre)
                        if const_expr(attention_ptpc):
                            reduce_rows(
                                1,
                                acc,
                                emit_ptpc(QKV_A_TILE, t * QKV_A_TILE, r_sqa, qa_scales),
                                group_count,
                            )
                        else:
                            reduce_rows(1, acc, emit_out(QKV_A_TILE), group_count)
                        stamp("qkv_a", t, 3)
                        gpu.barrier()
                        if tid < group_count * QKV_A_TILE:
                            s = sample_base + tid // QKV_A_TILE
                            row = t * QKV_A_TILE + tid % QKV_A_TILE
                            v = lds_ld(outs, tid)
                            if row < Q_LORA:
                                put(mb("q_a"), s * Q_LORA + row, v)
                            else:
                                put(mb("kv_a"), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)

                        if const_expr(with_indexer):
                            IW_CPW = QA_NKC // WAVES

                            def u_index_w(c):
                                kc = wave * IW_CPW + c
                                return unit_bf16(
                                    r_wiw,
                                    t,
                                    kc,
                                    QA_NKC,
                                    (INDEX_BF16_OFF if const_expr(attention_ptpc) else 0)
                                    + (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                                )

                            if t < INDEX_DIM // QKV_A_TILE:

                                def u_index_k(c):
                                    kc = (wave * QA_UNITS + c) * 2
                                    return unit_fp8x2(
                                        r_wik,
                                        r_sik,
                                        t,
                                        kc,
                                        QA_NKC,
                                        HIDDEN,
                                        (INDEX_BF16_OFF if const_expr(attention_ptpc) else 0)
                                        + (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                                    )

                                ik_acc = run_units(u_index_k, QA_UNITS, QA_UNITS)
                                iw_acc = [fx.Float32(0.0) for _ in range(4)]
                                if t < INDEX_HEADS // QKV_A_TILE:
                                    iw_acc = run_units(u_index_w, IW_CPW, IW_CPW)
                                reduce_rows(1, ik_acc, emit_out(INDEX_TILE), group_count)
                                gpu.barrier()
                                if tid < group_count * INDEX_TILE:
                                    s = sample_base + tid // INDEX_TILE
                                    row = t * INDEX_TILE + tid % INDEX_TILE
                                    put(
                                        mb("index_k"),
                                        s * INDEX_DIM + row,
                                        lds_ld(outs, tid),
                                    )
                                if t < INDEX_HEADS // QKV_A_TILE:
                                    reduce_rows(1, iw_acc, emit_out(INDEX_TILE), group_count)
                                    gpu.barrier()
                                    if tid < group_count * INDEX_TILE:
                                        s = sample_base + tid // INDEX_TILE
                                        row = t * INDEX_TILE + tid % INDEX_TILE
                                        put(
                                            mb("index_w"),
                                            s * INDEX_HEADS + row,
                                            lds_ld(outs, tid),
                                        )
                    stamp("qkv_a", t, 4)

            def ld_qa(sks):
                v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
                return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

            # ===================== 2. one q_a RMSNorm CTA per sample, shared downstream
            for s_norm in range(start("q_norm"), S, G):
                s_norm = fx.Int32(s_norm)
                stamp("q_norm", s_norm, 0)
                hint_wait(
                    Q_LORA // QKV_A_TILE,
                    lambda k: (
                        mb("q_a"),
                        s_norm * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1,
                    ),
                    mark=("q_norm", s_norm),
                )

                def ld_qa_one(sks):
                    return ld_qa([(s_norm, k) for _, k in sks])

                stage_x_rmsnorm(ld_qa_one, Q_LORA, g_q, count=1)
                stamp("q_norm", s_norm, 2)
                gpu.barrier()
                k = tid * 4
                w0 = lds_ld(xs, k // 2)
                w1 = lds_ld(xs, k // 2 + 1)
                a0, a1 = bf2_f32(w0.bitcast(fx.Int32))
                a2, a3 = bf2_f32(w1.bitcast(fx.Int32))
                put_bf(mb("q_an"), s_norm * Q_LORA + k, [a0, a1, a2, a3])
                stamp("q_norm", s_norm, 4)

            # ================ 3. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
            for t in range(start("cache"), S // CACHE_TILE, G):
                sample_base = fx.Int32(t) * CACHE_TILE if const_expr(CACHE_TILE < S) else fx.Int32(0)
                stamp("cache", t, 0)
                r_kv = _rsrc(kv_cache)
                r_pe = _rsrc(pe_cache)
                # gamma and the RoPE factors are issued ahead of the wait
                g = ld_bf16(_rsrc(g_kv), tid)
                tpe = tid % (PE_DIM // 2)
                cs = [
                    ld_rope(
                        _rsrc(rope_cos),
                        row_position(sample_base + s) * (PE_DIM // 2) + tpe,
                    )
                    for s in range(CACHE_TILE)
                ]
                sns = [
                    ld_rope(
                        _rsrc(rope_sin),
                        row_position(sample_base + s) * (PE_DIM // 2) + tpe,
                    )
                    for s in range(CACHE_TILE)
                ]
                hint_wait(
                    (KV_LORA + PE_DIM) // QKV_A_TILE,
                    lambda k: (
                        mb("kv_a"),
                        (sample_base + CACHE_TILE - 1) * (KV_LORA + PE_DIM) + k * QKV_A_TILE + QKV_A_TILE - 1,
                    ),
                    mark=("cache", t),
                )
                # This task's query tile shares one poll and one block reduction.
                vs = getf_many([(mb("kv_a"), (sample_base + s) * (KV_LORA + PE_DIM) + tid) for s in range(CACHE_TILE)])
                pes = get2_many(
                    [
                        (
                            mb("kv_a"),
                            (sample_base + s) * (KV_LORA + PE_DIM) + KV_LORA + (tid % (PE_DIM // 2)) * 2,
                        )
                        for s in range(CACHE_TILE)
                    ]
                )
                stamp("cache", t, 2)
                ssq = block_sums([v * v for v in vs])
                for local_s in range_constexpr(CACHE_TILE):
                    s = sample_base + local_s
                    pos = row_position(s)
                    slot = row_slot(s)
                    active = row_writes_cache(s) & (not DEFER_OFF_CACHE)
                    value = vs[local_s] * _rsq(ssq[local_s] * (1.0 / KV_LORA) + EPS) * g
                    if const_expr(cache_fp8):
                        value = bf16_round(value)
                        kvn, _ = _fp8_roundtrip(value, value)
                    else:
                        kvn = bf16_round(value)
                    if const_expr(use_atom_kv_cache):
                        if active:
                            if const_expr(cache_fp8):
                                pair = fx.Int32(
                                    rocdl.cvt_pk_fp8_f32(T.i32, kvn, _xshfl(kvn, 1), fx.Int32(0), False)
                                ) & fx.Int32(0xFFFF)
                                upper_pair = bpermute_i32(fp8_kv_upper_pair_lane(lane) * 4, pair)
                                if lane % 4 == 0:
                                    bo.buffer_store(
                                        pair | (upper_pair << 16),
                                        r_kv,
                                        slot * (QK_DIM // 4) + tid // 4,
                                    )
                            else:
                                bo.buffer_store(kvn.to(fx.BFloat16), r_kv, slot * QK_DIM + tid)
                    else:
                        bo.buffer_store(kvn.to(fx.BFloat16), r_kv, pos * KV_LORA + tid)
                    put(mb("kvnew"), s * KV_LORA + tid, kvn)
                    if tid < PE_DIM // 2:
                        x0, x1 = pes[local_s]
                        c, sn = cs[local_s], sns[local_s]
                        p0, p1 = x0 * c - x1 * sn, x0 * sn + x1 * c
                        if const_expr(cache_fp8):
                            p0, p1 = bf16_round(p0), bf16_round(p1)
                            p0, p1 = _fp8_roundtrip(p0, p1)
                        else:
                            p0, p1 = bf16_round(p0), bf16_round(p1)
                        if const_expr(use_atom_kv_cache):
                            if active:
                                if const_expr(cache_fp8):
                                    pair = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, fx.Int32(0), False)) & fx.Int32(
                                        0xFFFF
                                    )
                                    upper_pair = bpermute_i32(fp8_pe_upper_pair_lane(lane) * 4, pair)
                                    if lane % 2 == 0:
                                        bo.buffer_store(
                                            pair | (upper_pair << 16),
                                            r_pe,
                                            slot * (QK_DIM // 4) + KV_LORA // 4 + tid // 2,
                                        )
                                else:
                                    pe_offset = slot * QK_DIM + KV_LORA + tid * 2
                                    bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                                    bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                        else:
                            pe_offset = pos * PE_DIM + tid * 2
                            bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                            bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                        put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)

                if const_expr(with_indexer):
                    # Index keys use LayerNorm (not RMSNorm), interleaved RoPE on
                    # the first 64 dimensions, and BF16 cache storage. Hadamard is
                    # omitted because applying the same orthogonal transform to Q
                    # and K leaves their dot products unchanged.
                    r_gik, r_bik = _rsrc(index_arg(5)), _rsrc(index_arg(6))
                    r_index_cache = _rsrc(indices)
                    for local_s in range_constexpr(CACHE_TILE):
                        s = sample_base + local_s
                        ik = getf(mb("index_k"), s * INDEX_DIM + fx.min(tid, INDEX_DIM - 1))
                        live = tid < INDEX_DIM
                        iv = live.select(ik, fx.Float32(0.0))
                        mean = block_sum(iv) * (1.0 / INDEX_DIM)
                        centered = live.select(ik - mean, fx.Float32(0.0))
                        rstd = _rsq(block_sum(centered * centered) * (1.0 / INDEX_DIM) + 1.0e-6)
                        if tid < INDEX_DIM // 2:
                            i0 = tid * 2
                            k0, k1 = get2(mb("index_k"), s * INDEX_DIM + i0)
                            v0 = (k0 - mean) * rstd * ld_f32(r_gik, i0) + ld_f32(r_bik, i0)
                            v1 = (k1 - mean) * rstd * ld_f32(r_gik, i0 + 1) + ld_f32(r_bik, i0 + 1)
                            if tid < PE_DIM // 2:
                                c, sn = cs[local_s], sns[local_s]
                                v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c
                            bo.buffer_store(
                                fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                                r_index_cache,
                                index_page_slot(s, row_position(s)) * INDEX_DIM + i0,
                            )
                            put_bf(mb("index_k_new"), s * INDEX_DIM + i0, [v0, v1])
                        gpu.barrier()
                        if tid == 0:
                            put(mb("index_ready"), s, fx.Int32(1))
                stamp("cache", t, 4)

            # =============================================== 4. normalized q_a -> q_b (+RoPE)
            def emit_index_query(qs, qt, qr):
                values = [lds_ld(outs, qs * INDEX_TILE + qr + j) for j in range(4)]
                offset = qs * INDEX_Q_ROWS + qt * INDEX_TILE + qr
                put_bf(mb("index_q"), offset, values)
                if const_expr(INDEX_PRE_ROPE):
                    # Keep the first BF16 rounding used by the raw query mailbox.
                    rounded = [bf16_round(v) for v in values]
                    rotated = []
                    for pair in range_constexpr(2):
                        q0, q1 = rounded[pair * 2], rounded[pair * 2 + 1]
                        qk = (qt * INDEX_TILE + qr + pair * 2) % INDEX_DIM
                        if qk < PE_DIM:
                            qc = ld_rope(_rsrc(rope_cos), row_position(qs) * (PE_DIM // 2) + qk // 2)
                            qsn = ld_rope(_rsrc(rope_sin), row_position(qs) * (PE_DIM // 2) + qk // 2)
                            q0, q1 = q0 * qc - q1 * qsn, q0 * qsn + q1 * qc
                        rotated += [q0, q1]
                    # Keep rotated queries above the later mid quantization region.
                    # Its readiness tags must never alias within this epoch.
                    put_bf(mb("ugp"), 2 * INDEX_ROT_OFF + offset, rotated)

            r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
            QB_NKC = Q_LORA // 64
            QB_UNITS = QB_NKC // (attention_k_chunks_per_unit * WAVES)
            if const_expr(with_indexer):
                r_wiq, r_siq = _rsrc(index_arg(3)), _rsrc(index_arg(4))

            if const_expr(GLOBAL_QB40):
                # All output tiles cover the same global 16-row chunk. Each
                # output element retains its original eight-wave K reduction.
                for qb_chunk in range((total_samples + 15) // 16):
                    qb_chunk = fx.Int32(qb_chunk)
                    qb_first = qb_chunk * 16
                    qb_count = fx.min(fx.Int32(16), total_samples - qb_first)
                    t = full_bid
                    # Reuse the original per-cohort timeline allocation. Three
                    # task rounds occupy its first192 Q_B slots;64 stay unused.
                    qb_stamp = bid + qb_chunk * G
                    stamp("q_b", qb_stamp, 0)

                    def u_global_qb(c):
                        kc = (wave * QB_UNITS + c) * 2
                        return unit_fp8x2(
                            r_wqb,
                            r_sqb,
                            t,
                            kc,
                            QB_NKC,
                            Q_LORA,
                            fx.min(lane % 16, qb_count - 1) * (Q_LORA // 2 + 4) + kc * 32,
                        )

                    pre = [u_global_qb(c) for c in range(QB_UNITS)]
                    gpu.barrier()
                    stamp("q_b", qb_stamp, 1)
                    # Four whole rows per batch bounds live mailbox values.
                    # The last chunk stages eight rows; MFMA padding duplicates
                    # row seven and publishes no padded output.
                    for qb_batch in range_constexpr(0, 16, 4):
                        if qb_batch < qb_count:
                            qb_specs = []
                            for qb_part in range_constexpr(4):
                                qb_row = qb_first + qb_batch + qb_part
                                qb_src = full_scratch + fx.Int64(qb_row // S) * TILE_SCRATCH_BYTES
                                qb_specs.append((qb_src + SC["q_an"], ((qb_row % S) * Q_LORA + tid * 4) // 2, 2))
                            qb_values = poll(qb_specs)
                            for qb_part in range_constexpr(4):
                                for qb_pair in range_constexpr(2):
                                    lds_st(
                                        xs,
                                        (qb_batch + qb_part) * (Q_LORA // 2 + 4) + tid * 2 + qb_pair,
                                        qb_values[qb_part][qb_pair].bitcast(fx.Float32),
                                    )
                    gpu.barrier()
                    stamp("q_b", qb_stamp, 2)
                    acc = run_units(u_global_qb, QB_UNITS, QB_UNITS, pre)
                    reduce_rows(1, acc, emit_out(Q_B_TILE), count=16)
                    stamp("q_b", qb_stamp, 3)
                    gpu.barrier()
                    head = t // QB_PER_HEAD
                    hoff = (t % QB_PER_HEAD) * Q_B_TILE
                    if hoff < NOPE_DIM:
                        if tid < qb_count * Q_B_TILE // 4:
                            qb_local = tid // (Q_B_TILE // 4)
                            qb_row = qb_first + qb_local
                            qr = tid % (Q_B_TILE // 4) * 4
                            qb_dst = full_scratch + fx.Int64(qb_row // S) * TILE_SCRATCH_BYTES
                            put_bf(
                                qb_dst + SC["q_nope"],
                                ((qb_row % S) * H + head) * NOPE_DIM + hoff + qr,
                                [lds_ld(outs, qb_local * Q_B_TILE + qr + j) for j in range(4)],
                            )
                    else:
                        if tid < qb_count * Q_B_TILE // 2:
                            qb_local = tid // (Q_B_TILE // 2)
                            qb_row = qb_first + qb_local
                            qr = tid % (Q_B_TILE // 2) * 2
                            qk = hoff - NOPE_DIM + qr
                            qb_dst = full_scratch + fx.Int64(qb_row // S) * TILE_SCRATCH_BYTES
                            qpos = fx.Int32(bo.buffer_load(_rsrc(full_positions), qb_row * 2, vec_width=1, dtype=T.i32))
                            qb_begin = fx.Int32(bo.buffer_load(_rsrc(full_indptr), qb_row, vec_width=1, dtype=T.i32))
                            qb_end = fx.Int32(bo.buffer_load(_rsrc(full_indptr), qb_row + 1, vec_width=1, dtype=T.i32))
                            qb_slot = fx.Int32(bo.buffer_load(_rsrc(full_slots), qb_row * 2, vec_width=1, dtype=T.i32))
                            qpos = ((qb_end > qb_begin) | (qb_slot >= 0)).select(qpos, fx.Int32(0))
                            qcos = ld_rope(_rsrc(rope_cos), qpos * (PE_DIM // 2) + qk // 2)
                            qsin = ld_rope(_rsrc(rope_sin), qpos * (PE_DIM // 2) + qk // 2)
                            q0 = lds_ld(outs, qb_local * Q_B_TILE + qr)
                            q1 = lds_ld(outs, qb_local * Q_B_TILE + qr + 1)
                            put_bf(
                                qb_dst + SC["q_pe"],
                                ((qb_row % S) * H + head) * PE_DIM + qk,
                                [q0 * qcos - q1 * qsin, q0 * qsin + q1 * qcos],
                            )
                    stamp("q_b", qb_stamp, 4)

                    if const_expr(with_indexer):
                        # Raw and precomputed rotated index queries retain their
                        # BF16 rounding and ten-row cohort mailbox addresses.
                        gpu.barrier()
                        stamp("index_q", qb_stamp, 0)

                        def u_global_index_q(c):
                            kc = (wave * QB_UNITS + c) * 2
                            return unit_fp8x2(
                                r_wiq,
                                r_siq,
                                t,
                                kc,
                                QB_NKC,
                                Q_LORA,
                                fx.min(lane % 16, qb_count - 1) * (Q_LORA // 2 + 4) + kc * 32,
                            )

                        index_acc = run_units(u_global_index_q, QB_UNITS, QB_UNITS)
                        reduce_rows(1, index_acc, emit_out(INDEX_TILE), count=16)
                        stamp("index_q", qb_stamp, 3)
                        gpu.barrier()
                        if tid < qb_count * INDEX_TILE // 4:
                            index_local = tid // (INDEX_TILE // 4)
                            index_row = qb_first + index_local
                            index_r = tid % (INDEX_TILE // 4) * 4
                            index_dst = full_scratch + fx.Int64(index_row // S) * TILE_SCRATCH_BYTES
                            index_offset = (index_row % S) * INDEX_Q_ROWS + t * INDEX_TILE + index_r
                            index_values = [lds_ld(outs, index_local * INDEX_TILE + index_r + j) for j in range(4)]
                            put_bf(index_dst + SC["index_q"], index_offset, index_values)
                            rounded = [bf16_round(v) for v in index_values]
                            rotated = []
                            for pair in range_constexpr(2):
                                q0, q1 = rounded[pair * 2], rounded[pair * 2 + 1]
                                qk = (t * INDEX_TILE + index_r + pair * 2) % INDEX_DIM
                                if qk < PE_DIM:
                                    qp = fx.Int32(
                                        bo.buffer_load(_rsrc(full_positions), index_row * 2, vec_width=1, dtype=T.i32)
                                    )
                                    qb = fx.Int32(
                                        bo.buffer_load(_rsrc(full_indptr), index_row, vec_width=1, dtype=T.i32)
                                    )
                                    qe = fx.Int32(
                                        bo.buffer_load(_rsrc(full_indptr), index_row + 1, vec_width=1, dtype=T.i32)
                                    )
                                    qs = fx.Int32(
                                        bo.buffer_load(_rsrc(full_slots), index_row * 2, vec_width=1, dtype=T.i32)
                                    )
                                    qp = ((qe > qb) | (qs >= 0)).select(qp, fx.Int32(0))
                                    qc = ld_rope(_rsrc(rope_cos), qp * (PE_DIM // 2) + qk // 2)
                                    qsn = ld_rope(_rsrc(rope_sin), qp * (PE_DIM // 2) + qk // 2)
                                    q0, q1 = q0 * qc - q1 * qsn, q0 * qsn + q1 * qc
                                rotated += [q0, q1]
                            put_bf(index_dst + SC["ugp"], 2 * INDEX_ROT_OFF + index_offset, rotated)
                        stamp("index_q", qb_stamp, 4)
            else:
                for t in range(start("q_b"), N_QB, G):
                    t = fx.Int32(t)
                    stamp("q_b", t, 0)

                    def u_qb(c):
                        kc = (wave * QB_UNITS + c) * attention_k_chunks_per_unit
                        return unit_attention(
                            r_wqb,
                            r_sqb,
                            t,
                            kc,
                            QB_NKC,
                            Q_LORA,
                            128,
                            attention_b_word(n_sel() * Q_LORA + kc * 64),
                        )

                    pre = [u_qb(c) for c in range(QB_UNITS)]
                    hint_wait(
                        S,
                        lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2),
                        mark=("q_b", t),
                    )
                    if const_expr(attention_ptpc):
                        qb_scales = stage_x_pairs_ptpc("q_an", S, Q_LORA, lambda k: k, retain_bf16=with_indexer)
                    else:
                        stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
                    stamp("q_b", t, 2)
                    gpu.barrier()
                    acc = run_units(u_qb, QB_UNITS, QB_UNITS, pre)
                    if const_expr(attention_ptpc):
                        reduce_rows(1, acc, emit_ptpc(Q_B_TILE, t * Q_B_TILE, r_sqb, qb_scales))
                    else:
                        reduce_rows(1, acc, emit_out(Q_B_TILE))
                    stamp("q_b", t, 3)
                    gpu.barrier()
                    head = t // QB_PER_HEAD
                    hoff = (t % QB_PER_HEAD) * Q_B_TILE
                    if hoff < NOPE_DIM:
                        if tid < S * Q_B_TILE // 4:
                            s = tid // (Q_B_TILE // 4)
                            r = (tid % (Q_B_TILE // 4)) * 4
                            put_bf(
                                mb("q_nope"),
                                (s * H + head) * NOPE_DIM + hoff + r,
                                [lds_ld(outs, s * Q_B_TILE + r + j) for j in range(4)],
                            )
                    else:
                        if tid < S * Q_B_TILE // 2:
                            s = tid // (Q_B_TILE // 2)
                            pr = tid % (Q_B_TILE // 2)
                            i = hoff - NOPE_DIM + pr * 2
                            x0 = lds_ld(outs, s * Q_B_TILE + pr * 2)
                            x1 = lds_ld(outs, s * Q_B_TILE + pr * 2 + 1)
                            c = ld_rope(_rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + i // 2)
                            sn = ld_rope(_rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + i // 2)
                            put_bf(
                                mb("q_pe"),
                                (s * H + head) * PE_DIM + i,
                                [x0 * c - x1 * sn, x0 * sn + x1 * c],
                            )
                    stamp("q_b", t, 4)

                    if const_expr(with_indexer and not SHARD_INDEX_Q):
                        # Reuse this CTA's normalized q_lora tile for one index-query
                        # row tile.  Complementary CTAs compute the remaining half below.
                        gpu.barrier()
                        iq_t = t
                        stamp("index_q", iq_t, 0)

                        def u_index_q(c):
                            kc = (wave * QB_UNITS + c) * 2
                            return unit_fp8x2(
                                r_wiq,
                                r_siq,
                                iq_t,
                                kc,
                                QB_NKC,
                                Q_LORA,
                                (INDEX_Q_BF16_OFF if const_expr(attention_ptpc) else 0)
                                + (n_sel() * Q_LORA + kc * 64) // 2,
                            )

                        iq_acc = run_units(u_index_q, QB_UNITS, QB_UNITS)
                        reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                        stamp("index_q", iq_t, 3)
                        gpu.barrier()
                        if tid < S * INDEX_TILE // 4:
                            s = tid // (INDEX_TILE // 4)
                            r = (tid % (INDEX_TILE // 4)) * 4
                            emit_index_query(s, iq_t, r)
                        stamp("index_q", iq_t, 4)

            if const_expr(with_indexer):
                # The 128 CTAs without q_b work produce the other 128 index-query
                # tiles concurrently.  They reload q_a, but remove one full GEMV
                # from the q_b CTAs' serialized critical path.
                N_INDEX_Q_EXTRA = (
                    INDEX_Q_ROWS // INDEX_TILE // CP_RANKS if SHARD_INDEX_Q else INDEX_Q_ROWS // INDEX_TILE - N_QB
                )
                for tt in range(start("index_q"), N_INDEX_Q_EXTRA, G):
                    tt = fx.Int32(tt)
                    iq_t = cp_local_rank() * N_INDEX_Q_EXTRA + tt if const_expr(SHARD_INDEX_Q) else N_QB + tt
                    iq_task = tt if const_expr(SHARD_INDEX_Q) else iq_t
                    stamp("index_q", iq_task, 0)

                    def u_index_q_extra(c):
                        kc = (wave * QB_UNITS + c) * 2
                        return unit_fp8x2(
                            r_wiq,
                            r_siq,
                            iq_t,
                            kc,
                            QB_NKC,
                            Q_LORA,
                            (n_sel() * Q_LORA + kc * 64) // 2,
                        )

                    pre = [u_index_q_extra(c) for c in range(QB_UNITS)]
                    hint_wait(
                        S,
                        lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2),
                        mark=("index_q", iq_task),
                    )
                    stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
                    stamp("index_q", iq_task, 2)
                    gpu.barrier()
                    iq_acc = run_units(u_index_q_extra, QB_UNITS, QB_UNITS, pre)
                    reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                    stamp("index_q", iq_task, 3)
                    gpu.barrier()
                    if const_expr(SHARD_INDEX_Q):
                        if (wave < CP_RANKS) & (lane < S * INDEX_TILE // 4):
                            s = lane // (INDEX_TILE // 4)
                            r = (lane % (INDEX_TILE // 4)) * 4
                            put_bf(
                                cp_buffer(
                                    "index_cp_q",
                                    (cp_peer(wave) if const_expr(CP_RANKS < W) else peer_dst),
                                ),
                                s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                                [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                                CM_SYS,
                            )
                    else:
                        if tid < S * INDEX_TILE // 4:
                            s = tid // (INDEX_TILE // 4)
                            r = (tid % (INDEX_TILE // 4)) * 4
                            emit_index_query(s, iq_t, r)
                    stamp("index_q", iq_task, 4)

            # ==================================== 4. absorbed query: q_lat = W_UK^T q_nope
            # 8 row groups (128 latent rows of one head) x 3 chunks: one row group per wave
            r_wuk, r_suk = _rsrc(w_uk), _rsrc(s_uk)
            UK_NKC = NOPE_DIM // 64
            for t in range(start("uk"), N_UK, G):
                t = fx.Int32(t)
                stamp("uk", t, 0)
                head = t // UK_PER_HEAD

                def u_uk(c):
                    return unit_attention(
                        r_wuk,
                        r_suk,
                        t * WAVES + wave,
                        c,
                        UK_NKC,
                        NOPE_DIM,
                        64,
                        attention_b_word(n_sel() * NOPE_DIM + c * 64),
                        coef=(
                            (
                                lambda: pick(
                                    [uk_scales[sample * ((NOPE_DIM + 127) // 128) + c // 2] for sample in range(S)],
                                    n_sel(),
                                )
                            )
                            if const_expr(attention_ptpc)
                            else None
                        ),
                    )

                pre = [u_uk(c) for c in range(UK_NKC)]
                hint_wait(
                    NOPE_DIM // Q_B_TILE,
                    lambda k: (
                        mb("q_nope"),
                        ((S - 1) * H + head) * NOPE_DIM + k * Q_B_TILE + Q_B_TILE - 1,
                    ),
                    mark=("uk", t),
                )
                if const_expr(attention_ptpc):
                    uk_scales = stage_x_pairs_ptpc(
                        "q_nope",
                        S,
                        NOPE_DIM,
                        lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                        group=128,
                    )
                else:
                    stage_x_pairs(
                        "q_nope",
                        S * NOPE_DIM,
                        lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                    )
                stamp("uk", t, 2)
                gpu.barrier()
                acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
                fx.ptr_store(
                    fx.Vector.from_elements(acc, fx.Float32),
                    red + (wave * 64 + lane) * 4,
                )
                stamp("uk", t, 3)
                gpu.barrier()
                if tid < S * UK_TILE // 4:
                    k = tid * 4
                    s = k // UK_TILE
                    r0 = k % UK_TILE
                    vals = []
                    if const_expr(attention_ptpc):
                        ptpc_scale = ld_f32(r_suk, 0)
                    for j in range_constexpr(4):
                        r = r0 + j
                        ww = r // 16
                        value = lds_ld(red, (ww * 64 + s + 16 * ((r % 16) // 4)) * 4 + r % 4)
                        if const_expr(attention_ptpc):
                            value = bf16_round(value * ptpc_scale)
                        vals.append(value)
                    put_bf(
                        mb("q_lat"),
                        (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + r0,
                        vals,
                    )
                stamp("uk", t, 4)

            if const_expr(COHORT_ROW_TILES):
                # Every cohort finishes its front tasks before cross-tile reads.
                # Indexed mode publishes cache stores; external-index mode keeps
                # persistent cache intact and reads current versions from scratch.
                fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                gpu.barrier()
                cohort_arrivals = full_scratch + fx.Int64(ROW_TILES * TILE_SCRATCH_BYTES)
                if tid == 0:
                    put(cohort_arrivals, full_bid, fx.Int32(1))
                if full_bid == 0:
                    if tid < BLOCKS:
                        get(cohort_arrivals, tid)
                        fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
                    gpu.barrier()
                    if tid == 0:
                        put(cohort_arrivals, BLOCKS, fx.Int32(1))
                if tid == 0:
                    get(cohort_arrivals, BLOCKS)
                gpu.barrier()
                fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")

            # ====================== 4b. fused sparse index score + exact top-2048
            if const_expr(with_indexer):
                r_index_cache = _rsrc(indices)
                N_INDEX_SPLIT = INDEX_CAPACITY // INDEX_KEYS_PER_TASK

                def use_long_index(s):
                    if const_expr(indexer_cp):
                        return row_position(s) + 1 > LOCAL_INDEX_CAPACITY
                    return fx.Boolean(True)

                def use_short_index(s):
                    # A short compiled capacity cannot cross the selector boundary.
                    # Preserve the original branch-free short path with CP off.
                    if const_expr(not TILED_INDEXER):
                        return fx.Boolean(True)
                    return row_position(s) + 1 <= LOCAL_INDEX_CAPACITY

                def index_shard(s):
                    bound = row_position(s) + 1
                    if const_expr(indexer_cp):
                        stride = (
                            (bound + CP_RANKS * INDEX_KEYS_PER_TASK - 1) // (CP_RANKS * INDEX_KEYS_PER_TASK)
                        ) * INDEX_KEYS_PER_TASK
                        begin = cp_local_rank() * stride
                        return begin, fx.max(fx.Int32(0), fx.min(stride, bound - begin))
                    return fx.Int32(0), bound

                def load_index_q8(head, k):
                    words = fx.Vector.from_elements(
                        [lds_ld(xs, (head * INDEX_DIM + k) // 2 + j) for j in range(4)],
                        fx.Float32,
                    )
                    return words.bitcast(fx.BFloat16)

                for index_group in range(start("index_score"), S * (N_INDEX_SPLIT // INDEX_SCORE_REUSE), G):
                    index_group = fx.Int32(index_group)
                    s = index_group // (N_INDEX_SPLIT // INDEX_SCORE_REUSE)
                    index_tile_base = (index_group % (N_INDEX_SPLIT // INDEX_SCORE_REUSE)) * INDEX_SCORE_REUSE
                    tile = index_tile_base
                    tt = s * N_INDEX_SPLIT + tile
                    stamp("index_score", tt, 0)
                    bound = row_position(s) + 1
                    request_start = index_request_start(s)
                    first_position = row_position(request_start)
                    shard_begin, shard_size = index_shard(s)
                    if tile * INDEX_KEYS_PER_TASK < shard_size:
                        get(mb("index_ready"), s)
                        if tid < INDEX_HEADS:
                            lds_st(keys, tid, get(mb("index_w"), s * INDEX_HEADS + tid))
                        # Every key group reuses the same 32x128 query.  Stage and RoPE
                        # it once per scoring CTA instead of polling and rotating it in
                        # each of the four key-group wave pairs.
                        # Peer reads benefit from independent outstanding polls.
                        # Keep the original one-pair staging order when CP is off.
                        q_batch = (INDEX_Q_ROWS // 2) // THREADS if indexer_cp else 1
                        for batch in range_constexpr(0, (INDEX_Q_ROWS // 2) // THREADS, q_batch):
                            q_pairs = [tid + (batch + b) * THREADS for b in range(q_batch)]
                            q_source = (
                                cp_buffer("index_cp_q", sym)
                                if const_expr(SHARD_INDEX_Q)
                                else (mb("ugp") + fx.Int64(INDEX_ROT_OFF * 8) if INDEX_PRE_ROPE else mb("index_q"))
                            )
                            q_words = poll(
                                [(q_source, s * INDEX_Q_ROWS // 2 + pair, 1) for pair in q_pairs],
                                scope="system" if SHARD_INDEX_Q else "agent",
                            )
                            for q_pair, q_word in zip(q_pairs, q_words):
                                if const_expr(INDEX_PRE_ROPE):
                                    lds_st(xs, q_pair, q_word[0].bitcast(fx.Float32))
                                else:
                                    q_elem = q_pair * 2
                                    kq = q_elem % INDEX_DIM
                                    q0, q1 = bf2_f32(q_word[0])
                                    if kq < PE_DIM:
                                        c = ld_rope(
                                            _rsrc(rope_cos),
                                            row_position(s) * (PE_DIM // 2) + kq // 2,
                                        )
                                        sn = ld_rope(
                                            _rsrc(rope_sin),
                                            row_position(s) * (PE_DIM // 2) + kq // 2,
                                        )
                                        q0, q1 = q0 * c - q1 * sn, q0 * sn + q1 * c
                                    lds_st(xs, q_pair, bf16_pair(q0, q1))
                        gpu.barrier()
                        # Keep the query tile live while scoring adjacent key tiles.
                        for index_part in range_constexpr(INDEX_SCORE_REUSE):
                            tile = index_tile_base + index_part
                            tt = s * N_INDEX_SPLIT + tile
                            if const_expr(index_part > 0):
                                stamp("index_score", tt, 0)
                            if tile * INDEX_KEYS_PER_TASK < shard_size:
                                key_group = wave // 2
                                head_group = wave % 2
                                local_key = tile * INDEX_KEYS_PER_TASK + key_group * 16 + lane % 16
                                key_pos = shard_begin + local_key
                                safe_key = fx.min(key_pos, bound - 1)
                                head = head_group * 16 + lane % 16
                                score_frag = fx.Vector.filled(4, 0.0, fx.Float32)
                                for k32 in range_constexpr(INDEX_DIM // 32):
                                    k = k32 * 32 + (lane // 16) * 8
                                    qv = load_index_q8(head, k)
                                    kv = fx.Vector(
                                        bo.buffer_load(
                                            r_index_cache,
                                            (index_page_slot(s, safe_key) * INDEX_DIM + k) // 2,
                                            vec_width=4,
                                            dtype=T.i32,
                                        )
                                    ).bitcast(fx.BFloat16)
                                    if safe_key >= first_position:
                                        sn = request_start + safe_key - first_position
                                        kv_pairs = get_bf2_many(
                                            [(mb("index_k_new"), sn * INDEX_DIM + k + j * 2) for j in range(4)]
                                        )
                                        kv_values = []
                                        for pair in kv_pairs:
                                            kv_values += list(pair)
                                        kv = fx.Vector.from_elements(kv_values, fx.Float32).to(fx.BFloat16)
                                    score_frag = fx.Vector(
                                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [qv, kv, score_frag])
                                    )
                                partial = fx.Float32(0.0)
                                for e in range_constexpr(4):
                                    index_head = head_group * 16 + (lane // 16) * 4 + e
                                    weight = lds_ld(keys, index_head).bitcast(fx.Float32)
                                    partial = partial + fx.max(score_frag[e], fx.Float32(0.0)) * weight
                                partial = _xred(partial, 16, lambda a, b: a + b)
                                partial = _xred(partial, 32, lambda a, b: a + b)
                                if lane < 16:
                                    lds_st(red, wave * 16 + lane, partial)
                                gpu.barrier()
                                if (wave % 2 == 0) & (lane < 16) & (key_pos < bound):
                                    score = lds_ld(red, wave * 16 + lane) + lds_ld(red, (wave + 1) * 16 + lane)
                                    if const_expr(indexer_cp):
                                        if bound <= LOCAL_INDEX_CAPACITY:
                                            lds_st(red, 128 + key_group * 16 + lane, score)
                                        else:
                                            put(
                                                mb("index_scores"),
                                                s * INDEX_CAPACITY + local_key,
                                                score,
                                            )
                                    else:
                                        put(
                                            mb("index_scores"),
                                            s * INDEX_CAPACITY + local_key,
                                            score,
                                        )
                                if const_expr(indexer_cp):
                                    if bound <= LOCAL_INDEX_CAPACITY:
                                        # One wave per peer, one complete score exchange.
                                        gpu.barrier()
                                        out_key = shard_begin + tile * INDEX_KEYS_PER_TASK + lane
                                        if (wave < CP_RANKS) & (out_key < bound):
                                            put(
                                                cp_buffer(
                                                    "index_cp_scores",
                                                    (cp_peer(wave) if const_expr(CP_RANKS < W) else peer_dst),
                                                ),
                                                s * LOCAL_INDEX_CAPACITY + out_key,
                                                lds_ld(red, 128 + lane),
                                                CM_SYS,
                                            )
                            stamp("index_score", tt, 4)
                            if const_expr(index_part + 1 < INDEX_SCORE_REUSE):
                                # All readers finish before another key tile reuses red.
                                gpu.barrier()
                    else:
                        for index_part in range_constexpr(INDEX_SCORE_REUSE):
                            tt = s * N_INDEX_SPLIT + index_tile_base + index_part
                            if const_expr(index_part > 0):
                                stamp("index_score", tt, 0)
                            stamp("index_score", tt, 4)

                if const_expr(TILED_INDEXER):
                    # Context-parallel exact radix selection. Scoring owns a
                    # contiguous, 64-key-aligned shard of the *live* context on each
                    # rank. Histograms and compaction use bounded 4K-key tasks;
                    # neither LDS nor the compiled loop body scales with context.
                    # Each rank completes its local stage before entering the next
                    # one. All producers are resident, including rank-0 coordinators.
                    def cp_get(name, i):
                        return poll([(cp_buffer(name, sym), i, 1)], scope="system")[0][0]

                    def cp_threshold(digit, s):
                        values = poll(
                            [
                                (
                                    cp_buffer("index_cp_threshold", sym),
                                    (digit * S + s) * 2,
                                    2,
                                )
                            ],
                            scope="system",
                        )[0]
                        return fx.Uint32(_uniform(values[0])), _uniform(values[1])

                    def score_key(s, i):
                        bits = get(mb("index_scores"), s * INDEX_CAPACITY + i)
                        return fx.Uint32((bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits))

                    def cp_scan(value):
                        """Exclusive scan in ascending thread order; all threads participate."""
                        inclusive = fx.coop.warp_inclusive_scan(value, fx.ReductionOp.ADD, width=64)
                        if lane == 63:
                            lds_st(keys, 256 + wave, inclusive)
                        gpu.barrier()
                        before = fx.Int32(0)
                        total = fx.Int32(0)
                        for w in range_constexpr(WAVES):
                            count = lds_ld(keys, 256 + w)
                            before = before + (wave > w).select(count, fx.Int32(0))
                            total = total + count
                        offset = before + inclusive - value
                        gpu.barrier()
                        return offset, total

                    def cp_select_digit(shift, prefix, remain):
                        digit = 255 - fx.min(tid, 255)
                        count = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                        above, _ = cp_scan(count)
                        hit = (tid < 256) & (above < remain) & (above + count >= remain)
                        if hit:
                            lds_st(
                                keys,
                                256,
                                fx.Int32(prefix | (fx.Uint32(digit) << shift)),
                            )
                            lds_st(keys, 257, remain - above)
                        gpu.barrier()
                        result = fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)
                        gpu.barrier()
                        return result

                    cp_rank = cp_local_rank() if const_expr(indexer_cp) else fx.Int32(0)
                    owner = cp_peer(fx.Int32(0))
                    for digit in range_constexpr(INDEX_RADIX_DIGITS):
                        shift = 24 - digit * 8
                        mask = (0xFFFFFFFF << (shift + 8)) & 0xFFFFFFFF
                        for tt in range(start(f"index_hist_{digit}"), S * CP_PARTS, G):
                            tt = fx.Int32(tt)
                            s, part = tt // CP_PARTS, tt % CP_PARTS
                            stamp(f"index_hist_{digit}", tt, 0)
                            if use_long_index(s):
                                _, size = index_shard(s)
                                prefix = fx.Uint32(0)
                                if const_expr(digit > 0):
                                    prefix, _ = cp_threshold(digit - 1, s)
                                if tid < INDEX_RADIX_BINS:
                                    lds_st(keys, tid, fx.Int32(0))
                                gpu.barrier()
                                for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                                    i = part * INDEX_CP_TILE + batch * THREADS + tid
                                    if i < size:
                                        key = score_key(s, i)
                                        if (key & fx.Uint32(mask)) == prefix:
                                            bucket = fx.Int32((key >> shift) & fx.Uint32(255))
                                            fx.atomic_add(
                                                keys + bucket,
                                                fx.Int32(1),
                                                syncscope="workgroup",
                                            )
                                gpu.barrier()
                                if tid < INDEX_RADIX_BINS:
                                    put(
                                        mb("index_cp_hist"),
                                        (digit * S * CP_PARTS + tt) * 256 + tid,
                                        lds_ld(keys, tid),
                                    )
                                gpu.barrier()
                            stamp(f"index_hist_{digit}", tt, 4)

                        for s in range(start(f"index_prefix_{digit}"), S, G):
                            s = fx.Int32(s)
                            stamp(f"index_prefix_{digit}", s, 0)
                            if use_long_index(s):
                                if tid < INDEX_RADIX_BINS:
                                    count = fx.Int32(0)
                                    for part in range(CP_PARTS):
                                        count = count + get(
                                            mb("index_cp_hist"),
                                            ((digit * S + s) * CP_PARTS + part) * 256 + tid,
                                        )
                                    put(
                                        cp_buffer("index_cp_hist", owner),
                                        ((cp_rank * INDEX_RADIX_DIGITS + digit) * S + s) * 256 + tid,
                                        count,
                                        CM_SYS,
                                    )
                                gpu.barrier()
                                if cp_rank == 0:
                                    if tid < INDEX_RADIX_BINS:
                                        count = fx.Int32(0)
                                        for src in range_constexpr(CP_RANKS):
                                            count = count + cp_get(
                                                "index_cp_hist",
                                                ((src * INDEX_RADIX_DIGITS + digit) * S + s) * 256 + tid,
                                            )
                                        lds_st(keys, tid, count)
                                    gpu.barrier()
                                    prefix = fx.Uint32(0)
                                    remain = fx.min(fx.Int32(topk), row_position(s) + 1)
                                    if const_expr(digit > 0):
                                        prefix, remain = cp_threshold(digit - 1, s)
                                    prefix, remain = cp_select_digit(shift, prefix, remain)
                                    if tid == 0:
                                        for dst in range_constexpr(CP_RANKS):
                                            address = cp_buffer(
                                                "index_cp_threshold",
                                                cp_peer(fx.Int32(dst)),
                                            )
                                            put(
                                                address,
                                                (digit * S + s) * 2,
                                                fx.Int32(prefix),
                                                CM_SYS,
                                            )
                                            put(
                                                address,
                                                (digit * S + s) * 2 + 1,
                                                remain,
                                                CM_SYS,
                                            )
                                gpu.barrier()
                            stamp(f"index_prefix_{digit}", s, 4)

                    # Count strict winners and threshold ties separately. Prefixes
                    # in rank/partition/token order preserve the original selector's
                    # output: ascending token IDs above threshold, then earliest ties.
                    for tt in range(start("index_count"), S * CP_PARTS, G):
                        tt = fx.Int32(tt)
                        s, part = tt // CP_PARTS, tt % CP_PARTS
                        stamp("index_count", tt, 0)
                        if use_long_index(s):
                            _, size = index_shard(s)
                            threshold, _ = cp_threshold(3, s)
                            n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                            for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                                i = part * INDEX_CP_TILE + batch * THREADS + tid
                                if i < size:
                                    key = score_key(s, i)
                                    n_gt = n_gt + (key > threshold).select(fx.Int32(1), fx.Int32(0))
                                    n_eq = n_eq + (key == threshold).select(fx.Int32(1), fx.Int32(0))
                            _, total_gt = cp_scan(n_gt)
                            _, total_eq = cp_scan(n_eq)
                            if tid == 0:
                                put(mb("index_cp_counts"), tt * 2, total_gt)
                                put(mb("index_cp_counts"), tt * 2 + 1, total_eq)
                            gpu.barrier()
                        stamp("index_count", tt, 4)

                    for s in range(start("index_offsets"), S, G):
                        s = fx.Int32(s)
                        stamp("index_offsets", s, 0)
                        if use_long_index(s):
                            total_gt, total_eq = fx.Int32(0), fx.Int32(0)
                            for batch in range_constexpr((CP_PARTS + THREADS - 1) // THREADS):
                                part = batch * THREADS + tid
                                n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                                if part < CP_PARTS:
                                    n_gt = get(mb("index_cp_counts"), (s * CP_PARTS + part) * 2)
                                    n_eq = get(
                                        mb("index_cp_counts"),
                                        (s * CP_PARTS + part) * 2 + 1,
                                    )
                                off_gt, batch_gt = cp_scan(n_gt)
                                off_eq, batch_eq = cp_scan(n_eq)
                                if part < CP_PARTS:
                                    put(
                                        mb("index_cp_offsets"),
                                        (s * CP_PARTS + part) * 2,
                                        total_gt + off_gt,
                                    )
                                    put(
                                        mb("index_cp_offsets"),
                                        (s * CP_PARTS + part) * 2 + 1,
                                        total_eq + off_eq,
                                    )
                                total_gt = total_gt + batch_gt
                                total_eq = total_eq + batch_eq
                            if tid == 0:
                                put(
                                    cp_buffer("index_cp_counts", owner),
                                    (cp_rank * S + s) * 2,
                                    total_gt,
                                    CM_SYS,
                                )
                                put(
                                    cp_buffer("index_cp_counts", owner),
                                    (cp_rank * S + s) * 2 + 1,
                                    total_eq,
                                    CM_SYS,
                                )
                            gpu.barrier()
                            if cp_rank == 0:
                                before_gt, before_eq = fx.Int32(0), fx.Int32(0)
                                for src in range_constexpr(CP_RANKS):
                                    n_gt = cp_get("index_cp_counts", (src * S + s) * 2)
                                    n_eq = cp_get("index_cp_counts", (src * S + s) * 2 + 1)
                                    if tid == 0:
                                        address = cp_buffer("index_cp_offsets", cp_peer(fx.Int32(src)))
                                        put(address, s * 2, before_gt, CM_SYS)
                                        put(address, s * 2 + 1, before_eq, CM_SYS)
                                    before_gt = before_gt + n_gt
                                    before_eq = before_eq + n_eq
                            gpu.barrier()
                        stamp("index_offsets", s, 4)

                    for tt in range(start("index_emit"), S * CP_PARTS, G):
                        tt = fx.Int32(tt)
                        s, part = tt // CP_PARTS, tt % CP_PARTS
                        stamp("index_emit", tt, 0)
                        if use_long_index(s):
                            begin, size = index_shard(s)
                            threshold, need_eq = cp_threshold(3, s)
                            before_gt = cp_get("index_cp_offsets", s * 2) + get(mb("index_cp_offsets"), tt * 2)
                            before_eq = cp_get("index_cp_offsets", s * 2 + 1) + get(mb("index_cp_offsets"), tt * 2 + 1)
                            out_gt = fx.min(fx.Int32(topk), row_position(s) + 1) - need_eq
                            for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                                i = part * INDEX_CP_TILE + batch * THREADS + tid
                                is_gt, is_eq = fx.Int32(0), fx.Int32(0)
                                if i < size:
                                    key = score_key(s, i)
                                    is_gt = (key > threshold).select(fx.Int32(1), fx.Int32(0))
                                    is_eq = (key == threshold).select(fx.Int32(1), fx.Int32(0))
                                off_gt, count_gt = cp_scan(is_gt)
                                off_eq, count_eq = cp_scan(is_eq)
                                eq_index = before_eq + off_eq
                                selected = (is_gt != 0) | ((is_eq != 0) & (eq_index < need_eq))
                                slot = (is_gt != 0).select(before_gt + off_gt, out_gt + eq_index)
                                if selected:
                                    for dst in range_constexpr(CP_RANKS):
                                        put(
                                            cp_buffer(
                                                "index_cp_indices",
                                                cp_peer(fx.Int32(dst)),
                                            ),
                                            s * topk + slot,
                                            begin + i,
                                            CM_SYS,
                                        )
                                before_gt = before_gt + count_gt
                                before_eq = before_eq + count_eq
                            gpu.barrier()
                        stamp("index_emit", tt, 4)

                    for s in range(start("index_select"), S, G):
                        s = fx.Int32(s)
                        stamp("index_select", s, 0)
                        if use_long_index(s):
                            bound = row_position(s) + 1
                            for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                                j = batch * THREADS + tid
                                value = fx.Int32(0)
                                if j < fx.min(fx.Int32(topk), bound):
                                    value = cp_get("index_cp_indices", s * topk + j)
                                bo.buffer_store(
                                    value,
                                    _rsrc(mb("indices")),
                                    s * topk + j,
                                    cache_modifier=CM_DEV,
                                )
                            fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                            gpu.barrier()
                            if tid == 0:
                                put(mb("indices_ready"), s, fx.Int32(1))
                        stamp("index_select", s, 4)

                if const_expr(not TILED_INDEXER or indexer_cp):
                    for s in range(start("index_select"), S, G):
                        s = fx.Int32(s)
                        if use_short_index(s):
                            stamp("index_select", s, 0)
                            bound = row_position(s) + 1

                            def select_digit(shift, prefix, remain):
                                digit = 255 - fx.min(tid, 255)
                                count = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                                inclusive = fx.coop.warp_inclusive_scan(count, fx.ReductionOp.ADD, width=64)
                                wave_total = read_lane_i32(inclusive, 63)
                                if (wave < 4) & (lane == 63):
                                    lds_st(keys, 256 + wave, wave_total)
                                gpu.barrier()
                                before_wave = fx.Int32(0)
                                for w in range_constexpr(4):
                                    before_wave = before_wave + (wave > w).select(lds_ld(keys, 256 + w), fx.Int32(0))
                                above = before_wave + inclusive - count
                                hit = (tid < 256) & (above < remain) & (above + count >= remain)
                                gpu.barrier()
                                if hit:
                                    lds_st(
                                        keys,
                                        256,
                                        fx.Int32(prefix | (fx.Uint32(digit) << shift)),
                                    )
                                    lds_st(keys, 257, remain - above)
                                gpu.barrier()
                                result = fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)
                                # The following scan reuses these words for wave totals.
                                # All waves must read the final threshold before any
                                # wave overwrites it, or compaction can exceed topk.
                                gpu.barrier()
                                return result

                            # Transform scores to monotonic integer keys and build the high
                            # byte histogram in the same pass, avoiding one extra 4-K LDS
                            # scan before the remaining radix digits.
                            if tid < 256:
                                lds_st(keys, tid, fx.Int32(0))
                            gpu.barrier()

                            def stage_index_key(i, bits):
                                if i < bound:
                                    key = (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)
                                    # The selector CTA no longer needs the large GEMV staging
                                    # region, so reuse it for the 4-K radix keys instead of
                                    # increasing the monokernel's LDS allocation.
                                    lds_st(xs, i, key.bitcast(fx.Float32))
                                    digit = fx.Int32((fx.Uint32(key) >> 24) & fx.Uint32(255))
                                    fx.atomic_add(keys + digit, fx.Int32(1), syncscope="workgroup")

                            if const_expr(indexer_cp):
                                # Issue several independent peer-mailbox reads together.
                                # Tail lanes poll a valid key but never enter the histogram.
                                score_batches = (LOCAL_INDEX_CAPACITY + THREADS - 1) // THREADS
                                for batch in range_constexpr(0, score_batches, 4):
                                    ids = [tid + b * THREADS for b in range(batch, min(batch + 4, score_batches))]
                                    values = poll(
                                        [
                                            (
                                                cp_buffer("index_cp_scores", sym),
                                                s * LOCAL_INDEX_CAPACITY + fx.min(i, bound - 1),
                                                1,
                                            )
                                            for i in ids
                                        ],
                                        scope="system",
                                    )
                                    for i, value in zip(ids, values):
                                        stage_index_key(i, value[0])
                            else:
                                for batch in range_constexpr((LOCAL_INDEX_CAPACITY + THREADS - 1) // THREADS):
                                    i = tid + batch * THREADS
                                    if i < bound:
                                        stage_index_key(
                                            i,
                                            getf(
                                                mb("index_scores"),
                                                s * LOCAL_INDEX_CAPACITY + i,
                                            ).bitcast(fx.Int32),
                                        )
                            gpu.barrier()
                            prefix, remain = select_digit(24, fx.Uint32(0), fx.min(fx.Int32(topk), bound))
                            prefix_mask = 255 << 24
                            for shift in (16, 8, 0):
                                if tid < 256:
                                    lds_st(keys, tid, fx.Int32(0))
                                gpu.barrier()
                                for batch in range_constexpr((LOCAL_INDEX_CAPACITY + THREADS - 1) // THREADS):
                                    i = tid + batch * THREADS
                                    if i < bound:
                                        key = fx.Uint32(lds_ld(xs, i).bitcast(fx.Int32))
                                        if (key & fx.Uint32(prefix_mask)) == prefix:
                                            digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                            fx.atomic_add(
                                                keys + digit,
                                                fx.Int32(1),
                                                syncscope="workgroup",
                                            )
                                gpu.barrier()
                                prefix, remain = select_digit(shift, prefix, remain)
                                prefix_mask |= 255 << shift

                            threshold = prefix
                            r_index_out = _rsrc(mb("indices"))
                            items = (LOCAL_INDEX_CAPACITY + THREADS - 1) // THREADS
                            item_indices = [tid * items + j for j in range_constexpr(items)]
                            item_keys = [
                                fx.Uint32(lds_ld(xs, fx.min(i, bound - 1)).bitcast(fx.Int32)) for i in item_indices
                            ]
                            gt = [(i < bound) & (key > threshold) for i, key in zip(item_indices, item_keys)]
                            eq = [(i < bound) & (key == threshold) for i, key in zip(item_indices, item_keys)]

                            def scan_flags(flags):
                                """Thread-major exclusive offsets for eight flags without extra LDS.

                                The radix histogram already reserves keys[256:264] for wave
                                totals, so the scan can reuse those words and keep S=8 at the
                                96-KiB static-LDS boundary.
                                """
                                local = fx.Int32(0)
                                local_offsets = []
                                for flag in flags:
                                    local_offsets.append(local)
                                    local = local + flag.select(fx.Int32(1), fx.Int32(0))
                                inclusive = fx.coop.warp_inclusive_scan(local, fx.ReductionOp.ADD, width=64)
                                wave_total = read_lane_i32(inclusive, 63)
                                if lane == 63:
                                    lds_st(keys, 256 + wave, wave_total)
                                gpu.barrier()
                                before_wave = fx.Int32(0)
                                total = fx.Int32(0)
                                for w in range_constexpr(WAVES):
                                    wave_count = lds_ld(keys, 256 + w)
                                    before_wave = before_wave + (wave > w).select(wave_count, fx.Int32(0))
                                    total = total + wave_count
                                thread_base = before_wave + inclusive - local
                                return [thread_base + off for off in local_offsets], total

                            gt_offsets, out_gt = scan_flags(gt)
                            gpu.barrier()
                            eq_offsets, _ = scan_flags(eq)
                            gpu.barrier()
                            for j in range_constexpr(items):
                                if gt[j]:
                                    bo.buffer_store(
                                        fx.Int32(item_indices[j]),
                                        r_index_out,
                                        s * topk + gt_offsets[j],
                                        cache_modifier=CM_DEV,
                                    )
                                if eq[j] & (eq_offsets[j] < topk):
                                    lds_st(
                                        red,
                                        eq_offsets[j],
                                        item_indices[j].bitcast(fx.Float32),
                                    )
                            gpu.barrier()
                            need_eq = fx.min(fx.Int32(topk), bound) - out_gt
                            for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                                j = tid + batch * THREADS
                                if j >= bound:
                                    bo.buffer_store(
                                        fx.Int32(0),
                                        r_index_out,
                                        s * topk + j,
                                        cache_modifier=CM_DEV,
                                    )
                                if j < need_eq:
                                    i = lds_ld(red, j).bitcast(fx.Int32)
                                    bo.buffer_store(
                                        i,
                                        r_index_out,
                                        s * topk + out_gt + j,
                                        cache_modifier=CM_DEV,
                                    )
                            # Every lane contributed compact indices.  Make all of those
                            # device-memory stores visible before lane 0 publishes the one
                            # readiness tag consumed by the attention CTAs.
                            fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                            gpu.barrier()
                            if tid == 0:
                                put(mb("indices_ready"), s, fx.Int32(1))
                            stamp("index_select", s, 4)

            # ================================== 5. sparse MLA split: 32 keys x 8 heads
            if const_expr(SPLIT_ALL_HEADS):
                r_kv = _rsrc(kv_cache)
                r_pe = _rsrc(pe_cache)
                r_idx = _rsrc(indices)
                KPW = SPLIT_KEYS // WAVES

                def split_keys(t, s):
                    """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS."""
                    if const_expr(use_atom_kv_cache):
                        index_base, index_end = row_index_bounds(s)
                        nkeys = index_end - index_base
                        sparse = index_end > index_base
                    else:
                        index_base = s * topk
                        kv_len = pos0 + s + 1
                        sparse = kv_len > topk
                        nkeys = sparse.select(fx.Int32(topk), kv_len)
                    if const_expr(with_indexer):
                        if sparse:
                            if wave == 0:
                                if lane == 0:
                                    get(mb("indices_ready"), s)
                                # Only wave 0 consumes the compact index payload; its
                                # divergent ready poll reconverges before this acquire.
                                fx.memory_fence(
                                    ordering=fx.AtomicOrdering.Acquire,
                                    syncscope="agent",
                                )
                    if wave == 0:
                        if lane < SPLIT_KEYS:
                            k_pos = t * SPLIT_KEYS + lane
                            k_cl = (k_pos < nkeys).select(k_pos, 0)
                            if const_expr(with_indexer):
                                idx = k_cl
                                if sparse:
                                    idx = fx.Int32(
                                        bo.buffer_load(
                                            _rsrc(mb("indices")),
                                            s * topk + k_cl,
                                            vec_width=1,
                                            dtype=T.i32,
                                            cache_modifier=CM_DEV,
                                        )
                                    )
                                lds_st(attn_keys, lane, index_page_slot(s, idx))
                            elif const_expr(use_atom_kv_cache):
                                idx = fx.Int32(0)
                                if sparse:
                                    idx = fx.Int32(
                                        bo.buffer_load(
                                            r_idx,
                                            index_base + k_cl,
                                            vec_width=1,
                                            dtype=T.i32,
                                        )
                                    )
                                lds_st(attn_keys, lane, idx)
                            else:
                                idx = fx.Int32(
                                    bo.buffer_load(
                                        r_idx,
                                        index_base + k_cl,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                                lds_st(attn_keys, lane, sparse.select(idx, k_cl))
                    return nkeys, sparse

                def gather_old_kv():
                    """Copy persistent KV/PE rows to LDS before patching incoming cache versions."""
                    krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
                    if const_expr(COHORT_ROW_TILES and not with_indexer and native_fp4_mfma and npes == 4):
                        # Two independent persistent-cache rows outstanding per wave.
                        # Complete the original gather before any q/patch dependency.
                        for gather_base in range_constexpr(0, KPW, 2):
                            gather_kv = [
                                fx.Vector(
                                    bo.buffer_load(
                                        r_kv,
                                        krows[gather_base + part] * (QK_DIM // 2) + lane * 4,
                                        vec_width=4,
                                        dtype=T.i32,
                                    )
                                )
                                for part in range(2)
                            ]
                            for gather_part in range_constexpr(2):
                                fx.ptr_store(
                                    gather_kv[gather_part].bitcast(fx.Float32),
                                    ktile + ((wave * KPW + gather_base + gather_part) * KS + lane * 4),
                                )
                            if lane < PE_DIM // 2:
                                gather_pe = [
                                    ld_f32(r_pe, krows[gather_base + part] * (QK_DIM // 2) + KV_LORA // 2 + lane)
                                    for part in range(2)
                                ]
                                for gather_part in range_constexpr(2):
                                    lds_st(
                                        petile,
                                        (wave * KPW + gather_base + gather_part) * PS + lane,
                                        gather_pe[gather_part],
                                    )
                    else:
                        for jj in range_constexpr(KPW):
                            j = wave * KPW + jj
                            if const_expr(cache_fp8):
                                row = krows[jj] * (QK_DIM // 4)
                                raw = fx.Vector(bo.buffer_load(r_kv, row + lane * 2, vec_width=2, dtype=T.i32))
                                fx.ptr_store(
                                    _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                    ktile + (j * KS + lane * 4),
                                )
                                if lane < PE_DIM // 8:
                                    pe_raw = fx.Vector(
                                        bo.buffer_load(
                                            r_pe,
                                            row + KV_LORA // 4 + lane * 2,
                                            vec_width=2,
                                            dtype=T.i32,
                                        )
                                    )
                                    fx.ptr_store(
                                        _fp8_to_bf16x8(pe_raw[0], pe_raw[1]).bitcast(fx.Float32),
                                        petile + (j * PS + lane * 4),
                                    )
                            else:
                                kv_row_words = QK_DIM // 2 if const_expr(use_atom_kv_cache) else KV_LORA // 2
                                kv8 = fx.Vector(
                                    bo.buffer_load(
                                        r_kv,
                                        krows[jj] * kv_row_words + lane * 4,
                                        vec_width=4,
                                        dtype=T.i32,
                                    )
                                )
                                fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 4))
                                if lane < PE_DIM // 2:
                                    pe_row = (
                                        krows[jj] * (QK_DIM // 2) + KV_LORA // 2 + lane
                                        if const_expr(use_atom_kv_cache)
                                        else krows[jj] * (PE_DIM // 2) + lane
                                    )
                                    lds_st(petile, j * PS + lane, ld_f32(r_pe, pe_row))

                def patch_new_kv():
                    """Patch the cache version visible at this tile's serial boundary."""
                    if const_expr(DEFER_OFF_CACHE):
                        # One coalesced incoming-slot load per lane, then one ballot
                        # per key replaces a serial forty-slot search in wave zero.
                        incoming_slot = fx.Int32(
                            bo.buffer_load(
                                _rsrc(full_slots), fx.min(lane, total_samples - 1) * 2, vec_width=1, dtype=T.i32
                            )
                        )
                        visible_row = (lane < row_offset + S) & (lane < total_samples) & (incoming_slot >= 0)
                        for jj in range_constexpr(KPW):
                            j = wave * KPW + jj
                            physical_key = _uniform(lds_ld(attn_keys, j))
                            version_mask = fx.Int64(rocdl.ballot(T.i64, visible_row & (incoming_slot == physical_key)))
                            if version_mask != 0:
                                incoming_row = fx.Int32(63) - fx.Int32(fmath.ctlz(version_mask))
                                source_tile = incoming_row // S
                                source_row = incoming_row % S
                                source_scratch = full_scratch + fx.Int64(source_tile) * TILE_SCRATCH_BYTES
                                kvp = get2_many(
                                    [
                                        (source_scratch + SC["kvnew"], source_row * KV_LORA + lane * 8 + m * 2)
                                        for m in range(4)
                                    ]
                                )
                                words = [bf16_pair(a0, a1) for a0, a1 in kvp]
                                fx.ptr_store(
                                    fx.Vector.from_elements(words, fx.Float32),
                                    ktile + (j * KS + lane * 4),
                                )
                                if lane < PE_DIM // 2:
                                    a0, a1 = get2(source_scratch + SC["penew"], source_row * PE_DIM + lane * 2)
                                    lds_st(petile, j * PS + lane, bf16_pair(a0, a1))
                    else:
                        if const_expr(use_atom_kv_cache):
                            new_active = [row_writes_cache(new_s) for new_s in range(S)]
                            new_slots = [row_slot(new_s) for new_s in range(S)]
                        for jj in range_constexpr(KPW):
                            j = wave * KPW + jj
                            kr = lds_ld(attn_keys, j)
                            if const_expr(use_atom_kv_cache):
                                sn = fx.Int32(-1)
                                for new_s in range_constexpr(S):
                                    match = new_active[new_s] & (kr == new_slots[new_s])
                                    sn = match.select(fx.Int32(new_s), sn)
                                is_new = sn >= 0
                            else:
                                is_new = kr >= pos0
                                sn = kr - pos0
                            if is_new:
                                kvp = get2_many([(mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2) for m in range(4)])
                                w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                                fx.ptr_store(
                                    fx.Vector.from_elements(w, fx.Float32),
                                    ktile + (j * KS + lane * 4),
                                )
                                if lane < PE_DIM // 2:
                                    a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                                    lds_st(petile, j * PS + lane, bf16_pair(a0, a1))

                N_HEAD_GROUPS = 1
                for tt in range(start("split"), S * N_HEAD_GROUPS * N_SPLIT, G):
                    tt = fx.Int32(tt)
                    stamp("split", tt, 0)
                    s = tt // (N_HEAD_GROUPS * N_SPLIT)
                    head_group = (tt // N_SPLIT) % N_HEAD_GROUPS
                    t = tt % N_SPLIT
                    h = wave
                    nkeys, sparse = split_keys(t, s)
                    gpu.barrier()
                    gather_old_kv()  # before waiting for q: these rows are from earlier launches
                    if const_expr(True):
                        N_PE_T = PE_DIM // Q_B_TILE
                        hint_wait(
                            N_UK + H * N_PE_T + 1,
                            lambda k: (
                                (k < N_UK).select(
                                    fx.Int64(SC["q_lat"]),
                                    (k < N_UK + H * N_PE_T).select(fx.Int64(SC["q_pe"]), fx.Int64(SC["penew"])),
                                )
                                + scratch,
                                (k < N_UK).select(
                                    (s * H + k // UK_PER_HEAD) * KV_LORA + (k % UK_PER_HEAD) * UK_TILE + UK_TILE - 1,
                                    (k < N_UK + H * N_PE_T).select(
                                        (s * H + (k - N_UK) // N_PE_T) * PE_DIM
                                        + ((k - N_UK) % N_PE_T) * Q_B_TILE
                                        + Q_B_TILE
                                        - 1,
                                        s * PE_DIM + PE_DIM - 1,
                                    ),
                                ),
                            ),
                            mark=("split", tt),
                        )
                    # q of all heads -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64
                    NQ = H * KV_LORA // 4 // THREADS
                    tpe = fx.min(tid, H * PE_DIM // 4 - 1)
                    qv = poll(
                        [
                            (
                                mb("q_lat"),
                                (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2,
                                2,
                            )
                            for i in range(NQ)
                        ]
                        + [(mb("q_pe"), (s * H * PE_DIM + tpe * 4) // 2, 2)]
                    )
                    for i in range_constexpr(NQ):
                        w4 = tid + i * THREADS
                        qw = (w4 // (KV_LORA // 4)) * QS + (w4 % (KV_LORA // 4)) * 2
                        lds_st(xs, qw, qv[i][0].bitcast(fx.Float32))
                        lds_st(xs, qw + 1, qv[i][1].bitcast(fx.Float32))
                    if tid < H * PE_DIM // 4:
                        hh = tid // (PE_DIM // 4)
                        qw = hh * QS + KV_LORA // 2 + (tid % (PE_DIM // 4)) * 2
                        lds_st(xs, qw, qv[NQ][0].bitcast(fx.Float32))
                        lds_st(xs, qw + 1, qv[NQ][1].bitcast(fx.Float32))
                    patch_new_kv()
                    if const_expr(True):
                        stamp("split", tt, 2)
                    gpu.barrier()
                    stamp("split", tt, 5)
                    # scores = K Q^T on MFMA.  The 64-key tile maps the eight waves to
                    # (four row groups, two K halves); the batch-8 32-key tile uses two
                    # row groups and four waves per K half.  All waves subsequently own
                    # one attention head for softmax and P@V.
                    hn = lane % 16
                    rgk = wave % (SPLIT_KEYS // 16)
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    for st in range_constexpr(QK_DIM // 32 // 2):
                        kst = ((wave // (SPLIT_KEYS // 16)) % 2) * (QK_DIM // 32 // 2) + st
                        key = rgk * 16 + lane % 16
                        kw = (kst < KV_LORA // 32).select(
                            KT_OFF + key * KS + kst * 16,
                            PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                        )
                        a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                        b = fx.ptr_load(
                            xs + (hn * QS + kst * 16 + (lane // 16) * 4),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    if const_expr(SPLIT_KEYS == 64) or wave < 4:
                        fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                    gpu.barrier()
                    stamp("split", tt, 6)
                    # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
                    kidx = t * SPLIT_KEYS + lane
                    valid = (lane < SPLIT_KEYS) & (kidx < nkeys)
                    r16 = lane % 16

                    def softmax_one(softmax_head):
                        cl = split_score_column(softmax_head, lane)
                        key_rg = fx.min(lane // 16, SPLIT_KEYS // 16 - 1)
                        half_stride = SPLIT_KEYS // 16
                        raw = lds_ld(red, (key_rg * 64 + cl) * 4 + r16 % 4) + lds_ld(
                            red, ((key_rg + half_stride) * 64 + cl) * 4 + r16 % 4
                        )
                        sc_v = valid.select(raw * scale, fx.Float32(NEG))
                        m = wave_max(sc_v)
                        p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                        lsum = wave_sum(p)
                        p_n = _xshfl(p, 1)
                        if (lane < SPLIT_KEYS) & (lane % 2 == 0):
                            lds_st(
                                pl,
                                softmax_head * (SPLIT_KEYS // 2) + lane // 2,
                                bf16_pair(p, p_n),
                            )
                        return m, lsum

                    m0, l0 = softmax_one(wave)
                    m1, l1 = softmax_one(wave + WAVES)
                    gpu.barrier()
                    stamp("split", tt, 3)
                    # O = P V on MFMA: heads M, keys K, latent dims N.  Each V word holds
                    # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
                    # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
                    for g in range_constexpr(KV_LORA // 32 // WAVES):
                        dw = (wave * (KV_LORA // 32 // WAVES) + g) * 16 + lane % 16  # dim pair word
                        c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                        c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                        for js in range_constexpr(SPLIT_KEYS // 32):
                            a = fx.ptr_load(
                                pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                                result_type=v4f,
                            ).bitcast(fx.BFloat16)
                            ws = [
                                fx.ptr_load(ktile + ((js * 32 + (lane // 16) * 8 + i) * KS + dw)).bitcast(fx.Int32)
                                for i in range(8)
                            ]
                            w_lo = [(ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)]
                            w_hi = [fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536) for i in range(4)]
                            b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                            b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                            c0 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0]))
                            c1 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1]))
                        if lane < 64:
                            for e in range_constexpr(4):
                                hh = (lane // 16) * 4 + e
                                put_bf(
                                    mb("sp_acc"),
                                    ((s * N_SPLIT + t) * H + hh) * KV_LORA + dw * 2,
                                    [c0[e], c1[e]],
                                )
                    if lane == 0:  # written last: the merge's readiness hint
                        put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m0)
                        put(mb("sp_m"), (s * N_SPLIT + t) * H + h + WAVES, m1)
                        put(mb("sp_l"), (s * N_SPLIT + t) * H + h, l0)
                        put(mb("sp_l"), (s * N_SPLIT + t) * H + h + WAVES, l1)
                    stamp("split", tt, 4)
            else:
                r_kv = _rsrc(kv_cache)
                r_pe = _rsrc(pe_cache)
                r_idx = _rsrc(indices)
                KPW = SPLIT_GATHER_KEYS // WAVES

                def split_keys(t, s):
                    """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS."""
                    if const_expr(use_atom_kv_cache):
                        index_base, index_end = row_index_bounds(s)
                        nkeys = index_end - index_base
                        sparse = index_end > index_base
                    else:
                        index_base = s * topk
                        kv_len = pos0 + s + 1
                        sparse = kv_len > topk
                        nkeys = sparse.select(fx.Int32(topk), kv_len)
                    if const_expr(with_indexer):
                        if sparse:
                            if wave == 0:
                                if lane == 0:
                                    get(mb("indices_ready"), s)
                                # Only wave 0 consumes the compact index payload; its
                                # divergent ready poll reconverges before this acquire.
                                fx.memory_fence(
                                    ordering=fx.AtomicOrdering.Acquire,
                                    syncscope="agent",
                                )
                    if wave == 0:
                        if lane < SPLIT_GATHER_KEYS:
                            k_pos = t * SPLIT_GATHER_KEYS + lane
                            k_cl = (k_pos < nkeys).select(k_pos, 0)
                            if const_expr(with_indexer):
                                idx = k_cl
                                if sparse:
                                    idx = fx.Int32(
                                        bo.buffer_load(
                                            _rsrc(mb("indices")),
                                            s * topk + k_cl,
                                            vec_width=1,
                                            dtype=T.i32,
                                            cache_modifier=CM_DEV,
                                        )
                                    )
                                lds_st(attn_keys, lane, index_page_slot(s, idx))
                            elif const_expr(use_atom_kv_cache):
                                idx = fx.Int32(0)
                                if sparse:
                                    idx = fx.Int32(
                                        bo.buffer_load(
                                            r_idx,
                                            index_base + k_cl,
                                            vec_width=1,
                                            dtype=T.i32,
                                        )
                                    )
                                lds_st(attn_keys, lane, idx)
                            else:
                                idx = fx.Int32(
                                    bo.buffer_load(
                                        r_idx,
                                        index_base + k_cl,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                                lds_st(attn_keys, lane, sparse.select(idx, k_cl))
                    return nkeys, sparse

                def gather_old_kv():
                    """Copy persistent KV/PE rows to LDS before patching incoming cache versions."""
                    krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
                    for jj in range_constexpr(KPW):
                        j = wave * KPW + jj
                        if const_expr(cache_fp8):
                            row = krows[jj] * (QK_DIM // 4)
                            raw = fx.Vector(bo.buffer_load(r_kv, row + lane * 2, vec_width=2, dtype=T.i32))
                            fx.ptr_store(
                                _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                ktile + (j * KS + lane * 4),
                            )
                            if lane < PE_DIM // 8:
                                pe_raw = fx.Vector(
                                    bo.buffer_load(
                                        r_pe,
                                        row + KV_LORA // 4 + lane * 2,
                                        vec_width=2,
                                        dtype=T.i32,
                                    )
                                )
                                fx.ptr_store(
                                    _fp8_to_bf16x8(pe_raw[0], pe_raw[1]).bitcast(fx.Float32),
                                    petile + (j * PS + lane * 4),
                                )
                        else:
                            kv_row_words = QK_DIM // 2 if const_expr(use_atom_kv_cache) else KV_LORA // 2
                            kv8 = fx.Vector(
                                bo.buffer_load(
                                    r_kv,
                                    krows[jj] * kv_row_words + lane * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 4))
                            if lane < PE_DIM // 2:
                                pe_row = (
                                    krows[jj] * (QK_DIM // 2) + KV_LORA // 2 + lane
                                    if const_expr(use_atom_kv_cache)
                                    else krows[jj] * (PE_DIM // 2) + lane
                                )
                                lds_st(petile, j * PS + lane, ld_f32(r_pe, pe_row))

                def patch_new_kv():
                    """Patch the cache version visible at this tile's serial boundary."""
                    if const_expr(DEFER_OFF_CACHE):
                        # One coalesced incoming-slot load per lane, then one ballot
                        # per key replaces a serial forty-slot search in wave zero.
                        incoming_slot = fx.Int32(
                            bo.buffer_load(
                                _rsrc(full_slots), fx.min(lane, total_samples - 1) * 2, vec_width=1, dtype=T.i32
                            )
                        )
                        visible_row = (lane < row_offset + S) & (lane < total_samples) & (incoming_slot >= 0)
                        for jj in range_constexpr(KPW):
                            j = wave * KPW + jj
                            physical_key = _uniform(lds_ld(attn_keys, j))
                            version_mask = fx.Int64(rocdl.ballot(T.i64, visible_row & (incoming_slot == physical_key)))
                            if version_mask != 0:
                                incoming_row = fx.Int32(63) - fx.Int32(fmath.ctlz(version_mask))
                                source_tile = incoming_row // S
                                source_row = incoming_row % S
                                source_scratch = full_scratch + fx.Int64(source_tile) * TILE_SCRATCH_BYTES
                                kvp = get2_many(
                                    [
                                        (source_scratch + SC["kvnew"], source_row * KV_LORA + lane * 8 + m * 2)
                                        for m in range(4)
                                    ]
                                )
                                words = [bf16_pair(a0, a1) for a0, a1 in kvp]
                                fx.ptr_store(
                                    fx.Vector.from_elements(words, fx.Float32),
                                    ktile + (j * KS + lane * 4),
                                )
                                if lane < PE_DIM // 2:
                                    a0, a1 = get2(source_scratch + SC["penew"], source_row * PE_DIM + lane * 2)
                                    lds_st(petile, j * PS + lane, bf16_pair(a0, a1))
                    else:
                        if const_expr(use_atom_kv_cache):
                            new_active = [row_writes_cache(new_s) for new_s in range(S)]
                            new_slots = [row_slot(new_s) for new_s in range(S)]
                        for jj in range_constexpr(KPW):
                            j = wave * KPW + jj
                            kr = lds_ld(attn_keys, j)
                            if const_expr(use_atom_kv_cache):
                                sn = fx.Int32(-1)
                                for new_s in range_constexpr(S):
                                    match = new_active[new_s] & (kr == new_slots[new_s])
                                    sn = match.select(fx.Int32(new_s), sn)
                                is_new = sn >= 0
                            else:
                                is_new = kr >= pos0
                                sn = kr - pos0
                            if is_new:
                                kvp = get2_many([(mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2) for m in range(4)])
                                w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                                fx.ptr_store(
                                    fx.Vector.from_elements(w, fx.Float32),
                                    ktile + (j * KS + lane * 4),
                                )
                                if lane < PE_DIM // 2:
                                    a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                                    lds_st(petile, j * PS + lane, bf16_pair(a0, a1))

                N_HEAD_GROUPS = H // WAVES
                for tt in range(start("split"), S * N_HEAD_GROUPS * N_SPLIT_TASKS, G):
                    tt = fx.Int32(tt)
                    stamp("split", tt, 0)
                    s = tt // (N_HEAD_GROUPS * N_SPLIT_TASKS)
                    head_group = (tt // N_SPLIT_TASKS) % N_HEAD_GROUPS
                    t = tt % N_SPLIT_TASKS
                    h = head_group * WAVES + wave
                    nkeys, sparse = split_keys(t, s)
                    gpu.barrier()
                    gather_old_kv()  # before waiting for q: these rows are from earlier launches
                    if const_expr(True):
                        N_PE_T = PE_DIM // Q_B_TILE
                        hint_wait(
                            N_UK + H * N_PE_T + 1,
                            lambda k: (
                                (k < N_UK).select(
                                    fx.Int64(SC["q_lat"]),
                                    (k < N_UK + H * N_PE_T).select(fx.Int64(SC["q_pe"]), fx.Int64(SC["penew"])),
                                )
                                + scratch,
                                (k < N_UK).select(
                                    (s * H + k // UK_PER_HEAD) * KV_LORA + (k % UK_PER_HEAD) * UK_TILE + UK_TILE - 1,
                                    (k < N_UK + H * N_PE_T).select(
                                        (s * H + (k - N_UK) // N_PE_T) * PE_DIM
                                        + ((k - N_UK) % N_PE_T) * Q_B_TILE
                                        + Q_B_TILE
                                        - 1,
                                        s * PE_DIM + PE_DIM - 1,
                                    ),
                                ),
                            ),
                            mark=("split", tt),
                        )
                    # q of all heads -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64
                    NQ = H * KV_LORA // 4 // THREADS
                    tpe = fx.min(tid, H * PE_DIM // 4 - 1)
                    qv = poll(
                        [
                            (
                                mb("q_lat"),
                                (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2,
                                2,
                            )
                            for i in range(NQ)
                        ]
                        + [(mb("q_pe"), (s * H * PE_DIM + tpe * 4) // 2, 2)]
                    )
                    for i in range_constexpr(NQ):
                        w4 = tid + i * THREADS
                        qw = (w4 // (KV_LORA // 4)) * QS + (w4 % (KV_LORA // 4)) * 2
                        lds_st(xs, qw, qv[i][0].bitcast(fx.Float32))
                        lds_st(xs, qw + 1, qv[i][1].bitcast(fx.Float32))
                    if tid < H * PE_DIM // 4:
                        hh = tid // (PE_DIM // 4)
                        qw = hh * QS + KV_LORA // 2 + (tid % (PE_DIM // 4)) * 2
                        lds_st(xs, qw, qv[NQ][0].bitcast(fx.Float32))
                        lds_st(xs, qw + 1, qv[NQ][1].bitcast(fx.Float32))
                    patch_new_kv()
                    if const_expr(True):
                        stamp("split", tt, 2)
                    gpu.barrier()
                    stamp("split", tt, 5)
                    # scores = K Q^T on MFMA.  The 64-key tile maps the eight waves to
                    # (four row groups, two K halves); the batch-8 32-key tile uses two
                    # row groups and four waves per K half.  All waves subsequently own
                    # one attention head for softmax and P@V.
                    hn = head_group * WAVES + fx.min(lane % 16, WAVES - 1)
                    rgk = wave % (SPLIT_KEYS // 16)
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    for st in range_constexpr(QK_DIM // 32 // 2):
                        kst = ((wave // (SPLIT_KEYS // 16)) % 2) * (QK_DIM // 32 // 2) + st
                        key = rgk * 16 + lane % 16 + (wave // 4) * SPLIT_KEYS * (SPLIT_TASK_PAIR - 1)
                        kw = (kst < KV_LORA // 32).select(
                            KT_OFF + key * KS + kst * 16,
                            PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                        )
                        a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                        b = fx.ptr_load(
                            xs + (hn * QS + kst * 16 + (lane // 16) * 4),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    if const_expr(SPLIT_KEYS == 64 or SPLIT_TASK_PAIR == 2) or wave < 4:
                        fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                    gpu.barrier()
                    stamp("split", tt, 6)
                    for split_part in range_constexpr(SPLIT_TASK_PAIR):
                        # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
                        logical_split = t * SPLIT_TASK_PAIR + split_part
                        kidx = logical_split * SPLIT_KEYS + lane
                        valid = (lane < SPLIT_KEYS) & (kidx < nkeys)
                        r16 = lane % 16
                        cl = split_score_column(wave, lane)
                        key_rg = fx.min(lane // 16, SPLIT_KEYS // 16 - 1)
                        half_stride = SPLIT_KEYS // 16
                        raw = lds_ld(red, ((key_rg + split_part * 4) * 64 + cl) * 4 + r16 % 4) + lds_ld(
                            red, ((key_rg + half_stride + split_part * 4) * 64 + cl) * 4 + r16 % 4
                        )
                        sc_v = valid.select(raw * scale, fx.Float32(NEG))
                        m = wave_max(sc_v)
                        p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                        lsum = wave_sum(p)
                        p_n = _xshfl(p, 1)
                        if (lane < SPLIT_KEYS) & (lane % 2 == 0):
                            lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
                        gpu.barrier()
                        stamp("split", tt, 3)
                        # O = P V on MFMA: heads M, keys K, latent dims N.  Each V word holds
                        # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
                        # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
                        for g in range_constexpr(KV_LORA // 32 // WAVES):
                            dw = (wave * (KV_LORA // 32 // WAVES) + g) * 16 + lane % 16  # dim pair word
                            c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                            c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                            for js in range_constexpr(SPLIT_KEYS // 32):
                                a = fx.ptr_load(
                                    pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                                    result_type=v4f,
                                ).bitcast(fx.BFloat16)
                                ws = [
                                    fx.ptr_load(
                                        ktile + ((split_part * SPLIT_KEYS + js * 32 + (lane // 16) * 8 + i) * KS + dw)
                                    ).bitcast(fx.Int32)
                                    for i in range(8)
                                ]
                                w_lo = [(ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)]
                                w_hi = [
                                    fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536) for i in range(4)
                                ]
                                b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                                b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                                c0 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0]))
                                c1 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1]))
                            if lane < 32:
                                for e in range_constexpr(4):
                                    hh = split_acc_head(head_group, lane // 16, e)
                                    put_bf(
                                        mb("sp_acc"),
                                        ((s * N_SPLIT + logical_split) * H + hh) * KV_LORA + dw * 2,
                                        [c0[e], c1[e]],
                                    )
                        if lane == 0:  # written last: the merge's readiness hint
                            put(mb("sp_m"), (s * N_SPLIT + logical_split) * H + h, m)
                            put(mb("sp_l"), (s * N_SPLIT + logical_split) * H + h, lsum)
                        if const_expr(split_part + 1 < SPLIT_TASK_PAIR):
                            gpu.barrier()
                    stamp("split", tt, 4)

            # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
            # 4 row groups x 8 chunks: 2 waves per row group, 4 chunks each
            r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
            UV_NKC = KV_LORA // 64
            UV_R = UV_TILE // 16
            UV_WPR = WAVES // UV_R
            UV_UNITS = UV_NKC // (attention_k_chunks_per_unit * UV_WPR)
            if const_expr(GLOBAL_UV40):
                # Retired input-norm payloads already have the ordinary epoch.
                # A separate phase bit prevents UV from accepting them as ready.
                uv_tag = tag ^ fx.Int32(-2147483648)

            def uv_merged_pair(base_addr, i):
                def load_pair():
                    return fx.Vector(
                        bo.buffer_load(_rsrc(base_addr), i * 2, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
                    )

                words = load_pair()
                while (words[1] != uv_tag) | (words[3] != uv_tag):
                    spin_pause()
                    words = load_pair()
                return words[0], words[2]

            for tt in range(start("uv"), S * N_UV // UV_REUSE, G):
                tt = fx.Int32(tt)
                stamp("uv", tt, 0)
                s = tt // (N_UV // UV_REUSE)
                t = (tt % (N_UV // UV_REUSE)) * UV_REUSE
                local_t = dcp_local_uv_tile(t, L)
                head = t // (V_DIM // UV_TILE)

                def u_uv_at(c, uv_local_t):
                    kc = ((wave % UV_WPR) * UV_UNITS + c) * attention_k_chunks_per_unit
                    return unit_attention(
                        r_wuv,
                        r_suv,
                        uv_local_t * UV_R + wave // UV_WPR,
                        kc,
                        UV_NKC,
                        KV_LORA,
                        128,
                        attention_b_word(kc * 64),
                        scale_rows=uv_scale_rows,
                        coef=((lambda: lds_ld(misc, N_SPLIT + 2 + kc // 2)) if const_expr(attention_ptpc) else None),
                    )

                if const_expr(not GLOBAL_UV40):
                    pre = [u_uv_at(c, local_t) for c in range(UV_UNITS)]
                hint_wait(
                    N_SPLIT,
                    lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head),
                    mark=("uv", tt),
                )
                pre_poll(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
                stamp("uv", tt, 5)
                # Four 16-split quarters keep each poll and its live payload bounded.
                # Every thread owns one (quarter, half-of-latent) pair and processes
                # its two latent-pair positions sequentially; red holds quarter sums.
                SPH = N_SPLIT // 4
                dp_lo = tid % (KV_LORA // 4)
                hf = tid // (KV_LORA // 4)
                spi = fx.min(lane, N_SPLIT - 1)
                ml = (s * N_SPLIT + spi) * H + head
                ml_got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)])
                if wave == 0:  # per-split weights exp(m - M) / L for this head -> misc[sp]
                    ok_sp = lane < N_SPLIT
                    m_sp = ok_sp.select(ml_got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                    l_sp = ok_sp.select(ml_got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                    m_all = wave_max(m_sp)
                    w_sp = _exp(m_sp - m_all)
                    den = wave_sum(l_sp * w_sp)
                    if ok_sp:
                        lds_st(
                            misc,
                            lane,
                            w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                        )
                    if lane == 0:
                        lds_st(misc, N_SPLIT, m_all)
                        lds_st(misc, N_SPLIT + 1, den)
                stamp("uv", tt, 2)
                gpu.barrier()
                for dh in range_constexpr(2):
                    dp = dp_lo + dh * (KV_LORA // 4)
                    got = poll(
                        [
                            (
                                mb("sp_acc"),
                                ((s * N_SPLIT + hf * SPH + j) * H + head) * (KV_LORA // 2) + dp,
                                1,
                            )
                            for j in range(SPH)
                        ]
                    )
                    o0 = fx.Float32(0.0)
                    o1 = fx.Float32(0.0)
                    for j in range_constexpr(SPH):
                        wj = lds_ld(misc, hf * SPH + j)
                        a0, a1 = bf2_f32(got[j][0])
                        o0 = o0 + a0 * wj
                        o1 = o1 + a1 * wj
                    lds_st(red, (hf * (KV_LORA // 2) + dp) * 2, o0)
                    lds_st(red, (hf * (KV_LORA // 2) + dp) * 2 + 1, o1)
                gpu.barrier()
                uv_valid = tid < KV_LORA // 2
                uv0, uv1 = fx.Float32(0.0), fx.Float32(0.0)
                if uv_valid:
                    o0 = fx.Float32(0.0)
                    o1 = fx.Float32(0.0)
                    for q in range_constexpr(4):
                        o0 = o0 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2)
                        o1 = o1 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2 + 1)
                    uv0, uv1 = bf16_round(o0), bf16_round(o1)
                    if const_expr(not attention_ptpc):
                        lds_st(xs, tid, bf16_pair(uv0, uv1))
                if const_expr(attention_ptpc):
                    uv_max = wave_max(uv_valid.select(fx.max(fmath.absf(uv0), fmath.absf(uv1)), fx.Float32(0.0)))
                    if (lane == 0) & (wave < KV_LORA // 128):
                        lds_st(
                            misc,
                            N_SPLIT + 2 + wave,
                            (uv_max == 0.0).select(fx.Float32(1.0), uv_max * (1.0 / FP8_MAX)),
                        )
                gpu.barrier()
                if const_expr(attention_ptpc):
                    if uv_valid:
                        uv_scale = lds_ld(misc, N_SPLIT + 2 + tid // 64)
                        uv_rcp = 1.0 / uv_scale
                        st_ptpc(
                            tid * 2,
                            div_rn(uv0, uv_scale, uv_rcp),
                            div_rn(uv1, uv_scale, uv_rcp),
                        )
                    gpu.barrier()
                if const_expr(GLOBAL_UV40):
                    # This range is disjoint from packed mid values/scales.
                    # Input-norm readers retire at the global front-stage
                    # barrier. The distinct UV phase tag prevents consumers from
                    # accepting those old payloads before these stores complete.
                    if uv_valid:
                        uv_packed = bf16_pair(uv0, uv1).bitcast(fx.Int32)
                        bo.buffer_store(
                            fx.Vector.from_elements([uv_packed, uv_tag], fx.Int32),
                            _rsrc(mb("ugp")),
                            (UV_MERGE_OFF + (s * H + head) * (KV_LORA // 2) + tid) * 2,
                            cache_modifier=CM_DEV,
                        )
                    gpu.barrier()
                    stamp("uv", tt, 3)
                else:
                    owner = dcp_merge_latent(s, t)
                    if owner:
                        for uv_pass in range_constexpr(UV_REUSE):
                            local_t = dcp_local_uv_tile(t + uv_pass, L)
                            if const_expr(uv_pass > 0):
                                pre = [u_uv_at(c, local_t) for c in range(UV_UNITS)]
                            acc = run_units(lambda c: u_uv_at(c, local_t), UV_UNITS, UV_UNITS, pre)
                            if const_expr(attention_ptpc):
                                reduce_rows(UV_R, acc, emit_ptpc_scalar(UV_TILE, ld_f32(r_suv, 0)), count=1)
                            else:
                                reduce_rows(UV_R, acc, emit_out(UV_TILE), count=1)
                            stamp("uv", tt, 3)
                            gpu.barrier()
                            if tid < UV_TILE // 4:
                                r = tid * 4
                                put_bf(
                                    mb("o"),
                                    s * O_K + local_t * UV_TILE + r,
                                    [lds_ld(outs, r + j) for j in range(4)],
                                )
                            # All output readers finish before the next pass reuses outs.
                            if const_expr(UV_REUSE > 1):
                                gpu.barrier()
                stamp("uv", tt, 4)

            if const_expr(GLOBAL_UV40):
                # Every CTA has finished its merge producers before waiting here.
                # Timeline columns 6/7 use the first 64 slots/cohort for 192 global
                # projector tasks; columns0..5 remain the original merge tasks.
                if full_bid < ((total_samples + 15) // 16) * N_UV:
                    uv_task = full_bid
                    uv_first = uv_task // N_UV * 16
                    uv_count = fx.min(fx.Int32(16), total_samples - uv_first)
                    uv_t = uv_task % N_UV
                    uv_head = uv_t // (V_DIM // UV_TILE)
                    stamp("uv", bid, 6)

                    def global_uv_unit(c):
                        kc = ((wave % UV_WPR) * UV_UNITS + c) * attention_k_chunks_per_unit
                        return unit_attention(
                            r_wuv,
                            r_suv,
                            uv_t * UV_R + wave // UV_WPR,
                            kc,
                            UV_NKC,
                            KV_LORA,
                            128,
                            fx.min(lane % 16, uv_count - 1) * 260 + kc * 32,
                            scale_rows=uv_scale_rows,
                        )

                    uv_pre = [global_uv_unit(c) for c in range(UV_UNITS)]
                    for uv_batch in range_constexpr(0, 16, 4):
                        if uv_batch < uv_count:
                            uv_local = uv_batch + tid // 128
                            uv_row = uv_first + uv_local
                            uv_src = full_scratch + fx.Int64(uv_row // S) * TILE_SCRATCH_BYTES
                            uv_words = uv_merged_pair(
                                uv_src + SC["ugp"],
                                UV_MERGE_OFF + ((uv_row % S) * H + uv_head) * (KV_LORA // 2) + (tid % 128) * 2,
                            )
                            lds_st(xs, uv_local * 260 + (tid % 128) * 2, uv_words[0].bitcast(fx.Float32))
                            lds_st(xs, uv_local * 260 + (tid % 128) * 2 + 1, uv_words[1].bitcast(fx.Float32))
                    gpu.barrier()
                    uv_acc = run_units(global_uv_unit, UV_UNITS, UV_UNITS, uv_pre)
                    fx.ptr_store(fx.Vector.from_elements(uv_acc, fx.Float32), red + (wave * 64 + lane) * 4)
                    gpu.barrier()
                    if tid < uv_count * UV_TILE // 2:
                        uv_local = tid // (UV_TILE // 2)
                        uv_r = (tid % (UV_TILE // 2)) * 2
                        uv_v0, uv_v1 = fx.Float32(0.0), fx.Float32(0.0)
                        for uv_w in range_constexpr(UV_WPR):
                            uv_ww = (uv_r // 16) * UV_WPR + uv_w
                            uv_address = (uv_ww * 64 + uv_local + 16 * ((uv_r % 16) // 4)) * 4 + uv_r % 4
                            uv_v0 = uv_v0 + lds_ld(red, uv_address)
                            uv_v1 = uv_v1 + lds_ld(red, uv_address + 1)
                        uv_row = uv_first + uv_local
                        uv_dst = full_scratch + fx.Int64(uv_row // S) * TILE_SCRATCH_BYTES
                        put_bf(uv_dst + SC["o"], (uv_row % S) * O_K + uv_t * UV_TILE + uv_r, [uv_v0, uv_v1])
                    gpu.barrier()
                    stamp("uv", bid, 7)

            # ====================== 7. W_o + attention TP peer reduce + residual -> a
            # 2 row groups x 32 chunks: 4 waves per row group, 8 chunks each
            r_wo, r_so = _rsrc(w_o), _rsrc(s_o)
            O_NKC = O_K // 64
            O_R = ROW_TILE // 16
            O_WPR = WAVES // O_R
            O_UNITS = O_NKC // (attention_k_chunks_per_unit * O_WPR)
            for t in range(start("o"), N_ROW_TILES, G):
                t = fx.Int32(t)
                stamp("o", t, 0)

                def u_o(c):
                    kc = ((wave % O_WPR) * O_UNITS + c) * attention_k_chunks_per_unit
                    return unit_attention(
                        r_wo,
                        r_so,
                        t * O_R + wave // O_WPR,
                        kc,
                        O_NKC,
                        O_K,
                        128,
                        attention_b_word(n_sel() * O_K + kc * 64),
                    )

                pre = [u_o(c) for c in range(O_UNITS)]
                hint_wait(
                    S * N_UV,
                    lambda k: (
                        mb("o"),
                        (k // N_UV) * O_K + (k % N_UV) * UV_TILE + UV_TILE - 1,
                    ),
                    mark=("o", t),
                )
                if const_expr(attention_ptpc):
                    o_scales = stage_x_pairs_ptpc("o", S, O_K, lambda k: k)
                else:
                    stage_x_pairs("o", S * O_K, lambda k: k)
                stamp("o", t, 2)
                gpu.barrier()
                acc = run_units(u_o, O_UNITS, O_UNITS, pre)
                if const_expr(attention_ptpc):
                    reduce_rows(O_R, acc, emit_ptpc(ROW_TILE, t * ROW_TILE, r_so, o_scales))
                else:
                    reduce_rows(O_R, acc, emit_out(ROW_TILE))
                stamp("o", t, 3)
                gpu.barrier()

                def resid_h(s, row):
                    w = fx.Vector.from_elements(
                        [
                            fx.Int32(
                                bo.buffer_load(
                                    r_h,
                                    (s * HIDDEN + row) // 2,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                        ],
                        fx.Int32,
                    )
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    return v[0], v[1]

                if const_expr(DN_UNION_CANDIDATE and index_request_width == 6 and not indexer_cp):
                    o_owner_reduce(
                        t,
                        resid_h,
                        lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
                    )
                else:
                    peer_reduce(
                        "attn",
                        t,
                        resid_h,
                        lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
                    )
                stamp("o", t, 4)

            if const_expr(DN_UNION_CANDIDATE and index_request_width == 6 and not indexer_cp):
                # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
                # Two samples per CTA, retaining the original per-sample K-fold and rounding.
                # Reuse weights across two live MFMA columns; 192 tasks retain the original CTA base.
                r_wr = _rsrc(w_r)
                R_NKC = HIDDEN // 64
                for tt in range(start("router"), (S // 2) * N_ROUTER, G):
                    tt = fx.Int32(tt)
                    t = tt % N_ROUTER
                    router_sample = (tt // N_ROUTER) * 2
                    stamp("router", tt, 0)

                    # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
                    # columns 8..15 the second, so every loaded weight row is distinct and the
                    # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
                    r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
                    r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
                    R_CPW = R_NKC // WAVES // 2
                    r_fold = (lane % 16) // ROUTER_TILE
                    r_ns = fx.min(lane % ROUTER_TILE, 1)

                    def u_r(c):
                        kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                        return unit_bf16(
                            r_wr,
                            t * ROUTER_TILE // 16,
                            kc,
                            R_NKC,
                            (r_ns * HIDDEN + kc * 64) // 2,
                            r_ln,
                        )

                    pre = [u_r(c) for c in range(R_CPW)]
                    hint_wait(
                        N_ROW_TILES,
                        lambda k: (
                            mb("a"),
                            router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1,
                        ),
                        mark=("router", tt),
                    )
                    # this task's FP8 activation block inputs ride along with the staging loads:
                    # wave w quantizes block w * N_ROUTER + t of this CTA's sample.
                    r_gp = _rsrc(g_post)
                    x_blk = (wave % XQ_WAVES) * N_ROUTER + t
                    x_local_sample = fx.min(wave // XQ_WAVES, 1)
                    x_s = router_sample + x_local_sample
                    x_ok = (wave < 2 * XQ_WAVES) & (x_blk < XQ_BLOCKS)
                    xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
                    xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
                    xa = []

                    def ld_a(sks):
                        specs = [(mb("a"), ((router_sample + s) * HIDDEN + k) // 2, 2) for s, k in sks]
                        specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                        v = poll(specs, batch=len(specs))
                        stamp("router", tt, 5, lead=THREADS - 64)
                        xa.append(bf2_f32(v[-1][0]))
                        return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

                    rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, mark=("router", tt), count=2)
                    stamp("router", tt, 2)
                    # this task's FP8 activation blocks go out ahead of the gate GEMV
                    if x_ok:
                        x_rstd = (x_local_sample == 0).select(rstds[0], rstds[1])
                        a0, a1 = xa[0]
                        if const_expr(native_fp4_mfma):
                            q0, q1, qs = quant_mxfp8(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                        else:
                            q0, q1, qs = quant_scaled(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                        w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                        w8n = _xshfl(w8, 1)
                        if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                            put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                        d0, d1 = _fp8_roundtrip(q0, q1)
                        qscale = (qs << 23).bitcast(fx.Float32) if const_expr(native_fp4_mfma) else qs
                        bo.buffer_store(
                            fx.Vector.from_elements([d0 * qscale, d1 * qscale], fx.Float32),
                            _rsrc(mb("xqd")),
                            x_s * HIDDEN + xk,
                        )
                        if const_expr(native_fp4_mfma):
                            if lane % 16 == 0:
                                put(
                                    mb("xqs"),
                                    x_s * XQ_GROUPS + x_blk * 4 + lane // 16,
                                    qs,
                                )
                        else:
                            if lane == 0:
                                put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                    gpu.barrier()
                    acc = run_units(u_r, R_CPW, R_CPW, pre)
                    fx.ptr_store(
                        fx.Vector.from_elements(acc, fx.Float32),
                        red + (wave * 64 + lane) * 4,
                    )
                    gpu.barrier()
                    stamp("router", tt, 3)
                    if tid < 2 * ROUTER_TILE:
                        r = tid % ROUTER_TILE
                        n = tid // ROUTER_TILE
                        logit = fx.Float32(0.0)
                        for w in range_constexpr(WAVES):
                            for f in range_constexpr(2):
                                m = f * ROUTER_TILE + r
                                logit = logit + lds_ld(
                                    red,
                                    (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4,
                                )
                        put(
                            mb("scores"),
                            (router_sample + n) * N_EXPERTS + t * ROUTER_TILE + r,
                            _rcp(1.0 + _exp(-logit)),
                        )
                    stamp("router", tt, 4)

            else:
                # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
                # One sample per CTA: 1 row group x 96 chunks (bf16), 8 waves split K.
                # S > 1 gets S times as many independent router CTAs instead of serializing
                # every sample's normalization and output columns inside one CTA.
                r_wr = _rsrc(w_r)
                R_NKC = HIDDEN // 64
                for tt in range(start("router"), S * N_ROUTER, G):
                    tt = fx.Int32(tt)
                    t = tt if const_expr(S == 1) else tt % N_ROUTER
                    router_sample = fx.Int32(0) if const_expr(S == 1) else tt // N_ROUTER
                    stamp("router", tt, 0)

                    # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
                    # columns 8..15 the second, so every loaded weight row is distinct and the
                    # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
                    r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
                    r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
                    R_CPW = R_NKC // WAVES // 2
                    r_fold = (lane % 16) // ROUTER_TILE
                    r_ns = fx.Int32(0)

                    def u_r(c):
                        kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                        return unit_bf16(
                            r_wr,
                            t * ROUTER_TILE // 16,
                            kc,
                            R_NKC,
                            (r_ns * HIDDEN + kc * 64) // 2,
                            r_ln,
                        )

                    pre = [u_r(c) for c in range(R_CPW)]
                    hint_wait(
                        N_ROW_TILES,
                        lambda k: (
                            mb("a"),
                            router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1,
                        ),
                        mark=("router", tt),
                    )
                    # this task's FP8 activation block inputs ride along with the staging loads:
                    # wave w quantizes block w * N_ROUTER + t of this CTA's sample.
                    r_gp = _rsrc(g_post)
                    x_blk = wave * N_ROUTER + t
                    x_s = router_sample
                    x_ok = (wave < XQ_WAVES) & (x_blk < XQ_BLOCKS)
                    xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
                    xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
                    xa = []

                    def ld_a(sks):
                        specs = [(mb("a"), (router_sample * HIDDEN + k) // 2, 2) for s, k in sks]
                        specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                        v = poll(specs, batch=len(specs))
                        stamp("router", tt, 5, lead=THREADS - 64)
                        xa.append(bf2_f32(v[-1][0]))
                        return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

                    rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, mark=("router", tt), count=1)
                    stamp("router", tt, 2)
                    # this task's FP8 activation blocks go out ahead of the gate GEMV
                    if x_ok:
                        x_rstd = rstds[0]
                        a0, a1 = xa[0]
                        if const_expr(native_fp4_mfma):
                            q0, q1, qs = quant_mxfp8(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                        else:
                            q0, q1, qs = quant_scaled(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                        w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                        w8n = _xshfl(w8, 1)
                        if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                            put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                        d0, d1 = _fp8_roundtrip(q0, q1)
                        qscale = (qs << 23).bitcast(fx.Float32) if const_expr(native_fp4_mfma) else qs
                        bo.buffer_store(
                            fx.Vector.from_elements([d0 * qscale, d1 * qscale], fx.Float32),
                            _rsrc(mb("xqd")),
                            x_s * HIDDEN + xk,
                        )
                        if const_expr(native_fp4_mfma):
                            if lane % 16 == 0:
                                put(
                                    mb("xqs"),
                                    x_s * XQ_GROUPS + x_blk * 4 + lane // 16,
                                    qs,
                                )
                        else:
                            if lane == 0:
                                put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                    gpu.barrier()
                    acc = run_units(u_r, R_CPW, R_CPW, pre)
                    fx.ptr_store(
                        fx.Vector.from_elements(acc, fx.Float32),
                        red + (wave * 64 + lane) * 4,
                    )
                    gpu.barrier()
                    stamp("router", tt, 3)
                    if tid < ROUTER_TILE:
                        r = tid % ROUTER_TILE
                        n = fx.Int32(0)
                        logit = fx.Float32(0.0)
                        for w in range_constexpr(WAVES):
                            for f in range_constexpr(2):
                                m = f * ROUTER_TILE + r
                                logit = logit + lds_ld(
                                    red,
                                    (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4,
                                )
                        put(
                            mb("scores"),
                            router_sample * N_EXPERTS + t * ROUTER_TILE + r,
                            _rcp(1.0 + _exp(-logit)),
                        )
                    stamp("router", tt, 4)

            def dn_route(bs):
                """Expert-down routing: one whole wave per sample, in wave-sized batches."""
                for sample_batch in range_constexpr(sample_wave_batches(S)):
                    route_sample = wave + sample_batch * WAVES
                    if route_sample < S:
                        e, w = route_top8(route_sample, bs=bs)
                        if lane < MOE_SLOTS:  # slot 0: the shared expert, then pick lane (slot lane + 1)
                            q = route_sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                            lds_st(
                                keys,
                                q,
                                (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e),
                            )
                            lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

            # ================================ 9. expert up/gate + SiLU
            # 2 row groups (16 gate + 16 up rows) x 96 chunks: 4 waves per group, 24 chunks each
            UG_NKC = HIDDEN // 64
            UG_W_BYTES = 2 * expert_inter * HIDDEN // (2 if expert_mxfp4 else 1)
            UG_S_BYTES = (
                2 * expert_inter * (HIDDEN // 32)
                if expert_mxfp4
                else 2 * expert_inter // SCALE_BM * (HIDDEN // 128) * 4
            )

            if const_expr(S == 1):
                # Each CTA handles one eight-intermediate tile per task round;
                # the first routed slot also computes the shared expert. TP4 has
                # two rounds, TP8 one. All eight waves split the K dimension.
                UG8 = NATIVE_UG_TILE
                UG8_CPW = UG_NKC // WAVES
                for u in range(start("ug"), expert_inter, G):
                    u = fx.Int32(u)
                    stamp("ug", u, 0)
                    s_u, c = fx.Int32(0), u % (expert_inter // UG8)
                    has_sh = u < expert_inter // UG8
                    slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8))
                    # the FP8 activation is computed here from the post-attention state (in
                    # parallel with the router): RMSNorm, then per-128 quant with one wave per
                    # block -> X[0] (fp8 values in bf16), block scales -> misc[8:]
                    NB = XQ_BLOCKS // WAVES
                    ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                    r_gp = _rsrc(g_post)
                    gps = [(ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_]  # issued ahead of the wait
                    bs = load_bias()
                    w_rg = (
                        c if paired_ug_rows else ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                    )  # MFMA rows 0-7 gate, 8-15 up
                    w_ln = lane if paired_ug_rows else (lane & -16) | ((c % 2) * 8 + lane % 8)
                    s_rg = (lane // 32) * (expert_inter // 16) + c // 2  # this lane's output rows

                    def u_ug8(cc, e, live=None):  # expert e's weights (loads return 0 unless live)
                        nw = None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                        ns = None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                        r_wug = bo.create_buffer_resource_from_addr(
                            w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                            num_records_bytes=nw,
                        )
                        r_sug = bo.create_buffer_resource_from_addr(
                            s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                            num_records_bytes=ns,
                        )
                        unit = wave * (UG8_CPW // 2) + cc
                        kc = unit * 2
                        if const_expr(expert_mxfp4):
                            if const_expr(native_fp4_mfma):
                                return unit_native_mxfp4(
                                    r_wug,
                                    r_sug,
                                    w_rg,
                                    unit,
                                    HIDDEN,
                                    unit * 32,
                                    8 + unit * 4 + lane // 16,
                                    ln=w_ln,
                                )
                            return unit_mxfp4(
                                r_wug,
                                r_sug,
                                w_rg,
                                unit,
                                HIDDEN,
                                unit * 32,
                                lambda: _uniform_f32(lds_ld(misc, 8 + unit)),
                                w_ln,
                            )
                        wv = [
                            fx.Vector(
                                bo.buffer_load(
                                    r_wug,
                                    ((w_rg * UG_NKC + kc + h) * 64 + w_ln) * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            for h in range(2)
                        ]
                        sc = ld_f32(r_sug, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)
                        return (
                            "f8f8",
                            wv,
                            lambda: sc * _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                            kc * 16 + (lane // 16) * 4,
                        )

                    # the shared expert's weights do not depend on routing: prefetch them (the
                    # later zero-weight MMAs of the other tasks are cheaper than a branch)
                    pre = [u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh) for cc in range(UG8_CPW // 2)]
                    hint_wait(
                        N_ROW_TILES,
                        lambda k: (mb("a"), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
                        mark=("ug", u),
                    )
                    # the sum of squares takes the router's element partition and order
                    # (stage_x_rmsnorm), so rstd -- and every FP8 rounding -- is bit-identical
                    NQ4 = HIDDEN // (4 * THREADS)
                    got = poll(
                        [(mb("a"), (s_u * HIDDEN + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ4)]
                        + [(mb("a"), (s_u * HIDDEN + k) // 2, 1) for k in ks_]
                    )
                    av = [bf2_f32(w[0]) for w in got[NQ4:]]
                    ss = fx.Float32(0.0)
                    for w in got[:NQ4]:
                        for a in list(bf2_f32(w[0])) + list(bf2_f32(w[1])):
                            ss = ss + a * a
                    rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                    for j in range_constexpr(NB):
                        if const_expr(native_fp4_mfma):
                            q0, q1, qs = quant_mxfp8(
                                av[j][0] * rstd * gps[j][0],
                                av[j][1] * rstd * gps[j][1],
                            )
                            st_mxfp8(ks_[j], q0, q1)
                            if lane % 16 == 0:
                                lds_st(
                                    misc,
                                    8 + (wave + j * WAVES) * 4 + lane // 16,
                                    qs.bitcast(fx.Float32),
                                )
                        else:
                            q0, q1, qs = quant_scaled(
                                av[j][0] * rstd * gps[j][0],
                                av[j][1] * rstd * gps[j][1],
                            )
                            st_f8(ks_[j], q0, q1)
                            if lane == 0:
                                lds_st(misc, 8 + wave + j * WAVES, qs)
                    if wave == 0:
                        e, w = route_top8(s_u, bs=bs)
                        if lane == slot - 1:
                            lds_st(keys, 0, e)
                            lds_st(misc, 0, w)
                    stamp("ug", u, 2)
                    gpu.barrier()
                    e_sel = _uniform(lds_ld(keys, 0))
                    post = [u_ug8(cc, e_sel) for cc in range(UG8_CPW // 2)]
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], post),
                        lambda rl, n, v: lds_st(outs, 16 + rl, v),
                    )
                    stamp("ug", u, 3)
                    gpu.barrier()
                    if tid < UG8:  # threads 0-3: the shared expert's rows, 4-7: the routed slot's
                        r = (tid % (UG8 // 2)) * 2
                        o = (tid // (UG8 // 2)) * 16
                        g0, g1 = lds_ld(outs, o + r), lds_ld(outs, o + r + 1)
                        u0, u1 = lds_ld(outs, o + UG8 + r), lds_ld(outs, o + UG8 + r + 1)
                        if has_sh | (tid >= UG8 // 2):
                            put2(
                                mb("mid"),
                                (tid < UG8 // 2).select(fx.Int32(0), slot) * expert_inter + c * UG8 + r,
                                g0 * _rcp(1.0 + _exp(-g0)) * u0,
                                g1 * _rcp(1.0 + _exp(-g1)) * u1,
                            )
                    if (c == 0) & (tid == 0):  # routing record (debug / tests)
                        put(mb("sel"), slot, e_sel)
                        put(mb("prob"), slot, lds_ld(misc, 0))
                        if has_sh:
                            put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                            put(mb("prob"), 0, fx.Float32(1.0))
                    stamp("ug", u, 4)
                    if const_expr(expert_inter > G):
                        gpu.barrier()
            elif const_expr(S > 1):
                # One eight-intermediate tile per CTA and task round. Shared-expert weights
                # feed all sample columns of one MFMA, while routed-expert weights are
                # prefetched one sample ahead.  This avoids the segmented partial
                # tiles and mailbox reduction used by the older S=2/4 schedule.
                UG8 = NATIVE_UG_TILE
                UG8_UNITS = (HIDDEN // 128) // WAVES
                XW = UG_X_ROW

                def ug8_units(e, sample, c, live=None):
                    w_rg = c if paired_ug_rows else ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                    w_ln = lane if paired_ug_rows else (lane & -16) | ((c % 2) * 8 + lane % 8)
                    s_rg = (lane // 32) * (expert_inter // 16) + c // 2
                    if const_expr(live is None):
                        rw = _rsrc(w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES))
                        rs = _rsrc(s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES))
                    else:
                        rw = bo.create_buffer_resource_from_addr(
                            w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                            num_records_bytes=live.select(fx.Int32(UG_W_BYTES), fx.Int32(0)),
                        )
                        rs = bo.create_buffer_resource_from_addr(
                            s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                            num_records_bytes=live.select(fx.Int32(UG_S_BYTES), fx.Int32(0)),
                        )
                    sn = n_sel() if sample is None else fx.Int32(sample)
                    units = []
                    for cc in range_constexpr(UG8_UNITS):
                        unit = wave * UG8_UNITS + cc
                        kc = unit * 2
                        if const_expr(expert_mxfp4):
                            if const_expr(native_fp4_mfma):
                                units.append(
                                    unit_native_mxfp4(
                                        rw,
                                        rs,
                                        w_rg,
                                        unit,
                                        HIDDEN,
                                        sn * XW + unit * 32,
                                        8 + sn * UG_SCALE_ROW + unit * 4 + lane // 16,
                                        ln=w_ln,
                                    )
                                )
                            else:
                                units.append(
                                    unit_mxfp4(
                                        rw,
                                        rs,
                                        w_rg,
                                        unit,
                                        HIDDEN,
                                        sn * XW + unit * 32,
                                        lambda unit=unit, sn=sn: lds_ld(misc, 8 + sn * XQ_BLOCKS + unit),
                                        w_ln,
                                    )
                                )
                            continue
                        wv = [
                            fx.Vector(
                                bo.buffer_load(
                                    rw,
                                    ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            for j in range(2)
                        ]
                        sc = ld_f32(rs, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)
                        units.append(
                            (
                                "f8f8",
                                wv,
                                lambda sc=sc, kc=kc, sn=sn: (sc * lds_ld(misc, 8 + sn * XQ_BLOCKS + kc // 2)),
                                sn * XW + kc * 16 + (lane // 16) * 4,
                            )
                        )
                    return units

                dn_route(load_bias())
                gpu.barrier()
                stage_ug_first = atom_experts or (expert_mxfp4 and S > 4)
                if const_expr(stage_ug_first and not GLOBAL_UG40):
                    stage_xq(list(range(S)))
                    gpu.barrier()
                if const_expr(GLOBAL_UG40):
                    # Publish the original route once per cohort before global work.
                    if (bid == 0) & (tid < S * MOE_SLOTS):
                        put(mb("sel"), tid, lds_ld(keys, tid))
                        put(mb("prob"), tid, lds_ld(dnw, tid))
                    gpu.barrier()
                    if tid < total_samples * MOE_SLOTS:
                        gu_row = tid // MOE_SLOTS
                        gu_local = (gu_row % S) * MOE_SLOTS + tid % MOE_SLOTS
                        gu_scratch = full_scratch + fx.Int64(gu_row // S) * TILE_SCRATCH_BYTES
                        gu_route = get(gu_scratch + fx.Int64(SC["sel"]), gu_local)
                        gu_prob = get(gu_scratch + fx.Int64(SC["prob"]), gu_local)
                        lds_st(keys, tid, gu_route)
                        lds_st(dnw, tid, gu_prob.bitcast(fx.Float32))
                    gpu.barrier()
                    gu_experts = keys + total_samples * MOE_SLOTS
                    gu_lo = gu_experts + N_EXPERTS + 1
                    gu_hi = gu_lo + N_EXPERTS + 1
                    gu_counts = gu_hi + N_EXPERTS + 1
                    gu_selected = gu_counts + N_EXPERTS // 64 + 1
                    gu_red = fx.recast_iter(fx.Int32, red)
                    for gu_i in range_constexpr(2 * WAVES * N_EXPERTS // THREADS):
                        lds_st(gu_red, tid + gu_i * THREADS, fx.Int32(0))
                    gpu.barrier()
                    for gu_batch in range_constexpr((total_samples + WAVES - 1) // WAVES):
                        gu_sample = wave + gu_batch * WAVES
                        if (gu_sample < total_samples) & (lane < TOP_K):
                            gu_expert = fx.Int32(lds_ld(keys, gu_sample * MOE_SLOTS + lane + 1))
                            gu_address = (gu_sample // 32) * WAVES * N_EXPERTS + wave * N_EXPERTS + gu_expert
                            lds_st(gu_red, gu_address, lds_ld(gu_red, gu_address) | (fx.Int32(1) << (gu_sample % 32)))
                        gpu.barrier()
                    gu_mask_lo, gu_mask_hi = fx.Int32(0), fx.Int32(0)
                    if tid < N_EXPERTS:
                        for gu_w in range_constexpr(WAVES):
                            gu_mask_lo = gu_mask_lo | lds_ld(gu_red, gu_w * N_EXPERTS + tid)
                            gu_mask_hi = gu_mask_hi | lds_ld(gu_red, (WAVES + gu_w) * N_EXPERTS + tid)
                    gu_ballot = fx.Int64(rocdl.ballot(T.i64, (gu_mask_lo | gu_mask_hi) != 0))
                    gu_below = fx.Int32(fmath.ctpop(gu_ballot & ((fx.Int64(1) << fx.Int64(lane)) - 1)))
                    gu_count = fx.Int32(fmath.ctpop(gu_ballot))
                    if (tid < N_EXPERTS) & (lane == 0):
                        lds_st(gu_counts, wave, gu_count)
                    gpu.barrier()
                    gu_offset = fx.Int32(0)
                    for gu_w in range_constexpr(N_EXPERTS // 64):
                        gu_offset = gu_offset + (wave > gu_w).select(fx.Int32(lds_ld(gu_counts, gu_w)), fx.Int32(0))
                    if (tid < N_EXPERTS) & ((gu_mask_lo | gu_mask_hi) != 0):
                        lds_st(gu_experts, gu_offset + gu_below, tid)
                        lds_st(gu_lo, gu_offset + gu_below, gu_mask_lo)
                        lds_st(gu_hi, gu_offset + gu_below, gu_mask_hi)
                    if tid == N_EXPERTS - 1:
                        gu_total = gu_offset + gu_count
                        lds_st(gu_experts, gu_total, fx.Int32(SHARED_EXPERT))
                        lds_st(gu_lo, gu_total, fx.Int32(-1))
                        lds_st(gu_hi, gu_total, fx.Int32((1 << (total_samples - 32)) - 1))
                        lds_st(gu_counts, N_EXPERTS // 64, gu_total + 1)
                    gpu.barrier()
                    gu_nunion = _uniform(lds_ld(gu_counts, N_EXPERTS // 64))
                    # One global task contains one <=S-token chunk, including shared.
                    # The retired reduction arena holds only prefix-scan totals here.
                    gu_chunk_map = gu_selected + total_samples
                    gu_prefix_index = fx.min(tid, gu_nunion - 1)
                    gu_prefix_bits = fx.Int64(fx.Uint32(lds_ld(gu_lo, gu_prefix_index))) | (
                        fx.Int64(fx.Uint32(lds_ld(gu_hi, gu_prefix_index))) << 32
                    )
                    gu_chunk_count = (tid < gu_nunion).select(
                        (fx.Int32(fmath.ctpop(gu_prefix_bits)) + S - 1) // S, fx.Int32(0)
                    )
                    gu_chunk_scan = gu_chunk_count
                    for gu_shift in range_constexpr(6):
                        gu_distance = 1 << gu_shift
                        gu_previous = bpermute_i32(fx.max(lane - gu_distance, fx.Int32(0)) * 4, gu_chunk_scan)
                        gu_chunk_scan = gu_chunk_scan + (lane >= gu_distance).select(gu_previous, fx.Int32(0))
                    if lane == 63:
                        lds_st(gu_red, wave, gu_chunk_scan)
                    gpu.barrier()
                    gu_chunk_offset = gu_chunk_scan - gu_chunk_count
                    for gu_scan_wave in range_constexpr(WAVES):
                        gu_chunk_offset = gu_chunk_offset + (wave > gu_scan_wave).select(
                            lds_ld(gu_red, gu_scan_wave), fx.Int32(0)
                        )
                    if tid == THREADS - 1:
                        lds_st(gu_red, WAVES, gu_chunk_offset + gu_chunk_count)
                    for gu_chunk_id in range_constexpr((total_samples + S - 1) // S):
                        if gu_chunk_id < gu_chunk_count:
                            lds_st(gu_chunk_map, gu_chunk_offset + gu_chunk_id, tid * ROW_TILES + gu_chunk_id)
                    gpu.barrier()
                    gu_nchunks = _uniform(lds_ld(gu_red, WAVES))
                    gu_tiles = expert_inter // UG8
                    for gu_task in range(full_bid, gu_nchunks * (gu_tiles // (GLOBAL_UG_WIDTH // 8)), BLOCKS):
                        gu_task = fx.Int32(gu_task)
                        gu_chunk_record = _uniform(
                            lds_ld(gu_chunk_map, gu_task // (gu_tiles // (GLOBAL_UG_WIDTH // 8)))
                        )
                        gu_union = gu_chunk_record // ROW_TILES
                        gu_segment = gu_chunk_record % ROW_TILES
                        gu_cbase = (gu_task % (gu_tiles // (GLOBAL_UG_WIDTH // 8))) * (GLOBAL_UG_WIDTH // 8)
                        gu_stamp_base = (gu_task // BLOCKS * G + bid) * (GLOBAL_UG_WIDTH // 8)
                        gu_expert = _uniform(lds_ld(gu_experts, gu_union))
                        gu_bits = fx.Int64(fx.Uint32(_uniform(lds_ld(gu_lo, gu_union)))) | (
                            fx.Int64(fx.Uint32(_uniform(lds_ld(gu_hi, gu_union)))) << 32
                        )
                        gu_nselected = fx.Int32(fmath.ctpop(gu_bits))
                        if tid < total_samples:
                            gu_bit = fx.Int64(1) << fx.Int64(tid)
                            if (gu_bits & gu_bit) != 0:
                                gu_index = fx.Int32(fmath.ctpop(gu_bits & (gu_bit - 1)))
                                lds_st(gu_selected, gu_index, tid)
                        gpu.barrier()
                        gu_valid = fx.min(fx.Int32(S), gu_nselected - gu_segment * S)
                        gu_tokens = [
                            _uniform(lds_ld(gu_selected, gu_segment * S + fx.min(fx.Int32(gu_col_j), gu_valid - 1)))
                            for gu_col_j in range(S)
                        ]
                        gu_bases = [
                            full_scratch + fx.Int64(gu_xtoken // S) * TILE_SCRATCH_BYTES for gu_xtoken in gu_tokens
                        ]
                        gu_nxw = HIDDEN // 4 // THREADS
                        gu_got = poll(
                            [
                                (
                                    gu_xbase + fx.Int64(SC["xq"]),
                                    (gu_xtoken % S) * (HIDDEN // 4) + tid + gu_load_i * THREADS,
                                    1,
                                )
                                for gu_xtoken, gu_xbase in zip(gu_tokens, gu_bases)
                                for gu_load_i in range(gu_nxw)
                            ]
                            + [
                                (
                                    gu_xbase + fx.Int64(SC["xqs"]),
                                    (gu_xtoken % S) * XQ_GROUPS + fx.min(tid, XQ_GROUPS - 1),
                                    1,
                                )
                                for gu_xtoken, gu_xbase in zip(gu_tokens, gu_bases)
                            ]
                        )
                        for gu_j in range_constexpr(S):
                            for gu_i in range_constexpr(gu_nxw):
                                lds_st(
                                    xs,
                                    gu_j * UG_X_ROW + tid + gu_i * THREADS,
                                    gu_got[gu_j * gu_nxw + gu_i][0].bitcast(fx.Float32),
                                )
                            if tid < XQ_GROUPS:
                                lds_st(
                                    misc,
                                    8 + gu_j * UG_SCALE_ROW + tid,
                                    gu_got[S * gu_nxw + gu_j][0].bitcast(fx.Float32),
                                )
                        gpu.barrier()
                        # Reuse selected activations across the configured output tile.
                        for gu_pair in range_constexpr(GLOBAL_UG_WIDTH // 16):
                            gu_units = [
                                ug8_units(gu_expert, None, gu_cbase + gu_pair * 2 + gu_part) for gu_part in range(2)
                            ]
                            for gu_part in range_constexpr(2):
                                stamp("ug", gu_stamp_base + gu_pair * 2 + gu_part, 0)
                                gu_acc = mma_units([fx.Float32(0.0) for _ in range(4)], gu_units[gu_part])
                                fx.ptr_store(
                                    fx.Vector.from_elements(gu_acc, fx.Float32),
                                    red + ((gu_part * WAVES + wave) * 64 + lane) * 4,
                                )
                            gpu.barrier()
                            if tid < S * 32:
                                gu_part = tid // (S * 16)
                                gu_row = tid % (S * 16)
                                gu_n, gu_r = gu_row // 16, gu_row % 16
                                gu_sum = fx.Float32(0.0)
                                for gu_w in range_constexpr(WAVES):
                                    gu_sum = gu_sum + lds_ld(
                                        red, ((gu_part * WAVES + gu_w) * 64 + gu_n + 16 * (gu_r // 4)) * 4 + gu_r % 4
                                    )
                                lds_st(outs, tid, gu_sum)
                            gpu.barrier()
                            if tid < gu_valid * UG8:
                                gu_sample = tid // UG8
                                gu_element = (tid % UG8) * 2
                                gu_part, gu_r = gu_element // UG8, gu_element % UG8
                                gu_token = fx.Int32(lds_ld(gu_selected, gu_segment * S + gu_sample))
                                gu_slot = fx.Int32(0)
                                for gu_j in range_constexpr(1, MOE_SLOTS):
                                    gu_slot = (fx.Int32(lds_ld(keys, gu_token * MOE_SLOTS + gu_j)) == gu_expert).select(
                                        fx.Int32(gu_j), gu_slot
                                    )
                                gu_o = gu_part * (S * 16) + gu_sample * 16 + gu_r
                                gu_g0, gu_g1 = lds_ld(outs, gu_o), lds_ld(outs, gu_o + 1)
                                gu_v0, gu_v1 = lds_ld(outs, gu_o + UG8), lds_ld(outs, gu_o + UG8 + 1)
                                gu_dst = (
                                    full_scratch + fx.Int64(gu_token // S) * TILE_SCRATCH_BYTES + fx.Int64(SC["mid"])
                                )
                                put2(
                                    gu_dst,
                                    ((gu_token % S) * MOE_SLOTS + gu_slot) * expert_inter
                                    + (gu_cbase + gu_pair * 2 + gu_part) * UG8
                                    + gu_r,
                                    gu_g0 * _rcp(1.0 + _exp(-gu_g0)) * gu_v0,
                                    gu_g1 * _rcp(1.0 + _exp(-gu_g1)) * gu_v1,
                                )
                            gpu.barrier()
                            for gu_part in range_constexpr(2):
                                stamp("ug", gu_stamp_base + gu_pair * 2 + gu_part, 4)
                    # Restore this CTA's original cohort-local route layout for down.
                    gu_my_key = lds_ld(keys, fx.min(tid, S * MOE_SLOTS - 1) + row_tile * S * MOE_SLOTS)
                    gu_my_prob = lds_ld(dnw, fx.min(tid, S * MOE_SLOTS - 1) + row_tile * S * MOE_SLOTS)
                    gpu.barrier()
                    if tid < S * MOE_SLOTS:
                        lds_st(keys, tid, gu_my_key)
                        lds_st(dnw, tid, gu_my_prob)
                    gpu.barrier()
                    if const_expr(DN_UNION_ROWS40):
                        # Rebuild cohort-local expert metadata expected by ordered down union;
                        # wave-local masks only occupy the otherwise-idle reduction workspace.
                        union_experts = keys + S * MOE_SLOTS
                        union_masks = union_experts + UG_UNION_MAX
                        union_counts = union_masks + UG_UNION_MAX
                        mask_red = fx.recast_iter(fx.Int32, red)
                        for i in range_constexpr(WAVES * N_EXPERTS // THREADS):
                            lds_st(mask_red, tid + i * THREADS, fx.Int32(0))
                        gpu.barrier()
                        for batch in range_constexpr(sample_wave_batches(S)):
                            sample = wave + batch * WAVES
                            if (sample < S) & (lane < TOP_K):
                                expert = fx.Int32(lds_ld(keys, sample * MOE_SLOTS + lane + 1))
                                address = wave * N_EXPERTS + expert
                                previous = lds_ld(mask_red, address)
                                lds_st(mask_red, address, previous | (fx.Int32(1) << sample))
                            gpu.barrier()
                        mask = fx.Int32(0)
                        if tid < N_EXPERTS:
                            for w in range_constexpr(WAVES):
                                mask = mask | lds_ld(mask_red, w * N_EXPERTS + tid)
                        ballot = fx.Int64(rocdl.ballot(T.i64, mask != 0))
                        below = fx.Int32(fmath.ctpop(ballot & ((fx.Int64(1) << fx.Int64(lane)) - 1)))
                        count = fx.Int32(fmath.ctpop(ballot))
                        if (tid < N_EXPERTS) & (lane == 0):
                            lds_st(union_counts, wave, count)
                        gpu.barrier()
                        offset = fx.Int32(0)
                        for w in range_constexpr(N_EXPERTS // 64):
                            offset = offset + (wave > w).select(fx.Int32(lds_ld(union_counts, w)), fx.Int32(0))
                        if (tid < N_EXPERTS) & (mask != 0):
                            lds_st(union_experts, offset + below, tid)
                            lds_st(union_masks, offset + below, mask)
                        if tid == N_EXPERTS - 1:
                            total = offset + count
                            lds_st(union_experts, total, fx.Int32(SHARED_EXPERT))
                            lds_st(union_masks, total, fx.Int32((1 << S) - 1))
                            lds_st(union_counts, N_EXPERTS // 64, total + 1)
                        gpu.barrier()
                elif const_expr(GROUPED_UG):
                    # Reuse the indexer's retired key arena for the compact union and masks;
                    # wave-local masks only occupy the otherwise-idle reduction workspace.
                    union_experts = keys + S * MOE_SLOTS
                    union_masks = union_experts + UG_UNION_MAX
                    union_counts = union_masks + UG_UNION_MAX
                    mask_red = fx.recast_iter(fx.Int32, red)
                    for i in range_constexpr(WAVES * N_EXPERTS // THREADS):
                        lds_st(mask_red, tid + i * THREADS, fx.Int32(0))
                    gpu.barrier()
                    for batch in range_constexpr(sample_wave_batches(S)):
                        sample = wave + batch * WAVES
                        if (sample < S) & (lane < TOP_K):
                            expert = fx.Int32(lds_ld(keys, sample * MOE_SLOTS + lane + 1))
                            address = wave * N_EXPERTS + expert
                            previous = lds_ld(mask_red, address)
                            lds_st(mask_red, address, previous | (fx.Int32(1) << sample))
                        gpu.barrier()
                    mask = fx.Int32(0)
                    if tid < N_EXPERTS:
                        for w in range_constexpr(WAVES):
                            mask = mask | lds_ld(mask_red, w * N_EXPERTS + tid)
                    ballot = fx.Int64(rocdl.ballot(T.i64, mask != 0))
                    below = fx.Int32(fmath.ctpop(ballot & ((fx.Int64(1) << fx.Int64(lane)) - 1)))
                    count = fx.Int32(fmath.ctpop(ballot))
                    if (tid < N_EXPERTS) & (lane == 0):
                        lds_st(union_counts, wave, count)
                    gpu.barrier()
                    offset = fx.Int32(0)
                    for w in range_constexpr(N_EXPERTS // 64):
                        offset = offset + (wave > w).select(fx.Int32(lds_ld(union_counts, w)), fx.Int32(0))
                    if (tid < N_EXPERTS) & (mask != 0):
                        lds_st(union_experts, offset + below, tid)
                        lds_st(union_masks, offset + below, mask)
                    if tid == N_EXPERTS - 1:
                        total = offset + count
                        lds_st(union_experts, total, fx.Int32(SHARED_EXPERT))
                        lds_st(union_masks, total, fx.Int32((1 << S) - 1))
                        lds_st(union_counts, N_EXPERTS // 64, total + 1)
                    gpu.barrier()
                    n_union = _uniform(lds_ld(union_counts, N_EXPERTS // 64))

                    # This native body writes UG8 elements, unlike the generic UG_TILE=16 path.
                    # Cover the complete intermediate row so every down mailbox gets published.
                    grouped_tiles = expert_inter // UG8
                    # Only populated union entries execute MFMA; preserve per-expert K order.
                    if const_expr(S == 12 or COHORT_ROW_TILES):
                        assert grouped_tiles % 2 == 0
                        ugpair_tiles = grouped_tiles // 2
                        for ugpair_task in range(start("ug"), n_union * ugpair_tiles, G):
                            ugpair_task = fx.Int32(ugpair_task)
                            ugpair_union = ugpair_task // ugpair_tiles
                            ugpair_cbase = (ugpair_task % ugpair_tiles) * 2
                            ugpair_expert = _uniform(lds_ld(union_experts, ugpair_union))
                            ugpair_mask = _uniform(lds_ld(union_masks, ugpair_union))
                            ugpair_units = [ug8_units(ugpair_expert, None, ugpair_cbase + p) for p in range(2)]
                            for ugpair_part in range_constexpr(2):
                                ugpair_u = ugpair_union * grouped_tiles + ugpair_cbase + ugpair_part
                                stamp("ug", ugpair_u, 0)
                                ugpair_acc = mma_units(
                                    [fx.Float32(0.0) for _ in range(4)],
                                    ugpair_units[ugpair_part],
                                )
                                fx.ptr_store(
                                    fx.Vector.from_elements(ugpair_acc, fx.Float32),
                                    red + ((ugpair_part * WAVES + wave) * 64 + lane) * 4,
                                )
                            gpu.barrier()
                            # Two independent reductions, each retaining the exact original
                            # wave0..7 addition order; one synchronization covers both tiles.
                            if tid < S * 32:
                                ugpair_part = tid // (S * 16)
                                ugpair_row = tid % (S * 16)
                                ugpair_n = ugpair_row // 16
                                ugpair_r = ugpair_row % 16
                                ugpair_total = fx.Float32(0.0)
                                for ugpair_w in range_constexpr(WAVES):
                                    ugpair_total = ugpair_total + lds_ld(
                                        red,
                                        ((ugpair_part * WAVES + ugpair_w) * 64 + ugpair_n + 16 * (ugpair_r // 4)) * 4
                                        + ugpair_r % 4,
                                    )
                                lds_st(outs, tid, ugpair_total)
                            gpu.barrier()
                            if tid < S * UG8:
                                ugpair_sample = tid // UG8
                                ugpair_element = (tid % UG8) * 2
                                ugpair_part = ugpair_element // UG8
                                ugpair_r = ugpair_element % UG8
                                if ((ugpair_mask >> ugpair_sample) & 1) != 0:
                                    ugpair_slot = fx.Int32(0)
                                    for ugpair_j in range_constexpr(1, MOE_SLOTS):
                                        ugpair_slot = (
                                            fx.Int32(
                                                lds_ld(
                                                    keys,
                                                    ugpair_sample * MOE_SLOTS + ugpair_j,
                                                )
                                            )
                                            == ugpair_expert
                                        ).select(fx.Int32(ugpair_j), ugpair_slot)
                                    ugpair_o = ugpair_part * (S * 16) + ugpair_sample * 16 + ugpair_r
                                    ugpair_g0, ugpair_g1 = lds_ld(outs, ugpair_o), lds_ld(outs, ugpair_o + 1)
                                    ugpair_v0, ugpair_v1 = lds_ld(outs, ugpair_o + UG8), lds_ld(
                                        outs, ugpair_o + UG8 + 1
                                    )
                                    ugpair_c = ugpair_cbase + ugpair_part
                                    put2(
                                        mb("mid"),
                                        (ugpair_sample * MOE_SLOTS + ugpair_slot) * expert_inter
                                        + ugpair_c * UG8
                                        + ugpair_r,
                                        ugpair_g0 * _rcp(1.0 + _exp(-ugpair_g0)) * ugpair_v0,
                                        ugpair_g1 * _rcp(1.0 + _exp(-ugpair_g1)) * ugpair_v1,
                                    )
                                    if (ugpair_c == 0) & (ugpair_r == 0):
                                        put(
                                            mb("sel"),
                                            ugpair_sample * MOE_SLOTS + ugpair_slot,
                                            ugpair_expert,
                                        )
                                        put(
                                            mb("prob"),
                                            ugpair_sample * MOE_SLOTS + ugpair_slot,
                                            lds_ld(
                                                dnw,
                                                ugpair_sample * MOE_SLOTS + ugpair_slot,
                                            ),
                                        )
                            for ugpair_part in range_constexpr(2):
                                stamp(
                                    "ug",
                                    ugpair_union * grouped_tiles + ugpair_cbase + ugpair_part,
                                    4,
                                )
                            gpu.barrier()
                    else:
                        for task in range(start("ug"), n_union * grouped_tiles, G):
                            u = fx.Int32(task)
                            c = u % grouped_tiles
                            expert = _uniform(lds_ld(union_experts, u // grouped_tiles))
                            token_mask = _uniform(lds_ld(union_masks, u // grouped_tiles))
                            stamp("ug", u, 0)
                            units = ug8_units(expert, None, c)
                            reduce_rows(
                                1,
                                mma_units([fx.Float32(0.0) for _ in range(4)], units),
                                emit_out(16),
                            )
                            gpu.barrier()
                            if tid < S * UG8 // 2:
                                sample = tid // (UG8 // 2)
                                r = (tid % (UG8 // 2)) * 2
                                if ((token_mask >> sample) & 1) != 0:
                                    slot = fx.Int32(0)
                                    for j in range_constexpr(1, MOE_SLOTS):
                                        slot = (fx.Int32(lds_ld(keys, sample * MOE_SLOTS + j)) == expert).select(
                                            fx.Int32(j), slot
                                        )
                                    g0, g1 = lds_ld(outs, sample * 16 + r), lds_ld(outs, sample * 16 + r + 1)
                                    v0, v1 = lds_ld(outs, sample * 16 + UG8 + r), lds_ld(
                                        outs, sample * 16 + UG8 + r + 1
                                    )
                                    put2(
                                        mb("mid"),
                                        (sample * MOE_SLOTS + slot) * expert_inter + c * UG8 + r,
                                        g0 * _rcp(1.0 + _exp(-g0)) * v0,
                                        g1 * _rcp(1.0 + _exp(-g1)) * v1,
                                    )
                                    if (c == 0) & (r == 0):
                                        put(mb("sel"), sample * MOE_SLOTS + slot, expert)
                                        put(
                                            mb("prob"),
                                            sample * MOE_SLOTS + slot,
                                            lds_ld(dnw, sample * MOE_SLOTS + slot),
                                        )
                            stamp("ug", u, 4)
                            gpu.barrier()
                else:
                    u0 = start("ug")
                    for task_round in range_constexpr(UG_TASK_ROUNDS):
                        u = fx.Int32(u0 + task_round * G)
                        c = u % (expert_inter // UG8)
                        has_sh = u < expert_inter // UG8
                        slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8))

                        def ug8_emit(sample, shared):
                            if tid < (S if shared else 1) * UG8 // 2:
                                n = tid // (UG8 // 2)
                                r = (tid % (UG8 // 2)) * 2
                                g0, g1 = (
                                    lds_ld(outs, n * 16 + r),
                                    lds_ld(outs, n * 16 + r + 1),
                                )
                                v0 = lds_ld(outs, n * 16 + UG8 + r)
                                v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                                sn = n if shared else fx.Int32(sample)
                                sl = fx.Int32(0) if shared else slot
                                put2(
                                    mb("mid"),
                                    (sn * MOE_SLOTS + sl) * expert_inter + c * UG8 + r,
                                    g0 * _rcp(1.0 + _exp(-g0)) * v0,
                                    g1 * _rcp(1.0 + _exp(-g1)) * v1,
                                )
                            if (c == 0) & (tid < S if shared else tid == 0):
                                sn = tid if shared else fx.Int32(sample)
                                sl = fx.Int32(0) if shared else slot
                                put(
                                    mb("sel"),
                                    sn * MOE_SLOTS + sl,
                                    lds_ld(keys, sn * MOE_SLOTS + sl),
                                )
                                put(
                                    mb("prob"),
                                    sn * MOE_SLOTS + sl,
                                    lds_ld(dnw, sn * MOE_SLOTS + sl),
                                )

                        shared_pre = ug8_units(fx.Int32(SHARED_EXPERT), None, c, has_sh)
                        cur = ug8_units(_uniform(lds_ld(keys, slot)), 0, c)
                        if const_expr(not stage_ug_first and task_round == 0):
                            # Native weights can be prefetched while router activations
                            # arrive. ATOM tiles retain their lower-register staging order.
                            stage_xq(list(range(S)))
                            gpu.barrier()
                        if has_sh:
                            reduce_rows(
                                1,
                                mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre),
                                emit_out(16),
                            )
                            gpu.barrier()
                            ug8_emit(0, True)
                        for sample in range_constexpr(S):
                            stamp("ug", sample * expert_inter + u, 0)
                            pre = cur
                            if const_expr(sample + 1 < S):
                                cur = ug8_units(
                                    _uniform(lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)),
                                    sample + 1,
                                    c,
                                )
                            reduce_rows(
                                1,
                                mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                                emit_out(16),
                            )
                            gpu.barrier()
                            ug8_emit(sample, False)
                            stamp("ug", sample * expert_inter + u, 4)
                        if const_expr(task_round + 1 < UG_TASK_ROUNDS):
                            gpu.barrier()

            if const_expr(GROUPED_UG):
                # One owner wave per 128-value block; all producers remain resident.
                # Preserve UG8 scheduling and original FP32->MXFP8 arithmetic, while
                # removing the same quantization from every down CTA.
                for mid_chunk in range(bid * WAVES + wave, S * MOE_SLOTS * expert_inter // 128, G * WAVES):
                    mid_chunk = fx.Int32(mid_chunk)
                    a0, a1 = get2(mb("mid"), mid_chunk * 128 + lane * 2)
                    q0, q1, mid_exponent = quant_mxfp8(a0, a1)
                    packed_word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                    packed_neighbor = _xshfl(packed_word, 1)
                    if lane % 2 == 0:
                        put(
                            mb("ugp"),
                            mid_chunk * 32 + lane // 2,
                            packed_word | (packed_neighbor << 16),
                        )
                    if lane % 16 == 0:
                        put(
                            mb("ugp"),
                            MIDQ_WORDS + mid_chunk * 4 + lane // 16,
                            mid_exponent,
                        )
                gpu.barrier()

            # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
            # 2 row groups x (sample tile * 9 slots * 4) chunks: 4 waves per group.
            # S=8 is evaluated as two four-sample groups so its FP8 mid tile and
            # in-flight weight batches fit comfortably in LDS/VGPRs.
            DN_NKC = expert_inter // 64
            DN_R = (DN_TILE + 15) // 16  # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
            DN_WPR = WAVES // DN_R
            DN_BATCH = down_prefetch_batch(S, native_fp4_mfma, atom_experts, expert_inter)
            DN_W_BYTES = HIDDEN * expert_inter // (2 if expert_mxfp4 else 1)
            DN_S_BYTES = (
                HIDDEN * (expert_inter // 32) if expert_mxfp4 else HIDDEN // SCALE_BM * (expert_inter // 128) * 4
            )

            def dn_union_tile(t):

                mode = "union_lut4"
                hidden, INTER, SLOTS = HIDDEN, 512, DU_SLOTS
                XROW, SCROW, UNION_MAX = DU_XROW, DU_SCROW, DU_UMAX
                MSOFF, KEYOFF, DWOFF = DU_MSOFF, DU_KEYOFF, DU_DWOFF
                CROFF, CWORDS = DU_CROFF, DU_CWORDS
                gu, gw = wave // 4, wave % 4
                tile = t
                arena = lds.work.ptr
                old_keys = fx.recast_iter(fx.Int32, arena + KEYS_OFF)
                old_dnw = arena + DNW_OFF
                incoming_key = lds_ld(old_keys, fx.min(tid, DU_KEY_WORDS - 1))
                incoming_prob = lds_ld(old_dnw, fx.min(tid, DU_ROUTE_WORDS - 1))
                gpu.barrier()
                xs, misc = arena, arena + MSOFF
                keys = fx.recast_iter(fx.Int32, arena + KEYOFF)
                dnw, contributions = arena + DWOFF, arena + CROFF
                red = contributions

                def ld(p, i):
                    return fx.ptr_load(p + i)

                def st(p, i, v):
                    fx.ptr_store(v, p + i)

                def contribution_index(token, slot, kc, row):
                    if const_expr(mode in ("union_transpose", "union_batch4", "union_lut4")):
                        return ((slot * 4 + kc) * 32 + row) * S + token
                    return ((token * SLOTS + slot) * 4 + kc) * 32 + row

                def unit(expert, slot, kc, coefficient):
                    sn = fx.min(lane % 16, S - 1)
                    row = sn * SLOTS + slot
                    wb = _rsrc(w_dn + fx.Int64(expert) * fx.Int64(hidden * INTER // 2))
                    sb = _rsrc(s_dn + fx.Int64(expert) * fx.Int64(hidden * INTER // 32))
                    rg = tile * 2 + gu
                    raw = fx.Vector(
                        bo.buffer_load(
                            wb,
                            (rg * 8 + kc * 2) * 128 + lane * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    scale = fx.Int32(
                        bo.buffer_load(
                            sb,
                            atom_mxfp4_scale_index(rg * 16 + lane % 16, kc * 4 + lane // 16, 16),
                            vec_width=1,
                            dtype=T.i8,
                        )
                    )
                    return (
                        raw,
                        scale,
                        row * XROW + kc * 32,
                        row * SCROW + kc * 4 + lane // 16,
                        coefficient,
                    )

                def partial(u):
                    raw, scale, bw, sw, coefficient = u
                    words = [
                        fx.Vector(fx.ptr_load(xs + bw + 16 * half + 4 * (lane // 16), result_type=v4f)).bitcast(
                            fx.Int32
                        )
                        for half in range(2)
                    ]
                    b = fx.Vector.from_elements(
                        [words[half][e] for half in range(2) for e in range(4)],
                        fx.Int32,
                    )
                    asc = ld(misc, sw).bitcast(fx.Int32)
                    zero = fx.Vector.filled(4, 0.0, fx.Float32)
                    return fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [raw, b, zero, 4, 0, 0, scale, 0, asc])
                    )

                if tid < DU_KEY_WORDS:
                    st(keys, tid, incoming_key)
                if tid < DU_ROUTE_WORDS:
                    st(dnw, tid, incoming_prob)
                gpu.barrier()

                MID_PAIR_ROUNDS = (MIDQ_WORDS + 2 * THREADS - 1) // (2 * THREADS)
                for pack_base in range_constexpr(0, MID_PAIR_ROUNDS, 4):
                    packed = poll(
                        [
                            (
                                mb("ugp"),
                                fx.min((tid + j * THREADS) * 2, MIDQ_WORDS - 2),
                                2,
                            )
                            for j in range(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS))
                        ]
                    )
                    if const_expr(pack_base == 0):
                        stamp("down", t, 2)
                    for j in range_constexpr(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS)):
                        word = (tid + j * THREADS) * 2
                        if word < MIDQ_WORDS:
                            target_word = word
                            if const_expr(GROUPED_DN):
                                target_word = (word // (expert_inter // 4)) * DN_X_ROW + word % (expert_inter // 4)
                            st(
                                xs,
                                target_word,
                                packed[j - pack_base][0].bitcast(fx.Float32),
                            )
                            st(
                                xs,
                                target_word + 1,
                                packed[j - pack_base][1].bitcast(fx.Float32),
                            )
                for j in range_constexpr((MID_SCALE_WORDS + THREADS - 1) // THREADS):
                    index = tid + j * THREADS
                    mid_exponent = get(mb("ugp"), MIDQ_WORDS + fx.min(index, MID_SCALE_WORDS - 1))
                    if index < MID_SCALE_WORDS:
                        target_scale = index
                        if const_expr(GROUPED_DN):
                            target_scale = (index // (expert_inter // 32)) * DN_SCALE_ROW + index % (expert_inter // 32)
                        st(misc, target_scale, mid_exponent.bitcast(fx.Float32))

                gpu.barrier()

                ue = keys + S * SLOTS
                um = ue + UNION_MAX
                uc = um + UNION_MAX

                if const_expr(mode == "union_lut4"):
                    slot_lut = fx.recast_iter(fx.Int8, arena + CROFF + CWORDS)
                    for j in range_constexpr((257 * S + THREADS - 1) // THREADS):
                        address = tid + j * THREADS
                        if address < 257 * S:
                            st(slot_lut, address, fx.Int8(0))
                    gpu.barrier()
                    if tid < S * SLOTS:
                        lut_expert = ld(keys, tid)
                        st(
                            slot_lut,
                            lut_expert * S + tid // SLOTS,
                            fx.Int8(tid % SLOTS),
                        )
                    gpu.barrier()

                n_union = _uniform(ld(uc, 4))

                sn = fx.min(lane % 16, S - 1)

                BATCH = 4 if mode in ("union_batch4", "union_lut4") else 1

                for batch in range((n_union + BATCH - 1) // BATCH):
                    units, slots, active_columns = [], [], []
                    for part in range_constexpr(BATCH):
                        index = fx.min(batch * BATCH + part, n_union - 1)
                        expert = _uniform(ld(ue, index))
                        if const_expr(mode == "union_lut4"):
                            slot = fx.Int32(ld(slot_lut, expert * S + sn))
                        else:
                            slot = fx.Int32(0)
                            for j in range_constexpr(SLOTS):
                                slot = (ld(keys, sn * SLOTS + j) == expert).select(fx.Int32(j), slot)
                        slots.append(slot)
                        active_columns.append((lane % 16 < S) & (ld(keys, sn * SLOTS + slot) == expert))
                        units.append(unit(expert, slot, gw, fx.Float32(1.0)))
                    for part in range_constexpr(BATCH):
                        if (batch * BATCH + part < n_union) | (part == 0 and index_request_width == 6):
                            c = partial(units[part])
                            if active_columns[part]:
                                for e in range_constexpr(4):
                                    row = gu * 16 + lane // 16 * 4 + e
                                    st(
                                        contributions,
                                        contribution_index(sn, slots[part], gw, row),
                                        c[e],
                                    )

                gpu.barrier()

                acc = fx.Vector.filled(4, 0.0, fx.Float32)

                if const_expr(DN_UNION_ROWS40):
                    # S10 assigns 90 contiguous slot/K units to each wave.
                    # Tokens2/7 cross a wave boundary at slot4, K chunk2.
                    if lane % 16 < S:
                        for slot in range_constexpr(SLOTS):
                            coefficient = ld(dnw, sn * SLOTS + slot)
                            for kc in range_constexpr(4):
                                if ((sn * SLOTS + slot) * 4 + kc) // 90 == gw:
                                    vals = [
                                        ld(
                                            contributions,
                                            contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4 + e),
                                        )
                                        for e in range(4)
                                    ]
                                    acc = fx.Vector.from_elements(
                                        [acc[e] + vals[e] * coefficient for e in range(4)], fx.Float32
                                    )
                else:
                    if (lane % 16 < S) & (sn // 3 == gw):
                        for slot in range_constexpr(SLOTS):
                            coefficient = ld(dnw, sn * SLOTS + slot)
                            for kc in range_constexpr(4):
                                vals = [
                                    ld(
                                        contributions,
                                        contribution_index(sn, slot, kc, gu * 16 + lane // 16 * 4 + e),
                                    )
                                    for e in range(4)
                                ]
                                acc = fx.Vector.from_elements(
                                    [acc[e] + vals[e] * coefficient for e in range(4)],
                                    fx.Float32,
                                )

                gpu.barrier()

                fx.ptr_store(acc, red + (wave * 64 + lane) * 4)

                gpu.barrier()

                if tid < S * 32:
                    output_row, output_token = tid % 32, tid // 32
                    tot = fx.Float32(0.0)
                    for w in range_constexpr(4):
                        ww = output_row // 16 * 4 + w
                        tot = tot + ld(
                            red,
                            (ww * 64 + output_token + 16 * ((output_row % 16) // 4)) * 4 + output_row % 4,
                        )
                    lds_st(outs, output_token * 32 + output_row, tot)

                if const_expr(DN_UNION_ROWS40):
                    # Cohorts execute three down tasks per CTA. The union arena
                    # overlaps the original keys, so restore them after the
                    # last contribution read and before the next task round.
                    restore_key = ld(keys, fx.min(tid, DU_KEY_WORDS - 1))
                    restore_prob = ld(dnw, fx.min(tid, DU_ROUTE_WORDS - 1))
                    gpu.barrier()
                    if tid < DU_KEY_WORDS:
                        st(old_keys, tid, restore_key)
                    if tid < DU_ROUTE_WORDS:
                        st(old_dnw, tid, restore_prob)
                    gpu.barrier()

            def down_inputs_needed(t):
                if const_expr(DN_INPUT_REUSE40):
                    return t == start("down")
                return fx.Boolean(True)

            for t in range(start("down"), N_DN_TILES, G):
                t = fx.Int32(t)
                stamp("down", t, 0)
                if const_expr(DN_UNION_CANDIDATE or DN_UNION_ROWS40):
                    dn_union_tile(t)
                else:
                    if down_inputs_needed(t):
                        if const_expr(ug_split(S, expert_inter) is None):  # retain the tuned down-stage route schedule
                            dn_route(load_bias())
                    gpu.barrier()
                    gu = wave // DN_WPR
                    dn_rg = t * DN_TILE // 16
                    dn_off = t * DN_TILE % 16
                    # this lane's row, as a tile row; rows outside the tile load their lane ^ 8 twin
                    # (same cache lines) and are dropped in the output
                    dn_lr = gu * 16 + lane % 16 - dn_off
                    dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

                    DN_NU = S * MOE_SLOTS * DN_NKC // 2
                    DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
                    DN_BLK = S * MOE_SLOTS * expert_inter // 128

                    def u_dn(cc):  # cc: 128-k chunk of this wave
                        qu = (wave % DN_WPR) * DN_UPW + cc
                        live = qu < DN_NU
                        unit = fx.min(qu, DN_NU - 1)
                        q = unit * 2  # 64-k chunk index over (s, slot, kc)
                        s_q = q // (MOE_SLOTS * DN_NKC)
                        slot_q = (q // DN_NKC) % MOE_SLOTS
                        kc = q % DN_NKC
                        e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                        wb = bo.create_buffer_resource_from_addr(
                            w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                            num_records_bytes=(
                                None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                            ),
                        )
                        sb = bo.create_buffer_resource_from_addr(
                            s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                            num_records_bytes=(
                                None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))
                            ),
                        )

                        def coef():  # mid block scale * route weight, only in this sample's column
                            return (lane % 16 == s_q).select(_uniform_f32(lds_ld(misc, q // 2)), fx.Float32(0.0))

                        if const_expr(expert_mxfp4):

                            def bf16_coef():
                                return (lane % 16 == s_q).select(
                                    _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)),
                                    fx.Float32(0.0),
                                )

                            if const_expr(native_fp4_mfma):
                                return unit_native_mxfp4(
                                    wb,
                                    sb,
                                    dn_rg + gu,
                                    unit % (expert_inter // 128),
                                    expert_inter,
                                    unit * 32,
                                    unit * 4 + lane // 16,
                                    bf16_coef,
                                    dn_ln,
                                )
                            return unit_mxfp4_bf16(
                                wb,
                                sb,
                                dn_rg + gu,
                                unit % (expert_inter // 128),
                                expert_inter,
                                unit * 64,
                                bf16_coef,
                                dn_ln,
                            )
                        return unit_f8f8(
                            wb,
                            sb,
                            dn_rg + gu,
                            kc,
                            DN_NKC,
                            expert_inter,
                            q * 16,
                            coef,
                            dn_ln,
                        )

                    if const_expr(GROUPED_DN and S == 12):
                        # keys/dnw are ready before mid payloads. Build in the disjoint red
                        # arena now, so producer progress can overlap this scheduling work.
                        # Preserve the old contiguous sample partition of each row-group
                        # wave. Reorder only independent token columns, never one token's
                        # nonzero slot/K additions or its final wave-reduction position.
                        assert S % DN_WPR == 0
                        DN_GROUP_SAMPLES = S // DN_WPR
                        DN_GROUP_MAX = DN_GROUP_SAMPLES * MOE_SLOTS
                        DN_COUNT_OFF = DN_WPR * DN_GROUP_MAX * 3
                        assert DN_COUNT_OFF + DN_WPR <= RED_WORDS
                        dsg_dn_table = fx.recast_iter(fx.Int32, red)
                        dsg_count = fx.Int32(0)
                        dsg_mask = fx.Int32(0)
                        if wave < DN_WPR:
                            dsg_first_sample = wave * DN_GROUP_SAMPLES
                            for dsg_build_slot in range_constexpr(MOE_SLOTS):
                                dsg_experts = [
                                    _uniform(
                                        lds_ld(
                                            keys,
                                            (dsg_first_sample + j) * MOE_SLOTS + dsg_build_slot,
                                        )
                                    )
                                    for j in range(DN_GROUP_SAMPLES)
                                ]
                                for j in range_constexpr(DN_GROUP_SAMPLES):
                                    dsg_distinct = fx.Int32(1) == 1
                                    for dsg_previous in range_constexpr(j):
                                        dsg_distinct = dsg_distinct & (dsg_experts[j] != dsg_experts[dsg_previous])
                                    if dsg_distinct:
                                        dsg_mask = fx.Int32(0)
                                        for dsg_member in range_constexpr(DN_GROUP_SAMPLES):
                                            dsg_mask = dsg_mask | (dsg_experts[j] == dsg_experts[dsg_member]).select(
                                                fx.Int32(1) << (dsg_first_sample + dsg_member),
                                                fx.Int32(0),
                                            )
                                        if lane == 0:
                                            dsg_address = (wave * DN_GROUP_MAX + dsg_count) * 3
                                            lds_st(
                                                dsg_dn_table,
                                                dsg_address,
                                                dsg_experts[j],
                                            )
                                            lds_st(
                                                dsg_dn_table,
                                                dsg_address + 1,
                                                fx.Int32(dsg_build_slot),
                                            )
                                            lds_st(dsg_dn_table, dsg_address + 2, dsg_mask)
                                        dsg_count = dsg_count + 1
                            if lane == 0:
                                lds_st(dsg_dn_table, DN_COUNT_OFF + wave, dsg_count)
                        gpu.barrier()
                    if const_expr(S <= 4):
                        pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
                    # Later output tasks read the same route, packed mid values,
                    # and exponents. Their compute/reduction/TP bodies only read
                    # xs/misc/keys/dnw, so keep those LDS inputs until down ends.
                    if down_inputs_needed(t):
                        hint_wait(
                            N_UG,
                            lambda k: (
                                mb("mid"),
                                (k // (MOE_SLOTS * I_PER_SLOT) * MOE_SLOTS + (k // I_PER_SLOT) % MOE_SLOTS)
                                * expert_inter
                                + (k % I_PER_SLOT) * UG_TILE
                                + UG_TILE
                                - 1,
                            ),
                            mark=("down", t),
                        )
                        if const_expr(GROUPED_UG):
                            # Poll each tagged payload directly: no new global completion
                            # flag or relaxed data-publication assumption is introduced.
                            MID_PAIR_ROUNDS = (MIDQ_WORDS + 2 * THREADS - 1) // (2 * THREADS)
                            for pack_base in range_constexpr(0, MID_PAIR_ROUNDS, 4):
                                packed = poll(
                                    [
                                        (
                                            mb("ugp"),
                                            fx.min((tid + j * THREADS) * 2, MIDQ_WORDS - 2),
                                            2,
                                        )
                                        for j in range(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS))
                                    ]
                                )
                                if const_expr(pack_base == 0):
                                    stamp("down", t, 2)
                                for j in range_constexpr(pack_base, min(pack_base + 4, MID_PAIR_ROUNDS)):
                                    word = (tid + j * THREADS) * 2
                                    if word < MIDQ_WORDS:
                                        target_word = word
                                        if const_expr(GROUPED_DN):
                                            target_word = (word // (expert_inter // 4)) * DN_X_ROW + word % (
                                                expert_inter // 4
                                            )
                                        lds_st(
                                            xs,
                                            target_word,
                                            packed[j - pack_base][0].bitcast(fx.Float32),
                                        )
                                        lds_st(
                                            xs,
                                            target_word + 1,
                                            packed[j - pack_base][1].bitcast(fx.Float32),
                                        )
                            for j in range_constexpr((MID_SCALE_WORDS + THREADS - 1) // THREADS):
                                index = tid + j * THREADS
                                mid_exponent = get(
                                    mb("ugp"),
                                    MIDQ_WORDS + fx.min(index, MID_SCALE_WORDS - 1),
                                )
                                if index < MID_SCALE_WORDS:
                                    target_scale = index
                                    if const_expr(GROUPED_DN):
                                        target_scale = (index // (expert_inter // 32)) * DN_SCALE_ROW + index % (
                                            expert_inter // 32
                                        )
                                    lds_st(misc, target_scale, mid_exponent.bitcast(fx.Float32))
                        else:
                            # TP4 with 12 rows has 54 wave-local groups. Quantize bounded batches
                            # so the full set of mid values does not stay live in registers.
                            N_MID_BLOCKS = (DN_BLK + WAVES - 1) // WAVES
                            MID_BATCH = 8 if native_fp4_mfma and expert_inter > INTER and S == 12 else N_MID_BLOCKS
                            for mid_base in range_constexpr(0, N_MID_BLOCKS, MID_BATCH):
                                mids = get2_many(
                                    [
                                        (
                                            mb("mid"),
                                            fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2,
                                        )
                                        for b in range(
                                            mid_base,
                                            min(mid_base + MID_BATCH, N_MID_BLOCKS),
                                        )
                                    ]
                                )
                                if const_expr(mid_base == 0):
                                    stamp("down", t, 2)
                                for b in range_constexpr(mid_base, min(mid_base + MID_BATCH, N_MID_BLOCKS)):
                                    blk = wave + b * WAVES
                                    if blk < DN_BLK:
                                        if const_expr(expert_mxfp4):
                                            if const_expr(native_fp4_mfma):
                                                q0, q1, qs = quant_mxfp8(
                                                    mids[b - mid_base][0],
                                                    mids[b - mid_base][1],
                                                )
                                                st_mxfp8(blk * 128 + lane * 2, q0, q1)
                                                if lane % 16 == 0:
                                                    lds_st(
                                                        misc,
                                                        blk * 4 + lane // 16,
                                                        qs.bitcast(fx.Float32),
                                                    )
                                            else:
                                                lds_st(
                                                    xs,
                                                    blk * 64 + lane,
                                                    bf16_pair(
                                                        mids[b - mid_base][0],
                                                        mids[b - mid_base][1],
                                                    ),
                                                )
                                        else:
                                            q0, q1, qs = quant_scaled(mids[b - mid_base][0], mids[b - mid_base][1])
                                            st_f8(blk * 128 + lane * 2, q0, q1)
                                            if lane == 0:
                                                lds_st(
                                                    misc,
                                                    blk,
                                                    qs * lds_ld(dnw, blk // (expert_inter // 128)),
                                                )
                    gpu.barrier()
                    if const_expr(GROUPED_DN):
                        if const_expr(S != 12):
                            # Preserve the old contiguous sample partition of each row-group
                            # wave. Reorder only independent token columns, never one token's
                            # nonzero slot/K additions or its final wave-reduction position.
                            assert S % DN_WPR == 0
                            DN_GROUP_SAMPLES = S // DN_WPR
                            DN_GROUP_MAX = DN_GROUP_SAMPLES * MOE_SLOTS
                            DN_COUNT_OFF = DN_WPR * DN_GROUP_MAX * 3
                            assert DN_COUNT_OFF + DN_WPR <= RED_WORDS
                            dsg_dn_table = fx.recast_iter(fx.Int32, red)
                            dsg_count = fx.Int32(0)
                            dsg_mask = fx.Int32(0)
                            if wave < DN_WPR:
                                dsg_first_sample = wave * DN_GROUP_SAMPLES
                                for dsg_build_slot in range_constexpr(MOE_SLOTS):
                                    dsg_experts = [
                                        _uniform(
                                            lds_ld(
                                                keys,
                                                (dsg_first_sample + j) * MOE_SLOTS + dsg_build_slot,
                                            )
                                        )
                                        for j in range(DN_GROUP_SAMPLES)
                                    ]
                                    for j in range_constexpr(DN_GROUP_SAMPLES):
                                        dsg_distinct = fx.Int32(1) == 1
                                        for dsg_previous in range_constexpr(j):
                                            dsg_distinct = dsg_distinct & (dsg_experts[j] != dsg_experts[dsg_previous])
                                        if dsg_distinct:
                                            dsg_mask = fx.Int32(0)
                                            for dsg_member in range_constexpr(DN_GROUP_SAMPLES):
                                                dsg_mask = dsg_mask | (
                                                    dsg_experts[j] == dsg_experts[dsg_member]
                                                ).select(
                                                    fx.Int32(1) << (dsg_first_sample + dsg_member),
                                                    fx.Int32(0),
                                                )
                                            if lane == 0:
                                                dsg_address = (wave * DN_GROUP_MAX + dsg_count) * 3
                                                lds_st(
                                                    dsg_dn_table,
                                                    dsg_address,
                                                    dsg_experts[j],
                                                )
                                                lds_st(
                                                    dsg_dn_table,
                                                    dsg_address + 1,
                                                    fx.Int32(dsg_build_slot),
                                                )
                                                lds_st(
                                                    dsg_dn_table,
                                                    dsg_address + 2,
                                                    dsg_mask,
                                                )
                                            dsg_count = dsg_count + 1
                                if lane == 0:
                                    lds_st(dsg_dn_table, DN_COUNT_OFF + wave, dsg_count)
                            gpu.barrier()
                        dsg_group_wave = wave % DN_WPR
                        dsg_groups = _uniform(lds_ld(dsg_dn_table, DN_COUNT_OFF + dsg_group_wave))
                        dsg_down_acc = fx.Vector.filled(4, 0.0, fx.Float32)
                        DN_GROUP_BATCH = 2 if S == 12 else 1
                        for dsg_batch_index in range((dsg_groups + DN_GROUP_BATCH - 1) // DN_GROUP_BATCH):
                            dsg_units = []
                            for dsg_part in range_constexpr(DN_GROUP_BATCH):
                                group_index = dsg_batch_index
                                if const_expr(DN_GROUP_BATCH > 1):
                                    group_index = fx.min(
                                        dsg_batch_index * DN_GROUP_BATCH + dsg_part,
                                        dsg_groups - 1,
                                    )
                                dsg_address = (dsg_group_wave * DN_GROUP_MAX + group_index) * 3
                                dsg_expert = _uniform(lds_ld(dsg_dn_table, dsg_address))
                                dsg_route_slot = _uniform(lds_ld(dsg_dn_table, dsg_address + 1))
                                dsg_token_mask = _uniform(lds_ld(dsg_dn_table, dsg_address + 2))
                                if const_expr(DN_GROUP_BATCH > 1):
                                    dsg_token_mask = (dsg_batch_index * DN_GROUP_BATCH + dsg_part < dsg_groups).select(
                                        dsg_token_mask, fx.Int32(0)
                                    )
                                dsg_sn = n_sel()
                                dsg_mid_row = dsg_sn * MOE_SLOTS + dsg_route_slot
                                dsg_coefficient = (((dsg_token_mask >> dsg_sn) & 1) != 0).select(
                                    lds_ld(dnw, dsg_mid_row), fx.Float32(0.0)
                                )
                                dsg_wb = _rsrc(w_dn + fx.Int64(dsg_expert) * fx.Int64(DN_W_BYTES))
                                dsg_sb = _rsrc(s_dn + fx.Int64(dsg_expert) * fx.Int64(DN_S_BYTES))
                                dsg_units.extend(
                                    [
                                        unit_native_mxfp4(
                                            dsg_wb,
                                            dsg_sb,
                                            dn_rg + gu,
                                            kc,
                                            expert_inter,
                                            dsg_mid_row * DN_X_ROW + kc * 32,
                                            dsg_mid_row * DN_SCALE_ROW + kc * 4 + lane // 16,
                                            dsg_coefficient,
                                            dn_ln,
                                        )
                                        for kc in range(expert_inter // 128)
                                    ]
                                )
                            dsg_down_acc = fx.Vector.from_elements(
                                mma_units([dsg_down_acc[e] for e in range(4)], dsg_units),
                                fx.Float32,
                            )
                        acc = [dsg_down_acc[e] for e in range(4)]
                        # All waves must finish reading the schedule before reduce_rows
                        # reuses red for per-wave accumulator stores.
                        gpu.barrier()
                    else:
                        if const_expr(native_fp4_mfma and npes == 4 and S == 10 and expert_inter == 512):
                            # Bound the unrolled body while preserving every wave's
                            # slot/K order and two-batch software pipeline.
                            DN_LOOP_UNITS = 10
                            assert DN_UPW % DN_LOOP_UNITS == 0 and DN_LOOP_UNITS % DN_BATCH == 0
                            dn_loop_acc = fx.Vector.filled(4, 0.0, fx.Float32)
                            for dn_loop in range(DN_UPW // DN_LOOP_UNITS):
                                dn_base = dn_loop * DN_LOOP_UNITS
                                dn_cur = [u_dn(dn_base + cc) for cc in range(DN_BATCH)]
                                for dn_begin in range_constexpr(0, DN_LOOP_UNITS, DN_BATCH):
                                    dn_next = None
                                    if const_expr(dn_begin + DN_BATCH < DN_LOOP_UNITS):
                                        dn_next = [
                                            u_dn(dn_base + cc)
                                            for cc in range(
                                                dn_begin + DN_BATCH,
                                                dn_begin + 2 * DN_BATCH,
                                            )
                                        ]
                                    dn_loop_acc = fx.Vector.from_elements(
                                        mma_units([dn_loop_acc[e] for e in range(4)], dn_cur),
                                        fx.Float32,
                                    )
                                    dn_cur = dn_next
                            acc = [dn_loop_acc[e] for e in range(4)]
                        else:
                            if const_expr(S > 4):
                                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
                            acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

                    def emit_dn(rl, n, v):
                        if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                            lds_st(outs, n * DN_TILE + rl - dn_off, v)

                    reduce_rows(DN_R, acc, emit_dn)
                stamp("down", t, 3)
                gpu.barrier()

                def store_x(s, row, v0, v1):
                    bo.buffer_store(
                        fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                        _rsrc(x_out),
                        s * HIDDEN + row,
                    )

                def residual_a(s, row):
                    return bf2_f32(get(mb("a"), (s * HIDDEN + row) // 2))

                if const_expr(
                    native_fp4_mfma
                    and atom_experts
                    and npes == 4
                    and S == 12
                    and expert_inter == 512
                    and not with_indexer
                    and not indexer_cp
                    and dcp_size == 1
                ):
                    ffn_owner_reduce(t, residual_a, store_x)
                else:
                    peer_reduce("ffn", t, residual_a, store_x, tile=DN_TILE)
                gpu.barrier()
                stamp("down", t, 4)

            if const_expr(DEFER_OFF_CACHE):
                # Every reader finishes before persistent cache storage is overwritten.
                # This second barrier has a distinct slot at the same invocation tag.
                fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                gpu.barrier()
                commit_arrivals = full_scratch + fx.Int64(ROW_TILES * TILE_SCRATCH_BYTES + (BLOCKS + 1) * 8)
                if tid == 0:
                    put(commit_arrivals, full_bid, fx.Int32(1))
                if full_bid == 0:
                    if tid < BLOCKS:
                        get(commit_arrivals, tid)
                        fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
                    gpu.barrier()
                    if tid == 0:
                        put(commit_arrivals, BLOCKS, fx.Int32(1))
                if tid == 0:
                    get(commit_arrivals, BLOCKS)
                gpu.barrier()
                fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
                if bid < S:
                    commit_slot = row_slot(bid)
                    later_slot = fx.Int32(
                        bo.buffer_load(_rsrc(full_slots), fx.min(lane, total_samples - 1) * 2, vec_width=1, dtype=T.i32)
                    )
                    overwritten = (lane > row_offset + bid) & (lane < total_samples) & (later_slot == commit_slot)
                    overwritten_mask = fx.Int64(rocdl.ballot(T.i64, overwritten))
                    commit_owner = (commit_slot >= 0) & (overwritten_mask == 0)
                    if commit_owner:
                        committed_kv = getf(mb("kvnew"), bid * KV_LORA + tid)
                        bo.buffer_store(committed_kv.to(fx.BFloat16), _rsrc(kv_cache), commit_slot * QK_DIM + tid)
                        if tid < PE_DIM // 2:
                            committed_pe0, committed_pe1 = get2(mb("penew"), bid * PE_DIM + tid * 2)
                            bo.buffer_store(
                                fx.Vector.from_elements([committed_pe0, committed_pe1], fx.Float32).to(fx.BFloat16),
                                _rsrc(pe_cache),
                                commit_slot * QK_DIM + KV_LORA + tid * 2,
                            )

            if const_expr(ROW_TILES > 1 and not COHORT_ROW_TILES):
                if row_tile + 1 < ROW_TILES:
                    # Every wave publishes its cache/output stores before its CTA arrives.
                    # Separate tile payloads and barrier slots avoid overwriting readers.
                    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                    gpu.barrier()
                    arrivals = full_scratch + fx.Int64(ROW_TILES * TILE_SCRATCH_BYTES)
                    arrivals = arrivals + fx.Int64(row_tile) * ((G + 1) * 8)
                    if tid == 0:
                        put(arrivals, bid, fx.Int32(1))
                    if bid == 0:
                        if tid < G:
                            get(arrivals, tid)
                            fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
                        gpu.barrier()
                        if tid == 0:
                            put(arrivals, G, fx.Int32(1))
                    if tid == 0:
                        get(arrivals, G)
                    gpu.barrier()
                    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")

        if const_expr(COHORT_ROW_TILES):
            run_row_tile(full_bid // G)
        elif const_expr(ROW_TILES == 1):
            run_row_tile(0)
        elif const_expr(STATIC_ROW_TILES):
            for row_tile in range_constexpr(ROW_TILES):
                run_row_tile(row_tile)
        else:
            for row_tile in range(ROW_TILES):
                run_row_tile(row_tile)

    @flyc.jit
    def launch(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        glm5_monokernel_v19_router_pair2(
            h_in,
            x_out,
            cur_pos,
            positions,
            slot_mapping,
            sparse_kv_indptr,
            kv_cache,
            pe_cache,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            w_qkv_a,
            s_qkv_a,
            w_q_b,
            s_q_b,
            w_uk,
            s_uk,
            w_uv,
            s_uv,
            w_o,
            s_o,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            rank,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
