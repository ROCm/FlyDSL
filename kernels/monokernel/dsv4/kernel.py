# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""DeepSeek-V4 attention + MoE layer in ONE persistent launch per rank (TP8 decode).

V4 is shared-KV (MQA): ``wkv`` emits one ``HEAD_DIM`` vector per token, K and V
are the same tensor, and RoPE occupies its last ``ROPE_DIM`` lanes; there is no
nope/pe split and no absorbed ``W_UK``/``W_UV``.  ``split`` adds a per-head
softmax ``sink``; ``uv`` merges the splits and de-rotates the output's RoPE lanes;
``o`` is the grouped low-rank pair ``o_a`` then ``o_b``.

One launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs the
whole layer body for this rank's TP shard::

    input RMSNorm -> q_a / kv projection -> q_a RMSNorm -> q_b (+ head RMS, RoPE)
      -> KV RMSNorm / RoPE / FP8 round-trip -> sliding-window KV ring publish
      -> gather-sparse split softmax (+ sink) -> merge -> inverse RoPE
      -> o_a (grouped low rank) -> o_b
      -> attention TP peer reduce + residual                       (sym_attn)
      -> post-attention RMSNorm -> router sqrt-softplus + expert activation
      -> flat top-6 -> 1 shared + 6 routed expert up/gate/clamped SwiGLU
      -> expert down + route weighting
      -> MoE TP peer reduce + residual -> x_out                    (sym_ffn)

The KV compressor (HCA, ratio 128) and the lightning indexer (CSA, ratio 4) add
stages; hyper-connections (``hc_mult > 1``) replace the plain residual.

Scheduling: task ``t`` of a stage runs on CTA ``(stage_base + t) % 256`` and
every CTA walks the stages in order.  No grid-wide barrier: dependencies only
point to earlier stages and all CTAs are co-resident, so every spin wait
makes progress.

Mailboxes are tagged pairs ``(value, tag)`` stored with device- (``sc1``) or
system-coherent (``sc0 sc1``) 8 / 16-byte stores; a consumer polls the payload
until the tag matches this launch's epoch, so a hand-off is one round trip.

GEMVs run on the matrix cores: ``packing.py`` arranges weights so one wave loads
16 rows x 64 k as one contiguous 1 KB, FP8 is widened exactly to bf16 and fed to
``mfma_f32_16x16x32_bf16`` with the samples as N; each 64-k partial is scaled by
its f32 block scale.  Weight loads are issued before waiting for inputs.

Cross-GPU: each rank pushes BF16 partial rows plus a tag into every peer's
symmetric buffer; every rank sums the 8 partials in rank order, so all ranks
produce bit-identical hidden states (and routing).
"""

import enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv4.attention import attention_stages
from kernels.monokernel.dsv4.common import common_defs
from kernels.monokernel.dsv4.config import (
    BLOCK_TOKENS,
    HC_EPS,
    HC_SINKHORN_ITERS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    N_EXPERTS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    SOFTMAX_SCALE,
    SWIGLU_LIMIT,
    TOP_K,
    WINDOW,
    ExpertActivation,
    ExpertWeight,
    MoeMode,
    moe_format,
)
from kernels.monokernel.dsv4.ffn import ffn_stages
from kernels.monokernel.dsv4.hc import hc_attn_stages
from kernels.monokernel.dsv4.indexer import index_query_stages, index_select_stages
from kernels.monokernel.dsv4.plan import (
    BLOCKS,
    LAYER_SLOTS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    SPLIT_KEYS,
    THREADS,
    UG8,
    UG_TILE,
    WAVES,
    dn_tile,
    hc_shape,
    layout,
    n_index,
    n_topk_parts,
    qkv_a_rows,
    stage_tasks,
    uv_tile,
)
from kernels.monokernel.dsv4.qkv import q_b_stage, qkv_stages
from kernels.monokernel.ops import rsrc, uniform


def build_dsv4_kernel(
    S: int = 1,
    heads: int = 16,
    npes: int = 8,
    window: int = WINDOW,
    scale: float = SOFTMAX_SCALE,
    timeline: bool = False,
    moe_mode: MoeMode | str = MoeMode.A8W4,
    poll_timeout_us: int | None = None,
    tokens_per_seq: int = 1,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    n_experts: int = N_EXPERTS,
    top_k: int = TOP_K,
    inter: int = INTER,
    swiglu_limit: float = SWIGLU_LIMIT,
    hc_mult: int = 1,
    hc_sinkhorn_iters: int = HC_SINKHORN_ITERS,
    hc_eps: float = HC_EPS,
    compress_ratio: int = 0,
    window_rows: int | None = None,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    index_topk: int = 0,
    max_seq: int = 0,
    kv_fp8: bool = False,
    indexer_hadamard: bool = True,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole V4 layer.

    ``window`` keys come per sample from a ring cache (-1 = unwritten); ``compress_ratio``
    > 0 appends the compressed entries (all for HCA, the indexer's top-k for CSA).
    ``timeline=True`` records ``s_memrealtime`` stamps into int64 ``[tasks, TL_COLS]``.
    ``tokens_per_seq`` > 1 is an MTP verify step: runs of consecutive tokens of one
    sequence, whose compressor ring has ``C_ROWS + tokens_per_seq - 1`` rows so draft
    rows never alias the window a later step reads after a rejection.
    """
    assert (
        heads % WAVES == 0 and heads <= 16
    ), "the split-attention mapping needs a whole number of wave-groups per head, heads <= 16"
    assert window % SPLIT_KEYS == 0
    # what bounds the sample count: fail the build rather than hang on the GPU
    assert 1 <= S <= 16, "n_sel / reduce_rows put the samples in the MFMA's 16 B columns"
    assert S <= WAVES, f"dn_route routes sample s on wave s, and there are {WAVES} waves"
    assert S * (1 + top_k) <= SPLIT_KEYS, (
        f"dn_route packs S * MOE_SLOTS = {S * (1 + top_k)} expert ids into Smem.keys, "
        f"which the split stage sizes at SPLIT_KEYS = {SPLIT_KEYS}"
    )
    assert head_dim % 64 == 0, "the score MFMA walks HEAD_DIM in 32-wide steps over 2 wave halves"
    # too small a HEAD_DIM gives a wave no PV / gather work: an unfilled mailbox, a hang
    assert (
        head_dim % (32 * WAVES) == 0
    ), f"head_dim must be a multiple of {32 * WAVES} for the PV MFMA's per-wave dim groups, got {head_dim}"
    assert head_dim >= 128, "the KV gather needs at least one packed word per lane"
    assert (head_dim - ROPE_DIM) % 64 == 0, "the KV row's FP8 round-trip blocks the nope part by 64"
    assert heads % o_groups == 0, "o_a groups partition the concatenated heads"
    assert o_lora % ROW_TILE == 0 and (heads * head_dim // o_groups) % 64 == 0
    assert head_dim <= THREADS, "the cache / q_norm stages map one thread per head dim"
    assert (head_dim - ROPE_DIM) % 2 == 0, "interleaved RoPE pairs must align to the nope boundary"
    assert n_experts % 64 == 0, "route_topk packs n_experts // 64 selection keys per lane"
    assert n_experts <= 1 << 16, "the packed selection key needs room for the score above the id field"
    assert inter % UG_TILE == 0
    # shadow the module constants with this build's overrides (before any read of them)
    HIDDEN = hidden
    Q_LORA = q_lora
    HEAD_DIM = head_dim
    O_GROUPS = o_groups
    O_LORA = o_lora
    N_EXPERTS = n_experts
    TOP_K = top_k
    INTER = inter
    NOPE_DIM = HEAD_DIM - ROPE_DIM
    MOE_SLOTS = 1 + TOP_K
    assert TOP_K <= 8, "ug keeps the routing weights in misc[:8], below the quant scales"
    SHARED_EXPERT = N_EXPERTS
    QKV_A_ROWS = Q_LORA + HEAD_DIM  # the FP8 rows; the compressors' BF16 rows follow (qkv_c)
    N_QKV_A = QKV_A_ROWS // QKV_A_TILE
    N_QKV_C = (qkv_a_rows(Q_LORA, HEAD_DIM, compress_ratio, c_coff, index_head_dim) - QKV_A_ROWS) // QKV_A_TILE
    N_ROW_TILES = HIDDEN // ROW_TILE
    N_ROUTER = N_EXPERTS // ROUTER_TILE
    N_UG_PER_SLOT = INTER // UG_TILE
    fmt = moe_format(moe_mode)
    use_fp8_block128 = fmt.activation is ExpertActivation.FP8_BLOCK128
    use_mxfp8_block32 = fmt.activation is ExpertActivation.MXFP8_BLOCK32
    use_mxfp4_weight = fmt.weight is ExpertWeight.MXFP4_BLOCK32
    # with MXFP4 experts the shared expert stays FP8 128x128 (w_sug / w_sdn), as ATOM
    SHARED_FP8 = use_mxfp4_weight
    XQ_BLOCKS = 0 if fmt.activation_group is None else HIDDEN // fmt.activation_group
    PUBLISH_BLOCKS = HIDDEN // (32 if use_mxfp8_block32 else 128)
    XQ_WAVES = (
        (PUBLISH_BLOCKS + N_ROUTER * 4 - 1) // (N_ROUTER * 4)
        if use_mxfp8_block32
        else (PUBLISH_BLOCKS + N_ROUTER - 1) // N_ROUTER
    )
    assert XQ_WAVES <= WAVES
    # KV compression (CR == 0 compiles out); CSA's entries pool 2*CR tokens at stride CR (C_COFF = 2)
    CR = compress_ratio
    C_COFF = c_coff
    # the lightning indexer's own compressor (Hadamard, FP4); IHD == 0 compiles it out
    IHD = index_head_dim
    # ATOM's fp8 KV layout: a NoPE plane of KV_ROW_BYTES per row (NOPE_DIM FP8 bytes,
    # each 64-group's E8M0 byte twice, padding) plus a bf16 RoPE plane; else one bf16 plane
    KV_FP8 = kv_fp8
    BOUNDED_POLL = poll_timeout_us is not None
    POLL_TIMEOUT_TICKS = (poll_timeout_us or 0) * 100  # s_memrealtime runs at 100 MHz
    INDEXER_HADAMARD = indexer_hadamard
    if CR:
        assert BLOCK_TOKENS % CR == 0, "a block holds a whole number of compressed entries"
    KV_ROW_BYTES = 512
    if KV_FP8:
        assert NOPE_DIM % 64 == 0 and HEAD_DIM - NOPE_DIM == ROPE_DIM and THREADS == HEAD_DIM
    IW = C_COFF * IHD
    # Compressed entries are paged as ATOM's: entry e of sequence s is KV row
    # dest_rows[1, s] + block_tables[s][e // K_PB] * env_rows + e % K_PB. The indexer's
    # FP4 key pool: per block, codes [IHD / 32][K_PB][16 B] and E8M0 scales [IHD / 32][K_PB]
    # with the entry axis interleaved (byte (e % 16) * 4 + (e % K_PB) // 16).
    K_PB = BLOCK_TOKENS // CR if CR else 1
    IC_GRP_WORDS = K_PB * 16 // 4  # one 32-element group of one block, in dwords
    IC_BLK_WORDS = (IHD // 32) * IC_GRP_WORDS if IHD else 0
    IC_S_BLK = (IHD // 32) * K_PB if IHD else 0  # scale bytes of one block
    CW = C_COFF * HEAD_DIM  # width of one state row
    # compressor state: a ring indexed by position, as ATOM's (window at p: rows (p + 1 + i) % C_ROWS)
    C_ROWS = C_COFF * CR
    TOK = tokens_per_seq
    assert S % TOK == 0, "a launch holds whole runs of a sequence's tokens"
    assert TOK == 1 or not CR or TOK <= C_ROWS, "the in-launch tail of the window is at most its length"
    # the ring the state is stored in: C_ROWS plus the K draft positions of a step
    C_RING = C_ROWS + TOK - 1
    OVERLAP = C_COFF > 1
    # state rows the pooling loop loads per trip
    CMP_CHUNK = max(c for c in range(1, 9) if C_ROWS % c == 0) if C_ROWS else 1
    # with TOK > 1 the ring part of the window is its first C_ROWS - TOK rows
    CMP_CHUNK_T = max(c for c in range(1, 9) if (C_ROWS - TOK) % c == 0) if (C_ROWS and TOK > 1) else 1
    # window and compressed entries share one cache, the compressed half at `window`
    CACHE_ROWS = window if window_rows is None else window_rows
    N_KEYS = window if n_keys is None else n_keys
    if CR:
        assert CACHE_ROWS > window, "a compressing layer needs cache rows past the window"
        assert HEAD_DIM <= THREADS, "the compressor maps one thread per channel"

    # hyper-connections (HC == 1 is a plain residual and compiles out)
    HC = hc_mult
    HC_TASKS, HC_KSLICE, HC_ROWS, HC_VALS = hc_shape(HC, HIDDEN)
    HC_MIX = (2 + HC) * HC if HC > 1 else 0
    # waves sharing the coefficient poll, one poll batch (POLL_MAX) of hcd tasks each
    HC_PW = min(WAVES, -(-HC_TASKS // 12)) if HC > 1 else 1
    HC_TPW = -(-HC_TASKS // HC_PW) if HC > 1 else 1
    HC_COEF = 2 * HC + HC * HC if HC > 1 else 0
    # comb lane = j * HC + k: XOR offsets walk a row (low bits) or a column (high bits)
    HC_ROW_OFFS = tuple(1 << b for b in range(HC.bit_length() - 1)) if HC > 1 else ()
    HC_COL_OFFS = tuple(HC << b for b in range(HC.bit_length() - 1)) if HC > 1 else ()
    HC_NKC = HC_KSLICE // 64 if HC > 1 else 0
    HC_RG = HC_ROWS // 16 if HC > 1 else 0
    HC_WPR = WAVES // HC_RG if HC > 1 else 0
    HC_NKC_FULL = (HC * HIDDEN) // 64 if HC > 1 else 0
    if HC > 1:
        assert HC & (HC - 1) == 0, "hc_mult must be a power of two for the cross-lane Sinkhorn"
        assert HC * HC <= 64, "the Sinkhorn matrix must fit one wave"
        assert HC_NKC % HC_WPR == 0, "hc_pre chunks must divide over the waves of a row group"

    down_scale_words = 0 if fmt.activation_group is None else S * MOE_SLOTS * INTER // fmt.activation_group
    HC_MISC = 8 + max(S * XQ_BLOCKS, down_scale_words)
    # `uv` stores one per-split weight in misc, so it must hold N_SPLIT of them
    misc_words = max(HC_MISC + S * max(HC_COEF, 1), n_keys // SPLIT_KEYS)
    H = heads
    W = npes
    G = BLOCKS
    SC, SY = layout(
        S,
        H,
        W,
        window,
        moe_mode,
        hidden=HIDDEN,
        q_lora=Q_LORA,
        head_dim=HEAD_DIM,
        o_groups=O_GROUPS,
        o_lora=O_LORA,
        hc_mult=HC,
        compress_ratio=CR,
        n_keys=N_KEYS,
        c_coff=C_COFF,
        index_head_dim=IHD,
        index_heads=index_heads,
        index_topk=index_topk,
        max_seq=max_seq,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        inter=INTER,
    )
    assert N_KEYS % SPLIT_KEYS == 0, "the index list must be a whole number of key tiles"
    N_SPLIT = N_KEYS // SPLIT_KEYS
    # split/merge sized per launch by the live keys (see live_splits)
    LIVE_SPLITS = bool(compress_ratio) and not index_head_dim and N_KEYS > window
    # the `uv` merge gives each split one thread: one wave while they fit, else the block
    UV_WIDE = N_SPLIT > THREADS // WAVES
    UV_CHUNK = max(c for c in range(1, 9) if N_SPLIT % c == 0)
    assert N_SPLIT <= THREADS, (
        f"{N_SPLIT} key splits exceeds the {THREADS} threads the block-wide `uv` merge has. "
        f"n_keys {N_KEYS} = window {window} + the compressed list, so this is a max_seq limit"
    )
    N_QB = H * HEAD_DIM // Q_B_TILE
    QB_PER_HEAD = HEAD_DIM // Q_B_TILE
    IH = index_heads if IHD else 0
    N_IQB = IH * IHD // Q_B_TILE
    N_IHG = -(-IH // 16)  # 16-head MFMA column groups of the score
    N_COMP = (max_seq // CR) if (IHD and CR) else 0
    N_INDEX = n_index(max_seq, compress_ratio, index_head_dim, index_topk)
    # a top-k thread takes TK_PER consecutive candidates per trip (two 16-byte loads)
    TK_PER = 4
    TK_BINS = 256  # 8-bit radix digit; 4 passes cover the key
    # bin copies per lane group, so near-identical top digits do not serialize the atomics
    TK_REP = 16
    TK_BC = TK_BINS * TK_REP  # the four words past the bins that broadcast a pass's result
    TK_PARTS = n_topk_parts(max_seq, compress_ratio, index_head_dim)
    # trips PER PART: part q takes every TK_PARTS-th trip, starting at q
    TK_TRIPS = max(1, -(-N_COMP // (THREADS * TK_PER * max(TK_PARTS, 1)))) if N_COMP else 1
    # the index list pads to a whole key tile; the surplus is -1
    N_ISEL = N_KEYS - window
    if IHD:
        assert window % SPLIT_KEYS == 0, "a key tile must fall wholly inside or outside the window"
        assert N_KEYS >= window + N_INDEX, "the index list must hold the window and the pick"
        assert IHD % 8 == 0 and N_COMP > 0
        # keeps a clamped group 16-byte aligned and wholly past n_live (so masked)
        assert N_COMP % TK_PER == 0, "the top-k reads whole groups of TK_PER candidates"
        # every part polls every other part's bins: two parts on one CTA would hang
        assert S * TK_PARTS <= BLOCKS, "each top-k part needs its own CTA"
        # 32 per thread covers a 1M context
        assert TK_TRIPS * TK_PER <= 32, "the top-k's candidates per thread must fit in registers"
        assert IHD == 128, "the indexer's Hadamard is written for a 128-wide head"
        assert K_PB == 64, "the FP4 pool's scale interleave is written for 64 entries a block (4 runs of 16)"
        assert IH % WAVES == 0, "one wave takes a whole index head"
        assert IHD - ROPE_DIM == 64, "rope must fall entirely in the head's second half"
    UV_TILE = uv_tile(S, H, HEAD_DIM)  # shadows the module minimum
    N_UV = H * HEAD_DIM // UV_TILE
    UV_PER_HEAD = HEAD_DIM // UV_TILE
    OA_K = H * HEAD_DIM // O_GROUPS  # one group's slice of the concatenated heads
    N_OA = S * O_GROUPS * O_LORA // ROW_TILE
    OA_PER_GROUP = O_LORA // ROW_TILE
    OB_K = O_GROUPS * O_LORA
    # selection key: order-preserving score bits, low ID_BITS = ID_MASK - id (ties to the lower id)
    ID_BITS = max(8, (N_EXPERTS - 1).bit_length())
    ID_MASK = (1 << ID_BITS) - 1
    UG_PER_SLOT = INTER // UG8
    N_UG_TASKS = TOP_K * UG_PER_SLOT
    N_UG = S * MOE_SLOTS * N_UG_PER_SLOT
    # route weight sum over lanes 0..TOP_K-1 (lanes >= TOP_K hold 0)
    _off, TOPK_SUM_OFFS = 1, []
    while _off < TOP_K:
        TOPK_SUM_OFFS.append(_off)
        _off *= 2
    TOPK_SUM_OFFS = tuple(TOPK_SUM_OFFS)
    # one KV tile: RoPE is inside the HEAD_DIM vector
    QK_DIM = HEAD_DIM
    QS = QK_DIM // 2 + 4
    KS = HEAD_DIM // 2 + 4
    KT_OFF = H * QS
    XN = max(S * HIDDEN // 2, KT_OFF + SPLIT_KEYS * KS)
    ON = S * max(ROW_TILE, QKV_A_TILE, UG_TILE * 2)
    DN_TILE = dn_tile(S, HIDDEN)
    N_DN_TILES = HIDDEN // DN_TILE

    st_args = dict(
        window=window,
        hidden=HIDDEN,
        q_lora=Q_LORA,
        head_dim=HEAD_DIM,
        o_groups=O_GROUPS,
        o_lora=O_LORA,
        top_k=TOP_K,
        inter=INTER,
        hc_mult=HC,
        compress_ratio=CR,
        n_keys=N_KEYS,
        c_coff=C_COFF,
        index_head_dim=IHD,
        index_heads=index_heads,
        index_topk=index_topk,
        max_seq=max_seq,
        n_experts=N_EXPERTS,
    )
    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(S, H, **st_args):
        first[name] = acc
        acc += n
    # CTA placement: split is placed before q_b so its tasks land on CTAs freed by qkv_a
    tasks = dict(stage_tasks(S, H, **st_args))
    acc = 0
    for name in (
        "hcd_a",
        "hcc_a",
        "qkv_a",
        "qkv_c",
        "cache",
        "cmp",
        "i_cmp",
        "split",
        "q_b",
        "q_norm",
        "i_q_b",
        "i_q",
        "i_wp",
        "i_score",
        "i_topk",
        "uv",
        "o_a",
        "o_b",
        "hcd_f",
        "hcc_f",
        "router",
        "ug",
        "down",
    ):
        base[name] = acc % G
        acc += tasks[name]

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XN, 16]  # bf16 activations (pairs) / split q + KV tile
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, misc_words, 16]
        p: fx.Array[fx.Float32, H * SPLIT_KEYS, 16]
        keys: fx.Array[fx.Int32, SPLIT_KEYS, 16]
        dnw: fx.Array[fx.Float32, S * MOE_SLOTS, 16]  # expert-down route weights
        # radix-select bins, replicated, plus two words to broadcast the winning digit
        hist: fx.Array[fx.Int32, (TK_BC + 4) if IHD else 1, 16]

    # the stages read this build's constants from bc; _cache_tag puts them in the JIT cache key
    bc = dict(locals())
    _cache_tag = tuple(
        (n, v) for n, v in sorted(bc.items()) if isinstance(v, (bool, int, float, str, tuple, type(None), enum.Enum))
    )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def dsv4_kernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        kv_cache: Int64,
        kv_rope: Int64,
        dest_rows: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        attn_sink: Int64,
        ape: Int64,
        g_ckv: Int64,
        kv_state: Int64,
        score_state: Int64,
        i_ape: Int64,
        g_ickv: Int64,
        i_kv_state: Int64,
        i_score_state: Int64,
        i_cache: Int64,
        hc_attn_fn: Int64,
        hc_attn_sb: Int64,
        hc_ffn_fn: Int64,
        hc_ffn_sb: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_qkv_c: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_i_q_b: Int64,
        s_i_q_b: Int64,
        i_w: Int64,
        w_o_a: Int64,
        s_o_a: Int64,
        w_o_b: Int64,
        s_o_b: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sug: Int64,
        s_sug: Int64,
        w_sdn: Int64,
        s_sdn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        hang: Int64,
        state_slots: Int64,
        tok_ids: Int64,
        tid2eid: Int64,
        block_tables: Int64,
        i_cache_s: Int64,
        rank: Int32,
        layer: Int32,
        st_kv: Int32,
        st_i: Int32,
        st_ic: Int32,
        use_hash: Int32,
        bt_stride: Int32,
        env_rows: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.x.ptr
        outs = lds.out.ptr
        red = lds.red.ptr
        misc = lds.misc.ptr
        pl = lds.p.ptr
        keys = lds.keys.ptr
        dnw = lds.dnw.ptr
        hist = lds.hist.ptr
        ktile = xs + KT_OFF  # f32-typed view holding raw bf16 pairs
        v4f = fx.Vector.make_type(4, fx.Float32)

        r_h = rsrc(h_in)
        # this launch's epoch: a per-step device counter, unique per layer
        tag = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32)) * LAYER_SLOTS + layer + 1
        r_pos = rsrc(cur_pos)

        def ld_pos(s):
            """Sample ``s``'s position (``s`` is wave-uniform)."""
            return uniform(bo.buffer_load(r_pos, s, vec_width=1, dtype=T.i32))

        r_dest = rsrc(dest_rows)

        def ld_dest(j, s):
            """Plane rows for sample ``s``, supplied by the pool: j=0 the row this
            token's KV goes to, j=1 the base of its compressed entries (see comp_row)."""
            return uniform(bo.buffer_load(r_dest, j * S + s, vec_width=1, dtype=T.i32))

        def bt_block(s, e):
            """The physical block holding sequence ``s``'s compressed entry ``e``."""
            return fx.Int32(bo.buffer_load(rsrc(block_tables), s * bt_stride + e // K_PB, vec_width=1, dtype=T.i32))

        def comp_row(s, e):
            """Plane row of sequence ``s``'s compressed entry ``e`` (see K_PB)."""
            return ld_dest(1, s) + bt_block(s, e) * env_rows + e % K_PB

        r_slot = rsrc(state_slots)

        def ld_slot(s, stride):
            """Element offset of sample ``s``'s slot of a rolling-state pool; ``stride``
            is the whole pool entry, which interleaves several fields."""
            return uniform(bo.buffer_load(r_slot, s, vec_width=1, dtype=T.i32)) * stride

        # serving pools exceed a buffer resource's 4 GB reach: row / slot bases are 64-bit
        def slot_rsrc(ptr, s, stride):
            """A resource at sample ``s``'s slot of an f32 rolling-state pool."""
            slot = uniform(bo.buffer_load(r_slot, s, vec_width=1, dtype=T.i32))
            return rsrc(ptr + fx.Int64(slot) * fx.Int64(stride) * 4)

        def row_rsrc(ptr, row, row_bytes):
            """A resource at plane row ``row`` (wave-uniform) of rows ``row_bytes`` wide."""
            return rsrc(ptr + fx.Int64(uniform(row)) * row_bytes)

        r_peers = rsrc(peers)
        # each wave sends to one peer
        pv = fx.Vector(bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32))
        peer_dst = (fx.Int64(uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(uniform(pv[0])))

        _ = _cache_tag
        ctx = dict(bc)
        ctx.update(locals())
        ctx.update(common_defs(ctx))
        ctx.update(hc_attn_stages(ctx))
        ctx.update(qkv_stages(ctx))
        ctx.update(index_query_stages(ctx))
        ctx.update(q_b_stage(ctx))
        ctx.update(index_select_stages(ctx))
        ctx.update(attention_stages(ctx))
        ffn_stages(ctx)

    @flyc.jit
    def launch_dsv4(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        kv_cache: Int64,
        kv_rope: Int64,
        dest_rows: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        attn_sink: Int64,
        ape: Int64,
        g_ckv: Int64,
        kv_state: Int64,
        score_state: Int64,
        i_ape: Int64,
        g_ickv: Int64,
        i_kv_state: Int64,
        i_score_state: Int64,
        i_cache: Int64,
        hc_attn_fn: Int64,
        hc_attn_sb: Int64,
        hc_ffn_fn: Int64,
        hc_ffn_sb: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_qkv_c: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_i_q_b: Int64,
        s_i_q_b: Int64,
        i_w: Int64,
        w_o_a: Int64,
        s_o_a: Int64,
        w_o_b: Int64,
        s_o_b: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sug: Int64,
        s_sug: Int64,
        w_sdn: Int64,
        s_sdn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        hang: Int64,
        state_slots: Int64,
        tok_ids: Int64,
        tid2eid: Int64,
        block_tables: Int64,
        i_cache_s: Int64,
        rank: Int32,
        layer: Int32,
        st_kv: Int32,
        st_i: Int32,
        st_ic: Int32,
        use_hash: Int32,
        bt_stride: Int32,
        env_rows: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        dsv4_kernel(
            h_in,
            x_out,
            cur_pos,
            kv_cache,
            kv_rope,
            dest_rows,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            attn_sink,
            ape,
            g_ckv,
            kv_state,
            score_state,
            i_ape,
            g_ickv,
            i_kv_state,
            i_score_state,
            i_cache,
            hc_attn_fn,
            hc_attn_sb,
            hc_ffn_fn,
            hc_ffn_sb,
            w_qkv_a,
            s_qkv_a,
            w_qkv_c,
            w_q_b,
            s_q_b,
            w_i_q_b,
            s_i_q_b,
            i_w,
            w_o_a,
            s_o_a,
            w_o_b,
            s_o_b,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            w_sug,
            s_sug,
            w_sdn,
            s_sdn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            hang,
            state_slots,
            tok_ids,
            tid2eid,
            block_tables,
            i_cache_s,
            rank,
            layer,
            st_kv,
            st_i,
            st_ic,
            use_hash,
            bt_stride,
            env_rows,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    # LLVM's VectorCombine (foldShuffleToIdentity) goes exponential on the S == 1 up/gate
    return flyc.compile[{"llvm_options": {"disable-vector-combine": True}}](launch_dsv4)


# ---------------------------------------------------------------- step advance
SCRUB_PAIRS = THREADS  # mailbox pairs each step advance checks: one per thread, one round trip


def scrub_period(n_pairs: int) -> int:
    """Steps for the step advance to visit every one of ``n_pairs`` pairs: a power
    of two, so ``step % period`` stays continuous through the int32 wrap."""
    return 1 << max(0, -(-n_pairs // SCRUB_PAIRS) - 1).bit_length()


def build_advance_step(scr_pairs: int, sym_pairs: int):
    """The ``@flyc.jit`` step advance for one scratch: ``step += 1`` plus a scrub.

    Int32 tags wrap, and a mailbox left unwritten that long would read as fresh, so each
    advance zeroes (CAS) pairs older than the last step in one of ``scrub_period``
    slices; tags are never 0, and a peer one launch ahead only writes newer tags."""
    n = scr_pairs + sym_pairs
    period = scrub_period(n)
    chunk = -(-n // period)
    per_thread = -(-chunk // THREADS)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def advance_kernel(step: Int64, scratch: Int64, sym: Int64):
        tid = fx.thread_idx.x

        def scrub(addr, new_base):
            """Zero the pair at ``addr`` if its tag is nonzero and at most ``new_base``."""
            ptr = fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8), addr)
            old = fx.Int64(fx.generic_load(ptr, memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent"))
            tg = fx.Int32(old >> 32)  # a pair is (value, tag): the tag is the high word
            if (tg != 0) & ((new_base - tg) >= 0):
                fx.atomic_cas(ptr, old, fx.Int64(0))

        s = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32))
        new_base = s * LAYER_SLOTS  # every tag of steps < s is at most this
        base = (s & (period - 1)) * chunk
        for j in range_constexpr(per_thread):
            o = j * THREADS + tid
            i = base + o
            if (o < chunk) & (i < n):
                if i < scr_pairs:
                    scrub(scratch + fx.Int64(i) * 8, new_base)
                else:
                    scrub(sym + fx.Int64(i - scr_pairs) * 8, new_base)
        gpu.barrier()
        if tid == 0:
            bo.buffer_store(s + 1, rsrc(step), 0)

    @flyc.jit
    def advance_step(step: Int64, scratch: Int64, sym: Int64, stream: fx.Stream = fx.Stream(None)):
        advance_kernel(step, scratch, sym).launch(grid=(1,), block=(THREADS,), stream=stream)

    return advance_step
