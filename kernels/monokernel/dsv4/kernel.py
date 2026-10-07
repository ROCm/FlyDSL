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

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value
from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv4.config import (
    BLOCK_TOKENS,
    EPS,
    FP8_MAX,
    HC_EPS,
    HC_SINKHORN_ITERS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    MAX_LAYERS_PER_STEP,
    N_EXPERTS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    ROUTE_SCALE,
    SCALE_BM,
    SOFTMAX_SCALE,
    SWIGLU_LIMIT,
    TOP_K,
    WINDOW,
    ExpertActivation,
    ExpertWeight,
    MoeMode,
    moe_format,
)
from kernels.monokernel.layout import CM_DEV, CM_SYS, NEG
from kernels.monokernel.ops import (
    exp,
    f8_word,
    fp8_roundtrip,
    fp8_to_bf16x8,
    mem_realtime,
    mxfp4_to_bf16x8,
    mxfp8_to_bf16x8,
    rcp,
    rsq,
    rsrc,
    uniform,
    uniform_f32,
    wave_umax,
    write_lane_i32,
    xred,
    xshfl,
)

BLOCKS = 256
LAYER_SLOTS = MAX_LAYERS_PER_STEP
THREADS = 512
WAVES = THREADS // 64
QKV_A_TILE = 16
Q_B_TILE = 16
UV_TILE = 64  # output dims merged per uv task, at least: see uv_tile
ROW_TILE = 32  # rows per o_a / o_b / attention peer-reduce tile
ROUTER_TILE = 8  # experts per router task (a part of a 16-row MFMA group)
UG_TILE = 16  # intermediates per up/gate task (16 gate rows + 16 up rows)
UG8 = 8  # intermediates one up/gate task actually owns
SPLIT_KEYS = 64
HC_CPW = 2  # hc_pre 64-K chunks per wave; sets the K split across tasks
MIN_I32 = -(1 << 31)  # flips the sign bit: signed-ordered <-> unsigned-ordered

# task counts per stage
QKV_A_ROWS = Q_LORA + HEAD_DIM
N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE


def dn_tile(S: int, hidden: int = HIDDEN) -> int:
    """Hidden rows per expert-down / FFN peer-reduce task: 32 at S = 1, else about one
    task per CTA, rounded up to a multiple of 16 so ``emit_dn``'s tile never straddles
    the ``DN_R * 16`` rows ``reduce_rows`` produces (V4-Pro's 28 would drop rows)."""
    return 32 if S == 1 else max(16, -(-(hidden // BLOCKS) // 16) * 16)


def qkv_a_groups(n_tiles: int) -> int:
    """16-row groups per qkv_a task: two (four waves each splitting K) when the tiles
    outnumber the grid, so the input is not staged twice."""
    return 2 if n_tiles > BLOCKS and n_tiles % 2 == 0 else 1


def q_b_groups(n_tiles: int) -> int:
    """16-row groups per q_b task, by qkv_a_groups' rule."""
    return qkv_a_groups(n_tiles)


def uv_tile(S: int, heads: int, head_dim: int) -> int:
    """Output dims merged per uv task: UV_TILE, widened until the tasks fit one round
    of the grid (every task redoes its head's split weights)."""
    t = UV_TILE
    while S * heads * head_dim // t > BLOCKS and t * 2 <= head_dim:
        t *= 2
    return t


def o_a_spt(S: int, o_groups: int, o_lora: int) -> int:
    """Samples per o_a task: two (in two MFMA B columns) once the tasks outgrow the grid."""
    return 2 if S % 2 == 0 and S * o_groups * o_lora // ROW_TILE > BLOCKS else 1


def ffn_hcc(S: int, hc_mult: int, n_experts: int = N_EXPERTS) -> bool:
    """Whether the FFN contracts its hc_mult streams in a stage of its own (hcc_f,
    publishing ``ain``) rather than in every router task: only once router tasks
    carry more than one sample."""
    return hc_mult > 1 and router_spt(S, n_experts) > 1


def router_spt(S: int, n_experts: int = N_EXPERTS) -> int:
    """Samples per router task: the fewest (at most 8) that fit the grid in one round."""
    n_router = n_experts // ROUTER_TILE
    spt = 1
    while S * n_router > BLOCKS * spt and spt < ROUTER_TILE:
        spt *= 2
    return spt


N_UG_PER_SLOT = INTER // UG_TILE


# CPol scope: SC1 = device (past the per-XCD caches), SC0|SC1 = system (peers over XGMI)
POLL_MAX = 12  # mailbox specs polled per batch
# opt-in poll bound (poll_timeout_us): a poll waiting this long gives up and flags `hang`
POLL_TIMEOUT_US = 10_000_000
TL_COLS = 5  # timeline stamps per task: start, hint seen, inputs staged, compute done, end


def _align(n, a=256):
    return (n + a - 1) // a * a


def hc_shape(hc_mult: int, hidden: int):
    """(tasks per side, K per task, rows, values per task) for hc_pre: K = hc * hidden is
    split across tasks, each also publishing a partial sum of squares as an extra row."""
    if hc_mult <= 1:
        return 0, 0, 0, 0
    rows = ((2 + hc_mult) * hc_mult + 15) // 16 * 16
    k_total = hc_mult * hidden
    n_tasks = k_total // (64 * WAVES * HC_CPW)
    return n_tasks, k_total // n_tasks, rows, rows + 1


def layout(
    S: int,
    heads: int,
    npes: int,
    window: int = WINDOW,
    moe_mode: MoeMode | str = MoeMode.A8W4,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    hc_mult: int = 1,
    compress_ratio: int = 0,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    # unused here: all three builders take the same dims dict
    index_topk: int = 0,
    max_seq: int = 0,
    kv_fp8: bool = False,
    indexer_hadamard: bool = True,
    n_experts: int = N_EXPERTS,
    top_k: int = TOP_K,
    inter: int = INTER,
):
    """Byte offsets of the per-rank scratch and of the symmetric buffer.

    Every mailbox holds ``(value, tag)`` int32 pairs (8 bytes per element)."""
    fmt = moe_format(moe_mode)
    quant_group = fmt.activation_group
    xq_blocks = 0 if quant_group is None else hidden // quant_group
    n_split = (window if n_keys is None else n_keys) // SPLIT_KEYS
    hc_tasks, _, hc_rows, hc_vals = hc_shape(hc_mult, hidden)
    hc_coef = 2 * hc_mult + hc_mult * hc_mult  # pre | post | comb
    pr = 8
    items = [
        ("hc_d", S * 2 * max(hc_tasks, 1) * max(hc_vals, 1) * pr),
        ("hc_c", S * 2 * max(hc_coef, 1) * pr),
        # hc_pre's output: the hc_mult streams contracted by `pre` to one
        ("xin", S * hidden * pr if hc_mult > 1 else pr),
        ("ain", S * hidden * pr if ffn_hcc(S, hc_mult, n_experts) else pr),  # the FFN side's, see ffn_hcc
        ("q_a", S * q_lora * pr),
        ("kv_a", S * head_dim * pr),  # the single shared KV row, pre-norm
        # the compressor's kv / gate from the fused qkv_a GEMV (c_coff-wide: overlap)
        ("c_kv", S * c_coff * head_dim * pr if compress_ratio else pr),
        ("c_gate", S * c_coff * head_dim * pr if compress_ratio else pr),
        ("i_kv", S * c_coff * index_head_dim * pr if index_head_dim else pr),
        ("i_gate", S * c_coff * index_head_dim * pr if index_head_dim else pr),
        # rows this launch just wrote (cnew, kvnew, i_cnew): a CTA cannot rely on
        # seeing its own global store, so readers take them from these mailboxes
        ("cnew", S * head_dim * pr if compress_ratio else pr),
        ("kvnew", S * head_dim * pr),  # this launch's KV ring rows (bf16 values)
        ("q_raw", S * heads * head_dim * pr),  # q_b output, before the per-head RMS
        # the indexer's query, raw then rotated / Hadamard / FP4
        ("i_q_raw", S * index_heads * index_head_dim * pr if index_head_dim else pr),
        ("i_q", S * index_heads * index_head_dim * pr if index_head_dim else pr),
        ("i_wp", S * index_heads * pr if index_head_dim else pr),
        ("i_cnew", S * index_head_dim * pr if index_head_dim else pr),
        # one score per compressed entry; entries not yet written score NEG
        ("i_score", S * max(n_compressed(max_seq, compress_ratio), 1) * pr),
        # the indexer's picks, gathered instead of the caller's compressed indices
        ("i_sel", S * max((n_keys or window) - window, 1) * pr),
        # the top-k parts' bins, one set per radix digit
        ("tk_hist", S * 4 * n_topk_parts(max_seq, compress_ratio, index_head_dim) * 256 * pr or pr),
        ("q", S * heads * head_dim * pr),  # full per-head query: rope is inside it
        ("sp_acc", S * n_split * heads * head_dim * pr),
        ("sp_m", S * n_split * heads * pr),
        ("sp_l", S * n_split * heads * pr),
        ("o", S * heads * head_dim * pr),  # merged, de-rotated attention output
        ("o_lora", S * o_groups * o_lora * pr),
        ("a", S * max(hc_mult, 1) * hidden * pr),  # post-attention residual stream (bf16)
        ("scores", S * n_experts * pr),
        ("xq", S * hidden // (4 if quant_group is not None else 2) * pr),
        ("xqs", S * xq_blocks * pr),
        ("sel", S * (1 + top_k) * pr),
        ("prob", S * (1 + top_k) * pr),
        ("mid", S * (1 + top_k) * inter * pr),
        ("xqd", S * hidden * 4),  # debug: dequantized MoE activation (plain f32)
    ]
    off, scratch = 0, {}
    for name, size in items:
        scratch[name] = off
        off += _align(size)
    scratch["_bytes"] = off
    part = npes * S * hidden * pr
    sym = {"attn": 0, "ffn": part, "_bytes": 2 * part}
    return scratch, sym


def _wave_any(pred):
    """Whether ``pred`` holds on any lane of the wave (wave-uniform)."""
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, pred.ir_value())) != fx.Int64(0)


def _sqrt_softplus(x):
    """V4's router score sqrt(softplus(x)), softplus as max(x, 0) + log1p(exp(-|x|))."""
    sp = fx.max(x, fx.Float32(0.0)) + fmath.log1p(exp(-fmath.absf(x)))
    return fmath.sqrt(sp)


def _swiglu(g, u, limit):
    """SwiGLU with V4's clamp: ``up`` on both sides, ``gate`` only from above."""
    if const_expr(limit > 0):
        g = fx.min(g, fx.Float32(limit))
        u = fx.min(fx.max(u, fx.Float32(-limit)), fx.Float32(limit))
    return g * rcp(1.0 + exp(-g)) * u


def _sort_network(n):
    """Odd-even transposition compare-exchange pairs for ``n`` elements."""
    pairs = []
    for r in range(n):
        for i in range(r % 2, n - 1, 2):
            pairs.append((i, i + 1))
    return pairs


FP4_MAX = 6.0


def _fp4_roundtrip(a, b):
    """f32 pair -> E2M1 -> f32 pair (pre-scaled inputs), plus the codes word (``a``
    in the low nibble). Scaling is done in f32 around this; the scale operand is 1.0."""
    one = as_ir_value(fx.Float32(1.0))
    word = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, as_ir_value(fx.Int32(0)), a, b, one, 0))
    v2 = fx.Vector.make_type(2, fx.Float32)
    out = fx.Vector(rocdl.cvt_scalef32_pk_f32_fp4(res=v2, src=as_ir_value(word), scale=one, src_sel_index=0))
    return out[0], out[1], word


def _pow2_ceil(x):
    """Smallest power of two >= x: V4's FP4 block scale (reference.quant_dequant_fp4)."""
    bits = fx.Float32(x).bitcast(fx.Int32)
    man = bits & ((1 << 23) - 1)
    e = ((bits >> 23) & 0xFF) - 127 + (man != 0).select(fx.Int32(1), fx.Int32(0))
    return ((e + 127) << 23).bitcast(fx.Float32)


def _bf16x2_has_nan(w):
    """Whether either bf16 half of a word is NaN."""
    w = fx.Int32(w)
    return ((w & 0x7FFF) > fx.Int32(0x7F80)) | (((w >> 16) & 0x7FFF) > fx.Int32(0x7F80))


SCORE_TILE = 512  # one candidate per thread
IH_TASK = WAVES  # index heads per i_q / i_wp task (i_q: one per wave)


def n_compressed(max_seq: int, compress_ratio: int) -> int:
    return max_seq // compress_ratio if compress_ratio else 0


def n_index(max_seq: int, compress_ratio: int, index_head_dim: int, index_topk: int) -> int:
    """Compressed slots the attention can gather: the indexer's pick, capped."""
    if not index_head_dim:
        return 0
    return min(index_topk, n_compressed(max_seq, compress_ratio))


def n_topk_parts(max_seq: int, compress_ratio: int, index_head_dim: int) -> int:
    """CTAs the indexer's top-k splits its candidates over; the parts agree on each
    radix digit by summing one another's bins. Capped, since that exchange grows
    quadratically with the part count."""
    if not index_head_dim:
        return 0
    return min(16, max(1, n_compressed(max_seq, compress_ratio) // 4096))


def n_score_tiles(max_seq: int, compress_ratio: int, index_head_dim: int) -> int:
    """Tiles of compressed entries the indexer scores, sized for the whole cache;
    entries not yet written score NEG."""
    if not index_head_dim:
        return 0
    n = n_compressed(max_seq, compress_ratio)
    return (n + SCORE_TILE - 1) // SCORE_TILE


def qkv_a_rows(q_lora: int, head_dim: int, compress_ratio: int, c_coff: int, index_head_dim: int = 0) -> int:
    """Rows of the fused qkv_a GEMV: q_a, kv, the compressor's c_coff-wide pair and
    the indexer compressor's pair (reference.qkv_a_tail())."""
    if not compress_ratio:
        return q_lora + head_dim
    tail = 2 * c_coff * (head_dim + index_head_dim)
    return q_lora + head_dim + tail


def stage_tasks(
    S: int,
    heads: int,
    window: int = WINDOW,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    top_k: int = TOP_K,
    inter: int = INTER,
    hc_mult: int = 1,
    compress_ratio: int = 0,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    index_topk: int = 0,
    max_seq: int = 0,
    kv_fp8: bool = False,
    indexer_hadamard: bool = True,
    n_experts: int = N_EXPERTS,
):
    """[(stage name, task count)] in execution order."""
    n_qkv_a = (q_lora + head_dim) // QKV_A_TILE
    n_qkv_c = (qkv_a_rows(q_lora, head_dim, compress_ratio, c_coff, index_head_dim) - q_lora - head_dim) // QKV_A_TILE
    hc_tasks, _, _, _ = hc_shape(hc_mult, hidden)
    n_iqb = index_heads * index_head_dim // Q_B_TILE
    return [
        ("hcd_a", S * hc_tasks),
        ("hcc_a", (hidden // ROW_TILE) if hc_mult > 1 else 0),
        ("qkv_a", n_qkv_a // qkv_a_groups(n_qkv_a)),
        # the compressors' projections, BF16 as ATOM
        ("qkv_c", n_qkv_c // qkv_a_groups(n_qkv_c)),
        ("cache", 1),
        ("cmp", S if compress_ratio else 0),
        ("i_cmp", S if index_head_dim else 0),
        ("q_b", heads * head_dim // Q_B_TILE // q_b_groups(heads * head_dim // Q_B_TILE)),
        ("q_norm", S * heads),
        ("i_q_b", n_iqb // q_b_groups(n_iqb) if index_head_dim else 0),
        ("i_q", S * index_heads // IH_TASK if index_head_dim else 0),
        ("i_wp", S * index_heads // IH_TASK if index_head_dim else 0),
        ("i_score", S * n_score_tiles(max_seq, compress_ratio, index_head_dim)),
        ("i_topk", S * n_topk_parts(max_seq, compress_ratio, index_head_dim)),
        ("split", S * ((window if n_keys is None else n_keys) // SPLIT_KEYS)),
        ("uv", S * (heads * head_dim // uv_tile(S, heads, head_dim))),
        ("o_a", S * o_groups * o_lora // ROW_TILE // o_a_spt(S, o_groups, o_lora)),
        ("o_b", hidden // ROW_TILE),
        ("hcd_f", S * hc_tasks),
        ("hcc_f", (hidden // ROW_TILE) if ffn_hcc(S, hc_mult, n_experts) else 0),
        ("router", S * (n_experts // ROUTER_TILE) // router_spt(S, n_experts)),
        # one tile per (routed slot, 8 intermediates); the first INTER / UG8 also carry the shared expert
        ("ug", S * top_k * (inter // UG8)),
        ("down", hidden // dn_tile(S, hidden)),
    ]


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

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def ld_bf16(r, i):
            return fx.Float32(fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16)))

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        # ---- tagged-pair mailboxes
        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base_addr, i, v, cm=CM_DEV):
            """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), rsrc(base_addr), i * 2, cache_modifier=cm)

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [fx.Float32(v0).bitcast(fx.Int32), tag, fx.Float32(v1).bitcast(fx.Int32), tag], fx.Int32
            )
            bo.buffer_store(vec, rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
            i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store."""
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(base_addr), i, cache_modifier=cm)

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)

        def poll(specs, scope="agent", batch=POLL_MAX):
            """Batched poll of mailbox pairs ``specs`` = [(base_addr, pair index, npairs in
            {1, 2})]: re-load the batch until every tag matches; one Int32 list per spec.
            The s_nop in the retry loop keeps the loads from being hoisted."""
            if const_expr(len(specs) == 0):
                return []
            if const_expr(len(specs) > batch):  # bound live registers
                return poll(specs[:batch], scope, batch) + poll(specs[batch:], scope, batch)
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

            def load_all():
                words = []
                for b, i, n in specs:
                    w = fx.Vector(
                        bo.buffer_load(rsrc(b), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32, cache_modifier=cm)
                    )
                    words += [w[e] for e in range(2 * n)]
                return fx.Vector.from_elements(words, fx.Int32)

            nw = sum(2 * n for _, _, n in specs)

            def unpack(v):
                outs_, e = [], 0
                for _, _, n in specs:
                    outs_.append([v[e + 2 * q] for q in range(n)])
                    e += 2 * n
                return outs_

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad

            # bounded: a timed-out poll stores the tag into ``hang``; later polls seeing it give up
            v = load_all()
            if const_expr(BOUNDED_POLL):
                t0 = mem_realtime()
                stop = fx.Int32(0)
                while pending(v) & (stop == 0):
                    rocdl.s_nop(0)
                    v = load_all()
                    flagged = (
                        uniform(bo.buffer_load(rsrc(hang), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) == tag
                    )
                    late = (mem_realtime() - t0) > fx.Int64(POLL_TIMEOUT_TICKS)
                    stop = late.select(fx.Int32(2), flagged.select(fx.Int32(1), fx.Int32(0)))
                if stop == 2:
                    bo.buffer_store(tag, rsrc(hang), 0, cache_modifier=CM_DEV)
            else:
                while pending(v):
                    rocdl.s_nop(0)
                    v = load_all()
            return unpack(v)

        def hint_wait(n, addr_of, mark=None):
            """Block barrier before a stage polls its inputs; consumers poll their payload
            directly (tight per-wave spins). ``mark`` stamps timeline column 1."""
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
            return [(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in poll([(b, i, 2) for b, i in specs])]

        # ---- wave reductions
        def wave_sum(v):
            for sh in range_constexpr(6):
                v = xred(v, 32 >> sh, lambda a, b: a + b)
            return v

        def wave_max(v):
            for sh in range_constexpr(6):
                v = xred(v, 32 >> sh, fx.max)
            return v

        def subgroup16_max(v):
            for off in (8, 4, 2, 1):
                v = xred(v, off, fx.max)
            return v

        def _other_parts(part):
            """Every top-k part but this one, starting just after it (a part never reads
            its own global store back)."""
            return [(part + 1 + k) % TK_PARTS for k in range(TK_PARTS - 1)]

        def part_keys(sbase, part, n_live, n_parts):
            """(trip bases, unsigned order-preserving keys) of this thread's candidates in part
            ``part``, held in registers. Bases are clamped and trips past ``n_live`` poll
            candidate 0 (never unwritten); the caller masks on the unclamped index."""
            cbs = [((fx.Int32(j) * n_parts + part) * THREADS + tid) * TK_PER for j in range(TK_TRIPS)]
            specs = []
            for cb in cbs:
                a = (cb < n_live).select(fx.min(cb, fx.Int32(N_COMP - TK_PER)), fx.Int32(0))
                specs += [(mb("i_score"), sbase + a + 2 * q, 2) for q in range(TK_PER // 2)]
            ws = [w[e] for w in poll(specs) for e in range(2)]
            return cbs, [(w ^ ((w >> 31) & 0x7FFFFFFF)) ^ MIN_I32 for w in ws]

        def block_excl_scan(v):
            """Exclusive prefix sum of a per-thread int32 over the block, and the block
            total: an xor-butterfly scan per wave, then the wave totals through LDS."""
            x = fx.Float32(v)
            pre = fx.Float32(0.0)
            for sh in range_constexpr(6):
                off = 1 << sh
                p = xshfl(x, off)
                pre = ((lane & off) != 0).select(pre + p, pre)
                x = x + p  # every lane now holds the sum of its 2 * off block
            if lane == 0:
                lds_st(red, wave, x)
            gpu.barrier()
            tot = fx.Float32(0.0)
            for i in range_constexpr(WAVES):
                t = lds_ld(red, i)
                pre = (fx.Int32(i) < wave).select(pre + t, pre)
                tot = tot + t
            gpu.barrier()
            return fx.Int32(pre), fx.Int32(tot)

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

        def block_max(v):
            w = wave_max(v)
            if lane == 0:
                lds_st(red, wave, w)
            gpu.barrier()
            t = lds_ld(red, 0)
            for i in range_constexpr(1, WAVES):
                t = fx.max(t, lds_ld(red, i))
            gpu.barrier()
            return t

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
        def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None, ln=None):
            """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
            bf16 activation chunk starts at LDS word ``b_word``."""
            ln = lane if ln is None else ln
            wv = fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
            if const_expr(callable(coef)):  # factor known only after a later wait
                return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8", [wv], s, b_word + (lane // 16) * 4)

        def unit_f8f8(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
            """Issue one 128-k chunk (64-k chunks kc, kc + 1) of row group ``rg`` against the
            FP8 activation at LDS word ``b_word``; ``coef()`` = activation scale (x route weight)."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
            return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

        def unit_fp8mx(w_rsrc, s_rsrc, rg, s_rg, kc, K, b_word, coef=None, ln=None):
            """One 128-K chunk of row group ``rg`` of an FP8 128x128-scaled matrix (``s_rg`` =
            the row group of this lane's output rows) against bf16 at LDS word ``b_word``."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(w_rsrc, ((rg * (K // 64) + kc * 2 + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32)
                )
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (s_rg * 16 // SCALE_BM) * (K // 128) + kc)
            return ("fp8mx", wv, (s, coef), b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, ln=None):
            """Issue one 128-K tile of an MXFP4 bank in ATOM's gfx950 layout and its E8M0
            scale (lane ``16 * kl + r`` holds one 32-K scale block of row ``r``; the scale
            dword's four bytes cover tiles kc (even, odd) x row groups (even, odd))."""
            ln = lane if ln is None else ln
            k1n = -(-(K // 32) // 8)  # scale K blocks, padded to 8, in groups of 8
            raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            word = fx.Int32(bo.buffer_load(s_rsrc, ((rg // 2) * k1n + kc // 2) * 64 + ln, vec_width=1, dtype=T.i32))
            sc_byte = word.shrui(((kc % 2) * 2 + rg % 2) * 8) & fx.Int32(0xFF)
            return ("mxfp4", (raw, sc_byte), coef, b_word + (lane // 16) * 16)

        def mx_rg(c):
            """Up/gate tile ``c`` (8 intermediates: gate rows on lanes r < 8, up rows on
            r >= 8) as this lane's row group of the gate/up bank in ATOM's order."""
            return (c // 2) * 2 + (lane % 16) // 8

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for unit_format, wv, coef, bw in units:
                if const_expr(unit_format == "fp8mx"):
                    # one factor per unit, so the four K32 MFMAs chain into one partial
                    ws, f = coef
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    for sp in range_constexpr(4):
                        a = fp8_to_bf16x8(wv[sp // 2][(sp % 2) * 2], wv[sp // 2][(sp % 2) * 2 + 1])
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    f = ws if const_expr(f is None) else ws * (f() if const_expr(callable(f)) else f)
                    acc = [acc[e] + c[e] * f for e in range(4)]
                    continue
                if const_expr(callable(coef) and unit_format != "mxfp4"):
                    coef = coef()
                if const_expr(unit_format == "mxfp4"):
                    # step sp takes K 32 * kl + 8 * sp .. of the lane's block (B words 16 * kl + 4 * sp)
                    raw, sc_byte = wv
                    assert not isinstance(coef, list), "per-K32 factors are folded into the operands"
                    c = (
                        fx.Vector.from_elements(acc, fx.Float32)
                        if coef is None
                        else fx.Vector.filled(4, 0.0, fx.Float32)
                    )
                    sc = (sc_byte << fx.Int32(23)).bitcast(fx.Float32)
                    for sp in range_constexpr(4):
                        a = mxfp4_to_bf16x8(raw[sp], sc)
                        b = fx.ptr_load(xs + (bw + sp * 4), result_type=v4f).bitcast(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    if const_expr(coef is None):
                        acc = [c[e] for e in range(4)]
                    else:
                        f = coef() if const_expr(callable(coef)) else coef
                        acc = [acc[e] + c[e] * f for e in range(4)]
                    continue
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                if const_expr(unit_format == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                    a = fx.Vector.from_elements([wv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                    bv = [
                        fx.Vector(fx.ptr_load(xs + (bw + h * 16), result_type=v4f)).bitcast(fx.Int32) for h in range(2)
                    ]
                    b = fx.Vector.from_elements([bv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                    one = fx.Int32(127)
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one])
                    )
                for sp in range_constexpr(2 if unit_format != "f8f8" else 0):
                    if const_expr(unit_format == "fp8"):
                        a = fp8_to_bf16x8(wv[0][sp * 2], wv[0][sp * 2 + 1])
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

        def reduce_rows(R, acc, emit):
            """Sum the per-wave MFMA tiles of each of R row groups; emit(row_local, n, v) for n < S."""
            wpr = WAVES // R
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
            gpu.barrier()
            n_out = R * 16 * S
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

        def _rmsnorm_tail_ks(n):
            """This thread's group-of-4 starting indices of n elements. A ragged tail is
            clamped to the last group (idempotent rewrites); ``active`` masks its
            contribution to sums (None when there is no tail)."""
            nq = n // 4
            full = nq // THREADS
            ks = [(tid + i * THREADS) * 4 for i in range(full)]
            active = None
            if const_expr(nq % THREADS):
                w = tid + full * THREADS
                active = w < nq
                ks.append(fx.min(w, nq - 1) * 4)
            return ks, active

        def stage_x_rmsnorm(ld4s, n, gamma, loaded=None, count=S):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
            ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
            ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
            ks, active = _rmsnorm_tail_ks(n)
            per = len(ks)
            gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            sss = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for a in vals[s * per + i]:
                        term = a * a
                        if const_expr(active is not None and i == per - 1):
                            term = active.select(term, fx.Float32(0.0))
                        ss = ss + term
                sss.append(ss)
            rstds = [rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
            for s in range_constexpr(count):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (s * n + ks[i]) // 2 + j,
                            bf16_pair(a[2 * j] * rstds[s] * gs[i][2 * j], a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1]),
                        )
            return rstds

        def load_x_rmsnorm(ld4s, n, gamma, count=S):
            """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
            rg_ = rsrc(gamma)
            ks, _ = _rmsnorm_tail_ks(n)
            gs = []
            for k in ks:
                g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(count) for k in ks])

        def stage_x_pairs(name, n_total, src_of):
            """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
            (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements."""
            nq = n_total // 4
            full = nq // THREADS
            vals = poll([(mb(name), src_of((tid + i * THREADS) * 4) // 2, 2) for i in range(full)])
            for i in range_constexpr(full):
                for j in range_constexpr(2):
                    lds_st(xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32))
            if const_expr(nq % THREADS):
                w = tid + full * THREADS
                if w < nq:
                    v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
                    for j in range_constexpr(2):
                        lds_st(xs, w * 2 + j, v[j].bitcast(fx.Float32))

        def quant_scaled(a0, a1):
            """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
            amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
            inv = nz.select(rcp(amax) * FP8_MAX, fx.Float32(1.0))
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, qs

        def quant_mxfp8(a0, a1):
            """Per-16-lane/32-value MXFP8 quantization; the E8M0 scale rounds up so the
            block max never clips."""

            amax = subgroup16_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            scale = nz.select(_pow2_ceil(amax * (1.0 / FP8_MAX)), fx.Float32(1.0))
            inv = nz.select(rcp(scale), fx.Float32(1.0))
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            d0, d1 = fp8_roundtrip(q0, q1)
            return d0, d1, scale

        def stage_moe_input(samples):
            """Stage normalized expert inputs published by the router into LDS."""
            if const_expr(use_fp8_block128):
                # HIDDEN // 4 slots of 4 FP8 bytes; ragged tail clamped (see _rmsnorm_tail_ks)
                nq = HIDDEN // 4
                full = nq // THREADS
                xk = [tid + i * THREADS for i in range(full)]
                if const_expr(nq % THREADS):
                    xk.append(fx.min(tid + full * THREADS, nq - 1))
                nxw = len(xk)
                got = poll(
                    [(mb("xq"), sx * nq + k, 1) for sx in samples for k in xk]
                    + [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
                )
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(nxw):
                        wd = f8_word(xk[i] * 4)
                        lds_st(xs, j * nq + wd, got[j * nxw + i][0].bitcast(fx.Float32))
                    if tid < XQ_BLOCKS:
                        lds_st(
                            misc,
                            8 + j * XQ_BLOCKS + tid,
                            got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                        )
            elif const_expr(use_mxfp8_block32):
                chunks = HIDDEN // 8
                per_thread = (chunks + THREADS - 1) // THREADS
                # the power-of-two block scale folds exactly into the conversion
                data_specs, scale_specs = [], []
                for sx in samples:
                    for i in range_constexpr(per_thread):
                        chunk = fx.min(tid + i * THREADS, chunks - 1)
                        data_specs.append((mb("xq"), sx * (HIDDEN // 4) + chunk * 2, 2))
                        scale_specs.append((mb("xqs"), sx * XQ_BLOCKS + chunk // 4, 1))
                got = poll(data_specs + scale_specs)
                nd = len(data_specs)
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(per_thread):
                        chunk = tid + i * THREADS
                        if chunk < chunks:
                            words = got[j * per_thread + i]
                            qs = got[nd + j * per_thread + i][0].bitcast(fx.Float32)
                            values = mxfp8_to_bf16x8(words[0], words[1], qs)
                            for pair in range_constexpr(4):
                                lds_st(
                                    xs,
                                    j * (HIDDEN // 2) + chunk * 4 + pair,
                                    fx.Vector.from_elements(
                                        [values[2 * pair], values[2 * pair + 1]], fx.BFloat16
                                    ).bitcast(fx.Float32)[0],
                                )
            else:
                nxw = HIDDEN // 2 // THREADS
                got = poll(
                    [(mb("xq"), sx * (HIDDEN // 2) + tid + i * THREADS, 1) for sx in samples for i in range(nxw)]
                )
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(nxw):
                        lds_st(
                            xs,
                            j * (HIDDEN // 2) + tid + i * THREADS,
                            got[j * nxw + i][0].bitcast(fx.Float32),
                        )

        def st_f8(k, q0, q1):
            """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
            k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
            w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
            nb = xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

        def load_bias():
            """This lane's 4 expert biases (issue before the scores wait)."""
            return [ld_f32(rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

        def route_topk(s, raws=None, bs=None):
            """Flat top-``TOP_K`` of sample s (one whole wave): each round is one u32 wave
            max over packed keys of (score + bias). Returns (expert id, raw score / sum *
            ROUTE_SCALE) of pick ``lane``, valid in lanes < TOP_K. A hash-routed layer
            (``use_hash``, runtime) takes the ids from ``tid2eid[token]`` instead."""
            KPL = N_EXPERTS // 64  # selection keys held per lane
            if const_expr(bs is None):
                bs = load_bias()
            if const_expr(raws is None):
                raws = getf_many([(mb("scores"), s * N_EXPERTS + lane + i * 64) for i in range(KPL)])
            ks = []
            for i in range_constexpr(KPL):
                kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                ks.append(fx.Uint32((ok & fx.Int32(-(1 << ID_BITS))) | (ID_MASK - (lane + i * 64))))
            # sort each lane's keys descending; a round takes the wave max of the heads
            for a, b in _sort_network(KPL):
                ks[a], ks[b] = fx.max(ks[a], ks[b]), fx.min(ks[a], ks[b])
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            mv = fx.Int32(0)  # lane k: the key of pick k
            for k in range_constexpr(TOP_K):
                m = wave_umax(ks[0])
                hit = ks[0] == m
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(KPL)] + [ks[KPL]]
                mv = write_lane_i32(m, k, mv)
            e = ID_MASK - (mv & ID_MASK)
            # a scored layer passes null tables: zero records makes the loads return 0
            nrec = (use_hash != 0).select(fx.Int32(0x7FFFFFF0), fx.Int32(0))
            r_tok = bo.create_buffer_resource_from_addr(tok_ids, num_records_bytes=nrec)
            r_t2e = bo.create_buffer_resource_from_addr(tid2eid, num_records_bytes=nrec)
            tok = fx.Int32(bo.buffer_load(r_tok, s, vec_width=1, dtype=T.i32))
            hk = tok * TOP_K + fx.min(lane, fx.Int32(TOP_K - 1))
            e_hash = fx.Int32(bo.buffer_load(r_t2e, hk, vec_width=1, dtype=T.i32))
            e = (use_hash != 0).select(e_hash, e)
            src = (e % 64) * 4
            got = [fx.Int32(rocdl.ds_bpermute(T.i32, src.ir_value(), r.bitcast(fx.Int32).ir_value())) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, N_EXPERTS // 64):
                raw = (e // 64 == i).select(got[i], raw)
            raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
            tot = raw
            for off in TOPK_SUM_OFFS:
                tot = xred(tot, off, lambda a, b: a + b)
            return e, raw * (rcp(tot) * ROUTE_SCALE)

        def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
            """Push BF16 partials to every peer, then sum all ranks' in rank order.
            ``residual``: fn(s, row) -> (r0, r1), a mailbox base polled with the peers,
            or None when the caller adds its own (hc_post)."""
            if const_expr(W > 1):
                # one wave per destination peer
                if wave < W:
                    pair_count = S * tile // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = lane + batch * 64
                        if pair < pair_count:
                            si = pair // (tile // 2)
                            ri = (pair % (tile // 2)) * 2
                            put_bf(
                                peer_dst + fx.Int64(SY[region]),
                                (rank * S + si) * HIDDEN + t * tile + ri,
                                [lds_ld(outs, si * tile + ri), lds_ld(outs, si * tile + ri + 1)],
                                CM_SYS,
                            )
                gpu.barrier()
            if tid < S * tile // 2:
                s = tid // (tile // 2)
                r = (tid % (tile // 2)) * 2
                row = t * tile + r
                r0 = fx.Float32(0.0)
                r1 = fx.Float32(0.0)
                if const_expr(callable(residual)):
                    r0, r1 = residual(s, row)
                v0 = lds_ld(outs, s * tile + r)
                v1 = lds_ld(outs, s * tile + r + 1)
                if const_expr(W == 1):
                    parts = [(v0, v1)]
                    got = []
                    if const_expr(residual is not None and not callable(residual)):
                        got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
                else:
                    own = sym + fx.Int64(SY[region])
                    specs = [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
                    if const_expr(residual is not None and not callable(residual)):
                        specs.append((residual, (s * HIDDEN + row) // 2, 1))
                    got = poll(specs, "one-as")
                    parts = [bf2_f32(v[0]) for v in got[:W]]
                    got = got[W:]
                if const_expr(residual is not None and not callable(residual)):
                    r0, r1 = bf2_f32(got[0][0])
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                out_fn(s, row, r0 + t0, r1 + t1)

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        def stamp(name, t, which, pred=None):
            if const_expr(timeline):
                # nothing may cross the clock read (constrains timeline builds only)
                rocdl.sched_barrier(0)
                # a masked-off rep must not overwrite the live task's row
                ok = (tid == 0) if const_expr(pred is None) else ((tid == 0) & pred)
                if ok:
                    now = mem_realtime()
                    fx.generic_store(
                        fx.inttoptr(
                            fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                            timeline_buf + fx.Int64((first[name] + t) * TL_COLS + which) * 8,
                        ),
                        now,
                    )

        def n_sel():
            """This lane's MFMA B column (sample); columns >= S duplicate the last one."""
            return fx.min(lane % 16, S - 1)

        # ======== 0. hyper-connection pre-mix: partial dots + partial sum of squares
        def hc_stage_coef(sd):
            """Every sample's pre | post | comb into LDS."""
            if tid < S * HC_COEF:
                s_ = tid // HC_COEF
                lds_st(misc, HC_MISC + tid, getf(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + tid % HC_COEF))
            gpu.barrier()

        def hc_post(s, row, v0, v1, res_word, emit):
            """out[k] = post[k] * x + sum_j comb[j, k] * residual[j], for a row pair.
            ``v0``/``v1`` are rounded to bf16 first, as the model's all-reduce returns bf16."""
            v0 = bf16_round(v0)
            v1 = bf16_round(v1)
            rj = [bf2_f32(res_word(s, j, row)) for j in range_constexpr(HC)]
            for k in range_constexpr(HC):
                pk = lds_ld(misc, HC_MISC + s * HC_COEF + HC + k)
                o0 = pk * v0
                o1 = pk * v1
                for j in range_constexpr(HC):
                    cjk = lds_ld(misc, HC_MISC + s * HC_COEF + 2 * HC + j * HC + k)
                    o0 = o0 + cjk * rj[j][0]
                    o1 = o1 + cjk * rj[j][1]
                emit(s, k, row, o0, o1)

        def hc_pre_stages(side, sd, fn_ptr, sb_ptr, src_word, out_name, contract=True, src_pair=None):
            """hcd (the mixing projection's partials) and, with ``contract``, hcc (the
            contracted input). Returns the coefficient routine. ``src_pair(s, k)``: the
            source's (mailbox, pair), so hcc polls a row's streams in one batch."""
            r_fn = rsrc(fn_ptr)
            r_sb = rsrc(sb_ptr)  # [3 scales | HC_MIX bases]
            for tt in range(start(f"hcd_{side}"), S * HC_TASKS, G):
                tt = fx.Int32(tt)
                stamp(f"hcd_{side}", tt, 0)
                s = tt // HC_TASKS
                t = tt % HC_TASKS
                w = src_word(s, t * HC_KSLICE + tid * 2)
                lds_st(xs, tid, w.bitcast(fx.Float32))
                x0, x1 = bf2_f32(w)
                ssq = block_sum(x0 * x0 + x1 * x1)
                stamp(f"hcd_{side}", tt, 2)
                gpu.barrier()

                # the fp32 mixer is packed as bf16 hi / lo (pack_hc_fn), both into one tile
                HC_UPW = HC_NKC // HC_WPR

                def u_hc(c, t=t):
                    lo = 1 if c >= HC_UPW else 0
                    kc = (wave % HC_WPR) * HC_UPW + c % HC_UPW
                    return unit_bf16(r_fn, wave // HC_WPR + lo * HC_RG, t * HC_NKC + kc, HC_NKC_FULL, kc * 32)

                acc = run_units(u_hc, 2 * HC_UPW, 2 * HC_UPW)
                reduce_rows(HC_RG, acc, emit_out(HC_ROWS))
                stamp(f"hcd_{side}", tt, 3)
                gpu.barrier()
                base_i = ((s * 2 + sd) * HC_TASKS + t) * HC_VALS
                if tid < HC_ROWS:
                    put(mb("hc_d"), base_i + tid, lds_ld(outs, tid))
                if tid == HC_ROWS:
                    put(mb("hc_d"), base_i + HC_ROWS, ssq)
                stamp(f"hcd_{side}", tt, 4)

            def hc_coefficients(sd, publish):
                """Reduce the hcd partials, take the RMS scale, run the Sinkhorn; redone
                per hcc task (no extra stage), ``publish`` puts them for hc_post."""
                sc0 = ld_f32(r_sb, 0)
                sc1 = ld_f32(r_sb, 1)
                sc2 = ld_f32(r_sb, 2)
                # HC_PW waves poll one batch each; wave 0 adds them in wave order (rank-identical)
                NBLK = (S * HC_VALS + 63) // 64
                for blk in range_constexpr(NBLK):
                    idx = lane + blk * 64
                    if (wave < HC_PW) & (idx < S * HC_VALS):
                        s_ = idx // HC_VALS
                        j = idx % HC_VALS
                        tasks = [fx.min(wave * HC_TPW + i, HC_TASKS - 1) for i in range(HC_TPW)]
                        parts = poll([(mb("hc_d"), ((s_ * 2 + sd) * HC_TASKS + ti) * HC_VALS + j, 1) for ti in tasks])
                        tot = fx.Float32(0.0)
                        for i in range_constexpr(HC_TPW):
                            ok = (wave * HC_TPW + i) < HC_TASKS
                            tot = tot + ok.select(parts[i][0].bitcast(fx.Float32), fx.Float32(0.0))
                        lds_st(red, (wave * NBLK + blk) * 64 + lane, tot)
                gpu.barrier()
                for blk in range_constexpr(NBLK):
                    idx = lane + blk * 64
                    if (wave == 0) & (idx < S * HC_VALS):
                        tot = fx.Float32(0.0)
                        for w_ in range_constexpr(HC_PW):
                            tot = tot + lds_ld(red, (w_ * NBLK + blk) * 64 + lane)
                        lds_st(red, idx, tot)
                gpu.barrier()
                # one wave per sample: the samples' Sinkhorn chains run in parallel
                if wave < S:
                    s_ = wave
                    rstd = rsq(lds_ld(red, s_ * HC_VALS + HC_ROWS) * (1.0 / (HC * HIDDEN)) + EPS)

                    def coef(i, s_=s_, rstd=rstd):
                        return lds_ld(red, s_ * HC_VALS + i) * rstd

                    if lane < 2 * HC:  # pre then post share these lanes
                        m = coef(fx.min(lane, fx.Int32(HC_MIX - 1)))
                        b = ld_f32(r_sb, 3 + lane)
                        v = (lane < HC).select(
                            rcp(1.0 + exp(-(m * sc0 + b))) + hc_eps,
                            2.0 * rcp(1.0 + exp(-(m * sc1 + b))),
                        )
                        lds_st(misc, HC_MISC + s_ * HC_COEF + lane, v)
                        if publish:
                            put(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + lane, v)
                    if lane < HC * HC:
                        cb = coef(2 * HC + lane) * sc2 + ld_f32(r_sb, 3 + 2 * HC + lane)
                        rmax = cb
                        for off in HC_ROW_OFFS:
                            rmax = xred(rmax, off, fx.max)
                        c = exp(cb - rmax)
                        rsum = c
                        for off in HC_ROW_OFFS:
                            rsum = xred(rsum, off, lambda a, b: a + b)
                        c = c * rcp(rsum) + hc_eps
                        csum = c
                        for off in HC_COL_OFFS:
                            csum = xred(csum, off, lambda a, b: a + b)
                        c = c * rcp(csum + hc_eps)
                        for _ in range_constexpr(hc_sinkhorn_iters - 1):
                            rsum = c
                            for off in HC_ROW_OFFS:
                                rsum = xred(rsum, off, lambda a, b: a + b)
                            c = c * rcp(rsum + hc_eps)
                            csum = c
                            for off in HC_COL_OFFS:
                                csum = xred(csum, off, lambda a, b: a + b)
                            c = c * rcp(csum + hc_eps)
                        lds_st(misc, HC_MISC + s_ * HC_COEF + 2 * HC + lane, c)
                        if publish:
                            put(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + 2 * HC + lane, c)
                gpu.barrier()

            if const_expr(not contract):
                return hc_coefficients

            # --- contract the hc_mult streams by `pre` into the single-width input
            for t in range(start(f"hcc_{side}"), N_ROW_TILES, G):
                t = fx.Int32(t)
                stamp(f"hcc_{side}", t, 0)
                hc_coefficients(sd, t == 0)
                stamp(f"hcc_{side}", t, 2)
                if tid < S * ROW_TILE // 2:
                    s_ = tid // (ROW_TILE // 2)
                    r = (tid % (ROW_TILE // 2)) * 2
                    row = t * ROW_TILE + r
                    a0 = fx.Float32(0.0)
                    a1 = fx.Float32(0.0)
                    if const_expr(src_pair is not None):
                        ws = [v[0] for v in poll([src_pair(s_, j * HIDDEN + row) + (1,) for j in range(HC)])]
                    else:
                        ws = [src_word(s_, j * HIDDEN + row) for j in range(HC)]
                    for j in range_constexpr(HC):
                        pj = lds_ld(misc, HC_MISC + s_ * HC_COEF + j)
                        x0, x1 = bf2_f32(ws[j])
                        a0 = a0 + pj * x0
                        a1 = a1 + pj * x1
                    put_bf(mb(out_name), s_ * HIDDEN + row, [a0, a1])
                stamp(f"hcc_{side}", t, 4)
            return hc_coefficients

        if const_expr(HC > 1):
            hc_pre_stages(
                "a",
                0,
                hc_attn_fn,
                hc_attn_sb,
                lambda s, k: fx.Int32(bo.buffer_load(r_h, (s * HC * HIDDEN + k) // 2, vec_width=1, dtype=T.i32)),
                "xin",
            )

        # ================================================= 1. q_a / kv GEMV
        r_wqa, r_sqa = rsrc(w_qkv_a), rsrc(s_qkv_a)
        r_wqc = rsrc(w_qkv_c)
        QA_NKC = HIDDEN // 64

        def qkv_stage(name, n_tiles, row0, bf16_w):
            """One fused-GEMV stage over n_tiles 16-row groups from global row row0: the
            FP8 q_a | kv rows (qkv_a) or the compressors' BF16 rows (qkv_c)."""
            QA_R = qkv_a_groups(n_tiles)
            QA_WPR = WAVES // QA_R
            QA_UPW = QA_NKC // QA_WPR
            # units in flight per wave (larger BF16 batches spill VGPRs)
            QA_BATCH = max(1, QA_NKC // WAVES // (4 if bf16_w else 1))
            QA_ROWS = QKV_A_TILE * QA_R
            for t in range(start(name), n_tiles // QA_R, G):
                t = fx.Int32(t)
                stamp(name, t, 0)
                qa_rg = t * QA_R + wave // QA_WPR

                def u_qa(c):
                    kc = (wave % QA_WPR) * QA_UPW + c
                    if const_expr(bf16_w):
                        return unit_bf16(r_wqc, qa_rg, kc, QA_NKC, (n_sel() * HIDDEN + kc * 64) // 2)
                    return unit_fp8(r_wqa, r_sqa, qa_rg, kc, QA_NKC, HIDDEN, 128, (n_sel() * HIDDEN + kc * 64) // 2)

                def ld_h(sks):
                    if const_expr(HC > 1):  # hc_pre already contracted the streams
                        vals = poll([(mb("xin"), (s * HIDDEN + k) // 2, 2) for s, k in sks])
                        res = []
                        for i in range_constexpr(len(sks)):
                            a0, a1 = bf2_f32(vals[i][0])
                            b0, b1 = bf2_f32(vals[i][1])
                            res.append([a0, a1, b0, b1])
                        return res
                    res = []
                    for s, k in sks:
                        w = fx.Vector(bo.buffer_load(r_h, (s * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append([v[j] for j in range(4)])
                    return res

                # input loads go out before the weight stream (loads complete in order)
                h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in)
                pre = [u_qa(c) for c in range(QA_BATCH)]
                stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld)
                gpu.barrier()
                stamp(name, t, 2)
                acc = run_units(u_qa, QA_UPW, QA_BATCH, pre)
                reduce_rows(QA_R, acc, emit_out(QA_ROWS))
                stamp(name, t, 3)
                gpu.barrier()
                if tid < S * QA_ROWS:
                    s = tid // QA_ROWS
                    row = row0 + t * QA_ROWS + tid % QA_ROWS
                    v = lds_ld(outs, tid)
                    # column ranges of the fused GEMV (reference.qkv_a_split())
                    if row < Q_LORA:
                        put(mb("q_a"), s * Q_LORA + row, v)
                    elif row < Q_LORA + HEAD_DIM:
                        put(mb("kv_a"), s * HEAD_DIM + row - Q_LORA, v)
                    elif row < Q_LORA + HEAD_DIM + CW:
                        put(mb("c_kv"), s * CW + row - Q_LORA - HEAD_DIM, v)
                    elif row < Q_LORA + HEAD_DIM + 2 * CW:
                        put(mb("c_gate"), s * CW + row - Q_LORA - HEAD_DIM - CW, v)
                    elif row < Q_LORA + HEAD_DIM + 2 * CW + IW:
                        put(mb("i_kv"), s * IW + row - Q_LORA - HEAD_DIM - 2 * CW, v)
                    else:
                        put(mb("i_gate"), s * IW + row - Q_LORA - HEAD_DIM - 2 * CW - IW, v)
                stamp(name, t, 4)

        qkv_stage("qkv_a", N_QKV_A, 0, False)
        if const_expr(N_QKV_C):
            qkv_stage("qkv_c", N_QKV_C, QKV_A_ROWS, True)

        def kv_quant(nv):
            """This thread's KV channel through the NoPE FP8 round trip (one 64-group per
            wave, ue8m0 scale 2**ceil(log2(amax / 448)) as ATOM) -> (value, FP8 byte, E8M0 byte)."""
            amax = wave_max(fmath.absf(nv))
            sc = _pow2_ceil(fx.max(amax, fx.Float32(FP8_MAX * 2.0**-126)) * (1.0 / FP8_MAX))
            q = fx.min(fx.max(nv * rcp(sc), -FP8_MAX), FP8_MAX)  # rcp is exact on a power of two
            word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, q, fx.Int32(0), False))
            v2 = fx.Vector.make_type(2, fx.Float32)
            d = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))[0]
            return d * sc, word & 0xFF, (sc.bitcast(fx.Int32) >> 23) & 0xFF

        def put_kv_row(row, kvn, byte, e8):
            """Write one KV row (thread = channel; ``kvn`` its value, ``byte`` / ``e8``
            from kv_quant). CTA-uniform: the fp8 layout trades scale bytes through LDS."""
            if const_expr(KV_FP8):
                r_nope, r_rope = row_rsrc(kv_cache, row, KV_ROW_BYTES), row_rsrc(kv_rope, row, ROPE_DIM * 2)
                # four lanes' FP8 bytes to one dword, channel tid in byte tid % 4
                wb = byte << ((tid % 4) * 8)
                for off in (1, 2):
                    wb = xred(wb, off, lambda a, b: a | b)
                if (tid < NOPE_DIM) & (tid % 4 == 0):
                    bo.buffer_store(wb, r_nope, tid // 4)
                if tid >= NOPE_DIM:
                    bo.buffer_store(kvn.to(fx.BFloat16), r_rope, tid - NOPE_DIM)
                if (lane == 0) & (tid < NOPE_DIM):
                    lds_st(misc, wave, e8.bitcast(fx.Float32))
                gpu.barrier()
                # scale dword k: groups 2k and 2k + 1, each byte twice
                NG = NOPE_DIM // 64
                if tid < (NG + 1) // 2:
                    lo = lds_ld(misc, fx.min(2 * tid, fx.Int32(NG - 1))).bitcast(fx.Int32)
                    hi = (2 * tid + 1 < NG).select(
                        lds_ld(misc, fx.min(2 * tid + 1, fx.Int32(NG - 1))).bitcast(fx.Int32), fx.Int32(0)
                    )
                    sw = lo | (lo << 8) | (hi << 16) | (hi << 24)
                    bo.buffer_store(sw, r_nope, NOPE_DIM // 4 + tid)
                gpu.barrier()
            else:
                if tid < HEAD_DIM:
                    bo.buffer_store(kvn.to(fx.BFloat16), row_rsrc(kv_cache, row, HEAD_DIM * 2), tid)

        # ====== 2. KV RMSNorm + RoPE + FP8 round trip -> sliding-window ring cache
        # the NOPE_DIM lanes are FP8 round-tripped in 64-blocks, matching the checkpoint's QAT
        for t in range(start("cache"), 1, G):
            stamp("cache", t, 0)
            g = ld_bf16(rsrc(g_kv), fx.min(tid, HEAD_DIM - 1))
            ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
            ps = [ld_pos(sx) for sx in range(S)]
            cs = [ld_f32(rsrc(rope_cos), ps[sx] * (ROPE_DIM // 2) + ri) for sx in range(S)]
            sns = [ld_f32(rsrc(rope_sin), ps[sx] * (ROPE_DIM // 2) + ri) for sx in range(S)]
            hint_wait(
                HEAD_DIM // QKV_A_TILE,
                lambda k: (mb("kv_a"), (S - 1) * HEAD_DIM + k * QKV_A_TILE + QKV_A_TILE - 1),
                mark=("cache", t),
            )
            vs = getf_many([(mb("kv_a"), s * HEAD_DIM + fx.min(tid, HEAD_DIM - 1)) for s in range(S)])
            stamp("cache", t, 2)
            live = tid < HEAD_DIM
            ssq = block_sums([live.select(v * v, fx.Float32(0.0)) for v in vs])
            for s in range_constexpr(S):
                nv = vs[s] * rsq(ssq[s] * (1.0 / HEAD_DIM) + EPS) * g
                # rope tail: lane ^ 1 is the other half of this interleaved (2i, 2i+1) pair
                partner = xshfl(nv, 1)
                even = tid % 2 == 0
                rot = even.select(nv * cs[s] - partner * sns[s], partner * sns[s] + nv * cs[s])
                dq, byte, e8 = kv_quant(nv)
                kvn = bf16_round((tid < NOPE_DIM).select(dq, rot))
                put_kv_row(ld_dest(0, s), kvn, byte, e8)
                if live:
                    put(mb("kvnew"), s * HEAD_DIM + tid, kvn)
            stamp("cache", t, 4)

        # ============= 2b. KV compressor (HCA): rolling state, pooled on the boundary
        # each CR window's last token emits a per-channel softmax pool, normed / RoPE'd / FP8'd
        def window_pool(p, j_tok, ring, launch, own):
            """Online softmax over the compressor window ending at p -> (den, num). Element i
            comes from the ring, this token (``own``), or, for a token d back in this launch,
            its mailbox (``launch``): its state write may not be visible yet."""

            def fold(acc, svs, kvs):
                m, den, num = acc
                for e in range_constexpr(len(svs)):
                    m_new = fx.max(m, svs[e])
                    rescale = exp(m - m_new)
                    w = exp(svs[e] - m_new)
                    den = den * rescale + w
                    num = num * rescale + w * kvs[e]
                    m = m_new
                return [m, den, num]

            n_ring = C_ROWS if const_expr(TOK == 1) else C_ROWS - TOK
            chunk = CMP_CHUNK if const_expr(TOK == 1) else CMP_CHUNK_T
            for _i, acc in range(
                0, n_ring // chunk, fx.Int32(1), init=[fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)]
            ):
                ib = fx.Int32(_i) * chunk
                lds_ = [ring(ib + e) for e in range(chunk)]
                res = yield fold(
                    [fx.Float32(acc[0]), fx.Float32(acc[1]), fx.Float32(acc[2])],
                    [a for a, _ in lds_],
                    [b for _, b in lds_],
                )
            acc = [fx.Float32(res[0]), fx.Float32(res[1]), fx.Float32(res[2])]
            if const_expr(TOK > 1):
                svs, kvs = [], []
                for i in range_constexpr(C_ROWS - TOK, C_ROWS):
                    d = C_ROWS - 1 - i
                    if const_expr(d == 0):
                        sv, kv = own(i)
                    else:
                        r_sv, r_kv = ring(fx.Int32(i))
                        l_sv, l_kv = launch(d, i)
                        inl = fx.Int32(d) <= j_tok
                        sv, kv = inl.select(l_sv, r_sv), inl.select(l_kv, r_kv)
                    svs.append(sv)
                    kvs.append(kv)
                acc = fold(acc, svs, kvs)
            return acc[1], acc[2]

        def ring_row(p, i):
            """State-ring row of window element i (position p + 1 - C_ROWS + i)."""
            return (p + 1 + i + (C_RING - C_ROWS)) % C_RING

        if const_expr(CR):
            for tt in range(start("cmp"), S, G):
                tt = fx.Int32(tt)
                stamp("cmp", tt, 0)
                ch = fx.min(tid, HEAD_DIM - 1)
                live = tid < HEAD_DIM
                # tt is the sample; per-sample state keeps these unordered tasks from racing
                p = ld_pos(tt)
                rs_kv, rs_sc, sb = slot_rsrc(kv_state, tt, st_kv), slot_rsrc(score_state, tt, st_kv), 0
                slot = p % CR
                ap0 = [ld_f32(rsrc(ape), slot * CW + j * HEAD_DIM + ch) for j in range(C_COFF)]
                g = ld_bf16(rsrc(g_ckv), ch)
                # the window's first position (clamped: loaded unconditionally)
                anchor = fx.max(p + 1 - CR, fx.Int32(0))
                ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
                # one rope table per layer: a compressing layer's is on compress_rope_theta for all it rotates
                rc = ld_f32(rsrc(rope_cos), anchor * (ROPE_DIM // 2) + ri)
                rs = ld_f32(rsrc(rope_sin), anchor * (ROPE_DIM // 2) + ri)
                kvv = [getf(mb("c_kv"), (tt * C_COFF + j) * HEAD_DIM + ch) for j in range(C_COFF)]
                gtv = [getf(mb("c_gate"), (tt * C_COFF + j) * HEAD_DIM + ch) for j in range(C_COFF)]
                stamp("cmp", tt, 2)
                if live:
                    for j in range_constexpr(C_COFF):
                        w = sb + (p % C_RING) * CW + j * HEAD_DIM + ch
                        bo.buffer_store(kvv[j], rs_kv, w)
                        bo.buffer_store(gtv[j] + ap0[j], rs_sc, w)
                if (p + 1) % CR == 0:  # uniform across the CTA
                    j_tok = tt % TOK

                    # overlap: previous window's rows from their first half, current from the second
                    def c_half(i):
                        return (i >= CR).select(fx.Int32(1), fx.Int32(0)) if OVERLAP else 0

                    def c_ring(i):
                        wi = sb + ring_row(p, i) * CW + c_half(i) * HEAD_DIM + ch
                        return ld_f32(rs_sc, wi), ld_f32(rs_kv, wi)

                    def c_own(i):
                        h = 1 if (OVERLAP and i >= CR) else 0
                        return gtv[h] + ap0[h], kvv[h]

                    def c_launch(d, i):
                        h = 1 if (OVERLAP and i >= CR) else 0
                        sm = tt - fx.min(fx.Int32(d), j_tok)
                        q = fx.max(p - d, fx.Int32(0))
                        apq = ld_f32(rsrc(ape), (q % CR) * CW + h * HEAD_DIM + ch)
                        return (
                            getf(mb("c_gate"), (sm * C_COFF + h) * HEAD_DIM + ch) + apq,
                            getf(mb("c_kv"), (sm * C_COFF + h) * HEAD_DIM + ch),
                        )

                    den_, num_ = window_pool(p, j_tok, c_ring, c_launch, c_own)
                    pooled = num_ * rcp(den_)
                    pooled = bf16_round(pooled)
                    ssq = block_sum(live.select(pooled * pooled, fx.Float32(0.0)))
                    # the model's norm returns bf16
                    nv = bf16_round(pooled * rsq(ssq * (1.0 / HEAD_DIM) + EPS) * g)
                    partner = xshfl(nv, 1)
                    even = tid % 2 == 0
                    rot = even.select(nv * rc - partner * rs, partner * rs + nv * rc)
                    dq, byte, e8 = kv_quant(nv)
                    cv = bf16_round((tid < NOPE_DIM).select(dq, rot))
                    put_kv_row(comp_row(tt, p // CR), cv, byte, e8)
                    if live:
                        put(mb("cnew"), tt * HEAD_DIM + tid, cv)
                stamp("cmp", tt, 4)

        def had_pair(v0, v1, ln):
            """FWHT over IHD channels (lane ln holds ln and ln + 64), scaled by IHD**-0.5;
            the identity without INDEXER_HADAMARD (ATOM's indexer)."""
            if const_expr(not INDEXER_HADAMARD):
                return v0, v1
            # range_constexpr, not a while: the shuffle offset must stay a constant
            for h in range_constexpr(6):
                st = 1 << h
                p0, p1 = xshfl(v0, st), xshfl(v1, st)
                hi = (ln & st) != 0
                v0 = hi.select(p0 - v0, v0 + p0)
                v1 = hi.select(p1 - v1, v1 + p1)
            v0, v1 = v0 + v1, v0 - v1
            sc = float(IHD) ** -0.5
            return v0 * sc, v1 * sc

        def fp4_block(v):
            """FP4 round trip over 32-lane blocks, power-of-two scale -> (value, code, scale)."""
            amax = fmath.absf(v)
            for off in (16, 8, 4, 2, 1):
                amax = xred(amax, off, fx.max)
            sc = _pow2_ceil(fx.max(amax, fx.Float32(FP4_MAX * 2.0**-126)) * (1.0 / FP4_MAX))
            q = fx.min(fx.max(v * rcp(sc), -FP4_MAX), FP4_MAX)
            d, _, word = _fp4_roundtrip(q, fx.Float32(0.0))
            return d * sc, word & 0xF, sc

        # ========== 2c. the indexer's compressor: same pooling, Hadamard + FP4 tail
        if const_expr(IHD):
            for tt in range(start("i_cmp"), S, G):
                tt = fx.Int32(tt)
                stamp("i_cmp", tt, 0)
                # one WAVE covers the row: lane ln holds channels ln and ln + 64
                ln = lane
                ilive = wave == 0
                p = ld_pos(tt)
                rs_ikv, rs_isc, isb = slot_rsrc(i_kv_state, tt, st_i), slot_rsrc(i_score_state, tt, st_i), 0
                icb = ld_slot(tt, st_ic)  # the indexer's key cache slot (bytes)
                slot = p % CR
                chs = [ln, ln + 64]
                ap0 = [[ld_f32(rsrc(i_ape), slot * IW + j * IHD + c) for j in range(C_COFF)] for c in chs]
                gg = [ld_bf16(rsrc(g_ickv), c) for c in chs]
                anchor = fx.max(p + 1 - CR, fx.Int32(0))
                kvv = [[getf(mb("i_kv"), (tt * C_COFF + j) * IHD + c) for j in range(C_COFF)] for c in chs]
                gtv = [[getf(mb("i_gate"), (tt * C_COFF + j) * IHD + c) for j in range(C_COFF)] for c in chs]
                stamp("i_cmp", tt, 2)
                if ilive:
                    for e in range_constexpr(2):
                        for j in range_constexpr(C_COFF):
                            w = isb + (p % C_RING) * IW + j * IHD + chs[e]
                            bo.buffer_store(kvv[e][j], rs_ikv, w)
                            bo.buffer_store(gtv[e][j] + ap0[e][j], rs_isc, w)
                if (p + 1) % CR == 0:
                    j_tok = tt % TOK
                    pooled = []
                    for e in range_constexpr(2):

                        def i_ring(i, e=e):
                            coff = (i >= CR).select(fx.Int32(IHD), fx.Int32(0)) if OVERLAP else 0
                            wi = isb + ring_row(p, i) * IW + coff + chs[e]
                            return ld_f32(rs_isc, wi), ld_f32(rs_ikv, wi)

                        def i_own(i, e=e):
                            h = 1 if (OVERLAP and i >= CR) else 0
                            return gtv[e][h] + ap0[e][h], kvv[e][h]

                        def i_launch(d, i, e=e):
                            h = 1 if (OVERLAP and i >= CR) else 0
                            sm = tt - fx.min(fx.Int32(d), j_tok)
                            q = fx.max(p - d, fx.Int32(0))
                            apq = ld_f32(rsrc(i_ape), (q % CR) * IW + h * IHD + chs[e])
                            return (
                                getf(mb("i_gate"), (sm * C_COFF + h) * IHD + chs[e]) + apq,
                                getf(mb("i_kv"), (sm * C_COFF + h) * IHD + chs[e]),
                            )

                        den_, num_ = window_pool(p, j_tok, i_ring, i_launch, i_own)
                        pooled.append(bf16_round(num_ * rcp(den_)))
                    sq = pooled[0] * pooled[0] + pooled[1] * pooled[1]
                    for off in range_constexpr(6):
                        sq = xred(sq, 32 >> off, lambda a, b: a + b)
                    rs = rsq(sq * (1.0 / IHD) + EPS)
                    nv = [bf16_round(pooled[e] * rs * gg[e]) for e in range(2)]
                    # rope is entirely in the second half, so only channel ln + 64 rotates
                    rc = ld_f32(rsrc(rope_cos), anchor * (ROPE_DIM // 2) + ln // 2)
                    rs2 = ld_f32(rsrc(rope_sin), anchor * (ROPE_DIM // 2) + ln // 2)
                    partner = xshfl(nv[1], 1)
                    even = ln % 2 == 0
                    nv[1] = bf16_round(even.select(nv[1] * rc - partner * rs2, partner * rs2 + nv[1] * rc))
                    h0, h1 = had_pair(nv[0], nv[1], ln)
                    q0, q1 = bf16_round(h0), bf16_round(h1)
                    (o0, k0, s0), (o1, k1, s1) = fp4_block(q0), fp4_block(q1)
                    # eight lanes' codes OR into one word (nibble j); group g's exponent in byte g
                    cw = [k0 << ((ln % 8) * 4), k1 << ((ln % 8) * 4)]
                    for off in (1, 2, 4):
                        cw = [xred(w, off, lambda a, b: a | b) for w in cw]
                    e8 = [(sc.bitcast(fx.Int32) >> 23) & 0xFF for sc in (s0, s1)]
                    sw = (e8[0] << ((ln // 32) * 8)) | (e8[1] << ((ln // 32 + 2) * 8))
                    sw = xred(sw, 32, lambda a, b: a | b)
                    if ilive:
                        # FP4 pool (see K_PB): word w is group w // 4, dword w % 4 of its 16 bytes
                        e_i = p // CR
                        blk_i = bt_block(tt, e_i)
                        sl = e_i % K_PB
                        dbase = icb // 4 + blk_i * IC_BLK_WORDS + sl * 4
                        if ln % 8 == 0:
                            for h in range_constexpr(2):
                                w = ln // 8 + 8 * h
                                bo.buffer_store(cw[h], rsrc(i_cache), dbase + (w // 4) * IC_GRP_WORDS + w % 4)
                        if ln == 0:
                            sbase = icb // 16 + blk_i * IC_S_BLK + (sl % 16) * 4 + (sl % K_PB) // 16
                            for g in range_constexpr(IHD // 32):
                                bo.buffer_store(fx.Int8((sw >> (8 * g)) & 0xFF), rsrc(i_cache_s), sbase + g * K_PB)
                        put(mb("i_cnew"), tt * IHD + ln, bf16_round(o0))
                        put(mb("i_cnew"), tt * IHD + ln + 64, bf16_round(o1))
                stamp("i_cmp", tt, 4)

        # ========= 3b. the indexer's query: its own projection off q_a (no per-head RMS)
        if const_expr(IHD):
            r_wiqb, r_siqb = rsrc(w_i_q_b), rsrc(s_i_q_b)
            IQB_NKC = Q_LORA // 64
            IQB_R = q_b_groups(N_IQB)
            IQB_WPR = WAVES // IQB_R
            IQB_UPW = IQB_NKC // IQB_WPR
            IQB_ROWS = Q_B_TILE * IQB_R
            for t in range(start("i_q_b"), N_IQB // IQB_R, G):
                t = fx.Int32(t)
                stamp("i_q_b", t, 0)
                iqb_rg = t * IQB_R + wave // IQB_WPR

                def u_iqb(c):
                    kc = (wave % IQB_WPR) * IQB_UPW + c
                    return unit_fp8(r_wiqb, r_siqb, iqb_rg, kc, IQB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

                pre = [u_iqb(c) for c in range(IQB_NKC // WAVES)]
                hint_wait(
                    Q_LORA // QKV_A_TILE,
                    lambda k: (mb("q_a"), (S - 1) * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1),
                    mark=("i_q_b", t),
                )

                def ld_iqa(sks):
                    v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
                    return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

                stage_x_rmsnorm(ld_iqa, Q_LORA, g_q)
                stamp("i_q_b", t, 2)
                gpu.barrier()
                acc = run_units(u_iqb, IQB_UPW, IQB_NKC // WAVES, pre)
                reduce_rows(IQB_R, acc, emit_out(IQB_ROWS))
                stamp("i_q_b", t, 3)
                gpu.barrier()
                if tid < S * IQB_ROWS:
                    s = tid // IQB_ROWS
                    row = t * IQB_ROWS + tid % IQB_ROWS  # row of the [IH * IHD] query
                    put(mb("i_q_raw"), s * IH * IHD + row, lds_ld(outs, tid))
                stamp("i_q_b", t, 4)

            # ---- rope -> Hadamard -> FP4, one whole index head per wave
            for tt in range(start("i_q"), S * IH // IH_TASK, G):
                tt = fx.Int32(tt)
                stamp("i_q", tt, 0)
                s = tt // (IH // IH_TASK)
                ln = lane
                chs = [ln, ln + 64]
                ip = ld_pos(s)
                rc = ld_f32(rsrc(rope_cos), ip * (ROPE_DIM // 2) + ln // 2)
                rs2 = ld_f32(rsrc(rope_sin), ip * (ROPE_DIM // 2) + ln // 2)
                for k in range_constexpr(IH_TASK // WAVES):
                    ihead = (tt % (IH // IH_TASK)) * IH_TASK + wave + k * WAVES
                    base_i = (s * IH + ihead) * IHD
                    v = getf_many([(mb("i_q_raw"), base_i + c) for c in chs])
                    partner = xshfl(v[1], 1)
                    even = ln % 2 == 0
                    v[1] = bf16_round(even.select(v[1] * rc - partner * rs2, partner * rs2 + v[1] * rc))
                    v[0] = bf16_round(v[0])
                    h0, h1 = had_pair(v[0], v[1], ln)
                    o0, o1 = fp4_block(bf16_round(h0))[0], fp4_block(bf16_round(h1))[0]
                    put(mb("i_q"), base_i + chs[0], o0)
                    put(mb("i_q"), base_i + chs[1], o1)
                stamp("i_q", tt, 4)

        # ===== 3c. weights_proj: the per-head weight of the score's head sum (bf16, no MFMA)
        if const_expr(IHD):
            for tt in range(start("i_wp"), S * IH // IH_TASK, G):
                tt = fx.Int32(tt)
                stamp("i_wp", tt, 0)
                r_iw = rsrc(i_w)
                sw = tt // (IH // IH_TASK)  # the sample
                h0 = (tt % (IH // IH_TASK)) * IH_TASK  # its first head

                def ld_hw(sks, sw=sw):
                    # count=1 below: `sw` is the sample, the pair's `s` is always 0
                    if const_expr(HC > 1):
                        vals = poll([(mb("xin"), (sw * HIDDEN + k) // 2, 2) for _s, k in sks])
                        res = []
                        for i in range_constexpr(len(sks)):
                            a0, a1 = bf2_f32(vals[i][0])
                            b0, b1 = bf2_f32(vals[i][1])
                            res.append([a0, a1, b0, b1])
                        return res
                    res = []
                    for _s, k in sks:
                        w = fx.Vector(bo.buffer_load(r_h, (sw * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append([v[j] for j in range(4)])
                    return res

                # qkv_a's normed input, recomputed rather than republished
                ks, act = _rmsnorm_tail_ks(HIDDEN)
                gs, xv = load_x_rmsnorm(ld_hw, HIDDEN, g_in, count=1)
                ssq = fx.Float32(0.0)
                for i in range_constexpr(len(ks)):
                    for a in xv[i]:
                        term = a * a
                        if const_expr(act is not None and i == len(ks) - 1):
                            term = act.select(term, fx.Float32(0.0))
                        ssq = ssq + term
                rstd = rsq(block_sums([ssq])[0] * (1.0 / HIDDEN) + EPS)
                stamp("i_wp", tt, 2)
                parts = []
                for hh in range_constexpr(IH_TASK):
                    acc = fx.Float32(0.0)
                    for i in range_constexpr(len(ks)):
                        wv = (
                            fx.Vector(bo.buffer_load(r_iw, ((h0 + hh) * HIDDEN + ks[i]) // 2, vec_width=2, dtype=T.i32))
                            .bitcast(fx.BFloat16)
                            .to(fx.Float32)
                        )
                        for j in range_constexpr(4):
                            term = xv[i][j] * rstd * gs[i][j] * wv[j]
                            if const_expr(act is not None and i == len(ks) - 1):
                                term = act.select(term, fx.Float32(0.0))
                            acc = acc + term
                    parts.append(acc)
                tots = block_sums(parts)
                sc = float(IHD) ** -0.5 * float(IH) ** -0.5
                for hh in range_constexpr(IH_TASK):
                    if tid == 0:
                        put(mb("i_wp"), sw * IH + h0 + hh, bf16_round(tots[hh]) * sc)
                stamp("i_wp", tt, 4)

        # ==================================== 3. q_a RMSNorm -> q_b (raw f32 query)
        r_wqb, r_sqb = rsrc(w_q_b), rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        QB_R = q_b_groups(N_QB)
        QB_WPR = WAVES // QB_R
        QB_UPW = QB_NKC // QB_WPR
        QB_BATCH = QB_NKC // WAVES
        QB_ROWS = Q_B_TILE * QB_R
        for t in range(start("q_b"), N_QB // QB_R, G):
            t = fx.Int32(t)
            stamp("q_b", t, 0)
            qb_rg = t * QB_R + wave // QB_WPR

            def u_qb(c):
                kc = (wave % QB_WPR) * QB_UPW + c
                return unit_fp8(r_wqb, r_sqb, qb_rg, kc, QB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

            pre = [u_qb(c) for c in range(QB_BATCH)]
            hint_wait(
                Q_LORA // QKV_A_TILE,
                lambda k: (mb("q_a"), (S - 1) * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1),
                mark=("q_b", t),
            )

            def ld_qa(sks):
                v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
                return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

            stage_x_rmsnorm(ld_qa, Q_LORA, g_q)
            stamp("q_b", t, 2)
            gpu.barrier()
            acc = run_units(u_qb, QB_UPW, QB_BATCH, pre)
            reduce_rows(QB_R, acc, emit_out(QB_ROWS))
            stamp("q_b", t, 3)
            gpu.barrier()
            # f32: the per-head RMS sees the unrounded output; bf16 rounding comes after RoPE
            if tid < S * QB_ROWS:
                s = tid // QB_ROWS
                row = t * QB_ROWS + tid % QB_ROWS
                put(mb("q_raw"), (s * H + row // HEAD_DIM) * HEAD_DIM + row % HEAD_DIM, lds_ld(outs, tid))
            stamp("q_b", t, 4)

        if const_expr(IHD):
            # ===== 3d. score every compressed entry: score[c] = sum_h relu(q[h] . k[c]) * w[h]
            # only (sample, tile) tasks up to the batch's largest live-tile count run
            ISC_L = fx.Int32(0)
            for s_ in range_constexpr(S):
                nl = fx.min((ld_pos(s_) + 1) // CR, fx.Int32(N_COMP))
                ISC_L = fx.max(ISC_L, (nl > N_INDEX).select((nl + SCORE_TILE - 1) // SCORE_TILE, fx.Int32(0)))
            for tt in range(start("i_score"), S * ISC_L, G):
                tt = fx.Int32(tt)
                stamp("i_score", tt, 0)
                s = tt // ISC_L
                blk = tt % ISC_L
                # skip <= N_INDEX live entries (all kept, see i_topk) and tiles past the live ones
                n_live_s = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_COMP))
                if (n_live_s > N_INDEX) & (blk * SCORE_TILE < n_live_s):
                    # the query as f32, then as bf16 pairs at QBF (the MFMA B operand; exact)
                    QBF = IH * IHD
                    NQW = IH * IHD // 2  # element pairs of the query
                    qws = [fx.min(tid + i * THREADS, NQW - 1) for i in range((NQW + THREADS - 1) // THREADS)]
                    qv = get2_many([(mb("i_q"), s * IH * IHD + 2 * w2) for w2 in qws])
                    for i in range_constexpr(len(qws)):
                        q0, q1 = qv[i]
                        lds_st(xs, 2 * qws[i], q0)  # clamped duplicates rewrite the same value
                        lds_st(xs, 2 * qws[i] + 1, q1)
                        lds_st(xs, QBF + qws[i], bf16_pair(q0, q1))
                    if tid < IH:  # each head's weight polled once, read through LDS
                        lds_st(pl, tid, getf(mb("i_wp"), s * IH + tid))
                    gpu.barrier()
                    stamp("i_score", tt, 2)
                    c = blk * SCORE_TILE + tid
                    sp = ld_pos(s)
                    n_live = (sp + 1) // CR
                    r_ic2 = rsrc(i_cache)
                    r_ics = rsrc(i_cache_s)
                    icb = ld_slot(s, st_ic)  # bytes; the scale pool's base is 1/16 of it
                    j_tok = s % TOK
                    # MFMA: a wave's 64 entries as 4 x 16 A rows, the heads as B columns, 16 per group
                    hn = lane % 16
                    wcols, qbs = [], []
                    for hg in range_constexpr(N_IHG):
                        hd = hg * 16 + hn  # this lane's head in group hg
                        wcol = lds_ld(pl, fx.min(hd, fx.Int32(IH - 1)))
                        wcols.append((hd < IH).select(wcol, fx.Float32(0.0)))
                        qbs.append(QBF + (fx.min(hd, fx.Int32(IH - 1)) * IHD + (lane // 16) * 8) // 2)
                    for g in range_constexpr(4):
                        ec = fx.min(blk * SCORE_TILE + wave * 64 + g * 16 + hn, fx.Int32(N_COMP - 1))
                        blk_e = bt_block(s, ec)
                        sl_e = ec % K_PB
                        db = icb // 4 + blk_e * IC_BLK_WORDS + sl_e * 4 + lane // 16
                        sdb = icb // 64 + blk_e * (IC_S_BLK // 4) + sl_e % 16
                        kds = [
                            bo.buffer_load(r_ic2, db + kb * IC_GRP_WORDS, vec_width=1, dtype=T.i32)
                            for kb in range(IHD // 32)
                        ]
                        sds = [
                            bo.buffer_load(r_ics, sdb + kb * (K_PB // 4), vec_width=1, dtype=T.i32)
                            for kb in range(IHD // 32)
                        ]
                        # the keys decode once and serve every head group
                        ka = []
                        for kb in range_constexpr(IHD // 32):
                            bsc = (((fx.Int32(sds[kb]) >> ((sl_e // 16) * 8)) & 0xFF) << 23).bitcast(fx.Float32)
                            ka.append(mxfp4_to_bf16x8(fx.Int32(kds[kb]), bsc))
                        vs = [fx.Float32(0.0)] * 4
                        for hg in range_constexpr(N_IHG):
                            acc = fx.Vector.filled(4, 0.0, fx.Float32)
                            for kb in range_constexpr(IHD // 32):
                                b = fx.ptr_load(xs + (qbs[hg] + kb * 16), result_type=v4f).bitcast(fx.BFloat16)
                                acc = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [ka[kb], b, acc]))
                            # C[entry 4 * (lane // 16) + i][head lane % 16]: relu, weight
                            vs = [vs[i] + fx.max(acc[i], fx.Float32(0.0)) * wcols[hg] for i in range(4)]
                        # ... then sum the heads across the 16 lanes
                        for i in range_constexpr(4):
                            v = vs[i]
                            for off in (1, 2, 4, 8):
                                v = xred(v, off, lambda x, y: x + y)
                            if hn == i:
                                lds_st(red, wave * 64 + g * 16 + 4 * (lane // 16) + i, v)
                    gpu.barrier()
                    # entries this launch wrote may not be visible in the cache: their wave
                    # rescores them from i_cnew (token s - d wrote entry (sp - d) // CR)
                    for d in range_constexpr(TOK):
                        sd = s - fx.min(fx.Int32(d), j_tok)
                        pd = sp - d
                        ne = fx.max(pd, fx.Int32(0)) // CR
                        has_new = (fx.Int32(d) <= j_tok) & ((pd + 1) % CR == 0) & (blk == ne // SCORE_TILE)
                        if has_new & (wave == (ne - blk * SCORE_TILE) // 64):
                            nv = getf_many([(mb("i_cnew"), sd * IHD + lane + 64 * hf) for hf in range(IHD // 64)])
                            sc_n = fx.Float32(0.0)
                            for hh in range_constexpr(IH):
                                part = fx.Float32(0.0)
                                for hf in range_constexpr(IHD // 64):
                                    part = part + nv[hf] * lds_ld(xs, hh * IHD + lane + 64 * hf)
                                sc_n = sc_n + fx.max(wave_sum(part), fx.Float32(0.0)) * lds_ld(pl, hh)
                            if lane == 0:
                                lds_st(red, ne - blk * SCORE_TILE, sc_n)
                    gpu.barrier()
                    sc_t = lds_ld(red, tid)
                    live = (c < n_live) & (c < N_COMP)
                    stamp("i_score", tt, 3)
                    if c < N_COMP:
                        put(mb("i_score"), s * N_COMP + c, live.select(sc_t, fx.Float32(NEG)))
                stamp("i_score", tt, 4)

        # ============ 3e. top-k: which compressed entries the attention gathers
        # Exact radix select (8-bit digits, MSB first) over scores every rank computes
        # identically (the indexer is replicated), so every rank picks the same set.
        if const_expr(IHD):
            for tt in range(start("i_topk"), S * TK_PARTS, G):
                tt = fx.Int32(tt)
                stamp("i_topk", tt, 0)
                s = tt // TK_PARTS
                part = tt % TK_PARTS
                n_live = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_COMP))
                k_want = fx.min(n_live, fx.Int32(N_INDEX))
                sbase = s * N_COMP
                # <= N_INDEX live: take all; while they fit one part, part 0 selects alone
                n_parts = (n_live <= TK_TRIPS * THREADS * TK_PER).select(fx.Int32(1), fx.Int32(TK_PARTS))
                if (n_live > N_INDEX) & (part < n_parts):
                    tk_cbs, tk_keys = part_keys(sbase, part, n_live, n_parts)

                    def trip_live(j):
                        """Whether trip ``j`` of this part holds any live candidate (CTA-uniform)."""
                        return (fx.Int32(j) * n_parts + part) * (THREADS * TK_PER) < n_live

                    stamp("i_topk", tt, 2)

                    pfx = fx.Int32(0)  # the digits already fixed, in the unsigned domain
                    gt = fx.Int32(0)  # candidates ranking strictly above that prefix
                    # of those (and, after the last digit, of the ties), how many earlier parts hold
                    gt_b = fx.Int32(0)
                    eq_b = fx.Int32(0)
                    for d in range_constexpr(4):
                        sh = 24 - 8 * d
                        # the bits above this digit, folded at trace time (no 32-wide shift)
                        mk = (~((1 << (sh + 8)) - 1)) & 0xFFFFFFFF
                        hi = fx.Int32(mk - (1 << 32) if mk >= (1 << 31) else mk)
                        for z in range_constexpr(-(-(TK_BC + 4) // THREADS)):
                            zi = fx.Int32(tid) + z * THREADS
                            if zi < TK_BC + 4:
                                lds_st(hist, zi, fx.Int32(0))
                        gpu.barrier()
                        for j in range_constexpr(TK_TRIPS):
                            if trip_live(j):
                                for q in range_constexpr(TK_PER):
                                    c = tk_cbs[j] + q
                                    ok = c < n_live
                                    u = ok.select(tk_keys[j * TK_PER + q], fx.Int32(0))
                                    if ok & (((u ^ pfx) & hi) == 0):
                                        fx.atomic_add(
                                            hist + ((u >> sh) & (TK_BINS - 1)) * TK_REP + (tid & (TK_REP - 1)),
                                            fx.Int32(1),
                                            syncscope=fx.rocdl.SyncScope.Workgroup,
                                        )
                        gpu.barrier()
                        # thread t takes bin TK_BINS - 1 - t: the exclusive scan counts those above it
                        need = k_want - gt
                        cnt = fx.Int32(0)
                        bn = TK_BINS - 1 - tid
                        if tid < TK_BINS:
                            for r in range_constexpr(TK_REP):
                                cnt = cnt + lds_ld(hist, bn * TK_REP + r)
                        if const_expr(TK_PARTS > 1):
                            # the parts trade bins so all pick the same digit
                            if (tid < TK_BINS) & (n_parts > 1):
                                put(mb("tk_hist"), ((s * 4 + d) * TK_PARTS + part) * TK_BINS + bn, cnt)
                            tot = cnt
                            before = fx.Int32(0)  # this bin's count in the parts before this one
                            if (tid < TK_BINS) & (n_parts > 1):
                                vs = poll(
                                    [
                                        (mb("tk_hist"), ((s * 4 + d) * TK_PARTS + pp) * TK_BINS + bn, 1)
                                        for pp in _other_parts(part)
                                    ]
                                )
                                others = _other_parts(part)
                                for k in range_constexpr(TK_PARTS - 1):
                                    tot = tot + vs[k][0]
                                    before = before + (others[k] < part).select(vs[k][0], fx.Int32(0))
                            cnt = tot
                        above, _tot = block_excl_scan(cnt)
                        if const_expr(TK_PARTS > 1):
                            # at the chosen bin: how many candidates above it earlier parts hold
                            above_b, _tb = block_excl_scan(before)
                        if (tid < TK_BINS) & (above < need) & ((above + cnt) >= need):
                            lds_st(hist, TK_BC, bn)
                            lds_st(hist, TK_BC + 1, above)
                            if const_expr(TK_PARTS > 1):
                                lds_st(hist, TK_BC + 2, above_b)
                                lds_st(hist, TK_BC + 3, before)
                        gpu.barrier()
                        pfx = pfx | (lds_ld(hist, TK_BC) << sh)
                        gt = gt + lds_ld(hist, TK_BC + 1)
                        if const_expr(TK_PARTS > 1):
                            gt_b = gt_b + lds_ld(hist, TK_BC + 2)
                            eq_b = lds_ld(hist, TK_BC + 3)  # the last digit's is the ties'
                        gpu.barrier()
                    thr = pfx ^ MIN_I32  # back to the signed-comparable domain

                    # Compaction: picks above the threshold go to [gt_b, ..) of [0, gt), ties
                    # to [gt + eq_b, ..) below k_want. The gather polls every i_sel slot, so
                    # a radix bug that leaves one unwritten is a hang, not a wrong answer.
                    stamp("i_topk", tt, 3)
                    if tid == 0:
                        lds_st(hist, TK_BC, fx.Int32(0))
                        lds_st(hist, TK_BC + 1, fx.Int32(0))
                    gpu.barrier()
                    for j in range_constexpr(TK_TRIPS):
                        if trip_live(j):
                            for q in range_constexpr(TK_PER):
                                c = tk_cbs[j] + q
                                ok = c < n_live
                                sk = tk_keys[j * TK_PER + q] ^ MIN_I32
                                if ok & (sk > thr):
                                    w = fx.Int32(
                                        fx.atomic_add(hist + TK_BC, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Workgroup)
                                    )
                                    put(mb("i_sel"), s * N_ISEL + gt_b + w, comp_row(s, c))
                                if ok & (sk == thr):
                                    w = fx.Int32(
                                        fx.atomic_add(
                                            hist + TK_BC + 1, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Workgroup
                                        )
                                    )
                                    if (gt + eq_b + w) < k_want:
                                        put(mb("i_sel"), s * N_ISEL + gt + eq_b + w, comp_row(s, c))
                    # the next task's first digit re-zeroes these counters
                    gpu.barrier()
                    # part 0 fills the tail (every part writes only below k_want)
                    if part == 0:
                        for j in range_constexpr((N_ISEL + THREADS - 1) // THREADS):
                            o = fx.Int32(tid) + j * THREADS
                            if o < N_ISEL:
                                if o >= k_want:
                                    put(mb("i_sel"), s * N_ISEL + o, fx.Int32(-1))
                else:
                    if part == 0:
                        for j in range_constexpr((N_ISEL + THREADS - 1) // THREADS):
                            o = fx.Int32(tid) + j * THREADS
                            if o < N_ISEL:
                                row = comp_row(s, fx.max(fx.min(o, n_live - 1), fx.Int32(0)))
                                put(mb("i_sel"), s * N_ISEL + o, (o < n_live).select(row, fx.Int32(-1)))
                stamp("i_topk", tt, 4)

        # ============== 4. per-head query RMS (no weight) + RoPE -> bf16 query
        for tt in range(start("q_norm"), S * H, G):
            tt = fx.Int32(tt)
            stamp("q_norm", tt, 0)
            s = tt // H
            head = tt % H
            ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
            sp = ld_pos(s)
            c = ld_f32(rsrc(rope_cos), sp * (ROPE_DIM // 2) + ri)
            sn = ld_f32(rsrc(rope_sin), sp * (ROPE_DIM // 2) + ri)
            hint_wait(
                QB_PER_HEAD,
                lambda k: (mb("q_raw"), (s * H + head) * HEAD_DIM + k * Q_B_TILE + Q_B_TILE - 1),
                mark=("q_norm", tt),
            )
            qv = getf(mb("q_raw"), (s * H + head) * HEAD_DIM + fx.min(tid, HEAD_DIM - 1))
            stamp("q_norm", tt, 2)
            live = tid < HEAD_DIM
            ssq = block_sum(live.select(qv * qv, fx.Float32(0.0)))
            nv = qv * rsq(ssq * (1.0 / HEAD_DIM) + EPS)
            partner = xshfl(nv, 1)
            even = tid % 2 == 0
            rot = even.select(nv * c - partner * sn, partner * sn + nv * c)
            if const_expr(KV_FP8):
                # ATOM's fp8 attention takes the NoPE query as FP8 too
                nv = kv_quant(nv)[0]
            qn = (tid < NOPE_DIM).select(nv, rot)
            other = xshfl(qn, 1)
            if live & (tid % 2 == 0):
                put(mb("q"), ((s * H + head) * HEAD_DIM + tid) // 2, bf16_pair(qn, other))
            stamp("q_norm", tt, 4)

        # ============== 5. gather-sparse sliding-window split: 64 keys x H heads
        r_idx = rsrc(indices)
        KPW = SPLIT_KEYS // WAVES
        EPL = HEAD_DIM // 64  # KV elements one lane owns of a key's row
        WPL = EPL // 2

        def split_keys(t, s):
            """Wave 0 writes this split's 64 cache rows to LDS keys (-1 = unwritten); with an
            indexer the compressed half comes from the top-k's i_sel."""
            if wave == 0:
                k_pos = t * SPLIT_KEYS + lane
                if const_expr(LIVE_SPLITS):
                    # a tile group can overrun the list; those tiles hold no keys
                    k_at = s * N_KEYS + fx.min(k_pos, fx.Int32(N_KEYS - 1))
                    kk = fx.Int32(bo.buffer_load(r_idx, k_at, vec_width=1, dtype=T.i32))
                    lds_st(keys, lane, (t < N_SPLIT).select(kk, fx.Int32(-1)))
                elif const_expr(IHD):
                    if t * SPLIT_KEYS >= window:
                        lds_st(keys, lane, get(mb("i_sel"), s * N_ISEL + k_pos - window))
                    else:
                        lds_st(
                            keys,
                            lane,
                            fx.Int32(bo.buffer_load(r_idx, s * N_KEYS + k_pos, vec_width=1, dtype=T.i32)),
                        )
                else:
                    lds_st(
                        keys,
                        lane,
                        fx.Int32(bo.buffer_load(r_idx, s * N_KEYS + k_pos, vec_width=1, dtype=T.i32)),
                    )

        def gather_old_kv():
            """Each wave copies its KPW keys' KV rows (absolute plane rows) into the tile;
            unwritten slots (-1) are clamped to 0 here and masked in the softmax."""
            krows = [fx.max(lds_ld(keys, wave * KPW + jj), fx.Int32(0)) for jj in range(KPW)]
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                if const_expr(KV_FP8):
                    # NoPE lanes: 8 FP8 bytes x E8M0 (exact in bf16); the rest the RoPE plane
                    r_row = row_rsrc(kv_cache, krows[jj], KV_ROW_BYTES)
                    q8 = fx.Vector(
                        bo.buffer_load(r_row, fx.min(lane, fx.Int32(NOPE_DIM // EPL - 1)) * 2, vec_width=2, dtype=T.i32)
                    )
                    g = fx.min(lane, fx.Int32(NOPE_DIM // EPL - 1)) // (64 // EPL)  # this lane's 64-group
                    sw = fx.Int32(bo.buffer_load(r_row, NOPE_DIM // 4 + g // 2, vec_width=1, dtype=T.i32))
                    sb = (sw >> ((g % 2) * 16)) & 0xFF
                    sc = (sb << 23).bitcast(fx.Float32)
                    nope = (fp8_to_bf16x8(q8[0], q8[1]).to(fx.Float32) * sc).to(fx.BFloat16)
                    rope = fx.Vector(
                        bo.buffer_load(
                            row_rsrc(kv_rope, krows[jj], ROPE_DIM * 2),
                            fx.max(lane - NOPE_DIM // EPL, fx.Int32(0)) * WPL,
                            vec_width=WPL,
                            dtype=T.i32,
                        )
                    )
                    nw = nope.bitcast(fx.Int32)
                    is_n = lane < NOPE_DIM // EPL
                    # ATOM leaves some rows unwritten (0xFF) or NaN; as its attention, mask
                    # the key (-1) and zero it. A 0xFF scale is checked on its own (inf x FP8
                    # is not NaN); one bad lane masks the whole key.
                    words = [is_n.select(nw[m], rope[m]) for m in range(WPL)]
                    bad = sb == fx.Int32(0xFF)
                    for w_ in words:
                        bad = bad | _bf16x2_has_nan(w_)
                    kv8 = fx.Vector.from_elements([bad.select(fx.Int32(0), w_) for w_ in words], fx.Int32)
                    if _wave_any(bad) & (lane == 0):
                        lds_st(keys, j, fx.Int32(-1))
                else:
                    kv8 = fx.Vector(
                        bo.buffer_load(
                            row_rsrc(kv_cache, krows[jj], HEAD_DIM * 2), lane * WPL, vec_width=WPL, dtype=T.i32
                        )
                    )
                fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * WPL))

        def patch_new_kv(s):
            """Patch the rows this launch wrote (kvnew, and cnew on a boundary) from the
            mailboxes, for sample ``s``'s own sequence only (with TOK > 1, its run's
            tokens up to ``s``). ``kr`` is wave-uniform, so whole waves reach the polls."""
            sp = ld_pos(s)
            j_tok = s % TOK
            for d in range_constexpr(TOK):
                sd = s - fx.min(fx.Int32(d), j_tok)  # clamped onto s; masked by `inl`
                inl = fx.Int32(d) <= j_tok
                pd = sp - d
                w_row = ld_dest(0, sd)
                c_row = comp_row(sd, fx.max(pd, fx.Int32(0)) // CR) if const_expr(CR) else fx.Int32(0)
                for jj in range_constexpr(KPW):
                    j = wave * KPW + jj
                    kr = lds_ld(keys, j)
                    if inl & (kr == w_row):
                        kvp = get2_many([(mb("kvnew"), sd * HEAD_DIM + lane * EPL + m * 2) for m in range(WPL)])
                        w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                        fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * KS + lane * WPL))
                    if const_expr(CR):
                        if inl & ((pd + 1) % CR == 0) & (kr == c_row):
                            cvp = get2_many([(mb("cnew"), sd * HEAD_DIM + lane * EPL + m * 2) for m in range(WPL)])
                            # NaN when ATOM's compressor state is: masked (see gather_old_kv)
                            words = [bf16_pair(a0, a1).bitcast(fx.Int32) for a0, a1 in cvp]
                            bad = _bf16x2_has_nan(words[0])
                            for w_ in words[1:]:
                                bad = bad | _bf16x2_has_nan(w_)
                            w = [bad.select(fx.Int32(0), w_).bitcast(fx.Float32) for w_ in words]
                            fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * KS + lane * WPL))
                            if _wave_any(bad) & (lane == 0):
                                lds_st(keys, j, fx.Int32(-1))

        def live_splits(s):
            """Splits of sample ``s`` that can hold a live key: the window's plus the
            ``(pos + 1) // CR`` compressed keys so far (HCA only, else N_SPLIT)."""
            n = N_SPLIT
            if const_expr(LIVE_SPLITS):
                nc = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_KEYS - window))
                n = (window + nc + SPLIT_KEYS - 1) // SPLIT_KEYS
            return n

        # (sample, split) tasks over the batch's largest live-split count
        SPL_L = N_SPLIT
        if const_expr(LIVE_SPLITS):
            SPL_L = fx.Int32(0)
            for s_ in range_constexpr(S):
                SPL_L = fx.max(SPL_L, live_splits(s_))
        # past one round of the grid a task folds TPT tiles (flash-style) into one partial
        TPT = 1
        NG = SPL_L
        if const_expr(LIVE_SPLITS):
            TPT = fx.min((S * SPL_L + G - 1) // G, fx.Int32(N_SPLIT))
            NG = (SPL_L + TPT - 1) // TPT
        NDG = HEAD_DIM // 32 // WAVES  # 32-dim output groups a wave owns in P V
        HPW = H // WAVES  # heads a wave runs the softmax for
        FAC = H * (SPLIT_KEYS // 2)  # words of `p` past P^T: per-head rescale factors

        def load_tile(t, s):
            split_keys(t, s)
            gpu.barrier()
            gather_old_kv()

        def fold_head(carry, hg, h, m, lsum):
            """Fold one head's tile max / sum into the running pair; leave the two
            rescale factors in LDS for the P V accumulator rows of that head."""
            m_run, l_run = fx.Float32(carry[2 * hg]), fx.Float32(carry[2 * hg + 1])
            m_new = fx.max(m_run, m)
            f_old = exp(m_run - m_new)
            f_new = exp(m - m_new)
            if lane == 0:
                lds_st(pl, FAC + 2 * h, f_old)
                lds_st(pl, FAC + 2 * h + 1, f_new)
            return [m_new, l_run * f_old + lsum * f_new]

        def publish_folded(s, t, fin):
            """The folded group's partial, as tile group ``t``."""
            for hg in range_constexpr(HPW):
                if lane == 0:
                    h = wave + hg * WAVES
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h, fin[2 * hg])
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h, fin[2 * hg + 1])
            for g in range_constexpr(NDG):
                dw = (wave * NDG + g) * 16 + lane % 16
                if lane < 16 * (H // 4):  # rows (heads) 4 * (lane // 16) + e < H
                    for e in range_constexpr(4):
                        hh = (lane // 16) * 4 + e
                        o0 = fin[2 * HPW + (g * 2) * 4 + e]
                        o1 = fin[2 * HPW + (g * 2 + 1) * 4 + e]
                        put_bf(mb("sp_acc"), ((s * N_SPLIT + t) * H + hh) * HEAD_DIM + dw * 2, [o0, o1])

        def run_tiles(s, t):
            """TPT tiles from t * TPT (the first already loaded), folded."""
            init = [fx.Float32(NEG), fx.Float32(0.0)] * HPW + [fx.Float32(0.0)] * (8 * NDG)
            for i_t, carry in range(0, TPT, fx.Int32(1), init=init):
                i_t = fx.Int32(i_t)
                if i_t > 0:
                    gpu.barrier()  # the previous tile's P V is done with the tile
                    load_tile(t * TPT + i_t, s)
                nxt = compute_tile(s, t, carry, True)
                res = yield nxt
            return [fx.Float32(v) for v in res]

        def compute_tile(s, t, carry, multi):
            """Scores, softmax and P V of the tile in LDS: folded into ``carry`` (running
            max / sum per head + P V accumulator) if ``multi``, else published as tile ``t``."""
            patch_new_kv(s)
            gpu.barrier()
            # scores = K Q^T on MFMA: keys M (4 row groups), HEAD_DIM K (two wave halves), heads N
            hn = fx.min(lane % 16, H - 1)
            rgk = wave % 4
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(QK_DIM // 32 // 2):
                kst = (wave // 4) * (QK_DIM // 32 // 2) + st
                key = rgk * 16 + lane % 16
                kw = KT_OFF + key * KS + kst * 16
                a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                b = fx.ptr_load(xs + (hn * QS + kst * 16 + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            fx.ptr_store(c, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            # split-local softmax: wave h, lane = key j (score = sum of the two K halves).
            ml = []
            for hg in range_constexpr(HPW):
                h = wave + hg * WAVES
                valid = lds_ld(keys, lane) >= 0
                r16 = lane % 16
                cl = h + 16 * (r16 // 4)
                raw = lds_ld(red, ((lane // 16) * 64 + cl) * 4 + r16 % 4) + lds_ld(
                    red, ((lane // 16 + 4) * 64 + cl) * 4 + r16 % 4
                )
                sc_v = valid.select(raw * scale, fx.Float32(NEG))
                m = wave_max(sc_v)
                p = valid.select(exp(sc_v - m), fx.Float32(0.0))
                lsum = wave_sum(p)
                p_n = xshfl(p, 1)
                if lane % 2 == 0:  # P^T bf16 [h][64 keys] (words h * 32 + j / 2)
                    lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
                # (trace-time: a conditional expression, since the tracer turns `if` into a branch)
                ml += fold_head(carry, hg, h, m, lsum) if multi else [m, lsum]
                if not multi:  # trace-time; nothing assigned here is used after it
                    if lane == 0:  # written last: the merge's readiness hint
                        put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                        put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
            gpu.barrier()
            # O = P V on MFMA: heads M, keys K (2 steps), dims N; V is the K tile
            ov = []
            f_o = [lds_ld(pl, FAC + 2 * ((lane // 16) * 4 + e)) for e in range(4)] if multi else None
            f_n = [lds_ld(pl, FAC + 2 * ((lane // 16) * 4 + e) + 1) for e in range(4)] if multi else None
            for g in range_constexpr(NDG):
                dw = (wave * NDG + g) * 16 + lane % 16
                c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                for js in range_constexpr(SPLIT_KEYS // 32):
                    a = fx.ptr_load(
                        pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4), result_type=v4f
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
                if not multi:
                    if lane < 16 * (H // 4):
                        for e in range_constexpr(4):
                            hh = (lane // 16) * 4 + e
                            put_bf(mb("sp_acc"), ((s * N_SPLIT + t) * H + hh) * HEAD_DIM + dw * 2, [c0[e], c1[e]])
                for e in range_constexpr(4):
                    k0 = 2 * HPW + (g * 2) * 4 + e
                    k1 = 2 * HPW + (g * 2 + 1) * 4 + e
                    ov += (
                        [
                            fx.Float32(carry[k0]) * f_o[e] + c0[e] * f_n[e],
                            fx.Float32(carry[k1]) * f_o[e] + c1[e] * f_n[e],
                        ]
                        if multi
                        else [c0[e], c1[e]]
                    )
            out = list(ml)
            for g in range_constexpr(NDG):
                out += [ov[(g * 4 + e) * 2] for e in range(4)] + [ov[(g * 4 + e) * 2 + 1] for e in range(4)]
            return out

        for tt in range(start("split"), S * NG, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s = tt // NG
            t = tt % NG
            load_tile(t * TPT, s)  # before waiting for q: these rows are from earlier launches
            hint_wait(
                H,
                lambda k: (mb("q"), ((s * H + k) * HEAD_DIM + HEAD_DIM - 2) // 2),
                mark=("split", tt),
            )
            # q of all heads -> bf16 Q[h][HEAD_DIM] at words h * QS + d / 2 (padded stride)
            NQ_TOT = H * HEAD_DIM // 4
            NQ = NQ_TOT // THREADS

            def stage_q(w4, v):
                qw = (w4 // (HEAD_DIM // 4)) * QS + (w4 % (HEAD_DIM // 4)) * 2
                lds_st(xs, qw, v[0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, v[1].bitcast(fx.Float32))

            qv = poll([(mb("q"), (s * H * HEAD_DIM + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ)])
            for i in range_constexpr(NQ):
                stage_q(tid + i * THREADS, qv[i])
            if const_expr(NQ_TOT % THREADS):
                w4 = tid + NQ * THREADS
                if w4 < NQ_TOT:
                    stage_q(w4, poll([(mb("q"), (s * H * HEAD_DIM + w4 * 4) // 2, 2)])[0])
            stamp("split", tt, 2)
            if const_expr(LIVE_SPLITS):
                if TPT == 1:
                    compute_tile(s, t, None, False)
                else:
                    publish_folded(s, t, run_tiles(s, t))
            else:
                compute_tile(s, t, None, False)
            stamp("split", tt, 4)

        # ============ 6. split merge (+ attention sink) + inverse RoPE -> o
        # the sink enters the denominator only; RoPE lanes are de-rotated (V is the RoPE'd K)
        UV_PAIRS = UV_TILE // 2
        for tt in range(start("uv"), S * N_UV, G):
            tt = fx.Int32(tt)
            stamp("uv", tt, 0)
            s = tt // N_UV
            t = tt % N_UV  # UV_TILE-dim tile
            head = t // UV_PER_HEAD
            doff = (t % UV_PER_HEAD) * UV_TILE
            sink = ld_f32(rsrc(attn_sink), head)
            n_sp = (live_splits(s) + TPT - 1) // TPT  # the tile groups written: see the split stage
            hint_wait(n_sp, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head), mark=("uv", tt))
            pre_poll(n_sp, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            dp = fx.min(tid, UV_PAIRS - 1)
            # one thread per split: a lane of wave 0, or of the block when UV_WIDE
            sp_id = tid if UV_WIDE else lane
            spi = fx.min(sp_id, n_sp - 1)  # clamped so the spare threads read a live slot
            ml = (s * N_SPLIT + spi) * H + head
            got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)], batch=2)
            # per-split weights exp(m - M) / L for this head -> misc[split]
            ok_sp = sp_id < n_sp
            m_sp = ok_sp.select(got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
            l_sp = ok_sp.select(got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
            if UV_WIDE:
                mx = block_max(m_sp)
                w_sp = exp(m_sp - mx)
                den = block_sum(l_sp * w_sp) + exp(sink - mx)
                if ok_sp:
                    lds_st(misc, sp_id, w_sp * rcp(den))
            else:
                if wave == 0:
                    mx = wave_max(m_sp)
                    w_sp = exp(m_sp - mx)
                    den = wave_sum(l_sp * w_sp) + exp(sink - mx)
                    if ok_sp:
                        lds_st(misc, sp_id, w_sp * rcp(den))
            stamp("uv", tt, 2)
            gpu.barrier()
            # runtime loop over UV_CHUNK splits; past the live ones re-read the last at weight 0
            for _c, acc in range(
                0, (n_sp + UV_CHUNK - 1) // UV_CHUNK, fx.Int32(1), init=[fx.Float32(0.0), fx.Float32(0.0)]
            ):
                cb = fx.Int32(_c) * UV_CHUNK
                gc = poll(
                    [
                        (
                            mb("sp_acc"),
                            (
                                (s * N_SPLIT + (fx.min(cb + e, n_sp - 1) if const_expr(LIVE_SPLITS) else cb + e)) * H
                                + head
                            )
                            * (HEAD_DIM // 2)
                            + doff // 2
                            + dp,
                            1,
                        )
                        for e in range(UV_CHUNK)
                    ],
                    batch=UV_CHUNK,
                )
                o0 = fx.Float32(acc[0])
                o1 = fx.Float32(acc[1])
                for e in range_constexpr(UV_CHUNK):
                    wj = lds_ld(misc, cb + e)
                    if const_expr(LIVE_SPLITS):
                        wj = (cb + e < n_sp).select(wj, fx.Float32(0.0))
                    a0, a1 = bf2_f32(gc[e][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                res = yield [o0, o1]
            o0 = fx.Float32(res[0])
            o1 = fx.Float32(res[1])
            # de-rotate the RoPE lanes (inverse rotation: sin negated)
            d0 = doff + dp * 2
            ri = fx.max(d0 - NOPE_DIM, fx.Int32(0)) // 2
            sp = ld_pos(s)
            c = ld_f32(rsrc(rope_cos), sp * (ROPE_DIM // 2) + ri)
            sn = ld_f32(rsrc(rope_sin), sp * (ROPE_DIM // 2) + ri)
            rot = d0 >= NOPE_DIM
            v0 = rot.select(o0 * c + o1 * sn, o0)
            v1 = rot.select(o1 * c - o0 * sn, o1)
            if tid < UV_PAIRS:
                put(mb("o"), ((s * H + head) * HEAD_DIM + doff) // 2 + dp, bf16_pair(v0, v1))
            stamp("uv", tt, 4)

        # ================= 7a. o_a: grouped low-rank output projection (group = OA_K slice)
        r_woa, r_soa = rsrc(w_o_a), rsrc(s_o_a)
        OA_NKC = OA_K // 64
        OA_R = ROW_TILE // 16
        OA_WPR = WAVES // OA_R
        OA_SPT = o_a_spt(S, O_GROUPS, O_LORA)
        for tt in range(start("o_a"), N_OA // OA_SPT, G):
            tt = fx.Int32(tt)
            stamp("o_a", tt, 0)
            s = (tt // (O_GROUPS * OA_PER_GROUP)) * OA_SPT
            t = tt % (O_GROUPS * OA_PER_GROUP)
            grp = t // OA_PER_GROUP
            # B column n holds sample s + n (columns past OA_SPT repeat the last)
            oa_col = fx.min(lane % 16, OA_SPT - 1)

            def u_oa(c):
                kc = (wave % OA_WPR) * (OA_NKC // OA_WPR) + c
                return unit_fp8(
                    r_woa, r_soa, t * OA_R + wave // OA_WPR, kc, OA_NKC, OA_K, 128, (oa_col * OA_K + kc * 64) // 2
                )

            pre = [u_oa(c) for c in range(OA_NKC // OA_WPR)]
            hint_wait(
                H // O_GROUPS,
                lambda k: (mb("o"), ((s * H + grp * (H // O_GROUPS) + k) * HEAD_DIM + HEAD_DIM - 2) // 2),
                mark=("o_a", tt),
            )
            stage_x_pairs("o", OA_SPT * OA_K, lambda k: ((s + k // OA_K) * H) * HEAD_DIM + grp * OA_K + k % OA_K)
            stamp("o_a", tt, 2)
            gpu.barrier()
            acc = run_units(u_oa, OA_NKC // OA_WPR, OA_NKC // OA_WPR, pre)
            reduce_rows(OA_R, acc, emit_out(ROW_TILE))
            stamp("o_a", tt, 3)
            gpu.barrier()
            if tid < OA_SPT * ROW_TILE // 4:
                n = tid // (ROW_TILE // 4)
                r = tid % (ROW_TILE // 4) * 4
                put_bf(
                    mb("o_lora"),
                    (s + n) * OB_K + t * ROW_TILE + r,
                    [lds_ld(outs, n * ROW_TILE + r + j) for j in range(4)],
                )
            stamp("o_a", tt, 4)

        # ============= 7b. o_b + attention TP peer reduce + residual -> a
        r_wob, r_sob = rsrc(w_o_b), rsrc(s_o_b)
        OB_NKC = OB_K // 64
        OB_R = ROW_TILE // 16
        OB_WPR = WAVES // OB_R
        for t in range(start("o_b"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("o_b", t, 0)

            def u_ob(c):
                kc = (wave % OB_WPR) * (OB_NKC // OB_WPR) + c
                return unit_fp8(
                    r_wob, r_sob, t * OB_R + wave // OB_WPR, kc, OB_NKC, OB_K, 128, (n_sel() * OB_K + kc * 64) // 2
                )

            pre = [u_ob(c) for c in range(OB_NKC // OB_WPR)]
            hint_wait(
                S * O_GROUPS * OA_PER_GROUP,
                lambda k: (
                    mb("o_lora"),
                    (k // (O_GROUPS * OA_PER_GROUP)) * OB_K + (k % (O_GROUPS * OA_PER_GROUP)) * ROW_TILE + ROW_TILE - 1,
                ),
                mark=("o_b", t),
            )
            stage_x_pairs("o_lora", S * OB_K, lambda k: k)
            stamp("o_b", t, 2)
            gpu.barrier()
            acc = run_units(u_ob, OB_NKC // OB_WPR, OB_NKC // OB_WPR, pre)
            reduce_rows(OB_R, acc, emit_out(ROW_TILE))
            stamp("o_b", t, 3)
            gpu.barrier()

            def resid_h(s, row):
                w = fx.Vector.from_elements(
                    [fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32))], fx.Int32
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            if const_expr(HC > 1):
                hc_stage_coef(0)
                peer_reduce(
                    "attn",
                    t,
                    None,  # hc_post owns the combination
                    lambda s, row, v0, v1: hc_post(
                        s,
                        row,
                        v0,
                        v1,
                        lambda s_, j, r_: fx.Int32(
                            bo.buffer_load(r_h, ((s_ * HC + j) * HIDDEN + r_) // 2, vec_width=1, dtype=T.i32)
                        ),
                        lambda s_, k, r_, o0, o1: put_bf(mb("a"), (s_ * HC + k) * HIDDEN + r_, [o0, o1]),
                    ),
                )
            else:
                peer_reduce(
                    "attn",
                    t,
                    resid_h,
                    lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
                )
            stamp("o_b", t, 4)
        # FFN side: the router contracts the streams itself unless ffn_hcc
        hc_coef_f = None
        FFN_HCC = ffn_hcc(S, HC, N_EXPERTS)
        if const_expr(FFN_HCC):
            hc_pre_stages(
                "f",
                1,
                hc_ffn_fn,
                hc_ffn_sb,
                lambda s, k: get(mb("a"), (s * HC * HIDDEN + k) // 2),
                "ain",
                src_pair=lambda s, k: (mb("a"), (s * HC * HIDDEN + k) // 2),
            )
        elif const_expr(HC > 1):
            hc_coef_f = hc_pre_stages(
                "f",
                1,
                hc_ffn_fn,
                hc_ffn_sb,
                lambda s, k: get(mb("a"), (s * HC * HIDDEN + k) // 2),
                None,
                contract=False,
            )

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        r_wr = rsrc(w_r)
        R_NKC = HIDDEN // 64
        SPT = router_spt(S, N_EXPERTS)
        assert SPT <= ROUTER_TILE
        for tt in range(start("router"), S * N_ROUTER // SPT, G):
            tt = fx.Int32(tt)
            t = tt % N_ROUTER
            rs0 = (tt // N_ROUTER) * SPT
            stamp("router", tt, 0)

            # K-fold: rows / columns 0..7 take the first K half, 8..15 the second
            r_sub = t * ROUTER_TILE % 16
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.min(lane % 8, fx.Int32(SPT - 1))

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, (r_ns * HIDDEN + kc * 64) // 2, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(0, None, mark=("router", tt))
            # this task's expert-activation blocks ride along (MXFP8: 4 16-lane groups a wave)
            r_gp = rsrc(g_post)
            if const_expr(use_mxfp8_block32):
                x_blk = (wave * N_ROUTER + t) * 4 + lane // 16
                xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 32 + lane % 16 * 2
            else:
                x_blk = wave * N_ROUTER + t
                xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 128 + lane * 2
            x_ok = (wave < XQ_WAVES) & (x_blk < PUBLISH_BLOCKS)
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                # sks = (local sample s, k) pairs; local sample s is sample rs0 + s
                n4 = len(sks)
                if const_expr(HC == 1 or FFN_HCC):  # one stream: `a` itself, or hcc_f's `ain`
                    x_src = "ain" if FFN_HCC else "a"
                    specs = [(mb(x_src), ((rs0 + s) * HIDDEN + k) // 2, 2) for s, k in sks]
                    specs += [(mb(x_src), ((rs0 + s) * HIDDEN + xk) // 2, 1) for s in range(SPT)]
                    v = poll(specs, batch=len(specs))
                    for s in range_constexpr(SPT):
                        xa.append(bf2_f32(v[n4 + s][0]))
                    return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:n4]]
                # x = sum_j pre[j] * stream j, contracted here in hcc's order and rounding
                specs = [
                    (mb("a"), ((rs0 + s) * HC * HIDDEN + j * HIDDEN + k) // 2, 2) for s, k in sks for j in range(HC)
                ]
                specs += [
                    (mb("a"), ((rs0 + s) * HC * HIDDEN + j * HIDDEN + xk) // 2, 1)
                    for s in range(SPT)
                    for j in range(HC)
                ]
                v = poll(specs)
                hc_coef_f(1, tt == 0)  # task 0 also publishes post / comb for down
                pjs = [[lds_ld(misc, HC_MISC + (rs0 + s) * HC_COEF + j) for j in range(HC)] for s in range(SPT)]

                def mix(words, pj):
                    """bf16(sum_j pre[j] * x_j) for each element of the words' streams."""
                    xs_ = [list(bf2_f32(w[0])) + (list(bf2_f32(w[1])) if len(w) > 1 else []) for w in words]
                    out = []
                    for e in range_constexpr(len(xs_[0])):
                        acc = fx.Float32(0.0)
                        for j in range_constexpr(HC):
                            acc = acc + pj[j] * xs_[j][e]
                        out.append(bf16_round(acc))
                    return out

                for s in range_constexpr(SPT):
                    xa.append(tuple(mix(v[(n4 + s) * HC : (n4 + s + 1) * HC], pjs[s])))
                return [mix(v[i * HC : (i + 1) * HC], pjs[sks[i][0]]) for i in range(n4)]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, count=SPT)
            stamp("router", tt, 2)
            for s_l in range_constexpr(SPT):
                if x_ok:
                    x_s = rs0 + s_l
                    x_rstd = rstds[s_l]
                    a0, a1 = xa[s_l]
                    v0, v1 = a0 * x_rstd * xg[0], a1 * x_rstd * xg[1]
                    if const_expr(use_fp8_block128):
                        q0, q1, qs = quant_scaled(v0, v1)
                        w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                        w8n = xshfl(w8, 1)
                        if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                            put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                        d0, d1 = fp8_roundtrip(q0, q1)
                        d0, d1 = d0 * qs, d1 * qs
                        if lane == 0:
                            put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                    elif const_expr(use_mxfp8_block32):
                        d0, d1, qs = quant_mxfp8(v0, v1)
                        w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, d0, d1, fx.Int32(0), False)) & 0xFFFF
                        w8n = xshfl(w8, 1)
                        if lane % 2 == 0:
                            put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                        if lane % 16 == 0:
                            put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                        d0, d1 = d0 * qs, d1 * qs
                    else:
                        d0, d1 = bf16_round(v0), bf16_round(v1)
                        put(mb("xq"), (x_s * HIDDEN + xk) // 2, bf16_pair(d0, d1))
                    bo.buffer_store(fx.Vector.from_elements([d0, d1], fx.Float32), rsrc(mb("xqd")), x_s * HIDDEN + xk)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
            gpu.barrier()
            stamp("router", tt, 3)
            if tid < ROUTER_TILE * SPT:
                r = tid % ROUTER_TILE
                n = tid // ROUTER_TILE
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(red, (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4)
                put(
                    mb("scores"), (rs0 + n) * N_EXPERTS + t * ROUTER_TILE + r, _sqrt_softplus(bf16_round(logit))
                )  # bf16 gate logits, as ATOM's
            stamp("router", tt, 4)

        def dn_route(bs):
            """Expert-down routing (wave s -> sample s): expert ids -> keys[s * 9 + slot],
            route weights -> dnw[]; the scores must have landed."""
            if wave < S:
                e, w = route_topk(wave, bs=bs)
                if lane < MOE_SLOTS:  # slot 0: the shared expert, then pick lane (slot lane + 1)
                    q = wave * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                    lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e))
                    lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ================================ 9. expert up/gate + SiLU
        # one 16-row group (8 gate + 8 up rows) per tile, all eight waves splitting K
        UG_NKC = HIDDEN // 64
        UG_UNIT_K = 128 if (use_fp8_block128 or use_mxfp4_weight) else 64
        UG_W_BYTES = 2 * INTER * HIDDEN // (2 if use_mxfp4_weight else 1)
        UG_S_BYTES = (  # an MXFP4 bank's scales pad K / 32 to a multiple of 8 (pack_mxfp4_scales)
            2 * INTER * (-(-(HIDDEN // 32) // 8) * 8)
            if use_mxfp4_weight
            else 2 * INTER // SCALE_BM * (HIDDEN // 128) * 4
        )
        SUG_S_BYTES = 2 * INTER // SCALE_BM * (HIDDEN // 128) * 4  # the FP8 shared expert's scales

        if const_expr(S == 1):
            # task u: 8 intermediates of routed slot u // UG_PER_SLOT (u < UG_PER_SLOT: also shared)
            UG8_UNITS = (HIDDEN // UG_UNIT_K) // WAVES
            for u in range(start("ug"), N_UG_TASKS, G):
                u = fx.Int32(u)
                stamp("ug", u, 0)
                s_u, c = fx.Int32(0), u % UG_PER_SLOT
                has_sh = u < UG_PER_SLOT
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // UG_PER_SLOT)
                bs = load_bias()
                w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2  # MFMA rows 0-7 gate, 8-15 up
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                s_rg = (lane // 32) * (INTER // 16) + c // 2

                def u_ug8(cc, e, live=None):  # expert e's weights (loads return 0 unless live)
                    nw = None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                    ns = None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                    r_wug = bo.create_buffer_resource_from_addr(
                        w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw
                    )
                    r_sug = bo.create_buffer_resource_from_addr(
                        s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns
                    )
                    unit = wave * UG8_UNITS + cc
                    if const_expr(use_mxfp4_weight):
                        coefficients = None  # MXFP8 scales are folded into the staged activations
                        return unit_mxfp4(
                            r_wug,
                            r_sug,
                            mx_rg(c),
                            unit,
                            HIDDEN,
                            unit * 64,
                            coefficients,
                            w_ln,
                        )
                    kc = unit * (2 if use_fp8_block128 else 1)
                    nwc = 2 if use_fp8_block128 else 1
                    wv = [
                        fx.Vector(
                            bo.buffer_load(r_wug, ((w_rg * UG_NKC + kc + h) * 64 + w_ln) * 4, vec_width=4, dtype=T.i32)
                        )
                        for h in range(nwc)
                    ]
                    sc = ld_f32(r_sug, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)
                    if const_expr(use_fp8_block128):
                        return (
                            "f8f8",
                            wv,
                            lambda: sc * uniform_f32(lds_ld(misc, 8 + kc // 2)),
                            kc * 16 + (lane // 16) * 4,
                        )
                    return ("fp8", wv, sc, kc * 32 + (lane // 16) * 4)

                # the shared expert's weights do not depend on routing: prefetch them
                if const_expr(SHARED_FP8):

                    def u_ug8_sh(cc, live):
                        unit = wave * UG8_UNITS + cc
                        coefficients = None
                        return unit_fp8mx(
                            bo.create_buffer_resource_from_addr(
                                w_sug, num_records_bytes=live.select(fx.Int32(2 * INTER * HIDDEN), fx.Int32(0))
                            ),
                            bo.create_buffer_resource_from_addr(
                                s_sug, num_records_bytes=live.select(fx.Int32(SUG_S_BYTES), fx.Int32(0))
                            ),
                            w_rg,
                            s_rg,
                            unit,
                            HIDDEN,
                            unit * 64,
                            coefficients,
                            w_ln,
                        )

                    pre = [u_ug8_sh(cc, has_sh) for cc in range(UG8_UNITS)]
                else:
                    pre = [u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh) for cc in range(UG8_UNITS)]
                # at S == 1 only a CTA's first task stages input and routing; later ones reuse LDS
                if u == fx.Int32(start("ug")):
                    hint_wait(0, None, mark=("ug", u))
                    stage_moe_input([0])
                    if wave == 0:
                        e, w = route_topk(s_u, bs=bs)
                        # every pick: a later task on this CTA reads its own from here
                        if lane < TOP_K:
                            lds_st(keys, lane, e)
                            lds_st(misc, lane, w)
                stamp("ug", u, 2)
                gpu.barrier()
                e_sel = uniform(lds_ld(keys, slot - 1))
                post = [u_ug8(cc, e_sel) for cc in range(UG8_UNITS)]
                reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16))
                gpu.barrier()
                reduce_rows(
                    1, mma_units([fx.Float32(0.0) for _ in range(4)], post), lambda rl, n, v: lds_st(outs, 16 + rl, v)
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
                            (tid < UG8 // 2).select(fx.Int32(0), slot) * INTER + c * UG8 + r,
                            _swiglu(g0, u0, swiglu_limit),
                            _swiglu(g1, u1, swiglu_limit),
                        )
                if (c == 0) & (tid == 0):  # routing record (debug / tests)
                    put(mb("sel"), slot, e_sel)
                    put(mb("prob"), slot, lds_ld(misc, slot - 1))
                    if has_sh:
                        put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                        put(mb("prob"), 0, fx.Float32(1.0))
                stamp("ug", u, 4)
        elif const_expr(S > 1):
            # (tile, sample) items, unrolled at build time (the unit lists cannot be
            # scf.for-carried). An item past the end is masked by `live` (zero
            # num_records, no publishes), not skipped, so every barrier stays uniform;
            # every item must be covered or `down` polls its `mid` slots forever.
            UG8_UNITS = (HIDDEN // UG_UNIT_K) // WAVES
            XW = HIDDEN // (4 if use_fp8_block128 else 2)
            u0 = fx.Int32(start("ug"))

            def ug8_units(c, w_rg, w_ln, s_rg, e, sample, live=None):
                nw = None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                ns = None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                rw = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw
                )
                rs = bo.create_buffer_resource_from_addr(
                    s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns
                )
                sn = n_sel() if sample is None else fx.Int32(sample)
                units = []
                for cc in range_constexpr(UG8_UNITS):
                    unit = wave * UG8_UNITS + cc
                    if const_expr(use_mxfp4_weight):
                        coefficients = None
                        units.append(
                            unit_mxfp4(
                                rw,
                                rs,
                                mx_rg(c),
                                unit,
                                HIDDEN,
                                sn * XW + unit * 64,
                                coefficients,
                                w_ln,
                            )
                        )
                        continue
                    kc = unit * (2 if use_fp8_block128 else 1)
                    nwc = 2 if use_fp8_block128 else 1
                    wv = [
                        fx.Vector(
                            bo.buffer_load(rw, ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4, vec_width=4, dtype=T.i32)
                        )
                        for j in range(nwc)
                    ]
                    sc = ld_f32(rs, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)

                    if const_expr(use_fp8_block128):

                        def coefficient(sc=sc, kb=kc // 2, sn=sn):
                            return sc * lds_ld(misc, 8 + sn * XQ_BLOCKS + kb)

                        units.append(("f8f8", wv, coefficient, sn * XW + kc * 16 + (lane // 16) * 4))
                    else:
                        units.append(("fp8", wv, sc, sn * XW + kc * 32 + (lane // 16) * 4))
                return units

            def ug8_units_sh(c, w_rg, w_ln, s_rg, live):
                """The FP8 shared expert's units: every sample at once (lane column = sample)."""
                rw = bo.create_buffer_resource_from_addr(
                    w_sug, num_records_bytes=live.select(fx.Int32(2 * INTER * HIDDEN), fx.Int32(0))
                )
                rs = bo.create_buffer_resource_from_addr(
                    s_sug, num_records_bytes=live.select(fx.Int32(SUG_S_BYTES), fx.Int32(0))
                )
                sn = n_sel()
                units = []
                for cc in range_constexpr(UG8_UNITS):
                    unit = wave * UG8_UNITS + cc
                    coefficients = None
                    units.append(unit_fp8mx(rw, rs, w_rg, s_rg, unit, HIDDEN, sn * XW + unit * 64, coefficients, w_ln))
                return units

            def shared_units(c, w_rg, w_ln, s_rg, live):
                if const_expr(SHARED_FP8):
                    return ug8_units_sh(c, w_rg, w_ln, s_rg, live)
                return ug8_units(c, w_rg, w_ln, s_rg, fx.Int32(SHARED_EXPERT), None, live)

            def ug8_emit(c, slot, sample, shared, live):
                if tid < (S if shared else 1) * UG8 // 2:
                    n = tid // (UG8 // 2)
                    r = (tid % (UG8 // 2)) * 2
                    g0, g1 = lds_ld(outs, n * 16 + r), lds_ld(outs, n * 16 + r + 1)
                    v0, v1 = lds_ld(outs, n * 16 + UG8 + r), lds_ld(outs, n * 16 + UG8 + r + 1)
                    sn = n if shared else fx.Int32(sample)
                    sl = fx.Int32(0) if shared else slot
                    if live:
                        put2(
                            mb("mid"),
                            (sn * MOE_SLOTS + sl) * INTER + c * UG8 + r,
                            _swiglu(g0, v0, swiglu_limit),
                            _swiglu(g1, v1, swiglu_limit),
                        )
                if (c == 0) & (tid < S if shared else tid == 0):
                    sn = tid if shared else fx.Int32(sample)
                    sl = fx.Int32(0) if shared else slot
                    if live:
                        put(mb("sel"), sn * MOE_SLOTS + sl, lds_ld(keys, sn * MOE_SLOTS + sl))
                        put(mb("prob"), sn * MOE_SLOTS + sl, lds_ld(dnw, sn * MOE_SLOTS + sl))

            def ug8_tile(u):
                """Tile ``u`` (clamped; ``live`` masks a dead one): (live, uu, c, slot, has_sh, w_rg, w_ln, s_rg)."""
                live = u < N_UG_TASKS
                uu = fx.min(u, fx.Int32(N_UG_TASKS - 1))
                c = uu % UG_PER_SLOT
                has_sh = uu < UG_PER_SLOT
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), uu // UG_PER_SLOT)
                w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                s_rg = (lane // 32) * (INTER // 16) + c // 2
                return live, uu, c, slot, has_sh, w_rg, w_ln, s_rg

            # a shared-expert tile serves every sample at once (B columns); routed work is items
            N_UG_ITEMS = S * N_UG_TASKS
            UG_ITEMS = (N_UG_ITEMS + G - 1) // G

            def ug8_item(k):
                """This CTA's k-th routed item, clamped as ug8_tile: (live, uu, sample, c, ...)."""
                w = u0 + k * G
                live = w < N_UG_ITEMS
                ww = fx.min(w, fx.Int32(N_UG_ITEMS - 1))
                uu = ww % N_UG_TASKS
                sample = ww // N_UG_TASKS
                _l, _u, c, slot, _h, w_rg, w_ln, s_rg = ug8_tile(uu)
                return live, uu, sample, c, slot, w_rg, w_ln, s_rg

            live0, uu0, c0, slot0, has_sh0, wr0, wl0, sr0 = ug8_tile(u0)
            shared_pre = shared_units(c0, wr0, wl0, sr0, has_sh0 & live0)
            dn_route(load_bias())
            gpu.barrier()
            items = [ug8_item(0)]
            lv, uu_, sm, c_, sl, wr, wl, sr = items[0]
            cur = ug8_units(c_, wr, wl, sr, uniform(lds_ld(keys, sm * MOE_SLOTS + sl)), sm, lv)
            stage_moe_input(list(range(S)))
            gpu.barrier()
            if has_sh0 & live0:
                reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre), emit_out(16))
                gpu.barrier()
                ug8_emit(c0, slot0, 0, True, live0)
            for k in range_constexpr(UG_ITEMS):
                live, uu, sample, c, slot, w_rg, w_ln, s_rg = items[k]
                stamp("ug", sample * N_UG_TASKS + uu, 0, pred=live)
                pre = cur
                if const_expr(k + 1 < UG_ITEMS):
                    items.append(ug8_item(k + 1))
                    lv, uu_, sm, c_, sl, wr, wl, sr = items[k + 1]
                    cur = ug8_units(c_, wr, wl, sr, uniform(lds_ld(keys, sm * MOE_SLOTS + sl)), sm, lv)
                reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16))
                gpu.barrier()
                ug8_emit(c, slot, sample, False, live)
                stamp("ug", sample * N_UG_TASKS + uu, 4, pred=live)

        # =============== 10. expert down + route weighting + MoE TP reduce
        DN_NKC = INTER // 64
        # see dn_tile: every tile starts at row 0 of a group
        assert DN_TILE % 16 == 0 and HIDDEN % DN_TILE == 0, f"down tile {DN_TILE} must divide {HIDDEN} by 16s"
        DN_R = DN_TILE // 16
        DN_WPR = WAVES // DN_R
        DN_UNIT_K = 128 if (use_fp8_block128 or use_mxfp4_weight) else 64
        DN_UNITS_PER_SLOT = INTER // DN_UNIT_K
        # routed units over (sample, slot, K chunk), then any FP8 shared-expert units
        DN_SLOTS = TOP_K if SHARED_FP8 else MOE_SLOTS
        DN_NU = S * DN_SLOTS * DN_UNITS_PER_SLOT
        DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
        # the shared expert's units: one per K chunk, every sample in its own B column
        DN_SH_NU = DN_UNITS_PER_SLOT if SHARED_FP8 else 0
        DN_SH_UPW = (DN_SH_NU + DN_WPR - 1) // DN_WPR
        DN_CPW = DN_UPW + DN_SH_UPW
        DN_BLK = S * MOE_SLOTS * INTER // 128
        DN_W_BYTES = HIDDEN * INTER // (2 if use_mxfp4_weight else 1)
        DN_S_BYTES = (
            HIDDEN * (-(-(INTER // 32) // 8) * 8) if use_mxfp4_weight else HIDDEN // SCALE_BM * (INTER // 128) * 4
        )
        SDN_S_BYTES = HIDDEN // SCALE_BM * (INTER // 128) * 4  # the FP8 shared expert's scales
        DN_BATCH = 9  # 128-k chunks per wave in flight / prefetched before the mid wait
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stamp("down", t, 0)
            if const_expr(S == 1):  # multi-sample routing was staged before up/gate
                dn_route(load_bias())
            gpu.barrier()
            gu = wave // DN_WPR
            dn_rg = t * DN_TILE // 16
            dn_off = t * DN_TILE % 16
            # rows outside the tile load their lane ^ 8 twin (same lines) and are dropped
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            def dn_coefficients(q, s_q, slot_q):
                """A unit's factor: its route weight, in its sample's column only. A VGPR LDS
                broadcast, not a readfirstlane (SGPR copies per unit cause spills)."""

                def coefficient():
                    return (lane % 16 == s_q).select(lds_ld(dnw, s_q * MOE_SLOTS + slot_q), fx.Float32(0.0))

                return coefficient

            def u_dn_sh(cc):
                qs = (wave % DN_WPR) * DN_SH_UPW + cc
                live = qs < DN_SH_NU
                kc = fx.min(qs, DN_SH_NU - 1)
                # B column n_sel() reads its sample's slot-0 mid; the route weight is 1
                q = n_sel() * MOE_SLOTS * DN_UNITS_PER_SLOT + kc
                masked = DN_SH_NU % DN_WPR != 0
                wb = bo.create_buffer_resource_from_addr(
                    w_sdn, num_records_bytes=live.select(fx.Int32(HIDDEN * INTER), fx.Int32(0)) if masked else None
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_sdn, num_records_bytes=live.select(fx.Int32(SDN_S_BYTES), fx.Int32(0)) if masked else None
                )
                rg = dn_rg + gu
                return unit_fp8mx(wb, sb, rg, rg, kc, INTER, q * 64, None, dn_ln)

            def u_dn(cc):
                if const_expr(cc >= DN_UPW):
                    return u_dn_sh(cc - DN_UPW)
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                r = fx.min(qu, DN_NU - 1)
                s_q = r // (DN_SLOTS * DN_UNITS_PER_SLOT)
                slot_q = (r // DN_UNITS_PER_SLOT) % DN_SLOTS + (MOE_SLOTS - DN_SLOTS)
                kc = r % DN_UNITS_PER_SLOT
                q = (s_q * MOE_SLOTS + slot_q) * DN_UNITS_PER_SLOT + kc
                e = uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0)),
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                    num_records_bytes=None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0)),
                )

                if const_expr(use_mxfp4_weight):
                    coefficients = dn_coefficients(q, s_q, slot_q)
                    return unit_mxfp4(
                        wb,
                        sb,
                        dn_rg + gu,
                        kc,
                        INTER,
                        q * 64,
                        coefficients,
                        dn_ln,
                    )

                kc64 = kc * (2 if use_fp8_block128 else 1)
                if const_expr(use_fp8_block128):

                    def coef():  # mid block scale * route weight, only in this sample's column
                        return (lane % 16 == s_q).select(uniform_f32(lds_ld(misc, q)), fx.Float32(0.0))

                    return unit_f8f8(wb, sb, dn_rg + gu, kc64, DN_NKC, INTER, q * 32, coef, dn_ln)

                def coef():
                    return (lane % 16 == s_q).select(uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)), 0.0)

                return unit_fp8(wb, sb, dn_rg + gu, kc64, DN_NKC, INTER, 128, q * 32, coef, dn_ln)

            # the experts are known: stream their down weights while up/gate finishes
            pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_CPW))]
            hint_wait(
                N_UG,
                lambda k: (
                    mb("mid"),
                    (k // (MOE_SLOTS * N_UG_PER_SLOT) * MOE_SLOTS + (k // N_UG_PER_SLOT) % MOE_SLOTS) * INTER
                    + (k % N_UG_PER_SLOT) * UG_TILE
                    + UG_TILE
                    - 1,
                ),
                mark=("down", t),
            )
            mids = get2_many(
                [
                    (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2)
                    for b in range((DN_BLK + WAVES - 1) // WAVES)
                ]
            )
            stamp("down", t, 2)
            for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    if const_expr(use_fp8_block128):
                        q0, q1, qs = quant_scaled(mids[b][0], mids[b][1])
                        st_f8(blk * 128 + lane * 2, q0, q1)
                        if lane == 0:
                            lds_st(misc, blk, qs * lds_ld(dnw, blk // (INTER // 128)))
                    elif const_expr(use_mxfp8_block32):
                        d0, d1, qs = quant_mxfp8(mids[b][0], mids[b][1])
                        lds_st(xs, blk * 64 + lane, bf16_pair(d0 * qs, d1 * qs))
                    else:
                        lds_st(xs, blk * 64 + lane, bf16_pair(mids[b][0], mids[b][1]))
            gpu.barrier()
            acc = run_units(u_dn, DN_CPW, DN_BATCH, pre)

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            reduce_rows(DN_R, acc, emit_dn)
            stamp("down", t, 3)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16), rsrc(x_out), s * HIDDEN + row
                )

            if const_expr(HC > 1):
                hc_stage_coef(1)
                peer_reduce(
                    "ffn",
                    t,
                    None,
                    lambda s, row, v0, v1: hc_post(
                        s,
                        row,
                        v0,
                        v1,
                        lambda s_, j, r_: get(mb("a"), ((s_ * HC + j) * HIDDEN + r_) // 2),
                        lambda s_, k, r_, o0, o1: bo.buffer_store(
                            fx.Vector.from_elements([o0, o1], fx.Float32).to(fx.BFloat16),
                            rsrc(x_out),
                            (s_ * HC + k) * HIDDEN + r_,
                        ),
                    ),
                    tile=DN_TILE,
                )
            else:
                peer_reduce("ffn", t, mb("a"), store_x, tile=DN_TILE)
            gpu.barrier()
            stamp("down", t, 4)

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
