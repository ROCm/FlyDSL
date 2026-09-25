# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared/reuse decode kernels in one persistent launch per rank.

For GLM-5, one launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X
CU) runs the whole layer body for this rank's TP shard::

    input RMSNorm -> q_a / kv_a projection -> q_a RMSNorm -> q_b (+RoPE)
      -> KV RMSNorm / k_pe RoPE -> KV/PE cache publish
      -> absorbed q (W_UK) -> sparse MLA split softmax -> merge -> W_UV -> W_o
      -> attention TP8 peer reduce + residual                      (sym_attn)
      -> post-attention RMSNorm -> router sigmoid + expert activation staging
      -> top-8 -> 1 shared + 8 routed expert up/gate/SiLU
      -> expert down + route weighting
      -> MoE TP8 peer reduce + residual -> x_out                   (sym_ffn)

The Kimi-K3 profile runs the full-attention MLA portion through the attention
TP reduce and returns there. Its input is already normalized by the caller and
its residual/attention-residual handling remains outside this kernel.

Scheduling: every stage is a list of tasks; task ``t`` of a stage runs on CTA
``(stage_base + t) % 256`` and every CTA walks the stages in order.  There is
no grid-wide barrier: dependencies only point to earlier stages and all CTAs
are co-resident, so every spin wait makes progress.

Mailboxes are *tagged pairs*: every 32-bit value a task hands to another CTA
(or GPU) is stored next to this launch's epoch tag, ``(value, tag)``, with
device- (``sc1``) or system-coherent (``sc0 sc1``) 8 / 16-byte stores.  A
consumer polls the payload itself until the tags match, so a hand-off costs
one memory round trip: no store drain, no separate flag, no second load.

GEMVs run on the matrix cores: ``packing.py`` arranges weights so one wave
loads 16 rows x 64 k as one contiguous 1 KB. FP8 is
widened exactly to bf16 and fed to ``mfma_f32_16x16x32_bf16`` with the samples
as the N dimension.  Each 64-k chunk's partial is scaled by its f32 block
scale (times any activation scale / route weight) into the accumulator, so the
math is exact block-scaled FP8 on bf16 activations.  Weight loads that do not
depend on upstream results are issued before the task waits for its inputs.

Cross-GPU: each rank rounds partial rows to BF16 and pushes packed pairs plus
an epoch tag into every peer's symmetric buffer and polls its own; every rank sums the 8 partials in rank
order, so all ranks produce bit-identical hidden states (and routing).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value
from kernels.common import buffer_ops as bo
from kernels.common.dpp_utils import update_dpp_i32
from kernels.mla_moe_layer.config import (
    EPS,
    FP8_MAX,
    GLM5_CONFIG,
    MAX_LAYERS_PER_STEP,
    SCALE_BM,
    AttentionWeight,
    ExpertActivation,
    ExpertWeight,
    LayerConfig,
    MoeMode,
    as_layer_config,
    moe_format,
)

BLOCKS = 256
LAYER_SLOTS = MAX_LAYERS_PER_STEP
THREADS = 512
WAVES = THREADS // 64
QKV_A_TILE = 16
Q_B_TILE = 16
UK_TILE = 128
UV_TILE = 64
ROW_TILE = 32  # hidden rows per W_o / attention peer-reduce tile
ROUTER_TILE = 8  # experts per router task (a part of a 16-row MFMA group)
UG_TILE = 16  # intermediates per up/gate task (16 gate rows + 16 up rows)
SPLIT_KEYS = 64
NEG = -1.0e30


def dn_tile(S: int, hidden: int = GLM5_CONFIG.hidden) -> int:
    """Hidden rows per expert-down / FFN peer-reduce task: 32 at S = 1 (192 tasks,
    placed off the router CTAs, whose up/gate task finishes last, so every down task
    streams its weights during the mid wait); 24 above (one task per CTA), where
    each down task already streams S x 9 experts."""
    return 32 if S == 1 else hidden // BLOCKS


# gfx94x/95x cache policy bits (LLVM CPol): SC0 = 1, NT = 2, SC1 = 16.  SC1:SC0 is
# the coherence scope of the access itself: SC1 = device (past the per-XCD
# non-coherent caches), SC0|SC1 = system (peer GPUs over XGMI).
CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12  # mailbox specs polled per batch
TL_COLS = 8  # timeline stamps per task: 5 phases + 3 free debug marks


def _align(n, a=256):
    return (n + a - 1) // a * a


def layout(
    S: int,
    heads: int,
    npes: int,
    topk: int,
    moe_mode: MoeMode | str = MoeMode.W8A8,
    model_config: LayerConfig | str = GLM5_CONFIG,
    attention_only: bool = False,
):
    """Byte offsets of the per-rank scratch and of the symmetric buffer.

    Every mailbox holds ``(value, tag)`` int32 pairs (8 bytes per element)."""
    config = as_layer_config(model_config)
    hidden = config.hidden
    q_lora = config.q_lora
    kv_lora = config.kv_lora
    pe_dim = config.pe_dim
    nope_dim = config.nope_dim
    v_dim = config.v_dim
    n_experts = config.n_experts
    moe_slots = config.moe_slots
    inter = config.inter
    fmt = moe_format(moe_mode)
    quant_group = fmt.activation_group
    xq_blocks = 0 if quant_group is None else hidden // quant_group
    n_split = topk // SPLIT_KEYS
    pr = 8
    items = [
        ("q_a", S * q_lora * pr),
        ("kv_a", S * (kv_lora + pe_dim) * pr),
        ("gate", S * heads * v_dim * pr if config.attention_output_gate else 0),
        ("kvnew", S * kv_lora * pr),  # this launch's KV cache rows (bf16 values)
        ("penew", S * pe_dim * pr),
        ("q_nope", S * heads * nope_dim * pr),
        ("q_pe", S * heads * pe_dim * pr),
        ("q_lat", S * heads * kv_lora * pr),
        ("sp_acc", S * n_split * heads * kv_lora * pr),
        ("sp_m", S * n_split * heads * pr),
        ("sp_l", S * n_split * heads * pr),
        ("o", S * heads * v_dim * pr),
        ("a", S * hidden * pr),  # post-attention hidden (bf16 values)
    ]
    if not attention_only:
        items += [
            ("scores", S * n_experts * pr),
            ("xq", S * hidden // (4 if quant_group is not None else 2) * pr),
            ("xqs", S * xq_blocks * pr),
            ("sel", S * moe_slots * pr),
            ("prob", S * moe_slots * pr),
            ("mid", S * moe_slots * inter * pr),
            ("ugp", BLOCKS * S * 2 * UG_TILE * pr),  # up/gate K-segment partial sums
            ("xqd", S * hidden * 4),  # debug: dequantized MoE activation (plain f32)
        ]
    off, scratch = 0, {}
    for name, size in items:
        scratch[name] = off
        off += _align(size)
    scratch["_bytes"] = off
    part = npes * S * hidden * pr
    sym = {"attn": 0, "ffn": part, "_bytes": part if attention_only else 2 * part}
    return scratch, sym


def _rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def _uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def _uniform_f32(v):
    return _uniform(fx.Float32(v).bitcast(fx.Int32)).bitcast(fx.Float32)


def _hw_f32(name, x):
    """One hardware transcendental (v_rsq / v_rcp / v_exp): ~1 ulp, no libm range fixups."""
    return fx.Float32(llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], []))


def _rsq(x):
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def _rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def _exp(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", fx.Float32(x) * 1.4426950408889634)


def _xshfl(v, off):
    """Value of lane ``lane ^ off``.  Offsets 32 / 16 lower to v_permlane*_swap; the
    in-row offsets use DPP (VALU latency) instead of ds_swizzle (LDS latency)."""
    if off >= 16:
        return v.shuffle_xor(off, 64)
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    if off == 8:  # row_shr:8 into banks 2-3, row_shl:8 into banks 0-1
        y = fx.Int32(update_dpp_i32(x, x, 0x118, 0xF, 0xC, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x108, 0xF, 0x3, False))
    elif off == 4:
        y = fx.Int32(update_dpp_i32(x, x, 0x114, 0xF, 0xA, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x104, 0xF, 0x5, False))
    elif off == 2:  # quad_perm [2, 3, 0, 1]
        y = fx.Int32(update_dpp_i32(x, x, 0x4E, 0xF, 0xF, False))
    else:  # quad_perm [1, 0, 3, 2]
        y = fx.Int32(update_dpp_i32(x, x, 0xB1, 0xF, 0xF, False))
    return y.bitcast(fx.Float32) if is_f else y


def _wave_umax(v):
    """Unsigned max over the fully active wave as a wave-uniform Int32."""
    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(v), fx.ReductionOp.MAX, width=64))


def _xred(v, off, op):
    """op(v, value of lane ``lane ^ off``) for a symmetric op.  Offsets 32 / 16 take
    both halves of one v_permlane*_swap as the operands (no select needed)."""
    if off < 16:
        return op(v, _xshfl(v, off))
    is_f = isinstance(v, fx.Float32)
    x = as_ir_value(v.bitcast(fx.Int32) if is_f else fx.Int32(v))
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(llvm.StructType.get_literal([T.i32, T.i32]), x, x, False, False)
    a, b = (fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))
    if is_f:
        return op(a.bitcast(fx.Float32), b.bitcast(fx.Float32))
    return op(type(v)(a), type(v)(b))


def _ballot(pred):
    return fx.Int64(rocdl.ballot(T.i64, fx.Boolean(pred).ir_value()))


def _popc(mask):
    return fx.Int32(fx.Int64(fmath.ctpop(mask)))


def _mbcnt(mask):
    """Number of set bits of the 64-bit lane mask below this lane."""
    lo = llvm.call_intrinsic(
        T.i32, "llvm.amdgcn.mbcnt.lo", [fx.Int32(mask & 0xFFFFFFFF).ir_value(), fx.Int32(0).ir_value()], [], []
    )
    return fx.Int32(llvm.call_intrinsic(T.i32, "llvm.amdgcn.mbcnt.hi", [fx.Int32(mask >> 32).ir_value(), lo], [], []))


def _fp8_roundtrip(a, b):
    """f32 pair -> E4M3FN -> f32 pair (inputs already scaled into range)."""
    word = rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False)
    v2 = fx.Vector.make_type(2, fx.Float32)
    lo = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))
    return lo[0], lo[1]


def f8_word(k):
    """LDS word of FP8 activation byte k: each 64-k chunk is stored so that the 16 bytes
    lane group g needs (k = 8 g + [0, 8) and 32 + 8 g + [0, 8), the packed weight order)
    are contiguous."""
    return (k // 64) * 16 + ((k % 32) // 8) * 4 + ((k % 64) // 32) * 2 + (k % 8) // 4


def _fp8_to_bf16x8(w0, w1):
    """Two dwords of 8 FP8 -> vector<8 x bf16> (exact: E4M3 is a subset of bf16).

    ``cvt_scalef32_pk_bf16_fp8`` only honours the scale's exponent, so the scale
    is 1 here and the f32 block scale is applied to the MFMA partials instead.
    """
    one = as_ir_value(fx.Float32(1.0))
    parts = []
    for w in (w0, w1):
        for half in range_constexpr(2):
            pr = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp8(T.vec(2, T.bf16), as_ir_value(w), one, bool(half)))
            parts += [pr[0], pr[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def _mxfp4_to_bf16x8(word, scale):
    """One packed dword of eight E2M1 values -> scaled BF16 MFMA operand."""

    parts = []
    for select in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), as_ir_value(word), as_ir_value(scale), select)
        )
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def stage_tasks(
    S: int,
    heads: int,
    topk: int,
    model_config: LayerConfig | str = GLM5_CONFIG,
    attention_only: bool = False,
):
    """[(stage name, task count)] in execution order."""
    config = as_layer_config(model_config)
    head_groups = (heads + WAVES - 1) // WAVES
    split_ctas_per_tile = head_groups if S == 1 else 1
    tasks = [
        ("qkv_a", config.qkv_a_rows // QKV_A_TILE),
        ("cache", 1),
        ("q_b", heads * (config.nope_dim + config.pe_dim) // Q_B_TILE),
        ("uk", heads * config.kv_lora // UK_TILE),
        ("split", S * (topk // SPLIT_KEYS) * split_ctas_per_tile),
        ("uv", S * (heads * config.v_dim // UV_TILE)),
        ("o", config.hidden // ROW_TILE),
    ]
    if attention_only:
        return tasks
    tasks += [
        ("router", S * (config.n_experts // ROUTER_TILE)),
        (
            "ug",
            (BLOCKS if S == 1 else S * BLOCKS),
        ),
        ("down", config.hidden // dn_tile(S, config.hidden)),
    ]
    return tasks


def build_shared_reuse_kernel(
    S: int = 1,
    heads: int = 8,
    npes: int = 8,
    topk: int = 2048,
    scale: float | None = None,
    timeline: bool = False,
    moe_mode: MoeMode | str = MoeMode.W8A8,
    model_config: LayerConfig | str = GLM5_CONFIG,
    attention_only: bool = False,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer.

    ``timeline=True`` records ``s_memrealtime`` (100 MHz) at the start and end of
    every task, and once its inputs have arrived, into the ``timeline`` buffer:
    int64 ``[sum(task counts), TL_COLS]`` (start, hint seen, inputs staged, compute
    done, end, then free debug marks) in ``stage_tasks`` order.
    """
    config = as_layer_config(model_config)
    HIDDEN = config.hidden
    Q_LORA = config.q_lora
    KV_LORA = config.kv_lora
    PE_DIM = config.pe_dim
    NOPE_DIM = config.nope_dim
    V_DIM = config.v_dim
    QKV_A_ROWS = config.qkv_a_rows
    N_EXPERTS = config.n_experts
    TOP_K = config.top_k
    MOE_SLOTS = config.moe_slots
    SHARED_EXPERT = config.shared_expert
    INTER = config.inter
    ROUTE_SCALE = config.route_scale
    SOFTMAX_SCALE = config.softmax_scale
    if scale is None:
        scale = SOFTMAX_SCALE
    N_QKV_A = QKV_A_ROWS // QKV_A_TILE
    N_ROW_TILES = HIDDEN // ROW_TILE
    N_ROUTER = N_EXPERTS // ROUTER_TILE
    N_UG_PER_SLOT = INTER // UG_TILE
    HEAD_GROUPS = (heads + WAVES - 1) // WAVES
    SPLIT_CTAS_PER_TILE = HEAD_GROUPS if S == 1 else 1
    HEAD_GROUPS_PER_CTA = 1 if S == 1 else HEAD_GROUPS
    PAD_HEADS = HEAD_GROUPS * WAVES
    attention_bf16 = config.attention_weight is AttentionWeight.BF16
    attention_output_gate = config.attention_output_gate
    attention_input_norm = config.attention_input_norm
    attention_residual = config.attention_residual

    assert heads == config.local_heads, f"{config.name} requires {config.local_heads} local heads"
    assert topk % SPLIT_KEYS == 0 and 1 <= S <= 8
    fmt = moe_format(moe_mode)
    use_fp8_block128 = fmt.activation is ExpertActivation.FP8_BLOCK128
    use_mxfp8_block32 = fmt.activation is ExpertActivation.MXFP8_BLOCK32
    use_mxfp4_weight = fmt.weight is ExpertWeight.MXFP4_BLOCK32
    XQ_BLOCKS = 0 if fmt.activation_group is None else HIDDEN // fmt.activation_group
    PUBLISH_BLOCKS = HIDDEN // (32 if use_mxfp8_block32 else 128)
    XQ_WAVES = (
        (PUBLISH_BLOCKS + N_ROUTER * 4 - 1) // (N_ROUTER * 4)
        if use_mxfp8_block32
        else (PUBLISH_BLOCKS + N_ROUTER - 1) // N_ROUTER
    )
    assert XQ_WAVES <= WAVES
    down_scale_words = 0 if fmt.activation_group is None else S * MOE_SLOTS * INTER // fmt.activation_group
    misc_words = max(topk // SPLIT_KEYS, 8 + max(S * XQ_BLOCKS, down_scale_words))
    H = heads
    W = npes
    G = BLOCKS
    SC, SY = layout(S, H, W, topk, moe_mode, config, attention_only)
    N_SPLIT = topk // SPLIT_KEYS
    QB_ROWS = H * (NOPE_DIM + PE_DIM)
    N_QB = QB_ROWS // Q_B_TILE
    QB_PER_HEAD = (NOPE_DIM + PE_DIM) // Q_B_TILE
    N_UK = H * KV_LORA // UK_TILE
    UK_PER_HEAD = KV_LORA // UK_TILE
    N_UV = H * V_DIM // UV_TILE
    O_K = H * V_DIM
    N_UG = S * MOE_SLOTS * N_UG_PER_SLOT
    QK_DIM = KV_LORA + PE_DIM
    # split LDS: bf16 q of all heads, then the KV latent / k_pe tiles (bf16 pairs); row
    # strides are padded by 4 words so the MFMA operand rows spread over the banks
    QS = QK_DIM // 2 + 4
    KS = KV_LORA // 2 + 4
    PS = PE_DIM // 2 + 4
    KT_OFF = H * QS
    PT_OFF = KT_OFF + SPLIT_KEYS * KS
    XN = max(S * HIDDEN // 2, PT_OFF + SPLIT_KEYS * PS)
    ON = S * UK_TILE
    DN_TILE = dn_tile(S, HIDDEN)
    N_DN_TILES = HIDDEN // DN_TILE

    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(S, H, topk, config, attention_only):
        first[name] = acc
        acc += n
    # CTA placement: split before uk, so every split tile lands on a CTA freed by
    # qkv_a (uk shares the q_b CTAs it waits on anyway)
    tasks = dict(stage_tasks(S, H, topk, config, attention_only))
    acc = 0
    stage_order = ["qkv_a", "cache", "q_b", "split", "uk", "uv", "o"]
    if not attention_only:
        stage_order += ["router", "ug", "down"]
    for name in stage_order:
        base[name] = acc % G
        acc += tasks[name]

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XN, 16]  # bf16 activations (pairs) / split q + KV tile
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, misc_words, 16]
        p: fx.Array[fx.Float32, PAD_HEADS * SPLIT_KEYS, 16]
        keys: fx.Array[fx.Int32, SPLIT_KEYS, 16]
        dnw: fx.Array[fx.Float32, S * MOE_SLOTS, 16]  # expert-down route weights

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def shared_reuse_kernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
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
        ktile = xs + KT_OFF  # f32-typed views holding raw bf16 pairs
        petile = xs + PT_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        r_h = _rsrc(h_in)
        # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
        # device counter bumped once per decode step (graph friendly); ``layer``
        # makes it unique per layer within the step.
        tag = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32)) * LAYER_SLOTS + layer + 1
        pos0 = _uniform(bo.buffer_load(_rsrc(cur_pos), 0, vec_width=1, dtype=T.i32))
        r_peers = _rsrc(peers)
        # Each wave sends to one peer, so retain only that wave's destination.
        pv = fx.Vector(bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32))
        peer_dst = (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))

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
            bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [fx.Float32(v0).bitcast(fx.Int32), tag, fx.Float32(v1).bitcast(fx.Int32), tag], fx.Int32
            )
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
            i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store."""
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), _rsrc(base_addr), i, cache_modifier=cm)

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)

        def _qptr(addr):
            return fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8), fx.Int64(addr))

        def _ld_pair(addr, scope):
            """One (value, tag) pair as a single 64-bit relaxed atomic load: never hoisted,
            coherent at ``scope`` (agent -> sc1, system -> sc0 sc1)."""
            return fx.generic_load(_qptr(addr), memory_order=fx.AtomicOrdering.Monotonic, syncscope=scope)

        def poll(specs, scope="agent", batch=POLL_MAX):
            """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

            All pairs are loaded together with plain 8 / 16-byte coherent buffer loads
            (sc1 locally, sc0 sc1 for peer memory); while any tag is not this launch's
            the whole batch is re-loaded, so a batch costs one round trip after its
            last producer lands.  A side-effecting scheduling op in the retry loop
            keeps the loads from being hoisted.  Returns one list of Int32 value bits
            per spec."""
            if const_expr(len(specs) == 0):
                return []
            if const_expr(len(specs) > batch):  # bound live registers
                return poll(specs[:batch], scope, batch) + poll(specs[batch:], scope, batch)
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

            def load_all():
                words = []
                for b, i, n in specs:
                    w = fx.Vector(
                        bo.buffer_load(_rsrc(b), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32, cache_modifier=cm)
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
                rocdl.s_nop(0)
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
            return [(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in poll([(b, i, 2) for b, i in specs])]

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

        def subgroup16_max(v):
            for off in (8, 4, 2, 1):
                v = _xred(v, off, fx.max)
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
            """Issue one 128-k chunk (packed 64-k chunks kc, kc + 1; kc even) of row group
            ``rg`` against the FP8 activation of LDS words ``b_word`` + [0, 32) (``f8_word``
            order); ``coef()`` = activation block scale (times route weight).  ``ln``
            = the lane whose weights are loaded (default: own lane)."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
            return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, ln=None):
            """Issue one packed 128-K MXFP4 tile and its four per-row E8M0 scales."""

            ln = lane if ln is None else ln
            raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            row = rg * 16 + ln % 16
            packed_scale = fx.Int32(bo.buffer_load(s_rsrc, row * (K // 128) + kc, vec_width=1, dtype=T.i32))
            scales = [
                ((packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
                for sp in range_constexpr(4)
            ]
            return ("mxfp4", (raw, scales), coef, b_word + (lane // 16) * 4)

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def unit_attention(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, ln=None):
            """Issue one configured attention-weight chunk."""

            if const_expr(attention_bf16):
                return unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln)
            return unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, ln=ln)

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for unit_format, wv, coef, bw in units:
                if const_expr(callable(coef) and unit_format != "mxfp4"):
                    coef = coef()
                if const_expr(unit_format == "mxfp4"):
                    raw, scales = wv
                    for sp in range_constexpr(4):
                        a = _mxfp4_to_bf16x8(raw[sp], scales[sp])
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                        c = fx.Vector.filled(4, 0.0, fx.Float32)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                        part_coef = coef[sp] if const_expr(isinstance(coef, list)) else coef
                        if const_expr(callable(part_coef)):
                            part_coef = part_coef()
                        if const_expr(part_coef is None):
                            acc = [acc[e] + c[e] for e in range(4)]
                        else:
                            acc = [acc[e] + c[e] * part_coef for e in range(4)]
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
                        a = _fp8_to_bf16x8(wv[0][sp * 2], wv[0][sp * 2 + 1])
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

        def stage_x_rmsnorm(ld4s, n, gamma, mark=None, loaded=None, count=S):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
            ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
            ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
            per = (n + 4 * THREADS - 1) // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            sss = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    valid = ks[i] < n
                    for a in vals[s * per + i]:
                        ss = ss + valid.select(a * a, fx.Float32(0.0))
                sss.append(ss)
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 6)
            rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 7)
            for s in range_constexpr(count):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    if ks[i] < n:
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
            per = (n + 4 * THREADS - 1) // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            safe_ks = [fx.min(k, n - 4) for k in ks]
            gs = []
            for k in safe_ks:
                g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(count) for k in safe_ks])

        def load_x_bf16(ld4s, n, count=S):
            """Issue direct BF16 loads for an activation normalized by the caller."""

            per = (n + 4 * THREADS - 1) // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            safe_ks = [fx.min(k, n - 4) for k in ks]
            return ld4s([(s, k) for s in range(count) for k in safe_ks])

        def stage_x_bf16(ld4s, n, loaded=None, count=S):
            """Stage a pre-normalized BF16 activation without applying another norm."""

            per = (n + 4 * THREADS - 1) // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            vals = loaded if loaded is not None else load_x_bf16(ld4s, n, count)
            for s in range_constexpr(count):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    if ks[i] < n:
                        for j in range_constexpr(2):
                            lds_st(xs, (s * n + ks[i]) // 2 + j, bf16_pair(a[2 * j], a[2 * j + 1]))

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

        def stage_attention_output():
            """Stage the BF16 attention output; Kimi-K3 gating is fused at W_UV."""

            stage_x_pairs("o", S * O_K, lambda k: k)

        def quant_scaled(a0, a1):
            """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
            amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
            inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))  # hardware rcp, no IEEE divide
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, qs

        def quant_block(a0, a1):
            """quant_scaled, values returned as the FP8-rounded f32s."""
            q0, q1, qs = quant_scaled(a0, a1)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, qs

        def quant_mxfp8(a0, a1):
            """Per-16-lane/32-value MXFP8 quantization with an E8M0 scale."""

            amax = subgroup16_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            raw_scale = amax * (1.0 / FP8_MAX)
            bits = raw_scale.bitcast(fx.Int32)
            exponent = (bits.shrui(fx.Int32(23))) & fx.Int32(0xFF)
            round_up = ((bits & fx.Int32(0x400000)) != 0) & (
                ((bits & fx.Int32(0x200000)) != 0) | ((bits & fx.Int32(0x1FFFFF)) != 0) | (exponent > 0)
            )
            exponent = exponent + round_up.select(fx.Int32(1), fx.Int32(0))
            scale = nz.select((exponent << fx.Int32(23)).bitcast(fx.Float32), fx.Float32(1.0))
            inv = nz.select(_rcp(scale), fx.Float32(1.0))
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, scale

        def stage_moe_input(samples):
            """Stage normalized expert inputs published by the router into LDS."""
            if const_expr(use_fp8_block128):
                nxw = HIDDEN // 4 // THREADS
                got = poll(
                    [(mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1) for sx in samples for i in range(nxw)]
                    + [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
                )
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(nxw):
                        wd = f8_word((tid + i * THREADS) * 4)
                        lds_st(xs, j * (HIDDEN // 4) + wd, got[j * nxw + i][0].bitcast(fx.Float32))
                    if tid < XQ_BLOCKS:
                        lds_st(
                            misc,
                            8 + j * XQ_BLOCKS + tid,
                            got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                        )
            elif const_expr(use_mxfp8_block32):
                chunks = HIDDEN // 8
                per_thread = (chunks + THREADS - 1) // THREADS
                data_specs = []
                for sx in samples:
                    for i in range_constexpr(per_thread):
                        chunk = fx.min(tid + i * THREADS, chunks - 1)
                        data_specs.append((mb("xq"), sx * (HIDDEN // 4) + chunk * 2, 2))
                scale_specs = [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
                got = poll(data_specs + scale_specs)
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(per_thread):
                        chunk = tid + i * THREADS
                        if chunk < chunks:
                            words = got[j * per_thread + i]
                            values = _fp8_to_bf16x8(words[0], words[1])
                            for pair in range_constexpr(4):
                                lds_st(
                                    xs,
                                    j * (HIDDEN // 2) + chunk * 4 + pair,
                                    fx.Vector.from_elements(
                                        [values[2 * pair], values[2 * pair + 1]], fx.BFloat16
                                    ).bitcast(fx.Float32)[0],
                                )
                    if tid < XQ_BLOCKS:
                        lds_st(
                            misc,
                            8 + j * XQ_BLOCKS + tid,
                            got[len(samples) * per_thread + j][0].bitcast(fx.Float32),
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
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

        def load_bias():
            """This lane's 4 expert biases (issue before the scores wait)."""
            return [ld_f32(_rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

        def route_top8(s, raws=None, bs=None):
            """Top-8 of sample s (call from one whole wave, after the router scores landed).

            Packed-key argmax: key = order-preserving bits of (sigmoid + bias) with the
            low byte replaced by 255 - expert id (unique; near-ties go to the lower id),
            so each of the 8 rounds is one u32 wave max (candidate i of this lane is
            expert lane + 64 i).  Returns (expert id, route weight = raw score / sum of
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
                ks.append(fx.Uint32((ok & fx.Int32(-256)) | (255 - (lane + i * 64))))
            # sort this lane's 4 keys descending; each round then takes the wave max of
            # the lane heads and shifts the winning lane's list (0 is below every key)
            for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
                ks[a], ks[b] = fx.max(ks[a], ks[b]), fx.min(ks[a], ks[b])
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            mv = fx.Int32(0)  # lane k: the key of pick k
            for k in range_constexpr(TOP_K):
                m = _wave_umax(ks[0])
                hit = ks[0] == m
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(4)] + [ks[4]]
                mv = fx.Int32(
                    llvm.call_intrinsic(
                        T.i32,
                        "llvm.amdgcn.writelane.i32",
                        [m.ir_value(), fx.Int32(k).ir_value(), mv.ir_value()],
                        [],
                        [],
                    )
                )
            e = 255 - (mv & 255)
            src = (e % 64) * 4
            got = [fx.Int32(rocdl.ds_bpermute(T.i32, src.ir_value(), r.bitcast(fx.Int32).ir_value())) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, N_EXPERTS // 64):
                raw = (e // 64 == i).select(got[i], raw)
            raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
            tot = raw
            for off in (1, 2, 4):
                tot = _xred(tot, off, lambda a, b: a + b)
            return e, raw * (_rcp(tot) * ROUTE_SCALE)

        def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
            """Push BF16 partials in tagged pairs to every peer, then sum all
            ranks' pairs from the own symmetric buffer in rank order (W = 1: no exchange).  ``residual`` is
            either fn(s, row) -> (r0, r1) (plain loads, issued first) or a mailbox base
            (pairs s * HIDDEN + row, polled in the same batch as the peers)."""
            if const_expr(W > 1):
                # One wave per destination: peer pointers are wave-uniform, and the
                # destinations progress concurrently instead of eight serial stores
                # from the output wave. All waves consume outs before it is reused.
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
                    own = sym + fx.Int64(SY[region])
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

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        def stamp(name, t, which, lead=0):
            if const_expr(timeline):
                if tid == lead:
                    now = fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
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

        # ================================================= 1. q_a / kv_a GEMV
        # 1 row group x 96 chunks: 8 waves split K, 12 chunks each (all prefetched)
        r_wqa, r_sqa = _rsrc(w_qkv_a), _rsrc(s_qkv_a)
        QA_NKC = HIDDEN // 64
        for t in range(start("qkv_a"), N_QKV_A, G):
            t = fx.Int32(t)
            stamp("qkv_a", t, 0)

            def u_qa(c):
                kc = wave * (QA_NKC // WAVES) + c
                return unit_attention(
                    r_wqa,
                    r_sqa,
                    t,
                    kc,
                    QA_NKC,
                    HIDDEN,
                    128,
                    (n_sel() * HIDDEN + kc * 64) // 2,
                )

            def ld_h(sks):
                res = []
                for s, k in sks:
                    w = fx.Vector(bo.buffer_load(r_h, (s * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            # the (small) input loads go out before the weight stream: loads complete in order
            h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in) if const_expr(attention_input_norm) else load_x_bf16(ld_h, HIDDEN)
            pre = [u_qa(c) for c in range(QA_NKC // WAVES)]
            if const_expr(attention_input_norm):
                stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld)
            else:
                stage_x_bf16(ld_h, HIDDEN, loaded=h_ld)
            gpu.barrier()
            stamp("qkv_a", t, 2)
            acc = run_units(u_qa, QA_NKC // WAVES, QA_NKC // WAVES, pre)
            reduce_rows(1, acc, emit_out(QKV_A_TILE))
            stamp("qkv_a", t, 3)
            gpu.barrier()
            if tid < S * QKV_A_TILE:
                s = tid // QKV_A_TILE
                row = t * QKV_A_TILE + tid % QKV_A_TILE
                v = lds_ld(outs, tid)
                if row < Q_LORA:
                    put(mb("q_a"), s * Q_LORA + row, v)
                elif row < Q_LORA + KV_LORA + PE_DIM:
                    put(mb("kv_a"), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)
                else:
                    put(mb("gate"), s * O_K + row - (Q_LORA + KV_LORA + PE_DIM), v)
            stamp("qkv_a", t, 4)

        # ================ 2. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
        for t in range(start("cache"), 1, G):
            stamp("cache", t, 0)
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            # gamma and the RoPE factors are issued ahead of the wait
            g = ld_bf16(_rsrc(g_kv), tid)
            tpe = tid % (PE_DIM // 2)
            cs = [ld_f32(_rsrc(rope_cos), (pos0 + s) * (PE_DIM // 2) + tpe) for s in range(S)]
            sns = [ld_f32(_rsrc(rope_sin), (pos0 + s) * (PE_DIM // 2) + tpe) for s in range(S)]
            hint_wait(
                (KV_LORA + PE_DIM) // QKV_A_TILE,
                lambda k: (mb("kv_a"), (S - 1) * (KV_LORA + PE_DIM) + k * QKV_A_TILE + QKV_A_TILE - 1),
                mark=("cache", t),
            )
            # every sample's kv latent and k_pe pair in one poll, one block reduction
            vs = getf_many([(mb("kv_a"), s * (KV_LORA + PE_DIM) + tid) for s in range(S)])
            pes = get2_many(
                [(mb("kv_a"), s * (KV_LORA + PE_DIM) + KV_LORA + (tid % (PE_DIM // 2)) * 2) for s in range(S)]
            )
            stamp("cache", t, 2)
            ssq = block_sums([v * v for v in vs])
            for s in range_constexpr(S):
                pos = pos0 + s
                kvn = bf16_round(vs[s] * _rsq(ssq[s] * (1.0 / KV_LORA) + EPS) * g)
                bo.buffer_store(kvn.to(fx.BFloat16), r_kv, pos * KV_LORA + tid)
                put(mb("kvnew"), s * KV_LORA + tid, kvn)
                if tid < PE_DIM // 2:
                    x0, x1 = pes[s]
                    c, sn = cs[s], sns[s]
                    p0 = bf16_round(x0 * c - x1 * sn)
                    p1 = bf16_round(x0 * sn + x1 * c)
                    bo.buffer_store(p0.to(fx.BFloat16), r_pe, pos * PE_DIM + tid * 2)
                    bo.buffer_store(p1.to(fx.BFloat16), r_pe, pos * PE_DIM + tid * 2 + 1)
                    put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)
            stamp("cache", t, 4)

        # ============================================ 3. q_a RMSNorm -> q_b (+RoPE)
        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        for t in range(start("q_b"), N_QB, G):
            t = fx.Int32(t)
            stamp("q_b", t, 0)

            def u_qb(c):
                kc = wave * (QB_NKC // WAVES) + c
                return unit_attention(
                    r_wqb,
                    r_sqb,
                    t,
                    kc,
                    QB_NKC,
                    Q_LORA,
                    128,
                    (n_sel() * Q_LORA + kc * 64) // 2,
                )

            pre = [u_qb(c) for c in range(QB_NKC // WAVES)]
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
            acc = run_units(u_qb, QB_NKC // WAVES, QB_NKC // WAVES, pre)
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
                    c = ld_f32(_rsrc(rope_cos), (pos0 + s) * (PE_DIM // 2) + i // 2)
                    sn = ld_f32(_rsrc(rope_sin), (pos0 + s) * (PE_DIM // 2) + i // 2)
                    put_bf(mb("q_pe"), (s * H + head) * PE_DIM + i, [x0 * c - x1 * sn, x0 * sn + x1 * c])
            stamp("q_b", t, 4)

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
                    r_wuk, r_suk, t * WAVES + wave, c, UK_NKC, NOPE_DIM, 64, (n_sel() * NOPE_DIM + c * 64) // 2
                )

            pre = [u_uk(c) for c in range(UK_NKC)]
            hint_wait(
                NOPE_DIM // Q_B_TILE,
                lambda k: (mb("q_nope"), ((S - 1) * H + head) * NOPE_DIM + k * Q_B_TILE + Q_B_TILE - 1),
                mark=("uk", t),
            )
            stage_x_pairs("q_nope", S * NOPE_DIM, lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM)
            stamp("uk", t, 2)
            gpu.barrier()
            acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
            reduce_rows(WAVES, acc, emit_out(UK_TILE))
            stamp("uk", t, 3)
            gpu.barrier()
            if tid < S * UK_TILE // 4:
                k = tid * 4
                s = k // UK_TILE
                put_bf(
                    mb("q_lat"),
                    (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + k % UK_TILE,
                    [lds_ld(outs, k + j) for j in range(4)],
                )
            stamp("uk", t, 4)

        # ================================== 5. sparse MLA split: 64 keys x 8 heads
        r_kv = _rsrc(kv_cache)
        r_pe = _rsrc(pe_cache)
        r_idx = _rsrc(indices)
        KPW = SPLIT_KEYS // WAVES

        def split_keys(t, s):
            """(nkeys, sparse) of sample s; wave 0 writes this split's 64 cache rows to LDS keys."""
            kv_len = pos0 + s + 1
            sparse = kv_len > topk
            nkeys = sparse.select(fx.Int32(topk), kv_len)
            if wave == 0:
                k_pos = t * SPLIT_KEYS + lane
                k_cl = (k_pos < nkeys).select(k_pos, 0)
                lds_st(
                    keys,
                    lane,
                    sparse.select(fx.Int32(bo.buffer_load(r_idx, s * topk + k_cl, vec_width=1, dtype=T.i32)), k_cl),
                )
            return nkeys, sparse

        def gather_old_kv():
            """Each wave copies its 8 keys' KV latent (1 KB) + k_pe (128 B) cache rows
            into the LDS tiles (rows of this launch are patched in by patch_new_kv)."""
            krows = [lds_ld(keys, wave * KPW + jj) for jj in range(KPW)]
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kv8 = fx.Vector(bo.buffer_load(r_kv, krows[jj] * (KV_LORA // 2) + lane * 4, vec_width=4, dtype=T.i32))
                fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 4))
                if lane < PE_DIM // 2:
                    lds_st(petile, j * PS + lane, ld_f32(r_pe, krows[jj] * (PE_DIM // 2) + lane))

        def patch_new_kv():
            """Rows appended by this launch come from the cache task's kvnew / penew pairs."""
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kr = lds_ld(keys, j)
                if kr >= pos0:
                    sn = kr - pos0
                    kvp = get2_many([(mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2) for m in range(4)])
                    w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                    fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * KS + lane * 4))
                    if lane < PE_DIM // 2:
                        a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                        lds_st(petile, j * PS + lane, bf16_pair(a0, a1))

        for tt in range(start("split"), S * N_SPLIT * SPLIT_CTAS_PER_TILE, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s = tt // (N_SPLIT * SPLIT_CTAS_PER_TILE)  # sample
            split_group = tt % (N_SPLIT * SPLIT_CTAS_PER_TILE)
            t = split_group // SPLIT_CTAS_PER_TILE  # 64-key chunk
            task_head_group = split_group % SPLIT_CTAS_PER_TILE
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
                [(mb("q_lat"), (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ)]
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
            # scores = K Q^T on MFMA: keys are M (4 row groups), the 576 dims K
            # (18 steps of 32, split in two halves), heads N.  wave = (row group, half)
            score_head_base = task_head_group * WAVES
            hn = fx.min(score_head_base + lane % 16, H - 1)
            rgk = wave % 4
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(QK_DIM // 32 // 2):
                kst = (wave // 4) * (QK_DIM // 32 // 2) + st
                key = rgk * 16 + lane % 16
                kw = (kst < KV_LORA // 32).select(
                    KT_OFF + key * KS + kst * 16,
                    PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                )
                a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                b = fx.ptr_load(xs + (hn * QS + kst * 16 + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            fx.ptr_store(c, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            stamp("split", tt, 6)
            # The score MFMA above produces 16 head columns in one pass.  Process
            # those columns in 8-wave groups so Kimi-K3's heads 8..11 reuse the
            # already-staged Q/KV and scores instead of launching a second CTA.
            for local_head_group in range_constexpr(HEAD_GROUPS_PER_CTA):
                head_base = (task_head_group + local_head_group) * WAVES
                h = head_base + wave
                # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
                kidx = t * SPLIT_KEYS + lane
                valid = kidx < nkeys
                r16 = lane % 16
                score_head = local_head_group * WAVES + wave
                cl = score_head + 16 * (r16 // 4)
                raw = lds_ld(red, ((lane // 16) * 64 + cl) * 4 + r16 % 4) + lds_ld(
                    red, ((lane // 16 + 4) * 64 + cl) * 4 + r16 % 4
                )
                sc_v = valid.select(raw * scale, fx.Float32(NEG))
                m = wave_max(sc_v)
                p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                lsum = wave_sum(p)
                p_n = _xshfl(p, 1)
                if lane % 2 == 0:  # P^T bf16 [h][64 keys] (words h * 32 + j / 2)
                    lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
                gpu.barrier()
                stamp("split", tt, 3)
                # O = P V on MFMA: heads M, keys K (2 steps), latent dims N.  Each V word holds
                # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
                # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
                hn = fx.min(head_base + lane % 16, H - 1)
                for g in range_constexpr(KV_LORA // 32 // WAVES):
                    dw = (wave * (KV_LORA // 32 // WAVES) + g) * 16 + lane % 16  # dim pair word
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
                    if lane < 32:  # rows (heads) 4 * (lane // 16) + e < 8
                        for e in range_constexpr(4):
                            hh = head_base + (lane // 16) * 4 + e
                            if hh < H:
                                put_bf(
                                    mb("sp_acc"),
                                    ((s * N_SPLIT + t) * H + hh) * KV_LORA + dw * 2,
                                    [c0[e], c1[e]],
                                )
                if (lane == 0) & (h < H):  # written last: the merge's readiness hint
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
                gpu.barrier()
            stamp("split", tt, 4)

        # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
        # 4 row groups x 8 chunks: 2 waves per row group, 4 chunks each
        r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
        UV_NKC = KV_LORA // 64
        UV_R = UV_TILE // 16
        UV_WPR = WAVES // UV_R
        for tt in range(start("uv"), S * N_UV, G):
            tt = fx.Int32(tt)
            stamp("uv", tt, 0)
            s = tt // N_UV  # sample
            t = tt % N_UV  # 64-row tile
            head = t // (V_DIM // UV_TILE)

            def u_uv(c):
                kc = (wave % UV_WPR) * (UV_NKC // UV_WPR) + c
                return unit_attention(
                    r_wuv,
                    r_suv,
                    t * UV_R + wave // UV_WPR,
                    kc,
                    UV_NKC,
                    KV_LORA,
                    128,
                    (kc * 64) // 2,
                )

            pre = [u_uv(c) for c in range(UV_NKC // UV_WPR)]
            hint_wait(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head), mark=("uv", tt))
            pre_poll(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            stamp("uv", tt, 5)
            # one batched poll: thread = (d pair dp, split half hf) -> its SPH splits' acc
            # pairs, plus lane's split (m, l) so wave 0 can form the merge weights
            SPH = N_SPLIT // 2
            dp = tid % (KV_LORA // 2)
            hf = tid // (KV_LORA // 2)
            spi = fx.min(lane, N_SPLIT - 1)
            ml = (s * N_SPLIT + spi) * H + head
            got = poll(
                [(mb("sp_acc"), ((s * N_SPLIT + hf * SPH + j) * H + head) * (KV_LORA // 2) + dp, 1) for j in range(SPH)]
                + [(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)],
                batch=SPH + 2,
            )
            if wave == 0:  # per-split weights exp(m - M) / L for this head -> misc[sp]
                ok_sp = lane < N_SPLIT
                m_sp = ok_sp.select(got[SPH][0].bitcast(fx.Float32), fx.Float32(NEG))
                l_sp = ok_sp.select(got[SPH + 1][0].bitcast(fx.Float32), fx.Float32(0.0))
                w_sp = _exp(m_sp - wave_max(m_sp))
                den = wave_sum(l_sp * w_sp)
                if ok_sp:
                    lds_st(misc, lane, w_sp * _rcp(den))
            stamp("uv", tt, 2)
            gpu.barrier()
            o0 = fx.Float32(0.0)
            o1 = fx.Float32(0.0)
            for j in range_constexpr(SPH):
                wj = lds_ld(misc, hf * SPH + j)
                a0, a1 = bf2_f32(got[j][0])
                o0 = o0 + a0 * wj
                o1 = o1 + a1 * wj
            if hf == 1:
                lds_st(red, dp * 2, o0)
                lds_st(red, dp * 2 + 1, o1)
            gpu.barrier()
            if hf == 0:
                lds_st(xs, dp, bf16_pair(o0 + lds_ld(red, dp * 2), o1 + lds_ld(red, dp * 2 + 1)))
            gpu.barrier()
            acc = run_units(u_uv, UV_NKC // UV_WPR, UV_NKC // UV_WPR, pre)
            reduce_rows(UV_R, acc, emit_out(UV_TILE))
            stamp("uv", tt, 3)
            gpu.barrier()
            if tid < UV_TILE // 4:
                r = tid * 4
                values = [lds_ld(outs, r + j) for j in range(4)]
                if const_expr(attention_output_gate):
                    gate = getf_many([(mb("gate"), s * O_K + t * UV_TILE + r + j) for j in range(4)])
                    values = [bf16_round(values[j]) * _rcp(fx.Float32(1.0) + _exp(-gate[j])) for j in range(4)]
                put_bf(mb("o"), s * O_K + t * UV_TILE + r, values)
            stamp("uv", tt, 4)

        # ====================== 7. W_o + attention TP peer reduce + residual -> a
        # 2 row groups x 32 chunks: 4 waves per row group, 8 chunks each
        r_wo, r_so = _rsrc(w_o), _rsrc(s_o)
        O_NKC = O_K // 64
        O_R = ROW_TILE // 16
        O_WPR = WAVES // O_R
        for t in range(start("o"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("o", t, 0)

            def u_o(c):
                kc = (wave % O_WPR) * (O_NKC // O_WPR) + c
                return unit_attention(
                    r_wo, r_so, t * O_R + wave // O_WPR, kc, O_NKC, O_K, 128, (n_sel() * O_K + kc * 64) // 2
                )

            pre = [u_o(c) for c in range(O_NKC // O_WPR)]
            hint_wait(
                S * N_UV, lambda k: (mb("o"), (k // N_UV) * O_K + (k % N_UV) * UV_TILE + UV_TILE - 1), mark=("o", t)
            )
            stage_attention_output()
            stamp("o", t, 2)
            gpu.barrier()
            acc = run_units(u_o, O_NKC // O_WPR, O_NKC // O_WPR, pre)
            reduce_rows(O_R, acc, emit_out(ROW_TILE))
            stamp("o", t, 3)
            gpu.barrier()

            def resid_h(s, row):
                if const_expr(not attention_residual):
                    return fx.Float32(0.0), fx.Float32(0.0)
                w = fx.Vector.from_elements(
                    [fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32))], fx.Int32
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            def store_attention(s, row, v0, v1):
                put_bf(mb("a"), s * HIDDEN + row, [v0, v1])
                if const_expr(attention_only):
                    bo.buffer_store(
                        fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                        _rsrc(x_out),
                        s * HIDDEN + row,
                    )

            peer_reduce(
                "attn",
                t,
                resid_h,
                store_attention,
            )
            stamp("o", t, 4)

        if const_expr(attention_only):
            return

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # One sample per CTA: 1 row group x 96 chunks (bf16), 8 waves split K
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt % N_ROUTER
            router_sample = tt // N_ROUTER
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
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, (r_ns * HIDDEN + kc * 64) // 2, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (mb("a"), router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
                mark=("router", tt),
            )
            # This task's expert-activation block inputs ride along with the staging
            # loads. MXFP8 uses four independent 16-lane groups per wave.
            r_gp = _rsrc(g_post)
            if const_expr(use_mxfp8_block32):
                x_blk = (wave * N_ROUTER + t) * 4 + lane // 16
                xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 32 + lane % 16 * 2
            else:
                x_blk = wave * N_ROUTER + t
                xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 128 + lane * 2
            x_s = router_sample
            x_ok = (wave < XQ_WAVES) & (x_blk < PUBLISH_BLOCKS)
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
            # This task's normalized expert input goes out ahead of the gate GEMV.
            if x_ok:
                x_rstd = rstds[0]
                a0, a1 = xa[0]
                v0, v1 = a0 * x_rstd * xg[0], a1 * x_rstd * xg[1]
                if const_expr(use_fp8_block128):
                    q0, q1, qs = quant_scaled(v0, v1)
                    w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                    w8n = _xshfl(w8, 1)
                    if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                        put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                    d0, d1 = _fp8_roundtrip(q0, q1)
                    d0, d1 = d0 * qs, d1 * qs
                    if lane == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                elif const_expr(use_mxfp8_block32):
                    d0, d1, qs = quant_mxfp8(v0, v1)
                    w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, d0, d1, fx.Int32(0), False)) & 0xFFFF
                    w8n = _xshfl(w8, 1)
                    if lane % 2 == 0:
                        put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                    if lane % 16 == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                    d0, d1 = d0 * qs, d1 * qs
                else:
                    d0, d1 = bf16_round(v0), bf16_round(v1)
                    put(mb("xq"), (x_s * HIDDEN + xk) // 2, bf16_pair(d0, d1))
                bo.buffer_store(fx.Vector.from_elements([d0, d1], fx.Float32), _rsrc(mb("xqd")), x_s * HIDDEN + xk)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
            gpu.barrier()
            stamp("router", tt, 3)
            if tid < ROUTER_TILE:
                r = tid % ROUTER_TILE
                n = fx.Int32(0)
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(red, (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4)
                put(mb("scores"), router_sample * N_EXPERTS + t * ROUTER_TILE + r, _rcp(1.0 + _exp(-logit)))
            stamp("router", tt, 4)

        def dn_route(bs):
            """Expert-down routing (wave s -> sample s): expert ids -> keys[s * 9 + slot],
            route weights -> dnw[]; the scores must have landed."""
            if wave < S:
                e, w = route_top8(wave, bs=bs)
                if lane < MOE_SLOTS:  # slot 0: the shared expert, then pick lane (slot lane + 1)
                    q = wave * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                    lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e))
                    lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ================================ 9. expert up/gate + SiLU
        # One 16-row group (8 gate + 8 up rows), with all eight waves splitting K.
        UG_NKC = HIDDEN // 64
        UG_UNIT_K = 128 if (use_fp8_block128 or use_mxfp4_weight) else 64
        UG_W_BYTES = 2 * INTER * HIDDEN // (2 if use_mxfp4_weight else 1)
        UG_S_BYTES = 2 * INTER * (HIDDEN // 32) if use_mxfp4_weight else 2 * INTER // SCALE_BM * (HIDDEN // 128) * 4

        if const_expr(S == 1):
            # one task per CTA: task u takes intermediates (u % 32) * 8 of routed slot
            # u // 32 (slot 8 for u < 32, which also take the shared expert's); the 8 gate
            # + 8 up rows are one MFMA row group and all waves split K
            UG8 = 8
            UG8_UNITS = (HIDDEN // UG_UNIT_K) // WAVES
            for u in range(start("ug"), G, G):
                u = fx.Int32(u)
                stamp("ug", u, 0)
                s_u, c = fx.Int32(0), u % (INTER // UG8)
                has_sh = u < INTER // UG8
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (INTER // UG8))
                # The expert activation is recomputed here in parallel with the router.
                # MXFP8 uses the four independent 16-lane groups in each wave.
                if const_expr(use_mxfp8_block32):
                    NB = XQ_BLOCKS // (WAVES * 4)
                    ks_ = [((wave + j * WAVES) * 4 + lane // 16) * 32 + lane % 16 * 2 for j in range(NB)]
                else:
                    NB = PUBLISH_BLOCKS // WAVES
                    ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                r_gp = _rsrc(g_post)
                gps = [(ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_]  # issued ahead of the wait
                bs = load_bias()
                w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2  # MFMA rows 0-7 gate, 8-15 up
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                s_rg = (lane // 32) * (INTER // 16) + c // 2  # this lane's output rows

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
                        coefficients = None
                        if const_expr(use_mxfp8_block32):
                            coefficients = []
                            for sp in range_constexpr(4):

                                def coefficient(sp=sp, unit=unit):
                                    return _uniform_f32(lds_ld(misc, 8 + unit * 4 + sp))

                                coefficients.append(coefficient)
                        return unit_mxfp4(
                            r_wug,
                            r_sug,
                            w_rg,
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
                            lambda: sc * _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                            kc * 16 + (lane // 16) * 4,
                        )
                    return ("fp8", wv, sc, kc * 32 + (lane // 16) * 4)

                # the shared expert's weights do not depend on routing: prefetch them (the
                # later zero-weight MMAs of the other tasks are cheaper than a branch)
                pre = [u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh) for cc in range(UG8_UNITS)]
                hint_wait(N_ROW_TILES, lambda k: (mb("a"), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1), mark=("ug", u))
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
                    v0, v1 = av[j][0] * rstd * gps[j][0], av[j][1] * rstd * gps[j][1]
                    if const_expr(use_fp8_block128):
                        q0, q1, qs = quant_scaled(v0, v1)
                        st_f8(ks_[j], q0, q1)
                        if lane == 0:
                            lds_st(misc, 8 + wave + j * WAVES, qs)
                    elif const_expr(use_mxfp8_block32):
                        d0, d1, qs = quant_mxfp8(v0, v1)
                        lds_st(xs, ks_[j] // 2, bf16_pair(d0, d1))
                        if lane % 16 == 0:
                            block = (wave + j * WAVES) * 4 + lane // 16
                            lds_st(misc, 8 + block, qs)
                    else:
                        lds_st(xs, ks_[j] // 2, bf16_pair(v0, v1))
                if wave == 0:
                    e, w = route_top8(s_u, bs=bs)
                    if lane == slot - 1:
                        lds_st(keys, 0, e)
                        lds_st(misc, 0, w)
                stamp("ug", u, 2)
                gpu.barrier()
                e_sel = _uniform(lds_ld(keys, 0))
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
        elif const_expr(S > 1):
            # One eight-intermediate tile per CTA and sample. Shared weights use
            # the sample columns of one MFMA; routed tiles pipeline over samples.
            UG8 = 8
            UG8_UNITS = (HIDDEN // UG_UNIT_K) // WAVES
            XW = HIDDEN // (4 if use_fp8_block128 else 2)
            u = fx.Int32(start("ug"))
            c = u % (INTER // UG8)
            has_sh = u < INTER // UG8
            slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (INTER // UG8))
            w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2
            w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
            s_rg = (lane // 32) * (INTER // 16) + c // 2

            def ug8_units(e, sample, live=None):
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
                        if const_expr(use_mxfp8_block32):
                            coefficients = []
                            for sp in range_constexpr(4):

                                def coefficient(sp=sp, unit=unit, sn=sn):
                                    return lds_ld(misc, 8 + sn * XQ_BLOCKS + unit * 4 + sp)

                                coefficients.append(coefficient)
                        units.append(
                            unit_mxfp4(
                                rw,
                                rs,
                                w_rg,
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

                    # Bind each chunk's operands; the deferred scale follows staging.
                    if const_expr(use_fp8_block128):

                        def coefficient(sc=sc, kb=kc // 2, sn=sn):
                            return sc * lds_ld(misc, 8 + sn * XQ_BLOCKS + kb)

                        units.append(("f8f8", wv, coefficient, sn * XW + kc * 16 + (lane // 16) * 4))
                    else:
                        units.append(("fp8", wv, sc, sn * XW + kc * 32 + (lane // 16) * 4))
                return units

            def ug8_emit(sample, shared):
                if tid < (S if shared else 1) * UG8 // 2:
                    n = tid // (UG8 // 2)
                    r = (tid % (UG8 // 2)) * 2
                    g0, g1 = lds_ld(outs, n * 16 + r), lds_ld(outs, n * 16 + r + 1)
                    v0, v1 = lds_ld(outs, n * 16 + UG8 + r), lds_ld(outs, n * 16 + UG8 + r + 1)
                    sn = n if shared else fx.Int32(sample)
                    sl = fx.Int32(0) if shared else slot
                    put2(
                        mb("mid"),
                        (sn * MOE_SLOTS + sl) * INTER + c * UG8 + r,
                        g0 * _rcp(1.0 + _exp(-g0)) * v0,
                        g1 * _rcp(1.0 + _exp(-g1)) * v1,
                    )
                if (c == 0) & (tid < S if shared else tid == 0):
                    sn = tid if shared else fx.Int32(sample)
                    sl = fx.Int32(0) if shared else slot
                    put(mb("sel"), sn * MOE_SLOTS + sl, lds_ld(keys, sn * MOE_SLOTS + sl))
                    put(mb("prob"), sn * MOE_SLOTS + sl, lds_ld(dnw, sn * MOE_SLOTS + sl))

            shared_pre = ug8_units(fx.Int32(SHARED_EXPERT), None, has_sh)
            dn_route(load_bias())
            gpu.barrier()
            cur = ug8_units(_uniform(lds_ld(keys, slot)), 0)
            stage_moe_input(list(range(S)))
            gpu.barrier()
            if has_sh:
                reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre), emit_out(16))
                gpu.barrier()
                ug8_emit(0, True)
            for sample in range_constexpr(S):
                stamp("ug", sample * G + u, 0)
                pre = cur
                if const_expr(sample + 1 < S):
                    cur = ug8_units(_uniform(lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)), sample + 1)
                reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16))
                gpu.barrier()
                ug8_emit(sample, False)
                stamp("ug", sample * G + u, 4)

        # =============== 10. expert down + route weighting + MoE TP reduce
        DN_NKC = INTER // 64
        DN_R = (DN_TILE + 15) // 16  # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
        DN_WPR = WAVES // DN_R
        DN_UNIT_K = 128 if (use_fp8_block128 or use_mxfp4_weight) else 64
        DN_UNITS_PER_SLOT = INTER // DN_UNIT_K
        DN_NU = S * MOE_SLOTS * DN_UNITS_PER_SLOT
        DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
        DN_BLK = S * MOE_SLOTS * INTER // 128
        DN_W_BYTES = HIDDEN * INTER // (2 if use_mxfp4_weight else 1)
        DN_S_BYTES = HIDDEN * (INTER // 32) if use_mxfp4_weight else HIDDEN // SCALE_BM * (INTER // 128) * 4
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
            # this lane's row, as a tile row; rows outside the tile load their lane ^ 8 twin
            # (same cache lines) and are dropped in the output
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            def u_dn(cc):  # cc: 128-k chunk of this wave
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                q = fx.min(qu, DN_NU - 1)  # unit index over (sample, slot, K chunk)
                s_q = q // (MOE_SLOTS * DN_UNITS_PER_SLOT)
                slot_q = (q // DN_UNITS_PER_SLOT) % MOE_SLOTS
                kc = q % DN_UNITS_PER_SLOT
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0)),
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                    num_records_bytes=None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0)),
                )

                if const_expr(use_mxfp4_weight):
                    coefficients = []
                    if const_expr(use_mxfp8_block32):
                        for sp in range_constexpr(4):

                            def coefficient(sp=sp, q=q, s_q=s_q):
                                return (lane % 16 == s_q).select(
                                    _uniform_f32(lds_ld(misc, q * 4 + sp)), fx.Float32(0.0)
                                )

                            coefficients.append(coefficient)
                    else:

                        def coefficient():
                            return (lane % 16 == s_q).select(
                                _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)), fx.Float32(0.0)
                            )

                        coefficients = coefficient
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
                        return (lane % 16 == s_q).select(_uniform_f32(lds_ld(misc, q)), fx.Float32(0.0))

                    return unit_f8f8(wb, sb, dn_rg + gu, kc64, DN_NKC, INTER, q * 32, coef, dn_ln)

                def coef():
                    return (lane % 16 == s_q).select(_uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)), 0.0)

                return unit_fp8(wb, sb, dn_rg + gu, kc64, DN_NKC, INTER, 128, q * 32, coef, dn_ln)

            # the experts are known: stream their down weights while up/gate finishes
            pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
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
            # Stage expert intermediates in the activation format selected for this mode.
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
                        lds_st(xs, blk * 64 + lane, bf16_pair(d0, d1))
                        if lane % 16 == 0:
                            scale_group = blk * 4 + lane // 16
                            lds_st(misc, scale_group, qs * lds_ld(dnw, blk // (INTER // 128)))
                    else:
                        lds_st(xs, blk * 64 + lane, bf16_pair(mids[b][0], mids[b][1]))
            gpu.barrier()
            acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            reduce_rows(DN_R, acc, emit_dn)
            stamp("down", t, 3)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16), _rsrc(x_out), s * HIDDEN + row
                )

            peer_reduce("ffn", t, mb("a"), store_x, tile=DN_TILE)
            gpu.barrier()
            stamp("down", t, 4)

    @flyc.jit
    def launch_shared_reuse(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
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
        shared_reuse_kernel(
            h_in,
            x_out,
            cur_pos,
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
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch_shared_reuse
