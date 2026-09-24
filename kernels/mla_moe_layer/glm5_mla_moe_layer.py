# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GLM-5 shared/reuse MoE layer in ONE persistent launch per rank (TP8 decode).

One launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs the
whole layer body for this rank's TP shard::

    input RMSNorm -> q_a / kv_a projection -> q_a RMSNorm -> q_b (+RoPE)
      -> KV RMSNorm / k_pe RoPE -> KV/PE cache publish
      -> absorbed q (W_UK) -> sparse MLA split softmax -> merge -> W_UV -> W_o
      -> attention TP8 peer reduce + residual                      (sym_attn)
      -> post-attention RMSNorm -> router sigmoid + activation FP8 quant
      -> top-8 -> 1 shared + 8 routed expert up/gate/SiLU
      -> mid FP8 quant -> expert down + route weighting
      -> MoE TP8 peer reduce + residual -> x_out                   (sym_ffn)

Scheduling: every stage is a list of tasks; task ``t`` of a stage runs on CTA
``(stage_base + t) % 256`` and every CTA walks the stages in order.  There is
no grid-wide barrier: dependencies only point to earlier stages and all CTAs
are co-resident, so every spin wait makes progress.

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

Cross-GPU: each rank pushes its partial rows as tagged pairs into every peer's
symmetric buffer and polls its own; every rank sums the 8 partials in rank
order, so all ranks produce bit-identical hidden states (and routing).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value
from kernels.common import buffer_ops as bo
from kernels.common.dpp_utils import update_dpp_i32
from kernels.mla_moe_layer.reference import (
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
    QKV_A_ROWS,
    ROUTE_SCALE,
    SCALE_BM,
    SHARED_EXPERT,
    SOFTMAX_SCALE,
    TOP_K,
    V_DIM,
)

BLOCKS = 256
LAYER_SLOTS = 128  # max layers per step sharing one scratch / symmetric buffer
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

# task counts per stage
N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE


def dn_tile(S: int) -> int:
    """Hidden rows per expert-down / FFN peer-reduce task: 64 at S = 1 so up/gate and
    down tasks fit on distinct CTAs (down prefetches while up/gate runs); 24 above
    (one task per CTA), where each down task already streams S x 9 experts."""
    return 64 if S == 1 else HIDDEN // BLOCKS


N_ROUTER = N_EXPERTS // ROUTER_TILE
N_UG_PER_SLOT = INTER // UG_TILE


def ug_split(S: int):
    """Balanced S in (2, 4) up/gate schedule, or None: the shared expert's tiles are computed
    once for all samples, so there are (S * TOP_K + 1) * 16 tiles; every CTA runs
    ``nf`` whole tiles plus one of ``seg`` K-segments of a leftover tile.  Returns
    (nf, seg) when the leftover tiles split evenly over the CTAs."""
    nf, r = divmod((S * TOP_K + 1) * N_UG_PER_SLOT, BLOCKS)
    if S not in (2, 4) or r == 0 or BLOCKS % r:
        return None
    seg = BLOCKS // r
    if (HIDDEN // 128) % seg or (HIDDEN // 128) // seg > WAVES // 2:
        return None
    return nf, seg


XQ_BLOCKS = HIDDEN // 128  # MoE activation quant blocks
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER  # router task t quantizes blocks t, t + N_ROUTER, ..
assert XQ_WAVES * 4 <= WAVES  # one (block, sample) per wave up to S = 4

# gfx94x/95x cache policy bits (LLVM CPol): SC0 = 1, NT = 2, SC1 = 16.  SC1:SC0 is
# the coherence scope of the access itself: SC1 = device (past the per-XCD
# non-coherent caches), SC0|SC1 = system (peer GPUs over XGMI).
CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12  # mailbox specs polled per batch
TL_COLS = 8  # timeline stamps per task: 5 phases + 3 free debug marks


def _align(n, a=256):
    return (n + a - 1) // a * a


def layout(S: int, heads: int, npes: int, topk: int):
    """Byte offsets of the per-rank scratch and of the symmetric buffer.

    Every mailbox holds ``(value, tag)`` int32 pairs (8 bytes per element)."""
    n_split = topk // SPLIT_KEYS
    pr = 8
    items = [
        ("q_a", S * Q_LORA * pr),
        ("kv_a", S * (KV_LORA + PE_DIM) * pr),
        ("kvnew", S * KV_LORA * pr),  # this launch's KV cache rows (bf16 values)
        ("penew", S * PE_DIM * pr),
        ("q_nope", S * heads * NOPE_DIM * pr),
        ("q_pe", S * heads * PE_DIM * pr),
        ("q_lat", S * heads * KV_LORA * pr),
        ("sp_acc", S * n_split * heads * KV_LORA * pr),
        ("sp_m", S * n_split * heads * pr),
        ("sp_l", S * n_split * heads * pr),
        ("o", S * heads * V_DIM * pr),
        ("a", S * HIDDEN * pr),  # post-attention hidden (bf16 values)
        ("scores", S * N_EXPERTS * pr),
        ("xq", S * HIDDEN // 4 * pr),  # FP8-quantized MoE activation (4 packed FP8 per pair)
        ("xqs", S * XQ_BLOCKS * pr),  # its per-128 block scales
        ("sel", S * MOE_SLOTS * pr),
        ("prob", S * MOE_SLOTS * pr),
        ("mid", S * MOE_SLOTS * INTER * pr),
        ("ugp", BLOCKS * S * 2 * UG_TILE * pr),  # up/gate K-segment partial sums
        ("xqd", S * HIDDEN * 4),  # debug: dequantized MoE activation (plain f32)
    ]
    off, scratch = 0, {}
    for name, size in items:
        scratch[name] = off
        off += _align(size)
    scratch["_bytes"] = off
    part = npes * S * HIDDEN * pr
    sym = {"attn": 0, "ffn": part, "_bytes": 2 * part}
    return scratch, sym


def pack_fp8(q: torch.Tensor) -> torch.Tensor:
    """FP8 ``[..., N, K]`` -> MFMA-native ``[..., N/16, K/64, 64 lanes, 16 B]``.

    Lane ``l`` of a 64-k chunk holds row ``l % 16``, k ``sp*32 + (l//16)*8 + i``
    for sp in (0, 1), i < 8: the A operands of two ``mfma_f32_16x16x32_bf16``.
    """
    *lead, N, K = q.shape
    w8 = q.view(torch.uint8).reshape(*lead, N // 16, 16, K // 64, 2, 4, 8)  # rg r kc sp lg i
    nl = len(lead)
    perm = list(range(nl)) + [nl + p for p in (0, 2, 4, 1, 3, 5)]  # rg kc lg r sp i
    return w8.permute(*perm).contiguous().view(-1)


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """bf16 ``[N, K]`` -> ``[N/16, K/64, 2 (sp), 64 lanes, 8 bf16]``."""
    N, K = w.shape
    w16 = w.view(torch.int16).reshape(N // 16, 16, K // 64, 2, 4, 8)  # rg r kc sp lg i
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


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


_UMAX_ASM = "\n".join(
    ["s_nop 1\nv_max_u32_dpp $0, $0, $0 " + c for c in (
        "row_shr:1 bound_ctrl:0",
        "row_shr:2 bound_ctrl:0",
        "row_shr:4 bound_ctrl:0",
        "row_shr:8 bound_ctrl:0",
        "row_bcast:15 row_mask:0xa",
        "row_bcast:31 row_mask:0xc",
    )]

)


def _wave_umax(v):
    """Unsigned max over the (fully active) wave as a wave-uniform Int32: fused DPP
    max steps (out-of-row sources read 0) leave it in lane 63, read into an SGPR."""
    x = llvm.InlineAsmOp(T.i32, [fx.Int32(v).ir_value()], _UMAX_ASM, "=v,0").result
    return fx.Int32(rocdl.readlane(T.i32, x, fx.Int32(63).ir_value()))


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


def stage_tasks(S: int, heads: int, topk: int):
    """[(stage name, task count)] in execution order."""
    return [
        ("qkv_a", N_QKV_A),
        ("cache", 1),
        ("q_b", heads * (NOPE_DIM + PE_DIM) // Q_B_TILE),
        ("uk", heads * KV_LORA // UK_TILE),
        ("split", S * (topk // SPLIT_KEYS)),
        ("uv", S * (heads * V_DIM // UV_TILE)),
        ("o", N_ROW_TILES),
        ("router", N_ROUTER),
        ("ug", S * MOE_SLOTS * N_UG_PER_SLOT if ug_split(S) is None else (ug_split(S)[0] + 1) * BLOCKS),
        ("down", HIDDEN // dn_tile(S)),
    ]


def build_layer(
    S: int = 1, heads: int = 8, npes: int = 8, topk: int = 2048, scale: float = SOFTMAX_SCALE, timeline: bool = False
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer.

    ``timeline=True`` records ``s_memrealtime`` (100 MHz) at the start and end of
    every task, and once its inputs have arrived, into the ``timeline`` buffer:
    int64 ``[sum(task counts), TL_COLS]`` (start, hint seen, inputs staged, compute
    done, end, then free debug marks) in ``stage_tasks`` order.
    """
    assert heads == 8, "the split-attention mapping uses one wave per local head"
    assert topk % SPLIT_KEYS == 0 and 1 <= S <= 4
    H = heads
    W = npes
    G = BLOCKS
    SC, SY = layout(S, H, W, topk)
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
    DN_TILE = dn_tile(S)
    N_DN_TILES = HIDDEN // DN_TILE

    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(S, H, topk):
        first[name] = acc
        acc += n
    # CTA placement: split before uk, so every split tile lands on a CTA freed by
    # qkv_a (uk shares the q_b CTAs it waits on anyway)
    tasks = dict(stage_tasks(S, H, topk))
    acc = 0
    for name in ("qkv_a", "cache", "q_b", "split", "uk", "uv", "o", "router", "ug", "down"):
        base[name] = acc % G
        acc += tasks[name]

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XN, 16]  # bf16 activations (pairs) / split q + KV tile
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, 8 + S * XQ_BLOCKS, 16]
        p: fx.Array[fx.Float32, H * SPLIT_KEYS, 16]
        keys: fx.Array[fx.Int32, SPLIT_KEYS, 16]
        dnw: fx.Array[fx.Float32, S * MOE_SLOTS, 16]  # expert-down route weights

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def layer_kernel(
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
        # wave-uniform peer bases: buffer descriptors must live in SGPRs
        peer_addr = []
        for p in range_constexpr(W):
            pv = fx.Vector(bo.buffer_load(r_peers, p * 2, vec_width=2, dtype=T.i32))
            lo = fx.Int64(fx.Uint32(_uniform(pv[0])))
            hi = fx.Int64(_uniform(pv[1]))
            peer_addr.append((hi << 32) | lo)

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
                llvm.InlineAsmOp(None, [], "s_nop 0", "", has_side_effects=True)
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
        def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None):
            """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
            bf16 activation chunk starts at LDS word ``b_word``."""
            wv = fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc) * 64 + lane) * 4, vec_width=4, dtype=T.i32))
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

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4, vec_width=4, dtype=T.i32)
                )
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for fmt, wv, coef, bw in units:
                if const_expr(callable(coef)):
                    coef = coef()
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                if const_expr(fmt == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                    a = fx.Vector.from_elements([wv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                    bv = [
                        fx.Vector(fx.ptr_load(xs + (bw + h * 16), result_type=v4f)).bitcast(fx.Int32) for h in range(2)
                    ]
                    b = fx.Vector.from_elements([bv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                    one = fx.Int32(127)
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one])
                    )
                for sp in range_constexpr(2 if fmt != "f8f8" else 0):
                    if const_expr(fmt == "fp8"):
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

        def stage_x_rmsnorm(ld4s, n, gamma, mark=None, loaded=None):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
            ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
            ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma)
            sss = []
            for s in range_constexpr(S):
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
            for s in range_constexpr(S):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (s * n + ks[i]) // 2 + j,
                            bf16_pair(a[2 * j] * rstds[s] * gs[i][2 * j], a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1]),
                        )
            return rstds

        def load_x_rmsnorm(ld4s, n, gamma):
            """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
            rg_ = _rsrc(gamma)
            ks = [(tid + i * THREADS) * 4 for i in range(n // (4 * THREADS))]
            gs = []
            for k in ks:
                g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(S) for k in ks])

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
            inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))  # hardware rcp, no IEEE divide
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, qs

        def quant_block(a0, a1):
            """quant_scaled, values returned as the FP8-rounded f32s."""
            q0, q1, qs = quant_scaled(a0, a1)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, qs

        def stage_xq(samples):
            """Poll the router's packed FP8 activation + block scales of ``samples``
            (sample list, or one runtime sample) into LDS words s * HIDDEN / 4 (``f8_word``
            order; slot 0 for a single runtime sample) and misc[8 + s * XQ_BLOCKS:]."""
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
                    lds_st(misc, 8 + j * XQ_BLOCKS + tid, got[len(samples) * nxw + j][0].bitcast(fx.Float32))

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
                        T.i32, "llvm.amdgcn.writelane.i32", [m.ir_value(), fx.Int32(k).ir_value(), mv.ir_value()], [], []
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
            return e, raw / tot * ROUTE_SCALE

        def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
            """Push outs[s * tile + r] as tagged pairs to every peer, then sum all
            ranks' pairs from the own symmetric buffer in rank order (W = 1: no exchange).  ``residual`` is
            either fn(s, row) -> (r0, r1) (plain loads, issued first) or a mailbox base
            (pairs s * HIDDEN + row, polled in the same batch as the peers)."""
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
                    for p in range_constexpr(W):
                        put2(
                            peer_addr[p] + fx.Int64(SY[region]), (rank * S + s) * HIDDEN + t * tile + r, v0, v1, CM_SYS
                        )
                    own = sym + fx.Int64(SY[region])
                    specs = [(own, (src * S + s) * HIDDEN + row, 2) for src in range(W)]
                    if const_expr(not callable(residual)):  # packed bf16 pair
                        specs.append((residual, (s * HIDDEN + row) // 2, 1))
                    got = poll(specs, "one-as")
                    parts = [(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in got[:W]]
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
            return (bid + (G - base[name])) % G

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
                return unit_fp8(r_wqa, r_sqa, t, kc, QA_NKC, HIDDEN, 128, (n_sel() * HIDDEN + kc * 64) // 2)

            def ld_h(sks):
                res = []
                for s, k in sks:
                    w = fx.Vector(bo.buffer_load(r_h, (s * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            # the (small) input loads go out before the weight stream: loads complete in order
            h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in)
            pre = [u_qa(c) for c in range(QA_NKC // WAVES)]
            stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld)
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
                else:
                    put(mb("kv_a"), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)
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
                return unit_fp8(r_wqb, r_sqb, t, kc, QB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

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
                return unit_fp8(
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

        for tt in range(start("split"), S * N_SPLIT, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s = tt // N_SPLIT  # sample
            t = tt % N_SPLIT  # 64-key chunk
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
            hn = fx.min(lane % 16, H - 1)
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
            # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
            kidx = t * SPLIT_KEYS + lane
            valid = kidx < nkeys
            r16 = lane % 16
            cl = h + 16 * (r16 // 4)
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
                        hh = (lane // 16) * 4 + e
                        put_bf(mb("sp_acc"), ((s * N_SPLIT + t) * H + hh) * KV_LORA + dw * 2, [c0[e], c1[e]])
            if lane == 0:  # written last: the merge's readiness hint
                put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
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
                return unit_fp8(r_wuv, r_suv, t * UV_R + wave // UV_WPR, kc, UV_NKC, KV_LORA, 128, (kc * 64) // 2)

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
                    lds_st(misc, lane, w_sp / den)
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
                put_bf(mb("o"), s * O_K + t * UV_TILE + r, [lds_ld(outs, r + j) for j in range(4)])
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
                return unit_fp8(
                    r_wo, r_so, t * O_R + wave // O_WPR, kc, O_NKC, O_K, 128, (n_sel() * O_K + kc * 64) // 2
                )

            pre = [u_o(c) for c in range(O_NKC // O_WPR)]
            hint_wait(
                S * N_UV, lambda k: (mb("o"), (k // N_UV) * O_K + (k % N_UV) * UV_TILE + UV_TILE - 1), mark=("o", t)
            )
            stage_x_pairs("o", S * O_K, lambda k: k)
            stamp("o", t, 2)
            gpu.barrier()
            acc = run_units(u_o, O_NKC // O_WPR, O_NKC // O_WPR, pre)
            reduce_rows(O_R, acc, emit_out(ROW_TILE))
            stamp("o", t, 3)
            gpu.barrier()

            def resid_h(s, row):
                w = fx.Vector.from_elements(
                    [fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32))], fx.Int32
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            peer_reduce(
                "attn",
                t,
                resid_h,
                lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
            )
            stamp("o", t, 4)

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # 1 row group x 96 chunks (bf16): 8 waves split K
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for t in range(start("router"), N_ROUTER, G):
            t = fx.Int32(t)
            stamp("router", t, 0)

            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.min(lane % ROUTER_TILE, S - 1)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, (r_ns * HIDDEN + kc * 64) // 2, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES, lambda k: (mb("a"), (S - 1) * HIDDEN + k * ROW_TILE + ROW_TILE - 1), mark=("router", t)
            )
            # this task's FP8 activation block inputs ride along with the staging loads:
            # wave w quantizes block (w // S) * N_ROUTER + t of sample w % S
            r_gp = _rsrc(g_post)
            x_blk = wave // S * N_ROUTER + t
            x_s = wave % S
            x_ok = (wave < XQ_WAVES * S) & (x_blk < XQ_BLOCKS)
            xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                specs = [(mb("a"), (s * HIDDEN + k) // 2, 2) for s, k in sks]
                specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                v = poll(specs, batch=len(specs))
                stamp("router", t, 5, lead=THREADS - 64)
                xa.append(bf2_f32(v[-1][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, mark=("router", t))
            stamp("router", t, 2)
            # this task's FP8 activation blocks go out ahead of the gate GEMV
            if x_ok:
                x_rstd = rstds[0]
                for s in range_constexpr(1, S):
                    x_rstd = (x_s == s).select(rstds[s], x_rstd)
                a0, a1 = xa[0]
                q0, q1, qs = quant_scaled(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                w8n = _xshfl(w8, 1)
                if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                    put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                d0, d1 = _fp8_roundtrip(q0, q1)
                bo.buffer_store(
                    fx.Vector.from_elements([d0 * qs, d1 * qs], fx.Float32), _rsrc(mb("xqd")), x_s * HIDDEN + xk
                )
                if lane == 0:
                    put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
            gpu.barrier()
            stamp("router", t, 3)
            if tid < S * ROUTER_TILE:
                r = tid % ROUTER_TILE
                n = tid // ROUTER_TILE
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(red, (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4)
                put(mb("scores"), n * N_EXPERTS + t * ROUTER_TILE + r, _rcp(1.0 + _exp(-logit)))
            stamp("router", t, 4)

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
        # 2 row groups (16 gate + 16 up rows) x 96 chunks: 4 waves per group, 24 chunks each
        UG_NKC = HIDDEN // 64
        UG_CPW = UG_NKC // (WAVES // 2)
        UG_W_BYTES = 2 * INTER * HIDDEN
        UG_S_BYTES = 2 * INTER // SCALE_BM * (HIDDEN // 128) * 4

        def ug_units(e_sel, c, live=None):
            """Unit maker of up/gate tile c of expert e_sel; ``live`` False -> empty
            buffers (loads return 0 without memory traffic)."""
            if const_expr(live is None):
                r_wug = _rsrc(w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES))
                r_sug = _rsrc(s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES))
            else:
                r_wug = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES),
                    num_records_bytes=live.select(fx.Int32(UG_W_BYTES), fx.Int32(0)),
                )
                r_sug = bo.create_buffer_resource_from_addr(
                    s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES),
                    num_records_bytes=live.select(fx.Int32(UG_S_BYTES), fx.Int32(0)),
                )
            gate_up = wave // (WAVES // 2)  # waves 0-3: gate rows, 4-7: up rows

            def u_ug(cc):  # cc: 128-k chunk of this wave
                kc = (wave % (WAVES // 2)) * UG_CPW + cc * 2
                return unit_f8f8(
                    r_wug,
                    r_sug,
                    gate_up * (INTER // 16) + c,
                    kc,
                    UG_NKC,
                    HIDDEN,
                    kc * 16,
                    lambda: _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                )

            return u_ug

        def ug_finish(u, s_u, slot, c, e_sel, prob, u_ug, pre):
            """MFMA the staged activation (X, scales in misc[8:]) against the tile,
            SiLU(gate) * up -> mid."""
            acc = run_units(u_ug, UG_CPW // 2, UG_CPW // 2, pre)
            reduce_rows(2, acc, emit_out(UG_TILE * 2))
            stamp("ug", u, 3)
            gpu.barrier()
            if tid < UG_TILE // 2:
                r = tid * 2
                g0, g1 = lds_ld(outs, r), lds_ld(outs, r + 1)
                u0, u1 = lds_ld(outs, UG_TILE + r), lds_ld(outs, UG_TILE + r + 1)
                put2(
                    mb("mid"),
                    (s_u * MOE_SLOTS + slot) * INTER + c * UG_TILE + r,
                    g0 * _rcp(1.0 + _exp(-g0)) * u0,
                    g1 * _rcp(1.0 + _exp(-g1)) * u1,
                )
            if (c == 0) & (tid == 0):  # routing record (debug / tests)
                put(mb("sel"), s_u * MOE_SLOTS + slot, e_sel)
                put(mb("prob"), s_u * MOE_SLOTS + slot, prob())
            stamp("ug", u, 4)

        def ug_task(u):
            s_u = u // (MOE_SLOTS * N_UG_PER_SLOT)
            return s_u, (u // N_UG_PER_SLOT) % MOE_SLOTS, u % N_UG_PER_SLOT

        if const_expr(S == 1):
            for u in range(start("ug"), N_UG, G):
                u = fx.Int32(u)
                stamp("ug", u, 0)
                s_u, slot, c = ug_task(u)
                # the FP8 activation is computed here from the post-attention state (in
                # parallel with the router): RMSNorm, then per-128 quant with one wave per
                # block -> X[0] (fp8 values in bf16), block scales -> misc[8:]
                NB = XQ_BLOCKS // WAVES
                ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                r_gp = _rsrc(g_post)
                gps = [(ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_]  # issued ahead of the wait
                bs = load_bias()
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
                    q0, q1, qs = quant_scaled(av[j][0] * rstd * gps[j][0], av[j][1] * rstd * gps[j][1])
                    st_f8(ks_[j], q0, q1)
                    if lane == 0:
                        lds_st(misc, 8 + wave + j * WAVES, qs)
                if tid == 0:  # slot 0: the shared expert, weight 1 (does not wait for routing)
                    lds_st(keys, 0, fx.Int32(SHARED_EXPERT))
                    lds_st(misc, 0, fx.Float32(1.0))
                if (slot > 0) & (wave == 0):
                    e, w = route_top8(s_u, bs=bs)
                    if lane == slot - 1:
                        lds_st(keys, 0, e)
                        lds_st(misc, 0, w)
                stamp("ug", u, 2)
                gpu.barrier()
                e_sel = _uniform(lds_ld(keys, 0))
                ug_finish(u, s_u, slot, c, e_sel, lambda: lds_ld(misc, 0), ug_units(e_sel, c), None)
        elif const_expr(ug_split(S) is not None):
            # S = 2, 4 (the router already quantized every sample's activation): job 0 is
            # this CTA's K-segment of a leftover tile (partial sums -> ugp; the segment-0
            # CTA sums them after its own tiles), jobs 1..NF whole tiles.  Tile x: expert
            # slot x // 16 (0 = the shared expert with MFMA column n = sample n, then
            # sample-major routed slots), intermediates (x % 16) * 16.  Job k + 1 is
            # routed and its weights are in flight while job k computes.
            NF, SEG = ug_split(S)
            KB = XQ_BLOCKS // SEG  # 128-k blocks per segment and row group
            XW = HIDDEN // 4  # LDS words of one sample's FP8 activation
            gu_row = (wave // (WAVES // 2)) * (INTER // 16)  # waves 0-3: gate rows, 4-7: up rows
            wq = wave % (WAVES // 2)
            seg = bid % SEG
            x_seg = NF * G + bid // SEG

            def ug_job(k):
                x = fx.Int32(x_seg if k == 0 else bid + (k - 1) * G)
                es = x // N_UG_PER_SLOT
                c = x % N_UG_PER_SLOT
                shared = es == 0
                s_u = fx.max(es - 1, 0) // TOP_K
                slot = shared.select(fx.Int32(0), (es - 1) % TOP_K + 1)
                e_sel = _uniform(lds_ld(keys, s_u * MOE_SLOTS + slot))  # routed once by dn_route
                prob = lds_ld(dnw, s_u * MOE_SLOTS + slot)
                bsel = shared.select(n_sel(), s_u)  # this lane's activation sample
                if const_expr(k == 0):  # waves wq < KB each own one 128-k block
                    kbs = [seg * KB + fx.min(wq, KB - 1)]
                    nrec = (wq < KB).select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                else:
                    kbs = [wq * (UG_CPW // 2) + cc for cc in range(UG_CPW // 2)]
                    nrec = fx.Int32(UG_W_BYTES)
                r_w = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES), num_records_bytes=nrec
                )
                r_s = _rsrc(s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES))

                def mk(kb):
                    return unit_f8f8(
                        r_w,
                        r_s,
                        gu_row + c,
                        kb * 2,
                        UG_NKC,
                        HIDDEN,
                        bsel * XW + kb * 32,
                        lambda: lds_ld(misc, 8 + bsel * XQ_BLOCKS + kb),
                    )

                return (shared, s_u, slot, c, e_sel, prob, [mk(kb) for kb in kbs])

            def ug_mid(shared, s_u, slot, c, e_sel, prob):
                """outs (per column: 16 gate then 16 up sums) -> SiLU(gate) * up -> mid of
                sample s_u, or of every sample n (column n) for the shared expert."""
                if tid < S * UG_TILE // 2:
                    n = tid // (UG_TILE // 2)
                    r = (tid % (UG_TILE // 2)) * 2
                    if shared | (n == 0):
                        g0, g1 = lds_ld(outs, n * 2 * UG_TILE + r), lds_ld(outs, n * 2 * UG_TILE + r + 1)
                        v0 = lds_ld(outs, n * 2 * UG_TILE + UG_TILE + r)
                        v1 = lds_ld(outs, n * 2 * UG_TILE + UG_TILE + r + 1)
                        put2(
                            mb("mid"),
                            (shared.select(n, s_u) * MOE_SLOTS + slot) * INTER + c * UG_TILE + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * v0,
                            g1 * _rcp(1.0 + _exp(-g1)) * v1,
                        )
                if (c == 0) & (tid < S):  # routing record (debug / tests)
                    if shared | (tid == 0):
                        put(mb("sel"), shared.select(tid, s_u) * MOE_SLOTS + slot, e_sel)
                        put(mb("prob"), shared.select(tid, s_u) * MOE_SLOTS + slot, prob)

            stamp("ug", bid, 5)
            dn_route(load_bias())  # every sample's top-8 (waves < S in parallel), for up/gate and down
            gpu.barrier()
            stamp("ug", bid, 6)
            cur = ug_job(0)
            stamp("ug", bid, 0)
            stage_xq(list(range(S)))
            stamp("ug", bid, 2)
            gpu.barrier()
            job0 = cur[:6]
            for k in range_constexpr(NF + 1):
                shared, s_u, slot, c, e_sel, prob, pre = cur
                if const_expr(k > 0):
                    stamp("ug", k * G + bid, 0)
                if const_expr(k < NF):
                    cur = ug_job(k + 1)
                acc = mma_units([fx.Float32(0.0) for _ in range(4)], pre)
                reduce_rows(2, acc, emit_out(UG_TILE * 2))
                stamp("ug", k * G + bid, 3)
                gpu.barrier()
                if const_expr(k == 0):
                    if tid < S * 2 * UG_TILE:
                        put(mb("ugp"), ((x_seg - NF * G) * SEG + seg) * S * 2 * UG_TILE + tid, lds_ld(outs, tid))
                else:
                    ug_mid(shared, s_u, slot, c, e_sel, prob)
                stamp("ug", k * G + bid, 4)
            if seg == 0:  # sum the leftover tile's K-segments
                gpu.barrier()
                if tid < S * 2 * UG_TILE:
                    parts = getf_many(
                        [((mb("ugp")), ((x_seg - NF * G) * SEG + j) * S * 2 * UG_TILE + tid) for j in range(SEG)]
                    )
                    tot_p = parts[0]
                    for j in range_constexpr(1, SEG):
                        tot_p = tot_p + parts[j]
                    lds_st(outs, tid, tot_p)
                gpu.barrier()
                ug_mid(*job0)
        else:
            # S > 1 (the router already quantized every sample's activation): this CTA's
            # tasks are software pipelined -- task k+1 is routed (by every wave on its
            # own) and its weights are in flight while task k computes
            UG_NT = (N_UG + G - 1) // G
            NSC = N_EXPERTS // 64
            u0 = start("ug")

            def ug_prep(k):
                u = fx.Int32(u0 + k * G)
                live = u < N_UG
                s_u, slot, c = ug_task(fx.min(u, N_UG - 1))
                bs = load_bias()
                raws = getf_many([(mb("scores"), s_u * N_EXPERTS + lane + i * 64) for i in range(NSC)])
                e, w = route_top8(s_u, raws, bs)
                i_pk = fx.max(slot - 1, 0).ir_value()
                e_sel = _uniform((slot == 0).select(fx.Int32(SHARED_EXPERT), fx.Int32(rocdl.readlane(T.i32, e.ir_value(), i_pk))))
                prob = (slot == 0).select(
                    fx.Float32(1.0), fx.Int32(rocdl.readlane(T.i32, w.bitcast(fx.Int32).ir_value(), i_pk)).bitcast(fx.Float32)
                )
                u_ug = ug_units(e_sel, c, live)
                return (u, live, s_u, slot, c, e_sel, prob, u_ug, [u_ug(cc) for cc in range(UG_CPW // 2)])

            cur = ug_prep(0)
            for k in range_constexpr(UG_NT):
                u, live, s_u, slot, c, e_sel, prob, u_ug, pre = cur
                if live:
                    stamp("ug", u, 0)
                    stage_xq([s_u])
                    stamp("ug", u, 2)
                gpu.barrier()
                if const_expr(k + 1 < UG_NT):
                    cur = ug_prep(k + 1)
                if live:
                    ug_finish(u, s_u, slot, c, e_sel, lambda: prob, u_ug, pre)

        # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
        # 2 row groups x (S * 9 slots * 4) chunks: 4 waves per group
        DN_NKC = INTER // 64
        DN_R = (DN_TILE + 15) // 16  # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
        DN_WPR = WAVES // DN_R
        DN_CPW = S * MOE_SLOTS * DN_NKC // DN_WPR
        DN_BLK = S * MOE_SLOTS * INTER // 128
        DN_BATCH = 9 if const_expr(S == 1) else 3  # 128-k chunks in flight per wave (VGPR budget)
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stamp("down", t, 0)
            if const_expr(ug_split(S) is None):  # else routed before up/gate
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
                q = (wave % DN_WPR) * DN_CPW + cc * 2  # 64-k chunk index over (s, slot, kc)
                s_q = q // (MOE_SLOTS * DN_NKC)
                slot_q = (q // DN_NKC) % MOE_SLOTS
                kc = q % DN_NKC
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = _rsrc(w_dn + fx.Int64(e) * fx.Int64(HIDDEN * INTER))
                sb = _rsrc(s_dn + fx.Int64(e) * fx.Int64(HIDDEN // SCALE_BM * (INTER // 128) * 4))

                def coef():  # mid block scale * route weight, only in this sample's column
                    return (lane % 16 == s_q).select(_uniform_f32(lds_ld(misc, q // 2)), fx.Float32(0.0))

                return unit_f8f8(wb, sb, dn_rg + gu, kc, DN_NKC, INTER, q * 16, coef, dn_ln)

            # the experts are known: stream their down weights while up/gate finishes
            pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_CPW // 2))]
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
            # mids -> per-128 FP8 (values in bf16) in X[(s * 9 + slot) * 256 + k];
            # block scale * route weight in misc[block]
            for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    q0, q1, qs = quant_scaled(mids[b][0], mids[b][1])
                    st_f8(blk * 128 + lane * 2, q0, q1)
                    if lane == 0:
                        lds_st(misc, blk, qs * lds_ld(dnw, blk // (INTER // 128)))
            gpu.barrier()
            acc = run_units(u_dn, DN_CPW // 2, DN_BATCH, pre)
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
    def launch(
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
        layer_kernel(
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

    return launch
