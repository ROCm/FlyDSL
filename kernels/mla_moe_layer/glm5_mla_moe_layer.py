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
ROW_TILE = 32  # hidden rows per W_o / down / peer-reduce tile
ROUTER_TILE = 16
UG_TILE = 16  # intermediates per up/gate task (16 gate rows + 16 up rows)
SPLIT_KEYS = 64
NEG = -1.0e30

# task counts per stage
N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
N_UG_PER_SLOT = INTER // UG_TILE
XQ_BLOCKS = HIDDEN // 128  # MoE activation quant blocks
XQ_PER_ROUTER = XQ_BLOCKS // N_ROUTER

# gfx94x/95x cache policy bits (LLVM CPol): SC0 = 1, NT = 2, SC1 = 16.  SC1:SC0 is
# the coherence scope of the access itself: SC1 = device (past the per-XCD
# non-coherent caches), SC0|SC1 = system (peer GPUs over XGMI).
CM_DEV = 16
CM_SYS = 17


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
        ("xq", S * HIDDEN * pr),  # FP8-quantized MoE activation (fp8 values)
        ("xqs", S * XQ_BLOCKS * pr),  # its per-128 block scales
        ("sel", S * MOE_SLOTS * pr),
        ("prob", S * MOE_SLOTS * pr),
        ("mid", S * MOE_SLOTS * INTER * pr),
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
        ("split", topk // SPLIT_KEYS),
        ("uv", heads * V_DIM // UV_TILE),
        ("o", N_ROW_TILES),
        ("router", N_ROUTER),
        ("topk", S),
        ("ug", S * MOE_SLOTS * N_UG_PER_SLOT),
        ("down", N_ROW_TILES),
    ]


def build_layer(
    S: int = 1, heads: int = 8, npes: int = 8, topk: int = 2048, scale: float = SOFTMAX_SCALE, timeline: bool = False
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer.

    ``timeline=True`` records ``s_memrealtime`` (100 MHz) at the start and end of
    every task, and once its inputs have arrived, into the ``timeline`` buffer:
    int64 ``[sum(task counts), 3]`` (start, inputs ready, end) in ``stage_tasks``
    order.
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
    KT_OFF = H * QK_DIM  # split: bf16 q of all heads, then the KV latent / k_pe tiles (bf16 pairs)
    PT_OFF = KT_OFF + SPLIT_KEYS * KV_LORA // 2
    XN = max(S * HIDDEN // 2, PT_OFF + SPLIT_KEYS * PE_DIM // 2)
    ON = S * UK_TILE

    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(S, H, topk):
        base[name] = acc % G
        first[name] = acc
        acc += n

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XN, 16]  # bf16 activations (pairs) / split q + KV tile
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, 128, 16]
        p: fx.Array[fx.Float32, H * SPLIT_KEYS, 16]
        keys: fx.Array[fx.Int32, SPLIT_KEYS, 16]

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

        def _qptr(addr):
            return fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8), fx.Int64(addr))

        def _ld_pair(addr, scope):
            """One (value, tag) pair as a single 64-bit relaxed atomic load: never hoisted,
            coherent at ``scope`` (agent -> sc1, system -> sc0 sc1)."""
            return fx.generic_load(_qptr(addr), memory_order=fx.AtomicOrdering.Monotonic, syncscope=scope)

        def poll(specs, scope="agent"):
            """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

            Every iteration re-issues all loads together and the loop exits only when
            every tag is this launch's, so a batch costs one round trip after its
            last producer lands.  Returns one list of Int32 value bits per spec."""
            if const_expr(len(specs) == 0):
                return []
            addrs = []
            for b, i, n in specs:
                for q in range(n):
                    addrs.append(b + (fx.Int64(i) + q) * 8)

            def load_all():
                return fx.Vector.from_elements([_ld_pair(a, scope) for a in addrs], fx.Int64)

            def pending(v):
                bad = fx.Int32(v[0] >> 32) != tag
                for e in range_constexpr(1, len(addrs)):
                    bad = bad | (fx.Int32(v[e] >> 32) != tag)
                return bad

            v = load_all()
            while pending(v):
                v = load_all()
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([fx.Int32(v[e + q]) for q in range(n)])
                e += n
            return outs_

        def hint_wait(n, addr_of):
            """Cheap pre-wait: wave 0 polls (with backoff) one pair per producer task --
            the last pair that producer writes -- then the CTA barriers.  Payload
            loads afterwards still verify every tag; this only keeps hundreds of
            waiting CTAs from flooding memory with full-batch polls."""
            if wave == 0:
                for j in range_constexpr((n + 63) // 64):
                    k = lane + j * 64
                    if k < n:
                        b, i = addr_of(k)
                        a = b + fx.Int64(i) * 8
                        v = _ld_pair(a, "agent")
                        while fx.Int32(v >> 32) != tag:
                            rocdl.s_sleep(1)
                            v = _ld_pair(a, "agent")
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

        # ---- wave reductions
        def wave_sum(v):
            for sh in range_constexpr(6):
                v = v + _xshfl(v, 32 >> sh)
            return v

        def wave_max(v):
            for sh in range_constexpr(6):
                v = fx.max(v, _xshfl(v, 32 >> sh))
            return v

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

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word):
            wv = [
                fx.Vector(
                    bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + lane) * 4, vec_width=4, dtype=T.i32)
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
                for sp in range_constexpr(2):
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
            for e in range_constexpr(4):
                lds_st(red, (wave * 64 + lane) * 4 + e, acc[e])
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

        def stage_x_rmsnorm(ld2s, n, gamma, s):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x) * gamma) where ld2s(ks) -> [(x[k], x[k+1])]."""
            per = n // (2 * THREADS)
            rg_ = _rsrc(gamma)
            vals = ld2s([(tid + i * THREADS) * 2 for i in range(per)])
            ss = fx.Float32(0.0)
            for i in range_constexpr(per):
                a0, a1 = vals[i]
                ss = ss + a0 * a0 + a1 * a1
            rstd = fmath.rsqrt(block_sum(ss) / float(n) + EPS)
            for i in range_constexpr(per):
                k = (tid + i * THREADS) * 2
                a0, a1 = vals[i]
                lds_st(xs, (s * n + k) // 2, bf16_pair(a0 * rstd * ld_bf16(rg_, k), a1 * rstd * ld_bf16(rg_, k + 1)))
            return rstd

        def stage_x_pairs(name, n_total, src_of):
            """LDS bf16 X[k] = bf16(mailbox ``name`` at src_of(k)) for k < n_total (src_of(k) even)."""
            nw = n_total // 2
            full = nw // THREADS
            vals = get2_many([(mb(name), src_of((tid + i * THREADS) * 2)) for i in range(full)])
            for i in range_constexpr(full):
                lds_st(xs, tid + i * THREADS, bf16_pair(vals[i][0], vals[i][1]))
            if const_expr(nw % THREADS):
                w = tid + full * THREADS
                if w < nw:
                    a0, a1 = get2(mb(name), src_of(w * 2))
                    lds_st(xs, w, bf16_pair(a0, a1))

        def quant_block(a0, a1):
            """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (q0, q1, scale)."""
            amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            qs = (amax > 0.0).select(amax / FP8_MAX, fx.Float32(1.0))
            q0 = fx.min(fx.max(a0 / qs, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 / qs, -FP8_MAX), FP8_MAX)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, qs

        def peer_reduce(region, t, residual_fn, out_fn):
            """Push outs[s * ROW_TILE + r] as tagged pairs to every peer, then sum all
            ranks' pairs from the own symmetric buffer in rank order."""
            if tid < S * ROW_TILE // 2:
                s = tid // (ROW_TILE // 2)
                r = (tid % (ROW_TILE // 2)) * 2
                v0 = lds_ld(outs, s * ROW_TILE + r)
                v1 = lds_ld(outs, s * ROW_TILE + r + 1)
                for p in range_constexpr(W):
                    put2(
                        peer_addr[p] + fx.Int64(SY[region]), (rank * S + s) * HIDDEN + t * ROW_TILE + r, v0, v1, CM_SYS
                    )
                own = sym + fx.Int64(SY[region])
                parts = [
                    (v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32))
                    for v in poll([(own, (src * S + s) * HIDDEN + t * ROW_TILE + r, 2) for src in range(W)], "one-as")
                ]
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                row = t * ROW_TILE + r
                r0, r1 = residual_fn(s, row)
                out_fn(s, row, r0 + t0, r1 + t1)

        def start(name):
            return (bid + (G - base[name])) % G

        def stamp(name, t, which):
            if const_expr(timeline):
                if tid == 0:
                    now = fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                    fx.generic_store(
                        fx.inttoptr(
                            fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                            timeline_buf + fx.Int64((first[name] + t) * 3 + which) * 8,
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

            pre = [u_qa(c) for c in range(QA_NKC // WAVES)]
            for s in range_constexpr(S):

                def ld_h(ks, s=s):
                    res = []
                    for k in ks:
                        w = fx.Vector.from_elements(
                            [fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + k) // 2, vec_width=1, dtype=T.i32))], fx.Int32
                        )
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append((v[0], v[1]))
                    return res

                stage_x_rmsnorm(ld_h, HIDDEN, g_in, s)
            gpu.barrier()
            acc = run_units(u_qa, QA_NKC // WAVES, QA_NKC // WAVES, pre)
            reduce_rows(1, acc, emit_out(QKV_A_TILE))
            gpu.barrier()
            if tid < S * QKV_A_TILE:
                s = tid // QKV_A_TILE
                row = t * QKV_A_TILE + tid % QKV_A_TILE
                v = lds_ld(outs, tid)
                if row < Q_LORA:
                    put(mb("q_a"), s * Q_LORA + row, v)
                else:
                    put(mb("kv_a"), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)
            stamp("qkv_a", t, 2)

        # ================ 2. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
        for t in range(start("cache"), 1, G):
            stamp("cache", t, 0)
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            hint_wait(
                (KV_LORA + PE_DIM) // QKV_A_TILE,
                lambda k: (mb("kv_a"), (S - 1) * (KV_LORA + PE_DIM) + k * QKV_A_TILE + QKV_A_TILE - 1),
            )
            for s in range_constexpr(S):
                pos = pos0 + s
                v = getf(mb("kv_a"), s * (KV_LORA + PE_DIM) + tid)
                if const_expr(s == 0):
                    stamp("cache", t, 1)
                rstd = fmath.rsqrt(block_sum(v * v) / float(KV_LORA) + EPS)
                kvn = bf16_round(v * rstd * ld_bf16(_rsrc(g_kv), tid))
                bo.buffer_store(kvn.to(fx.BFloat16), r_kv, pos * KV_LORA + tid)
                put(mb("kvnew"), s * KV_LORA + tid, kvn)
                if tid < PE_DIM // 2:
                    x0, x1 = get2(mb("kv_a"), s * (KV_LORA + PE_DIM) + KV_LORA + tid * 2)
                    c = ld_f32(_rsrc(rope_cos), pos * (PE_DIM // 2) + tid)
                    sn = ld_f32(_rsrc(rope_sin), pos * (PE_DIM // 2) + tid)
                    p0 = bf16_round(x0 * c - x1 * sn)
                    p1 = bf16_round(x0 * sn + x1 * c)
                    bo.buffer_store(p0.to(fx.BFloat16), r_pe, pos * PE_DIM + tid * 2)
                    bo.buffer_store(p1.to(fx.BFloat16), r_pe, pos * PE_DIM + tid * 2 + 1)
                    put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)
            stamp("cache", t, 2)

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
            hint_wait(Q_LORA // QKV_A_TILE, lambda k: (mb("q_a"), (S - 1) * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1))
            for s in range_constexpr(S):
                stage_x_rmsnorm(lambda ks, s=s: get2_many([(mb("q_a"), s * Q_LORA + k) for k in ks]), Q_LORA, g_q, s)
            stamp("q_b", t, 1)
            gpu.barrier()
            acc = run_units(u_qb, QB_NKC // WAVES, QB_NKC // WAVES, pre)
            reduce_rows(1, acc, emit_out(Q_B_TILE))
            gpu.barrier()
            head = t // QB_PER_HEAD
            hoff = (t % QB_PER_HEAD) * Q_B_TILE
            if hoff < NOPE_DIM:
                if tid < S * Q_B_TILE:
                    s = tid // Q_B_TILE
                    put(mb("q_nope"), (s * H + head) * NOPE_DIM + hoff + tid % Q_B_TILE, lds_ld(outs, tid))
            else:
                if tid < S * Q_B_TILE // 2:
                    s = tid // (Q_B_TILE // 2)
                    pr = tid % (Q_B_TILE // 2)
                    i = hoff - NOPE_DIM + pr * 2
                    x0 = lds_ld(outs, s * Q_B_TILE + pr * 2)
                    x1 = lds_ld(outs, s * Q_B_TILE + pr * 2 + 1)
                    c = ld_f32(_rsrc(rope_cos), (pos0 + s) * (PE_DIM // 2) + i // 2)
                    sn = ld_f32(_rsrc(rope_sin), (pos0 + s) * (PE_DIM // 2) + i // 2)
                    put2(mb("q_pe"), (s * H + head) * PE_DIM + i, x0 * c - x1 * sn, x0 * sn + x1 * c)
            stamp("q_b", t, 2)

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
            )
            stage_x_pairs("q_nope", S * NOPE_DIM, lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM)
            stamp("uk", t, 1)
            gpu.barrier()
            acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
            reduce_rows(WAVES, acc, emit_out(UK_TILE))
            gpu.barrier()
            for i in range_constexpr((S * UK_TILE + THREADS - 1) // THREADS):
                k = tid + i * THREADS
                if k < S * UK_TILE:
                    s = k // UK_TILE
                    put(
                        mb("q_lat"),
                        (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + k % UK_TILE,
                        lds_ld(outs, k),
                    )
            stamp("uk", t, 2)

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
                fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * (KV_LORA // 2) + lane * 4))
                if lane < PE_DIM // 2:
                    lds_st(petile, j * (PE_DIM // 2) + lane, ld_f32(r_pe, krows[jj] * (PE_DIM // 2) + lane))

        def patch_new_kv():
            """Rows appended by this launch come from the cache task's kvnew / penew pairs."""
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kr = lds_ld(keys, j)
                if kr >= pos0:
                    sn = kr - pos0
                    kvp = get2_many([(mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2) for m in range(4)])
                    w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                    fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * (KV_LORA // 2) + lane * 4))
                    if lane < PE_DIM // 2:
                        a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                        lds_st(petile, j * (PE_DIM // 2) + lane, bf16_pair(a0, a1))

        for t in range(start("split"), N_SPLIT, G):
            t = fx.Int32(t)
            stamp("split", t, 0)
            h = wave
            for s in range_constexpr(S):
                if const_expr(s > 0):
                    gpu.barrier()
                nkeys, sparse = split_keys(t, s)
                gpu.barrier()
                gather_old_kv()  # before waiting for q: these rows are from earlier launches
                if const_expr(s == 0):
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
                                ((S - 1) * H + k // UK_PER_HEAD) * KV_LORA + (k % UK_PER_HEAD) * UK_TILE + UK_TILE - 1,
                                (k < N_UK + H * N_PE_T).select(
                                    ((S - 1) * H + (k - N_UK) // N_PE_T) * PE_DIM
                                    + ((k - N_UK) % N_PE_T) * Q_B_TILE
                                    + Q_B_TILE
                                    - 1,
                                    (S - 1) * PE_DIM + PE_DIM - 1,
                                ),
                            ),
                        ),
                    )
                # q of all heads -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64
                NQ = H * KV_LORA // 4 // THREADS
                qv = get2_many(
                    [
                        (mb("q_lat"), s * H * KV_LORA + (tid + (i // 2) * THREADS) * 4 + (i % 2) * 2)
                        for i in range(2 * NQ)
                    ]
                )
                for i in range_constexpr(NQ):
                    w4 = tid + i * THREADS
                    qw = (w4 // (KV_LORA // 4)) * (QK_DIM // 2) + (w4 % (KV_LORA // 4)) * 2
                    lds_st(xs, qw, bf16_pair(qv[2 * i][0], qv[2 * i][1]))
                    lds_st(xs, qw + 1, bf16_pair(qv[2 * i + 1][0], qv[2 * i + 1][1]))
                if tid < H * PE_DIM // 4:
                    hh = tid // (PE_DIM // 4)
                    (a0, a1), (a2, a3) = get2_many(
                        [(mb("q_pe"), s * H * PE_DIM + tid * 4), (mb("q_pe"), s * H * PE_DIM + tid * 4 + 2)]
                    )
                    qw = hh * (QK_DIM // 2) + KV_LORA // 2 + (tid % (PE_DIM // 4)) * 2
                    lds_st(xs, qw, bf16_pair(a0, a1))
                    lds_st(xs, qw + 1, bf16_pair(a2, a3))
                patch_new_kv()
                if const_expr(s == 0):
                    stamp("split", t, 1)
                gpu.barrier()
                # scores = K Q^T on MFMA: keys are M (4 row groups), the 576 dims K
                # (18 steps of 32, split in two halves), heads N.  wave = (row group, half)
                hn = fx.min(lane % 16, H - 1)
                rgk = wave % 4
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                for st in range_constexpr(QK_DIM // 32 // 2):
                    kst = (wave // 4) * (QK_DIM // 32 // 2) + st
                    key = rgk * 16 + lane % 16
                    kw = (kst < KV_LORA // 32).select(
                        KT_OFF + key * (KV_LORA // 2) + kst * 16,
                        PT_OFF + key * (PE_DIM // 2) + (kst - KV_LORA // 32) * 16,
                    )
                    a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                    b = fx.ptr_load(xs + (hn * (QK_DIM // 2) + kst * 16 + (lane // 16) * 4), result_type=v4f).bitcast(
                        fx.BFloat16
                    )
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                for e in range_constexpr(4):
                    lds_st(red, (wave * 64 + lane) * 4 + e, c[e])
                gpu.barrier()
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
                p = valid.select(fmath.exp(sc_v - m), fx.Float32(0.0))
                lsum = wave_sum(p)
                p_n = _xshfl(p, 1)
                if lane % 2 == 0:  # P^T bf16 [h][64 keys] (words h * 32 + j / 2)
                    lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
                gpu.barrier()
                # O = P V on MFMA: heads M, keys K (2 steps), 16 latent dims N per group;
                # each wave owns 4 of the 32 dim groups.  V is read key-strided from the tile.
                for g in range_constexpr(KV_LORA // 16 // WAVES):
                    dg = wave * (KV_LORA // 16 // WAVES) + g
                    d = dg * 16 + lane % 16
                    sh = (d % 2) * 16
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    for js in range_constexpr(SPLIT_KEYS // 32):
                        a = fx.ptr_load(
                            pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4), result_type=v4f
                        ).bitcast(fx.BFloat16)
                        vv = []
                        for i in range_constexpr(8):
                            j = js * 32 + (lane // 16) * 8 + i
                            word = fx.ptr_load(ktile + (j * (KV_LORA // 2) + d // 2)).bitcast(fx.Int32)
                            vv.append(fx.Int32((word >> sh) << 16).bitcast(fx.Float32))
                        b = fx.Vector.from_elements(vv, fx.Float32).to(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    if lane < 32:  # rows (heads) 4 * (lane // 16) + e < 8
                        for e in range_constexpr(4):
                            hh = (lane // 16) * 4 + e
                            put(mb("sp_acc"), ((s * N_SPLIT + t) * H + hh) * KV_LORA + d, c[e])
                if lane == 0:  # written last: the merge's readiness hint
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
            stamp("split", t, 2)

        # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
        # 4 row groups x 8 chunks: 2 waves per row group, 4 chunks each
        r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
        UV_NKC = KV_LORA // 64
        UV_R = UV_TILE // 16
        UV_WPR = WAVES // UV_R
        for t in range(start("uv"), N_UV, G):
            t = fx.Int32(t)
            stamp("uv", t, 0)
            head = t // (V_DIM // UV_TILE)

            def u_uv(c):
                kc = (wave % UV_WPR) * (UV_NKC // UV_WPR) + c
                return unit_fp8(
                    r_wuv,
                    r_suv,
                    t * UV_R + wave // UV_WPR,
                    kc,
                    UV_NKC,
                    KV_LORA,
                    128,
                    (n_sel() * KV_LORA + kc * 64) // 2,
                )

            pre = [u_uv(c) for c in range(UV_NKC // UV_WPR)]
            hint_wait(N_SPLIT, lambda k: (mb("sp_l"), ((S - 1) * N_SPLIT + k) * H + head))
            for s in range_constexpr(S):
                # wave 0: per-split weights exp(m - M) / L for this head -> misc[sp]
                if wave == 0:
                    ok_sp = lane < N_SPLIT
                    spi = ok_sp.select(lane, 0)
                    m_raw, l_raw = getf_many(
                        [(mb("sp_m"), (s * N_SPLIT + spi) * H + head), (mb("sp_l"), (s * N_SPLIT + spi) * H + head)]
                    )
                    m_sp = ok_sp.select(m_raw, fx.Float32(NEG))
                    l_sp = ok_sp.select(l_raw, fx.Float32(0.0))
                    w_sp = fmath.exp(m_sp - wave_max(m_sp))
                    den = wave_sum(l_sp * w_sp)
                    if ok_sp:
                        lds_st(misc, lane, w_sp / den)
                accs_sp = getf_many(
                    [(mb("sp_acc"), ((s * N_SPLIT + sp) * H + head) * KV_LORA + tid) for sp in range(N_SPLIT)]
                )
                if const_expr(s == 0):
                    stamp("uv", t, 1)
                gpu.barrier()
                o_l = fx.Float32(0.0)
                for sp in range_constexpr(N_SPLIT):
                    o_l = o_l + accs_sp[sp] * lds_ld(misc, sp)
                o_n = _xshfl(o_l, 1)
                if tid % 2 == 0:
                    lds_st(xs, (s * KV_LORA + tid) // 2, bf16_pair(o_l, o_n))
                gpu.barrier()
            acc = run_units(u_uv, UV_NKC // UV_WPR, UV_NKC // UV_WPR, pre)
            reduce_rows(UV_R, acc, emit_out(UV_TILE))
            gpu.barrier()
            if tid < S * UV_TILE // 2:
                s = tid // (UV_TILE // 2)
                r = (tid % (UV_TILE // 2)) * 2
                put2(
                    mb("o"), s * O_K + t * UV_TILE + r, lds_ld(outs, s * UV_TILE + r), lds_ld(outs, s * UV_TILE + r + 1)
                )
            stamp("uv", t, 2)

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
            hint_wait(N_UV, lambda k: (mb("o"), (S - 1) * O_K + k * UV_TILE + UV_TILE - 1))
            stage_x_pairs("o", S * O_K, lambda k: k)
            stamp("o", t, 1)
            gpu.barrier()
            acc = run_units(u_o, O_NKC // O_WPR, O_NKC // O_WPR, pre)
            reduce_rows(O_R, acc, emit_out(ROW_TILE))
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
                lambda s, row, v0, v1: put2(mb("a"), s * HIDDEN + row, bf16_round(v0), bf16_round(v1)),
            )
            stamp("o", t, 2)

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # 1 row group x 96 chunks (bf16): 8 waves split K
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for t in range(start("router"), N_ROUTER, G):
            t = fx.Int32(t)
            stamp("router", t, 0)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + c
                return unit_bf16(r_wr, t, kc, R_NKC, (n_sel() * HIDDEN + kc * 64) // 2)

            pre = [u_r(c) for c in range(R_NKC // WAVES)]
            hint_wait(N_ROW_TILES, lambda k: (mb("a"), (S - 1) * HIDDEN + k * ROW_TILE + ROW_TILE - 1))
            rstds = []
            for s in range_constexpr(S):
                rstds.append(
                    stage_x_rmsnorm(
                        lambda ks, s=s: get2_many([(mb("a"), s * HIDDEN + k) for k in ks]), HIDDEN, g_post, s
                    )
                )
            stamp("router", t, 1)
            # FP8-quantize this task's activation blocks (one wave per block)
            r_gp = _rsrc(g_post)
            for s in range_constexpr(S):
                blk = t * XQ_PER_ROUTER + wave
                if wave < XQ_PER_ROUTER:
                    k = blk * 128 + lane * 2
                    a0, a1 = get2(mb("a"), s * HIDDEN + k)
                    d0, d1, qs = quant_block(a0 * rstds[s] * ld_bf16(r_gp, k), a1 * rstds[s] * ld_bf16(r_gp, k + 1))
                    put2(mb("xq"), s * HIDDEN + k, d0, d1)
                    bo.buffer_store(
                        fx.Vector.from_elements([d0 * qs, d1 * qs], fx.Float32), _rsrc(mb("xqd")), s * HIDDEN + k
                    )
                    if lane == 0:
                        put(mb("xqs"), s * XQ_BLOCKS + blk, qs)
            gpu.barrier()
            acc = run_units(u_r, R_NKC // WAVES, R_NKC // WAVES, pre)
            reduce_rows(1, acc, emit_out(ROUTER_TILE))
            gpu.barrier()
            if tid < S * ROUTER_TILE:
                s = tid // ROUTER_TILE
                logit = lds_ld(outs, tid)
                put(mb("scores"), s * N_EXPERTS + t * ROUTER_TILE + tid % ROUTER_TILE, 1.0 / (1.0 + fmath.exp(-logit)))
            stamp("router", t, 2)

        # ======================= 9a. top-8 routing (one task per sample) -> sel / prob
        for t in range(start("topk"), S, G):
            t = fx.Int32(t)
            stamp("topk", t, 0)
            hint_wait(N_ROUTER, lambda k: (mb("scores"), t * N_EXPERTS + k * ROUTER_TILE + ROUTER_TILE - 1))
            if wave == 0:
                # threshold select: binary-search the 8th largest order-preserving key
                # with ballot counts, then take keys above it plus the lowest-index ties.
                # Slots 1..8 hold the winners in ascending expert id.
                r_b = _rsrc(bias)
                raws = getf_many([(mb("scores"), t * N_EXPERTS + lane + i * 64) for i in range(N_EXPERTS // 64)])
                keys_ = []
                for i in range_constexpr(N_EXPERTS // 64):
                    kb = (raws[i] + ld_f32(r_b, lane + i * 64)).bitcast(fx.Int32)
                    keys_.append(fx.Uint32((kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)))
                stamp("topk", t, 1)
                thr = fx.Uint32(0)
                for b in range_constexpr(31, -1, -1):
                    cand = thr | fx.Uint32(1 << b)
                    cnt = fx.Int32(0)
                    for i in range_constexpr(N_EXPERTS // 64):
                        cnt = cnt + _popc(_ballot(keys_[i] >= cand))
                    thr = (cnt >= TOP_K).select(cand, thr)
                n_gt = fx.Int32(0)
                for i in range_constexpr(N_EXPERTS // 64):
                    n_gt = n_gt + _popc(_ballot(keys_[i] > thr))
                eq_before = fx.Int32(0)
                sel_before = fx.Int32(0)
                tot = fx.Float32(0.0)
                picks = []
                for i in range_constexpr(N_EXPERTS // 64):
                    eqm = _ballot(keys_[i] == thr)
                    take = (keys_[i] > thr) | ((keys_[i] == thr) & (eq_before + _mbcnt(eqm) < TOP_K - n_gt))
                    selm = _ballot(take)
                    picks.append((take, sel_before + _mbcnt(selm)))
                    eq_before = eq_before + _popc(eqm)
                    sel_before = sel_before + _popc(selm)
                    tot = tot + take.select(raws[i], fx.Float32(0.0))
                tot = wave_sum(tot)
                for i in range_constexpr(N_EXPERTS // 64):
                    if picks[i][0]:
                        slot_i = t * MOE_SLOTS + 1 + picks[i][1]
                        put(mb("sel"), slot_i, lane + i * 64)
                        put(mb("prob"), slot_i, raws[i] / tot * ROUTE_SCALE)
                if lane == 0:  # slot 0: shared expert, weight 1
                    put(mb("sel"), t * MOE_SLOTS, fx.Int32(SHARED_EXPERT))
                    put(mb("prob"), t * MOE_SLOTS, fx.Float32(1.0))
            stamp("topk", t, 2)

        # ================================ 9. expert up/gate + SiLU
        # 2 row groups (16 gate + 16 up rows) x 96 chunks: 4 waves per group, 24 chunks each
        UG_NKC = HIDDEN // 64
        UG_CPW = UG_NKC // (WAVES // 2)
        for u in range(start("ug"), N_UG, G):
            u = fx.Int32(u)
            stamp("ug", u, 0)
            s_u = u // (MOE_SLOTS * N_UG_PER_SLOT)
            slot = (u // N_UG_PER_SLOT) % MOE_SLOTS
            c = u % N_UG_PER_SLOT
            hint_wait(N_ROUTER, lambda k: (mb("scores"), s_u * N_EXPERTS + k * ROUTER_TILE + ROUTER_TILE - 1))
            # this sample's FP8 activation (values in bf16) -> X[0]; block scales -> misc[8:]
            stage_x_pairs("xq", HIDDEN, lambda k: s_u * HIDDEN + k)
            if tid < XQ_BLOCKS:
                lds_st(misc, 8 + tid, getf(mb("xqs"), s_u * XQ_BLOCKS + tid))
            if tid == 0:  # slot 0 (shared expert) does not wait for routing
                lds_st(keys, 0, fx.Int32(SHARED_EXPERT))
            if (tid == 0) & (slot > 0):
                lds_st(keys, 0, get(mb("sel"), s_u * MOE_SLOTS + slot))
            stamp("ug", u, 1)
            gpu.barrier()
            e_sel = _uniform(lds_ld(keys, 0))
            wbase = w_ug + fx.Int64(e_sel) * fx.Int64(2 * INTER * HIDDEN)
            sbase = s_ug + fx.Int64(e_sel) * fx.Int64(2 * INTER // SCALE_BM * (HIDDEN // 128) * 4)
            r_wug, r_sug = _rsrc(wbase), _rsrc(sbase)
            gate_up = wave // (WAVES // 2)  # waves 0-3: gate rows, 4-7: up rows

            def u_ug(cc):
                kc = (wave % (WAVES // 2)) * UG_CPW + cc
                xsc = _uniform_f32(lds_ld(misc, 8 + kc // 2))
                return unit_fp8(r_wug, r_sug, gate_up * (INTER // 16) + c, kc, UG_NKC, HIDDEN, 128, kc * 32, xsc)

            acc = run_units(u_ug, UG_CPW, 24)
            reduce_rows(2, acc, emit_out(UG_TILE * 2))
            gpu.barrier()
            if tid < UG_TILE // 2:
                r = tid * 2
                g0, g1 = lds_ld(outs, r), lds_ld(outs, r + 1)
                u0, u1 = lds_ld(outs, UG_TILE + r), lds_ld(outs, UG_TILE + r + 1)
                put2(
                    mb("mid"),
                    (s_u * MOE_SLOTS + slot) * INTER + c * UG_TILE + r,
                    g0 / (1.0 + fmath.exp(-g0)) * u0,
                    g1 / (1.0 + fmath.exp(-g1)) * u1,
                )
            stamp("ug", u, 2)

        # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
        # 2 row groups x (S * 9 slots * 4) chunks: 4 waves per group
        DN_NKC = INTER // 64
        DN_CPW = S * MOE_SLOTS * DN_NKC // (WAVES // 2)
        DN_BLK = S * MOE_SLOTS * INTER // 128
        for t in range(start("down"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("down", t, 0)
            # routing: expert ids -> keys[], route weights -> misc[80:] (block scales use misc[:72])
            if tid < S * MOE_SLOTS:
                e_v, p_v = poll([(mb("sel"), tid, 1), (mb("prob"), tid, 1)])
                lds_st(keys, tid, e_v[0])
                lds_st(misc, 80 + tid, p_v[0].bitcast(fx.Float32))
            gpu.barrier()
            gu = wave // (WAVES // 2)

            def u_dn(cc):
                q = (wave % (WAVES // 2)) * DN_CPW + cc  # chunk index over (s, slot, kc)
                s_q = q // (MOE_SLOTS * DN_NKC)
                slot_q = (q // DN_NKC) % MOE_SLOTS
                kc = q % DN_NKC
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = _rsrc(w_dn + fx.Int64(e) * fx.Int64(HIDDEN * INTER))
                sb = _rsrc(s_dn + fx.Int64(e) * fx.Int64(HIDDEN // SCALE_BM * (INTER // 128) * 4))

                def coef():  # mid block scale * route weight, only in this sample's column
                    return (lane % 16 == s_q).select(_uniform_f32(lds_ld(misc, q // 2)), fx.Float32(0.0))

                return unit_fp8(wb, sb, t * 2 + gu, kc, DN_NKC, INTER, 128, q * 32, coef)

            # the experts are known: stream their down weights while up/gate finishes
            pre = [u_dn(cc) for cc in range(min(9, DN_CPW))]
            hint_wait(
                N_UG,
                lambda k: (
                    mb("mid"),
                    (k // (MOE_SLOTS * N_UG_PER_SLOT) * MOE_SLOTS + (k // N_UG_PER_SLOT) % MOE_SLOTS) * INTER
                    + (k % N_UG_PER_SLOT) * UG_TILE
                    + UG_TILE
                    - 1,
                ),
            )
            mids = get2_many(
                [
                    (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2)
                    for b in range((DN_BLK + WAVES - 1) // WAVES)
                ]
            )
            stamp("down", t, 1)
            # mids -> per-128 FP8 (values in bf16) in X[(s * 9 + slot) * 256 + k];
            # block scale * route weight in misc[block]
            for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    d0, d1, qs = quant_block(mids[b][0], mids[b][1])
                    lds_st(xs, blk * 64 + lane, bf16_pair(d0, d1))
                    if lane == 0:
                        lds_st(misc, blk, qs * lds_ld(misc, 80 + blk // (INTER // 128)))
            gpu.barrier()
            acc = run_units(u_dn, DN_CPW, 9, pre)
            reduce_rows(2, acc, emit_out(ROW_TILE))
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16), _rsrc(x_out), s * HIDDEN + row
                )

            peer_reduce("ffn", t, lambda s, row: get2(mb("a"), s * HIDDEN + row), store_x)
            gpu.barrier()
            stamp("down", t, 2)

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
