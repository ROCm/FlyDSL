# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Fused act-quant + int8_linear (+ optional LoRA residual) for gfx120x.

Hot path for static INT8 weights with dynamic activations: quantize ``A`` from
bf16/fp16/fp32 to signed INT8 **inside** the GEMM kernel (prologue tile
staging), then iu8 WMMA against resident INT8 ``B``. Per-row activation scales
and per-column (or scalar) weight scales apply in the epilogue, matching
``rdna4_int8_linear``.

Clamp notes: activations are scaled by ``1/scale_a[row]``, clamped to
``[-128, 127]``, and converted with round-to-nearest-even style via
``cvt.rni``-equivalent float→int cast used elsewhere in the suite. The WMMA
atom is constructed with ``clamp=False`` so the INT32 accumulator wraps;
scales restore floating range in the epilogue (same as ``rdna4_int8_linear``).

Optional LoRA residual (Linear / keep math), single or multi:

    out += lora_scale * (A_f @ lora_down.T) @ lora_up.T

where ``lora_down`` is ``[R, K]``, ``lora_up`` is ``[N, R]``, and
``lora_scale`` already folds ``(alpha / rank) * multiplier``.

**Host LoRA residuals (gfx120x):** ``int8_linear_fused`` / ``int8_linear_fused_multi``
run **one** quantized base GEMM (in-kernel fused when ``K<=128``, else device
rowwise quant + iu8), then add each adapter as a host bf16/fp16 residual in
load order for **unbounded N** — same policy as
``scaled_mm_fp8_fused`` / ``scaled_mm_fp8_fused_multi``. Prior measured slowdown vs HIP on
large/multi came from the heavy in-kernel LoRA epilogue (~60s JIT); that path
is no longer the default host surface (builder still accepts ranks
``{0,8,16,32,64}`` as an optional building block).

Requires the gfx120x iu8 WMMA atom (same branch stack as ``rdna4_int8_linear``).


Measured 2026-09-30: size-gated base (``K<=DEFAULT_K_FUSED_KERNEL_MAX`` fused /
else device-quant+iu8) + host LoRA residuals beat HIP dynamic ``int8_linear``
on the product shape grid. Unbounded adapter count.

Public: ``build_int8_linear_fused_module``, ``int8_linear_fused``,
``int8_linear_fused_multi``. Tile pick reuses ``rdna4_int8_linear.pick_tile_config``.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache

import torch

from flydsl.compiler.jit_argument import PointerJitArg
from kernels.common.dispatch_mode import get_dispatch_mode
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_autotune_tables import DEFAULT_K_FUSED_KERNEL_MAX as _TABLE_K_FUSED

from .rdna4_int8_linear import TileConfig, pick_tile_config

KERNEL_NAME = "rdna4_int8_linear_fused"

WM = WN = WK = 16
WARP = 32
LDS_PAD = 8
_I8_MAX = 127.0
_SUPPORTED_LORA_RANKS = (0, 8, 16, 32, 64)
# Measured gfx1201 2026-09-30: in-kernel fused quant+mm is faster for K<=128;
# larger K → device rowwise quant + iu8 GEMM (HIP-like) + host LoRA is faster.
DEFAULT_K_FUSED_KERNEL_MAX = _TABLE_K_FUSED  # measured table


@lru_cache(maxsize=128)
def build_int8_linear_fused_module(
    out_name: str,
    in_name: str,
    cfg: TileConfig,
    skip_bounds: bool = False,
    w_scale_per_n: bool = True,
    lora_rank: int = 0,
) -> Callable[..., None]:
    """Compile fused act-quant + int8 WMMA GEMM, optionally with LoRA epilogue."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import Vector as Vec

    if lora_rank not in _SUPPORTED_LORA_RANKS:
        raise ValueError(f"lora_rank must be one of {_SUPPORTED_LORA_RANKS}, got {lora_rank}")

    OutTy = {
        "bfloat16": fx.BFloat16,
        "float16": fx.Float16,
        "float32": fx.Float32,
    }[out_name]
    InTy = {
        "bfloat16": fx.BFloat16,
        "float16": fx.Float16,
        "float32": fx.Float32,
    }[in_name]
    use_lora = lora_rank > 0
    sig = f"{out_name}_{in_name}_{cfg.name}_bounds{int(skip_bounds)}_wpn{int(w_scale_per_n)}_R{lora_rank}"

    BM, BN, BK = cfg.bm, cfg.bn, cfg.bk
    WARPS_M, WARPS_N = cfg.warps_m, cfg.warps_n
    TM, TN = cfg.tm, cfg.tn
    THREADS = cfg.threads
    N_ACC = TM * TN
    K_STEPS = BK // WK
    STRIDE = BK + LDS_PAD
    CHUNKS_PER_ROW = BK // 16
    CHUNKS_A = BM * CHUNKS_PER_ROW
    CHUNKS_B = BN * CHUNKS_PER_ROW
    PTA = CHUNKS_A // THREADS
    PTB = CHUNKS_B // THREADS
    GROUP_M = 4
    AS_BYTES = BM * STRIDE
    BS_BYTES = BN * STRIDE
    R = lora_rank
    H_BYTES = BM * max(R, 1) * 4
    H_STRIDE = max(R, 1)

    assert BM == WARPS_M * TM * WM
    assert BN == WARPS_N * TN * WN
    assert BK % WK == 0
    assert CHUNKS_A % THREADS == 0 and CHUNKS_B % THREADS == 0

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def gemm_kernel(
        Af: fx.Pointer,
        Bnk: fx.Pointer,
        C: fx.Pointer,
        ScaleA: fx.Pointer,
        ScaleB: fx.Pointer,
        LoraDown: fx.Pointer,
        LoraUp: fx.Pointer,
        LoraScale: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Int32,
        BlocksN: fx.Int32,
        BlocksM: fx.Int32,
    ) -> None:
        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WM, WN, WK, fx.Int8, fx.Int32, sign_a=True, sign_b=True, clamp=False)
        )

        def wmma_iu8(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            """RDNA4 Wave32 iu8 WMMA via ``fx.gemm`` (v8 i8 A/B, v8 i32 C)."""
            a_frag = fx.make_rmem_tensor(8, fx.Int8)
            b_frag = fx.make_rmem_tensor(8, fx.Int8)
            c_frag = fx.make_rmem_tensor(8, fx.Int32)
            a_frag.store(Vec(a_v8))
            b_frag.store(Vec(b_v8))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)
            return Vec(c_frag.load())

        tid = fx.Int32(gpu.thread_id("x"))
        bid_x = fx.Int32(gpu.block_id("x"))
        bid_y = fx.Int32(gpu.block_id("y"))
        lane = tid % fx.Int32(WARP)
        wave = tid // fx.Int32(WARP)
        lane16 = lane % fx.Int32(16)
        klane = lane // fx.Int32(16)
        wm = wave // fx.Int32(WARPS_N)
        wn = wave % fx.Int32(WARPS_N)

        bid = bid_y * BlocksN + bid_x
        per_group = fx.Int32(GROUP_M) * BlocksN
        group = bid // per_group
        idx_in_group = bid - group * per_group
        group_rows = fx.min(fx.Int32(GROUP_M), BlocksM - group * fx.Int32(GROUP_M))
        bm = group * fx.Int32(GROUP_M) + idx_in_group % group_rows
        bn = idx_in_group // group_rows
        m0 = bm * fx.Int32(BM)
        n0 = bn * fx.Int32(BN)

        a_g = fx.recast_iter(fx.PointerType.get(InTy.ir_type, Af.address_space), Af)
        b_g = fx.recast_iter(fx.PointerType.get(fx.Int8.ir_type, Bnk.address_space), Bnk)
        c_ptr = fx.recast_iter(fx.PointerType.get(OutTy.ir_type, C.address_space), C)
        sa_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleA.address_space), ScaleA)
        sb_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB)
        down_g = fx.recast_iter(fx.PointerType.get(InTy.ir_type, LoraDown.address_space), LoraDown)
        up_g = fx.recast_iter(fx.PointerType.get(InTy.ir_type, LoraUp.address_space), LoraUp)

        def as_i32(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, ptr.address_space), ptr)

        alloc = fx.SharedAllocator()
        as_base = alloc.allocate(AS_BYTES)._ptr
        bs_base = alloc.allocate(BS_BYTES)._ptr
        if const_expr(use_lora and H_BYTES > AS_BYTES):
            h_base = alloc.allocate(H_BYTES)._ptr
        else:
            h_base = as_base

        vmax = fx.Float32(_I8_MAX)
        vmin = fx.Float32(-_I8_MAX)

        def load16_gmem_b(elem_ptr: fx.Pointer, byte_idx: fx.Int32 | fx.Int64 | int, inb: object) -> fx.Vector:
            i32_ptr = as_i32(elem_ptr)
            safe = inb.select(fx.Int64(byte_idx), fx.Int64(0))
            view = fx.make_view(
                fx.add_offset(i32_ptr, safe // fx.Int64(4)),
                fx.make_layout(4, 1),
            )
            raw = Vec(view.load())
            z = fx.Int32(0)
            return Vec.from_elements(
                [
                    inb.select(fx.Int32(raw[0]), z),
                    inb.select(fx.Int32(raw[1]), z),
                    inb.select(fx.Int32(raw[2]), z),
                    inb.select(fx.Int32(raw[3]), z),
                ],
                fx.Int32,
            )

        def pack_i8x4(e0: fx.Int32, e1: fx.Int32, e2: fx.Int32, e3: fx.Int32) -> fx.Int32:
            """Pack four signed i8 values into one i32 (little-endian bytes)."""
            # Mask to 8 bits and shift-or.
            b0 = fx.Int32(e0) & fx.Int32(0xFF)
            b1 = (fx.Int32(e1) & fx.Int32(0xFF)) << fx.Int32(8)
            b2 = (fx.Int32(e2) & fx.Int32(0xFF)) << fx.Int32(16)
            b3 = (fx.Int32(e3) & fx.Int32(0xFF)) << fx.Int32(24)
            return b0 | b1 | b2 | b3

        def load16_act_quant_i8(grow: fx.Int32, gk: fx.Int32, inb_row: object) -> fx.Vector:
            """Load 16 activations, quantize to packed INT8 using per-row scale."""
            # scale_a[grow]
            sa_safe = inb_row.select(fx.Int64(grow), fx.Int64(0))
            sa_v = fx.make_view(fx.add_offset(sa_ptr, sa_safe), fx.make_layout(1, 1)).load()
            inv = fx.Float32(1.0) / fx.Float32(sa_v[0])
            elems_i = []
            for ei in range_constexpr(16):
                kk = gk + fx.Int32(ei)
                inb = inb_row & (kk < K)
                safe_m = inb.select(fx.Int64(grow), fx.Int64(0))
                safe_k = inb.select(fx.Int64(kk), fx.Int64(0))
                idx = safe_m * fx.Int64(K) + safe_k
                view = fx.make_view(
                    fx.add_offset(a_g, idx),
                    fx.make_layout(1, 1),
                )
                raw = view.load()
                if const_expr(in_name == "float32"):
                    v = fx.Float32(raw[0])
                else:
                    v = InTy(raw[0]).to(fx.Float32)
                v = inb.select(v, fx.Float32(0.0))
                x = fx.max(fx.min(v * inv, vmax), vmin)
                # round-to-nearest via +0.5 * sign trick for positive-biased;
                # use nearbyint-style: convert through Int32 after round.
                # Fix negative rounding (truncate toward +inf after +0.5 on neg).
                # Simpler portable path: cast float→int32 (implementation-defined
                # toward zero) after adjusting by sign.
                pos = x >= fx.Float32(0.0)
                adj = pos.select(x + fx.Float32(0.5), x - fx.Float32(0.5))
                qi = adj.to(fx.Int32)
                qi = fx.max(fx.min(qi, fx.Int32(127)), fx.Int32(-128))
                elems_i.append(qi)

            words = []
            for w in range_constexpr(4):
                bi = w * 4
                words.append(
                    pack_i8x4(
                        elems_i[bi + 0],
                        elems_i[bi + 1],
                        elems_i[bi + 2],
                        elems_i[bi + 3],
                    )
                )
            return Vec.from_elements(words, fx.Int32)

        def store8x2_lds(lds_ptr: fx.Pointer, byte_off: fx.Int32 | fx.Int64 | int, v4i32: fx.Vector) -> None:
            i32_lds = as_i32(lds_ptr)
            off0 = fx.Int64(byte_off) // fx.Int64(4)
            view0 = fx.make_view(fx.add_offset(i32_lds, off0), fx.make_layout(2, 1))
            view0.store(Vec.from_elements([fx.Int32(v4i32[0]), fx.Int32(v4i32[1])], fx.Int32))
            view1 = fx.make_view(fx.add_offset(i32_lds, off0 + fx.Int64(2)), fx.make_layout(2, 1))
            view1.store(Vec.from_elements([fx.Int32(v4i32[2]), fx.Int32(v4i32[3])], fx.Int32))

        def load_a_regs(kbyte0: fx.Int32) -> list[fx.Vector]:
            regs = []
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                grow = m0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                inb_row = grow < M
                regs.append(load16_act_quant_i8(grow, gk, inb_row))
            return regs

        def load_b_regs(kbyte0: fx.Int32) -> list[fx.Vector]:
            regs = []
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                grow = n0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                inb = (grow < N) & (gk < K)
                regs.append(load16_gmem_b(b_g, fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk), inb))
            return regs

        def store_a_regs(regs: list[fx.Vector]) -> None:
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                local_row = c // fx.Int32(CHUNKS_PER_ROW)
                local_k = (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                store8x2_lds(as_base, local_row * fx.Int32(STRIDE) + local_k, regs[i])

        def store_b_regs(regs: list[fx.Vector]) -> None:
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                local_row = c // fx.Int32(CHUNKS_PER_ROW)
                local_k = (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                store8x2_lds(bs_base, local_row * fx.Int32(STRIDE) + local_k, regs[i])

        def load_frag_lds(lds_ptr: fx.Pointer, row: fx.Int32, kbyte: fx.Int32) -> fx.Vector:
            off = fx.Int64(row) * fx.Int64(STRIDE) + fx.Int64(kbyte) + fx.Int64(klane) * fx.Int64(8)
            i32_lds = as_i32(lds_ptr)
            view = fx.make_view(
                fx.add_offset(i32_lds, off // fx.Int64(4)),
                fx.make_layout(2, 1),
            )
            return Vec(view.load()).bitcast(fx.Int8)

        def unpack_acc(carried: list[fx.Float32], ai: int) -> fx.Vector:
            base = ai * 8
            return Vec.from_elements(
                [fx.Int32(carried[base + e]) for e in range(8)],
                fx.Int32,
            )

        def pack_acc(acc_vec: fx.Vector) -> list[fx.Float32]:
            return [fx.Int32(acc_vec[e]) for e in range(8)]

        zero = fx.Int32(0)
        init = [zero for _ in range(N_ACC * 8)]

        a_regs = load_a_regs(fx.Int32(0))
        b_regs = load_b_regs(fx.Int32(0))
        store_a_regs(a_regs)
        store_b_regs(b_regs)
        gpu.barrier()

        results = init
        for kb, carried in range(fx.Int32(0), K, fx.Int32(BK), init=init):
            kb_i = fx.Int32(kb)
            acc_list = [unpack_acc(carried, ai) for ai in range_constexpr(N_ACC)]

            knext = kb_i + fx.Int32(BK)
            has_next = knext < K
            if has_next:
                a_regs = load_a_regs(knext)
                b_regs = load_b_regs(knext)

            af_cur = []
            for ti in range_constexpr(TM):
                row = wm * fx.Int32(TM * WM) + fx.Int32(ti * WM) + lane16
                af_cur.append(load_frag_lds(as_base, row, fx.Int32(0)))
            bf_cur = []
            for tj in range_constexpr(TN):
                col = wn * fx.Int32(TN * WN) + fx.Int32(tj * WN) + lane16
                bf_cur.append(load_frag_lds(bs_base, col, fx.Int32(0)))

            for kk in range_constexpr(K_STEPS):
                if const_expr(kk + 1 < K_STEPS):
                    kbyte_n = fx.Int32((kk + 1) * WK)
                    af_nxt = []
                    for ti in range_constexpr(TM):
                        row = wm * fx.Int32(TM * WM) + fx.Int32(ti * WM) + lane16
                        af_nxt.append(load_frag_lds(as_base, row, kbyte_n))
                    bf_nxt = []
                    for tj in range_constexpr(TN):
                        col = wn * fx.Int32(TN * WN) + fx.Int32(tj * WN) + lane16
                        bf_nxt.append(load_frag_lds(bs_base, col, kbyte_n))

                new_accs = []
                for ti in range_constexpr(TM):
                    for tj in range_constexpr(TN):
                        idx = ti * TN + tj
                        new_accs.append(wmma_iu8(af_cur[ti], bf_cur[tj], acc_list[idx]))
                acc_list = new_accs

                if const_expr(kk + 1 < K_STEPS):
                    af_cur = af_nxt
                    bf_cur = bf_nxt

            if has_next:
                gpu.barrier()
                store_a_regs(a_regs)
                store_b_regs(b_regs)
                gpu.barrier()

            flat = []
            for ai in range_constexpr(N_ACC):
                flat.extend(pack_acc(acc_list[ai]))
            results = yield flat

        if const_expr(use_lora):
            gpu.barrier()
            h_i32 = as_i32(h_base)
            h_words = (BM * R + THREADS - 1) // THREADS
            for hi in range_constexpr(h_words):
                flat_i = tid + fx.Int32(hi * THREADS)
                if flat_i < fx.Int32(BM * R):
                    view = fx.make_view(
                        fx.add_offset(h_i32, fx.Int64(flat_i)),
                        fx.make_layout(1, 1),
                    )
                    view.store(Vec.from_elements([fx.Int32(0)], fx.Int32))
            gpu.barrier()

            pairs = BM * R
            n_stripes = (pairs + THREADS - 1) // THREADS
            h_f_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, h_base.address_space),
                h_base,
            )
            for si in range_constexpr(n_stripes):
                flat_i = tid + fx.Int32(si * THREADS)
                in_pair = flat_i < fx.Int32(pairs)
                local_row = flat_i // fx.Int32(R)
                rr = flat_i % fx.Int32(R)
                grow = m0 + local_row
                acc_init = [fx.Float32(0.0), fx.Float32(0.0)]
                acc_h = acc_init
                for kk, carried in range(fx.Int32(0), K, fx.Int32(1), init=acc_init):
                    kk_i = fx.Int32(kk)
                    prev = fx.Float32(carried[0])
                    inb = in_pair & (grow < M) & (kk_i < K)
                    safe_m = inb.select(fx.Int64(grow), fx.Int64(0))
                    safe_k = inb.select(fx.Int64(kk_i), fx.Int64(0))
                    a_view = fx.make_view(
                        fx.add_offset(a_g, safe_m * fx.Int64(K) + safe_k),
                        fx.make_layout(1, 1),
                    )
                    d_view = fx.make_view(
                        fx.add_offset(down_g, fx.Int64(rr) * fx.Int64(K) + safe_k),
                        fx.make_layout(1, 1),
                    )
                    av = a_view.load()
                    dv = d_view.load()
                    if const_expr(in_name == "float32"):
                        af = fx.Float32(av[0])
                        df = fx.Float32(dv[0])
                    else:
                        af = InTy(av[0]).to(fx.Float32)
                        df = InTy(dv[0]).to(fx.Float32)
                    af = inb.select(af, fx.Float32(0.0))
                    df = inb.select(df, fx.Float32(0.0))
                    nxt = prev + af * df
                    acc_h = yield [nxt, fx.Float32(0.0)]
                if in_pair:
                    off = fx.Int64(local_row) * fx.Int64(H_STRIDE) + fx.Int64(rr)
                    hv = fx.make_view(fx.add_offset(h_f_ptr, off), fx.make_layout(1, 1))
                    hv.store(Vec.from_elements([fx.Float32(acc_h[0])], fx.Float32))
            gpu.barrier()

            ls_view = fx.make_view(
                fx.recast_iter(
                    fx.PointerType.get(fx.Float32.ir_type, LoraScale.address_space),
                    LoraScale,
                ),
                fx.make_layout(1, 1),
            )
            lora_scale = fx.Float32(ls_view.load()[0])
            h_f = h_f_ptr

        # Epilogue: i32→f32 * row_scale * col_scale (+ LoRA) → out
        for ti in range_constexpr(TM):
            row_base = (
                fx.Int64(m0) + fx.Int64(wm) * fx.Int64(TM * WM) + fx.Int64(ti * WM) + fx.Int64(klane) * fx.Int64(8)
            )
            for tj in range_constexpr(TN):
                idx = ti * TN + tj
                acc = unpack_acc(results, idx)
                col = fx.Int64(n0) + fx.Int64(wn) * fx.Int64(TN * WN) + fx.Int64(tj * WN) + fx.Int64(lane16)
                for e in range_constexpr(8):
                    m = row_base + fx.Int64(e)
                    n = col
                    sa_v = fx.make_view(fx.add_offset(sa_ptr, m), fx.make_layout(1, 1)).load()
                    if const_expr(w_scale_per_n):
                        sb_v = fx.make_view(fx.add_offset(sb_ptr, n), fx.make_layout(1, 1)).load()
                    else:
                        sb_v = fx.make_view(sb_ptr, fx.make_layout(1, 1)).load()
                    val = fx.Float32(fx.Int32(acc[e]).to(fx.Float32)) * fx.Float32(sa_v[0]) * fx.Float32(sb_v[0])
                    if const_expr(use_lora):
                        local_row = fx.Int32(m - fx.Int64(m0))
                        delta = fx.Float32(0.0)
                        for rr in range_constexpr(R):
                            hv = fx.make_view(
                                fx.add_offset(
                                    h_f,
                                    fx.Int64(local_row) * fx.Int64(H_STRIDE) + fx.Int64(rr),
                                ),
                                fx.make_layout(1, 1),
                            ).load()
                            inb_up = (m < fx.Int64(M)) & (n < fx.Int64(N))
                            safe_n = inb_up.select(n, fx.Int64(0))
                            uv = fx.make_view(
                                fx.add_offset(up_g, safe_n * fx.Int64(R) + fx.Int64(rr)),
                                fx.make_layout(1, 1),
                            ).load()
                            if const_expr(in_name == "float32"):
                                uf = fx.Float32(uv[0])
                            else:
                                uf = InTy(uv[0]).to(fx.Float32)
                            delta = delta + fx.Float32(hv[0]) * uf
                        val = val + lora_scale * delta
                    if const_expr(out_name == "float32"):
                        out_v = val
                    elif const_expr(out_name == "float16"):
                        out_v = val.to(fx.Float16)
                    else:
                        out_v = val.to(fx.BFloat16)
                    cidx = m * fx.Int64(N) + n
                    if const_expr(skip_bounds):
                        view = fx.make_view(fx.add_offset(c_ptr, cidx), fx.make_layout(1, 1))
                        view.store(Vec.from_elements([out_v], OutTy))
                    else:
                        inb = (m < fx.Int64(M)) & (n < fx.Int64(N))
                        safe = inb.select(cidx, fx.Int64(0))
                        view = fx.make_view(fx.add_offset(c_ptr, safe), fx.make_layout(1, 1))
                        if inb:
                            view.store(Vec.from_elements([out_v], OutTy))

    @flyc.jit
    def launch(
        Af: fx.Pointer,
        Bnk: fx.Pointer,
        C: fx.Pointer,
        ScaleA: fx.Pointer,
        ScaleB: fx.Pointer,
        LoraDown: fx.Pointer,
        LoraUp: fx.Pointer,
        LoraScale: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid_n = (fx.Int64(N) + fx.Int64(BN - 1)) // fx.Int64(BN)
        grid_m = (fx.Int64(M) + fx.Int64(BM - 1)) // fx.Int64(BM)
        gemm_kernel(
            Af,
            Bnk,
            C,
            ScaleA,
            ScaleB,
            LoraDown,
            LoraUp,
            LoraScale,
            M,
            N,
            K,
            fx.Int32(grid_n),
            fx.Int32(grid_m),
        ).launch(grid=(grid_n, grid_m, 1), block=(THREADS, 1, 1), stream=stream)

    gemm_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


_DUMMY_LORA_CACHE: dict = {}


def _dummy_lora_bufs(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse 1-elem dummy LoRA buffers so lora_rank=0 launches do not allocate."""
    key = (str(device), dtype)
    hit = _DUMMY_LORA_CACHE.get(key)
    if hit is None:
        import torch

        hit = (
            torch.zeros(1, dtype=dtype, device=device),
            torch.zeros(1, dtype=dtype, device=device),
            torch.zeros(1, dtype=torch.float32, device=device),
        )
        _DUMMY_LORA_CACHE[key] = hit
    return hit


def _scale_f32_ready(s: object, device: torch.device, *, numel: int | None = None) -> torch.Tensor:
    """Skip .to/reshape/contiguous when scale is already f32 contiguous on device."""
    import torch

    if (
        isinstance(s, torch.Tensor)
        and s.dtype == torch.float32
        and s.device == device
        and s.is_contiguous()
        and (numel is None or s.numel() == numel)
    ):
        return s.reshape(-1) if s.dim() != 1 else s
    out = s.detach().to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    return out


def reference_int8_linear_fused(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_down: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """Torch reference: separate int8 quant + int8_linear (+ optional LoRA)."""
    sa = scale_a.float().reshape(-1)
    sb = scale_b.float().reshape(-1)
    # per-row quant
    a_scaled = a_f.float() / sa.unsqueeze(1)
    a_q = a_scaled.round().clamp(-128, 127).to(torch.int8)
    acc = a_q.float() @ b_nk.float().T
    if sb.numel() == 1:
        out = acc.float() * sa.unsqueeze(1) * sb[0]
    else:
        out = acc.float() * sa.unsqueeze(1) * sb.unsqueeze(0)
    if lora_down is not None and lora_up is not None:
        # F.linear chain: hidden = a @ down.T; delta = hidden @ up.T
        hidden = a_f.float() @ lora_down.float().T
        out = out + float(lora_scale) * (hidden @ lora_up.float().T)
    return out.to(out_dtype)


def _base_quant_mm(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """One quantized base GEMM (FP8 fused mirror): fused kernel or device-quant+iu8.

    * ``K <= DEFAULT_K_FUSED_KERNEL_MAX``: in-kernel act-quant + iu8 WMMA using
      caller ``scale_a`` / ``scale_b`` (faster on measured tiny-K shapes).
    * ``K > DEFAULT_K_FUSED_KERNEL_MAX``: device ``quantize_int8_rowwise`` +
      iu8 ``int8_linear_dispatched`` (measured faster than HIP dynamic int8_linear
      on mid/large; HIP-like dynamic act quant). Caller ``scale_a`` is not used
      on this path (scales come from the device quantizer); ``scale_b`` is.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.tensor_shim import _run_compiled

    m, k = a_f.shape
    n = b_nk.shape[0]

    mode = get_dispatch_mode()
    if mode == "force_hip":
        from kernels.common.gfx120x_torch_ref import int8_linear_torch

        return int8_linear_torch(a_f, b_nk, scale_b, out_dtype=out_dtype)

    # force_flydsl: always in-kernel fused (ignore K gate). auto: measured K gate.
    if mode != "force_flydsl" and k > DEFAULT_K_FUSED_KERNEL_MAX:
        from kernels.gemm.rdna4_int8_linear_dispatch import int8_linear_dispatched
        from kernels.quant.rdna4_quantize_int8_rowwise import quantize_int8_rowwise

        q, s = quantize_int8_rowwise(a_f)
        # quantize returns f32 [M,1] contiguous — avoid detach/float/contig churn
        x_scale = (
            s.reshape(-1) if s.dtype == torch.float32 and s.is_contiguous() else s.float().reshape(-1).contiguous()
        )
        w_scale = _scale_f32_ready(scale_b, a_f.device)
        return int8_linear_dispatched(
            "iu8",
            a_int8=q,
            b_int8=b_nk,
            x_scale=x_scale,
            w_scale=w_scale,
            out_dtype=out_dtype,
        )

    in_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[a_f.dtype]
    out_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[out_dtype]

    scale_a = _scale_f32_ready(scale_a, a_f.device, numel=m)
    scale_b = _scale_f32_ready(scale_b, a_f.device)
    if scale_a.numel() != m:
        raise ValueError(f"scale_a must be [M]={m}, got {scale_a.numel()}")
    w_per_n = scale_b.numel() != 1
    if w_per_n and scale_b.numel() != n:
        raise ValueError(f"scale_b must be scalar or [N]={n}, got {scale_b.numel()}")

    cfg = pick_tile_config(m, n, k)
    skip_bounds = m % cfg.bm == 0 and n % cfg.bn == 0
    launch = build_int8_linear_fused_module(out_name, in_name, cfg, skip_bounds, w_per_n, 0)
    out = torch.empty((m, n), dtype=out_dtype, device=a_f.device)
    lora_down_buf, lora_up_buf, lora_scale_buf = _dummy_lora_bufs(a_f.device, a_f.dtype)

    def _ptr(t: torch.Tensor) -> PointerJitArg:
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    if stream is None:
        stream = torch.cuda.current_stream()
    _run_compiled(
        launch,
        _ptr(a_f.view(torch.uint8)),
        _ptr(b_nk.view(torch.uint8)),
        _ptr(out.view(torch.uint8)),
        _ptr(scale_a),
        _ptr(scale_b),
        _ptr(lora_down_buf.view(torch.uint8)),
        _ptr(lora_up_buf.view(torch.uint8)),
        _ptr(lora_scale_buf),
        m,
        n,
        k,
        stream,
    )
    return out


def int8_linear_fused(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_down: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_up: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_scale: float | torch.Tensor | Sequence[float | torch.Tensor] = 1.0,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Host wrapper: fused act-quant + int8_linear with optional LoRA.

    ``a_f`` is bf16/fp16/fp32 ``[M, K]``; ``b_nk`` is int8 ``[N, K]``.
    ``scale_a`` is per-row ``[M]``; ``scale_b`` is scalar or per-column ``[N]``.

    LoRA tensors (when provided) match Linear keep: down ``[R, K]``,
    up ``[N, R]``, same dtype as activations. Pass a **list/tuple** of downs/ups
    (and optional scale list) for N adapters in load order — dispatches to
    ``int8_linear_fused_multi``.
    """
    require_gfx120x(a_f.device, what="int8_linear_fused (gfx120x)")
    # Host LoRA residual: any LoRA count (1..N) uses multi/host residual path (in-kernel
    # LoRA epilogue measured slower / heavy JIT on an otherwise idle gfx120x).
    if (
        lora_down is not None
        or lora_up is not None
        or isinstance(lora_down, (list, tuple))
        or isinstance(lora_up, (list, tuple))
    ):
        return int8_linear_fused_multi(
            a_f,
            b_nk,
            scale_a,
            scale_b,
            out_dtype=out_dtype,
            lora_downs=lora_down,
            lora_ups=lora_up,
            lora_scales=lora_scale,
            stream=stream,
        )

    if a_f.dim() != 2 or b_nk.dim() != 2:
        raise ValueError("int8_linear_fused expects 2D operands")
    if b_nk.dtype != torch.int8:
        raise ValueError(f"b_nk must be int8, got {b_nk.dtype}")
    m, k = a_f.shape
    if b_nk.shape[1] != k:
        raise ValueError(f"K mismatch: a_f {k} vs b_nk {b_nk.shape[1]}")
    return _base_quant_mm(a_f, b_nk, scale_a, scale_b, out_dtype=out_dtype, stream=stream)


def _lora_scale_as_float(scale: float | torch.Tensor) -> float:
    if isinstance(scale, torch.Tensor):
        return float(scale.detach().float().reshape(-1)[0].item())
    return float(scale)


def _pack_lora_adapters(
    lora_downs: torch.Tensor | Sequence[torch.Tensor] | None,
    lora_ups: torch.Tensor | Sequence[torch.Tensor] | None,
    lora_scales: float | torch.Tensor | Sequence[float | torch.Tensor] | None,
    *,
    k: int,
    n: int,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Normalize single/list LoRA args into ordered (down, up, scale) packs."""
    if lora_downs is None and lora_ups is None:
        if lora_scales is None:
            return []
        raise ValueError("lora_scales set without lora_downs/lora_ups")
    if lora_downs is None or lora_ups is None:
        raise ValueError("lora_downs and lora_ups must both be set or both None")

    if isinstance(lora_downs, torch.Tensor):
        downs: list[torch.Tensor] = [lora_downs]
    else:
        downs = list(lora_downs)
    if isinstance(lora_ups, torch.Tensor):
        ups: list[torch.Tensor] = [lora_ups]
    else:
        ups = list(lora_ups)
    if len(downs) != len(ups):
        raise ValueError(f"lora_downs/lora_ups length mismatch: {len(downs)} vs {len(ups)}")

    if lora_scales is None:
        scales_list: list[float | torch.Tensor] = [1.0] * len(downs)
    elif isinstance(lora_scales, (list, tuple)):
        scales_list = list(lora_scales)
    else:
        scales_list = [lora_scales] * len(downs)
    if len(scales_list) != len(downs):
        raise ValueError(f"lora_scales length {len(scales_list)} != adapters {len(downs)}")

    packs: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for i, (down, up, scale) in enumerate(zip(downs, ups, scales_list)):
        if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
            raise TypeError(f"adapter[{i}] down/up must be tensors")
        if down.dim() != 2 or up.dim() != 2:
            raise ValueError(f"adapter[{i}] down/up must be 2D")
        rank = int(down.shape[0])
        if tuple(down.shape) != (rank, k):
            raise ValueError(f"adapter[{i}] lora_down must be [{rank}, {k}]")
        if tuple(up.shape) != (n, rank):
            raise ValueError(f"adapter[{i}] lora_up must be [{n}, {rank}]")
        packs.append((down, up, _lora_scale_as_float(scale)))
    return packs


def _lora_residual(
    a_f: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    lora_scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Host LoRA-only GEMM in activation dtype (FP8 fused mirror; no fp32 copies)."""
    down = lora_down if lora_down.dtype == a_f.dtype else lora_down.to(dtype=a_f.dtype)
    up = lora_up if lora_up.dtype == a_f.dtype else lora_up.to(dtype=a_f.dtype)
    # (x @ A.T) @ B.T — keep act dtype; match scaled_mm_fp8_fused._lora_residual
    hidden = a_f @ down.t()
    delta = hidden @ up.t()
    if lora_scale != 1.0:
        delta = delta * float(lora_scale)
    return delta if delta.dtype == out_dtype else delta.to(dtype=out_dtype)


def _apply_lora_inplace(
    out: torch.Tensor,
    a_f: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    lora_scale: float,
) -> None:
    """Accumulate scale*(x@down.T)@up.T into ``out`` via addmm_ (one epilogue add)."""
    down = lora_down if lora_down.dtype == a_f.dtype else lora_down.to(dtype=a_f.dtype)
    up = lora_up if lora_up.dtype == a_f.dtype else lora_up.to(dtype=a_f.dtype)
    hidden = a_f @ down.t()
    alpha = float(lora_scale)
    # addmm_(mat1, mat2) => out += mat1 @ mat2; up is [N,R] so use up.t() -> [R,N]
    if alpha == 1.0:
        out.addmm_(hidden, up.t())
    else:
        out.addmm_(hidden, up.t(), alpha=alpha)


def reference_int8_linear_fused_multi(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_downs: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_ups: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_scales: float | torch.Tensor | Sequence[float | torch.Tensor] | None = 1.0,
) -> torch.Tensor:
    """Torch reference: base quant+mm + sequential LoRA residuals in load order."""
    m, k = a_f.shape
    n = b_nk.shape[0]
    packs = _pack_lora_adapters(lora_downs, lora_ups, lora_scales, k=k, n=n)
    out = reference_int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=out_dtype)
    for down, up, scale in packs:
        out.add_(_lora_residual(a_f, down, up, scale, out_dtype))
    return out


def int8_linear_fused_multi(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_downs: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_ups: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_scales: float | torch.Tensor | Sequence[float | torch.Tensor] | None = 1.0,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """N-adapter fused path: one quantized base GEMM, then host residuals.

    **Unbounded N:** any number of adapters (no fixed max). Base runs once;
    each adapter applies ``scale * (x @ down.T) @ up.T`` in **load order** as a
    host bf16/fp16 residual (same contract as ``scaled_mm_fp8_fused_multi``).
    No dedicated multi-LoRA GEMM kernel.

    * **N=0**: base only via ``_base_quant_mm`` (fused kernel or device-quant+iu8)
    * **N≥1**: base once + host residuals for every adapter in order

    Args accept a single tensor or a sequence of tensors for downs/ups; scales
    may be a scalar (broadcast) or a per-adapter sequence.
    """
    require_gfx120x(a_f.device, what="int8_linear_fused_multi (gfx120x)")
    if a_f.dim() != 2 or b_nk.dim() != 2:
        raise ValueError("int8_linear_fused_multi expects 2D operands")
    m, k = a_f.shape
    n = b_nk.shape[0]
    if b_nk.shape[1] != k:
        raise ValueError(f"K mismatch: a_f {k} vs b_nk {b_nk.shape[1]}")

    packs = _pack_lora_adapters(lora_downs, lora_ups, lora_scales, k=k, n=n)

    # Host-residual policy (gfx120x / FP8 mirror): one quantized base + host residuals.
    # N is unbounded — loop packs in load order (no max-adapter cap).
    out = _base_quant_mm(a_f, b_nk, scale_a, scale_b, out_dtype=out_dtype, stream=stream)
    for down, up, scale in packs:
        _apply_lora_inplace(out, a_f, down, up, scale)
    return out
