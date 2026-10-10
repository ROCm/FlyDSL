# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Fused act-quant + FP8 e4m3/e5m2 scaled_mm (+ optional LoRA residual) for gfx120x.

Hot path for static FP8 weights with dynamic activations: quantize ``A`` from
bf16/fp16/fp32 to e4m3 or e5m2 **inside** the GEMM kernel (prologue tile
staging; bf8 cvt when ``e5m2``), then WMMA against resident FP8 ``B``, applying
tensorwise ``scale_a * scale_b`` in the epilogue. Avoids a separate
``fp8_quant`` launch before ``scaled_mm``. Format parity with
``rdna4_fp8_quant`` / ``rdna4_stoch_fp8`` (max 448 vs 57344).

Optional LoRA residual (Linear math), single or multi::

    out += lora_scale * (A_f @ lora_down.T) @ lora_up.T

where ``lora_down`` is ``[R, K]``, ``lora_up`` is ``[N, R]``, and
``lora_scale`` already folds ``(alpha / rank) * multiplier``.
``build_scaled_mm_fp8_fused_module(..., lora_rank)`` can specialize ranks
``{8,16,32,64}`` as an in-kernel LoRA epilogue building block, but the public
host wrappers do **not** launch that path: any LoRA (N>=1) uses the device
residual policy below (in-kernel LoRA was slower than HIP on idle gfx120x).

**Multi-adapter (any N, load order):** ``scaled_mm_fp8_fused_multi`` (also via
list or single ``lora_down``/``lora_up`` on ``scaled_mm_fp8_fused``) runs base
fused quant+mm once (``lora_rank=0`` / no in-kernel LoRA), then adds each
residual in pack order via device GEMMs (``gemm_bf16_nmajor_lds`` +
``add_same``). N=0 is plain fused. Correctness = sequential sum in load
order (same as ``sum h_i(x)``).

Does **not** require the gfx120x iu8 atom (FP8 WMMA only).

Public: ``build_scaled_mm_fp8_fused_module``, ``scaled_mm_fp8_fused``,
``scaled_mm_fp8_fused_multi``. Tile pick reuses ``rdna4_scaled_mm_fp8.pick_tile_config``.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache

import torch

from flydsl.compiler.jit_argument import PointerJitArg
from kernels.common.gfx120x_arch import require_gfx120x

from .rdna4_scaled_mm_fp8 import TileConfig, pick_tile_config

KERNEL_NAME = "rdna4_scaled_mm_fp8_fused"

WM = WN = WK = 16
WARP = 32
LDS_PAD = 8
_F8_E4M3_MAX = 448.0
_F8_E5M2_MAX = 57344.0
_SUPPORTED_LORA_RANKS = (0, 8, 16, 32, 64)
_FP8_TORCH = (torch.float8_e4m3fn, torch.float8_e5m2)


@lru_cache(maxsize=128)
def build_scaled_mm_fp8_fused_module(
    out_name: str,
    in_name: str,
    cfg: TileConfig,
    skip_bounds: bool = False,
    lora_rank: int = 0,
    e5m2: bool = False,
    *,
    k_tail: int = 0,
    scale_b_per_n: bool = False,
) -> Callable[..., None]:
    """Compile fused act-quant + FP8 WMMA GEMM, optionally with LoRA epilogue.

    Parameters
    ----------
    out_name:
        ``bfloat16`` / ``float16`` / ``float32`` output dtype name.
    in_name:
        Activation dtype name (``bfloat16`` / ``float16`` / ``float32``).
    cfg:
        Tile configuration (same family as ``rdna4_scaled_mm_fp8``).
    skip_bounds:
        Skip OOB guards when M/N are tile-aligned.
    lora_rank:
        ``0`` disables LoRA fusion; otherwise one of ``{8,16,32,64}``.
    e5m2:
        ``False`` → e4m3 (``cvt_pk_fp8_f32``, max 448, ``Float8E4M3FN`` WMMA).
        ``True`` → e5m2 (``cvt_pk_bf8_f32``, max 57344, ``Float8E5M2`` WMMA).
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import T
    from flydsl.expr.typing import Vector as Vec

    if lora_rank not in _SUPPORTED_LORA_RANKS:
        raise ValueError(f"lora_rank must be one of {_SUPPORTED_LORA_RANKS}, got {lora_rank}")
    if not isinstance(k_tail, int) or not 0 <= k_tail < 16:
        raise ValueError(f"k_tail must be an int in 0..15, got {k_tail!r}")

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
    fp8_max = _F8_E5M2_MAX if e5m2 else _F8_E4M3_MAX
    cvt_pk = fx.rocdl.cvt_pk_bf8_f32 if e5m2 else fx.rocdl.cvt_pk_fp8_f32
    Fp8Ty = fx.Float8E5M2 if e5m2 else fx.Float8E4M3FN
    sig = (
        f"{out_name}_{in_name}_{cfg.name}_bounds{int(skip_bounds)}_R{lora_rank}"
        f"_e5m2{int(e5m2)}_tail{k_tail}_sb{int(scale_b_per_n)}"
    )

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
    # LoRA H staging in LDS: BM * R f32 (reuse A staging after GEMM).
    H_BYTES = BM * max(R, 1) * 4
    H_STRIDE = max(R, 1)

    assert BM == WARPS_M * TM * WM
    assert BN == WARPS_N * TN * WN
    assert BK % WK == 0
    assert CHUNKS_A % THREADS == 0 and CHUNKS_B % THREADS == 0
    if use_lora:
        assert H_BYTES <= max(AS_BYTES, BS_BYTES) or H_BYTES <= AS_BYTES + BS_BYTES

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
        down_g = fx.recast_iter(fx.PointerType.get(InTy.ir_type, LoraDown.address_space), LoraDown)
        up_g = fx.recast_iter(fx.PointerType.get(InTy.ir_type, LoraUp.address_space), LoraUp)

        def as_i32(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, ptr.address_space), ptr)

        alloc = fx.SharedAllocator()
        as_base = alloc.allocate(AS_BYTES)._ptr
        bs_base = alloc.allocate(BS_BYTES)._ptr
        # Extra LDS only when H does not fit in A staging alone.
        if const_expr(use_lora and H_BYTES > AS_BYTES):
            h_base = alloc.allocate(H_BYTES)._ptr
        else:
            h_base = as_base

        sa_view = fx.make_view(
            fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleA.address_space), ScaleA),
            fx.make_layout(1, 1),
        )
        sb_view = fx.make_view(
            fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB),
            fx.make_layout(1, 1),
        )
        scale_a = fx.Float32(sa_view.load()[0])
        if const_expr(not scale_b_per_n):
            scale_b = fx.Float32(sb_view.load()[0])
            scale_ab = scale_a * scale_b
        else:
            sb_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB)
        inv_a = fx.Float32(1.0) / scale_a
        vmax = fx.Float32(fp8_max)
        vmin = fx.Float32(-fp8_max)

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

        def load16_act_quant(grow: fx.Int32, gk: fx.Int32, inb_row: object) -> fx.Vector:
            """Load 16 activations at (grow, gk..gk+15), quantize to packed FP8."""
            elems = []
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
                x = fx.max(fx.min(v * inv_a, vmax), vmin)
                # Match rdna4_fp8_quant: cvt_pk_* expects MLIR Values (bf8 is strict).
                elems.append(x.ir_value())

            words = []
            for w in range_constexpr(4):
                bi = w * 4
                pk = fx.Int32(0).ir_value()
                pk = cvt_pk(T.i32, elems[bi + 0], elems[bi + 1], pk, 0)
                pk = cvt_pk(T.i32, elems[bi + 2], elems[bi + 3], pk, 1)
                words.append(fx.Int32(pk))
            return Vec.from_elements(words, fx.Int32)

        def load16_bytes(elem_ptr: fx.Pointer, byte_idx: fx.Int64, n_valid: fx.Int32) -> fx.Vector:
            # Byte loads. row*K is not 16-aligned when K % 16 != 0.
            bvals = []
            for b in range_constexpr(16):
                take = n_valid > fx.Int32(b)
                off = take.select(byte_idx + fx.Int64(b), fx.Int64(0))
                view = fx.make_view(fx.add_offset(elem_ptr, off), fx.make_layout(1, 1))
                raw = fx.Int32(Vec(view.load())[0]) & fx.Int32(255)
                bvals.append(take.select(raw, fx.Int32(0)))
            words = []
            for w in range_constexpr(4):
                acc = fx.Int32(0)
                for t in range_constexpr(4):
                    acc = acc | (bvals[w * 4 + t] << fx.Int32(8 * t))
                words.append(acc)
            return Vec.from_elements(words, fx.Int32)

        def chunk_n_valid(grow: fx.Int32, gk: fx.Int32, rows: fx.Int32) -> fx.Int32:
            inb = (grow < rows) & (gk < K)
            return inb.select(fx.min(K - gk, fx.Int32(16)), fx.Int32(0))

        def store8x2_lds(lds_ptr: fx.Pointer, byte_off: fx.Int32 | fx.Int64 | int, v4i32: fx.Vector) -> None:
            i32_lds = as_i32(lds_ptr)
            off0 = fx.Int64(byte_off) // fx.Int64(4)
            view0 = fx.make_view(
                fx.add_offset(i32_lds, off0),
                fx.make_layout(2, 1),
            )
            view0.store(Vec.from_elements([fx.Int32(v4i32[0]), fx.Int32(v4i32[1])], fx.Int32))
            view1 = fx.make_view(
                fx.add_offset(i32_lds, off0 + fx.Int64(2)),
                fx.make_layout(2, 1),
            )
            view1.store(Vec.from_elements([fx.Int32(v4i32[2]), fx.Int32(v4i32[3])], fx.Int32))

        def load_a_regs(kbyte0: fx.Int32) -> list[fx.Vector]:
            regs = []
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                grow = m0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                inb_row = grow < M
                regs.append(load16_act_quant(grow, gk, inb_row))
            return regs

        def load_b_regs(kbyte0: fx.Int32) -> list[fx.Vector]:
            regs = []
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                grow = n0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                if const_expr(k_tail == 0):
                    inb = (grow < N) & (gk < K)
                    regs.append(load16_gmem_b(b_g, fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk), inb))
                else:
                    addr = fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk)
                    regs.append(load16_bytes(b_g, addr, chunk_n_valid(grow, gk, N)))
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

        wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WM, WN, WK, Fp8Ty, fx.Float32))

        def wmma_acc(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            a_frag = fx.make_rmem_tensor(8, fx.Int8)
            b_frag = fx.make_rmem_tensor(8, fx.Int8)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
            a_frag.store(Vec(a_v8))
            b_frag.store(Vec(b_v8))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)
            return Vec(c_frag.load())

        def unpack_acc(carried: list[fx.Float32], ai: int) -> fx.Vector:
            base = ai * 8
            return Vec.from_elements(
                [fx.Float32(carried[base + e]) for e in range(8)],
                fx.Float32,
            )

        def pack_acc(acc_vec: fx.Vector) -> list[fx.Float32]:
            return [fx.Float32(acc_vec[e]) for e in range(8)]

        zero = fx.Float32(0.0)
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
                        new_accs.append(wmma_acc(af_cur[ti], bf_cur[tj], acc_list[idx]))
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

        # Optional LoRA: stage H[BM, R] = A_f[m0:m0+BM] @ lora_down.T into LDS.
        if const_expr(use_lora):
            gpu.barrier()
            h_i32 = as_i32(h_base)
            # Zero H
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

            # Accumulate H[local_row, r] over K (one (row,r) per thread stripe).
            # Stripe across BM*R pairs; each pair does a carried K reduction.
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

        # Epilogue: Wave32 column-distributed store + optional LoRA add.
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
                    inb = (m < fx.Int64(M)) & (n < fx.Int64(N))
                    safe_m = inb.select(m, fx.Int64(0))
                    safe_n = inb.select(n, fx.Int64(0))
                    if const_expr(not scale_b_per_n):
                        val = fx.Float32(acc[e]) * scale_ab
                    else:
                        sb_v = fx.make_view(fx.add_offset(sb_ptr, safe_n), fx.make_layout(1, 1)).load()
                        val = fx.Float32(acc[e]) * scale_a * fx.Float32(sb_v[0])
                    if const_expr(use_lora):
                        local_row = inb.select(fx.Int32(m - fx.Int64(m0)), fx.Int32(0))
                        delta = fx.Float32(0.0)
                        for rr in range_constexpr(R):
                            hv = fx.make_view(
                                fx.add_offset(
                                    h_f,
                                    fx.Int64(local_row) * fx.Int64(H_STRIDE) + fx.Int64(rr),
                                ),
                                fx.make_layout(1, 1),
                            ).load()
                            # up[n, r]
                            uv = fx.make_view(
                                fx.add_offset(up_g, safe_n * fx.Int64(R) + fx.Int64(rr)),
                                fx.make_layout(1, 1),
                            ).load()
                            if const_expr(in_name == "float32"):
                                uf = fx.Float32(uv[0])
                            else:
                                uf = InTy(uv[0]).to(fx.Float32)
                            delta = delta + fx.Float32(hv[0]) * uf
                        val = inb.select(val + lora_scale * delta, val)
                    if const_expr(out_name == "float32"):
                        out_v = val
                    elif const_expr(out_name == "float16"):
                        out_v = val.to(fx.Float16)
                    else:
                        out_v = val.to(fx.BFloat16)
                    cidx = m * fx.Int64(N) + n
                    if const_expr(skip_bounds):
                        view = fx.make_view(
                            fx.add_offset(c_ptr, cidx),
                            fx.make_layout(1, 1),
                        )
                        view.store(Vec.from_elements([out_v], OutTy))
                    else:
                        safe = inb.select(cidx, fx.Int64(0))
                        view = fx.make_view(
                            fx.add_offset(c_ptr, safe),
                            fx.make_layout(1, 1),
                        )
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
        stream: fx.Stream = fx.Stream(None),
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


def _fp8_max_for(dtype: torch.dtype) -> float:
    if dtype == torch.float8_e5m2:
        return _F8_E5M2_MAX
    return _F8_E4M3_MAX


def _scale1(s: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Flat 1xf32 scale; skip .to/reshape/contiguous when already hot-path ready.

    Docs: playbook lean host (flat scales / skip redundant contiguous); same
    pattern as flydsl_quant._scale_operand and flydsl_scaled_mm._scale1.
    """
    if s.dtype == torch.float32 and s.device == device and s.is_contiguous() and s.numel() == 1:
        return s
    return s.detach().to(device=device, dtype=torch.float32).reshape(1).contiguous()


def scaled_mm_fp8_fused(
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
    e5m2: bool | None = None,
) -> torch.Tensor:
    """Host wrapper: fused act-quant + FP8 scaled_mm with optional LoRA.

    ``a_f`` is bf16/fp16/fp32 ``[M, K]``; ``b_nk`` is ``float8_e4m3fn`` or
    ``float8_e5m2`` ``[N, K]``. Format is inferred from ``b_nk.dtype`` unless
    ``e5m2`` is set explicitly. Act-quant prologue uses bf8 cvt + max 57344
    when e5m2, else fp8 cvt + max 448.

    LoRA tensors (when provided) match Linear keep: down ``[R, K]``,
    up ``[N, R]``, same dtype as activations. Pass a **list/tuple** of downs/ups
    (and optional scale list) for N adapters in load order — dispatches to
    ``scaled_mm_fp8_fused_multi``.
    """
    require_gfx120x(a_f.device, what="scaled_mm_fp8_fused (gfx120x)")
    # Device LoRA residual: any LoRA count (1..N) uses multi/device residual path (in-kernel
    # LoRA epilogue measured slower than HIP on an otherwise idle gfx120x).
    if (
        lora_down is not None
        or lora_up is not None
        or isinstance(lora_down, (list, tuple))
        or isinstance(lora_up, (list, tuple))
    ):
        return scaled_mm_fp8_fused_multi(
            a_f,
            b_nk,
            scale_a,
            scale_b,
            out_dtype=out_dtype,
            lora_downs=lora_down,
            lora_ups=lora_up,
            lora_scales=lora_scale,
            stream=stream,
            e5m2=e5m2,
        )

    import flydsl.compiler as flyc
    import flydsl.expr as fx

    # LoRA already diverted above; this path is base fused act-quant + scaled_mm only.
    if a_f.dim() != 2 or b_nk.dim() != 2:
        raise ValueError("scaled_mm_fp8_fused expects 2D operands")
    if b_nk.dtype not in _FP8_TORCH:
        raise ValueError(f"b_nk must be float8_e4m3fn or float8_e5m2, got {b_nk.dtype}")
    if e5m2 is None:
        e5m2 = b_nk.dtype == torch.float8_e5m2
    elif e5m2 != (b_nk.dtype == torch.float8_e5m2):
        want = torch.float8_e5m2 if e5m2 else torch.float8_e4m3fn
        raise ValueError(f"e5m2={e5m2} disagrees with b_nk.dtype={b_nk.dtype} (want {want})")
    m, k = a_f.shape
    n = b_nk.shape[0]
    if b_nk.shape[1] != k:
        raise ValueError(f"K mismatch: a_f {k} vs b_nk {b_nk.shape[1]}")
    if m == 0 or n == 0:
        return torch.empty((m, n), dtype=out_dtype, device=a_f.device)
    k_tail = int(k) % 16

    # Soft-contig for pointer ABI (transpose/slice views otherwise OOB).
    from kernels.common.gfx120x_pad import ensure_contiguous

    a_f = ensure_contiguous(a_f, stream=stream)
    b_nk = ensure_contiguous(b_nk, stream=stream)

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

    cfg = pick_tile_config(m, n, k, device=a_f.device)
    skip_bounds = m % cfg.bm == 0 and n % cfg.bn == 0
    scale_a = _scale1(scale_a, a_f.device)
    if scale_b.numel() == 1:
        scale_b = _scale1(scale_b, a_f.device)
        scale_b_per_n = False
    else:
        scale_b = scale_b.detach().to(device=a_f.device, dtype=torch.float32).reshape(-1).contiguous()
        scale_b_per_n = True
        if scale_b.numel() != n:
            raise ValueError(f"scale_b must be a scalar or [N]={n}, got {scale_b.numel()}")
    if int(k) == 0:
        return torch.zeros((m, n), dtype=out_dtype, device=a_f.device)
    launch = build_scaled_mm_fp8_fused_module(
        out_name, in_name, cfg, skip_bounds, 0, e5m2, k_tail=k_tail, scale_b_per_n=scale_b_per_n
    )
    out = torch.empty((m, n), dtype=out_dtype, device=a_f.device)

    # Dummy 1-element LoRA buffers so the launcher signature stays fixed (rank=0).
    lora_down_buf = torch.zeros(1, dtype=a_f.dtype, device=a_f.device)
    lora_up_buf = torch.zeros(1, dtype=a_f.dtype, device=a_f.device)
    lora_scale_buf = torch.zeros(1, dtype=torch.float32, device=a_f.device)

    def _ptr(t: torch.Tensor) -> PointerJitArg:
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    args = (
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
    )
    if stream is None:
        launch(*args)
    else:
        launch(*args, stream)
    return out


def _lora_scale_norm(scale: float | torch.Tensor, device: torch.device) -> float | torch.Tensor:
    """Python float stays float; tensor scales stay on device (no .item())."""
    if isinstance(scale, torch.Tensor):
        return _scale1(scale, device)
    return float(scale)


def _pack_lora_adapters(
    lora_downs: torch.Tensor | Sequence[torch.Tensor] | None,
    lora_ups: torch.Tensor | Sequence[torch.Tensor] | None,
    lora_scales: float | torch.Tensor | Sequence[float | torch.Tensor] | None,
    *,
    k: int,
    n: int,
) -> list[tuple[torch.Tensor, torch.Tensor, float | torch.Tensor]]:
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

    packs: list[tuple[torch.Tensor, torch.Tensor, float | torch.Tensor]] = []
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
        packs.append((down, up, _lora_scale_norm(scale, down.device)))
    return packs


def _lora_residual(
    a_f: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    lora_scale: float | torch.Tensor,
    out_dtype: torch.dtype,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Live LoRA residual via gfx120x GEMM: (x @ down.T) @ up.T."""
    from kernels.common.gfx120x_row_bias import mul_by_scale1
    from kernels.gemm.rdna4_fused_mlp_nmajor import gemm_bf16_nmajor_lds

    compute = a_f.dtype if a_f.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    down = lora_down if lora_down.dtype == compute else lora_down.to(dtype=compute)
    up = lora_up if lora_up.dtype == compute else lora_up.to(dtype=compute)
    act = a_f if a_f.dtype == compute else a_f.to(dtype=compute)
    # Fold scale into up on device (tensor scales: no .item(); float 1.0: skip).
    if isinstance(lora_scale, torch.Tensor):
        up = mul_by_scale1(up, lora_scale, stream=stream)
    elif float(lora_scale) != 1.0:
        scale_buf = torch.tensor([float(lora_scale)], device=a_f.device, dtype=torch.float32)
        up = mul_by_scale1(up, scale_buf, stream=stream)
    hidden = gemm_bf16_nmajor_lds(act, down, stream=stream)
    delta = gemm_bf16_nmajor_lds(hidden, up, stream=stream)
    return delta if delta.dtype == out_dtype else delta.to(dtype=out_dtype)


def _apply_lora_inplace(
    out: torch.Tensor,
    a_f: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    lora_scale: float | torch.Tensor,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Accumulate scale*(x@down.T)@up.T into ``out`` via device GEMM + add_same."""
    from kernels.common.gfx120x_row_bias import add_same

    delta = _lora_residual(a_f, lora_down, lora_up, lora_scale, out.dtype, stream=stream)
    summed = add_same(out, delta, stream=stream)
    if stream is None:
        out.copy_(summed)
    else:
        with torch.cuda.stream(stream):
            out.copy_(summed)


def scaled_mm_fp8_fused_multi(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_downs: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_ups: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_scales: float | torch.Tensor | Sequence[float | torch.Tensor] | None = None,
    stream: torch.cuda.Stream | None = None,
    e5m2: bool | None = None,
) -> torch.Tensor:
    """N-adapter fused path: one quantized base GEMM, then device LoRA residuals.

    **Unbounded N:** any number of adapters (no fixed max). Base fused
    quant+mm runs once; each adapter applies ``scale * (x @ down.T) @ up.T``
    in **load order** via ``gemm_bf16_nmajor_lds`` + ``add_same``.

    * **N=0** (no LoRA): omit ``lora_downs``/``lora_ups`` (and leave
      ``lora_scales=None``). Delegates to plain ``scaled_mm_fp8_fused``.
      Passing ``lora_scales`` alone without downs/ups raises.
    * **N≥1**: base fused (no in-kernel LoRA) + device residuals in pack
      order. Omitted ``lora_scales`` defaults to ``1.0`` per adapter.

    Args accept a single tensor or a sequence of tensors for downs/ups; scales
    may be a scalar (broadcast) or a per-adapter sequence.
    """
    require_gfx120x(a_f.device, what="scaled_mm_fp8_fused_multi (gfx120x)")
    if a_f.dim() != 2 or b_nk.dim() != 2:
        raise ValueError("scaled_mm_fp8_fused_multi expects 2D operands")
    m, k = a_f.shape
    n = b_nk.shape[0]
    if b_nk.shape[1] != k:
        raise ValueError(f"K mismatch: a_f {k} vs b_nk {b_nk.shape[1]}")
    if m == 0 or n == 0:
        return torch.empty((m, n), dtype=out_dtype, device=a_f.device)
    if int(k) == 0:
        _pack_lora_adapters(lora_downs, lora_ups, lora_scales, k=k, n=n)
        return torch.zeros((m, n), dtype=out_dtype, device=a_f.device)

    packs = _pack_lora_adapters(lora_downs, lora_ups, lora_scales, k=k, n=n)

    if len(packs) == 0:
        return scaled_mm_fp8_fused(
            a_f,
            b_nk,
            scale_a,
            scale_b,
            out_dtype=out_dtype,
            stream=stream,
            e5m2=e5m2,
        )

    # Device-residual policy (gfx120x): always base fused (no in-kernel LoRA) +
    # gemm_bf16_nmajor_lds/add_same residuals in load order. In-kernel LoRA
    # epilogue measured slower than HIP quant+mm+host-LoRA on R9700; device
    # residual after fused base wins.
    out = scaled_mm_fp8_fused(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
        stream=stream,
        e5m2=e5m2,
    )
    for down, up, scale in packs:
        _apply_lora_inplace(out, a_f, down, up, scale, stream=stream)
    return out
