# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Per-1x32 MXFP4 block-scaled GEMM for gfx120x (RDNA4).

Computes ``C[m, n] = sum_kb scaleA[m, kb] * scaleB[n, kb] *
(A_fp4[m, kb*32:(kb+1)*32] @ B_fp4[n, kb*32:(kb+1)*32].T)``. A and B
are uint8, two E2M1 values per byte, low nibble = even K. E8M0 scales
are applied in software after the block product.

gfx120x has no fp4 WMMA. Every E2M1 code is exact in e4m3, so each
nibble is rewritten to that byte and the product uses the fp8 WMMA.
Quantization lives in ``kernels.quant.rdna4_mxfp4_e2m1``. K is a
multiple of 32. The tile matches ``rdna4_mxfp8_block_gemm``.
"""

from collections.abc import Callable
from functools import lru_cache

import torch

from kernels.common.gfx120x_arch import require_gfx120x

KERNEL_NAME = "rdna4_mxfp4_block_gemm"

# E2M1 nibble -> OCP e4m3fn byte. Same order as rdna4_mxfp4_e2m1._LUT.
_E4M3_FROM_E2M1 = (0, 48, 56, 60, 64, 68, 72, 76, 128, 176, 184, 188, 192, 196, 200, 204)

WM = WN = WK = 16
WARP = 32
LDS_PAD = 8
_GROUP = 32
_BM = _BN = 64
_BK = 32
_WARPS_M = _WARPS_N = 2
_TM = _TN = 2
_THREADS = _WARPS_M * _WARPS_N * WARP


def _e8m0_to_f32(exponent):
    """Match ``kernels.quant.rdna4_mxfp8_e8m0._e8m0_to_f32``."""
    import flydsl.expr as fx

    bits = exponent << fx.Int32(23)
    bits = (exponent == fx.Int32(0)).select(fx.Int32(0x00400000), bits)
    bits = (exponent == fx.Int32(0xFF)).select(fx.Int32(0x7F800001), bits)
    return bits.bitcast(fx.Float32)


@lru_cache(maxsize=16)
def build_mxfp4_block_gemm_module(out_name: str, skip_bounds: bool = False) -> Callable[..., None]:
    """Compile one (out_dtype) MXFP4 block GEMM.

    A/B are packed E2M1, two values per byte. ``K`` is the logical value
    count and a multiple of 32. Scales are uint8 E8M0.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import Vector as Vec

    OutTy = {"bfloat16": fx.BFloat16, "float16": fx.Float16, "float32": fx.Float32}[out_name]
    Fp8Ty = fx.Float8E4M3FN
    BM, BN, BK = _BM, _BN, _BK
    WARPS_M, WARPS_N = _WARPS_M, _WARPS_N
    TM, TN = _TM, _TN
    THREADS = _THREADS
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

    assert BM == WARPS_M * TM * WM
    assert BN == WARPS_N * TN * WN
    assert BK == _GROUP and BK % WK == 0
    assert CHUNKS_A % THREADS == 0 and CHUNKS_B % THREADS == 0

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def gemm_kernel(
        A: fx.Pointer,
        Bnk: fx.Pointer,
        C: fx.Pointer,
        ScaleA: fx.Pointer,
        ScaleB: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Int32,
        ScaleCols: fx.Int32,
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

        a_g = fx.recast_iter(fx.PointerType.get(fx.Int8.ir_type, A.address_space), A)
        b_g = fx.recast_iter(fx.PointerType.get(fx.Int8.ir_type, Bnk.address_space), Bnk)
        c_ptr = fx.recast_iter(fx.PointerType.get(OutTy.ir_type, C.address_space), C)
        sa_g = fx.recast_iter(fx.PointerType.get(fx.Uint8.ir_type, ScaleA.address_space), ScaleA)
        sb_g = fx.recast_iter(fx.PointerType.get(fx.Uint8.ir_type, ScaleB.address_space), ScaleB)

        # LDS via allocate(nbytes) — same pattern as rdna4_scaled_mm_fp8.
        alloc = fx.SharedAllocator()
        as_base = alloc.allocate(AS_BYTES)._ptr
        bs_base = alloc.allocate(BS_BYTES)._ptr

        def as_i32(elem_ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, elem_ptr.address_space), elem_ptr)

        def nibble_to_e4m3(nibble):
            byte = fx.Int32(_E4M3_FROM_E2M1[0])
            for index, bits in enumerate(_E4M3_FROM_E2M1):
                byte = (nibble == fx.Int32(index)).select(fx.Int32(bits), byte)
            return byte

        def load16_fp4(elem_ptr: fx.Pointer, row, logical_k, inb) -> fx.Vector:
            """16 E2M1 values (8 packed bytes) rewritten as 16 e4m3 bytes."""
            packed_k = K // fx.Int32(2)
            byte_idx = fx.Int64(row) * fx.Int64(packed_k) + fx.Int64(logical_k) // fx.Int64(2)
            safe = inb.select(byte_idx, fx.Int64(0))
            view = fx.make_view(fx.add_offset(as_i32(elem_ptr), safe // fx.Int64(4)), fx.make_layout(2, 1))
            raw = Vec(view.load())
            z = fx.Int32(0)
            elems = []
            for word_i in range_constexpr(2):
                word = inb.select(fx.Int32(raw[word_i]), z)
                for shift in range_constexpr(4):
                    byte = (word >> fx.Int32(8 * shift)) & fx.Int32(255)
                    elems.append(nibble_to_e4m3(byte & fx.Int32(15)))
                    elems.append(nibble_to_e4m3((byte >> fx.Int32(4)) & fx.Int32(15)))
            return _pack_i8_16(elems)

        def store8x2_lds(lds_ptr: fx.Pointer, byte_off, v4i32: fx.Vector) -> None:
            i32_lds = as_i32(lds_ptr)
            off0 = fx.Int64(byte_off) // fx.Int64(4)
            view0 = fx.make_view(fx.add_offset(i32_lds, off0), fx.make_layout(2, 1))
            view0.store(Vec.from_elements([fx.Int32(v4i32[0]), fx.Int32(v4i32[1])], fx.Int32))
            view1 = fx.make_view(fx.add_offset(i32_lds, off0 + fx.Int64(2)), fx.make_layout(2, 1))
            view1.store(Vec.from_elements([fx.Int32(v4i32[2]), fx.Int32(v4i32[3])], fx.Int32))

        def _pack_i8_16(elems: list) -> fx.Vector:
            words = []
            for w in range_constexpr(4):
                b = w * 4
                word = fx.Int32(0)
                for s in range_constexpr(4):
                    word = word | ((fx.Int32(elems[b + s]) & fx.Int32(255)) << fx.Int32(8 * s))
                words.append(word)
            return Vec.from_elements(words, fx.Int32)

        def load_a_regs(kbyte0: fx.Int32) -> list:
            regs = []
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                grow = m0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                inb = (grow < M) & (gk < K)
                regs.append(load16_fp4(a_g, grow, gk, inb))
            return regs

        def load_b_regs(kbyte0: fx.Int32) -> list:
            regs = []
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                grow = n0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                inb = (grow < N) & (gk < K)
                regs.append(load16_fp4(b_g, grow, gk, inb))
            return regs

        def store_a_regs(regs) -> None:
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                local_row = c // fx.Int32(CHUNKS_PER_ROW)
                local_k = (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                store8x2_lds(as_base, local_row * fx.Int32(STRIDE) + local_k, regs[i])

        def store_b_regs(regs) -> None:
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                local_row = c // fx.Int32(CHUNKS_PER_ROW)
                local_k = (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                store8x2_lds(bs_base, local_row * fx.Int32(STRIDE) + local_k, regs[i])

        def load_frag_lds(lds_ptr: fx.Pointer, row: fx.Int32, kbyte: fx.Int32) -> fx.Vector:
            off = fx.Int64(row) * fx.Int64(STRIDE) + fx.Int64(kbyte) + fx.Int64(klane) * fx.Int64(8)
            view = fx.make_view(fx.add_offset(as_i32(lds_ptr), off // fx.Int64(4)), fx.make_layout(2, 1))
            return Vec(view.load()).bitcast(fx.Int8)

        wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WM, WN, WK, Fp8Ty, fx.Float32))

        def wmma_acc(a_v8, b_v8, c_v8) -> fx.Vector:
            a_frag = fx.make_rmem_tensor(8, fx.Int8)
            b_frag = fx.make_rmem_tensor(8, fx.Int8)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
            a_frag.store(Vec(a_v8))
            b_frag.store(Vec(b_v8))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)
            return Vec(c_frag.load())

        def unpack_acc(carried, ai: int) -> fx.Vector:
            base = ai * 8
            return Vec.from_elements([fx.Float32(carried[base + e]) for e in range(8)], fx.Float32)

        def pack_acc(acc_vec) -> list:
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
            # Fresh partial for this E8M0 block; scale after the two WMMA K=16 steps.
            acc_list = [Vec.from_elements([zero] * 8, fx.Float32) for _ in range_constexpr(N_ACC)]

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

            # Software E8M0: scale the block partial, then fold into the running sum.
            kb_scale = kb_i // fx.Int32(_GROUP)
            scaled = []
            for ti in range_constexpr(TM):
                for tj in range_constexpr(TN):
                    idx = ti * TN + tj
                    row_base = m0 + wm * fx.Int32(TM * WM) + fx.Int32(ti * WM) + klane * fx.Int32(8)
                    col = n0 + wn * fx.Int32(TN * WN) + fx.Int32(tj * WN) + lane16
                    elems = []
                    for e in range_constexpr(8):
                        m = row_base + fx.Int32(e)
                        n = col
                        sa_off = fx.Int64(m) * fx.Int64(ScaleCols) + fx.Int64(kb_scale)
                        sb_off = fx.Int64(n) * fx.Int64(ScaleCols) + fx.Int64(kb_scale)
                        sa_inb = (m < M) & (kb_scale < ScaleCols)
                        sb_inb = (n < N) & (kb_scale < ScaleCols)
                        sa_safe = sa_inb.select(sa_off, fx.Int64(0))
                        sb_safe = sb_inb.select(sb_off, fx.Int64(0))
                        sa_v = fx.make_view(fx.add_offset(sa_g, sa_safe), fx.make_layout(1, 1))
                        sb_v = fx.make_view(fx.add_offset(sb_g, sb_safe), fx.make_layout(1, 1))
                        sa_f = _e8m0_to_f32(fx.Int32(sa_v.load()[0]))
                        sb_f = _e8m0_to_f32(fx.Int32(sb_v.load()[0]))
                        scale = sa_f * sb_f
                        prev = fx.Float32(carried[idx * 8 + e])
                        elems.append(prev + fx.Float32(acc_list[idx][e]) * scale)
                    scaled.extend(elems)

            if has_next:
                gpu.barrier()
                store_a_regs(a_regs)
                store_b_regs(b_regs)
                gpu.barrier()

            results = yield scaled

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
                    val = fx.Float32(acc[e])
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
        A: fx.Pointer,
        Bnk: fx.Pointer,
        C: fx.Pointer,
        ScaleA: fx.Pointer,
        ScaleB: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Int32,
        ScaleCols: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_n = (fx.Int64(N) + fx.Int64(BN - 1)) // fx.Int64(BN)
        grid_m = (fx.Int64(M) + fx.Int64(BM - 1)) // fx.Int64(BM)
        gemm_kernel(A, Bnk, C, ScaleA, ScaleB, M, N, K, ScaleCols, fx.Int32(grid_n), fx.Int32(grid_m)).launch(
            grid=(grid_n, grid_m, 1), block=(THREADS, 1, 1), stream=stream
        )

    gemm_kernel.__name__ = f"{KERNEL_NAME}_{out_name}_bounds{int(skip_bounds)}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{out_name}"
    return launch


def mxfp4_block_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """MXFP4 block GEMM. ``a``/``b`` are packed E2M1 ``[M, K//2]`` / ``[N, K//2]``; scales are uint8 E8M0."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.tensor_shim import _run_compiled

    require_gfx120x(a.device, what="mxfp4_block_gemm (gfx120x)")
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        raise ValueError(f"A/B must be uint8 packed E2M1, got {a.dtype}, {b.dtype}")
    if scale_a.dtype != torch.uint8 or scale_b.dtype != torch.uint8:
        raise ValueError(f"scales must be uint8 E8M0, got {scale_a.dtype}, {scale_b.dtype}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"A/B must be rank-2, got {a.shape}, {b.shape}")
    m, packed_k = a.shape
    n, packed_k_b = b.shape
    if packed_k != packed_k_b:
        raise ValueError(f"packed K mismatch: A {a.shape}, B {b.shape}")
    if m == 0 or n == 0 or packed_k == 0:
        return torch.zeros((m, n), device=a.device, dtype=out_dtype)
    if packed_k % 16 != 0:
        raise ValueError(f"MXFP4 GEMM K must be a multiple of 32 values ({packed_k * 2}), got packed K {packed_k}")
    k = packed_k * 2
    need_scale_cols = k // _GROUP
    if tuple(scale_a.shape) != (m, need_scale_cols) or tuple(scale_b.shape) != (n, need_scale_cols):
        raise ValueError(
            f"scale shapes must be ({m}, {need_scale_cols}) and ({n}, {need_scale_cols}), "
            f"got {tuple(scale_a.shape)}, {tuple(scale_b.shape)}"
        )
    scale_cols = need_scale_cols
    from kernels.common.gfx120x_pad import ensure_contiguous

    a = ensure_contiguous(a, stream=stream)
    b = ensure_contiguous(b, stream=stream)
    scale_a = ensure_contiguous(scale_a, stream=stream)
    scale_b = ensure_contiguous(scale_b, stream=stream)
    out_name = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[out_dtype]
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    skip_bounds = (m % _BM == 0) and (n % _BN == 0)
    launch = build_mxfp4_block_gemm_module(out_name, skip_bounds=skip_bounds)

    def _ptr(t: torch.Tensor):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    _run_compiled(
        launch,
        _ptr(a.view(torch.uint8)),
        _ptr(b.view(torch.uint8)),
        _ptr(out.view(torch.uint8)),
        _ptr(scale_a),
        _ptr(scale_b),
        m,
        n,
        k,
        scale_cols,
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out
