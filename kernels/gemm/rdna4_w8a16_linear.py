# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""W8A16 linear GEMM for gfx120x (bf16/fp16 acts × 8-bit weights, float WMMA).

Weights may be ``int8``, ``float8_e4m3fn``, or ``float8_e5m2``. Activations stay
bf16/fp16. Pattern: cast 8-bit weights → bf16/fp16 in-register (same float WMMA
path for all three); apply ``w_scale`` in the epilogue. FP8 format parity with
``rdna4_scaled_mm_fp8`` / ``rdna4_scaled_mm_fp8_fused`` (e4m3 max 448 vs e5m2
max 57344). Does **not** require the gfx120x iu8 WMMA atom.

Measured R9700: tiny/mid faster than HIP; large slower (~×0.57). For large-K
iu8 WMMA use ``rdna4_int8_linear`` explicitly (no size/K auto router).

Public: ``build_w8a16_linear_module``, ``pick_tile_config``, ``w8a16_gemm``,
``w8a16_linear``.
"""

# NOTE: do NOT add postponed-eval annotations future-import — @fx.struct / Array sizes need live types.

from collections.abc import Callable  # noqa: E402
from functools import lru_cache

from flydsl.compiler.jit_argument import PointerJitArg  # noqa: E402
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.gemm.rdna4_tile import TileConfig

KERNEL_NAME = "rdna4_w8a16_linear"

WM = WN = WK = 16
WARP = 32
LDS_PAD = 8
_DEFAULT_WGPS = 32

# Mirror iu8 int8 linear launch tiles.
_CFG_128_128_64 = TileConfig(128, 128, 64, 4, 2, 2, 4)
_CFG_256_128_64 = TileConfig(256, 128, 64, 4, 2, 4, 4)
_CFG_64_64_64 = TileConfig(64, 64, 64, 2, 2, 2, 2)


_WGP_COUNT_CACHE: dict[int, int] = {}


def _wgp_count(device=None) -> int:
    """Cached CU/WGP count for ``device`` (not always cuda:0)."""
    try:
        import torch  # noqa: E402

        if device is None:
            idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        else:
            idx = int(device.index) if getattr(device, "index", None) is not None else int(torch.cuda.current_device())
        if idx not in _WGP_COUNT_CACHE:
            _WGP_COUNT_CACHE[idx] = int(torch.cuda.get_device_properties(idx).multi_processor_count) or _DEFAULT_WGPS
        return _WGP_COUNT_CACHE[idx]
    except Exception:  # noqa: BLE001
        return _DEFAULT_WGPS


def pick_tile_config(M: int, N: int, K: int, wgps: int | None = None, device=None) -> TileConfig:
    """Return the W8A16 tile for this M, N, and K."""
    if wgps is None:
        wgps = _wgp_count(device)
    blocks_128 = ((M + 127) // 128) * ((N + 127) // 128)
    skinny = M <= 64 or N <= 64
    if not skinny and blocks_128 >= wgps:
        return _CFG_256_128_64
    if not skinny and (M >= 128 or N >= 128):
        return _CFG_128_128_64
    return _CFG_64_64_64


@lru_cache(maxsize=64)
def build_w8a16_linear_module(
    act_name: str,
    out_name: str,
    cfg: TileConfig,
    skip_bounds: bool = False,
    w_scale_per_n: bool = True,
    w_dtype_name: str = "int8",
    k_tail: int = 0,
) -> Callable[..., None]:
    """Compile W8A16 linear: bf16/fp16 A + 8-bit B cast-in-reg; w_scale in epilogue.

    ``w_dtype_name`` is ``"int8"``, ``"float8_e4m3fn"``, or ``"float8_e5m2"``.
    Does NOT require iu8 WMMA patch.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import T
    from flydsl.expr.typing import Vector as Vec

    ActTy = {
        "bfloat16": fx.BFloat16,
        "float16": fx.Float16,
    }[act_name]
    OutTy = {
        "bfloat16": fx.BFloat16,
        "float16": fx.Float16,
        "float32": fx.Float32,
    }[out_name]
    # WMMA always bf16/fp16 frags (acts match WmmaTy; 8-bit weights cast in-reg).
    WmmaTy = fx.BFloat16 if act_name == "bfloat16" else fx.Float16
    if w_dtype_name not in ("int8", "float8_e4m3fn", "float8_e5m2"):
        raise ValueError(f"w_dtype_name must be int8/float8_e4m3fn/float8_e5m2, got {w_dtype_name!r}")
    # FP8→f32 must use ROCDL cvt (arith.extf / f8 bitcast does not lower to LLVM).
    # Parity with rdna4_fp8_quant / scaled_mm_fp8: e4m3 → cvt_f32_fp8; e5m2 → cvt_f32_bf8.
    _is_fp8_w = w_dtype_name != "int8"
    _cvt_f32_f8 = fx.rocdl.cvt_f32_bf8 if w_dtype_name == "float8_e5m2" else fx.rocdl.cvt_f32_fp8

    BM, BN, BK = cfg.bm, cfg.bn, cfg.bk
    WARPS_M, WARPS_N = cfg.warps_m, cfg.warps_n
    TM, TN = cfg.tm, cfg.tn
    THREADS = cfg.threads
    N_ACC = TM * TN
    K_STEPS = BK // WK

    # A is 2-byte elems; B is 1-byte (int8 or FP8). Separate strides / chunking.
    STRIDE_A = BK + LDS_PAD  # elements (bf16/fp16)
    STRIDE_B = BK + LDS_PAD  # elements (int8/fp8) == bytes
    AS_BYTES = BM * STRIDE_A * 2
    BS_BYTES = BN * STRIDE_B

    # 16B vector loads: A → 8 elems, B → 16 elems
    CHUNKS_PER_ROW_A = BK // 8
    CHUNKS_PER_ROW_B = BK // 16
    CHUNKS_A = BM * CHUNKS_PER_ROW_A
    CHUNKS_B = BN * CHUNKS_PER_ROW_B
    PTA = CHUNKS_A // THREADS
    PTB = CHUNKS_B // THREADS
    GROUP_M = 4

    assert BM == WARPS_M * TM * WM
    assert BN == WARPS_N * TN * WN
    assert BK % WK == 0
    assert BK % 16 == 0
    assert CHUNKS_A % THREADS == 0 and CHUNKS_B % THREADS == 0
    assert PTA >= 1 and PTB >= 1

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def gemm_kernel(
        A: fx.Pointer,
        Bnk: fx.Pointer,
        C: fx.Pointer,
        ScaleB: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Int32,
        BlocksN: fx.Int32,
        BlocksM: fx.Int32,
    ) -> None:
        wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WM, WN, WK, WmmaTy, fx.Float32))

        def wmma_f16(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            """Float WMMA (bf16 or fp16) via ``fx.gemm`` on rank-1 rmem fragments."""
            a_frag = fx.make_rmem_tensor(8, WmmaTy)
            b_frag = fx.make_rmem_tensor(8, WmmaTy)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
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

        a_g = fx.recast_iter(fx.PointerType.get(ActTy.ir_type, A.address_space), A)
        b_g = fx.recast_iter(fx.PointerType.get(fx.Int8.ir_type, Bnk.address_space), Bnk)
        c_ptr = fx.recast_iter(fx.PointerType.get(OutTy.ir_type, C.address_space), C)
        sb_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB)

        def as_i32(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, ptr.address_space), ptr)

        alloc = fx.SharedAllocator()
        as_base = alloc.allocate(AS_BYTES)._ptr
        bs_base = alloc.allocate(BS_BYTES)._ptr

        def load16_gmem_bytes(elem_ptr: fx.Pointer, byte_idx: fx.Int32 | fx.Int64 | int, inb: object) -> fx.Vector:
            """Load 16B as 4×i32 from a byte offset into an element-typed pointer."""
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

        def store8x2_lds(lds_ptr: fx.Pointer, byte_off: fx.Int32 | fx.Int64 | int, v4i32: fx.Vector) -> None:
            i32_lds = as_i32(lds_ptr)
            off0 = fx.Int64(byte_off) // fx.Int64(4)
            view0 = fx.make_view(fx.add_offset(i32_lds, off0), fx.make_layout(2, 1))
            view0.store(Vec.from_elements([fx.Int32(v4i32[0]), fx.Int32(v4i32[1])], fx.Int32))
            view1 = fx.make_view(fx.add_offset(i32_lds, off0 + fx.Int64(2)), fx.make_layout(2, 1))
            view1.store(Vec.from_elements([fx.Int32(v4i32[2]), fx.Int32(v4i32[3])], fx.Int32))

        def _pack_act8(elems: list) -> fx.Vector:
            words = []
            for w in range_constexpr(4):
                lo = fx.Int32(elems[w * 2].bitcast(fx.Int16)) & fx.Int32(0xFFFF)
                hi = fx.Int32(elems[w * 2 + 1].bitcast(fx.Int16)) & fx.Int32(0xFFFF)
                words.append(lo | (hi << fx.Int32(16)))
            return Vec.from_elements(words, fx.Int32)

        def _pack_i8_16(elems: list) -> fx.Vector:
            words = []
            for w in range_constexpr(4):
                b = w * 4
                word = fx.Int32(0)
                for s in range_constexpr(4):
                    word = word | ((fx.Int32(elems[b + s]) & fx.Int32(255)) << fx.Int32(8 * s))
                words.append(word)
            return Vec.from_elements(words, fx.Int32)

        def load_a_regs(k0: fx.Int32) -> list[fx.Vector]:
            """Coop-load A bf16/fp16 tiles: 8 elems (16B) per chunk along K."""
            if const_expr(k_tail != 0):
                # A 16-byte load that starts before K and ends after it reads the
                # next row. Odd K loads one element and zero-fills the rest.
                regs = []
                for i in range_constexpr(PTA):
                    c = tid * fx.Int32(PTA) + fx.Int32(i)
                    grow = m0 + c // fx.Int32(CHUNKS_PER_ROW_A)
                    gk = k0 + (c % fx.Int32(CHUNKS_PER_ROW_A)) * fx.Int32(8)
                    elems = []
                    for e in range_constexpr(8):
                        one = ActTy(0)
                        gk_e = gk + fx.Int32(e)
                        if (grow < M) & (gk_e < K):
                            idx = fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk_e)
                            one = Vec(fx.make_view(fx.add_offset(a_g, idx), fx.make_layout(1, 1)).load())[0]
                        elems.append(one)
                    regs.append(_pack_act8(elems))
                return regs
            regs = []
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                grow = m0 + c // fx.Int32(CHUNKS_PER_ROW_A)
                gk = k0 + (c % fx.Int32(CHUNKS_PER_ROW_A)) * fx.Int32(8)
                inb = (grow < M) & (gk < K)
                # byte offset into ActTy buffer
                byte_idx = (fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk)) * fx.Int64(2)
                regs.append(load16_gmem_bytes(a_g, byte_idx, inb))
            return regs

        def load_b_regs(k0: fx.Int32) -> list[fx.Vector]:
            """Coop-load B 8-bit tiles: 16 elems (16B) per chunk along K."""
            if const_expr(k_tail != 0):
                regs = []
                for i in range_constexpr(PTB):
                    c = tid * fx.Int32(PTB) + fx.Int32(i)
                    grow = n0 + c // fx.Int32(CHUNKS_PER_ROW_B)
                    gk = k0 + (c % fx.Int32(CHUNKS_PER_ROW_B)) * fx.Int32(16)
                    elems = []
                    for e in range_constexpr(16):
                        one = fx.Int8(0)
                        gk_e = gk + fx.Int32(e)
                        if (grow < N) & (gk_e < K):
                            idx = fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk_e)
                            one = Vec(fx.make_view(fx.add_offset(b_g, idx), fx.make_layout(1, 1)).load())[0]
                        elems.append(one)
                    regs.append(_pack_i8_16(elems))
                return regs
            regs = []
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                grow = n0 + c // fx.Int32(CHUNKS_PER_ROW_B)
                gk = k0 + (c % fx.Int32(CHUNKS_PER_ROW_B)) * fx.Int32(16)
                inb = (grow < N) & (gk < K)
                byte_idx = fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk)
                regs.append(load16_gmem_bytes(b_g, byte_idx, inb))
            return regs

        def store_a_regs(regs: list[fx.Vector]) -> None:
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                local_row = c // fx.Int32(CHUNKS_PER_ROW_A)
                local_k = (c % fx.Int32(CHUNKS_PER_ROW_A)) * fx.Int32(8)
                # A LDS in bytes: (row * STRIDE_A + k) * 2
                byte_off = (local_row * fx.Int32(STRIDE_A) + local_k) * fx.Int32(2)
                store8x2_lds(as_base, byte_off, regs[i])

        def store_b_regs(regs: list[fx.Vector]) -> None:
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                local_row = c // fx.Int32(CHUNKS_PER_ROW_B)
                local_k = (c % fx.Int32(CHUNKS_PER_ROW_B)) * fx.Int32(16)
                byte_off = local_row * fx.Int32(STRIDE_B) + local_k
                store8x2_lds(bs_base, byte_off, regs[i])

        def load_frag_a(row: fx.Int32, kbyte: fx.Int32) -> fx.Vector:
            """Load 8×ActTy (=16B) WMMA A fragment from LDS."""
            off_elems = fx.Int64(row) * fx.Int64(STRIDE_A) + fx.Int64(kbyte) + fx.Int64(klane) * fx.Int64(8)
            off_bytes = off_elems * fx.Int64(2)
            i32_lds = as_i32(as_base)
            view = fx.make_view(
                fx.add_offset(i32_lds, off_bytes // fx.Int64(4)),
                fx.make_layout(4, 1),
            )
            return Vec(view.load()).bitcast(ActTy)

        def load_frag_b(col: fx.Int32, kbyte: fx.Int32) -> fx.Vector:
            """Load 8×w8 from LDS, cast to WmmaTy in VGPR (no w_scale; epi-fold).

            int8: sitofp → WmmaTy. FP8 e4m3/e5m2: ROCDL ``cvt_f32_fp8`` /
            ``cvt_f32_bf8`` on packed i32 words (same as ``rdna4_fp8_quant``
            dequant), then → WmmaTy. All share the float WMMA path.
            """
            off = fx.Int64(col) * fx.Int64(STRIDE_B) + fx.Int64(kbyte) + fx.Int64(klane) * fx.Int64(8)
            i32_lds = as_i32(bs_base)
            view = fx.make_view(
                fx.add_offset(i32_lds, off // fx.Int64(4)),
                fx.make_layout(2, 1),
            )
            raw = Vec(view.load())  # 2×i32 = 8×w8
            if const_expr(not _is_fp8_w):
                b_i8 = raw.bitcast(fx.Int8)
                e0 = fx.Int8(b_i8[0]).to(fx.Float32).to(WmmaTy)
                e1 = fx.Int8(b_i8[1]).to(fx.Float32).to(WmmaTy)
                e2 = fx.Int8(b_i8[2]).to(fx.Float32).to(WmmaTy)
                e3 = fx.Int8(b_i8[3]).to(fx.Float32).to(WmmaTy)
                e4 = fx.Int8(b_i8[4]).to(fx.Float32).to(WmmaTy)
                e5 = fx.Int8(b_i8[5]).to(fx.Float32).to(WmmaTy)
                e6 = fx.Int8(b_i8[6]).to(fx.Float32).to(WmmaTy)
                e7 = fx.Int8(b_i8[7]).to(fx.Float32).to(WmmaTy)
            else:
                w0 = fx.Int32(raw[0]).ir_value()
                w1 = fx.Int32(raw[1]).ir_value()
                e0 = fx.Float32(_cvt_f32_f8(T.f32, w0, 0)).to(WmmaTy)
                e1 = fx.Float32(_cvt_f32_f8(T.f32, w0, 1)).to(WmmaTy)
                e2 = fx.Float32(_cvt_f32_f8(T.f32, w0, 2)).to(WmmaTy)
                e3 = fx.Float32(_cvt_f32_f8(T.f32, w0, 3)).to(WmmaTy)
                e4 = fx.Float32(_cvt_f32_f8(T.f32, w1, 0)).to(WmmaTy)
                e5 = fx.Float32(_cvt_f32_f8(T.f32, w1, 1)).to(WmmaTy)
                e6 = fx.Float32(_cvt_f32_f8(T.f32, w1, 2)).to(WmmaTy)
                e7 = fx.Float32(_cvt_f32_f8(T.f32, w1, 3)).to(WmmaTy)
            return Vec.from_elements([e0, e1, e2, e3, e4, e5, e6, e7], WmmaTy)

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
                af_cur.append(load_frag_a(row, fx.Int32(0)))
            bf_cur = []
            for tj in range_constexpr(TN):
                col = wn * fx.Int32(TN * WN) + fx.Int32(tj * WN) + lane16
                bf_cur.append(load_frag_b(col, fx.Int32(0)))

            for kk in range_constexpr(K_STEPS):
                if const_expr(kk + 1 < K_STEPS):
                    kbyte_n = fx.Int32((kk + 1) * WK)
                    af_nxt = []
                    for ti in range_constexpr(TM):
                        row = wm * fx.Int32(TM * WM) + fx.Int32(ti * WM) + lane16
                        af_nxt.append(load_frag_a(row, kbyte_n))
                    bf_nxt = []
                    for tj in range_constexpr(TN):
                        col = wn * fx.Int32(TN * WN) + fx.Int32(tj * WN) + lane16
                        bf_nxt.append(load_frag_b(col, kbyte_n))

                new_accs = []
                for ti in range_constexpr(TM):
                    for tj in range_constexpr(TN):
                        idx = ti * TN + tj
                        # If act is fp16 and WMMA is fp16, frags match; bf16 path same.
                        a_frag = af_cur[ti]
                        if const_expr(act_name != "bfloat16" and act_name == "float16"):
                            # already WmmaTy == Float16
                            pass
                        new_accs.append(wmma_f16(a_frag, bf_cur[tj], acc_list[idx]))
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

        # Epilogue: f32 acc × w_scale[n] → out. Wave32 column layout.
        # Math: y = (x @ w8.T.float()) * s  ≡  x @ (w8.float()*s).T  (per-N/scalar).
        for ti in range_constexpr(TM):
            row_base = (
                fx.Int64(m0) + fx.Int64(wm) * fx.Int64(TM * WM) + fx.Int64(ti * WM) + fx.Int64(klane) * fx.Int64(8)
            )
            for tj in range_constexpr(TN):
                idx = ti * TN + tj
                acc = unpack_acc(results, idx)
                col = fx.Int64(n0) + fx.Int64(wn) * fx.Int64(TN * WN) + fx.Int64(tj * WN) + fx.Int64(lane16)
                # Load w_scale once per output column (not per K-frag).
                if const_expr(w_scale_per_n):
                    if const_expr(skip_bounds):
                        sb_v = fx.make_view(fx.add_offset(sb_ptr, col), fx.make_layout(1, 1)).load()
                    else:
                        safe_n = (col < fx.Int64(N)).select(col, fx.Int64(0))
                        sb_v = fx.make_view(fx.add_offset(sb_ptr, safe_n), fx.make_layout(1, 1)).load()
                else:
                    sb_v = fx.make_view(sb_ptr, fx.make_layout(1, 1)).load()
                sc = fx.Float32(sb_v[0])
                for e in range_constexpr(8):
                    m = row_base + fx.Int64(e)
                    n = col
                    val = fx.Float32(acc[e]) * sc
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
        ScaleB: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_n = (fx.Int64(N) + fx.Int64(BN - 1)) // fx.Int64(BN)
        grid_m = (fx.Int64(M) + fx.Int64(BM - 1)) // fx.Int64(BM)
        gemm_kernel(A, Bnk, C, ScaleB, M, N, K, fx.Int32(grid_n), fx.Int32(grid_m)).launch(
            grid=(grid_n, grid_m, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# Host API (install hooks stripped — kernel surface only)
# ---------------------------------------------------------------------------


import torch  # noqa: E402

# FP8 format max parity with rdna4_scaled_mm_fp8(_fused) / rdna4_fp8_quant.
_F8_E4M3_MAX = 448.0
_F8_E5M2_MAX = 57344.0
_W8_TORCH = (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2)
_FP8_TORCH = (torch.float8_e4m3fn, torch.float8_e5m2)


def fp8_max_for(dtype: torch.dtype) -> float:
    """Finite max for e4m3fn (448) or e5m2 (57344); mirrors scaled_mm_fp8 helpers."""
    if dtype == torch.float8_e5m2:
        return _F8_E5M2_MAX
    if dtype == torch.float8_e4m3fn:
        return _F8_E4M3_MAX
    raise ValueError(f"fp8_max_for expects float8_e4m3fn/e5m2, got {dtype}")


def _w_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.int8:
        return "int8"
    if dtype == torch.float8_e5m2:
        return "float8_e5m2"
    if dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    raise ValueError(f"W8A16 weights must be int8/float8_e4m3fn/float8_e5m2, got {dtype}")


def _ptr(t: torch.Tensor) -> PointerJitArg:
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def w8a16_gemm(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """W8A16 linear GEMM: A[M,K] (bf16/fp16) @ B[N,K].T (int8/FP8); w_scale in epi → out[M,N].

    ``b_nk`` may be ``torch.int8``, ``torch.float8_e4m3fn``, or ``torch.float8_e5m2``.
    Weights are cast to act precision in-register; ``w_scale`` (scalar or [N])
    folds in the epilogue. Odd K stays in the kernel (any weight dtype including e5m2).
    ``stream`` is ``torch.cuda.Stream`` (host); launch uses ``fx.Stream``.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(a.device, what="w8a16_gemm (gfx120x)")
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"W8A16 linear requires bf16/fp16 acts, got {a.dtype}")
    if b_nk.dtype not in _W8_TORCH:
        raise ValueError(f"W8A16 linear requires int8/float8_e4m3fn/float8_e5m2 weights, got {b_nk.dtype}")
    if a.ndim != 2 or b_nk.ndim != 2:
        raise ValueError("W8A16 linear expects 2D A[M,K], B[N,K]")
    if b_nk.device != a.device:
        raise ValueError(f"a/b_nk device mismatch: {a.device} vs {b_nk.device}")
    m, k = a.shape
    n, k2 = b_nk.shape
    if k != k2:
        raise ValueError(f"K mismatch {k} vs {k2}")
    if m == 0 or n == 0:
        od = out_dtype if out_dtype is not None else a.dtype
        return torch.empty(m, n, device=a.device, dtype=od)
    # Odd K is loaded inside the kernel. K % 16 == 0 keeps the 16-byte path.
    k_tail = int(k) % 16
    if out_dtype is None:
        out_dtype = a.dtype
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported out_dtype {out_dtype}")

    w_scale = ensure_contiguous(w_scale.to(device=a.device, dtype=torch.float32).reshape(-1), stream=stream)
    w_per_n = w_scale.numel() != 1
    if w_per_n and w_scale.numel() != n:
        raise ValueError(f"w_scale must be scalar or [N]={n}, got {w_scale.numel()}")
    if int(k) == 0:
        store_dtype = torch.float32 if bias is not None else out_dtype
        out = torch.zeros((m, n), dtype=store_dtype, device=a.device)
        if bias is not None:
            from kernels.common.gfx120x_row_bias import add_row_bias

            return add_row_bias(out, bias, out_dtype=out_dtype, stream=stream)
        return out

    a = ensure_contiguous(a, stream=stream)
    b_nk = ensure_contiguous(b_nk, stream=stream)
    store_dtype = torch.float32 if bias is not None else out_dtype
    out = torch.empty(m, n, device=a.device, dtype=store_dtype)

    cfg = pick_tile_config(m, n, k, device=a.device)
    skip = (m % cfg.bm == 0) and (n % cfg.bn == 0)
    act_name = "bfloat16" if a.dtype == torch.bfloat16 else "float16"
    out_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[store_dtype]
    w_name = _w_dtype_name(b_nk.dtype)
    launch = build_w8a16_linear_module(
        act_name,
        out_name,
        cfg,
        skip_bounds=skip,
        w_scale_per_n=w_per_n,
        w_dtype_name=w_name,
        k_tail=k_tail,
    )
    if stream is None:
        launch(
            _ptr(a.view(torch.uint8)),
            _ptr(b_nk.view(torch.uint8)),
            _ptr(out.view(torch.uint8)),
            _ptr(w_scale),
            int(m),
            int(n),
            int(k),
        )
    else:
        launch(
            _ptr(a.view(torch.uint8)),
            _ptr(b_nk.view(torch.uint8)),
            _ptr(out.view(torch.uint8)),
            _ptr(w_scale),
            int(m),
            int(n),
            int(k),
            stream,
        )
    if bias is not None:
        from kernels.common.gfx120x_row_bias import add_row_bias

        out = add_row_bias(out, bias, out_dtype=out_dtype, stream=stream)
    elif store_dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def w8a16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """W8A16 linear: bf16/fp16 x @ int8/FP8 weight.T (cast in-reg; w_scale in epi).

    Soft-pads odd K; ``stream`` is ``torch.cuda.Stream`` (host).
    """
    require_gfx120x(x.device, what="w8a16_linear (gfx120x)")
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    if out_dtype is None:
        out_dtype = x.dtype
    n = weight.shape[0]
    if bias is not None:
        bias = bias.to(device=x.device).reshape(-1)
        if bias.numel() != n:
            raise ValueError(f"bias must be [N]={n}")
    out = w8a16_gemm(x2d, weight, weight_scale, bias=bias, out_dtype=out_dtype, stream=stream)
    return out.reshape(*orig_shape[:-1], n)
