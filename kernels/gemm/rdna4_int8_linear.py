# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X iu8 int8 linear GEMM (gfx120x / RDNA4).

Computes ``C[M, N] = A[M, K] @ B_T[N, K].T`` with a per-row activation scale
and either a per-column or scalar weight scale in the epilogue. Int8
activations × int8 weights → bf16/fp16 out via the gfx120x **iu8** WMMA atom.

Requires that atom on ``MmaOpGFX120X_WMMAType``. Stock FlyDSL without integer
GFX120X WMMA cannot compile this module. For bf16 activations with 8-bit
weights (no iu8), use ``rdna4_w8a16_linear``; for an M / M×K size pick between
the two, use ``rdna4_int8_linear_dispatch`` / ``rdna4_int8_linear_auto``.

The WMMA atom is constructed with ``clamp=False`` (default): the INT32
accumulator wraps on overflow. ``clamp=True`` would saturate to the input
element type range per AMD IU8 semantics; this kernel keeps wrap for exact
int32 accumulation before the scale epilogue.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

KERNEL_NAME = "rdna4_int8_linear"

WM = WN = WK = 16
WARP = 32
LDS_PAD = 8
_DEFAULT_WGPS = 32


@dataclass(frozen=True)
class TileConfig:
    """WMMA launch tile. ``threads`` is the block size. ``name`` is the tile label."""

    bm: int
    bn: int
    bk: int
    warps_m: int
    warps_n: int
    tm: int
    tn: int

    @property
    def threads(self) -> int:
        """Threads in the block: warps_m * warps_n * 32."""
        return self.warps_m * self.warps_n * WARP

    @property
    def name(self) -> str:
        """Tile label, for logs and the autotune table."""
        return f"{self.bm}x{self.bn}x{self.bk}_w{self.warps_m}x{self.warps_n}_t{self.tm}x{self.tn}"


# Launch tiles: include deeper K on fat shapes where that tile beat the HIP baseline.
_CFG_128_128_64 = TileConfig(128, 128, 64, 4, 2, 2, 4)
_CFG_128_128_128 = TileConfig(128, 128, 128, 4, 2, 2, 4)
_CFG_256_128_64 = TileConfig(256, 128, 64, 4, 2, 4, 4)
_CFG_256_128_128 = TileConfig(256, 128, 128, 4, 2, 4, 4)
_CFG_64_64_64 = TileConfig(64, 64, 64, 2, 2, 2, 2)


_WGP_COUNT_CACHE: int | None = None


def _wgp_count() -> int:
    """Cached CU/WGP count — pick_tile runs on the short-launch hot path."""
    global _WGP_COUNT_CACHE
    if _WGP_COUNT_CACHE is not None:
        return _WGP_COUNT_CACHE
    try:
        import torch

        _WGP_COUNT_CACHE = int(torch.cuda.get_device_properties(0).multi_processor_count) or _DEFAULT_WGPS
    except Exception:  # noqa: BLE001
        _WGP_COUNT_CACHE = _DEFAULT_WGPS
    return _WGP_COUNT_CACHE


def pick_tile_config(M: int, N: int, K: int, wgps: int | None = None) -> TileConfig:
    """Size-based tile pick; prefer IU8 128×128×128 for deep K.

    Design notes (measured on gfx1201 / R9700, 2026-09-30):
      - FlyDSL ``docs/kernel_tuning_guide.md`` (deeper ``tile_k`` for reuse; LDS budget)
      - FlyDSL ``docs/testing_benchmarking_guide.md`` (CUDA-event median timing)
      - IU8 BK=128 on 128×128 when K≥2048 (LDS about 34 KiB)
      - HIP ``launch_gemm_wmma`` tile thresholds (WGP-aware)
      - Large production-like shapes (e.g. M=N=K around 4k, or 1024×5120×5120)
        prefer 128×128×128 over tall-M 256×128×128

    Re-bench: tall-M 256×128×128 is slower than or ties the HIP baseline on those large shapes;
    128×128×128 is faster on large (~1.60×) and on 1024×5120×5120 (~1.11×) vs the HIP baseline.
    Keep 256 tiles for tall-M mid-K only.
    """
    if wgps is None:
        wgps = _wgp_count()
    blocks_128 = ((M + 127) // 128) * ((N + 127) // 128)
    skinny = M <= 64 or N <= 64
    # Deep-K first: prefer 128×128×128 over tall-M 256 on large/deep-K shapes.
    if not skinny and M >= 128 and N >= 128 and K >= 2048:
        return _CFG_128_128_128
    if not skinny and M >= 512 and blocks_128 >= wgps:
        if K >= 512:
            return _CFG_256_128_128
        return _CFG_256_128_64
    if not skinny and M >= 128 and N >= 128:
        if K >= 512:
            return _CFG_128_128_128
        return _CFG_128_128_64
    return _CFG_64_64_64


@lru_cache(maxsize=64)
def build_int8_linear_module(
    out_name: str,
    cfg: TileConfig,
    skip_bounds: bool = False,
    w_scale_per_n: bool = True,
) -> Callable[..., None]:
    """Compile multi-wave LDS int8 WMMA GEMM for one (out_dtype, tile, scale mode).

    REQUIRES FlyDSL gfx120x iu8 WMMA atom support.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import Vector as Vec

    OutTy = {
        "bfloat16": fx.BFloat16,
        "float16": fx.Float16,
        "float32": fx.Float32,
    }[out_name]

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

    assert BM == WARPS_M * TM * WM
    assert BN == WARPS_N * TN * WN
    assert BK % WK == 0
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
        BlocksN: fx.Int32,
        BlocksM: fx.Int32,
    ) -> None:
        # Create WMMA atom inside the kernel so an MLIR Context is active
        # (same pattern as flydsl_scaled_mm float8 path).
        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WM, WN, WK, fx.Int8, fx.Int32, sign_a=True, sign_b=True, clamp=False)
        )

        def wmma_iu8(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            """RDNA4 Wave32 iu8 WMMA via ``fx.gemm`` on rank-1 rmem fragments.

            v8 ABI: A/B are 8 signed i8 (packed as the atom's v2i32), C is v8i32.
            ``fx.gemm`` is the single-atom form. ``clamp=False`` on the atom
            leaves i32 wrap before the scale epilogue.
            """
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

        a_g = fx.recast_iter(fx.PointerType.get(fx.Int8.ir_type, A.address_space), A)
        b_g = fx.recast_iter(fx.PointerType.get(fx.Int8.ir_type, Bnk.address_space), Bnk)
        c_ptr = fx.recast_iter(fx.PointerType.get(OutTy.ir_type, C.address_space), C)
        sa_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleA.address_space), ScaleA)
        sb_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB)

        def as_i32(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, ptr.address_space), ptr)

        alloc = fx.SharedAllocator()
        as_base = alloc.allocate(AS_BYTES)._ptr
        bs_base = alloc.allocate(BS_BYTES)._ptr

        def load16_gmem(elem_ptr: fx.Pointer, byte_idx: fx.Int32 | fx.Int64 | int, inb: object) -> fx.Vector:
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

        def load_a_regs(kbyte0: fx.Int32) -> list[fx.Vector]:
            regs = []
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                grow = m0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                inb = (grow < M) & (gk < K)
                regs.append(load16_gmem(a_g, fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk), inb))
            return regs

        def load_b_regs(kbyte0: fx.Int32) -> list[fx.Vector]:
            regs = []
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                grow = n0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                inb = (grow < N) & (gk < K)
                regs.append(load16_gmem(b_g, fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk), inb))
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

        # Epilogue: i32→f32 * row_scale * col_scale → out; Wave32 column-distributed.
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
                    # scales
                    sa_v = fx.make_view(fx.add_offset(sa_ptr, m), fx.make_layout(1, 1)).load()
                    if const_expr(w_scale_per_n):
                        sb_v = fx.make_view(fx.add_offset(sb_ptr, n), fx.make_layout(1, 1)).load()
                    else:
                        sb_v = fx.make_view(sb_ptr, fx.make_layout(1, 1)).load()
                    val = fx.Float32(fx.Int32(acc[e]).to(fx.Float32)) * fx.Float32(sa_v[0]) * fx.Float32(sb_v[0])
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
        stream: fx.Stream,
    ) -> None:
        grid_n = (fx.Int64(N) + fx.Int64(BN - 1)) // fx.Int64(BN)
        grid_m = (fx.Int64(M) + fx.Int64(BM - 1)) // fx.Int64(BM)
        gemm_kernel(A, Bnk, C, ScaleA, ScaleB, M, N, K, fx.Int32(grid_n), fx.Int32(grid_m)).launch(
            grid=(grid_n, grid_m, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch


# Keep create_* spelling aligned with the rdna3 GEMM family (alias of build_*).
def create_wmma_int8_linear_module(
    out_dtype: str = "bfloat16",
    cfg: TileConfig | tuple | None = None,
    *,
    skip_bounds: bool = False,
    w_scale_per_n: bool = True,
) -> Callable[..., None]:
    """Create a gfx120x int8-linear launcher for one tile configuration."""
    if cfg is None:
        cfg = _CFG_128_128_64
    if isinstance(cfg, tuple):
        cfg = TileConfig(*cfg)
    return build_int8_linear_module(out_dtype, cfg, skip_bounds, w_scale_per_n)
