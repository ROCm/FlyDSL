# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""FP8 e4m3 / e5m2 tensorwise scaled_mm / GEMM for gfx120x (RDNA4).

Computes ``C = scale_a * scale_b * (A_fp8 @ B_fp8.T)`` with per-tensor scales
using multi-wave LDS-pipelined FP8 WMMA and a Wave32 epilogue. Supports both
``float8_e4m3fn`` (fp8 cvt / WMMA) and ``float8_e5m2`` (bf8 cvt / WMMA) via the
``e5m2`` specialize flag -- parity with ``rdna4_fp8_quant`` / ``rdna4_stoch_fp8``.
Does **not** require the gfx120x iu8 atom (FP8 WMMA is already present on
GFX120X).

Public launcher: ``build_scaled_mm_fp8_module``. Tile configs trade LDS and
register pressure for occupancy on typical diffusion / LLM activation shapes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

KERNEL_NAME = "rdna4_scaled_mm_fp8"

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


_CFG_128_128_64 = TileConfig(128, 128, 64, 4, 2, 2, 4)
_CFG_128_128_128_A = TileConfig(128, 128, 128, 4, 4, 2, 2)
_CFG_128_128_128_B = TileConfig(128, 128, 128, 4, 2, 2, 4)
_CFG_256_128_64 = TileConfig(256, 128, 64, 4, 2, 4, 4)  # tall-M
_CFG_64_64_64 = TileConfig(64, 64, 64, 2, 2, 2, 2)
_CFG_64_64_128 = TileConfig(64, 64, 128, 2, 2, 2, 2)


_WGP_COUNT_CACHE: int | None = None


def _wgp_count() -> int:
    """Cached CU/WGP count — pick_tile runs on the short-launch hot path (lean host).

    Docs: playbook lean host / HIP performance guidelines (avoid repeated host
    queries on short launches); mirrors kernels/gemm/rdna4_int8_linear.py.
    """
    global _WGP_COUNT_CACHE
    if _WGP_COUNT_CACHE is not None:
        return _WGP_COUNT_CACHE
    try:
        _WGP_COUNT_CACHE = int(torch.cuda.get_device_properties(0).multi_processor_count) or _DEFAULT_WGPS
    except Exception:  # noqa: BLE001
        _WGP_COUNT_CACHE = _DEFAULT_WGPS
    return _WGP_COUNT_CACHE


def pick_tile_config(M: int, N: int, K: int, wgps: int | None = None) -> TileConfig:
    """Size-based tile pick mirroring HIP ``launch_gemm_wmma`` (gemm_wmma.h)."""
    if wgps is None:
        wgps = _wgp_count()
    blocks_128 = ((M + 127) // 128) * ((N + 127) // 128)
    skinny = M <= 64 or N <= 64
    if not skinny and blocks_128 >= wgps:
        if K >= 4096:
            if blocks_128 <= 4 * wgps:
                return _CFG_128_128_128_A
            return _CFG_128_128_128_B
        # Deepen BK for K>=512 on fat grids: fewer LDS round-trips. Occupancy
        # drops (≈34KB LDS) but nets a win on K=512 fat shapes vs HIP's BK=64.
        if K >= 512:
            return _CFG_128_128_128_B
        # Prefer 128x128 BK=64 (HIP default for K<4096).
        return _CFG_128_128_64
    # Docs:  picks (tiny→tile/BK deepen, not blind WGP bump);
    # docs/kernel_tuning_guide.md (larger tile_k for latency-bound small M);
    # docs/testing_benchmarking_guide.md CUDA-event median vs HIP;
    # https://rocm.docs.amd.com/projects/FlyDSL/en/latest/
    # Measured 2026-09-30: 64x64x128 tiny ~1.43× faster than HIP on gfx1201 / R9700.
    if K >= 128:
        return _CFG_64_64_128
    return _CFG_64_64_64


@lru_cache(maxsize=64)
def build_scaled_mm_fp8_module(
    out_name: str, cfg: TileConfig, skip_bounds: bool = False, e5m2: bool = False
) -> Callable[..., None]:
    """Compile multi-wave LDS FP8 WMMA GEMM for one (out_dtype, tile, format) triple.

    ``e5m2=False`` (default) uses e4m3 / ``Float8E4M3FN`` WMMA; ``e5m2=True``
    uses e5m2 / ``Float8E5M2`` WMMA (bf8 path). Hosts pass matching packed
    ``float8_e4m3fn`` or ``float8_e5m2`` operands.
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
    Fp8Ty = fx.Float8E5M2 if e5m2 else fx.Float8E4M3FN
    # Keep cache-safe names local to FlyDSL; no external kernel-signature helper is
    # intentionally not a dependency of this standalone repository kernel.
    sig = f"{out_name}_{cfg.name}_bounds{int(skip_bounds)}_e5m2{int(e5m2)}"

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
        tid = fx.Int32(gpu.thread_id("x"))
        bid_x = fx.Int32(gpu.block_id("x"))
        bid_y = fx.Int32(gpu.block_id("y"))
        lane = tid % fx.Int32(WARP)
        wave = tid // fx.Int32(WARP)
        # Wave32 column-distributed accum (): lane%16→col, lane//16→row group.
        # Wrong mapping silently transposed tiles and forced an LDS round-trip — do not "fix" in LDS.
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

        def as_i32(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, ptr.address_space), ptr)

        # LDS via allocate(nbytes) — avoids @fx.struct + postponed-annotations issues
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

        # pad=8 breaks 16B LDS alignment — reference stores as two uint2s.
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

        sa_view = fx.make_view(
            fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleA.address_space), ScaleA),
            fx.make_layout(1, 1),
        )
        sb_view = fx.make_view(
            fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB),
            fx.make_layout(1, 1),
        )
        scale_ab = fx.Float32(sa_view.load()[0]) * fx.Float32(sb_view.load()[0])

        wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WM, WN, WK, Fp8Ty, fx.Float32))

        def wmma_acc(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            a_frag = fx.make_rmem_tensor(8, fx.Int8)
            b_frag = fx.make_rmem_tensor(8, fx.Int8)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
            a_frag.store(Vec(a_v8))
            b_frag.store(Vec(b_v8))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)  # FlyDSL 0.3.4.1: a/b are Sequence
            return Vec(c_frag.load())

        def unpack_acc(carried: list[fx.Float32], ai: int) -> fx.Vector:
            base = ai * 8
            return Vec.from_elements(
                [
                    fx.Float32(carried[base + 0]),
                    fx.Float32(carried[base + 1]),
                    fx.Float32(carried[base + 2]),
                    fx.Float32(carried[base + 3]),
                    fx.Float32(carried[base + 4]),
                    fx.Float32(carried[base + 5]),
                    fx.Float32(carried[base + 6]),
                    fx.Float32(carried[base + 7]),
                ],
                fx.Float32,
            )

        def pack_acc(acc_vec: fx.Vector) -> list[fx.Float32]:
            return [
                fx.Float32(acc_vec[0]),
                fx.Float32(acc_vec[1]),
                fx.Float32(acc_vec[2]),
                fx.Float32(acc_vec[3]),
                fx.Float32(acc_vec[4]),
                fx.Float32(acc_vec[5]),
                fx.Float32(acc_vec[6]),
                fx.Float32(acc_vec[7]),
            ]

        zero = fx.Float32(0.0)
        init = [zero for _ in range(N_ACC * 8)]

        # Prologue: stage tile 0 into LDS (TileStager load+store).
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
            # Prefetch next tile's GMEM into regs (overlaps current-tile math).
            # Guarded: only issue loads when a next tile exists.
            if has_next:
                a_regs = load_a_regs(knext)
                b_regs = load_b_regs(knext)

            # Intra-tile K frag pipeline: af[2]/bf[2] (gemm_wmma.h).
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

        # Epilogue: Wave32 column-distributed store (lane16→N/col, klane*8+e→M/row). See §10.1.
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
                    val = fx.Float32(acc[e]) * scale_ab
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
                        inb = (m < fx.Int64(M)) & (n < fx.Int64(N))
                        safe = inb.select(cidx, fx.Int64(0))
                        view = fx.make_view(
                            fx.add_offset(c_ptr, safe),
                            fx.make_layout(1, 1),
                        )
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

    gemm_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch
