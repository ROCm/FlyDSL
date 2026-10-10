# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""FP8 e4m3 / e5m2 tensorwise scaled_mm / GEMM for gfx120x (RDNA4).

Computes ``C = scale_a * scale_b * (A_fp8 @ B_fp8.T)`` with per-tensor scales
using multi-wave LDS-pipelined FP8 WMMA and a Wave32 epilogue. Supports both
``float8_e4m3fn`` (fp8 cvt / WMMA) and ``float8_e5m2`` (bf8 cvt / WMMA) via the
``e5m2`` specialize flag -- parity with ``rdna4_fp8_quant`` / ``rdna4_stoch_fp8``.
Does **not** require the gfx120x iu8 atom (FP8 WMMA is already present on
GFX120X).

Public: ``scaled_mm_fp8`` (product host; odd K is handled in the kernel),
``build_scaled_mm_fp8_module``, ``pick_tile_config``. Tile configs trade LDS
and register pressure for occupancy on typical diffusion / LLM activation shapes.
"""

from collections.abc import Callable
from functools import lru_cache

import torch

from kernels.gemm.rdna4_tile import TileConfig

KERNEL_NAME = "rdna4_scaled_mm_fp8"

WM = WN = WK = 16
WARP = 32
LDS_PAD = 8
_DEFAULT_WGPS = 32

_CFG_128_128_64 = TileConfig(128, 128, 64, 4, 2, 2, 4)
_CFG_128_128_128_A = TileConfig(128, 128, 128, 4, 4, 2, 2)
_CFG_128_128_128_B = TileConfig(128, 128, 128, 4, 2, 2, 4)
_CFG_256_128_64 = TileConfig(256, 128, 64, 4, 2, 4, 4)  # tall-M
_CFG_64_64_64 = TileConfig(64, 64, 64, 2, 2, 2, 2)
_CFG_64_64_128 = TileConfig(64, 64, 128, 2, 2, 2, 2)


_WGP_COUNT_CACHE: dict[int, int] = {}


def _wgp_count(device=None) -> int:
    """Cached CU/WGP count for ``device`` (defaults to current / 0).

    Multi-GPU: query the tensor's device index, not always ``cuda:0``.
    """
    try:
        import torch

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
    """Size-based tile pick mirroring HIP ``launch_gemm_wmma`` (gemm_wmma.h)."""
    if wgps is None:
        wgps = _wgp_count(device)
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


@lru_cache(maxsize=128)
def build_scaled_mm_fp8_module(
    out_name: str,
    cfg: TileConfig,
    skip_bounds: bool = False,
    e5m2: bool = False,
    k_tail: int = 0,
    *,
    kind: str = "fp8",
    scale_b_per_n: bool = False,
    sign_a: bool = True,
    sign_b: bool = True,
    kernel_name: str | None = None,
) -> Callable[..., None]:
    """Compile the shared gfx120x quant GEMM for one specialization.

    ``kind="fp8"`` uses e4m3 (``e5m2=False``) or e5m2 WMMA into f32.
    ``kind="int8"`` uses the iu8 WMMA into i32 (``sign_a`` / ``sign_b`` select
    the ISA NEG modifier, ``clamp=False``). int8 activation scale is per row.
    fp8 activation scale is per tensor. ``scale_b_per_n`` selects a per-column
    weight scale; otherwise scale B is one f32.

    ``k_tail`` is ``K % 16``. Zero keeps the 16-byte global loads and is only
    legal when ``K % 16 == 0``. A non-zero tail loads each K-chunk as bytes.
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
    if kind not in ("fp8", "int8"):
        raise ValueError(f"kind must be 'fp8' or 'int8', got {kind!r}")
    if not isinstance(k_tail, int) or not 0 <= k_tail < 16:
        raise ValueError(f"k_tail must be an int in 0..15, got {k_tail!r}")
    sig = (
        f"{kind}_{out_name}_{cfg.name}_bounds{int(skip_bounds)}_e5m2{int(e5m2)}"
        f"_tail{k_tail}_sb{int(scale_b_per_n)}_sg{int(sign_a)}{int(sign_b)}"
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

        def load16_bytes(elem_ptr: fx.Pointer, byte_idx: fx.Int64, n_valid: fx.Int32) -> fx.Vector:
            # One byte at a time. A 16-byte load needs a 16-byte-aligned address,
            # and row*K is not 16-aligned when K % 16 != 0. Bytes past n_valid
            # reload offset 0 and are discarded, so the load stays inside the allocation.
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
                if const_expr(k_tail == 0):
                    inb = (grow < M) & (gk < K)
                    regs.append(load16_gmem(a_g, fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk), inb))
                else:
                    addr = fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk)
                    regs.append(load16_bytes(a_g, addr, chunk_n_valid(grow, gk, M)))
            return regs

        def load_b_regs(kbyte0: fx.Int32) -> list[fx.Vector]:
            regs = []
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                grow = n0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk = kbyte0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(16)
                if const_expr(k_tail == 0):
                    inb = (grow < N) & (gk < K)
                    regs.append(load16_gmem(b_g, fx.Int64(grow) * fx.Int64(K) + fx.Int64(gk), inb))
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

        if const_expr(kind == "fp8" and not scale_b_per_n):
            sa_view = fx.make_view(
                fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleA.address_space), ScaleA),
                fx.make_layout(1, 1),
            )
            sb_view = fx.make_view(
                fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB),
                fx.make_layout(1, 1),
            )
            scale_ab = fx.Float32(sa_view.load()[0]) * fx.Float32(sb_view.load()[0])
        elif const_expr(kind == "fp8"):
            sa_view = fx.make_view(
                fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleA.address_space), ScaleA),
                fx.make_layout(1, 1),
            )
            scale_a = fx.Float32(sa_view.load()[0])
            sb_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB)
        else:
            sa_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleA.address_space), ScaleA)
            sb_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ScaleB.address_space), ScaleB)

        if const_expr(kind == "fp8"):
            wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WM, WN, WK, Fp8Ty, fx.Float32))
        else:
            wmma_atom = fx.make_mma_atom(
                fx.rocdl.WMMA(WM, WN, WK, fx.Int8, fx.Int32, sign_a=sign_a, sign_b=sign_b, clamp=False)
            )

        def wmma_acc(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            a_frag = fx.make_rmem_tensor(8, fx.Int8)
            b_frag = fx.make_rmem_tensor(8, fx.Int8)
            if const_expr(kind == "fp8"):
                c_frag = fx.make_rmem_tensor(8, fx.Float32)
            else:
                c_frag = fx.make_rmem_tensor(8, fx.Int32)
            a_frag.store(Vec(a_v8))
            b_frag.store(Vec(b_v8))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, [a_frag], [b_frag], c_frag)  # FlyDSL 0.3.4.1: a/b are Sequence
            return Vec(c_frag.load())

        def unpack_acc(carried: list, ai: int) -> fx.Vector:
            base = ai * 8
            if const_expr(kind == "fp8"):
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
            return Vec.from_elements([fx.Int32(carried[base + e]) for e in range(8)], fx.Int32)

        def pack_acc(acc_vec: fx.Vector) -> list:
            if const_expr(kind == "fp8"):
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
            return [fx.Int32(acc_vec[e]) for e in range(8)]

        if const_expr(kind == "fp8"):
            zero = fx.Float32(0.0)
        else:
            zero = fx.Int32(0)
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
                    if const_expr(kind == "fp8" and not scale_b_per_n):
                        val = fx.Float32(acc[e]) * scale_ab
                    elif const_expr(kind == "fp8"):
                        inb_mn = (m < fx.Int64(M)) & (n < fx.Int64(N))
                        safe_n = inb_mn.select(n, fx.Int64(0))
                        if const_expr(skip_bounds):
                            sb_v = fx.make_view(fx.add_offset(sb_ptr, n), fx.make_layout(1, 1)).load()
                        else:
                            sb_v = fx.make_view(fx.add_offset(sb_ptr, safe_n), fx.make_layout(1, 1)).load()
                        val = fx.Float32(acc[e]) * scale_a * fx.Float32(sb_v[0])
                    else:
                        inb_mn = (m < fx.Int64(M)) & (n < fx.Int64(N))
                        safe_m = inb_mn.select(m, fx.Int64(0))
                        safe_n = inb_mn.select(n, fx.Int64(0))
                        sa_v = fx.make_view(fx.add_offset(sa_ptr, safe_m), fx.make_layout(1, 1)).load()
                        if const_expr(scale_b_per_n):
                            sb_v = fx.make_view(fx.add_offset(sb_ptr, safe_n), fx.make_layout(1, 1)).load()
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
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_n = (fx.Int64(N) + fx.Int64(BN - 1)) // fx.Int64(BN)
        grid_m = (fx.Int64(M) + fx.Int64(BM - 1)) // fx.Int64(BM)
        gemm_kernel(A, Bnk, C, ScaleA, ScaleB, M, N, K, fx.Int32(grid_n), fx.Int32(grid_m)).launch(
            grid=(grid_n, grid_m, 1), block=(THREADS, 1, 1), stream=stream
        )

    launched_as = kernel_name or KERNEL_NAME
    gemm_kernel.__name__ = f"{launched_as}_{sig}"
    launch.__name__ = f"launch_{launched_as}_{sig}"
    return launch


def _out_dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[dtype]


def scaled_mm_fp8(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    e5m2: bool | None = None,
    wgps: int | None = None,
    tile: TileConfig | None = None,
    bias: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Product host: contiguify, pick or use ``tile``, then launch.

    ``a`` / ``b_nk`` are packed FP8 (``float8_e4m3fn`` or ``float8_e5m2``); ``b_nk``
    is [N, K]. ``scale_a`` is one f32. ``scale_b`` is one f32 or ``[N]``.
    ``e5m2`` must match the tensor dtype (``None`` auto-infers). Odd K is read
    in the kernel. ``bias`` is ``[N]`` and uses the same row-bias add as int8.
    ``stream`` is ``torch.cuda.Stream`` (host).
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.compiler.jit_argument import PointerJitArg
    from kernels.common.gfx120x_arch import require_gfx120x
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(a.device, what="scaled_mm_fp8 (gfx120x)")
    if a.ndim != 2 or b_nk.ndim != 2:
        raise ValueError("a and b_nk must be 2D [M,K] and [N,K]")
    _fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    if a.dtype not in _fp8 or b_nk.dtype not in _fp8:
        raise ValueError(f"a/b_nk must be float8_e4m3fn or float8_e5m2, got {a.dtype}, {b_nk.dtype}")
    if a.dtype != b_nk.dtype:
        raise ValueError(f"a/b_nk FP8 formats must match, got {a.dtype} vs {b_nk.dtype}")
    # Derive e5m2 from tensors; None auto-infers (same rule as scaled_mm_fp8_fused).
    e5m2_eff = a.dtype == torch.float8_e5m2
    if e5m2 is None:
        e5m2 = e5m2_eff
    elif e5m2 != e5m2_eff:
        want = torch.float8_e5m2 if e5m2 else torch.float8_e4m3fn
        raise ValueError(f"e5m2={e5m2} disagrees with a.dtype={a.dtype} (want {want})")
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b_nk.shape[0])
    if int(b_nk.shape[1]) != k:
        raise ValueError(f"inner dim mismatch: a K={k}, b K={b_nk.shape[1]}")
    if m == 0 or n == 0:
        return torch.empty((m, n), dtype=out_dtype, device=a.device)
    k_tail = k % 16
    a = ensure_contiguous(a, stream=stream)
    b_nk = ensure_contiguous(b_nk, stream=stream)

    cfg = tile if tile is not None else pick_tile_config(m, n, k, wgps=wgps, device=a.device)
    skip_bounds = m % cfg.bm == 0 and n % cfg.bn == 0
    scale_a = ensure_contiguous(scale_a.to(device=a.device, dtype=torch.float32).reshape(-1), stream=stream)
    scale_b = ensure_contiguous(scale_b.to(device=a.device, dtype=torch.float32).reshape(-1), stream=stream)
    if scale_a.numel() != 1:
        raise ValueError(f"scale_a must be a scalar, got {scale_a.numel()}")
    scale_b_per_n = scale_b.numel() != 1
    if scale_b_per_n and scale_b.numel() != n:
        raise ValueError(f"scale_b must be a scalar or [N]={n}, got {scale_b.numel()}")
    if k == 0:
        # An empty K still has an [M, N] product. Launching would load byte 0
        # of that empty allocation. The product is zero; bias is added after.
        store_dtype = torch.float32 if bias is not None else out_dtype
        out = torch.zeros((m, n), dtype=store_dtype, device=a.device)
        if bias is not None:
            from kernels.common.gfx120x_row_bias import add_row_bias

            return add_row_bias(out, bias, out_dtype=out_dtype, stream=stream)
        return out
    store_dtype = torch.float32 if bias is not None else out_dtype
    launch = build_scaled_mm_fp8_module(
        _out_dtype_name(store_dtype),
        cfg,
        skip_bounds,
        e5m2=e5m2,
        k_tail=k_tail,
        scale_b_per_n=scale_b_per_n,
    )
    out = torch.empty((m, n), dtype=store_dtype, device=a.device)

    def _ptr(t: torch.Tensor) -> PointerJitArg:
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    args = (
        _ptr(a.view(torch.uint8)),
        _ptr(b_nk.view(torch.uint8)),
        _ptr(out.view(torch.uint8)),
        _ptr(scale_a),
        _ptr(scale_b),
        m,
        n,
        k,
    )
    if stream is None:
        launch(*args)
    else:
        launch(*args, stream)
    if bias is not None:
        from kernels.common.gfx120x_row_bias import add_row_bias

        out = add_row_bias(out, bias, out_dtype=out_dtype, stream=stream)
    return out
