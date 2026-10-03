# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X native iu4 WMMA GEMM (packed int4 A/B → scaled float or i32 out).

Uses gfx12 ``wmma_i32_16x16x16_iu4`` with scalar i32 A/B packing. Pack layout
matches ``kernels.quant.rdna4_int4_codec`` (low nibble = even column).

**Default:** prefer native iu4 when ``K % 16 == 0``. Pass
``prefer_native=False`` to force unpack→iu8. This is the GEMM used by ConvRot
when ``linear_dtype="int4"``.

Measured faster than HIP on those shapes (~×1.05–1.22). See
``docs/prebuilt_kernels_guide.md`` (gfx120x iu4 / ConvRot).
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

from flydsl.compiler.jit_argument import PointerJitArg
from kernels.common.gfx120x_arch import require_gfx120x

KERNEL_NAME = "rdna4_iu4_gemm"

WM = WN = WK = 16
WARP = 32
LDS_PAD = 8  # packed-byte pad
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


# Conservative tiles — LDS holds packed bytes (half of iu8 for same logical BK).
_CFG_64_64_64 = TileConfig(64, 64, 64, 2, 2, 2, 2)
_CFG_128_128_64 = TileConfig(128, 128, 64, 4, 2, 2, 4)
_CFG_128_128_128 = TileConfig(128, 128, 128, 4, 2, 2, 4)

_WGP_COUNT_CACHE: int | None = None


def _wgp_count() -> int:
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
    """Size-based tile pick for native iu4 (logical M/N/K)."""
    if wgps is None:
        wgps = _wgp_count()
    del wgps  # reserved for future WGP-aware pick
    skinny = M <= 64 or N <= 64
    if not skinny and M >= 128 and N >= 128 and K >= 512:
        return _CFG_128_128_128
    if not skinny and M >= 128 and N >= 128:
        return _CFG_128_128_64
    return _CFG_64_64_64


def shapes_ok_for_native_iu4(M: int, N: int, K: int) -> bool:
    """Gate: K%16==0 and positive dims. Partial M/N OK (bounds path)."""
    return M > 0 and N > 0 and K > 0 and (K % 16 == 0)


@lru_cache(maxsize=64)
def build_iu4_gemm_module(
    out_name: str,
    cfg: TileConfig,
    skip_bounds: bool = False,
    w_scale_per_n: bool = True,
    sign_a: bool = True,
    sign_b: bool = True,
) -> Callable[..., None]:
    """Compile multi-wave LDS iu4 WMMA GEMM for one (out_dtype, tile, scale mode).

    REQUIRES FlyDSL gfx120x iu4 WMMA atom (scalar i32 A/B).
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import Vector as Vec

    OutTy = {
        "bfloat16": fx.BFloat16,
        "float16": fx.Float16,
        "float32": fx.Float32,
        "int32": fx.Int32,
    }[out_name]

    BM, BN, BK = cfg.bm, cfg.bn, cfg.bk
    WARPS_M, WARPS_N = cfg.warps_m, cfg.warps_n
    TM, TN = cfg.tm, cfg.tn
    THREADS = cfg.threads
    N_ACC = TM * TN
    K_STEPS = BK // WK
    # Packed-byte LDS stride (logical BK nibbles → BK/2 bytes).
    STRIDE = BK // 2 + LDS_PAD
    CHUNKS_PER_ROW = BK // WK  # each chunk = WK=16 nibbles = 8 packed bytes
    CHUNKS_A = BM * CHUNKS_PER_ROW
    CHUNKS_B = BN * CHUNKS_PER_ROW
    PTA = CHUNKS_A // THREADS
    PTB = CHUNKS_B // THREADS
    GROUP_M = 4
    AS_BYTES = BM * STRIDE
    BS_BYTES = BN * STRIDE
    PACKED_CHUNK = WK // 2  # 8

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
        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(
                WM,
                WN,
                WK,
                fx.Int4,
                fx.Int32,
                sign_a=sign_a,
                sign_b=sign_b,
                clamp=False,
            )
        )

        def wmma_iu4(a_i32: fx.Vector, b_i32: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            """RDNA4 Wave32 iu4 WMMA via ``fx.gemm``: scalar i32 A/B (8 nibbles), v8i32 C."""
            a_frag = fx.make_rmem_tensor(8, fx.Int4)
            b_frag = fx.make_rmem_tensor(8, fx.Int4)
            c_frag = fx.make_rmem_tensor(8, fx.Int32)
            a_frag.store(Vec(a_i32).bitcast(fx.Int4))
            b_frag.store(Vec(b_i32).bitcast(fx.Int4))
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

        # GMEM is nibble-packed int8 (K_packed = K/2).
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

        def load8_gmem(elem_ptr: fx.Pointer, byte_idx: fx.Int32 | fx.Int64 | int, inb: object) -> fx.Vector:
            """Load one K=16 chunk as 2×i32 (8 packed bytes / 16 nibbles)."""
            i32_ptr = as_i32(elem_ptr)
            safe = inb.select(fx.Int64(byte_idx), fx.Int64(0))
            view = fx.make_view(
                fx.add_offset(i32_ptr, safe // fx.Int64(4)),
                fx.make_layout(2, 1),
            )
            raw = Vec(view.load())
            z = fx.Int32(0)
            return Vec.from_elements(
                [
                    inb.select(fx.Int32(raw[0]), z),
                    inb.select(fx.Int32(raw[1]), z),
                ],
                fx.Int32,
            )

        def store2_lds(lds_ptr: fx.Pointer, byte_off: fx.Int32 | int, v2i32: fx.Vector) -> None:
            i32_lds = as_i32(lds_ptr)
            off0 = fx.Int64(byte_off) // fx.Int64(4)
            view0 = fx.make_view(fx.add_offset(i32_lds, off0), fx.make_layout(2, 1))
            view0.store(Vec.from_elements([fx.Int32(v2i32[0]), fx.Int32(v2i32[1])], fx.Int32))

        # k0_logical is the logical K start; packed byte = k0_logical // 2.
        def load_a_regs(k0_logical: fx.Int32) -> list[fx.Vector]:
            regs = []
            k_packed0 = k0_logical // fx.Int32(2)
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                grow = m0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk_logical = k0_logical + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(WK)
                gk_packed = k_packed0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(PACKED_CHUNK)
                inb = (grow < M) & (gk_logical < K)
                row_stride = K // fx.Int32(2)
                regs.append(
                    load8_gmem(
                        a_g,
                        fx.Int64(grow) * fx.Int64(row_stride) + fx.Int64(gk_packed),
                        inb,
                    )
                )
            return regs

        def load_b_regs(k0_logical: fx.Int32) -> list[fx.Vector]:
            regs = []
            k_packed0 = k0_logical // fx.Int32(2)
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                grow = n0 + c // fx.Int32(CHUNKS_PER_ROW)
                gk_logical = k0_logical + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(WK)
                gk_packed = k_packed0 + (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(PACKED_CHUNK)
                inb = (grow < N) & (gk_logical < K)
                row_stride = K // fx.Int32(2)
                regs.append(
                    load8_gmem(
                        b_g,
                        fx.Int64(grow) * fx.Int64(row_stride) + fx.Int64(gk_packed),
                        inb,
                    )
                )
            return regs

        def store_a_regs(regs: list[fx.Vector]) -> None:
            for i in range_constexpr(PTA):
                c = tid * fx.Int32(PTA) + fx.Int32(i)
                local_row = c // fx.Int32(CHUNKS_PER_ROW)
                local_k_packed = (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(PACKED_CHUNK)
                store2_lds(as_base, local_row * fx.Int32(STRIDE) + local_k_packed, regs[i])

        def store_b_regs(regs: list[fx.Vector]) -> None:
            for i in range_constexpr(PTB):
                c = tid * fx.Int32(PTB) + fx.Int32(i)
                local_row = c // fx.Int32(CHUNKS_PER_ROW)
                local_k_packed = (c % fx.Int32(CHUNKS_PER_ROW)) * fx.Int32(PACKED_CHUNK)
                store2_lds(bs_base, local_row * fx.Int32(STRIDE) + local_k_packed, regs[i])

        def load_frag_lds(lds_ptr: fx.Pointer, row: fx.Int32, k_packed0: fx.Int32) -> fx.Vector:
            """One scalar i32 fragment (8 nibbles) for this lane's K-half."""
            off = fx.Int64(row) * fx.Int64(STRIDE) + fx.Int64(k_packed0) + fx.Int64(klane) * fx.Int64(4)
            i32_lds = as_i32(lds_ptr)
            view = fx.make_view(
                fx.add_offset(i32_lds, off // fx.Int64(4)),
                fx.make_layout(1, 1),
            )
            return Vec.from_elements([fx.Int32(Vec(view.load())[0])], fx.Int32)

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
                    k_packed_n = fx.Int32((kk + 1) * PACKED_CHUNK)
                    af_nxt = []
                    for ti in range_constexpr(TM):
                        row = wm * fx.Int32(TM * WM) + fx.Int32(ti * WM) + lane16
                        af_nxt.append(load_frag_lds(as_base, row, k_packed_n))
                    bf_nxt = []
                    for tj in range_constexpr(TN):
                        col = wn * fx.Int32(TN * WN) + fx.Int32(tj * WN) + lane16
                        bf_nxt.append(load_frag_lds(bs_base, col, k_packed_n))

                new_accs = []
                for ti in range_constexpr(TM):
                    for tj in range_constexpr(TN):
                        idx = ti * TN + tj
                        new_accs.append(wmma_iu4(af_cur[ti], bf_cur[tj], acc_list[idx]))
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

        # Epilogue: i32→out; optional scale. Wave32 column-distributed store.
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
                    if const_expr(out_name == "int32"):
                        out_v = fx.Int32(acc[e])
                    else:
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


def create_wmma_iu4_gemm_module(
    out_dtype: str = "bfloat16",
    cfg: TileConfig | tuple | None = None,
    *,
    skip_bounds: bool = False,
    w_scale_per_n: bool = True,
    sign_a: bool = True,
    sign_b: bool = True,
) -> Callable[..., None]:
    """Create a gfx120x iu4 GEMM launcher for one tile configuration."""
    if cfg is None:
        cfg = _CFG_64_64_64
    if isinstance(cfg, tuple):
        cfg = TileConfig(*cfg)
    return build_iu4_gemm_module(out_dtype, cfg, skip_bounds, w_scale_per_n, sign_a, sign_b)


def _ptr(t: torch.Tensor) -> PointerJitArg:
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _out_name(dtype: torch.dtype) -> str:
    import torch

    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.int32: "int32",
    }[dtype]


def iu4_gemm(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    prefer_native: bool = True,
    sign_a: bool = True,
    sign_b: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Native iu4 GEMM: packed int4 A/B → ``[M, N]`` (bf16/fp16/fp32 or i32).

    Typical call::

        from kernels.quant.rdna4_int4_codec import pack_int4_row_major
        a_packed = pack_int4_row_major(a_logical)  # [M, K] int4 → [M, K//2] int8
        b_packed = pack_int4_row_major(b_logical)  # [N, K] int4 → [N, K//2] int8
        y = iu4_gemm(a_packed, b_packed, scale_a, scale_b, out_dtype=torch.bfloat16)

    Args:
        a_packed: ``[M, K//2]`` int8 (low nibble = even K). Pack with
            ``pack_int4_row_major``.
        b_packed: ``[N, K//2]`` int8, same pack (row-major N×K logical).
        scale_a: ``[M]`` fp32 (default ones); ignored when ``out_dtype=int32``.
        scale_b: ``[N]`` or ``[1]`` fp32 (default ones).
        prefer_native: ``True`` (default) uses iu4 WMMA when
            ``shapes_ok_for_native_iu4(M,N,K)`` (``K % 16 == 0``). ``False``
            unpacks to int8 and runs the iu8 linear path.

    Returns:
        ``[M, N]`` in ``out_dtype`` (default bf16; use int32 for raw accumulator).

    See ``docs/prebuilt_kernels_guide.md`` for call examples.
    """
    require_gfx120x(a_packed.device, what="iu4_gemm (gfx120x)")
    import torch

    from kernels.common.tensor_shim import _run_compiled
    from kernels.quant.rdna4_int4_codec import unpack_int4_row_major

    if a_packed.dim() != 2 or b_packed.dim() != 2:
        raise ValueError("a_packed / b_packed must be 2D")
    m, k_half = int(a_packed.shape[0]), int(a_packed.shape[1])
    n, k_half_b = int(b_packed.shape[0]), int(b_packed.shape[1])
    if k_half != k_half_b:
        raise ValueError(f"K mismatch: A packed {k_half} vs B {k_half_b}")
    k = k_half * 2
    if out_dtype is None:
        out_dtype = torch.bfloat16
    if stream is None:
        stream = torch.cuda.current_stream()

    use_native = prefer_native and shapes_ok_for_native_iu4(m, n, k)
    if not use_native:
        # Fallback: unpack → iu8 linear (or torch i32 ref when out is int32).
        a_i8 = unpack_int4_row_major(a_packed).contiguous()
        b_i8 = unpack_int4_row_major(b_packed).contiguous()
        if out_dtype == torch.int32:
            a_ref = a_i8.to(torch.int32)
            b_ref = b_i8.to(torch.int32)
            if not sign_a:
                a_ref = a_i8.to(torch.uint8).to(torch.int32)
            if not sign_b:
                b_ref = b_i8.to(torch.uint8).to(torch.int32)
            acc = (a_ref.to(torch.float32) @ b_ref.to(torch.float32).T).to(torch.int32)
            if out is None:
                return acc
            out.copy_(acc)
            return out
        from kernels.gemm.rdna4_int8_linear import (
            create_wmma_int8_linear_module,
        )
        from kernels.gemm.rdna4_int8_linear import (
            pick_tile_config as pick_i8,
        )

        if scale_a is None:
            scale_a = torch.ones(m, device=a_packed.device, dtype=torch.float32)
        if scale_b is None:
            scale_b = torch.ones(n, device=a_packed.device, dtype=torch.float32)
        scale_a = scale_a.to(device=a_packed.device, dtype=torch.float32).reshape(-1).contiguous()
        scale_b = scale_b.to(device=a_packed.device, dtype=torch.float32).reshape(-1).contiguous()
        w_per_n = scale_b.numel() != 1
        if out is None:
            out = torch.empty((m, n), device=a_packed.device, dtype=out_dtype)
        cfg = pick_i8(m, n, k)
        launch = create_wmma_int8_linear_module(
            _out_name(out_dtype),
            cfg,
            skip_bounds=(m % cfg.bm == 0 and n % cfg.bn == 0),
            w_scale_per_n=w_per_n,
        )
        _run_compiled(
            launch,
            _ptr(a_i8),
            _ptr(b_i8),
            _ptr(out),
            _ptr(scale_a),
            _ptr(scale_b),
            m,
            n,
            k,
            stream,
        )
        return out

    if scale_a is None:
        scale_a = torch.ones(m, device=a_packed.device, dtype=torch.float32)
    if scale_b is None:
        scale_b = torch.ones(n, device=a_packed.device, dtype=torch.float32)
    scale_a = scale_a.to(device=a_packed.device, dtype=torch.float32).reshape(-1).contiguous()
    scale_b = scale_b.to(device=a_packed.device, dtype=torch.float32).reshape(-1).contiguous()
    w_per_n = scale_b.numel() != 1
    if scale_a.numel() != m:
        raise ValueError(f"scale_a must have {m} elems, got {scale_a.numel()}")
    if w_per_n and scale_b.numel() != n:
        raise ValueError(f"scale_b must be scalar or [N]={n}, got {scale_b.numel()}")

    if out is None:
        out = torch.empty((m, n), device=a_packed.device, dtype=out_dtype)
    a_c = a_packed.contiguous()
    b_c = b_packed.contiguous()
    cfg = pick_tile_config(m, n, k)
    launch = create_wmma_iu4_gemm_module(
        _out_name(out_dtype),
        cfg,
        skip_bounds=(m % cfg.bm == 0 and n % cfg.bn == 0),
        w_scale_per_n=w_per_n,
        sign_a=sign_a,
        sign_b=sign_b,
    )
    _run_compiled(
        launch,
        _ptr(a_c),
        _ptr(b_c),
        _ptr(out),
        _ptr(scale_a),
        _ptr(scale_b),
        m,
        n,
        k,
        stream,
    )
    return out


__all__ = [
    "KERNEL_NAME",
    "TileConfig",
    "pick_tile_config",
    "shapes_ok_for_native_iu4",
    "build_iu4_gemm_module",
    "create_wmma_iu4_gemm_module",
    "iu4_gemm",
]
