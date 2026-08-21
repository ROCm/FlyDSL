# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Independent flash/flex-attention forward on the FlyDSL layout API (gfx950).

This is a from-scratch attention kernel written on the new CuTe-style layout API
(``fx.make_tiled_mma`` / ``make_fragment_{A,B,C}`` / ``fx.gemm`` / ``fx.copy`` /
swizzled LDS views), modelled on ``flydsl-examples/kernels/hgemm_layout_gfx950.py``
for the MMA/pipeline structure and ``kernels/norm/softmax_kernel.py`` for the
softmax numerics. It does NOT reuse the legacy raw-MFMA ``flash_attn_generic``
path.

Phase 0 (this file): a correct DENSE forward, single-stage, no flex hooks. One
workgroup computes one ``[BLOCK_M, D]`` query tile: load Q resident, loop over KV
``[BLOCK_N, D]`` tiles doing GEMM1 (S = Q@K^T), online softmax, the C->A bridge,
then GEMM2 (O += P@V); epilogue normalizes O by the row sum and stores it.

Target arch: gfx950 (CDNA4). Uses the cdna4 LDS transpose-read atom and the
gfx950 LDS swizzles; it is NOT expected to run on gfx942.
"""

from typing import Optional
import warnings

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl, arith

from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from kernels.attention.pipeline import (
    pipeline_stagger_enabled,
)

try:
    from flydsl.expr.rocdl.universal import make_buffer_ptr as _make_buffer_ptr
except ImportError:
    from flydsl.expr import buffer_ops
    from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace

    def _make_buffer_ptr(ptr, num_records_bytes=None):
        if num_records_bytes is None:
            num_records_bytes = fx.Int64(0xFFFFFFFF)
        elif not isinstance(num_records_bytes, fx.Int64):
            num_records_bytes = fx.Int64(num_records_bytes)
        buf_ptr_ty = fx.PointerType.get(
            elem_ty=ptr.element_type.ir_type,
            address_space=TargetAddressSpace.BufferDesc,
            alignment=ptr.alignment,
        )
        return fx.make_ptr(
            buf_ptr_ty,
            [
                ptr,
                fx.Int16(0).ir_value(),
                num_records_bytes.ir_value(),
                fx.Int32(buffer_ops._get_buffer_flags()).ir_value(),
            ],
        )

GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16
FLEX_DTYPE_BF16 = 2
FLEX_DTYPE_FP16 = 3

_LOG2E = 1.4426950408889634

MASK_NONE = 0
MASK_CAUSAL = 1
MASK_SLIDING_WINDOW = 2

SCORE_NONE = 0
SCORE_ALIBI = 1


class FlexMod:
    has_mask = False
    has_score = False
    needs_safe_norm = False

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        return fx.Int32(0), fx.Int32(n_kv_tiles)

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        return fx.Int32(0) != fx.Int32(0)

    def apply_mask(self, score, q_idx, kv_idx):
        return score

    def apply_score(self, score, b, h, q_idx, kv_idx):
        return score


class CausalMask(FlexMod):
    has_mask = True
    needs_safe_norm = True

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        raw_hi = (q_max_wg + fx.Int32(block_n)) // fx.Int32(block_n)
        kv_hi = fx.Int32(arith.minsi(raw_hi.ir_value(), fx.Int32(n_kv_tiles).ir_value()))
        return fx.Int32(0), kv_hi

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
        return kv_tile_end > q_idx

    def apply_mask(self, score, q_idx, kv_idx):
        return (kv_idx <= q_idx).select(score, fx.Float32(-1e9))


class SlidingWindowMask(FlexMod):
    has_mask = True
    needs_safe_norm = True

    def __init__(self, window):
        self.window = window

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        raw_hi = (q_max_wg + fx.Int32(block_n)) // fx.Int32(block_n)
        kv_hi = fx.Int32(arith.minsi(raw_hi.ir_value(), fx.Int32(n_kv_tiles).ir_value()))
        raw_lo = (q_min_wg - fx.Int32(self.window)) // fx.Int32(block_n)
        kv_lo = fx.Int32(arith.maxsi(raw_lo.ir_value(), fx.Int32(0).ir_value()))
        return kv_lo, kv_hi

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
        kv_tile_start = kv_tile_idx * fx.Int32(block_n)
        too_far = kv_tile_end > q_idx
        out_of_window = (q_idx - kv_tile_start) > fx.Int32(self.window)
        return too_far | out_of_window

    def apply_mask(self, score, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        in_window = (q_idx - kv_idx) <= fx.Int32(self.window)
        return (causal & in_window).select(score, fx.Float32(-1e9))


class AlibiScore(FlexMod):
    has_score = True

    def __init__(self, slope):
        self.slope = slope

    def apply_score(self, score, b, h, q_idx, kv_idx):
        bias = (kv_idx - q_idx).to(fx.Float32) * fx.Float32(self.slope)
        return fx.Float32(score) + bias


class CompositeMod(FlexMod):
    def __init__(self, score_mod, mask_mod):
        self._score = score_mod
        self._mask = mask_mod
        self.has_score = score_mod.has_score
        self.has_mask = mask_mod.has_mask
        self.needs_safe_norm = mask_mod.needs_safe_norm

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        return self._mask.kv_range(q_min_wg, q_max_wg, n_kv_tiles, block_n)

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        return self._mask.tile_needs_mask(kv_tile_idx, q_idx, block_n)

    def apply_mask(self, score, q_idx, kv_idx):
        return self._mask.apply_mask(score, q_idx, kv_idx)

    def apply_score(self, score, b, h, q_idx, kv_idx):
        return self._score.apply_score(score, b, h, q_idx, kv_idx)


def _build_mod(mask_type, score_type, mask_window=0, score_alibi_slope=0.0):
    _mask = {
        MASK_NONE: FlexMod(),
        MASK_CAUSAL: CausalMask(),
        MASK_SLIDING_WINDOW: SlidingWindowMask(mask_window),
    }[mask_type]
    _score = {
        SCORE_NONE: FlexMod(),
        SCORE_ALIBI: AlibiScore(score_alibi_slope),
    }[score_type]
    if _mask.has_mask or _score.has_score:
        return CompositeMod(_score, _mask)
    return FlexMod()


@fx.struct
class FlexAttnParam:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    head_dim: fx.Constexpr[int]
    num_heads_q: fx.Constexpr[int]
    num_heads_kv: fx.Constexpr[int]
    # wave tiling
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    # num_groups independent 2-wave query subtiles per workgroup, all sharing the
    # same KV loop. Each group runs the validated 32-row body on rows
    # [group*block_m : (group+1)*block_m); K/V are meant to be loaded once and reused
    # across all groups (strategy A). Total query rows per workgroup = num_groups*block_m.
    num_groups: fx.Constexpr[int]
    # mma shape (bf16/f16: 16x16x32 to match hgemm)
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    # derived
    group_threads: fx.Constexpr[int]  # threads per group = m_waves*n_waves*wave_size
    block_threads: fx.Constexpr[int]  # = num_groups * group_threads
    gqa_group: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    n_kv_tiles: fx.Constexpr[int]  # seqlen_kv // block_n (KV loop is compile-time unrolled)
    pipe_depth: fx.Constexpr[int]  # 1 = monolithic, 2 = decomposed pipeline
    pipe_stages: fx.Constexpr[int]  # deprecated: stagger follows num_groups/pipe_depth/m_waves
    # True = exact per-row softmax; False = approximate column softmax (mma_m=32 only)
    accurate_softmax: fx.Constexpr[bool]
    # flex mods: integer type IDs (MASK_NONE/CAUSAL/SLIDING_WINDOW, SCORE_NONE/ALIBI)
    mask_type: fx.Constexpr[int]
    score_type: fx.Constexpr[int]
    mask_window: fx.Constexpr[int]  # sliding window size (only used when mask_type==MASK_SLIDING_WINDOW)
    score_alibi_slope: fx.Constexpr[float]  # alibi slope (only used when score_type==SCORE_ALIBI)


def make_flex_attn_param(
    seqlen_kv: int,
    dtype_id: int = FLEX_DTYPE_BF16,
    block_m: int = 32,
    block_n: int = 32,
    head_dim: int = 128,
    num_heads_q: int = 8,
    num_heads_kv: int = 8,
    m_waves: int = 1,
    n_waves: int = 1,
    num_groups: int = 1,
    mma_m: int = 32,
    mma_n: int = 32,
    mma_k: int = 16,
    pipe_depth: int = 1,
    pipe_stages: int = 1,
    accurate_softmax: bool = True,
    mask_type: int = MASK_NONE,
    score_type: int = SCORE_NONE,
    mask_window: int = 0,
    score_alibi_slope: float = 0.0,
) -> FlexAttnParam:
    if dtype_id not in (FLEX_DTYPE_BF16, FLEX_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m <= 0 or block_n <= 0 or head_dim <= 0:
        raise ValueError("block_m, block_n, head_dim must be positive")
    # Phase 0 proven config (see plan progress log): fx.gemm with MFMA 16x16x16 has
    # a lowering bug on this build, so both GEMMs use mma_k=32 -> block_n multiple of
    # 32. The C-fragment slot->row map is only locked for block_m=32 (2 M-waves x
    # mma_m=16), one N-wave; larger block_m needs a per-slot row map (TODO perf).
    if not (mma_m == mma_n and (mma_m, mma_k) in ((16, 32), (16, 16), (32, 16), (32, 8))):
        raise ValueError(f"unsupported MMA shape {mma_m}x{mma_n}x{mma_k}")
    if block_m % (m_waves * mma_m) != 0:
        raise ValueError(f"block_m ({block_m}) must be divisible by m_waves*mma_m ({m_waves * mma_m})")
    if block_n % (n_waves * mma_n) != 0:
        raise ValueError(f"block_n ({block_n}) must be divisible by n_waves*mma_n ({n_waves * mma_n})")
    if n_waves != 1:
        raise ValueError("n_waves must be 1 (softmax row reduction requires all N-lanes in one wave)")
    if not accurate_softmax and mma_m != 32:
        raise ValueError("accurate_softmax=False (approximate column softmax) requires mma_m=32")
    if num_groups < 1:
        raise ValueError("num_groups must be >= 1")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("num_heads_q must be divisible by num_heads_kv (GQA)")
    if head_dim % mma_k != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by mma_k ({mma_k})")
    if seqlen_kv % block_n != 0:
        raise ValueError(f"seqlen_kv ({seqlen_kv}) must be a multiple of block_n ({block_n})")
    if pipe_stages not in (1, 2):
        raise ValueError("pipe_stages must be 1 or 2")
    if pipe_stages >= 2 and pipe_depth < 2:
        raise ValueError("pipe_stages=2 requires pipe_depth>=2 (decomposed pipeline)")
    if pipe_depth >= 2 and not pipeline_stagger_enabled(
        depth=pipe_depth, num_groups=num_groups, m_waves=m_waves,
    ):
        raise ValueError(
            "pipe_depth>=2 requires pipeline stagger: num_groups>=2 and m_waves>=2 "
            f"(got num_groups={num_groups}, m_waves={m_waves})"
        )
    if pipe_stages >= 2 and not pipeline_stagger_enabled(
        depth=pipe_depth, num_groups=num_groups, m_waves=m_waves,
    ):
        warnings.warn(
            "pipe_stages=2 does not enable stagger; stagger is auto when "
            "pipe_depth>=2, num_groups>=2, and m_waves>=2",
            stacklevel=2,
        )

    in_dbytes = 2  # bf16/f16
    group_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    block_threads = num_groups * group_threads
    _max_waves = 8
    if block_threads > _max_waves * GFX950_WAVE_SIZE:
        raise ValueError(
            f"block_threads ({block_threads}) exceeds {_max_waves} SIMDs/CU limit "
            f"({_max_waves * GFX950_WAVE_SIZE} threads); reduce num_groups or m_waves"
        )

    return FlexAttnParam(
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        m_waves=m_waves,
        n_waves=n_waves,
        num_groups=num_groups,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        group_threads=group_threads,
        block_threads=block_threads,
        gqa_group=num_heads_q // num_heads_kv,
        in_data_bytes=in_dbytes,
        n_kv_tiles=seqlen_kv // block_n,
        pipe_depth=pipe_depth,
        pipe_stages=pipe_stages,
        accurate_softmax=accurate_softmax,
        mask_type=mask_type,
        score_type=score_type,
        mask_window=mask_window,
        score_alibi_slope=score_alibi_slope,
    )


def make_flex_attn_kernel_name(param: FlexAttnParam) -> str:
    dtype_str = "fp16" if param.dtype_id == FLEX_DTYPE_FP16 else "bf16"
    name = f"flex_attn_{dtype_str}_m{param.block_m}n{param.block_n}d{param.head_dim}"
    name += f"_w{param.m_waves}x{param.n_waves}g{param.num_groups}"
    name += "_dense"
    name += "_rsm" if param.accurate_softmax else "_csm"
    name += f"_pd{param.pipe_depth}"
    if pipeline_stagger_enabled(
        depth=int(param.pipe_depth),
        num_groups=int(param.num_groups),
        m_waves=int(param.m_waves),
    ):
        name += "_stg"
    return name


_FM = fx.arith.FastMathFlags.fast


def _elem_dtype(dtype_id):
    return fx.Float16 if dtype_id == FLEX_DTYPE_FP16 else fx.BFloat16


def _size_scalar(shape) -> int:
    s = fx.size(shape)
    if hasattr(s, "unpack"):
        return s.unpack()
    if hasattr(s, "is_static") and s.is_static:
        v = s.to_py_value()
        if isinstance(v, tuple):
            return int(v[0]) if len(v) == 1 else int(v)
        return int(v)
    raise TypeError(f"cannot get static size from {type(s)!r}")


def _to_elem(val, elem_ty):
    if hasattr(val, "to"):
        return val.to(elem_ty)
    return fx.Float32(val).to(elem_ty)



def _hw_exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def _permlane32_reduce(x, mode):
    """Cross-half-wave reduce via permlane32_swap (1 instruction)."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as _arith
    from flydsl._mlir.dialects import llvm
    v_i32 = fx.Int32(_arith.bitcast(T.i32, fx.Float32(x).ir_value()))
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32.ir_value(), v_i32.ir_value(), False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    lhs = fx.Float32(_arith.bitcast(T.f32, lhs_i32))
    rhs = fx.Float32(_arith.bitcast(T.f32, rhs_i32))
    if mode == "max":
        return lhs.maximumf(rhs)
    else:
        return lhs.addf(rhs, fastmath=_FM)



def _mfma_acc(a, b, c, mma_atom):
    """Single MFMA call: C += A × B. Returns updated accumulator."""
    from flydsl._mlir.dialects import fly
    acc_ty = c.type
    return fly.mma_atom_call_ssa([acc_ty], mma_atom, a, b, c)



@flyc.kernel
def flex_attn_fwd_gfx950_kernel(
    o: fx.Tensor,       # [B, Sq, Hq, D]
    q: fx.Tensor,       # [B, Sq, Hq, D]
    k: fx.Tensor,       # [B, Skv, Hkv, D]
    v: fx.Tensor,       # [B, Skv, Hkv, D]
    seqlen_q: fx.Int32,
    seqlen_kv: fx.Int32,
    num_batches: fx.Int32,
    scale: fx.Float32,
    tiled_mma_qk: fx.TiledMma,
    tiled_mma_pv: fx.TiledMma,
    param: FlexAttnParam,
):
    block_m = param.block_m
    block_n = param.block_n
    head_dim = param.head_dim
    elem_dtype = _elem_dtype(param.dtype_id)

    tid = fx.thread_idx.x
    # Strategy A: num_groups independent 2-wave query subtiles per workgroup, all
    # driving the SAME KV loop so K/V (staged in LDS) is reused across groups. Each
    # group runs the validated 128-thread body via local_tid; group g owns query rows
    # [(q_tile*num_groups + g)*block_m : +block_m).
    num_groups = param.num_groups
    group_threads = param.group_threads  # 128 (m_waves*n_waves*wave_size)
    group = tid // group_threads
    local_tid = tid % group_threads
    # grid.x = super-tile index (num_groups query subtiles); grid.y = head; grid.z = batch.
    q_tile = fx.block_idx.x
    h_idx = fx.block_idx.y
    b_idx = fx.block_idx.z
    kv_head = h_idx // param.gqa_group

    q_start = (q_tile * num_groups + group) * block_m

    n_kv_tiles = param.n_kv_tiles  # compile-time: seqlen_kv // block_n (validated on host)

    # ── LDS: K/V staging (shared across all groups) + per-group P bridge ──────
    kv_tile_elems = block_n * head_dim
    _lds_ring_slots = max(2, int(param.pipe_depth))

    @fx.struct
    class SharedStorage:
        k_lds_0: fx.Array[elem_dtype, kv_tile_elems, 16]
        k_lds_1: fx.Array[elem_dtype, kv_tile_elems, 16]
        v_lds_0: fx.Array[elem_dtype, kv_tile_elems, 16]
        v_lds_1: fx.Array[elem_dtype, kv_tile_elems, 16]
        p: fx.Array[elem_dtype, num_groups * block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    sK_ptr = [storage.k_lds_0.peek().ptr, storage.k_lds_1.peek().ptr]
    sV_ptr = [storage.v_lds_0.peek().ptr, storage.v_lds_1.peek().ptr]

    _k_swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
    _k_base_layout = fx.make_composed_layout(
        _k_swizzle, fx.make_layout((block_n, head_dim), (head_dim, 1))
    )
    sK = [fx.make_view(sK_ptr[i], _k_base_layout) for i in range_constexpr(_lds_ring_slots)]
    # V LDS: 4 compact sub-tiles [block_n, 32]:(32, 1) per D-chunk.
    _v_subtile_elems = block_n * 32
    # Per-group P-bridge region: group g uses [g*block_m*block_n : +block_m*block_n).
    sP = fx.make_view(
        storage.p.peek().ptr + group * fx.Int32(block_m * block_n),
        fx.make_layout((block_m, block_n), (block_n, 1)),
    )

    # ── per-(batch,head) [S, D] views of the BSHD tensors ─────────────────────
    # Element (b,s,h,d) at b*Sq*Hq*D + s*Hq*D + h*D + d.  q/o slice: base offset
    # b*Sq*Hq*D + h*D + q_start*Hq*D, row-stride Hq*D. k slice uses Hkv/kv_head.
    hq = param.num_heads_q
    hkv = param.num_heads_kv
    q_off = b_idx * seqlen_q * hq * head_dim + h_idx * head_dim + q_start * hq * head_dim
    o_off = q_off
    k_off = b_idx * seqlen_kv * hkv * head_dim + kv_head * head_dim
    # V is [B, Skv, Hkv, D] (un-transposed): element (b,s,h,d) at
    # b*Skv*Hkv*D + s*Hkv*D + h*D + d.  This head's base:
    v_off = b_idx * seqlen_kv * hkv * head_dim + kv_head * head_dim

    # Bounded Q descriptor: the tiled copy B (BufferCopy128b) can overshoot
    # head_dim for the last K-group's final 128b load. Use total tensor size
    # as num_records so the HW clamps OOB reads to 0.
    _q_total_bytes = num_batches * seqlen_q * fx.Int32(hq * head_dim * param.in_data_bytes)
    q_it = _make_buffer_ptr(
        fx.recast_iter(elem_dtype, fx.get_iter(q)),
        num_records_bytes=_q_total_bytes,
    )
    gQ = fx.make_view(q_it + fx.Int32(q_off), fx.make_layout((block_m, head_dim), (hq * head_dim, 1)))

    # Each group runs the validated 128-thread MMA partition via local_tid.
    thr_qk = tiled_mma_qk.thr_slice(local_tid)

    ca = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    uca = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

    # Q resident: load once into the GEMM1 B-fragment (reused every KV tile).
    # QK uses K=A, Q=B so C's M-rows = score indices, allowing register C→B pack for PV.
    tcB_q = fx.make_tiled_copy_B(ca, tiled_mma_qk).get_slice(local_tid)
    frag_Q = thr_qk.make_fragment_B(gQ)
    fx.copy(ca, tcB_q.partition_S(gQ), tcB_q.retile(frag_Q))
    _is_32x32 = int(param.mma_m) == 32
    if const_expr(_is_32x32):
        n_q = _size_scalar(frag_Q.shape)
        for qi in range_constexpr(n_q):
            frag_Q[qi] = _to_elem(_to_elem(frag_Q[qi], fx.Float32) * scale, elem_dtype)

    # Persistent O accumulator: 4 × v16f32 (one per D-chunk).
    # With V=A, P=B PV GEMM: each v16f32 has 16 D-values at 1 query-row per lane.
    Vec = fx.Vector
    _n_d_chunks = head_dim // 32
    o_accs_init = [Vec.filled(16, 0.0, fx.Float32).ir_value()
                   for _ in range_constexpr(_n_d_chunks)]

    # Per-slot row map: thr_qk.partition_C partitions by THIS thread's wave, so
    # n_c is always this lane's slot count (not the full tile). For MFMA 16x16
    # with n_waves=1, each lane has 4 M-values × (block_n/mma_n) N-repeats slots.
    # The first half and second half are the two column-groups of the same rows,
    # so npair = n_c // 2 gives the number of distinct row-indices this lane owns,
    # and i % npair maps each slot to its row. This holds for any m_waves because
    # thr_slice already selects the per-wave partition.
    n_c = _size_scalar(thr_qk.partition_C(sP).shape)
    # After QK operand swap (K=A, Q=B), C's M-rows = score indices, N-cols = query.
    # Each lane has 16 score values at 1 query column. npair=1: single max/sum per lane.
    # This gives exact per-query-row softmax (permlane32 combines the two score halves).
    if const_expr(_is_32x32):
        npair = 1
    else:
        npair = n_c // 2

    if const_expr(_is_32x32):
        scale_log2e = fx.Float32(_LOG2E)
    else:
        scale_log2e = scale * fx.Float32(_LOG2E)

    # m_i lives in log2-scaled space (pre-multiplied by scale_log2e) so exp2
    # in the softmax hot loop is just subtract + exp2 with no per-element multiply.
    _M_NEG_FLOOR_SCALED = -60.0 * _LOG2E
    m_i = [fx.Float32(_M_NEG_FLOOR_SCALED) for _ in range_constexpr(npair)]
    l_i = [fx.Float32(0.0) for _ in range_constexpr(npair)]

    # ── KV-loop via stage closures ──────────────────────────────────────
    # Each closure below is one pipeline stage, containing the actual
    # layout-API code extracted from the original monolithic loop body.
    # The closures capture all kernel-local objects (copy atoms, fragments,
    # LDS views, DMA descriptors) so the pipeline module never needs them.

    # Copy builders and fragments for double-buffered K/V
    # K is now the A operand of QK GEMM (K=A, Q=B).
    tcA_k_lds = [fx.make_tiled_copy_A(uca, tiled_mma_qk).get_slice(local_tid) for _ in range_constexpr(_lds_ring_slots)]
    frag_K = [thr_qk.make_fragment_A(sK[i]) for i in range_constexpr(_lds_ring_slots)]

    # V is loaded as A operand for PV GEMM (V=A, P=B).
    # V LDS has 4 compact sub-tiles [block_n, 32]:(32, 1). LDSReadTrans16_64b
    # transposes each [block_n, 32] → [32, block_n] = A[M=D_chunk, K=score].
    _v_tr_atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype)
    # View sub-tiles as [M=32(D), K=block_n(score)]:(1, 32) — column-major.
    # The transpose atom reads score-contiguous data from LDS and delivers A[M=D, K=score].
    # DMA infrastructure
    block_threads = param.block_threads
    _dma_bytes = GFX950_DMA_BYTES
    _kv_tile_bytes = kv_tile_elems * param.in_data_bytes
    _dma_ops_per_thread = _kv_tile_bytes // (block_threads * _dma_bytes)
    dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    _k_row_stride_bytes = hkv * head_dim * param.in_data_bytes
    _k_row_bytes = head_dim * param.in_data_bytes
    # V DMA: sub-tile rows are 32 bf16 = 64 bytes. Global stride between score rows
    # is hkv * head_dim elements (V is [B, Skv, Hkv, D], D contiguous per score row).
    _v_subtile_row_bytes = 32 * param.in_data_bytes  # 64 bytes per sub-tile row
    _v_row_stride_bytes = hkv * head_dim * param.in_data_bytes
    gK_flat = fx.rocdl.make_buffer_tensor(
        fx.Tensor(fx.make_view(fx.recast_iter(fx.Int8, fx.get_iter(k)), fx.make_layout(0x7FFFFFFF, 1))),
        max_size=True,
    )
    gV_flat = fx.rocdl.make_buffer_tensor(
        fx.Tensor(fx.make_view(fx.recast_iter(fx.Int8, fx.get_iter(v)), fx.make_layout(0x7FFFFFFF, 1))),
        max_size=True,
    )
    k_div = fx.logical_divide(gK_flat, fx.make_layout(1, 1))
    v_div = fx.logical_divide(gV_flat, fx.make_layout(1, 1))
    sK_i8 = [fx.recast_iter(fx.Int8, sK_ptr[i]) for i in range_constexpr(_lds_ring_slots)]
    sV_i8 = [fx.recast_iter(fx.Int8, sV_ptr[i]) for i in range_constexpr(_lds_ring_slots)]
    # ── Stage: DMA K+V global → LDS ─────────────────────────────────────
    def _k_swizzled_col(tile_row, tile_col_elem):
        """Apply K swizzle to get the global column index for a given LDS position."""
        elem_off = fx.get_scalar(fx.crd2idx(
            fx.make_int_tuple((tile_row, tile_col_elem)), _k_base_layout
        ))
        return elem_off % head_dim

    def _stage_kv_to_lds(kv_idx, buf, do_k, do_v):
        wave_off = rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes))
        _step_bytes = block_threads * _dma_bytes
        if const_expr(do_k):
            k_global_base = k_off * param.in_data_bytes + kv_idx * block_n * _k_row_stride_bytes
            lds_k = fx.add_offset(sK_i8[buf], wave_off)
            for i in range_constexpr(_dma_ops_per_thread):
                if const_expr(i > 0):
                    lds_k = fx.add_offset(lds_k, _step_bytes)
                flat_byte = i * block_threads * _dma_bytes + tid * _dma_bytes
                tile_row = flat_byte // _k_row_bytes
                tile_col_elem = (flat_byte % _k_row_bytes) // param.in_data_bytes
                swiz_col = _k_swizzled_col(tile_row, tile_col_elem)
                gmem_byte = k_global_base + tile_row * _k_row_stride_bytes + swiz_col * param.in_data_bytes
                fx.copy(dma_atom, fx.slice(k_div, (None, fx.Int32(gmem_byte))),
                        fx.make_view(lds_k, fx.make_layout(1, 1)))
        if const_expr(do_v):
            v_global_base = v_off * param.in_data_bytes + kv_idx * fx.Int32(block_n * _v_row_stride_bytes)
            lds_v = fx.add_offset(sV_i8[buf], wave_off)
            for i in range_constexpr(_dma_ops_per_thread):
                if const_expr(i > 0):
                    lds_v = fx.add_offset(lds_v, _step_bytes)
                flat_byte = i * block_threads * _dma_bytes + tid * _dma_bytes
                tile_row = flat_byte // _v_subtile_row_bytes
                tile_col_byte = flat_byte % _v_subtile_row_bytes
                dc = tile_row // block_n
                score_row = tile_row % block_n
                d_global_byte = dc * 32 * param.in_data_bytes + tile_col_byte
                gmem_byte = fx.Int32(v_global_base) + score_row * fx.Int32(_v_row_stride_bytes) + d_global_byte
                fx.copy(dma_atom, fx.slice(v_div, (None, fx.Int32(gmem_byte))),
                        fx.make_view(lds_v, fx.make_layout(1, 1)))

    from kernels.attention.pipeline import (
        InfraContext,
        pipeline_stagger_enabled,
    )

    _pd_for_sched = int(param.pipe_depth)

    class _LayoutSchedTraits:
        HEAD_DIM = int(param.head_dim)

    def load_kv(tile_idx, slot):
        _stage_kv_to_lds(tile_idx, slot, True, True)
        return []

    def load_k(tile_idx, slot):
        _stage_kv_to_lds(tile_idx, slot, True, False)
        return []

    def load_v(tile_idx, slot):
        _stage_kv_to_lds(tile_idx, slot, False, True)
        return []

    # V transpose read using swa's copy-atom pattern on compact [32, 32] sub-tiles.
    _v_tr_layout = fx.make_layout(4, 1)  # 4 bf16 per transpose read
    _v_smem_layout = fx.make_layout((block_n, 32), (32, 1))  # sub-tile layout
    # Lane address decomposition (same as swa for V sub-tiles):
    _v_row_off = ((local_tid % 16) // 4) + ((local_tid // 32) * 4)
    _v_col_off = ((local_tid % 4) * 4) + (16 * ((local_tid % 32) // 16))
    _v_lane_elem = fx.Int32(fx.get_scalar(
        fx.crd2idx(fx.make_int_tuple((fx.Int32(_v_row_off), fx.Int32(_v_col_off % 32))),
                   _v_smem_layout)
    ))

    def _make_read_v(slot):
        _slot_v_ptr = sV_ptr[slot]
        def _read():
            base_ptr = fx.add_offset(_slot_v_ptr, fx.make_int_tuple(_v_lane_elem))
            halves = [[None] * 4 for _ in range_constexpr(_n_d_chunks)]
            for k_sub in range_constexpr(4):
                for dc in range_constexpr(_n_d_chunks):
                    off = (dc * 4 + k_sub) * _v_subtile_elems // 4
                    src = fx.make_view(
                        fx.add_offset(base_ptr, fx.make_int_tuple(off)),
                        _v_tr_layout,
                    )
                    dst = fx.make_rmem_tensor(_v_tr_layout, elem_dtype)
                    fx.copy(_v_tr_atom, src, dst)
                    halves[dc][k_sub] = Vec(dst.load())
            v_lo_out = [None] * _n_d_chunks
            v_hi_out = [None] * _n_d_chunks
            for dc in range_constexpr(_n_d_chunks):
                v_lo_out[dc] = halves[dc][0].shuffle(halves[dc][1], list(range(8))).ir_value()
                v_hi_out[dc] = halves[dc][2].shuffle(halves[dc][3], list(range(8))).ir_value()
            return v_lo_out, v_hi_out
        return _read

    read_v_slot = [_make_read_v(i) for i in range_constexpr(_lds_ring_slots)]

    def read_k_work(slot):
        fx.copy(uca, tcA_k_lds[slot].partition_S(sK[slot]), tcA_k_lds[slot].retile(frag_K[0]))
        return []

    def gemm1_qk(frag_Q_in, frag_K_in):
        frag_S_out = thr_qk.make_fragment_C(sP)
        frag_S_out.fill(0.0)
        fx.gemm(tiled_mma_qk, frag_S_out, frag_K_in, frag_Q_in, frag_S_out)
        return [frag_S_out]

    # ── Flex score/mask mod application ────────────────────────────────────
    # MFMA 32x32x16 C fragment with K=A, Q=B swap:
    #   q_idx = q_start + local_tid % 32 (same for all 16 elements)
    #   kv_in_tile(e) = 8*(e//4) + e%4 + 4*(local_tid//32)
    flex_mod = _build_mod(int(param.mask_type), int(param.score_type),
                          int(param.mask_window), float(param.score_alibi_slope))
    mod_has_score = flex_mod.has_score
    mod_has_mask = flex_mod.has_mask
    _mod_apply_score = flex_mod.apply_score
    _mod_apply_mask = flex_mod.apply_mask
    _mod_tile_needs_mask = flex_mod.tile_needs_mask
    b_i32 = fx.Int32(arith.index_cast(T.i32, b_idx))
    h_i32 = fx.Int32(arith.index_cast(T.i32, h_idx))
    q_idx_mod = fx.Int32(arith.index_cast(T.i32, q_start)) + fx.Int32(local_tid % 32)
    lane_group_off = fx.Int32((local_tid // 32) * 4)
    kv_offsets = [8 * (e // 4) + (e % 4) for e in range(n_c)]

    def apply_mods(frag_S_in, kv_tile_idx):
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        if const_expr(mod_has_score):
            for e in range_constexpr(n_c):
                kv_idx = kv_base + fx.Int32(kv_offsets[e])
                frag_S_in[e] = _mod_apply_score(frag_S_in[e], b_i32, h_i32, q_idx_mod, kv_idx)
        if const_expr(mod_has_mask):
            if _mod_tile_needs_mask(kv_tile_idx, q_idx_mod, block_n):
                for e in range_constexpr(n_c):
                    kv_idx = kv_base + fx.Int32(kv_offsets[e])
                    frag_S_in[e] = _mod_apply_mask(frag_S_in[e], q_idx_mod, kv_idx)

    # _n_d_chunks defined above as head_dim // 32 (= 4 for D=128).

    def _scale_o_vec(o_accs_in, scale_scalar):
        """Vectorized O rescale: broadcast scalar to vec16, multiply per D-chunk."""
        scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
        for dc in range_constexpr(_n_d_chunks):
            o_vec = Vec(o_accs_in[dc])
            o_accs_in[dc] = (o_vec * scale_vec).ir_value()

    def softmax_start(frag_S_in, m_i_in):
        # Pre-scale S by log2e, return scaled scores as a list of SSA values.
        _sl2e_vec = Vec.from_elements([scale_log2e], fx.Float32).broadcast_to(16)
        s_elems = [frag_S_in[i] for i in range_constexpr(n_c)]
        s_scaled = Vec.from_elements(s_elems, fx.Float32) * _sl2e_vec
        s_out = [s_scaled[i] for i in range_constexpr(n_c)]
        # Max reduce + cross-lane
        tile_max = s_out[0]
        for i in range_constexpr(1, n_c):
            tile_max = tile_max.maximumf(s_out[i])
        tile_max = _permlane32_reduce(tile_max, "max")
        # m_new and correction factor
        m_new = m_i_in[0].maximumf(tile_max)
        corr_scalar = _hw_exp2(m_i_in[0] - m_new)
        m_i_in[0] = m_new
        return corr_scalar, s_out

    def softmax_finish(s_scaled, m_i_in, l_i_in, o_accs_in, corr_scalar):
        # s_scaled: list of n_c pre-scaled SSA values from softmax_start.
        m_new = m_i_in[0]
        p_elems = [_hw_exp2(s_scaled[i] - m_new) for i in range_constexpr(n_c)]
        # Sum reduce + cross-lane
        p_vec = Vec.from_elements(p_elems, fx.Float32)
        local_sum = p_vec.reduce("add", init_val=fx.Float32(0.0), fastmath=_FM)
        local_sum = _permlane32_reduce(local_sum, "sum")
        # l_i update + O rescale
        corr = [corr_scalar]
        l_i_in[0] = l_i_in[0] * corr_scalar + local_sum
        _scale_o_vec(o_accs_in, corr_scalar)
        return [p_elems, m_i_in, l_i_in, o_accs_in, corr]

    # ── Register-only PV GEMM (V=A, P=B) ──────────────────────────────────
    # After QK swap (K=A, Q=B), C's M-rows = score indices.
    # C→B is register-local: pack 16 f32 → 2 × v8bf16.
    # V is loaded as A from LDS per D-chunk.
    _pv_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))

    _is_bf16 = int(param.dtype_id) == FLEX_DTYPE_BF16

    def _pack_8_f32_to_v8elem(vals_8):
        """Pack 8 f32 values into v8 of elem_dtype (bf16 or f16)."""
        if const_expr(_is_bf16):
            pairs = []
            for j in range_constexpr(4):
                pairs.append(rocdl.cvt_pk_bf16_f32(vals_8[j * 2], vals_8[j * 2 + 1]))
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
        else:
            elems = []
            for j in range_constexpr(8):
                elems.append(fx.Float32(vals_8[j]).to(elem_dtype))
            return Vec.from_elements(elems, elem_dtype).ir_value()

    def _pack_p_b(frag_P_in):
        """Pack 16 C-fragment f32 values into 2 v8elem MFMA B packs."""
        p_lo = _pack_8_f32_to_v8elem([frag_P_in[i] for i in range(8)])
        p_hi = _pack_8_f32_to_v8elem([frag_P_in[8 + i] for i in range(8)])
        return p_lo, p_hi


    def pv_gemm_register(frag_P_in, v_lo_regs, v_hi_regs, o_accs):
        """PV GEMM without LDS P bridge: P packed as B, V pre-read as A.

        v_lo_regs/v_hi_regs: lists of v8elem per D-chunk, pre-read from LDS
        in the non-contiguous C-score order by _read_v_scattered.
        """
        p_lo, p_hi = _pack_p_b(frag_P_in)
        for dc in range_constexpr(_n_d_chunks):
            o_accs[dc] = _mfma_acc(v_lo_regs[dc], p_lo, o_accs[dc], _pv_mma_atom)
            o_accs[dc] = _mfma_acc(v_hi_regs[dc], p_hi, o_accs[dc], _pv_mma_atom)

    def dualwave_cluster_sync(cluster_index):
        if const_expr(cluster_index == 1 and _pd_for_sched >= 2):
            rocdl.sched_barrier(0)
        else:
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

    def stagger_open(stagger_i32):
        from flydsl._mlir import ir
        from flydsl._mlir.dialects import scf
        from flydsl.expr import arith
        from flydsl.expr.typing import T
        is_group_b = arith.cmpi(
            arith.CmpIPredicate.ne, stagger_i32, arith.constant(0, type=T.i32),
        )
        _if = scf.IfOp(is_group_b, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            rocdl.s_barrier()
            scf.YieldOp([])

    def stagger_close(stagger_i32):
        from flydsl._mlir import ir
        from flydsl._mlir.dialects import scf
        from flydsl.expr import arith
        from flydsl.expr.typing import T
        is_group_a = arith.cmpi(
            arith.CmpIPredicate.eq, stagger_i32, arith.constant(0, type=T.i32),
        )
        _if = scf.IfOp(is_group_a, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            rocdl.s_barrier()
            scf.YieldOp([])

    if const_expr(_is_32x32):
        _enable_stagger = int(num_groups) >= 2
    else:
        _enable_stagger = pipeline_stagger_enabled(
            depth=int(param.pipe_depth),
            num_groups=int(num_groups),
            m_waves=int(param.m_waves),
        )

    infra = InfraContext()
    infra.head_dim = head_dim
    infra.tiled_mma_qk = tiled_mma_qk
    infra.tiled_mma_pv = tiled_mma_pv
    infra.elem_dtype = elem_dtype
    infra.n_kv_tiles = n_kv_tiles
    if const_expr(_enable_stagger):
        if const_expr(_is_32x32):
            _half_groups = max(1, int(num_groups) // 2)
            _group_idx = fx.Int32(tid // int(param.group_threads))
            infra.stagger_i32 = rocdl.readfirstlane(
                fx.Int32.ir_type, _group_idx // fx.Int32(_half_groups),
            )
        else:
            infra.stagger_i32 = rocdl.readfirstlane(
                fx.Int32.ir_type, fx.Int32(local_tid // GFX950_WAVE_SIZE),
            )

    infra.traits = _LayoutSchedTraits()

    single_tile = True
    if const_expr(single_tile):
        # KV tile range: clamp to the mask's valid range to skip fully-masked tiles.
        _q_min_wg = fx.Int32(arith.index_cast(T.i32, q_tile)) * fx.Int32(num_groups * block_m)
        _q_max_wg = _q_min_wg + fx.Int32(num_groups * block_m - 1)
        _kv_lo, _kv_hi = flex_mod.kv_range(_q_min_wg, _q_max_wg, n_kv_tiles, block_n)

        # Double-buffered KV loop via scf.for with loop-carried m/l/O state.
        load_kv(_kv_lo, 0)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

        if const_expr(_enable_stagger):
            stagger_open(infra.stagger_i32)

        # Unrolled-by-2 loop for compile-time slot selection with split LDS globals.
        # Software-pipelined KV loop: overlap prev tile's softmax_finish (VALU)
        # with current tile's QK GEMM (MFMA).  Unrolled by 2 for compile-time
        # slot selection (split LDS globals need static slot indices).

        o_accs = o_accs_init
        _o = 2 * npair

        def _kv_iter_body(kv_i32, m_i, l_i, o_accs, prev_corr, s_prev,
                         v_lo_prev, v_hi_prev, read_slot, dma_slot,
                         valid=None):
            """One software-pipelined iteration.

            Processes prev tile (softmax_finish + PV GEMM) while computing
            current tile's QK GEMM.  v_lo/hi_prev carry the previous tile's V
            regs; read_slot/dma_slot are compile-time constants.
            """
            has_next = (kv_i32 + fx.Int32(1)) < _kv_hi

            # C0 (mem): DMA next tile, read current K/V.
            rocdl.s_setprio(1)
            rocdl.s_waitcnt(0)
            if has_next:
                load_kv(kv_i32 + fx.Int32(1), dma_slot)
            read_k_work(read_slot)
            v_lo_cur, v_hi_cur = read_v_slot[read_slot]()
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(0)

            dualwave_cluster_sync(0)

            # C1: softmax_finish(prev) + QK GEMM(current) — VALU || MFMA.
            rocdl.s_setprio(0)
            out_sm = softmax_finish(s_prev, m_i, l_i, o_accs, prev_corr)
            frag_P, m_i, l_i, o_accs, _ = (
                out_sm[0], out_sm[1], out_sm[2], out_sm[3], out_sm[4],
            )
            frag_S, = gemm1_qk(frag_Q, frag_K[0])
            pv_gemm_register(frag_P, v_lo_prev, v_hi_prev, o_accs)

            # Prepare current scores for next iteration.
            if const_expr(mod_has_score or mod_has_mask):
                apply_mods(frag_S, kv_i32)
            if valid is not None:
                _neg_inf = fx.Float32(-1e9)
                for i in range_constexpr(n_c):
                    frag_S[i] = valid.select(frag_S[i], _neg_inf)
            corr_scalar, s_scaled = softmax_start(frag_S, m_i)
            dualwave_cluster_sync(1)

            return m_i, l_i, o_accs, corr_scalar, s_scaled, v_lo_cur, v_hi_cur

        # ── Prologue: QK GEMM tile 0 ──
        read_k_work(0)
        v_lo_last, v_hi_last = read_v_slot[0]()
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(0)
        dualwave_cluster_sync(0)
        frag_S, = gemm1_qk(frag_Q, frag_K[0])
        if const_expr(mod_has_score or mod_has_mask):
            apply_mods(frag_S, _kv_lo)
        prev_corr, s_prev = softmax_start(frag_S, m_i)
        _has_tile1 = (_kv_lo + fx.Int32(1)) < _kv_hi
        if _has_tile1:
            load_kv(_kv_lo + fx.Int32(1), 1)
        dualwave_cluster_sync(1)

        # ── Steady-state: unrolled by 2, padded to pairs ──
        _kv_body_lo = _kv_lo + fx.Int32(1)
        _kv_body_range = _kv_hi - _kv_body_lo
        _kv_pairs = (_kv_body_range + fx.Int32(1)) // fx.Int32(2)

        _c = _o + _n_d_chunks
        _fs = _c + 1
        _vlo = _fs + n_c
        _vhi = _vlo + _n_d_chunks
        init_args = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [prev_corr]
            + [s_prev[i] for i in range_constexpr(n_c)]
            + [v_lo_last[dc] for dc in range_constexpr(_n_d_chunks)]
            + [v_hi_last[dc] for dc in range_constexpr(_n_d_chunks)]
        )
        loop_results = init_args

        for kv_pair, loop_args in range(
            fx.Int32(0),
            _kv_pairs,
            fx.Int32(1),
            init=init_args,
        ):
            m_i = [loop_args[r] for r in range_constexpr(npair)]
            l_i = [loop_args[npair + r] for r in range_constexpr(npair)]
            o_accs = [loop_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
            prev_corr = loop_args[_c]
            s_prev = [loop_args[_fs + i] for i in range_constexpr(n_c)]
            v_lo_last = [loop_args[_vlo + dc] for dc in range_constexpr(_n_d_chunks)]
            v_hi_last = [loop_args[_vhi + dc] for dc in range_constexpr(_n_d_chunks)]

            kv_even = _kv_body_lo + fx.Int32(arith.index_cast(T.i32, kv_pair)) * fx.Int32(2)
            ##########################################################
            # Even: read slot 1, DMA to 0.
            m_i, l_i, o_accs, prev_corr, s_prev, v_lo_last, v_hi_last = _kv_iter_body(
                kv_even, m_i, l_i, o_accs, prev_corr, s_prev,
                v_lo_last, v_hi_last, 1, 0)
            ##########################################################
            # Odd: read slot 0, DMA to 1. Mask invalid when past _kv_hi.
            kv_odd = kv_even + fx.Int32(1)
            m_i, l_i, o_accs, prev_corr, s_prev, v_lo_last, v_hi_last = _kv_iter_body(
                kv_odd, m_i, l_i, o_accs, prev_corr, s_prev,
                v_lo_last, v_hi_last, 0, 1,
                valid=kv_odd < _kv_hi)

            loop_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
                + [prev_corr]
                + [s_prev[i] for i in range_constexpr(n_c)]
                + [v_lo_last[dc] for dc in range_constexpr(_n_d_chunks)]
                + [v_hi_last[dc] for dc in range_constexpr(_n_d_chunks)]
            )

        m_i = [loop_results[r] for r in range_constexpr(npair)]
        l_i = [loop_results[npair + r] for r in range_constexpr(npair)]
        o_accs = [loop_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        prev_corr = loop_results[_c]
        s_prev = [loop_results[_fs + i] for i in range_constexpr(n_c)]
        v_lo_last = [loop_results[_vlo + dc] for dc in range_constexpr(_n_d_chunks)]
        v_hi_last = [loop_results[_vhi + dc] for dc in range_constexpr(_n_d_chunks)]

        # ── Epilogue: process last tile's deferred scores ──
        out_sm = softmax_finish(s_prev, m_i, l_i, o_accs, prev_corr)
        frag_P, m_i, l_i, o_accs, _ = (
            out_sm[0], out_sm[1], out_sm[2], out_sm[3], out_sm[4],
        )
        pv_gemm_register(frag_P, v_lo_last, v_hi_last, o_accs)

        if const_expr(_enable_stagger):
            stagger_close(infra.stagger_i32)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

    # After QK swap with npair=1: l_i[0] already has the correct per-query-row sum
    # (permlane32 in softmax combines the two score halves). No shuffle_xor needed.

    # O normalization: divide each v16f32 by l_i[0].
    # Guard against fully-masked rows (l_i == 0) producing NaN.
    if const_expr(flex_mod.needs_safe_norm):
        _safe_l = l_i[0].maximumf(fx.Float32(1e-12))
        inv_l = fx.Float32(1.0) / _safe_l
    else:
        inv_l = fx.Float32(1.0) / l_i[0]
    inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(_n_d_chunks):
        o_accs[dc] = (Vec(o_accs[dc]) * inv_l_vec).ir_value()

    # O store: each thread writes 16 D-values at 1 query-row per D-chunk.
    # C fragment layout (M=D, N=query): lane L has query_row = L%32.
    # Elements [4k..4k+3] map to 4 contiguous D columns at offset 8k,
    # so each group of 4 can be stored as one 64-bit buffer store.
    _qrow = fx.Int32(local_tid % 32)
    _group_d_base = fx.Int32((local_tid // 32) * 4)
    _o_row_stride = hq * head_dim

    _o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem_dtype)
    o_store_reg = fx.make_rmem_tensor(fx.make_layout(4, 1), elem_dtype)
    o_div = fx.logical_divide(
        fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.recast_iter(elem_dtype, fx.get_iter(o)),
                                   fx.make_layout(0x7FFFFFFF, 1))),
            max_size=True,
        ),
        fx.make_layout(1, 1),
    )

    o_base = o_off + _qrow * _o_row_stride
    for dc in range_constexpr(_n_d_chunks):
        o_vec = Vec(o_accs[dc])
        for k in range_constexpr(4):
            col = dc * 32 + _group_d_base + fx.Int32(k * 8)
            elems = [o_vec[k * 4 + e] for e in range_constexpr(4)]
            vbf = Vec.from_elements(elems, fx.Float32).to(elem_dtype)
            off = o_base + col
            fx.memref_store_vec(vbf, o_store_reg)
            fx.copy(_o_store_atom, o_store_reg, fx.slice(o_div, (None, fx.Int32(off))))


@flyc.jit
def launch_flex_attn_gfx950(
    o: fx.Tensor,
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    scale: fx.Float32,
    param: FlexAttnParam,
    stream: fx.Stream = fx.Stream(None),
):
    b = fx.Int32(fx.get_scalar(q.shape[0]))
    seqlen_q = fx.Int32(fx.get_scalar(q.shape[1]))
    hq = fx.Int32(fx.get_scalar(q.shape[2]))
    seqlen_kv = fx.Int32(fx.get_scalar(k.shape[1]))

    elem_dtype = _elem_dtype(param.dtype_id)

    # Both GEMMs use the same MFMA atom + wave layout as the validated probe:
    # make_tiled_mma(atom, (m_waves, n_waves, 1)) with no K-permutation tile.
    wave_layout = fx.make_layout(
        (param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)
    )
    mma_atom_qk = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    tiled_mma_qk = fx.make_tiled_mma(mma_atom_qk, wave_layout)
    mma_atom_pv = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    tiled_mma_pv = fx.make_tiled_mma(mma_atom_pv, wave_layout)

    # Each workgroup covers num_groups query subtiles of block_m rows.
    rows_per_wg = param.block_m * param.num_groups
    num_q_tiles = (seqlen_q + rows_per_wg - 1) // rows_per_wg

    flex_attn_fwd_gfx950_kernel._known_block_size = [param.block_threads, 1, 1]
    flex_attn_fwd_gfx950_kernel._func.__name__ = make_flex_attn_kernel_name(param)
    _total_waves = int(param.block_threads) // GFX950_WAVE_SIZE
    _waves_per_eu = max(1, _total_waves // 4)
    flex_attn_fwd_gfx950_kernel(
        o, q, k, v, seqlen_q, seqlen_kv, b, scale, tiled_mma_qk, tiled_mma_pv, param,
        value_attrs={
            "rocdl.waves_per_eu": _waves_per_eu,
            "rocdl.flat_work_group_size": f"{param.block_threads},{param.block_threads}",
        },
    ).launch(
        grid=(num_q_tiles, hq, b),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


# fast_fp_math breaks pipe_depth=2 when seqlen_kv == block_n (single KV tile); omit it.
_flex_attn_compile_hints = {
    "waves_per_eu": 2,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}
launch_flex_attn_gfx950.compile_hints = dict(_flex_attn_compile_hints)


def flydsl_flex_attention_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = 32,
    block_n: int = 32,
    num_groups: int = 2,
    pipe_depth: int = 1,
    pipe_stages: int = 1,
    accurate_softmax: bool = True,
    mask_type: int = MASK_NONE,
    score_type: int = SCORE_NONE,
    mask_window: int = 0,
    score_alibi_slope: float = 0.0,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Flash-attention forward on the layout API (gfx950) with flex score/mask mods.

    q/k/v: ``[B, S, H, D]`` (BSHD), bf16/f16. Returns ``[B, Sq, Hq, D]``.

    **Pipeline depth 2 (two tiles in flight, Option A):**

    - Set ``pipe_depth=2`` and ``num_groups >= 2`` (Strategy A stagger; default
      param uses ``m_waves=2``).
    - One KV tile per outer loop iteration. Overlap is from **LDS double-buffering**,
      **one-ahead K/V DMA prefetch**, **loop-carried softmax** (``SoftmaxFinish``
      / ``SoftmaxStart``), and optional **inter-tile readahead** (ReadK/ReadV for
      the next tile hoisted before C3 on the prior tile), not from advancing two tile
      indices per trip.
    - For meaningful steady-state overlap, use ``Skv / block_n >= 2``. A single KV
      tile (``Skv == block_n``) is supported but is mostly prologue/epilogue drain.
    - ``emit_prologue`` + ``emit_main_loop`` + ``emit_epilogue``. Cluster debug dumps
      list static wiring; ``LoadV`` is prefetch-only during normal C2 execution (V from
      prior prefetch or prologue for tile 0).

    ``pipe_depth >= 2`` requires ``num_groups >= 2`` (staggered Strategy A pipeline).
    """
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"flex_attention_layout targets gfx950; got {arch!r}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must share dtype")
    if q.dim() != 4:
        raise ValueError(f"q must be 4D [B,S,H,D], got {q.dim()}D")

    dtype_id = FLEX_DTYPE_FP16 if q.dtype is torch.float16 else FLEX_DTYPE_BF16
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}")

    B, Sq, Hq, D = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    if num_kv_heads is not None and num_kv_heads != Hkv:
        raise ValueError(f"num_kv_heads {num_kv_heads} != k head count {Hkv}")
    rows_per_wg = block_m * num_groups
    if Sq % rows_per_wg != 0:
        raise ValueError(
            f"seqlen_q ({Sq}) must be a multiple of block_m*num_groups ({rows_per_wg})"
        )
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    if stream is None:
        stream = torch.cuda.current_stream()
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    if pipe_depth >= 2 and num_groups < 2:
        raise ValueError(
            "pipe_depth>=2 requires num_groups>=2 (Strategy A staggered pipeline)"
        )

    param = make_flex_attn_param(
        seqlen_kv=Skv,
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=D,
        num_heads_q=Hq,
        num_heads_kv=Hkv,
        num_groups=num_groups,
        pipe_depth=pipe_depth,
        pipe_stages=pipe_stages,
        accurate_softmax=accurate_softmax,
        mask_type=mask_type,
        score_type=score_type,
        mask_window=mask_window,
        score_alibi_slope=score_alibi_slope,
    )
    # V stays in BSHD layout [B, Skv, Hkv, D]; the kernel tiles V into compact
    # [block_n, 32] D-chunk sub-tiles in LDS and uses LDSReadTrans16_64b to transpose.
    launch_flex_attn_gfx950(
        out.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(),
        fx.Float32(scale), param, stream,
    )
    return out
