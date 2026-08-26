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
from kernels.attention.flash_attn_utils import (
    _stagger_extra_barrier_if_one,
    _stagger_extra_barrier_if_zero,
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

# When True, _do_tile_passthrough prints per-thread K LDS b128 bank info (first KV tile only).
_DEBUG_LDS_BANKS = True
GFX950_DMA_BYTES = 16
# Ring slot 1 base skew vs slot 0: rotate LDS banks during ping-pong DMA/read overlap.
# 16 bytes = 4 banks; keeps BufferCopyLDS128b 16-byte aligned. Experimental only.
_LDS_RING_BANK_SKEW_BYTES = 0
_LDS_RING_BANK_SKEW_ELEMS = _LDS_RING_BANK_SKEW_BYTES // 2  # bf16/f16 element padding
# Upper D-half (ki >= head_dim/2/mma_k) skew within a K tile so ki=4..7 use different banks than ki=0..3.
_K_HALF_BANK_SKEW_BYTES = 0
_K_HALF_BANK_SKEW_ELEMS = _K_HALF_BANK_SKEW_BYTES // 2
FLEX_DTYPE_BF16 = 2
FLEX_DTYPE_FP16 = 3
FLEX_DTYPE_FP8 = 4

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


def _make_k_lds_layout(block_n, head_dim):
    # GEMM make_transposed_lds_layout XOR swizzle on D-contiguous storage.
    # Keep (block_n, head_dim) shape so QK MFMA fragment A matches loop_m=block_n.
    base_layout = fx.make_layout((block_n, head_dim), (head_dim, 1))
    if const_expr(head_dim == 128):
        k_swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
        return fx.make_composed_layout(k_swizzle, base_layout)
    return base_layout


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
    if dtype_id not in (FLEX_DTYPE_BF16, FLEX_DTYPE_FP16, FLEX_DTYPE_FP8):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m <= 0 or block_n <= 0 or head_dim <= 0:
        raise ValueError("block_m, block_n, head_dim must be positive")
    # Phase 0 proven config (see plan progress log): fx.gemm with MFMA 16x16x16 has
    # a lowering bug on this build, so both GEMMs use mma_k=32 -> block_n multiple of
    # 32. The C-fragment slot->row map is only locked for block_m=32 (2 M-waves x
    # mma_m=16), one N-wave; larger block_m needs a per-slot row map (TODO perf).
    _valid_mma = ((16, 32), (16, 16), (32, 16), (32, 8))
    if dtype_id == FLEX_DTYPE_FP8:
        _valid_mma = ((32, 64),)  # scaled MFMA 32x32x64
    if not (mma_m == mma_n and (mma_m, mma_k) in _valid_mma):
        raise ValueError(f"unsupported MMA shape {mma_m}x{mma_n}x{mma_k} for dtype_id={dtype_id}")
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

    in_dbytes = 1 if dtype_id == FLEX_DTYPE_FP8 else 2

    group_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    block_threads = num_groups * group_threads
    print("m_waves", m_waves, "n_waves", n_waves, "num_groups", num_groups)
    print("block_threads", block_threads)
    print("group_threads", group_threads)
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


def _flex_stagger_divisor(block_threads: int) -> int:
    """Waves per stagger half (flash-style ``wave_id // N`` for 32×32 MFMA)."""
    total_waves = block_threads // GFX950_WAVE_SIZE
    return max(1, total_waves // 2)


def flex_layout_stagger_enabled(param: FlexAttnParam) -> bool:
    """True when wave-group stagger is active for this param."""
    total_waves = int(param.block_threads) // GFX950_WAVE_SIZE
    if int(param.mma_m) == 32:
        # Flash-style stagger needs >=2 waves per half (e.g. 4+4 at 512 threads).
        return total_waves >= 4
    return pipeline_stagger_enabled(
        depth=int(param.pipe_depth),
        num_groups=int(param.num_groups),
        m_waves=int(param.m_waves),
    )


def make_flex_attn_kernel_name(param: FlexAttnParam) -> str:
    dtype_str = "fp16" if param.dtype_id == FLEX_DTYPE_FP16 else "bf16"
    name = f"flex_attn_{dtype_str}_m{param.block_m}n{param.block_n}d{param.head_dim}"
    name += f"_w{param.m_waves}x{param.n_waves}g{param.num_groups}"
    name += "_dense"
    name += "_rsm" if param.accurate_softmax else "_csm"
    name += f"_pd{param.pipe_depth}"
    if flex_layout_stagger_enabled(param):
        name += "_stg"
    return name


_FM = fx.arith.FastMathFlags.fast
_FM_CONTRACT = fx.arith.FastMathFlags.contract


def _elem_dtype(dtype_id):
    if dtype_id == FLEX_DTYPE_FP16:
        return fx.Float16
    if dtype_id == FLEX_DTYPE_FP8:
        return fx.Float8E4M3FN
    return fx.BFloat16


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

def _fsub(a, b):
    return fx.Float32(arith.subf(fx.Float32(a).ir_value(), fx.Float32(b).ir_value(), fastmath=_FM_CONTRACT))

def _fmul(a, b):
    return fx.Float32(arith.mulf(fx.Float32(a).ir_value(), fx.Float32(b).ir_value(), fastmath=_FM_CONTRACT))

def _fadd(a, b):
    return fx.Float32(arith.addf(fx.Float32(a).ir_value(), fx.Float32(b).ir_value(), fastmath=_FM_CONTRACT))

def _fdiv(a, b):
    return fx.Float32(arith.divf(fx.Float32(a).ir_value(), fx.Float32(b).ir_value(), fastmath=_FM_CONTRACT))

def _vmulf(a_ir, b_ir):
    """Vector mulf with contract fastmath — enables FMA fusion, avoids v_mul_f32_e64."""
    return arith.mulf(a_ir, b_ir, fastmath=_FM_CONTRACT)


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
    o: fx.Tensor,       # [B, Sq, Hq, D]  — output, always bf16 for fp8
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
    q_descale: fx.Float32 = fx.Float32(1.0),  # fp8 per-tensor Q descale
    k_descale: fx.Float32 = fx.Float32(1.0),  # fp8 per-tensor K descale
    v_descale: fx.Float32 = fx.Float32(1.0),  # fp8 per-tensor V descale
):
    block_m = param.block_m
    block_n = param.block_n
    head_dim = param.head_dim
    elem_dtype = _elem_dtype(param.dtype_id)
    _is_fp8 = int(param.dtype_id) == FLEX_DTYPE_FP8

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
    _v_subtile_elems = block_n * 32
    _v_step_elems = _v_subtile_elems // 4  # 256 elements per step (8 rows × 32 cols)
    _lds_ring_slots = max(2, int(param.pipe_depth))

    _k_lds_pad_elems = _K_HALF_BANK_SKEW_ELEMS + _LDS_RING_BANK_SKEW_ELEMS

    @fx.struct
    class SharedStorage:
        k_lds_0: fx.Array[elem_dtype, kv_tile_elems + _K_HALF_BANK_SKEW_ELEMS, 16]
        k_lds_1: fx.Array[elem_dtype, kv_tile_elems + _k_lds_pad_elems, 16]
        v_lds_0: fx.Array[elem_dtype, kv_tile_elems, 16]
        v_lds_1: fx.Array[elem_dtype, kv_tile_elems + _LDS_RING_BANK_SKEW_ELEMS, 16]
        p: fx.Array[elem_dtype, num_groups * block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    _k1_ptr = storage.k_lds_1.peek().ptr
    _v1_ptr = storage.v_lds_1.peek().ptr
    if _LDS_RING_BANK_SKEW_BYTES > 0:
        _skew = fx.make_int_tuple(_LDS_RING_BANK_SKEW_ELEMS)
        _k1_ptr = fx.add_offset(_k1_ptr, _skew)
        _v1_ptr = fx.add_offset(_v1_ptr, _skew)
    sK_ptr = [storage.k_lds_0.peek().ptr, _k1_ptr]
    sV_ptr = [storage.v_lds_0.peek().ptr, _v1_ptr]

    # K LDS: D-contiguous tile with GEMM-style XOR swizzle (Swizzle 2,4,3 when D=128).
    _k_base_layout = _make_k_lds_layout(block_n, head_dim)
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
    _is_32x32 = int(param.mma_m) == 32
    tcB_q = fx.make_tiled_copy_B(ca, tiled_mma_qk).get_slice(local_tid)
    frag_Q = thr_qk.make_fragment_B(gQ)
    fx.copy(ca, tcB_q.partition_S(gQ), tcB_q.retile(frag_Q))
    if const_expr(_is_32x32 and not _is_fp8):
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

    if const_expr(_is_fp8):
        scale_log2e = scale * fx.Float32(_LOG2E) * q_descale * k_descale
    elif const_expr(_is_32x32):
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

    # ── K LDS read (QK GEMM A operand) ─────────────────────────────────────
    # LDS logical tile: [block_n score, head_dim D] D-contiguous + Swizzle(3,3,3).
    # NO transpose — UniversalCopy128b → ds_read_b128 (8 bf16 / lane / ki).
    #
    # QK uses K=A, Q=B with MFMA 32×32×16, m_waves=2 (128 threads / query group):
    #   • M = 32 score rows; each wave owns 16 rows (local_tid // 64 → wave 0|1).
    #   • K depth = head_dim; one ki index = one mma_k=16 panel (D cols [ki*16, ki*16+15]).
    #   • _k_iters = head_dim/16 = 8; read_k_work_split loads _k_half=4 ki per call.
    #
    # tcA_k_lds[slot].partition_S(sK[slot]) gives this lane's LDS source coords
    #   (score_row, d_col) for each ki — layout from tiled_copy_A × swizzled sK view.
    # retile(frag_K[slot]) is the MFMA A fragment register target for gemm1_qk.
    #
    # Upper-D half (ki >= 4): DMA wrote with +16B skew → read via sK_upper (+16B base).
    tcA_k_lds = [fx.make_tiled_copy_A(uca, tiled_mma_qk).get_slice(local_tid) for _ in range_constexpr(_lds_ring_slots)]
    frag_K = [thr_qk.make_fragment_A(sK[i]) for i in range_constexpr(_lds_ring_slots)]

    # V is loaded as A operand for PV GEMM (V=A, P=B).
    # V LDS has 4 compact sub-tiles [block_n, 32]:(32, 1). LDSReadTrans16_64b
    # transposes each [block_n, 32] → [32, block_n] = A[M=D_chunk, K=score].
    if const_expr(_is_fp8):
        _v_tr_atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans8_64b(), elem_dtype)
    else:
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
    _k_half_d = int(head_dim) // 2  # 64 for D=128; ki 0..3 = D-lo, ki 4..7 = D-hi
    # sK_upper: same layout as sK, base + _K_HALF_BANK_SKEW_BYTES (16B) for ki>=4 reads.
    sK_upper_ptr = [
        fx.recast_iter(
            elem_dtype,
            fx.add_offset(sK_i8[i], fx.Int32(_K_HALF_BANK_SKEW_BYTES)),
        )
        for i in range_constexpr(_lds_ring_slots)
    ]
    sK_upper = [
        fx.make_view(sK_upper_ptr[i], _k_base_layout)
        for i in range_constexpr(_lds_ring_slots)
    ]

    def _k_swizzled_col(tile_row, tile_col_elem):
        """Apply K swizzle to get the global column index for a given LDS position."""
        elem_off = fx.get_scalar(fx.crd2idx(
            fx.make_int_tuple((tile_row, tile_col_elem)), _k_base_layout
        ))
        return elem_off % head_dim

    # ── Stage: DMA K+V global → LDS ─────────────────────────────────────
    def _stage_kv_to_lds(kv_idx, buf, do_k, do_v, ops=_dma_ops_per_thread, op_offset=0):
        wave_off = rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes))
        _step_bytes = block_threads * _dma_bytes
        if const_expr(do_k):
            k_global_base = k_off * param.in_data_bytes + kv_idx * block_n * _k_row_stride_bytes
            lds_k = fx.add_offset(sK_i8[buf], wave_off + op_offset * _step_bytes)
            for i in range_constexpr(ops):
                if const_expr(i > 0):
                    lds_k = fx.add_offset(lds_k, _step_bytes)
                flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
                tile_row = flat_byte // _k_row_bytes
                tile_col_elem = (flat_byte % _k_row_bytes) // param.in_data_bytes
                swiz_col = _k_swizzled_col(tile_row, tile_col_elem)
                gmem_byte = k_global_base + tile_row * _k_row_stride_bytes + swiz_col * param.in_data_bytes
                fx.copy(dma_atom, fx.slice(k_div, (None, fx.Int32(gmem_byte))),
                        fx.make_view(lds_k, fx.make_layout(1, 1)))
        if const_expr(do_v):
            v_global_base = v_off * param.in_data_bytes + kv_idx * fx.Int32(block_n * _v_row_stride_bytes)
            _v_step_bytes = _v_step_elems * param.in_data_bytes
            for i in range_constexpr(ops):
                flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
                tile_row = flat_byte // _v_subtile_row_bytes
                tile_col_byte = flat_byte % _v_subtile_row_bytes
                dc = tile_row // block_n
                score_row = tile_row % block_n
                v_step = dc * 4 + score_row // 8
                row_in_step = score_row % 8
                dc_shift_bytes = 0
                lds_byte = v_step * _v_step_bytes + row_in_step * _v_subtile_row_bytes + dc_shift_bytes + tile_col_byte
                lds_v = fx.add_offset(sV_i8[buf], lds_byte)
                d_global_byte = dc * 32 * param.in_data_bytes + tile_col_byte
                gmem_byte = fx.Int32(v_global_base) + score_row * fx.Int32(_v_row_stride_bytes) + d_global_byte
                fx.copy(dma_atom, fx.slice(v_div, (None, fx.Int32(gmem_byte))),
                        fx.make_view(lds_v, fx.make_layout(1, 1)))

    # stride_phase: 0 = K D-lo, 1 = K D-hi, 2 = V tile (for split prefetch vs K reads).
    def _stage_kv_to_lds_strided(kv_idx, buf, stride_phase, ops=_dma_ops_per_thread, op_offset=0):
        from flydsl._mlir import ir
        from flydsl._mlir.dialects import scf
        from flydsl.expr import arith
        from flydsl.expr.typing import T

        if const_expr(stride_phase == 0 or stride_phase == 1):
            k_global_base = k_off * param.in_data_bytes + kv_idx * block_n * _k_row_stride_bytes
            _half_d_i32 = fx.Int32(_k_half_d)
            for i in range_constexpr(ops):
                flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
                tile_row = flat_byte // _k_row_bytes
                tile_col_elem = (flat_byte % _k_row_bytes) // param.in_data_bytes
                swiz_col = _k_swizzled_col(tile_row, tile_col_elem)
                gmem_byte = k_global_base + tile_row * _k_row_stride_bytes + swiz_col * param.in_data_bytes
                if const_expr(stride_phase == 0):
                    in_phase = arith.cmpi(
                        arith.CmpIPredicate.ult, tile_col_elem, _half_d_i32,
                    )
                else:
                    in_phase = arith.cmpi(
                        arith.CmpIPredicate.uge, tile_col_elem, _half_d_i32,
                    )
                _if = scf.IfOp(in_phase, [], has_else=False)
                with ir.InsertionPoint(_if.then_block):
                    _wave_off = rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes))
                    lds_k = fx.add_offset(sK_i8[buf], _wave_off + (op_offset + i) * block_threads * _dma_bytes)
                    fx.copy(
                        dma_atom,
                        fx.slice(k_div, (None, fx.Int32(gmem_byte))),
                        fx.make_view(lds_k, fx.make_layout(1, 1)),
                    )
                    scf.YieldOp([])
        if const_expr(stride_phase == 2):
            v_global_base = v_off * param.in_data_bytes + kv_idx * fx.Int32(block_n * _v_row_stride_bytes)
            _v_step_bytes = _v_step_elems * param.in_data_bytes
            for i in range_constexpr(ops):
                flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
                tile_row = flat_byte // _v_subtile_row_bytes
                tile_col_byte = flat_byte % _v_subtile_row_bytes
                dc = tile_row // block_n
                score_row = tile_row % block_n
                v_step = dc * 4 + score_row // 8
                row_in_step = score_row % 8
                dc_shift_bytes = 0
                lds_byte = (
                    v_step * _v_step_bytes
                    + row_in_step * _v_subtile_row_bytes
                    + dc_shift_bytes
                    + tile_col_byte
                )
                lds_v = fx.add_offset(sV_i8[buf], lds_byte)
                d_global_byte = dc * 32 * param.in_data_bytes + tile_col_byte
                gmem_byte = fx.Int32(v_global_base) + score_row * fx.Int32(_v_row_stride_bytes) + d_global_byte
                fx.copy(
                    dma_atom,
                    fx.slice(v_div, (None, fx.Int32(gmem_byte))),
                    fx.make_view(lds_v, fx.make_layout(1, 1)),
                )

    from kernels.attention.pipeline import (
        InfraContext,
        pipeline_stagger_enabled,
    )

    _pd_for_sched = int(param.pipe_depth)

    class _LayoutSchedTraits:
        HEAD_DIM = int(param.head_dim)

    def load_kv(tile_idx, slot, ops=_dma_ops_per_thread, op_offset=0):
        _stage_kv_to_lds(tile_idx, slot, True, True, ops=ops, op_offset=op_offset)
        return []

    def load_k(tile_idx, slot, ops=_dma_ops_per_thread, op_offset=0):
        _stage_kv_to_lds(tile_idx, slot, True, False, ops=ops, op_offset=op_offset)
        return []

    def load_v(tile_idx, slot, ops=_dma_ops_per_thread, op_offset=0):
        _stage_kv_to_lds(tile_idx, slot, False, True, ops=ops, op_offset=op_offset)
        return []

    def load_k_strided(tile_idx, slot, stride_phase, ops=_dma_ops_per_thread, op_offset=0):
        _stage_kv_to_lds_strided(tile_idx, slot, stride_phase, ops=ops, op_offset=op_offset)
        return []

    def load_v_strided(tile_idx, slot, ops=_dma_ops_per_thread, op_offset=0):
        _stage_kv_to_lds_strided(tile_idx, slot, 2, ops=ops, op_offset=op_offset)
        return []

    # ── V transpose read (swa pattern) ─────────────────────────────────────
    # LDS stores V as padded [block_n score, 32 D] sub-tiles per dc (D-chunk).
    # read path: LDS [score,D] ──ds_read_tr16_b64──► 4 bf16/lane ──shuffle──► v8elem MFMA A.
    #
    # ds_read_tr16_b64 copy atom (LDSReadTrans16_64b):
    #   • 16 consecutive lanes (local_tid // 16, lanes local_tid % 16) cooperate per op.
    #   • Each lane reads 64b (4 bf16) from LDS; HW transposes a 16×16 bf16 tile.
    #   • 128 threads → 8 tr16 groups per (k_sub, dc) iteration.
    #
    # Per-lane LDS origin within a [32,32] sub-tile (score row, D col in elems):
    #   score_row = _v_row_off                          (0..15; //32 adds +4 per quarter)
    #   d_col     = _v_col_off                          (0,4,8,12 or +16 for upper half)
    #   elem      = score_row * 32 + d_col + _v_pair_off
    #
    # Example (local_tid → score_row, d_col) for first tr16 group (local_tid 0..15):
    #   tid  0→(0, 0)   1→(0, 4)   2→(0, 8)   3→(0,12)
    #   tid  4→(1, 0)   5→(1, 4)   6→(1, 8)   7→(1,12)
    #   tid  8→(2, 0)   9→(2, 4)  10→(2, 8)  11→(2,12)
    #   tid 12→(3, 0)  13→(3, 4)  14→(3, 8)  15→(3,12)
    # Second tr16 group (local_tid 16..31): score rows 0..3, d_col + 16:
    #   tid 16→(0,16)  17→(0,20) …  31→(3,28)
    # Quarter-wave row bias (local_tid // 32 → ×4 on score_row):
    #   tid 0..31   → score rows 0..3    (wave 0, top half of 32 scores)
    #   tid 32..63  → score rows 4..7
    #   tid 64..95  → score rows 8..11   (wave 1)
    #   tid 96..127 → score rows 12..15
    _v_tr_layout = fx.make_layout(4, 1)  # dst/src tile: 4 bf16 (64b) per lane per copy
    _v_smem_layout = fx.make_layout((block_n, 32), (32, 1))  # logical sub-tile [score, D]
    _v_row_off = ((local_tid % 16) // 4) + ((local_tid // 32) * 4)
    _v_col_off = ((local_tid % 4) * 4) + (16 * ((local_tid % 32) // 16))
    _v_lane_elem = fx.Int32(_v_row_off * 32 + (_v_col_off % 32))

    def _make_read_v(slot):
        """Build V LDS→register transpose read for PV GEMM A operand.

        Returns (v_lo_regs, v_hi_regs): lists of length _n_d_chunks (head_dim/32).
        Each entry is one v8elem (8 bf16) consumed by pv_gemm_register as MFMA A.

        Register flow per lane (local_tid):
          fx.copy(_v_tr_atom)  → dst[0:3]   4 bf16  halves[dc][k_sub]
          shuffle k_sub 0,1    → v_lo_out[dc]  8 bf16  MFMA A, K-lo (with p_lo)
          shuffle k_sub 2,3    → v_hi_out[dc]  8 bf16  MFMA A, K-hi (with p_hi)

        Loop axes (head_dim=128 → _n_d_chunks=4):
          dc    — D-chunk index (0..3), each covers D cols [dc*32, dc*32+31] in global V.
          k_sub — score/K sub-step (0..3); steps along score axis within padded LDS step.
          off   — elem offset from lane base into padded V LDS (v_step / dc_shift layout).
        """
        _slot_v_ptr = sV_ptr[slot]
        def _read():
            # Lane-specific base into [32,32] sub-tile for score_row/d_col above.
            base_ptr = fx.add_offset(_slot_v_ptr, fx.make_int_tuple(_v_lane_elem))
            # halves[dc][k_sub] = this lane's 4 bf16 after one ds_read_tr16_b64 + transpose.
            halves = [[None] * 4 for _ in range_constexpr(_n_d_chunks)]
            for k_sub in range_constexpr(4):
                for dc in range_constexpr(_n_d_chunks):
                    # Jump to dc-th D-chunk × k_sub-th score strip in padded LDS.
                    off = (dc * 4 + k_sub) * _v_step_elems 
                    src = fx.make_view(
                        fx.add_offset(base_ptr, fx.make_int_tuple(off)),
                        _v_tr_layout,
                    )
                    dst = fx.make_rmem_tensor(_v_tr_layout, elem_dtype)
                    # Emits ds_read_tr16_b64; stores transposed 4 bf16 into dst[0:3].
                    fx.copy(_v_tr_atom, src, dst)
                    halves[dc][k_sub] = Vec(dst.load())
            # Pack 4×(4 bf16) → 2×(8 bf16) MFMA A vectors per D-chunk, per lane.
            v_lo_out = [None] * _n_d_chunks
            v_hi_out = [None] * _n_d_chunks
            for dc in range_constexpr(_n_d_chunks):
                # k_sub 0,1 → v_lo_regs[dc] fed to _mfma_acc(..., p_lo, ...)  (PV K-lo)
                v_lo_out[dc] = halves[dc][0].shuffle(halves[dc][1], list(range(8))).ir_value()
                # k_sub 2,3 → v_hi_regs[dc] fed to _mfma_acc(..., p_hi, ...)  (PV K-hi)
                v_hi_out[dc] = halves[dc][2].shuffle(halves[dc][3], list(range(8))).ir_value()
            return v_lo_out, v_hi_out
        return _read

    read_v_slot = [_make_read_v(i) for i in range_constexpr(_lds_ring_slots)]

    def read_k_work(slot):
        """Per-wave serpentine K read: wave 0 forward, wave 1 reversed.

        Wave 0: (0,1,3,2,4,5,7,6), wave 1: (6,7,5,4,2,3,1,0).
        At any given step the two waves read different K-groups → disjoint banks.
        """
        from flydsl._mlir import ir
        from flydsl._mlir.dialects import scf
        _is_wave0 = (fx.Int32(local_tid // GFX950_WAVE_SIZE) & fx.Int32(1)) == fx.Int32(0)
        _if = scf.IfOp(_is_wave0.ir_value(), [], has_else=True)
        with ir.InsertionPoint(_if.then_block):
            for idx in range_constexpr(_k_iters):
                read_k_work_split(ki_count=1, ki_offset=_k_serpentine[idx], slot=slot)
            scf.YieldOp([])
        with ir.InsertionPoint(_if.else_block):
            for idx in range_constexpr(_k_iters):
                read_k_work_split(ki_count=1, ki_offset=_k_serpentine_rev[idx], slot=slot)
            scf.YieldOp([])
        return []

    _k_iters = int(param.head_dim) // int(param.mma_k)   # 128/16 → 8 ki panels
    _k_half  = _k_iters // 2                             # 4 ki per half (D-lo / D-hi)
    _k_serpentine = tuple(
        c + (1 - j) if (c // 2) % 2 else c + j
        for c in range(0, _k_iters, 2) for j in range(2)
    )
    _k_serpentine_rev = tuple(reversed(_k_serpentine))
    _k_frag_retile_0 = tcA_k_lds[0].retile(frag_K[0])
    _k_frag_retile_1 = tcA_k_lds[1].retile(frag_K[1])

    # Passthrough LDS bank debug phase ids (see _debug_lds_phase / _debug_print_k_b128).
    _PH_K_HALF0 = 0
    _PH_DMA0 = 1
    _PH_K_HALF1 = 2
    _PH_DMA1 = 3
    _PH_V_READ = 4
    _PH_QK_GEMM = 5
    _PH_PV_GEMM = 6

    def _debug_print_k_b128(phase, ki, slot, k_src):
        """Log ds_read_b128 source: LDS byte offset and LDS banks [b0..b3] for 128b read."""
        if const_expr(not _DEBUG_LDS_BANKS):
            return
        # crd2idx on the per-thread K partition layout (not ptrtoint — LDS iter bases
        # are not distinct per lane until lowering).
        elem_off = fx.get_scalar(
            fx.crd2idx(fx.make_int_tuple((0, 0, ki)), fx.get_layout(k_src))
        )
        lds_byte = elem_off * fx.Int32(param.in_data_bytes)
        bank0 = (lds_byte // fx.Int32(4)) % fx.Int32(32)
        bank3 = ((lds_byte + fx.Int32(12)) // fx.Int32(4)) % fx.Int32(32)
        wave = fx.Int32(local_tid // GFX950_WAVE_SIZE)
        lane = fx.Int32(local_tid % GFX950_WAVE_SIZE)
        fx.printf(
            "LDS_K tid={} w={} lane={} ph={} ki={} slot={} byte={} b0={} b3={}\n",
            fx.Int32(local_tid),
            wave,
            lane,
            fx.Int32(phase),
            fx.Int32(ki),
            fx.Int32(slot),
            lds_byte,
            bank0,
            bank3,
        )

    def _debug_lds_phase(phase, kv_i32, slot):
        """Mark pipeline step boundaries in passthrough (no bank math)."""
        if const_expr(not _DEBUG_LDS_BANKS):
            return
        wave = fx.Int32(local_tid // GFX950_WAVE_SIZE)
        fx.printf(
            "PHASE tid={} w={} ph={} kv={} slot={}\n",
            fx.Int32(local_tid),
            wave,
            fx.Int32(phase),
            kv_i32,
            fx.Int32(slot),
        )

    def read_k_work_split(ki_count=_k_half, ki_offset=0, slot=0, phase=_PH_K_HALF0, emit_debug=False):
        """Read ki_count K-panels from LDS into frag_K[slot] for QK MFMA A.

        ki_offset / k_idx — which D-panel along head_dim (each panel is 32×16 scores×D):
          ki 0..3  D cols [0,63]   read from sK[slot]       (no half skew)
          ki 4..7  D cols [64,127] read from sK_upper[slot] (+16B LDS base)

        Per lane (local_tid), each ki issues one ds_read_b128:
          src = k_src[None, None, k_idx]  — this lane's 8 bf16 for that ki panel
          dst = _k_frag_retile_{slot}[None, None, k_idx]  — MFMA A fragment slot

        Wave split (m_waves=2): lanes 0..63 cover score rows 0..15 of the 32×16 panel;
          lanes 64..127 cover score rows 16..31 (same ki, complementary M rows).

        emit_debug + _debug_print_k_b128: log LDS byte + banks [b0..b3] per (ki, lane).
        """
        _use_k_half_skew = _K_HALF_BANK_SKEW_BYTES > 0 and ki_offset >= _k_half
        if const_expr(_use_k_half_skew):
            k_src = tcA_k_lds[slot].partition_S(sK_upper[slot])  # D-hi: skewed base
        else:
            k_src = tcA_k_lds[slot].partition_S(sK[slot])       # D-lo: normal base
        for ki in range_constexpr(ki_count):
            k_idx = ki_offset + ki
            if const_expr(slot == 0):
                # Emits ds_read_b128; stores 8 bf16 into frag_K[0] for this ki.
                fx.copy(uca, k_src[None, None, k_idx], _k_frag_retile_0[None, None, k_idx])
            else:
                fx.copy(uca, k_src[None, None, k_idx], _k_frag_retile_1[None, None, k_idx])
        return []



    if const_expr(_is_fp8):
        _qk_mma_atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, elem_dtype))
    else:
        _qk_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))

    def _frag_reps(tensor, mode):
        return fx.size(fx.get_shape(tensor)[mode]).to_py_value()

    _qk_k_reps = _frag_reps(frag_K[0], 2)
    _qk_a_m_reps = _frag_reps(frag_K[0], 1)
    _qk_b_n_reps = _frag_reps(frag_Q, 1)

    def gemm1_qk(frag_Q_in, frag_K_in):
        frag_S_out = thr_qk.make_fragment_C(sP)
        frag_S_out.fill(0.0)
        fx.gemm(tiled_mma_qk, frag_S_out, frag_K_in, frag_Q_in, frag_S_out)
        return [frag_S_out]

    def gemm1_qk_mfma(frag_S_acc, frag_Q_in, frag_K_in, ki):
        """All M×N MFMAs for one K-group ki. Caller controls ki scheduling."""
        for m in range_constexpr(_qk_a_m_reps):
            for n in range_constexpr(_qk_b_n_reps):
                fx.mma_atom_call(
                    _qk_mma_atom,
                    frag_S_acc[None, m, n],
                    frag_K_in[None, m, ki],
                    frag_Q_in[None, n, ki],
                    frag_S_acc[None, m, n],
                )

    def gemm1_qk_unrolled(frag_Q_in, frag_K_in):
        """QK GEMM with explicit per-ki MFMA calls (register-only, no bank concerns)."""
        frag_S_out = thr_qk.make_fragment_C(sP)
        frag_S_out.fill(0.0)
        for ki in range_constexpr(_qk_k_reps):
            gemm1_qk_mfma(frag_S_out, frag_Q_in, frag_K_in, ki)
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
            for e in range_constexpr(n_c):
                kv_idx = kv_base + fx.Int32(kv_offsets[e])
                frag_S_in[e] = _mod_apply_mask(frag_S_in[e], q_idx_mod, kv_idx)

    # _n_d_chunks defined above as head_dim // 32 (= 4 for D=128).

    def _scale_o_vec(o_accs_in, scale_scalar):
        """Vectorized O rescale: broadcast scalar to vec16, multiply per D-chunk."""
        scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
        o_out = []
        for dc in range_constexpr(_n_d_chunks):
            o_vec = Vec(o_accs_in[dc])
            o_out.append((o_vec * scale_vec).ir_value())
        return o_out

    def softmax_start(frag_S_in, m_i_in):
        _sl2e_vec = Vec.from_elements([scale_log2e], fx.Float32).broadcast_to(16)
        s_elems = [frag_S_in[i] for i in range_constexpr(n_c)]
        s_scaled = Vec.from_elements(s_elems, fx.Float32) * _sl2e_vec
        s_out = [s_scaled[i] for i in range_constexpr(n_c)]
        tile_max = s_out[0]
        for i in range_constexpr(1, n_c):
            tile_max = tile_max.maximumf(s_out[i])
        tile_max = _permlane32_reduce(tile_max, "max")
        m_new = m_i_in[0].maximumf(tile_max)
        corr_scalar = _hw_exp2(m_i_in[0] - m_new)
        return corr_scalar, s_out, m_new

    def softmax_finish(s_scaled, m_i_in, l_i_in, o_accs_in, corr_scalar):
        m_new = m_i_in[0]
        p_elems = [_hw_exp2(s_scaled[i] - m_new) for i in range_constexpr(n_c)]
        p_vec = Vec.from_elements(p_elems, fx.Float32)
        local_sum = p_vec.reduce("add", init_val=fx.Float32(0.0), fastmath=_FM)
        local_sum = _permlane32_reduce(local_sum, "sum")
        corr = [corr_scalar]
        l_new = l_i_in[0] * corr_scalar + local_sum
        l_i_out = [l_new] + [l_i_in[r] for r in range_constexpr(1, npair)]
        o_accs_out = _scale_o_vec(o_accs_in, corr_scalar)
        return [p_elems, m_i_in, l_i_out, o_accs_out, corr]

    # ── Register-only PV GEMM (V=A, P=B) ──────────────────────────────────
    # After QK swap (K=A, Q=B), C's M-rows = score indices.
    # C→B is register-local: pack 16 f32 → 2 × v8bf16.
    # V is loaded as A from LDS per D-chunk.
    # PV GEMM uses bf16 MFMA for fp8 (HIPREC mode) or native dtype MFMA for bf16/f16.
    if const_expr(_is_fp8):
        _pv_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16))
    else:
        _pv_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))

    _is_bf16 = int(param.dtype_id) == FLEX_DTYPE_BF16

    def _pack_8_f32_to_v8elem(vals_8):
        """Pack 8 f32 values into v8 of elem_dtype (bf16 or f16).

        For FP8: uses bf16 packing — the PV GEMM currently uses the bf16
        MFMA path even for fp8 input (HIPREC mode). Full fp8 PV with
        cvt_pk_fp8_f32 + scaled MFMA is a follow-up.
        """
        if const_expr(_is_bf16 or _is_fp8):
            pairs = []
            for j in range_constexpr(4):
                pairs.append(rocdl.cvt_pk_bf16_f32(vals_8[j * 2], vals_8[j * 2 + 1]))
            return Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16).ir_value()
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
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)


    if const_expr(_is_32x32):
        _total_waves = int(param.block_threads) // GFX950_WAVE_SIZE
        _enable_stagger =  True#_total_waves >= 4
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
            _stagger_div = _flex_stagger_divisor(int(param.block_threads))
            _wave_id = fx.Int32(tid // GFX950_WAVE_SIZE)
            infra.stagger_i32 = rocdl.readfirstlane(
                fx.Int32.ir_type,
                _wave_id // fx.Int32(_stagger_div),
            )
        else:
            infra.stagger_i32 = rocdl.readfirstlane(
                fx.Int32.ir_type, fx.Int32(local_tid // GFX950_WAVE_SIZE),
            )

    infra.traits = _LayoutSchedTraits()

    single_tile = True
    print("n_kv_tiles: ", n_kv_tiles)
    if const_expr(single_tile):
        # KV tile range: clamp to the mask's valid range to skip fully-masked tiles.
        _q_min_wg = fx.Int32(arith.index_cast(T.i32, q_tile)) * fx.Int32(num_groups * block_m)
        _q_max_wg = _q_min_wg + fx.Int32(num_groups * block_m - 1)
        _kv_lo, _kv_hi = flex_mod.kv_range(_q_min_wg, _q_max_wg, n_kv_tiles, block_n)
        rocdl.s_barrier()
        rocdl.s_barrier()

        # Double-buffered KV loop via scf.for with loop-carried m/l/O state.
        load_kv(_kv_lo, 0)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

        if const_expr(_enable_stagger):
            rocdl.sched_barrier(0)
            _stagger_extra_barrier_if_one(infra.stagger_i32)

        # Unrolled-by-2 KV loop with split LDS globals for compile-time slot
        # selection.  Each iteration does: read K/V → QK GEMM → softmax →
        # DMA next tile → PV GEMM.  DMA and LDS reads target separate per-slot
        # globals so LLVM can prove non-aliasing.

        o_accs = o_accs_init
        _o = 2 * npair

        def _do_tile(kv_i32, m_i, l_i, o_accs, read_slot, dma_slot, has_next,
                    valid=None):
            """One non-pipelined iteration on a compile-time slot.

            valid: runtime i1 flag; when False the tile's scores are masked to
            -inf so softmax produces all-zero P and PV GEMM is a no-op.
            """
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()
            read_k_work(read_slot)
            rocdl.sched_barrier(0)
            if has_next:
                load_kv(kv_i32 + fx.Int32(1), dma_slot)
            v_lo_regs, v_hi_regs = read_v_slot[read_slot]()
            rocdl.s_waitcnt(lgkmcnt=0)
            dualwave_cluster_sync(0)

            rocdl.s_setprio(0)
            frag_S, = gemm1_qk_unrolled(frag_Q, frag_K[read_slot])
            s_raw = [frag_S[i] for i in range_constexpr(n_c)]
            if const_expr(mod_has_score or mod_has_mask):
                apply_mods(s_raw, kv_i32)
            if valid is not None:
                _neg_inf = fx.Float32(-1e9)
                s_raw = [valid.select(s_raw[i], _neg_inf) for i in range_constexpr(n_c)]
            corr_scalar, s_scaled, m_new = softmax_start(s_raw, m_i)
            m_i = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
            out_sm = softmax_finish(s_scaled, m_i, l_i, o_accs, corr_scalar)
            frag_P = out_sm[0]
            l_i, o_accs = out_sm[2], out_sm[3]
            pv_gemm_register(frag_P, v_lo_regs, v_hi_regs, o_accs)
            dualwave_cluster_sync(1)
            return m_i, l_i, o_accs

        def _do_tile_overlapping_softmax(kv_i32, m_i, l_i, o_accs, read_slot, dma_slot, has_next,
                    s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev, has_softmax_prev,
                    m_i_prev, odd_valid):
            """Two-tile iteration with deferred PV GEMM overlapping QK GEMM.

            PV GEMM from the previous tile runs alongside the current tile's
            QK GEMM (both are MFMA-bound, writing different accumulators).
            softmax_finish waits for the deferred PV to complete before
            rescaling o_accs.  The current tile's PV is deferred to the next call.

            m_i_prev: the m_i snapshot from when the deferred softmax_start ran.
            odd_valid: runtime flag — False when kv_i32+1 >= _kv_hi (odd tile OOB).
            """
            # ── Cluster 0: mem tile 0 ──
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()
            read_k_work(0)
            v_lo_regs_0, v_hi_regs_0 = read_v_slot[0]()
            if odd_valid:
                load_kv(kv_i32 + fx.Int32(1), 1)
            rocdl.s_waitcnt(lgkmcnt=0)
            dualwave_cluster_sync(0)

            # ── Cluster 1: QK GEMM tile 0 + deferred PV from prev ──
            frag_S, = gemm1_qk_unrolled(frag_Q, frag_K[0])
            if has_softmax_prev:
                out_sm_prev = softmax_finish(s_scaled_prev, m_i_prev, l_i, o_accs, corr_scalar_prev)
                pv_gemm_register(out_sm_prev[0], v_lo_prev, v_hi_prev, out_sm_prev[3])
                l_i, o_accs = out_sm_prev[2], out_sm_prev[3]
            s_raw = [frag_S[i] for i in range_constexpr(n_c)]
            if const_expr(mod_has_score or mod_has_mask):
                apply_mods(s_raw, kv_i32)
            corr_scalar_0, s_scaled_0, m_new = softmax_start(s_raw, m_i)
            m_i_at_tile0 = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
            m_i = m_i_at_tile0
            dualwave_cluster_sync(1)

            # ── Cluster 2: mem tile 1 ──
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()
            read_k_work(1)
            v_lo_regs_1, v_hi_regs_1 = read_v_slot[1]()
            if has_next:
                load_kv(kv_i32 + fx.Int32(2), 0)
            rocdl.s_waitcnt(lgkmcnt=0)
            dualwave_cluster_sync(0)

            # ── Cluster 3: QK GEMM tile 1 + PV from tile 0 ──
            frag_S, = gemm1_qk_unrolled(frag_Q, frag_K[1])
            out_sm_0 = softmax_finish(s_scaled_0, m_i_at_tile0, l_i, o_accs, corr_scalar_0)
            pv_gemm_register(out_sm_0[0], v_lo_regs_0, v_hi_regs_0, out_sm_0[3])
            l_i, o_accs = out_sm_0[2], out_sm_0[3]
            s_raw = [frag_S[i] for i in range_constexpr(n_c)]
            if const_expr(mod_has_score or mod_has_mask):
                apply_mods(s_raw, kv_i32 + fx.Int32(1))
            # Mask tile 1 scores when odd tile is past _kv_hi
            _neg_inf = fx.Float32(-1e9)
            s_raw = [odd_valid.select(s_raw[i], _neg_inf) for i in range_constexpr(n_c)]
            corr_scalar_1, s_scaled_1, m_new = softmax_start(s_raw, m_i)
            m_i_at_tile1 = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
            m_i = m_i_at_tile1
            dualwave_cluster_sync(1)

            return (
                m_i, l_i, o_accs,
                s_scaled_1, corr_scalar_1,
                v_lo_regs_1, v_hi_regs_1,
                fx.Boolean(True),
                m_i_at_tile1,
            )

        def _do_tile_passthrough(kv_i32, m_i, l_i, o_accs, read_slot, dma_slot, has_next):
            """Passthrough iteration: memory transfers + GEMMs only, no softmax.

            Measures theoretical peak by skipping score mods, softmax, and O
            rescaling.  Raw QK scores are fed directly as P into PV GEMM.
            m_i / l_i are passed through unchanged.
            """
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()
            read_k_work(read_slot)
            v_lo_regs, v_hi_regs = read_v_slot[read_slot]()
            if has_next:
                load_kv(kv_i32 + fx.Int32(1), dma_slot)
            rocdl.s_waitcnt(lgkmcnt=8)
            dualwave_cluster_sync(0)
            frag_S, = gemm1_qk(frag_Q, frag_K[read_slot])

            frag_P = [frag_S[i] for i in range_constexpr(n_c)]
            pv_gemm_register(frag_P, v_lo_regs, v_hi_regs, o_accs)
            dualwave_cluster_sync(1)
            return m_i, l_i, o_accs

        # Pad tile count to pairs: ceil(range / 2).  When range is odd the
        # last pair's odd call lands past _kv_hi — _do_tile's has_next=False
        # skips the DMA, and the QK/softmax on stale LDS is harmless because
        # the NEXT iteration (which would consume those scores) never runs.
        # For a 1-tile range the loop executes once: even is the real tile,
        # odd is a no-op past _kv_hi whose scores never feed a subsequent PV.
        _kv_range = _kv_hi - _kv_lo
        _kv_pairs = (_kv_range + fx.Int32(1)) // fx.Int32(2)

        s_scaled_prev = [fx.Float32(_M_NEG_FLOOR_SCALED) for _ in range_constexpr(n_c)]
        corr_scalar_prev = fx.Float32(1.0)
        has_softmax_prev = fx.Boolean(False)
        v_lo_prev, v_hi_prev = read_v_slot[0]()

        m_i_prev = [fx.Float32(_M_NEG_FLOOR_SCALED) for _ in range_constexpr(npair)]
        _sm_base = _o + _n_d_chunks
        init_args = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [s_scaled_prev[i] for i in range_constexpr(n_c)]
            + [corr_scalar_prev, has_softmax_prev]
            + [v_lo_prev[dc] for dc in range_constexpr(_n_d_chunks)]
            + [v_hi_prev[dc] for dc in range_constexpr(_n_d_chunks)]
            + [m_i_prev[r] for r in range_constexpr(npair)]
        )
        loop_results = init_args

        overlap_softmax = True
        for kv_pair, loop_args in range(
            fx.Int32(0),
            _kv_pairs,
            fx.Int32(1),
            init=init_args,
        ):
            m_i = [loop_args[r] for r in range_constexpr(npair)]
            l_i = [loop_args[npair + r] for r in range_constexpr(npair)]
            o_accs = [loop_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
            s_scaled_prev = [
                loop_args[_sm_base + i] for i in range_constexpr(n_c)
            ]
            corr_scalar_prev = loop_args[_sm_base + n_c]
            has_softmax_prev = loop_args[_sm_base + n_c + 1]
            _v_base = _sm_base + n_c + 2
            v_lo_prev = [
                loop_args[_v_base + dc] for dc in range_constexpr(_n_d_chunks)
            ]
            v_hi_prev = [
                loop_args[_v_base + _n_d_chunks + dc]
                for dc in range_constexpr(_n_d_chunks)
            ]
            _mi_prev_base = _v_base + 2 * _n_d_chunks
            m_i_prev = [loop_args[_mi_prev_base + r] for r in range_constexpr(npair)]

            kv_even = _kv_lo + fx.Int32(arith.index_cast(T.i32, kv_pair)) * fx.Int32(2)
            kv_odd = kv_even + fx.Int32(1)
            has_next_even = (kv_even + fx.Int32(1)) < _kv_hi
            has_next_odd = (kv_odd + fx.Int32(1)) < _kv_hi

            passthrough = False
            if const_expr(overlap_softmax):
                odd_valid = kv_odd < _kv_hi
                has_next_even = (kv_even + fx.Int32(2)) < _kv_hi
                m_i, l_i, o_accs, s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev, has_softmax_prev, m_i_prev = (
                    _do_tile_overlapping_softmax(
                        kv_even, m_i, l_i, o_accs, 0, 1, has_next_even,
                        s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev,
                        has_softmax_prev, m_i_prev, odd_valid,
                    )
                )
            elif const_expr(passthrough):
                m_i, l_i, o_accs = _do_tile_passthrough(kv_even, m_i, l_i, o_accs, 0, 1, has_next_even)
                m_i, l_i, o_accs = _do_tile_passthrough(kv_odd, m_i, l_i, o_accs, 1, 0, has_next_odd)
            else:
                odd_valid = kv_odd < _kv_hi
                ##########################################################
                # Even iteration: slot 0 → DMA next into slot 1.
                m_i, l_i, o_accs = _do_tile(kv_even, m_i, l_i, o_accs, 0, 1, has_next_even)
                ##########################################################
                # Odd iteration: slot 1 → DMA next into slot 0 (if exists).
                # When range is odd, last pair's odd tile is past _kv_hi — mask
                # its scores to -inf so softmax/PV are a no-op.
                m_i, l_i, o_accs = _do_tile(kv_odd, m_i, l_i, o_accs, 1, 0,
                                            has_next_odd, valid=odd_valid)

            loop_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
                + [s_scaled_prev[i] for i in range_constexpr(n_c)]
                + [corr_scalar_prev, has_softmax_prev]
                + [v_lo_prev[dc] for dc in range_constexpr(_n_d_chunks)]
                + [v_hi_prev[dc] for dc in range_constexpr(_n_d_chunks)]
                + [m_i_prev[r] for r in range_constexpr(npair)]
            )

        m_i = [loop_results[r] for r in range_constexpr(npair)]
        l_i = [loop_results[npair + r] for r in range_constexpr(npair)]
        o_accs = [loop_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        s_scaled_prev = [loop_results[_sm_base + i] for i in range_constexpr(n_c)]
        corr_scalar_prev = loop_results[_sm_base + n_c]
        has_softmax_prev = loop_results[_sm_base + n_c + 1]
        _v_base = _sm_base + n_c + 2
        v_lo_prev = [loop_results[_v_base + dc] for dc in range_constexpr(_n_d_chunks)]
        v_hi_prev = [
            loop_results[_v_base + _n_d_chunks + dc]
            for dc in range_constexpr(_n_d_chunks)
        ]
        _mi_prev_base = _v_base + 2 * _n_d_chunks
        m_i_prev = [loop_results[_mi_prev_base + r] for r in range_constexpr(npair)]

        if const_expr(overlap_softmax):
            if has_softmax_prev:
                out_sm_final = softmax_finish(s_scaled_prev, m_i_prev, l_i, o_accs, corr_scalar_prev)
                pv_gemm_register(out_sm_final[0], v_lo_prev, v_hi_prev, out_sm_final[3])
                l_i, o_accs = out_sm_final[2], out_sm_final[3]

        if const_expr(_enable_stagger):
            _stagger_extra_barrier_if_zero(infra.stagger_i32)
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
    if const_expr(_is_fp8):
        inv_l = inv_l * v_descale
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
    _out_elem_dtype = fx.BFloat16 if const_expr(_is_fp8) else elem_dtype

    _o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), _out_elem_dtype)
    o_store_reg = fx.make_rmem_tensor(fx.make_layout(4, 1), _out_elem_dtype)
    o_div = fx.logical_divide(
        fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.recast_iter(_out_elem_dtype, fx.get_iter(o)),
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
            vbf = Vec.from_elements(elems, fx.Float32).to(_out_elem_dtype)
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
    q_descale: fx.Float32 = fx.Float32(1.0),
    k_descale: fx.Float32 = fx.Float32(1.0),
    v_descale: fx.Float32 = fx.Float32(1.0),
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
    _is_fp8_launch = int(param.dtype_id) == FLEX_DTYPE_FP8
    if const_expr(_is_fp8_launch):
        mma_atom_qk = fx.make_mma_atom(
            rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
        )
        mma_atom_pv = fx.make_mma_atom(
            rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
        )
    else:
        mma_atom_qk = fx.make_mma_atom(
            fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
        )
        mma_atom_pv = fx.make_mma_atom(
            fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
        )
    tiled_mma_qk = fx.make_tiled_mma(mma_atom_qk, wave_layout)
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
        q_descale, k_descale, v_descale,
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


def flydsl_flex_attention_layout_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: float,
    k_descale: float,
    v_descale: float,
    *,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = 32,
    block_n: int = 32,
    num_groups: int = 2,
    accurate_softmax: bool = True,
    mask_type: int = MASK_NONE,
    score_type: int = SCORE_NONE,
    mask_window: int = 0,
    score_alibi_slope: float = 0.0,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """FP8 flex-attention forward on the layout API (gfx950).

    q/k/v: ``[B, S, H, D]`` (BSHD), float8_e4m3fn.
    q/k/v_descale: per-tensor f32 descale scalars.
    Returns ``[B, Sq, Hq, D]`` in bf16.
    """
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"flex_attention_layout targets gfx950; got {arch!r}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be CUDA tensors")
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(f"expected float8_e4m3fn, got {q.dtype}")
    if q.dim() != 4:
        raise ValueError(f"q must be 4D [B,S,H,D], got {q.dim()}D")

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
        out = torch.empty(B, Sq, Hq, D, dtype=torch.bfloat16, device=q.device)

    param = make_flex_attn_param(
        seqlen_kv=Skv,
        dtype_id=FLEX_DTYPE_FP8,
        block_m=block_m,
        block_n=block_n,
        head_dim=D,
        num_heads_q=Hq,
        num_heads_kv=Hkv,
        num_groups=num_groups,
        mma_m=32,
        mma_n=32,
        mma_k=64,
        accurate_softmax=accurate_softmax,
        mask_type=mask_type,
        score_type=score_type,
        mask_window=mask_window,
        score_alibi_slope=score_alibi_slope,
    )
    launch_flex_attn_gfx950(
        out.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(),
        fx.Float32(scale), param, stream,
        fx.Float32(q_descale), fx.Float32(k_descale), fx.Float32(v_descale),
    )
    return out
