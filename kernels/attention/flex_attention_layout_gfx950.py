# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Independent flash/flex-attention forward on the FlyDSL layout API (gfx950).

This is an attention kernel written on the CuTe-style layout API
(``fx.make_tiled_mma`` / ``make_fragment_{A,B,C}`` / ``fx.copy`` /
swizzled LDS views).

One workgroup computes ``num_groups`` independent ``[BLOCK_M, D]`` query tiles:
load Q resident, loop over KV ``[BLOCK_N, D]`` tiles doing GEMM1 (S = Q@K^T),
online softmax, the C->B bridge (scores packed as MFMA B operand), then
GEMM2 (O += P@V with V=A, P=B); epilogue normalizes O by the row sum and
stores it.  Supports optional flex score/mask mods (causal, sliding window, prefix LM,
alibi) and an overlapping-softmax pipeline for the KV loop.

Target arch: gfx950 (CDNA4). Uses the cdna4 LDS transpose-read atom and the
gfx950 LDS swizzles; it is NOT expected to run on gfx942.
"""

from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.attention.flash_attn_utils import (
    _stagger_extra_barrier_if_one,
    _stagger_extra_barrier_if_zero,
)


def pipeline_stagger_enabled(*, depth: int, num_groups: int, m_waves: int) -> bool:
    return depth >= 2 and num_groups >= 2 and m_waves >= 2


class _InfraContext:
    stagger_i32: object = None

try:
    from flydsl.expr.rocdl.universal import make_buffer_ptr as _make_buffer_ptr
except ImportError:
    from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
    from flydsl.expr import buffer_ops

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
# Ring slot 1 base skew vs slot 0: rotate LDS banks during ping-pong DMA/read overlap.
# 16 bytes = 4 banks; keeps BufferCopyLDS128b 16-byte aligned. Experimental only.
_LDS_RING_BANK_SKEW_BYTES = 0
_LDS_RING_BANK_SKEW_ELEMS = _LDS_RING_BANK_SKEW_BYTES // 2  # bf16/f16 element padding
# Upper D-half (ki >= head_dim/2/mma_k) skew within a K tile so ki=4..7 use different banks than ki=0..3.
_K_HALF_BANK_SKEW_BYTES = 0
_K_HALF_BANK_SKEW_ELEMS = _K_HALF_BANK_SKEW_BYTES // 2
FLEX_DTYPE_BF16 = 2
FLEX_DTYPE_FP16 = 3

_LOG2E = 1.4426950408889634

MASK_NONE = 0
MASK_CAUSAL = 1
MASK_SLIDING_WINDOW = 2
MASK_PREFIX_LM = 3

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


class PrefixLMMask(FlexMod):
    has_mask = True
    needs_safe_norm = True

    def __init__(self, prefix_len):
        self.prefix_len = prefix_len

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        raw_hi = (q_max_wg + fx.Int32(block_n)) // fx.Int32(block_n)
        kv_hi = fx.Int32(arith.minsi(raw_hi.ir_value(), fx.Int32(n_kv_tiles).ir_value()))
        return fx.Int32(0), kv_hi

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
        return kv_tile_end > q_idx

    def apply_mask(self, score, q_idx, kv_idx):
        visible = (kv_idx <= q_idx) | (kv_idx < fx.Int32(self.prefix_len))
        return visible.select(score, fx.Float32(-1e9))


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


def _build_mod(mask_type, score_type, mask_window=0, score_alibi_slope=0.0, mask_prefix_len=0):
    _mask = {
        MASK_NONE: FlexMod(),
        MASK_CAUSAL: CausalMask(),
        MASK_SLIDING_WINDOW: SlidingWindowMask(mask_window),
        MASK_PREFIX_LM: PrefixLMMask(mask_prefix_len),
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
    # num_groups independent query subtiles per workgroup, all sharing the same KV
    # loop. Each group runs the validated 32-row body on rows
    # [group*block_m : (group+1)*block_m); K/V are loaded once and reused across all
    # groups (strategy A). Total query rows per workgroup = num_groups*block_m.
    # Default 8: fills all 8 SIMDs/CU (8 groups × 1 wave × 64 threads = 512) and
    # enables wave-group stagger for overlapping DMA with compute.
    num_groups: fx.Constexpr[int]
    # mma shape
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    # derived
    group_threads: fx.Constexpr[int]  # threads per group = m_waves*n_waves*wave_size
    block_threads: fx.Constexpr[int]  # = num_groups * group_threads
    gqa_group: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    n_kv_tiles: fx.Constexpr[int]  # seqlen_kv // block_n
    pipe_depth: fx.Constexpr[int]  # 1 = monolithic, 2 = decomposed pipeline
    pipe_stages: fx.Constexpr[int]  # deprecated: stagger follows num_groups/pipe_depth/m_waves
    # True = exact per-row softmax; False = approximate column softmax (mma_m=32 only)
    accurate_softmax: fx.Constexpr[bool]
    # flex mods: integer type IDs (MASK_NONE/CAUSAL/SLIDING_WINDOW/PREFIX_LM, SCORE_NONE/ALIBI)
    mask_type: fx.Constexpr[int]
    score_type: fx.Constexpr[int]
    mask_window: fx.Constexpr[int]  # sliding window size (only used when mask_type==MASK_SLIDING_WINDOW)
    mask_prefix_len: fx.Constexpr[int]  # prefix length (only used when mask_type==MASK_PREFIX_LM)
    score_alibi_slope: fx.Constexpr[float]  # alibi slope (only used when score_type==SCORE_ALIBI)
    num_kv_splits: fx.Constexpr[int]  # split-K: partition KV range across this many WGs (1=disabled)
    paged: fx.Constexpr[bool]  # True = paged KV cache, False = contiguous


_PAGED_BT_LDS_SIZE = 2048


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
    num_groups: int = 8,
    mma_m: int = 32,
    mma_n: int = 32,
    mma_k: int = 16,
    pipe_depth: int = 1,
    pipe_stages: int = 1,
    accurate_softmax: bool = True,
    mask_type: int = MASK_NONE,
    score_type: int = SCORE_NONE,
    mask_window: int = 0,
    mask_prefix_len: int = 0,
    score_alibi_slope: float = 0.0,
    num_kv_splits: int = 1,
    paged: bool = False,
) -> FlexAttnParam:
    if dtype_id not in (FLEX_DTYPE_BF16, FLEX_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m <= 0 or block_n <= 0 or head_dim <= 0:
        raise ValueError("block_m, block_n, head_dim must be positive")
    _valid_mma = ((16, 32), (16, 16), (32, 16), (32, 8))
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
        depth=pipe_depth,
        num_groups=num_groups,
        m_waves=m_waves,
    ):
        raise ValueError(
            "pipe_depth>=2 requires pipeline stagger: num_groups>=2 and m_waves>=2 "
            f"(got num_groups={num_groups}, m_waves={m_waves})"
        )

    in_dbytes = 2

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
        mask_prefix_len=mask_prefix_len,
        score_alibi_slope=score_alibi_slope,
        num_kv_splits=num_kv_splits,
        paged=paged,
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
    o: fx.Tensor,  # [B, Sq, Hq, D]
    q: fx.Tensor,  # [B, Sq, Hq, D]
    k: fx.Tensor,  # [B, Skv, Hkv, D]
    v: fx.Tensor,  # [B, Skv, Hkv, D]
    seqlen_q: fx.Int32,
    seqlen_kv: fx.Int32,
    num_batches: fx.Int32,
    scale: fx.Float32,
    tiled_mma_qk: fx.TiledMma,
    param: FlexAttnParam,
    ws_o: fx.Tensor = fx.Tensor,
    ws_ml: fx.Tensor = fx.Tensor,
    block_table: fx.Tensor = fx.Tensor,  # [B * max_pages_per_seq] i32, flat
    block_table_stride: fx.Int32 = fx.Int32(0),
    context_lens: fx.Tensor = fx.Tensor,  # [B] i32
):
    block_m = param.block_m
    block_n = param.block_n
    head_dim = param.head_dim
    elem_dtype = _elem_dtype(param.dtype_id)
    _paged = bool(param.paged)

    tid = fx.thread_idx.x
    # Strategy A: num_groups independent 2-wave query subtiles per workgroup, all
    # driving the SAME KV loop so K/V (staged in LDS) is reused across groups. Each
    # group runs the validated 128-thread body via local_tid; group g owns query rows
    # [(q_tile*num_groups + g)*block_m : +block_m).
    num_groups = param.num_groups
    group_threads = param.group_threads  # 128 (m_waves*n_waves*wave_size)
    group = tid // group_threads
    local_tid = tid % group_threads
    # grid.x = q_tile; grid.y = head; grid.z = batch (or batch * num_kv_splits if split-K).
    _SPLITK = int(param.num_kv_splits) > 1
    _num_kv_splits = int(param.num_kv_splits)
    _is_causal = int(param.mask_type) in (MASK_CAUSAL, MASK_PREFIX_LM)
    if const_expr(_is_causal):
        _num_q_tiles = (seqlen_q + fx.Int32(num_groups * block_m - 1)) // fx.Int32(num_groups * block_m)
        q_tile = fx.Index(
            arith.index_cast(T.index, _num_q_tiles - fx.Int32(1) - fx.Int32(arith.index_cast(T.i32, fx.block_idx.x)))
        )
    else:
        q_tile = fx.block_idx.x
    h_idx = fx.block_idx.y
    if const_expr(_SPLITK):
        b_idx = fx.block_idx.z // fx.Index(_num_kv_splits)
        split_idx = fx.Int32(arith.index_cast(T.i32, fx.block_idx.z % fx.Index(_num_kv_splits)))
    else:
        b_idx = fx.block_idx.z
    kv_head = h_idx // param.gqa_group

    q_start = (q_tile * num_groups + group) * block_m

    if const_expr(_paged):
        _ctx_len_it = fx.recast_iter(fx.Int32, fx.get_iter(context_lens))
        _ctx_len = fx.Int32(fx.ptr_load(_ctx_len_it + fx.Int32(arith.index_cast(T.i32, b_idx))))
        n_kv_tiles = (_ctx_len + fx.Int32(block_n - 1)) // fx.Int32(block_n)
    else:
        n_kv_tiles = param.n_kv_tiles

    # ── LDS: K/V staging (shared across all groups) + per-group P bridge ──────
    kv_tile_elems = block_n * head_dim
    _v_subtile_elems = block_n * 32
    _v_step_elems = _v_subtile_elems // 4  # 256 elements per step (8 rows × 32 cols)
    _lds_ring_slots = max(2, int(param.pipe_depth))

    _k_lds_pad_elems = _K_HALF_BANK_SKEW_ELEMS + _LDS_RING_BANK_SKEW_ELEMS

    if const_expr(_paged):

        @fx.struct
        class SharedStorage:
            k_lds_0: fx.Array[elem_dtype, kv_tile_elems + _K_HALF_BANK_SKEW_ELEMS, 16]
            k_lds_1: fx.Array[elem_dtype, kv_tile_elems + _k_lds_pad_elems, 16]
            v_lds_0: fx.Array[elem_dtype, kv_tile_elems, 16]
            v_lds_1: fx.Array[elem_dtype, kv_tile_elems + _LDS_RING_BANK_SKEW_ELEMS, 16]
            p: fx.Array[elem_dtype, num_groups * block_m * block_n, 16]
            bt: fx.Array[fx.Int32, _PAGED_BT_LDS_SIZE, 16]

    else:

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
    if const_expr(_is_32x32):
        n_q = _size_scalar(frag_Q.shape)
        for qi in range_constexpr(n_q):
            frag_Q[qi] = _to_elem(_to_elem(frag_Q[qi], fx.Float32) * scale, elem_dtype)

    # Persistent O accumulator: 4 × v16f32 (one per D-chunk).
    # With V=A, P=B PV GEMM: each v16f32 has 16 D-values at 1 query-row per lane.
    Vec = fx.Vector
    _n_d_chunks = head_dim // 32
    o_accs_init = [Vec.filled(16, 0.0, fx.Float32).ir_value() for _ in range_constexpr(_n_d_chunks)]

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

    # ── KV-loop helpers ────────────────────────────────────────────────

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
    # retile(frag_K[slot]) is the MFMA A fragment register target for gemm1_qk_unrolled.
    #
    # Upper-D half (ki >= 4): sK_upper has +skew base when _K_HALF_BANK_SKEW_BYTES > 0 (currently disabled).
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
    # Paged KV cache: [num_blocks, block_n, Hkv, D] — page stride and head offset.
    _page_byte_stride = block_n * hkv * head_dim * param.in_data_bytes
    _kv_head_byte_offset = fx.Int32(arith.index_cast(T.i32, kv_head)) * fx.Int32(head_dim * param.in_data_bytes)
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
    sK_upper = [fx.make_view(sK_upper_ptr[i], _k_base_layout) for i in range_constexpr(_lds_ring_slots)]

    def _k_swizzled_col(tile_row, tile_col_elem):
        """Apply K swizzle to get the global column index for a given LDS position."""
        elem_off = fx.get_scalar(fx.crd2idx(fx.make_int_tuple((tile_row, tile_col_elem)), _k_base_layout))
        return elem_off % head_dim

    # ── Stage: DMA K+V global → LDS ─────────────────────────────────────
    def _stage_kv_to_lds_contiguous(kv_idx, buf, do_k, do_v, ops=_dma_ops_per_thread, op_offset=0):
        wave_off = rocdl.readfirstlane(
            fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes)
        )
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
                fx.copy(
                    dma_atom, fx.slice(k_div, (None, fx.Int32(gmem_byte))), fx.make_view(lds_k, fx.make_layout(1, 1))
                )
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
                lds_byte = v_step * _v_step_bytes + row_in_step * _v_subtile_row_bytes + tile_col_byte
                lds_v = fx.add_offset(sV_i8[buf], lds_byte)
                d_global_byte = dc * 32 * param.in_data_bytes + tile_col_byte
                gmem_byte = fx.Int32(v_global_base) + score_row * fx.Int32(_v_row_stride_bytes) + d_global_byte
                fx.copy(
                    dma_atom, fx.slice(v_div, (None, fx.Int32(gmem_byte))), fx.make_view(lds_v, fx.make_layout(1, 1))
                )

    def _stage_kv_to_lds_paged(page_id, buf, ops=_dma_ops_per_thread, op_offset=0):
        wave_off = rocdl.readfirstlane(
            fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes)
        )
        _step_bytes = block_threads * _dma_bytes
        k_global_base = page_id * fx.Int32(_page_byte_stride) + _kv_head_byte_offset
        lds_k = fx.add_offset(sK_i8[buf], wave_off + op_offset * _step_bytes)
        for i in range_constexpr(ops):
            if const_expr(i > 0):
                lds_k = fx.add_offset(lds_k, _step_bytes)
            flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
            tile_row = flat_byte // _k_row_bytes
            tile_col_elem = (flat_byte % _k_row_bytes) // param.in_data_bytes
            swiz_col = _k_swizzled_col(tile_row, tile_col_elem)
            gmem_byte = k_global_base + tile_row * _k_row_stride_bytes + swiz_col * param.in_data_bytes
            fx.copy(dma_atom, fx.slice(k_div, (None, fx.Int32(gmem_byte))), fx.make_view(lds_k, fx.make_layout(1, 1)))
        v_global_base = page_id * fx.Int32(_page_byte_stride) + _kv_head_byte_offset
        _v_step_bytes = _v_step_elems * param.in_data_bytes
        for i in range_constexpr(ops):
            flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
            tile_row = flat_byte // _v_subtile_row_bytes
            tile_col_byte = flat_byte % _v_subtile_row_bytes
            dc = tile_row // block_n
            score_row = tile_row % block_n
            v_step = dc * 4 + score_row // 8
            row_in_step = score_row % 8
            lds_byte = v_step * _v_step_bytes + row_in_step * _v_subtile_row_bytes + tile_col_byte
            lds_v = fx.add_offset(sV_i8[buf], lds_byte)
            d_global_byte = dc * 32 * param.in_data_bytes + tile_col_byte
            gmem_byte = fx.Int32(v_global_base) + score_row * fx.Int32(_v_row_stride_bytes) + d_global_byte
            fx.copy(dma_atom, fx.slice(v_div, (None, fx.Int32(gmem_byte))), fx.make_view(lds_v, fx.make_layout(1, 1)))

    # stride_phase: 0 = K D-lo, 1 = K D-hi, 2 = V tile (for split prefetch vs K reads).
    def _stage_kv_to_lds_strided(kv_idx, buf, stride_phase, ops=_dma_ops_per_thread, op_offset=0):
        from flydsl._mlir import ir
        from flydsl._mlir.dialects import scf
        from flydsl.expr import arith

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
                        arith.CmpIPredicate.ult,
                        tile_col_elem,
                        _half_d_i32,
                    )
                else:
                    in_phase = arith.cmpi(
                        arith.CmpIPredicate.uge,
                        tile_col_elem,
                        _half_d_i32,
                    )
                _if = scf.IfOp(in_phase, [], has_else=False)
                with ir.InsertionPoint(_if.then_block):
                    _wave_off = rocdl.readfirstlane(
                        fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes)
                    )
                    _lds_off = _wave_off + (op_offset + i) * block_threads * _dma_bytes
                    if const_expr(_K_HALF_BANK_SKEW_BYTES > 0 and stride_phase == 1):
                        _lds_off = _lds_off + _K_HALF_BANK_SKEW_BYTES
                    lds_k = fx.add_offset(sK_i8[buf], _lds_off)
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
                lds_byte = v_step * _v_step_bytes + row_in_step * _v_subtile_row_bytes + tile_col_byte
                lds_v = fx.add_offset(sV_i8[buf], lds_byte)
                d_global_byte = dc * 32 * param.in_data_bytes + tile_col_byte
                gmem_byte = fx.Int32(v_global_base) + score_row * fx.Int32(_v_row_stride_bytes) + d_global_byte
                fx.copy(
                    dma_atom,
                    fx.slice(v_div, (None, fx.Int32(gmem_byte))),
                    fx.make_view(lds_v, fx.make_layout(1, 1)),
                )

    def load_kv(tile_idx, slot, ops=_dma_ops_per_thread, op_offset=0):
        if const_expr(_paged):
            _pid = _load_page_id(tile_idx)
            _stage_kv_to_lds_paged(_pid, slot, ops=ops, op_offset=op_offset)
        elif _K_HALF_BANK_SKEW_BYTES > 0:
            _stage_kv_to_lds_strided(tile_idx, slot, 0, ops=ops, op_offset=op_offset)
            _stage_kv_to_lds_strided(tile_idx, slot, 1, ops=ops, op_offset=op_offset)
            _stage_kv_to_lds_strided(tile_idx, slot, 2, ops=ops, op_offset=op_offset)
        else:
            _stage_kv_to_lds_contiguous(tile_idx, slot, True, True, ops=ops, op_offset=op_offset)
        return []

    # ── V transpose read ────────────────────────────────────────────────────
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
    #   elem      = score_row * 32 + d_col
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

    _k_iters = int(param.head_dim) // int(param.mma_k)  # 128/16 → 8 ki panels
    _k_half = _k_iters // 2  # 4 ki per half (D-lo / D-hi)
    _k_serpentine = tuple(c + (1 - j) if (c // 2) % 2 else c + j for c in range(0, _k_iters, 2) for j in range(2))
    _k_serpentine_rev = tuple(reversed(_k_serpentine))
    _k_frag_retile_0 = tcA_k_lds[0].retile(frag_K[0])
    _k_frag_retile_1 = tcA_k_lds[1].retile(frag_K[1])

    def read_k_work_split(ki_count=_k_half, ki_offset=0, slot=0):
        """Read ki_count K-panels from LDS into frag_K[slot] for QK MFMA A.

        ki_offset / k_idx — which D-panel along head_dim (each panel is 32×16 scores×D):
          ki 0..3  D cols [0,63]   read from sK[slot]       (no half skew)
          ki 4..7  D cols [64,127] read from sK_upper[slot] (+16B LDS base)

        Per lane (local_tid), each ki issues one ds_read_b128:
          src = k_src[None, None, k_idx]  — this lane's 8 bf16 for that ki panel
          dst = _k_frag_retile_{slot}[None, None, k_idx]  — MFMA A fragment slot

        Wave split (m_waves=2): lanes 0..63 cover score rows 0..15 of the 32×16 panel;
          lanes 64..127 cover score rows 16..31 (same ki, complementary M rows).
        """
        _use_k_half_skew = _K_HALF_BANK_SKEW_BYTES > 0 and ki_offset >= _k_half
        if const_expr(_use_k_half_skew):
            k_src = tcA_k_lds[slot].partition_S(sK_upper[slot])  # D-hi: skewed base
        else:
            k_src = tcA_k_lds[slot].partition_S(sK[slot])  # D-lo: normal base
        for ki in range_constexpr(ki_count):
            k_idx = ki_offset + ki
            if const_expr(slot == 0):
                # Emits ds_read_b128; stores 8 bf16 into frag_K[0] for this ki.
                fx.copy(uca, k_src[None, None, k_idx], _k_frag_retile_0[None, None, k_idx])
            else:
                fx.copy(uca, k_src[None, None, k_idx], _k_frag_retile_1[None, None, k_idx])
        return []

    _qk_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))

    def _frag_reps(tensor, mode):
        return fx.size(fx.get_shape(tensor)[mode]).to_py_value()

    _qk_k_reps = _frag_reps(frag_K[0], 2)
    _qk_a_m_reps = _frag_reps(frag_K[0], 1)
    _qk_b_n_reps = _frag_reps(frag_Q, 1)

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
    flex_mod = _build_mod(
        int(param.mask_type),
        int(param.score_type),
        int(param.mask_window),
        float(param.score_alibi_slope),
        int(param.mask_prefix_len),
    )
    mod_has_score = flex_mod.has_score
    mod_has_mask = flex_mod.has_mask
    _mod_apply_score = flex_mod.apply_score
    _mod_apply_mask = flex_mod.apply_mask
    b_i32 = fx.Int32(arith.index_cast(T.i32, b_idx))
    h_i32 = fx.Int32(arith.index_cast(T.i32, h_idx))
    q_idx_mod = fx.Int32(arith.index_cast(T.i32, q_start)) + fx.Int32(local_tid % 32)
    _q_start_i32 = fx.Int32(arith.index_cast(T.i32, q_start))
    lane_group_off = fx.Int32((local_tid // 32) * 4)
    kv_offsets = [8 * (e // 4) + (e % 4) for e in range(n_c)]

    def apply_score_mods(frag_S_in, kv_tile_idx):
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        for e in range_constexpr(n_c):
            kv_idx = kv_base + fx.Int32(kv_offsets[e])
            frag_S_in[e] = _mod_apply_score(frag_S_in[e], b_i32, h_i32, q_idx_mod, kv_idx)

    def apply_mask_mods(frag_S_in, kv_tile_idx):
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        for e in range_constexpr(n_c):
            kv_idx = kv_base + fx.Int32(kv_offsets[e])
            frag_S_in[e] = _mod_apply_mask(frag_S_in[e], q_idx_mod, kv_idx)

    def apply_mods(frag_S_in, kv_tile_idx):
        if const_expr(mod_has_score):
            apply_score_mods(frag_S_in, kv_tile_idx)
        if const_expr(mod_has_mask):
            _kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
            if _kv_tile_end > _q_start_i32:
                apply_mask_mods(frag_S_in, kv_tile_idx)

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
    _pv_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))

    _is_bf16 = int(param.dtype_id) == FLEX_DTYPE_BF16

    def _pack_8_f32_to_v8elem(vals_8):
        """Pack 8 f32 values into v8 of elem_dtype (bf16 or f16)."""
        if const_expr(_is_bf16):
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
        in the non-contiguous C-score order by _make_read_v.
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
        _enable_stagger = True
    else:
        _enable_stagger = pipeline_stagger_enabled(
            depth=int(param.pipe_depth),
            num_groups=int(num_groups),
            m_waves=int(param.m_waves),
        )

    infra = _InfraContext()
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
                fx.Int32.ir_type,
                fx.Int32(local_tid // GFX950_WAVE_SIZE),
            )

    # ── Paged KV: load block table into LDS ──────────────────────────────
    if const_expr(_paged):
        from flydsl._mlir.dialects import llvm as _llvm

        _bt_lds_ptr = storage.bt.peek().ptr
        _bt_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        _bt_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(block_table)), fx.make_layout(0x7FFFFFFF, 1))),
            max_size=True,
        )
        _bt_div = fx.logical_divide(_bt_flat, fx.make_layout(1, 1))
        _bt_batch_off = fx.Int32(arith.index_cast(T.i32, b_idx)) * block_table_stride
        _bt_entries = n_kv_tiles
        for _bt_pass in range_constexpr((_PAGED_BT_LDS_SIZE + block_threads - 1) // block_threads):
            _bt_local = fx.Int32(_bt_pass) * fx.Int32(block_threads) + fx.Int32(tid)
            _bt_in_range = _bt_local < _bt_entries
            _bt_global = _bt_batch_off + _bt_local
            _bt_lds_dst = fx.add_offset(fx.recast_iter(fx.Int32, _bt_lds_ptr), fx.make_int_tuple(_bt_local))
            if _bt_in_range:
                fx.copy(
                    _bt_copy_atom,
                    fx.slice(_bt_div, (None, fx.Int32(_bt_global))),
                    fx.make_view(_bt_lds_dst, fx.make_layout(1, 1)),
                )
            else:
                fx.ptr_store(fx.Int32(0), _bt_lds_dst)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

        def _load_page_id(tile_idx):
            _byte_off = tile_idx * fx.Int32(4)
            _lds_i8 = fx.recast_iter(fx.Int8, _bt_lds_ptr)
            _raw = _llvm.LoadOp(T.i32, fx.to_llvm_ptr(fx.add_offset(_lds_i8, _byte_off)))
            rocdl.s_waitcnt(lgkmcnt=0)
            return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, _raw.result))

    else:

        def _load_page_id(tile_idx):
            return fx.Int32(0)

    # KV tile range: clamp to the mask's valid range to skip fully-masked tiles.
    _q_min_wg = fx.Int32(arith.index_cast(T.i32, q_tile)) * fx.Int32(num_groups * block_m)
    _q_max_wg = _q_min_wg + fx.Int32(num_groups * block_m - 1)
    _kv_lo, _kv_hi = flex_mod.kv_range(_q_min_wg, _q_max_wg, n_kv_tiles, block_n)
    if const_expr(_SPLITK):
        _total_tiles = _kv_hi - _kv_lo
        _chunk = (_total_tiles + fx.Int32(_num_kv_splits - 1)) // fx.Int32(_num_kv_splits)
        _kv_lo = _kv_lo + split_idx * _chunk
        _kv_hi_split = _kv_lo + _chunk
        _kv_hi = fx.Int32(arith.minsi(_kv_hi_split.ir_value(), _kv_hi.ir_value()))
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

    def _do_tile_overlapping_softmax_prologue(kv_i32, m_i, l_i, o_accs):
        """First pair: no deferred PV from previous. Has odd_valid/has_next guards."""
        odd_valid = (kv_i32 + fx.Int32(1)) < _kv_hi
        has_next = (kv_i32 + fx.Int32(2)) < _kv_hi
        # ── Cluster 0: mem tile 0 ──
        rocdl.s_waitcnt(vmcnt=0)
        rocdl.s_barrier()
        read_k_work(0)
        v_lo_regs_0, v_hi_regs_0 = read_v_slot[0]()
        if odd_valid:
            load_kv(kv_i32 + fx.Int32(1), 1)
        rocdl.s_waitcnt(lgkmcnt=0)
        dualwave_cluster_sync(0)

        # ── Cluster 1: QK GEMM tile 0, no deferred PV ──
        (frag_S,) = gemm1_qk_unrolled(frag_Q, frag_K[0])
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
        dualwave_cluster_sync(2)

        # ── Cluster 3: QK GEMM tile 1 + PV from tile 0 ──
        (frag_S,) = gemm1_qk_unrolled(frag_Q, frag_K[1])
        out_sm_0 = softmax_finish(s_scaled_0, m_i_at_tile0, l_i, o_accs, corr_scalar_0)
        pv_gemm_register(out_sm_0[0], v_lo_regs_0, v_hi_regs_0, out_sm_0[3])
        l_i, o_accs = out_sm_0[2], out_sm_0[3]
        s_raw = [frag_S[i] for i in range_constexpr(n_c)]
        if const_expr(mod_has_score or mod_has_mask):
            apply_mods(s_raw, kv_i32 + fx.Int32(1))
        _neg_inf = fx.Float32(-1e9)
        s_raw = [odd_valid.select(s_raw[i], _neg_inf) for i in range_constexpr(n_c)]
        corr_scalar_1, s_scaled_1, m_new = softmax_start(s_raw, m_i)
        m_i_at_tile1 = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
        m_i = m_i_at_tile1
        dualwave_cluster_sync(3)

        return (m_i, l_i, o_accs, s_scaled_1, corr_scalar_1, v_lo_regs_1, v_hi_regs_1, m_i_at_tile1)

    def _do_tile_overlapping_softmax_main(
        kv_i32, m_i, l_i, o_accs, s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev, m_i_prev
    ):
        """Steady-state: always has deferred PV, always has odd tile and next."""
        # ── Cluster 0: mem tile 0 ──
        rocdl.s_waitcnt(vmcnt=0)
        read_k_work(0)
        v_lo_regs_0, v_hi_regs_0 = read_v_slot[0]()
        load_kv(kv_i32 + fx.Int32(1), 1)
        rocdl.s_waitcnt(lgkmcnt=0)
        dualwave_cluster_sync(0)

        # ── Cluster 1: QK GEMM tile 0 + deferred PV from prev ──
        (frag_S,) = gemm1_qk_unrolled(frag_Q, frag_K[0])
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
        read_k_work(1)
        v_lo_regs_1, v_hi_regs_1 = read_v_slot[1]()
        load_kv(kv_i32 + fx.Int32(2), 0)
        rocdl.s_waitcnt(lgkmcnt=0)
        dualwave_cluster_sync(2)

        # ── Cluster 3: QK GEMM tile 1 + PV from tile 0 ──
        (frag_S,) = gemm1_qk_unrolled(frag_Q, frag_K[1])
        out_sm_0 = softmax_finish(s_scaled_0, m_i_at_tile0, l_i, o_accs, corr_scalar_0)
        pv_gemm_register(out_sm_0[0], v_lo_regs_0, v_hi_regs_0, out_sm_0[3])
        l_i, o_accs = out_sm_0[2], out_sm_0[3]
        s_raw = [frag_S[i] for i in range_constexpr(n_c)]
        if const_expr(mod_has_score or mod_has_mask):
            apply_mods(s_raw, kv_i32 + fx.Int32(1))
        corr_scalar_1, s_scaled_1, m_new = softmax_start(s_raw, m_i)
        m_i_at_tile1 = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
        m_i = m_i_at_tile1
        dualwave_cluster_sync(3)

        return (m_i, l_i, o_accs, s_scaled_1, corr_scalar_1, v_lo_regs_1, v_hi_regs_1, m_i_at_tile1)

    def _do_tile_overlapping_softmax_epilogue(
        kv_i32, m_i, l_i, o_accs, s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev, m_i_prev
    ):
        """Last pair: has deferred PV, odd tile may be invalid, no next DMA."""
        odd_valid = (kv_i32 + fx.Int32(1)) < _kv_hi
        has_next = (kv_i32 + fx.Int32(2)) < _kv_hi
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
        (frag_S,) = gemm1_qk_unrolled(frag_Q, frag_K[0])
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
        dualwave_cluster_sync(2)

        # ── Cluster 3: QK GEMM tile 1 + PV from tile 0 ──
        (frag_S,) = gemm1_qk_unrolled(frag_Q, frag_K[1])
        out_sm_0 = softmax_finish(s_scaled_0, m_i_at_tile0, l_i, o_accs, corr_scalar_0)
        pv_gemm_register(out_sm_0[0], v_lo_regs_0, v_hi_regs_0, out_sm_0[3])
        l_i, o_accs = out_sm_0[2], out_sm_0[3]
        s_raw = [frag_S[i] for i in range_constexpr(n_c)]
        if const_expr(mod_has_score or mod_has_mask):
            apply_mods(s_raw, kv_i32 + fx.Int32(1))
        _neg_inf = fx.Float32(-1e9)
        s_raw = [odd_valid.select(s_raw[i], _neg_inf) for i in range_constexpr(n_c)]
        corr_scalar_1, s_scaled_1, m_new = softmax_start(s_raw, m_i)
        m_i_at_tile1 = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
        m_i = m_i_at_tile1
        dualwave_cluster_sync(3)

        return (m_i, l_i, o_accs, s_scaled_1, corr_scalar_1, v_lo_regs_1, v_hi_regs_1, m_i_at_tile1)

    # Pad tile count to pairs: ceil(range / 2).  When range is odd the
    # last pair's odd tile lands past _kv_hi — has_next=False skips the
    # DMA, and the QK/softmax on stale LDS is harmless because the NEXT
    # iteration (which would consume those scores) never runs.
    # For a 1-tile range the loop executes once: even is the real tile,
    # odd is a no-op past _kv_hi whose scores never feed a subsequent PV.
    _kv_range = _kv_hi - _kv_lo
    _kv_pairs = (_kv_range + fx.Int32(1)) // fx.Int32(2)

    _sm_base = _o + _n_d_chunks
    _v_base = _sm_base + n_c + 1
    _mi_prev_base = _v_base + 2 * _n_d_chunks

    # ── Prologue: first pair, no deferred PV ──
    # Uses the dedicated prologue function which skips the deferred PV
    # entirely. It keeps odd_valid/has_next runtime guards for edge cases
    # (_kv_pairs==1: odd tile may be invalid, no next pair to DMA).
    m_i, l_i, o_accs, s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev, m_i_prev = (
        _do_tile_overlapping_softmax_prologue(_kv_lo, m_i, l_i, o_accs)
    )

    # ── Main loop: pairs 1 .. _kv_pairs-2, no runtime branches ──
    # All middle pairs have valid odd tile + guaranteed has_next.
    _main_loop_count = _kv_pairs - fx.Int32(2)  # 0 when _kv_pairs <= 2
    init_args = (
        [m_i[r] for r in range_constexpr(npair)]
        + [l_i[r] for r in range_constexpr(npair)]
        + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        + [s_scaled_prev[i] for i in range_constexpr(n_c)]
        + [corr_scalar_prev]
        + [v_lo_prev[dc] for dc in range_constexpr(_n_d_chunks)]
        + [v_hi_prev[dc] for dc in range_constexpr(_n_d_chunks)]
        + [m_i_prev[r] for r in range_constexpr(npair)]
    )
    loop_results = init_args

    for kv_mid, loop_args in range(
        fx.Int32(0),
        _main_loop_count,
        fx.Int32(1),
        init=init_args,
    ):
        m_i = [loop_args[r] for r in range_constexpr(npair)]
        l_i = [loop_args[npair + r] for r in range_constexpr(npair)]
        o_accs = [loop_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        s_scaled_prev = [loop_args[_sm_base + i] for i in range_constexpr(n_c)]
        corr_scalar_prev = loop_args[_sm_base + n_c]
        v_lo_prev = [loop_args[_v_base + dc] for dc in range_constexpr(_n_d_chunks)]
        v_hi_prev = [loop_args[_v_base + _n_d_chunks + dc] for dc in range_constexpr(_n_d_chunks)]
        m_i_prev = [loop_args[_mi_prev_base + r] for r in range_constexpr(npair)]

        kv_even = _kv_lo + (fx.Int32(arith.index_cast(T.i32, kv_mid)) + fx.Int32(1)) * fx.Int32(2)
        m_i, l_i, o_accs, s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev, m_i_prev = (
            _do_tile_overlapping_softmax_main(
                kv_even,
                m_i,
                l_i,
                o_accs,
                s_scaled_prev,
                corr_scalar_prev,
                v_lo_prev,
                v_hi_prev,
                m_i_prev,
            )
        )

        loop_results = yield (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [s_scaled_prev[i] for i in range_constexpr(n_c)]
            + [corr_scalar_prev]
            + [v_lo_prev[dc] for dc in range_constexpr(_n_d_chunks)]
            + [v_hi_prev[dc] for dc in range_constexpr(_n_d_chunks)]
            + [m_i_prev[r] for r in range_constexpr(npair)]
        )

    m_i = [loop_results[r] for r in range_constexpr(npair)]
    l_i = [loop_results[npair + r] for r in range_constexpr(npair)]
    o_accs = [loop_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
    s_scaled_prev = [loop_results[_sm_base + i] for i in range_constexpr(n_c)]
    corr_scalar_prev = loop_results[_sm_base + n_c]
    v_lo_prev = [loop_results[_v_base + dc] for dc in range_constexpr(_n_d_chunks)]
    v_hi_prev = [loop_results[_v_base + _n_d_chunks + dc] for dc in range_constexpr(_n_d_chunks)]
    m_i_prev = [loop_results[_mi_prev_base + r] for r in range_constexpr(npair)]

    # ── Epilogue: last pair (runs 0 or 1 times), has odd_valid/has_next guards ──
    # Uses scf.for with trip count min(_kv_pairs-1, 1) to avoid scf.if dominance issues.
    # When _kv_pairs==1, prologue handled the only pair; skip straight to final PV.
    _epilogue_count = (_kv_pairs > fx.Int32(1)).select(fx.Int32(1), fx.Int32(0))
    epi_init = (
        [m_i[r] for r in range_constexpr(npair)]
        + [l_i[r] for r in range_constexpr(npair)]
        + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        + [s_scaled_prev[i] for i in range_constexpr(n_c)]
        + [corr_scalar_prev]
        + [v_lo_prev[dc] for dc in range_constexpr(_n_d_chunks)]
        + [v_hi_prev[dc] for dc in range_constexpr(_n_d_chunks)]
        + [m_i_prev[r] for r in range_constexpr(npair)]
    )
    epi_results = epi_init

    for _epi_i, epi_args in range(
        fx.Int32(0),
        _epilogue_count,
        fx.Int32(1),
        init=epi_init,
    ):
        m_i = [epi_args[r] for r in range_constexpr(npair)]
        l_i = [epi_args[npair + r] for r in range_constexpr(npair)]
        o_accs = [epi_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        s_scaled_prev = [epi_args[_sm_base + i] for i in range_constexpr(n_c)]
        corr_scalar_prev = epi_args[_sm_base + n_c]
        v_lo_prev = [epi_args[_v_base + dc] for dc in range_constexpr(_n_d_chunks)]
        v_hi_prev = [epi_args[_v_base + _n_d_chunks + dc] for dc in range_constexpr(_n_d_chunks)]
        m_i_prev = [epi_args[_mi_prev_base + r] for r in range_constexpr(npair)]

        kv_last = _kv_lo + (_kv_pairs - fx.Int32(1)) * fx.Int32(2)
        m_i, l_i, o_accs, s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev, m_i_prev = (
            _do_tile_overlapping_softmax_epilogue(
                kv_last,
                m_i,
                l_i,
                o_accs,
                s_scaled_prev,
                corr_scalar_prev,
                v_lo_prev,
                v_hi_prev,
                m_i_prev,
            )
        )

        epi_results = yield (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [s_scaled_prev[i] for i in range_constexpr(n_c)]
            + [corr_scalar_prev]
            + [v_lo_prev[dc] for dc in range_constexpr(_n_d_chunks)]
            + [v_hi_prev[dc] for dc in range_constexpr(_n_d_chunks)]
            + [m_i_prev[r] for r in range_constexpr(npair)]
        )

    m_i = [epi_results[r] for r in range_constexpr(npair)]
    l_i = [epi_results[npair + r] for r in range_constexpr(npair)]
    o_accs = [epi_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
    s_scaled_prev = [epi_results[_sm_base + i] for i in range_constexpr(n_c)]
    corr_scalar_prev = epi_results[_sm_base + n_c]
    v_lo_prev = [epi_results[_v_base + dc] for dc in range_constexpr(_n_d_chunks)]
    v_hi_prev = [epi_results[_v_base + _n_d_chunks + dc] for dc in range_constexpr(_n_d_chunks)]
    m_i_prev = [epi_results[_mi_prev_base + r] for r in range_constexpr(npair)]

    # Final deferred PV from the last tile
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
    _out_elem_dtype = elem_dtype

    if const_expr(_SPLITK):
        # Split-K: write normalized partial O (f32) + (m, l) to workspace.
        _ws_o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        ws_o_reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        ws_o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.recast_iter(fx.Float32, fx.get_iter(ws_o)), fx.make_layout(0x7FFFFFFF, 1))),
                max_size=True,
            ),
            fx.make_layout(1, 1),
        )
        # ws_o layout: [num_splits, B, Hq, Sq, D] — compute flat offset.
        _ws_sq = seqlen_q
        _ws_o_row_stride = fx.Int32(head_dim)
        _ws_o_head_stride = _ws_sq * _ws_o_row_stride
        _ws_o_batch_stride = hq * _ws_o_head_stride
        _ws_o_split_stride = num_batches * _ws_o_batch_stride
        _ws_o_base = (
            split_idx * _ws_o_split_stride
            + b_idx * _ws_o_batch_stride
            + fx.Int32(arith.index_cast(T.i32, h_idx)) * _ws_o_head_stride
            + (q_start + _qrow) * _ws_o_row_stride
        )
        for dc in range_constexpr(_n_d_chunks):
            o_vec = Vec(o_accs[dc])
            for k in range_constexpr(4):
                col = dc * 32 + _group_d_base + fx.Int32(k * 8)
                elems = [o_vec[k * 4 + e] for e in range_constexpr(4)]
                v4f = Vec.from_elements(elems, fx.Float32)
                off = _ws_o_base + col
                fx.memref_store_vec(v4f, ws_o_reg)
                fx.copy(_ws_o_store_atom, ws_o_reg, fx.slice(ws_o_div, (None, fx.Int32(off))))

        # Store (m, l) per query row. Each lane in the wave has the same m_i/l_i
        # (permlane32 reduced), so only one lane per row writes.
        _ws_ml_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float32)
        ws_ml_reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Float32)
        ws_ml_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.recast_iter(fx.Float32, fx.get_iter(ws_ml)), fx.make_layout(0x7FFFFFFF, 1))),
                max_size=True,
            ),
            fx.make_layout(1, 1),
        )
        _ws_ml_row_stride = fx.Int32(2)
        _ws_ml_head_stride = _ws_sq * _ws_ml_row_stride
        _ws_ml_batch_stride = hq * _ws_ml_head_stride
        _ws_ml_split_stride = num_batches * _ws_ml_batch_stride
        _ws_ml_base = (
            split_idx * _ws_ml_split_stride
            + b_idx * _ws_ml_batch_stride
            + fx.Int32(arith.index_cast(T.i32, h_idx)) * _ws_ml_head_stride
            + (q_start + _qrow) * _ws_ml_row_stride
        )
        ml_vec = Vec.from_elements([m_i[0], l_i[0]], fx.Float32)
        fx.memref_store_vec(ml_vec, ws_ml_reg)
        fx.copy(_ws_ml_store_atom, ws_ml_reg, fx.slice(ws_ml_div, (None, fx.Int32(_ws_ml_base))))
    else:
        _o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), _out_elem_dtype)
        o_store_reg = fx.make_rmem_tensor(fx.make_layout(4, 1), _out_elem_dtype)
        o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.recast_iter(_out_elem_dtype, fx.get_iter(o)), fx.make_layout(0x7FFFFFFF, 1))),
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


_COMBINE_BLOCK = 256


@flyc.kernel(known_block_size=[_COMBINE_BLOCK, 1, 1])
def flex_splitk_combine_kernel(
    o: fx.Tensor,  # [B, Sq, Hq, D] output (bf16/f16)
    ws_o: fx.Tensor,  # [num_splits, B, Hq, Sq, D] f32 partial O
    ws_ml: fx.Tensor,  # [num_splits, B, Hq, Sq, 2] f32 (m, l) per split
    num_splits: fx.Constexpr[int],
    head_dim: fx.Constexpr[int],
    out_dtype_id: fx.Constexpr[int],
):
    """Combine split-K partial outputs into final O.

    Each thread handles 4 D-values at one query row. Block covers
    head_dim/4 lanes × (256 / (head_dim/4)) rows per workgroup.
    """
    from flydsl.expr.primitive import const_expr, range_constexpr

    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    _lanes_per_row = head_dim // 4
    _rows_per_block = _COMBINE_BLOCK // _lanes_per_row
    lane_in_row = tid % _lanes_per_row
    row_in_block = tid // _lanes_per_row

    # Global row = bid * rows_per_block + row_in_block
    # This row maps to (batch, head, seq_pos) in the flattened [B*Hq*Sq] space.
    global_row = fx.Int32(arith.index_cast(T.i32, bid)) * fx.Int32(_rows_per_block) + fx.Int32(row_in_block)

    # ws_o is [num_splits, B, Hq, Sq, D] — stride for split dim = B*Hq*Sq*D
    _total_rows = (
        fx.Int32(fx.get_scalar(ws_ml.shape[1]))
        * fx.Int32(fx.get_scalar(ws_ml.shape[2]))
        * fx.Int32(fx.get_scalar(ws_ml.shape[3]))
    )
    _split_o_stride = _total_rows * fx.Int32(head_dim)
    _split_ml_stride = _total_rows * fx.Int32(2)

    # D column offset: lane_in_row * 4
    d_col = fx.Int32(lane_in_row) * fx.Int32(4)

    # Load per-split (m, l) and find m_max
    _M_NEG_INF = fx.Float32(-1e30)
    m_max = _M_NEG_INF

    # ws_ml descriptor
    ws_ml_it = fx.recast_iter(fx.Float32, fx.get_iter(ws_ml))
    ws_o_it = fx.recast_iter(fx.Float32, fx.get_iter(ws_o))

    # First pass: find m_max across all splits
    for s in range_constexpr(num_splits):
        ml_off = fx.Int32(s) * _split_ml_stride + global_row * fx.Int32(2)
        m_s = fx.Float32(fx.ptr_load(ws_ml_it + fx.Int32(ml_off)))
        m_max = m_max.maximumf(m_s)

    # Second pass: weighted accumulate O and denominator
    Vec = fx.Vector
    acc = Vec.filled(4, 0.0, fx.Float32)
    den = fx.Float32(0.0)
    for s in range_constexpr(num_splits):
        ml_off = fx.Int32(s) * _split_ml_stride + global_row * fx.Int32(2)
        m_s = fx.Float32(fx.ptr_load(ws_ml_it + fx.Int32(ml_off)))
        l_s = fx.Float32(fx.ptr_load(ws_ml_it + fx.Int32(ml_off + fx.Int32(1))))
        w_s = _hw_exp2(m_s - m_max) * l_s

        o_off = fx.Int32(s) * _split_o_stride + global_row * fx.Int32(head_dim) + d_col
        o_vals = [fx.Float32(fx.ptr_load(ws_o_it + fx.Int32(o_off + fx.Int32(e)))) for e in range_constexpr(4)]
        o_vec = Vec.from_elements(o_vals, fx.Float32)
        w_vec = Vec.from_elements([w_s], fx.Float32).broadcast_to(4)
        acc = acc + o_vec * w_vec
        den = den + w_s

    # Normalize and store
    inv_den = fx.Float32(1.0) / den.maximumf(fx.Float32(1e-12))
    inv_vec = Vec.from_elements([inv_den], fx.Float32).broadcast_to(4)
    result = acc * inv_vec

    # Write to output O [B, Sq, Hq, D] — map global_row back to BSHD offset.
    # global_row indexes [B, Hq, Sq] (workspace layout), O is [B, Sq, Hq, D].
    _Hq = fx.Int32(fx.get_scalar(ws_ml.shape[2]))
    _Sq = fx.Int32(fx.get_scalar(ws_ml.shape[3]))
    _b = global_row // (_Hq * _Sq)
    _rem = global_row % (_Hq * _Sq)
    _h = _rem // _Sq
    _sq = _rem % _Sq
    _o_off = _b * _Sq * _Hq * fx.Int32(head_dim) + _sq * _Hq * fx.Int32(head_dim) + _h * fx.Int32(head_dim) + d_col

    _out_elem = fx.BFloat16 if const_expr(out_dtype_id == FLEX_DTYPE_BF16) else fx.Float16
    o_it = fx.recast_iter(_out_elem, fx.get_iter(o))
    for e in range_constexpr(4):
        val = result[e].to(_out_elem)
        fx.ptr_store(val, o_it + fx.Int32(_o_off + fx.Int32(e)))


@flyc.jit
def launch_flex_attn_gfx950(
    o: fx.Tensor,
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    scale: fx.Float32,
    param: FlexAttnParam,
    stream: fx.Stream = fx.Stream(None),
    ws_o: fx.Tensor = fx.Tensor,
    ws_ml: fx.Tensor = fx.Tensor,
    block_table: fx.Tensor = fx.Tensor,
    block_table_stride: fx.Int32 = fx.Int32(0),
    context_lens: fx.Tensor = fx.Tensor,
    max_seqlen_kv: fx.Int32 = fx.Int32(0),
):
    b = fx.Int32(fx.get_scalar(q.shape[0]))
    seqlen_q = fx.Int32(fx.get_scalar(q.shape[1]))
    hq = fx.Int32(fx.get_scalar(q.shape[2]))
    _paged = bool(param.paged)
    if const_expr(_paged):
        seqlen_kv = max_seqlen_kv
    else:
        seqlen_kv = fx.Int32(fx.get_scalar(k.shape[1]))

    elem_dtype = _elem_dtype(param.dtype_id)
    _SPLITK = int(param.num_kv_splits) > 1
    _num_kv_splits = int(param.num_kv_splits)

    wave_layout = fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0))
    mma_atom_qk = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))
    tiled_mma_qk = fx.make_tiled_mma(mma_atom_qk, wave_layout)

    rows_per_wg = param.block_m * param.num_groups
    num_q_tiles = (seqlen_q + rows_per_wg - 1) // rows_per_wg

    flex_attn_fwd_gfx950_kernel._known_block_size = [param.block_threads, 1, 1]
    flex_attn_fwd_gfx950_kernel._func.__name__ = make_flex_attn_kernel_name(param)
    _total_waves = int(param.block_threads) // GFX950_WAVE_SIZE
    _waves_per_eu = max(1, _total_waves // 4)

    if const_expr(_SPLITK):
        grid_z = b * fx.Int32(_num_kv_splits)
    else:
        grid_z = b

    flex_attn_fwd_gfx950_kernel(
        o,
        q,
        k,
        v,
        seqlen_q,
        seqlen_kv,
        b,
        scale,
        tiled_mma_qk,
        param,
        ws_o,
        ws_ml,
        block_table,
        block_table_stride,
        context_lens,
        value_attrs={
            "rocdl.waves_per_eu": _waves_per_eu,
            "rocdl.flat_work_group_size": f"{param.block_threads},{param.block_threads}",
        },
    ).launch(
        grid=(num_q_tiles, hq, grid_z),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )

    if const_expr(_SPLITK):
        _head_dim = int(param.head_dim)
        _lanes_per_row = _head_dim // 4
        _rows_per_block = _COMBINE_BLOCK // _lanes_per_row
        _total_rows = b * hq * seqlen_q
        _combine_blocks = (_total_rows + fx.Int32(_rows_per_block - 1)) // fx.Int32(_rows_per_block)
        flex_splitk_combine_kernel(
            o,
            ws_o,
            ws_ml,
            _num_kv_splits,
            _head_dim,
            int(param.dtype_id),
        ).launch(
            grid=(_combine_blocks, fx.Index(1), fx.Index(1)),
            block=(_COMBINE_BLOCK, 1, 1),
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
    num_groups: int = 8,
    pipe_depth: int = 1,
    pipe_stages: int = 1,
    accurate_softmax: bool = True,
    mask_type: int = MASK_NONE,
    score_type: int = SCORE_NONE,
    mask_window: int = 0,
    mask_prefix_len: int = 0,
    score_alibi_slope: float = 0.0,
    num_kv_splits: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Flash-attention forward on the layout API (gfx950) with flex score/mask mods.

    q/k/v: ``[B, S, H, D]`` (BSHD), bf16/f16. Returns ``[B, Sq, Hq, D]``.

    The KV loop uses an overlapping-softmax pipeline that processes tile pairs:
    a dedicated prologue (first pair, no deferred PV), a steady-state main loop,
    and an epilogue (last pair with validity guards). LDS is double-buffered
    with one-ahead K/V DMA prefetch.

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
        raise ValueError(f"seqlen_q ({Sq}) must be a multiple of block_m*num_groups ({rows_per_wg})")
    if scale is None:
        scale = 1.0 / (D**0.5)

    if stream is None:
        stream = torch.cuda.current_stream()
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    if pipe_depth >= 2 and num_groups < 2:
        raise ValueError("pipe_depth>=2 requires num_groups>=2 (Strategy A staggered pipeline)")

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
        mask_prefix_len=mask_prefix_len,
        score_alibi_slope=score_alibi_slope,
        num_kv_splits=num_kv_splits,
    )

    if num_kv_splits > 1:
        ws_o = torch.zeros(num_kv_splits, B, Hq, Sq, D, dtype=torch.float32, device=q.device)
        ws_ml = torch.full((num_kv_splits, B, Hq, Sq, 2), -1e30, dtype=torch.float32, device=q.device)
    else:
        ws_o = torch.empty(1, dtype=torch.float32, device=q.device)
        ws_ml = torch.empty(1, dtype=torch.float32, device=q.device)

    _dummy_bt = torch.empty(1, dtype=torch.int32, device=q.device)
    _dummy_ctx = torch.empty(1, dtype=torch.int32, device=q.device)
    launch_flex_attn_gfx950(
        out.contiguous(),
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        fx.Float32(scale),
        param,
        stream,
        ws_o=ws_o,
        ws_ml=ws_ml,
        block_table=_dummy_bt,
        block_table_stride=fx.Int32(0),
        context_lens=_dummy_ctx,
        max_seqlen_kv=fx.Int32(Skv),
    )
    return out


def flydsl_flex_attention_layout_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = 32,
    num_groups: int = 8,
    accurate_softmax: bool = True,
    mask_type: int = MASK_NONE,
    score_type: int = SCORE_NONE,
    mask_window: int = 0,
    mask_prefix_len: int = 0,
    score_alibi_slope: float = 0.0,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Paged-KV-cache flex attention forward (gfx950).

    q: ``[B, Sq, Hq, D]`` bf16/f16.
    k_cache/v_cache: ``[num_blocks, page_size, Hkv, D]`` bf16/f16 (linear layout).
    block_table: ``[B, max_pages_per_seq]`` i32 physical page IDs.
    context_lens: ``[B]`` i32 per-sequence KV context length.
    Returns ``[B, Sq, Hq, D]``.
    """
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"flex_attention_layout_paged targets gfx950; got {arch!r}")
    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("q/k_cache/v_cache must be CUDA tensors")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q/k_cache/v_cache must share dtype")
    if q.dim() != 4:
        raise ValueError(f"q must be 4D [B,Sq,H,D], got {q.dim()}D")
    if k_cache.dim() != 4:
        raise ValueError(f"k_cache must be 4D [num_blocks,page_size,Hkv,D], got {k_cache.dim()}D")

    dtype_id = FLEX_DTYPE_FP16 if q.dtype is torch.float16 else FLEX_DTYPE_BF16
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}")

    B, Sq, Hq, D = q.shape
    page_size = k_cache.shape[1]
    Hkv = k_cache.shape[2]
    block_n = page_size

    if num_kv_heads is not None and num_kv_heads != Hkv:
        raise ValueError(f"num_kv_heads {num_kv_heads} != k_cache head count {Hkv}")
    rows_per_wg = block_m * num_groups
    if Sq % rows_per_wg != 0:
        raise ValueError(f"seqlen_q ({Sq}) must be a multiple of block_m*num_groups ({rows_per_wg})")
    if scale is None:
        scale = 1.0 / (D**0.5)
    if stream is None:
        stream = torch.cuda.current_stream()
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    max_ctx = int(context_lens.max().item())
    max_seqlen_kv = ((max_ctx + block_n - 1) // block_n) * block_n

    bt_i32 = block_table.to(torch.int32).contiguous().reshape(-1)
    bt_stride = block_table.shape[1]
    ctx_i32 = context_lens.to(torch.int32).contiguous()

    param = make_flex_attn_param(
        seqlen_kv=max_seqlen_kv,
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=D,
        num_heads_q=Hq,
        num_heads_kv=Hkv,
        num_groups=num_groups,
        accurate_softmax=accurate_softmax,
        mask_type=mask_type,
        score_type=score_type,
        mask_window=mask_window,
        mask_prefix_len=mask_prefix_len,
        score_alibi_slope=score_alibi_slope,
        paged=True,
    )

    ws_o = torch.empty(1, dtype=torch.float32, device=q.device)
    ws_ml = torch.empty(1, dtype=torch.float32, device=q.device)

    launch_flex_attn_gfx950(
        out.contiguous(),
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        fx.Float32(scale),
        param,
        stream,
        ws_o=ws_o,
        ws_ml=ws_ml,
        block_table=bt_i32,
        block_table_stride=fx.Int32(bt_stride),
        context_lens=ctx_i32,
        max_seqlen_kv=fx.Int32(max_seqlen_kv),
    )
    return out


FLEX_DTYPE_FP8 = 4  # stub for test imports


def flydsl_flex_attention_layout_fp8(*args, **kwargs):
    raise NotImplementedError("FP8 flex attention is not supported in this build")
