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

import struct
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from flydsl.runtime.device import get_rocm_arch
from kernels.attention.flash_attn_utils import (
    _attn_mask_vec2_imm,
    _bitcast_f32,
    _bitcast_i32,
    _fused_o_128_dwords,
    _read_exec_i64,
    _stagger_extra_barrier_if_one,
    _stagger_extra_barrier_if_zero,
    _waitcnt_vm_n,
    scf_if_dispatch,
)

if not hasattr(fx, "max"):
    from flydsl._mlir.dialects import arith as _arith_mlir

    def _fx_max(a, b):
        if hasattr(a, "maximumf"):
            return a.maximumf(b)
        return type(a)(_arith_mlir.maxsi(a.ir_value(), b.ir_value()))

    def _fx_min(a, b):
        if hasattr(a, "minimumf"):
            return a.minimumf(b)
        return type(a)(_arith_mlir.minsi(a.ir_value(), b.ir_value()))

    fx.max = _fx_max
    fx.min = _fx_min


def _i32_min(a, b):
    from flydsl._mlir.dialects import arith as _am

    def _v(x):
        return x.ir_value() if hasattr(x, "ir_value") else x

    return fx.Int32(_am.minsi(_v(a), _v(b)))


def _i32_max(a, b):
    from flydsl._mlir.dialects import arith as _am

    def _v(x):
        return x.ir_value() if hasattr(x, "ir_value") else x

    return fx.Int32(_am.maxsi(_v(a), _v(b)))


def _f32_max(a, b):
    if hasattr(a, "maximumf"):
        return a.maximumf(b)
    return fx.Float32(a).maximumf(fx.Float32(b))


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
# Split KV schedule: LDS-K / QK / LDS-V / PV so K and V are not live together.
# Mem clusters pair full-tile LDS reads with complementary DMA, using flash
# dual-wave's wave-linear padded physical K/V layouts. Softmax VALU overlaps
# the compute clusters (start after QK, finish before PV). Set True to drop
# softmax for a GEMM-only experiment (numerics off).
_GEMM_ONLY_NO_SOFTMAX = False
_NO_SOFTMAX_MATH = False
# n64 flash-LDS software pipeline:
# C0 K-LDS + finish exp(P[n-1]); C1 QK[n] + sum(P[n-1]);
# C2 V-LDS[n-1] + max/rescale[n]; C3 PV[n-1] + first exp(P[n]).
_FLASH_DEFERRED_SOFTMAX = True
# n64 flash LDS: issue V reads + next-K DMA, then compute half of P and
# rescale O while memory is outstanding. C3 finishes P/l and runs PV.
_FLASH_SPLIT_SOFTMAX_ACROSS_V_MEM = False
# Alternate the LDS burst with the softmax arithmetic inside the n64 memory
# clusters. Left to itself the scheduler emits the whole K (or V) burst first
# and the arithmetic after it, so the wave eats the full ds_read issue
# backpressure (~13 cycles per ds_read_b128) before any VALU runs.
# Bit 0 = C0 (K reads vs exp2/O-rescale), bit 1 = C2 (V reads vs max chain).
_FLASH_SCHED_LDS_VALU = 3
# Keep the LDS burst on the near side of s_waitcnt. The scheduler may sink a
# load past the wait, which would leave it uncovered.
_FLASH_SCHED_PIN_LDS = True
# C1 has two independent dataflows: QK[n] and sum(P[n-1]). Without a grouping
# hint LLVM emits nearly all QK MFMAs before starting the deferred reduction.
_FLASH_SCHED_QK_SUM = True
# C3 runs PV[n-1] alongside the P[n] subtract/exp2/pack chain. Those do not feed
# the PV accumulator, so they belong in the MFMA shadow rather than in the
# EXP7/EXP8 runs LLVM otherwise leaves stranded between single MFMAs.
_FLASH_SCHED_PV_SOFTMAX = True
# Flash's wave-uniform lazy-rescale fast path. Keep the old max while every
# lane's tile max is within this log2 headroom, then skip O/l rebasing.
_FLASH_LAZY_O_RESCALE = True
_FLASH_LAZY_RESCALE_THRESHOLD = 8.0
# Match flash's priority window around the long PV + softmax cluster.
_FLASH_SETPRIO_PV = True
# Causal only: apply lazy O rescale at the start of C3 (before PV[n-1]).
# Dense C3 preamble (blob before PV) regresses ~7 us.
_FLASH_SCALE_O_IN_C3 = True
# Dense: apply corr_pending O-scale in C1 (QK MFMA shadow) instead of C0.
# QK does not read O, so the 4 x v16 muls can hide behind 16 QK MFMAs.
# Causal stays on C3. Epilogue has no QK and still scales in C0.
_FLASH_SCALE_O_IN_C1 = False
# Perf-only: skip the lazy rebase (no ballot, no m/l update, no O *= corr).
# Scores still reduce-max so C2 VALU stays; results are numerically wrong.
_FLASH_ELIDE_RESCALE = False
# Flash-style steady state: consume pending PV pack 0, establish the current
# tile max, lazily rebase O/l/pending-P, then consume pending PV packs 1..3.
_FLASH_C3_LAZY_RESCALE = True
_FLASH_8C_SCHED = True
# 8c only: C0/C4 read K that a previous cluster staged via buffer_load_lds.
# Drain that DMA (leaving this cluster's V in flight) before the LDS read.
# Without it the read can beat the write and the output is wrong.
# The steady-state trips do not strictly need it: each memory cluster closes on
# vmcnt(2 tiles), which retires that K two memory clusters before its reader
# runs. Only the first trip is unordered, because the prologue leaves its
# K[lo+1] in flight, so PRELOOP_VMWAIT alone is sufficient for ordering.
# Keep the per-trip drains enabled: warm and power-capped ATT both show that
# they improve downstream scheduling by about 0.25% (roughly 240 cycles/wave).
# Front/back power measurements both remain pinned at the same 1000 W cap.
_FLASH_8C_KREAD_VMWAIT = True
_FLASH_8C_KREAD_PRELOOP_VMWAIT = False
# Power probe: skip v_mfma (C passthrough) while still consuming A/B so LDS/DMA
# are not DCE'd. Numerically wrong. Default off.
_FLASH_8C_MFMA_PASSTHROUGH = False
# Power probe: one s_nop IMM after every cluster barrier. -1 = off, 0..15 = ISA imm
# (s_nop 0 waits 1 cycle, s_nop 15 waits 16).
_FLASH_8C_CLUSTER_NOP_IMM = -1
# Power probe: 64× s_nop 15 in exactly one 8c cluster (0..7). -1 = off.
_FLASH_8C_CLUSTER_STALL_ONLY = -1
_FLASH_8C_CLUSTER_STALL_NOPS = 64
# s_waitcnt: lgkmcnt=0, vmcnt=63, expcnt=7. Drain LDS without killing in-flight DMA.
_LGKMCNT_0_ONLY = 0xC07F
FLEX_DTYPE_BF16 = 2
FLEX_DTYPE_FP16 = 3

_LOG2E = 1.4426950408889634
_MAX_BUFFER_BYTES = 0x7FFFFFFF

MASK_NONE = 0
MASK_CAUSAL = 1
MASK_SLIDING_WINDOW = 2
MASK_PREFIX_LM = 3
# Compile-time 8-cluster cutoff. Masked paths skip KV tiles, so the 8c body
# only pays once the tensor Skv is long enough for the last Q tiles.
_LONG_SEQ_8C_SKV_DENSE = 768
_LONG_SEQ_8C_SKV_MASKED = 2048
# Causal mask: WG-uniform skip (flash's q_min_wg predicate) + packed
# attn_mask_vec2_imm on the taken path. Per-wave q_start made the scf.if
# diverge across dualwave groups before s_barrier.
_CAUSAL_WG_UNIFORM_PACKED_MASK = True
# Keep the C0..C7 steady state branch-free: only pairs wholly below the
# bottom-right causal diagonal use 8c.  The short diagonal band falls through
# to the 4c deferred step, which applies the packed mask before softmax.
_CAUSAL_8C_FULL_TILES_ONLY = True
_CAUSAL_NEG_INF_F32_BITS = struct.unpack("<I", struct.pack("<f", -1e9))[0]

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
        kv_hi = _i32_min(raw_hi, fx.Int32(n_kv_tiles))
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
        kv_hi = _i32_min(raw_hi, fx.Int32(n_kv_tiles))
        raw_lo = (q_min_wg - fx.Int32(self.window)) // fx.Int32(block_n)
        kv_lo = _i32_max(raw_lo, fx.Int32(0))
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
        kv_hi = _i32_min(raw_hi, fx.Int32(n_kv_tiles))
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
        bias = (kv_idx - q_idx).to(fx.Float32) * fx.Float32(self.slope) * fx.Float32(_LOG2E)
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
    # Dedicated flash-shaped 8-cluster schedule for long n64 sequences.
    long_seq_8c: fx.Constexpr[bool]
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
    long_seq_8c: Optional[bool] = None,
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
        long_seq_8c=(
            (
                seqlen_kv
                >= (
                    _LONG_SEQ_8C_SKV_MASKED
                    if mask_type
                    in (MASK_CAUSAL, MASK_SLIDING_WINDOW, MASK_PREFIX_LM)
                    else _LONG_SEQ_8C_SKV_DENSE
                )
                if long_seq_8c is None
                else bool(long_seq_8c)
            )
            and block_n == 64
            and head_dim == 128
            and block_threads == 512
            and not paged
        ),
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
    if bool(param.long_seq_8c):
        name += "_8c"
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
        return _f32_max(lhs, rhs)
    else:
        return lhs.addf(rhs, fastmath=_FM)


def _cluster_power_stall(cluster_index):
    """Insert a large s_nop burst at the start of one named 8c cluster."""
    if _FLASH_8C_CLUSTER_STALL_ONLY != cluster_index:
        return
    n = int(_FLASH_8C_CLUSTER_STALL_NOPS)
    if n <= 0:
        return
    llvm.inline_asm(
        None,
        [],
        "\n".join(["s_nop 15"] * n),
        "",
        has_side_effects=True,
    )
    rocdl.sched_barrier(0)


def _mfma_passthrough(a, b, c):
    """Keep A/B/C live without issuing v_mfma. Returns C unchanged."""
    av = as_mlir_value(a)
    bv = as_mlir_value(b)
    cv = as_mlir_value(c)
    llvm.inline_asm(None, [av, bv, cv], "", "v,v,v", has_side_effects=True)
    return c


def _mfma_acc(a, b, c, mma_atom):
    """Single MFMA call: C += A × B. Returns updated accumulator."""
    from flydsl._mlir.dialects import fly

    if _FLASH_8C_MFMA_PASSTHROUGH:
        return _mfma_passthrough(a, b, c)
    acc_ty = c.type
    return fly.mma_atom_call_ssa([acc_ty], mma_atom, a, b, c)


def _idx_to_i32(idx):
    return fx.Int32(arith.index_cast(T.i32, idx))


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
    # One SGPR wave id for the whole kernel. Deriving wave from `tid` inside
    # each buffer_load_lds makes LLVM emit v_readfirstlane per DMA (~66/wave).
    _wave_id_uni_s = rocdl.readfirstlane(
        fx.Int32.ir_type,
        fx.Int32(tid // GFX950_WAVE_SIZE).ir_value(),
    )
    _wave_id_uni = fx.Int32(_wave_id_uni_s)
    # Head-fast grid so linear_id % 8 (XCD) is the head, not the Q tile.
    # gfx950 L2 is private per XCD: Q tiles of one head then replay the same KV
    # on one XCD.  Causal/prefix still reverse q_tile on grid.y so the longest
    # KV loops of every head enqueue first.  grid.z = batch (or batch * splits).
    _SPLITK = int(param.num_kv_splits) > 1
    _num_kv_splits = int(param.num_kv_splits)
    _is_causal = int(param.mask_type) in (MASK_CAUSAL, MASK_PREFIX_LM)
    h_idx = fx.block_idx.x
    if const_expr(_is_causal):
        _num_q_tiles = (seqlen_q + fx.Int32(num_groups * block_m - 1)) // fx.Int32(num_groups * block_m)
        q_tile = fx.Index(arith.index_cast(T.index, _num_q_tiles - fx.Int32(1) - _idx_to_i32(fx.block_idx.y)))
    else:
        q_tile = fx.block_idx.y
    if const_expr(_SPLITK):
        b_idx = fx.block_idx.z // fx.Index(_num_kv_splits)
        split_idx = _idx_to_i32(fx.block_idx.z % fx.Index(_num_kv_splits))
    else:
        b_idx = fx.block_idx.z
    kv_head = h_idx // param.gqa_group

    q_start = (q_tile * num_groups + group) * block_m
    # Match flash-attention's bottom-right causal alignment for contiguous
    # cross-seqlen inputs.  When Skv < Sq, the first Sq-Skv query rows are
    # fully masked; the launcher removes whole dead workgroups and the output
    # store below zeros any dead rows in the first partially-live workgroup.
    # Paged KV keeps its existing top-left convention until it has a per-batch
    # active guard (context lengths may differ across batches).
    _bottom_right_causal = int(param.mask_type) == MASK_CAUSAL and not _paged
    if const_expr(_bottom_right_causal):
        _causal_q_shift = seqlen_kv - seqlen_q
    else:
        _causal_q_shift = fx.Int32(0)

    if const_expr(_paged):
        _ctx_len_it = fx.recast_iter(fx.Int32, fx.get_iter(context_lens))
        _ctx_len = fx.Int32(fx.ptr_load(_ctx_len_it + _idx_to_i32(b_idx)))
        n_kv_tiles = (_ctx_len + fx.Int32(block_n - 1)) // fx.Int32(block_n)
    else:
        n_kv_tiles = param.n_kv_tiles

    # ── LDS: K/V staging (shared across all groups) + per-group P bridge ──────
    kv_tile_elems = block_n * head_dim
    _v_subtile_elems = block_n * 32
    # A V transpose step is always 8 score rows x 32 D cols; block_n decides how
    # many steps tile one D-chunk (4 at block_n=32, 8 at block_n=64).
    _v_score_rows_per_step = 8
    _v_step_elems = _v_score_rows_per_step * 32  # 256 elements per step
    _v_transpose_steps = block_n // _v_score_rows_per_step
    _lds_ring_slots = max(2, int(param.pipe_depth))
    # Match flash_attn_dualwave_swp's physical LDS map on n64 (512-thread CTA):
    # 8 wave-linear repetitions, 2 D repetitions, with one 16B pad per K
    # line and one 64B pad per V line. n32 keeps the compact swizzled layout
    # and the overlapping-softmax pair schedule.
    _flash_lds = bool(block_n == 64 and head_dim == 128 and param.block_threads == 512)
    _flash_smem_n_rpt = 8
    _flash_smem_d_rpt = 2
    _flash_k_line = 520  # 512 bf16 payload + 8 bf16 (16B) pad
    _flash_v_line = 544  # 512 bf16 payload + 32 bf16 (64B) pad
    _flash_k_tile_elems = _flash_smem_n_rpt * _flash_smem_d_rpt * _flash_k_line
    _flash_v_tile_elems = _flash_smem_n_rpt * _flash_smem_d_rpt * _flash_v_line
    _k_storage_elems = _flash_k_tile_elems if _flash_lds else kv_tile_elems
    _v_storage_elems = _flash_v_tile_elems if _flash_lds else kv_tile_elems

    _k_lds_pad_elems = _K_HALF_BANK_SKEW_ELEMS + _LDS_RING_BANK_SKEW_ELEMS

    if const_expr(_paged):

        @fx.struct
        class SharedStorage:
            k_lds_0: fx.Array[elem_dtype, _k_storage_elems + _K_HALF_BANK_SKEW_ELEMS, 16]
            k_lds_1: fx.Array[elem_dtype, _k_storage_elems + _k_lds_pad_elems, 16]
            v_lds_0: fx.Array[elem_dtype, _v_storage_elems, 16]
            v_lds_1: fx.Array[elem_dtype, _v_storage_elems + _LDS_RING_BANK_SKEW_ELEMS, 16]
            p: fx.Array[elem_dtype, num_groups * block_m * block_n, 16]
            bt: fx.Array[fx.Int32, _PAGED_BT_LDS_SIZE, 16]

    else:

        @fx.struct
        class SharedStorage:
            k_lds_0: fx.Array[elem_dtype, _k_storage_elems + _K_HALF_BANK_SKEW_ELEMS, 16]
            k_lds_1: fx.Array[elem_dtype, _k_storage_elems + _k_lds_pad_elems, 16]
            v_lds_0: fx.Array[elem_dtype, _v_storage_elems, 16]
            v_lds_1: fx.Array[elem_dtype, _v_storage_elems + _LDS_RING_BANK_SKEW_ELEMS, 16]
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
    # QK C-fragment template. After the operand swap (K=A, Q=B) the score tile is
    # [M=block_n score, N=block_m query], so the view must be (block_n, block_m) --
    # square only when block_n==block_m. No data flows through here: PV packs P
    # straight from registers, so this exists purely to derive the C layout.
    sP = fx.make_view(
        storage.p.peek().ptr + group * fx.Int32(block_m * block_n),
        fx.make_layout((block_n, block_m), (block_m, 1)),
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

    # Q is loaded once into VGPRs. Cached (0): SC1/NT Q did not beat this
    # occupancy-held and 18s-clock sweep.
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
        _scale_log2e_f32 = scale * fx.Float32(_LOG2E)
        for qi in range_constexpr(n_q):
            frag_Q[qi] = _to_elem(_to_elem(frag_Q[qi], fx.Float32) * _scale_log2e_f32, elem_dtype)

    # Persistent O accumulator: 4 × v16f32 (one per D-chunk).
    # With V=A, P=B PV GEMM: each v16f32 has 16 D-values at 1 query-row per lane.
    Vec = fx.Vector
    _n_d_chunks = head_dim // 32
    # One PV MFMA consumes mma_k score positions, so block_n sets how many
    # A/B packs the PV GEMM needs: 2 at block_n=32, 4 (16 MFMAs) at block_n=64.
    _pv_packs = block_n // int(param.mma_k)
    # Width of the deferred-PV V register carry: one v8elem per pack per D-chunk
    # (8 regs at block_n=32, 16 at block_n=64).
    _v_carry = _pv_packs * _n_d_chunks
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
        scale_log2e = fx.Float32(1.0)
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
    _kv_head_byte_offset = _idx_to_i32(kv_head) * fx.Int32(head_dim * param.in_data_bytes)

    def _stage_flash_lds(tile_idx, buf, do_k, do_v):
        """Flash dual-wave's non-vectorized wave-linear padded K/V DMA map."""
        lane = fx.Int32(tid % GFX950_WAVE_SIZE)
        wave = _wave_id_uni
        n_in_warp = lane // fx.Int32(8)
        d_bucket = lane % fx.Int32(8)
        tile_row = n_in_warp * fx.Int32(8) + wave
        if const_expr(_paged):
            page_id = _load_page_id(tile_idx)
            src_base = page_id * fx.Int32(_page_byte_stride) + _kv_head_byte_offset
        else:
            src_base = tile_idx * fx.Int32(block_n * _k_row_stride_bytes)
            src_base = src_base + _idx_to_i32(k_off) * fx.Int32(param.in_data_bytes)
        for d in range_constexpr(_flash_smem_d_rpt):
            d_col = d_bucket * fx.Int32(8) + fx.Int32(d * 64)
            src_byte = src_base + tile_row * fx.Int32(_k_row_stride_bytes) + d_col * fx.Int32(param.in_data_bytes)
            if const_expr(do_k):
                # buffer_load_lds adds lane*16B; m0 is one padded line per wave.
                k_m0 = wave * fx.Int32(_flash_k_line * param.in_data_bytes)
                k_m0 = k_m0 + fx.Int32(d * _flash_smem_n_rpt * _flash_k_line * param.in_data_bytes)
                k_dst = fx.add_offset(sK_i8[buf], k_m0)
                fx.copy(
                    dma_atom,
                    fx.slice(k_div, (None, fx.Int32(src_byte))),
                    fx.make_view(k_dst, fx.make_layout(1, 1)),
                )
            if const_expr(do_v):
                v_m0 = wave * fx.Int32(_flash_v_line * param.in_data_bytes)
                v_m0 = v_m0 + fx.Int32(d * _flash_smem_n_rpt * _flash_v_line * param.in_data_bytes)
                v_dst = fx.add_offset(sV_i8[buf], v_m0)
                fx.copy(
                    dma_atom,
                    fx.slice(v_div, (None, fx.Int32(src_byte))),
                    fx.make_view(v_dst, fx.make_layout(1, 1)),
                )

    gK_flat = fx.rocdl.make_buffer_tensor(
        fx.Tensor(fx.make_view(fx.recast_iter(fx.Int8, fx.get_iter(k)), fx.make_layout(_MAX_BUFFER_BYTES, 1))),
        max_size=True,
    )
    gV_flat = fx.rocdl.make_buffer_tensor(
        fx.Tensor(fx.make_view(fx.recast_iter(fx.Int8, fx.get_iter(v)), fx.make_layout(_MAX_BUFFER_BYTES, 1))),
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
        wave_off = _wave_id_uni * fx.Int32(GFX950_WAVE_SIZE * _dma_bytes)
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
                v_step = dc * _v_transpose_steps + score_row // _v_score_rows_per_step
                row_in_step = score_row % _v_score_rows_per_step
                lds_byte = v_step * _v_step_bytes + row_in_step * _v_subtile_row_bytes + tile_col_byte
                lds_v = fx.add_offset(sV_i8[buf], lds_byte)
                d_global_byte = dc * 32 * param.in_data_bytes + tile_col_byte
                gmem_byte = fx.Int32(v_global_base) + score_row * fx.Int32(_v_row_stride_bytes) + d_global_byte
                fx.copy(
                    dma_atom, fx.slice(v_div, (None, fx.Int32(gmem_byte))), fx.make_view(lds_v, fx.make_layout(1, 1))
                )

    def _stage_kv_to_lds_paged(page_id, buf, ops=_dma_ops_per_thread, op_offset=0, do_k=True, do_v=True):
        wave_off = _wave_id_uni * fx.Int32(GFX950_WAVE_SIZE * _dma_bytes)
        _step_bytes = block_threads * _dma_bytes
        if const_expr(do_k):
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
        if const_expr(do_v):
            v_global_base = page_id * fx.Int32(_page_byte_stride) + _kv_head_byte_offset
            _v_step_bytes = _v_step_elems * param.in_data_bytes
            for i in range_constexpr(ops):
                flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
                tile_row = flat_byte // _v_subtile_row_bytes
                tile_col_byte = flat_byte % _v_subtile_row_bytes
                dc = tile_row // block_n
                score_row = tile_row % block_n
                v_step = dc * _v_transpose_steps + score_row // _v_score_rows_per_step
                row_in_step = score_row % _v_score_rows_per_step
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
                    _wave_off = _wave_id_uni * fx.Int32(GFX950_WAVE_SIZE * _dma_bytes)
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
                v_step = dc * _v_transpose_steps + score_row // _v_score_rows_per_step
                row_in_step = score_row % _v_score_rows_per_step
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
        if const_expr(_flash_lds):
            _stage_flash_lds(tile_idx, slot, True, True)
        elif const_expr(_paged):
            _pid = _load_page_id(tile_idx)
            _stage_kv_to_lds_paged(_pid, slot, ops=ops, op_offset=op_offset)
        elif _K_HALF_BANK_SKEW_BYTES > 0:
            _stage_kv_to_lds_strided(tile_idx, slot, 0, ops=ops, op_offset=op_offset)
            _stage_kv_to_lds_strided(tile_idx, slot, 1, ops=ops, op_offset=op_offset)
            _stage_kv_to_lds_strided(tile_idx, slot, 2, ops=ops, op_offset=op_offset)
        else:
            _stage_kv_to_lds_contiguous(tile_idx, slot, True, True, ops=ops, op_offset=op_offset)
        return []

    def load_k(tile_idx, slot, ops=_dma_ops_per_thread, op_offset=0):
        """DMA only K into LDS[slot]. Used in the V-read mem cluster (flash C2)."""
        if const_expr(_flash_lds):
            _stage_flash_lds(tile_idx, slot, True, False)
        elif const_expr(_paged):
            _pid = _load_page_id(tile_idx)
            _stage_kv_to_lds_paged(_pid, slot, ops=ops, op_offset=op_offset, do_k=True, do_v=False)
        elif _K_HALF_BANK_SKEW_BYTES > 0:
            _stage_kv_to_lds_strided(tile_idx, slot, 0, ops=ops, op_offset=op_offset)
            _stage_kv_to_lds_strided(tile_idx, slot, 1, ops=ops, op_offset=op_offset)
        else:
            _stage_kv_to_lds_contiguous(tile_idx, slot, True, False, ops=ops, op_offset=op_offset)
        return []

    def load_v(tile_idx, slot, ops=_dma_ops_per_thread, op_offset=0):
        """DMA only V into LDS[slot]. Used in the K-read mem cluster (flash C0)."""
        if const_expr(_flash_lds):
            _stage_flash_lds(tile_idx, slot, False, True)
        elif const_expr(_paged):
            _pid = _load_page_id(tile_idx)
            _stage_kv_to_lds_paged(_pid, slot, ops=ops, op_offset=op_offset, do_k=False, do_v=True)
        elif _K_HALF_BANK_SKEW_BYTES > 0:
            _stage_kv_to_lds_strided(tile_idx, slot, 2, ops=ops, op_offset=op_offset)
        else:
            _stage_kv_to_lds_contiguous(tile_idx, slot, False, True, ops=ops, op_offset=op_offset)
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
        """Build compact (n32) V LDS→register transpose read for PV GEMM A.

        Returns (v_lo_regs, v_hi_regs): lists of length _n_d_chunks. Each entry
        is one v8elem consumed by pv_gemm_register_pair as MFMA A.

        n64 flash LDS does not use this path (see read_v_flash).
        """
        _slot_v_ptr = sV_ptr[slot]

        def _read():
            base_ptr = fx.add_offset(_slot_v_ptr, fx.make_int_tuple(_v_lane_elem))
            halves = [[None] * 4 for _ in range_constexpr(_n_d_chunks)]
            for k_sub in range_constexpr(4):
                for dc in range_constexpr(_n_d_chunks):
                    off = (dc * 4 + k_sub) * _v_step_elems
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
                if const_expr(_FLASH_8C_MFMA_PASSTHROUGH):
                    _mfma_passthrough(
                        frag_K_in[None, m, ki].load(),
                        frag_Q_in[None, n, ki].load(),
                        frag_S_acc[None, m, n].load(),
                    )
                else:
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

    def _read_k_ki_into_frag(ki, slot):
        read_k_work_split(ki_count=1, ki_offset=ki, slot=slot)

    def read_k_prefetch(slot, start_ki=0, ki_count=2):
        """Issue the first K panels in the MEM cluster so C1 GEMM is not LDS-bound."""
        for ki in range_constexpr(ki_count):
            _read_k_ki_into_frag(start_ki + ki, slot)

    def gemm1_qk_k_from_lds(frag_Q_in, slot=0):
        """QK GEMM: consume prefetched ki=0,1 while later panels stream from LDS."""
        frag_S_out = thr_qk.make_fragment_C(sP)
        frag_S_out.fill(0.0)
        rocdl.s_waitcnt(lgkmcnt=0)
        for ki_pair in range_constexpr(_qk_k_reps // 2):
            ki = ki_pair * 2
            if const_expr(ki + 2 < _qk_k_reps):
                _read_k_ki_into_frag(ki + 2, slot)
                _read_k_ki_into_frag(ki + 3, slot)
            gemm1_qk_mfma(frag_S_out, frag_Q_in, frag_K[slot], ki)
            gemm1_qk_mfma(frag_S_out, frag_Q_in, frag_K[slot], ki + 1)
            if const_expr(ki + 2 < _qk_k_reps):
                rocdl.s_waitcnt(lgkmcnt=0)
        return [frag_S_out]

    def read_k_flash(slot):
        """Read full K using flash's padded wave-linear LDS addresses."""
        lane = fx.Int32(local_tid % GFX950_WAVE_SIZE)
        lane_mod_32 = lane % fx.Int32(32)
        lane_div_32 = lane // fx.Int32(32)
        lane_base = (lane_mod_32 % fx.Int32(8)) * fx.Int32(_flash_k_line)
        lane_base = lane_base + (lane_mod_32 // fx.Int32(8)) * fx.Int32(64)
        lane_base = lane_base + lane_div_32 * fx.Int32(8)
        pack_ty = Vec.make_type(8, elem_dtype)
        for ki in range_constexpr(_qk_k_reps):
            ki_off = (ki // 4) * (_flash_smem_n_rpt * _flash_k_line) + (ki % 4) * 16
            lo_ptr = fx.add_offset(sK_ptr[slot], fx.make_int_tuple(lane_base + fx.Int32(ki_off)))
            hi_ptr = fx.add_offset(lo_ptr, fx.make_int_tuple(fx.Int32(256)))
            lo = Vec(fx.ptr_load(lo_ptr, result_type=pack_ty))
            hi = Vec(fx.ptr_load(hi_ptr, result_type=pack_ty))
            if const_expr(slot == 0):
                _k_frag_retile_0[None, 0, ki].store(lo)
                _k_frag_retile_0[None, 1, ki].store(hi)
            else:
                _k_frag_retile_1[None, 0, ki].store(lo)
                _k_frag_retile_1[None, 1, ki].store(hi)
        return frag_K[slot]

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
    b_i32 = _idx_to_i32(b_idx)
    h_i32 = _idx_to_i32(h_idx)
    q_idx_mod = _idx_to_i32(q_start) + fx.Int32(local_tid % 32) + _causal_q_shift
    lane_group_off = fx.Int32((local_tid // 32) * 4)
    kv_offsets = [8 * (e // 4) + (e % 4) for e in range(n_c)]
    # Same predicate as flash causal_mask_prologue_if_needed: first live row of
    # the whole WG, not this group's q_start. Uniform across the 8 dualwave groups.
    _causal_wg_q_min = _idx_to_i32(q_tile) * fx.Int32(num_groups * block_m) + _causal_q_shift
    _causal_neg_inf_i32 = fx.Int32(_CAUSAL_NEG_INF_F32_BITS)
    _causal_pair_thresholds = [(kv_offsets[i], kv_offsets[i + 1]) for i in range(0, n_c, 2)]

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

    def _mask_scores(s_in, kv_tile_idx):
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        return [
            _mod_apply_mask(
                s_in[e], q_idx_mod, kv_base + fx.Int32(kv_offsets[e])
            )
            for e in range_constexpr(n_c)
        ]

    def _mask_scores_packed(s_in, kv_tile_idx):
        """Flash attn_mask_vec2_imm on flex's C-fragment KV offsets."""
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        rel_i32 = q_idx_mod - kv_base
        s_out = [s_in[e] for e in range_constexpr(n_c)]
        for p in range_constexpr(len(_causal_pair_thresholds)):
            thr_x, thr_y = _causal_pair_thresholds[p]
            idx_x = p * 2
            idx_y = p * 2 + 1
            new_x, new_y = _attn_mask_vec2_imm(
                rel_i32,
                _causal_neg_inf_i32,
                thr_x,
                thr_y,
                _bitcast_i32(s_out[idx_x]),
                _bitcast_i32(s_out[idx_y]),
            )
            s_out[idx_x] = _bitcast_f32(new_x)
            s_out[idx_y] = _bitcast_f32(new_y)
        return s_out

    def apply_mods(frag_S_in, kv_tile_idx):
        if const_expr(mod_has_score):
            apply_score_mods(frag_S_in, kv_tile_idx)
        if const_expr(mod_has_mask):
            if const_expr(int(param.mask_type) == MASK_CAUSAL):
                s_out = [frag_S_in[e] for e in range_constexpr(n_c)]
                if const_expr(_CAUSAL_WG_UNIFORM_PACKED_MASK):
                    # Flash predicate: mask if any row in the WG can see this tile.
                    needs_mask = flex_mod.tile_needs_mask(
                        kv_tile_idx, _causal_wg_q_min, block_n
                    )
                else:
                    needs_mask = flex_mod.tile_needs_mask(
                        kv_tile_idx, _idx_to_i32(q_start) + _causal_q_shift, block_n
                    )
                if needs_mask:
                    if const_expr(_CAUSAL_WG_UNIFORM_PACKED_MASK and n_c % 2 == 0):
                        s_out = _mask_scores_packed(s_out, kv_tile_idx)
                    else:
                        s_out = _mask_scores(s_out, kv_tile_idx)
                for e in range_constexpr(n_c):
                    frag_S_in[e] = s_out[e]
            elif const_expr(int(param.mask_type) == MASK_SLIDING_WINDOW):
                # Interior band tiles need no mask. Use q_start for the causal
                # edge and the last row in the group for the left window edge.
                s_out = [frag_S_in[e] for e in range_constexpr(n_c)]
                q_lo = _idx_to_i32(q_start)
                q_hi = q_lo + fx.Int32(int(block_m) - 1)
                kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
                kv_tile_start = kv_tile_idx * fx.Int32(block_n)
                too_far = kv_tile_end > q_lo
                out_of_window = (q_hi - kv_tile_start) > fx.Int32(int(param.mask_window))
                needs_mask = too_far | out_of_window
                if needs_mask:
                    s_out = _mask_scores(s_out, kv_tile_idx)
                for e in range_constexpr(n_c):
                    frag_S_in[e] = s_out[e]
            else:
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

    def _scale_o_if_needed(o_accs_in, scale_scalar):
        """Wave-uniform fast path for the overwhelmingly common corr == 1 case."""
        if const_expr(not _FLASH_LAZY_O_RESCALE):
            return _scale_o_vec(o_accs_in, scale_scalar)

        @flyc.jit
        def _run(o_accs, corr):
            is_identity = fx.Float32(corr) == fx.Float32(1.0)
            identity_lanes = rocdl.ballot(T.i64, as_mlir_value(is_identity))
            all_identity = arith.cmpi(
                arith.CmpIPredicate.eq,
                as_mlir_value(identity_lanes),
                _read_exec_i64(),
            )
            all_identity = llvm.intr_expect(
                all_identity,
                arith.constant(1, type=ir.IntegerType.get_signless(1)),
            )
            state = [as_mlir_value(o_accs[dc]) for dc in range(_n_d_chunks)]
            names = tuple(f"_lazy_o{dc}" for dc in range(_n_d_chunks))

            def _rescale(_n, *_st):
                scaled = _scale_o_vec(o_accs, corr)
                return [as_mlir_value(scaled[dc]) for dc in range(_n_d_chunks)]

            return list(
                scf_if_dispatch(
                    all_identity,
                    lambda *_a: None,
                    _rescale,
                    state_names=names,
                    state_values=state,
                )
            )

        return _run(o_accs_in, scale_scalar)

    _prescaled_q = const_expr(_is_32x32)

    def softmax_start(frag_S_in, m_i_in):
        s_elems = [frag_S_in[i] for i in range_constexpr(n_c)]
        if const_expr(_NO_SOFTMAX_MATH):
            return fx.Float32(1.0), s_elems, m_i_in[0]
        if const_expr(not _prescaled_q):
            _sl2e_vec = Vec.from_elements([scale_log2e], fx.Float32).broadcast_to(16)
            s_scaled = Vec.from_elements(s_elems, fx.Float32) * _sl2e_vec
            s_out = [s_scaled[i] for i in range_constexpr(n_c)]
        else:
            s_out = s_elems
        tile_max = s_out[0]
        for i in range_constexpr(1, n_c):
            tile_max = _f32_max(tile_max, s_out[i])
        tile_max = _permlane32_reduce(tile_max, "max")
        m_new = _f32_max(m_i_in[0], tile_max)
        corr_scalar = _hw_exp2(m_i_in[0] - m_new)
        return corr_scalar, s_out, m_new

    def softmax_finish(s_scaled, m_i_in, l_i_in, o_accs_in, corr_scalar):
        if const_expr(_NO_SOFTMAX_MATH):
            return [s_scaled, m_i_in, l_i_in, o_accs_in, [fx.Float32(1.0)]]
        m_new = m_i_in[0]
        p_elems = [_hw_exp2(s_scaled[i] - m_new) for i in range_constexpr(n_c)]
        p_vec = Vec.from_elements(p_elems, fx.Float32)
        local_sum = p_vec.reduce("add", init_val=fx.Float32(0.0), fastmath=_FM)
        local_sum = _permlane32_reduce(local_sum, "sum")
        corr = [corr_scalar]
        l_new = fx.Float32(fx.fma(l_i_in[0], corr_scalar, local_sum, fastmath=_FM))
        l_i_out = [l_new] + [l_i_in[r] for r in range_constexpr(1, npair)]
        o_accs_out = _scale_o_vec(o_accs_in, corr_scalar)
        return [p_elems, m_i_in, l_i_out, o_accs_out, corr]

    _flash_p_mem_elems = (3 * n_c) // 4

    def softmax_finish_mem_part(s_scaled, m_i_in, o_accs_in, corr_scalar):
        """Independent C2 work placed after V LDS/DMA issue and before wait."""
        m_new = m_i_in[0]
        p_mem = [
            _hw_exp2(s_scaled[i] - m_new)
            for i in range_constexpr(_flash_p_mem_elems)
        ]
        p_mem_vec = Vec.from_elements(p_mem, fx.Float32)
        sum_mem = p_mem_vec.reduce("add", init_val=fx.Float32(0.0), fastmath=_FM)
        return p_mem, sum_mem, _scale_o_vec(o_accs_in, corr_scalar)

    def softmax_finish_compute_part(s_scaled, m_i_in, l_i_in, corr_scalar, p_mem, sum_mem):
        """Finish P/l in C3 after the C2 memory wait."""
        m_new = m_i_in[0]
        p_compute = [
            _hw_exp2(s_scaled[_flash_p_mem_elems + i] - m_new)
            for i in range_constexpr(n_c - _flash_p_mem_elems)
        ]
        p_compute_vec = Vec.from_elements(p_compute, fx.Float32)
        local_sum = sum_mem + p_compute_vec.reduce(
            "add", init_val=fx.Float32(0.0), fastmath=_FM
        )
        local_sum = _permlane32_reduce(local_sum, "sum")
        l_new = fx.Float32(fx.fma(l_i_in[0], corr_scalar, local_sum, fastmath=_FM))
        l_i_out = [l_new] + [l_i_in[r] for r in range_constexpr(1, npair)]
        return p_mem + p_compute, l_i_out

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
        """Pack the C fragment into one v8elem MFMA B pack per PV step."""
        return [
            _pack_8_f32_to_v8elem([frag_P_in[pk * 8 + i] for i in range(8)])
            for pk in range_constexpr(_pv_packs)
        ]

    def _p_packs_to_vec(p_packs):
        return Vec.from_elements(
            [
                Vec(p_packs[pk])[i]
                for pk in range_constexpr(_pv_packs)
                for i in range_constexpr(8)
            ],
            elem_dtype,
        ).ir_value()

    def _p_vec_to_packs(p_all):
        p_vec = Vec(p_all, (n_c,), elem_dtype)
        return [
            p_vec.shuffle(p_vec, [pk * 8 + i for i in range(8)]).ir_value()
            for pk in range_constexpr(_pv_packs)
        ]

    def _scale_p_packs(p_packs, corr):
        p_all = _p_packs_to_vec(p_packs)
        p_f32_ty = Vec.make_type(n_c, fx.Float32)
        p_elem_ty = Vec.make_type(n_c, elem_dtype)
        fm_fast = ir.Attribute.parse("#llvm.fastmath<fast>")
        p_f32_op = llvm.FPExtOp(p_f32_ty, as_mlir_value(p_all))
        p_f32_op.operation.attributes["fastmathFlags"] = fm_fast
        corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(n_c)
        p_scaled = as_mlir_value(corr_vec * Vec(p_f32_op.result))
        p_elem_op = llvm.FPTruncOp(p_elem_ty, p_scaled)
        p_elem_op.operation.attributes["fastmathFlags"] = fm_fast
        return _p_vec_to_packs(p_elem_op.result)

    def _anchor_p_packs(p_packs):
        p_all = as_mlir_value(_p_packs_to_vec(p_packs))
        anchored = llvm.inline_asm(
            p_all.type,
            [p_all],
            "",
            "=v,0",
            has_side_effects=True,
        )
        return _p_vec_to_packs(anchored)

    def _flash_c3_lazy_rescale(o_accs, m_i, l_i, tile_max, p_packs):
        """Flash-style wave-uniform rebase of O/l and pending packed P."""

        @flyc.jit
        def _run(o_in, m_row, l_row, tile_max_i, p_in):
            below = (fx.Float32(tile_max_i) - fx.Float32(m_row)) <= fx.Float32(
                _FLASH_LAZY_RESCALE_THRESHOLD
            )
            below_lanes = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq,
                as_mlir_value(below_lanes),
                _read_exec_i64(),
            )
            all_below = llvm.intr_expect(
                all_below,
                arith.constant(1, type=ir.IntegerType.get_signless(1)),
            )
            p_all = _p_packs_to_vec(p_in)
            state = [as_mlir_value(o_in[dc]) for dc in range(_n_d_chunks)]
            state += [
                as_mlir_value(p_all),
                as_mlir_value(l_row),
                as_mlir_value(m_row),
            ]
            names = tuple(f"_c3_lr{i}" for i in range(_n_d_chunks + 3))

            def _rescale(_n, *_st):
                m_new = _f32_max(m_row, tile_max_i)
                corr = _hw_exp2(fx.Float32(m_row) - m_new)
                o_scaled = _scale_o_vec(o_in, corr)
                p_scaled = _scale_p_packs(p_in, corr)
                out = [as_mlir_value(o_scaled[dc]) for dc in range(_n_d_chunks)]
                out += [
                    as_mlir_value(_p_packs_to_vec(p_scaled)),
                    as_mlir_value(fx.Float32(l_row) * corr),
                    as_mlir_value(m_new),
                ]
                return out

            result = scf_if_dispatch(
                all_below,
                lambda *_a: None,
                _rescale,
                state_names=names,
                state_values=state,
            )
            o_out = list(result[:_n_d_chunks])
            p_out = _p_vec_to_packs(result[_n_d_chunks])
            l_out = result[_n_d_chunks + 1]
            m_out = result[_n_d_chunks + 2]
            return o_out, p_out, l_out, m_out

        o_out, p_out, l_out, m_out = _run(
            o_accs, m_i[0], l_i[0], tile_max, p_packs
        )
        return (
            o_out,
            [fx.Float32(m_out)] + [m_i[r] for r in range_constexpr(1, npair)],
            [fx.Float32(l_out)] + [l_i[r] for r in range_constexpr(1, npair)],
            p_out,
        )

    def pv_gemm_register_packs(frag_P_in, v_regs, o_accs):
        """PV GEMM for flattened V packs (n64 flash LDS)."""
        p_packs = _pack_p_b(frag_P_in)
        for pk in range_constexpr(_pv_packs):
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(
                    v_regs[pk * _n_d_chunks + dc], p_packs[pk], o_accs[dc], _pv_mma_atom
                )

    def pv_gemm_register_pair(frag_P_in, v_lo_regs, v_hi_regs, o_accs):
        """PV GEMM for n32 overlapping softmax: two K-packs per D-chunk."""
        p_packs = _pack_p_b(frag_P_in)
        p_lo, p_hi = p_packs[0], p_packs[1]
        for dc in range_constexpr(_n_d_chunks):
            o_accs[dc] = _mfma_acc(v_lo_regs[dc], p_lo, o_accs[dc], _pv_mma_atom)
            o_accs[dc] = _mfma_acc(v_hi_regs[dc], p_hi, o_accs[dc], _pv_mma_atom)

    def read_v_mfma_step(slot, pk):
        """One PV pack of V from LDS: 2 transpose steps × all D-chunks."""
        if const_expr(_flash_lds):
            lane = fx.Int32(local_tid % GFX950_WAVE_SIZE)
            lane_div_32 = lane // fx.Int32(32)
            lane_base = lane_div_32 * fx.Int32(2176)
            lane_base = lane_base + ((lane % fx.Int32(16)) // fx.Int32(4)) * fx.Int32(_flash_v_line)
            lane_base = lane_base + ((lane // fx.Int32(16)) % fx.Int32(2)) * fx.Int32(16)
            lane_base = lane_base + (lane % fx.Int32(4)) * fx.Int32(4)
            base_ptr = fx.add_offset(sV_ptr[slot], fx.make_int_tuple(lane_base))
        else:
            base_ptr = fx.add_offset(sV_ptr[slot], fx.make_int_tuple(_v_lane_elem))
        v_regs = [None] * _n_d_chunks
        for dc in range_constexpr(_n_d_chunks):
            if const_expr(_flash_lds):
                # Flash non-vectorized V: k-substep stride 128, D pair stride
                # 8*544, inner D-chunk stride 32, transpose pair stride 64.
                off_a = pk * 128 + (dc // 2) * (_flash_smem_n_rpt * _flash_v_line) + (dc % 2) * 32
                off_b = off_a + 64
            else:
                off_a = (dc * _v_transpose_steps + 2 * pk) * _v_step_elems
                off_b = off_a + _v_step_elems
            src_a = fx.make_view(fx.add_offset(base_ptr, fx.make_int_tuple(off_a)), _v_tr_layout)
            src_b = fx.make_view(fx.add_offset(base_ptr, fx.make_int_tuple(off_b)), _v_tr_layout)
            dst_a = fx.make_rmem_tensor(_v_tr_layout, elem_dtype)
            dst_b = fx.make_rmem_tensor(_v_tr_layout, elem_dtype)
            fx.copy(_v_tr_atom, src_a, dst_a)
            fx.copy(_v_tr_atom, src_b, dst_b)
            v_regs[dc] = Vec(dst_a.load()).shuffle(Vec(dst_b.load()), list(range(8))).ir_value()
        return v_regs

    def pv_gemm_register_streamed(frag_P_in, slot, o_accs, v_regs0):
        """PV GEMM; C2 already issued pack 0. Later V packs issue under MFMA."""
        v_regs = v_regs0
        for pk in range_constexpr(_pv_packs):
            p_pack = _pack_8_f32_to_v8elem([frag_P_in[pk * 8 + i] for i in range(8)])
            if const_expr(pk + 1 < _pv_packs):
                v_next = read_v_mfma_step(slot, pk + 1)
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(v_regs[dc], p_pack, o_accs[dc], _pv_mma_atom)
            if const_expr(pk + 1 < _pv_packs):
                rocdl.s_waitcnt(lgkmcnt=0)
                v_regs = v_next

    def read_v_flash(slot):
        """Read all four PV packs using flash's padded V LDS map."""
        v_regs = [None] * (_pv_packs * _n_d_chunks)
        for pk in range_constexpr(_pv_packs):
            pack = read_v_mfma_step(slot, pk)
            for dc in range_constexpr(_n_d_chunks):
                v_regs[pk * _n_d_chunks + dc] = pack[dc]
        return v_regs

    def dualwave_cluster_sync(cluster_index):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        if const_expr(_FLASH_8C_CLUSTER_NOP_IMM >= 0):
            llvm.inline_asm(
                None,
                [],
                f"s_nop {int(_FLASH_8C_CLUSTER_NOP_IMM)}",
                "",
                has_side_effects=True,
            )
            rocdl.sched_barrier(0)

    def sched_interleave_lds_math(groups, *, ds, trans=0, valu=0, sync_id=0):
        """Alternate LDS reads with softmax arithmetic in a memory cluster.

        The AMDGPU scheduler counts transcendentals (`v_exp_f32`) in their own
        group, so exp2 has to be asked for separately from plain VALU.
        """
        if const_expr(_FLASH_SCHED_LDS_VALU & (1 << sync_id)):
            for _ in range_constexpr(groups):
                rocdl.sched_group_barrier(0x100, ds, sync_id)
                if const_expr(trans):
                    rocdl.sched_group_barrier(0x400, trans, sync_id)
                if const_expr(valu):
                    rocdl.sched_group_barrier(0x002, valu, sync_id)
        if const_expr(_FLASH_SCHED_PIN_LDS):
            rocdl.sched_barrier(0)

    def sched_interleave_qk_sum(groups, *, mfma, valu):
        """Distribute the deferred P reduction through the independent QK GEMM."""
        if const_expr(_FLASH_SCHED_QK_SUM):
            for _ in range_constexpr(groups):
                rocdl.sched_group_barrier(0x008, mfma, 2)
                rocdl.sched_group_barrier(0x002, valu, 2)

    def sched_interleave_pv_softmax(groups, *, mfma, valu=0, trans=0):
        """Hide the P[n] softmax chain in the PV GEMM's MFMA shadow."""
        if const_expr(_FLASH_SCHED_PV_SOFTMAX):
            for _ in range_constexpr(groups):
                rocdl.sched_group_barrier(0x008, mfma, 3)
                if const_expr(valu):
                    rocdl.sched_group_barrier(0x002, valu, 3)
                if const_expr(trans):
                    rocdl.sched_group_barrier(0x400, trans, 3)

    def sched_flash_pairs(pairs, *, valu=0, trans=0, group):
        """Flash dualwave's exact MFMA/VALU-or-EXP scheduling recipe."""
        if const_expr(_FLASH_8C_SCHED):
            for _ in range_constexpr(pairs):
                rocdl.sched_group_barrier(0x008, 1, group)
                if const_expr(valu):
                    rocdl.sched_group_barrier(0x002, valu, group)
                if const_expr(trans):
                    rocdl.sched_group_barrier(0x400, trans, group)

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
            infra.stagger_i32 = arith.divsi(
                _wave_id_uni_s, as_mlir_value(fx.Int32(_stagger_div))
            )
        else:
            infra.stagger_i32 = arith.remsi(
                _wave_id_uni_s,
                as_mlir_value(fx.Int32(group_threads // GFX950_WAVE_SIZE)),
            )

    # ── Paged KV: load block table into LDS ──────────────────────────────
    if const_expr(_paged):
        from flydsl._mlir.dialects import llvm as _llvm

        _bt_lds_ptr = storage.bt.peek().ptr
        _bt_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        _bt_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(block_table)), fx.make_layout(_MAX_BUFFER_BYTES, 1))
            ),
            max_size=True,
        )
        _bt_div = fx.logical_divide(_bt_flat, fx.make_layout(1, 1))
        _bt_batch_off = _idx_to_i32(b_idx) * block_table_stride
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
    _q_min_wg = _idx_to_i32(q_tile) * fx.Int32(num_groups * block_m) + _causal_q_shift
    _q_max_wg = _q_min_wg + fx.Int32(num_groups * block_m - 1)
    _kv_lo, _kv_hi = flex_mod.kv_range(_q_min_wg, _q_max_wg, n_kv_tiles, block_n)
    if const_expr(_SPLITK):
        _total_tiles = _kv_hi - _kv_lo
        _chunk = (_total_tiles + fx.Int32(_num_kv_splits - 1)) // fx.Int32(_num_kv_splits)
        _kv_lo = _kv_lo + split_idx * _chunk
        _kv_hi_split = _kv_lo + _chunk
        _kv_hi = _i32_min(_kv_hi_split, _kv_hi)
    _causal_full_tile_8c = (
        _CAUSAL_8C_FULL_TILES_ONLY
        and int(param.mask_type) == MASK_CAUSAL
        and bool(param.long_seq_8c)
        and not _SPLITK
    )
    if const_expr(_causal_full_tile_8c):
        # Tile t is fully visible to every row in the WG iff
        # t*block_n + block_n-1 <= q_min.  Therefore the number of full tiles
        # before the diagonal is floor((q_min+1)/block_n).  Clamp it to this
        # WG's live range; _kv_hi already excludes wholly masked future tiles.
        _kv_full_hi_raw = (_q_min_wg + fx.Int32(1)) // fx.Int32(block_n)
        _kv_full_hi = _i32_min(
            _i32_max(_kv_full_hi_raw, _kv_lo),
            _kv_hi,
        )
    else:
        _kv_full_hi = _kv_hi
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

    def _split_one_tile(kv_i32, slot, o_accs, *, prefetch_mode, extra_mem_barrier, m_i=None, l_i=None, odd_mask=False, odd_valid=None):
        """One KV tile as mem/compute/mem/compute (same dual-wave split as flash).

        Flash C0/C2 pair LDS of current K or V with complementary next-tile DMA.
        C1 is QK (+ softmax_start when softmax is on). C3 is PV (+ softmax_finish
        so P/O rescale sit with the PV GEMM, not with QK).
        """
        next_slot = 1 - int(slot)

        # ── Cluster 0 (mem): full LDS K + DMA next V (flash C0) ──
        rocdl.s_waitcnt(vmcnt=0)
        if const_expr(extra_mem_barrier):
            rocdl.s_barrier()
        if const_expr(_flash_lds):
            k_regs = read_k_flash(slot)
        else:
            read_k_prefetch(slot)
        if const_expr(prefetch_mode == 1):
            load_v(kv_i32 + fx.Int32(1), next_slot)
        elif const_expr(prefetch_mode == 2):
            has_next = (kv_i32 + fx.Int32(1)) < _kv_hi
            if has_next:
                load_v(kv_i32 + fx.Int32(1), next_slot)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(_dma_ops_per_thread)
        dualwave_cluster_sync(0)

        # ── Cluster 1 (compute): register-only QK + softmax start ──
        if const_expr(_flash_lds):
            (frag_S,) = gemm1_qk_unrolled(frag_Q, k_regs)
        else:
            (frag_S,) = gemm1_qk_k_from_lds(frag_Q, slot)
        s_raw = [frag_S[i] for i in range_constexpr(n_c)]
        if const_expr(not _GEMM_ONLY_NO_SOFTMAX):
            if const_expr(mod_has_score or mod_has_mask):
                apply_mods(s_raw, kv_i32)
            if const_expr(odd_mask):
                _neg_inf = fx.Float32(-1e9)
                s_raw = [odd_valid.select(s_raw[i], _neg_inf) for i in range_constexpr(n_c)]
            corr_scalar, s_scaled, m_new = softmax_start(s_raw, m_i)
            m_i = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
        dualwave_cluster_sync(1)

        # Python-level seeds keep values defined across constexpr branches.
        p_mem = s_raw[:_flash_p_mem_elems]
        sum_mem = fx.Float32(0.0)

        # ── Cluster 2 (mem+VALU): LDS V + DMA next K + half P/O rescale ──
        if const_expr(_flash_lds):
            v_regs = read_v_flash(slot)
        else:
            v_regs0 = read_v_mfma_step(slot, 0)
        if const_expr(prefetch_mode == 1):
            load_k(kv_i32 + fx.Int32(1), next_slot)
        elif const_expr(prefetch_mode == 2):
            has_next = (kv_i32 + fx.Int32(1)) < _kv_hi
            if has_next:
                load_k(kv_i32 + fx.Int32(1), next_slot)
        if (
            const_expr(_FLASH_SPLIT_SOFTMAX_ACROSS_V_MEM)
            and const_expr(_flash_lds)
            and const_expr(not _GEMM_ONLY_NO_SOFTMAX)
        ):
            p_mem, sum_mem, o_accs = softmax_finish_mem_part(
                s_scaled, m_i, o_accs, corr_scalar
            )
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(_dma_ops_per_thread)
        dualwave_cluster_sync(2)

        # ── Cluster 3 (compute): finish P/l + register-only PV ──
        if const_expr(not _GEMM_ONLY_NO_SOFTMAX):
            if const_expr(_FLASH_SPLIT_SOFTMAX_ACROSS_V_MEM) and const_expr(_flash_lds):
                p_elems, l_i = softmax_finish_compute_part(
                    s_scaled, m_i, l_i, corr_scalar, p_mem, sum_mem
                )
                pv_gemm_register_packs(p_elems, v_regs, o_accs)
            else:
                out_sm = softmax_finish(s_scaled, m_i, l_i, o_accs, corr_scalar)
                p_elems, l_i, o_accs = out_sm[0], out_sm[2], out_sm[3]
                if const_expr(_flash_lds):
                    pv_gemm_register_packs(p_elems, v_regs, o_accs)
                else:
                    pv_gemm_register_streamed(p_elems, slot, o_accs, v_regs0)
        elif const_expr(_flash_lds):
            pv_gemm_register_packs(s_raw, v_regs, o_accs)
        else:
            pv_gemm_register_streamed(s_raw, slot, o_accs, v_regs0)
        dualwave_cluster_sync(3)
        return m_i, l_i, o_accs

    def _do_pair_gemm_only(kv_i32, o_accs, *, guarded, extra_mem_barrier):
        """Even tile slot0 then odd tile slot1, each LDS / QK / LDS / PV."""
        if const_expr(guarded):
            _, _, o_accs = _split_one_tile(
                kv_i32, 0, o_accs, prefetch_mode=2, extra_mem_barrier=extra_mem_barrier
            )
            odd_valid = (kv_i32 + fx.Int32(1)) < _kv_hi
            if odd_valid:
                _, _, o_accs = _split_one_tile(
                    kv_i32 + fx.Int32(1), 1, o_accs, prefetch_mode=2, extra_mem_barrier=extra_mem_barrier
                )
        else:
            _, _, o_accs = _split_one_tile(
                kv_i32, 0, o_accs, prefetch_mode=1, extra_mem_barrier=False
            )
            _, _, o_accs = _split_one_tile(
                kv_i32 + fx.Int32(1), 1, o_accs, prefetch_mode=1, extra_mem_barrier=False
            )
        return o_accs

    def _do_pair_split_softmax(kv_i32, m_i, l_i, o_accs, *, guarded, extra_mem_barrier):
        """Split schedule with softmax overlapping the QK and PV compute clusters."""
        if const_expr(guarded):
            m_i, l_i, o_accs = _split_one_tile(
                kv_i32, 0, o_accs, prefetch_mode=2, extra_mem_barrier=extra_mem_barrier, m_i=m_i, l_i=l_i
            )
            odd_valid = (kv_i32 + fx.Int32(1)) < _kv_hi
            m_i, l_i, o_accs = _split_one_tile(
                kv_i32 + fx.Int32(1),
                1,
                o_accs,
                prefetch_mode=2,
                extra_mem_barrier=extra_mem_barrier,
                m_i=m_i,
                l_i=l_i,
                odd_mask=True,
                odd_valid=odd_valid,
            )
        else:
            m_i, l_i, o_accs = _split_one_tile(
                kv_i32, 0, o_accs, prefetch_mode=1, extra_mem_barrier=False, m_i=m_i, l_i=l_i
            )
            m_i, l_i, o_accs = _split_one_tile(
                kv_i32 + fx.Int32(1), 1, o_accs, prefetch_mode=1, extra_mem_barrier=False, m_i=m_i, l_i=l_i
            )
        return m_i, l_i, o_accs

    _flash_p_half = n_c // 2

    def _flash_apply_mods_and_mask(s_raw, kv_i32, *, invalid=False, valid=None):
        if const_expr(mod_has_score or mod_has_mask):
            apply_mods(s_raw, kv_i32)
        if const_expr(invalid):
            neg_inf = fx.Float32(-1e9)
            s_raw = [valid.select(s_raw[i], neg_inf) for i in range_constexpr(n_c)]
        return s_raw

    def _flash_finish_deferred_exp(p_mixed_vec):
        """C0: first half is P; second half still holds (S-m)."""
        p_mixed = Vec(p_mixed_vec)
        p_out = [p_mixed[i] for i in range_constexpr(_flash_p_half)]
        p_out += [
            _hw_exp2(p_mixed[_flash_p_half + i])
            for i in range_constexpr(n_c - _flash_p_half)
        ]
        return p_out

    def _flash_add_deferred_sum(p_prev, l_i):
        """C1: add sum(P[n-1]); l was already rescaled in prior C2."""
        p_vec = Vec.from_elements(p_prev, fx.Float32)
        local_sum = p_vec.reduce("add", init_val=fx.Float32(0.0), fastmath=_FM)
        local_sum = _permlane32_reduce(local_sum, "sum")
        l_new = l_i[0] + local_sum
        return [l_new] + [l_i[r] for r in range_constexpr(1, npair)]

    def _flash_rescale_for_current(s_raw, m_i, l_i, o_accs, p_prev):
        """C2: update max for S[n]; defer O rescale until after PV[n-1]."""
        if const_expr(not _prescaled_q):
            scale_vec = Vec.from_elements([scale_log2e], fx.Float32).broadcast_to(n_c)
            scaled_vec = Vec.from_elements(s_raw, fx.Float32) * scale_vec
            s_scaled = [scaled_vec[i] for i in range_constexpr(n_c)]
        else:
            s_scaled = [s_raw[i] for i in range_constexpr(n_c)]
        if const_expr(_FLASH_C3_LAZY_RESCALE):
            return s_scaled, m_i, l_i, o_accs, p_prev, fx.Float32(1.0)
        tile_max = s_scaled[0]
        for i in range_constexpr(1, n_c):
            tile_max = _f32_max(tile_max, s_scaled[i])
        tile_max = _permlane32_reduce(tile_max, "max")
        if const_expr(_FLASH_ELIDE_RESCALE):
            llvm.inline_asm(
                None,
                [as_mlir_value(tile_max)],
                "",
                "v",
                has_side_effects=True,
            )
            return s_scaled, m_i, l_i, o_accs, p_prev, fx.Float32(1.0)

        @flyc.jit
        def _lazy_state(m_row, l_row, tile_max_i):
            below = (fx.Float32(tile_max_i) - fx.Float32(m_row)) <= fx.Float32(
                _FLASH_LAZY_RESCALE_THRESHOLD
            )
            below_lanes = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq,
                as_mlir_value(below_lanes),
                _read_exec_i64(),
            )
            all_below = llvm.intr_expect(
                all_below,
                arith.constant(1, type=ir.IntegerType.get_signless(1)),
            )
            one = fx.Float32(1.0)
            state = [as_mlir_value(l_row), as_mlir_value(m_row), as_mlir_value(one)]

            def _rescale(_n, *_st):
                m_new_i = _f32_max(m_row, tile_max_i)
                corr_i = _hw_exp2(fx.Float32(m_row) - m_new_i)
                return [
                    as_mlir_value(fx.Float32(l_row) * corr_i),
                    as_mlir_value(m_new_i),
                    as_mlir_value(corr_i),
                ]

            return scf_if_dispatch(
                all_below,
                lambda *_a: None,
                _rescale,
                state_names=("_lazy_l", "_lazy_m", "_lazy_corr"),
                state_values=state,
            )

        l_scaled, m_new, corr_scalar = _lazy_state(m_i[0], l_i[0], tile_max)
        l_i_out = [fx.Float32(l_scaled)] + [l_i[r] for r in range_constexpr(1, npair)]
        m_i_out = [fx.Float32(m_new)] + [m_i[r] for r in range_constexpr(1, npair)]
        return s_scaled, m_i_out, l_i_out, o_accs, p_prev, corr_scalar

    def _pin_in_cluster(values):
        """Give pure VALU results a use here so they cannot sink downstream.

        MachineSink runs before the `sched_barrier` in `dualwave_cluster_sync`
        is honoured, so it is free to move the exp2 chain across the cluster
        boundary to shorten live ranges. It does exactly that in one of the two
        unrolled copies of the KV loop, which dumps C3's 16 exp2 into the next
        C0 on top of its own 16. An empty side-effecting asm consuming the
        results anchors them in this block. Emit it after the
        `sched_group_barrier` requests, since it also ends the scheduling
        region.
        """
        vals = [as_mlir_value(v) for v in values]
        llvm.inline_asm(
            None,
            vals,
            "",
            ",".join("v" for _ in vals),
            has_side_effects=True,
        )

    def _flash_make_deferred_p(s_scaled, m_i):
        """C3: begin P[n], leaving its second half for next C0."""
        m_new = m_i[0]
        shifted = [s_scaled[i] - m_new for i in range_constexpr(n_c)]
        p_head = [_hw_exp2(shifted[i]) for i in range_constexpr(_flash_p_half)]
        p_mixed = p_head + [
            shifted[i] for i in range_constexpr(_flash_p_half, n_c)
        ]
        return Vec.from_elements(p_mixed, fx.Float32).ir_value(), p_head

    def _flash_deferred_prologue(kv_i32, m_i, l_i, o_accs):
        """Prime S/P for tile 0; K0/V0 are already resident in slot 0."""
        rocdl.s_waitcnt(vmcnt=0)
        rocdl.s_barrier()
        k_regs = read_k_flash(0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        dualwave_cluster_sync(0)

        (frag_S,) = gemm1_qk_unrolled(frag_Q, k_regs)
        s_raw = [frag_S[i] for i in range_constexpr(n_c)]
        s_raw = _flash_apply_mods_and_mask(s_raw, kv_i32)
        dualwave_cluster_sync(1)

        has_next = (kv_i32 + fx.Int32(1)) < _kv_hi
        if has_next:
            load_k(kv_i32 + fx.Int32(1), 1)
        # There is no pending P yet; only establish tile-0 max/state.
        corr_scalar, s_scaled, m_new = softmax_start(s_raw, m_i)
        m_i = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
        l_i = [l_i[0] * corr_scalar] + [l_i[r] for r in range_constexpr(1, npair)]
        _waitcnt_vm_n(_dma_ops_per_thread)
        dualwave_cluster_sync(2)

        p_mixed, p_head = _flash_make_deferred_p(s_scaled, m_i)
        _pin_in_cluster(p_head)
        dualwave_cluster_sync(3)
        return m_i, l_i, o_accs, p_mixed, corr_scalar

    def _flash_deferred_step(
        kv_i32, cur_slot, prev_slot, m_i, l_i, o_accs, p_mixed_prev, corr_pending
    ):
        """Steady tile n: QK/P preparation for n while consuming PV[n-1]."""
        # C0: launch V[n], issue K[n] LDS, then fill LDS latency with exp(P[n-1]).
        rocdl.s_waitcnt(vmcnt=0)
        load_v(kv_i32, cur_slot)
        k_regs = read_k_flash(cur_slot)
        p_prev = _flash_finish_deferred_exp(p_mixed_prev)
        if const_expr(
            not _FLASH_ELIDE_RESCALE
            and not _FLASH_C3_LAZY_RESCALE
            and not (_FLASH_SCALE_O_IN_C3 and _is_causal)
            and not (_FLASH_SCALE_O_IN_C1 and not _is_causal)
        ):
            # Fallback: corr on O in C0 (neither C1 dense nor C3 causal).
            o_accs = _scale_o_if_needed(o_accs, corr_pending)
        # Lazy rescale leaves eight independent exp2 operations in this region.
        # Pair two K reads with each exp: asking for 16 EXP groups delays the
        # last read and creates a ~44-cycle lgkmcnt tail.
        sched_interleave_lds_math(_qk_k_reps, ds=2, trans=1, sync_id=0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(_dma_ops_per_thread)
        dualwave_cluster_sync(0)

        # C1: independent QK[n] and sum(P[n-1]).
        (frag_S,) = gemm1_qk_unrolled(frag_Q, k_regs)
        s_raw = [frag_S[i] for i in range_constexpr(n_c)]
        s_raw = _flash_apply_mods_and_mask(s_raw, kv_i32)
        l_i = _flash_add_deferred_sum(p_prev, l_i)
        if const_expr(
            not _FLASH_ELIDE_RESCALE
            and not _FLASH_C3_LAZY_RESCALE
            and _FLASH_SCALE_O_IN_C1
            and not _is_causal
        ):
            # O is idle during QK. Hide corr_pending behind the 16 QK MFMAs
            # instead of paying it next to K LDS / exp2 in C0.
            o_accs = _scale_o_if_needed(o_accs, corr_pending)
        # One requested VALU expands to roughly two ISA ops, spreading the
        # reduction across all 16 QK MFMAs without leaving an MFMA-only tail.
        sched_interleave_qk_sum(16, mfma=1, valu=1)
        dualwave_cluster_sync(1)

        # C2: launch K[n+1], issue V[n-1] LDS, then max/rescale for S[n].
        has_next = (kv_i32 + fx.Int32(1)) < _kv_hi
        if has_next:
            load_k(kv_i32 + fx.Int32(1), prev_slot)
        v_prev = read_v_flash(prev_slot)
        s_scaled, m_i, l_i, o_accs, p_prev, corr_scalar = _flash_rescale_for_current(
            s_raw, m_i, l_i, o_accs, p_prev
        )
        sched_interleave_lds_math(_pv_packs * 2, ds=_n_d_chunks, valu=2, sync_id=1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(_dma_ops_per_thread)
        dualwave_cluster_sync(2)

        # C3: consume P[n-1] at the old max, then lazily rescale the combined
        # accumulator in the next C0 (or here, just before PV).
        if const_expr(_FLASH_SETPRIO_PV):
            rocdl.s_setprio(1)
        if const_expr(_FLASH_C3_LAZY_RESCALE):
            p_packs = _pack_p_b(p_prev)
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(
                    v_prev[dc], p_packs[0], o_accs[dc], _pv_mma_atom
                )
            tile_max = s_scaled[0]
            for i in range_constexpr(1, n_c):
                tile_max = _f32_max(tile_max, s_scaled[i])
            tile_max = _permlane32_reduce(tile_max, "max")
            o_accs, m_i, l_i, p_packs = _flash_c3_lazy_rescale(
                o_accs, m_i, l_i, tile_max, p_packs
            )
            for pk in range_constexpr(1, _pv_packs):
                for dc in range_constexpr(_n_d_chunks):
                    o_accs[dc] = _mfma_acc(
                        v_prev[pk * _n_d_chunks + dc],
                        p_packs[pk],
                        o_accs[dc],
                        _pv_mma_atom,
                    )
        else:
            if const_expr(
                not _FLASH_ELIDE_RESCALE and _FLASH_SCALE_O_IN_C3 and _is_causal
            ):
                o_accs = _scale_o_if_needed(o_accs, corr_pending)
            pv_gemm_register_packs(p_prev, v_prev, o_accs)
        p_mixed, p_head = _flash_make_deferred_p(s_scaled, m_i)
        # Only the exp2 chain pays here. Pinning the subtracts and bf16 packs
        # too costs ~1%, the same way it did for the O rescale in C0.
        sched_interleave_pv_softmax(_pv_packs * _n_d_chunks, mfma=1, trans=1)
        _pin_in_cluster(p_head)
        if const_expr(_FLASH_SETPRIO_PV):
            rocdl.s_setprio(0)
        dualwave_cluster_sync(3)
        return m_i, l_i, o_accs, p_mixed, corr_scalar

    def _flash_deferred_step_8c(
        kv_odd, m_i, l_i, o_accs, p_mixed_prev
    ):
        """Flash-shaped C0..C7 pipeline for two consecutive KV tiles."""
        kv_even = kv_odd + fx.Int32(1)

        # C0: V[odd] DMA and resident K[odd] read. K[even] is already in slot 0.
        _cluster_power_stall(0)
        llvm.inline_asm(None, [], "s_nop 7", "", has_side_effects=True)
        rocdl.sched_barrier(0)
        load_v(kv_odd, 1)
        # K in slot 1 was staged earlier; leave only this cluster's V DMA
        # outstanding before issuing its LDS read.
        if const_expr(_FLASH_8C_KREAD_VMWAIT):
            _waitcnt_vm_n(_dma_ops_per_thread)
        k_odd = read_k_flash(1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(2 * _dma_ops_per_thread)
        dualwave_cluster_sync(0)

        # C1: QK[odd], finish/sum P[odd-1], and carry packed P into C3.
        _cluster_power_stall(1)
        (frag_s_odd,) = gemm1_qk_unrolled(frag_Q, k_odd)
        s_odd = [frag_s_odd[i] for i in range_constexpr(n_c)]
        if const_expr(_causal_full_tile_8c and mod_has_score):
            apply_score_mods(s_odd, kv_odd)
        elif const_expr(not _causal_full_tile_8c):
            s_odd = _flash_apply_mods_and_mask(s_odd, kv_odd)
        p_prev = _flash_finish_deferred_exp(p_mixed_prev)
        l_i = _flash_add_deferred_sum(p_prev, l_i)
        p_prev_packs = _anchor_p_packs(_pack_p_b(p_prev))
        sched_flash_pairs(6, trans=3, group=1)
        sched_flash_pairs(10, valu=5, group=1)
        dualwave_cluster_sync(1)

        # C2: prefetch K[odd+2] into the consumed odd slot and read V[odd-1].
        _cluster_power_stall(2)
        llvm.inline_asm(None, [], "s_nop 7", "", has_side_effects=True)
        rocdl.sched_barrier(0)
        if (kv_odd + fx.Int32(2)) < _kv_hi:
            load_k(kv_odd + fx.Int32(2), 1)
        v_prev = read_v_flash(0)
        s_odd, m_i, l_i, o_accs, _p_prev, _corr = (
            _flash_rescale_for_current(s_odd, m_i, l_i, o_accs, p_prev)
        )
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(2 * _dma_ops_per_thread)
        dualwave_cluster_sync(2)

        # C3: PV[odd-1], lazy rebase from S[odd], then begin P[odd].
        _cluster_power_stall(3)
        if const_expr(_FLASH_SETPRIO_PV):
            rocdl.s_setprio(1)
        for dc in range_constexpr(_n_d_chunks):
            o_accs[dc] = _mfma_acc(
                v_prev[dc], p_prev_packs[0], o_accs[dc], _pv_mma_atom
            )
        tile_max_odd = s_odd[0]
        for i in range_constexpr(1, n_c):
            tile_max_odd = _f32_max(tile_max_odd, s_odd[i])
        tile_max_odd = _permlane32_reduce(tile_max_odd, "max")
        sched_flash_pairs(4, valu=6, group=2)
        o_accs, m_i, l_i, p_prev_packs = _flash_c3_lazy_rescale(
            o_accs, m_i, l_i, tile_max_odd, p_prev_packs
        )
        for pk in range_constexpr(1, _pv_packs):
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(
                    v_prev[pk * _n_d_chunks + dc],
                    p_prev_packs[pk],
                    o_accs[dc],
                    _pv_mma_atom,
                )
        p_mixed_odd, p_head_odd = _flash_make_deferred_p(s_odd, m_i)
        sched_flash_pairs(6, valu=6, group=2)
        sched_flash_pairs(6, trans=3, group=2)
        _pin_in_cluster(p_head_odd)
        if const_expr(_FLASH_SETPRIO_PV):
            rocdl.s_setprio(0)
        dualwave_cluster_sync(3)

        # C4: V[even] DMA and resident K[even] read.
        _cluster_power_stall(4)
        llvm.inline_asm(None, [], "s_nop 7", "", has_side_effects=True)
        rocdl.sched_barrier(0)
        load_v(kv_even, 0)
        # K in slot 0 was staged earlier; leave only this cluster's V DMA
        # outstanding before issuing its LDS read.
        if const_expr(_FLASH_8C_KREAD_VMWAIT):
            _waitcnt_vm_n(_dma_ops_per_thread)
        k_even = read_k_flash(0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(2 * _dma_ops_per_thread)
        dualwave_cluster_sync(4)

        # C5: QK[even], finish/sum P[odd], and carry packed P into C7.
        _cluster_power_stall(5)
        (frag_s_even,) = gemm1_qk_unrolled(frag_Q, k_even)
        s_even = [frag_s_even[i] for i in range_constexpr(n_c)]
        if const_expr(_causal_full_tile_8c and mod_has_score):
            apply_score_mods(s_even, kv_even)
        elif const_expr(not _causal_full_tile_8c):
            s_even = _flash_apply_mods_and_mask(s_even, kv_even)
        p_odd = _flash_finish_deferred_exp(p_mixed_odd)
        l_i = _flash_add_deferred_sum(p_odd, l_i)
        p_odd_packs = _anchor_p_packs(_pack_p_b(p_odd))
        sched_flash_pairs(6, trans=3, group=3)
        sched_flash_pairs(10, valu=5, group=3)
        dualwave_cluster_sync(5)

        # C6: prefetch K[even+2] into the consumed even slot and read V[odd].
        _cluster_power_stall(6)
        llvm.inline_asm(None, [], "s_nop 7", "", has_side_effects=True)
        rocdl.sched_barrier(0)
        if (kv_even + fx.Int32(2)) < _kv_hi:
            load_k(kv_even + fx.Int32(2), 0)
        v_odd = read_v_flash(1)
        s_even, m_i, l_i, o_accs, _p_odd, corr_scalar = (
            _flash_rescale_for_current(s_even, m_i, l_i, o_accs, p_odd)
        )
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(2 * _dma_ops_per_thread)
        dualwave_cluster_sync(6)

        # C7: PV[odd], lazy rebase from S[even], then begin P[even].
        _cluster_power_stall(7)
        if const_expr(_FLASH_SETPRIO_PV):
            rocdl.s_setprio(1)
        for dc in range_constexpr(_n_d_chunks):
            o_accs[dc] = _mfma_acc(
                v_odd[dc], p_odd_packs[0], o_accs[dc], _pv_mma_atom
            )
        tile_max_even = s_even[0]
        for i in range_constexpr(1, n_c):
            tile_max_even = _f32_max(tile_max_even, s_even[i])
        tile_max_even = _permlane32_reduce(tile_max_even, "max")
        sched_flash_pairs(4, valu=6, group=4)
        o_accs, m_i, l_i, p_odd_packs = _flash_c3_lazy_rescale(
            o_accs, m_i, l_i, tile_max_even, p_odd_packs
        )
        for pk in range_constexpr(1, _pv_packs):
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(
                    v_odd[pk * _n_d_chunks + dc],
                    p_odd_packs[pk],
                    o_accs[dc],
                    _pv_mma_atom,
                )
        p_mixed_even, p_head_even = _flash_make_deferred_p(s_even, m_i)
        sched_flash_pairs(6, valu=5, group=4)
        sched_flash_pairs(6, trans=3, group=4)
        _pin_in_cluster(p_head_even)
        if const_expr(_FLASH_SETPRIO_PV):
            rocdl.s_setprio(0)
        dualwave_cluster_sync(7)
        return m_i, l_i, o_accs, p_mixed_even, corr_scalar

    def _flash_deferred_epilogue(last_slot, m_i, l_i, o_accs, p_mixed, corr_pending):
        """Drain the final pending P/V tile."""
        p_last = _flash_finish_deferred_exp(p_mixed)
        if const_expr(not _FLASH_ELIDE_RESCALE and not _FLASH_C3_LAZY_RESCALE):
            o_accs = _scale_o_if_needed(o_accs, corr_pending)
        dualwave_cluster_sync(0)
        l_i = _flash_add_deferred_sum(p_last, l_i)
        dualwave_cluster_sync(1)
        v_last = read_v_flash(last_slot)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        dualwave_cluster_sync(2)
        pv_gemm_register_packs(p_last, v_last, o_accs)
        dualwave_cluster_sync(3)
        return m_i, l_i, o_accs

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
        pv_gemm_register_pair(out_sm_0[0], v_lo_regs_0, v_hi_regs_0, out_sm_0[3])
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
        pv_gemm_register_pair(out_sm_prev[0], v_lo_prev, v_hi_prev, out_sm_prev[3])
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
        pv_gemm_register_pair(out_sm_0[0], v_lo_regs_0, v_hi_regs_0, out_sm_0[3])
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
        pv_gemm_register_pair(out_sm_prev[0], v_lo_prev, v_hi_prev, out_sm_prev[3])
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
        pv_gemm_register_pair(out_sm_0[0], v_lo_regs_0, v_hi_regs_0, out_sm_0[3])
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

    if const_expr(_GEMM_ONLY_NO_SOFTMAX):
        # Numerics-off: no softmax. Persistent O. Flash physical LDS layout and
        # flash-style mem clusters (LDS K + DMA V / LDS V + DMA K).
        o_accs = _do_pair_gemm_only(_kv_lo, o_accs, guarded=True, extra_mem_barrier=True)

        _main_loop_count = _kv_pairs - fx.Int32(2)
        init_args = [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        loop_results = init_args
        for kv_mid, loop_args in range(
            fx.Int32(0),
            _main_loop_count,
            fx.Int32(1),
            init=init_args,
        ):
            o_accs = [loop_args[dc] for dc in range_constexpr(_n_d_chunks)]
            kv_even = _kv_lo + (_idx_to_i32(kv_mid) + fx.Int32(1)) * fx.Int32(2)
            o_accs = _do_pair_gemm_only(kv_even, o_accs, guarded=False, extra_mem_barrier=False)
            loop_results = yield [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        o_accs = [loop_results[dc] for dc in range_constexpr(_n_d_chunks)]

        _epilogue_count = (_kv_pairs > fx.Int32(1)).select(fx.Int32(1), fx.Int32(0))
        epi_init = [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        epi_results = epi_init
        for _epi_i, epi_args in range(
            fx.Int32(0),
            _epilogue_count,
            fx.Int32(1),
            init=epi_init,
        ):
            o_accs = [epi_args[dc] for dc in range_constexpr(_n_d_chunks)]
            kv_last = _kv_lo + (_kv_pairs - fx.Int32(1)) * fx.Int32(2)
            o_accs = _do_pair_gemm_only(kv_last, o_accs, guarded=True, extra_mem_barrier=True)
            epi_results = yield [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        o_accs = [epi_results[dc] for dc in range_constexpr(_n_d_chunks)]

        if const_expr(_enable_stagger):
            _stagger_extra_barrier_if_zero(infra.stagger_i32)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()
    elif const_expr(_flash_lds) and const_expr(_FLASH_DEFERRED_SOFTMAX):
        # Cross-tile pipeline. Each main step consumes PV[n-1] while preparing
        # scores/P[n], distributing softmax across both LDS and MFMA clusters.
        m_i, l_i, o_accs, p_mixed, corr_pending = _flash_deferred_prologue(
            _kv_lo, m_i, l_i, o_accs
        )
        if const_expr(param.long_seq_8c):
            # Flash keeps K two tiles ahead: K0/K1 are primed by the prologue,
            # and K2 is launched before C0 starts consuming K1.
            # For causal full-tile splitting, only prime K2 when an 8c pair
            # actually exists.  Otherwise the first 4c tail step will stage it.
            if (_kv_lo + fx.Int32(2)) < _kv_full_hi:
                load_k(_kv_lo + fx.Int32(2), 0)
                # Retire the prologue's K[lo+1] before the first C0 reads it,
                # leaving only this K[lo+2] in flight. Later trips are ordered
                # by the memory clusters' own closing waits. A shorter range
                # than this leaves _steady_pairs at 0, so no C0 read happens.
                if const_expr(_FLASH_8C_KREAD_PRELOOP_VMWAIT):
                    _waitcnt_vm_n(_dma_ops_per_thread)

        # Dense keeps the old all-8c range.  Bottom-right causal stops 8c at
        # _kv_full_hi, so C1/C5 never need a mask predicate.  The following
        # 4c loop consumes the diagonal band without draining p_mixed.
        _full_remaining = _kv_full_hi - (_kv_lo + fx.Int32(1))
        _full_remaining = _i32_max(_full_remaining, fx.Int32(0))
        _steady_pairs = _full_remaining // fx.Int32(2)
        _p_arg = _o + _n_d_chunks
        _corr_arg = _p_arg + 1
        init_args = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [p_mixed, corr_pending]
        )
        pair_results = init_args
        for pair_i, pair_args in range(
            fx.Int32(0),
            _steady_pairs,
            fx.Int32(1),
            init=init_args,
        ):
            m_i = [pair_args[r] for r in range_constexpr(npair)]
            l_i = [pair_args[npair + r] for r in range_constexpr(npair)]
            o_accs = [pair_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
            p_mixed = pair_args[_p_arg]
            corr_pending = pair_args[_corr_arg]
            kv_odd = _kv_lo + fx.Int32(1) + _idx_to_i32(pair_i) * fx.Int32(2)
            if const_expr(param.long_seq_8c):
                m_i, l_i, o_accs, p_mixed, corr_pending = (
                    _flash_deferred_step_8c(
                        kv_odd, m_i, l_i, o_accs, p_mixed
                    )
                )
            else:
                m_i, l_i, o_accs, p_mixed, corr_pending = _flash_deferred_step(
                    kv_odd, 1, 0, m_i, l_i, o_accs, p_mixed, corr_pending
                )
                m_i, l_i, o_accs, p_mixed, corr_pending = _flash_deferred_step(
                    kv_odd + fx.Int32(1), 0, 1, m_i, l_i, o_accs, p_mixed, corr_pending
                )
            pair_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
                + [p_mixed, corr_pending]
            )

        m_i = [pair_results[r] for r in range_constexpr(npair)]
        l_i = [pair_results[npair + r] for r in range_constexpr(npair)]
        o_accs = [pair_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        p_mixed = pair_results[_p_arg]
        corr_pending = pair_results[_corr_arg]

        # 8c pairs leave the pending even tile in slot 0 and the next K in
        # slot 1.  Consume the rest (an optional unpaired full tile followed by
        # the causal diagonal) as ordinary 4c pairs.  apply_mods cheaply skips
        # the optional full tile and masks each diagonal tile before softmax.
        _tail_start = _kv_lo + fx.Int32(1) + _steady_pairs * fx.Int32(2)
        _tail_remaining = _kv_hi - _tail_start
        _tail_pairs = _tail_remaining // fx.Int32(2)
        tail_pair_init = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [p_mixed, corr_pending]
        )
        tail_pair_results = tail_pair_init
        for tail_pair_i, tail_pair_args in range(
            fx.Int32(0),
            _tail_pairs,
            fx.Int32(1),
            init=tail_pair_init,
        ):
            m_i = [tail_pair_args[r] for r in range_constexpr(npair)]
            l_i = [tail_pair_args[npair + r] for r in range_constexpr(npair)]
            o_accs = [
                tail_pair_args[_o + dc] for dc in range_constexpr(_n_d_chunks)
            ]
            p_mixed = tail_pair_args[_p_arg]
            corr_pending = tail_pair_args[_corr_arg]
            kv_tail_even = _tail_start + _idx_to_i32(tail_pair_i) * fx.Int32(2)
            m_i, l_i, o_accs, p_mixed, corr_pending = _flash_deferred_step(
                kv_tail_even, 1, 0, m_i, l_i, o_accs, p_mixed, corr_pending
            )
            m_i, l_i, o_accs, p_mixed, corr_pending = _flash_deferred_step(
                kv_tail_even + fx.Int32(1),
                0,
                1,
                m_i,
                l_i,
                o_accs,
                p_mixed,
                corr_pending,
            )
            tail_pair_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
                + [p_mixed, corr_pending]
            )

        m_i = [tail_pair_results[r] for r in range_constexpr(npair)]
        l_i = [tail_pair_results[npair + r] for r in range_constexpr(npair)]
        o_accs = [
            tail_pair_results[_o + dc] for dc in range_constexpr(_n_d_chunks)
        ]
        p_mixed = tail_pair_results[_p_arg]
        corr_pending = tail_pair_results[_corr_arg]

        # An odd tail leaves one tile in slot 1. Process and drain it.
        _tail_count = _tail_remaining - _tail_pairs * fx.Int32(2)
        tail_init = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [p_mixed, corr_pending]
        )
        tail_results = tail_init
        for _tail_i, tail_args in range(
            fx.Int32(0),
            _tail_count,
            fx.Int32(1),
            init=tail_init,
        ):
            m_i = [tail_args[r] for r in range_constexpr(npair)]
            l_i = [tail_args[npair + r] for r in range_constexpr(npair)]
            o_accs = [tail_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
            p_mixed = tail_args[_p_arg]
            corr_pending = tail_args[_corr_arg]
            kv_tail = _tail_start + _tail_pairs * fx.Int32(2)
            m_i, l_i, o_accs, p_mixed, corr_pending = _flash_deferred_step(
                kv_tail, 1, 0, m_i, l_i, o_accs, p_mixed, corr_pending
            )
            m_i, l_i, o_accs = _flash_deferred_epilogue(
                1, m_i, l_i, o_accs, p_mixed, corr_pending
            )
            tail_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
                + [p_mixed, corr_pending]
            )

        m_i = [tail_results[r] for r in range_constexpr(npair)]
        l_i = [tail_results[npair + r] for r in range_constexpr(npair)]
        o_accs = [tail_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        p_mixed = tail_results[_p_arg]
        corr_pending = tail_results[_corr_arg]

        # No tail means the final pending tile is in slot 0 (including 1 tile).
        _no_tail_count = (_tail_count == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
        drain_init = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [p_mixed, corr_pending]
        )
        drain_results = drain_init
        for _drain_i, drain_args in range(
            fx.Int32(0),
            _no_tail_count,
            fx.Int32(1),
            init=drain_init,
        ):
            m_i = [drain_args[r] for r in range_constexpr(npair)]
            l_i = [drain_args[npair + r] for r in range_constexpr(npair)]
            o_accs = [drain_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
            p_mixed = drain_args[_p_arg]
            corr_pending = drain_args[_corr_arg]
            m_i, l_i, o_accs = _flash_deferred_epilogue(
                0, m_i, l_i, o_accs, p_mixed, corr_pending
            )
            drain_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
                + [p_mixed, corr_pending]
            )

        m_i = [drain_results[r] for r in range_constexpr(npair)]
        l_i = [drain_results[npair + r] for r in range_constexpr(npair)]
        o_accs = [drain_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]

        if const_expr(_enable_stagger):
            _stagger_extra_barrier_if_zero(infra.stagger_i32)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()
    elif const_expr(_flash_lds):
        # n64 split schedule with softmax overlapping QK (start) and PV (finish).
        m_i, l_i, o_accs = _do_pair_split_softmax(
            _kv_lo, m_i, l_i, o_accs, guarded=True, extra_mem_barrier=True
        )

        _main_loop_count = _kv_pairs - fx.Int32(2)
        init_args = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
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
            kv_even = _kv_lo + (_idx_to_i32(kv_mid) + fx.Int32(1)) * fx.Int32(2)
            m_i, l_i, o_accs = _do_pair_split_softmax(
                kv_even, m_i, l_i, o_accs, guarded=False, extra_mem_barrier=False
            )
            loop_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            )
        m_i = [loop_results[r] for r in range_constexpr(npair)]
        l_i = [loop_results[npair + r] for r in range_constexpr(npair)]
        o_accs = [loop_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]

        _epilogue_count = (_kv_pairs > fx.Int32(1)).select(fx.Int32(1), fx.Int32(0))
        epi_init = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
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
            kv_last = _kv_lo + (_kv_pairs - fx.Int32(1)) * fx.Int32(2)
            m_i, l_i, o_accs = _do_pair_split_softmax(
                kv_last, m_i, l_i, o_accs, guarded=True, extra_mem_barrier=True
            )
            epi_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            )
        m_i = [epi_results[r] for r in range_constexpr(npair)]
        l_i = [epi_results[npair + r] for r in range_constexpr(npair)]
        o_accs = [epi_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]

        if const_expr(_enable_stagger):
            _stagger_extra_barrier_if_zero(infra.stagger_i32)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()
    else:
        _sm_base = _o + _n_d_chunks
        _v_base = _sm_base + n_c + 1
        _mi_prev_base = _v_base + 2 * _n_d_chunks

        m_i, l_i, o_accs, s_scaled_prev, corr_scalar_prev, v_lo_prev, v_hi_prev, m_i_prev = (
            _do_tile_overlapping_softmax_prologue(_kv_lo, m_i, l_i, o_accs)
        )

        _main_loop_count = _kv_pairs - fx.Int32(2)
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

            kv_even = _kv_lo + (_idx_to_i32(kv_mid) + fx.Int32(1)) * fx.Int32(2)
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

        out_sm_final = softmax_finish(s_scaled_prev, m_i_prev, l_i, o_accs, corr_scalar_prev)
        pv_gemm_register_pair(out_sm_final[0], v_lo_prev, v_hi_prev, out_sm_final[3])
        l_i, o_accs = out_sm_final[2], out_sm_final[3]

        if const_expr(_enable_stagger):
            _stagger_extra_barrier_if_zero(infra.stagger_i32)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

    if const_expr(not _GEMM_ONLY_NO_SOFTMAX) and const_expr(not _NO_SOFTMAX_MATH):
        l0 = l_i[0]
        if const_expr(flex_mod.needs_safe_norm):
            l0 = _f32_max(l0, fx.Float32(1e-12))
        l0_ssa = l0.ir_value() if hasattr(l0, "ir_value") else l0
        inv_l = fx.Float32(rocdl.rcp(T.f32, l0_ssa))
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
                fx.Tensor(
                    fx.make_view(fx.recast_iter(fx.Float32, fx.get_iter(ws_o)), fx.make_layout(_MAX_BUFFER_BYTES, 1))
                ),
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
            + _idx_to_i32(h_idx) * _ws_o_head_stride
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
                fx.Tensor(
                    fx.make_view(fx.recast_iter(fx.Float32, fx.get_iter(ws_ml)), fx.make_layout(_MAX_BUFFER_BYTES, 1))
                ),
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
            + _idx_to_i32(h_idx) * _ws_ml_head_stride
            + (q_start + _qrow) * _ws_ml_row_stride
        )
        ml_vec = Vec.from_elements([m_i[0], l_i[0]], fx.Float32)
        fx.memref_store_vec(ml_vec, ws_ml_reg)
        fx.copy(_ws_ml_store_atom, ws_ml_reg, fx.slice(ws_ml_div, (None, fx.Int32(_ws_ml_base))))
    else:
        # Fuse the complementary 32-lane halves of each wave so every lane
        # issues eight contiguous 128-bit writes instead of sixteen 64-bit
        # writes.  The two halves own alternating 4-element D groups for the
        # same query row, so permlane32 can assemble each contiguous 8-tuple
        # without LDS or a workgroup synchronization.
        _o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), _out_elem_dtype)
        o_store_reg = fx.make_rmem_tensor(fx.make_layout(8, 1), _out_elem_dtype)
        o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(fx.recast_iter(_out_elem_dtype, fx.get_iter(o)), fx.make_layout(_MAX_BUFFER_BYTES, 1))
                ),
                max_size=True,
            ),
            fx.make_layout(1, 1),
        )
        o_base = o_off + _qrow * _o_row_stride
        _causal_row_active = (
            _idx_to_i32(q_start) + _qrow + _causal_q_shift
        ) >= fx.Int32(0)
        _lane_div_32 = fx.Index(local_tid // 32)
        for dc in range_constexpr(_n_d_chunks):
            o_vec = Vec(o_accs[dc])
            if const_expr(_bottom_right_causal):
                o_vec = Vec.from_elements(
                    [
                        _causal_row_active.select(o_vec[e], fx.Float32(0.0))
                        for e in range_constexpr(16)
                    ]
                )
            for g in range_constexpr(2):
                dwords = []
                for sg in range_constexpr(2):
                    k = 2 * g + sg
                    elems = [o_vec[k * 4 + e] for e in range_constexpr(4)]
                    packed = Vec.from_elements(elems, fx.Float32).to(_out_elem_dtype).bitcast(fx.Int32)
                    dwords.extend([packed[0], packed[1]])
                fused = _fused_o_128_dwords(
                    _lane_div_32,
                    dwords[0],
                    dwords[1],
                    dwords[2],
                    dwords[3],
                )
                vbf = Vec.from_elements([fx.Int32(w) for w in fused], fx.Int32).bitcast(_out_elem_dtype)
                col = dc * 32 + fx.Int32((2 * g) * 8) + fx.Int32(local_tid // 32) * fx.Int32(8)
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
    global_row = _idx_to_i32(bid) * fx.Int32(_rows_per_block) + fx.Int32(row_in_block)

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
        m_max = _f32_max(m_max, m_s)

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
    inv_den = fx.Float32(1.0) / _f32_max(den, fx.Float32(1e-12))
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
    _causal_grid = int(param.mask_type) in (MASK_CAUSAL, MASK_PREFIX_LM)
    if const_expr(_causal_grid):
        if const_expr(int(param.mask_type) == MASK_CAUSAL and not _paged):
            # Bottom-right causal has max(Sq-Skv, 0) fully-masked leading
            # query rows.  Drop every workgroup wholly contained in that
            # prefix.  Kernel-side reverse mapping still starts from the last
            # full-grid q_tile, so reducing grid.y naturally selects the live
            # tail without another kernel argument.
            _dead_q_rows = seqlen_q - seqlen_kv
            _dead_q_rows = (_dead_q_rows > fx.Int32(0)).select(
                _dead_q_rows, fx.Int32(0)
            )
            _first_live_q_tile = _dead_q_rows // fx.Int32(rows_per_wg)
            _active_q_tiles = num_q_tiles - _first_live_q_tile
            _grid = (hq, _active_q_tiles, grid_z)
        else:
            _grid = (hq, num_q_tiles, grid_z)
    else:
        # Dense (and other non-causal masks): same head-fast mapping as causal
        # so Q tiles of one head share an XCD and replay KV from its L2.
        _grid = (hq, num_q_tiles, grid_z)

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
        grid=_grid,
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
    long_seq_8c: Optional[bool] = None,
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
        long_seq_8c=long_seq_8c,
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
