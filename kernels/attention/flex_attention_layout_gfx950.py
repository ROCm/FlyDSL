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

from dataclasses import dataclass
from typing import Callable, Optional
import warnings
import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, arith
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.runtime.device import get_rocm_arch

from kernels.attention.pipeline import (
    InfraContext,
    PipelineScheduler,
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


@fx.struct
class FlexAttnParam:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    head_dim: fx.Constexpr[int]
    num_heads_q: fx.Constexpr[int]
    num_heads_kv: fx.Constexpr[int]
    causal: fx.Constexpr[bool]
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


def make_flex_attn_param(
    seqlen_kv: int,
    dtype_id: int = FLEX_DTYPE_BF16,
    block_m: int = 32,
    block_n: int = 32,
    head_dim: int = 128,
    num_heads_q: int = 8,
    num_heads_kv: int = 8,
    causal: bool = False,
    m_waves: int = 2,
    n_waves: int = 1,
    num_groups: int = 1,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
    pipe_depth: int = 1,
    pipe_stages: int = 1,
) -> FlexAttnParam:
    if dtype_id not in (FLEX_DTYPE_BF16, FLEX_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m <= 0 or block_n <= 0 or head_dim <= 0:
        raise ValueError("block_m, block_n, head_dim must be positive")
    # Phase 0 proven config (see plan progress log): fx.gemm with MFMA 16x16x16 has
    # a lowering bug on this build, so both GEMMs use mma_k=32 -> block_n multiple of
    # 32. The C-fragment slot->row map is only locked for block_m=32 (2 M-waves x
    # mma_m=16), one N-wave; larger block_m needs a per-slot row map (TODO perf).
    if (mma_m, mma_n, mma_k) != (16, 16, 32):
        raise ValueError("Phase 0 requires mma=16x16x32 (16x16x16 fx.gemm lowering bug)")
    if block_m % (m_waves * mma_m) != 0:
        raise ValueError(f"block_m ({block_m}) must be divisible by m_waves*mma_m ({m_waves * mma_m})")
    if block_n % (n_waves * mma_n) != 0:
        raise ValueError(f"block_n ({block_n}) must be divisible by n_waves*mma_n ({n_waves * mma_n})")
    if n_waves != 1:
        raise ValueError("n_waves must be 1 (softmax row reduction requires all N-lanes in one wave)")
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

    return FlexAttnParam(
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        causal=causal,
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
    )


def make_flex_attn_kernel_name(param: FlexAttnParam) -> str:
    dtype_str = "fp16" if param.dtype_id == FLEX_DTYPE_FP16 else "bf16"
    name = f"flex_attn_{dtype_str}_m{param.block_m}n{param.block_n}d{param.head_dim}"
    name += f"_w{param.m_waves}x{param.n_waves}g{param.num_groups}"
    name += "_causal" if param.causal else "_dense"
    name += f"_pd{param.pipe_depth}"
    if pipeline_stagger_enabled(
        depth=int(param.pipe_depth),
        num_groups=int(param.num_groups),
        m_waves=int(param.m_waves),
    ):
        name += "_stg"
    return name


def summarize_flex_wg_thread_layout(param: FlexAttnParam) -> str:
    """Thread / wave layout inside one HIP workgroup (constexpr from ``FlexAttnParam``)."""
    waves_per_group = int(param.m_waves) * int(param.n_waves)
    waves_per_wg = int(param.num_groups) * waves_per_group
    rows_per_wg = int(param.block_m) * int(param.num_groups)
    lds_ring = max(2, int(param.pipe_depth))
    stagger = pipeline_stagger_enabled(
        depth=int(param.pipe_depth),
        num_groups=int(param.num_groups),
        m_waves=int(param.m_waves),
    )
    return "\n".join([
        "flex_attn workgroup layout:",
        f"  block_threads={param.block_threads} "
        f"({param.num_groups} query groups × {param.group_threads} threads/group)",
        f"  waves_per_group={waves_per_group} "
        f"(m_waves={param.m_waves} × n_waves={param.n_waves}, lane_width={GFX950_WAVE_SIZE})",
        f"  waves_per_workgroup={waves_per_wg} (= num_groups × waves_per_group)",
        f"  Q rows: {param.block_m} per group, {rows_per_wg} per workgroup (Strategy A shared KV loop)",
        f"  KV iterations per WG: n_kv_tiles={param.n_kv_tiles} (seqlen_kv/block_n at compile time)",
        f"  LDS K/V ring slots={lds_ring} (pipe_depth={param.pipe_depth})",
        f"  pipeline_stagger (dual-wave phase shift within each group)={stagger}",
        "  tid → group=tid//group_threads, local_tid=tid%group_threads; "
        "stagger_i32=local_tid//64 when stagger enabled",
        "  All groups in a WG share K/V LDS and the same kv loop; groups differ in Q/O rows only",
        "  Grid blocks are mapped to CUs by the HIP scheduler (no fixed tiles/CU in kernel code)",
        f"  Launch grid: (num_q_tiles=ceil(seqlen_q/{rows_per_wg}), heads={param.num_heads_q}, batch)",
    ])


def summarize_flex_attn_launch(
    param: FlexAttnParam,
    *,
    seqlen_q: Optional[int] = None,
    seqlen_kv: Optional[int] = None,
    batch: Optional[int] = None,
) -> str:
    """Human-readable launch config: ``FlexAttnParam`` plus pipeline emit notes."""
    stagger = pipeline_stagger_enabled(
        depth=int(param.pipe_depth),
        num_groups=int(param.num_groups),
        m_waves=int(param.m_waves),
    )
    dtype_str = "fp16" if int(param.dtype_id) == FLEX_DTYPE_FP16 else "bf16"
    lines = [
        f"flex_attn launch [{make_flex_attn_kernel_name(param)}]",
        f"  dtype={dtype_str} block_m={param.block_m} block_n={param.block_n} head_dim={param.head_dim}",
        f"  m_waves={param.m_waves} n_waves={param.n_waves} num_groups={param.num_groups} "
        f"group_threads={param.group_threads} block_threads={param.block_threads}",
        f"  mma={param.mma_m}x{param.mma_n}x{param.mma_k} gqa_group={param.gqa_group} "
        f"n_kv_tiles={param.n_kv_tiles}",
        f"  pipe_depth={param.pipe_depth} pipe_stages={param.pipe_stages} "
        f"pipeline_stagger={stagger} causal={param.causal}",
        summarize_flex_wg_thread_layout(param),
    ]
    rows_per_wg = int(param.block_m) * int(param.num_groups)
    if seqlen_q is not None and seqlen_kv is not None:
        num_q_tiles = (int(seqlen_q) + rows_per_wg - 1) // rows_per_wg
        lines.append(
            f"  seqlen_q={seqlen_q} seqlen_kv={seqlen_kv} rows_per_wg={rows_per_wg} "
            f"num_q_tiles={num_q_tiles}"
        )
        if batch is not None:
            total_wgs = num_q_tiles * int(param.num_heads_q) * int(batch)
            lines.append(
                f"  grid=(x={num_q_tiles}, y={param.num_heads_q}, z={batch}) "
                f"total_workgroups={total_wgs}"
            )
    if int(param.pipe_depth) >= 2:
        lines.append("  PipelineScheduler: emit_prologue + emit_main_loop + emit_epilogue")
        lines.append(
            f"    _LayoutSchedTraits: HEAD_DIM={param.head_dim} "
            f"LGKMCNT_0_ONLY=0x{LGKMCNT_0_ONLY:X} "
            f"SCHED_MFMA=0x008 SCHED_VALU=0x002 SCHED_EXP=0x400"
        )
    return "\n".join(lines)


def print_flex_attn_launch_summary(
    param: FlexAttnParam,
    *,
    seqlen_q: Optional[int] = None,
    seqlen_kv: Optional[int] = None,
    batch: Optional[int] = None,
) -> None:
    print(summarize_flex_attn_launch(param, seqlen_q=seqlen_q, seqlen_kv=seqlen_kv, batch=batch), flush=True)


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


def _row_reduce(x, mode):
    # Per-query-row reduction over the key dimension: the 16 keys of an N-wave are
    # spread across the low 4 lane bits, so a shuffle_xor butterfly over offsets
    # 8/4/2/1 combines them. block_n stays within one N-wave (n_waves_n == 1).
    w = x
    for off in (8, 4, 2, 1):
        peer = w.shuffle_xor(off, GFX950_WAVE_SIZE)
        w = w.maximumf(peer) if mode == "max" else w.addf(peer, fastmath=_FM)
    return w


def _scale_sched_pairs(pairs, head_dim):
    return max(1, (pairs + 1) // 2) if head_dim == 64 else pairs


def _sched_barrier_pairs(traits, pairs, valu_cnt, group):
    """Emit ``pairs`` × {1 MFMA + valu_cnt VALU} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_VALU_MASK, valu_cnt, group)


def _sched_barrier_exp_pairs(traits, pairs, exp_cnt, group):
    """Emit ``pairs`` × {1 MFMA + exp_cnt EXP} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_EXP_MASK, exp_cnt, group)



LGKMCNT_0_ONLY = 0xC07F

# rocprof PC sampling phase tags (s_nop 1..7 in pd2 loop). Set FLEX_ATTN_PC_MARKERS=1 before JIT.
_FLEX_ATTN_PC_MARKERS = os.environ.get("FLEX_ATTN_PC_MARKERS", "0") == "1"


def _flex_pc_marker_m0_load() -> None:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    llvm.InlineAsmOp(ir.Type.parse("!llvm.void"), [], "s_nop 1", "", has_side_effects=True)


def _flex_pc_marker_m1_read() -> None:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    llvm.InlineAsmOp(ir.Type.parse("!llvm.void"), [], "s_nop 2", "", has_side_effects=True)


def _flex_pc_marker_c1_qk() -> None:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    llvm.InlineAsmOp(ir.Type.parse("!llvm.void"), [], "s_nop 3", "", has_side_effects=True)


def _flex_pc_marker_c2_softmax() -> None:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    llvm.InlineAsmOp(ir.Type.parse("!llvm.void"), [], "s_nop 4", "", has_side_effects=True)


def _flex_pc_marker_m2_pwrite() -> None:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    llvm.InlineAsmOp(ir.Type.parse("!llvm.void"), [], "s_nop 5", "", has_side_effects=True)


def _flex_pc_marker_c3_pv() -> None:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    llvm.InlineAsmOp(ir.Type.parse("!llvm.void"), [], "s_nop 6", "", has_side_effects=True)


def _flex_pc_marker_tail_sync() -> None:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm

    llvm.InlineAsmOp(ir.Type.parse("!llvm.void"), [], "s_nop 7", "", has_side_effects=True)


@flyc.kernel
def flex_attn_fwd_gfx950_kernel(
    o: fx.Tensor,       # [B, Sq, Hq, D]
    q: fx.Tensor,       # [B, Sq, Hq, D]
    k: fx.Tensor,       # [B, Skv, Hkv, D]
    v: fx.Tensor,       # [B, Skv, Hkv, D]
    seqlen_q: fx.Int32,
    seqlen_kv: fx.Int32,
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
    n_kv_tiles = (seqlen_kv + block_n - 1) // block_n

    n_kv_tiles = param.n_kv_tiles  # compile-time: seqlen_kv // block_n (validated on host)
    mma_k = param.mma_k

    # ── LDS: K/V staging (shared across all groups) + per-group P bridge ──────
    kv_tile_elems = block_n * head_dim  # elements in one K or V tile
    _lds_ring_slots = max(2, int(param.pipe_depth))
    _lds_kv_elems = _lds_ring_slots * kv_tile_elems

    @fx.struct
    class SharedStorage:
        k_lds: fx.Array[elem_dtype, _lds_kv_elems, 16]
        v_lds: fx.Array[elem_dtype, _lds_kv_elems, 16]
        p: fx.Array[elem_dtype, num_groups * block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    sK_ptr = storage.k_lds.peek().ptr
    sV_ptr = storage.v_lds.peek().ptr

    # K LDS: [block_n, head_dim] row-major (head_dim contiguous).
    sK = [
        fx.make_view(sK_ptr + i * kv_tile_elems, fx.make_layout((block_n, head_dim), (head_dim, 1)))
        for i in range_constexpr(_lds_ring_slots)
    ]
    # V LDS: [head_dim, block_n] row-major (block_n contiguous) — V is host-transposed.
    sV = [
        fx.make_view(sV_ptr + i * kv_tile_elems, fx.make_layout((head_dim, block_n), (block_n, 1)))
        for i in range_constexpr(_lds_ring_slots)
    ]
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
    # Vt is [B, Hkv, D, Skv] (host-transposed per head): element (b,h,d,s) at
    # b*Hkv*D*Skv + h*D*Skv + d*Skv + s.  This head's [D, Skv] slice:
    v_off = b_idx * hkv * head_dim * seqlen_kv + kv_head * head_dim * seqlen_kv

    # Per-(batch,head) sub-views wrapped as BUFFER pointers so BufferCopy128b
    # legalizes (a plain make_view over a global ptr is not buffer-backed).
    q_it = _make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(q)) + fx.Int32(q_off))
    o_it = _make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(o)) + fx.Int32(o_off))
    k_it = _make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(k)) + fx.Int32(k_off))
    vt_it = _make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(v)) + fx.Int32(v_off))

    gQ = fx.make_view(q_it, fx.make_layout((block_m, head_dim), (hq * head_dim, 1)))
    gO = fx.make_view(o_it, fx.make_layout((block_m, head_dim), (hq * head_dim, 1)))
    gK = fx.make_view(k_it, fx.make_layout((seqlen_kv, head_dim), (hkv * head_dim, 1)))
    gV = fx.make_view(vt_it, fx.make_layout((head_dim, seqlen_kv), (seqlen_kv, 1)))

    # Each group runs the validated 128-thread MMA partition via local_tid.
    thr_qk = tiled_mma_qk.thr_slice(local_tid)
    thr_pv = tiled_mma_pv.thr_slice(local_tid)

    ca = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    uca = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

    # Q resident: load once into the GEMM1 A-fragment (reused every KV tile).
    tcA = fx.make_tiled_copy_A(ca, tiled_mma_qk).get_slice(local_tid)
    frag_Q = thr_qk.make_fragment_A(gQ)
    fx.copy(ca, tcA.partition_S(gQ), tcA.retile(frag_Q))

    # Persistent O accumulator [block_m, head_dim] in registers across the KV loop.
    frag_O = thr_pv.make_fragment_C(gO)
    frag_O.fill(0.0)

    # Per-slot row map: thr_qk.partition_C partitions by THIS thread's wave, so
    # n_c is always this lane's slot count (not the full tile). For MFMA 16x16
    # with n_waves=1, each lane has 4 M-values × (block_n/mma_n) N-repeats slots.
    # The first half and second half are the two column-groups of the same rows,
    # so npair = n_c // 2 gives the number of distinct row-indices this lane owns,
    # and i % npair maps each slot to its row. This holds for any m_waves because
    # thr_slice already selects the per-wave partition.
    n_c = _size_scalar(thr_qk.partition_C(sP).shape)
    npair = n_c // 2
    _s_slot_to_row_idx = [i % npair for i in range(n_c)]

    _M_NEG_FLOOR = -60.0  # large negative, finite; avoids -inf * scale = NaN under fastmath
    m_i = [fx.Float32(_M_NEG_FLOOR) for _ in range_constexpr(npair)]
    l_i = [fx.Float32(0.0) for _ in range_constexpr(npair)]

    n_o = _size_scalar(frag_O.shape)
    o_slot_row = [i % npair for i in range(n_o)]

    # Fold the softmax scale into the exp2 multiplier: keep frag_S in raw (unscaled)
    # units and multiply by scale*log2e inside every exp2. softmax is invariant to a
    # positive scalar applied before the exp, so this is equivalent (no per-tile
    # scale loop). m_i/tile_m therefore track raw scores.
    scale_log2e = scale * fx.Float32(_LOG2E)

    # ── KV-loop via stage closures ──────────────────────────────────────
    # Each closure below is one pipeline stage, containing the actual
    # layout-API code extracted from the original monolithic loop body.
    # The closures capture all kernel-local objects (copy atoms, fragments,
    # LDS views, DMA descriptors) so the pipeline module never needs them.

    # Copy builders and fragments for double-buffered K/V
    tcA2 = fx.make_tiled_copy_A(uca, tiled_mma_pv).get_slice(local_tid)
    tcB_lds = [fx.make_tiled_copy_B(uca, tiled_mma_qk).get_slice(local_tid) for _ in range_constexpr(_lds_ring_slots)]
    tcB2_lds = [fx.make_tiled_copy_B(uca, tiled_mma_pv).get_slice(local_tid) for _ in range_constexpr(_lds_ring_slots)]
    frag_K = [thr_qk.make_fragment_B(sK[i]) for i in range_constexpr(_lds_ring_slots)]
    frag_V = [thr_pv.make_fragment_B(sV[i]) for i in range_constexpr(_lds_ring_slots)]

    # DMA infrastructure
    block_threads = param.block_threads
    _dma_bytes = GFX950_DMA_BYTES
    _kv_tile_bytes = kv_tile_elems * param.in_data_bytes
    _dma_ops_per_thread = _kv_tile_bytes // (block_threads * _dma_bytes)
    dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    _k_row_stride_bytes = hkv * head_dim * param.in_data_bytes
    _k_row_bytes = head_dim * param.in_data_bytes
    _v_row_stride_bytes = seqlen_kv * fx.Int32(param.in_data_bytes)
    _v_row_bytes = block_n * param.in_data_bytes
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
    sK_i8 = [fx.recast_iter(fx.Int8, sK_ptr) + i * fx.Int32(kv_tile_elems * param.in_data_bytes)
             for i in range_constexpr(_lds_ring_slots)]
    sV_i8 = [fx.recast_iter(fx.Int8, sV_ptr) + i * fx.Int32(kv_tile_elems * param.in_data_bytes)
             for i in range_constexpr(_lds_ring_slots)]
    _step_bytes = block_threads * _dma_bytes

    # P-bridge: write via QK thread's partition of the P LDS tile.
    thr_p_lds = thr_qk.partition_C(sP)

    # Row-slot map for softmax (compile-time)
    _row_slots = [[s for s in range(n_c) if _s_slot_to_row_idx[s] == r] for r in range(npair)]

    print(summarize_flex_wg_thread_layout(param), flush=True)

    # ── Stage: DMA K+V global → LDS ─────────────────────────────────────
    def _stage_kv_to_lds(kv_idx, buf):
        wave_off = rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes))
        k_global_base = k_off * param.in_data_bytes + kv_idx * block_n * _k_row_stride_bytes
        lds_k = fx.add_offset(sK_i8[buf], wave_off)
        for i in range_constexpr(_dma_ops_per_thread):
            if const_expr(i > 0):
                lds_k = fx.add_offset(lds_k, _step_bytes)
            flat_byte = i * block_threads * _dma_bytes + tid * _dma_bytes
            tile_row = flat_byte // _k_row_bytes
            tile_col_byte = flat_byte % _k_row_bytes
            gmem_byte = k_global_base + tile_row * _k_row_stride_bytes + tile_col_byte
            fx.copy(dma_atom, fx.slice(k_div, (None, fx.Int32(gmem_byte))),
                    fx.make_view(lds_k, fx.make_layout(1, 1)))
        v_global_base = v_off * param.in_data_bytes + kv_idx * block_n * param.in_data_bytes
        lds_v = fx.add_offset(sV_i8[buf], wave_off)
        for i in range_constexpr(_dma_ops_per_thread):
            if const_expr(i > 0):
                lds_v = fx.add_offset(lds_v, _step_bytes)
            flat_byte = i * block_threads * _dma_bytes + tid * _dma_bytes
            tile_row = flat_byte // _v_row_bytes
            tile_col_byte = flat_byte % _v_row_bytes
            gmem_byte = fx.Int32(v_global_base) + tile_row * _v_row_stride_bytes + fx.Int32(tile_col_byte)
            fx.copy(dma_atom, fx.slice(v_div, (None, fx.Int32(gmem_byte))),
                    fx.make_view(lds_v, fx.make_layout(1, 1)))

    def _stage_k_to_lds(kv_idx, buf):
        wave_off = rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes))
        k_global_base = k_off * param.in_data_bytes + kv_idx * block_n * _k_row_stride_bytes
        lds_k = fx.add_offset(sK_i8[buf], wave_off)
        for i in range_constexpr(_dma_ops_per_thread):
            if const_expr(i > 0):
                lds_k = fx.add_offset(lds_k, _step_bytes)
            flat_byte = i * block_threads * _dma_bytes + tid * _dma_bytes
            tile_row = flat_byte // _k_row_bytes
            tile_col_byte = flat_byte % _k_row_bytes
            gmem_byte = k_global_base + tile_row * _k_row_stride_bytes + tile_col_byte
            fx.copy(dma_atom, fx.slice(k_div, (None, fx.Int32(gmem_byte))),
                    fx.make_view(lds_k, fx.make_layout(1, 1)))

    def _stage_v_to_lds(kv_idx, buf):
        wave_off = rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes))
        v_global_base = v_off * param.in_data_bytes + kv_idx * block_n * param.in_data_bytes
        lds_v = fx.add_offset(sV_i8[buf], wave_off)
        for i in range_constexpr(_dma_ops_per_thread):
            if const_expr(i > 0):
                lds_v = fx.add_offset(lds_v, _step_bytes)
            flat_byte = i * block_threads * _dma_bytes + tid * _dma_bytes
            tile_row = flat_byte // _v_row_bytes
            tile_col_byte = flat_byte % _v_row_bytes
            gmem_byte = fx.Int32(v_global_base) + tile_row * _v_row_stride_bytes + fx.Int32(tile_col_byte)
            fx.copy(dma_atom, fx.slice(v_div, (None, fx.Int32(gmem_byte))),
                    fx.make_view(lds_v, fx.make_layout(1, 1)))

    # ── Pipeline: wire stages ───────────────────────────────────────────
    from kernels.attention.pipeline import (
        PipelineStage, PipelineScheduler, Wire, InfraContext,
        StageKind, ResourceDecl, SubStage,
        pipeline_stagger_enabled,
        _rocdl_waitcnt_vm_n,
        _rocdl_waitcnt_vmcnt0,
    )

    class _LayoutSchedTraits:
        """Minimal traits for flash-style sched_group_barrier emitters."""
        HEAD_DIM = int(param.head_dim)
        SCHED_MFMA_MASK = 0x008
        SCHED_VALU_MASK = 0x002
        SCHED_EXP_MASK = 0x400
        LGKMCNT_0_ONLY = LGKMCNT_0_ONLY

    # IGroupLP groups mirror flash dualwave C1/C3 compute clusters (not cluster.index).
    _SCHED_GROUP_C1 = 1
    _SCHED_GROUP_C3 = 2
    _pd_for_sched = int(param.pipe_depth)

    def _sched_after_softmax_finish(infra, _cluster_index, _entry_index):
        t = infra.traits
        _sched_barrier_exp_pairs(t, 6, 3, _SCHED_GROUP_C1)
        _sched_barrier_pairs(t, 10, 5, _SCHED_GROUP_C1)

    def _sched_after_softmax_start(infra, _cluster_index, _entry_index):
        t = infra.traits
        _sched_barrier_pairs(t, 4, 6, _SCHED_GROUP_C3)
        _sched_barrier_pairs(t, 6, 6, _SCHED_GROUP_C3)
        _sched_barrier_exp_pairs(t, 6, 3, _SCHED_GROUP_C3)

    def _sched_after_gemm2_pv(infra, _cluster_index, _entry_index):
        t = infra.traits
        _sched_barrier_pairs(t, 4, 6, _SCHED_GROUP_C3)
        _sched_barrier_pairs(t, 6, 6, _SCHED_GROUP_C3)

    def _flex_wait_lgkm_after_lds_write():
        """Drain LDS after ds writes (P bridge / Gemm2WriteP) without a full WG barrier."""
        if const_expr(_pd_for_sched >= 2):
            rocdl.s_waitcnt(LGKMCNT_0_ONLY)

    # ── Stage bodies: explicit parameters, list returns (no shared_regs dict) ──

    def load_k(tile_idx, slot):
        _stage_k_to_lds(tile_idx, slot)
        return []

    def load_v(tile_idx, slot):
        _stage_v_to_lds(tile_idx, slot)
        return []

    def load_kv(tile_idx, slot):
        _stage_kv_to_lds(tile_idx, slot)
        return []

    def read_k(slot):
        fx.copy(uca, tcB_lds[slot].partition_S(sK[slot]), tcB_lds[slot].retile(frag_K[slot]))
        return [frag_K[slot]]

    def read_v(slot):
        fx.copy(uca, tcB2_lds[slot].partition_S(sV[slot]), tcB2_lds[slot].retile(frag_V[slot]))
        return [frag_V[slot]]

    def read_kv(slot):
        read_k(slot)
        read_v(slot)
        return [frag_K[slot], frag_V[slot]]

    def read_kv_work(slot):
        # pd2 scf.for: dynamic ring slot, fixed register tiles (avoid fragment SSA merge).
        fx.copy(uca, tcB_lds[slot].partition_S(sK[slot]), tcB_lds[slot].retile(frag_K[0]))
        fx.copy(uca, tcB2_lds[slot].partition_S(sV[slot]), tcB2_lds[slot].retile(frag_V[0]))
        return []

    def gemm1_qk(frag_Q_in, frag_K_in):
        frag_S_out = thr_qk.make_fragment_C(sP)
        frag_S_out.fill(0.0)
        fx.gemm(tiled_mma_qk, frag_S_out, frag_Q_in, frag_K_in, frag_S_out)
        return [frag_S_out]

    def softmax_full(frag_S_in, m_i_in, l_i_in, frag_O_in):
        corr = [fx.Float32(1.0) for _ in range_constexpr(npair)]
        for r in range_constexpr(npair):
            slots = _row_slots[r]
            row_max = frag_S_in[slots[0]]
            for si in range_constexpr(1, len(slots)):
                row_max = row_max.maximumf(frag_S_in[slots[si]])
            tile_m = _row_reduce(row_max, "max")
            m_new = m_i_in[r].maximumf(tile_m)
            corr[r] = fmath.exp2((m_i_in[r] - m_new) * scale_log2e, fastmath=_FM)
            for si in range_constexpr(len(slots)):
                frag_S_in[slots[si]] = fmath.exp2(
                    (frag_S_in[slots[si]] - m_new) * scale_log2e, fastmath=_FM
                )
            row_sum = frag_S_in[slots[0]]
            for si in range_constexpr(1, len(slots)):
                row_sum = row_sum.addf(frag_S_in[slots[si]], fastmath=_FM)
            tile_l = _row_reduce(row_sum, "sum")
            l_i_in[r] = l_i_in[r] * corr[r] + tile_l
            m_i_in[r] = m_new
        for i in range_constexpr(n_o):
            frag_O_in[i] = frag_O_in[i] * corr[o_slot_row[i]]
        return [frag_S_in, m_i_in, l_i_in, frag_O_in, corr]

    def softmax_finish(v_p_partial_in, l_i_in):
        for r in range_constexpr(npair):
            slots = _row_slots[r]
            row_sum = v_p_partial_in[slots[0]]
            for si in range_constexpr(1, len(slots)):
                row_sum = row_sum.addf(v_p_partial_in[slots[si]], fastmath=_FM)
            tile_l = _row_reduce(row_sum, "sum")
            l_i_in[r] = l_i_in[r] + tile_l
        return [l_i_in]

    def softmax_start(frag_S_in, m_i_in, l_i_in, v_p_partial_in, p_pv_snap_in, frag_O_in):
        for i in range_constexpr(n_c):
            p_pv_snap_in[i] = v_p_partial_in[i]
        corr = [fx.Float32(1.0) for _ in range_constexpr(npair)]
        for r in range_constexpr(npair):
            slots = _row_slots[r]
            row_max = frag_S_in[slots[0]]
            for si in range_constexpr(1, len(slots)):
                row_max = row_max.maximumf(frag_S_in[slots[si]])
            tile_m = _row_reduce(row_max, "max")
            m_new = m_i_in[r].maximumf(tile_m)
            corr[r] = fmath.exp2((m_i_in[r] - m_new) * scale_log2e, fastmath=_FM)
            l_i_in[r] = l_i_in[r] * corr[r]
            for si in range_constexpr(len(slots)):
                v_p_partial_in[slots[si]] = v_p_partial_in[slots[si]] * corr[r]
            for si in range_constexpr(len(slots)):
                frag_S_in[slots[si]] = fmath.exp2(
                    (frag_S_in[slots[si]] - m_new) * scale_log2e, fastmath=_FM
                )
            m_i_in[r] = m_new
        for i in range_constexpr(n_o):
            frag_O_in[i] = frag_O_in[i] * corr[o_slot_row[i]]
        return [v_p_partial_in, frag_S_in, m_i_in, l_i_in, frag_O_in, corr]

    def bridge_p_write_from_v_partial(v_p_partial_in):
        for i in range_constexpr(n_c):
            thr_p_lds[i] = _to_elem(v_p_partial_in[i], elem_dtype)
        return []

    def bridge_p_write_from_frag_p(frag_P_in):
        for i in range_constexpr(n_c):
            thr_p_lds[i] = _to_elem(frag_P_in[i], elem_dtype)
        return []

    def bridge_p_load():
        _flex_wait_lgkm_after_lds_write()
        frag_P_a_out = thr_pv.make_fragment_A(sP)
        fx.copy(uca, tcA2.partition_S(sP), tcA2.retile(frag_P_a_out))
        return [frag_P_a_out]

    def gemm2_write_p(v_p_for_gemm_in, softmax_corr_in):
        for i in range_constexpr(n_c):
            pi = _to_elem(v_p_for_gemm_in[i], fx.Float32) * softmax_corr_in[_s_slot_to_row_idx[i]]
            thr_p_lds[i] = _to_elem(pi, elem_dtype)
        return []

    def gemm2_pv(frag_Vt_in, frag_Vt_next_in, frag_O_in):
        _flex_wait_lgkm_after_lds_write()
        frag_P_a_local = thr_pv.make_fragment_A(sP)
        fx.copy(uca, tcA2.partition_S(sP), tcA2.retile(frag_P_a_local))
        fx.gemm(tiled_mma_pv, frag_O_in, frag_P_a_local, frag_Vt_in, frag_O_in)
        return [frag_O_in, frag_Vt_next_in]

    def gemm2_pd1(frag_P_a_in, frag_Vt_next_in, frag_O_in):
        fx.gemm(tiled_mma_pv, frag_O_in, frag_P_a_in, frag_Vt_next_in, frag_O_in)
        return [frag_O_in]

    def dualwave_cluster_sync(cluster_index):
        if const_expr(cluster_index == 1 and _pd_for_sched >= 2):
            rocdl.sched_barrier(0)
        else:
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

    def mem_cluster_exit(vm_at_most):
        rocdl.s_waitcnt(LGKMCNT_0_ONLY)
        if const_expr(vm_at_most == 0):
            _rocdl_waitcnt_vmcnt0()
        else:
            _rocdl_waitcnt_vm_n(vm_at_most)

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

    # ── Pipeline adapters (infra + dict) for PipelineScheduler wiring only ──

    def _pipeline_load_k(infra, **_):
        load_k(infra.tile_idx, infra.buf_slot)
        return {}

    def _pipeline_load_v(infra, **_):
        load_v(infra.tile_idx, infra.buf_slot)
        return {}

    def _pipeline_read_k(infra, **_):
        out = read_k(infra.buf_slot)
        return {"frag_K": out[0]}

    def _pipeline_read_v(infra, **_):
        out = read_v(infra.buf_slot)
        return {"frag_Vt_next": out[0]}

    def _pipeline_bridge_p_write(infra, **regs):
        if const_expr(param.pipe_depth >= 2):
            bridge_p_write_from_v_partial(regs["v_p_partial"])
        else:
            bridge_p_write_from_frag_p(regs["frag_P"])
        return {}

    def _pipeline_bridge_p_load(infra, **_):
        out = bridge_p_load()
        return {"frag_P_a": out[0]}

    def _pipeline_gemm2_write_p(infra, **regs):
        gemm2_write_p(regs["frag_P"], regs["softmax_corr"])
        return {}

    def _pipeline_gemm2_pv(infra, **regs):
        out = gemm2_pv(regs["frag_Vt"], regs["frag_Vt_next"], regs["frag_O"])
        return {"frag_O": out[0], "frag_Vt": out[1]}

    def _pipeline_gemm2_pd1(infra, *, frag_P_a, frag_Vt_next, frag_O, **_):
        out = gemm2_pd1(frag_P_a, frag_Vt_next, frag_O)
        return {"frag_O": out[0]}

    def _pipeline_bridge_p_epilogue(infra, **shared_regs):
        _pipeline_bridge_p_write(infra, **shared_regs)
        return _pipeline_bridge_p_load(infra)

    # ── LoadKV: DMA K+V global → LDS. min_slots=1, max_slots=2. ─────────
    class _StageLoadKV(PipelineStage):
        name = "LoadKV"
        kind = StageKind.MEMORY

        @property
        def resources(self):
            return ResourceDecl(dma_count=_dma_ops_per_thread * 2)

        @property
        def min_slots(self):
            return 1

        @property
        def max_slots(self):
            return 2

        def execute(self, infra, **_):
            load_kv(infra.tile_idx, infra.buf_slot)
            return {}

        def decompose(self, allocated_slots):
            if allocated_slots == 1:
                return [
                    SubStage(
                        "LoadKV",
                        self.execute,
                        ResourceDecl(dma_count=_dma_ops_per_thread * 2),
                    ),
                ]
            return [
                SubStage("LoadK", _pipeline_load_k, ResourceDecl(dma_count=_dma_ops_per_thread)),
                SubStage(
                    "LoadV",
                    _pipeline_load_v,
                    ResourceDecl(dma_count=_dma_ops_per_thread),
                    prefetch_only=True,
                ),
            ]

    # ── ReadKV: LDS → register fragments. min_slots=1, max_slots=2. ─────
    class _StageReadKV(PipelineStage):
        name = "ReadKV"
        kind = StageKind.MEMORY
        @property
        def resources(self):
            return ResourceDecl(lds_read_count=8)
        @property
        def min_slots(self):
            return 1
        @property
        def max_slots(self):
            return 2
        def execute(self, infra, **_):
            read_kv(infra.buf_slot)
            return {}
        def decompose(self, allocated_slots):
            if allocated_slots == 1:
                return [SubStage("ReadKV", self.execute, self.resources)]
            return [
                SubStage("ReadK", _pipeline_read_k, ResourceDecl(lds_read_count=4)),
                SubStage("ReadV", _pipeline_read_v, ResourceDecl(lds_read_count=4)),
            ]

    # ── Gemm1_QK: S = Q @ K^T. Single stage, no split. ──────────────────
    class _StageGemm1(PipelineStage):
        name = "Gemm1_QK"
        kind = StageKind.COMPUTE
        resources = ResourceDecl(mfma_count=head_dim // mma_k * 2)
        def execute(self, infra, *, frag_K, frag_Q, **_):
            out = gemm1_qk(frag_Q, frag_K)
            return {"frag_S": out[0]}

    # ── Softmax: online softmax. min_slots=1, max_slots=2. ──────────────
    class _StageSoftmax(PipelineStage):
        name = "Softmax"
        kind = StageKind.COMPUTE
        @property
        def resources(self):
            return ResourceDecl(valu_count=npair * 12, exp_count=n_c)
        @property
        def min_slots(self):
            return 1
        @property
        def max_slots(self):
            return 2
        def execute(self, infra, *, frag_S, m_i, l_i, frag_O, **_):
            out = softmax_full(frag_S, m_i, l_i, frag_O)
            return {
                "frag_P": out[0], "m_i": out[1], "l_i": out[2], "frag_O": out[3],
                "softmax_corr": out[4],
            }
        def decompose(self, allocated_slots):
            if allocated_slots == 1:
                return [SubStage("Softmax", self.execute, self.resources)]
            return [
                SubStage(
                    "SoftmaxFinish", self._finish_pipeline,
                    ResourceDecl(valu_count=npair * 4),
                    epilogue_drain=True,
                    sched_after=_sched_after_softmax_finish if _pd_for_sched >= 2 else None,
                ),
                SubStage(
                    "SoftmaxStart", self._start_pipeline,
                    ResourceDecl(valu_count=npair * 8, exp_count=n_c),
                    sched_after=_sched_after_softmax_start if _pd_for_sched >= 2 else None,
                ),
            ]
        def _finish_pipeline(self, infra, *, v_p_partial, l_i, **_):
            out = softmax_finish(v_p_partial, l_i)
            return {"l_i": out[0]}
        def _start_pipeline(self, infra, *, frag_S, m_i, l_i, v_p_partial, frag_P, frag_O, **_):
            out = softmax_start(frag_S, m_i, l_i, v_p_partial, frag_P, frag_O)
            return {
                "v_p_partial": out[1],
                "m_i": out[2],
                "l_i": out[3],
                "frag_O": out[4],
                "softmax_corr": out[5],
            }

    class _StageBridgeP(PipelineStage):
        name = "BridgeP"
        kind = StageKind.MEMORY
        resources = ResourceDecl(lds_read_count=4)

        @property
        def min_slots(self):
            return 2

        @property
        def max_slots(self):
            return 2

        @property
        def epilogue_drain(self):
            return True

        def decompose(self, allocated_slots):
            assert allocated_slots == 2
            return [
                SubStage(
                    "BridgePWrite", _pipeline_bridge_p_write,
                    ResourceDecl(valu_count=n_c),
                    defer_to_next_cluster=False,
                ),
                SubStage(
                    "BridgePLoad", _pipeline_bridge_p_load,
                    ResourceDecl(lds_read_count=4),
                    epilogue_drain=True,
                    epilogue_fn=_pipeline_bridge_p_epilogue,
                    defer_to_next_cluster=False,
                ),
            ]

        def execute(self, infra, **shared_regs):
            _pipeline_bridge_p_write(infra, **shared_regs)
            return _pipeline_bridge_p_load(infra)

    # ── Gemm2_PV: O += P @ V. ──────────────────────────────────────────
    # depth=1: P and V are from the same tile → use frag_Vt_next directly.
    # depth>=2: P is from the PREVIOUS tile → use frag_Vt (lagged carry).
    _pd = int(param.pipe_depth)
    _enable_stagger = pipeline_stagger_enabled(
        depth=_pd,
        num_groups=int(num_groups),
        m_waves=int(param.m_waves),
    )

    class _StageGemm2(PipelineStage):
        name = "Gemm2_PV"
        kind = StageKind.COMPUTE
        resources = ResourceDecl(mfma_count=head_dim // mma_k * 2)

        @property
        def min_slots(self):
            return 2 if _pd >= 2 else 1

        @property
        def max_slots(self):
            return 2 if _pd >= 2 else 1

        @property
        def epilogue_drain(self):
            return _pd >= 2

        @property
        def epilogue_fn(self):
            return _pipeline_gemm2_pd1 if _pd >= 2 else None

        def decompose(self, allocated_slots):
            if allocated_slots == 1:
                return [SubStage("Gemm2_PV", _pipeline_gemm2_pd1, self.resources)]
            return [
                SubStage(
                    "Gemm2WriteP", _pipeline_gemm2_write_p,
                    ResourceDecl(valu_count=n_c),
                    defer_to_next_cluster=False,
                ),
                SubStage(
                    "Gemm2PV", _pipeline_gemm2_pv,
                    ResourceDecl(mfma_count=head_dim // mma_k * 2),
                    epilogue_drain=True,
                    epilogue_fn=_pipeline_gemm2_pd1,
                    defer_to_next_cluster=False,
                    sched_after=_sched_after_gemm2_pv if _pd_for_sched >= 2 else None,
                ),
            ]

        def execute(self, infra, **shared_regs):
            if const_expr(_pd >= 2):
                gemm2_write_p(shared_regs["frag_P"], shared_regs["softmax_corr"])
                out = gemm2_pv(
                    shared_regs["frag_Vt"],
                    shared_regs["frag_Vt_next"],
                    shared_regs["frag_O"],
                )
                return {"frag_O": out[0], "frag_Vt": out[1]}
            out = gemm2_pd1(
                shared_regs["frag_P_a"],
                shared_regs["frag_Vt_next"],
                shared_regs["frag_O"],
            )
            return {"frag_O": out[0]}

    # ── Assemble pipeline ───────────────────────────────────────────
    # Initialize shared registers: pre-loop values + initial v_p_partial.
    # The first tile's SoftmaxFinish reads v_p_partial (loop-carried from
    # "previous tile"). We seed it with zeros so reduce_sum adds 0 to l_i,
    # pd2+: frag_P (frag_P_carry) holds the pre-rescale P snap for GEMM2WriteP;
    # v_p_partial carries the split-softmax partial (in frag_S after SoftmaxStart).
    frag_P_init = thr_qk.make_fragment_C(sP)
    frag_P_init.fill(0.0)
    frag_P_carry = thr_qk.make_fragment_C(sP)
    frag_P_carry.fill(0.0)
    v_p_partial = frag_P_init
    frag_P = frag_P_carry
    frag_Vt = thr_pv.make_fragment_B(sV[0])
    softmax_corr = [fx.Float32(1.0) for _ in range_constexpr(npair)]
    frag_Vt_next = frag_V[0]
    frag_S = frag_P_init
    frag_P_a = thr_pv.make_fragment_A(sP)

    infra = InfraContext()
    infra.head_dim = head_dim
    infra.tiled_mma_qk = tiled_mma_qk
    infra.tiled_mma_pv = tiled_mma_pv
    infra.elem_dtype = elem_dtype
    infra.n_kv_tiles = n_kv_tiles
    if const_expr(_enable_stagger):
        _wave_id_uni_i32 = rocdl.readfirstlane(
            fx.Int32.ir_type,
            fx.Int32(local_tid // GFX950_WAVE_SIZE),
        )
        infra.stagger_i32 = _wave_id_uni_i32

    infra.traits = _LayoutSchedTraits()
    _kv_vm_in_flight = _dma_ops_per_thread * 2

    if const_expr(param.pipe_depth == 1):
        for kv in range_constexpr(n_kv_tiles):
            rb = kv % _lds_ring_slots
            load_kv(kv, rb)
            rocdl.s_waitcnt(0)
            rocdl.s_barrier()
            read_kv(rb)
            frag_Vt_next = frag_V[rb]
            frag_S, = gemm1_qk(frag_Q, frag_K[rb])
            out_sm = softmax_full(frag_S, m_i, l_i, frag_O)
            frag_P, m_i, l_i, frag_O, softmax_corr = (
                out_sm[0], out_sm[1], out_sm[2], out_sm[3], out_sm[4],
            )
            bridge_p_write_from_frag_p(frag_P)
            rocdl.s_waitcnt(0)
            rocdl.s_barrier()
            frag_P_a, = bridge_p_load()
            frag_O, = gemm2_pd1(frag_P_a, frag_Vt_next, frag_O)
            rocdl.s_waitcnt(0)
            rocdl.s_barrier()
    elif const_expr(param.pipe_depth == 2):
        # scf.for KV loop (one loop body in IR). Lockstep phases; ring 0/1 via dynamic if.
        load_kv(fx.Int32(0), 0)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()
        if const_expr(_enable_stagger):
            stagger_open(infra.stagger_i32)

        init_args = (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [frag_O]
        )
        _o = 2 * npair
        loop_results = init_args
        for kv, loop_args in range(
            fx.Int32(0),
            fx.Int32(n_kv_tiles),
            fx.Int32(1),
            init=init_args,
        ):
            m_i = [loop_args[r] for r in range_constexpr(npair)]
            l_i = [loop_args[npair + r] for r in range_constexpr(npair)]
            frag_O = loop_args[_o]

            kv_i32 = fx.Int32(arith.index_cast(T.i32, kv))
            _ring_slots = fx.Int32(_lds_ring_slots)
            ring0 = (kv_i32 % _ring_slots) == fx.Int32(0)
            write_ring0 = ((kv_i32 + fx.Int32(1)) % _ring_slots) == fx.Int32(0)
            has_next = (kv_i32 + fx.Int32(1)) < fx.Int32(n_kv_tiles)

            if ring0:
                read_kv_work(0)
            else:
                read_kv_work(1)

            frag_S, = gemm1_qk(frag_Q, frag_K[0])
            out_sm = softmax_full(frag_S, m_i, l_i, frag_O)
            frag_P, m_i, l_i, frag_O, softmax_corr = (
                out_sm[0], out_sm[1], out_sm[2], out_sm[3], out_sm[4],
            )
            bridge_p_write_from_frag_p(frag_P)
            rocdl.s_waitcnt(0)
            rocdl.s_barrier()
            frag_P_a, = bridge_p_load()
            frag_O, = gemm2_pd1(frag_P_a, frag_V[0], frag_O)
            rocdl.s_waitcnt(0)
            rocdl.s_barrier()
            if has_next:
                if write_ring0:
                    load_kv(kv_i32 + fx.Int32(1), 0)
                else:
                    load_kv(kv_i32 + fx.Int32(1), 1)

            loop_results = yield (
                [m_i[r] for r in range_constexpr(npair)]
                + [l_i[r] for r in range_constexpr(npair)]
                + [frag_O]
            )

        m_i = [loop_results[r] for r in range_constexpr(npair)]
        l_i = [loop_results[npair + r] for r in range_constexpr(npair)]
        frag_O = loop_results[_o]
        if const_expr(_enable_stagger):
            stagger_close(infra.stagger_i32)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()
    else:
        shared_regs = {
            "frag_Q": frag_Q,
            "m_i": m_i,
            "l_i": l_i,
            "frag_O": frag_O,
            "v_p_partial": v_p_partial,
            "frag_P": frag_P,
            "frag_Vt": frag_Vt,
            "frag_Vt_next": frag_Vt_next,
            "softmax_corr": softmax_corr,
            "frag_S": frag_S,
            "frag_P_a": frag_P_a,
            "frag_K": frag_K[0],
        }
        pipeline = PipelineScheduler([
            Wire(_StageLoadKV(),  out=()),
            Wire(_StageReadKV(),  out=("frag_K", "frag_Vt_next")),
            Wire(_StageGemm1(),   inp=("frag_K", "frag_Q"),  out=("frag_S",),
                  carry=("frag_Q",)),
            Wire(_StageSoftmax(), inp=("frag_S", "m_i", "l_i", "frag_O", "frag_P"),
                  out=("v_p_partial", "m_i", "l_i", "frag_O", "softmax_corr"),
                  carry=("v_p_partial", "m_i", "l_i", "frag_O", "frag_P")),
            Wire(_StageBridgeP(), inp=("frag_P", "v_p_partial"), out=("frag_P_a",)),
            Wire(_StageGemm2(),
                 inp=("frag_Vt", "frag_Vt_next", "frag_O", "frag_P", "softmax_corr"),
                 out=("frag_O", "frag_Vt"),
                 carry=("frag_Vt",)),
        ], force_depth=param.pipe_depth, enable_stagger=True,
           num_groups=int(num_groups), m_waves=int(param.m_waves),
           light_c1_c2_boundary=(_pd_for_sched >= 2),
           inter_tile_readahead=(_pd_for_sched >= 2))
        pipeline.emit_prologue(infra, shared_regs)
        pipeline.emit_main_loop(infra, shared_regs, n_kv_tiles)
        pipeline.emit_epilogue(infra, shared_regs)
        frag_O = shared_regs["frag_O"]
        l_i = shared_regs["l_i"]


    #pipeline._dump_config()

    for i in range_constexpr(n_o):
        frag_O[i] = frag_O[i] * (fx.Float32(1.0) / l_i[o_slot_row[i]])
    thr_o = thr_pv.partition_C(gO)
    for i in range_constexpr(n_o):
        thr_o[i] = _to_elem(frag_O[i], elem_dtype)


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
    flex_attn_fwd_gfx950_kernel(
        o, q, k, v, seqlen_q, seqlen_kv, scale, tiled_mma_qk, tiled_mma_pv, param,
    ).launch(
        grid=(num_q_tiles, hq, b),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


# fast_fp_math breaks pipe_depth=2 when seqlen_kv == block_n (single KV tile); omit it.
_flex_attn_compile_hints = {
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
    causal: bool = False,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = 32,
    block_n: int = 32,
    num_groups: int = 2,
    pipe_depth: int = 1,
    pipe_stages: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Dense flash-attention forward on the layout API (gfx950). Phase 0: no mods.

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
        causal=causal,
        num_groups=num_groups,
        pipe_depth=pipe_depth,
        pipe_stages=pipe_stages,
    )
    #if os.environ.get("FLEX_ATTN_PRINT_TRAITS"):
    #    print_flex_attn_launch_summary(param, seqlen_q=Sq, seqlen_kv=Skv)
    # V is pre-transposed on the host to [B, Hkv, D, Skv] (contiguous) so GEMM2's
    # B operand has block_n (K-dim) contiguous. In-kernel LDSReadTrans is a future opt.
    vt = v.permute(0, 2, 3, 1).contiguous()
    launch_flex_attn_gfx950(
        out.contiguous(), q.contiguous(), k.contiguous(), vt,
        fx.Float32(scale), param, stream,
    )
    return out
