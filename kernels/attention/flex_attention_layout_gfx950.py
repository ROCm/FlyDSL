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

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch

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
    )


def make_flex_attn_kernel_name(param: FlexAttnParam) -> str:
    dtype_str = "fp16" if param.dtype_id == FLEX_DTYPE_FP16 else "bf16"
    name = f"flex_attn_{dtype_str}_m{param.block_m}n{param.block_n}d{param.head_dim}"
    name += f"_w{param.m_waves}x{param.n_waves}g{param.num_groups}"
    name += "_causal" if param.causal else "_dense"
    name += f"_pd{param.pipe_depth}"
    return name


_FM = fx.arith.FastMathFlags.fast


def _elem_dtype(dtype_id):
    return fx.Float16 if dtype_id == FLEX_DTYPE_FP16 else fx.BFloat16


def _row_reduce(x, mode):
    # Per-query-row reduction over the key dimension: the 16 keys of an N-wave are
    # spread across the low 4 lane bits, so a shuffle_xor butterfly over offsets
    # 8/4/2/1 combines them. block_n stays within one N-wave (n_waves_n == 1).
    w = x
    for off in (8, 4, 2, 1):
        peer = w.shuffle_xor(off, GFX950_WAVE_SIZE)
        w = w.maximumf(peer) if mode == "max" else w.addf(peer, fastmath=_FM)
    return w


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

    @fx.struct
    class SharedStorage:
        k_lds: fx.Array[elem_dtype, 2 * kv_tile_elems, 16]
        v_lds: fx.Array[elem_dtype, 2 * kv_tile_elems, 16]
        p: fx.Array[elem_dtype, num_groups * block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    sK_ptr = storage.k_lds.peek().ptr
    sV_ptr = storage.v_lds.peek().ptr

    # K LDS: [block_n, head_dim] row-major (head_dim contiguous).
    sK = [
        fx.make_view(sK_ptr + i * kv_tile_elems, fx.make_layout((block_n, head_dim), (head_dim, 1)))
        for i in range_constexpr(2)
    ]
    # V LDS: [head_dim, block_n] row-major (block_n contiguous) — V is host-transposed.
    sV = [
        fx.make_view(sV_ptr + i * kv_tile_elems, fx.make_layout((head_dim, block_n), (block_n, 1)))
        for i in range_constexpr(2)
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
    q_it = fx.rocdl.make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(q)) + fx.Int32(q_off))
    o_it = fx.rocdl.make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(o)) + fx.Int32(o_off))
    k_it = fx.rocdl.make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(k)) + fx.Int32(k_off))
    vt_it = fx.rocdl.make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(v)) + fx.Int32(v_off))

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
    n_c = fx.size(thr_qk.partition_C(sP).shape).unpack()
    npair = n_c // 2
    _s_slot_to_row_idx = [i % npair for i in range(n_c)]

    _M_NEG_FLOOR = -60.0  # large negative, finite; avoids -inf * scale = NaN under fastmath
    m_i = [fx.Float32(_M_NEG_FLOOR) for _ in range_constexpr(npair)]
    l_i = [fx.Float32(0.0) for _ in range_constexpr(npair)]

    n_o = fx.size(frag_O.shape).unpack()
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
    tcB_lds = [fx.make_tiled_copy_B(uca, tiled_mma_qk).get_slice(local_tid) for _ in range_constexpr(2)]
    tcB2_lds = [fx.make_tiled_copy_B(uca, tiled_mma_pv).get_slice(local_tid) for _ in range_constexpr(2)]
    frag_K = [thr_qk.make_fragment_B(sK[i]) for i in range_constexpr(2)]
    frag_V = [thr_pv.make_fragment_B(sV[i]) for i in range_constexpr(2)]

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
             for i in range_constexpr(2)]
    sV_i8 = [fx.recast_iter(fx.Int8, sV_ptr) + i * fx.Int32(kv_tile_elems * param.in_data_bytes)
             for i in range_constexpr(2)]
    _step_bytes = block_threads * _dma_bytes

    # P-bridge coordinate maps (compile-time, reused every tile)
    tr = thr_qk.partition_C(fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0))))
    tc = thr_qk.partition_C(fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1))))

    # Row-slot map for softmax (compile-time)
    _row_slots = [[s for s in range(n_c) if _s_slot_to_row_idx[s] == r] for r in range(npair)]

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

    # ── Pipeline: wire stages ───────────────────────────────────────────
    from kernels.attention.pipeline import (
        PipelineStage, PipelineScheduler, Wire, InfraContext,
        StageKind, ResourceDecl, SubStage,
    )

    # ── LoadKV: DMA K+V global → LDS. Single stage, no split. ───────────
    class _StageLoadKV(PipelineStage):
        name = "LoadKV"
        kind = StageKind.MEMORY
        resources = ResourceDecl(dma_count=_dma_ops_per_thread * 2)
        def execute(self, infra, **kw):
            _stage_kv_to_lds(infra.tile_idx, infra.buf_slot)
            return {}

    # ── ReadKV: LDS → register fragments. min_slots=1, max_slots=2. ─────
    # At 1 slot: reads both K and V in the same memory cluster.
    # At 2 slots: ReadK in C0, ReadV in C2 — K available for GEMM1 earlier,
    # V loaded later (closer to GEMM2), matching the dualwave pattern.
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
        def execute(self, infra, **kw):
            fx.copy(uca, tcB_lds[infra.buf_slot].partition_S(sK[infra.buf_slot]),
                    tcB_lds[infra.buf_slot].retile(frag_K[infra.buf_slot]))
            fx.copy(uca, tcB2_lds[infra.buf_slot].partition_S(sV[infra.buf_slot]),
                    tcB2_lds[infra.buf_slot].retile(frag_V[infra.buf_slot]))
            return {"frag_K": frag_K[infra.buf_slot], "frag_Vt_next": frag_V[infra.buf_slot]}
        def decompose(self, allocated_slots):
            if allocated_slots == 1:
                return [SubStage("ReadKV", self.execute, self.resources)]
            return [
                SubStage("ReadK", self._read_k, ResourceDecl(lds_read_count=4)),
                SubStage("ReadV", self._read_v, ResourceDecl(lds_read_count=4)),
            ]
        def _read_k(self, infra, **kw):
            buf = infra.buf_slot
            fx.copy(uca, tcB_lds[buf].partition_S(sK[buf]), tcB_lds[buf].retile(frag_K[buf]))
            return {"frag_K": frag_K[buf]}
        def _read_v(self, infra, **kw):
            buf = infra.buf_slot
            fx.copy(uca, tcB2_lds[buf].partition_S(sV[buf]), tcB2_lds[buf].retile(frag_V[buf]))
            return {"frag_Vt_next": frag_V[buf]}

    # ── Gemm1_QK: S = Q @ K^T. Single stage, no split. ──────────────────
    class _StageGemm1(PipelineStage):
        name = "Gemm1_QK"
        kind = StageKind.COMPUTE
        resources = ResourceDecl(mfma_count=head_dim // mma_k * 2)
        def execute(self, infra, *, frag_K, frag_Q, **_):
            frag_S = thr_qk.make_fragment_C(sP)
            frag_S.fill(0.0)
            fx.gemm(tiled_mma_qk, frag_S, frag_Q, frag_K, frag_S)
            return {"frag_S": frag_S}

    # ── Softmax: online softmax. min_slots=1, max_slots=2. ──────────────
    # At 1 slot: full softmax (reduce_max + rescale_O + exp2 + reduce_sum)
    #            runs monolithically in one compute cluster.
    # At 2 slots (dualwave pattern):
    #   Sub-stage 0 "SoftmaxFinish" (C1): finish PREVIOUS tile's exp2
    #     second half + reduce_sum + cast → frag_P. Runs alongside GEMM1.
    #   Sub-stage 1 "SoftmaxStart" (C3): reduce_max + rescale_O + sub_m +
    #     exp2 first half → v_p_partial. Runs alongside GEMM2.
    #   v_p_partial is loop-carried across tile iterations.
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
            corr = [fx.Float32(1.0) for _ in range_constexpr(npair)]
            for r in range_constexpr(npair):
                slots = _row_slots[r]
                row_max = frag_S[slots[0]]
                for si in range_constexpr(1, len(slots)):
                    row_max = row_max.maximumf(frag_S[slots[si]])
                tile_m = _row_reduce(row_max, "max")
                m_new = m_i[r].maximumf(tile_m)
                corr[r] = fmath.exp2((m_i[r] - m_new) * scale_log2e, fastmath=_FM)
                for si in range_constexpr(len(slots)):
                    frag_S[slots[si]] = fmath.exp2((frag_S[slots[si]] - m_new) * scale_log2e, fastmath=_FM)
                row_sum = frag_S[slots[0]]
                for si in range_constexpr(1, len(slots)):
                    row_sum = row_sum.addf(frag_S[slots[si]], fastmath=_FM)
                tile_l = _row_reduce(row_sum, "sum")
                l_i[r] = l_i[r] * corr[r] + tile_l
                m_i[r] = m_new
            for i in range_constexpr(n_o):
                frag_O[i] = frag_O[i] * corr[o_slot_row[i]]
            return {
                "frag_P": frag_S, "m_i": m_i, "l_i": l_i, "frag_O": frag_O,
                "softmax_corr": corr,
            }
        def decompose(self, allocated_slots):
            if allocated_slots == 1:
                return [SubStage("Softmax", self.execute, self.resources)]
            # Finish first (C1 with Gemm1: sum previous tile's v_p into l_i),
            # Start deferred (C3 with Gemm2: max/rescale/exp2 for current tile).
            return [
                SubStage("SoftmaxFinish", self._finish,
                         ResourceDecl(valu_count=npair * 4)),
                SubStage("SoftmaxStart", self._start,
                         ResourceDecl(valu_count=npair * 8, exp_count=n_c)),
            ]
        def _finish(self, infra, *, v_p_partial, l_i, **_):
            for r in range_constexpr(npair):
                slots = _row_slots[r]
                row_sum = v_p_partial[slots[0]]
                for si in range_constexpr(1, len(slots)):
                    row_sum = row_sum.addf(v_p_partial[slots[si]], fastmath=_FM)
                tile_l = _row_reduce(row_sum, "sum")
                l_i[r] = l_i[r] + tile_l
            return {"l_i": l_i}
        def _start(self, infra, *, frag_S, m_i, l_i, v_p_partial, v_p_for_gemm, frag_O, **_):
            for i in range_constexpr(n_c):
                v_p_for_gemm[i] = v_p_partial[i]
            corr = [fx.Float32(1.0) for _ in range_constexpr(npair)]
            for r in range_constexpr(npair):
                slots = _row_slots[r]
                row_max = frag_S[slots[0]]
                for si in range_constexpr(1, len(slots)):
                    row_max = row_max.maximumf(frag_S[slots[si]])
                tile_m = _row_reduce(row_max, "max")
                m_new = m_i[r].maximumf(tile_m)
                corr[r] = fmath.exp2((m_i[r] - m_new) * scale_log2e, fastmath=_FM)
                l_i[r] = l_i[r] * corr[r]
                for si in range_constexpr(len(slots)):
                    v_p_partial[slots[si]] = v_p_partial[slots[si]] * corr[r]
                for si in range_constexpr(len(slots)):
                    frag_S[slots[si]] = fmath.exp2((frag_S[slots[si]] - m_new) * scale_log2e, fastmath=_FM)
                m_i[r] = m_new
            for i in range_constexpr(n_o):
                frag_O[i] = frag_O[i] * corr[o_slot_row[i]]
            return {
                "frag_P": v_p_partial, "v_p_partial": frag_S, "m_i": m_i, "l_i": l_i,
                "frag_O": frag_O, "softmax_corr": corr,
            }

    # ── BridgeP: C→A bridge through LDS. Single stage, no split. ────────
    def _bridge_p_to_lds(infra, p_frag):
        rocdl.sched_barrier(0)
        fx.gpu.barrier()
        for i in range_constexpr(n_c):
            sP[fx.get_scalar(tr[i]), fx.get_scalar(tc[i])] = p_frag[i].to(elem_dtype)
        fx.gpu.barrier()
        rocdl.sched_barrier(0)
        frag_P_a = thr_pv.make_fragment_A(sP)
        fx.copy(uca, tcA2.partition_S(sP), tcA2.retile(frag_P_a))
        return frag_P_a

    def _bridge_p_epilogue(infra, *, v_p_partial, **_):
        return {"frag_P_a": _bridge_p_to_lds(infra, v_p_partial)}

    class _StageBridgeP(PipelineStage):
        name = "BridgeP"
        kind = StageKind.MEMORY
        resources = ResourceDecl(lds_read_count=4)
        def execute(self, infra, **shared_regs):
            # depth>=2: Bridge runs before SoftmaxStart; v_p_partial still holds
            # the previous tile's exp2(S) (P for lagged Gemm2). frag_P is a stale
            # rescale carry and must not be bridged.
            if const_expr(param.pipe_depth >= 2):
                p_frag = shared_regs["v_p_partial"]
            else:
                p_frag = shared_regs["frag_P"]
            return {"frag_P_a": _bridge_p_to_lds(infra, p_frag)}

    # ── Gemm2_PV: O += P @ V. ──────────────────────────────────────────
    # depth=1: P and V are from the same tile → use frag_Vt_next directly.
    # depth>=2: P is from the PREVIOUS tile → use frag_Vt (lagged carry).
    _pd = int(param.pipe_depth)

    def _gemm2_d1(infra, *, frag_P_a, frag_Vt_next, frag_O, **_):
        fx.gemm(tiled_mma_pv, frag_O, frag_P_a, frag_Vt_next, frag_O)
        return {"frag_O": frag_O}

    def _gemm2_d2(infra, *, frag_Vt, frag_Vt_next, frag_O, v_p_for_gemm, softmax_corr, **_):
        # Snapshot of previous tile's exp2(S), scaled by this tile's row corr (matches O).
        rocdl.sched_barrier(0)
        fx.gpu.barrier()
        for i in range_constexpr(n_c):
            pi = v_p_for_gemm[i].to(fx.Float32) * softmax_corr[_s_slot_to_row_idx[i]]
            sP[fx.get_scalar(tr[i]), fx.get_scalar(tc[i])] = pi.to(elem_dtype)
        fx.gpu.barrier()
        rocdl.sched_barrier(0)
        frag_P_a = thr_pv.make_fragment_A(sP)
        fx.copy(uca, tcA2.partition_S(sP), tcA2.retile(frag_P_a))
        fx.gemm(tiled_mma_pv, frag_O, frag_P_a, frag_Vt, frag_O)
        return {"frag_O": frag_O, "frag_Vt": frag_Vt_next}

    class _StageGemm2(PipelineStage):
        name = "Gemm2_PV"
        kind = StageKind.COMPUTE
        resources = ResourceDecl(mfma_count=head_dim // mma_k * 2)
        execute = staticmethod(_gemm2_d2 if _pd >= 2 else _gemm2_d1)

    # ── Assemble pipeline ────────────────────────────────────────────────
    # The scheduler auto-assigns clusters by StageKind alternation.
    # With Softmax.max_slots=2 and ReadKV.max_slots=2, the scheduler
    # decomposes them to produce the 4-cluster dualwave structure:
    #   C0 (mem):  LoadKV + ReadK
    #   C1 (comp): Gemm1_QK + SoftmaxFinish
    #   C2 (mem):  BridgeP + ReadV
    #   C3 (comp): SoftmaxStart + Gemm2_PV
    # With max_slots=1 (or force_depth=1), everything stays monolithic.
    pipeline = PipelineScheduler([
        Wire(_StageLoadKV(),  out=()),
        Wire(_StageReadKV(),  out=("frag_K", "frag_Vt_next")),
        Wire(_StageGemm1(),   inp=("frag_K", "frag_Q"),  out=("frag_S",),
                              carry=("frag_Q",)),
        Wire(_StageSoftmax(), inp=("frag_S", "m_i", "l_i", "frag_O", "v_p_for_gemm"),
                              out=("frag_P", "v_p_partial", "m_i", "l_i", "frag_O", "softmax_corr"),
                              carry=("v_p_partial", "m_i", "l_i", "frag_O", "v_p_for_gemm")),
        Wire(_StageBridgeP(), inp=("frag_P", "v_p_partial"), out=("frag_P_a",)),
        Wire(_StageGemm2(),
             inp=("frag_Vt", "frag_Vt_next", "frag_O", "v_p_for_gemm", "softmax_corr") if _pd >= 2
                 else ("frag_P_a", "frag_Vt_next", "frag_O"),
             out=("frag_O", "frag_Vt") if _pd >= 2 else ("frag_O",),
             carry=("frag_Vt",) if _pd >= 2 else ()),
    ], force_depth=param.pipe_depth, enable_stagger=False)

    # Initialize shared registers: pre-loop values + initial v_p_partial.
    # The first tile's SoftmaxFinish reads v_p_partial (loop-carried from
    # "previous tile"). We seed it with zeros so reduce_sum adds 0 to l_i,
    # and the initial frag_P is zero (GEMM2 accumulates 0 for the non-existent
    # "tile -1"). The real first contribution comes from tile 0's own
    # RescaleAndGemm2 → v_p_partial, consumed by tile 1's SoftmaxFinish.
    frag_P_init = thr_qk.make_fragment_C(sP)
    frag_P_init.fill(0.0)
    frag_P_carry = thr_qk.make_fragment_C(sP)
    frag_P_carry.fill(0.0)
    v_p_for_gemm_init = thr_qk.make_fragment_C(sP)
    v_p_for_gemm_init.fill(0.0)
    shared_regs = {
        "frag_Q": frag_Q,
        "m_i": m_i,
        "l_i": l_i,
        "frag_O": frag_O,
        "v_p_partial": frag_P_init,
        "frag_P": frag_P_carry,
        "v_p_for_gemm": v_p_for_gemm_init,
    }
    if _pd >= 2:
        shared_regs["frag_Vt"] = thr_pv.make_fragment_B(sV[0])  # dummy; first P is zeros
        shared_regs["softmax_corr"] = [fx.Float32(1.0) for _ in range_constexpr(npair)]

    infra = InfraContext()
    infra.tiled_mma_qk = tiled_mma_qk
    infra.tiled_mma_pv = tiled_mma_pv
    infra.elem_dtype = elem_dtype
    infra.n_kv_tiles = n_kv_tiles

    # ── Pipeline: prologue → main tile loop → epilogue ───────────────────
    # The scheduler handles: buffer priming, prefetch timing, cluster
    # execution, barrier placement, vmcnt management, and stagger.
    pipeline.emit_prologue(infra, shared_regs)
    pipeline.emit_main_loop(infra, shared_regs, n_kv_tiles)
    if const_expr(param.pipe_depth >= 2):
        # Inline tail drain (one KV tile of lagged softmax mass + P@V).
        _softmax = _StageSoftmax()
        shared_regs.update(_softmax._finish(infra, **shared_regs))
        shared_regs.update(_bridge_p_epilogue(infra, **shared_regs))
        # Last tile: P and V are aligned (not lagged like the main loop).
        shared_regs.update(_gemm2_d1(infra, **shared_regs))
        rocdl.s_barrier()
    else:
        pipeline.emit_epilogue(infra, shared_regs)
    
    frag_O = shared_regs["frag_O"]
    l_i = shared_regs["l_i"]

    # ── Output: O /= l_i (per row), store ───────────────────────────────
    for i in range_constexpr(n_o):
        frag_O[i] = frag_O[i] * (fx.Float32(1.0) / l_i[o_slot_row[i]])
    thr_o = thr_pv.partition_C(gO)
    for i in range_constexpr(n_o):
        thr_o[i] = frag_O[i].to(elem_dtype)


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
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Dense flash-attention forward on the layout API (gfx950). Phase 0: no mods.

    q/k/v: ``[B, S, H, D]`` (BSHD), bf16/f16. Returns ``[B, Sq, Hq, D]``.
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
    )
    # V is pre-transposed on the host to [B, Hkv, D, Skv] (contiguous) so GEMM2's
    # B operand has block_n (K-dim) contiguous. In-kernel LDSReadTrans is a future opt.
    vt = v.permute(0, 2, 3, 1).contiguous()
    launch_flex_attn_gfx950(
        out.contiguous(), q.contiguous(), k.contiguous(), vt,
        fx.Float32(scale), param, stream,
    )
    return out
