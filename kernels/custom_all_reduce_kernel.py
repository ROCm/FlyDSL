# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL all-reduce kernels using signal protocol for multi-GPU communication.

Implements 1-stage and 2-stage (reduce-scatter + all-gather) kernels.
Signal buffers are hipDeviceMallocUncached (bypasses L1/TCP cache).
Memory ordering uses GFX942 inline assembly for XGMI/HBM visibility.
"""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith as ea, gpu, range_constexpr, buffer_ops
from flydsl.expr import const_expr
from flydsl.expr.typing import T, Int32, Int64, Stream
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, llvm, rocdl
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from kernels.custom_all_reduce import _KMAXBLOCKS as _MAX_BLOCKS


# ---------------------------------------------------------------------------
# Low-level memory helpers — all operate on raw i64 device addresses.
#
# Cache modifier bits for buffer_load / buffer_store (AMD GFX942 aux field):
#   bit 0 = SC0  — bypass L1/TCP cache
#   bit 1 = SC1  — bypass L2/TCC cache
#   bit 2 = NT   — nontemporal (bypass hardware prefetcher)
# ---------------------------------------------------------------------------
_CM_CACHED  = 0  # normal cached access
_CM_SC1     = 2  # bypass L2 only  (reads from signal bufs across GPUs)
_CM_SC0_SC1 = 3  # bypass L1+L2   (writes to signal bufs: fully uncached)
_CM_NT      = 4  # nontemporal    (bulk data writes, bypasses L2 prefetch)


# ---- buffer resource descriptor helper ------------------------------------

def _make_rsrc(addr_i64):
    """Create buffer resource descriptor from a wave-uniform i64 base address."""
    return buffer_ops.create_buffer_resource_from_addr(addr_i64)


# ---- bulk data: 16-byte (128-bit) load / store ----------------------------
# These accept a pre-built rsrc descriptor and a per-lane element offset (i32).

def _load_v4i32(rsrc, elem_off_i32):
    """Buffer-load vector<4xi32> (16 bytes) with pre-built descriptor."""
    raw = buffer_ops.buffer_load(rsrc, elem_off_i32, vec_width=4, dtype=T.i32)
    return fx.Vector(raw)


def _store_v4i32(rsrc, elem_off_i32, data):
    """Buffer-store vector<4xi32> (16 bytes), cached."""
    buffer_ops.buffer_store(data, rsrc, elem_off_i32,
                            cache_modifier=_CM_CACHED)


def _store_v4i32_nt(rsrc, elem_off_i32, v4i32_val):
    """Buffer-store vector<4xi32> nontemporal — bypasses L2 prefetcher."""
    buffer_ops.buffer_store(v4i32_val, rsrc, elem_off_i32,
                            cache_modifier=_CM_NT)
    rocdl.s_waitcnt(0)


# ---- signal buffer: i32 load / store --------------------------------------

def _store_i32(rsrc, val_i32):
    """Store i32 with default caching via pre-built rsrc descriptor."""
    buffer_ops.buffer_store(val_i32, rsrc, 0,
                            cache_modifier=_CM_CACHED)


def _load_i32_uncached(rsrc):
    """Load i32 bypassing L2 (sc1) via pre-built rsrc descriptor."""
    val = buffer_ops.buffer_load(rsrc, 0,
                                 vec_width=1, dtype=T.i32,
                                 cache_modifier=_CM_SC1)
    rocdl.s_waitcnt(0)
    return val


def _store_i32_uncached(rsrc, val_i32):
    """Store i32 bypassing L1+L2 (sc0+sc1) via pre-built rsrc descriptor."""
    buffer_ops.buffer_store(val_i32, rsrc, 0,
                            cache_modifier=_CM_SC0_SC1)
    rocdl.s_waitcnt(0)


def _invalidate_l1():
    """Invalidate L1 scalar cache (buffer_inv sc1).

    Call inside a polling loop after an uncached load to discard stale L1
    lines so the next iteration sees fresh data from L2/HBM.
    """
    llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)


def _store_i32_uncached_flush(rsrc, val_i32):
    """Store i32 with L2 writeback then sc0+sc1 store via pre-built rsrc.

    buffer_wbl2 flushes dirty L2 lines to HBM before the signal store.
    """
    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
    buffer_ops.buffer_store(val_i32, rsrc, 0,
                            cache_modifier=_CM_SC0_SC1)
    rocdl.s_waitcnt(0)


# ---- pointer array helpers -----------------------------------------------

def _pack_i64_vec(values):
    """Pack preloaded i64 values into vector<Nxi64> for contiguous VGPR storage.

    On AMDGPU the subsequent ``fx.Vector.__getitem__`` with a dynamic index lowers
    through ``ConvertVectorToLLVM`` to ``llvm.extractelement`` which the
    backend emits as ``v_movrels_b32`` (VGPR-relative addressing, ~3 insns)
    instead of a chained ``arith.select`` costing 2*(N-1) insns.
    """
    return fx.Vector.from_elements(values, dtype=fx.Int64)


def _extract_i64(vec, index):
    """Extract i64 from a packed vector by dynamic index (VGPR-relative)."""
    if not isinstance(vec, fx.Vector):
        vec = fx.Vector(vec)
    return vec[index]


def _load_device_ptr(array_base_i64, index):
    """Load i64 pointer from a device-side pointer array at *index*.

    Uses buffer_load(dtype=i64): offset is in elements so buffer_load
    automatically scales by 8 bytes internally.
    """
    rsrc = buffer_ops.create_buffer_resource_from_addr(array_base_i64)
    return buffer_ops.buffer_load(rsrc, index, vec_width=1, dtype=T.i64)


# Signal buffer layout offsets (bytes), derived from _MAX_BLOCKS.
# start[_MAX_BLOCKS][8] of uint32 | end[_MAX_BLOCKS][8] of uint32 | flag[_MAX_BLOCKS] of uint32
_SG_START_OFF_B = 0
_SG_END_OFF_B = _MAX_BLOCKS * 8 * 4            # 2560 when _MAX_BLOCKS=80
_SG_FLAG_OFF_B = _MAX_BLOCKS * 8 * 4 * 2       # 5120 when _MAX_BLOCKS=80


# ---------------------------------------------------------------------------
# Element type helpers
# ---------------------------------------------------------------------------

_BYTES_PER_PACK = 16  # sizeof(vector<4xi32>), the atomic load/store unit
_ELEMS_PER_PACK = _BYTES_PER_PACK // 4  # i32 elements per pack


def _elem_bytes(dtype_str: str) -> int:
    """Return byte width of one scalar element for the given dtype."""
    d = (dtype_str or "").strip().lower()
    if d in {"f32", "fp32"}:
        return 4
    if d in {"f16", "fp16", "bf16"}:
        return 2
    raise ValueError(f"unsupported dtype_str: {dtype_str!r}")


def _elem_type(dtype_str: str) -> ir.Type:
    d = (dtype_str or "").strip().lower()
    if d in {"f16", "fp16"}:
        return T.f16
    if d in {"bf16"}:
        return T.bf16
    if d in {"f32", "fp32"}:
        return T.f32
    raise ValueError(f"unsupported dtype_str: {dtype_str!r}")


def _pack_elems(dtype_str: str) -> int:
    """Number of elements per pack, derived from _BYTES_PER_PACK."""
    return _BYTES_PER_PACK // _elem_bytes(dtype_str)


def _u(v):
    """Tag ArithValue as unsigned for //, %, <, <=, >, >=, >> ops."""
    return v.with_signedness(False)


def _raw(v):
    """Unwrap FlyDSL wrapper values when low-level MLIR ops need raw ir.Value."""
    return v.ir_value() if hasattr(v, "ir_value") else v


def _smem_store(smem_ptr: SmemPtr, value, idx):
    """Store one vector lane into shared memory by scalar index."""
    smem_ptr.store(value, [idx])


def _smem_load(smem_ptr: SmemPtr, idx):
    """Load one vector lane from shared memory by scalar index."""
    return smem_ptr.load([idx])


def _c64(v):
    """Create i64 constant with concise syntax."""
    return ea.constant(v, type=T.i64)


# ---------------------------------------------------------------------------
# Signal synchronization primitives
# ---------------------------------------------------------------------------

def _signal_start_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int):
    """Start-sync: write start flag to all peers, wait for all to arrive."""
    i32, i64 = T.i32, T.i64

    flag_addr = self_sg_i64 + _c64(_SG_FLAG_OFF_B) + bid_i32.extui(i64) * _c64(4)
    flag_rsrc = _make_rsrc(flag_addr)
    flag = _load_i32_uncached(flag_rsrc) + 1

    bid8 = bid_i32 * 8
    lin_lane = bid8 + lane_i32
    start_wait_addr = self_sg_i64 + _c64(_SG_START_OFF_B) + lin_lane.extui(i64) * _c64(4)
    wait_rsrc = _make_rsrc(start_wait_addr)
    lin_rank = bid8 + rank_i32
    start_rank_off = _c64(_SG_START_OFF_B) + lin_rank.extui(i64) * _c64(4)

    is_lane = _u(lane_i32) < ngpus
    if_op = scf.IfOp(_raw(is_lane), results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _extract_i64(_pack_i64_vec(sgs_i64), lane_i32)
        peer_rsrc = _make_rsrc(peer_sg + start_rank_off)
        _store_i32_uncached(peer_rsrc, flag)
        init_cur = _load_i32_uncached(wait_rsrc)
        w = scf.WhileOp([i32], [init_cur])
        wb = ir.Block.create_at_start(w.before, [i32])
        wa = ir.Block.create_at_start(w.after, [i32])
        with ir.InsertionPoint(wb):
            cur = wb.arguments[0]
            need_wait = _u(cur) < flag
            scf.ConditionOp(_raw(need_wait), [cur])
        with ir.InsertionPoint(wa):
            scf.YieldOp([_load_i32_uncached(wait_rsrc)])
        scf.YieldOp([])

    gpu.barrier()
    is_t0 = lane_i32 == 0
    if_t0 = scf.IfOp(_raw(is_t0), results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        _store_i32(flag_rsrc, flag)
        scf.YieldOp([])
    return flag_addr


def _signal_end_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64,
                     ngpus: int):
    """End-sync: write end flag to all peers, wait for all to finish."""

    i32, i64 = T.i32, T.i64

    flag_addr = self_sg_i64 + _c64(_SG_FLAG_OFF_B) + bid_i32.extui(i64) * _c64(4)
    flag_rsrc = _make_rsrc(flag_addr)
    flag = _load_i32_uncached(flag_rsrc) + 1

    bid8 = bid_i32 * 8
    lin_lane = bid8 + lane_i32
    end_wait_addr = self_sg_i64 + _c64(_SG_END_OFF_B) + lin_lane.extui(i64) * _c64(4)
    wait_rsrc = _make_rsrc(end_wait_addr)
    lin_rank = bid8 + rank_i32
    end_rank_off = _c64(_SG_END_OFF_B) + lin_rank.extui(i64) * _c64(4)

    is_lane = _u(lane_i32) < ngpus
    if_op = scf.IfOp(_raw(is_lane), results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _extract_i64(_pack_i64_vec(sgs_i64), lane_i32)
        peer_rsrc = _make_rsrc(peer_sg + end_rank_off)
        _store_i32_uncached(peer_rsrc, flag)
        init_cur = _load_i32_uncached(wait_rsrc)
        w = scf.WhileOp([i32], [init_cur])
        wb = ir.Block.create_at_start(w.before, [i32])
        wa = ir.Block.create_at_start(w.after, [i32])
        with ir.InsertionPoint(wb):
            cur = wb.arguments[0]
            need_wait = _u(cur) < flag
            scf.ConditionOp(_raw(need_wait), [cur])
        with ir.InsertionPoint(wa):
            nxt = _load_i32_uncached(wait_rsrc)
            _invalidate_l1()
            scf.YieldOp([nxt])
        scf.YieldOp([])

    gpu.barrier()
    is_t0 = lane_i32 == 0
    if_t0 = scf.IfOp(_raw(is_t0), results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        _store_i32(flag_rsrc, flag)
        scf.YieldOp([])


# ---------------------------------------------------------------------------
# Kernel work group size attribute helper
# ---------------------------------------------------------------------------

def _current_gpu_func():
    """Return the enclosing gpu.func op for current kernel emission."""
    entry_block = ir.InsertionPoint.current.block
    gpu_func_op = entry_block.owner
    return gpu_func_op


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_allreduce_kernels(*, N: int, dtype_str: str, world_size: int, threads: int = 512):
    """Build and return compiled allreduce launcher functions.

    Captures compile-time constants as closures, returns a dict with:
      "run_1stage_arr"        -- CUDAGraph-compatible 1-stage allreduce (small N)
      "run_2stage_arr"        -- CUDAGraph-compatible 2-stage allreduce
      "run_2stage_write_mode" -- Large-tensor 2-stage allreduce (N > 512*4096, ws=8)

    Args:
        N:          Total number of elements to reduce.
        dtype_str:  "f16" or "f32".
        world_size: Number of GPUs (2, 4, or 8).
        threads:    Threads per block (must be divisible by world_size).
    """
    if world_size not in {2, 4, 8}:
        raise ValueError(f"world_size must be one of {{2,4,8}}, got {world_size}")
    if threads <= 0 or threads % world_size != 0:
        raise ValueError(f"threads={threads} must be > 0 and divisible by world_size={world_size}")
    tnum_gpu_check = threads // world_size
    if tnum_gpu_check & (tnum_gpu_check - 1) != 0:
        raise ValueError(
            f"threads/world_size must be a power of 2, got "
            f"threads={threads}, world_size={world_size}, "
            f"threads/world_size={tnum_gpu_check}"
        )

    pack_elems = _pack_elems(dtype_str)
    if N <= 0 or N % pack_elems != 0:
        raise ValueError(f"N={N} must be > 0 and a multiple of pack_elems={pack_elems}")

    # Compile-time constants captured by closures
    num_packs = N // pack_elems
    part_p = num_packs // world_size
    largest_part_p = part_p + (num_packs % world_size)
    tnum_gpu = threads // world_size
    is_f32 = dtype_str.lower().strip() in {"f32", "fp32"}
    is_bf16 = dtype_str.lower().strip() in {"bf16"}
    # Vectorized gather path: requires perfect partition + no world_size=6
    vec_ok = (num_packs % world_size == 0) and (world_size != 6)

    # Adaptive LDS buffer strategy for 2-stage Stage 1:
    #   Single buffer (8KB, 2 barriers/iter): halves LDS usage, doubles block
    #   occupancy per CU, improves latency-hiding for many-iteration workloads.
    #   Double buffer (16KB, 1 barrier/iter): saves 1 barrier per iteration,
    #   better for small tensors where the kernel runs only 1-2 iterations and
    #   occupancy is already saturated by register usage rather than LDS.
    # Threshold: use single buffer when estimated iterations per block >= 3.
    _est_iters_2stage = max(1, (max(1, part_p) + _MAX_BLOCKS * tnum_gpu - 1)
                            // (_MAX_BLOCKS * tnum_gpu))
    _use_single_buf_2stage = (_est_iters_2stage >= 3)

    # -----------------------------------------------------------------------
    # GPU Kernel: 1-stage arr (full allreduce in one pass, CUDAGraph-compatible)
    # -----------------------------------------------------------------------
    @flyc.kernel(known_block_size=[threads, 1, 1])
    def allreduce_1stage_arr(
        rank: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        in_ptrs: Int64,
        out_ptr: Int64,
    ):
        """1-stage allreduce using shared memory.

        Each warp loads data from one rank into shared memory, then warp 0
        reduces across all warps and writes the result to global memory.
        """
        i32, i64 = T.i32, T.i64
        v4i32 = T.i32x4
        if const_expr(not is_f32):
            half_dtype = fx.BFloat16 if is_bf16 else fx.Float16

        gpu_func_op = _current_gpu_func()

        lane_i32 = fx.thread_idx.x
        bid_i32 = fx.block_idx.x
        rank_i32 = rank
        self_sg_i64 = self_sg
        sg_ptrs_i64 = sg_ptrs
        in_ptrs_i64 = in_ptrs
        out_ptr_i64 = out_ptr

        sgs = [_load_device_ptr(sg_ptrs_i64, i) for i in range(world_size)]
        in_ptrs_arr = [_load_device_ptr(in_ptrs_i64, i) for i in range(world_size)]
        in_ptrs_vec = _pack_i64_vec(in_ptrs_arr)

        smem_sym = f"allreduce_1s_smem_ws{world_size}_t{threads}"
        n_smem = 2 * threads
        allocator = SmemAllocator(None, global_sym_name=smem_sym)
        smem_off = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_off + n_smem * _BYTES_PER_PACK
        with ir.InsertionPoint.at_block_begin(gpu_func_op.operation.block):
            allocator.finalize()
        smem_ptr = SmemPtr(allocator.get_base(), smem_off, v4i32, shape=(n_smem,))
        smem_ptr.get()

        tnum_gpu_i32 = tnum_gpu
        warp_id = _u(lane_i32) // tnum_gpu_i32
        lane_id = _u(lane_i32) % tnum_gpu_i32

        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # Grid-stride loop: each warp loads from its assigned rank,
        # then warp 0 reduces and writes output.
        tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
        stride_pack = fx.grid_dim.x * tnum_gpu_i32

        out_rsrc = _make_rsrc(out_ptr_i64)
        in_rsrc = _make_rsrc(_extract_i64(in_ptrs_vec, warp_id))

        parity = 0
        for p in range(tid_pack, num_packs, stride_pack):
            off_i32 = p * _ELEMS_PER_PACK
            raw = _load_v4i32(in_rsrc, off_i32)
            sm_base = parity * threads
            sm_idx = fx.Index(sm_base + lane_i32)
            _smem_store(smem_ptr, raw, sm_idx)
            gpu.barrier()

            # Warp 0 reduces across all warps and writes to output
            if warp_id == 0:
                acc = None
                for wi in range_constexpr(world_size):
                    sm_i_idx = ea.index_cast(
                        T.index, wi * tnum_gpu_i32 + lane_id + sm_base)
                    raw_i = fx.Vector(_smem_load(smem_ptr, sm_i_idx))
                    if const_expr(is_f32):
                        vf = raw_i.bitcast(fx.Float32)
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = raw_i.bitcast(half_dtype)
                        v32 = v16.to(fx.Float32)
                        acc = v32 if acc is None else acc + v32
                if const_expr(is_f32):
                    out_bits = acc.bitcast(fx.Int32)
                else:
                    out_bits = acc.to(half_dtype).bitcast(fx.Int32)
                dst_off_i32 = p * _ELEMS_PER_PACK
                _store_v4i32(out_rsrc, dst_off_i32, out_bits)
            parity = 1 - parity

        # 1-stage does not use end_sync to avoid hangs.

    # -----------------------------------------------------------------------
    # GPU Kernel: 2-stage arr (reduce-scatter + all-gather)
    # -----------------------------------------------------------------------
    @flyc.kernel(known_block_size=[threads, 1, 1])
    def allreduce_2stage_arr(
        rank: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        in_ptrs: Int64,
        tmp_ptrs: Int64,
        out_ptr: Int64,
    ):
        i32, i64 = T.i32, T.i64
        v4i32 = T.i32x4
        if const_expr(not is_f32):
            half_dtype = fx.BFloat16 if is_bf16 else fx.Float16

        gpu_func_op = _current_gpu_func()

        lane_i32 = fx.thread_idx.x
        bid_i32 = fx.block_idx.x
        rank_i32 = rank
        self_sg_i64 = self_sg
        sg_ptrs_i64 = sg_ptrs
        in_ptrs_i64 = in_ptrs
        tmp_ptrs_i64 = tmp_ptrs
        out_ptr_i64 = out_ptr

        sgs = [_load_device_ptr(sg_ptrs_i64, i) for i in range(world_size)]
        in_ptrs_arr = [_load_device_ptr(in_ptrs_i64, i) for i in range(world_size)]
        tmp_ptrs_arr = [_load_device_ptr(tmp_ptrs_i64, i) for i in range(world_size)]
        in_ptrs_vec = _pack_i64_vec(in_ptrs_arr)

        # Compute pack range for this rank's reduce-scatter partition
        start_p = rank_i32 * part_p
        is_last = rank_i32 == (world_size - 1)
        end_p = is_last.select(
            num_packs,
            start_p + part_p,
        )

        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        tnum_gpu_i32 = tnum_gpu
        warp_id = _u(lane_i32) // tnum_gpu_i32
        lane_id = _u(lane_i32) % tnum_gpu_i32
        tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
        stride_pack = fx.grid_dim.x * tnum_gpu_i32

        _buf_tag = "1b" if _use_single_buf_2stage else "2b"
        smem_sym = f"allreduce_smem_ws{world_size}_t{threads}_{_buf_tag}"
        smem_slots = threads if _use_single_buf_2stage else 2 * threads
        allocator = SmemAllocator(None, global_sym_name=smem_sym)
        smem_off = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_off + smem_slots * _BYTES_PER_PACK
        with ir.InsertionPoint.at_block_begin(gpu_func_op.operation.block):
            allocator.finalize()
        smem_ptr = SmemPtr(allocator.get_base(), smem_off, v4i32, shape=(smem_slots,))
        smem_ptr.get()
        tmp_out_rsrc = _make_rsrc(tmp_ptrs_arr[0])

        # ---- Stage 1: reduce-scatter ----
        # Two implementations selected at compile time via _use_single_buf_2stage:
        #   Single-buffer (large tensor): 8KB LDS, 2 barriers/iter, higher occupancy.
        #   Double-buffer (small tensor): 16KB LDS, 1 barrier/iter (parity trick).
        in_rsrc = _make_rsrc(_extract_i64(in_ptrs_vec, warp_id))

        def _build_reduce_body(cur, smem_base_expr=None):
            """Emit reduce body: load → smem → barrier1 → warp0 reduce → [barrier2]."""
            off_i32 = cur * _ELEMS_PER_PACK
            raw = _load_v4i32(in_rsrc, off_i32)
            if const_expr(smem_base_expr is None):
                sm_idx = fx.Index(lane_i32)
            else:
                sm_idx = fx.Index(smem_base_expr + lane_i32)
            _smem_store(smem_ptr, raw, sm_idx)
            gpu.barrier()  # barrier 1: all warps have written smem

            if warp_id == 0:
                acc = None
                for wi in range_constexpr(world_size):
                    if const_expr(smem_base_expr is None):
                        sm_r_idx = fx.Index(wi * tnum_gpu_i32 + lane_id)
                    else:
                        sm_r_idx = fx.Index(
                            wi * tnum_gpu_i32 + lane_id + smem_base_expr
                        )
                    raw_i = fx.Vector(_smem_load(smem_ptr, sm_r_idx))
                    if const_expr(is_f32):
                        vf = raw_i.bitcast(fx.Float32)
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = raw_i.bitcast(half_dtype)
                        v32 = v16.to(fx.Float32)
                        acc = v32 if acc is None else acc + v32
                if const_expr(is_f32):
                    out_raw = acc.bitcast(fx.Int32)
                else:
                    out_raw = acc.to(half_dtype).bitcast(fx.Int32)
                rel_p = cur - start_p
                rel_off_i32 = rel_p * _ELEMS_PER_PACK
                _store_v4i32(tmp_out_rsrc, rel_off_i32, out_raw)

        idx_p = start_p + tid_pack
        if const_expr(_use_single_buf_2stage):
            # Single buffer: 8KB LDS, 2 barriers per iteration.
            for cur in range(idx_p, end_p, stride_pack):
                _build_reduce_body(cur, smem_base_expr=None)
                gpu.barrier()  # barrier 2: protect smem before next iter's writes
        else:
            # Double buffer: 16KB LDS, 1 barrier per iteration (parity trick).
            # The parity alternates between the two smem halves so warp-0 reads
            # from half-A while all warps write the next pack to half-B.
            parity = 0
            for cur in range(idx_p, end_p, stride_pack):
                sm_base = parity * threads
                _build_reduce_body(cur, smem_base_expr=sm_base)
                # No barrier 2: parity ensures next iteration writes to opposite
                # smem half, so warp-0 reads and all-warp writes are disjoint.
                parity = 1 - parity

        gpu.barrier()
        _signal_end_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                         self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # ---- Stage 2: all-gather ----
        out_rsrc = _make_rsrc(out_ptr_i64)

        if const_expr(vec_ok):
            tmp_ptrs_vec = _pack_i64_vec(tmp_ptrs_arr)
            tid_pack2 = bid_i32 * tnum_gpu_i32 + lane_id
            stride_pack2 = fx.grid_dim.x * tnum_gpu_i32
            tmp_rsrc = _make_rsrc(_extract_i64(tmp_ptrs_vec, warp_id))
            for cur in range(tid_pack2, part_p, stride_pack2):
                sum_rw = rank_i32 + warp_id
                if const_expr(world_size in {2, 4, 8}):
                    dst_rank = sum_rw & (world_size - 1)
                else:
                    dst_rank = _u(sum_rw) % world_size
                src_off_i32 = cur * _ELEMS_PER_PACK
                raw = _load_v4i32(tmp_rsrc, src_off_i32)
                dst_pack = dst_rank * part_p + cur
                dst_off_i32 = dst_pack * _ELEMS_PER_PACK
                _store_v4i32(out_rsrc, dst_off_i32, raw)
        else:
            tmp_rsrcs = [_make_rsrc(tmp_ptrs_arr[i]) for i in range(world_size)]
            tid_i32 = bid_i32 * threads + lane_i32
            stride_i32 = fx.grid_dim.x * threads

            for cur in range(tid_i32, largest_part_p, stride_i32):
                for p in range_constexpr(world_size):
                    if const_expr(p == world_size - 1):
                        ok = True
                    else:
                        ok = _u(cur) < part_p
                    if ok:
                        src_off_i32 = cur * _ELEMS_PER_PACK
                        raw = _load_v4i32(tmp_rsrcs[p], src_off_i32)
                        dst_pack_idx = (p * part_p) + cur
                        dst_off_i32 = dst_pack_idx * _ELEMS_PER_PACK
                        _store_v4i32(out_rsrc, dst_off_i32, raw)

    # -----------------------------------------------------------------------
    # GPU Kernel: 2-stage write-mode (large tensors, writes reduced result
    # directly to REMOTE output buffers via XGMI)
    # -----------------------------------------------------------------------
    @flyc.kernel(known_block_size=[threads, 1, 1])
    def allreduce_2stage_write_mode(
        rank: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        inp_ptr: Int64,
        out_ptrs: Int64,
        tmp_ptrs: Int64,
    ):
        i32, i64 = T.i32, T.i64
        v4i32 = T.i32x4
        if const_expr(not is_f32):
            half_dtype = fx.BFloat16 if is_bf16 else fx.Float16

        gpu_func_op = _current_gpu_func()

        lane_i32 = fx.thread_idx.x
        bid_i32 = fx.block_idx.x
        rank_i32 = rank
        self_sg_i64 = self_sg
        sg_ptrs_i64 = sg_ptrs
        inp_ptr_i64 = inp_ptr
        out_ptrs_i64 = out_ptrs
        tmp_ptrs_i64 = tmp_ptrs

        sgs = [_load_device_ptr(sg_ptrs_i64, i) for i in range(world_size)]
        out_ptrs_arr = [_load_device_ptr(out_ptrs_i64, i) for i in range(world_size)]
        tmp_ptrs_arr = [_load_device_ptr(tmp_ptrs_i64, i) for i in range(world_size)]
        tmp_ptrs_vec = _pack_i64_vec(tmp_ptrs_arr)
        out_ptrs_vec = _pack_i64_vec(out_ptrs_arr)

        tnum_gpu_i32 = tnum_gpu
        log2_tnum = int(math.log2(tnum_gpu))
        warp_id = _u(lane_i32) >> log2_tnum
        warp_base = warp_id * tnum_gpu_i32
        lane_id = lane_i32 - warp_base
        tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
        stride_pack = fx.grid_dim.x * tnum_gpu_i32

        smem_sym_wm = f"allreduce_smem_wm_ws{world_size}_t{threads}"
        n_smem_wm = 2 * threads
        allocator_wm = SmemAllocator(None, global_sym_name=smem_sym_wm)
        smem_wm_off = allocator_wm._align(allocator_wm.ptr, 16)
        allocator_wm.ptr = smem_wm_off + n_smem_wm * _BYTES_PER_PACK
        with ir.InsertionPoint.at_block_begin(gpu_func_op.operation.block):
            allocator_wm.finalize()
        smem_ptr = SmemPtr(allocator_wm.get_base(), smem_wm_off, v4i32, shape=(n_smem_wm,))
        smem_ptr.get()
        tmp_out_i64 = _extract_i64(tmp_ptrs_vec, rank_i32)

        # ---- Stage 1: scatter local input to REMOTE tmp buffers ----
        inp_rsrc = _make_rsrc(inp_ptr_i64)

        start_w = warp_id * part_p
        is_last_w = warp_id == (world_size - 1)
        end_w = is_last_w.select(
            num_packs,
            start_w + part_p,
        )

        dst_tmp = _extract_i64(tmp_ptrs_vec, warp_id)
        is_tmp_null = dst_tmp == _c64(0)
        dst_tmp_low4 = dst_tmp & _c64(0xF)
        is_tmp_misaligned = dst_tmp_low4 != _c64(0)
        bad_tmp_addr = is_tmp_null | is_tmp_misaligned
        dst_tmp_rsrc = _make_rsrc(dst_tmp)

        idx_s1 = start_w + tid_pack
        for cur in range(idx_s1, end_w, stride_pack):
            cur_off_i32 = cur * _ELEMS_PER_PACK
            raw = _load_v4i32(inp_rsrc, cur_off_i32)
            rel_idx = cur - start_w
            dst_off = rank_i32 * part_p + rel_idx
            if not bad_tmp_addr:
                dst_off_i32 = dst_off * _ELEMS_PER_PACK
                _store_v4i32(dst_tmp_rsrc, dst_off_i32, raw)

        # Signal all ranks that stage 1 is complete
        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # ---- Stage 2: reduce local tmp and write to REMOTE outputs ----
        tmp_out_rsrc = _make_rsrc(tmp_out_i64)
        part_p_i32 = ea.constant(part_p, type=T.i32)
        largest_part_p_i32 = ea.constant(largest_part_p, type=T.i32)
        is_last_rank_s2 = rank_i32 == (world_size - 1)
        end_s2 = is_last_rank_s2.select(largest_part_p_i32, part_p_i32)

        is_tmpout_null = tmp_out_i64 == _c64(0)
        tmpout_low4 = tmp_out_i64 & _c64(0xF)
        is_load_misaligned = tmpout_low4 != _c64(0)
        bad_load_addr = is_tmpout_null | is_load_misaligned

        dst_ptr = _extract_i64(out_ptrs_vec, warp_id)
        dst_out_rsrc = _make_rsrc(dst_ptr)
        is_out_null = dst_ptr == _c64(0)
        dst_ptr_low4 = dst_ptr & _c64(0xF)
        is_out_misaligned = dst_ptr_low4 != _c64(0)
        bad_out_addr = is_out_null | is_out_misaligned

        for cur in range(tid_pack, end_s2, stride_pack):
            # All warps load their chunk from tmp into smem
            src_off = warp_id * part_p + cur
            src_off_i32 = src_off * _ELEMS_PER_PACK
            raw = fx.Vector.filled(4, 0, fx.Int32)
            if not bad_load_addr:
                raw = _load_v4i32(tmp_out_rsrc, src_off_i32)

            sm_idx = fx.Index(lane_i32)
            _smem_store(smem_ptr, raw, sm_idx)
            gpu.barrier()

            # Warp 0 reduces across all warps, writes result to res area
            # (smem[threads .. threads+tnum_gpu-1]).  Two-barrier pattern
            # matching aiter: barrier1 guards tmp_smem, barrier2 guards
            # res_smem; between iterations tmp and res are disjoint so no
            # WAR hazard exists.
            if warp_id == 0:
                acc = None
                for wi in range_constexpr(world_size):
                    sm_i_idx = ea.index_cast(
                        T.index, (wi * tnum_gpu) + lane_id)
                    raw_i = fx.Vector(_smem_load(smem_ptr, sm_i_idx))
                    if const_expr(is_f32):
                        vf = raw_i.bitcast(fx.Float32)
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = raw_i.bitcast(half_dtype)
                        v32 = v16.to(fx.Float32)
                        acc = v32 if acc is None else acc + v32
                if const_expr(is_f32):
                    out_raw = acc.bitcast(fx.Int32)
                else:
                    out_raw = acc.to(half_dtype).bitcast(fx.Int32)
                res_idx = fx.Index(threads + lane_id)
                _smem_store(smem_ptr, out_raw, res_idx)

            gpu.barrier()

            # All warps read the same reduced result from res area and
            # nontemporal-write to their respective remote output buffers.
            res_read_idx = fx.Index(threads + lane_id)
            reduced_val = _smem_load(smem_ptr, res_read_idx)

            dst_out_off = rank_i32 * part_p + cur
            dst_off_i32 = dst_out_off * _ELEMS_PER_PACK

            if not bad_out_addr:
                _store_v4i32_nt(dst_out_rsrc, dst_off_i32, reduced_val)

        gpu.barrier()
        _signal_end_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                         self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

    # -----------------------------------------------------------------------
    # Host launchers (@flyc.jit)
    # -----------------------------------------------------------------------
    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def run_1stage_arr(
        rank: Int32,
        grid_x: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        in_ptrs: Int64,
        out_ptr: Int64,
        stream: Stream = Stream(None),
    ):
        allreduce_1stage_arr(
            rank,
            self_sg,
            sg_ptrs,
            in_ptrs,
            out_ptr,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def run_2stage_arr(
        rank: Int32,
        grid_x: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        in_ptrs: Int64,
        tmp_ptrs: Int64,
        out_ptr: Int64,
        stream: Stream = Stream(None),
    ):
        """Launch 2-stage allreduce (arr variant, CUDAGraph-compatible)."""
        allreduce_2stage_arr(
            rank,
            self_sg,
            sg_ptrs,
            in_ptrs,
            tmp_ptrs,
            out_ptr,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def run_2stage_write_mode(
        rank: Int32,
        grid_x: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        inp_ptr: Int64,
        out_ptrs: Int64,
        tmp_ptrs: Int64,
        stream: Stream = Stream(None),
    ):
        """Launch 2-stage write-mode allreduce (large tensors)."""
        allreduce_2stage_write_mode(
            rank,
            self_sg,
            sg_ptrs,
            inp_ptr,
            out_ptrs,
            tmp_ptrs,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    # Unique function names per (N, dtype_str, world_size, threads) to prevent
    # file-cache collisions (N is baked into kernel body, not the cache key).
    _suffix = f"_N{N}_{dtype_str}_ws{world_size}_t{threads}"
    run_1stage_arr.func.__name__        = f"run_1stage_arr{_suffix}"
    run_2stage_arr.func.__name__        = f"run_2stage_arr{_suffix}"
    run_2stage_write_mode.func.__name__ = f"run_2stage_write_mode{_suffix}"

    return {
        "run_1stage_arr": run_1stage_arr,
        "run_2stage_arr": run_2stage_arr,
        "run_2stage_write_mode": run_2stage_write_mode,
    }
