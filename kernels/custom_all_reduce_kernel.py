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
from flydsl.expr import arith as ea, gpu, range_constexpr, vector as ev, buffer_ops
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
    return buffer_ops.buffer_load(rsrc, elem_off_i32,
                                  vec_width=4, dtype=T.i32)


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
    buffer_ops.buffer_store(val_i32, rsrc, ea.constant(0, type=T.i32),
                            cache_modifier=_CM_CACHED)


def _load_i32_uncached(rsrc):
    """Load i32 bypassing L2 (sc1) via pre-built rsrc descriptor."""
    val = buffer_ops.buffer_load(rsrc, ea.constant(0, type=T.i32),
                                 vec_width=1, dtype=T.i32,
                                 cache_modifier=_CM_SC1)
    rocdl.s_waitcnt(0)
    return val


def _store_i32_uncached(rsrc, val_i32):
    """Store i32 bypassing L1+L2 (sc0+sc1) via pre-built rsrc descriptor."""
    buffer_ops.buffer_store(val_i32, rsrc, ea.constant(0, type=T.i32),
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
    buffer_ops.buffer_store(val_i32, rsrc, ea.constant(0, type=T.i32),
                            cache_modifier=_CM_SC0_SC1)
    rocdl.s_waitcnt(0)


# ---- pointer array helpers -----------------------------------------------

def _pack_i64_vec(values):
    """Pack preloaded i64 values into vector<Nxi64> for contiguous VGPR storage.

    On AMDGPU the subsequent ``ev.extract`` with a dynamic index lowers
    through ``ConvertVectorToLLVM`` to ``llvm.extractelement`` which the
    backend emits as ``v_movrels_b32`` (VGPR-relative addressing, ~3 insns)
    instead of a chained ``arith.select`` costing 2*(N-1) insns.
    """
    vec_type = T.vec(len(values), T.i64)
    return ev.from_elements(vec_type, values)


def _extract_i64(vec, index):
    """Extract i64 from a packed vector by dynamic index (VGPR-relative)."""
    idx = ea.index_cast(T.index, index)
    return ev.extract(vec, dynamic_position=[idx])


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


# ---------------------------------------------------------------------------
# Signal synchronization primitives
# ---------------------------------------------------------------------------

def _signal_start_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int):
    """Start-sync: write start flag to all peers, wait for all to arrive."""
    i32, i64 = T.i32, T.i64

    flag_addr = (self_sg_i64 + ea.constant(_SG_FLAG_OFF_B, type=i64)
                 + bid_i32.extui(i64) * ea.constant(4, type=i64))
    flag_rsrc = _make_rsrc(flag_addr)
    flag = _load_i32_uncached(flag_rsrc) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    start_wait_addr = (self_sg_i64 + ea.constant(_SG_START_OFF_B, type=i64)
                       + lin_lane.extui(i64) * ea.constant(4, type=i64))
    wait_rsrc = _make_rsrc(start_wait_addr)
    lin_rank = bid8 + rank_i32
    start_rank_off = (ea.constant(_SG_START_OFF_B, type=i64)
                      + lin_rank.extui(i64) * ea.constant(4, type=i64))

    is_lane = _u(lane_i32) < ea.constant(ngpus, type=i32)
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
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
            scf.ConditionOp(need_wait, [cur])
        with ir.InsertionPoint(wa):
            scf.YieldOp([_load_i32_uncached(wait_rsrc)])
        scf.YieldOp([])

    gpu.barrier()
    is_t0 = lane_i32 == ea.constant(0, type=i32)
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        _store_i32(flag_rsrc, flag)
        scf.YieldOp([])
    return flag_addr


def _signal_end_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64,
                     ngpus: int):
    """End-sync: write end flag to all peers, wait for all to finish."""

    i32, i64 = T.i32, T.i64

    flag_addr = (self_sg_i64 + ea.constant(_SG_FLAG_OFF_B, type=i64)
                 + bid_i32.extui(i64) * ea.constant(4, type=i64))
    flag_rsrc = _make_rsrc(flag_addr)
    flag = _load_i32_uncached(flag_rsrc) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    end_wait_addr = (self_sg_i64 + ea.constant(_SG_END_OFF_B, type=i64)
                     + lin_lane.extui(i64) * ea.constant(4, type=i64))
    wait_rsrc = _make_rsrc(end_wait_addr)
    lin_rank = bid8 + rank_i32
    end_rank_off = (ea.constant(_SG_END_OFF_B, type=i64)
                    + lin_rank.extui(i64) * ea.constant(4, type=i64))

    is_lane = _u(lane_i32) < ea.constant(ngpus, type=i32)
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
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
            scf.ConditionOp(need_wait, [cur])
        with ir.InsertionPoint(wa):
            nxt = _load_i32_uncached(wait_rsrc)
            _invalidate_l1()
            scf.YieldOp([nxt])
        scf.YieldOp([])

    gpu.barrier()
    is_t0 = lane_i32 == ea.constant(0, type=i32)
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        _store_i32(flag_rsrc, flag)
        scf.YieldOp([])


# ---------------------------------------------------------------------------
# Kernel work group size attribute helper
# ---------------------------------------------------------------------------

def _set_workgroup_size(threads: int):
    """Get the enclosing gpu.func and set rocdl.flat_work_group_size.

    rocdl.reqd_work_group_size is set by the MLIR lowering from the
    known_block_size attribute declared on the @flyc.kernel decorator;
    setting it here as well would create a duplicate attribute in the
    LLVMFuncOp and trigger a DictionaryAttr assertion failure.
    """
    entry_block = ir.InsertionPoint.current.block
    gpu_func_op = entry_block.owner
    gpu_func_op.operation.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{threads},{threads}")
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
        if is_f32:
            v4f32 = T.f32x4
        else:
            v8half = T.bf16x8 if is_bf16 else T.f16x8
            v8f32 = T.vec(8, T.f32)

        gpu_func_op = _set_workgroup_size(threads)

        lane_i32    = ea.index_cast(i32, gpu.thread_id("x"))
        bid_i32     = ea.index_cast(i32, gpu.block_id("x"))
        rank_i32    = rank.ir_value()
        self_sg_i64 = self_sg.ir_value()
        sg_ptrs_i64 = sg_ptrs.ir_value()
        in_ptrs_i64 = in_ptrs.ir_value()
        out_ptr_i64 = out_ptr.ir_value()

        sgs         = [_load_device_ptr(sg_ptrs_i64, ea.constant(i, type=i32)) for i in range(world_size)]
        in_ptrs_arr = [_load_device_ptr(in_ptrs_i64, ea.constant(i, type=i32)) for i in range(world_size)]
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

        tnum_gpu_i32 = ea.constant(tnum_gpu, type=i32)
        warp_id = _u(lane_i32) // tnum_gpu_i32
        lane_id = _u(lane_i32) % tnum_gpu_i32

        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # Grid-stride loop: each warp loads from its assigned rank,
        # then warp 0 reduces and writes output.
        tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
        stride_pack = gpu.grid_dim.x.ir_value() * tnum_gpu_i32

        out_rsrc = _make_rsrc(out_ptr_i64)
        in_rsrc = _make_rsrc(_extract_i64(in_ptrs_vec, warp_id))

        loop = scf.WhileOp([i32, i32], [tid_pack, ea.constant(0, type=i32)])
        bfor = ir.Block.create_at_start(loop.before, [i32, i32])
        afor = ir.Block.create_at_start(loop.after,  [i32, i32])
        with ir.InsertionPoint(bfor):
            p = bfor.arguments[0]
            cond = _u(p) < ea.constant(num_packs, type=i32)
            scf.ConditionOp(cond, [p, bfor.arguments[1]])
        with ir.InsertionPoint(afor):
            p = afor.arguments[0]
            parity = afor.arguments[1]

            off_i32 = p * ea.constant(_ELEMS_PER_PACK, type=i32)
            raw = _load_v4i32(in_rsrc, off_i32)
            sm_base = parity * ea.constant(threads, type=i32)
            sm_idx = ea.index_cast(T.index, sm_base + lane_i32)
            smem_ptr.store(raw, [sm_idx])
            gpu.barrier()

            # Warp 0 reduces across all warps and writes to output
            is_w0 = warp_id == ea.constant(0, type=i32)
            ifw0 = scf.IfOp(is_w0, results_=[], has_else=False)
            with ir.InsertionPoint(ifw0.then_block):
                acc = None
                for wi in range_constexpr(world_size):
                    sm_i_idx = ea.index_cast(
                        T.index, ea.constant(wi, type=i32) * tnum_gpu_i32 + lane_id + sm_base)
                    raw_i = smem_ptr.load([sm_i_idx])
                    if is_f32:
                        vf = ev.bitcast(v4f32, raw_i)
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = ev.bitcast(v8half, raw_i)
                        v32 = v16.extf(v8f32)
                        acc = v32 if acc is None else acc + v32
                if is_f32:
                    out_bits = ev.bitcast(v4i32, acc)
                else:
                    out_bits = ev.bitcast(v4i32, acc.truncf(v8half))
                dst_off_i32 = p * ea.constant(_ELEMS_PER_PACK, type=i32)
                _store_v4i32(out_rsrc, dst_off_i32, out_bits)
                scf.YieldOp([])

            # No barrier 2 needed: parity double-buffer ensures next iteration
            # writes to the opposite smem half, so warp-0 reads from parity_N half
            # are disjoint from all-warp writes to (1-parity_N) half in the next
            # iteration. The barrier at the top of the next iteration guarantees
            # warp-0 finishes before any thread reads the new data.
            scf.YieldOp([p + stride_pack, ea.constant(1, type=i32) - parity])

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
        if is_f32:
            v4f32 = T.f32x4
        else:
            v8half = T.bf16x8 if is_bf16 else T.f16x8
            v8f32 = T.vec(8, T.f32)

        gpu_func_op = _set_workgroup_size(threads)

        lane_i32 = ea.index_cast(i32, gpu.thread_id("x"))
        bid_i32 = ea.index_cast(i32, gpu.block_id("x"))
        rank_i32 = rank.ir_value()
        self_sg_i64 = self_sg.ir_value()
        sg_ptrs_i64 = sg_ptrs.ir_value()
        in_ptrs_i64 = in_ptrs.ir_value()
        tmp_ptrs_i64 = tmp_ptrs.ir_value()
        out_ptr_i64 = out_ptr.ir_value()

        sgs = [_load_device_ptr(sg_ptrs_i64, ea.constant(i, type=i32)) for i in range(world_size)]
        in_ptrs_arr = [_load_device_ptr(in_ptrs_i64, ea.constant(i, type=i32)) for i in range(world_size)]
        tmp_ptrs_arr = [_load_device_ptr(tmp_ptrs_i64, ea.constant(i, type=i32)) for i in range(world_size)]
        in_ptrs_vec = _pack_i64_vec(in_ptrs_arr)

        # Compute pack range for this rank's reduce-scatter partition
        start_p = rank_i32 * ea.constant(part_p, type=i32)
        is_last = rank_i32 == ea.constant(world_size - 1, type=i32)
        end_p = ea.select(is_last, ea.constant(num_packs, type=i32),
                          start_p + ea.constant(part_p, type=i32))

        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        tnum_gpu_i32 = ea.constant(tnum_gpu, type=i32)
        warp_id = _u(lane_i32) // tnum_gpu_i32
        lane_id = _u(lane_i32) % tnum_gpu_i32
        tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
        stride_pack = gpu.grid_dim.x.ir_value() * tnum_gpu_i32

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
            off_i32 = cur * ea.constant(_ELEMS_PER_PACK, type=i32)
            raw = _load_v4i32(in_rsrc, off_i32)
            if smem_base_expr is None:
                sm_idx = ea.index_cast(T.index, lane_i32)
            else:
                sm_idx = ea.index_cast(T.index, smem_base_expr + lane_i32)
            smem_ptr.store(raw, [sm_idx])
            gpu.barrier()  # barrier 1: all warps have written smem

            is_w0 = warp_id == ea.constant(0, type=i32)
            ifw0 = scf.IfOp(is_w0, results_=[], has_else=False)
            with ir.InsertionPoint(ifw0.then_block):
                acc = None
                for wi in range_constexpr(world_size):
                    if smem_base_expr is None:
                        sm_r_idx = ea.index_cast(T.index, ea.constant(wi, type=i32) * tnum_gpu_i32 + lane_id)
                    else:
                        sm_r_idx = ea.index_cast(T.index, ea.constant(wi, type=i32) * tnum_gpu_i32 + lane_id + smem_base_expr)
                    raw_i = smem_ptr.load([sm_r_idx])
                    if is_f32:
                        vf = ev.bitcast(v4f32, raw_i)
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = ev.bitcast(v8half, raw_i)
                        v32 = v16.extf(v8f32)
                        acc = v32 if acc is None else acc + v32
                if is_f32:
                    out_raw = ev.bitcast(v4i32, acc)
                else:
                    out_raw = ev.bitcast(v4i32, acc.truncf(v8half))
                rel_p = cur - start_p
                rel_off_i32 = rel_p * ea.constant(_ELEMS_PER_PACK, type=i32)
                _store_v4i32(tmp_out_rsrc, rel_off_i32, out_raw)
                scf.YieldOp([])

        idx_p = start_p + tid_pack
        if _use_single_buf_2stage:
            # Single buffer: 8KB LDS, 2 barriers per iteration.
            loop1 = scf.WhileOp([i32], [idx_p])
            b1 = ir.Block.create_at_start(loop1.before, [i32])
            a1 = ir.Block.create_at_start(loop1.after, [i32])
            with ir.InsertionPoint(b1):
                cur = b1.arguments[0]
                cond = _u(cur) < end_p
                scf.ConditionOp(cond, [cur])
            with ir.InsertionPoint(a1):
                cur = a1.arguments[0]
                _build_reduce_body(cur, smem_base_expr=None)
                gpu.barrier()  # barrier 2: protect smem before next iter's writes
                scf.YieldOp([cur + stride_pack])
        else:
            # Double buffer: 16KB LDS, 1 barrier per iteration (parity trick).
            # The parity alternates between the two smem halves so warp-0 reads
            # from half-A while all warps write the next pack to half-B.
            loop1 = scf.WhileOp([i32, i32], [idx_p, ea.constant(0, type=i32)])
            b1 = ir.Block.create_at_start(loop1.before, [i32, i32])
            a1 = ir.Block.create_at_start(loop1.after, [i32, i32])
            with ir.InsertionPoint(b1):
                cur = b1.arguments[0]
                cond = _u(cur) < end_p
                scf.ConditionOp(cond, [cur, b1.arguments[1]])
            with ir.InsertionPoint(a1):
                cur = a1.arguments[0]
                parity = a1.arguments[1]
                sm_base = parity * ea.constant(threads, type=i32)
                _build_reduce_body(cur, smem_base_expr=sm_base)
                # No barrier 2: parity ensures next iteration writes to opposite
                # smem half, so warp-0 reads and all-warp writes are disjoint.
                scf.YieldOp([cur + stride_pack, ea.constant(1, type=i32) - parity])

        gpu.barrier()
        _signal_end_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                         self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # ---- Stage 2: all-gather ----
        out_rsrc = _make_rsrc(out_ptr_i64)

        if vec_ok:
            tmp_ptrs_vec = _pack_i64_vec(tmp_ptrs_arr)
            tid_pack2 = bid_i32 * tnum_gpu_i32 + lane_id
            stride_pack2 = gpu.grid_dim.x.ir_value() * tnum_gpu_i32
            tmp_rsrc = _make_rsrc(_extract_i64(tmp_ptrs_vec, warp_id))
            loop2 = scf.WhileOp([i32], [tid_pack2])
            b2 = ir.Block.create_at_start(loop2.before, [i32])
            a2 = ir.Block.create_at_start(loop2.after, [i32])
            with ir.InsertionPoint(b2):
                cur = b2.arguments[0]
                cond = _u(cur) < ea.constant(part_p, type=i32)
                scf.ConditionOp(cond, [cur])
            with ir.InsertionPoint(a2):
                cur = a2.arguments[0]
                sum_rw = rank_i32 + warp_id
                if world_size in {2, 4, 8}:
                    dst_rank = sum_rw & ea.constant(world_size - 1, type=i32)
                else:
                    dst_rank = _u(sum_rw) % ea.constant(world_size, type=i32)
                src_off_i32 = cur * ea.constant(_ELEMS_PER_PACK, type=i32)
                raw = _load_v4i32(tmp_rsrc, src_off_i32)
                dst_pack = dst_rank * ea.constant(part_p, type=i32) + cur
                dst_off_i32 = dst_pack * ea.constant(_ELEMS_PER_PACK, type=i32)
                _store_v4i32(out_rsrc, dst_off_i32, raw)
                scf.YieldOp([cur + stride_pack2])
        else:
            tmp_rsrcs = [_make_rsrc(tmp_ptrs_arr[i]) for i in range(world_size)]
            tid_i32 = bid_i32 * ea.constant(threads, type=i32) + lane_i32
            stride_i32 = gpu.grid_dim.x.ir_value() * ea.constant(threads, type=i32)

            loop2 = scf.WhileOp([i32], [tid_i32])
            b2 = ir.Block.create_at_start(loop2.before, [i32])
            a2 = ir.Block.create_at_start(loop2.after, [i32])
            with ir.InsertionPoint(b2):
                cur = b2.arguments[0]
                cond = _u(cur) < ea.constant(largest_part_p, type=i32)
                scf.ConditionOp(cond, [cur])
            with ir.InsertionPoint(a2):
                cur = a2.arguments[0]
                for p in range_constexpr(world_size):
                    if p == world_size - 1:
                        ok = ea.constant(1, type=T.bool())
                    else:
                        ok = _u(cur) < ea.constant(part_p, type=i32)
                    ifp = scf.IfOp(ok, results_=[], has_else=False)
                    with ir.InsertionPoint(ifp.then_block):
                        src_off_i32 = cur * ea.constant(_ELEMS_PER_PACK, type=i32)
                        raw = _load_v4i32(tmp_rsrcs[p], src_off_i32)
                        dst_pack_idx = ea.constant(p * part_p, type=i32) + cur
                        dst_off_i32 = dst_pack_idx * ea.constant(_ELEMS_PER_PACK, type=i32)
                        _store_v4i32(out_rsrc, dst_off_i32, raw)
                        scf.YieldOp([])
                scf.YieldOp([cur + stride_i32])

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
        if is_f32:
            v4f32 = T.f32x4
        else:
            v8half = T.bf16x8 if is_bf16 else T.f16x8
            v8f32 = T.vec(8, T.f32)

        gpu_func_op = _set_workgroup_size(threads)

        lane_i32 = ea.index_cast(i32, gpu.thread_id("x"))
        bid_i32 = ea.index_cast(i32, gpu.block_id("x"))
        rank_i32 = rank.ir_value()
        self_sg_i64 = self_sg.ir_value()
        sg_ptrs_i64 = sg_ptrs.ir_value()
        inp_ptr_i64 = inp_ptr.ir_value()
        out_ptrs_i64 = out_ptrs.ir_value()
        tmp_ptrs_i64 = tmp_ptrs.ir_value()

        sgs = [_load_device_ptr(sg_ptrs_i64, ea.constant(i, type=i32)) for i in range(world_size)]
        out_ptrs_arr = [_load_device_ptr(out_ptrs_i64, ea.constant(i, type=i32)) for i in range(world_size)]
        tmp_ptrs_arr = [_load_device_ptr(tmp_ptrs_i64, ea.constant(i, type=i32)) for i in range(world_size)]
        tmp_ptrs_vec = _pack_i64_vec(tmp_ptrs_arr)
        out_ptrs_vec = _pack_i64_vec(out_ptrs_arr)

        tnum_gpu_i32 = ea.constant(tnum_gpu, type=i32)
        log2_tnum = int(math.log2(tnum_gpu))
        warp_id = _u(lane_i32) >> ea.constant(log2_tnum, type=i32)
        warp_base = warp_id * tnum_gpu_i32
        lane_id = lane_i32 - warp_base
        tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
        stride_pack = gpu.grid_dim.x.ir_value() * tnum_gpu_i32

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

        start_w = warp_id * ea.constant(part_p, type=i32)
        is_last_w = warp_id == ea.constant(world_size - 1, type=i32)
        end_w_if = scf.IfOp(is_last_w, results_=[i32], has_else=True)
        with ir.InsertionPoint(end_w_if.then_block):
            scf.YieldOp([ea.constant(num_packs, type=i32)])
        with ir.InsertionPoint(end_w_if.else_block):
            scf.YieldOp([start_w + ea.constant(part_p, type=i32)])
        end_w = end_w_if.results[0]

        dst_tmp = _extract_i64(tmp_ptrs_vec, warp_id)
        is_tmp_null = dst_tmp == ea.constant(0, type=i64)
        dst_tmp_low4 = dst_tmp & ea.constant(0xF, type=i64)
        is_tmp_misaligned = dst_tmp_low4 != ea.constant(0, type=i64)
        bad_tmp_addr = is_tmp_null | is_tmp_misaligned
        dst_tmp_rsrc = _make_rsrc(dst_tmp)

        idx_s1 = start_w + tid_pack
        loop_s1 = scf.WhileOp([i32, i32], [idx_s1, stride_pack])
        bs1 = ir.Block.create_at_start(loop_s1.before, [i32, i32])
        as1 = ir.Block.create_at_start(loop_s1.after, [i32, i32])
        with ir.InsertionPoint(bs1):
            cur = bs1.arguments[0]
            cond = _u(cur) < end_w
            scf.ConditionOp(cond, [cur, bs1.arguments[1]])
        with ir.InsertionPoint(as1):
            cur = as1.arguments[0]
            stride_s1 = as1.arguments[1]
            cur_off_i32 = cur * ea.constant(_ELEMS_PER_PACK, type=i32)
            raw = _load_v4i32(inp_rsrc, cur_off_i32)
            rel_idx = cur - start_w
            dst_off = rank_i32 * ea.constant(part_p, type=i32) + rel_idx
            if_tmp_ok = scf.IfOp(bad_tmp_addr, results_=[], has_else=True)
            with ir.InsertionPoint(if_tmp_ok.then_block):
                scf.YieldOp([])
            with ir.InsertionPoint(if_tmp_ok.else_block):
                dst_off_i32 = dst_off * ea.constant(_ELEMS_PER_PACK, type=i32)
                _store_v4i32(dst_tmp_rsrc, dst_off_i32, raw)
                scf.YieldOp([])
            scf.YieldOp([cur + stride_s1, stride_s1])

        # Signal all ranks that stage 1 is complete
        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # ---- Stage 2: reduce local tmp and write to REMOTE outputs ----
        tmp_out_rsrc = _make_rsrc(tmp_out_i64)
        part_p_i32 = ea.constant(part_p, type=i32)
        is_last_rank_s2 = rank_i32 == ea.constant(world_size - 1, type=i32)
        end_s2_if = scf.IfOp(is_last_rank_s2, results_=[i32], has_else=True)
        with ir.InsertionPoint(end_s2_if.then_block):
            scf.YieldOp([ea.constant(largest_part_p, type=i32)])
        with ir.InsertionPoint(end_s2_if.else_block):
            scf.YieldOp([part_p_i32])
        end_s2 = end_s2_if.results[0]

        is_tmpout_null = tmp_out_i64 == ea.constant(0, type=i64)
        tmpout_low4 = tmp_out_i64 & ea.constant(0xF, type=i64)
        is_load_misaligned = tmpout_low4 != ea.constant(0, type=i64)
        bad_load_addr = is_tmpout_null | is_load_misaligned

        dst_ptr = _extract_i64(out_ptrs_vec, warp_id)
        dst_out_rsrc = _make_rsrc(dst_ptr)
        is_out_null = dst_ptr == ea.constant(0, type=i64)
        dst_ptr_low4 = dst_ptr & ea.constant(0xF, type=i64)
        is_out_misaligned = dst_ptr_low4 != ea.constant(0, type=i64)
        bad_out_addr = is_out_null | is_out_misaligned

        loop_s2 = scf.WhileOp([i32, i32], [tid_pack, stride_pack])
        bs2 = ir.Block.create_at_start(loop_s2.before, [i32, i32])
        as2 = ir.Block.create_at_start(loop_s2.after, [i32, i32])
        with ir.InsertionPoint(bs2):
            cur = bs2.arguments[0]
            cond = _u(cur) < end_s2
            scf.ConditionOp(cond, [cur, bs2.arguments[1]])
        with ir.InsertionPoint(as2):
            cur = as2.arguments[0]
            stride_s2 = as2.arguments[1]

            # All warps load their chunk from tmp into smem
            src_off = warp_id * ea.constant(part_p, type=i32) + cur
            src_off_i32 = src_off * ea.constant(_ELEMS_PER_PACK, type=i32)
            raw_if = scf.IfOp(bad_load_addr, results_=[v4i32], has_else=True)
            with ir.InsertionPoint(raw_if.then_block):
                scf.YieldOp([ea.constant_vector(0, v4i32)])
            with ir.InsertionPoint(raw_if.else_block):
                scf.YieldOp([_load_v4i32(tmp_out_rsrc, src_off_i32)])
            raw = raw_if.results[0]

            sm_idx = ea.index_cast(T.index, lane_i32)
            smem_ptr.store(raw, [sm_idx])
            gpu.barrier()

            # Warp 0 reduces across all warps, writes result to res area
            # (smem[threads .. threads+tnum_gpu-1]).  Two-barrier pattern
            # matching aiter: barrier1 guards tmp_smem, barrier2 guards
            # res_smem; between iterations tmp and res are disjoint so no
            # WAR hazard exists.
            is_w0 = warp_id == ea.constant(0, type=i32)
            ifw0 = scf.IfOp(is_w0, results_=[], has_else=False)
            with ir.InsertionPoint(ifw0.then_block):
                acc = None
                for wi in range_constexpr(world_size):
                    sm_i_idx = ea.index_cast(
                        T.index, ea.constant(wi * tnum_gpu, type=i32) + lane_id)
                    raw_i = smem_ptr.load([sm_i_idx])
                    if is_f32:
                        vf = ev.bitcast(v4f32, raw_i)
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = ev.bitcast(v8half, raw_i)
                        v32 = v16.extf(v8f32)
                        acc = v32 if acc is None else acc + v32
                if is_f32:
                    out_raw = ev.bitcast(v4i32, acc)
                else:
                    out_raw = ev.bitcast(v4i32, acc.truncf(v8half))
                res_idx = ea.index_cast(T.index, ea.constant(threads, type=i32) + lane_id)
                smem_ptr.store(out_raw, [res_idx])
                scf.YieldOp([])

            gpu.barrier()

            # All warps read the same reduced result from res area and
            # nontemporal-write to their respective remote output buffers.
            res_read_idx = ea.index_cast(T.index, ea.constant(threads, type=i32) + lane_id)
            reduced_val = smem_ptr.load([res_read_idx])

            dst_out_off = rank_i32 * ea.constant(part_p, type=i32) + cur
            dst_off_i32 = dst_out_off * ea.constant(_ELEMS_PER_PACK, type=i32)

            if_out_ok = scf.IfOp(bad_out_addr, results_=[], has_else=True)
            with ir.InsertionPoint(if_out_ok.then_block):
                scf.YieldOp([])
            with ir.InsertionPoint(if_out_ok.else_block):
                _store_v4i32_nt(dst_out_rsrc, dst_off_i32, reduced_val)
                scf.YieldOp([])

            scf.YieldOp([cur + stride_s2, stride_s2])

        gpu.barrier()
        _signal_end_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                         self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

    # -----------------------------------------------------------------------
    # Host launchers (@flyc.jit)
    # -----------------------------------------------------------------------

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
        allreduce_1stage_arr(rank, self_sg, sg_ptrs, in_ptrs, out_ptr).launch(
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
        allreduce_2stage_arr(rank, self_sg, sg_ptrs, in_ptrs, tmp_ptrs, out_ptr).launch(
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
        allreduce_2stage_write_mode(rank, self_sg, sg_ptrs, inp_ptr, out_ptrs, tmp_ptrs).launch(
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
