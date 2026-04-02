# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL all-reduce kernels using signal protocol for multi-GPU communication.

Implements 1-stage and 2-stage (reduce-scatter + all-gather) kernels.
Signal buffers are hipDeviceMallocUncached (bypasses L1/TCP cache).
Memory ordering uses GFX942 inline assembly for XGMI/HBM visibility.
"""

from __future__ import annotations

import flydsl.compiler as flyc
from flydsl.expr import arith as ea, gpu, range_constexpr, mem_ops, vector as ev
from flydsl.expr.typing import T, Int32, Int64, Stream
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from kernels.custom_all_reduce import _KMAXBLOCKS as _MAX_BLOCKS

# Signal buffer layout offsets (bytes), derived from _MAX_BLOCKS.
# start[_MAX_BLOCKS][8] of uint32 | end[_MAX_BLOCKS][8] of uint32 | flag[_MAX_BLOCKS] of uint32
_SG_START_OFF_B = 0
_SG_END_OFF_B = _MAX_BLOCKS * 8 * 4            # 2560 when _MAX_BLOCKS=80
_SG_FLAG_OFF_B = _MAX_BLOCKS * 8 * 4 * 2       # 5120 when _MAX_BLOCKS=80


# ---------------------------------------------------------------------------
# Element type helpers
# ---------------------------------------------------------------------------

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
    d = (dtype_str or "").strip().lower()
    if d in {"f32", "fp32"}:
        return 4
    if d in {"f16", "fp16", "bf16"}:
        return 8
    raise ValueError(f"unsupported dtype_str: {dtype_str!r}")


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
    flag = mem_ops.load_i32_uncached(flag_addr) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    start_wait_addr = (self_sg_i64 + ea.constant(_SG_START_OFF_B, type=i64)
                       + lin_lane.extui(i64) * ea.constant(4, type=i64))
    lin_rank = bid8 + rank_i32
    start_rank_off = (ea.constant(_SG_START_OFF_B, type=i64)
                      + lin_rank.extui(i64) * ea.constant(4, type=i64))

    is_lane = _u(lane_i32) < ea.constant(ngpus, type=i32)
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = ea.select_by_index(lane_i32, sgs_i64)
        mem_ops.store_i32_uncached_flush(peer_sg + start_rank_off, flag)
        init_cur = mem_ops.load_i32_uncached(start_wait_addr)
        w = scf.WhileOp([i32], [init_cur])
        wb = ir.Block.create_at_start(w.before, [i32])
        wa = ir.Block.create_at_start(w.after, [i32])
        with ir.InsertionPoint(wb):
            cur = wb.arguments[0]
            need_wait = _u(cur) < flag
            scf.ConditionOp(need_wait, [cur])
        with ir.InsertionPoint(wa):
            scf.YieldOp([mem_ops.load_i32_uncached(start_wait_addr)])
        scf.YieldOp([])

    gpu.barrier()
    is_t0 = lane_i32 == ea.constant(0, type=i32)
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        mem_ops.store_i32(flag_addr, flag)
        scf.YieldOp([])
    return flag_addr


def _signal_end_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64,
                     ngpus: int, need_wbl2: bool = False):
    """End-sync: write end flag to all peers, wait for all to finish.

    Args:
        need_wbl2: True  → use st_xgmi_u32 (buffer_wbl2 + signal store).
                           Required after cached stores (st_global_16b) so
                           that L2 dirty lines reach HBM before the signal.
                   False → use st_signal_u32 (signal store only, no wbl2).
                           For nt data stores (st_nt_16b) which already bypass
                           L2; uses ATOMIC_RELAXED + MEMORY_SCOPE_SYSTEM.
    """


    i32, i64 = T.i32, T.i64

    gpu.barrier()
    flag_addr = (self_sg_i64 + ea.constant(_SG_FLAG_OFF_B, type=i64)
                 + bid_i32.extui(i64) * ea.constant(4, type=i64))
    flag = mem_ops.load_i32_uncached(flag_addr) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    end_wait_addr = (self_sg_i64 + ea.constant(_SG_END_OFF_B, type=i64)
                     + lin_lane.extui(i64) * ea.constant(4, type=i64))
    lin_rank = bid8 + rank_i32
    end_rank_off = (ea.constant(_SG_END_OFF_B, type=i64)
                    + lin_rank.extui(i64) * ea.constant(4, type=i64))

    is_lane = _u(lane_i32) < ea.constant(ngpus, type=i32)
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = ea.select_by_index(lane_i32, sgs_i64)
        if need_wbl2:
            mem_ops.store_i32_uncached_flush(peer_sg + end_rank_off, flag)
        else:
            mem_ops.store_i32_uncached(peer_sg + end_rank_off, flag)
        init_cur = mem_ops.load_i32_uncached(end_wait_addr)
        w = scf.WhileOp([i32], [init_cur])
        wb = ir.Block.create_at_start(w.before, [i32])
        wa = ir.Block.create_at_start(w.after, [i32])
        with ir.InsertionPoint(wb):
            cur = wb.arguments[0]
            need_wait = _u(cur) < flag
            scf.ConditionOp(need_wait, [cur])
        with ir.InsertionPoint(wa):
            nxt = mem_ops.load_i32_uncached(end_wait_addr)
            mem_ops.invalidate_l1()
            scf.YieldOp([nxt])
        scf.YieldOp([])

    gpu.barrier()
    is_t0 = lane_i32 == ea.constant(0, type=i32)
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        mem_ops.store_i32(flag_addr, flag)
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
        idx = ir.IndexType.get()
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

        sgs         = [mem_ops.load_device_ptr(sg_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]
        in_ptrs_arr = [mem_ops.load_device_ptr(in_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]

        smem_sym = f"allreduce_1s_smem_ws{world_size}_t{threads}"
        n_smem = 2 * threads
        allocator = SmemAllocator(None, global_sym_name=smem_sym)
        smem_off = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_off + n_smem * 16
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

            # Each warp loads data from its rank into shared memory
            in_base = ea.select_by_index(warp_id, in_ptrs_arr)
            off16 = p.extui(i64) * ea.constant(16, type=i64)
            raw = mem_ops.load_v4i32(in_base + off16)
            sm_base = parity * ea.constant(threads, type=i32)
            sm_idx = ea.index_cast(idx, sm_base + lane_i32)
            smem_ptr.store(raw, [sm_idx])
            gpu.barrier()

            # Warp 0 reduces across all warps and writes to output
            is_w0 = warp_id == ea.constant(0, type=i32)
            ifw0 = scf.IfOp(is_w0, results_=[], has_else=False)
            with ir.InsertionPoint(ifw0.then_block):
                acc = None
                for wi in range_constexpr(world_size):
                    sm_i_idx = ea.index_cast(
                        idx, ea.constant(wi, type=i32) * tnum_gpu_i32 + lane_id + sm_base)
                    raw_i = smem_ptr.load([sm_i_idx])
                    if is_f32:
                        vf = raw_i.bitcast(v4f32)
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = ev.bitcast(v8half, raw_i)
                        v32 = v16.extf(v8f32)
                        acc = v32 if acc is None else acc + v32
                if is_f32:
                    out_bits = acc.bitcast(v4i32)
                else:
                    out_bits = ev.bitcast(v4i32, acc.truncf(v8half))
                dst_off = p.extui(i64) * ea.constant(16, type=i64)
                mem_ops.store_v4i32(out_ptr_i64 + dst_off, out_bits)
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
        idx = ir.IndexType.get()
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

        sgs = [mem_ops.load_device_ptr(sg_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]
        in_ptrs_arr = [mem_ops.load_device_ptr(in_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]
        tmp_ptrs_arr = [mem_ops.load_device_ptr(tmp_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]

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
        allocator.ptr = smem_off + smem_slots * 16
        with ir.InsertionPoint.at_block_begin(gpu_func_op.operation.block):
            allocator.finalize()
        smem_ptr = SmemPtr(allocator.get_base(), smem_off, v4i32, shape=(smem_slots,))
        smem_ptr.get()
        tmp_out_i64 = tmp_ptrs_arr[0]

        # ---- Stage 1: reduce-scatter ----
        # Two implementations selected at compile time via _use_single_buf_2stage:
        #   Single-buffer (large tensor): 8KB LDS, 2 barriers/iter, higher occupancy.
        #   Double-buffer (small tensor): 16KB LDS, 1 barrier/iter (parity trick).

        def _build_reduce_body(cur, smem_base_expr=None):
            """Emit reduce body: load → smem → barrier1 → warp0 reduce → [barrier2]."""
            in_base = ea.select_by_index(warp_id, in_ptrs_arr)
            raw = mem_ops.load_v4i32(in_base + cur.extui(i64) * ea.constant(16, type=i64))
            if smem_base_expr is None:
                sm_idx = ea.index_cast(idx, lane_i32)
            else:
                sm_idx = ea.index_cast(idx, smem_base_expr + lane_i32)
            smem_ptr.store(raw, [sm_idx])
            gpu.barrier()  # barrier 1: all warps have written smem

            is_w0 = warp_id == ea.constant(0, type=i32)
            ifw0 = scf.IfOp(is_w0, results_=[], has_else=False)
            with ir.InsertionPoint(ifw0.then_block):
                acc = None
                for wi in range_constexpr(world_size):
                    if smem_base_expr is None:
                        sm_r_idx = ea.index_cast(idx, ea.constant(wi, type=i32) * tnum_gpu_i32 + lane_id)
                    else:
                        sm_r_idx = ea.index_cast(idx, ea.constant(wi, type=i32) * tnum_gpu_i32 + lane_id + smem_base_expr)
                    raw_i = smem_ptr.load([sm_r_idx])
                    if is_f32:
                        vf = raw_i.bitcast(v4f32)
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = ev.bitcast(v8half, raw_i)
                        v32 = v16.extf(v8f32)
                        acc = v32 if acc is None else acc + v32
                if is_f32:
                    out_raw = acc.bitcast(v4i32)
                else:
                    out_raw = ev.bitcast(v4i32, acc.truncf(v8half))
                rel_p = cur - start_p
                mem_ops.store_v4i32(tmp_out_i64 + rel_p.extui(i64) * ea.constant(16, type=i64),
                        out_raw)
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

        _signal_end_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                         self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # ---- Stage 2: all-gather ----
        if vec_ok:
            tid_pack2 = bid_i32 * tnum_gpu_i32 + lane_id
            stride_pack2 = gpu.grid_dim.x.ir_value() * tnum_gpu_i32

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
                tmp_base = ea.select_by_index(warp_id, tmp_ptrs_arr)
                raw = mem_ops.load_v4i32(tmp_base + cur.extui(i64) * ea.constant(16, type=i64))
                dst_pack = dst_rank * ea.constant(part_p, type=i32) + cur
                mem_ops.store_v4i32(out_ptr_i64 + dst_pack.extui(i64) * ea.constant(16, type=i64),
                        raw)
                scf.YieldOp([cur + stride_pack2])
        else:
            # Non-vectorized fallback (world_size=6 or num_packs % world_size != 0)
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
                        src_off = cur.extui(i64) * ea.constant(16, type=i64)
                        raw = mem_ops.load_v4i32(tmp_ptrs_arr[p] + src_off)
                        dst_pack_idx = ea.constant(p * part_p, type=i32) + cur
                        dst_off = dst_pack_idx.extui(i64) * ea.constant(16, type=i64)
                        mem_ops.store_v4i32(out_ptr_i64 + dst_off, raw)
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
        import math
    

        i32, i64 = T.i32, T.i64
        idx = ir.IndexType.get()
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

        sgs = [mem_ops.load_device_ptr(sg_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]
        out_ptrs_arr = [mem_ops.load_device_ptr(out_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]

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
        allocator_wm.ptr = smem_wm_off + n_smem_wm * 16
        with ir.InsertionPoint.at_block_begin(gpu_func_op.operation.block):
            allocator_wm.finalize()
        smem_ptr = SmemPtr(allocator_wm.get_base(), smem_wm_off, v4i32, shape=(n_smem_wm,))
        smem_ptr.get()
        tmp_out_i64 = mem_ops.load_device_ptr(tmp_ptrs_i64, rank_i32)

        # ---- Stage 1: scatter local input to REMOTE tmp buffers ----
        start_w = warp_id * ea.constant(part_p, type=i32)
        is_last_w = warp_id == ea.constant(world_size - 1, type=i32)
        end_w_if = scf.IfOp(is_last_w, results_=[i32], has_else=True)
        with ir.InsertionPoint(end_w_if.then_block):
            scf.YieldOp([ea.constant(num_packs, type=i32)])
        with ir.InsertionPoint(end_w_if.else_block):
            scf.YieldOp([start_w + ea.constant(part_p, type=i32)])
        end_w = end_w_if.results[0]

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
            raw = mem_ops.load_v4i32(inp_ptr_i64 + cur.extui(i64) * ea.constant(16, type=i64))
            rel_idx = cur - start_w
            dst_off = rank_i32 * ea.constant(part_p, type=i32) + rel_idx
            dst_tmp = mem_ops.load_device_ptr(tmp_ptrs_i64, warp_id)
            tmp_addr = dst_tmp + dst_off.extui(i64) * ea.constant(16, type=i64)
            is_tmp_null = dst_tmp == ea.constant(0, type=i64)
            tmp_low4 = tmp_addr & ea.constant(0xF, type=i64)
            is_tmp_misaligned = tmp_low4 != ea.constant(0, type=i64)
            bad_tmp_addr = is_tmp_null | is_tmp_misaligned
            if_tmp_ok = scf.IfOp(bad_tmp_addr, results_=[], has_else=True)
            with ir.InsertionPoint(if_tmp_ok.then_block):
                scf.YieldOp([])
            with ir.InsertionPoint(if_tmp_ok.else_block):
                mem_ops.store_v4i32(tmp_addr, raw)
                scf.YieldOp([])
            scf.YieldOp([cur + stride_s1, stride_s1])

        # Signal all ranks that stage 1 is complete
        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # ---- Stage 2: reduce local tmp and write to REMOTE outputs ----
        part_p_i32 = ea.constant(part_p, type=i32)
        # The last rank's output partition has largest_part_p elements
        # (= part_p + num_packs % world_size).  Use a runtime branch so that
        # when num_packs is evenly divisible the overhead is minimal (same value).
        is_last_rank_s2 = rank_i32 == ea.constant(world_size - 1, type=i32)
        end_s2_if = scf.IfOp(is_last_rank_s2, results_=[i32], has_else=True)
        with ir.InsertionPoint(end_s2_if.then_block):
            scf.YieldOp([ea.constant(largest_part_p, type=i32)])
        with ir.InsertionPoint(end_s2_if.else_block):
            scf.YieldOp([part_p_i32])
        end_s2 = end_s2_if.results[0]
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

            src_off = warp_id * ea.constant(part_p, type=i32) + cur
            load_addr = tmp_out_i64 + src_off.extui(i64) * ea.constant(16, type=i64)
            is_tmpout_null = tmp_out_i64 == ea.constant(0, type=i64)
            load_low4 = load_addr & ea.constant(0xF, type=i64)
            is_load_misaligned = load_low4 != ea.constant(0, type=i64)
            bad_load_addr = is_tmpout_null | is_load_misaligned
            raw_if = scf.IfOp(bad_load_addr, results_=[v4i32], has_else=True)
            with ir.InsertionPoint(raw_if.then_block):
                scf.YieldOp([ea.constant_vector(0, v4i32)])
            with ir.InsertionPoint(raw_if.else_block):
                scf.YieldOp([mem_ops.load_v4i32(load_addr)])
            raw = raw_if.results[0]

            sm_idx = ea.index_cast(idx, lane_i32)
            smem_ptr.store(raw, [sm_idx])
            gpu.barrier()

            warp_id_local = _u(lane_i32) >> ea.constant(log2_tnum, type=i32)
            lane_id_local = lane_i32 - warp_id_local * ea.constant(tnum_gpu, type=i32)

            raw_vals = []
            for wi in range_constexpr(world_size):
                sm_i_idx = ea.index_cast(idx, ea.constant(wi * tnum_gpu, type=i32) + lane_id_local)
                raw_vals.append(smem_ptr.load([sm_i_idx]))

            acc = None
            for wi in range_constexpr(world_size):
                raw_i = raw_vals[wi]
                if is_f32:
                    vf = raw_i.bitcast(v4f32)
                    acc = vf if acc is None else acc + vf
                else:
                    v16 = ev.bitcast(v8half, raw_i)
                    v32 = v16.extf(v8f32)
                    acc = v32 if acc is None else acc + v32
            if is_f32:
                out_raw = acc.bitcast(v4i32)
            else:
                out_raw = ev.bitcast(v4i32, acc.truncf(v8half))

            dst_out_off = rank_i32 * ea.constant(part_p, type=i32) + cur
            dst_byte_off = dst_out_off.extui(i64) * ea.constant(16, type=i64)

            # Each warp writes its reduced partition directly to the target
            # output via flat_store_dwordx4 nt. The nt hint bypasses L1/L2
            # and works for all memory types (including IPC-mapped addresses).
            dst_ptr = out_ptrs_arr[0]
            for w in range_constexpr(1, world_size):
                is_warp_w = warp_id_local == ea.constant(w, type=i32)
                dst_ptr = ea.select(is_warp_w, out_ptrs_arr[w], dst_ptr)
            out_addr = dst_ptr + dst_byte_off
            is_out_null = dst_ptr == ea.constant(0, type=i64)
            out_low4 = out_addr & ea.constant(0xF, type=i64)
            is_out_misaligned = out_low4 != ea.constant(0, type=i64)
            bad_out_addr = is_out_null | is_out_misaligned
            if_out_ok = scf.IfOp(bad_out_addr, results_=[], has_else=True)
            with ir.InsertionPoint(if_out_ok.then_block):
                scf.YieldOp([])
            with ir.InsertionPoint(if_out_ok.else_block):
                mem_ops.store_v4i32_nt(out_addr, out_raw)
                scf.YieldOp([])

            scf.YieldOp([cur + stride_s2, stride_s2])

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
