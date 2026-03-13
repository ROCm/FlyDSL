"""FlyDSL all-reduce kernels using signal protocol for multi-GPU communication.

Implements 2-stage (reduce-scatter + all-gather) kernels.
Signal buffers are hipDeviceMallocUncached (bypasses L1/TCP cache).
Memory ordering uses GFX942 inline assembly for XGMI/HBM visibility.
"""

from __future__ import annotations

import flydsl.compiler as flyc
from flydsl.expr.typing import Int32, Int64, Stream
from flydsl._mlir import ir
from flydsl.expr import arith as ea  # Registers ArithValue caster; enables operator overloading
from flydsl.expr import range_constexpr  # Compile-time loop unrolling in @flyc.kernel

# Signal buffer layout offsets (bytes) within the per-rank signal buffer
_SG_START_OFF_B = 0
_SG_END_OFF_B = 2560
_SG_FLAG_OFF_B = 5120

_MAX_BLOCKS = 80


# ---------------------------------------------------------------------------
# Element type helpers
# ---------------------------------------------------------------------------

def _elem_type(dtype_str: str) -> ir.Type:
    d = (dtype_str or "").strip().lower()
    if d in {"f16", "fp16"}:
        return ir.F16Type.get()
    if d in {"f32", "fp32"}:
        return ir.F32Type.get()
    raise ValueError(f"unsupported dtype_str: {dtype_str!r}")


def _pack_elems(dtype_str: str) -> int:
    d = (dtype_str or "").strip().lower()
    if d in {"f32", "fp32"}:
        return 4
    if d in {"f16", "fp16"}:
        return 8
    raise ValueError(f"unsupported dtype_str: {dtype_str!r}")


def _i32() -> ir.Type:
    return ir.IntegerType.get_signless(32)


def _i64() -> ir.Type:
    return ir.IntegerType.get_signless(64)


# ---------------------------------------------------------------------------
# Low-level uncached global memory primitives (operate on raw ir.Value / ArithValue)
# All functions assume ArithValue caster is registered (ea import at top).
# ---------------------------------------------------------------------------

def _ld_u32(addr_i64):
    """Load u32 from uncached global address (signal buffer, bypasses L1)."""
    from flydsl._mlir.dialects import llvm, rocdl
    v = llvm.InlineAsmOp(
        _i32(), [addr_i64],
        "global_load_dword $0, $1, off sc1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


def _st_u32_xgmi(addr_i64, val_i32):
    """Store u32 to peer GPU signal buffer, flushing L2 for XGMI visibility."""
    from flydsl._mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def _ld_16b(addr_i64):
    """Load 16 bytes (vector<4xi32>) from global address."""
    from flydsl._mlir.dialects import llvm, rocdl
    v4i32 = ir.VectorType.get([4], _i32())
    v = llvm.InlineAsmOp(
        v4i32, [addr_i64],
        "flat_load_dwordx4 $0, $1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


def _st_16b(addr_i64, v4i32_val):
    """Store 16 bytes (vector<4xi32>) to global address."""
    from flydsl._mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(
        None, [addr_i64, v4i32_val],
        "global_store_dwordx4 $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def _st_u32_local(addr_i64, val_i32):
    """Store u32 to local (same-GPU) signal buffer, no XGMI flush needed."""
    from flydsl._mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


# ---------------------------------------------------------------------------
# Pointer arithmetic helpers
# ---------------------------------------------------------------------------

def _select_by_lane(lane_i32, vals_i64):
    """Select one of vals_i64[0..7] by lane index via chained arith.select."""
    from flydsl._mlir.dialects import arith
    i32 = _i32()
    out = vals_i64[0]
    for i in range(1, len(vals_i64)):
        pred = arith.CmpIOp(arith.CmpIPredicate.eq, lane_i32, ea.constant(i, type=i32)).result
        out = arith.SelectOp(pred, vals_i64[i], out).result
    return out


def _load_ptr_from_array(array_base_i64, index_i32):
    """Load an i64 pointer from a device-side array at index_i32."""
    from flydsl._mlir.dialects import llvm, arith
    i64 = _i64()
    elem_addr = array_base_i64 + arith.ExtUIOp(i64, index_i32).result * ea.constant(8, type=i64)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), elem_addr).result
    return llvm.LoadOp(i64, ptr).result


# ---------------------------------------------------------------------------
# Signal synchronization primitives
# ---------------------------------------------------------------------------

def _spin_wait_ge(addr_i64, target_u32):
    """Spin-wait until *addr >= target (uncachable signal buffer)."""
    from flydsl._mlir.dialects import arith, scf
    i32 = _i32()
    init_cur = _ld_u32(addr_i64)
    w = scf.WhileOp([i32], [init_cur])
    before = ir.Block.create_at_start(w.before, [i32])
    after = ir.Block.create_at_start(w.after, [i32])
    with ir.InsertionPoint(before):
        cur = before.arguments[0]
        need_wait = arith.CmpIOp(arith.CmpIPredicate.ult, cur, target_u32).result
        scf.ConditionOp(need_wait, [cur])
    with ir.InsertionPoint(after):
        scf.YieldOp([_ld_u32(addr_i64)])


def _signal_start_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int):
    """Start-sync: write start flag to all peers, wait for all to arrive."""
    from flydsl._mlir.dialects import arith, gpu, scf
    i32, i64 = _i32(), _i64()

    flag_addr = (self_sg_i64 + ea.constant(_SG_FLAG_OFF_B, type=i64)
                 + arith.ExtUIOp(i64, bid_i32).result * ea.constant(4, type=i64))
    flag = _ld_u32(flag_addr) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    start_wait_addr = (self_sg_i64 + ea.constant(_SG_START_OFF_B, type=i64)
                       + arith.ExtUIOp(i64, lin_lane).result * ea.constant(4, type=i64))
    lin_rank = bid8 + rank_i32
    start_rank_off = (ea.constant(_SG_START_OFF_B, type=i64)
                      + arith.ExtUIOp(i64, lin_rank).result * ea.constant(4, type=i64))

    is_lane = arith.CmpIOp(arith.CmpIPredicate.ult, lane_i32, ea.constant(ngpus, type=i32)).result
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _select_by_lane(lane_i32, sgs_i64)
        _st_u32_xgmi(peer_sg + start_rank_off, flag)
        _spin_wait_ge(start_wait_addr, flag)
        scf.YieldOp([])

    gpu.BarrierOp()
    is_t0 = arith.CmpIOp(arith.CmpIPredicate.eq, lane_i32, ea.constant(0, type=i32)).result
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        _st_u32_local(flag_addr, flag)
        scf.YieldOp([])
    return flag_addr


def _signal_end_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int):
    """End-sync: write end flag to all peers, wait for all to finish."""
    from flydsl._mlir.dialects import arith, gpu, scf
    i32, i64 = _i32(), _i64()

    gpu.BarrierOp()
    flag_addr = (self_sg_i64 + ea.constant(_SG_FLAG_OFF_B, type=i64)
                 + arith.ExtUIOp(i64, bid_i32).result * ea.constant(4, type=i64))
    flag = _ld_u32(flag_addr) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    end_wait_addr = (self_sg_i64 + ea.constant(_SG_END_OFF_B, type=i64)
                     + arith.ExtUIOp(i64, lin_lane).result * ea.constant(4, type=i64))
    lin_rank = bid8 + rank_i32
    end_rank_off = (ea.constant(_SG_END_OFF_B, type=i64)
                    + arith.ExtUIOp(i64, lin_rank).result * ea.constant(4, type=i64))

    is_lane = arith.CmpIOp(arith.CmpIPredicate.ult, lane_i32, ea.constant(ngpus, type=i32)).result
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _select_by_lane(lane_i32, sgs_i64)
        _st_u32_xgmi(peer_sg + end_rank_off, flag)
        _spin_wait_ge(end_wait_addr, flag)
        scf.YieldOp([])

    gpu.BarrierOp()
    is_t0 = arith.CmpIOp(arith.CmpIPredicate.eq, lane_i32, ea.constant(0, type=i32)).result
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        _st_u32_local(flag_addr, flag)
        scf.YieldOp([])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_allreduce_kernels(*, N: int, dtype_str: str, world_size: int, threads: int = 256):
    """Build and return compiled allreduce launcher functions.

    Captures compile-time constants as closures, returns a dict with:
      "run_2stage_arr"        — CUDAGraph-compatible 2-stage allreduce
      "run_2stage_write_mode" — Large-tensor 2-stage allreduce (N > 512*4096, ws=8)

    Args:
        N:          Total number of elements to reduce.
        dtype_str:  "f16" or "f32".
        world_size: Number of GPUs (2, 4, 6, or 8).
        threads:    Threads per block (must be divisible by world_size).
    """
    if world_size not in {2, 4, 6, 8}:
        raise ValueError(f"world_size must be one of {{2,4,6,8}}, got {world_size}")
    if threads <= 0 or threads % world_size != 0:
        raise ValueError(f"threads={threads} must be > 0 and divisible by world_size={world_size}")

    pack_elems = _pack_elems(dtype_str)
    if N <= 0 or N % pack_elems != 0:
        raise ValueError(f"N={N} must be > 0 and a multiple of pack_elems={pack_elems}")

    # Compile-time constants captured by closures
    num_packs = N // pack_elems
    part_p = num_packs // world_size
    largest_part_p = part_p + (num_packs % world_size)
    tnum_gpu = threads // world_size
    is_f32 = dtype_str.lower().strip() in {"f32", "fp32"}
    # Vectorized gather path: requires perfect partition + no world_size=6
    vec_ok = (num_packs % world_size == 0) and (world_size != 6)

    # -----------------------------------------------------------------------
    # GPU Kernel: 2-stage arr (reduce-scatter + all-gather)
    # Loads pointer arrays inside the kernel -> CUDAGraph-compatible.
    # Args: rank, self_sg, sg_ptrs, in_ptrs, tmp_ptrs, out_ptr  (all raw i64)
    # -----------------------------------------------------------------------
    @flyc.kernel
    def allreduce_2stage_arr(
        rank: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        in_ptrs: Int64,
        tmp_ptrs: Int64,
        out_ptr: Int64,
    ):
        from flydsl._mlir.dialects import arith, gpu, memref, scf, vector
        from flydsl.expr.buffer_ops import _unwrap_value

        # Set required work group size attributes (needed for correct LDS allocation)
        _entry_block = ir.InsertionPoint.current.block
        _gpu_func_op = _entry_block.owner
        _gpu_func_op.operation.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([threads, 1, 1])
        _gpu_func_op.operation.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{threads},{threads}")

        # MLIR types (created inside JIT context)
        i32 = _i32()
        i64 = _i64()
        idx = ir.IndexType.get()
        v4i32 = ir.VectorType.get([4], i32)
        lds_space = ir.Attribute.parse("#gpu.address_space<workgroup>")
        smem_ty = ir.MemRefType.get([2 * threads], v4i32, memory_space=lds_space)
        if is_f32:
            v4f32 = ir.VectorType.get([4], ir.F32Type.get())
        else:
            v8f16 = ir.VectorType.get([8], ir.F16Type.get())
            v8f32 = ir.VectorType.get([8], ir.F32Type.get())

        # Unwrap FlyDSL typed params -> raw ir.Value (with ArithValue operators)
        lane_i32 = ea.index_cast(i32, gpu.thread_id("x"))
        bid_i32 = ea.index_cast(i32, gpu.block_id("x"))
        rank_i32 = _unwrap_value(rank)
        self_sg_i64 = _unwrap_value(self_sg)
        sg_ptrs_i64 = _unwrap_value(sg_ptrs)
        in_ptrs_i64 = _unwrap_value(in_ptrs)
        tmp_ptrs_i64 = _unwrap_value(tmp_ptrs)
        out_ptr_i64 = _unwrap_value(out_ptr)

        # Load pointer arrays from device memory
        sgs = [_load_ptr_from_array(sg_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]
        in_ptrs_arr = [_load_ptr_from_array(in_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]
        tmp_ptrs_arr = [_load_ptr_from_array(tmp_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]

        # Compute pack range for this rank's reduce-scatter partition
        start_p = rank_i32 * ea.constant(part_p, type=i32)
        is_last = arith.CmpIOp(arith.CmpIPredicate.eq, rank_i32,
                               ea.constant(world_size - 1, type=i32)).result
        end_p = arith.SelectOp(is_last, ea.constant(num_packs, type=i32),
                               start_p + ea.constant(part_p, type=i32)).result

        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        tnum_gpu_i32 = ea.constant(tnum_gpu, type=i32)
        warp_id = arith.DivUIOp(lane_i32, tnum_gpu_i32).result
        lane_id = arith.RemUIOp(lane_i32, tnum_gpu_i32).result
        tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
        stride_pack = ea.index_cast(i32, gpu.grid_dim("x")) * tnum_gpu_i32

        # Declare LDS (shared memory) as a GlobalOp in the GPU module scope.
        # memref.GlobalOp with #gpu.address_space<workgroup> maps to LDS.
        # Insert it before the current gpu.func in the GPU module body.
        smem_sym = f"allreduce_smem_ws{world_size}_t{threads}"
        _entry_block = ir.InsertionPoint.current.block
        _gpu_func_op = _entry_block.owner
        _gpu_module_body_block = _gpu_func_op.operation.block
        with ir.InsertionPoint.at_block_begin(_gpu_module_body_block):
            memref.GlobalOp(
                sym_name=ir.StringAttr.get(smem_sym),
                type_=smem_ty,
                initial_value=None,
                constant=False,
                alignment=16,
            )
        smem = memref.GetGlobalOp(smem_ty, smem_sym).result
        tmp_out_i64 = tmp_ptrs_arr[0]

        # ---- Stage 1: reduce-scatter ----
        # Each warp reads its partition, reduces via smem, writes to local tmp
        idx_p = start_p + tid_pack
        loop1 = scf.WhileOp([i32, i32], [idx_p, ea.constant(0, type=i32)])
        b1 = ir.Block.create_at_start(loop1.before, [i32, i32])
        a1 = ir.Block.create_at_start(loop1.after, [i32, i32])
        with ir.InsertionPoint(b1):
            cur = b1.arguments[0]
            cond = arith.CmpIOp(arith.CmpIPredicate.ult, cur, end_p).result
            scf.ConditionOp(cond, [cur, b1.arguments[1]])
        with ir.InsertionPoint(a1):
            cur = a1.arguments[0]
            parity = a1.arguments[1]
            # Each warp loads its rank's input slice
            in_base = _select_by_lane(warp_id, in_ptrs_arr)
            raw = _ld_16b(in_base + arith.ExtUIOp(i64, cur).result * ea.constant(16, type=i64))
            sm_base = parity * ea.constant(threads, type=i32)
            sm_idx = ea.index_cast(idx, sm_base + lane_i32)
            memref.StoreOp(raw, smem, [sm_idx])
            gpu.BarrierOp()

            # Warp 0 reduces across all warps and stores to tmp
            is_w0 = arith.CmpIOp(arith.CmpIPredicate.eq, warp_id, ea.constant(0, type=i32)).result
            ifw0 = scf.IfOp(is_w0, results_=[], has_else=False)
            with ir.InsertionPoint(ifw0.then_block):
                acc = None
                for wi in range_constexpr(world_size):
                    sm_i_idx = ea.index_cast(
                        idx, ea.constant(wi, type=i32) * tnum_gpu_i32 + lane_id + sm_base)
                    raw_i = memref.LoadOp(smem, [sm_i_idx]).result
                    if is_f32:
                        vf = vector.BitCastOp(v4f32, raw_i).result
                        acc = vf if acc is None else acc + vf
                    else:
                        v16 = vector.BitCastOp(v8f16, raw_i).result
                        v32 = arith.ExtFOp(v8f32, v16).result
                        acc = v32 if acc is None else acc + v32
                if is_f32:
                    out_raw = vector.BitCastOp(v4i32, acc).result
                else:
                    from flydsl._mlir.dialects import llvm
                    out_raw = llvm.BitcastOp(v4i32, acc.truncf(v8f16)).result
                rel_p = cur - start_p
                _st_16b(tmp_out_i64 + arith.ExtUIOp(i64, rel_p).result * ea.constant(16, type=i64),
                        out_raw)
                scf.YieldOp([])

            scf.YieldOp([cur + stride_pack, ea.constant(1, type=i32) - parity])

        _signal_end_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                         self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # ---- Stage 2: all-gather ----
        if vec_ok:
            # Vectorized: each warp reads its own tmp slice and writes to the
            # destination rank's output partition (rotated assignment for bandwidth)
            tid_pack2 = bid_i32 * tnum_gpu_i32 + lane_id
            stride_pack2 = ea.index_cast(i32, gpu.grid_dim("x")) * tnum_gpu_i32

            loop2 = scf.WhileOp([i32], [tid_pack2])
            b2 = ir.Block.create_at_start(loop2.before, [i32])
            a2 = ir.Block.create_at_start(loop2.after, [i32])
            with ir.InsertionPoint(b2):
                cur = b2.arguments[0]
                cond = arith.CmpIOp(arith.CmpIPredicate.ult, cur,
                                    ea.constant(part_p, type=i32)).result
                scf.ConditionOp(cond, [cur])
            with ir.InsertionPoint(a2):
                cur = a2.arguments[0]
                sum_rw = arith.AddIOp(rank_i32, warp_id).result
                if world_size in {2, 4, 8}:
                    dst_rank = arith.AndIOp(sum_rw, ea.constant(world_size - 1, type=i32)).result
                else:
                    dst_rank = arith.RemUIOp(sum_rw, ea.constant(world_size, type=i32)).result
                tmp_base = _select_by_lane(warp_id, tmp_ptrs_arr)
                raw = _ld_16b(tmp_base + arith.ExtUIOp(i64, cur).result * ea.constant(16, type=i64))
                dst_pack = dst_rank * ea.constant(part_p, type=i32) + cur
                _st_16b(out_ptr_i64 + arith.ExtUIOp(i64, dst_pack).result * ea.constant(16, type=i64),
                        raw)
                scf.YieldOp([cur + stride_pack2])
        else:
            # Non-vectorized fallback (world_size=6 or num_packs % world_size != 0)
            # Iterates over largest_part_p packs; guards the last partition boundary
            tid_i32 = bid_i32 * ea.constant(threads, type=i32) + lane_i32
            stride_i32 = ea.index_cast(i32, gpu.grid_dim("x")) * ea.constant(threads, type=i32)

            loop2 = scf.WhileOp([i32], [tid_i32])
            b2 = ir.Block.create_at_start(loop2.before, [i32])
            a2 = ir.Block.create_at_start(loop2.after, [i32])
            with ir.InsertionPoint(b2):
                cur = b2.arguments[0]
                cond = arith.CmpIOp(arith.CmpIPredicate.ult, cur,
                                    ea.constant(largest_part_p, type=i32)).result
                scf.ConditionOp(cond, [cur])
            with ir.InsertionPoint(a2):
                cur = a2.arguments[0]
                for p in range_constexpr(world_size):
                    if p == world_size - 1:
                        ok = arith.ConstantOp(ir.IntegerType.get_signless(1), 1).result
                    else:
                        ok = arith.CmpIOp(arith.CmpIPredicate.ult, cur,
                                          ea.constant(part_p, type=i32)).result
                    ifp = scf.IfOp(ok, results_=[], has_else=False)
                    with ir.InsertionPoint(ifp.then_block):
                        src_off = arith.ExtUIOp(i64, cur).result * ea.constant(16, type=i64)
                        raw = _ld_16b(tmp_ptrs_arr[p] + src_off)
                        dst_pack_idx = ea.constant(p * part_p, type=i32) + cur
                        dst_off = arith.ExtUIOp(i64, dst_pack_idx).result * ea.constant(16, type=i64)
                        _st_16b(out_ptr_i64 + dst_off, raw)
                        scf.YieldOp([])
                scf.YieldOp([cur + stride_i32])

    # -----------------------------------------------------------------------
    # GPU Kernel: 2-stage write-mode (large tensors, writes reduced result
    # directly to REMOTE output buffers via XGMI)
    # Args: rank, self_sg, sg_ptrs, inp_ptr, out_ptrs, tmp_ptrs  (all raw i64)
    # -----------------------------------------------------------------------
    @flyc.kernel
    def allreduce_2stage_write_mode(
        rank: Int32,
        self_sg: Int64,
        sg_ptrs: Int64,
        inp_ptr: Int64,
        out_ptrs: Int64,
        tmp_ptrs: Int64,
    ):
        import math
        from flydsl._mlir.dialects import arith, gpu, memref, scf, vector, rocdl
        from flydsl.expr.buffer_ops import _unwrap_value

        # Set required work group size attributes
        _entry_block_wm = ir.InsertionPoint.current.block
        _gpu_func_op_wm = _entry_block_wm.owner
        _gpu_func_op_wm.operation.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([threads, 1, 1])
        _gpu_func_op_wm.operation.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{threads},{threads}")

        i32 = _i32()
        i64 = _i64()
        idx = ir.IndexType.get()
        v4i32 = ir.VectorType.get([4], i32)
        lds_space = ir.Attribute.parse("#gpu.address_space<workgroup>")
        smem_ty = ir.MemRefType.get([2 * threads], v4i32, memory_space=lds_space)
        if is_f32:
            v4f32 = ir.VectorType.get([4], ir.F32Type.get())
        else:
            v8f16 = ir.VectorType.get([8], ir.F16Type.get())
            v8f32 = ir.VectorType.get([8], ir.F32Type.get())

        lane_i32 = ea.index_cast(i32, gpu.thread_id("x"))
        bid_i32 = ea.index_cast(i32, gpu.block_id("x"))
        rank_i32 = _unwrap_value(rank)
        self_sg_i64 = _unwrap_value(self_sg)
        sg_ptrs_i64 = _unwrap_value(sg_ptrs)
        inp_ptr_i64 = _unwrap_value(inp_ptr)
        out_ptrs_i64 = _unwrap_value(out_ptrs)
        tmp_ptrs_i64 = _unwrap_value(tmp_ptrs)

        sgs = [_load_ptr_from_array(sg_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]
        out_ptrs_arr = [_load_ptr_from_array(out_ptrs_i64, ea.constant(i, type=i32)) for i in range(8)]

        tnum_gpu_i32 = ea.constant(tnum_gpu, type=i32)
        log2_tnum = int(math.log2(tnum_gpu))
        warp_id = arith.ShRUIOp(lane_i32, ea.constant(log2_tnum, type=i32)).result
        warp_base = warp_id * tnum_gpu_i32
        lane_id = arith.SubIOp(lane_i32, warp_base).result
        tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
        stride_pack = ea.index_cast(i32, gpu.grid_dim("x")) * tnum_gpu_i32

        smem_sym_wm = f"allreduce_smem_wm_ws{world_size}_t{threads}"
        _entry_block = ir.InsertionPoint.current.block
        _gpu_func_op = _entry_block.owner
        _gpu_module_body_block = _gpu_func_op.operation.block
        with ir.InsertionPoint.at_block_begin(_gpu_module_body_block):
            memref.GlobalOp(
                sym_name=ir.StringAttr.get(smem_sym_wm),
                type_=smem_ty,
                initial_value=None,
                constant=False,
                alignment=16,
            )
        smem = memref.GetGlobalOp(smem_ty, smem_sym_wm).result
        tmp_out_i64 = _load_ptr_from_array(tmp_ptrs_i64, rank_i32)

        # ---- Stage 1: scatter local input to REMOTE tmp buffers ----
        start_w = warp_id * ea.constant(part_p, type=i32)
        is_last_w = arith.CmpIOp(arith.CmpIPredicate.eq, warp_id,
                                 ea.constant(world_size - 1, type=i32)).result
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
            cond = arith.CmpIOp(arith.CmpIPredicate.ult, cur, end_w).result
            scf.ConditionOp(cond, [cur, bs1.arguments[1]])
        with ir.InsertionPoint(as1):
            cur = as1.arguments[0]
            stride_s1 = as1.arguments[1]
            raw = _ld_16b(inp_ptr_i64 + arith.ExtUIOp(i64, cur).result * ea.constant(16, type=i64))
            rel_idx = cur - start_w
            dst_off = rank_i32 * ea.constant(part_p, type=i32) + rel_idx
            dst_tmp = _load_ptr_from_array(tmp_ptrs_i64, warp_id)
            _st_16b(dst_tmp + arith.ExtUIOp(i64, dst_off).result * ea.constant(16, type=i64), raw)
            scf.YieldOp([cur + stride_s1, stride_s1])

        # Signal all ranks that stage 1 is complete
        _signal_start_sync(lane_i32=lane_i32, rank_i32=rank_i32, bid_i32=bid_i32,
                           self_sg_i64=self_sg_i64, sgs_i64=sgs, ngpus=world_size)

        # ---- Stage 2: reduce local tmp and write to REMOTE outputs ----
        part_p_i32 = ea.constant(part_p, type=i32)
        loop_s2 = scf.WhileOp([i32, i32], [tid_pack, stride_pack])
        bs2 = ir.Block.create_at_start(loop_s2.before, [i32, i32])
        as2 = ir.Block.create_at_start(loop_s2.after, [i32, i32])
        with ir.InsertionPoint(bs2):
            cur = bs2.arguments[0]
            cond = arith.CmpIOp(arith.CmpIPredicate.ult, cur, part_p_i32).result
            scf.ConditionOp(cond, [cur, bs2.arguments[1]])
        with ir.InsertionPoint(as2):
            cur = as2.arguments[0]
            stride_s2 = as2.arguments[1]

            src_off = warp_id * ea.constant(part_p, type=i32) + cur
            load_addr = tmp_out_i64 + arith.ExtUIOp(i64, src_off).result * ea.constant(16, type=i64)
            raw = _ld_16b(load_addr)

            sm_idx = ea.index_cast(idx, lane_i32)
            memref.StoreOp(raw, smem, [sm_idx])
            gpu.BarrierOp()

            warp_id_local = arith.ShRUIOp(lane_i32, ea.constant(log2_tnum, type=i32)).result
            lane_id_local = arith.SubIOp(
                lane_i32, warp_id_local * ea.constant(tnum_gpu, type=i32)).result

            acc = None
            for wi in range_constexpr(world_size):
                sm_i_idx = ea.index_cast(idx, ea.constant(wi * tnum_gpu, type=i32) + lane_id_local)
                raw_i = memref.LoadOp(smem, [sm_i_idx]).result
                if is_f32:
                    vf = vector.BitCastOp(v4f32, raw_i).result
                    acc = vf if acc is None else acc + vf
                else:
                    v16 = vector.BitCastOp(v8f16, raw_i).result
                    v32 = arith.ExtFOp(v8f32, v16).result
                    acc = v32 if acc is None else acc + v32
            if is_f32:
                out_raw = vector.BitCastOp(v4i32, acc).result
            else:
                out_raw = vector.BitCastOp(v4i32, acc.truncf(v8f16)).result

            dst_out_off = rank_i32 * ea.constant(part_p, type=i32) + cur
            dst_byte_off = arith.ExtUIOp(i64, dst_out_off).result * ea.constant(16, type=i64)

            # Each warp writes to its assigned output partition
            dst_ptr = out_ptrs_arr[0]
            for w in range_constexpr(1, world_size):
                is_warp_w = arith.CmpIOp(arith.CmpIPredicate.eq, warp_id_local,
                                          ea.constant(w, type=i32)).result
                dst_ptr = arith.SelectOp(is_warp_w, out_ptrs_arr[w], dst_ptr).result
            _st_16b(dst_ptr + dst_byte_off, out_raw)

            scf.YieldOp([cur + stride_s2, stride_s2])

        gpu.BarrierOp()
        rocdl.s_waitcnt(0)

    # -----------------------------------------------------------------------
    # Host launchers (@flyc.jit)
    # -----------------------------------------------------------------------

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
    run_2stage_arr.func.__name__ = f"run_2stage_arr{_suffix}"
    run_2stage_write_mode.func.__name__ = f"run_2stage_write_mode{_suffix}"

    return {
        "run_2stage_arr": run_2stage_arr,
        "run_2stage_write_mode": run_2stage_write_mode,
    }
