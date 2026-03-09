"""AIter-style all-reduce on ROCm using FlyDSL + MLIR.

Implements 1-stage and 2-stage (reduce-scatter + all-gather) kernels
following the AIter ROCm signal protocol (start/end/_flag).

tmp_buffer and signal_buffer are allocated as hipDeviceMallocUncached
(bypasses L1/TCP cache). Memory loads/stores use standard llvm ops; the
compiler generates optimal GFX942 instructions automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from flydsl._mlir import ir
from flydsl.expr import arith as ea  # Registers ArithValue caster; enables operator overloading

_AITER_KMAXBLOCKS = 80
_AITER_START_OFF_B = 0
_AITER_END_OFF_B = 2560
_AITER_FLAG_OFF_B = 5120


def _elem_type(dtype_str: str) -> ir.Type:
    d = (dtype_str or "").strip().lower()
    if d in {"f16", "fp16"}:
        return ir.F16Type.get()
    if d in {"f32", "fp32"}:
        return ir.F32Type.get()
    raise ValueError(f"unsupported dtype_str: {dtype_str}")


def _pack_elems(dtype_str: str) -> int:
    d = (dtype_str or "").strip().lower()
    if d in {"f32", "fp32"}:
        return 4
    if d in {"f16", "fp16"}:
        return 8
    raise ValueError(f"unsupported dtype_str: {dtype_str}")


def _i32() -> ir.Type:
    return ir.IntegerType.get_signless(32)


def _i64() -> ir.Type:
    return ir.IntegerType.get_signless(64)


def _compute_allreduce_vector(ins, out, base_idx, world_size, vec_e_ty, vec_f32_ty, elem_ty):
    """Reduce vectors from all ranks and store result."""
    from flydsl._mlir.dialects import arith, vector
    is_f32 = (elem_ty == ir.F32Type.get())
    acc = None
    for r in range(world_size):
        v_e = vector.LoadOp(vec_e_ty, ins[r], [base_idx]).result
        v_f = v_e if is_f32 else arith.ExtFOp(vec_f32_ty, v_e).result
        acc = v_f if acc is None else acc + v_f
    out_v = acc if is_f32 else acc.truncf(vec_e_ty)
    vector.StoreOp(out_v, out, [base_idx], alignment=16)


# ---- Low-level global memory primitives ----
# tmp_buffer and signal_buffer are hipDeviceMallocUncached (bypasses L1/TCP).
# For cross-GPU writes via XGMI, buffer_wbl2 flushes L2 (TCC) to HBM for
# XGMI fabric visibility. sc0/sc1 modifiers ensure store/load ordering.

def _global_ld_u32(addr_i64):
    """Load u32 from a raw global pointer (signal buffer, L1-bypassed)."""
    from flydsl._mlir.dialects import llvm, rocdl
    v = llvm.InlineAsmOp(
        _i32(), [addr_i64],
        "global_load_dword $0, $1, off sc1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


def _global_st_u32(addr_i64, val_i32):
    """Store u32 to peer GPU's signal buffer (flush L2 for XGMI visibility)."""
    from flydsl._mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def _global_ld_16b(addr_i64):
    """Load 16 bytes (vector<4xi32>) from a raw global pointer."""
    from flydsl._mlir.dialects import llvm, rocdl
    v4i32 = ir.VectorType.get([4], _i32())
    v = llvm.InlineAsmOp(
        v4i32, [addr_i64],
        "flat_load_dwordx4 $0, $1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


def _global_st_16b(addr_i64, v4i32_val):
    """Store 16 bytes (vector<4xi32>) to a raw global pointer."""
    from flydsl._mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(
        None, [addr_i64, v4i32_val],
        "global_store_dwordx4 $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def _same_gpu_st_u32(addr_i64, val_i32):
    """Store u32 to local (same-GPU) signal buffer (no XGMI flush needed)."""
    from flydsl._mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def _select_i64_by_lane(lane_i32, vals_i64):
    """Select one of vals_i64[0..7] by lane value via chained arith.select."""
    from flydsl._mlir.dialects import arith
    i32 = _i32()
    out = vals_i64[0]
    for i in range(1, 8):
        pred = arith.CmpIOp(arith.CmpIPredicate.eq, lane_i32, ea.constant(i, type=i32)).result
        out = arith.SelectOp(pred, vals_i64[i], out).result
    return out


def _load_ptr_from_array(array_base_i64, index_i32):
    """Load an i64 pointer value from a device memory array at index_i32."""
    from flydsl._mlir.dialects import llvm, arith
    i64 = _i64()
    elem_addr = array_base_i64 + arith.ExtUIOp(i64, index_i32).result * ea.constant(8, type=i64)
    ptr = llvm.IntToPtrOp(ir.Type.parse('!llvm.ptr'), elem_addr).result
    return llvm.LoadOp(i64, ptr).result


def _spin_wait_ge_u32(addr_i64, target_u32):
    """Spin-wait until *addr >= target (uncachable signal buffer)."""
    from flydsl._mlir.dialects import arith, scf
    i32 = _i32()
    init_cur = _global_ld_u32(addr_i64)
    w = scf.WhileOp([i32], [init_cur])
    before = ir.Block.create_at_start(w.before, [i32])
    after = ir.Block.create_at_start(w.after, [i32])
    with ir.InsertionPoint(before):
        cur = before.arguments[0]
        need_wait = arith.CmpIOp(arith.CmpIPredicate.ult, cur, target_u32).result
        scf.ConditionOp(need_wait, [cur])
    with ir.InsertionPoint(after):
        scf.YieldOp([_global_ld_u32(addr_i64)])


def _start_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int):
    """AIter ROCm start_sync: write start flag to all peers, wait for all to arrive."""
    from flydsl._mlir.dialects import arith, gpu, scf
    i32, i64 = _i32(), _i64()

    flag_addr = (self_sg_i64 + ea.constant(_AITER_FLAG_OFF_B, type=i64)
                 + arith.ExtUIOp(i64, bid_i32).result * ea.constant(4, type=i64))
    flag = _global_ld_u32(flag_addr) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    start_wait_addr = (self_sg_i64 + ea.constant(_AITER_START_OFF_B, type=i64)
                       + arith.ExtUIOp(i64, lin_lane).result * ea.constant(4, type=i64))
    lin_rank = bid8 + rank_i32
    start_rank_off = (ea.constant(_AITER_START_OFF_B, type=i64)
                      + arith.ExtUIOp(i64, lin_rank).result * ea.constant(4, type=i64))

    is_lane = arith.CmpIOp(arith.CmpIPredicate.ult, lane_i32, ea.constant(ngpus, type=i32)).result
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _select_i64_by_lane(lane_i32, sgs_i64)
        _global_st_u32(peer_sg + start_rank_off, flag)
        _spin_wait_ge_u32(start_wait_addr, flag)
        scf.YieldOp([])

    gpu.BarrierOp()
    is_t0 = arith.CmpIOp(arith.CmpIPredicate.eq, lane_i32, ea.constant(0, type=i32)).result
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        _same_gpu_st_u32(flag_addr, flag)
        scf.YieldOp([])
    return flag_addr


def _end_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int):
    """AIter ROCm end_sync: write end flag to all peers, wait for all to finish."""
    from flydsl._mlir.dialects import arith, gpu, scf
    i32, i64 = _i32(), _i64()

    gpu.BarrierOp()
    flag_addr = (self_sg_i64 + ea.constant(_AITER_FLAG_OFF_B, type=i64)
                 + arith.ExtUIOp(i64, bid_i32).result * ea.constant(4, type=i64))
    flag = _global_ld_u32(flag_addr) + ea.constant(1, type=i32)

    bid8 = bid_i32 * ea.constant(8, type=i32)
    lin_lane = bid8 + lane_i32
    end_wait_addr = (self_sg_i64 + ea.constant(_AITER_END_OFF_B, type=i64)
                     + arith.ExtUIOp(i64, lin_lane).result * ea.constant(4, type=i64))
    lin_rank = bid8 + rank_i32
    end_rank_off = (ea.constant(_AITER_END_OFF_B, type=i64)
                    + arith.ExtUIOp(i64, lin_rank).result * ea.constant(4, type=i64))

    is_lane = arith.CmpIOp(arith.CmpIPredicate.ult, lane_i32, ea.constant(ngpus, type=i32)).result
    if_op = scf.IfOp(is_lane, results_=[], has_else=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _select_i64_by_lane(lane_i32, sgs_i64)
        _global_st_u32(peer_sg + end_rank_off, flag)
        _spin_wait_ge_u32(end_wait_addr, flag)
        scf.YieldOp([])

    gpu.BarrierOp()
    is_t0 = arith.CmpIOp(arith.CmpIPredicate.eq, lane_i32, ea.constant(0, type=i32)).result
    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
    with ir.InsertionPoint(if_t0.then_block):
        _same_gpu_st_u32(flag_addr, flag)
        scf.YieldOp([])


def build_aiter_signal_allreduce_raw_module(*, N: int, dtype_str: str, world_size: int, threads: int = 256, meta_size_bytes: int = 5504, max_spin: int = 20000000) -> ir.Module:
    """Build an ir.Module with host entrypoints:
    - run_1stage(rank, grid_x, self_sg, sg0..sg7, in0..in7, out, stream_ptr)
    - run_2stage(rank, grid_x, self_sg, sg0..sg7, in0..in7, tmp0..tmp7, out, stream_ptr)
    - run_2stage_arr(rank, grid_x, self_sg, sg_ptrs_base, in_ptrs_base, tmp_ptrs_base, out_ptr, stream_ptr)
    - run_2stage_write_mode(rank, grid_x, self_sg, sg_ptrs_base, inp_ptr, out_ptrs_base, tmp_ptrs_base, stream_ptr)
    """
    if world_size not in {2, 4, 6, 8}:
        raise ValueError("world_size must be one of {2,4,6,8}")
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if N <= 0:
        raise ValueError("N must be > 0")

    from kernels.kernels_common import stream_i64_to_llvm_ptr
    from flydsl._mlir.dialects import arith, func, gpu, memref, scf, vector

    ctx = ir.Context()
    ctx.load_all_available_dialects()
    with ctx, ir.Location.unknown():
        elem_ty = _elem_type(dtype_str)
        pack_elems = _pack_elems(dtype_str)
        if N % pack_elems != 0:
            raise ValueError(f"N must be multiple of pack_elems={pack_elems}")
        num_packs = N // pack_elems
        part_p = num_packs // world_size
        largest_part_p = part_p + (num_packs % world_size)
        tmp_elems = largest_part_p * pack_elems

        i32 = _i32()
        i64 = _i64()
        idx = ir.IndexType.get()

        mem_in_ty = ir.MemRefType.get([N], elem_ty)
        mem_out_ty = ir.MemRefType.get([N], elem_ty)
        mem_tmp_ty = ir.MemRefType.get([tmp_elems], elem_ty)
        vec_e_ty = ir.VectorType.get([pack_elems], elem_ty)
        vec_f32_ty = ir.VectorType.get([pack_elems], ir.F32Type.get())
        v4i32 = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
        v4f32 = ir.VectorType.get([4], ir.F32Type.get())
        v8f16 = ir.VectorType.get([8], ir.F16Type.get()) if pack_elems == 8 else None
        v8f32 = ir.VectorType.get([8], ir.F32Type.get()) if pack_elems == 8 else None

        m = ir.Module.create()
        m.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
        with ir.InsertionPoint(m.body):
            gpu_mod = gpu.GPUModuleOp("aiter_signal")
            gpu_mod.bodyRegion.blocks.append()

            with ir.InsertionPoint(gpu_mod.bodyRegion.blocks[0]):
                if (threads % world_size) != 0:
                    raise ValueError(f"threads={threads} must be divisible by world_size={world_size}")
                lds_space = ir.Attribute.parse("#gpu.address_space<workgroup>")
                smem_ty = ir.MemRefType.get([2 * threads], v4i32, memory_space=lds_space)
                smem_sym = f"aiter_signal_smem_ws{world_size}_t{threads}"
                memref.GlobalOp(
                    sym_name=ir.StringAttr.get(smem_sym),
                    type_=smem_ty,
                    initial_value=None,
                    constant=False,
                    alignment=16,
                )

                ptrtbl_ty = ir.MemRefType.get([16], i64, memory_space=lds_space)
                ptrtbl_sym = f"aiter_signal_ptrtbl_ws{world_size}"
                memref.GlobalOp(
                    sym_name=ir.StringAttr.get(ptrtbl_sym),
                    type_=ptrtbl_ty,
                    initial_value=None,
                    constant=False,
                    alignment=16,
                )

                sg_ptrtbl_ty = ir.MemRefType.get([8], i64, memory_space=lds_space)
                sg_ptrtbl_sym = f"aiter_signal_sg_ptrtbl_ws{world_size}"
                memref.GlobalOp(
                    sym_name=ir.StringAttr.get(sg_ptrtbl_sym),
                    type_=sg_ptrtbl_ty,
                    initial_value=None,
                    constant=False,
                    alignment=16,
                )

                # ---- Kernel: 1-stage all-reduce ----
                k1_args = [i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_out_ty]
                k1_fty = ir.FunctionType.get(k1_args, [])
                k1 = gpu.GPUFuncOp(ir.TypeAttr.get(k1_fty), sym_name=f"aiter_signal_all_reduce_1stage_ws{world_size}", kernel=True)
                k1.operation.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([threads, 1, 1])
                k1.operation.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{threads},{threads}")
                k1_entry = ir.Block.create_at_start(k1.operation.regions[0], k1_args)
                with ir.InsertionPoint(k1_entry):
                    rank = k1_entry.arguments[0]
                    self_sg = k1_entry.arguments[1]
                    sgs = list(k1_entry.arguments[2:10]) + [k1_entry.arguments[2]] * max(0, 8 - 8)
                    ins = list(k1_entry.arguments[10:18])
                    out = k1_entry.arguments[18]

                    lane_i32 = ea.index_cast(i32, gpu.thread_id("x"))
                    bid_i32 = ea.index_cast(i32, gpu.block_id("x"))

                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                                self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    pack0 = bid_i32 * ea.constant(threads, type=i32) + lane_i32
                    stride_p = ea.index_cast(i32, gpu.grid_dim("x")) * ea.constant(threads, type=i32)
                    loop = scf.WhileOp([i32], [pack0])
                    bfor = ir.Block.create_at_start(loop.before, [i32])
                    afor = ir.Block.create_at_start(loop.after, [i32])
                    with ir.InsertionPoint(bfor):
                        p = bfor.arguments[0]
                        cond = arith.CmpIOp(arith.CmpIPredicate.ult, p, ea.constant(num_packs, type=i32)).result
                        scf.ConditionOp(cond, [p])
                    with ir.InsertionPoint(afor):
                        p = afor.arguments[0]
                        base_idx = ea.index_cast(idx, p * ea.constant(pack_elems, type=i32))
                        _compute_allreduce_vector(
                            ins=ins, out=out, base_idx=base_idx,
                            world_size=world_size, vec_e_ty=vec_e_ty,
                            vec_f32_ty=vec_f32_ty, elem_ty=elem_ty,
                        )
                        scf.YieldOp([p + stride_p])

                    _end_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                              self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)
                    gpu.ReturnOp([])

                # ---- Kernel: 2-stage (reduce-scatter + all-gather) ----
                k2_args = [i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_tmp_ty] * 8 + [mem_out_ty]
                k2_fty = ir.FunctionType.get(k2_args, [])
                k2 = gpu.GPUFuncOp(ir.TypeAttr.get(k2_fty), sym_name=f"aiter_signal_all_reduce_2stage_ws{world_size}", kernel=True)
                k2.operation.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([threads, 1, 1])
                k2.operation.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{threads},{threads}")
                k2_entry = ir.Block.create_at_start(k2.operation.regions[0], k2_args)
                with ir.InsertionPoint(k2_entry):
                    rank = k2_entry.arguments[0]
                    self_sg = k2_entry.arguments[1]
                    sgs = list(k2_entry.arguments[2:10])
                    ins = list(k2_entry.arguments[10:18])
                    tmps = list(k2_entry.arguments[18:26])
                    out = k2_entry.arguments[26]

                    lane_i32 = ea.index_cast(i32, gpu.thread_id("x"))
                    bid_i32 = ea.index_cast(i32, gpu.block_id("x"))
                    tid_i32 = bid_i32 * ea.constant(threads, type=i32) + lane_i32
                    stride_i32 = ea.index_cast(i32, gpu.grid_dim("x")) * ea.constant(threads, type=i32)

                    start_p = rank * ea.constant(part_p, type=i32)
                    is_last = arith.CmpIOp(arith.CmpIPredicate.eq, rank, ea.constant(world_size - 1, type=i32)).result
                    end_p = arith.SelectOp(is_last, ea.constant(num_packs, type=i32), start_p + ea.constant(part_p, type=i32)).result
                    tmp_out = tmps[0]

                    sg_ptrtbl = memref.GetGlobalOp(sg_ptrtbl_ty, sg_ptrtbl_sym).result

                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                                self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    tnum_gpu = int(threads // world_size)
                    tnum_gpu_i32 = ea.constant(tnum_gpu, type=i32)
                    warp_id = arith.DivUIOp(lane_i32, tnum_gpu_i32).result
                    lane_id = arith.RemUIOp(lane_i32, tnum_gpu_i32).result
                    tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
                    stride_pack = ea.index_cast(i32, gpu.grid_dim("x")) * tnum_gpu_i32

                    smem = memref.GetGlobalOp(smem_ty, smem_sym).result
                    ptrtbl = memref.GetGlobalOp(ptrtbl_ty, ptrtbl_sym).result
                    sg_ptrtbl = memref.GetGlobalOp(sg_ptrtbl_ty, sg_ptrtbl_sym).result

                    tmp_out_i64 = ea.index_cast(i64, memref.ExtractAlignedPointerAsIndexOp(tmp_out).result)

                    is_t0 = arith.CmpIOp(arith.CmpIPredicate.eq, lane_i32, ea.constant(0, type=i32)).result
                    if_t0 = scf.IfOp(is_t0, results_=[], has_else=False)
                    with ir.InsertionPoint(if_t0.then_block):
                        for i in range(world_size):
                            in_i64 = ea.index_cast(i64, memref.ExtractAlignedPointerAsIndexOp(ins[i]).result)
                            memref.StoreOp(in_i64, ptrtbl, [ea.index(i)])
                            tmp_i64 = ea.index_cast(i64, memref.ExtractAlignedPointerAsIndexOp(tmps[i]).result)
                            memref.StoreOp(tmp_i64, ptrtbl, [ea.index(8 + i)])
                        for i in range(world_size):
                            memref.StoreOp(sgs[i], sg_ptrtbl, [ea.index(i)])
                        scf.YieldOp([])
                    gpu.BarrierOp()

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

                        warp_idx = ea.index_cast(idx, warp_id)
                        in_i64 = memref.LoadOp(ptrtbl, [warp_idx]).result
                        raw = _global_ld_16b(in_i64 + arith.ExtUIOp(i64, cur).result * ea.constant(16, type=i64))
                        sm_base = parity * ea.constant(threads, type=i32)
                        sm_idx = ea.index_cast(idx, sm_base + lane_i32)
                        memref.StoreOp(raw, smem, [sm_idx])
                        gpu.BarrierOp()

                        is_w0 = arith.CmpIOp(arith.CmpIPredicate.eq, warp_id, ea.constant(0, type=i32)).result
                        ifw0 = scf.IfOp(is_w0, results_=[], has_else=False)
                        with ir.InsertionPoint(ifw0.then_block):
                            acc = None
                            for i in range(world_size):
                                sm_i_idx = ea.index_cast(idx, ea.constant(i, type=i32) * tnum_gpu_i32 + lane_id + sm_base)
                                raw_i = memref.LoadOp(smem, [sm_i_idx]).result
                                if dtype_str == "f32":
                                    vf = vector.BitCastOp(v4f32, raw_i).result
                                    acc = vf if acc is None else acc + vf
                                else:
                                    v16 = vector.BitCastOp(v8f16, raw_i).result
                                    v32 = arith.ExtFOp(v8f32, v16).result
                                    acc = v32 if acc is None else acc + v32
                            if dtype_str == "f32":
                                out_raw = vector.BitCastOp(v4i32, acc).result
                            else:
                                out_raw = vector.BitCastOp(v4i32, acc.truncf(v8f16)).result

                            rel_p = cur - start_p
                            _global_st_16b(tmp_out_i64 + arith.ExtUIOp(i64, rel_p).result * ea.constant(16, type=i64), out_raw)
                            scf.YieldOp([])

                        scf.YieldOp([cur + stride_pack, ea.constant(1, type=i32) - parity])

                    _end_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                              self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    tmp_ptrs_i64 = [ea.index_cast(i64, memref.ExtractAlignedPointerAsIndexOp(tmps[i]).result) for i in range(world_size)]

                    vec_ok = (num_packs % world_size == 0) and (world_size != 6)
                    if vec_ok:
                        tid_pack2 = bid_i32 * tnum_gpu_i32 + lane_id
                        stride_pack2 = ea.index_cast(i32, gpu.grid_dim("x")) * tnum_gpu_i32
                        out_i64 = ea.index_cast(i64, memref.ExtractAlignedPointerAsIndexOp(out).result)

                        loop2 = scf.WhileOp([i32], [tid_pack2])
                        b2 = ir.Block.create_at_start(loop2.before, [i32])
                        a2 = ir.Block.create_at_start(loop2.after, [i32])
                        with ir.InsertionPoint(b2):
                            cur = b2.arguments[0]
                            cond = arith.CmpIOp(arith.CmpIPredicate.ult, cur, ea.constant(part_p, type=i32)).result
                            scf.ConditionOp(cond, [cur])
                        with ir.InsertionPoint(a2):
                            cur = a2.arguments[0]
                            sum_rw = arith.AddIOp(rank, warp_id).result
                            if world_size in {2, 4, 8}:
                                dst_rank = arith.AndIOp(sum_rw, ea.constant(world_size - 1, type=i32)).result
                            else:
                                dst_rank = arith.RemUIOp(sum_rw, ea.constant(world_size, type=i32)).result
                            cur_idx = ea.index_cast(idx, cur)

                            def _load_from_tmp_memref(memref_val, idx_val):
                                if dtype_str == "f32":
                                    return vector.BitCastOp(v4i32, vector.LoadOp(v4f32, memref_val, [idx_val]).result).result
                                else:
                                    from flydsl._mlir.dialects import llvm
                                    return llvm.BitcastOp(v4i32, vector.LoadOp(v8f16, memref_val, [idx_val]).result).result

                            tmp_if_chain = None
                            for i in range(world_size - 1, -1, -1):
                                is_match = arith.CmpIOp(arith.CmpIPredicate.eq, warp_id, ea.constant(i, type=i32)).result
                                if tmp_if_chain is None:
                                    if_op = scf.IfOp(is_match, results_=[v4i32], has_else=True)
                                    with ir.InsertionPoint(if_op.then_block):
                                        scf.YieldOp([_load_from_tmp_memref(tmps[i], cur_idx)])
                                    with ir.InsertionPoint(if_op.else_block):
                                        scf.YieldOp([_load_from_tmp_memref(tmps[0], cur_idx)])
                                    tmp_if_chain = if_op.results[0]
                                else:
                                    if_op = scf.IfOp(is_match, results_=[v4i32], has_else=True)
                                    with ir.InsertionPoint(if_op.then_block):
                                        scf.YieldOp([_load_from_tmp_memref(tmps[i], cur_idx)])
                                    with ir.InsertionPoint(if_op.else_block):
                                        scf.YieldOp([tmp_if_chain])
                                    tmp_if_chain = if_op.results[0]
                            raw = tmp_if_chain

                            dst_pack = dst_rank * ea.constant(part_p, type=i32) + cur
                            dst_off16 = arith.ExtUIOp(i64, dst_pack).result * ea.constant(16, type=i64)
                            _global_st_16b(out_i64 + dst_off16, raw)
                            scf.YieldOp([cur + stride_pack2])
                    else:
                        loop2 = scf.WhileOp([i32], [tid_i32])
                        b2 = ir.Block.create_at_start(loop2.before, [i32])
                        a2 = ir.Block.create_at_start(loop2.after, [i32])
                        with ir.InsertionPoint(b2):
                            cur = b2.arguments[0]
                            cond = arith.CmpIOp(arith.CmpIPredicate.ult, cur, ea.constant(largest_part_p, type=i32)).result
                            scf.ConditionOp(cond, [cur])
                        with ir.InsertionPoint(a2):
                            cur = a2.arguments[0]
                            for p in range(world_size):
                                if p == world_size - 1:
                                    ok = arith.ConstantOp(ir.IntegerType.get_signless(1), 1).result
                                else:
                                    ok = arith.CmpIOp(arith.CmpIPredicate.ult, cur, ea.constant(part_p, type=i32)).result
                                ifp = scf.IfOp(ok, results_=[], has_else=False)
                                with ir.InsertionPoint(ifp.then_block):
                                    src_idx = ea.index_cast(idx, cur * ea.constant(pack_elems, type=i32))
                                    v_e = vector.LoadOp(vec_e_ty, tmps[p], [src_idx]).result
                                    dst_idx = ea.index_cast(idx, (ea.constant(p, type=i32) * ea.constant(part_p, type=i32) + cur) * ea.constant(pack_elems, type=i32))
                                    vector.StoreOp(v_e, out, [dst_idx])
                                    scf.YieldOp([])
                            scf.YieldOp([cur + stride_i32])
                    gpu.ReturnOp([])

                # ---- Kernel: 2-stage arr (CUDAGraph-compatible, loads pointers inside kernel) ----
                k2a_args = [i32, i64, i64, i64, i64, i64]
                k2a_fty = ir.FunctionType.get(k2a_args, [])
                k2a = gpu.GPUFuncOp(ir.TypeAttr.get(k2a_fty), sym_name=f"aiter_signal_all_reduce_2stage_arr_ws{world_size}", kernel=True)
                k2a.operation.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([threads, 1, 1])
                k2a.operation.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{threads},{threads}")
                k2a_entry = ir.Block.create_at_start(k2a.operation.regions[0], k2a_args)
                with ir.InsertionPoint(k2a_entry):
                    rank = k2a_entry.arguments[0]
                    self_sg = k2a_entry.arguments[1]
                    sg_ptrs_base = k2a_entry.arguments[2]
                    in_ptrs_base = k2a_entry.arguments[3]
                    tmp_ptrs_base = k2a_entry.arguments[4]
                    out_ptr_i64 = k2a_entry.arguments[5]

                    sgs = [_load_ptr_from_array(sg_ptrs_base, ea.constant(i, type=i32)) for i in range(8)]
                    in_ptrs = [_load_ptr_from_array(in_ptrs_base, ea.constant(i, type=i32)) for i in range(8)]
                    tmp_ptrs = [_load_ptr_from_array(tmp_ptrs_base, ea.constant(i, type=i32)) for i in range(8)]

                    lane_i32 = ea.index_cast(i32, gpu.thread_id("x"))
                    bid_i32 = ea.index_cast(i32, gpu.block_id("x"))

                    start_p = rank * ea.constant(part_p, type=i32)
                    is_last = arith.CmpIOp(arith.CmpIPredicate.eq, rank, ea.constant(world_size - 1, type=i32)).result
                    end_p = arith.SelectOp(is_last, ea.constant(num_packs, type=i32), start_p + ea.constant(part_p, type=i32)).result

                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                                self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    tnum_gpu = int(threads // world_size)
                    tnum_gpu_i32 = ea.constant(tnum_gpu, type=i32)
                    warp_id = arith.DivUIOp(lane_i32, tnum_gpu_i32).result
                    lane_id = arith.RemUIOp(lane_i32, tnum_gpu_i32).result
                    tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
                    stride_pack = ea.index_cast(i32, gpu.grid_dim("x")) * tnum_gpu_i32
                    smem = memref.GetGlobalOp(smem_ty, smem_sym).result
                    tmp_out_i64 = tmp_ptrs[0]

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

                        in_base = _select_i64_by_lane(warp_id, in_ptrs)
                        raw = _global_ld_16b(in_base + arith.ExtUIOp(i64, cur).result * ea.constant(16, type=i64))
                        sm_base = parity * ea.constant(threads, type=i32)
                        sm_idx = ea.index_cast(idx, sm_base + lane_i32)
                        memref.StoreOp(raw, smem, [sm_idx])
                        gpu.BarrierOp()

                        is_w0 = arith.CmpIOp(arith.CmpIPredicate.eq, warp_id, ea.constant(0, type=i32)).result
                        ifw0 = scf.IfOp(is_w0, results_=[], has_else=False)
                        with ir.InsertionPoint(ifw0.then_block):
                            acc = None
                            for i in range(world_size):
                                sm_i_idx = ea.index_cast(idx, ea.constant(i, type=i32) * tnum_gpu_i32 + lane_id + sm_base)
                                raw_i = memref.LoadOp(smem, [sm_i_idx]).result
                                if dtype_str == "f32":
                                    vf = vector.BitCastOp(v4f32, raw_i).result
                                    acc = vf if acc is None else acc + vf
                                else:
                                    v16 = vector.BitCastOp(v8f16, raw_i).result
                                    v32 = arith.ExtFOp(v8f32, v16).result
                                    acc = v32 if acc is None else acc + v32
                            if dtype_str == "f32":
                                out_raw = vector.BitCastOp(v4i32, acc).result
                            else:
                                from flydsl._mlir.dialects import llvm
                                out_raw = llvm.BitcastOp(v4i32, acc.truncf(v8f16)).result

                            rel_p = cur - start_p
                            _global_st_16b(tmp_out_i64 + arith.ExtUIOp(i64, rel_p).result * ea.constant(16, type=i64), out_raw)
                            scf.YieldOp([])

                        scf.YieldOp([cur + stride_pack, ea.constant(1, type=i32) - parity])

                    _end_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                              self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    vec_ok = (num_packs % world_size == 0) and (world_size != 6)
                    if vec_ok:
                        tid_pack2 = bid_i32 * tnum_gpu_i32 + lane_id
                        stride_pack2 = ea.index_cast(i32, gpu.grid_dim("x")) * tnum_gpu_i32

                        loop2 = scf.WhileOp([i32], [tid_pack2])
                        b2 = ir.Block.create_at_start(loop2.before, [i32])
                        a2 = ir.Block.create_at_start(loop2.after, [i32])
                        with ir.InsertionPoint(b2):
                            cur = b2.arguments[0]
                            cond = arith.CmpIOp(arith.CmpIPredicate.ult, cur, ea.constant(part_p, type=i32)).result
                            scf.ConditionOp(cond, [cur])
                        with ir.InsertionPoint(a2):
                            cur = a2.arguments[0]
                            sum_rw = arith.AddIOp(rank, warp_id).result
                            if world_size in {2, 4, 8}:
                                dst_rank = arith.AndIOp(sum_rw, ea.constant(world_size - 1, type=i32)).result
                            else:
                                dst_rank = arith.RemUIOp(sum_rw, ea.constant(world_size, type=i32)).result

                            tmp_base = _select_i64_by_lane(warp_id, tmp_ptrs)
                            raw = _global_ld_16b(tmp_base + arith.ExtUIOp(i64, cur).result * ea.constant(16, type=i64))
                            dst_pack = dst_rank * ea.constant(part_p, type=i32) + cur
                            _global_st_16b(out_ptr_i64 + arith.ExtUIOp(i64, dst_pack).result * ea.constant(16, type=i64), raw)
                            scf.YieldOp([cur + stride_pack2])

                    gpu.ReturnOp([])

                # ---- Kernel: 2-stage write_mode (large tensors, N > 512*4096) ----
                # Stage1: scatter local input to REMOTE tmp buffers.
                # Stage2: read local tmp (all ranks' data), reduce, write to REMOTE output.
                # end_sync omitted (causes GPU hang); host-side barrier handles visibility.
                k2w_args = [i32, i64, i64, i64, i64, i64]
                k2w_fty = ir.FunctionType.get(k2w_args, [])
                k2w = gpu.GPUFuncOp(ir.TypeAttr.get(k2w_fty), sym_name=f"aiter_signal_all_reduce_2stage_write_mode_ws{world_size}", kernel=True)
                k2w.operation.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([threads, 1, 1])
                k2w.operation.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{threads},{threads}")
                k2w_entry = ir.Block.create_at_start(k2w.operation.regions[0], k2w_args)
                with ir.InsertionPoint(k2w_entry):
                    rank = k2w_entry.arguments[0]
                    self_sg = k2w_entry.arguments[1]
                    sg_ptrs_base = k2w_entry.arguments[2]
                    inp_ptr_i64 = k2w_entry.arguments[3]
                    out_ptrs_base = k2w_entry.arguments[4]
                    tmp_ptrs_base = k2w_entry.arguments[5]

                    sgs = [_load_ptr_from_array(sg_ptrs_base, ea.constant(i, type=i32)) for i in range(8)]
                    out_ptrs = [_load_ptr_from_array(out_ptrs_base, ea.constant(i, type=i32)) for i in range(8)]

                    lane_i32 = ea.index_cast(i32, gpu.thread_id("x"))
                    bid_i32 = ea.index_cast(i32, gpu.block_id("x"))

                    tnum_gpu = int(threads // world_size)
                    tnum_gpu_i32 = ea.constant(tnum_gpu, type=i32)
                    import math
                    log2_tnum = int(math.log2(tnum_gpu))
                    warp_id = arith.ShRUIOp(lane_i32, ea.constant(log2_tnum, type=i32)).result
                    warp_base = warp_id * tnum_gpu_i32
                    lane_id = arith.SubIOp(lane_i32, warp_base).result
                    tid_pack = bid_i32 * tnum_gpu_i32 + lane_id
                    stride_pack = ea.index_cast(i32, gpu.grid_dim("x")) * tnum_gpu_i32

                    smem = memref.GetGlobalOp(smem_ty, smem_sym).result
                    tmp_out_i64 = _load_ptr_from_array(tmp_ptrs_base, rank)

                    # Stage1: write local input to REMOTE tmp buffers
                    start_w = warp_id * ea.constant(part_p, type=i32)
                    is_last_w = arith.CmpIOp(arith.CmpIPredicate.eq, warp_id, ea.constant(world_size - 1, type=i32)).result
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
                        stride_pack_s1 = as1.arguments[1]
                        raw = _global_ld_16b(inp_ptr_i64 + arith.ExtUIOp(i64, cur).result * ea.constant(16, type=i64))
                        rel_idx = cur - start_w
                        dst_off = rank * ea.constant(part_p, type=i32) + rel_idx
                        dst_tmp = _load_ptr_from_array(tmp_ptrs_base, warp_id)
                        _global_st_16b(dst_tmp + arith.ExtUIOp(i64, dst_off).result * ea.constant(16, type=i64), raw)
                        scf.YieldOp([cur + stride_pack_s1, stride_pack_s1])

                    # Signal all ranks that stage1 is complete
                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                                self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    # Stage2: read local tmp, reduce, write reduced result to REMOTE outputs
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
                        stride_pack_loop = as2.arguments[1]

                        src_off = warp_id * ea.constant(part_p, type=i32) + cur
                        load_addr = tmp_out_i64 + arith.ExtUIOp(i64, src_off).result * ea.constant(16, type=i64)
                        raw = _global_ld_16b(load_addr)

                        sm_idx = ea.index_cast(idx, lane_i32)
                        memref.StoreOp(raw, smem, [sm_idx])
                        gpu.BarrierOp()

                        warp_id_local = arith.ShRUIOp(lane_i32, ea.constant(log2_tnum, type=i32)).result
                        lane_id_local = arith.SubIOp(lane_i32, warp_id_local * ea.constant(tnum_gpu, type=i32)).result

                        raw_vals = []
                        for i in range(world_size):
                            sm_i_idx = ea.index_cast(idx, ea.constant(i * tnum_gpu, type=i32) + lane_id_local)
                            raw_vals.append(memref.LoadOp(smem, [sm_i_idx]).result)

                        acc = None
                        for i in range(world_size):
                            raw_i = raw_vals[i]
                            if dtype_str == "f32":
                                vf = vector.BitCastOp(v4f32, raw_i).result
                                acc = vf if acc is None else acc + vf
                            else:
                                v16 = vector.BitCastOp(v8f16, raw_i).result
                                v32 = arith.ExtFOp(v8f32, v16).result
                                acc = v32 if acc is None else acc + v32
                        if dtype_str == "f32":
                            out_raw = vector.BitCastOp(v4i32, acc).result
                        else:
                            out_raw = vector.BitCastOp(v4i32, acc.truncf(v8f16)).result

                        dst_out_off = rank * ea.constant(part_p, type=i32) + cur
                        dst_byte_off = arith.ExtUIOp(i64, dst_out_off).result * ea.constant(16, type=i64)

                        dst_ptr = out_ptrs[0]
                        for w in range(1, world_size):
                            is_warp_w = arith.CmpIOp(arith.CmpIPredicate.eq, warp_id_local, ea.constant(w, type=i32)).result
                            dst_ptr = arith.SelectOp(is_warp_w, out_ptrs[w], dst_ptr).result
                        _global_st_16b(dst_ptr + dst_byte_off, out_raw)

                        scf.YieldOp([cur + stride_pack_loop, stride_pack_loop])

                    gpu.BarrierOp()
                    from flydsl._mlir.dialects import rocdl
                    rocdl.s_waitcnt(0)
                    gpu.ReturnOp([])

            # ---- Host entrypoints ----

            run1_args = [i32, i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_out_ty] + [i64]
            run1_fty = ir.FunctionType.get(run1_args, [])
            run1 = func.FuncOp("run_1stage", run1_fty)
            run1.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            b = run1.add_entry_block()
            with ir.InsertionPoint(b):
                rank = b.arguments[0]
                grid_x_i32 = b.arguments[1]
                self_sg = b.arguments[2]
                sgs = list(b.arguments[3:11])
                ins = list(b.arguments[11:19])
                out = b.arguments[19]
                stream_ptr = b.arguments[20]
                gx = ea.index_cast(idx, grid_x_i32)
                one = ea.index(1)
                bx = ea.index(threads)
                kops = [rank, self_sg] + list(sgs) + list(ins) + [out]
                stream_obj = stream_i64_to_llvm_ptr(stream_ptr)
                gpu.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_1stage_ws{world_size}"],
                    grid_size=(gx, one, one), block_size=(bx, one, one),
                    kernel_operands=kops, async_object=stream_obj,
                )
                func.ReturnOp([])

            run2_args = [i32, i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_tmp_ty] * 8 + [mem_out_ty] + [i64]
            run2_fty = ir.FunctionType.get(run2_args, [])
            run2 = func.FuncOp("run_2stage", run2_fty)
            run2.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            b2 = run2.add_entry_block()
            with ir.InsertionPoint(b2):
                rank = b2.arguments[0]
                grid_x_i32 = b2.arguments[1]
                self_sg = b2.arguments[2]
                sgs = list(b2.arguments[3:11])
                ins = list(b2.arguments[11:19])
                tmps = list(b2.arguments[19:27])
                out = b2.arguments[27]
                stream_ptr = b2.arguments[28]
                gx = ea.index_cast(idx, grid_x_i32)
                one = ea.index(1)
                bx = ea.index(threads)
                kops = [rank, self_sg] + list(sgs) + list(ins) + list(tmps) + [out]
                stream_obj = stream_i64_to_llvm_ptr(stream_ptr)
                gpu.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_2stage_ws{world_size}"],
                    grid_size=(gx, one, one), block_size=(bx, one, one),
                    kernel_operands=kops, async_object=stream_obj,
                )
                func.ReturnOp([])

            run2a_args = [i32, i32, i64, i64, i64, i64, i64, i64]
            run2a_fty = ir.FunctionType.get(run2a_args, [])
            run2a = func.FuncOp("run_2stage_arr", run2a_fty)
            run2a.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            b2a = run2a.add_entry_block()
            with ir.InsertionPoint(b2a):
                rank = b2a.arguments[0]
                grid_x_i32 = b2a.arguments[1]
                self_sg = b2a.arguments[2]
                sg_ptrs_array_base = b2a.arguments[3]
                in_ptrs_array_base = b2a.arguments[4]
                tmp_ptrs_array_base = b2a.arguments[5]
                out_ptr_i64 = b2a.arguments[6]
                stream_ptr = b2a.arguments[7]
                gx = ea.index_cast(idx, grid_x_i32)
                one = ea.index(1)
                bx = ea.index(threads)
                kops = [rank, self_sg, sg_ptrs_array_base, in_ptrs_array_base, tmp_ptrs_array_base, out_ptr_i64]
                stream_obj = stream_i64_to_llvm_ptr(stream_ptr)
                gpu.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_2stage_arr_ws{world_size}"],
                    grid_size=(gx, one, one), block_size=(bx, one, one),
                    kernel_operands=kops, async_object=stream_obj,
                )
                func.ReturnOp([])

            # run_2stage_write_mode: rank, grid_x, self_sg, sg_ptrs_base, inp_ptr, out_ptrs_base, tmp_ptrs_base, stream_ptr
            run2w_args = [i32, i32, i64, i64, i64, i64, i64, i64]
            run2w_fty = ir.FunctionType.get(run2w_args, [])
            run2w = func.FuncOp("run_2stage_write_mode", run2w_fty)
            run2w.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            b2w = run2w.add_entry_block()
            with ir.InsertionPoint(b2w):
                rank = b2w.arguments[0]
                grid_x_i32 = b2w.arguments[1]
                self_sg = b2w.arguments[2]
                sg_ptrs_array_base = b2w.arguments[3]
                inp_ptr_i64 = b2w.arguments[4]
                out_ptrs_array_base = b2w.arguments[5]
                tmp_ptrs_array_base = b2w.arguments[6]
                stream_ptr = b2w.arguments[7]
                gx = ea.index_cast(idx, grid_x_i32)
                one = ea.index(1)
                bx = ea.index(threads)
                kops = [rank, self_sg, sg_ptrs_array_base, inp_ptr_i64, out_ptrs_array_base, tmp_ptrs_array_base]
                stream_obj = stream_i64_to_llvm_ptr(stream_ptr)
                gpu.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_2stage_write_mode_ws{world_size}"],
                    grid_size=(gx, one, one), block_size=(bx, one, one),
                    kernel_operands=kops, async_object=stream_obj,
                )
                func.ReturnOp([])

        return m


@dataclass
class _Fake1DTensor:
    """Minimal tensor-like object for ExecutionEngineExecutor memref packing."""

    _data_ptr: int
    _shape0: int
    _base_ptr: int | None = None

    @property
    def shape(self):
        return (int(self._shape0),)

    def stride(self):
        return (1,)

    def data_ptr(self):
        return int(self._data_ptr)

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        base = int(self._base_ptr) if self._base_ptr is not None else int(self._data_ptr)

        class _S:
            def __init__(self, p: int):
                self._p = int(p)

            def data_ptr(self):
                return int(self._p)

        return _S(base)


def make_fake_1d_memref(*, data_ptr: int, n_elems: int, base_ptr: int | None = None) -> _Fake1DTensor:
    return _Fake1DTensor(int(data_ptr), int(n_elems), int(base_ptr) if base_ptr is not None else None)
