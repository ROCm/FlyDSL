"""AIter-style all-reduce on ROCm using FlyDSL + MLIR.

Implements 1-stage and 2-stage (reduce-scatter + all-gather) kernels
following the AIter ROCm signal protocol (start/end/_flag).

tmp_buffer and signal_buffer are allocated as uncachable memory, so cache
coherency instructions (buffer_wbl2, buffer_inv, sc0/sc1 modifiers) are
not needed. Memory loads/stores use standard llvm ops; the compiler generates
optimal GFX942 instructions automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from _mlir import ir


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


def _c_i32(v: int):
    from _mlir.dialects import arith
    return _res(arith.ConstantOp(_i32(), v).result)


def _c_i64(v: int):
    from _mlir.dialects import arith
    return _res(arith.ConstantOp(_i64(), v).result)


def _c_index(v: int):
    from _mlir.dialects import arith
    return _res(arith.ConstantOp(ir.IndexType.get(), v).result)


def _compute_allreduce_vector(ins, out, base_idx, world_size, vec_e_ty, vec_f32_ty, elem_ty):
    """Reduce vectors from all ranks and store result."""
    from flydsl.dialects.ext import vector as flir_vector
    from _mlir.dialects import arith

    acc = None
    for r in range(world_size):
        v_e_raw = _res(flir_vector.load_op(vec_e_ty, ins[r], [base_idx]))
        if elem_ty == ir.F32Type.get():
            v_f = _res(v_e_raw)
        else:
            v_f = _res(arith.ExtFOp(vec_f32_ty, _res(v_e_raw)))
        if acc is None:
            acc = _res(v_f)
        else:
            acc = _res(arith.AddFOp(_res(acc), _res(v_f)))

    if elem_ty == ir.F32Type.get():
        out_v = _res(acc)
    else:
        out_v = _res(arith.TruncFOp(vec_e_ty, _res(acc)))
    flir_vector.store(_res(out_v), out, [base_idx], alignment=16)


def _res(x):
    """Extract MLIR Value from various wrappers/opviews."""
    if x is None:
        return None
    if hasattr(x, "value"):
        try:
            return x.value
        except Exception:
            pass
    if hasattr(x, "result"):
        try:
            return x.result
        except Exception:
            pass
    if hasattr(x, "results"):
        try:
            rs = list(x.results)
            if len(rs) == 1:
                return rs[0]
        except Exception:
            pass
    return x


def _addi(a, b):
    from _mlir.dialects import arith
    return _res(arith.AddIOp(_res(a), _res(b)))


def _subi(a, b):
    from _mlir.dialects import arith
    return _res(arith.SubIOp(_res(a), _res(b)))


def _muli(a, b):
    from _mlir.dialects import arith
    return _res(arith.MulIOp(_res(a), _res(b)))


def _cmp(pred, a, b):
    from _mlir.dialects import arith
    return _res(arith.CmpIOp(pred, _res(a), _res(b)))


def _select(cond, t, f):
    from _mlir.dialects import arith
    return _res(arith.SelectOp(_res(cond), _res(t), _res(f)))


# ---- Low-level global memory primitives ----
# tmp_buffer and signal_buffer are allocated as hipDeviceMallocUncached
# (bypasses L1/TCP cache). For cross-GPU writes via XGMI, buffer_wbl2 is
# still needed to flush L2 (TCC) to HBM for XGMI fabric visibility.
# buffer_inv and sc1/sc0 on instruction modifiers are removed since the
# compiler generates optimal ordering, and s_waitcnt handles completion.

def _global_ld_u32(addr_i64):
    """Load u32 from a raw global pointer (signal buffer, L1-bypassed)."""
    from _mlir.dialects import llvm, rocdl
    v = llvm.InlineAsmOp(
        _i32(), [_res(addr_i64)],
        "global_load_dword $0, $1, off sc1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return _res(v)


def _global_st_u32(addr_i64, val_i32):
    """Store u32 to peer GPU's signal buffer (flush L2 for XGMI visibility)."""
    from _mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
    llvm.InlineAsmOp(
        None, [_res(addr_i64), _res(val_i32)],
        "global_store_dword $0, $1, off sc0 sc1", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def _global_ld_16b(addr_i64):
    """Load 16 bytes (vector<4xi32>) from a raw global pointer."""
    from _mlir.dialects import llvm, rocdl
    v4i32 = ir.VectorType.get([4], _i32())
    v = llvm.InlineAsmOp(
        v4i32, [_res(addr_i64)],
        "flat_load_dwordx4 $0, $1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return _res(v)


def _global_st_16b(addr_i64, v4i32_val):
    """Store 16 bytes (vector<4xi32>) to a raw global pointer."""
    from _mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(
        None, [_res(addr_i64), _res(v4i32_val)],
        "global_store_dwordx4 $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def _same_gpu_st_u32(addr_i64, val_i32):
    """Store u32 to local (same-GPU) signal buffer (no XGMI flush needed)."""
    from _mlir.dialects import llvm, rocdl
    llvm.InlineAsmOp(
        None, [_res(addr_i64), _res(val_i32)],
        "global_store_dword $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def _select_i64_by_lane(lane_i32, vals_i64):
    """Select one of vals_i64[0..7] by lane value via chained arith.select."""
    from _mlir.dialects import arith as _arith

    assert len(vals_i64) == 8
    out = vals_i64[0]
    for i in range(1, 8):
        pred = _cmp(_arith.CmpIPredicate.eq, lane_i32, _c_i32(i))
        out = _select(pred, vals_i64[i], out)
    return out


def _load_ptr_from_array(array_base_i64, index_i32):
    """Load an i64 pointer value from a device memory array at index_i32."""
    from _mlir.dialects import llvm, arith

    index_i64 = _res(arith.ExtUIOp(_i64(), _res(index_i32)))
    elem_addr_i64 = _addi(array_base_i64, _muli(index_i64, _c_i64(8)))
    ptr = llvm.IntToPtrOp(ir.Type.parse('!llvm.ptr'), _res(elem_addr_i64)).result
    return _res(llvm.LoadOp(_i64(), ptr).result)


def _spin_wait_ge_u32(addr_i64, target_u32):
    """Spin-wait until *addr >= target (uncachable signal buffer, no cache flush needed)."""
    from _mlir.dialects import arith, scf

    init_cur = _global_ld_u32(addr_i64)
    w = scf.WhileOp([_i32()], [_res(init_cur)])
    before = ir.Block.create_at_start(w.before, [_i32()])
    after = ir.Block.create_at_start(w.after, [_i32()])

    with ir.InsertionPoint(before):
        cur = before.arguments[0]
        need_wait = _cmp(arith.CmpIPredicate.ult, cur, target_u32)
        scf.ConditionOp(_res(need_wait), [_res(cur)])

    with ir.InsertionPoint(after):
        cur_new = _global_ld_u32(addr_i64)
        scf.YieldOp([_res(cur_new)])


def _addr_add(base_i64, off_bytes_i64):
    from _mlir.dialects import arith
    return _res(arith.AddIOp(_res(base_i64), _res(off_bytes_i64)))


def _mul_i64(a_i64, b_i64):
    from _mlir.dialects import arith
    return _res(arith.MulIOp(_res(a_i64), _res(b_i64)))


def _extui_i64(x_i32):
    from _mlir.dialects import arith
    return _res(arith.ExtUIOp(_i64(), _res(x_i32)))


def _start_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int):
    """AIter ROCm start_sync: write start flag to all peers, wait for all to arrive."""
    from _mlir.dialects import arith, gpu, scf

    lane_i32 = _res(lane_i32)
    rank_i32 = _res(rank_i32)
    bid_i32 = _res(bid_i32)
    self_sg_i64 = _res(self_sg_i64)

    flag_addr = _addr_add(self_sg_i64, _c_i64(_AITER_FLAG_OFF_B))
    flag_addr = _addr_add(flag_addr, _mul_i64(_extui_i64(bid_i32), _c_i64(4)))
    flag = _addi(_global_ld_u32(flag_addr), _c_i32(1))

    bid8 = _muli(bid_i32, _c_i32(8))
    lin_lane = _addi(bid8, lane_i32)
    start_wait_addr = _addr_add(self_sg_i64, _c_i64(_AITER_START_OFF_B))
    start_wait_addr = _addr_add(start_wait_addr, _mul_i64(_extui_i64(lin_lane), _c_i64(4)))

    lin_rank = _addi(bid8, rank_i32)
    start_rank_off = _addr_add(_c_i64(_AITER_START_OFF_B), _mul_i64(_extui_i64(lin_rank), _c_i64(4)))

    is_lane = _cmp(arith.CmpIPredicate.ult, lane_i32, _c_i32(ngpus))
    if_op = scf.IfOp(_res(is_lane), results_=[], hasElse=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _select_i64_by_lane(lane_i32, sgs_i64)
        _global_st_u32(_addr_add(peer_sg, start_rank_off), flag)  # cross-GPU write
        _spin_wait_ge_u32(start_wait_addr, flag)
        scf.YieldOp([])

    gpu.BarrierOp()
    is_t0 = _cmp(arith.CmpIPredicate.eq, lane_i32, _c_i32(0))
    if_t0 = scf.IfOp(_res(is_t0), results_=[], hasElse=False)
    with ir.InsertionPoint(if_t0.then_block):
        _same_gpu_st_u32(flag_addr, flag)  # same-GPU write (no XGMI flush needed)
        scf.YieldOp([])
    return flag_addr


def _end_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int):
    """AIter ROCm end_sync: write end flag to all peers, wait for all to finish."""
    from _mlir.dialects import arith, gpu, scf

    lane_i32 = _res(lane_i32)
    rank_i32 = _res(rank_i32)
    bid_i32 = _res(bid_i32)
    self_sg_i64 = _res(self_sg_i64)

    gpu.BarrierOp()
    flag_addr = _addr_add(self_sg_i64, _c_i64(_AITER_FLAG_OFF_B))
    flag_addr = _addr_add(flag_addr, _mul_i64(_extui_i64(bid_i32), _c_i64(4)))
    flag = _addi(_global_ld_u32(flag_addr), _c_i32(1))

    bid8 = _muli(bid_i32, _c_i32(8))
    lin_lane = _addi(bid8, lane_i32)
    end_wait_addr = _addr_add(self_sg_i64, _c_i64(_AITER_END_OFF_B))
    end_wait_addr = _addr_add(end_wait_addr, _mul_i64(_extui_i64(lin_lane), _c_i64(4)))

    lin_rank = _addi(bid8, rank_i32)
    end_rank_off = _addr_add(_c_i64(_AITER_END_OFF_B), _mul_i64(_extui_i64(lin_rank), _c_i64(4)))

    is_lane = _cmp(arith.CmpIPredicate.ult, lane_i32, _c_i32(ngpus))
    if_op = scf.IfOp(_res(is_lane), results_=[], hasElse=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _select_i64_by_lane(lane_i32, sgs_i64)
        _global_st_u32(_addr_add(peer_sg, end_rank_off), flag)  # cross-GPU write
        _spin_wait_ge_u32(end_wait_addr, flag)
        scf.YieldOp([])

    gpu.BarrierOp()
    is_t0 = _cmp(arith.CmpIPredicate.eq, lane_i32, _c_i32(0))
    if_t0 = scf.IfOp(_res(is_t0), results_=[], hasElse=False)
    with ir.InsertionPoint(if_t0.then_block):
        _same_gpu_st_u32(flag_addr, flag)  # same-GPU write (no XGMI flush needed)
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

    from flydsl.compiler.compiler import ensure_flir_python_extensions
    from kernels.kernels_common import stream_ptr_to_async_token
    from _mlir.dialects import arith, func, gpu, memref, scf, vector
    from flydsl.dialects.ext import flir as flir_ext

    ctx = ir.Context()
    ensure_flir_python_extensions(ctx)
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
                k1 = gpu.GPUFuncOp(ir.TypeAttr.get(k1_fty))
                k1.operation.attributes["sym_name"] = ir.StringAttr.get(f"aiter_signal_all_reduce_1stage_ws{world_size}")
                k1.operation.attributes["gpu.kernel"] = ir.UnitAttr.get()
                k1.operation.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([threads, 1, 1])
                k1.operation.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{threads},{threads}")
                k1_entry = ir.Block.create_at_start(k1.operation.regions[0], k1_args)
                with ir.InsertionPoint(k1_entry):
                    rank = k1_entry.arguments[0]
                    self_sg = k1_entry.arguments[1]
                    sgs = list(k1_entry.arguments[2:10]) + [k1_entry.arguments[2]] * max(0, 8 - 8)
                    ins = list(k1_entry.arguments[10:18])
                    out = k1_entry.arguments[18]

                    lane_i32 = _res(arith.IndexCastOp(i32, _res(gpu.thread_id("x"))))
                    bid_i32 = _res(arith.IndexCastOp(i32, _res(gpu.block_id("x"))))

                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                                self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    pack0 = _addi(_muli(bid_i32, _c_i32(threads)), lane_i32)
                    stride_p = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), _c_i32(threads))
                    loop = scf.WhileOp([i32], [_res(pack0)])
                    bfor = ir.Block.create_at_start(loop.before, [i32])
                    afor = ir.Block.create_at_start(loop.after, [i32])
                    with ir.InsertionPoint(bfor):
                        p = bfor.arguments[0]
                        cond = _cmp(arith.CmpIPredicate.ult, p, _c_i32(num_packs))
                        scf.ConditionOp(_res(cond), [_res(p)])
                    with ir.InsertionPoint(afor):
                        p = afor.arguments[0]
                        base_elem_i32 = _muli(p, _c_i32(pack_elems))
                        base_idx = _res(arith.IndexCastOp(idx, _res(base_elem_i32)))
                        _compute_allreduce_vector(
                            ins=ins, out=out, base_idx=_res(base_idx),
                            world_size=world_size, vec_e_ty=vec_e_ty,
                            vec_f32_ty=vec_f32_ty, elem_ty=elem_ty,
                        )
                        p_next = _addi(p, stride_p)
                        scf.YieldOp([_res(p_next)])

                    _end_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                              self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)
                    gpu.ReturnOp([])

                # ---- Kernel: 2-stage (reduce-scatter + all-gather) ----
                k2_args = [i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_tmp_ty] * 8 + [mem_out_ty]
                k2_fty = ir.FunctionType.get(k2_args, [])
                k2 = gpu.GPUFuncOp(ir.TypeAttr.get(k2_fty))
                k2.operation.attributes["sym_name"] = ir.StringAttr.get(f"aiter_signal_all_reduce_2stage_ws{world_size}")
                k2.operation.attributes["gpu.kernel"] = ir.UnitAttr.get()
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

                    lane_i32 = _res(arith.IndexCastOp(i32, _res(gpu.thread_id("x"))))
                    bid_i32 = _res(arith.IndexCastOp(i32, _res(gpu.block_id("x"))))
                    tid_i32 = _addi(_muli(bid_i32, _c_i32(threads)), lane_i32)
                    stride_i32 = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), _c_i32(threads))

                    start_p = _muli(rank, _c_i32(part_p))
                    is_last = _cmp(arith.CmpIPredicate.eq, rank, _c_i32(world_size - 1))
                    end_p = _select(is_last, _c_i32(num_packs), _addi(start_p, _c_i32(part_p)))
                    tmp_out = tmps[0]

                    sg_ptrtbl = memref.GetGlobalOp(sg_ptrtbl_ty, sg_ptrtbl_sym).result

                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                                self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    tnum_gpu = int(threads // world_size)
                    tnum_gpu_i32 = _c_i32(tnum_gpu)
                    warp_id = _res(arith.DivUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                    lane_id = _res(arith.RemUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                    tid_pack = _addi(_muli(bid_i32, tnum_gpu_i32), lane_id)
                    stride_pack = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), tnum_gpu_i32)

                    smem = memref.GetGlobalOp(smem_ty, smem_sym).result
                    ptrtbl = memref.GetGlobalOp(ptrtbl_ty, ptrtbl_sym).result
                    sg_ptrtbl = memref.GetGlobalOp(sg_ptrtbl_ty, sg_ptrtbl_sym).result

                    tmp_out_ptr = memref.ExtractAlignedPointerAsIndexOp(tmp_out).result
                    tmp_out_i64 = _res(arith.IndexCastOp(i64, _res(tmp_out_ptr)))

                    is_t0 = _cmp(arith.CmpIPredicate.eq, lane_i32, _c_i32(0))
                    if_t0 = scf.IfOp(_res(is_t0), results_=[], hasElse=False)
                    with ir.InsertionPoint(if_t0.then_block):
                        for i in range(world_size):
                            ip = memref.ExtractAlignedPointerAsIndexOp(ins[i]).result
                            in_i64 = _res(arith.IndexCastOp(i64, _res(ip)))
                            memref.StoreOp(_res(in_i64), ptrtbl, [_res(_c_index(i))])
                            tp = memref.ExtractAlignedPointerAsIndexOp(tmps[i]).result
                            tmp_i64 = _res(arith.IndexCastOp(i64, _res(tp)))
                            memref.StoreOp(_res(tmp_i64), ptrtbl, [_res(_c_index(8 + i))])
                        for i in range(world_size):
                            memref.StoreOp(_res(sgs[i]), sg_ptrtbl, [_res(_c_index(i))])
                        scf.YieldOp([])
                    gpu.BarrierOp()

                    idx_p = _addi(start_p, tid_pack)
                    parity0 = _c_i32(0)
                    loop1 = scf.WhileOp([i32, i32], [_res(idx_p), _res(parity0)])
                    b1 = ir.Block.create_at_start(loop1.before, [i32, i32])
                    a1 = ir.Block.create_at_start(loop1.after, [i32, i32])
                    with ir.InsertionPoint(b1):
                        cur = b1.arguments[0]
                        cond = _cmp(arith.CmpIPredicate.ult, cur, end_p)
                        scf.ConditionOp(_res(cond), [_res(cur), _res(b1.arguments[1])])
                    with ir.InsertionPoint(a1):
                        cur = a1.arguments[0]
                        parity = a1.arguments[1]

                        warp_idx = _res(arith.IndexCastOp(idx, _res(warp_id)))
                        in_i64 = memref.LoadOp(ptrtbl, [_res(warp_idx)]).result
                        raw = _global_ld_16b(_addr_add(in_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(cur))), _c_i64(16))))
                        sm_base = _muli(parity, _c_i32(threads))
                        sm_idx_i32 = _addi(sm_base, lane_i32)
                        sm_idx = _res(arith.IndexCastOp(idx, _res(sm_idx_i32)))
                        memref.StoreOp(_res(raw), smem, [_res(sm_idx)])
                        gpu.BarrierOp()

                        is_w0 = _cmp(arith.CmpIPredicate.eq, warp_id, _c_i32(0))
                        ifw0 = scf.IfOp(_res(is_w0), results_=[], hasElse=False)
                        with ir.InsertionPoint(ifw0.then_block):
                            acc = None
                            for i in range(world_size):
                                sm_i = _addi(_muli(_c_i32(i), tnum_gpu_i32), lane_id)
                                sm_i = _addi(sm_i, sm_base)
                                sm_i_idx = _res(arith.IndexCastOp(idx, _res(sm_i)))
                                raw_i = memref.LoadOp(smem, [_res(sm_i_idx)]).result
                                if dtype_str == "f32":
                                    vf = _res(vector.BitCastOp(v4f32, _res(raw_i)))
                                    acc = vf if acc is None else _res(arith.AddFOp(_res(acc), _res(vf)))
                                else:
                                    v16 = _res(vector.BitCastOp(v8f16, _res(raw_i)))
                                    v32 = _res(arith.ExtFOp(v8f32, _res(v16)))
                                    acc = v32 if acc is None else _res(arith.AddFOp(_res(acc), _res(v32)))
                            if dtype_str == "f32":
                                out_raw = _res(vector.BitCastOp(v4i32, _res(acc)))
                            else:
                                out16 = _res(arith.TruncFOp(v8f16, _res(acc)))
                                out_raw = _res(vector.BitCastOp(v4i32, _res(out16)))

                            rel_p = _subi(cur, start_p)
                            _global_st_16b(_addr_add(tmp_out_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(rel_p))), _c_i64(16))), out_raw)
                            scf.YieldOp([])

                        nxt = _addi(cur, stride_pack)
                        parity_next = _subi(_c_i32(1), parity)
                        scf.YieldOp([_res(nxt), _res(parity_next)])

                    _end_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                              self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    tmp_ptrs_i64 = []
                    for i in range(world_size):
                        tp = memref.ExtractAlignedPointerAsIndexOp(tmps[i]).result
                        tmp_ptrs_i64.append(_res(arith.IndexCastOp(i64, _res(tp))))

                    vec_ok = (num_packs % world_size == 0) and (world_size != 6)
                    if vec_ok:
                        tid_pack2 = _addi(_muli(bid_i32, tnum_gpu_i32), lane_id)
                        stride_pack2 = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), tnum_gpu_i32)

                        out_ptr = memref.ExtractAlignedPointerAsIndexOp(out).result
                        out_i64 = _res(arith.IndexCastOp(i64, _res(out_ptr)))

                        idx2 = tid_pack2
                        loop2 = scf.WhileOp([i32], [_res(idx2)])
                        b2 = ir.Block.create_at_start(loop2.before, [i32])
                        a2 = ir.Block.create_at_start(loop2.after, [i32])
                        with ir.InsertionPoint(b2):
                            cur = b2.arguments[0]
                            cond = _cmp(arith.CmpIPredicate.ult, cur, _c_i32(part_p))
                            scf.ConditionOp(_res(cond), [_res(cur)])
                        with ir.InsertionPoint(a2):
                            cur = a2.arguments[0]
                            sum_rw = _res(arith.AddIOp(_res(rank), _res(warp_id)).result)
                            if world_size in {2, 4, 8}:
                                dst_rank = _res(arith.AndIOp(_res(sum_rw), _res(_c_i32(world_size - 1))).result)
                            else:
                                dst_rank = _res(arith.RemUIOp(_res(sum_rw), _res(_c_i32(world_size))).result)
                            cur_idx = _res(arith.IndexCastOp(idx, _res(cur)))

                            def _load_from_tmp_memref(memref_val, idx_val):
                                if dtype_str == "f32":
                                    vf = _res(vector.LoadOp(v4f32, memref_val, [_res(idx_val)]))
                                    return _res(vector.BitCastOp(v4i32, _res(vf)))
                                else:
                                    v16 = _res(vector.LoadOp(v8f16, memref_val, [_res(idx_val)]))
                                    from _mlir.dialects import llvm
                                    return llvm.BitcastOp(v4i32, _res(v16)).result

                            tmp_if_chain = None
                            for i in range(world_size - 1, -1, -1):
                                is_match = _cmp(arith.CmpIPredicate.eq, warp_id, _c_i32(i))
                                if tmp_if_chain is None:
                                    if_op = scf.IfOp(_res(is_match), results_=[v4i32], hasElse=True)
                                    with ir.InsertionPoint(if_op.then_block):
                                        v = _load_from_tmp_memref(tmps[i], cur_idx)
                                        scf.YieldOp([_res(v)])
                                    with ir.InsertionPoint(if_op.else_block):
                                        v_fallback = _load_from_tmp_memref(tmps[0], cur_idx)
                                        scf.YieldOp([_res(v_fallback)])
                                    tmp_if_chain = if_op.results[0]
                                else:
                                    if_op = scf.IfOp(_res(is_match), results_=[v4i32], hasElse=True)
                                    with ir.InsertionPoint(if_op.then_block):
                                        v = _load_from_tmp_memref(tmps[i], cur_idx)
                                        scf.YieldOp([_res(v)])
                                    with ir.InsertionPoint(if_op.else_block):
                                        scf.YieldOp([_res(tmp_if_chain)])
                                    tmp_if_chain = if_op.results[0]
                            raw = _res(tmp_if_chain)

                            dst_pack = _addi(_muli(dst_rank, _c_i32(part_p)), cur)
                            dst_i64 = _res(arith.ExtUIOp(i64, _res(dst_pack)))
                            dst_off16 = _mul_i64(dst_i64, _c_i64(16))
                            dst_addr = _addr_add(out_i64, dst_off16)
                            _global_st_16b(dst_addr, raw)

                            nxt = _addi(cur, stride_pack2)
                            scf.YieldOp([_res(nxt)])
                    else:
                        idx2 = tid_i32
                        loop2 = scf.WhileOp([i32], [_res(idx2)])
                        b2 = ir.Block.create_at_start(loop2.before, [i32])
                        a2 = ir.Block.create_at_start(loop2.after, [i32])
                        with ir.InsertionPoint(b2):
                            cur = b2.arguments[0]
                            cond = _cmp(arith.CmpIPredicate.ult, cur, _c_i32(largest_part_p))
                            scf.ConditionOp(_res(cond), [_res(cur)])
                        with ir.InsertionPoint(a2):
                            cur = a2.arguments[0]
                            for p in range(world_size):
                                ok = None
                                if p == world_size - 1:
                                    ok = _res(arith.ConstantOp(ir.IntegerType.get_signless(1), 1).result)
                                else:
                                    ok = _cmp(arith.CmpIPredicate.ult, cur, _c_i32(part_p))
                                ifp = scf.IfOp(_res(ok), results_=[], hasElse=False)
                                with ir.InsertionPoint(ifp.then_block):
                                    src_elem_i32 = _muli(cur, _c_i32(pack_elems))
                                    src_idx = _res(arith.IndexCastOp(idx, _res(src_elem_i32)))
                                    v_e = _res(vector.LoadOp(vec_e_ty, tmps[p], [_res(src_idx)]))
                                    dst_pack = _addi(_muli(_c_i32(p), _c_i32(part_p)), cur)
                                    dst_elem_i32 = _muli(dst_pack, _c_i32(pack_elems))
                                    dst_idx = _res(arith.IndexCastOp(idx, _res(dst_elem_i32)))
                                    vector.StoreOp(_res(v_e), out, [_res(dst_idx)])
                                    scf.YieldOp([])

                            nxt = _addi(cur, stride_i32)
                            scf.YieldOp([_res(nxt)])
                    gpu.ReturnOp([])

                # ---- Kernel: 2-stage arr (CUDAGraph-compatible, loads pointers inside kernel) ----
                k2a_args = [i32, i64, i64, i64, i64, i64]
                k2a_fty = ir.FunctionType.get(k2a_args, [])
                k2a = gpu.GPUFuncOp(ir.TypeAttr.get(k2a_fty))
                k2a.operation.attributes["sym_name"] = ir.StringAttr.get(f"aiter_signal_all_reduce_2stage_arr_ws{world_size}")
                k2a.operation.attributes["gpu.kernel"] = ir.UnitAttr.get()
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

                    sgs = [_load_ptr_from_array(sg_ptrs_base, _c_i32(i)) for i in range(8)]
                    in_ptrs = [_load_ptr_from_array(in_ptrs_base, _c_i32(i)) for i in range(8)]
                    tmp_ptrs = [_load_ptr_from_array(tmp_ptrs_base, _c_i32(i)) for i in range(8)]

                    lane_i32 = _res(arith.IndexCastOp(i32, _res(gpu.thread_id("x"))))
                    bid_i32 = _res(arith.IndexCastOp(i32, _res(gpu.block_id("x"))))

                    start_p = _muli(rank, _c_i32(part_p))
                    is_last = _cmp(arith.CmpIPredicate.eq, rank, _c_i32(world_size - 1))
                    end_p = _select(is_last, _c_i32(num_packs), _addi(start_p, _c_i32(part_p)))

                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                                self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    tnum_gpu = int(threads // world_size)
                    tnum_gpu_i32 = _c_i32(tnum_gpu)
                    warp_id = _res(arith.DivUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                    lane_id = _res(arith.RemUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                    tid_pack = _addi(_muli(bid_i32, tnum_gpu_i32), lane_id)
                    stride_pack = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), tnum_gpu_i32)
                    smem = memref.GetGlobalOp(smem_ty, smem_sym).result
                    tmp_out_i64 = tmp_ptrs[0]

                    idx_p = _addi(start_p, tid_pack)
                    parity0 = _c_i32(0)
                    loop1 = scf.WhileOp([i32, i32], [_res(idx_p), _res(parity0)])
                    b1 = ir.Block.create_at_start(loop1.before, [i32, i32])
                    a1 = ir.Block.create_at_start(loop1.after, [i32, i32])
                    with ir.InsertionPoint(b1):
                        cur = b1.arguments[0]
                        cond = _cmp(arith.CmpIPredicate.ult, cur, end_p)
                        scf.ConditionOp(_res(cond), [_res(cur), _res(b1.arguments[1])])
                    with ir.InsertionPoint(a1):
                        cur = a1.arguments[0]
                        parity = a1.arguments[1]

                        in_base = _select_i64_by_lane(warp_id, in_ptrs)
                        raw = _global_ld_16b(_addr_add(in_base, _mul_i64(_res(arith.ExtUIOp(i64, _res(cur))), _c_i64(16))))

                        sm_base = _muli(parity, _c_i32(threads))
                        sm_idx_i32 = _addi(sm_base, lane_i32)
                        sm_idx = _res(arith.IndexCastOp(idx, _res(sm_idx_i32)))
                        memref.StoreOp(_res(raw), smem, [_res(sm_idx)])
                        gpu.BarrierOp()

                        is_w0 = _cmp(arith.CmpIPredicate.eq, warp_id, _c_i32(0))
                        ifw0 = scf.IfOp(_res(is_w0), results_=[], hasElse=False)
                        with ir.InsertionPoint(ifw0.then_block):
                            acc = None
                            for i in range(world_size):
                                sm_i = _addi(_muli(_c_i32(i), tnum_gpu_i32), lane_id)
                                sm_i = _addi(sm_i, sm_base)
                                sm_i_idx = _res(arith.IndexCastOp(idx, _res(sm_i)))
                                raw_i = memref.LoadOp(smem, [_res(sm_i_idx)]).result
                                if dtype_str == "f32":
                                    vf = _res(vector.BitCastOp(v4f32, _res(raw_i)))
                                    acc = vf if acc is None else _res(arith.AddFOp(_res(acc), _res(vf)))
                                else:
                                    v16 = _res(vector.BitCastOp(v8f16, _res(raw_i)))
                                    v32 = _res(arith.ExtFOp(v8f32, _res(v16)))
                                    acc = v32 if acc is None else _res(arith.AddFOp(_res(acc), _res(v32)))
                            if dtype_str == "f32":
                                out_raw = _res(vector.BitCastOp(v4i32, _res(acc)))
                            else:
                                out16 = arith.TruncFOp(v8f16, _res(acc)).result
                                from _mlir.dialects import llvm
                                out_raw = llvm.BitcastOp(v4i32, _res(out16)).result

                            rel_p = _subi(cur, start_p)
                            _global_st_16b(_addr_add(tmp_out_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(rel_p))), _c_i64(16))), out_raw)
                            scf.YieldOp([])

                        nxt = _addi(cur, stride_pack)
                        parity_next = _subi(_c_i32(1), parity)
                        scf.YieldOp([_res(nxt), _res(parity_next)])

                    _end_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                              self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    vec_ok = (num_packs % world_size == 0) and (world_size != 6)
                    if vec_ok:
                        tid_pack2 = _addi(_muli(bid_i32, tnum_gpu_i32), lane_id)
                        stride_pack2 = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), tnum_gpu_i32)

                        idx2 = tid_pack2
                        loop2 = scf.WhileOp([i32], [_res(idx2)])
                        b2 = ir.Block.create_at_start(loop2.before, [i32])
                        a2 = ir.Block.create_at_start(loop2.after, [i32])
                        with ir.InsertionPoint(b2):
                            cur = b2.arguments[0]
                            cond = _cmp(arith.CmpIPredicate.ult, cur, _c_i32(part_p))
                            scf.ConditionOp(_res(cond), [_res(cur)])
                        with ir.InsertionPoint(a2):
                            cur = a2.arguments[0]
                            sum_rw = _res(arith.AddIOp(_res(rank), _res(warp_id)).result)
                            if world_size in {2, 4, 8}:
                                dst_rank = _res(arith.AndIOp(_res(sum_rw), _res(_c_i32(world_size - 1))).result)
                            else:
                                dst_rank = _res(arith.RemUIOp(_res(sum_rw), _res(_c_i32(world_size))).result)

                            tmp_base = _select_i64_by_lane(warp_id, tmp_ptrs)
                            raw = _global_ld_16b(_addr_add(tmp_base, _mul_i64(_res(arith.ExtUIOp(i64, _res(cur))), _c_i64(16))))

                            dst_pack = _addi(_muli(dst_rank, _c_i32(part_p)), cur)
                            _global_st_16b(_addr_add(out_ptr_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(dst_pack))), _c_i64(16))), raw)

                            nxt = _addi(cur, stride_pack2)
                            scf.YieldOp([_res(nxt)])

                    gpu.ReturnOp([])

                # ---- Kernel: 2-stage write_mode (large tensors, N > 512*4096) ----
                # Stage1: scatter local input to REMOTE tmp buffers.
                # Stage2: read local tmp (all ranks' data), reduce, write to REMOTE output.
                # end_sync is omitted (causes GPU hang); host-side barrier handles visibility.
                k2w_args = [i32, i64, i64, i64, i64, i64]
                k2w_fty = ir.FunctionType.get(k2w_args, [])
                k2w = gpu.GPUFuncOp(ir.TypeAttr.get(k2w_fty))
                k2w.operation.attributes["sym_name"] = ir.StringAttr.get(f"aiter_signal_all_reduce_2stage_write_mode_ws{world_size}")
                k2w.operation.attributes["gpu.kernel"] = ir.UnitAttr.get()
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

                    sgs = [_load_ptr_from_array(sg_ptrs_base, _c_i32(i)) for i in range(8)]
                    out_ptrs = [_load_ptr_from_array(out_ptrs_base, _c_i32(i)) for i in range(8)]

                    lane_i32 = _res(arith.IndexCastOp(i32, _res(gpu.thread_id("x"))))
                    bid_i32 = _res(arith.IndexCastOp(i32, _res(gpu.block_id("x"))))

                    tnum_gpu = int(threads // world_size)
                    tnum_gpu_i32 = _c_i32(tnum_gpu)
                    import math
                    log2_tnum = int(math.log2(tnum_gpu))
                    warp_id = _res(arith.ShRUIOp(_res(lane_i32), _c_i32(log2_tnum)))
                    warp_base = _muli(warp_id, tnum_gpu_i32)
                    lane_id = _res(arith.SubIOp(_res(lane_i32), _res(warp_base)))
                    tid_pack = _addi(_muli(bid_i32, tnum_gpu_i32), lane_id)
                    stride_pack = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), tnum_gpu_i32)

                    smem = memref.GetGlobalOp(smem_ty, smem_sym).result
                    tmp_out_i64 = _load_ptr_from_array(tmp_ptrs_base, rank)

                    # Stage1: write local input to REMOTE tmp buffers
                    start_w = _muli(warp_id, _c_i32(part_p))
                    is_last_w = _cmp(arith.CmpIPredicate.eq, warp_id, _c_i32(world_size - 1))
                    end_w_if = scf.IfOp(_res(is_last_w), results_=[i32], hasElse=True)
                    with ir.InsertionPoint(end_w_if.then_block):
                        scf.YieldOp([_res(_c_i32(num_packs))])
                    with ir.InsertionPoint(end_w_if.else_block):
                        scf.YieldOp([_res(_addi(start_w, _c_i32(part_p)))])
                    end_w = _res(end_w_if.results[0])

                    idx_s1 = _addi(start_w, tid_pack)
                    loop_s1 = scf.WhileOp([i32, i32], [_res(idx_s1), _res(stride_pack)])
                    bs1 = ir.Block.create_at_start(loop_s1.before, [i32, i32])
                    as1 = ir.Block.create_at_start(loop_s1.after, [i32, i32])
                    with ir.InsertionPoint(bs1):
                        cur = bs1.arguments[0]
                        cond = _cmp(arith.CmpIPredicate.ult, cur, end_w)
                        scf.ConditionOp(_res(cond), [_res(cur), _res(bs1.arguments[1])])
                    with ir.InsertionPoint(as1):
                        cur = as1.arguments[0]
                        stride_pack_s1 = as1.arguments[1]
                        raw = _global_ld_16b(_addr_add(inp_ptr_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(cur))), _c_i64(16))))
                        rel_idx = _subi(cur, start_w)
                        dst_off = _addi(_muli(rank, _c_i32(part_p)), rel_idx)
                        dst_tmp = _load_ptr_from_array(tmp_ptrs_base, warp_id)
                        _global_st_16b(_addr_add(dst_tmp, _mul_i64(_res(arith.ExtUIOp(i64, _res(dst_off))), _c_i64(16))), raw)
                        nxt = _addi(cur, stride_pack_s1)
                        scf.YieldOp([_res(nxt), _res(stride_pack_s1)])

                    # Signal all ranks that stage1 is complete
                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32,
                                self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size)

                    # Stage2: read local tmp, reduce, write reduced result to REMOTE outputs
                    idx_s2 = tid_pack
                    part_p_i32 = _c_i32(part_p)
                    loop_s2 = scf.WhileOp([i32, i32], [_res(idx_s2), _res(stride_pack)])
                    bs2 = ir.Block.create_at_start(loop_s2.before, [i32, i32])
                    as2 = ir.Block.create_at_start(loop_s2.after, [i32, i32])
                    with ir.InsertionPoint(bs2):
                        cur = bs2.arguments[0]
                        stride_pack_loop = bs2.arguments[1]
                        cond = _cmp(arith.CmpIPredicate.ult, cur, part_p_i32)
                        scf.ConditionOp(_res(cond), [_res(cur), _res(bs2.arguments[1])])
                    with ir.InsertionPoint(as2):
                        cur = as2.arguments[0]
                        stride_pack_loop = as2.arguments[1]

                        src_off = _addi(_muli(warp_id, _c_i32(part_p)), cur)
                        src_off_i64 = _res(arith.ExtUIOp(i64, _res(src_off)))
                        load_addr = _addr_add(tmp_out_i64, _mul_i64(src_off_i64, _c_i64(16)))
                        raw = _global_ld_16b(load_addr)

                        sm_idx = _res(arith.IndexCastOp(idx, _res(lane_i32)))
                        memref.StoreOp(_res(raw), smem, [_res(sm_idx)])
                        gpu.BarrierOp()

                        warp_id_local = _res(arith.ShRUIOp(_res(lane_i32), _c_i32(log2_tnum)))
                        warp_base_local = _muli(warp_id_local, _c_i32(tnum_gpu))
                        lane_id_local = _res(arith.SubIOp(_res(lane_i32), _res(warp_base_local)))

                        raw_vals = []
                        for i in range(world_size):
                            sm_i = _addi(_muli(_c_i32(i), _c_i32(tnum_gpu)), lane_id_local)
                            sm_i_idx = _res(arith.IndexCastOp(idx, _res(sm_i)))
                            raw_vals.append(memref.LoadOp(smem, [_res(sm_i_idx)]).result)

                        acc = None
                        for i in range(world_size):
                            raw_i = raw_vals[i]
                            if dtype_str == "f32":
                                vf = _res(vector.BitCastOp(v4f32, _res(raw_i)))
                                acc = vf if acc is None else _res(arith.AddFOp(_res(acc), _res(vf)))
                            else:
                                v16 = _res(vector.BitCastOp(v8f16, _res(raw_i)))
                                v32 = _res(arith.ExtFOp(v8f32, _res(v16)))
                                acc = v32 if acc is None else _res(arith.AddFOp(_res(acc), _res(v32)))
                        if dtype_str == "f32":
                            out_raw = _res(vector.BitCastOp(v4i32, _res(acc)))
                        else:
                            out16 = _res(arith.TruncFOp(v8f16, _res(acc)))
                            out_raw = _res(vector.BitCastOp(v4i32, _res(out16)))

                        dst_out_off = _addi(_muli(rank, _c_i32(part_p)), cur)
                        dst_byte_off = _mul_i64(_res(arith.ExtUIOp(i64, _res(dst_out_off))), _c_i64(16))

                        dst_ptr = out_ptrs[0]
                        for w in range(1, world_size):
                            is_warp_w = _cmp(arith.CmpIPredicate.eq, warp_id_local, _c_i32(w))
                            dst_ptr = _res(arith.SelectOp(_res(is_warp_w), _res(out_ptrs[w]), _res(dst_ptr)))
                        _global_st_16b(_addr_add(dst_ptr, dst_byte_off), out_raw)

                        nxt = _addi(cur, stride_pack_loop)
                        scf.YieldOp([_res(nxt), _res(stride_pack_loop)])

                    gpu.BarrierOp()
                    from _mlir.dialects import rocdl
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
                gx = _res(arith.IndexCastOp(idx, _res(grid_x_i32)))
                one = _c_index(1)
                bx = _c_index(threads)
                kops = [_res(rank), _res(self_sg)] + [_res(x) for x in sgs] + [_res(x) for x in ins] + [_res(out)]
                stream_token = stream_ptr_to_async_token(stream_ptr)
                flir_ext.gpu_ext.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_1stage_ws{world_size}"],
                    grid_size=(gx, one, one), block_size=(bx, one, one),
                    kernel_operands=kops, async_dependencies=[stream_token],
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
                gx = _res(arith.IndexCastOp(idx, _res(grid_x_i32)))
                one = _c_index(1)
                bx = _c_index(threads)
                kops = [_res(rank), _res(self_sg)] + [_res(x) for x in sgs] + [_res(x) for x in ins] + [_res(x) for x in tmps] + [_res(out)]
                stream_token = stream_ptr_to_async_token(stream_ptr)
                flir_ext.gpu_ext.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_2stage_ws{world_size}"],
                    grid_size=(gx, one, one), block_size=(bx, one, one),
                    kernel_operands=kops, async_dependencies=[stream_token],
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
                gx = _res(arith.IndexCastOp(idx, _res(grid_x_i32)))
                one = _c_index(1)
                bx = _c_index(threads)
                kops = [_res(rank), _res(self_sg), _res(sg_ptrs_array_base), _res(in_ptrs_array_base), _res(tmp_ptrs_array_base), _res(out_ptr_i64)]
                stream_token = stream_ptr_to_async_token(stream_ptr)
                flir_ext.gpu_ext.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_2stage_arr_ws{world_size}"],
                    grid_size=(gx, one, one), block_size=(bx, one, one),
                    kernel_operands=kops, async_dependencies=[stream_token],
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
                gx = _res(arith.IndexCastOp(idx, _res(grid_x_i32)))
                one = _c_index(1)
                bx = _c_index(threads)
                kops = [_res(rank), _res(self_sg), _res(sg_ptrs_array_base), _res(inp_ptr_i64), _res(out_ptrs_array_base), _res(tmp_ptrs_array_base)]
                stream_token = stream_ptr_to_async_token(stream_ptr)
                flir_ext.gpu_ext.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_2stage_write_mode_ws{world_size}"],
                    grid_size=(gx, one, one), block_size=(bx, one, one),
                    kernel_operands=kops, async_dependencies=[stream_token],
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
