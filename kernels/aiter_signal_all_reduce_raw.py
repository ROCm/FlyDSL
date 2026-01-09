"""Raw MLIR (no flydsl.dialects.ext wrappers) implementation of AIter-style all-reduce on ROCm.

Goal:
- Match AIter's ROCm signal protocol (start/end/_flag) for multi-GPU correctness.
- Provide 1-stage and 2-stage (reduce-scatter + all-gather) kernels.

This file intentionally uses only `_mlir.dialects.*` ODS ops so we don't hit the
Value-wrapper issues seen when mixing `flydsl.dialects.ext.*` with `llvm.inline_asm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from _mlir import ir


# AIter ROCm Signal layout (bytes). Matches the ISA dump for gfx942.
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


def _res(x):
    """Best-effort get an MLIR `Value` out of various wrappers/opviews."""
    if x is None:
        return None
    # FlyDSL ext wrappers.
    if hasattr(x, "value"):
        try:
            return x.value
        except Exception:
            pass
    # ODS opviews.
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


def _asm_void(asm: str):
    from _mlir.dialects import llvm

    # NOTE: this bindings requires `res=None` for void inline asm (res=[] errors).
    llvm.InlineAsmOp(
        None,
        [],
        asm,
        "",
        has_side_effects=True,
    )


def _asm_waitcnt():
    _asm_void("s_waitcnt vmcnt(0) lgkmcnt(0)")


def _asm_buffer_wbl2():
    _asm_void("buffer_wbl2 sc0 sc1")


def _asm_buffer_inv():
    _asm_void("buffer_inv sc1")


def _asm_ld_u32_sc1(addr_i64):
    from _mlir.dialects import llvm

    v = llvm.InlineAsmOp(
        _i32(),
        [_res(addr_i64)],
        "global_load_dword $0, $1, off sc1",
        "=v,v",
        has_side_effects=True,
    ).result
    _asm_waitcnt()
    return _res(v)


def _asm_st_u32_sc0sc1(addr_i64, val_i32):
    from _mlir.dialects import llvm

    llvm.InlineAsmOp(
        None,
        [_res(addr_i64), _res(val_i32)],
        "global_store_dword $0, $1, off sc0 sc1",
        "v,v",
        has_side_effects=True,
    )
    _asm_waitcnt()


def _asm_st_u32(addr_i64, val_i32):
    from _mlir.dialects import llvm

    llvm.InlineAsmOp(
        None,
        [_res(addr_i64), _res(val_i32)],
        "global_store_dword $0, $1, off",
        "v,v",
        has_side_effects=True,
    )
    _asm_waitcnt()


def _asm_ld_16b(addr_i64):
    """Load 16 bytes from a global address (flat_load_dwordx4). Returns vector<4xi32>."""
    from _mlir.dialects import llvm

    v4i32 = ir.VectorType.get([4], _i32())
    v = llvm.InlineAsmOp(
        v4i32,
        [_res(addr_i64)],
        "flat_load_dwordx4 $0, $1",
        "=v,v",
        has_side_effects=True,
    ).result
    _asm_waitcnt()
    return _res(v)


def _asm_st_16b(addr_i64, v4i32_val):
    """Store 16 bytes to a global address (global_store_dwordx4)."""
    from _mlir.dialects import llvm

    llvm.InlineAsmOp(
        None,
        [_res(addr_i64), _res(v4i32_val)],
        "global_store_dwordx4 $0, $1, off",
        "v,v",
        has_side_effects=True,
    )
    _asm_waitcnt()


def _select_i64_by_lane(lane_i32, vals_i64):
    """Select one of vals_i64[0..7] by lane value via chained arith.select."""
    from _mlir.dialects import arith as _arith

    assert len(vals_i64) == 8
    out = vals_i64[0]
    for i in range(1, 8):
        pred = _cmp(_arith.CmpIPredicate.eq, lane_i32, _c_i32(i))
        out = _select(pred, vals_i64[i], out)
    return out


def _spin_wait_ge_u32(addr_i64, target_u32, *, do_inv: bool, max_iters: int, debug_addr_i64=None, debug_tag_u32=None):
    """Spin until *addr >= target with a max-iter timeout.

    If `debug_addr_i64` is provided and timeout triggers, write a small debug record:
      [0]=tag, [1]=target, [2]=last_loaded
    """
    from _mlir.dialects import arith, scf

    init = _c_i32(0)
    w = scf.WhileOp([_i32(), _i32()], [_res(init), _res(init)])
    # This bindings does not auto-create region blocks for scf.while; create them explicitly.
    before = ir.Block.create_at_start(w.before, [_i32(), _i32()])
    after = ir.Block.create_at_start(w.after, [_i32(), _i32()])

    # before: condition
    with ir.InsertionPoint(before):
        cur = _asm_ld_u32_sc1(addr_i64)
        if do_inv:
            _asm_buffer_inv()
        # Continue spinning while (cur < target) and (it < max_iters).
        need_wait = _cmp(arith.CmpIPredicate.ult, cur, target_u32)
        it = before.arguments[0]
        last = before.arguments[1]
        it_next = _addi(it, _c_i32(1))
        ok_it = _cmp(arith.CmpIPredicate.ult, it, _c_i32(int(max_iters)))
        cont = _cmp(arith.CmpIPredicate.and_, need_wait, ok_it) if hasattr(arith.CmpIPredicate, "and_") else None
        # Some bindings don't have and_; emulate with arith.andi on i1.
        if cont is None:
            andv = arith.AndIOp(_res(need_wait), _res(ok_it)).result
            cont = andv
        scf.ConditionOp(_res(cont), [_res(it_next), _res(cur)])

    # after: just yield the carried value (keeps spinning)
    with ir.InsertionPoint(after):
        # If we exited because it reached max_iters (timeout), optionally dump debug.
        if debug_addr_i64 is not None:
            it = after.arguments[0]
            last = after.arguments[1]
            timed_out = _cmp(arith.CmpIPredicate.uge, it, _c_i32(int(max_iters)))
            ifop = scf.IfOp(_res(timed_out), results_=[], hasElse=False)
            with ir.InsertionPoint(ifop.then_block):
                tag = _c_i32(int(debug_tag_u32) if debug_tag_u32 is not None else 0)
                _asm_st_u32(debug_addr_i64, tag)
                _asm_st_u32(_addr_add(debug_addr_i64, _c_i64(4)), _res(target_u32))
                _asm_st_u32(_addr_add(debug_addr_i64, _c_i64(8)), _res(last))
                scf.YieldOp([])
        scf.YieldOp([_res(after.arguments[0]), _res(after.arguments[1])])


def _addr_add(base_i64, off_bytes_i64):
    from _mlir.dialects import arith

    return _res(arith.AddIOp(_res(base_i64), _res(off_bytes_i64)))


def _mul_i64(a_i64, b_i64):
    from _mlir.dialects import arith

    return _res(arith.MulIOp(_res(a_i64), _res(b_i64)))


def _extui_i64(x_i32):
    from _mlir.dialects import arith

    return _res(arith.ExtUIOp(_i64(), _res(x_i32)))


def _start_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int, meta_size_bytes: int, max_spin: int):
    """Emit AIter ROCm start_sync<ngpus>(sg, self_sg, rank)."""
    from _mlir.dialects import arith, gpu, scf
    lane_i32 = _res(lane_i32)
    rank_i32 = _res(rank_i32)
    bid_i32 = _res(bid_i32)
    self_sg_i64 = _res(self_sg_i64)

    # flag_addr = self_sg + FLAG_OFF + bid*4
    flag_addr = _addr_add(self_sg_i64, _c_i64(_AITER_FLAG_OFF_B))
    flag_addr = _addr_add(flag_addr, _mul_i64(_extui_i64(bid_i32), _c_i64(4)))
    flag0 = _asm_ld_u32_sc1(flag_addr)
    flag = _addi(flag0, _c_i32(1))

    is_lane = _cmp(arith.CmpIPredicate.ult, lane_i32, _c_i32(ngpus))
    # start_wait_addr = self_sg + START_OFF + (bid*8 + lane)*4
    bid8 = _muli(bid_i32, _c_i32(8))
    lin_lane = _addi(bid8, lane_i32)
    start_wait_addr = _addr_add(self_sg_i64, _c_i64(_AITER_START_OFF_B))
    start_wait_addr = _addr_add(start_wait_addr, _mul_i64(_extui_i64(lin_lane), _c_i64(4)))

    # start_rank_off = START_OFF + (bid*8 + rank)*4
    lin_rank = _addi(bid8, rank_i32)
    start_rank_off = _c_i64(_AITER_START_OFF_B)
    start_rank_off = _addr_add(start_rank_off, _mul_i64(_extui_i64(lin_rank), _c_i64(4)))

    # if lane < ngpus: store start to peer's start[bid][rank] and wait
    if_op = scf.IfOp(_res(is_lane), results_=[], hasElse=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _select_i64_by_lane(lane_i32, sgs_i64)
        peer_start_addr = _addr_add(peer_sg, start_rank_off)
        _asm_st_u32_sc0sc1(peer_start_addr, flag)
        # debug record lives in tmp region: self_sg + meta_size + 64 + (bid*8+lane)*16
        lin = _addi(_muli(bid8, _c_i32(1)), lane_i32)  # just reuse linear-ish
        dbg = _addr_add(self_sg_i64, _c_i64(int(meta_size_bytes) + 64))
        dbg = _addr_add(dbg, _mul_i64(_extui_i64(lin), _c_i64(16)))
        _spin_wait_ge_u32(start_wait_addr, flag, do_inv=False, max_iters=int(max_spin), debug_addr_i64=dbg, debug_tag_u32=0x53544152)  # 'STAR'
        scf.YieldOp([])

    gpu.BarrierOp()
    # tid0 updates flag
    is_t0 = _cmp(arith.CmpIPredicate.eq, lane_i32, _c_i32(0))
    if_t0 = scf.IfOp(_res(is_t0), results_=[], hasElse=False)
    with ir.InsertionPoint(if_t0.then_block):
        _asm_st_u32(flag_addr, flag)
        scf.YieldOp([])
    gpu.BarrierOp()
    return flag_addr


def _end_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int, final_sync: bool, meta_size_bytes: int, max_spin: int):
    """Emit AIter ROCm end_sync<ngpus, final_sync>(...)."""
    from _mlir.dialects import arith, gpu, scf
    lane_i32 = _res(lane_i32)
    rank_i32 = _res(rank_i32)
    bid_i32 = _res(bid_i32)
    self_sg_i64 = _res(self_sg_i64)

    gpu.BarrierOp()
    flag_addr = _addr_add(self_sg_i64, _c_i64(_AITER_FLAG_OFF_B))
    flag_addr = _addr_add(flag_addr, _mul_i64(_extui_i64(bid_i32), _c_i64(4)))
    flag0 = _asm_ld_u32_sc1(flag_addr)
    flag = _addi(flag0, _c_i32(1))

    is_lane = _cmp(arith.CmpIPredicate.ult, lane_i32, _c_i32(ngpus))
    bid8 = _muli(bid_i32, _c_i32(8))
    lin_lane = _addi(bid8, lane_i32)
    end_wait_addr = _addr_add(self_sg_i64, _c_i64(_AITER_END_OFF_B))
    end_wait_addr = _addr_add(end_wait_addr, _mul_i64(_extui_i64(lin_lane), _c_i64(4)))

    lin_rank = _addi(bid8, rank_i32)
    end_rank_off = _c_i64(_AITER_END_OFF_B)
    end_rank_off = _addr_add(end_rank_off, _mul_i64(_extui_i64(lin_rank), _c_i64(4)))

    if_op = scf.IfOp(_res(is_lane), results_=[], hasElse=False)
    with ir.InsertionPoint(if_op.then_block):
        peer_sg = _select_i64_by_lane(lane_i32, sgs_i64)
        peer_end_addr = _addr_add(peer_sg, end_rank_off)
        if not final_sync:
            _asm_buffer_wbl2()
        _asm_st_u32_sc0sc1(peer_end_addr, flag)
        lin = _addi(_muli(bid8, _c_i32(1)), lane_i32)
        dbg = _addr_add(self_sg_i64, _c_i64(int(meta_size_bytes) + 128))
        dbg = _addr_add(dbg, _mul_i64(_extui_i64(lin), _c_i64(16)))
        _spin_wait_ge_u32(end_wait_addr, flag, do_inv=(not final_sync), max_iters=int(max_spin), debug_addr_i64=dbg, debug_tag_u32=0x454E4421)  # 'END!'
        scf.YieldOp([])

    gpu.BarrierOp()
    is_t0 = _cmp(arith.CmpIPredicate.eq, lane_i32, _c_i32(0))
    if_t0 = scf.IfOp(_res(is_t0), results_=[], hasElse=False)
    with ir.InsertionPoint(if_t0.then_block):
        _asm_st_u32(flag_addr, flag)
        scf.YieldOp([])
    gpu.BarrierOp()


def build_aiter_signal_allreduce_raw_module(*, N: int, dtype_str: str, world_size: int, threads: int = 256, meta_size_bytes: int = 5504, max_spin: int = 20000000) -> ir.Module:
    """Build an `ir.Module` that exports host entrypoints:
    - `run_1stage(rank, grid_x, self_sg, sg0..sg7, in0..in7, out)`
    - `run_2stage(rank, grid_x, self_sg, sg0..sg7, in0..in7, tmp0..tmp7, out)`
    - `run_2stage_ptr(rank, grid_x, self_sg, sg0..sg7, in0_ptr..in7_ptr, tmp0_ptr..tmp7_ptr, out_ptr)`
    """
    if world_size not in {2, 4, 6, 8}:
        raise ValueError("world_size must be one of {2,4,6,8}")
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if N <= 0:
        raise ValueError("N must be > 0")

    from flydsl.compiler.compiler import ensure_flir_python_extensions
    from _mlir.dialects import arith, func, gpu, memref, scf, vector

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
        # Required by the GPU dialect for host-side `gpu.launch_func` wrappers.
        m.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
        with ir.InsertionPoint(m.body):
            gpu_mod = gpu.GPUModuleOp("aiter_signal")
            gpu_mod.bodyRegion.blocks.append()

            # ---- gpu.func kernels ----
            with ir.InsertionPoint(gpu_mod.bodyRegion.blocks[0]):
                # ---- LDS scratch for AIter-style shared-memory reduction ----
                # Layout: memref<(2*threads) x vector<4xi32>, #gpu.address_space<workgroup>>
                # Ping-pong (double-buffer) to reduce barriers in stage1:
                #   - writers store to buffer[parity]
                #   - one barrier
                #   - warp0 reduces from buffer[parity]
                # Next iter flips parity so writers won't clobber what warp0 is reading.
                # Note: threads must be divisible by world_size for 2stage shared-memory stage1.
                if (threads % world_size) != 0:
                    raise ValueError(f"threads={threads} must be divisible by world_size={world_size} for shared-memory stage1")
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

                # ---- LDS pointer table to avoid select chains ----
                # Layout: memref<16xi64, workgroup>
                # [0..7]  : rotated input base pointers (as i64)
                # [8..15] : rotated tmp base pointers (as i64)
                ptrtbl_ty = ir.MemRefType.get([16], i64, memory_space=lds_space)
                ptrtbl_sym = f"aiter_signal_ptrtbl_ws{world_size}"
                memref.GlobalOp(
                    sym_name=ir.StringAttr.get(ptrtbl_sym),
                    type_=ptrtbl_ty,
                    initial_value=None,
                    constant=False,
                    alignment=16,
                )

                # Kernel 1-stage
                k1_args = [i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_out_ty]
                k1_fty = ir.FunctionType.get(k1_args, [])
                k1 = gpu.GPUFuncOp(ir.TypeAttr.get(k1_fty))
                k1.operation.attributes["sym_name"] = ir.StringAttr.get(f"aiter_signal_all_reduce_1stage_ws{world_size}")
                k1.operation.attributes["gpu.kernel"] = ir.UnitAttr.get()
                k1_entry = ir.Block.create_at_start(k1.operation.regions[0], k1_args)
                with ir.InsertionPoint(k1_entry):
                    rank = k1_entry.arguments[0]
                    self_sg = k1_entry.arguments[1]
                    sgs = list(k1_entry.arguments[2:10]) + [k1_entry.arguments[2]] * max(0, 8 - 8)
                    ins = list(k1_entry.arguments[10:18])
                    out = k1_entry.arguments[18]

                    lane_i32 = _res(arith.IndexCastOp(i32, _res(gpu.thread_id("x"))))
                    bid_i32 = _res(arith.IndexCastOp(i32, _res(gpu.block_id("x"))))

                    # start_sync
                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32, self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size, meta_size_bytes=int(meta_size_bytes), max_spin=int(max_spin))

                    # compute packs with stride loop:
                    # for (p = bid*threads + lane; p < num_packs; p += gridDim*threads)
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

                        acc = None
                        for r in range(world_size):
                            v_e = _res(vector.LoadOp(vec_e_ty, ins[r], [_res(base_idx)]))
                            v_f = v_e if elem_ty == ir.F32Type.get() else _res(arith.ExtFOp(vec_f32_ty, _res(v_e)))
                            acc = v_f if acc is None else _res(arith.AddFOp(_res(acc), _res(v_f)))
                        out_v = acc if elem_ty == ir.F32Type.get() else _res(arith.TruncFOp(vec_e_ty, _res(acc)))
                        vector.StoreOp(_res(out_v), out, [_res(base_idx)])

                        p_next = _addi(p, stride_p)
                        scf.YieldOp([_res(p_next)])

                    # end_sync(final)
                    _end_sync(
                        lane_i32=lane_i32,
                        rank_i32=rank,
                        bid_i32=bid_i32,
                        self_sg_i64=self_sg,
                        sgs_i64=sgs,
                        ngpus=world_size,
                        final_sync=True,
                        meta_size_bytes=int(meta_size_bytes),
                        max_spin=int(max_spin),
                    )
                    gpu.ReturnOp([])

                # Kernel 2-stage (single kernel: stage1 + end_sync + stage2)
                k2_args = [i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_tmp_ty] * 8 + [mem_out_ty]
                k2_fty = ir.FunctionType.get(k2_args, [])
                k2 = gpu.GPUFuncOp(ir.TypeAttr.get(k2_fty))
                k2.operation.attributes["sym_name"] = ir.StringAttr.get(f"aiter_signal_all_reduce_2stage_ws{world_size}")
                k2.operation.attributes["gpu.kernel"] = ir.UnitAttr.get()
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

                    # start/end in packs for this rank
                    start_p = _muli(rank, _c_i32(part_p))
                    is_last = _cmp(arith.CmpIPredicate.eq, rank, _c_i32(world_size - 1))
                    end_p = _select(is_last, _c_i32(num_packs), _addi(start_p, _c_i32(part_p)))

                    # tmp_out: tmps are provided in rotated order at host:
                    #   tmps[i] corresponds to target=(rank+i)%ws, so tmps[0] is always this rank.
                    tmp_out = tmps[0]

                    # start_sync
                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32, self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size, meta_size_bytes=int(meta_size_bytes), max_spin=int(max_spin))

                    # stage1: reduce-scatter packs (AIter-style shared-memory)
                    # tnum_gpu = threads / ngpus; warp_id = tid / tnum_gpu; lane_id = tid % tnum_gpu.
                    tnum_gpu = int(threads // world_size)
                    tnum_gpu_i32 = _c_i32(tnum_gpu)
                    warp_id = _res(arith.DivUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                    lane_id = _res(arith.RemUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                    tid_pack = _addi(_muli(bid_i32, tnum_gpu_i32), lane_id)
                    stride_pack = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), tnum_gpu_i32)

                    # Get LDS scratch
                    smem = memref.GetGlobalOp(smem_ty, smem_sym).result
                    ptrtbl = memref.GetGlobalOp(ptrtbl_ty, ptrtbl_sym).result

                    # Precompute tmp_out base pointer (as i64) for inline-asm stores.
                    tmp_out_ptr = memref.ExtractAlignedPointerAsIndexOp(tmp_out).result
                    tmp_out_i64 = _res(arith.IndexCastOp(i64, _res(tmp_out_ptr)))

                    # Initialize pointer table once per block (thread0 only).
                    # This removes per-thread select chains in stage1/2.
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
                        scf.YieldOp([])
                    gpu.BarrierOp()

                    # p = start_p + tid_pack; p < end_p; p += stride_pack, with parity ping-pong.
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

                        # Base address = in_ptr(warp_id) + cur*16B
                        warp_idx = _res(arith.IndexCastOp(idx, _res(warp_id)))
                        in_i64 = memref.LoadOp(ptrtbl, [_res(warp_idx)]).result
                        cur_i64 = _res(arith.ExtUIOp(i64, _res(cur)))
                        pack_off = _mul_i64(cur_i64, _c_i64(16))
                        in_addr = _addr_add(in_i64, pack_off)

                        raw = _asm_ld_16b(in_addr)  # v4i32
                        # smem index = parity*threads + tid
                        sm_base = _muli(parity, _c_i32(threads))
                        sm_idx_i32 = _addi(sm_base, lane_i32)
                        sm_idx = _res(arith.IndexCastOp(idx, _res(sm_idx_i32)))
                        memref.StoreOp(_res(raw), smem, [_res(sm_idx)])
                        gpu.BarrierOp()

                        # warp0 reduces across GPUs and writes tmp_out
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
                            rel_i64 = _res(arith.ExtUIOp(i64, _res(rel_p)))
                            tmp_off = _mul_i64(rel_i64, _c_i64(16))
                            tmp_addr = _addr_add(tmp_out_i64, tmp_off)
                            _asm_st_16b(tmp_addr, out_raw)
                            scf.YieldOp([])

                        # No second barrier: flip parity so next iteration writers won't clobber current reads.
                        nxt = _addi(cur, stride_pack)
                        parity_next = _subi(_c_i32(1), parity)  # toggles 0<->1
                        scf.YieldOp([_res(nxt), _res(parity_next)])

                    # end_sync (not final): must match ISA: wbl2 before store, inv during wait
                    _end_sync(
                        lane_i32=lane_i32,
                        rank_i32=rank,
                        bid_i32=bid_i32,
                        self_sg_i64=self_sg,
                        sgs_i64=sgs,
                        ngpus=world_size,
                        final_sync=False,
                        meta_size_bytes=int(meta_size_bytes),
                        max_spin=int(max_spin),
                    )
                    # stage2: all-gather packs
                    #
                    # Fast path (matches AIter cross_device_reduce_2stage):
                    # - only valid when num_packs % ngpus == 0 (i.e., bytes % (ngpus*16) == 0)
                    #   and ngpus != 6 (AIter uses naive path for ws=6).
                    # - each thread gathers a single warp_id chunk:
                    #     dst_rank = (warp_id + rank) % ngpus
                    #     out[dst_rank*part + idx] = tmp_from(dst_rank)[idx]
                    vec_ok = (num_packs % world_size == 0) and (world_size != 6)
                    if vec_ok:
                        # Reuse warp_id/lane_id/tnum_gpu from stage1.
                        tnum_gpu = int(threads // world_size)
                        tnum_gpu_i32 = _c_i32(tnum_gpu)
                        warp_id = _res(arith.DivUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                        lane_id = _res(arith.RemUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                        tid_pack2 = _addi(_muli(bid_i32, tnum_gpu_i32), lane_id)
                        stride_pack2 = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), tnum_gpu_i32)

                        # Base pointers (i64) for output.
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
                            # dst_rank = (rank + warp_id) % ws
                            # For power-of-two ws (2/4/8), use bitmask to avoid integer division.
                            sum_rw = _res(arith.AddIOp(_res(rank), _res(warp_id)).result)
                            if world_size in {2, 4, 8}:
                                dst_rank = _res(arith.AndIOp(_res(sum_rw), _res(_c_i32(world_size - 1))).result)
                            else:
                                dst_rank = _res(arith.RemUIOp(_res(sum_rw), _res(_c_i32(world_size))).result)
                            # tmps are rotated at host: tmps[warp_id] corresponds to (rank+warp_id)%ws.
                            warp_idx = _res(arith.IndexCastOp(idx, _res(warp_id)))
                            tmp_tbl_idx = _res(arith.AddIOp(_res(_c_index(8)), _res(warp_idx)).result)
                            tmp_i64 = memref.LoadOp(ptrtbl, [_res(tmp_tbl_idx)]).result

                            cur_i64 = _res(arith.ExtUIOp(i64, _res(cur)))
                            off16 = _mul_i64(cur_i64, _c_i64(16))
                            src_addr = _addr_add(tmp_i64, off16)
                            raw = _asm_ld_16b(src_addr)

                            dst_pack = _addi(_muli(dst_rank, _c_i32(part_p)), cur)
                            dst_i64 = _res(arith.ExtUIOp(i64, _res(dst_pack)))
                            dst_off16 = _mul_i64(dst_i64, _c_i64(16))
                            dst_addr = _addr_add(out_i64, dst_off16)
                            _asm_st_16b(dst_addr, raw)

                            nxt = _addi(cur, stride_pack2)
                            scf.YieldOp([_res(nxt)])
                    else:
                        # Generic path: handles remainder in last rank by bounds checks.
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
                            # for each rank p: if p==last or cur < part_p
                            for p in range(world_size):
                                ok = None
                                if p == world_size - 1:
                                    ok = _res(arith.ConstantOp(ir.IntegerType.get_signless(1), 1).result)
                                else:
                                    ok = _cmp(arith.CmpIPredicate.ult, cur, _c_i32(part_p))
                                ifp = scf.IfOp(_res(ok), results_=[], hasElse=False)
                                with ir.InsertionPoint(ifp.then_block):
                                    # load tmp pack from rank p
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

                # Kernel 2-stage ptr variant: accept raw i64 base pointers for ins/tmps/out.
                # This avoids memref descriptor handling overhead in stage2.
                k2p_args = [i32, i64] + [i64] * 8 + [i64] * 8 + [i64] * 8 + [i64]
                k2p_fty = ir.FunctionType.get(k2p_args, [])
                k2p = gpu.GPUFuncOp(ir.TypeAttr.get(k2p_fty))
                k2p.operation.attributes["sym_name"] = ir.StringAttr.get(f"aiter_signal_all_reduce_2stage_ptr_ws{world_size}")
                k2p.operation.attributes["gpu.kernel"] = ir.UnitAttr.get()
                k2p_entry = ir.Block.create_at_start(k2p.operation.regions[0], k2p_args)
                with ir.InsertionPoint(k2p_entry):
                    rank = k2p_entry.arguments[0]
                    self_sg = k2p_entry.arguments[1]
                    sgs = list(k2p_entry.arguments[2:10])
                    in_ptrs = list(k2p_entry.arguments[10:18])
                    tmp_ptrs = list(k2p_entry.arguments[18:26])
                    out_ptr_i64 = k2p_entry.arguments[26]

                    lane_i32 = _res(arith.IndexCastOp(i32, _res(gpu.thread_id("x"))))
                    bid_i32 = _res(arith.IndexCastOp(i32, _res(gpu.block_id("x"))))

                    # start/end in packs for this rank
                    start_p = _muli(rank, _c_i32(part_p))
                    is_last = _cmp(arith.CmpIPredicate.eq, rank, _c_i32(world_size - 1))
                    end_p = _select(is_last, _c_i32(num_packs), _addi(start_p, _c_i32(part_p)))

                    # start_sync
                    _start_sync(
                        lane_i32=lane_i32,
                        rank_i32=rank,
                        bid_i32=bid_i32,
                        self_sg_i64=self_sg,
                        sgs_i64=sgs,
                        ngpus=world_size,
                        meta_size_bytes=int(meta_size_bytes),
                        max_spin=int(max_spin),
                    )

                    # stage1 (keep existing LDS approach but using in_ptrs/tmp_ptrs directly)
                    tnum_gpu = int(threads // world_size)
                    tnum_gpu_i32 = _c_i32(tnum_gpu)
                    warp_id = _res(arith.DivUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                    lane_id = _res(arith.RemUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                    tid_pack = _addi(_muli(bid_i32, tnum_gpu_i32), lane_id)
                    stride_pack = _muli(_res(arith.IndexCastOp(i32, _res(gpu.grid_dim("x")))), tnum_gpu_i32)

                    smem = memref.GetGlobalOp(smem_ty, smem_sym).result
                    # tmp_out is always tmp_ptrs[0] because host passes rotated order: tmp0 corresponds to self rank.
                    tmp_out_i64 = tmp_ptrs[0]

                    # p = start_p + tid_pack; p < end_p; p += stride_pack, with parity ping-pong.
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

                        # in_base = in_ptrs[warp_id] (rotated order)
                        in_base = _select_i64_by_lane(warp_id, in_ptrs)
                        cur_i64 = _res(arith.ExtUIOp(i64, _res(cur)))
                        off16 = _mul_i64(cur_i64, _c_i64(16))
                        in_addr = _addr_add(in_base, off16)
                        raw = _asm_ld_16b(in_addr)

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
                                # Use llvm.bitcast instead of vector.bitcast here; some environments are
                                # picky about operand wrappers for vector.bitcast.
                                from _mlir.dialects import llvm

                                out_raw = llvm.BitcastOp(v4i32, _res(out16)).result

                            rel_p = _subi(cur, start_p)
                            rel_i64 = _res(arith.ExtUIOp(i64, _res(rel_p)))
                            tmp_off = _mul_i64(rel_i64, _c_i64(16))
                            tmp_addr = _addr_add(tmp_out_i64, tmp_off)
                            _asm_st_16b(tmp_addr, out_raw)
                            scf.YieldOp([])

                        nxt = _addi(cur, stride_pack)
                        parity_next = _subi(_c_i32(1), parity)
                        scf.YieldOp([_res(nxt), _res(parity_next)])

                    # end_sync (not final)
                    _end_sync(
                        lane_i32=lane_i32,
                        rank_i32=rank,
                        bid_i32=bid_i32,
                        self_sg_i64=self_sg,
                        sgs_i64=sgs,
                        ngpus=world_size,
                        final_sync=False,
                        meta_size_bytes=int(meta_size_bytes),
                        max_spin=int(max_spin),
                    )

                    # stage2: all-gather fast path (16B) using raw pointers
                    vec_ok = (num_packs % world_size == 0) and (world_size != 6)
                    if vec_ok:
                        tnum_gpu = int(threads // world_size)
                        tnum_gpu_i32 = _c_i32(tnum_gpu)
                        warp_id = _res(arith.DivUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
                        lane_id = _res(arith.RemUIOp(_res(lane_i32), _res(tnum_gpu_i32)))
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
                            cur_i64 = _res(arith.ExtUIOp(i64, _res(cur)))
                            off16 = _mul_i64(cur_i64, _c_i64(16))
                            src_addr = _addr_add(tmp_base, off16)
                            raw = _asm_ld_16b(src_addr)

                            dst_pack = _addi(_muli(dst_rank, _c_i32(part_p)), cur)
                            dst_i64 = _res(arith.ExtUIOp(i64, _res(dst_pack)))
                            dst_off16 = _mul_i64(dst_i64, _c_i64(16))
                            dst_addr = _addr_add(out_ptr_i64, dst_off16)
                            _asm_st_16b(dst_addr, raw)

                            nxt = _addi(cur, stride_pack2)
                            scf.YieldOp([_res(nxt)])

                    gpu.ReturnOp([])

            # ---- host entrypoints (llvm.emit_c_interface) ----
            # kernel symbol refs
            k1_ref = ir.SymbolRefAttr.get(["aiter_signal", f"aiter_signal_all_reduce_1stage_ws{world_size}"])
            k2_ref = ir.SymbolRefAttr.get(["aiter_signal", f"aiter_signal_all_reduce_2stage_ws{world_size}"])
            k2p_ref = ir.SymbolRefAttr.get(["aiter_signal", f"aiter_signal_all_reduce_2stage_ptr_ws{world_size}"])

            # run_1stage
            run1_args = [i32, i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_out_ty]
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

                gx = _res(arith.IndexCastOp(idx, _res(grid_x_i32)))
                one = _c_index(1)
                bx = _c_index(threads)
                kops = [_res(rank), _res(self_sg)] + [_res(x) for x in sgs] + [_res(x) for x in ins] + [_res(out)]
                gpu.LaunchFuncOp(
                    None,
                    [],
                    k1_ref,
                    _res(gx),
                    _res(one),
                    _res(one),
                    _res(bx),
                    _res(one),
                    _res(one),
                    kops,
                )
                func.ReturnOp([])

            # run_2stage
            run2_args = [i32, i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_tmp_ty] * 8 + [mem_out_ty]
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
                gx = _res(arith.IndexCastOp(idx, _res(grid_x_i32)))
                one = _c_index(1)
                bx = _c_index(threads)
                kops = [_res(rank), _res(self_sg)] + [_res(x) for x in sgs] + [_res(x) for x in ins] + [_res(x) for x in tmps] + [_res(out)]
                gpu.LaunchFuncOp(None, [], k2_ref, _res(gx), _res(one), _res(one), _res(bx), _res(one), _res(one), kops)
                func.ReturnOp([])

            # run_2stage_ptr
            run2p_args = [i32, i32, i64] + [i64] * 8 + [i64] * 8 + [i64] * 8 + [i64]
            run2p_fty = ir.FunctionType.get(run2p_args, [])
            run2p = func.FuncOp("run_2stage_ptr", run2p_fty)
            run2p.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            b2p = run2p.add_entry_block()
            with ir.InsertionPoint(b2p):
                rank = b2p.arguments[0]
                grid_x_i32 = b2p.arguments[1]
                self_sg = b2p.arguments[2]
                sgs = list(b2p.arguments[3:11])
                in_ptrs = list(b2p.arguments[11:19])
                tmp_ptrs = list(b2p.arguments[19:27])
                out_ptr_i64 = b2p.arguments[27]
                gx = _res(arith.IndexCastOp(idx, _res(grid_x_i32)))
                one = _c_index(1)
                bx = _c_index(threads)
                kops = [_res(rank), _res(self_sg)] + [_res(x) for x in sgs] + [_res(x) for x in in_ptrs] + [_res(x) for x in tmp_ptrs] + [_res(out_ptr_i64)]
                gpu.LaunchFuncOp(None, [], k2p_ref, _res(gx), _res(one), _res(one), _res(bx), _res(one), _res(one), kops)
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

    # Let the executor prefer base_ptr if available.
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

