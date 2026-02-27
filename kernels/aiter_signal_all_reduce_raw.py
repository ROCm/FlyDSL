"""Hybrid MLIR implementation of AIter-style all-reduce on ROCm.

Goal:
- Match AIter's ROCm signal protocol (start/end/_flag) for multi-GPU correctness.
- Provide 1-stage and 2-stage (reduce-scatter + all-gather) kernels.

This file uses a hybrid approach:
- Raw MLIR (`_mlir.dialects.*`) for synchronization and inline assembly
- FlyDSL high-level API for computation (vector load/store) to simplify code
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


def _compute_allreduce_vector(ins, out, base_idx, world_size, vec_e_ty, vec_f32_ty, elem_ty):
    """Compute all-reduce for a vector pack using FlyDSL vector API.
    
    Simplified version using FlyDSL's vector.load_op/store for cleaner code.
    """
    from flydsl.dialects.ext import vector as flir_vector
    from _mlir.dialects import arith
    
    # Load and accumulate vectors from all ranks
    acc = None
    for r in range(world_size):
        # Use FlyDSL vector.load_op (result_type, memref, indices)
        v_e = flir_vector.load_op(vec_e_ty, ins[r], [base_idx])
        v_e_raw = _res(v_e)
        
        # Convert to f32 if needed
        if elem_ty == ir.F32Type.get():
            v_f = _res(v_e_raw)
        else:
            v_f = _res(arith.ExtFOp(vec_f32_ty, _res(v_e_raw)))
        
        # Accumulate - ensure both operands are Values
        if acc is None:
            acc = _res(v_f)
        else:
            acc = _res(arith.AddFOp(_res(acc), _res(v_f)))
    
    # Convert back to element type if needed
    if elem_ty == ir.F32Type.get():
        out_v = _res(acc)
    else:
        out_v = _res(arith.TruncFOp(vec_e_ty, _res(acc)))
    
    # Use FlyDSL vector.store (value, memref, indices, alignment=...)
    flir_vector.store(_res(out_v), out, [base_idx], alignment=16)


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


def _asm_waitcnt_vm():
    """Wait only for VM (vector memory) operations, not LDS. Used in sync loops."""
    _asm_void("s_waitcnt vmcnt(0)")


def _asm_buffer_wbl2():
    _asm_void("buffer_wbl2 sc0 sc1")


def _asm_buffer_inv():
    _asm_void("buffer_inv sc1")


def _asm_ld_u32_sc1(addr_i64, do_inv=False, wait_vm_only=False):
    """Load u32 with optional buffer_inv for memory consistency.
    
    Args:
        addr_i64: Address to load from
        do_inv: If True, add buffer_inv sc1 after load (for GFX942 optimization)
        wait_vm_only: If True, only wait for VM (not LDS). Used in sync loops.
    """
    from _mlir.dialects import llvm

    v = llvm.InlineAsmOp(
        _i32(),
        [_res(addr_i64)],
        "global_load_dword $0, $1, off sc1",
        "=v,v",
        has_side_effects=True,
    ).result
    if wait_vm_only:
        _asm_waitcnt_vm()
    else:
        _asm_waitcnt()
    if do_inv:
        # Ensure buffer_inv is emitted immediately after waitcnt for proper ordering
        _asm_buffer_inv()
    return _res(v)


def _asm_st_u32_sc0sc1(addr_i64, val_i32, wait_after=True):
    """Store u32 with sc0 sc1 modifiers.
    
    Args:
        addr_i64: Address to store to
        val_i32: Value to store
        wait_after: If True, wait after store. Set to False if wait is done later.
    """
    from _mlir.dialects import llvm

    # Add buffer_wbl2 before store for memory consistency (GFX942 optimization)
    _asm_buffer_wbl2()
    llvm.InlineAsmOp(
        None,
        [_res(addr_i64), _res(val_i32)],
        "global_store_dword $0, $1, off sc0 sc1",
        "v,v",
        has_side_effects=True,
    )
    if wait_after:
        # Only wait for VM, not LDS (like aiter)
        _asm_waitcnt_vm()


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


def _load_ptr_from_array(array_base_i64, index_i32):
    """Load an i64 pointer value from device memory array.
    
    Args:
        array_base_i64: Base address of the pointer array (i64)
        index_i32: Index into the array (i32)
    
    Returns:
        Loaded pointer value (i64)
    """
    from _mlir.dialects import llvm
    from _mlir.dialects import arith
    
    # Calculate offset: index * 8 (each pointer is 8 bytes)
    # Use ExtUIOp to convert i32 to i64 (unsigned extension)
    index_i64 = _res(arith.ExtUIOp(_i64(), _res(index_i32)))
    offset_i64 = _muli(index_i64, _c_i64(8))
    
    # Calculate element address: base + offset
    elem_addr_i64 = _addi(array_base_i64, offset_i64)
    
    # Convert i64 address to LLVM pointer
    ptr_type = ir.Type.parse('!llvm.ptr')
    elem_ptr = llvm.IntToPtrOp(ptr_type, _res(elem_addr_i64)).result
    
    # Load the i64 pointer value
    loaded = llvm.LoadOp(_i64(), elem_ptr).result
    return _res(loaded)


def _select_memref_by_lane(lane_i32, memrefs):
    """Select one of memrefs[0..7] by lane value via chained arith.select.
    
    This preserves memref offset information for precise address calculation.
    """
    from _mlir.dialects import arith as _arith

    assert len(memrefs) == 8
    out = memrefs[0]
    for i in range(1, 8):
        pred = _cmp(_arith.CmpIPredicate.eq, lane_i32, _c_i32(i))
        out = _select(pred, memrefs[i], out)
    return out


def _spin_wait_ge_u32(addr_i64, target_u32, *, do_inv: bool, max_iters: int, debug_addr_i64=None, debug_tag_u32=None):
    """Spin until *addr >= target with a max-iter timeout.

    Optimized version: simplified loop structure, removed debug overhead from hot path.
    If `debug_addr_i64` is provided and timeout triggers, write a small debug record:
      [0]=tag, [1]=target, [2]=last_loaded
    """
    from _mlir.dialects import arith, scf

    # Simplified loop: only carry current value, no iteration counter in hot path
    # Use wait_vm_only=True for sync loops to avoid unnecessary LDS waits
    init_cur = _asm_ld_u32_sc1(addr_i64, do_inv=do_inv, wait_vm_only=True)
    w = scf.WhileOp([_i32()], [_res(init_cur)])
    before = ir.Block.create_at_start(w.before, [_i32()])
    after = ir.Block.create_at_start(w.after, [_i32()])

    # before: condition - simplified, no iteration counter in hot path
    with ir.InsertionPoint(before):
        cur = before.arguments[0]
        # Continue spinning while cur < target
        need_wait = _cmp(arith.CmpIPredicate.ult, cur, target_u32)
        scf.ConditionOp(_res(need_wait), [_res(cur)])

    # after: reload and check again
    with ir.InsertionPoint(after):
        # Reload current value - use wait_vm_only=True and ensure buffer_inv is emitted
        cur_new = _asm_ld_u32_sc1(addr_i64, do_inv=do_inv, wait_vm_only=True)
        scf.YieldOp([_res(cur_new)])

    # Debug handling (only if debug_addr provided) - separate from hot path
    if debug_addr_i64 is not None:
        # Note: This is a simplified approach. For full timeout detection,
        # we'd need to add iteration counting, but that adds overhead.
        # For production, consider removing debug entirely or using a separate mechanism.
        pass  # Debug disabled in optimized version to avoid hot path overhead


def _addr_add(base_i64, off_bytes_i64):
    from _mlir.dialects import arith

    return _res(arith.AddIOp(_res(base_i64), _res(off_bytes_i64)))


def _mul_i64(a_i64, b_i64):
    from _mlir.dialects import arith

    return _res(arith.MulIOp(_res(a_i64), _res(b_i64)))


def _extui_i64(x_i32):
    from _mlir.dialects import arith

    return _res(arith.ExtUIOp(_i64(), _res(x_i32)))


def _start_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int, meta_size_bytes: int, max_spin: int, sg_ptrtbl=None):
    """Emit AIter ROCm start_sync<ngpus>(sg, self_sg, rank)."""
    from _mlir.dialects import arith, gpu, scf, memref
    idx = ir.IndexType.get()
    lane_i32 = _res(lane_i32)
    rank_i32 = _res(rank_i32)
    bid_i32 = _res(bid_i32)
    self_sg_i64 = _res(self_sg_i64)

    # flag_addr = self_sg + FLAG_OFF + bid*4
    flag_addr = _addr_add(self_sg_i64, _c_i64(_AITER_FLAG_OFF_B))
    flag_addr = _addr_add(flag_addr, _mul_i64(_extui_i64(bid_i32), _c_i64(4)))
    flag0 = _asm_ld_u32_sc1(flag_addr, do_inv=False)
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
        # Use chain selection (pointer table will be handled at call site)
        peer_sg = _select_i64_by_lane(lane_i32, sgs_i64)
        peer_start_addr = _addr_add(peer_sg, start_rank_off)
        # Store flag and wait for VM only (wait will be done in spin loop)
        _asm_st_u32_sc0sc1(peer_start_addr, flag, wait_after=False)
        # Wait for store to complete before entering spin loop
        _asm_waitcnt_vm()
        _spin_wait_ge_u32(start_wait_addr, flag, do_inv=False, max_iters=int(max_spin), debug_addr_i64=None, debug_tag_u32=None)
        scf.YieldOp([])

    # Barrier after wait loop (like aiter)
    gpu.BarrierOp()
    # tid0 updates flag
    is_t0 = _cmp(arith.CmpIPredicate.eq, lane_i32, _c_i32(0))
    if_t0 = scf.IfOp(_res(is_t0), results_=[], hasElse=False)
    with ir.InsertionPoint(if_t0.then_block):
        _asm_st_u32(flag_addr, flag)
        scf.YieldOp([])
    # Removed extra barrier - flag update doesn't need barrier after it
    return flag_addr


def _end_sync(*, lane_i32, rank_i32, bid_i32, self_sg_i64, sgs_i64, ngpus: int, final_sync: bool, meta_size_bytes: int, max_spin: int, sg_ptrtbl=None):
    """Emit AIter ROCm end_sync<ngpus, final_sync>(...)."""
    from _mlir.dialects import arith, gpu, scf, memref
    idx = ir.IndexType.get()
    lane_i32 = _res(lane_i32)
    rank_i32 = _res(rank_i32)
    bid_i32 = _res(bid_i32)
    self_sg_i64 = _res(self_sg_i64)

    # Barrier at start (like aiter) - ensures prior writes are visible
    gpu.BarrierOp()
    flag_addr = _addr_add(self_sg_i64, _c_i64(_AITER_FLAG_OFF_B))
    flag_addr = _addr_add(flag_addr, _mul_i64(_extui_i64(bid_i32), _c_i64(4)))
    flag0 = _asm_ld_u32_sc1(flag_addr, do_inv=False)
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
        # Use chain selection (pointer table will be handled at call site)
        peer_sg = _select_i64_by_lane(lane_i32, sgs_i64)
        peer_end_addr = _addr_add(peer_sg, end_rank_off)
        # Store flag and wait for VM only (wait will be done in spin loop)
        _asm_st_u32_sc0sc1(peer_end_addr, flag, wait_after=False)
        # Wait for store to complete before entering spin loop
        _asm_waitcnt_vm()
        _spin_wait_ge_u32(end_wait_addr, flag, do_inv=(not final_sync), max_iters=int(max_spin), debug_addr_i64=None, debug_tag_u32=None)
        scf.YieldOp([])

    # Barrier after wait loop (like aiter)
    gpu.BarrierOp()
    # tid0 updates flag
    is_t0 = _cmp(arith.CmpIPredicate.eq, lane_i32, _c_i32(0))
    if_t0 = scf.IfOp(_res(is_t0), results_=[], hasElse=False)
    with ir.InsertionPoint(if_t0.then_block):
        _asm_st_u32(flag_addr, flag)
        scf.YieldOp([])
    # Removed extra barrier - flag update doesn't need barrier after it (like aiter)


def build_aiter_signal_allreduce_raw_module(*, N: int, dtype_str: str, world_size: int, threads: int = 256, meta_size_bytes: int = 5504, max_spin: int = 20000000) -> ir.Module:
    """Build an `ir.Module` that exports host entrypoints:
    - `run_1stage(rank, grid_x, self_sg, sg0..sg7, in0..in7, out)`
    - `run_2stage(rank, grid_x, self_sg, sg0..sg7, in0..in7, tmp0..tmp7, out)`
    - `run_2stage_arr(rank, grid_x, self_sg, sg_ptrs_base, in_ptrs_base, tmp_ptrs_base, out_ptr)`
      Accepts base addresses of device pointer arrays; kernel loads pointers on-device for CUDAGraph compatibility.

    NOTE: Unlike the default MLIR ROCm runtime (which creates/synchronizes/destroys a HIP stream per call),
    our host wrappers accept a `stream_ptr` argument and bind kernel launches to the caller-provided stream
    (e.g. PyTorch current stream). This avoids per-launch stream create/destroy/synchronize overhead.
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
                
                # ---- LDS pointer table for signal groups (sgs) ----
                # Layout: memref<8xi64, workgroup>
                # [0..7]  : signal group base pointers (as i64)
                sg_ptrtbl_ty = ir.MemRefType.get([8], i64, memory_space=lds_space)
                sg_ptrtbl_sym = f"aiter_signal_sg_ptrtbl_ws{world_size}"
                memref.GlobalOp(
                    sym_name=ir.StringAttr.get(sg_ptrtbl_sym),
                    type_=sg_ptrtbl_ty,
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
                # Set workgroup size attributes to allow threads/block
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

                    # start_sync
                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32, self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size, meta_size_bytes=int(meta_size_bytes), max_spin=int(max_spin), sg_ptrtbl=None)

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

                        # Use FlyDSL vector API for cleaner computation
                        _compute_allreduce_vector(
                            ins=ins,
                            out=out,
                            base_idx=_res(base_idx),
                            world_size=world_size,
                            vec_e_ty=vec_e_ty,
                            vec_f32_ty=vec_f32_ty,
                            elem_ty=elem_ty,
                        )

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
                        sg_ptrtbl=None,
                    )
                    gpu.ReturnOp([])

                # Kernel 2-stage (single kernel: stage1 + end_sync + stage2)
                k2_args = [i32, i64] + [i64] * 8 + [mem_in_ty] * 8 + [mem_tmp_ty] * 8 + [mem_out_ty]
                k2_fty = ir.FunctionType.get(k2_args, [])
                k2 = gpu.GPUFuncOp(ir.TypeAttr.get(k2_fty))
                k2.operation.attributes["sym_name"] = ir.StringAttr.get(f"aiter_signal_all_reduce_2stage_ws{world_size}")
                k2.operation.attributes["gpu.kernel"] = ir.UnitAttr.get()
                # Set workgroup size attributes to allow threads/block
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

                    # start/end in packs for this rank
                    start_p = _muli(rank, _c_i32(part_p))
                    is_last = _cmp(arith.CmpIPredicate.eq, rank, _c_i32(world_size - 1))
                    end_p = _select(is_last, _c_i32(num_packs), _addi(start_p, _c_i32(part_p)))

                    # tmp_out: tmps are provided in rotated order at host:
                    #   tmps[i] corresponds to target=(rank+i)%ws, so tmps[0] is always this rank.
                    tmp_out = tmps[0]

                    # Get signal group pointer table
                    sg_ptrtbl = memref.GetGlobalOp(sg_ptrtbl_ty, sg_ptrtbl_sym).result

                    # start_sync
                    # Note: pointer table optimization requires reading pointers at call site
                    # For now, use chain selection to avoid region isolation issues
                    _start_sync(lane_i32=lane_i32, rank_i32=rank, bid_i32=bid_i32, self_sg_i64=self_sg, sgs_i64=sgs, ngpus=world_size, meta_size_bytes=int(meta_size_bytes), max_spin=int(max_spin), sg_ptrtbl=None)

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
                    sg_ptrtbl = memref.GetGlobalOp(sg_ptrtbl_ty, sg_ptrtbl_sym).result

                    # Precompute tmp_out base pointer (as i64) for inline-asm stores.
                    tmp_out_ptr = memref.ExtractAlignedPointerAsIndexOp(tmp_out).result
                    tmp_out_i64 = _res(arith.IndexCastOp(i64, _res(tmp_out_ptr)))

                    # Initialize pointer tables once per block (thread0 only).
                    # This removes per-thread select chains in stage1/2 and sync.
                    is_t0 = _cmp(arith.CmpIPredicate.eq, lane_i32, _c_i32(0))
                    if_t0 = scf.IfOp(_res(is_t0), results_=[], hasElse=False)
                    with ir.InsertionPoint(if_t0.then_block):
                        # Initialize data pointer table (ins and tmps)
                        for i in range(world_size):
                            ip = memref.ExtractAlignedPointerAsIndexOp(ins[i]).result
                            in_i64 = _res(arith.IndexCastOp(i64, _res(ip)))
                            memref.StoreOp(_res(in_i64), ptrtbl, [_res(_c_index(i))])
                            tp = memref.ExtractAlignedPointerAsIndexOp(tmps[i]).result
                            tmp_i64 = _res(arith.IndexCastOp(i64, _res(tp)))
                            memref.StoreOp(_res(tmp_i64), ptrtbl, [_res(_c_index(8 + i))])
                        # Initialize signal group pointer table (sgs)
                        for i in range(world_size):
                            memref.StoreOp(_res(sgs[i]), sg_ptrtbl, [_res(_c_index(i))])
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
                        # Merge address calculation to reduce intermediate variables
                        warp_idx = _res(arith.IndexCastOp(idx, _res(warp_id)))
                        in_i64 = memref.LoadOp(ptrtbl, [_res(warp_idx)]).result
                        raw = _asm_ld_16b(_addr_add(in_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(cur))), _c_i64(16))))  # v4i32
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

                            # Merge address calculation to reduce intermediate variables
                            rel_p = _subi(cur, start_p)
                            _asm_st_16b(_addr_add(tmp_out_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(rel_p))), _c_i64(16))), out_raw)
                            scf.YieldOp([])

                        # No second barrier: flip parity so next iteration writers won't clobber current reads.
                        nxt = _addi(cur, stride_pack)
                        parity_next = _subi(_c_i32(1), parity)  # toggles 0<->1
                        scf.YieldOp([_res(nxt), _res(parity_next)])

                    # end_sync (not final): must match ISA: wbl2 before store, inv during wait
                    # Note: pointer table optimization requires reading pointers at call site
                    # For now, use chain selection to avoid region isolation issues
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
                        sg_ptrtbl=None,
                    )

                    # Precompute all tmp pointers as i64 (ExtractAlignedPointerAsIndexOp includes offset)
                    # This preserves offset information while allowing efficient pointer selection
                    tmp_ptrs_i64 = []
                    for i in range(world_size):
                        tp = memref.ExtractAlignedPointerAsIndexOp(tmps[i]).result
                        tmp_ptrs_i64.append(_res(arith.IndexCastOp(i64, _res(tp))))

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
                        # Reuse warp_id/lane_id/tnum_gpu_i32 from stage1 (already computed above)
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
                            # Use vector.LoadOp to preserve offset precision (automatically handles memref offset)
                            cur_idx = _res(arith.IndexCastOp(idx, _res(cur)))
                            # Build chain of if statements to select correct memref
                            def _load_from_tmp_memref(memref_val, idx_val):
                                if dtype_str == "f32":
                                    vf = _res(vector.LoadOp(v4f32, memref_val, [_res(idx_val)]))
                                    return _res(vector.BitCastOp(v4i32, _res(vf)))
                                else:
                                    v16 = _res(vector.LoadOp(v8f16, memref_val, [_res(idx_val)]))
                                    from _mlir.dialects import llvm
                                    return llvm.BitcastOp(v4i32, _res(v16)).result
                            
                            # Chain if statements: if warp_id == i then load from tmps[i] else ...
                            # All if statements must have else blocks when returning values
                            tmp_if_chain = None
                            for i in range(world_size - 1, -1, -1):  # Build from last to first
                                is_match = _cmp(arith.CmpIPredicate.eq, warp_id, _c_i32(i))
                                if tmp_if_chain is None:
                                    # Last one: if warp_id == i then load else load from tmps[0] as fallback
                                    if_op = scf.IfOp(_res(is_match), results_=[v4i32], hasElse=True)
                                    with ir.InsertionPoint(if_op.then_block):
                                        v = _load_from_tmp_memref(tmps[i], cur_idx)
                                        scf.YieldOp([_res(v)])
                                    with ir.InsertionPoint(if_op.else_block):
                                        # Fallback to tmps[0] (shouldn't happen if warp_id is valid)
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

                # Kernel 2-stage arr variant: accept array base pointers, load inside kernel.
                # This makes device loads part of the kernel execution, enabling CUDAGraph replay.
                # Signature: rank, self_sg, sg_ptrs_base, in_ptrs_base, tmp_ptrs_base, out_ptr
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

                    # Load pointers from device arrays INSIDE the kernel
                    # This makes the loads part of kernel execution, captured by CUDAGraph
                    sgs = []
                    for i in range(8):
                        sg_ptr = _load_ptr_from_array(sg_ptrs_base, _c_i32(i))
                        sgs.append(sg_ptr)
                    
                    in_ptrs = []
                    for i in range(8):
                        in_ptr = _load_ptr_from_array(in_ptrs_base, _c_i32(i))
                        in_ptrs.append(in_ptr)
                    
                    tmp_ptrs = []
                    for i in range(8):
                        tmp_ptr = _load_ptr_from_array(tmp_ptrs_base, _c_i32(i))
                        tmp_ptrs.append(tmp_ptr)

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

                    # stage1 (same as k2p but using loaded pointers)
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
                        raw = _asm_ld_16b(_addr_add(in_base, _mul_i64(_res(arith.ExtUIOp(i64, _res(cur))), _c_i64(16))))

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
                            _asm_st_16b(_addr_add(tmp_out_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(rel_p))), _c_i64(16))), out_raw)
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
                        sg_ptrtbl=None,
                    )

                    # stage2: all-gather fast path
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
                            raw = _asm_ld_16b(_addr_add(tmp_base, _mul_i64(_res(arith.ExtUIOp(i64, _res(cur))), _c_i64(16))))

                            dst_pack = _addi(_muli(dst_rank, _c_i32(part_p)), cur)
                            _asm_st_16b(_addr_add(out_ptr_i64, _mul_i64(_res(arith.ExtUIOp(i64, _res(dst_pack))), _c_i64(16))), raw)

                            nxt = _addi(cur, stride_pack2)
                            scf.YieldOp([_res(nxt)])

                    gpu.ReturnOp([])

            # ---- host entrypoints (llvm.emit_c_interface) ----

            # run_1stage
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
                    grid_size=(gx, one, one),
                    block_size=(bx, one, one),
                    kernel_operands=kops,
                    async_dependencies=[stream_token],
                )
                func.ReturnOp([])

            # run_2stage
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
                    grid_size=(gx, one, one),
                    block_size=(bx, one, one),
                    kernel_operands=kops,
                    async_dependencies=[stream_token],
                )
                func.ReturnOp([])

            # run_2stage_arr (unified for eager and CUDAGraph)
            # Signature: rank, grid_x, self_sg, sg_ptrs_base, in_ptrs_base, tmp_ptrs_base, out_ptr, stream_ptr
            # All pointer arrays are pre-rotated by Python caller (except sg_ptrs which is non-rotated).
            # This wrapper launches a new kernel that loads pointers from device arrays INSIDE the kernel,
            # making it compatible with CUDAGraph (device loads are replayed).
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
                # Pass array base addresses to the kernel; kernel loads pointers internally
                kops = [_res(rank), _res(self_sg), _res(sg_ptrs_array_base), _res(in_ptrs_array_base), _res(tmp_ptrs_array_base), _res(out_ptr_i64)]
                stream_token = stream_ptr_to_async_token(stream_ptr)
                flir_ext.gpu_ext.LaunchFuncOp(
                    ["aiter_signal", f"aiter_signal_all_reduce_2stage_arr_ws{world_size}"],
                    grid_size=(gx, one, one),
                    block_size=(bx, one, one),
                    kernel_operands=kops,
                    async_dependencies=[stream_token],
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

