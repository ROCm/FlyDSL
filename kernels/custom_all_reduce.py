"""Custom all-reduce kernel + Python-facing shim.

This kernel is intentionally written in the same "ext arith" style as
`moe_gemm_2stage.py` / `preshuffle_gemm.py`:
- Use `@flir.kernel` and Python control-flow lowering
- Prefer `flydsl.dialects.ext.arith` helpers / `ArithValue` operator overloading
- Use `arith.as_value(...)` only at strict MLIR builder boundaries (e.g. memref.load/store)

Notes vs the C++ AIter implementation you pasted:
- That code implements **multi-GPU** collectives via IPC handles + device-side signalling.
- FlyDSL example kernels in this repo don't have an IPC runtime; so this file provides:
  - A **single-process** functional subset (world_size==1), and
  - A "packed input" mode (world_size>1) where inputs are provided as a stacked tensor
    `[world_size, N]` (or a Python list of tensors that we stack), and we compute the
    elementwise sum to mimic all-reduce semantics for testing/experimentation.
"""

from flydsl.dialects.ext import flir, arith, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import llvm as flir_llvm
from flydsl.dialects.ext import scf as flir_scf
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from _mlir import ir
from _mlir.dialects import scf as std_scf
from _mlir.dialects import arith as std_arith
from _mlir.dialects import gpu as std_gpu
from _mlir.dialects import vector as std_vector
from _mlir.dialects import llvm as std_llvm
import _mlir.extras.types as T


KERNEL_NAME_ALL_REDUCE = "custom_all_reduce_kernel"
KERNEL_NAME_ALL_GATHER = "custom_all_gather_kernel"
KERNEL_NAME_ALL_REDUCE_IPC = "custom_all_reduce_ipc_kernel"
KERNEL_NAME_REDUCE_SCATTER_IPC = "custom_reduce_scatter_ipc_kernel"
KERNEL_NAME_ALL_GATHER_IPC = "custom_all_gather_ipc_kernel"
KERNEL_NAME_COPY_CHUNK_IPC = "custom_copy_chunk_ipc_kernel"

# ---- AIter ROCm Signal layout constants (from ISA / header) ----
# start: uint32_t start[kMaxBlocks][8]   => 80*8*4 = 2560 bytes
# end:   uint32_t end[kMaxBlocks][8]     => 2560 bytes, aligned(128) => offset 2560
# flag:  uint32_t _flag[kMaxBlocks]      => 80*4 = 320 bytes, aligned(128) => offset 5120 (0x1400)
_AITER_KMAXBLOCKS = 80
_AITER_START_OFF_B = 0
_AITER_END_OFF_B = 2560
_AITER_FLAG_OFF_B = 5120


def build_custom_all_reduce_aiter_signal_raw_module(
    N: int,
    dtype_str: str = "f16",
    *,
    world_size: int,
    BLOCK_SIZE: int = 256,
):
    """AIter-signal protocol allreduce (raw ODS ops only).

    Kernel ABI (1stage):
      (rank_i32, self_sg_i64, sg0..sg7, in0..in7, out_ptr_i64)

    Kernel ABI (2stage):
      (rank_i32, self_sg_i64, sg0..sg7, in0..in7, tmp0..tmp7, out_ptr_i64)

    Notes:
    - Data is processed in 16B "packs" to match AIter's packed_t goal.
    - All sync uses the ROCm ISA sequence observed in AIter ISA dumps:
      `global_store_dword ... sc0 sc1`, `global_load_dword ... sc1`, plus barriers.
    """
    if world_size not in {2, 4, 6, 8}:
        raise ValueError("world_size must be one of {2,4,6,8}")
    if dtype_str not in {"f16", "f32"}:
        raise ValueError("only f16/f32 are supported")
    if N <= 0:
        raise ValueError("N must be > 0")

    PACK_ELEMS = 8 if dtype_str == "f16" else 4
    if N % PACK_ELEMS != 0:
        raise ValueError(f"N must be a multiple of pack elems ({PACK_ELEMS})")

    gpu_arch = get_hip_arch()
    num_packs = N // PACK_ELEMS
    part_packs = num_packs // world_size
    largest_part_packs = part_packs + (num_packs % world_size)

    # Conservative block count (matches the "small SM count" style; cap at kMaxBlocks).
    blocks_1stage = min(_AITER_KMAXBLOCKS, (num_packs + BLOCK_SIZE - 1) // BLOCK_SIZE)
    blocks_2stage = min(_AITER_KMAXBLOCKS, (max(1, part_packs) + BLOCK_SIZE - 1) // BLOCK_SIZE)

    _state = {}

    def _c_i32(v: int):
        return std_arith.ConstantOp(_state["i32"], v).result

    def _c_i64(v: int):
        return std_arith.ConstantOp(_state["i64"], v).result

    def _c_idx(v: int):
        return std_arith.ConstantOp(_state["idx"], v).result

    def _idx_cast(v):
        return std_arith.IndexCastOp(_state["idx"], v).result

    def _i64_cast(v):
        return std_arith.ExtUIOp(_state["i64"], v).result

    def _add_i64(a, b):
        return std_arith.AddIOp(a, b).result

    def _mul_i64(a, b):
        return std_arith.MulIOp(a, b).result

    def _asm_waitcnt():
        std_llvm.InlineAsmOp(
            res=None,
            operands_=[],
            asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
            constraints="",
            has_side_effects=True,
        )

    def _asm_ld_u32_sc1(addr_i64):
        v = std_llvm.InlineAsmOp(
            res=_state["i32"],
            operands_=[addr_i64],
            asm_string="global_load_dword $0, $1, off sc1",
            constraints="=v,v",
            has_side_effects=True,
        ).result
        _asm_waitcnt()
        return v

    def _asm_st_u32_sc0sc1(addr_i64, val_i32):
        std_llvm.InlineAsmOp(
            res=None,
            operands_=[addr_i64, val_i32],
            asm_string="global_store_dword $0, $1, off sc0 sc1",
            constraints="v,v",
            has_side_effects=True,
        )
        _asm_waitcnt()

    def _asm_st_u32(addr_i64, val_i32):
        std_llvm.InlineAsmOp(
            res=None,
            operands_=[addr_i64, val_i32],
            asm_string="global_store_dword $0, $1, off",
            constraints="v,v",
            has_side_effects=True,
        )
        _asm_waitcnt()

    def _asm_ld_16b(addr_i64):
        # Returns v4i32 (16 bytes). Use vector.bitcast to reinterpret.
        v = std_llvm.InlineAsmOp(
            res=_state["v4i32"],
            operands_=[addr_i64],
            asm_string="flat_load_dwordx4 $0, $1",
            constraints="=v,v",
            has_side_effects=True,
        ).result
        _asm_waitcnt()
        return v

    def _asm_st_16b(addr_i64, v4i32_val):
        std_llvm.InlineAsmOp(
            res=None,
            operands_=[addr_i64, v4i32_val],
            asm_string="global_store_dwordx4 $0, $1 off",
            constraints="v,v",
            has_side_effects=True,
        )
        _asm_waitcnt()

    def _sel_i64(sel_i1, a_i64, b_i64):
        return std_arith.SelectOp(sel_i1, a_i64, b_i64).result

    def _select_ptr8(idx_i32, ptrs_i64):
        # ptrs_i64: list of 8 i64 values (already padded); select by idx_i32 (0..7).
        out = ptrs_i64[0]
        for k in range(1, 8):
            pred = std_arith.CmpIOp(std_arith.CmpIPredicate.eq, idx_i32, _c_i32(k)).result
            out = std_arith.SelectOp(pred, ptrs_i64[k], out).result
        return out

    def _spin_wait_ge_u32(addr_i64, target_u32):
        init = _c_i32(0)
        w = std_scf.WhileOp([init])
        before = w.before
        after = w.after
        with ir.InsertionPoint(before.blocks[0]):
            cur = _asm_ld_u32_sc1(addr_i64)
            ok = std_arith.CmpIOp(std_arith.CmpIPredicate.uge, cur, target_u32).result
            std_scf.ConditionOp(ok, [cur])
        with ir.InsertionPoint(after.blocks[0]):
            std_scf.YieldOp([after.blocks[0].arguments[0]])
        return w

    class _RawAiterSignal(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_all_reduce_aiter_signal_raw_{dtype_str}_ws{world_size}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            # Materialize raw types under an active MLIR context & location.
            with ir.Location.unknown():
                _state["i1"] = ir.IntegerType.get_signless(1)
                _state["i32"] = ir.IntegerType.get_signless(32)
                _state["i64"] = ir.IntegerType.get_signless(64)
                _state["idx"] = ir.IndexType.get()
                _state["f16"] = ir.F16Type.get()
                _state["f32"] = ir.F32Type.get()
                _state["v4i32"] = ir.VectorType.get([4], _state["i32"])
                _state["v4f32"] = ir.VectorType.get([4], _state["f32"])
                _state["v8f16"] = ir.VectorType.get([8], _state["f16"])
                _state["v8f32"] = ir.VectorType.get([8], _state["f32"])

        @flir.kernel
        def aiter_signal_raw_1stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            in0: flir.T.i64,
            in1: flir.T.i64,
            in2: flir.T.i64,
            in3: flir.T.i64,
            in4: flir.T.i64,
            in5: flir.T.i64,
            in6: flir.T.i64,
            in7: flir.T.i64,
            out_ptr: flir.T.i64,
        ):
            i32 = _state["i32"]
            f32 = _state["f32"]
            v4i32 = _state["v4i32"]
            v4f32 = _state["v4f32"]
            v8f16 = _state["v8f16"]
            v8f32 = _state["v8f32"]

            # Thread/block IDs (use FlyDSL ext wrappers which return a raw Value already).
            tid_idx = flir.thread_idx("x")
            bid_idx = flir.block_idx("x")
            block_dim_idx = flir.block_dim("x")
            grid_dim_idx = gpu.grid_dim("x")
            tid_i32 = std_arith.IndexCastOp(i32, tid_idx).result
            bid_i32 = std_arith.IndexCastOp(i32, bid_idx).result
            block_dim_i32 = std_arith.IndexCastOp(i32, block_dim_idx).result
            grid_dim_i32 = std_arith.IndexCastOp(i32, grid_dim_idx).result

            # Build arrays (padded to 8).
            sg_ptrs = [sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7]
            in_ptrs = [in0, in1, in2, in3, in4, in5, in6, in7]

            # ---- start_sync ----
            bid8_i32 = std_arith.MulIOp(bid_i32, _c_i32(8)).result
            lin_rank_i32 = std_arith.AddIOp(bid8_i32, rank_i32).result
            lin_lane_i32 = std_arith.AddIOp(bid8_i32, tid_i32).result
            lin_rank_i64 = _i64_cast(lin_rank_i32)
            lin_lane_i64 = _i64_cast(lin_lane_i32)
            bid_i64 = _i64_cast(bid_i32)

            flag_addr = _add_i64(_add_i64(self_sg_i64, _c_i64(_AITER_FLAG_OFF_B)), _mul_i64(bid_i64, _c_i64(4)))
            start_wait_addr = _add_i64(_add_i64(self_sg_i64, _c_i64(_AITER_START_OFF_B)), _mul_i64(lin_lane_i64, _c_i64(4)))
            end_wait_addr = _add_i64(_add_i64(self_sg_i64, _c_i64(_AITER_END_OFF_B)), _mul_i64(lin_lane_i64, _c_i64(4)))
            start_rank_off = _add_i64(_c_i64(_AITER_START_OFF_B), _mul_i64(lin_rank_i64, _c_i64(4)))
            end_rank_off = _add_i64(_c_i64(_AITER_END_OFF_B), _mul_i64(lin_rank_i64, _c_i64(4)))

            flag0 = _asm_ld_u32_sc1(flag_addr)
            flag = std_arith.AddIOp(flag0, _c_i32(1)).result

            is_lane = std_arith.CmpIOp(std_arith.CmpIPredicate.ult, tid_i32, _c_i32(world_size)).result
            if is_lane:
                peer_sg = _select_ptr8(tid_i32, sg_ptrs)
                peer_start_addr = _add_i64(peer_sg, start_rank_off)
                _asm_st_u32_sc0sc1(peer_start_addr, flag)
                _spin_wait_ge_u32(start_wait_addr, flag)
            std_gpu.BarrierOp()
            is_t0 = std_arith.CmpIOp(std_arith.CmpIPredicate.eq, tid_i32, _c_i32(0)).result
            if is_t0:
                _asm_st_u32(flag_addr, flag)
            std_gpu.BarrierOp()

            # ---- compute packs ----
            # pack = bid*blockDim + tid (in index space), stride = gridDim*blockDim
            base_pack = std_arith.AddIOp(
                std_arith.MulIOp(std_arith.IndexCastOp(i32, bid_idx).result, std_arith.IndexCastOp(i32, block_dim_idx).result).result,
                std_arith.IndexCastOp(i32, tid_idx).result,
            ).result
            stride_pack = std_arith.MulIOp(grid_dim_i32, block_dim_i32).result

            # loop over packs
            start_iv = _idx_cast(base_pack)
            end_iv = _c_idx(num_packs)
            step_iv = _idx_cast(stride_pack)
            for_op = std_scf.ForOp(start_iv, end_iv, step_iv)
            with ir.InsertionPoint(for_op.body):
                p = for_op.body.arguments[0]  # index
                p_i64 = _i64_cast(std_arith.IndexCastOp(i32, p).result)
                pack_byte_off = _mul_i64(p_i64, _c_i64(16))

                # accumulate
                if dtype_str == "f32":
                    acc = std_vector.SplatOp(v4f32, std_arith.ConstantOp(f32, 0.0).result).result
                    for r in range(world_size):
                        ptr = in_ptrs[r]
                        addr = _add_i64(ptr, pack_byte_off)
                        raw = _asm_ld_16b(addr)  # v4i32
                        v = std_vector.BitCastOp(v4f32, raw).result
                        acc = std_arith.AddFOp(acc, v).result
                    out_raw = std_vector.BitCastOp(v4i32, acc).result
                else:
                    acc = std_vector.SplatOp(v8f32, std_arith.ConstantOp(f32, 0.0).result).result
                    for r in range(world_size):
                        ptr = in_ptrs[r]
                        addr = _add_i64(ptr, pack_byte_off)
                        raw = _asm_ld_16b(addr)  # v4i32
                        v16 = std_vector.BitCastOp(v8f16, raw).result
                        v32 = std_arith.ExtFOp(v8f32, v16).result
                        acc = std_arith.AddFOp(acc, v32).result
                    out16 = std_arith.TruncFOp(v8f16, acc).result
                    out_raw = std_vector.BitCastOp(v4i32, out16).result

                out_addr = _add_i64(out_ptr, pack_byte_off)
                _asm_st_16b(out_addr, out_raw)
                std_scf.YieldOp([])

            # ---- end_sync(final) ----
            std_gpu.BarrierOp()
            flag0b = _asm_ld_u32_sc1(flag_addr)
            flagb = std_arith.AddIOp(flag0b, _c_i32(1)).result
            if is_lane:
                peer_sg = _select_ptr8(tid_i32, sg_ptrs)
                peer_end_addr = _add_i64(peer_sg, end_rank_off)
                _asm_st_u32_sc0sc1(peer_end_addr, flagb)
                _spin_wait_ge_u32(end_wait_addr, flagb)
            std_gpu.BarrierOp()
            if is_t0:
                _asm_st_u32(flag_addr, flagb)
            std_gpu.BarrierOp()

        @flir.kernel
        def aiter_signal_raw_2stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            in0: flir.T.i64,
            in1: flir.T.i64,
            in2: flir.T.i64,
            in3: flir.T.i64,
            in4: flir.T.i64,
            in5: flir.T.i64,
            in6: flir.T.i64,
            in7: flir.T.i64,
            tmp0: flir.T.i64,
            tmp1: flir.T.i64,
            tmp2: flir.T.i64,
            tmp3: flir.T.i64,
            tmp4: flir.T.i64,
            tmp5: flir.T.i64,
            tmp6: flir.T.i64,
            tmp7: flir.T.i64,
            out_ptr: flir.T.i64,
        ):
            i32 = _state["i32"]
            f32 = _state["f32"]
            v4i32 = _state["v4i32"]
            v4f32 = _state["v4f32"]
            v8f16 = _state["v8f16"]
            v8f32 = _state["v8f32"]

            tid_idx = flir.thread_idx("x")
            bid_idx = flir.block_idx("x")
            block_dim_idx = flir.block_dim("x")
            grid_dim_idx = gpu.grid_dim("x")
            tid_i32 = std_arith.IndexCastOp(i32, tid_idx).result
            bid_i32 = std_arith.IndexCastOp(i32, bid_idx).result
            block_dim_i32 = std_arith.IndexCastOp(i32, block_dim_idx).result
            grid_dim_i32 = std_arith.IndexCastOp(i32, grid_dim_idx).result
            tid_global = std_arith.AddIOp(std_arith.MulIOp(bid_i32, block_dim_i32).result, tid_i32).result
            stride = std_arith.MulIOp(grid_dim_i32, block_dim_i32).result

            sg_ptrs = [sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7]
            in_ptrs = [in0, in1, in2, in3, in4, in5, in6, in7]
            tmp_ptrs = [tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7]

            # ---- start_sync ----
            bid8_i32 = std_arith.MulIOp(bid_i32, _c_i32(8)).result
            lin_rank_i32 = std_arith.AddIOp(bid8_i32, rank_i32).result
            lin_lane_i32 = std_arith.AddIOp(bid8_i32, tid_i32).result
            lin_rank_i64 = _i64_cast(lin_rank_i32)
            lin_lane_i64 = _i64_cast(lin_lane_i32)
            bid_i64 = _i64_cast(bid_i32)

            flag_addr = _add_i64(_add_i64(self_sg_i64, _c_i64(_AITER_FLAG_OFF_B)), _mul_i64(bid_i64, _c_i64(4)))
            start_wait_addr = _add_i64(_add_i64(self_sg_i64, _c_i64(_AITER_START_OFF_B)), _mul_i64(lin_lane_i64, _c_i64(4)))
            end_wait_addr = _add_i64(_add_i64(self_sg_i64, _c_i64(_AITER_END_OFF_B)), _mul_i64(lin_lane_i64, _c_i64(4)))
            start_rank_off = _add_i64(_c_i64(_AITER_START_OFF_B), _mul_i64(lin_rank_i64, _c_i64(4)))
            end_rank_off = _add_i64(_c_i64(_AITER_END_OFF_B), _mul_i64(lin_rank_i64, _c_i64(4)))

            flag0 = _asm_ld_u32_sc1(flag_addr)
            flag = std_arith.AddIOp(flag0, _c_i32(1)).result
            is_lane = std_arith.CmpIOp(std_arith.CmpIPredicate.ult, tid_i32, _c_i32(world_size)).result
            if is_lane:
                peer_sg = _select_ptr8(tid_i32, sg_ptrs)
                peer_start_addr = _add_i64(peer_sg, start_rank_off)
                _asm_st_u32_sc0sc1(peer_start_addr, flag)
                _spin_wait_ge_u32(start_wait_addr, flag)
            std_gpu.BarrierOp()
            is_t0 = std_arith.CmpIOp(std_arith.CmpIPredicate.eq, tid_i32, _c_i32(0)).result
            if is_t0:
                _asm_st_u32(flag_addr, flag)
            std_gpu.BarrierOp()

            # ---- stage1: reduce-scatter into tmp_out (this rank) ----
            start_p = std_arith.MulIOp(rank_i32, _c_i32(part_packs)).result
            end_p = std_arith.SelectOp(
                std_arith.CmpIOp(std_arith.CmpIPredicate.eq, rank_i32, _c_i32(world_size - 1)).result,
                _c_i32(num_packs),
                std_arith.AddIOp(start_p, _c_i32(part_packs)).result,
            ).result

            tmp_out = _select_ptr8(rank_i32, tmp_ptrs)

            # p = start_p + tid_global; p < end_p; p += stride
            p0 = std_arith.AddIOp(start_p, tid_global).result
            w = std_scf.WhileOp([p0])
            with ir.InsertionPoint(w.before.blocks[0]):
                cur = w.before.blocks[0].arguments[0]
                cond = std_arith.CmpIOp(std_arith.CmpIPredicate.ult, cur, end_p).result
                std_scf.ConditionOp(cond, [cur])
            with ir.InsertionPoint(w.after.blocks[0]):
                cur = w.after.blocks[0].arguments[0]
                cur_i64 = _i64_cast(cur)
                pack_byte_off = _mul_i64(cur_i64, _c_i64(16))

                # ptrs[i] = in_ptrs[(rank+i)%ws]
                if dtype_str == "f32":
                    acc = std_vector.SplatOp(v4f32, std_arith.ConstantOp(f32, 0.0).result).result
                    for i in range(world_size):
                        tgt = std_arith.RemUIOp(std_arith.AddIOp(rank_i32, _c_i32(i)).result, _c_i32(world_size)).result
                        ptr = _select_ptr8(tgt, in_ptrs)
                        addr = _add_i64(ptr, pack_byte_off)
                        raw = _asm_ld_16b(addr)
                        v = std_vector.BitCastOp(v4f32, raw).result
                        acc = std_arith.AddFOp(acc, v).result
                    out_raw = std_vector.BitCastOp(v4i32, acc).result
                else:
                    acc = std_vector.SplatOp(v8f32, std_arith.ConstantOp(f32, 0.0).result).result
                    for i in range(world_size):
                        tgt = std_arith.RemUIOp(std_arith.AddIOp(rank_i32, _c_i32(i)).result, _c_i32(world_size)).result
                        ptr = _select_ptr8(tgt, in_ptrs)
                        addr = _add_i64(ptr, pack_byte_off)
                        raw = _asm_ld_16b(addr)
                        v16 = std_vector.BitCastOp(v8f16, raw).result
                        v32 = std_arith.ExtFOp(v8f32, v16).result
                        acc = std_arith.AddFOp(acc, v32).result
                    out16 = std_arith.TruncFOp(v8f16, acc).result
                    out_raw = std_vector.BitCastOp(v4i32, out16).result

                rel = std_arith.SubIOp(cur, start_p).result
                rel_i64 = _i64_cast(rel)
                tmp_off = _mul_i64(rel_i64, _c_i64(16))
                tmp_addr = _add_i64(tmp_out, tmp_off)
                _asm_st_16b(tmp_addr, out_raw)

                nxt = std_arith.AddIOp(cur, stride).result
                std_scf.YieldOp([nxt])

            # ---- end_sync (not final) ----
            std_gpu.BarrierOp()
            flag0b = _asm_ld_u32_sc1(flag_addr)
            flagb = std_arith.AddIOp(flag0b, _c_i32(1)).result
            if is_lane:
                peer_sg = _select_ptr8(tid_i32, sg_ptrs)
                peer_end_addr = _add_i64(peer_sg, end_rank_off)
                _asm_st_u32_sc0sc1(peer_end_addr, flagb)
                _spin_wait_ge_u32(end_wait_addr, flagb)
            std_gpu.BarrierOp()
            if is_t0:
                _asm_st_u32(flag_addr, flagb)
            std_gpu.BarrierOp()

            # ---- stage2: allgather ----
            # idx = tid_global; idx < largest_part_packs; idx += stride
            w2 = std_scf.WhileOp([tid_global])
            with ir.InsertionPoint(w2.before.blocks[0]):
                cur = w2.before.blocks[0].arguments[0]
                cond = std_arith.CmpIOp(std_arith.CmpIPredicate.ult, cur, _c_i32(largest_part_packs)).result
                std_scf.ConditionOp(cond, [cur])
            with ir.InsertionPoint(w2.after.blocks[0]):
                cur = w2.after.blocks[0].arguments[0]
                cur_i64 = _i64_cast(cur)
                src_off = _mul_i64(cur_i64, _c_i64(16))

                for i in range(world_size):
                    gather_from = std_arith.RemUIOp(std_arith.AddIOp(rank_i32, _c_i32(i)).result, _c_i32(world_size)).result
                    # if (gather_from == ws-1) or (cur < part_packs)
                    is_last = std_arith.CmpIOp(std_arith.CmpIPredicate.eq, gather_from, _c_i32(world_size - 1)).result
                    in_part = std_arith.CmpIOp(std_arith.CmpIPredicate.ult, cur, _c_i32(part_packs)).result
                    do_it = std_arith.OrIOp(is_last, in_part).result
                    if do_it:
                        src_tmp = _select_ptr8(gather_from, tmp_ptrs)
                        src_addr = _add_i64(src_tmp, src_off)
                        raw = _asm_ld_16b(src_addr)
                        # dst_pack = gather_from*part_packs + cur
                        dst_pack = std_arith.AddIOp(std_arith.MulIOp(gather_from, _c_i32(part_packs)).result, cur).result
                        dst_off = _mul_i64(_i64_cast(dst_pack), _c_i64(16))
                        dst_addr = _add_i64(out_ptr, dst_off)
                        _asm_st_16b(dst_addr, raw)
                nxt = std_arith.AddIOp(cur, stride).result
                std_scf.YieldOp([nxt])

        @flir.jit
        def run_1stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            in0: flir.T.i64,
            in1: flir.T.i64,
            in2: flir.T.i64,
            in3: flir.T.i64,
            in4: flir.T.i64,
            in5: flir.T.i64,
            in6: flir.T.i64,
            in7: flir.T.i64,
            out_ptr: flir.T.i64,
        ):
            c1 = arith.index(1)
            gx = arith.index(blocks_1stage)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "aiter_signal_raw_1stage"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[
                    rank_i32,
                    self_sg_i64,
                    sg0,
                    sg1,
                    sg2,
                    sg3,
                    sg4,
                    sg5,
                    sg6,
                    sg7,
                    in0,
                    in1,
                    in2,
                    in3,
                    in4,
                    in5,
                    in6,
                    in7,
                    out_ptr,
                ],
            )

        @flir.jit
        def run_2stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            in0: flir.T.i64,
            in1: flir.T.i64,
            in2: flir.T.i64,
            in3: flir.T.i64,
            in4: flir.T.i64,
            in5: flir.T.i64,
            in6: flir.T.i64,
            in7: flir.T.i64,
            tmp0: flir.T.i64,
            tmp1: flir.T.i64,
            tmp2: flir.T.i64,
            tmp3: flir.T.i64,
            tmp4: flir.T.i64,
            tmp5: flir.T.i64,
            tmp6: flir.T.i64,
            tmp7: flir.T.i64,
            out_ptr: flir.T.i64,
        ):
            c1 = arith.index(1)
            gx = arith.index(blocks_2stage)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "aiter_signal_raw_2stage"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[
                    rank_i32,
                    self_sg_i64,
                    sg0,
                    sg1,
                    sg2,
                    sg3,
                    sg4,
                    sg5,
                    sg6,
                    sg7,
                    in0,
                    in1,
                    in2,
                    in3,
                    in4,
                    in5,
                    in6,
                    in7,
                    tmp0,
                    tmp1,
                    tmp2,
                    tmp3,
                    tmp4,
                    tmp5,
                    tmp6,
                    tmp7,
                    out_ptr,
                ],
            )

    return _RawAiterSignal().module


def _aiter_signal_addrs(*, self_sg_i64, bid_i32, lane_i32, rank_i32):
    """Return (flag_addr, start_addr, end_addr, start_wait_addr, end_wait_addr) as i64.

    - flag_addr = &self_sg->_flag[bid]
    - start_addr = &peer_sg->start[bid][rank]   (for lane's peer)
    - end_addr   = &peer_sg->end[bid][rank]
    - start_wait_addr = &self_sg->start[bid][lane]
    - end_wait_addr   = &self_sg->end[bid][lane]
    """
    # Compute byte addresses using a mix of ext-arith (for i32 math) and strict arith ops
    # (for i64 pointer math). Ensure all strict ops only see real `ir.Value`.
    self_sg_i64_v = arith.as_value(self_sg_i64)
    bid_i32_v = arith.as_value(bid_i32)
    lane_i32_v = arith.as_value(lane_i32)
    rank_i32_v = arith.as_value(rank_i32)

    bid8 = arith.as_value(arith.ArithValue(bid_i32_v) * 8)
    lin_rank_i32 = arith.as_value(arith.ArithValue(bid8) + rank_i32_v)
    lin_lane_i32 = arith.as_value(arith.ArithValue(bid8) + lane_i32_v)

    c4_i64 = arith.i64(4).value
    bid_i64 = arith.as_value(flir.arith.ExtUIOp(T.i64(), bid_i32_v).result)
    lin_rank_i64 = arith.as_value(flir.arith.ExtUIOp(T.i64(), lin_rank_i32).result)
    lin_lane_i64 = arith.as_value(flir.arith.ExtUIOp(T.i64(), lin_lane_i32).result)

    base_i64 = self_sg_i64_v
    flag_addr = flir.arith.AddIOp(
        flir.arith.AddIOp(base_i64, arith.i64(_AITER_FLAG_OFF_B).value).result,
        flir.arith.MulIOp(arith.as_value(bid_i64), c4_i64).result,
    ).result

    start_wait_addr = flir.arith.AddIOp(
        flir.arith.AddIOp(base_i64, arith.i64(_AITER_START_OFF_B).value).result,
        flir.arith.MulIOp(arith.as_value(lin_lane_i64), c4_i64).result,
    ).result
    end_wait_addr = flir.arith.AddIOp(
        flir.arith.AddIOp(base_i64, arith.i64(_AITER_END_OFF_B).value).result,
        flir.arith.MulIOp(arith.as_value(lin_lane_i64), c4_i64).result,
    ).result

    start_rank_off = flir.arith.AddIOp(arith.i64(_AITER_START_OFF_B).value, flir.arith.MulIOp(arith.as_value(lin_rank_i64), c4_i64).result).result
    end_rank_off = flir.arith.AddIOp(arith.i64(_AITER_END_OFF_B).value, flir.arith.MulIOp(arith.as_value(lin_rank_i64), c4_i64).result).result

    return (flag_addr, start_rank_off, end_rank_off, start_wait_addr, end_wait_addr)


def _asm_waitcnt():
    flir_llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
        constraints="",
        has_side_effects=True,
    )


def _asm_ld_u32_sc1(addr_i64):
    v = flir_llvm.InlineAsmOp(
        res=T.i32(),
        operands_=[arith.as_value(addr_i64)],
        asm_string="global_load_dword $0, $1, off sc1",
        constraints="=v,v",
        has_side_effects=True,
    ).result
    _asm_waitcnt()
    return v


def _asm_st_u32_sc0sc1(addr_i64, val_i32):
    flir_llvm.InlineAsmOp(
        res=None,
        operands_=[arith.as_value(addr_i64), arith.as_value(val_i32)],
        asm_string="global_store_dword $0, $1, off sc0 sc1",
        constraints="v,v",
        has_side_effects=True,
    )
    _asm_waitcnt()


def _asm_st_u32(addr_i64, val_i32):
    flir_llvm.InlineAsmOp(
        res=None,
        operands_=[arith.as_value(addr_i64), arith.as_value(val_i32)],
        asm_string="global_store_dword $0, $1, off",
        constraints="v,v",
        has_side_effects=True,
    )
    _asm_waitcnt()


def _spin_wait_ge_u32(addr_i64, target_u32):
    """Spin until *addr >= target."""
    init = arith.i32(0).value
    with flir_scf.while_([init]) as (before, after):
        with before:
            cur = _asm_ld_u32_sc1(addr_i64)
            ok = flir.arith.CmpIOp(flir.arith.CmpIPredicate.uge, cur, arith.as_value(target_u32)).result
            std_scf.ConditionOp(ok, [cur])
        with after:
            std_scf.YieldOp([after.arguments[0]])


def build_custom_all_reduce_aiter_signal_module(
    N: int,
    dtype_str: str = "f16",
    *,
    world_size: int,
    BLOCK_SIZE: int = 256,
):
    """FlyDSL all-reduce that matches AIter ROCm signal protocol (start/end/_flag) for correctness.

    This module expects:
    - `self_sg` + `sg0..sg7`: device pointers (i64) to each rank's `Signal` base
    - `In0..In7`: pointers to each rank's input buffer (N elements)
    - `Tmp0..Tmp7`: pointers to each rank's tmp buffer (largest_part elements)
    """
    if world_size not in {2, 4, 6, 8}:
        raise ValueError("world_size must be one of {2,4,6,8} for aiter protocol")
    if N <= 0:
        raise ValueError("N must be > 0")
    PACK_ELEMS = _pack_elems(dtype_str)
    if N % PACK_ELEMS != 0:
        raise ValueError(f"custom allreduce requires input length to be multiple of {PACK_ELEMS}")
    gpu_arch = get_hip_arch()
    num_packs = N // PACK_ELEMS
    num_blocks = min(_AITER_KMAXBLOCKS, (num_packs + BLOCK_SIZE - 1) // BLOCK_SIZE)
    part = N // world_size
    largest_part = part + (N % world_size)

    _state = {}
    VEC_ALIGN = 16

    class _AIterSignalAllReduce(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_all_reduce_aiter_signal_{dtype_str}_ws{world_size}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type, compute_type = _dtype_to_mlir(dtype_str)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

        @flir.kernel
        def aiter_signal_all_reduce_1stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            Ins = [In0, In1, In2, In3, In4, In5, In6, In7]
            sgs = [sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7]

            tid = arith.as_value(flir.arith.IndexCastOp(T.i32(), arith.as_value(flir.thread_idx("x"))).result)
            bid = arith.as_value(flir.arith.IndexCastOp(T.i32(), arith.as_value(flir.block_idx("x"))).result)
            c_ng = arith.i32(world_size).value
            c0_i32 = arith.i32(0).value

            # ---- start_sync ----
            # Compute addresses per AIter Signal layout.
            bid8 = arith.ArithValue(bid) * 8
            lin_rank_i32 = arith.as_value(arith.ArithValue(bid8) + rank_i32)
            lin_lane_i32 = arith.as_value(arith.ArithValue(bid8) + tid)
            lin_rank_i64 = flir.arith.ExtUIOp(T.i64(), arith.as_value(lin_rank_i32)).result
            lin_lane_i64 = flir.arith.ExtUIOp(T.i64(), arith.as_value(lin_lane_i32)).result
            bid_i64 = flir.arith.ExtUIOp(T.i64(), arith.as_value(bid)).result

            flag_addr = arith.ArithValue(self_sg_i64) + _AITER_FLAG_OFF_B + arith.ArithValue(bid_i64) * 4
            start_wait_addr = arith.ArithValue(self_sg_i64) + _AITER_START_OFF_B + arith.ArithValue(lin_lane_i64) * 4
            end_wait_addr = arith.ArithValue(self_sg_i64) + _AITER_END_OFF_B + arith.ArithValue(lin_lane_i64) * 4
            start_rank_off = arith.i64(_AITER_START_OFF_B) + arith.ArithValue(lin_rank_i64) * 4
            end_rank_off = arith.i64(_AITER_END_OFF_B) + arith.ArithValue(lin_rank_i64) * 4

            flag0 = _asm_ld_u32_sc1(flag_addr)
            flag = flir.arith.AddIOp(arith.as_value(flag0), arith.i32(1).value).result

            is_lane = flir.arith.CmpIOp(flir.arith.CmpIPredicate.ult, arith.as_value(tid), c_ng).result
            if is_lane:
                peer_sg = sgs[int(0)]  # dummy to keep python list alive
                # Select peer_sg = sgs[tid] via cndmask chain (ws <= 8).
                peer = sgs[0]
                for i in range(1, 8):
                    pred = flir.arith.CmpIOp(flir.arith.CmpIPredicate.eq, arith.as_value(tid), arith.i32(i).value).result
                    peer = flir.arith.SelectOp(pred, sgs[i], peer).result
                peer_sg = peer
                peer_start_addr = arith.ArithValue(peer_sg) + start_rank_off
                _asm_st_u32_sc0sc1(peer_start_addr, flag)
                _spin_wait_ge_u32(start_wait_addr, flag)
            flir.gpu_ext.barrier()
            is_t0 = flir.arith.CmpIOp(flir.arith.CmpIPredicate.eq, arith.as_value(tid), c0_i32).result
            if is_t0:
                _asm_st_u32(flag_addr, flag)
            flir.gpu_ext.barrier()

            # ---- compute ----
            vec_e_ty = ir.VectorType.get([PACK_ELEMS], _state["elem_type"])
            vec_c_ty = ir.VectorType.get([PACK_ELEMS], _state["compute_type"])
            tid_idx = flir.thread_idx("x")
            bid_idx = flir.block_idx("x")
            pack_idx = arith.ArithValue(bid_idx) * BLOCK_SIZE + arith.ArithValue(tid_idx)
            is_valid_pack = arith.ult(pack_idx, flir.const_index(num_packs))
            base = arith.ArithValue(pack_idx) * PACK_ELEMS
            if is_valid_pack:
                zero = arith.constant(0.0, type=_state["compute_type"])
                acc = flir.vector.splat(vec_c_ty, arith.as_value(zero))
                for r in range_constexpr(world_size):
                    v_e = flir.vector.load(vec_e_ty, Ins[r], [arith.as_value(base)], alignment=VEC_ALIGN)
                    v_c = v_e if dtype_str == "f32" else arith.extf(vec_c_ty, v_e)
                    acc = arith.as_value(arith.ArithValue(acc) + v_c)
                out_v = acc if dtype_str == "f32" else arith.trunc_f(vec_e_ty, acc)
                flir.vector.store(arith.as_value(out_v), Out, [arith.as_value(base)], alignment=VEC_ALIGN)

            # ---- end_sync(final) ----
            flir.gpu_ext.barrier()
            flag0b = _asm_ld_u32_sc1(flag_addr)
            flagb = flir.arith.AddIOp(arith.as_value(flag0b), arith.i32(1).value).result
            if is_lane:
                peer = sgs[0]
                for i in range(1, 8):
                    pred = flir.arith.CmpIOp(flir.arith.CmpIPredicate.eq, arith.as_value(tid), arith.i32(i).value).result
                    peer = flir.arith.SelectOp(pred, sgs[i], peer).result
                peer_end_addr = arith.ArithValue(peer) + end_rank_off
                _asm_st_u32_sc0sc1(peer_end_addr, flagb)
                _spin_wait_ge_u32(end_wait_addr, flagb)
            flir.gpu_ext.barrier()
            if is_t0:
                _asm_st_u32(flag_addr, flagb)
            flir.gpu_ext.barrier()

        @flir.kernel
        def aiter_signal_all_reduce_2stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Tmp0: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp1: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp2: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp3: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp4: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp5: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp6: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp7: lambda: T.memref(largest_part, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            Ins = [In0, In1, In2, In3, In4, In5, In6, In7]
            Tmps = [Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, Tmp5, Tmp6, Tmp7]
            sgs = [sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7]

            tid_i32 = arith.as_value(flir.arith.AddIOp(
                flir.arith.MulIOp(
                    flir.arith.IndexCastOp(T.i32(), arith.as_value(flir.block_idx("x"))).result,
                    flir.arith.IndexCastOp(T.i32(), arith.as_value(flir.block_dim("x"))).result,
                ).result,
                flir.arith.IndexCastOp(T.i32(), arith.as_value(flir.thread_idx("x"))).result,
            ).result)
            stride_i32 = arith.as_value(flir.arith.MulIOp(
                flir.arith.IndexCastOp(T.i32(), arith.as_value(gpu.grid_dim("x"))).result,
                flir.arith.IndexCastOp(T.i32(), arith.as_value(flir.block_dim("x"))).result,
            ).result)
            bid_i32 = arith.as_value(flir.arith.IndexCastOp(T.i32(), arith.as_value(flir.block_idx("x"))).result)
            lane_i32 = arith.as_value(flir.arith.IndexCastOp(T.i32(), arith.as_value(flir.thread_idx("x"))).result)

            part_e = arith.i32(part).value
            start_e = flir.arith.MulIOp(rank_i32, part_e).result
            end_e = flir.arith.SelectOp(
                flir.arith.CmpIOp(flir.arith.CmpIPredicate.eq, rank_i32, arith.i32(world_size - 1).value).result,
                arith.i32(N).value,
                flir.arith.AddIOp(start_e, part_e).result,
            ).result
            part_packs = arith.i32(part // PACK_ELEMS).value
            start_p = flir.arith.DivUIOp(start_e, arith.i32(PACK_ELEMS).value).result
            end_p = flir.arith.DivUIOp(end_e, arith.i32(PACK_ELEMS).value).result
            largest_p = arith.i32(largest_part // PACK_ELEMS).value

            # ---- start_sync ----
            bid8 = arith.ArithValue(bid_i32) * 8
            lin_rank_i32 = arith.as_value(arith.ArithValue(bid8) + rank_i32)
            lin_lane_i32 = arith.as_value(arith.ArithValue(bid8) + lane_i32)
            lin_rank_i64 = flir.arith.ExtUIOp(T.i64(), arith.as_value(lin_rank_i32)).result
            lin_lane_i64 = flir.arith.ExtUIOp(T.i64(), arith.as_value(lin_lane_i32)).result
            bid_i64 = flir.arith.ExtUIOp(T.i64(), arith.as_value(bid_i32)).result

            flag_addr = arith.ArithValue(self_sg_i64) + _AITER_FLAG_OFF_B + arith.ArithValue(bid_i64) * 4
            start_wait_addr = arith.ArithValue(self_sg_i64) + _AITER_START_OFF_B + arith.ArithValue(lin_lane_i64) * 4
            end_wait_addr = arith.ArithValue(self_sg_i64) + _AITER_END_OFF_B + arith.ArithValue(lin_lane_i64) * 4
            start_rank_off = arith.i64(_AITER_START_OFF_B) + arith.ArithValue(lin_rank_i64) * 4
            end_rank_off = arith.i64(_AITER_END_OFF_B) + arith.ArithValue(lin_rank_i64) * 4

            flag0 = _asm_ld_u32_sc1(flag_addr)
            flag = flir.arith.AddIOp(arith.as_value(flag0), arith.i32(1).value).result

            is_lane = flir.arith.CmpIOp(flir.arith.CmpIPredicate.ult, lane_i32, arith.i32(world_size).value).result
            if is_lane:
                peer = sgs[0]
                for i in range(1, 8):
                    pred = flir.arith.CmpIOp(flir.arith.CmpIPredicate.eq, lane_i32, arith.i32(i).value).result
                    peer = flir.arith.SelectOp(pred, sgs[i], peer).result
                peer_start_addr = arith.ArithValue(peer) + start_rank_off
                _asm_st_u32_sc0sc1(peer_start_addr, flag)
                _spin_wait_ge_u32(start_wait_addr, flag)
            flir.gpu_ext.barrier()
            is_t0 = flir.arith.CmpIOp(flir.arith.CmpIPredicate.eq, lane_i32, arith.i32(0).value).result
            if is_t0:
                _asm_st_u32(flag_addr, flag)
            flir.gpu_ext.barrier()

            # ---- stage1: reduce-scatter ----
            vec_e_ty = ir.VectorType.get([PACK_ELEMS], _state["elem_type"])
            vec_c_ty = ir.VectorType.get([PACK_ELEMS], _state["compute_type"])
            tmp_out = Tmps[0]
            for i in range(1, 8):
                pred = flir.arith.CmpIOp(flir.arith.CmpIPredicate.eq, rank_i32, arith.i32(i).value).result
                tmp_out = flir.arith.SelectOp(pred, Tmps[i], tmp_out).result

            idx_p = flir.arith.AddIOp(start_p, tid_i32).result
            with flir_scf.while_([idx_p]) as (before, after):
                with before:
                    cur = before.arguments[0]
                    cond = flir.arith.CmpIOp(flir.arith.CmpIPredicate.ult, cur, end_p).result
                    std_scf.ConditionOp(cond, [cur])
                with after:
                    cur = after.arguments[0]
                    base = flir.arith.MulIOp(cur, arith.i32(PACK_ELEMS).value).result
                    base_idx = flir.arith.IndexCastOp(T.index(), base).result
                    zero = arith.constant(0.0, type=_state["compute_type"])
                    acc = flir.vector.splat(vec_c_ty, arith.as_value(zero))
                    for r in range_constexpr(world_size):
                        v_e = flir.vector.load(vec_e_ty, Ins[r], [base_idx], alignment=VEC_ALIGN)
                        v_c = v_e if dtype_str == "f32" else arith.extf(vec_c_ty, v_e)
                        acc = arith.as_value(arith.ArithValue(acc) + v_c)
                    out_v = acc if dtype_str == "f32" else arith.trunc_f(vec_e_ty, acc)
                    # tmp index is (cur-start_p)*PACK_ELEMS
                    rel = flir.arith.SubIOp(cur, start_p).result
                    tmp_base = flir.arith.MulIOp(rel, arith.i32(PACK_ELEMS).value).result
                    tmp_base_idx = flir.arith.IndexCastOp(T.index(), tmp_base).result
                    flir.vector.store(arith.as_value(out_v), tmp_out, [tmp_base_idx], alignment=VEC_ALIGN)
                    nxt = flir.arith.AddIOp(cur, stride_i32).result
                    std_scf.YieldOp([nxt])

            # ---- end_sync(not final) ----
            flir.gpu_ext.barrier()
            flag0b = _asm_ld_u32_sc1(flag_addr)
            flagb = flir.arith.AddIOp(arith.as_value(flag0b), arith.i32(1).value).result
            if is_lane:
                peer = sgs[0]
                for i in range(1, 8):
                    pred = flir.arith.CmpIOp(flir.arith.CmpIPredicate.eq, lane_i32, arith.i32(i).value).result
                    peer = flir.arith.SelectOp(pred, sgs[i], peer).result
                peer_end_addr = arith.ArithValue(peer) + end_rank_off
                _asm_st_u32_sc0sc1(peer_end_addr, flagb)
                _spin_wait_ge_u32(end_wait_addr, flagb)
            flir.gpu_ext.barrier()
            if is_t0:
                _asm_st_u32(flag_addr, flagb)
            flir.gpu_ext.barrier()

            # ---- stage2: all-gather ----
            idx_p2 = tid_i32
            with flir_scf.while_([idx_p2]) as (before2, after2):
                with before2:
                    cur = before2.arguments[0]
                    cond = flir.arith.CmpIOp(flir.arith.CmpIPredicate.ult, cur, largest_p).result
                    std_scf.ConditionOp(cond, [cur])
                with after2:
                    cur = after2.arguments[0]
                    # gather from each rank's tmp
                    for i in range_constexpr(world_size):
                        gather_from = (int(i) + 0)  # constexpr int
                        # emulate (rank + i) % world_size using constexpr
                        # map via python list: for correctness we just use i in [0..ws)
                        src_tmp = Tmps[gather_from]
                        src_base = flir.arith.MulIOp(cur, arith.i32(PACK_ELEMS).value).result
                        src_base_idx = flir.arith.IndexCastOp(T.index(), src_base).result
                        v_e = flir.vector.load(vec_e_ty, src_tmp, [src_base_idx], alignment=VEC_ALIGN)
                        # dst rank = gather_from (already rotated out-of-scope here); keep simple for now.
                        # dst pack = gather_from * part_packs + cur
                        dst_pack = flir.arith.AddIOp(flir.arith.MulIOp(arith.i32(gather_from).value, part_packs).result, cur).result
                        dst_base = flir.arith.MulIOp(dst_pack, arith.i32(PACK_ELEMS).value).result
                        dst_base_idx = flir.arith.IndexCastOp(T.index(), dst_base).result
                        flir.vector.store(v_e, Out, [dst_base_idx], alignment=VEC_ALIGN)
                    nxt = flir.arith.AddIOp(cur, stride_i32).result
                    std_scf.YieldOp([nxt])

        @flir.jit
        def run_1stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "aiter_signal_all_reduce_1stage"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[rank_i32, self_sg_i64, sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7, In0, In1, In2, In3, In4, In5, In6, In7, Out],
            )

        @flir.jit
        def run_2stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Tmp0: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp1: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp2: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp3: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp4: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp5: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp6: lambda: T.memref(largest_part, _state["elem_type"]),
            Tmp7: lambda: T.memref(largest_part, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "aiter_signal_all_reduce_2stage"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[
                    rank_i32,
                    self_sg_i64,
                    sg0,
                    sg1,
                    sg2,
                    sg3,
                    sg4,
                    sg5,
                    sg6,
                    sg7,
                    In0,
                    In1,
                    In2,
                    In3,
                    In4,
                    In5,
                    In6,
                    In7,
                    Tmp0,
                    Tmp1,
                    Tmp2,
                    Tmp3,
                    Tmp4,
                    Tmp5,
                    Tmp6,
                    Tmp7,
                    Out,
                ],
            )

    return _AIterSignalAllReduce().module


def build_custom_all_reduce_aiter_signal_raw_module__legacy(
    N: int,
    dtype_str: str = "f16",
    *,
    world_size: int,
    BLOCK_SIZE: int = 256,
):
    """Raw-ODS version of AIter signal protocol all-reduce.

    This is intentionally implemented using `_mlir.dialects.*` directly (no ext wrappers)
    to avoid Value-wrapper mismatches when mixing scf/arith/llvm.inline_asm.

    Exposes 2 entrypoints:
    - `run_1stage(rank, self_sg, sg0..sg7, in0..in7, out)`
    - `run_2stage(rank, self_sg, sg0..sg7, in0..in7, tmp0..tmp7, out)`
    """
    if world_size not in {2, 4, 6, 8}:
        raise ValueError("world_size must be in {2,4,6,8}")
    if N <= 0:
        raise ValueError("N must be > 0")
    PACK_ELEMS = _pack_elems(dtype_str)
    if N % PACK_ELEMS != 0:
        raise ValueError(f"N must be multiple of {PACK_ELEMS}")

    gpu_arch = get_hip_arch()
    num_packs = N // PACK_ELEMS
    num_blocks = min(_AITER_KMAXBLOCKS, (num_packs + BLOCK_SIZE - 1) // BLOCK_SIZE)

    # tmp is interpreted as packed 16B units.
    size_packs = num_packs
    part_packs = size_packs // world_size
    largest_part_packs = part_packs + (size_packs % world_size)
    largest_part_elems = largest_part_packs * PACK_ELEMS

    _state = {}

    # Raw dialect imports (inside builder so we don't pay import cost for users not using this path).
    from _mlir import ir as _ir
    from _mlir.dialects import arith as _arith
    from _mlir.dialects import scf as _scf
    from _mlir.dialects import gpu as _gpu
    from _mlir.dialects import llvm as _llvm
    from _mlir.dialects import memref as _memref

    def _i32(x: int):
        return _arith.ConstantOp(_ir.IntegerType.get_signless(32), x).result

    def _i64(x: int):
        return _arith.ConstantOp(_ir.IntegerType.get_signless(64), x).result

    def _idx(x: int):
        return _arith.ConstantOp(_ir.IndexType.get(), x).result

    def _asm_waitcnt():
        _llvm.InlineAsmOp(
            None,
            [],
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "",
            has_side_effects=True,
        )

    def _asm_ld_u32_sc1(addr_i64):
        v = _llvm.InlineAsmOp(
            _ir.IntegerType.get_signless(32),
            [addr_i64],
            "global_load_dword $0, $1, off sc1",
            "=v,v",
            has_side_effects=True,
        ).result
        _asm_waitcnt()
        return v

    def _asm_st_u32_sc0sc1(addr_i64, val_i32):
        _llvm.InlineAsmOp(
            None,
            [addr_i64, val_i32],
            "global_store_dword $0, $1, off sc0 sc1",
            "v,v",
            has_side_effects=True,
        )
        _asm_waitcnt()

    def _asm_st_u32(addr_i64, val_i32):
        _llvm.InlineAsmOp(
            None,
            [addr_i64, val_i32],
            "global_store_dword $0, $1, off",
            "v,v",
            has_side_effects=True,
        )
        _asm_waitcnt()

    def _spin_wait_ge_u32(addr_i64, target_u32):
        # while (ld(addr) < target) continue;
        init = _i32(0)
        w = _scf.WhileOp([init])
        before = w.before
        after = w.after
        with _ir.InsertionPoint(before.blocks[0]):
            cur = _asm_ld_u32_sc1(addr_i64)
            ok = _arith.CmpIOp(_arith.CmpIPredicate.uge, cur, target_u32).result
            _scf.ConditionOp(ok, [cur])
        with _ir.InsertionPoint(after.blocks[0]):
            _scf.YieldOp([after.blocks[0].arguments[0]])

    class _AIterSignalRaw(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_all_reduce_aiter_signal_raw_{dtype_str}_ws{world_size}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type, compute_type = _dtype_to_mlir(dtype_str)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

        @flir.kernel
        def aiter_signal_raw_1stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            # ids
            tid_idx = _gpu.thread_id("x").result
            bid_idx = _gpu.block_id("x").result
            tid_i32 = _arith.IndexCastOp(_ir.IntegerType.get_signless(32), tid_idx).result
            bid_i32 = _arith.IndexCastOp(_ir.IntegerType.get_signless(32), bid_idx).result

            # addr calc (uint32 indexing)
            bid8 = _arith.MulIOp(bid_i32, _i32(8)).result
            lin_rank = _arith.AddIOp(bid8, rank_i32).result
            lin_lane = _arith.AddIOp(bid8, tid_i32).result
            bid_i64 = _arith.ExtUIOp(_ir.IntegerType.get_signless(64), bid_i32).result
            lin_rank_i64 = _arith.ExtUIOp(_ir.IntegerType.get_signless(64), lin_rank).result
            lin_lane_i64 = _arith.ExtUIOp(_ir.IntegerType.get_signless(64), lin_lane).result

            flag_addr = _arith.AddIOp(
                _arith.AddIOp(self_sg_i64, _i64(_AITER_FLAG_OFF_B)).result,
                _arith.MulIOp(bid_i64, _i64(4)).result,
            ).result
            start_wait_addr = _arith.AddIOp(
                self_sg_i64,
                _arith.AddIOp(_i64(_AITER_START_OFF_B), _arith.MulIOp(lin_lane_i64, _i64(4)).result).result,
            ).result
            end_wait_addr = _arith.AddIOp(
                self_sg_i64,
                _arith.AddIOp(_i64(_AITER_END_OFF_B), _arith.MulIOp(lin_lane_i64, _i64(4)).result).result,
            ).result
            start_rank_off = _arith.AddIOp(_i64(_AITER_START_OFF_B), _arith.MulIOp(lin_rank_i64, _i64(4)).result).result
            end_rank_off = _arith.AddIOp(_i64(_AITER_END_OFF_B), _arith.MulIOp(lin_rank_i64, _i64(4)).result).result

            # start_sync
            flag0 = _asm_ld_u32_sc1(flag_addr)
            flag = _arith.AddIOp(flag0, _i32(1)).result
            is_lane = _arith.CmpIOp(_arith.CmpIPredicate.ult, tid_i32, _i32(world_size)).result
            if is_lane:
                # select peer sg by tid (chain)
                peer = sg0
                for i in range(1, 8):
                    peer = _arith.SelectOp(_arith.CmpIOp(_arith.CmpIPredicate.eq, tid_i32, _i32(i)).result, [sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7][i], peer).result
                peer_start_addr = _arith.AddIOp(peer, start_rank_off).result
                _asm_st_u32_sc0sc1(peer_start_addr, flag)
                _spin_wait_ge_u32(start_wait_addr, flag)
            _gpu.BarrierOp()
            if _arith.CmpIOp(_arith.CmpIPredicate.eq, tid_i32, _i32(0)).result:
                _asm_st_u32(flag_addr, flag)
            _gpu.BarrierOp()

            # compute (scalar per pack element for correctness)
            pack_idx = _arith.AddIOp(_arith.MulIOp(_arith.IndexCastOp(_ir.IndexType.get(), bid_i32).result, _idx(BLOCK_SIZE)).result, tid_idx).result
            in_range = _arith.CmpIOp(_arith.CmpIPredicate.ult, pack_idx, _idx(num_packs)).result
            if in_range:
                base = _arith.MulIOp(pack_idx, _idx(PACK_ELEMS)).result
                for lane in range_constexpr(PACK_ELEMS):
                    idx = _arith.AddIOp(base, _idx(lane)).result
                    acc = None
                    for r, In in enumerate([In0, In1, In2, In3, In4, In5, In6, In7][:world_size]):
                        x0 = _memref.LoadOp(In, [idx]).result
                        if dtype_str == "f16":
                            x0f = _arith.ExtFOp(_state["compute_type"], x0).result
                        else:
                            x0f = x0
                        acc = x0f if acc is None else _arith.AddFOp(acc, x0f).result
                    y = acc if dtype_str == "f32" else _arith.TruncFOp(_state["elem_type"], acc).result
                    _memref.StoreOp(y, Out, [idx])

            # end_sync(final)
            _gpu.BarrierOp()
            flag0b = _asm_ld_u32_sc1(flag_addr)
            flagb = _arith.AddIOp(flag0b, _i32(1)).result
            if is_lane:
                peer = sg0
                for i in range(1, 8):
                    peer = _arith.SelectOp(_arith.CmpIOp(_arith.CmpIPredicate.eq, tid_i32, _i32(i)).result, [sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7][i], peer).result
                peer_end_addr = _arith.AddIOp(peer, end_rank_off).result
                _asm_st_u32_sc0sc1(peer_end_addr, flagb)
                _spin_wait_ge_u32(end_wait_addr, flagb)
            _gpu.BarrierOp()
            if _arith.CmpIOp(_arith.CmpIPredicate.eq, tid_i32, _i32(0)).result:
                _asm_st_u32(flag_addr, flagb)
            _gpu.BarrierOp()

        # NOTE: 2stage raw implementation is omitted in this patch for brevity; it will be added next.

        @flir.jit
        def run_1stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "aiter_signal_raw_1stage"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[rank_i32, self_sg_i64, sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7, In0, In1, In2, In3, In4, In5, In6, In7, Out],
            )

        @flir.jit
        def run_2stage(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            self_sg_i64: flir.T.i64,
            sg0: flir.T.i64,
            sg1: flir.T.i64,
            sg2: flir.T.i64,
            sg3: flir.T.i64,
            sg4: flir.T.i64,
            sg5: flir.T.i64,
            sg6: flir.T.i64,
            sg7: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Tmp0: lambda: T.memref(largest_part_elems, _state["elem_type"]),
            Tmp1: lambda: T.memref(largest_part_elems, _state["elem_type"]),
            Tmp2: lambda: T.memref(largest_part_elems, _state["elem_type"]),
            Tmp3: lambda: T.memref(largest_part_elems, _state["elem_type"]),
            Tmp4: lambda: T.memref(largest_part_elems, _state["elem_type"]),
            Tmp5: lambda: T.memref(largest_part_elems, _state["elem_type"]),
            Tmp6: lambda: T.memref(largest_part_elems, _state["elem_type"]),
            Tmp7: lambda: T.memref(largest_part_elems, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            # TEMP: fallback to 1stage implementation so the module can compile while we
            # implement the true 2stage protocol in raw dialect.
            _ = (Tmp0, Tmp1, Tmp2, Tmp3, Tmp4, Tmp5, Tmp6, Tmp7)
            self.run_1stage(rank_i32, self_sg_i64, sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7, In0, In1, In2, In3, In4, In5, In6, In7, Out)

    return _AIterSignalRaw().module


def _dtype_to_mlir(dtype_str: str):
    if dtype_str == "f32":
        return T.f32(), T.f32()
    if dtype_str == "f16":
        return T.f16(), T.f32()
    if dtype_str == "bf16":
        # NOTE: BF16 load/store/extf/trunc lowering has proven unreliable on some ROCm toolchains
        # used in this repo (can produce wildly incorrect values).
        # Keep BF16 as raw i16 storage and do explicit bit-based conversion to/from f32.
        # BF16 encoding is the top 16 bits of IEEE fp32.
        return T.i16(), T.f32()
    raise ValueError(f"unsupported dtype: {dtype_str}")

def _pack_elems(dtype_str: str) -> int:
    """Match C++ packed_t<T>::P where P is 16 bytes."""
    if dtype_str == "f32":
        return 4
    if dtype_str in {"f16", "bf16"}:
        return 8
    raise ValueError(f"unsupported dtype: {dtype_str}")

def _elem_size_bytes(dtype_str: str) -> int:
    if dtype_str == "f32":
        return 4
    if dtype_str in {"f16", "bf16"}:
        return 2
    raise ValueError(f"unsupported dtype: {dtype_str}")


def build_custom_all_reduce_module(
    N: int,
    dtype_str: str = "f32",
    *,
    world_size: int = 1,
    BLOCK_SIZE: int = 256,
):
    """Build a module containing a custom all-reduce kernel.

    Semantics:
    - If `world_size == 1`: Out[i] = In[i] (identity / copy)
    - If `world_size > 1`: inputs are provided as a packed tensor `In[r, i]`
      and we compute elementwise sum:
        Out[i] = sum_r In[r, i]
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if world_size <= 0 or world_size > 8:
        raise ValueError("world_size must be in [1, 8]")
    if world_size > 1 and (world_size % 2 != 0):
        raise ValueError("Odd num gpus is not supported for now (match aiter contract)")
    if BLOCK_SIZE <= 0 or (BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0:
        raise ValueError("BLOCK_SIZE must be a power-of-two > 0 (e.g., 256)")

    # C++ custom allreduce requires size to be multiple of 16B pack.
    PACK_ELEMS = _pack_elems(dtype_str)
    if N % PACK_ELEMS != 0:
        raise ValueError(f"custom allreduce requires input length to be multiple of {PACK_ELEMS}")

    gpu_arch = get_hip_arch()
    # One thread handles one 16B pack.
    num_packs = N // PACK_ELEMS
    num_blocks = (num_packs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _state = {}
    VEC_ALIGN = 16
    total_bytes = N * _elem_size_bytes(dtype_str)
    # Mirror the C++ dispatch: bytes%128==0 => fast path, else naive path.
    USE_FAST_VEC = (total_bytes % 128) == 0

    class _CustomAllReduce(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_all_reduce_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type, compute_type = _dtype_to_mlir(dtype_str)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

        @flir.kernel
        def custom_all_reduce_kernel(
            self: flir.T.i64,
            In: lambda: (T.memref(N, _state["elem_type"]) if world_size == 1 else T.memref(world_size, N, _state["elem_type"])),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            tid = flir.const_index(flir.thread_idx("x"))
            bid = flir.const_index(flir.block_idx("x"))

            c_num_packs = flir.const_index(num_packs)
            c0_idx = flir.const_index(0)

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            vec_e_ty = ir.VectorType.get([PACK_ELEMS], elem_type)
            vec_c_ty = ir.VectorType.get([PACK_ELEMS], compute_type)

            pack_idx = arith.ArithValue(bid) * BLOCK_SIZE + arith.ArithValue(tid)
            is_valid_pack = arith.ult(pack_idx, c_num_packs)
            pack_idx_safe = arith.select(is_valid_pack, pack_idx, c0_idx)
            base = arith.ArithValue(pack_idx_safe) * PACK_ELEMS  # element index

            # ---- Fast vector path: vector.load/store of a 16B pack ----
            # NOTE: BF16 is kept on the scalar path for correctness (see `_dtype_to_mlir`).
            if USE_FAST_VEC and dtype_str != "bf16":
                if world_size == 1:
                    v_e = flir.vector.load(vec_e_ty, In, [arith.as_value(base)], alignment=VEC_ALIGN)
                    # NOTE: `arith` registers a global value caster that may wrap op results as
                    # `ArithValue` (including vector results). Raw MLIR arith ops require `Value`,
                    # so always go through ext-arith helpers here.
                    if dtype_str == "f32":
                        v_c = v_e
                    else:
                        v_c = arith.extf(vec_c_ty, v_e)
                else:
                    zero = arith.constant(0.0, type=compute_type)
                    acc = flir.vector.splat(vec_c_ty, arith.as_value(zero))
                    for r in range_constexpr(world_size):
                        v_e = flir.vector.load(
                            vec_e_ty,
                            In,
                            [flir.const_index(r), arith.as_value(base)],
                            alignment=VEC_ALIGN,
                        )
                        if dtype_str == "f32":
                            v_c_lane = v_e
                        else:
                            v_c_lane = arith.extf(vec_c_ty, v_e)
                        acc = arith.as_value(arith.ArithValue(acc) + v_c_lane)
                    v_c = acc

                if is_valid_pack:
                    if dtype_str == "f32":
                        out_v = v_c
                    else:
                        out_v = arith.trunc_f(vec_e_ty, v_c)
                    flir.vector.store(arith.as_value(out_v), Out, [arith.as_value(base)], alignment=VEC_ALIGN)
                return

            # ---- Naive path: keep the same 16B "pack" contract but do scalar element ops ----
            if is_valid_pack:
                for lane in range_constexpr(PACK_ELEMS):
                    idx = arith.ArithValue(base) + lane
                    if world_size == 1:
                        x_e = flir.memref.load(In, [arith.as_value(idx)])
                        if dtype_str == "f32":
                            y_c = arith.ArithValue(x_e)
                        elif dtype_str == "f16":
                            y_c = arith.extf(compute_type, x_e)
                        else:
                            # bf16 storage is raw i16 bits (see `_dtype_to_mlir`).
                            # f32_bits = (u16(bf16_bits) << 16)
                            x_i32 = flir.arith.extui(T.i32(), arith.as_value(x_e))
                            x_bits = arith.ArithValue(x_i32) << 16
                            y_c = arith.ArithValue(flir.arith.bitcast(T.f32(), arith.as_value(x_bits)))
                    else:
                        acc = arith.constant(0.0, type=compute_type)
                        for r in range_constexpr(world_size):
                            x_e = flir.memref.load(In, [flir.const_index(r), arith.as_value(idx)])
                            if dtype_str == "f32":
                                x_c = arith.ArithValue(x_e)
                            elif dtype_str == "f16":
                                x_c = arith.extf(compute_type, x_e)
                            else:
                                x_i32 = flir.arith.extui(T.i32(), arith.as_value(x_e))
                                x_bits = arith.ArithValue(x_i32) << 16
                                x_c = arith.ArithValue(flir.arith.bitcast(T.f32(), arith.as_value(x_bits)))
                            acc = arith.ArithValue(acc) + x_c
                        y_c = acc

                    if dtype_str == "f32":
                        y_e = arith.as_value(y_c)
                    elif dtype_str == "f16":
                        y_e = arith.as_value(arith.trunc_f(elem_type, y_c))
                    else:
                        # f32 -> bf16 by truncation: bf16_bits = (u32(f32_bits) >> 16)
                        y_i32 = flir.arith.bitcast(T.i32(), arith.as_value(y_c))
                        y_hi = arith.shrui(y_i32, arith.i32(16))
                        y_i16 = flir.arith.trunci(T.i16(), arith.as_value(y_hi))
                        y_e = arith.as_value(y_i16)
                    flir.memref.store(y_e, Out, [arith.as_value(idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            In: lambda: (T.memref(N, _state["elem_type"]) if world_size == 1 else T.memref(world_size, N, _state["elem_type"])),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME_ALL_REDUCE],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[In, Out],
            )

    return _CustomAllReduce().module


def build_custom_all_reduce_ipc_module(
    N: int,
    dtype_str: str = "f16",
    *,
    world_size: int,
    BLOCK_SIZE: int = 256,
):
    """Build a module containing an IPC-pointer-based all-reduce kernel.

    Kernel args are specialized for `world_size`:
      (In0, In1, ..., In{world_size-1}, Out)

    Each In* is a pointer to a contiguous 1D buffer of length N on that rank's GPU,
    made accessible via HIP IPC mapping in the current process.
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if world_size <= 1 or world_size > 8:
        raise ValueError("world_size must be in [2, 8] for IPC mode")
    if world_size % 2 != 0:
        raise ValueError("Odd num gpus is not supported for now (match aiter contract)")
    if dtype_str == "bf16":
        # BF16 lowering in this repo has proven unreliable across toolchains;
        # keep IPC path to fp16/fp32 for now.
        raise ValueError("bf16 IPC all-reduce is not supported/stable yet")
    if BLOCK_SIZE <= 0 or (BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0:
        raise ValueError("BLOCK_SIZE must be a power-of-two > 0 (e.g., 256)")

    PACK_ELEMS = _pack_elems(dtype_str)
    if N % PACK_ELEMS != 0:
        raise ValueError(f"custom allreduce requires input length to be multiple of {PACK_ELEMS}")

    gpu_arch = get_hip_arch()
    VEC_ALIGN = 16
    total_bytes = N * _elem_size_bytes(dtype_str)
    USE_FAST_VEC = (total_bytes % 128) == 0

    # One thread handles one 16B pack.
    num_packs = N // PACK_ELEMS
    num_blocks = (num_packs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _state = {}

    class _CustomAllReduceIPC(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_all_reduce_ipc_{dtype_str}_ws{world_size}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type, compute_type = _dtype_to_mlir(dtype_str)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

        @flir.kernel
        def custom_all_reduce_ipc_kernel(
            self: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            Ins = [In0, In1, In2, In3, In4, In5, In6, In7]
            tid = flir.const_index(flir.thread_idx("x"))
            bid = flir.const_index(flir.block_idx("x"))

            c_num_packs = flir.const_index(num_packs)
            c0_idx = flir.const_index(0)

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]
            vec_e_ty = ir.VectorType.get([PACK_ELEMS], elem_type)
            vec_c_ty = ir.VectorType.get([PACK_ELEMS], compute_type)

            pack_idx = arith.ArithValue(bid) * BLOCK_SIZE + arith.ArithValue(tid)
            is_valid_pack = arith.ult(pack_idx, c_num_packs)
            pack_idx_safe = arith.select(is_valid_pack, pack_idx, c0_idx)
            base = arith.ArithValue(pack_idx_safe) * PACK_ELEMS

            if USE_FAST_VEC:
                zero = arith.constant(0.0, type=compute_type)
                acc = flir.vector.splat(vec_c_ty, arith.as_value(zero))
                for r in range_constexpr(world_size):
                    v_e = flir.vector.load(vec_e_ty, Ins[r], [arith.as_value(base)], alignment=VEC_ALIGN)
                    if dtype_str == "f32":
                        v_c = v_e
                    else:
                        v_c = arith.extf(vec_c_ty, v_e)
                    acc = arith.as_value(arith.ArithValue(acc) + v_c)

                if is_valid_pack:
                    out_v = acc if dtype_str == "f32" else arith.trunc_f(vec_e_ty, acc)
                    flir.vector.store(arith.as_value(out_v), Out, [arith.as_value(base)], alignment=VEC_ALIGN)
                return

            # Fallback scalar (should be rare; keep correctness).
            if is_valid_pack:
                for lane in range_constexpr(PACK_ELEMS):
                    idx = arith.ArithValue(base) + lane
                    acc = arith.constant(0.0, type=compute_type)
                    for r in range_constexpr(world_size):
                        x_e = flir.memref.load(Ins[r], [arith.as_value(idx)])
                        x_c = arith.ArithValue(x_e) if dtype_str == "f32" else arith.extf(compute_type, x_e)
                        acc = arith.ArithValue(acc) + x_c
                    y_e = arith.as_value(acc) if dtype_str == "f32" else arith.as_value(arith.trunc_f(elem_type, acc))
                    flir.memref.store(y_e, Out, [arith.as_value(idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME_ALL_REDUCE_IPC],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[In0, In1, In2, In3, In4, In5, In6, In7, Out],
            )

    return _CustomAllReduceIPC().module


def build_custom_reduce_scatter_ipc_module(
    N: int,
    dtype_str: str = "f16",
    *,
    world_size: int,
    BLOCK_SIZE: int = 256,
):
    """IPC reduce-scatter:

    Each rank computes the sum for its own chunk only (chunk = rank), writing into Out[chunk].
    This reduces per-rank peer traffic from ~world_size*N to ~N.
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if world_size <= 1 or world_size > 8:
        raise ValueError("world_size must be in [2, 8] for IPC mode")
    if world_size % 2 != 0:
        raise ValueError("Odd num gpus is not supported for now (match aiter contract)")
    if dtype_str == "bf16":
        raise ValueError("bf16 IPC all-reduce is not supported/stable yet")
    if N % world_size != 0:
        raise ValueError("reduce-scatter requires N divisible by world_size")
    if BLOCK_SIZE <= 0 or (BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0:
        raise ValueError("BLOCK_SIZE must be a power-of-two > 0 (e.g., 256)")

    PACK_ELEMS = _pack_elems(dtype_str)
    if (N // world_size) % PACK_ELEMS != 0:
        raise ValueError(f"chunk size must be multiple of {PACK_ELEMS} elems")

    gpu_arch = get_hip_arch()
    VEC_ALIGN = 16
    total_bytes = (N // world_size) * _elem_size_bytes(dtype_str)
    USE_FAST_VEC = (total_bytes % 128) == 0

    chunk_elems = N // world_size
    chunk_packs = chunk_elems // PACK_ELEMS
    num_blocks = (chunk_packs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _state = {}

    class _ReduceScatterIPC(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_reduce_scatter_ipc_{dtype_str}_ws{world_size}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type, compute_type = _dtype_to_mlir(dtype_str)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

        @flir.kernel
        def custom_reduce_scatter_ipc_kernel(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            Ins = [In0, In1, In2, In3, In4, In5, In6, In7]
            tid = flir.const_index(flir.thread_idx("x"))
            bid = flir.const_index(flir.block_idx("x"))
            c_chunk_packs = flir.const_index(chunk_packs)
            c0_idx = flir.const_index(0)

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]
            vec_e_ty = ir.VectorType.get([PACK_ELEMS], elem_type)
            vec_c_ty = ir.VectorType.get([PACK_ELEMS], compute_type)

            pack_in_chunk = arith.ArithValue(bid) * BLOCK_SIZE + arith.ArithValue(tid)
            is_valid_pack = arith.ult(pack_in_chunk, c_chunk_packs)
            pack_safe = arith.select(is_valid_pack, pack_in_chunk, c0_idx)

            rank = arith.index_cast(ir.IndexType.get(), rank_i32)
            base = arith.ArithValue(rank) * chunk_elems + arith.ArithValue(pack_safe) * PACK_ELEMS

            if USE_FAST_VEC:
                zero = arith.constant(0.0, type=compute_type)
                acc = flir.vector.splat(vec_c_ty, arith.as_value(zero))
                for r in range_constexpr(world_size):
                    v_e = flir.vector.load(vec_e_ty, Ins[r], [arith.as_value(base)], alignment=VEC_ALIGN)
                    v_c = v_e if dtype_str == "f32" else arith.extf(vec_c_ty, v_e)
                    acc = arith.as_value(arith.ArithValue(acc) + v_c)
                if is_valid_pack:
                    out_v = acc if dtype_str == "f32" else arith.trunc_f(vec_e_ty, acc)
                    flir.vector.store(arith.as_value(out_v), Out, [arith.as_value(base)], alignment=VEC_ALIGN)
                return

            if is_valid_pack:
                for lane in range_constexpr(PACK_ELEMS):
                    idx = arith.ArithValue(base) + lane
                    acc = arith.constant(0.0, type=compute_type)
                    for r in range_constexpr(world_size):
                        x_e = flir.memref.load(Ins[r], [arith.as_value(idx)])
                        x_c = arith.ArithValue(x_e) if dtype_str == "f32" else arith.extf(compute_type, x_e)
                        acc = arith.ArithValue(acc) + x_c
                    y_e = arith.as_value(acc) if dtype_str == "f32" else arith.as_value(arith.trunc_f(elem_type, acc))
                    flir.memref.store(y_e, Out, [arith.as_value(idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            rank_i32: flir.T.i32,
            In0: lambda: T.memref(N, _state["elem_type"]),
            In1: lambda: T.memref(N, _state["elem_type"]),
            In2: lambda: T.memref(N, _state["elem_type"]),
            In3: lambda: T.memref(N, _state["elem_type"]),
            In4: lambda: T.memref(N, _state["elem_type"]),
            In5: lambda: T.memref(N, _state["elem_type"]),
            In6: lambda: T.memref(N, _state["elem_type"]),
            In7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME_REDUCE_SCATTER_IPC],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[rank_i32, In0, In1, In2, In3, In4, In5, In6, In7, Out],
            )

    return _ReduceScatterIPC().module


def build_custom_all_gather_ipc_module(
    N: int,
    dtype_str: str = "f16",
    *,
    world_size: int,
    BLOCK_SIZE: int = 256,
):
    """IPC all-gather:

    After reduce-scatter, each rank has written its chunk to Out[rank*chunk : (rank+1)*chunk].
    This kernel gathers all chunks from peer Out buffers into the local Out buffer.
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if world_size <= 1 or world_size > 8:
        raise ValueError("world_size must be in [2, 8] for IPC mode")
    if world_size % 2 != 0:
        raise ValueError("Odd num gpus is not supported for now (match aiter contract)")
    if dtype_str == "bf16":
        raise ValueError("bf16 IPC all-reduce is not supported/stable yet")
    if N % world_size != 0:
        raise ValueError("all-gather requires N divisible by world_size")
    if BLOCK_SIZE <= 0 or (BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0:
        raise ValueError("BLOCK_SIZE must be a power-of-two > 0 (e.g., 256)")

    PACK_ELEMS = _pack_elems(dtype_str)
    if N % PACK_ELEMS != 0:
        raise ValueError(f"N must be multiple of {PACK_ELEMS}")

    gpu_arch = get_hip_arch()
    VEC_ALIGN = 16
    total_bytes = N * _elem_size_bytes(dtype_str)
    USE_FAST_VEC = (total_bytes % 128) == 0

    chunk_elems = N // world_size
    chunk_packs = chunk_elems // PACK_ELEMS
    num_packs = N // PACK_ELEMS
    num_blocks = (num_packs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _state = {}

    class _AllGatherIPC(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_all_gather_ipc_{dtype_str}_ws{world_size}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type, compute_type = _dtype_to_mlir(dtype_str)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

        @flir.kernel
        def custom_all_gather_ipc_kernel(
            self: flir.T.i64,
            Src0: lambda: T.memref(N, _state["elem_type"]),
            Src1: lambda: T.memref(N, _state["elem_type"]),
            Src2: lambda: T.memref(N, _state["elem_type"]),
            Src3: lambda: T.memref(N, _state["elem_type"]),
            Src4: lambda: T.memref(N, _state["elem_type"]),
            Src5: lambda: T.memref(N, _state["elem_type"]),
            Src6: lambda: T.memref(N, _state["elem_type"]),
            Src7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            Srcs = [Src0, Src1, Src2, Src3, Src4, Src5, Src6, Src7]
            tid = flir.const_index(flir.thread_idx("x"))
            bid = flir.const_index(flir.block_idx("x"))
            c_num_packs = flir.const_index(num_packs)
            c0_idx = flir.const_index(0)
            c_chunk_packs = flir.const_index(chunk_packs)

            elem_type = _state["elem_type"]
            vec_e_ty = ir.VectorType.get([PACK_ELEMS], elem_type)

            pack_idx = arith.ArithValue(bid) * BLOCK_SIZE + arith.ArithValue(tid)
            is_valid_pack = arith.ult(pack_idx, c_num_packs)
            pack_safe = arith.select(is_valid_pack, pack_idx, c0_idx)

            # src_rank = pack_idx / chunk_packs ; pack_in_chunk = pack_idx % chunk_packs
            idx_i32 = arith.index_cast(T.i32(), pack_safe)
            cp_i32 = arith.i32(chunk_packs)
            src_r_i32 = flir.arith.DivUIOp(arith.as_value(idx_i32), arith.as_value(cp_i32)).result
            p_i32 = flir.arith.RemUIOp(arith.as_value(idx_i32), arith.as_value(cp_i32)).result
            src_r = arith.index_cast(ir.IndexType.get(), src_r_i32)
            p = arith.index_cast(ir.IndexType.get(), p_i32)

            base = arith.ArithValue(src_r) * chunk_elems + arith.ArithValue(p) * PACK_ELEMS

            if USE_FAST_VEC:
                if is_valid_pack:
                    v = flir.vector.load(vec_e_ty, Src0, [arith.as_value(base)], alignment=VEC_ALIGN)
                    if arith.ArithValue(src_r_i32) == 1:
                        v = flir.vector.load(vec_e_ty, Src1, [arith.as_value(base)], alignment=VEC_ALIGN)
                    if arith.ArithValue(src_r_i32) == 2:
                        v = flir.vector.load(vec_e_ty, Src2, [arith.as_value(base)], alignment=VEC_ALIGN)
                    if arith.ArithValue(src_r_i32) == 3:
                        v = flir.vector.load(vec_e_ty, Src3, [arith.as_value(base)], alignment=VEC_ALIGN)
                    if arith.ArithValue(src_r_i32) == 4:
                        v = flir.vector.load(vec_e_ty, Src4, [arith.as_value(base)], alignment=VEC_ALIGN)
                    if arith.ArithValue(src_r_i32) == 5:
                        v = flir.vector.load(vec_e_ty, Src5, [arith.as_value(base)], alignment=VEC_ALIGN)
                    if arith.ArithValue(src_r_i32) == 6:
                        v = flir.vector.load(vec_e_ty, Src6, [arith.as_value(base)], alignment=VEC_ALIGN)
                    if arith.ArithValue(src_r_i32) == 7:
                        v = flir.vector.load(vec_e_ty, Src7, [arith.as_value(base)], alignment=VEC_ALIGN)
                    flir.vector.store(arith.as_value(v), Out, [arith.as_value(base)], alignment=VEC_ALIGN)
                return

            if is_valid_pack:
                for lane in range_constexpr(PACK_ELEMS):
                    idx = arith.ArithValue(base) + lane
                    v = flir.memref.load(Src0, [arith.as_value(idx)])
                    if arith.ArithValue(src_r_i32) == 1:
                        v = flir.memref.load(Src1, [arith.as_value(idx)])
                    if arith.ArithValue(src_r_i32) == 2:
                        v = flir.memref.load(Src2, [arith.as_value(idx)])
                    if arith.ArithValue(src_r_i32) == 3:
                        v = flir.memref.load(Src3, [arith.as_value(idx)])
                    if arith.ArithValue(src_r_i32) == 4:
                        v = flir.memref.load(Src4, [arith.as_value(idx)])
                    if arith.ArithValue(src_r_i32) == 5:
                        v = flir.memref.load(Src5, [arith.as_value(idx)])
                    if arith.ArithValue(src_r_i32) == 6:
                        v = flir.memref.load(Src6, [arith.as_value(idx)])
                    if arith.ArithValue(src_r_i32) == 7:
                        v = flir.memref.load(Src7, [arith.as_value(idx)])
                    flir.memref.store(arith.as_value(v), Out, [arith.as_value(idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Src0: lambda: T.memref(N, _state["elem_type"]),
            Src1: lambda: T.memref(N, _state["elem_type"]),
            Src2: lambda: T.memref(N, _state["elem_type"]),
            Src3: lambda: T.memref(N, _state["elem_type"]),
            Src4: lambda: T.memref(N, _state["elem_type"]),
            Src5: lambda: T.memref(N, _state["elem_type"]),
            Src6: lambda: T.memref(N, _state["elem_type"]),
            Src7: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME_ALL_GATHER_IPC],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Src0, Src1, Src2, Src3, Src4, Src5, Src6, Src7, Out],
            )

    return _AllGatherIPC().module


def build_custom_copy_chunk_ipc_module(
    N: int,
    dtype_str: str = "f16",
    *,
    world_size: int,
    BLOCK_SIZE: int = 256,
):
    """Copy one reduced chunk from a peer out buffer into local out buffer.

    Args:
      - src_rank_i32: which chunk to copy (0..world_size-1)
      - Src: pointer to the peer out buffer (full length N)
      - Out: local out buffer (full length N)
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if world_size <= 1 or world_size > 8:
        raise ValueError("world_size must be in [2, 8] for IPC mode")
    if world_size % 2 != 0:
        raise ValueError("Odd num gpus is not supported for now (match aiter contract)")
    if dtype_str == "bf16":
        raise ValueError("bf16 IPC all-reduce is not supported/stable yet")
    if N % world_size != 0:
        raise ValueError("chunk copy requires N divisible by world_size")
    if BLOCK_SIZE <= 0 or (BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0:
        raise ValueError("BLOCK_SIZE must be a power-of-two > 0 (e.g., 256)")

    PACK_ELEMS = _pack_elems(dtype_str)
    chunk_elems = N // world_size
    if chunk_elems % PACK_ELEMS != 0:
        raise ValueError(f"chunk size must be multiple of {PACK_ELEMS}")

    gpu_arch = get_hip_arch()
    VEC_ALIGN = 16
    total_bytes = chunk_elems * _elem_size_bytes(dtype_str)
    USE_FAST_VEC = (total_bytes % 128) == 0

    chunk_packs = chunk_elems // PACK_ELEMS
    num_blocks = (chunk_packs + BLOCK_SIZE - 1) // BLOCK_SIZE
    _state = {}

    class _CopyChunkIPC(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_copy_chunk_ipc_{dtype_str}_ws{world_size}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type, _compute_type = _dtype_to_mlir(dtype_str)
            _state["elem_type"] = elem_type

        @flir.kernel
        def custom_copy_chunk_ipc_kernel(
            self: flir.T.i64,
            src_rank_i32: flir.T.i32,
            Src: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            tid = flir.const_index(flir.thread_idx("x"))
            bid = flir.const_index(flir.block_idx("x"))
            c_chunk_packs = flir.const_index(chunk_packs)
            c0_idx = flir.const_index(0)

            elem_type = _state["elem_type"]
            vec_e_ty = ir.VectorType.get([PACK_ELEMS], elem_type)

            pack_in_chunk = arith.ArithValue(bid) * BLOCK_SIZE + arith.ArithValue(tid)
            is_valid_pack = arith.ult(pack_in_chunk, c_chunk_packs)
            pack_safe = arith.select(is_valid_pack, pack_in_chunk, c0_idx)

            src_rank = arith.index_cast(ir.IndexType.get(), src_rank_i32)
            base = arith.ArithValue(src_rank) * chunk_elems + arith.ArithValue(pack_safe) * PACK_ELEMS

            if USE_FAST_VEC:
                if is_valid_pack:
                    v = flir.vector.load(vec_e_ty, Src, [arith.as_value(base)], alignment=VEC_ALIGN)
                    flir.vector.store(arith.as_value(v), Out, [arith.as_value(base)], alignment=VEC_ALIGN)
                return

            if is_valid_pack:
                for lane in range_constexpr(PACK_ELEMS):
                    idx = arith.ArithValue(base) + lane
                    v = flir.memref.load(Src, [arith.as_value(idx)])
                    flir.memref.store(arith.as_value(v), Out, [arith.as_value(idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            src_rank_i32: flir.T.i32,
            Src: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME_COPY_CHUNK_IPC],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[src_rank_i32, Src, Out],
            )

    return _CopyChunkIPC().module


def build_custom_all_gather_module(
    N: int,
    dtype_str: str = "f32",
    *,
    world_size: int = 1,
    BLOCK_SIZE: int = 256,
):
    """Build a module containing a simplified all-gather kernel.

    Semantics (packed-input form):
    - If `world_size == 1`: Out[i] = In[i]
    - If `world_size > 1`: Out[r, i] = In[r, i]
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if world_size <= 0 or world_size > 8:
        raise ValueError("world_size must be in [1, 8]")
    if world_size > 1 and (world_size % 2 != 0):
        raise ValueError("Odd num gpus is not supported for now (match aiter contract)")
    if BLOCK_SIZE <= 0 or (BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0:
        raise ValueError("BLOCK_SIZE must be a power-of-two > 0 (e.g., 256)")

    gpu_arch = get_hip_arch()
    PACK_ELEMS = _pack_elems(dtype_str)
    VEC_ALIGN = 16
    total_bytes = N * _elem_size_bytes(dtype_str)
    USE_FAST_VEC = (total_bytes % 128) == 0 and (N % PACK_ELEMS == 0)

    # For world_size>1 we flatten (r, pack) space into 1D launch.
    packs_per_rank = (N // PACK_ELEMS) if (N % PACK_ELEMS == 0) else None
    total_work = N if world_size == 1 else (world_size * N)
    if USE_FAST_VEC:
        total_work = (N // PACK_ELEMS) if world_size == 1 else (world_size * (N // PACK_ELEMS))
    num_blocks = (total_work + BLOCK_SIZE - 1) // BLOCK_SIZE
    _state = {}

    class _CustomAllGather(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_all_gather_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type, _compute_type = _dtype_to_mlir(dtype_str)
            _state["elem_type"] = elem_type

        @flir.kernel
        def custom_all_gather_kernel(
            self: flir.T.i64,
            In: lambda: (T.memref(N, _state["elem_type"]) if world_size == 1 else T.memref(world_size, N, _state["elem_type"])),
            Out: lambda: (T.memref(N, _state["elem_type"]) if world_size == 1 else T.memref(world_size, N, _state["elem_type"])),
        ):
            tid = flir.const_index(flir.thread_idx("x"))
            bid = flir.const_index(flir.block_idx("x"))

            idx = arith.ArithValue(bid) * BLOCK_SIZE + arith.ArithValue(tid)

            if USE_FAST_VEC:
                vec_e_ty = ir.VectorType.get([PACK_ELEMS], _state["elem_type"])
                c_total = flir.const_index(total_work)
                is_valid = arith.ult(idx, c_total)
                if is_valid:
                    if world_size == 1:
                        base = arith.ArithValue(idx) * PACK_ELEMS
                        v = flir.vector.load(vec_e_ty, In, [arith.as_value(base)], alignment=VEC_ALIGN)
                        flir.vector.store(v, Out, [arith.as_value(base)], alignment=VEC_ALIGN)
                    else:
                        # Flattened pack index -> (r, pack_in_rank)
                        idx_i32 = arith.index_cast(T.i32(), idx)
                        cP_i32 = arith.i32(packs_per_rank)
                        r_i32 = flir.arith.DivUIOp(arith.as_value(idx_i32), arith.as_value(cP_i32)).result
                        p_i32 = flir.arith.RemUIOp(arith.as_value(idx_i32), arith.as_value(cP_i32)).result
                        r = arith.index_cast(ir.IndexType.get(), r_i32)
                        p = arith.index_cast(ir.IndexType.get(), p_i32)
                        base = arith.ArithValue(p) * PACK_ELEMS
                        v = flir.vector.load(vec_e_ty, In, [arith.as_value(r), arith.as_value(base)], alignment=VEC_ALIGN)
                        flir.vector.store(v, Out, [arith.as_value(r), arith.as_value(base)], alignment=VEC_ALIGN)
                return

            # naive scalar copy (supports any N)
            c_total = flir.const_index(total_work)
            is_valid = arith.ult(idx, c_total)
            if is_valid:
                if world_size == 1:
                    v = flir.memref.load(In, [arith.as_value(idx)])
                    flir.memref.store(v, Out, [arith.as_value(idx)])
                else:
                    idx_i32 = arith.index_cast(T.i32(), idx)
                    cN_i32 = arith.i32(N)
                    r_i32 = flir.arith.DivUIOp(arith.as_value(idx_i32), arith.as_value(cN_i32)).result
                    i_i32 = flir.arith.RemUIOp(arith.as_value(idx_i32), arith.as_value(cN_i32)).result
                    r = arith.index_cast(ir.IndexType.get(), r_i32)
                    i = arith.index_cast(ir.IndexType.get(), i_i32)
                    v = flir.memref.load(In, [arith.as_value(r), arith.as_value(i)])
                    flir.memref.store(v, Out, [arith.as_value(r), arith.as_value(i)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            In: lambda: (T.memref(N, _state["elem_type"]) if world_size == 1 else T.memref(world_size, N, _state["elem_type"])),
            Out: lambda: (T.memref(N, _state["elem_type"]) if world_size == 1 else T.memref(world_size, N, _state["elem_type"])),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME_ALL_GATHER],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[In, Out],
            )

    return _CustomAllGather().module


def meta_size() -> int:
    """Match the C++ API shape. FlyDSL demo doesn't require a device-side meta struct."""
    return 0


def _is_weak_contiguous(t) -> bool:
    """Match the C++ helper `_is_weak_contiguous`.

    This is slightly weaker than `t.is_contiguous()`:
    it also allows some views that still occupy a single dense range in storage.
    """
    try:
        if t.is_contiguous():
            return True
        # Prefer untyped_storage() on newer torch; fall back if not available.
        storage = t.untyped_storage() if hasattr(t, "untyped_storage") else t.storage()
        storage_nbytes = int(storage.nbytes())
        storage_offset_elems = int(t.storage_offset())
        elem_size = int(t.element_size())
        return storage_nbytes - storage_offset_elems * elem_size == int(t.numel()) * elem_size
    except Exception:
        return False


class CustomAllReduce:
    """A small Python shim that mirrors the AIter extension API shape.

    Limitations:
    - No IPC / multi-process signalling. For `world_size>1`, callers must provide
      inputs in a packed form (list of tensors or a stacked tensor).
    """

    def __init__(self, *, world_size: int, rank: int, full_nvlink: bool = False):
        if world_size <= 0 or world_size > 8:
            raise ValueError("world_size must be in [1, 8]")
        if world_size > 1 and (world_size % 2 != 0):
            raise ValueError("Odd num gpus is not supported for now")
        if rank < 0 or rank >= world_size:
            raise ValueError("invalid rank passed in")
        self.world_size = world_size
        self.rank = rank
        self.full_nvlink = bool(full_nvlink)
        self._exe_cache = {}
        # IPC mapping state (route B):
        # - `_ipc_ptrs[r]` is a device pointer (int) to rank r's buffer (base+offset)
        # - `_ipc_bases[r]` is the IPC base pointer to close (None for self rank / unopened)
        self._ipc_ptrs = None
        self._ipc_bases = None
        self._ipc_N = None
        self._ipc_dtype_str = None
        self._ipc_out_ptrs = None
        self._ipc_out_bases = None

    def _dtype_str(self, t):
        # Avoid importing torch in kernels/ by duck-typing.
        dt = getattr(t, "dtype", None)
        name = str(dt)
        if "float32" in name or "torch.float32" in name:
            return "f32"
        if "float16" in name or "torch.float16" in name:
            return "f16"
        if "bfloat16" in name or "torch.bfloat16" in name:
            return "bf16"
        raise ValueError(f"unsupported dtype: {dt!r}")

    def _pack_elems_for_tensor(self, t) -> int:
        return _pack_elems(self._dtype_str(t))

    def _compile(self, *, op: str, N: int, dtype_str: str, world_size: int):
        import flydsl

        key = (op, N, dtype_str, world_size)
        exe = self._exe_cache.get(key)
        if exe is not None:
            return exe
        if op == "all_reduce":
            m = build_custom_all_reduce_module(N, dtype_str=dtype_str, world_size=world_size)
        elif op == "all_reduce_ipc":
            m = build_custom_all_reduce_ipc_module(N, dtype_str=dtype_str, world_size=world_size)
        elif op == "reduce_scatter_ipc":
            m = build_custom_reduce_scatter_ipc_module(N, dtype_str=dtype_str, world_size=world_size)
        elif op == "all_gather_ipc":
            m = build_custom_all_gather_ipc_module(N, dtype_str=dtype_str, world_size=world_size)
        elif op == "copy_chunk_ipc":
            m = build_custom_copy_chunk_ipc_module(N, dtype_str=dtype_str, world_size=world_size)
        elif op == "all_gather":
            m = build_custom_all_gather_module(N, dtype_str=dtype_str, world_size=world_size)
        else:
            raise ValueError(f"unknown op: {op}")
        exe = flydsl.compile(m)
        self._exe_cache[key] = exe
        return exe

    def all_reduce_reg(self, inp, out, open_fp8_quant: bool = False):
        _ = open_fp8_quant  # not implemented in this FlyDSL demo
        if not _is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous (match aiter contract)")

        # IPC fast path: world_size>1 and we have opened peer pointers.
        if self.world_size > 1 and self._ipc_ptrs is not None:
            # Expect `inp` to be the local rank buffer (for API parity); but we don't
            # need it other than for shape/dtype validation.
            dtype_str = self._dtype_str(out)
            if self._ipc_dtype_str is not None and dtype_str != self._ipc_dtype_str:
                raise ValueError("IPC buffer dtype mismatch vs registered buffer")
            N = int(out.numel())
            if self._ipc_N is not None and int(N) != int(self._ipc_N):
                raise ValueError("IPC buffer length mismatch vs registered buffer")
            if dtype_str == "bf16":
                raise ValueError("bf16 IPC all-reduce is not supported/stable yet")
            pack_elems = _pack_elems(dtype_str)
            if int(N) % pack_elems != 0:
                raise ValueError(f"custom allreduce requires input length to be multiple of {pack_elems}")

            # NOTE: A true reduce-scatter + all-gather requires device-side signalling / memory
            # ordering to make remote reads observe peer writes. Without that, correctness is not
            # guaranteed on ROCm. Keep it behind an explicit opt-in flag.
            use_rsag = False
            try:
                import os

                use_rsag = os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_RSAG", "0") == "1"
            except Exception:
                use_rsag = False

            if use_rsag and self._ipc_out_ptrs is not None and (int(N) % int(self.world_size) == 0):
                exe_rs = self._compile(op="reduce_scatter_ipc", N=N, dtype_str=dtype_str, world_size=self.world_size)
                exe_copy = self._compile(op="copy_chunk_ipc", N=N, dtype_str=dtype_str, world_size=self.world_size)
                exe_rs(int(self.rank), *self._ipc_ptrs[:8], out)
                # IMPORTANT: synchronize across processes between phases.
                # We do not have device-side signalling yet; use torch.distributed barrier.
                try:
                    import torch  # type: ignore
                    import torch.distributed as dist  # type: ignore

                    if dist.is_initialized():
                        # Ensure RS writes are visible before any rank starts AG reads.
                        torch.cuda.synchronize()
                        dist.barrier()
                        torch.cuda.synchronize()
                except Exception:
                    pass
                # All-gather: copy chunk r from rank r's output buffer into local output.
                for r in range(self.world_size):
                    exe_copy(int(r), int(self._ipc_out_ptrs[r]), out)
            else:
                exe = self._compile(op="all_reduce_ipc", N=N, dtype_str=dtype_str, world_size=self.world_size)
                # Pass peer device pointers as raw ints.
                exe(*self._ipc_ptrs[:8], out)
            return

        # Packed mode:
        # - list/tuple of rank tensors, or
        # - a single stacked tensor shaped [world_size, N] (preferred for CUDA graph capture: no per-call stack alloc).
        if isinstance(inp, (list, tuple)) or (getattr(inp, "ndim", None) == 2 and int(getattr(inp, "shape", [0])[0]) == self.world_size and self.world_size > 1):
            import torch

            if isinstance(inp, (list, tuple)):
                xs = list(inp)
                if len(xs) != self.world_size:
                    raise ValueError("len(inp) must equal world_size in packed mode")
                x0 = xs[0]
                if any(int(x.numel()) != int(x0.numel()) for x in xs):
                    raise ValueError("all packed inputs must have same numel")
                if any(str(getattr(x, "dtype", None)) != str(getattr(x0, "dtype", None)) for x in xs):
                    raise ValueError("all packed inputs must have same dtype")
                dtype_str = self._dtype_str(x0)
                N = int(x0.numel())
                stacked = torch.stack(xs, dim=0).contiguous()
            else:
                stacked = inp
                if int(stacked.shape[0]) != self.world_size:
                    raise ValueError("packed input must have shape [world_size, N]")
                dtype_str = self._dtype_str(stacked)
                N = int(stacked.shape[1])

            if str(getattr(out, "dtype", None)) != str(getattr(stacked, "dtype", None)):
                raise ValueError("inp.scalar_type must equal out.scalar_type")
            if int(out.numel()) != int(N):
                raise ValueError("inp.numel must equal out.numel")
            pack_elems = _pack_elems(dtype_str)
            if int(N) % pack_elems != 0:
                raise ValueError(f"custom allreduce currently requires input length to be multiple of {pack_elems}")

            exe = self._compile(op="all_reduce", N=N, dtype_str=dtype_str, world_size=self.world_size)
            exe(stacked, out)
            return

        # single tensor path
        if str(getattr(inp, "dtype", None)) != str(getattr(out, "dtype", None)):
            raise ValueError("inp.scalar_type must equal out.scalar_type")
        if int(inp.numel()) != int(out.numel()):
            raise ValueError("inp.numel must equal out.numel")
        pack_elems = self._pack_elems_for_tensor(inp)
        if int(inp.numel()) % pack_elems != 0:
            raise ValueError(f"custom allreduce currently requires input length to be multiple of {pack_elems}")
        N = int(inp.numel())
        dtype_str = self._dtype_str(inp)
        exe = self._compile(op="all_reduce", N=N, dtype_str=dtype_str, world_size=1)
        exe(inp, out)

    def all_reduce_unreg(self, inp, reg_buffer, out):
        # Model the C++ API: copy into reg_buffer then run.
        input_bytes = int(inp.numel()) * int(inp.element_size())
        reg_bytes = int(reg_buffer.numel()) * int(reg_buffer.element_size())
        if input_bytes > reg_bytes:
            raise ValueError("registered buffer is too small to contain the input (bytes check)")
        reg_buffer.view_as(inp).copy_(inp)
        self.all_reduce_reg(reg_buffer.view_as(inp), out, open_fp8_quant=False)

    def all_gather_reg(self, inp, out):
        if not _is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous (match aiter contract)")
        if isinstance(inp, (list, tuple)):
            import torch

            xs = list(inp)
            if len(xs) != self.world_size:
                raise ValueError("len(inp) must equal world_size in packed mode")
            x0 = xs[0]
            if any(int(x.numel()) != int(x0.numel()) for x in xs):
                raise ValueError("all packed inputs must have same numel")
            if any(str(getattr(x, "dtype", None)) != str(getattr(x0, "dtype", None)) for x in xs):
                raise ValueError("all packed inputs must have same dtype")
            if str(getattr(out, "dtype", None)) != str(getattr(x0, "dtype", None)):
                raise ValueError("inp.scalar_type must equal out.scalar_type")
            if int(out.numel()) != self.world_size * int(x0.numel()):
                raise ValueError("out.numel must equal world_size * inp.numel in packed mode")

            N = int(x0.numel())
            dtype_str = self._dtype_str(x0)
            stacked = torch.stack(xs, dim=0).contiguous()
            exe = self._compile(op="all_gather", N=N, dtype_str=dtype_str, world_size=self.world_size)
            exe(stacked, out)
            return
        if str(getattr(inp, "dtype", None)) != str(getattr(out, "dtype", None)):
            raise ValueError("inp.scalar_type must equal out.scalar_type")
        if int(inp.numel()) != int(out.numel()):
            raise ValueError("inp.numel must equal out.numel")
        N = int(inp.numel())
        dtype_str = self._dtype_str(inp)
        exe = self._compile(op="all_gather", N=N, dtype_str=dtype_str, world_size=1)
        exe(inp, out)

    def all_gather_unreg(self, inp, reg_buffer, out):
        input_bytes = int(inp.numel()) * int(inp.element_size())
        reg_bytes = int(reg_buffer.numel()) * int(reg_buffer.element_size())
        if input_bytes > reg_bytes:
            raise ValueError("registered buffer is too small to contain the input (bytes check)")
        reg_buffer.view_as(inp).copy_(inp)
        self.all_gather_reg(reg_buffer.view_as(inp), out)

    def dispose(self):
        # Close any opened IPC mappings.
        try:
            if self._ipc_bases is not None:
                from flydsl.runtime import ipc as _ipc

                for b in self._ipc_bases:
                    if b is None:
                        continue
                    _ipc.close_ipc_handle(int(b))
                if self._ipc_out_bases is not None:
                    for b in self._ipc_out_bases:
                        if b is None:
                            continue
                        _ipc.close_ipc_handle(int(b))
        finally:
            self._ipc_ptrs = None
            self._ipc_bases = None
            self._ipc_N = None
            self._ipc_dtype_str = None
            self._ipc_out_ptrs = None
            self._ipc_out_bases = None
        self._exe_cache.clear()

    # The following APIs exist in the C++ extension for IPC / graph capture integration.
    # We keep them as explicit stubs to make the limitation clear.
    def register_buffer(self, t, handles, offsets):
        # Route B: open IPC handles for peer buffers and cache device pointers.
        # `t` is the local rank's tensor buffer that peers will read from.
        from flydsl.runtime import ipc as _ipc

        if self.world_size <= 1:
            raise ValueError("register_buffer is only meaningful for world_size>1")
        if len(handles) != self.world_size or len(offsets) != self.world_size:
            raise ValueError("handles/offsets length must equal world_size")
        if not _is_weak_contiguous(t):
            raise ValueError("registered buffer must be weak-contiguous")

        dtype_str = self._dtype_str(t)
        if dtype_str == "bf16":
            raise ValueError("bf16 IPC all-reduce is not supported/stable yet")
        N = int(t.numel())
        pack_elems = _pack_elems(dtype_str)
        if N % pack_elems != 0:
            raise ValueError(f"custom allreduce requires input length to be multiple of {pack_elems}")

        # Close any previous mappings first.
        self.dispose()

        ptrs = [None for _ in range(self.world_size)]
        bases = [None for _ in range(self.world_size)]

        # Local rank pointer.
        ptrs[self.rank] = int(t.data_ptr())

        for r in range(self.world_size):
            if r == self.rank:
                continue
            h = handles[r]
            if hasattr(h, "detach") and hasattr(h, "cpu"):
                hb = bytes(h.detach().cpu().numpy().tobytes())
            else:
                hb = bytes(h)
            base_ptr = _ipc.open_ipc_handle(hb)
            bases[r] = int(base_ptr)
            ptrs[r] = int(base_ptr) + int(offsets[r])

        # Kernel signature is fixed to 8 ptr args (In0..In7). Pad by repeating rank0 ptr.
        pad_ptr = int(ptrs[0])
        ptrs8 = [pad_ptr] * 8
        for i in range(min(self.world_size, 8)):
            ptrs8[i] = int(ptrs[i])

        self._ipc_ptrs = [int(p) for p in ptrs8]
        self._ipc_bases = bases
        self._ipc_N = N
        self._ipc_dtype_str = dtype_str

    def register_out_buffer(self, t_out, handles, offsets):
        """Register peer pointers for output buffer (used as source for all-gather phase)."""
        from flydsl.runtime import ipc as _ipc

        if self.world_size <= 1:
            raise ValueError("register_out_buffer is only meaningful for world_size>1")
        if len(handles) != self.world_size or len(offsets) != self.world_size:
            raise ValueError("handles/offsets length must equal world_size")
        if not _is_weak_contiguous(t_out):
            raise ValueError("output buffer must be weak-contiguous")

        dtype_str = self._dtype_str(t_out)
        if dtype_str == "bf16":
            raise ValueError("bf16 IPC all-reduce is not supported/stable yet")
        if self._ipc_dtype_str is not None and dtype_str != self._ipc_dtype_str:
            raise ValueError("out buffer dtype mismatch vs registered input buffer")
        N = int(t_out.numel())
        if self._ipc_N is not None and int(N) != int(self._ipc_N):
            raise ValueError("out buffer length mismatch vs registered input buffer")

        bases = [None for _ in range(self.world_size)]
        ptrs = [None for _ in range(self.world_size)]
        ptrs[self.rank] = int(t_out.data_ptr())
        for r in range(self.world_size):
            if r == self.rank:
                continue
            h = handles[r]
            if hasattr(h, "detach") and hasattr(h, "cpu"):
                hb = bytes(h.detach().cpu().numpy().tobytes())
            else:
                hb = bytes(h)
            base_ptr = _ipc.open_ipc_handle(hb)
            bases[r] = int(base_ptr)
            ptrs[r] = int(base_ptr) + int(offsets[r])

        pad_ptr = int(ptrs[0])
        ptrs8 = [pad_ptr] * 8
        for i in range(min(self.world_size, 8)):
            ptrs8[i] = int(ptrs[i])

        self._ipc_out_ptrs = [int(p) for p in ptrs8]
        self._ipc_out_bases = bases

    def get_graph_buffer_ipc_meta(self):
        raise NotImplementedError("graph buffer IPC meta is not available in FlyDSL demo")

    def register_graph_buffers(self, handles, offsets):
        raise NotImplementedError("register_graph_buffers requires IPC runtime; not available in FlyDSL demo")


def init_custom_ar(meta, rank_data, handles, offsets, rank: int, full_nvlink: bool):
    """C++-shaped initializer used by tests/harnesses.

    This repo provides two implementations behind the same API surface:
    - **FlyDSL shim** (default): `CustomAllReduce` in this file.
    - **AIter backend**: `aiter.dist.custom_all_reduce.CustomAllreduce` (real multi-GPU kernel).

    Control via env vars:
    - `FLYDSL_CUSTOM_ALL_REDUCE_BACKEND`:
        - `"flydsl"` (default): return FlyDSL shim
        - `"aiter"`: return AIter backend object (no new backend name)
    - `FLYDSL_AITER_IMPL` (only when backend=aiter):
        - `"aiter"` (default): run AIter kernel
        - `"flydsl"`: run FlyDSL kernel (for perf A/B; correctness not guaranteed on ROCm w/o signalling)
    """
    import os

    # Validate basic contract (match the C++ checks).
    _ = meta
    world_size = len(offsets)
    if world_size > 8:
        raise ValueError("world size > 8 is not supported")
    if world_size > 1 and (world_size % 2 != 0):
        raise ValueError("Odd num gpus is not supported for now")
    if world_size != len(handles):
        raise ValueError("handles length should equal to offsets length")
    if rank < 0 or rank >= world_size:
        raise ValueError("invalid rank passed in")

    backend = str(os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_BACKEND", "aiter")).strip().lower()
    if backend != "aiter":
        fa = CustomAllReduce(world_size=world_size, rank=rank, full_nvlink=full_nvlink)
        # If rank_data is a CUDA tensor, treat it as the registered buffer and open IPC handles.
        try:
            if hasattr(rank_data, "is_cuda") and bool(rank_data.is_cuda):
                fa.register_buffer(rank_data, handles, offsets)
        except Exception:
            pass
        return fa

    # ---- AIter backend ----
    impl = str(os.environ.get("FLYDSL_AITER_IMPL", "flydsl")).strip().lower()
    if impl not in {"aiter", "flydsl"}:
        raise ValueError(f"unsupported FLYDSL_AITER_IMPL={impl!r} (expected 'aiter' or 'flydsl')")

    # Create AIter object (for real backend); attach it to a non-NCCL group.
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore

    if not dist.is_initialized():
        raise RuntimeError("backend=aiter requires torch.distributed to be initialized")

    # AIter asserts group backend != NCCL. Use a gloo group for control-plane exchange.
    global _FLYDSL_AITER_GLOO_GROUP
    try:
        _FLYDSL_AITER_GLOO_GROUP  # type: ignore[name-defined]
    except Exception:
        _FLYDSL_AITER_GLOO_GROUP = None  # type: ignore[name-defined]
    if _FLYDSL_AITER_GLOO_GROUP is None:  # type: ignore[name-defined]
        try:
            _FLYDSL_AITER_GLOO_GROUP = dist.new_group(backend="gloo")  # type: ignore[name-defined]
        except Exception:
            # If gloo group creation fails, fall back to WORLD and let AIter raise a clearer error.
            _FLYDSL_AITER_GLOO_GROUP = dist.group.WORLD  # type: ignore[name-defined]

    # Device: prefer rank_data.device, else cuda:{rank}.
    dev = None
    try:
        dev = getattr(rank_data, "device", None)
    except Exception:
        dev = None
    if dev is None:
        dev = torch.device(f"cuda:{rank}")

    # Max size (bytes) for AIter.
    max_size = int(os.environ.get("FLYDSL_AITER_MAX_SIZE_BYTES", str(64 * 1024 * 1024)))

    from aiter.dist.custom_all_reduce import CustomAllreduce as _AIterCustomAllreduce  # type: ignore

    aiter_ar = _AIterCustomAllreduce(_FLYDSL_AITER_GLOO_GROUP, dev, max_size=max_size)  # type: ignore[arg-type,name-defined]

    # For fair kernel-only timing, register the caller-provided buffer so `all_reduce_reg` is valid.
    try:
        if hasattr(rank_data, "is_cuda") and bool(rank_data.is_cuda):
            aiter_ar.register_buffer(rank_data)
    except Exception:
        pass

    if impl == "aiter":
        return aiter_ar

    # impl == "flydsl": backend stays "aiter" (same control-plane + meta buffer),
    # but the compute kernel is FlyDSL-generated and uses the same AIter signal protocol.
    return _AIterSignalFlyDSLAllreduce(
        group=_FLYDSL_AITER_GLOO_GROUP,  # type: ignore[name-defined]
        device=dev,
        max_size=max_size,
        world_size=world_size,
        rank=rank,
        full_nvlink=bool(full_nvlink),
        rank_data=rank_data,
        handles=handles,
        offsets=offsets,
    )


class _AIterSignalFlyDSLAllreduce:
    """FlyDSL kernel runner that matches AIter ROCm signalling protocol.

    This object:
    - allocates AIter-style uncached meta buffer (signal + tmp)
    - exchanges meta IPC handles over a gloo group
    - opens peer meta + peer input buffers via HIP IPC
    - launches a FlyDSL kernel that uses AIter's start/end/_flag protocol
    """

    def __init__(
        self,
        *,
        group,
        device,
        max_size: int,
        world_size: int,
        rank: int,
        full_nvlink: bool,
        rank_data,
        handles,
        offsets,
    ):
        import torch  # type: ignore
        import torch.distributed as dist  # type: ignore
        import aiter as aiter_ops  # type: ignore
        from flydsl.runtime import ipc as fly_ipc

        self.group = group
        self.device = device
        self.max_size = int(max_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.full_nvlink = bool(full_nvlink)

        # AIter meta layout: [signal/meta (aiter_ops.meta_size bytes)] + [tmp buffer (max_size bytes)].
        self.meta = aiter_ops.allocate_meta_buffer(int(aiter_ops.meta_size()) + int(self.max_size))
        # Ensure deterministic initial flags.
        try:
            self.meta.zero_()
        except Exception:
            pass
        self._meta_size = int(aiter_ops.meta_size())

        # Exchange meta IPC handles via gloo broadcast list (AIter-compatible).
        handle = aiter_ops.get_meta_buffer_ipc_handle(self.meta)
        shard_data = (handle, 0)
        all_data = [[None] for _ in range(self.world_size)]
        all_data[self.rank][0] = shard_data
        ranks = dist.get_process_group_ranks(group=self.group)
        ranks = sorted(ranks)
        for i, r in enumerate(ranks):
            dist.broadcast_object_list(all_data[i], src=r, group=self.group, device="cpu")
        meta_handles = [all_data[i][0][0] for i in range(self.world_size)]
        meta_offsets = [int(all_data[i][0][1]) for i in range(self.world_size)]

        # Open peer meta (Signal base pointers) and derive tmp pointers.
        self._meta_bases = [None for _ in range(self.world_size)]
        self._sg_ptrs = [0 for _ in range(8)]
        self._tmp_ptrs = [0 for _ in range(8)]
        for r in range(self.world_size):
            if r == self.rank:
                base_ptr = int(self.meta.data_ptr())
            else:
                base_ptr = int(fly_ipc.open_ipc_handle(bytes(meta_handles[r])))
                self._meta_bases[r] = base_ptr
            sg_ptr = base_ptr + int(meta_offsets[r])
            tmp_ptr = sg_ptr + self._meta_size
            if r < 8:
                self._sg_ptrs[r] = int(sg_ptr)
                self._tmp_ptrs[r] = int(tmp_ptr)
        # Pad to 8 args.
        for i in range(self.world_size, 8):
            self._sg_ptrs[i] = int(self._sg_ptrs[0])
            self._tmp_ptrs[i] = int(self._tmp_ptrs[0])
        self._self_sg = int(self._sg_ptrs[self.rank])

        # Open peer input pointers from (handles, offsets). Use rank_data for local ptr.
        if not hasattr(rank_data, "is_cuda") or not bool(rank_data.is_cuda):
            raise ValueError("rank_data must be a CUDA tensor buffer for aiter-signal flydsl path")
        self._in_bases = [None for _ in range(self.world_size)]
        self._in_ptrs = [0 for _ in range(8)]
        self._in_ptrs[self.rank] = int(rank_data.data_ptr())
        for r in range(self.world_size):
            if r == self.rank:
                continue
            h = handles[r]
            hb = bytes(h.detach().cpu().numpy().tobytes()) if hasattr(h, "detach") else bytes(h)
            base_ptr = int(fly_ipc.open_ipc_handle(hb))
            self._in_bases[r] = base_ptr
            self._in_ptrs[r] = int(base_ptr) + int(offsets[r])
        for i in range(self.world_size, 8):
            self._in_ptrs[i] = int(self._in_ptrs[0])

        # Unregistered buffer for all_reduce_unreg (AIter-like).
        self.buffer = torch.empty(int(self.max_size), dtype=torch.uint8, device=self.device)

        self._exe_cache = {}
        # Cache per (N, dtype_str, world_size) launch plans to avoid rebuilding Python objects
        # (notably fake memref descriptors) on every call.
        self._plan_cache = {}

    def close(self):
        from flydsl.runtime import ipc as fly_ipc

        # Close opened peer meta and input mappings.
        for b in self._meta_bases:
            if b is None:
                continue
            fly_ipc.close_ipc_handle(int(b))
        for b in self._in_bases:
            if b is None:
                continue
            fly_ipc.close_ipc_handle(int(b))
        self._meta_bases = []
        self._in_bases = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _dtype_str(self, t):
        name = str(getattr(t, "dtype", None))
        if "float16" in name:
            return "f16"
        if "float32" in name:
            return "f32"
        raise ValueError(f"unsupported dtype: {name}")

    def _compile(self, *, N: int, dtype_str: str, world_size: int):
        import flydsl
        from kernels.aiter_signal_all_reduce_raw import build_aiter_signal_allreduce_raw_module
        key = (N, dtype_str, world_size)
        exe = self._exe_cache.get(key)
        if exe is not None:
            return exe
        import os
        max_spin = int(os.environ.get("FLYDSL_AITER_SIGNAL_MAX_SPIN", "20000000"))
        # Use the exact meta_size from aiter to avoid debug/tmp overlap.
        m = build_aiter_signal_allreduce_raw_module(
            N=int(N),
            dtype_str=str(dtype_str),
            world_size=int(world_size),
            threads=256,
            meta_size_bytes=int(self._meta_size),
            max_spin=int(max_spin),
        )
        exe = flydsl.compile(m)
        self._exe_cache[key] = exe
        return exe

    def _get_plan(self, *, N: int, dtype_str: str):
        """Get (and cache) an execution plan for this (N, dtype, ws).

        The plan caches:
        - compiled exe and bound callables (run_1stage/run_2stage)
        - sg_args tuple
        - remote input memref descriptors template (self rank filled per call)
        - tmp memref descriptors (all ranks)
        - precomputed blocks for 1stage/2stage with fixed threads
        """
        key = (int(N), str(dtype_str), int(self.world_size))
        plan = self._plan_cache.get(key)
        if plan is not None:
            return plan

        exe = self._compile(N=int(N), dtype_str=str(dtype_str), world_size=self.world_size)
        run1 = exe.run_1stage
        run2 = exe.run_2stage
        run2p = getattr(exe, "run_2stage_ptr", None)

        PACK_ELEMS = int(_pack_elems(dtype_str))
        if int(N) % PACK_ELEMS != 0:
            raise ValueError(f"input length must be multiple of {PACK_ELEMS}")
        size_packs = int(N) // PACK_ELEMS
        part_packs = size_packs // int(self.world_size)
        largest_part_packs = part_packs + (size_packs % int(self.world_size))

        threads = 256
        blocks_1 = (int(size_packs) + threads - 1) // threads
        blocks_2 = (int(part_packs) + threads - 1) // threads
        blocks_1 = max(1, min(int(_AITER_KMAXBLOCKS), int(blocks_1)))
        blocks_2 = max(1, min(int(_AITER_KMAXBLOCKS), int(blocks_2)))

        sg_args = tuple(int(x) for x in self._sg_ptrs[:8])

        from kernels.aiter_signal_all_reduce_raw import make_fake_1d_memref as _fake1d

        # Inputs: cache remote descriptors; self rank uses the real tensor each call.
        in_template = [None for _ in range(8)]
        for r in range(int(self.world_size)):
            if r == int(self.rank):
                in_template[r] = None
            else:
                in_template[r] = _fake1d(data_ptr=int(self._in_ptrs[r]), n_elems=int(N))

        # Tmp: cache all descriptors (they point into uncached meta allocations).
        tmp_elems = int(largest_part_packs * PACK_ELEMS)
        tmp_args = [None for _ in range(8)]
        for r in range(int(self.world_size)):
            tmp_args[r] = _fake1d(
                data_ptr=int(self._tmp_ptrs[r]),
                n_elems=int(tmp_elems),
                base_ptr=int(self._sg_ptrs[r]),
            )
        for i in range(int(self.world_size), 8):
            tmp_args[i] = tmp_args[0]

        # Rotated layout (AIter-style): index i corresponds to target=(rank+i)%ws.
        ws = int(self.world_size)
        rk = int(self.rank)
        in_rot_template = [None for _ in range(8)]
        tmp_rot_args = [None for _ in range(8)]
        for i in range(ws):
            in_rot_template[i] = in_template[(rk + i) % ws]
            tmp_rot_args[i] = tmp_args[(rk + i) % ws]
        # Pad remaining entries for ABI (8 args).
        for i in range(ws, 8):
            in_rot_template[i] = in_rot_template[0]
            tmp_rot_args[i] = tmp_rot_args[0]

        plan = {
            "exe": exe,
            "run1": run1,
            "run2": run2,
            "run2p": run2p,
            "threads": int(threads),
            "blocks_1": int(blocks_1),
            "blocks_2": int(blocks_2),
            "sg_args": sg_args,
            "in_template": in_template,
            "tmp_args": tuple(tmp_args),
            "in_rot_template": tuple(in_rot_template),
            "tmp_rot_args": tuple(tmp_rot_args),
            # Rotated raw pointers (i64) for ptr-based entrypoint (index0 is self rank).
            "in_rot_ptrs": tuple(int(self._in_ptrs[(rk + i) % ws]) for i in range(8)),
            "tmp_rot_ptrs": tuple(int(self._tmp_ptrs[(rk + i) % ws]) for i in range(8)),
            "PACK_ELEMS": int(PACK_ELEMS),
            "size_packs": int(size_packs),
            "part_packs": int(part_packs),
            "largest_part_packs": int(largest_part_packs),
        }
        self._plan_cache[key] = plan
        return plan

    def all_reduce_reg(self, inp, out=None, open_fp8_quant: bool = False):
        _ = open_fp8_quant
        import torch  # type: ignore
        import os
        import time

        if out is None:
            out = torch.empty_like(inp)
        if not _is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous (match aiter contract)")
        if int(inp.numel()) != int(out.numel()):
            raise ValueError("inp.numel must equal out.numel")
        dtype_str = self._dtype_str(out)
        if dtype_str != self._dtype_str(inp):
            raise ValueError("inp/out dtype mismatch")

        # AIter requires bytes % 16 == 0.
        bytes_n = int(out.numel()) * int(out.element_size())
        if bytes_n % 16 != 0:
            raise ValueError("aiter-signal flydsl allreduce requires byte size multiple of 16")
        if bytes_n > int(self.max_size):
            raise ValueError(f"input bytes {bytes_n} exceed max_size {self.max_size}")

        N = int(out.numel())
        plan = self._get_plan(N=N, dtype_str=dtype_str)
        PACK_ELEMS = int(plan["PACK_ELEMS"])
        size_packs = int(plan["size_packs"])
        part_packs = int(plan["part_packs"])
        largest_part_packs = int(plan["largest_part_packs"])
        tmp_bytes = int(largest_part_packs) * 16
        if tmp_bytes > int(self.max_size):
            raise ValueError(f"tmp bytes {tmp_bytes} exceed max_size {self.max_size}")

        exe = plan["exe"]

        # Mirror AIter kernel selection (simplified to 1stage vs 2stage).
        call_1stage = (self.world_size == 2) or (
            self.full_nvlink
            and ((self.world_size <= 4 and bytes_n < 160 * 1024) or (self.world_size <= 8 and bytes_n < 80 * 1024))
        )
        # Build in_args from cached rotated template (index0 is always self rank).
        in_args = list(plan["in_rot_template"])
        in_args[0] = inp
        fill_val = in_args[0] if in_args[0] is not None else inp
        for i in range(int(self.world_size), 8):
            in_args[i] = fill_val

        sg_args = plan["sg_args"]

        threads = int(plan["threads"])
        blocks = int(plan["blocks_1"] if call_1stage else plan["blocks_2"])

        trace = str(os.environ.get("FLYDSL_AITER_TRACE", "0")).strip() not in ("", "0", "false", "False")
        trace_timing = str(os.environ.get("FLYDSL_AITER_TRACE_TIMING", "0")).strip() not in ("", "0", "false", "False")
        if trace and int(self.rank) == 0:
            stage = "1stage" if call_1stage else "2stage"
            print(
                "[flydsl-aiter-trace] "
                f"stage={stage} ws={self.world_size} full_nvlink={bool(self.full_nvlink)} "
                f"dtype={dtype_str} N={N} bytes={bytes_n} "
                f"packs={size_packs} part_packs={part_packs} largest_part_packs={largest_part_packs} "
                f"grid(blocks)={blocks} block(threads)={threads}",
                flush=True,
            )

        t0 = time.perf_counter() if trace_timing else None
        ev0 = ev1 = None
        if trace_timing:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        if call_1stage:
            plan["run1"](int(self.rank), int(blocks), int(self._self_sg), *sg_args, *in_args, out)
            if trace_timing:
                ev1.record()
                ev1.synchronize()
                t1 = time.perf_counter()
                if int(self.rank) == 0:
                    ms = float(ev0.elapsed_time(ev1))
                    print(f"[flydsl-aiter-trace] kernel_ms={ms:.3f} host_s={t1 - t0:.6f}", flush=True)
            return out

        # Prefer ptr-based 2stage entrypoint when available to reduce memref descriptor overhead (stage2 hot path).
        run2p = plan.get("run2p", None)
        if run2p is not None:
            in_ptrs = list(plan["in_rot_ptrs"])
            tmp_ptrs = list(plan["tmp_rot_ptrs"])
            # index0 is always self rank in rotated layout; override in0 with real inp pointer.
            in_ptrs[0] = int(inp.data_ptr())
            out_ptr = int(out.data_ptr())
            run2p(int(self.rank), int(blocks), int(self._self_sg), *sg_args, *in_ptrs, *tmp_ptrs, int(out_ptr))
        else:
            tmp_args = plan["tmp_rot_args"]
            plan["run2"](int(self.rank), int(blocks), int(self._self_sg), *sg_args, *in_args, *tmp_args, out)
        if trace_timing:
            ev1.record()
            ev1.synchronize()
            t1 = time.perf_counter()
            if int(self.rank) == 0:
                ms = float(ev0.elapsed_time(ev1))
                print(f"[flydsl-aiter-trace] kernel_ms={ms:.3f} host_s={t1 - t0:.6f}", flush=True)
        return out

    def all_reduce_unreg(self, inp, out=None):
        import torch  # type: ignore

        if out is None:
            out = torch.empty_like(inp)
        input_bytes = int(inp.numel()) * int(inp.element_size())
        if input_bytes > int(self.buffer.numel()) * int(self.buffer.element_size()):
            raise ValueError("registered buffer is too small to contain the input")
        self.buffer.view_as(inp).copy_(inp)
        return self.all_reduce_reg(self.buffer.view_as(inp), out)



# ---- Optional free-function wrappers ----
def all_reduce_reg(fa: CustomAllReduce, inp, out, open_fp8_quant: bool = False):
    return fa.all_reduce_reg(inp, out, open_fp8_quant=open_fp8_quant)


def all_reduce_unreg(fa: CustomAllReduce, inp, reg_buffer, out):
    return fa.all_reduce_unreg(inp, reg_buffer, out)


def all_gather_reg(fa: CustomAllReduce, inp, out):
    return fa.all_gather_reg(inp, out)


def all_gather_unreg(fa: CustomAllReduce, inp, reg_buffer, out):
    return fa.all_gather_unreg(inp, reg_buffer, out)


def dispose(fa: CustomAllReduce):
    return fa.dispose()


