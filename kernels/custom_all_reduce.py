"""Custom all-reduce example kernel (block-level sum + broadcast).

This kernel is intentionally written in the same "ext arith" style as
`moe_gemm_2stage.py` / `preshuffle_gemm.py`:
- Use `@flir.kernel` and Python control-flow lowering
- Prefer `flydsl.dialects.ext.arith` helpers / `ArithValue` operator overloading
- Use `arith.as_value(...)` only at strict MLIR builder boundaries (e.g. gpu.ShuffleOp, memref.load/store)
"""

from flydsl.dialects.ext import flir, arith
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME = "custom_all_reduce_kernel"


def build_custom_all_reduce_module(N: int, dtype_str: str = "f32", *, BLOCK_SIZE: int = 256):
    """Build a module containing a block-level all-reduce sum kernel.

    Semantics:
    - Each block handles a contiguous segment of length `BLOCK_SIZE`
    - Compute sum over the segment (in f32 for f16/bf16)
    - Broadcast that sum to every element in the segment:
        Out[i] = sum(In[block_start:block_end])   for i in that block
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if BLOCK_SIZE <= 0 or (BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0:
        raise ValueError("BLOCK_SIZE must be a power-of-two > 0 (e.g., 256)")

    gpu_arch = get_hip_arch()
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Shared-memory scratch for cross-wave reduction (one slot per wave).
    WARP_SIZE = 64
    RED_SLOTS = max(1, (BLOCK_SIZE + WARP_SIZE - 1) // WARP_SIZE)
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _CustomAllReduce(flir.MlirModule):
        GPU_MODULE_NAME = f"custom_all_reduce_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            if dtype_str == "f32":
                elem_type = T.f32()
            elif dtype_str == "f16":
                elem_type = T.f16()
            elif dtype_str == "bf16":
                elem_type = ir.BF16Type.get()
            else:
                raise ValueError(f"unsupported dtype: {dtype_str}")

            compute_type = T.f32() if dtype_str != "f32" else elem_type
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

            _state["smem_red"] = allocator.allocate_array(compute_type, RED_SLOTS)
            allocator.finalize()

        @flir.kernel
        def custom_all_reduce_kernel(
            self: flir.T.i64,
            In: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            tid = flir.const_index(flir.thread_idx("x"))
            bid = flir.const_index(flir.block_idx("x"))

            c_N = flir.const_index(N)
            c0_idx = flir.const_index(0)

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            # Shared scratch.
            base_ptr = allocator.get_base()
            s_red = _state["smem_red"](base_ptr).get()

            # global index for this thread
            idx = arith.ArithValue(bid) * BLOCK_SIZE + arith.ArithValue(tid)
            is_valid = arith.ult(idx, c_N)

            # Predicated load to avoid SSA value escaping `scf.if` (flir-to-standard can be fragile here).
            idx_safe = arith.select(is_valid, idx, c0_idx)
            x_e = flir.memref.load(In, [arith.as_value(idx_safe)])
            x_c = arith.ArithValue(x_e) if dtype_str == "f32" else arith.extf(compute_type, x_e)
            x = arith.select(is_valid, x_c, arith.constant(0.0, type=compute_type))

            # ---- block reduce (sum) + broadcast ----
            tid_i32 = arith.index_cast(T.i32(), tid)
            c_warp = arith.i32(WARP_SIZE)
            lane_i32 = flir.arith.RemUIOp(arith.as_value(tid_i32), arith.as_value(c_warp)).result
            wave_i32 = flir.arith.DivUIOp(arith.as_value(tid_i32), arith.as_value(c_warp)).result

            width_i32 = arith.as_value(c_warp)

            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.as_value(arith.i32(sh))
                peer = flir.gpu_ext.ShuffleOp(arith.as_value(w), off, width_i32, mode="xor").shuffleResult
                w = arith.ArithValue(w) + peer

            is_lane0 = (arith.ArithValue(lane_i32) == 0)
            if is_lane0:
                wave_idx = arith.index_cast(ir.IndexType.get(), wave_i32)
                flir.memref.store(arith.as_value(w), s_red, [arith.as_value(wave_idx)])
            flir.gpu_ext.barrier()

            is_wave0 = (arith.ArithValue(wave_i32) == 0)
            if is_wave0:
                in_range = arith.ult(lane_i32, arith.i32(RED_SLOTS))
                lane_safe = arith.select(in_range, lane_i32, arith.i32(0))
                lane_safe_idx = arith.index_cast(ir.IndexType.get(), lane_safe)
                v = flir.memref.load(s_red, [arith.as_value(lane_safe_idx)])
                vv = arith.select(in_range, v, arith.as_value(arith.constant(0.0, type=compute_type)))

                ww = arith.ArithValue(vv)
                for sh in [32, 16, 8, 4, 2, 1]:
                    off = arith.as_value(arith.i32(sh))
                    peer = flir.gpu_ext.ShuffleOp(arith.as_value(ww), off, width_i32, mode="xor").shuffleResult
                    ww = arith.ArithValue(ww) + peer

                is_lane0_2 = (arith.ArithValue(lane_i32) == 0)
                if is_lane0_2:
                    flir.memref.store(arith.as_value(ww), s_red, [c0_idx])
            flir.gpu_ext.barrier()

            total = flir.memref.load(s_red, [c0_idx])  # broadcasted block sum

            if is_valid:
                if dtype_str == "f32":
                    y_e = total
                else:
                    y_e = arith.as_value(arith.trunc_f(elem_type, total))
                flir.memref.store(y_e, Out, [arith.as_value(idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            In: lambda: T.memref(N, _state["elem_type"]),
            Out: lambda: T.memref(N, _state["elem_type"]),
        ):
            c1 = arith.index(1)
            gx = arith.index(num_blocks)
            bx = arith.index(BLOCK_SIZE)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[In, Out],
            )

    return _CustomAllReduce().module


