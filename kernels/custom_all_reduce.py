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

from flydsl.dialects.ext import flir, arith
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME_ALL_REDUCE = "custom_all_reduce_kernel"
KERNEL_NAME_ALL_GATHER = "custom_all_gather_kernel"
KERNEL_NAME_ALL_REDUCE_IPC = "custom_all_reduce_ipc_kernel"


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
        finally:
            self._ipc_ptrs = None
            self._ipc_bases = None
            self._ipc_N = None
            self._ipc_dtype_str = None
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

    def get_graph_buffer_ipc_meta(self):
        raise NotImplementedError("graph buffer IPC meta is not available in FlyDSL demo")

    def register_graph_buffers(self, handles, offsets):
        raise NotImplementedError("register_graph_buffers requires IPC runtime; not available in FlyDSL demo")


def init_custom_ar(meta, rank_data, handles, offsets, rank: int, full_nvlink: bool):
    # Validate basic contract (match the C++ checks), but we don't consume IPC handles here.
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
    fa = CustomAllReduce(world_size=world_size, rank=rank, full_nvlink=full_nvlink)
    # If rank_data is a CUDA tensor, treat it as the registered buffer and open IPC handles.
    try:
        if hasattr(rank_data, "is_cuda") and bool(rank_data.is_cuda):
            fa.register_buffer(rank_data, handles, offsets)
    except Exception:
        pass
    return fa


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


