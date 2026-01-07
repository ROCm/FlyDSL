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


def _dtype_to_mlir(dtype_str: str):
    if dtype_str == "f32":
        return T.f32(), T.f32()
    if dtype_str == "f16":
        return T.f16(), T.f32()
    if dtype_str == "bf16":
        return ir.BF16Type.get(), T.f32()
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
            if USE_FAST_VEC:
                if world_size == 1:
                    v_e = flir.vector.load(vec_e_ty, In, [arith.as_value(base)], alignment=VEC_ALIGN)
                    v_c = v_e if dtype_str == "f32" else flir.arith.extf(vec_c_ty, (v_e))
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
                        v_c = v_e if dtype_str == "f32" else flir.arith.extf(vec_c_ty, (v_e))
                        acc = arith.as_value(arith.ArithValue(acc) + v_c)
                    v_c = acc

                if is_valid_pack:
                    if dtype_str == "f32":
                        out_v = v_c
                    else:
                        out_v = flir.arith.truncf(vec_e_ty, (v_c))
                    flir.vector.store(out_v, Out, [arith.as_value(base)], alignment=VEC_ALIGN)
                return

            # ---- Naive path: keep the same 16B "pack" contract but do scalar element ops ----
            if is_valid_pack:
                for lane in range_constexpr(PACK_ELEMS):
                    idx = arith.ArithValue(base) + lane
                    if world_size == 1:
                        x_e = flir.memref.load(In, [arith.as_value(idx)])
                        y_c = arith.ArithValue(x_e) if dtype_str == "f32" else arith.extf(compute_type, x_e)
                    else:
                        acc = arith.constant(0.0, type=compute_type)
                        for r in range_constexpr(world_size):
                            x_e = flir.memref.load(In, [flir.const_index(r), arith.as_value(idx)])
                            x_c = arith.ArithValue(x_e) if dtype_str == "f32" else arith.extf(compute_type, x_e)
                            acc = arith.ArithValue(acc) + x_c
                        y_c = acc

                    y_e = arith.as_value(y_c) if dtype_str == "f32" else arith.as_value(arith.trunc_f(elem_type, y_c))
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

        # Accept either a single tensor (world_size==1) or a list/tuple of tensors (packed mode).
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
            if int(out.numel()) != int(x0.numel()):
                raise ValueError("inp.numel must equal out.numel")
            # Match C++ allreduce hard requirement: size must be multiple of pack elems.
            pack_elems = self._pack_elems_for_tensor(x0)
            if int(x0.numel()) % pack_elems != 0:
                raise ValueError(f"custom allreduce currently requires input length to be multiple of {pack_elems}")

            N = int(x0.numel())
            dtype_str = self._dtype_str(x0)
            stacked = torch.stack(xs, dim=0).contiguous()
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
        self._exe_cache.clear()

    # The following APIs exist in the C++ extension for IPC / graph capture integration.
    # We keep them as explicit stubs to make the limitation clear.
    def register_buffer(self, t, handles, offsets):
        raise NotImplementedError("register_buffer requires IPC runtime; not available in FlyDSL demo")

    def get_graph_buffer_ipc_meta(self):
        raise NotImplementedError("graph buffer IPC meta is not available in FlyDSL demo")

    def register_graph_buffers(self, handles, offsets):
        raise NotImplementedError("register_graph_buffers requires IPC runtime; not available in FlyDSL demo")


def init_custom_ar(meta, rank_data, handles, offsets, rank: int, full_nvlink: bool):
    # Validate basic contract (match the C++ checks), but we don't consume IPC handles here.
    _ = meta
    _ = rank_data
    world_size = len(offsets)
    if world_size > 8:
        raise ValueError("world size > 8 is not supported")
    if world_size > 1 and (world_size % 2 != 0):
        raise ValueError("Odd num gpus is not supported for now")
    if world_size != len(handles):
        raise ValueError("handles length should equal to offsets length")
    if rank < 0 or rank >= world_size:
        raise ValueError("invalid rank passed in")
    return CustomAllReduce(world_size=world_size, rank=rank, full_nvlink=full_nvlink)


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


