"""Custom all-reduce kernel + Python-facing shim.

This kernel is intentionally written in the same "ext arith" style as
`moe_gemm_2stage.py` / `preshuffle_gemm.py`:
- Use `@flir.kernel` and Python control-flow lowering
- Prefer `flydsl.dialects.ext.arith` helpers / `ArithValue` operator overloading
- Use `arith.as_value(...)` only at strict MLIR builder boundaries (e.g. memref.load/store)

Notes vs the C++ AIter implementation you pasted:
- Multi-GPU collectives are provided via the optional AIter backend.
- This file provides:
  - A **single-process** functional subset (world_size==1), and
  - A "packed input" mode (world_size>1) where inputs are provided as a stacked tensor
    `[world_size, N]` (or a Python list of tensors that we stack), and we compute the
    elementwise sum to mimic all-reduce semantics for testing/experimentation.
"""

from contextlib import contextmanager

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

# ---- AIter ROCm Signal layout constants (from ISA / header) ----
# start: uint32_t start[kMaxBlocks][8]   => 80*8*4 = 2560 bytes
# end:   uint32_t end[kMaxBlocks][8]     => 2560 bytes, aligned(128) => offset 2560
# flag:  uint32_t _flag[kMaxBlocks]      => 80*4 = 320 bytes, aligned(128) => offset 5120 (0x1400)
_AITER_KMAXBLOCKS = 80
_AITER_START_OFF_B = 0
_AITER_END_OFF_B = 2560
_AITER_FLAG_OFF_B = 5120


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


def init_custom_ar(meta, rank_data, handles, offsets, rank: int, full_nvlink: bool, out=None):
    """C++-shaped initializer used by tests/harnesses.

    This function always uses the AIter backend. Control via env var:
    - `FLYDSL_AITER_IMPL`:
        - `"aiter"` (default): run AIter kernel
        - `"flydsl"`: run FlyDSL kernel that follows AIter signalling protocol

    Args:
        out: Optional output buffer to register for AIter backend. If provided,
            this buffer will be registered for use with registered_output=True.
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

    # AIter has moved CustomAllreduce across versions:
    # - old: aiter.dist.custom_all_reduce
    # - mid: aiter.dist.device_communicators.custom_all_reduce
    # Keep a compatibility import chain so tests don't depend on one exact layout.
    try:
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce as AIterCustomAllreduce  # type: ignore
    except ModuleNotFoundError:
        try:
            from aiter.dist.custom_all_reduce import CustomAllreduce as AIterCustomAllreduce  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Cannot import AIter CustomAllreduce. Tried "
                "'aiter.dist.device_communicators.custom_all_reduce' and "
                "'aiter.dist.custom_all_reduce'. "
                "Please ensure your aiter Python package/version is compatible."
            ) from e

    aiter_ar = AIterCustomAllreduce(_FLYDSL_AITER_GLOO_GROUP, dev, max_size=max_size)  # type: ignore[arg-type,name-defined]

    # For fair kernel-only timing, register the caller-provided buffer so registered_input can be used.
    # Also register the output buffer if provided for registered_output=True.
    try:
        if hasattr(rank_data, "is_cuda") and bool(rank_data.is_cuda):
            aiter_ar.register_input_buffer(rank_data)
            # Register output buffer if provided
        if out is not None and hasattr(out, "is_cuda") and bool(out.is_cuda):
            aiter_ar.register_output_buffer(out)
    except Exception:
        pass

    if impl == "aiter":
        return aiter_ar

    # impl == "flydsl": backend stays "aiter" (same control-plane group),
    # but compute uses FlyDSL-generated kernels that follow the AIter signal protocol.
    return FlyDSLAllreduce(
        group=_FLYDSL_AITER_GLOO_GROUP,  # type: ignore[name-defined]
        device=dev,
        max_size=max_size,
        world_size=world_size,
        rank=rank,
        full_nvlink=bool(full_nvlink),
        rank_data=rank_data,
    )


class FlyDSLAllreduce:
    """Run FlyDSL kernels that follow AIter signal protocol on ROCm.

    This path needs to (a) exchange per-rank meta handles (signal/tmp) and
    (b) map peer input buffers into the current process so the kernel can read them.
    """

    # HIP IPC handle is an opaque 64-byte blob.
    _HIP_IPC_HANDLE_BYTES = 64
    _HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
    _hip = None
    _hipIpcMemHandle_t = None

    @classmethod
    def _load_hip(cls):
        if cls._hip is not None:
            return cls._hip
        import ctypes

        for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                cls._hip = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if cls._hip is None:
            raise RuntimeError("Failed to load HIP runtime library (libamdhip64.so)")

        class hipIpcMemHandle_t(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_byte * cls._HIP_IPC_HANDLE_BYTES)]

        cls._hipIpcMemHandle_t = hipIpcMemHandle_t

        # Bind prototypes (best-effort; HIP uses C ABI).
        cls._hip.hipIpcGetMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(hipIpcMemHandle_t), ctypes.c_void_p]
        cls._hip.hipIpcOpenMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint]
        cls._hip.hipIpcCloseMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        cls._hip.hipGetErrorString.restype = ctypes.c_char_p
        cls._hip.hipGetErrorString.argtypes = [ctypes.c_int]
        return cls._hip

    @classmethod
    def _hip_check(cls, err: int, *, what: str):
        if int(err) == 0:
            return
        hip = cls._load_hip()
        try:
            s = hip.hipGetErrorString(int(err))
            msg = s.decode("utf-8", errors="replace") if s else f"hipError({err})"
        except Exception:
            msg = f"hipError({err})"
        raise RuntimeError(f"{what} failed: {msg}")

    @classmethod
    def _get_mem_handle_bytes(cls, base_ptr: int) -> bytes:
        import ctypes

        hip = cls._load_hip()
        h = cls._hipIpcMemHandle_t()  # type: ignore[misc]
        err = hip.hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcGetMemHandle")
        return bytes(ctypes.string_at(ctypes.byref(h), cls._HIP_IPC_HANDLE_BYTES))

    @classmethod
    def _open_mem_handle(cls, handle_bytes: bytes) -> int:
        import ctypes

        if len(handle_bytes) != cls._HIP_IPC_HANDLE_BYTES:
            raise ValueError(f"Expected {cls._HIP_IPC_HANDLE_BYTES}B handle, got {len(handle_bytes)}B")
        hip = cls._load_hip()
        h = cls._hipIpcMemHandle_t()  # type: ignore[misc]
        ctypes.memmove(ctypes.byref(h), bytes(handle_bytes), cls._HIP_IPC_HANDLE_BYTES)
        out_ptr = ctypes.c_void_p()
        err = hip.hipIpcOpenMemHandle(ctypes.byref(out_ptr), h, ctypes.c_uint(int(cls._HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)))
        cls._hip_check(err, what="hipIpcOpenMemHandle")
        return int(out_ptr.value)

    @classmethod
    def _close_mem_handle(cls, base_ptr: int) -> None:
        import ctypes

        hip = cls._load_hip()
        err = hip.hipIpcCloseMemHandle(ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcCloseMemHandle")

    @staticmethod
    def _gather_object_list_via_broadcast(group, shard_data):
        """AIter-style gather that works with gloo under inference mode."""
        import torch.distributed as dist  # type: ignore

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        all_data = [[None] for _ in range(world_size)]
        all_data[rank][0] = shard_data
        ranks = dist.get_process_group_ranks(group=group)
        ranks = sorted(ranks)
        for i, r in enumerate(ranks):
            dist.broadcast_object_list(all_data[i], src=r, group=group, device="cpu")
        return [all_data[i][0] for i in range(world_size)]

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
    ):
        import os
        import torch  # type: ignore
        import torch.distributed as dist  # type: ignore
        import aiter as aiter_ops  # type: ignore

        self.group = group
        self.device = device
        self.max_size = int(max_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.full_nvlink = bool(full_nvlink)

        if not dist.is_initialized():
            raise RuntimeError("FLYDSL_AITER_IMPL=flydsl requires torch.distributed to be initialized")
        if self.world_size <= 1:
            raise ValueError("FLYDSL_AITER_IMPL=flydsl is only meaningful for world_size>1")
        if not hasattr(rank_data, "is_cuda") or not bool(rank_data.is_cuda):
            raise ValueError("rank_data must be a CUDA tensor buffer for flydsl+aiter path")

        # AIter meta layout: [signal/meta] + [tmp buffer].
        self.meta = aiter_ops.allocate_meta_buffer(int(aiter_ops.meta_size()) + int(self.max_size))
        try:
            self.meta.zero_()
        except Exception:
            pass
        self._meta_size = int(aiter_ops.meta_size())

        # ---- Exchange and map meta buffers (signal/tmp) ----
        my_meta_h = aiter_ops.get_meta_buffer_ipc_handle(self.meta)
        my_meta_bytes = bytes(my_meta_h.detach().cpu().numpy().tobytes())
        all_meta = self._gather_object_list_via_broadcast(self.group, (my_meta_bytes, 0))

        self._meta_bases = [None for _ in range(self.world_size)]
        self._sg_ptrs = [0 for _ in range(8)]
        self._tmp_ptrs = [0 for _ in range(8)]
        for r in range(self.world_size):
            hb, off = all_meta[r]
            if r == self.rank:
                base_ptr = int(self.meta.data_ptr())
            else:
                base_ptr = int(self._open_mem_handle(bytes(hb)))
                self._meta_bases[r] = base_ptr
            sg_ptr = int(base_ptr) + int(off)
            tmp_ptr = int(sg_ptr) + int(self._meta_size)
            if r < 8:
                self._sg_ptrs[r] = int(sg_ptr)
                self._tmp_ptrs[r] = int(tmp_ptr)
        for i in range(self.world_size, 8):
            self._sg_ptrs[i] = int(self._sg_ptrs[0])
            self._tmp_ptrs[i] = int(self._tmp_ptrs[0])
        self._self_sg = int(self._sg_ptrs[self.rank])

        # ---- Exchange and map input buffers ----
        # Use raw HIP handle export (64B) to match hipIpcOpenMemHandle.
        storage = rank_data.untyped_storage() if hasattr(rank_data, "untyped_storage") else rank_data.storage()
        base_ptr = int(storage.data_ptr())
        ptr = int(rank_data.data_ptr())
        off_bytes = int(ptr - base_ptr)
        my_in_h = self._get_mem_handle_bytes(base_ptr)
        all_in = self._gather_object_list_via_broadcast(self.group, (my_in_h, off_bytes))

        self._in_bases = [None for _ in range(self.world_size)]
        self._in_ptrs = [0 for _ in range(8)]
        for r in range(self.world_size):
            hb, off = all_in[r]
            if r == self.rank:
                self._in_ptrs[r] = int(rank_data.data_ptr())
                continue
            peer_base = int(self._open_mem_handle(bytes(hb)))
            self._in_bases[r] = peer_base
            self._in_ptrs[r] = int(peer_base) + int(off)
        for i in range(self.world_size, 8):
            self._in_ptrs[i] = int(self._in_ptrs[0])

        # ---- Allocate and register input_buffer / output_buffer (eager path) ----
        self.input_buffer = torch.empty(self.max_size, dtype=torch.uint8, device=self.device)
        self.output_buffer = torch.empty(self.max_size, dtype=torch.uint8, device=self.device)
        # Register input_buffer: gather IPC (handle, offset) for all ranks
        inp_buf_storage = self.input_buffer.untyped_storage() if hasattr(self.input_buffer, "untyped_storage") else self.input_buffer.storage()
        inp_buf_base = int(inp_buf_storage.data_ptr())
        inp_buf_off = int(self.input_buffer.data_ptr()) - inp_buf_base
        my_inp_buf_h = self._get_mem_handle_bytes(inp_buf_base)
        all_inp_buf = self._gather_object_list_via_broadcast(self.group, (my_inp_buf_h, inp_buf_off))
        self._input_buffer_bases = [None for _ in range(self.world_size)]
        self._input_buffer_ptrs = [0 for _ in range(8)]
        for r in range(self.world_size):
            hb, off = all_inp_buf[r]
            if r == self.rank:
                self._input_buffer_ptrs[r] = int(self.input_buffer.data_ptr())
            else:
                peer_base = int(self._open_mem_handle(bytes(hb)))
                self._input_buffer_bases[r] = peer_base
                self._input_buffer_ptrs[r] = int(peer_base) + int(off)
        for i in range(self.world_size, 8):
            self._input_buffer_ptrs[i] = int(self._input_buffer_ptrs[0])
        # Register output_buffer
        out_buf_storage = self.output_buffer.untyped_storage() if hasattr(self.output_buffer, "untyped_storage") else self.output_buffer.storage()
        out_buf_base = int(out_buf_storage.data_ptr())
        out_buf_off = int(self.output_buffer.data_ptr()) - out_buf_base
        my_out_buf_h = self._get_mem_handle_bytes(out_buf_base)
        all_out_buf = self._gather_object_list_via_broadcast(self.group, (my_out_buf_h, out_buf_off))
        self._output_buffer_bases = [None for _ in range(self.world_size)]
        self._output_buffer_ptrs = [0 for _ in range(8)]
        for r in range(self.world_size):
            hb, off = all_out_buf[r]
            if r == self.rank:
                self._output_buffer_ptrs[r] = int(self.output_buffer.data_ptr())
            else:
                peer_base = int(self._open_mem_handle(bytes(hb)))
                self._output_buffer_bases[r] = peer_base
                self._output_buffer_ptrs[r] = int(peer_base) + int(off)
        for i in range(self.world_size, 8):
            self._output_buffer_ptrs[i] = int(self._output_buffer_ptrs[0])
        # Rotated input_buffer ptrs for eager kernel (index i = (rank+i)%world_size)
        _ws, _rk = int(self.world_size), int(self.rank)
        rotated_input_buf_ptrs = [int(self._input_buffer_ptrs[(_rk + i) % _ws]) for i in range(8)]
        self._gpu_input_buffer_ptrs_array = torch.tensor(rotated_input_buf_ptrs, dtype=torch.int64, device=self.device)

        # CUDAGraph state
        self._IS_CAPTURING = False
        self._graph_inp = None
        self._graph_out = None
        self._gpu_graph_in_ptrs_array = None
        self._graph_in_bases = []

        # Pre-compute rotated pointer arrays and copy to GPU tensors
        # Rotated layout (AIter-style): index i corresponds to target=(rank+i)%ws.
        ws = int(self.world_size)
        rk = int(self.rank)
        # Compute rotated in_ptrs (base rotation, index 0 will be updated per call)
        rotated_in_ptrs_base = [int(self._in_ptrs[(rk + i) % ws]) for i in range(8)]
        
        # Copy to GPU tensors (int64 for pointer values) as device memory arrays
        # sg_ptrs are NOT rotated (passed as-is), tmp_ptrs rotation will be handled in kernel
        self._gpu_sg_ptrs_array = torch.tensor([int(x) for x in self._sg_ptrs[:8]], dtype=torch.int64, device=self.device)
        self._gpu_in_ptrs_array = torch.tensor(rotated_in_ptrs_base, dtype=torch.int64, device=self.device)
        
        # Store tmp_offset and world_size for computing tmp_ptrs in kernel
        # tmp_ptrs[i] = sg_ptrs[(rank + i) % world_size] + tmp_offset (rotation handled in kernel)
        self._tmp_offset = int(self._meta_size)

        self._exe_cache = {}
        self._threads = 512  # Match AIter's cross_device_reduce_2stage_write_mode
        self._max_spin = int(os.environ.get("FLYDSL_AITER_SIGNAL_MAX_SPIN", "20000000"))
        
        # Compile kernel once during initialization
        N = int(rank_data.numel())
        dtype_str = self._dtype_str(rank_data)
        self._exe = self._compile(N=N, dtype_str=dtype_str)

    def close(self):
        # Close mapped peer meta and input handles.
        for b in getattr(self, "_meta_bases", []):
            if b is None:
                continue
            self._close_mem_handle(int(b))
        for b in getattr(self, "_in_bases", []):
            if b is None:
                continue
            self._close_mem_handle(int(b))
        for b in getattr(self, "_input_buffer_bases", []):
            if b is None:
                continue
            self._close_mem_handle(int(b))
        for b in getattr(self, "_output_buffer_bases", []):
            if b is None:
                continue
            self._close_mem_handle(int(b))
        for b in getattr(self, "_graph_in_bases", []):
            if b is None:
                continue
            self._close_mem_handle(int(b))
        self._meta_bases = []
        self._in_bases = []
        self._input_buffer_bases = []
        self._output_buffer_bases = []
        self._graph_in_bases = []

    @contextmanager
    def capture(self):
        """Context manager for CUDA graph capture. On exit, calls register_graph_buffers()."""
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            self.register_graph_buffers()

    def register_graph_buffers(self):
        """Gather IPC for graph-captured (inp, out) and build _gpu_graph_in_ptrs_array for replay."""
        import torch  # type: ignore
        if self._graph_inp is None:
            return
        # Gather IPC for each rank's _graph_inp (the inp tensor used during capture)
        storage = self._graph_inp.untyped_storage() if hasattr(self._graph_inp, "untyped_storage") else self._graph_inp.storage()
        base_ptr = int(storage.data_ptr())
        off = int(self._graph_inp.data_ptr()) - base_ptr
        my_handle = self._get_mem_handle_bytes(base_ptr)
        all_graph_in = self._gather_object_list_via_broadcast(self.group, (my_handle, off))
        ws = int(self.world_size)
        rk = int(self.rank)
        self._graph_in_bases = [None for _ in range(self.world_size)]
        graph_in_ptrs = [0 for _ in range(8)]
        for r in range(self.world_size):
            hb, o = all_graph_in[r]
            if r == self.rank:
                graph_in_ptrs[r] = int(self._graph_inp.data_ptr())
            else:
                peer_base = int(self._open_mem_handle(bytes(hb)))
                self._graph_in_bases[r] = peer_base
                graph_in_ptrs[r] = int(peer_base) + int(o)
        for i in range(self.world_size, 8):
            graph_in_ptrs[i] = int(graph_in_ptrs[0])
        rotated = [int(graph_in_ptrs[(rk + i) % ws]) for i in range(8)]
        self._gpu_graph_in_ptrs_array = torch.tensor(rotated, dtype=torch.int64, device=self.device)

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

    def _compile(self, *, N: int, dtype_str: str):
        import flydsl
        from kernels.aiter_signal_all_reduce_raw import build_aiter_signal_allreduce_raw_module
        print("start compile ****************************")

        # ABI tag: wrapper signatures include `stream_ptr` (avoid per-launch stream create/sync/destroy).
        key = (int(N), str(dtype_str), int(self.world_size), "abi_stream_v1")
        exe = self._exe_cache.get(key)
        if exe is not None:
            return exe
        m = build_aiter_signal_allreduce_raw_module(
            N=int(N),
            dtype_str=str(dtype_str),
            world_size=int(self.world_size),
            threads=int(self._threads),
            meta_size_bytes=int(self._meta_size),
            max_spin=int(self._max_spin),
        )
        exe = flydsl.compile(m)
        self._exe_cache[key] = exe
        return exe

    def _run_kernel_2stage(self, N: int, dtype_str: str, gpu_in_ptrs_array, out_ptr: int, *, capture_self_inp_ptr: int = None):
        """Launch 2-stage allreduce kernel. When capturing, pass capture_self_inp_ptr=inp.data_ptr() so host has our rank's in ptr."""
        import torch  # type: ignore
        pack_elems = 8 if dtype_str == "f16" else 4
        num_packs = int(N) // int(pack_elems)
        part_p = int(num_packs) // int(self.world_size)
        grid_x = max(1, min(int(_AITER_KMAXBLOCKS), (max(1, int(part_p)) + int(self._threads) - 1) // int(self._threads)))
        stream_ptr = torch.cuda.current_stream().cuda_stream
        # During capture, never compile (could trigger sync/alloc); use cached exe only.
        if torch.cuda.is_current_stream_capturing():
            key = (int(N), str(dtype_str), int(self.world_size), "abi_stream_v1")
            exe = self._exe_cache.get(key)
            if exe is None:
                raise RuntimeError(
                    "allreduce exe not in cache during graph capture; run warmup (eager allreduce) before capture"
                )
        else:
            exe = self._compile(N=N, dtype_str=dtype_str)
        # During HIP graph capture: no device loads, no stream wait. Use run_2stage_direct with host ptrs.
        if torch.cuda.is_current_stream_capturing():
            if not hasattr(exe, "run_2stage_direct"):
                raise RuntimeError("run_2stage_direct not found; rebuild with graph-capture support")
            ws, rk = int(self.world_size), int(self.rank)
            sg_direct = [int(self._sg_ptrs[(rk + i) % ws]) for i in range(8)]
            self_inp = int(capture_self_inp_ptr) if capture_self_inp_ptr is not None else int(self._in_ptrs[rk])
            in_direct = [
                self_inp if (rk + i) % ws == rk else int(self._in_ptrs[(rk + i) % ws])
                for i in range(8)
            ]
            tmp_direct = [int(self._tmp_ptrs[(rk + i) % ws]) for i in range(8)]
            exe.run_2stage_direct(
                int(self.rank),
                int(grid_x),
                int(self._self_sg),
                *sg_direct,
                *in_direct,
                *tmp_direct,
                int(out_ptr),
                int(stream_ptr),
            )
        else:
            exe.run_2stage_ptr(
                int(self.rank),
                int(grid_x),
                int(self._self_sg),
                int(self._gpu_sg_ptrs_array.data_ptr()),
                int(gpu_in_ptrs_array.data_ptr()),
                int(self._tmp_offset),
                int(out_ptr),
                int(stream_ptr),
            )

    def all_reduce(
        self,
        inp,
        *,
        out=None,
        use_new: bool = True,
        open_fp8_quant: bool = False,
        registered_input: bool = False,
        registered_output: bool = False,
    ):
        """Out-of-place all reduce. Eager: copy inp->input_buffer, kernel, copy output_buffer->out. Registered/graph: use inp/out directly."""
        import torch  # type: ignore
        _ = use_new
        _ = open_fp8_quant
        if out is None:
            out = torch.empty_like(inp)
        if int(inp.numel()) != int(out.numel()):
            raise ValueError("inp.numel must equal out.numel")
        if not _is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous")
        dtype_str = self._dtype_str(inp)
        if dtype_str != self._dtype_str(out):
            raise ValueError("inp/out dtype mismatch")
        bytes_n = int(inp.numel()) * int(inp.element_size())
        if bytes_n % 16 != 0:
            raise ValueError("byte size must be multiple of 16")
        if bytes_n > int(self.max_size):
            raise ValueError(f"input bytes {bytes_n} exceed max_size {self.max_size}")
        N = int(out.numel())

        if registered_input and registered_output:
            # Registered path (or graph replay): use inp/out pointers
            if self._IS_CAPTURING and torch.cuda.is_current_stream_capturing():
                # Capture: record (inp, out) and run kernel with host-side ptrs (no device load)
                self._graph_inp = inp
                self._graph_out = out
                self._gpu_in_ptrs_array[0] = int(inp.data_ptr())
                self._run_kernel_2stage(
                    N, dtype_str, self._gpu_in_ptrs_array, int(out.data_ptr()),
                    capture_self_inp_ptr=int(inp.data_ptr()),
                )
                return out
            if getattr(self, "_gpu_graph_in_ptrs_array", None) is not None and not self._IS_CAPTURING:
                # Replay: use graph-registered in_ptrs and current out (same tensor as capture)
                self._run_kernel_2stage(N, dtype_str, self._gpu_graph_in_ptrs_array, int(out.data_ptr()))
                return out
            # Registered but not graph: use _gpu_in_ptrs_array (rank_data ptrs) with [0]=inp
            self._gpu_in_ptrs_array[0] = int(inp.data_ptr())
            self._run_kernel_2stage(N, dtype_str, self._gpu_in_ptrs_array, int(out.data_ptr()))
            return out

        # Eager path: memcpy inp -> input_buffer, kernel -> output_buffer, memcpy output_buffer -> out
        inp_u8 = inp.view(torch.uint8)
        self.input_buffer[:bytes_n].copy_(inp_u8)
        self._run_kernel_2stage(N, dtype_str, self._gpu_input_buffer_ptrs_array, int(self.output_buffer.data_ptr()))
        out.view(torch.uint8)[:bytes_n].copy_(self.output_buffer[:bytes_n])
        return out

    def custom_all_reduce(self, inp, use_new: bool = True, open_fp8_quant: bool = False):
        """Like aiter: in capture and capturing use registered; else eager."""
        import torch  # type: ignore
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                print("**********capture****************")
                return self.all_reduce(inp, use_new=use_new, open_fp8_quant=open_fp8_quant, registered_input=True, registered_output=True)
            out = torch.empty_like(inp)
            out.zero_()
            return out
        return self.all_reduce(inp, use_new=use_new, open_fp8_quant=open_fp8_quant, registered_input=False, registered_output=False)

    def all_reduce_reg(self, inp, out=None, open_fp8_quant: bool = False):
        """Eager path: uses input_buffer/output_buffer and memcpy."""
        return self.all_reduce(inp, out=out, open_fp8_quant=open_fp8_quant, registered_input=False, registered_output=False)
