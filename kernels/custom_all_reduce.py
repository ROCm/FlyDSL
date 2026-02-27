"""Custom all-reduce kernel + Python-facing shim.

This file provides FlyDSL-generated allreduce kernels that follow the AIter
signal protocol for multi-GPU communication on ROCm.
"""

from contextlib import contextmanager
import torch

# ---- AIter ROCm Signal layout constants ----
_AITER_KMAXBLOCKS = 80


def meta_size() -> int:
    """Return meta buffer size (for API compatibility with aiter)."""
    return 0


def _is_weak_contiguous(t) -> bool:
    """Check if tensor occupies a single dense range in storage."""
    try:
        if t.is_contiguous():
            return True
        storage = t.untyped_storage() if hasattr(t, "untyped_storage") else t.storage()
        storage_nbytes = int(storage.nbytes())
        storage_offset_elems = int(t.storage_offset())
        elem_size = int(t.element_size())
        return storage_nbytes - storage_offset_elems * elem_size == int(t.numel()) * elem_size
    except Exception:
        return False


def init_custom_ar(meta, rank_data, handles, offsets, rank: int, full_nvlink: bool, out=None):
    """Initialize allreduce with AIter or FlyDSL backend.

    Backend controlled by env var FLYDSL_AITER_IMPL:
    - "aiter" (default): use AIter kernel
    - "flydsl": use FlyDSL kernel with AIter signal protocol
    """
    import os
    import torch.distributed as dist

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

    impl = str(os.environ.get("FLYDSL_AITER_IMPL", "flydsl")).strip().lower()
    if impl not in {"aiter", "flydsl"}:
        raise ValueError(f"unsupported FLYDSL_AITER_IMPL={impl!r}")

    if not dist.is_initialized():
        raise RuntimeError("backend=aiter requires torch.distributed to be initialized")

    # Create gloo group for control-plane exchange (AIter requires non-NCCL group)
    global _FLYDSL_AITER_GLOO_GROUP
    try:
        _FLYDSL_AITER_GLOO_GROUP
    except NameError:
        _FLYDSL_AITER_GLOO_GROUP = None
    if _FLYDSL_AITER_GLOO_GROUP is None:
        try:
            _FLYDSL_AITER_GLOO_GROUP = dist.new_group(backend="gloo")
        except Exception:
            _FLYDSL_AITER_GLOO_GROUP = dist.group.WORLD

    dev = getattr(rank_data, "device", None) or torch.device(f"cuda:{rank}")
    max_size = int(os.environ.get("FLYDSL_AITER_MAX_SIZE_BYTES", str(64 * 1024 * 1024)))

    # Import AIter CustomAllreduce
    try:
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce as AIterCustomAllreduce
    except ModuleNotFoundError:
        try:
            from aiter.dist.custom_all_reduce import CustomAllreduce as AIterCustomAllreduce
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Cannot import AIter CustomAllreduce") from e

    aiter_ar = AIterCustomAllreduce(_FLYDSL_AITER_GLOO_GROUP, dev, max_size=max_size)

    try:
        if hasattr(rank_data, "is_cuda") and bool(rank_data.is_cuda):
            aiter_ar.register_input_buffer(rank_data)
        if out is not None and hasattr(out, "is_cuda") and bool(out.is_cuda):
            aiter_ar.register_output_buffer(out)
    except Exception:
        pass

    if impl == "aiter":
        return aiter_ar

    return FlyDSLAllreduce(
        group=_FLYDSL_AITER_GLOO_GROUP,
        device=dev,
        max_size=max_size,
        world_size=world_size,
        rank=rank,
        full_nvlink=bool(full_nvlink),
    )


class FlyDSLAllreduce:
    """FlyDSL kernels following AIter signal protocol on ROCm."""

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
            raise RuntimeError("Failed to load HIP runtime library")

        class hipIpcMemHandle_t(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_byte * cls._HIP_IPC_HANDLE_BYTES)]
        cls._hipIpcMemHandle_t = hipIpcMemHandle_t

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
        h = cls._hipIpcMemHandle_t()
        err = hip.hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcGetMemHandle")
        return bytes(ctypes.string_at(ctypes.byref(h), cls._HIP_IPC_HANDLE_BYTES))

    @classmethod
    def _open_mem_handle(cls, handle_bytes: bytes) -> int:
        import ctypes
        if len(handle_bytes) != cls._HIP_IPC_HANDLE_BYTES:
            raise ValueError(f"Expected {cls._HIP_IPC_HANDLE_BYTES}B handle")
        hip = cls._load_hip()
        h = cls._hipIpcMemHandle_t()
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
        import torch.distributed as dist
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        all_data = [[None] for _ in range(world_size)]
        all_data[rank][0] = shard_data
        ranks = sorted(dist.get_process_group_ranks(group=group))
        for i, r in enumerate(ranks):
            dist.broadcast_object_list(all_data[i], src=r, group=group, device="cpu")
        return [all_data[i][0] for i in range(world_size)]

    def __init__(self, *, group, device, max_size: int, world_size: int, rank: int, full_nvlink: bool):
        import os
        import torch.distributed as dist
        import aiter as aiter_ops

        self.group = group
        self.device = device
        self.max_size = int(max_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.full_nvlink = bool(full_nvlink)

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        if self.world_size <= 1:
            raise ValueError("world_size must be > 1")

        # Allocate meta buffer (signal + tmp)
        self.meta = aiter_ops.allocate_meta_buffer(int(aiter_ops.meta_size()) + int(self.max_size))
        try:
            self.meta.zero_()
        except Exception:
            pass
        self._meta_size = int(aiter_ops.meta_size())

        # Exchange meta buffers
        my_meta_h = aiter_ops.get_meta_buffer_ipc_handle(self.meta)
        my_meta_bytes = bytes(my_meta_h.detach().cpu().numpy().tobytes())
        all_meta = self._gather_object_list_via_broadcast(self.group, (my_meta_bytes, 0))

        self._meta_bases = [None] * self.world_size
        self._sg_ptrs = [0] * 8
        self._tmp_ptrs = [0] * 8
        for r in range(self.world_size):
            hb, off = all_meta[r]
            base_ptr = int(self.meta.data_ptr()) if r == self.rank else int(self._open_mem_handle(bytes(hb)))
            if r != self.rank:
                self._meta_bases[r] = base_ptr
            sg_ptr = base_ptr + off
            tmp_ptr = sg_ptr + self._meta_size
            if r < 8:
                self._sg_ptrs[r] = sg_ptr
                self._tmp_ptrs[r] = tmp_ptr
        for i in range(self.world_size, 8):
            self._sg_ptrs[i] = self._sg_ptrs[0]
            self._tmp_ptrs[i] = self._tmp_ptrs[0]
        self._self_sg = self._sg_ptrs[self.rank]
        self._gpu_sg_ptrs_array = torch.tensor(self._sg_ptrs[:8], dtype=torch.int64, device=self.device)

        # Allocate eager buffers
        self.input_buffer = torch.empty(self.max_size, dtype=torch.uint8, device=self.device)
        self.output_buffer = torch.empty(self.max_size, dtype=torch.uint8, device=self.device)
        
        # Register input_buffer
        inp_buf_storage = self.input_buffer.untyped_storage() if hasattr(self.input_buffer, "untyped_storage") else self.input_buffer.storage()
        inp_buf_base = int(inp_buf_storage.data_ptr())
        inp_buf_off = int(self.input_buffer.data_ptr()) - inp_buf_base
        my_inp_buf_h = self._get_mem_handle_bytes(inp_buf_base)
        all_inp_buf = self._gather_object_list_via_broadcast(self.group, (my_inp_buf_h, inp_buf_off))
        self._input_buffer_bases = [None] * self.world_size
        self._input_buffer_ptrs = [0] * 8
        for r in range(self.world_size):
            hb, off = all_inp_buf[r]
            if r == self.rank:
                self._input_buffer_ptrs[r] = int(self.input_buffer.data_ptr())
            else:
                peer_base = int(self._open_mem_handle(bytes(hb)))
                self._input_buffer_bases[r] = peer_base
                self._input_buffer_ptrs[r] = peer_base + off
        for i in range(self.world_size, 8):
            self._input_buffer_ptrs[i] = self._input_buffer_ptrs[0]

        # Rotated input_buffer ptrs for eager kernel
        ws, rk = self.world_size, self.rank
        rotated_input_buf_ptrs = [self._input_buffer_ptrs[(rk + i) % ws] for i in range(8)]
        self._gpu_input_buffer_ptrs_array = torch.tensor(rotated_input_buf_ptrs, dtype=torch.int64, device=self.device)

        # Rotated tmp ptrs (computed once, shared by eager and CUDAGraph)
        rotated_tmp_ptrs = [self._tmp_ptrs[(rk + i) % ws] for i in range(8)]
        self._gpu_tmp_ptrs_array = torch.tensor(rotated_tmp_ptrs, dtype=torch.int64, device=self.device)

        # CUDAGraph state
        self._IS_CAPTURING = False
        self._graph_inp = None
        self._graph_out = None
        # Pre-allocate graph input pointer array with input_buffer addresses (updated at capture end)
        # Initialize with input_buffer ptrs so warmup works; updated to user tensor ptrs after capture
        self._gpu_graph_in_ptrs_array = torch.tensor(rotated_input_buf_ptrs, dtype=torch.int64, device=self.device)
        self._graph_in_bases = []

        self._exe_cache = {}
        self._threads = 512
        self._max_spin = int(os.environ.get("FLYDSL_AITER_SIGNAL_MAX_SPIN", "20000000"))
        self._grid_x_cache = {}

        self._reuse_out_default = str(os.environ.get("FLYDSL_AITER_REUSE_OUT", "0")).strip().lower() in {"1", "true", "yes", "y"}
        self._cached_out = None

    def close(self):
        """Release IPC memory handles for peer GPU buffers."""
        for bases in [self._meta_bases, self._input_buffer_bases, self._graph_in_bases]:
            for b in bases:
                if b is not None:
                    self._close_mem_handle(int(b))
        self._meta_bases = []
        self._input_buffer_bases = []
        self._graph_in_bases = []

    @contextmanager
    def capture(self):
        """Context manager for CUDA graph capture.
        
        Usage:
            with fa.capture():
                with torch.cuda.graph(g):
                    fa.custom_all_reduce(inp, out=out)
            # After exiting capture(), IPC handles are exchanged for recorded tensors
        """
        try:
            self._IS_CAPTURING = True
            self._graph_inp = None
            self._graph_out = None
            yield
        finally:
            self._IS_CAPTURING = False
            if self._graph_inp is not None:
                self._register_graph_tensors()

    def _register_graph_tensors(self):
        """Register input/output tensors for CUDAGraph replay (called at capture end).
        
        Updates _gpu_graph_in_ptrs_array with correct cross-process pointers to user tensors.
        The kernel loads pointers from this array during execution, so replay uses user tensor addresses.
        """
        inp = self._graph_inp
        if inp is None:
            return
        
        # Exchange IPC handles for inp tensor across all ranks
        storage = inp.untyped_storage() if hasattr(inp, "untyped_storage") else inp.storage()
        base_ptr = int(storage.data_ptr())
        off = int(inp.data_ptr()) - base_ptr
        my_handle = self._get_mem_handle_bytes(base_ptr)
        all_graph_in = self._gather_object_list_via_broadcast(self.group, (my_handle, off))
        
        ws, rk = self.world_size, self.rank
        self._graph_in_bases = [None] * self.world_size
        graph_in_ptrs = [0] * 8
        for r in range(self.world_size):
            hb, o = all_graph_in[r]
            if r == self.rank:
                graph_in_ptrs[r] = int(inp.data_ptr())
            else:
                peer_base = int(self._open_mem_handle(bytes(hb)))
                self._graph_in_bases[r] = peer_base
                graph_in_ptrs[r] = peer_base + o
        for i in range(self.world_size, 8):
            graph_in_ptrs[i] = graph_in_ptrs[0]
        
        # Update _gpu_graph_in_ptrs_array with rotated user tensor pointers for replay
        rotated = [graph_in_ptrs[(rk + i) % ws] for i in range(8)]
        self._gpu_graph_in_ptrs_array.copy_(torch.tensor(rotated, dtype=torch.int64, device=self.device))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # Class-level cache for dtype string conversion
    _DTYPE_STR_CACHE = {}

    def _dtype_str(self, t):
        dtype = getattr(t, "dtype", None)
        # Fast path: check cache first
        if dtype in self._DTYPE_STR_CACHE:
            return self._DTYPE_STR_CACHE[dtype]
        return self._dtype_str_from_torch(dtype)

    def _dtype_str_from_torch(self, dtype):
        # Check cache first
        if dtype in self._DTYPE_STR_CACHE:
            return self._DTYPE_STR_CACHE[dtype]
        name = str(dtype)
        if "float16" in name:
            result = "f16"
        elif "float32" in name:
            result = "f32"
        else:
            raise ValueError(f"unsupported dtype: {name}")
        # Cache the result
        self._DTYPE_STR_CACHE[dtype] = result
        return result

    def _compile(self, *, N: int, dtype_str: str):
        import flydsl
        from kernels.aiter_signal_all_reduce_raw import build_aiter_signal_allreduce_raw_module

        key = (N, dtype_str, self.world_size, "flydsl_allreduce")
        exe = self._exe_cache.get(key)
        if exe is not None:
            return exe
        m = build_aiter_signal_allreduce_raw_module(
            N=N,
            dtype_str=dtype_str,
            world_size=self.world_size,
            threads=self._threads,
            meta_size_bytes=self._meta_size,
            max_spin=self._max_spin,
        )
        exe = flydsl.compile(m)
        self._exe_cache[key] = exe
        return exe

    def _run_kernel(
        self,
        N: int,
        dtype_str: str,
        gpu_in_ptrs_array,
        out_ptr: int,
        *,
        stream_ptr: int | None = None,
    ):
        """Launch 2-stage allreduce kernel (unified for eager and CUDAGraph).
        
        Uses run_2stage_arr which loads pointers from device arrays inside the kernel,
        making it compatible with CUDAGraph replay.
        """
        # Cache grid_x for stable shapes (common in training loops).
        try:
            grid_x = self._grid_x_cache[(int(N), str(dtype_str))]
        except Exception:
            pack_elems = 8 if dtype_str == "f16" else 4
            num_packs = int(N) // int(pack_elems)
            part_p = int(num_packs) // int(self.world_size)
            grid_x = int(max(1, min(_AITER_KMAXBLOCKS, (max(1, part_p) + self._threads - 1) // self._threads)))
            self._grid_x_cache[(int(N), str(dtype_str))] = int(grid_x)

        if stream_ptr is None:
            stream_ptr = int(torch.cuda.current_stream().cuda_stream)

        exe = self._compile(N=N, dtype_str=dtype_str)

        # Unified kernel: run_2stage_arr loads pointers from device arrays inside kernel
        # This makes device loads part of kernel execution, captured by CUDAGraph
        exe.run_2stage_arr(
            self.rank, grid_x, self._self_sg,
            int(self._gpu_sg_ptrs_array.data_ptr()),
            int(gpu_in_ptrs_array.data_ptr()),
            int(self._gpu_tmp_ptrs_array.data_ptr()),
            out_ptr, stream_ptr,
        )

    def custom_all_reduce(
        self,
        inp,
        *,
        out=None,
        use_new: bool = True,
        open_fp8_quant: bool = False,
        validate: bool = True,
        stream_ptr: int | None = None,
    ):
        """Unified interface for all_reduce (eager and cudagraph).

        Automatically selects between eager and registered paths based on context.
        """
        _ = use_new
        _ = open_fp8_quant

        if out is None:
            if self._reuse_out_default and (self._cached_out is not None) and self._cached_out.shape == inp.shape and self._cached_out.dtype == inp.dtype and self._cached_out.device == inp.device:
                out = self._cached_out
            else:
                out = torch.empty_like(inp)
                if self._reuse_out_default:
                    self._cached_out = out

        if validate:
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
            if bytes_n > self.max_size:
                raise ValueError(f"input bytes {bytes_n} exceed max_size {self.max_size}")
        else:
            dtype_str = self._dtype_str(inp)
            bytes_n = int(inp.numel()) * int(inp.element_size())
        N = int(out.numel())

        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                # Capture phase: record tensors, use _gpu_graph_in_ptrs_array
                # Kernel loads pointers from device array during execution, captured by CUDAGraph.
                # After capture ends, _register_graph_tensors() updates array with user tensor addresses.
                self._graph_inp = inp
                self._graph_out = out
                self._graph_bytes_n = bytes_n
                self._run_kernel(N, dtype_str, self._gpu_graph_in_ptrs_array, int(out.data_ptr()),
                                 stream_ptr=stream_ptr)
                return out
            else:
                # Warmup inside capture context but before actual capture:
                # Run eager path to populate exe cache
                inp_u8 = inp.view(torch.uint8)
                self.input_buffer[:bytes_n].copy_(inp_u8)
                self._run_kernel(
                    N,
                    dtype_str,
                    self._gpu_input_buffer_ptrs_array,
                    int(self.output_buffer.data_ptr()),
                    stream_ptr=stream_ptr,
                )
                out.view(torch.uint8)[:bytes_n].copy_(self.output_buffer[:bytes_n])
                return out

        # Eager path: memcpy input -> kernel writes directly to out
        inp_u8 = inp.view(torch.uint8)
        self.input_buffer[:bytes_n].copy_(inp_u8)
        self._run_kernel(
            N,
            dtype_str,
            self._gpu_input_buffer_ptrs_array,
            int(out.data_ptr()),
            stream_ptr=stream_ptr,
        )
        return out
