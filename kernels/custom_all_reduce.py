"""Custom all-reduce kernel + Python-facing shim.

Provides FlyDSL-generated allreduce kernels following the AIter signal
protocol for multi-GPU communication on ROCm.
"""

from contextlib import contextmanager
import torch

_AITER_KMAXBLOCKS = 80


def meta_size() -> int:
    """Return meta buffer size (for API compatibility with aiter)."""
    return 0


def _is_weak_contiguous(t) -> bool:
    """Check if tensor occupies a single dense range in storage."""
    try:
        if t.is_contiguous():
            return True
        storage = t.untyped_storage()
        return int(storage.nbytes()) - int(t.storage_offset()) * int(t.element_size()) == int(t.numel()) * int(t.element_size())
    except Exception:
        return False


_FLYDSL_AITER_GLOO_GROUP = None


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
        raise RuntimeError("torch.distributed must be initialized")

    global _FLYDSL_AITER_GLOO_GROUP
    if _FLYDSL_AITER_GLOO_GROUP is None:
        try:
            _FLYDSL_AITER_GLOO_GROUP = dist.new_group(backend="gloo")
        except Exception:
            _FLYDSL_AITER_GLOO_GROUP = dist.group.WORLD

    dev = getattr(rank_data, "device", None) or torch.device(f"cuda:{rank}")
    max_size = int(os.environ.get("FLYDSL_AITER_MAX_SIZE_BYTES", str(64 * 1024 * 1024)))

    if impl == "flydsl":
        return FlyDSLAllreduce(
            group=_FLYDSL_AITER_GLOO_GROUP,
            device=dev,
            max_size=max_size,
            world_size=world_size,
            rank=rank,
            full_nvlink=bool(full_nvlink),
        )

    # impl == "aiter"
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
    return aiter_ar


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

        self.meta = aiter_ops.allocate_meta_buffer(int(aiter_ops.meta_size()) + int(self.max_size))
        try:
            self.meta.zero_()
        except Exception:
            pass
        self._meta_size = int(aiter_ops.meta_size())

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

        self.input_buffer = torch.empty(self.max_size, dtype=torch.uint8, device=self.device)
        self.output_buffer = torch.empty(self.max_size, dtype=torch.uint8, device=self.device)

        inp_buf_base = int(self.input_buffer.untyped_storage().data_ptr())
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

        ws, rk = self.world_size, self.rank
        rotated_input_buf_ptrs = [self._input_buffer_ptrs[(rk + i) % ws] for i in range(8)]
        self._gpu_input_buffer_ptrs_array = torch.tensor(rotated_input_buf_ptrs, dtype=torch.int64, device=self.device)

        rotated_tmp_ptrs = [self._tmp_ptrs[(rk + i) % ws] for i in range(8)]
        self._gpu_tmp_ptrs_array = torch.tensor(rotated_tmp_ptrs, dtype=torch.int64, device=self.device)

        out_buf_base = int(self.output_buffer.untyped_storage().data_ptr())
        out_buf_off = int(self.output_buffer.data_ptr()) - out_buf_base
        my_out_buf_h = self._get_mem_handle_bytes(out_buf_base)
        all_out_buf = self._gather_object_list_via_broadcast(self.group, (my_out_buf_h, out_buf_off))
        self._output_buffer_bases = [None] * self.world_size
        self._output_buffer_ptrs = [0] * 8
        for r in range(self.world_size):
            hb, off = all_out_buf[r]
            if r == self.rank:
                self._output_buffer_ptrs[r] = int(self.output_buffer.data_ptr())
            else:
                peer_base = int(self._open_mem_handle(bytes(hb)))
                self._output_buffer_bases[r] = peer_base
                self._output_buffer_ptrs[r] = peer_base + off
        for i in range(self.world_size, 8):
            self._output_buffer_ptrs[i] = self._output_buffer_ptrs[0]

        self._gpu_output_buffer_ptrs_array = torch.tensor(self._output_buffer_ptrs[:8], dtype=torch.int64, device=self.device)
        self._gpu_tmp_ptrs_nonrotated_array = torch.tensor(self._tmp_ptrs[:8], dtype=torch.int64, device=self.device)

        self._IS_CAPTURING = False
        self._graph_inp = None
        self._graph_out = None
        self._gpu_graph_in_ptrs_array = torch.tensor(rotated_input_buf_ptrs, dtype=torch.int64, device=self.device)
        self._graph_in_bases = []
        self._gpu_graph_out_ptrs_array = torch.tensor(self._output_buffer_ptrs[:8], dtype=torch.int64, device=self.device)
        self._graph_out_bases = []

        self._exe_cache = {}
        self._threads = 512
        self._max_spin = int(os.environ.get("FLYDSL_AITER_SIGNAL_MAX_SPIN", "20000000"))
        self._grid_x_cache = {}

        self._reuse_out_default = str(os.environ.get("FLYDSL_AITER_REUSE_OUT", "0")).strip().lower() in {"1", "true", "yes", "y"}
        self._cached_out = None

    def close(self):
        """Release IPC memory handles for peer GPU buffers."""
        for bases in [self._meta_bases, self._input_buffer_bases, self._output_buffer_bases, self._graph_in_bases, self._graph_out_bases]:
            for b in bases:
                if b is not None:
                    self._close_mem_handle(int(b))
        self._meta_bases = []
        self._input_buffer_bases = []
        self._output_buffer_bases = []
        self._graph_in_bases = []
        self._graph_out_bases = []

    @contextmanager
    def capture(self):
        """Context manager for CUDA graph capture."""
        try:
            self._IS_CAPTURING = True
            self._graph_inp = None
            self._graph_out = None
            yield
        finally:
            self._IS_CAPTURING = False
            if self._graph_inp is not None:
                self._register_graph_tensors()

    @classmethod
    def _get_alloc_base_ptr(cls, dev_ptr: int) -> int:
        """Get the hipMalloc allocation base for a device pointer."""
        import ctypes
        hip = cls._load_hip()
        base = ctypes.c_void_p()
        _RANGE_START_ADDR = 11
        if not hasattr(hip, '_pga_setup'):
            hip.hipPointerGetAttribute.restype = ctypes.c_int
            hip.hipPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            hip._pga_setup = True
        err = hip.hipPointerGetAttribute(
            ctypes.byref(base),
            ctypes.c_int(_RANGE_START_ADDR),
            ctypes.c_void_p(int(dev_ptr)),
        )
        cls._hip_check(err, what="hipPointerGetAttribute(RANGE_START_ADDR)")
        return int(base.value)

    def _register_graph_tensors(self):
        """Exchange IPC handles for captured tensors; update pointer arrays for replay."""
        ws, rk = self.world_size, self.rank

        inp = self._graph_inp
        if inp is not None:
            alloc_base = self._get_alloc_base_ptr(int(inp.data_ptr()))
            off = int(inp.data_ptr()) - alloc_base
            my_handle = self._get_mem_handle_bytes(alloc_base)
            all_graph_in = self._gather_object_list_via_broadcast(self.group, (my_handle, off))

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
            rotated_in = [graph_in_ptrs[(rk + i) % ws] for i in range(8)]
            self._gpu_graph_in_ptrs_array.copy_(torch.tensor(rotated_in, dtype=torch.int64, device=self.device))

        out = self._graph_out
        if out is not None:
            alloc_base = self._get_alloc_base_ptr(int(out.data_ptr()))
            off = int(out.data_ptr()) - alloc_base
            my_handle = self._get_mem_handle_bytes(alloc_base)
            all_graph_out = self._gather_object_list_via_broadcast(self.group, (my_handle, off))

            self._graph_out_bases = [None] * self.world_size
            graph_out_ptrs = [0] * 8
            for r in range(self.world_size):
                hb, o = all_graph_out[r]
                if r == self.rank:
                    graph_out_ptrs[r] = int(out.data_ptr())
                else:
                    peer_base = int(self._open_mem_handle(bytes(hb)))
                    self._graph_out_bases[r] = peer_base
                    graph_out_ptrs[r] = peer_base + o
            for i in range(self.world_size, 8):
                graph_out_ptrs[i] = graph_out_ptrs[0]
            self._gpu_graph_out_ptrs_array.copy_(torch.tensor(graph_out_ptrs[:8], dtype=torch.int64, device=self.device))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    _SUPPORTED_WORLD_SIZES = {2, 4, 8}
    _SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}

    def should_custom_ar(self, inp, *, open_fp8_quant: bool = False) -> bool:
        """Check whether the custom allreduce kernel can handle this input.

        Returns False (caller should fall back to NCCL) when any of these
        conditions is violated:
          1. world_size ∈ {2, 4, 8}
          2. inp byte-size is a multiple of 16
          3. dtype ∈ {float32, float16, bfloat16}
          4. inp byte-size ≤ max_size / 2  (2-stage write-mode uses 2× tmp)
          5. fp8 quantisation is not requested
          6. full_nvlink (fully_connected) is True, or world_size == 2
        """
        from flydsl.utils import log

        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            log().error("custom allreduce unsupported: world_size=%d, "
                        "expected one of %s", self.world_size,
                        sorted(self._SUPPORTED_WORLD_SIZES))
            return False

        inp_size = int(inp.numel()) * int(inp.element_size())
        if inp_size % 16 != 0:
            log().error("custom allreduce unsupported: inp_size=%d "
                        "is not a multiple of 16", inp_size)
            return False

        if inp.dtype not in self._SUPPORTED_DTYPES:
            log().error("custom allreduce unsupported: dtype=%s, "
                        "expected one of {%s}", inp.dtype,
                        ", ".join(str(d) for d in sorted(self._SUPPORTED_DTYPES, key=str)))
            return False

        if inp_size > self.max_size // 2:
            log().error("custom allreduce unsupported: inp_size=%d "
                        "exceeds max_size/2=%d", inp_size, self.max_size // 2)
            return False

        if open_fp8_quant:
            log().error("custom allreduce unsupported: fp8 quantisation "
                        "is not supported")
            return False

        if self.world_size > 2 and not self.full_nvlink:
            log().error("custom allreduce unsupported: fully_connected=false "
                        "is not supported for world_size=%d", self.world_size)
            return False

        return True

    _DTYPE_STR_CACHE = {}

    def _dtype_str(self, t) -> str:
        dtype = getattr(t, "dtype", None)
        if dtype in self._DTYPE_STR_CACHE:
            return self._DTYPE_STR_CACHE[dtype]
        name = str(dtype)
        if "bfloat16" in name:
            result = "bf16"
        elif "float16" in name:
            result = "f16"
        elif "float32" in name:
            result = "f32"
        else:
            raise ValueError(f"unsupported dtype: {name}")
        self._DTYPE_STR_CACHE[dtype] = result
        return result

    def _compile(self, *, N: int, dtype_str: str):
        from flydsl.compiler.jit_function import MlirCompiler
        from flydsl.compiler.executor import ExecutionEngineExecutor
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
        )
        with m.context:
            lowered = MlirCompiler.compile(m)
        exe = ExecutionEngineExecutor(lowered)
        self._exe_cache[key] = exe
        return exe

    def _run_kernel(
        self,
        N: int,
        dtype_str: str,
        *,
        gpu_in_ptrs_array=None,
        gpu_out_ptrs_array=None,
        inp_ptr: int = 0,
        out_ptr: int = 0,
        use_write_mode: bool = False,
        stream_ptr: int | None = None,
    ):
        """Launch allreduce kernel (auto-selects 1-stage or 2-stage by data size)."""
        from flydsl.expr.typing import Int32, Int64, Stream

        # Auto-select stage by data size (match aiter thresholds):
        #   world_size == 2              → always 1-stage
        #   world_size <= 4, bytes < 160KB → 1-stage
        #   world_size <= 8, bytes < 80KB  → 1-stage
        #   otherwise                      → 2-stage
        elem_bytes = 2 if dtype_str in ("f16", "bf16") else 4
        bytes_n = N * elem_bytes
        if self.world_size == 2:
            _stage = "1"
        elif (self.world_size <= 4 and bytes_n < 160 * 1024) or bytes_n < 80 * 1024:
            _stage = "1"
        else:
            _stage = "2"

        try:
            grid_x = self._grid_x_cache[(int(N), str(dtype_str), _stage)]
        except Exception:
            pack_elems = 8 if dtype_str in ("f16", "bf16") else 4
            num_packs = int(N) // int(pack_elems)
            if _stage == "1":
                # 1-stage: tnum_gpu threads per warp handle one pack each (match aiter)
                tnum_gpu = self._threads // self.world_size
                grid_x = int(max(1, min(_AITER_KMAXBLOCKS, (num_packs + tnum_gpu - 1) // tnum_gpu)))
            else:
                part_p = int(num_packs) // int(self.world_size)
                tnum_gpu = self._threads // self.world_size
                grid_x = int(max(1, min(_AITER_KMAXBLOCKS, (max(1, part_p) + tnum_gpu - 1) // tnum_gpu)))
            self._grid_x_cache[(int(N), str(dtype_str), _stage)] = int(grid_x)

        if stream_ptr is None:
            stream_ptr = int(torch.cuda.current_stream().cuda_stream)

        exe = self._compile(N=N, dtype_str=dtype_str)

        if _stage == "1" and not use_write_mode:
            fns["run_1stage_arr"](
                Int32(self.rank),
                Int32(grid_x),
                Int64(self._self_sg),
                Int64(int(self._gpu_sg_ptrs_array.data_ptr())),
                Int64(int(gpu_in_ptrs_array.data_ptr())),
                Int64(int(out_ptr)),
                stream=stream_obj,
            )
        elif use_write_mode:
            fns["run_2stage_write_mode"](
                Int32(self.rank),
                Int32(grid_x),
                Int64(self._self_sg),
                Int64(int(self._gpu_sg_ptrs_array.data_ptr())),
                Int64(int(inp_ptr)),
                Int64(int(gpu_out_ptrs_array.data_ptr())),
                Int64(int(self._gpu_tmp_ptrs_nonrotated_array.data_ptr())),
                stream=stream_obj,
            )
        else:
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
        """Unified all-reduce (eager and cudagraph).

        Returns None when the input is not supported by the custom kernel
        (caller should fall back to NCCL).
        Selects write_mode kernel when N > 512*4096 and world_size == 8.
        """
        if not self.should_custom_ar(inp, open_fp8_quant=open_fp8_quant):
            return None

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

        use_write_mode = (bytes_n > 512 * 4096 * 2 and self.world_size == 8)

        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                self._graph_inp = inp
                self._graph_out = out
                self._graph_bytes_n = bytes_n

                if use_write_mode:
                    self._graph_use_write_mode = True
                    self._run_kernel(
                        N, dtype_str,
                        gpu_out_ptrs_array=self._gpu_graph_out_ptrs_array,
                        inp_ptr=int(inp.data_ptr()),
                        use_write_mode=True,
                        stream_ptr=stream_ptr,
                    )
                else:
                    self._graph_use_write_mode = False
                    self._run_kernel(
                        N, dtype_str,
                        gpu_in_ptrs_array=self._gpu_graph_in_ptrs_array,
                        out_ptr=int(out.data_ptr()),
                        use_write_mode=False,
                        stream_ptr=stream_ptr,
                    )
                return out
            else:
                if use_write_mode:
                    self._run_kernel(
                        N, dtype_str,
                        gpu_out_ptrs_array=self._gpu_output_buffer_ptrs_array,
                        inp_ptr=int(inp.data_ptr()),
                        use_write_mode=True,
                        stream_ptr=stream_ptr,
                    )
                    out.view(torch.uint8)[:bytes_n].copy_(self.output_buffer[:bytes_n])
                else:
                    self.input_buffer[:bytes_n].copy_(inp.view(torch.uint8))
                    self._run_kernel(
                        N, dtype_str,
                        gpu_in_ptrs_array=self._gpu_input_buffer_ptrs_array,
                        out_ptr=int(self.output_buffer.data_ptr()),
                        use_write_mode=False,
                        stream_ptr=stream_ptr,
                    )
                    out.view(torch.uint8)[:bytes_n].copy_(self.output_buffer[:bytes_n])
                return out

        if use_write_mode:
            self._run_kernel(
                N, dtype_str,
                gpu_out_ptrs_array=self._gpu_output_buffer_ptrs_array,
                inp_ptr=int(inp.data_ptr()),
                use_write_mode=True,
                stream_ptr=stream_ptr,
            )
            # Host-side barrier: ensures all remote XGMI writes to local
            # output_buffer are complete before the copy.
            torch.cuda.current_stream().synchronize()
            import torch.distributed as dist
            dist.barrier(group=self.group)
            out.view(torch.uint8)[:bytes_n].copy_(self.output_buffer[:bytes_n])
        else:
            self.input_buffer[:bytes_n].copy_(inp.view(torch.uint8))
            self._run_kernel(
                N, dtype_str,
                gpu_in_ptrs_array=self._gpu_input_buffer_ptrs_array,
                out_ptr=int(out.data_ptr()),
                use_write_mode=False,
                stream_ptr=stream_ptr,
            )
        return out

    def all_reduce_reg(self, inp, out, open_fp8_quant: bool = False):
        if isinstance(inp, (list, tuple)):
            import functools
            result = functools.reduce(torch.add, inp)
            out.copy_(result)
            return out
        return self.custom_all_reduce(inp, out=out, open_fp8_quant=open_fp8_quant)

    def all_gather_reg(self, inp, out):
        if isinstance(inp, (list, tuple)):
            stacked = torch.stack(list(inp), dim=0)
            out.copy_(stacked)
        elif self.world_size == 1:
            out.copy_(inp)
        else:
            import torch.distributed as dist
            dist.all_gather_into_tensor(out, inp, group=self.group)
        return out
