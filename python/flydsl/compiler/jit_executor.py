import ctypes
import threading
from functools import lru_cache
from pathlib import Path
from typing import List

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from .protocol import fly_pointers


@lru_cache(maxsize=1)
def _resolve_runtime_libs() -> List[str]:
    mlir_libs_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    return [str(mlir_libs_dir / "libfly_jit_runtime.so")]


class _ArgPacker:
    """Thread-local buffer for packing C pointer arguments."""

    def __init__(self):
        self._tls = threading.local()

    def pack(self, ptrs: List[ctypes.c_void_p]):
        size = len(ptrs)
        buf = getattr(self._tls, "packed_args", None)
        capacity = getattr(self._tls, "capacity", 0)
        if buf is None or capacity < size:
            buf = (ctypes.c_void_p * size)()
            self._tls.packed_args = buf
            self._tls.capacity = size
        for i, ptr in enumerate(ptrs):
            buf[i] = ptr
        return buf


class _HipKernelFunc:
    """Launch a GPU kernel via HIP API, bypassing LLVM ExecutionEngine."""

    def __init__(self, hip, func_ptr, shared_mem, block_x, entry_name,
                 n_kernel_args=0, kernel_arg_kinds=None, tile_n=128):
        self._hip = hip
        self._func = func_ptr
        self._shared_mem = shared_mem
        self._block_x = block_x
        self._entry = entry_name
        self._n_kernel_args = n_kernel_args
        self._kernel_arg_kinds = kernel_arg_kinds or []
        self._tile_n = max(1, int(tile_n))

    def __call__(self, *jit_args):
        """Called with JitArguments: TensorAdaptor/Int32/Stream objects.

        The host launcher has (N_ptr + M_int + stream_i32 + stream_ptr) args.
        The GPU kernel only has (N_ptr + M_int) args — no stream.
        We extract all non-stream args, then truncate to n_kernel_args.
        """
        import torch
        from ..expr.typing import Stream as StreamType

        ptr_vals = []
        i32_vals_src = []
        for arg in jit_args:
            if isinstance(arg, StreamType):
                continue
            if hasattr(arg, '_tensor_keepalive'):
                t = arg._tensor_keepalive
                ptr_vals.append(ctypes.c_void_p(t.data_ptr()))
            elif hasattr(arg, 'value') and isinstance(arg.value, int):
                i32_vals_src.append(int(arg.value))
            else:
                if hasattr(arg, 'data_ptr'):
                    ptr_vals.append(ctypes.c_void_p(arg.data_ptr()))
                else:
                    ptr_vals.append(ctypes.c_void_p(0))

        # Build kernel args by LLVM signature kinds (ptr/i32), not by heuristics.
        # This safely drops host-side extras like stream-as-int32.
        c_args = []
        i32_kernel = []
        p_idx = 0
        i_idx = 0
        if self._kernel_arg_kinds:
            for kind in self._kernel_arg_kinds:
                if kind == "ptr":
                    if p_idx < len(ptr_vals):
                        c_args.append(ptr_vals[p_idx])
                        p_idx += 1
                    else:
                        c_args.append(ctypes.c_void_p(0))
                elif kind == "i32":
                    if i_idx < len(i32_vals_src):
                        v = i32_vals_src[i_idx]
                        i_idx += 1
                    else:
                        v = 0
                    i32_kernel.append(v)
                    c_args.append(ctypes.c_int32(v))
        else:
            # Backward-compatible fallback
            c_args.extend(ptr_vals)
            for v in i32_vals_src:
                i32_kernel.append(v)
                c_args.append(ctypes.c_int32(v))

        if self._n_kernel_args > 0:
            c_args = c_args[:self._n_kernel_args]

        n = len(c_args)
        if n == 0:
            return

        arg_ptrs = (ctypes.c_void_p * n)()
        for i, a in enumerate(c_args):
            arg_ptrs[i] = ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)

        stream = torch.cuda.current_stream().cuda_stream

        # Grid from kernel-bound i32 args: [tokens_in, n_or_inter, k_in, size_expert_ids]
        gy = i32_kernel[3] if len(i32_kernel) >= 4 else 1
        gx = (i32_kernel[1] // self._tile_n) if len(i32_kernel) >= 2 else 1
        gx = max(1, int(gx))
        gy = max(1, int(gy))

        status = self._hip.hipModuleLaunchKernel(
            self._func,
            ctypes.c_uint(gx), ctypes.c_uint(gy), ctypes.c_uint(1),
            ctypes.c_uint(self._block_x), ctypes.c_uint(1), ctypes.c_uint(1),
            ctypes.c_uint(self._shared_mem),
            ctypes.c_void_p(stream),
            arg_ptrs,
            ctypes.c_void_p(0),
        )
        if status != 0:
            raise RuntimeError(f"hipModuleLaunchKernel failed: {status}")


class CompiledArtifact:
    def __init__(
        self,
        compiled_module: ir.Module,
        func_name: str,
        source_ir: str = None,
    ):
        self._ir_text = str(compiled_module)
        self._entry = func_name
        self._source_ir = source_ir
        self._module = None
        self._engine = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def __getstate__(self):
        return {
            "ir_text": self._ir_text,
            "entry": self._entry,
            "source_ir": self._source_ir,
        }

    def __setstate__(self, state):
        self._ir_text = state["ir_text"]
        self._entry = state["entry"]
        self._source_ir = state["source_ir"]
        self._module = None
        self._engine = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None or self._hip_func is not None:
                return

            # Try HIP-based launcher first (avoids LLVM symbol conflicts)
            hip_func = self._try_hip_launcher()
            if hip_func is not None:
                self._hip_func = hip_func
                return

            with ir.Context() as ctx:
                ctx.load_all_available_dialects()
                self._module = ir.Module.parse(self._ir_text)
                self._engine = ExecutionEngine(
                    self._module,
                    opt_level=3,
                    shared_libs=_resolve_runtime_libs(),
                )
                self._engine.initialize()

    def _try_hip_launcher(self):
        """Extract GPU binary from compiled MLIR and create HIP launcher."""
        import re
        bin_match = re.search(r'bin\s*=\s*"((?:[^"\\]|\\.)*)"', self._ir_text, re.DOTALL)
        if not bin_match:
            return None

        # Parse GPU kernel name from compiled MLIR
        kern_match = re.search(r'kernel_metadata<"(\w+)"', self._ir_text)
        if not kern_match:
            kern_match = re.search(r'gpu\.func\s+@(\w+)', self._ir_text)
        if not kern_match:
            return None
        kernel_name = kern_match.group(1)

        # Parse grid computation from the launcher
        # gx = n_in / tile_n (extracted from MLIR)
        # gy = size_expert_ids
        # These are computed from i32 arguments at runtime

        # Extract ELF binary
        escaped = bin_match.group(1)
        raw = bytearray()
        i = 0
        while i < len(escaped):
            if escaped[i] == '\\' and i + 1 < len(escaped):
                i += 1; c = escaped[i]
                if c == '\\': raw.append(0x5C)
                elif c == '"': raw.append(0x22)
                elif c == 'n': raw.append(0x0A)
                elif c == 't': raw.append(0x09)
                else:
                    try: raw.append(int(escaped[i:i+2], 16)); i += 1
                    except ValueError: raw.append(ord(c))
            else:
                raw.append(ord(escaped[i]))
            i += 1

        binary = bytes(raw)
        if len(binary) < 100:
            return None

        # Load via HIP
        hip = ctypes.CDLL("libamdhip64.so")
        module = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(binary)
        status = hip.hipModuleLoadData(ctypes.byref(module), buf)
        if status != 0:
            return None

        func_ptr = ctypes.c_void_p()
        status = hip.hipModuleGetFunction(
            ctypes.byref(func_ptr), module, kernel_name.encode())
        if status != 0:
            return None

        # Parse shared memory size from MLIR metadata
        smem_match = re.search(r'group_segment_fixed_size\s*=\s*(\d+)', self._ir_text)
        shared_mem = int(smem_match.group(1)) if smem_match else 8192

        # Parse grid dims from launch_func in MLIR
        # gpu.launch_func ... blocks in (%gx, %gy, 1) threads in (256, 1, 1)
        grid_match = re.search(
            r'gpu\.launch_func.*blocks\s+in\s*\(([^)]+)\).*threads\s+in\s*\((\d+)',
            self._ir_text)
        block_x = int(grid_match.group(2)) if grid_match else 256

        # Parse GPU kernel arg count from MLIR (kernel func signature)
        # The kernel has N ptr args + M i32 args (no stream)
        import re as _re
        kern_sig = _re.search(
            r'llvm\.func\s+@' + kernel_name + r'\(([^)]*)\)',
            self._ir_text)
        n_kernel_args = 0
        kernel_arg_kinds = []
        if kern_sig:
            args_str = kern_sig.group(1).strip()
            if args_str:
                parsed_args = [a.strip() for a in args_str.split(',') if a.strip()]
                n_kernel_args = len(parsed_args)
                for ty in parsed_args:
                    ty_l = ty.lower()
                    if "ptr" in ty_l:
                        kernel_arg_kinds.append("ptr")
                    elif "i32" in ty_l:
                        kernel_arg_kinds.append("i32")
                    else:
                        # Keep conservative default for unknown scalar kinds.
                        kernel_arg_kinds.append("ptr")

        # Best-effort tile_n infer from division-by-constant in launch IR.
        tile_n = 128
        div_consts = re.findall(
            r'arith\.(?:divsi|divui|floordivsi|ceildivsi)\s+%\w+\s*,\s*%c(\d+)',
            self._ir_text,
        )
        if div_consts:
            try:
                tile_n = int(div_consts[-1])
            except ValueError:
                tile_n = 128

        return _HipKernelFunc(hip, func_ptr, shared_mem, block_x, self._entry,
                              n_kernel_args=n_kernel_args,
                              kernel_arg_kinds=kernel_arg_kinds,
                              tile_n=tile_n)

    def __call__(self, *args, **kwargs):
        if self._engine is None and not hasattr(self, '_hip_func'):
            self._hip_func = None
        if self._engine is None and (not hasattr(self, '_hip_func') or self._hip_func is None):
            self._ensure_engine()

        if hasattr(self, '_hip_func') and self._hip_func is not None:
            return self._hip_func(*args)

        all_c_ptrs: List[ctypes.c_void_p] = []
        for arg in args:
            all_c_ptrs.extend(fly_pointers(arg))

        func_ptr = self._engine.raw_lookup(self._entry)
        func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)

        packed_args = self._packer.pack(all_c_ptrs)

        return func_exe(packed_args)

    def dump(self, compiled: bool = True):
        if compiled:
            print("=" * 60)
            print("Compiled MLIR IR:")
            print("=" * 60)
            print(self._ir_text)
        else:
            if self._source_ir is None:
                print("Original IR not available")
            else:
                print("=" * 60)
                print("Original MLIR IR:")
                print("=" * 60)
                print(self._source_ir)

    @property
    def ir(self) -> str:
        return self._ir_text

    @property
    def source_ir(self) -> str:
        return self._source_ir
