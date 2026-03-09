"""
FlyDSL shmem kernel compilation helper.

``compile_shmem_kernel()`` compiles a ``@flyc.kernel`` function to a HIP
module with the mori shmem bitcode linked in, returning a ``ShmemKernel``
callable that dispatches via ctypes.

Pipeline
--------
``@flyc.kernel`` Python body
  → FlyDSL MLIR passes (up to ``reconcile-unrealized-casts``)
  → extract GPU module LLVM IR
  → ``mlir-translate --mlir-to-llvmir``
  → ``llvm-link`` + ``libmori_shmem_device.bc``
  → ``ROCm clang -target amdgcn-amd-amdhsa`` → ``.hsaco``
  → ``hipModuleLoad`` + ``shmem_module_init`` (via install_hook)
  → ``ShmemKernel`` callable
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Post-compile hook (set by mori.ir.flydsl.runtime.install_hook)
_shmem_post_compile_hook: Optional[Any] = None

# ---------------------------------------------------------------------------
# Tool paths
# ---------------------------------------------------------------------------
_ROCM_PATH    = os.environ.get("ROCM_PATH", "/opt/rocm")
_LLVM_LINK    = os.environ.get(
    "LLVM_LINK", os.path.join(_ROCM_PATH, "lib/llvm/bin/llvm-link"))
_ROCM_CLANG   = os.environ.get(
    "ROCM_CLANG", os.path.join(_ROCM_PATH, "llvm/bin/clang"))
_MLIR_TRANSLATE = (
    os.environ.get("MLIR_TRANSLATE")
    or shutil.which("mlir-translate-20")
    or shutil.which("mlir-translate")
    or "/mnt/data/xiaobing/llvm-project/buildmlir/bin/mlir-translate"
)

_HIP = ctypes.CDLL("libamdhip64.so")


# ---------------------------------------------------------------------------
# HIP helpers
# ---------------------------------------------------------------------------
def _hip_check(err: int, msg: str = "") -> None:
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")


def _hip_module_load(path: str) -> ctypes.c_void_p:
    mod = ctypes.c_void_p()
    _hip_check(_HIP.hipModuleLoad(ctypes.byref(mod), path.encode()),
               f"hipModuleLoad({path})")
    return mod


def _hip_get_function(mod: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
    func = ctypes.c_void_p()
    _hip_check(_HIP.hipModuleGetFunction(ctypes.byref(func), mod, name.encode()), name)
    return func


# ---------------------------------------------------------------------------
# MLIR → LLVM IR extraction
# ---------------------------------------------------------------------------
def _extract_gpu_module_mlir(mlir_asm: str, ctx) -> str:
    """Extract GPU kernel functions from MLIR after lowering passes.

    After ``reconcile-unrealized-casts``, the host module contains::

        module attributes {gpu.container_module} {
          gpu.module @kernels [#rocdl.target<chip="gfx942">] attributes {...} {
            llvm.func @ep_dispatch_intranode_0(...) attributes {gpu.kernel} {
              ...
            }
          }
          llvm.func @_shmem_stub(...) { gpu.launch_func ... }
        }

    We extract the ``gpu.module`` body, wrap in a standalone MLIR module,
    and pass to mlir-translate to get the GPU LLVM IR.
    """
    from .._mlir import ir as _ir

    full_module = _ir.Module.parse(mlir_asm, context=ctx)

    gpu_mod_op = None
    for op in full_module.body.operations:
        if op.name == "gpu.module":
            gpu_mod_op = op
            break

    if gpu_mod_op is None:
        raise RuntimeError(
            "No gpu.module found in MLIR after lowering. "
            "Ensure the kernel was compiled via @flyc.kernel."
        )

    # Collect data layout
    data_layout = ""
    try:
        dl = str(gpu_mod_op.operation.attributes["llvm.data_layout"])
        data_layout = dl.strip('"')
    except KeyError:
        pass

    # Collect GPU function ASMs
    func_asms = [
        op.operation.get_asm(enable_debug_info=False)
        for op in gpu_mod_op.regions[0].blocks[0].operations
    ]

    if not func_asms:
        raise RuntimeError("GPU module is empty — no kernel functions found.")

    dl_attr = f'llvm.data_layout = "{data_layout}"' if data_layout else ""
    attrs   = f"attributes {{{dl_attr}}}" if dl_attr else ""
    return f"module {attrs} {{\n" + "\n".join(func_asms) + "\n}\n"


def _mlir_to_llvm_ir(mlir_text: str) -> str:
    """Run ``mlir-translate --mlir-to-llvmir`` and return LLVM IR text."""
    result = subprocess.run(
        [_MLIR_TRANSLATE, "--mlir-to-llvmir"],
        input=mlir_text, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mlir-translate failed:\n"
            f"stderr: {result.stderr[:800]}"
        )
    return result.stdout


def _link_and_compile_hsaco(llvm_ir: str, chip: str, out_path: str, shmem_bc: str) -> str:
    """Link LLVM IR with shmem bitcode and compile to .hsaco."""
    tmpdir = tempfile.mkdtemp(prefix="flydsl_shmem_")
    try:
        kernel_ll = os.path.join(tmpdir, "kernel.ll")
        with open(kernel_ll, "w") as f:
            f.write(llvm_ir)

        linked_bc = os.path.join(tmpdir, "linked.bc")
        r = subprocess.run(
            [_LLVM_LINK, kernel_ll, shmem_bc, "-o", linked_bc],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"llvm-link failed:\n{r.stderr[:1000]}")

        r = subprocess.run(
            [_ROCM_CLANG, "-x", "ir", linked_bc,
             "-target", "amdgcn-amd-amdhsa",
             f"-mcpu={chip}", "-O3", "-o", out_path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ROCm clang failed:\n{r.stderr[:1000]}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path


# ---------------------------------------------------------------------------
# ShmemKernel callable
# ---------------------------------------------------------------------------
class ShmemKernel:
    """Compiled shmem kernel — thread-safe lazy HIP module loading."""

    def __init__(self, hsaco_path: str, kernel_name: str):
        self._hsaco_path  = hsaco_path
        self._kernel_name = kernel_name
        self._mod:  Optional[ctypes.c_void_p] = None
        self._func: Optional[ctypes.c_void_p] = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._func is not None:
                return
            mod  = _hip_module_load(self._hsaco_path)
            func = _hip_get_function(mod, self._kernel_name)
            if _shmem_post_compile_hook is not None:
                _shmem_post_compile_hook(mod.value)
            self._mod  = mod
            self._func = func

    def launch(
        self,
        *,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args: Sequence[Any],
        shared_mem: int = 0,
        stream: Optional[int] = None,
    ) -> None:
        """Launch the GPU kernel via HIP."""
        self._ensure_loaded()
        ptrs = (ctypes.c_void_p * len(args))()
        for i, a in enumerate(args):
            ptrs[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
        sp = ctypes.c_void_p(stream) if stream else ctypes.c_void_p(0)
        gx, gy, gz = grid
        bx, by, bz = block
        _hip_check(
            _HIP.hipModuleLaunchKernel(
                self._func, gx, gy, gz, bx, by, bz, shared_mem, sp, ptrs, None,
            ),
            "hipModuleLaunchKernel",
        )

    @property
    def hsaco_path(self) -> str:
        return self._hsaco_path

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    def __repr__(self) -> str:
        return f"ShmemKernel(kernel={self._kernel_name!r}, hsaco={self._hsaco_path!r})"


# ---------------------------------------------------------------------------
# compile_shmem_kernel — public API
# ---------------------------------------------------------------------------
def compile_shmem_kernel(
    kernel_fn,
    dummy_args: Sequence[Any],
    const_args: Optional[Dict[str, Any]] = None,
    chip: Optional[str] = None,
    shmem_bc: Optional[str] = None,
    out_path: Optional[str] = None,
) -> ShmemKernel:
    """Compile a ``@flyc.kernel`` function with mori shmem bitcode.

    Parameters
    ----------
    kernel_fn:
        ``KernelFunction`` produced by ``@flyc.kernel``.
    dummy_args:
        Example non-constexpr arguments with the correct types/shapes.
        Supports ``torch.Tensor`` and ``fx.Int32(value)`` / similar DSL types.
    const_args:
        Compile-time constant arguments
        (e.g. ``{"rank": 0, "npes": 8, "hidden_dim": 7168}``).
    chip:
        AMD GPU target.  Auto-detected from ``rocm_agent_enumerator``.
    shmem_bc:
        Path to ``libmori_shmem_device.bc``.  Auto-detected via mori.
    out_path:
        Destination ``.hsaco``.  A temp file is used if not given.

    Returns
    -------
    ShmemKernel
        Callable that lazily loads the HSACO and dispatches via ctypes.
    """
    import torch

    from .._mlir import ir
    from .._mlir._mlir_libs import _mlirRegisterEverything as _reg
    from .._mlir.dialects import func, gpu
    from .._mlir.passmanager import PassManager
    from ..runtime.device import get_rocm_arch
    from .kernel_function import (
        CompilationContext,
        create_gpu_module,
        get_gpu_module_body,
        KernelLauncher,
    )
    from .protocol import fly_types, fly_construct
    from ..expr.typing import Stream
    from .jit_argument import JitArgumentRegistry

    if chip is None:
        chip = get_rocm_arch()
    if shmem_bc is None:
        from mori.ir.bitcode import find_bitcode
        shmem_bc = find_bitcode()
    if const_args is None:
        const_args = {}

    if out_path is None:
        fn_name = getattr(
            getattr(kernel_fn, "_func", None), "__name__", "shmem_kernel")
        out_path = os.path.join(
            tempfile.gettempdir(), f"{fn_name}_{chip}.hsaco")

    # -----------------------------------------------------------------------
    # Convert dummy_args to (JitArgument, DslType) pairs for the MLIR pipeline
    # -----------------------------------------------------------------------
    jit_args_list  = []   # JitArgument objects (TensorAdaptor, Int32, ...)
    dsl_types_list = []   # DslType classes (Tensor, Int32_type, ...)

    for arg in dummy_args:
        if hasattr(arg, "__fly_types__") and hasattr(arg, "__fly_construct__"):
            # Already a proper DslType+JitArgument (e.g. fx.Int32(42))
            jit_args_list.append(arg)
            dsl_types_list.append(type(arg))
        elif hasattr(arg, "__fly_types__"):
            # JitArgument without __fly_construct__: get dsl_type from registry
            dsl_type = JitArgumentRegistry.get_dsl_type(type(arg))
            jit_args_list.append(arg)
            dsl_types_list.append(dsl_type or type(arg))
        elif isinstance(arg, torch.Tensor):
            ctor, dsl_type = JitArgumentRegistry.get(torch.Tensor)
            if ctor is None:
                raise TypeError("torch.Tensor not registered in JitArgumentRegistry")
            jit_args_list.append(ctor(arg))
            dsl_types_list.append(dsl_type)
        else:
            raise TypeError(
                f"compile_shmem_kernel: cannot handle dummy arg of type "
                f"{type(arg).__name__}.  Pass torch.Tensor or fx.Int32(val)."
            )

    # -----------------------------------------------------------------------
    # 1. Build MLIR module: GPU module + host stub
    # -----------------------------------------------------------------------
    reg_all = ir.DialectRegistry()
    _reg.register_dialects(reg_all)

    with ir.Context() as ctx:
        ctx.append_dialect_registry(reg_all)
        ctx.load_all_available_dialects()

        with ir.Location.unknown():
            module = ir.Module.create()
            module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

            # Stream appended for host function signature
            all_jit = jit_args_list + [Stream(None)]
            ir_types = fly_types(all_jit)

            with ir.InsertionPoint(module.body):
                # GPU module
                gpu_module = create_gpu_module(
                    "kernels",
                    targets=[f'#rocdl.target<chip = "{chip}">'],
                )

                # Minimal host function
                host_func = func.FuncOp("_shmem_stub", (ir_types, []))
                host_func.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                host_entry = host_func.add_entry_block()

                with CompilationContext.create() as comp_ctx:
                    comp_ctx.gpu_module_op   = gpu_module
                    comp_ctx.gpu_module_body = get_gpu_module_body(gpu_module)

                    with ir.InsertionPoint(host_entry):
                        host_block_args = list(
                            host_func.regions[0].blocks[0].arguments)
                        stream_arg = host_block_args[-1]
                        comp_ctx.stream_arg = stream_arg

                        # Construct DSL wrapper objects from block args
                        dsl_objs = []
                        idx = 0
                        for jit_arg, dsl_type in zip(jit_args_list, dsl_types_list):
                            n = len(fly_types(jit_arg))
                            dsl_obj = fly_construct(
                                dsl_type, jit_arg,
                                host_block_args[idx : idx + n],
                            )
                            dsl_objs.append(dsl_obj)
                            idx += n

                        # Build args dict in the kernel's exact parameter order.
                        # Constexpr params → plain Python value from const_args.
                        # Non-constexpr params → DSL wrapper from dsl_objs.
                        # Then call _emit_kernel with no positional args (avoids
                        # "multiple values" errors when const params are mixed in).
                        import inspect as _inspect2
                        import typing as _typing
                        from ..expr.typing import Constexpr as _CxType
                        sig2 = _inspect2.signature(kernel_fn._func)

                        def _is_cx(ann) -> bool:
                            """Check if annotation indicates a Constexpr param.

                            ASTRewriter stores annotations as *strings* (forward
                            references), so we handle both string and type forms.
                            """
                            if ann is _inspect2.Parameter.empty:
                                return False
                            # String annotation (ASTRewriter path)
                            if isinstance(ann, str):
                                return "Constexpr" in ann
                            # Actual type annotation path
                            if ann is _CxType:
                                return True
                            return _typing.get_origin(ann) is _CxType

                        ordered_kw: Dict[str, Any] = {}
                        nc_idx = 0
                        for pname, param in sig2.parameters.items():
                            ann = param.annotation
                            if ann is not _inspect2.Parameter.empty and _is_cx(ann):
                                if pname not in const_args:
                                    raise ValueError(
                                        f"Constexpr param '{pname}' not found in const_args. "
                                        f"Available: {list(const_args.keys())}"
                                    )
                                ordered_kw[pname] = const_args[pname]
                            else:
                                if nc_idx >= len(dsl_objs):
                                    raise ValueError(
                                        f"Not enough dummy_args: needed arg #{nc_idx} "
                                        f"for param '{pname}'."
                                    )
                                ordered_kw[pname] = dsl_objs[nc_idx]
                                nc_idx += 1

                        launcher_vals = kernel_fn._emit_kernel(
                            comp_ctx,
                            (),           # no positional args (all via kwargs)
                            ordered_kw,   # all params in correct order
                        )

                        # Emit gpu.launch_func (required for pass validation)
                        kl = KernelLauncher(
                            kernel_fn._kernel_name,
                            launcher_vals,
                        )
                        kl.launch(
                            grid=(1, 1, 1),
                            block=(64, 1, 1),
                            stream=stream_arg,
                        )
                        func.ReturnOp([])

            # ---------------------------------------------------------------
            # 2. Run FlyDSL passes (all EXCEPT gpu-module-to-binary)
            # ---------------------------------------------------------------
            pre_binary = [
                "gpu-kernel-outlining{data-layout-str=}",
                "fly-canonicalize",
                "fly-layout-lowering",
                "convert-fly-to-rocdl",
                "canonicalize",
                (f"gpu.module(convert-scf-to-cf,cse,"
                 f"convert-gpu-to-rocdl{{chipset={chip} index-bitwidth=0 "
                 f"runtime=HIP use-bare-ptr-memref-call-conv=true}})"),
                (f"rocdl-attach-target{{O=2 abi=600 chip={chip} "
                 f"correct-sqrt=true daz=false fast=false features= "
                 f"finite-only=false module= triple=amdgcn-amd-amdhsa "
                 f"unsafe-math=false wave64=true}}"),
                "convert-scf-to-cf",
                "convert-cf-to-llvm",
                ("fly-gpu-to-llvm{use-bare-pointers-for-host=true "
                 "use-bare-pointers-for-kernels=true}"),
                "convert-arith-to-llvm",
                "convert-func-to-llvm",
                "reconcile-unrealized-casts",
            ]
            pm = PassManager.parse(f"builtin.module({','.join(pre_binary)})")
            pm.run(module.operation)

            mlir_asm = module.operation.get_asm(enable_debug_info=False)

        # ---------------------------------------------------------------
        # 3. Extract GPU LLVM IR → link with shmem bc → compile to HSACO
        # ---------------------------------------------------------------
        gpu_mlir = _extract_gpu_module_mlir(mlir_asm, ctx)
        llvm_ir  = _mlir_to_llvm_ir(gpu_mlir)

    _link_and_compile_hsaco(llvm_ir, chip, out_path, shmem_bc)

    # -----------------------------------------------------------------------
    # 4. Return callable (loads HSACO lazily on first launch)
    # -----------------------------------------------------------------------
    return ShmemKernel(out_path, kernel_fn._kernel_name)
