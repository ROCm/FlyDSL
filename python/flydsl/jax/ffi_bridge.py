# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Bridge between FlyDSL compiled kernels and JAX's XLA custom-call interface.

This module compiles ``@flyc.jit`` functions via the normal FlyDSL MLIR
pipeline, then registers the resulting native function pointer as an XLA
custom-call target so that ``jax.jit`` can invoke it.

Architecture
------------
A compiled C trampoline (``_xla_bridge.so``) translates between XLA's
GPU custom-call convention and FlyDSL's bare-pointer convention:

XLA GPU custom call (``API_VERSION_STATUS_RETURNING_UNIFIED``)::

    void fn(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len)

Note: the api_version numbering differs between the XLA registration API
(``xla_client.register_custom_call_target``, where 0 = untyped custom call)
and StableHLO (``CustomCallOp``, where 2 = STATUS_RETURNING_UNIFIED for GPU).
Both refer to the same calling convention.

FlyDSL bare-pointer convention::

    void fn(void** ptrs)  // ptrs[i] = &storage[i], storage[i] = device_ptr or stream

The trampoline uses the ``opaque`` bytes to look up the registered FlyDSL
function and buffer count, then repacks the XLA buffers + stream into the
FlyDSL layout on the C stack.  This avoids Python callbacks entirely —
critical because XLA dispatches custom calls from C++ threads.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError(
        "JAX is required for flydsl.jax.  Install with:\n"
        "  pip install jax[rocm]"
    ) from exc


# ---------------------------------------------------------------------------
# Load (or build) the C trampoline shared library
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BRIDGE_SO = _THIS_DIR / "_xla_bridge.so"
_BRIDGE_C = _THIS_DIR / "_xla_bridge.c"


def _ensure_bridge_lib() -> ctypes.CDLL:
    """Load ``_xla_bridge.so``, compiling from source if necessary.

    Recompiles if the .c source is newer than the .so to avoid stale binaries.
    """
    needs_compile = not _BRIDGE_SO.exists()
    if not needs_compile and _BRIDGE_C.exists():
        needs_compile = _BRIDGE_C.stat().st_mtime > _BRIDGE_SO.stat().st_mtime
    if needs_compile:
        if not _BRIDGE_C.exists():
            raise FileNotFoundError(
                f"Cannot find XLA bridge source: {_BRIDGE_C}\n"
                f"Please rebuild or reinstall flydsl."
            )
        subprocess.check_call(
            ["cc", "-shared", "-fPIC", "-O2", "-lpthread",
             "-o", str(_BRIDGE_SO), str(_BRIDGE_C)],
            cwd=str(_THIS_DIR),
        )
    lib = ctypes.CDLL(str(_BRIDGE_SO))

    # int flydsl_xla_register(void *func_ptr, int n_buffers, int n_scalars, int64_t *scalar_vals)
    lib.flydsl_xla_register.restype = ctypes.c_int
    lib.flydsl_xla_register.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

    # void *flydsl_xla_get_bridge(int idx)
    lib.flydsl_xla_get_bridge.restype = ctypes.c_void_p
    lib.flydsl_xla_get_bridge.argtypes = [ctypes.c_int]

    return lib


_bridge_lib: Optional[ctypes.CDLL] = None


def _get_bridge_lib() -> ctypes.CDLL:
    global _bridge_lib
    if _bridge_lib is None:
        _bridge_lib = _ensure_bridge_lib()
    return _bridge_lib


# ---------------------------------------------------------------------------
# Thread-safe registry of compiled kernels
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_registered_targets: Dict[str, "_RegisteredTarget"] = {}


class _RegisteredTarget:
    """Bookkeeping for a registered XLA custom-call target."""

    __slots__ = ("name", "slot_idx", "n_buffers", "opaque_bytes",
                 "artifact", "bridge_func_ptr")

    def __init__(self, name: str, artifact, n_buffers: int, slot_idx: int, bridge_ptr: int):
        self.name = name
        self.artifact = artifact
        self.n_buffers = n_buffers
        self.slot_idx = slot_idx
        self.opaque_bytes = struct.pack("i", slot_idx)  # 4-byte opaque for XLA
        self.bridge_func_ptr = bridge_ptr


# ---------------------------------------------------------------------------
# Compilation + registration
# ---------------------------------------------------------------------------


def compile_and_register(
    flyc_func: Callable,
    *,
    input_shapes: List[Tuple[Tuple[int, ...], Any]],
    output_shapes: List[Tuple[Tuple[int, ...], Any]],
    constexpr_kwargs: Optional[dict] = None,
    runtime_scalars: Optional[dict] = None,
) -> str:
    """Compile a ``@flyc.jit`` function and register it as an XLA custom-call target.

    Parameters
    ----------
    flyc_func : callable
        A FlyDSL ``@flyc.jit``-decorated function (``JitFunction``).
    input_shapes : list of (shape, dtype)
        Shape and dtype of each input tensor.
    output_shapes : list of (shape, dtype)
        Shape and dtype of each output tensor.
    constexpr_kwargs : dict, optional
        Compile-time constant keyword arguments (``Constexpr`` parameters).
    runtime_scalars : dict, optional
        Runtime scalar arguments that are not tensors (e.g. ``n: Int32``).
        Keys are parameter names, values are representative values used
        during compilation tracing.

    Returns
    -------
    str
        The registered custom-call target name.
    """
    if constexpr_kwargs is None:
        constexpr_kwargs = {}
    if runtime_scalars is None:
        runtime_scalars = {}

    # Build a unique name based on function + shapes + constexprs.
    func_name = flyc_func.func.__name__ if hasattr(flyc_func, "func") else str(flyc_func)
    sig_parts = [func_name]
    for shape, dtype in input_shapes:
        sig_parts.append(f"i{shape}:{dtype}")
    for shape, dtype in output_shapes:
        sig_parts.append(f"o{shape}:{dtype}")
    for k, v in sorted(constexpr_kwargs.items()):
        sig_parts.append(f"c{k}={v}")
    for k, v in sorted(runtime_scalars.items()):
        sig_parts.append(f"r{k}={v}")

    name_hash = hashlib.sha256("|".join(sig_parts).encode()).hexdigest()[:16]
    target_name = f"flydsl_{name_hash}"

    # Hold the lock through the full compile+register path to prevent
    # concurrent threads from compiling and registering the same target.
    with _lock:
        if target_name in _registered_targets:
            return target_name

        # Create concrete JAX arrays for each tensor argument.
        from .adapter import from_jax

        all_arrays = []
        for shape, dtype in list(input_shapes) + list(output_shapes):
            all_arrays.append(jnp.zeros(shape, dtype=dtype))

        jit_args = [from_jax(a) for a in all_arrays]

        # Build the full argument list for the @flyc.jit function.
        call_args = list(jit_args)
        for _name, val in sorted(runtime_scalars.items()):
            call_args.append(val)

        # Force a fresh compilation by temporarily clearing the JitFunction's
        # caches.  Without this, a prior eager call with different adaptor
        # options (e.g. mark_layout_dynamic) may produce a cache hit with an
        # artifact whose function signature doesn't match the XLA bridge's
        # calling convention.
        saved_mem = flyc_func._mem_cache
        saved_call = flyc_func._call_state_cache
        flyc_func._mem_cache = {}
        flyc_func._call_state_cache = {}
        try:
            flyc_func(*call_args, **constexpr_kwargs)
            artifact = flyc_func.get_last_artifact()
        finally:
            # Merge the fresh compilation into the saved caches and restore.
            saved_mem.update(flyc_func._mem_cache)
            saved_call.update(flyc_func._call_state_cache)
            flyc_func._mem_cache = saved_mem
            flyc_func._call_state_cache = saved_call

        if artifact is None:
            raise RuntimeError(
                "FlyDSL compilation did not produce a cached artifact.  "
                "Ensure the function is a @flyc.jit-decorated function."
            )

        # Get the native function pointer from the compiled artifact.
        func_exe = artifact._get_func_exe()
        fly_func_ptr = ctypes.cast(func_exe, ctypes.c_void_p).value

        # Register with the C trampoline.
        # Scalar values are baked into the trampoline slot so they're inserted
        # between the tensor buffers and the stream at dispatch time.
        n_buffers = len(input_shapes) + len(output_shapes)
        scalar_values = [v for _name, v in sorted(runtime_scalars.items())]
        n_scalars = len(scalar_values)

        # Pack scalar values as int64 array for the C bridge.
        if n_scalars > 0:
            ScalarArray = ctypes.c_int64 * n_scalars
            scalar_arr = ScalarArray(*scalar_values)
            scalar_ptr = ctypes.cast(scalar_arr, ctypes.c_void_p)
        else:
            scalar_ptr = None

        lib = _get_bridge_lib()
        slot_idx = lib.flydsl_xla_register(fly_func_ptr, n_buffers, n_scalars, scalar_ptr)
        if slot_idx < 0:
            raise RuntimeError("Failed to register FlyDSL kernel in XLA bridge (too many targets?)")

        bridge_ptr = lib.flydsl_xla_get_bridge(slot_idx)

        target = _RegisteredTarget(target_name, artifact, n_buffers, slot_idx, bridge_ptr)

        # Register with JAX's XLA custom-call mechanism.
        _register_with_xla(target)

        _registered_targets[target_name] = target

    return target_name


def _register_with_xla(target: _RegisteredTarget) -> None:
    """Register the C bridge function as an XLA custom-call target."""
    # Ensure the JAX backend is initialized so the custom-call handler is
    # registered.  Without this, registrations are queued but never flushed.
    import jax as _jax
    _jax.default_backend()

    # Use JAX's own pycapsule to ensure the capsule name matches what XLA expects.
    from jax import ffi as _jax_ffi
    capsule = _jax_ffi.pycapsule(ctypes.c_void_p(target.bridge_func_ptr))

    # Use the internal xla_client directly to bypass JAX's platform name
    # mapping which maps "gpu" -> "CUDA" (wrong for ROCm).
    from jax._src.lib import xla_client as _xla_client

    # Detect the XLA internal platform name.
    if "ROCM" in _xla_client._custom_callback_handler:
        xla_platform_name = "ROCM"
    elif "CUDA" in _xla_client._custom_callback_handler:
        xla_platform_name = "CUDA"
    else:
        # Fallback: try both
        xla_platform_name = "ROCM"

    # Registration api_version=0 selects the untyped custom-call convention:
    #   void fn(stream, void** buffers, const char* opaque, size_t opaque_len)
    # This corresponds to StableHLO api_version=2 (STATUS_RETURNING_UNIFIED)
    # used in the CustomCallOp emitted by primitive.py.
    _xla_client.register_custom_call_target(
        target.name, capsule, xla_platform_name, api_version=0,
    )


def get_opaque_for(target_name: str) -> bytes:
    """Return the opaque bytes that XLA should pass to the custom call.

    The opaque encodes the slot index so the C trampoline can look up
    the correct FlyDSL function.
    """
    with _lock:
        target = _registered_targets.get(target_name)
    if target is None:
        raise KeyError(f"Target {target_name!r} not registered")
    return target.opaque_bytes
