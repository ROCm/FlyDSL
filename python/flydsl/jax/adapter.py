# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""JAX array adapter for FlyDSL's JitArgument protocol.

Wraps ``jax.Array`` objects via DLPack so they can be passed to
``@flyc.jit`` functions in eager mode.
"""

import ctypes
import threading
from typing import Optional

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError(
        "JAX is required for flydsl.jax.  Install it with:\n"
        "  pip install jax[rocm]  # or jax[cuda12] for NVIDIA GPUs"
    ) from exc

from .._mlir._mlir_libs._fly import DLTensorAdaptor


# JAX float8 dtypes that need uint8 view treatment (analogous to PyTorch float8).
_JAX_FLOAT8_DTYPES = tuple(
    dt
    for dt in (
        getattr(jnp, "float8_e3m4", None),
        getattr(jnp, "float8_e4m3", None),
        getattr(jnp, "float8_e4m3fn", None),
        getattr(jnp, "float8_e4m3fnuz", None),
        getattr(jnp, "float8_e4m3b11fnuz", None),
        getattr(jnp, "float8_e5m2", None),
        getattr(jnp, "float8_e5m2fnuz", None),
        getattr(jnp, "float8_e8m0fnu", None),
    )
    if dt is not None
)

# Lazy registration flag — we register JaxTensorAdaptor with
# JitArgumentRegistry only when torch (and therefore jit_argument.py)
# is actually importable.  This avoids a hard torch dependency.
_registered = False
_register_lock = threading.Lock()


def _ensure_registered():
    """Register JaxTensorAdaptor with FlyDSL's JitArgumentRegistry.

    Called lazily so that ``import flydsl.jax`` works even when
    torch is not installed.  Registration is only needed for the
    eager path (``@flyc.jit`` calls with JAX arrays).
    """
    global _registered
    if _registered:
        return
    with _register_lock:
        if _registered:
            return
        try:
            from ..compiler.jit_argument import JitArgumentRegistry
            from ..expr.typing import Tensor

            JitArgumentRegistry.register(jax.Array, dsl_type=Tensor)(JaxTensorAdaptor)
        except ImportError:
            pass  # torch not available — eager path won't work, but jax.jit path can
        _registered = True


class JaxTensorAdaptor:
    """Adapt a ``jax.Array`` to FlyDSL's JitArgument protocol via DLPack.

    Parameters
    ----------
    array : jax.Array
        A JAX array on a GPU device.
    assumed_align : int, optional
        Override pointer alignment assumption (bytes).
    use_32bit_stride : bool
        Use 32-bit strides in the MLIR memref descriptor.

    Notes
    -----
    - The array must reside on a single GPU device (no sharded arrays).
    - JAX does not expose HIP streams directly.  The caller is responsible
      for synchronization (``jax.block_until_ready`` before launch, and
      device synchronization after).
    - Float8 arrays are handled by extracting DLPack from a uint8 view.
    """

    def __init__(
        self,
        array: "jax.Array",
        assumed_align: Optional[int] = None,
        use_32bit_stride: bool = False,
    ):
        if not isinstance(array, jax.Array):
            raise TypeError(f"Expected jax.Array, got {type(array).__name__}")

        # Ensure the array is materialized and on a single device.
        array = jax.device_put(array)
        if callable(getattr(array, "is_deleted", None)) and array.is_deleted():
            raise ValueError("Cannot adapt a deleted JAX array")

        self._array_keepalive = array
        self._orig_dtype = array.dtype
        self._orig_shape = array.shape
        self._orig_strides = _jax_strides(array)
        self.assumed_align = assumed_align
        self.use_32bit_stride = use_32bit_stride
        self._dynamic_leading_dim = None
        self._dynamic_divisibility = None

        # Float8 arrays: extract DLPack from a uint8 view.
        dlpack_array = array
        if _JAX_FLOAT8_DTYPES and array.dtype in _JAX_FLOAT8_DTYPES:
            dlpack_array = array.view(jnp.uint8)
            self._array_keepalive = dlpack_array

        # Extract DLPack capsule.
        # JAX __dlpack__(stream=) semantics: stream=0 means "may be used on any
        # stream" (JAX default stream).  We pass 0 rather than a specific stream.
        dl_capsule = dlpack_array.__dlpack__(stream=0)
        self.tensor_adaptor = DLTensorAdaptor(dl_capsule, assumed_align, use_32bit_stride)

    # ------------------------------------------------------------------
    # JitArgument protocol
    # ------------------------------------------------------------------

    def _build_desc(func):
        """Decorator: ensure memref descriptor is built before access."""

        def wrapper(self, *args, **kwargs):
            self.tensor_adaptor.build_memref_desc()
            return func(self, *args, **kwargs)

        return wrapper

    @_build_desc
    def __fly_types__(self):
        return [self.tensor_adaptor.get_memref_type()]

    @_build_desc
    def __fly_ptrs__(self):
        return self.tensor_adaptor.get_c_pointers()

    # ------------------------------------------------------------------
    # Cache signature (same structure as TensorAdaptor)
    # ------------------------------------------------------------------

    def __cache_signature__(self):
        return (
            self._orig_dtype,
            self.assumed_align,
            self.use_32bit_stride,
            self._dynamic_leading_dim,
            self._dynamic_divisibility,
        )

    # ------------------------------------------------------------------
    # Fast-path reuse
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_data_ptr(arg):
        """Extract the raw device pointer from a jax.Array."""
        # For single-device arrays, unsafe_buffer_pointer gives the device ptr.
        buf = arg.addressable_data(0)
        return buf.unsafe_buffer_pointer()

    @classmethod
    def _reusable_slot_spec(cls, arg):
        if not isinstance(arg, jax.Array):
            return None
        return ctypes.c_void_p, cls._extract_data_ptr

    # ------------------------------------------------------------------
    # Layout dynamism (mirrors TensorAdaptor)
    # ------------------------------------------------------------------

    def mark_layout_dynamic(self, leading_dim: Optional[int] = None, divisibility: int = 1):
        """Mark dimensions as dynamic for shape-polymorphic compilation."""
        if leading_dim is None:
            leading_dim = -1
        self._dynamic_leading_dim = leading_dim
        self._dynamic_divisibility = divisibility
        self.tensor_adaptor.mark_layout_dynamic(leading_dim, divisibility)
        return self


def _jax_strides(array: "jax.Array"):
    """Return element strides for a JAX array (like torch.Tensor.stride()).

    JAX arrays are always contiguous (C-order by default), so we compute
    strides from the shape.
    """
    shape = array.shape
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def from_jax(
    array: "jax.Array",
    *,
    assumed_align: Optional[int] = None,
    use_32bit_stride: bool = False,
) -> "JaxTensorAdaptor":
    """Wrap a JAX array for use with ``@flyc.jit`` functions.

    Parameters
    ----------
    array : jax.Array
        A JAX array residing on a GPU device.
    assumed_align : int, optional
        Override pointer alignment hint (bytes).
    use_32bit_stride : bool
        Use 32-bit strides in the memref descriptor.

    Returns
    -------
    JaxTensorAdaptor
        An adapter implementing the FlyDSL ``JitArgument`` protocol.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import flydsl.compiler as flyc
    >>> from flydsl.jax import from_jax
    >>> a = jnp.ones(1024, dtype=jnp.float32)
    >>> ta = from_jax(a)
    >>> my_jit_func(ta, ...)  # pass to @flyc.jit function
    """
    _ensure_registered()
    return JaxTensorAdaptor(array, assumed_align, use_32bit_stride)
