# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import abc
import ctypes
import inspect
import struct as _struct
import threading
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Type, get_origin

import torch

from .._mlir import ir
from .._mlir._mlir_libs._mlirDialectsFly import DLTensorAdaptor, MemRefType
from .._mlir.extras import types as T
from ..expr.numeric import Numeric
from ..expr.typing import (
    AddressSpace,
    Boolean,
    Constexpr,
    Float32,
    Int32,
    Pointer,
    PointerType,
    Stream,
    Tensor,
    address_space_from_attr,
)
from .protocol import DslType, JitArgument

_RESOLVE_SIG_WARNED = set()


def resolve_signature(func):
    """``inspect.signature`` with PEP 563 string annotations resolved; warn once on NameError fallback."""
    try:
        return inspect.signature(func, eval_str=True)
    except NameError as exc:
        key = getattr(func, "__qualname__", repr(func))
        if key not in _RESOLVE_SIG_WARNED:
            _RESOLVE_SIG_WARNED.add(key)
            warnings.warn(f"FlyDSL: unresolved annotation in {key!r} ({exc}); cache key may degrade.", stacklevel=2)
        return inspect.signature(func)


class JitArgumentRegistry:
    registry: Dict[type, Tuple[Callable, Type[DslType]]] = {}
    jit_arg2dsl_type: Dict[type, Type[DslType]] = {}

    @classmethod
    def register(cls, py_type: type, *, dsl_type: Type[DslType] = None):
        def decorator(jit_arg_constructor: Callable):
            if py_type in cls.registry:
                raise ValueError(f"JitArgumentConstructor for {py_type} already registered")

            if dsl_type is not None:
                dest_dsl_type = dsl_type
            elif isinstance(jit_arg_constructor, type) and isinstance(jit_arg_constructor, DslType):
                dest_dsl_type = jit_arg_constructor
            else:
                raise ValueError(f"Invalid dsl_type for {py_type}: {dsl_type}")

            cls.registry[py_type] = (jit_arg_constructor, dest_dsl_type)
            cls.jit_arg2dsl_type[jit_arg_constructor] = dest_dsl_type
            return jit_arg_constructor

        return decorator

    @classmethod
    def register_jit_arg(cls, jit_arg: type, dsl_type: Type[DslType]):
        if not issubclass(jit_arg, JitArgument):
            raise ValueError(f"JitArgument must implement JitArgument protocol, got {jit_arg}")
        if jit_arg in cls.jit_arg2dsl_type:
            raise ValueError(f"JitArgument {jit_arg} already registered")
        cls.jit_arg2dsl_type[jit_arg] = dsl_type

    @classmethod
    def get(cls, py_type: type) -> Optional[Tuple[Callable, Type[DslType]]]:
        result = cls.registry.get(py_type, None)
        if result is not None:
            return result
        # Fallback: check base classes (e.g., torch.nn.Parameter -> torch.Tensor)
        for registered_type, entry in cls.registry.items():
            if isinstance(registered_type, type) and issubclass(py_type, registered_type):
                return entry
        return (None, None)

    @classmethod
    def get_dsl_type(cls, jit_arg_type: type) -> Type[DslType]:
        return cls.jit_arg2dsl_type[jit_arg_type]


def is_type_param_annotation(annotation) -> bool:
    """Check if annotation is Type, Type[T]."""
    origin = get_origin(annotation)
    return annotation is Type or annotation is type or origin is Type or origin is type


def convert_to_jit_arguments(
    sig: inspect.Signature, bound
) -> tuple[List[str], List[JitArgument], List[DslType], dict[str, any]]:
    param_names: List[str] = []
    jit_args: List[JitArgument] = []
    dsl_types: List[DslType] = []
    constexpr_values: dict[str, any] = {}

    for param_name, value in bound.arguments.items():
        param = sig.parameters[param_name]
        annotation = param.annotation

        if annotation is not inspect.Parameter.empty and Constexpr.is_constexpr_annotation(annotation):
            constexpr_values[param_name] = value
            continue

        if annotation is not inspect.Parameter.empty and is_type_param_annotation(annotation):
            constexpr_values[param_name] = value
            continue

        is_jit_arg = isinstance(value, JitArgument)
        is_dsl_type = isinstance(value, DslType)
        if is_jit_arg and is_dsl_type:
            jit_arg = value
            dsl_type = type(value)
        elif is_jit_arg:
            jit_arg = value
            dsl_type = JitArgumentRegistry.get_dsl_type(type(value))
            if dsl_type is None:
                raise TypeError(
                    f"No DslType registered for JitArgument type {type(value).__name__} (parameter '{param_name}')"
                )
        elif isinstance(annotation, type) and issubclass(annotation, JitArgument):
            # Annotation is a JitArgument (e.g. ``Stream``)
            try:
                jit_arg = annotation(value)
            except Exception as e:
                raise TypeError(f"Failed to construct JitArgument for parameter '{param_name}': {e}") from e
            dsl_type = annotation
        else:
            jit_arg_constructor, dsl_type = JitArgumentRegistry.get(type(value))
            if jit_arg_constructor is None:
                raise TypeError(f"No JitArgument registered for type {type(value).__name__} (parameter '{param_name}')")
            try:
                jit_arg = jit_arg_constructor(value)
            except Exception as e:
                raise TypeError(f"Failed to construct JitArgument for parameter '{param_name}': {e}") from e

        param_names.append(param_name)
        jit_args.append(jit_arg)
        dsl_types.append(dsl_type)
    return param_names, jit_args, dsl_types, constexpr_values


# ================================ Common useful JitArguments ================================


class _LayoutPlan:
    """Single source of the dynamic-layout buffer's byte contract.

    The buffer is dynamic-shape i32's then dynamic-stride i32/i64's, contiguous,
    ascending index.  Both groups are packed by one pre-compiled
    ``struct.Struct`` at offset 0 (``<`` = no padding), so a fill is a single
    ``pack_into``.
    """

    __slots__ = ("buf_ctype", "codec", "shape", "stride")

    def __init__(self, shape, stride, use_32bit_stride):
        self.shape = shape
        self.stride = stride
        struct_fmt = "<" + "i" * len(shape) + ("i" if use_32bit_stride else "q") * len(stride)
        self.codec = _struct.Struct(struct_fmt)
        self.buf_ctype = ctypes.c_byte * self.codec.size


class MemRefSpec:
    # shape[i] / stride[i] hold the *encoded* per-dim value: a non-negative value
    # is a static size/stride; a negative value ``-div`` marks a dynamic dim with
    # divisibility ``div``.  This is exactly the cache-signature encoding, so
    # get_cache_signature returns it directly and the dyn-index masks are ``v < 0``.
    __slots__ = ("alignment", "use_32bit_stride", "ndim", "shape", "stride")

    def __init__(self, element_bits, shape, strides, alignment=None, use_32bit_stride=False):
        if len(shape) != len(strides):
            raise RuntimeError("MemRefSpec: shape and strides must have equal rank")
        n = len(shape)
        if n == 0:
            raise RuntimeError("MemRefSpec: must have at least one dimension")
        self.alignment = alignment if alignment is not None else (element_bits + 7) // 8
        if self.alignment < 1:
            raise RuntimeError("Alignment must be at least 1")
        self.use_32bit_stride = use_32bit_stride
        self.ndim = n
        self.shape = [int(s) for s in shape]  # encoded, all static initially
        self.stride = [int(s) for s in strides]

    def mark_layout_dynamic(self, leading_dim=-1, divisibility=1):
        if leading_dim == -1:
            leading_dim = next((i for i in range(self.ndim) if self.stride[i] == 1), -1)
        if leading_dim < 0 or leading_dim >= self.ndim:
            raise RuntimeError("tensor has no axis with stride == 1; layout-dynamic memref requires one")
        if self.stride[leading_dim] != 1:
            raise RuntimeError("Leading dimension must have stride 1")
        for i in range(self.ndim):
            self.shape[i] = -1  # all shapes dynamic, divisibility 1
        for i in range(self.ndim):
            if i != leading_dim:
                self.stride[i] = -divisibility  # non-leading strides dynamic
        return self

    def mark_shape_dynamic(self, dims, divisibilities):
        for idx, div in zip(dims, divisibilities):
            if idx < 0 or idx >= self.ndim:
                raise RuntimeError("markDynamic: dimension index out of range")
            self.shape[idx] = -int(div)
        return self

    def mark_stride_dynamic(self, dims, divisibilities):
        for idx, div in zip(dims, divisibilities):
            if idx < 0 or idx >= self.ndim:
                raise RuntimeError("markDynamic: dimension index out of range")
            self.stride[idx] = -int(div)
        return self

    def get_cache_signature(self):
        # shape / stride already hold the encoded values -- direct read, no scan.
        return (self.alignment, self.use_32bit_stride, tuple(self.shape), tuple(self.stride))

    @property
    def shape_dyn_indices(self):
        return tuple(i for i, v in enumerate(self.shape) if v < 0)

    @property
    def stride_dyn_indices(self):
        return tuple(i for i, v in enumerate(self.stride) if v < 0)

    def get_memref_type(self, element_type):
        return MemRefType.get(element_type, self.shape, self.stride, self.use_32bit_stride, self.alignment)


class MemRefJitArg(abc.ABC):
    """Framework-neutral base for arguments whose bottom IR type is a ``memref``.

    Owns the honest, single-source contract: layout-dynamic configuration, memref
    IR-type derivation (via a metadata-driven ``MemRefSpec``), the cache
    signature, and the dynamic-layout *byte contract* (see :class:`_LayoutPlan`).
    It is framework-agnostic: it never reads the live argument itself, and leaves
    ``__c_abi_spec__`` abstract.

    A concrete framework subclass (e.g. :class:`TorchTensorJitArg`) implements
    ``__c_abi_spec__``: it builds the fills inline -- reading the framework object
    directly (torch ``data_ptr`` / ``shape`` / ``stride``, or numpy's byte-stride
    normalization, etc.) and exec-unrolling the layout pack per ``_LayoutPlan``.
    This split is what lets the protocol stay honest (neutral contract here) while
    the fill stays fast (direct framework reads there).
    """

    def __init__(
        self,
        *,
        element_bits: int,
        shape,
        strides,
        dtype,
        assumed_align: Optional[int] = None,
        use_32bit_stride: bool = False,
        dynamic_layout: bool = True,
    ):
        self.element_bits = element_bits
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.assumed_align = assumed_align
        self.use_32bit_stride = use_32bit_stride
        self.dtype = dtype
        self.rank = len(self.shape)
        self.dynamic_layout = dynamic_layout
        # Lazy: the MemRefSpec object is constructed only when the compile path
        # (__get_ir_types__) or an explicit mark_* actually needs it.
        self.spec = None
        self.is_layout_dynamic = dynamic_layout

        # Validate eagerly so a no-unit-stride tensor fails at wrap time (same
        # timing as before) with the same actionable message.
        if dynamic_layout and 1 not in self.strides:
            raise RuntimeError(
                f"cannot auto-mark layout-dynamic for tensor "
                f"shape={self.shape} strides={self.strides}: tensor has no axis "
                f"with stride == 1; layout-dynamic memref requires one. "
                "Use flyc.from_dlpack(t) to wrap as a static memref instead."
            )

    def _ensure_spec(self):
        if self.spec is None:
            spec = MemRefSpec(self.element_bits, self.shape, self.strides, self.assumed_align, self.use_32bit_stride)
            if self.dynamic_layout:
                spec.mark_layout_dynamic()
            self.spec = spec
        return self.spec

    @property
    def shape_dyn_indices(self) -> Tuple[int, ...]:
        return self._ensure_spec().shape_dyn_indices

    @property
    def stride_dyn_indices(self) -> Tuple[int, ...]:
        return self._ensure_spec().stride_dyn_indices

    @abc.abstractmethod
    def __c_abi_spec__(self): ...

    @abc.abstractmethod
    def element_type(self):
        """Build the MLIR element type in the active (compile) context.
        Framework-specific (torch dtype map / dlpack)."""

    def __get_ir_types__(self):
        return [self._ensure_spec().get_memref_type(self.element_type)]

    def __cache_signature__(self):
        # TODO: ``type(self)`` + framework ``dtype`` make TorchTensorJitArg and
        # DLTensorJitArg wrap distinct keys though they lower to the same memref;
        # a framework-neutral dtype id + memref-family tag could share the module.
        if self.spec is not None:
            return (type(self), self.dtype) + self.spec.get_cache_signature()
        align = self.assumed_align if self.assumed_align is not None else (self.element_bits + 7) // 8
        n = self.rank
        if self.dynamic_layout:
            unit = self.strides.index(1)  # validated in __init__
            shape = (-1,) * n
            stride = tuple(self.strides[i] if i == unit else -1 for i in range(n))
        else:
            shape = self.shape
            stride = self.strides
        return (type(self), self.dtype, align, self.use_32bit_stride, shape, stride)

    def _normalize_dims_div(self, dims, divisibility, what: str):
        """Normalize the ``(dims, divisibility)`` argument forms.

        * ``dims=int,  divisibility=int``  — single dimension, single divisibility.
        * ``dims=list, divisibility=list`` — one-to-one; lists must be equal length.
        * ``dims=list, divisibility=int``  — the divisibility is broadcast to every dim.

        Negative dimension indices are accepted (Python-style, ``idx + rank``).
        Returns ``(idx_list, div_list)`` of equal length.
        """
        dim_list = [dims] if isinstance(dims, int) else list(dims)
        if isinstance(divisibility, int):
            div_list = [divisibility] * len(dim_list)
        else:
            if isinstance(dims, int):
                raise ValueError(f"{what}: divisibility must be an int when dims is a single int")
            div_list = list(divisibility)
            if len(div_list) != len(dim_list):
                raise ValueError(
                    f"{what}: dims (len {len(dim_list)}) and divisibility "
                    f"(len {len(div_list)}) must have equal length"
                )

        normalized = []
        for d in dim_list:
            idx = int(d)
            if idx < 0:
                idx += self.rank
            if idx < 0 or idx >= self.rank:
                raise ValueError(f"{what}: dimension index {d} out of range for rank {self.rank}")
            normalized.append(idx)
        divs = [int(x) for x in div_list]
        for v in divs:
            if v <= 0 or (v & (v - 1)) != 0:
                raise ValueError(f"{what}: divisibility {v} must be a power of two")
        return normalized, divs

    def mark_layout_dynamic(self, leading_dim: Optional[int] = None, divisibility: int = 1):
        self._ensure_spec().mark_layout_dynamic(-1 if leading_dim is None else leading_dim, divisibility)
        self.is_layout_dynamic = True
        return self

    def mark_shape_dynamic(self, dims, divisibility=1):
        """Mark the *shape* leaf of the given dimension(s) dynamic.
        Strides and all other dims are left untouched.

        ``dims`` is an int or list of ints (negative indices allowed).
        ``divisibility`` (a power of two, default 1) is the compile-time
        alignment guaranteed on each dynamic size, given per-dim or broadcast.

        Examples::

            # GEMM whose M varies per batch: mark M dynamic, keep K static.
            flyc.from_dlpack(a).mark_shape_dynamic(0)

            # dims 0 and 2 dynamic, both guaranteed multiples of 8.
            t.mark_shape_dynamic([0, 2], divisibility=8)

            # per-dim divisibility (mode 0 multiple of 16, mode 1 of 8).
            t.mark_shape_dynamic([0, 1], [16, 8])
        """
        idxs, divs = self._normalize_dims_div(dims, divisibility, "mark_shape_dynamic")
        self._ensure_spec().mark_shape_dynamic(idxs, divs)
        self.is_layout_dynamic = True
        return self

    def mark_stride_dynamic(self, dims, divisibility=1):
        """Mark the *stride* leaf of the given dimension(s) dynamic. Shapes and all
        other dims are left untouched.

        ``dims`` is an int or list of ints (negative indices allowed).
        ``divisibility`` (a power of two, default 1) is the compile-time
        alignment guaranteed on each dynamic stride, given per-dim or broadcast.

        Examples::

            # Row stride varies but is always a multiple of 16 (e.g. padded rows).
            flyc.from_dlpack(a).mark_stride_dynamic(0, divisibility=16)

            # Combine with mark_shape_dynamic: M dynamic *and* its stride dynamic.
            t.mark_shape_dynamic(0).mark_stride_dynamic([0, 1], divisibility=8)
        """
        idxs, divs = self._normalize_dims_div(dims, divisibility, "mark_stride_dynamic")
        self._ensure_spec().mark_stride_dynamic(idxs, divs)
        self.is_layout_dynamic = True
        return self


class DLTensorJitArg(MemRefJitArg):
    """Generic dlpack-backed memref arg: works with *any* ``__dlpack__`` object
    (torch, numpy, jax, cupy, ...) through the DLPack protocol alone.

    It never touches a framework-specific API. All metadata (shape, stride,
    dtype, element type) is read through :class:`DLTensorAdaptor` off the DLPack
    capsule, and the per-launch fill re-reads ``data_ptr`` (and any dynamic dims)
    the same way. This portability costs one ``__dlpack__()`` + capsule parse per
    launch -- the price of going through DLPack rather than a native handle; use
    :class:`TorchTensorJitArg` (``from_torch_tensor``) when torch-native speed
    matters.
    """

    def __init__(
        self,
        dltensor,
        assumed_align: Optional[int] = None,
        use_32bit_stride: bool = False,
        dynamic_layout: bool = True,
    ):
        self.dltensor = dltensor
        try:
            dl = dltensor.__dlpack__(stream=-1)
            with_stream = True
        except Exception:
            with_stream = False
            dl = dltensor.__dlpack__()
        dladaptor = DLTensorAdaptor(dl)
        self.dladaptor = dladaptor
        self.with_stream_dlpack = with_stream
        super().__init__(
            element_bits=dladaptor.element_bits,
            shape=dladaptor.shape,
            strides=dladaptor.stride,
            dtype=dladaptor.dtype_id,
            assumed_align=assumed_align,
            use_32bit_stride=use_32bit_stride,
            dynamic_layout=dynamic_layout,
        )

    @property
    def element_type(self):
        # The dtype as an ir Type, built in the active (compile) context.
        return self.dladaptor.dtype

    def __c_abi_spec__(self):
        with_stream = self.with_stream_dlpack

        def _open(a):
            ad = getattr(a, "dladaptor", None)
            if ad is not None:
                return ad
            t = a.dltensor if hasattr(a, "dltensor") else a
            return DLTensorAdaptor(t.__dlpack__(stream=-1) if with_stream else t.__dlpack__())

        if not self.is_layout_dynamic:

            def ptr_fill(a, s, _open=_open):
                s.value = _open(a).data_ptr

            return [(ctypes.c_void_p, ptr_fill)]

        # Layout-dynamic: the pointer and layout slots are dispatched back to back
        # for this arg, so they share a single ``__dlpack__()`` per launch. The
        # pointer fill opens the live tensor once and hands the dladaptor to the
        # layout fill through a thread-local (thread-safe; never mutates the arg).
        plan = _LayoutPlan(self.shape_dyn_indices, self.stride_dyn_indices, bool(self.use_32bit_stride))
        shared = threading.local()

        def ptr_fill(a, s, _open=_open, _shared=shared):
            ad = _open(a)
            _shared.dladaptor = ad
            s.value = ad.data_ptr

        body = ["    _ad = _shared.dladaptor"]
        terms = []
        if plan.shape:
            body.append("    sh = _ad.shape")
            terms += [f"sh[{d}]" for d in plan.shape]
        if plan.stride:
            body.append("    st = _ad.stride")
            terms += [f"st[{d}]" for d in plan.stride]
        body.append(f"    _codec.pack_into(s, 0, {', '.join(terms)})")
        src = "def fill(a, s, _codec=_codec, _shared=_shared):\n" + "\n".join(body) + "\n"
        ns = {"_codec": plan.codec, "_shared": shared}
        exec(compile(src, "<flydsl-cabi-fill>", "exec"), ns)
        return [(ctypes.c_void_p, ptr_fill), (plan.buf_ctype, ns["fill"])]


_TORCH_DTYPE_TO_MLIR_BUILDER = {
    torch.float16: T.f16,
    torch.bfloat16: T.bf16,
    torch.float32: T.f32,
    torch.float64: T.f64,
    torch.bool: lambda: ir.IntegerType.get_signless(1),
    torch.uint8: lambda: ir.IntegerType.get_signless(8),
    torch.int8: lambda: ir.IntegerType.get_signless(8),
    torch.int16: lambda: ir.IntegerType.get_signless(16),
    torch.int32: lambda: ir.IntegerType.get_signless(32),
    torch.int64: lambda: ir.IntegerType.get_signless(64),
}
for _torch_name, _mlir_ctor in (
    ("float8_e5m2", ir.Float8E5M2Type),
    ("float8_e4m3fn", ir.Float8E4M3FNType),
    ("float8_e5m2fnuz", ir.Float8E5M2FNUZType),
    ("float8_e4m3fnuz", ir.Float8E4M3FNUZType),
):
    _torch_dt = getattr(torch, _torch_name, None)
    if _torch_dt is not None:
        _TORCH_DTYPE_TO_MLIR_BUILDER[_torch_dt] = _mlir_ctor.get
del _torch_name, _mlir_ctor, _torch_dt


def torch_dtype_to_mlir_type(dtype):
    builder = _TORCH_DTYPE_TO_MLIR_BUILDER.get(dtype)
    if builder is None:
        raise TypeError(f"unsupported torch dtype for memref element type: {dtype}")
    return builder()


@JitArgumentRegistry.register(torch.Tensor, dsl_type=Tensor)
class TorchTensorJitArg(MemRefJitArg):
    def __init__(
        self,
        tensor: torch.Tensor,
        assumed_align: Optional[int] = None,
        use_32bit_stride: bool = False,
        dynamic_layout: bool = True,
    ):
        self.torch_tensor = tensor
        super().__init__(
            element_bits=tensor.element_size() * 8,
            shape=tensor.shape,
            strides=tensor.stride(),
            dtype=tensor.dtype,
            assumed_align=assumed_align,
            use_32bit_stride=use_32bit_stride,
            dynamic_layout=dynamic_layout,
        )

    @property
    def element_type(self):
        return torch_dtype_to_mlir_type(self.dtype)

    def __c_abi_spec__(self):
        def ptr_fill(a, s):
            t = a.torch_tensor if hasattr(a, "torch_tensor") else a
            s.value = t.data_ptr()

        slots = [(ctypes.c_void_p, ptr_fill)]
        if self.is_layout_dynamic:
            plan = _LayoutPlan(self.shape_dyn_indices, self.stride_dyn_indices, bool(self.use_32bit_stride))
            body = ["    t = a.torch_tensor if hasattr(a, 'torch_tensor') else a"]
            terms = []
            if plan.shape:
                body.append("    sh = t.shape")
                terms += [f"sh[{d}]" for d in plan.shape]
            if plan.stride:
                body.append("    st = t.stride()")
                terms += [f"st[{d}]" for d in plan.stride]
            body.append(f"    _codec.pack_into(s, 0, {', '.join(terms)})")
            src = "def fill(a, s, _codec=_codec):\n" + "\n".join(body) + "\n"
            ns = {"_codec": plan.codec}
            exec(compile(src, "<flydsl-cabi-fill>", "exec"), ns)
            slots.append((plan.buf_ctype, ns["fill"]))
        return slots


class PointerJitArg:
    def __init__(
        self,
        element_type: Type[Numeric],
        pointer: ctypes.c_void_p | int | None,
        address_space=AddressSpace.Global,
        alignment: Optional[int] = None,
    ):
        address_space = address_space_from_attr(address_space)
        self.pointer = pointer if isinstance(pointer, ctypes.c_void_p) else ctypes.c_void_p(pointer)
        self.address_space = address_space
        self.element_type = element_type
        if alignment is None:
            alignment = self._trivial_alignment_bytes(element_type)
        self.alignment = alignment

    @staticmethod
    def _trivial_alignment_bytes(element_type) -> int:
        # Matches AlignAttr::getTrivialAlignment
        if isinstance(element_type, type) and issubclass(element_type, Numeric):
            width = element_type.width
        else:
            width = element_type.getIntOrFloatBitWidth()
        return (width + 7) // 8

    def __get_ir_types__(self):
        ir_type = self.element_type
        if isinstance(ir_type, type) and issubclass(ir_type, Numeric):
            ir_type = self.element_type.ir_type
        return [PointerType.get(ir_type, self.address_space, self.alignment)]

    def __cache_signature__(self):
        return (type(self), self.element_type, str(self.address_space), self.alignment)

    def __c_abi_spec__(self):
        def fill(a, s):
            if isinstance(a, PointerJitArg):
                s.value = a.pointer.value
            elif isinstance(a, ctypes.c_void_p):
                s.value = a.value
            else:
                s.value = int(a)

        return [(ctypes.c_void_p, fill)]


def from_dlpack(
    tensor,
    *,
    assumed_align: Optional[int] = None,
    use_32bit_stride: bool = False,
) -> DLTensorJitArg:
    return DLTensorJitArg(tensor, assumed_align, use_32bit_stride, dynamic_layout=False)


def from_torch_tensor(
    tensor: torch.Tensor,
    *,
    assumed_align: Optional[int] = None,
    use_32bit_stride: bool = False,
) -> TorchTensorJitArg:
    return TorchTensorJitArg(tensor, assumed_align, use_32bit_stride, dynamic_layout=False)


def from_c_void_p(
    element_type: Type[Numeric],
    pointer: ctypes.c_void_p | int | None,
    *,
    address_space=AddressSpace.Global,
    assumed_align: Optional[int] = None,
) -> PointerJitArg:
    return PointerJitArg(element_type, pointer, address_space, assumed_align)


JitArgumentRegistry.register(bool)(Boolean)
JitArgumentRegistry.register(int)(Int32)
JitArgumentRegistry.register(float)(Float32)
JitArgumentRegistry.register(torch.cuda.Stream)(Stream)

JitArgumentRegistry.register_jit_arg(PointerJitArg, Pointer)
JitArgumentRegistry.register_jit_arg(DLTensorJitArg, Tensor)
