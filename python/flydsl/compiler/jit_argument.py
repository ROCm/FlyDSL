# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import inspect
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Type, get_origin

import torch

from .._mlir._mlir_libs._mlirDialectsFly import DLTensorAdaptor
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
from .protocol import DslType, JitArgument, is_dsl_type, is_jit_argument

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


_FLOAT8_DTYPES = tuple(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
)


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

        is_jit_arg = is_jit_argument(value)
        is_dsl = is_dsl_type(value)
        if is_jit_arg and is_dsl:
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


@JitArgumentRegistry.register(torch.Tensor, dsl_type=Tensor)
class TensorAdaptor:
    def __init__(
        self,
        tensor: torch.Tensor,
        assumed_align: Optional[int] = None,
        use_32bit_stride: bool = False,
        dynamic_layout: bool = True,
    ):
        # Forward-only interop: DLPack export from torch rejects tensors that
        # still participate in autograd, so detach before crossing into FlyDSL.
        dlpack_tensor = tensor.detach() if tensor.requires_grad else tensor

        # torch < 2.12 cannot export fp8 dtypes through DLPack (raises "float8 types are not supported by dlpack").
        # Reinterpret as uint8 for transport; the original dtype is preserved in ``_orig_dtype``
        # below and re-prepended to the cache signature so e4m3 / e5m2 / etc. don't collide.
        #
        # TODO: Drop both this view and ``_orig_dtype`` once the minimum torch version reaches 2.12 — DLPack 1.0 (PR
        # pytorch/pytorch#145000) wires every fp8 code through DLConvertor.
        if _FLOAT8_DTYPES and dlpack_tensor.dtype in _FLOAT8_DTYPES:
            dlpack_tensor = dlpack_tensor.view(torch.uint8)
        self._tensor_keepalive = dlpack_tensor

        try:
            dl = dlpack_tensor.__dlpack__(stream=-1)
        except Exception:
            # CPU tensors (e.g. COMPILE_ONLY AOT) don't accept stream arg
            dl = dlpack_tensor.__dlpack__()
        self.tensor_adaptor = DLTensorAdaptor(dl, assumed_align, use_32bit_stride)
        self.assumed_align = assumed_align
        self.use_32bit_stride = use_32bit_stride
        self._orig_dtype = tensor.dtype
        self._orig_shape = tensor.shape
        self._orig_strides = tensor.stride()
        self._dyn_leading_dim = -1
        self._is_layout_dynamic = False

        # TODO: this duplicates state the C++ DLTensorAdaptor already owns. The
        # reusable-slot fast path keeps it Python-side only to avoid building a
        # heavy DLPack-backed adaptor per launch. Refactor to read the masks
        # from C++ (single source of truth) once that path is reworked.
        self._shape_dyn_indices: Tuple[int, ...] = ()
        self._stride_dyn_indices: Tuple[int, ...] = ()

        if dynamic_layout:
            try:
                self._mark_layout_dynamic(leading_dim=-1, divisibility=1)
            except RuntimeError as e:
                raise RuntimeError(
                    f"cannot auto-mark layout-dynamic for tensor "
                    f"shape={tuple(tensor.shape)} strides={tuple(tensor.stride())}: {e}. "
                    "Use flyc.from_dlpack(t) to wrap as a static memref instead."
                ) from e

    @staticmethod
    def _extract_data_ptr(arg):
        if hasattr(arg, "_tensor_keepalive"):
            return arg._tensor_keepalive.data_ptr()
        return arg.data_ptr()

    @staticmethod
    def _pick_unit_stride_axis(strides) -> int:
        """Return the index of the first axis whose stride is one.

        Raises ``RuntimeError`` if no axis qualifies, so callers do not have
        to handle a None return.
        """
        candidates = [idx for idx, val in enumerate(strides) if int(val) == 1]
        if not candidates:
            raise RuntimeError("tensor has no axis with stride == 1; layout-dynamic memref requires one")
        return candidates[0]

    @classmethod
    def _reusable_slot_spec(cls, arg):
        """Reusable slot(s) for a tensor argument.

        Returns ``(ctype, extract)`` for static memref (data ptr only), or a
        list of such tuples for dynamic memref (data ptr + a layout-buffer
        slot carrying the runtime shape / non-leading stride values).
        Buffer slots use the in-place protocol: ``extract(arg, storage)``
        writes into ``storage`` via ``struct.pack_into``.
        """
        if not hasattr(arg, "data_ptr") and not isinstance(arg, cls):
            return None

        adaptor = arg if isinstance(arg, cls) else cls(arg)
        if not getattr(adaptor, "_is_layout_dynamic", False):
            return ctypes.c_void_p, cls._extract_data_ptr

        # Dynamic memref: pre-compute the layout-buffer packing plan.
        # Layout matches C++ buildMemRefDesc: dynamic-shape i32's (ascending
        # index) then dynamic-stride i32/i64's (ascending index)
        shape_dim_indices = adaptor._shape_dyn_indices
        stride_dim_indices = adaptor._stride_dyn_indices
        use_32bit_stride = bool(adaptor.use_32bit_stride)
        shape_size = len(shape_dim_indices) * 4
        stride_elem = 4 if use_32bit_stride else 8
        buf_ctype = ctypes.c_byte * (shape_size + len(stride_dim_indices) * stride_elem)

        import struct as _struct

        shape_codec = _struct.Struct("<" + "i" * len(shape_dim_indices)) if shape_dim_indices else None
        if stride_dim_indices:
            stride_codec = _struct.Struct("<" + ("i" if use_32bit_stride else "q") * len(stride_dim_indices))
        else:
            stride_codec = None

        def pack_layout_buffer(
            t,
            storage,
            _shape_codec=shape_codec,
            _stride_codec=stride_codec,
            _shape_dims=shape_dim_indices,
            _stride_dims=stride_dim_indices,
            _shape_size=shape_size,
        ):
            tens = t._tensor_keepalive if isinstance(t, cls) else t
            if _shape_codec is not None:
                _shape_codec.pack_into(storage, 0, *[tens.shape[d] for d in _shape_dims])
            if _stride_codec is not None:
                st = tens.stride()
                _stride_codec.pack_into(storage, _shape_size, *[st[d] for d in _stride_dims])

        return [
            (ctypes.c_void_p, cls._extract_data_ptr),
            (buf_ctype, pack_layout_buffer),
        ]

    def requires_memref_desc(func):
        def wrapper(self, *args, **kwargs):
            self.tensor_adaptor.build_memref_desc()
            return func(self, *args, **kwargs)

        return wrapper

    @requires_memref_desc
    def __get_ir_types__(self):
        return [self.tensor_adaptor.get_memref_type()]

    @requires_memref_desc
    def __get_c_pointers__(self):
        return self.tensor_adaptor.get_c_pointers()

    def __cache_signature__(self):
        return (type(self), self._orig_dtype) + self.tensor_adaptor.get_cache_signature()

    def _mark_layout_dynamic(self, leading_dim: int, divisibility: int):
        # Always pass a concrete axis index down. The DLPack stride view that
        # the backend sees can disagree with the framework view for tensors
        # with zero-size or unit-size axes (DLPack often coerces such strides
        # to 1), so we resolve on the framework strides here.
        resolved = self._pick_unit_stride_axis(self._orig_strides) if leading_dim == -1 else int(leading_dim)
        self.tensor_adaptor.mark_layout_dynamic(resolved, divisibility)
        self._dyn_leading_dim = resolved
        self._is_layout_dynamic = True
        rank = len(self._orig_shape)
        self._shape_dyn_indices = tuple(range(rank))
        self._stride_dyn_indices = tuple(d for d in range(rank) if d != resolved)
        return self

    def _normalize_dims_div(self, dims, divisibility, what: str):
        """Normalize the ``(dims, divisibility)`` argument forms.

        * ``dims=int,  divisibility=int``  — single dimension, single divisibility.
        * ``dims=list, divisibility=list`` — one-to-one; lists must be equal length.
        * ``dims=list, divisibility=int``  — the divisibility is broadcast to every dim.

        Negative dimension indices are accepted (Python-style, ``idx + rank``).
        Returns ``(idx_list, div_list)`` of equal length.
        """
        rank = len(self._orig_shape)

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
                idx += rank
            if idx < 0 or idx >= rank:
                raise ValueError(f"{what}: dimension index {d} out of range for rank {rank}")
            normalized.append(idx)
        divs = [int(x) for x in div_list]
        for v in divs:
            if v <= 0 or (v & (v - 1)) != 0:
                raise ValueError(f"{what}: divisibility {v} must be a power of two")
        return normalized, divs

    def mark_layout_dynamic(self, leading_dim: Optional[int] = None, divisibility: int = 1):
        # TODO: C++ markLayoutDynamic accumulates dynamic flags across calls
        # without resetting -- a 2nd call with a *different* leading_dim
        # leaves the previous call's stride[leading] dynamic, and the
        # Python-cached ``_dyn_leading_dim`` (used by ``_reusable_slot_spec``
        # to lay out the layout buffer) diverges from the C++ ABI.
        # Temporary guard: forbid 2nd call with a different leading_dim.
        # Fix path: make C++ reset all dynamic flags before re-marking.
        if leading_dim is None:
            leading_dim = -1
        if self._is_layout_dynamic and leading_dim not in (-1, self._dyn_leading_dim):
            raise NotImplementedError(
                f"mark_layout_dynamic(leading_dim={leading_dim}) conflicts with "
                f"auto-detected leading_dim={self._dyn_leading_dim} from __init__.  "
                "Re-binding leading_dim is not supported yet (see TODO in jit_argument.py)."
            )
        return self._mark_layout_dynamic(leading_dim, divisibility)

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
        self.tensor_adaptor.mark_shape_dynamic(idxs, divs)
        self._shape_dyn_indices = tuple(sorted(set(self._shape_dyn_indices) | set(idxs)))
        self._is_layout_dynamic = bool(self._shape_dyn_indices or self._stride_dyn_indices)
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
        self.tensor_adaptor.mark_stride_dynamic(idxs, divs)
        self._stride_dyn_indices = tuple(sorted(set(self._stride_dyn_indices) | set(idxs)))
        self._is_layout_dynamic = bool(self._shape_dyn_indices or self._stride_dyn_indices)
        return self


class PointerAdaptor:
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

    def __get_c_pointers__(self):
        return [ctypes.cast(ctypes.pointer(self.pointer), ctypes.c_void_p)]

    def __cache_signature__(self):
        return (type(self), self.element_type, str(self.address_space), self.alignment)

    @staticmethod
    def _extract_pointer(arg):
        if isinstance(arg, PointerAdaptor):
            return arg.pointer.value
        if isinstance(arg, ctypes.c_void_p):
            return arg.value
        return int(arg)

    @classmethod
    def _reusable_slot_spec(cls, arg):
        return ctypes.c_void_p, cls._extract_pointer


def from_dlpack(
    tensor: torch.Tensor,
    *,
    assumed_align: Optional[int] = None,
    use_32bit_stride: bool = False,
) -> TensorAdaptor:
    return TensorAdaptor(tensor, assumed_align, use_32bit_stride, dynamic_layout=False)


def from_c_void_p(
    element_type: Type[Numeric],
    pointer: ctypes.c_void_p | int | None,
    *,
    address_space=AddressSpace.Global,
    assumed_align: Optional[int] = None,
) -> PointerAdaptor:
    return PointerAdaptor(element_type, pointer, address_space, assumed_align)


JitArgumentRegistry.register(bool)(Boolean)
JitArgumentRegistry.register(int)(Int32)
JitArgumentRegistry.register(float)(Float32)
JitArgumentRegistry.register(torch.cuda.Stream)(Stream)

JitArgumentRegistry.register_jit_arg(PointerAdaptor, Pointer)
