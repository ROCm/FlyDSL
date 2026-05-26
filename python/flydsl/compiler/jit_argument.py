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

        is_jit_arg = hasattr(value, "__get_ir_types__") and hasattr(value, "__get_c_pointers__")
        is_dsl_type = hasattr(value, "__construct_from_ir_values__") and hasattr(value, "__extract_to_ir_values__")
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
        else:
            if isinstance(value, int) and annotation is Stream:
                jit_arg = Stream(value)
                dsl_type = Stream
            else:
                jit_arg_constructor, dsl_type = JitArgumentRegistry.get(type(value))
                if jit_arg_constructor is None:
                    raise TypeError(
                        f"No JitArgument registered for type {type(value).__name__} (parameter '{param_name}')"
                    )
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
        # Marked-static dims contribute their (shape, stride) to the cache key.
        self._cached_dims: frozenset = frozenset()
        self._dyn_leading_dim = -1
        self._dynamic_divisibility = 1
        self._cache_dynamic_shape = bool(dynamic_layout)
        self._is_layout_dynamic = False

        if dynamic_layout:
            try:
                self._mark_layout_dynamic(leading_dim=-1, divisibility=1)
            except RuntimeError:
                # Keep the original fully-static layout when the leading
                # stride-1 dim cannot be inferred unambiguously.
                pass

    @staticmethod
    def _extract_data_ptr(arg):
        if hasattr(arg, "_tensor_keepalive"):
            return arg._tensor_keepalive.data_ptr()
        return arg.data_ptr()

    @staticmethod
    def _auto_leading_dim(strides) -> Optional[int]:
        leading_dims = [i for i, stride in enumerate(strides) if int(stride) == 1]
        return leading_dims[0] if len(leading_dims) == 1 else None

    @staticmethod
    def _dynamic_cache_signature(tensor: torch.Tensor, assumed_align: Optional[int], use_32bit_stride: bool):
        leading_dim = TensorAdaptor._auto_leading_dim(tensor.stride())
        base = (tensor.dtype, assumed_align, use_32bit_stride)
        shape_sig = tuple(int(d) for d in tensor.shape)
        stride_sig = tuple(int(s) for s in tensor.stride())
        if leading_dim is None:
            return base + ("static", shape_sig, stride_sig)
        return base + ("dynamic", tensor.dim(), leading_dim, 1, shape_sig, stride_sig)

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
        # Layout matches C++ buildMemRefDesc: shape i32's then non-leading
        # stride i32/i64's, little-endian packed.
        rank = len(adaptor._orig_shape)
        leading = adaptor._dyn_leading_dim
        use_32bit_stride = bool(adaptor.use_32bit_stride)
        stride_dim_indices = tuple(d for d in range(rank) if d != leading)
        shape_size = rank * 4
        stride_elem = 4 if use_32bit_stride else 8
        buf_ctype = ctypes.c_byte * (shape_size + len(stride_dim_indices) * stride_elem)

        import struct as _struct

        shape_codec = _struct.Struct("<" + "i" * rank) if rank else None
        if stride_dim_indices:
            stride_codec = _struct.Struct("<" + ("i" if use_32bit_stride else "q") * len(stride_dim_indices))
        else:
            stride_codec = None

        def pack_layout_buffer(
            t,
            storage,
            _shape_codec=shape_codec,
            _stride_codec=stride_codec,
            _stride_dims=stride_dim_indices,
            _shape_size=shape_size,
        ):
            tens = t._tensor_keepalive if isinstance(t, cls) else t
            mv = memoryview(storage).cast("b")
            if _shape_codec is not None:
                _shape_codec.pack_into(mv, 0, *tens.shape)
            if _stride_codec is not None:
                _stride_codec.pack_into(mv, _shape_size, *(tens.stride(d) for d in _stride_dims))

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

    @staticmethod
    def raw_cache_signature(tensor: torch.Tensor):
        """Lightweight cache sig from a raw tensor, no DLPack overhead.

        Raw tensors are auto-adapted to dynamic memrefs, but helper code may
        still bake shape-derived layout facts while tracing. Keep shape/stride
        in the key unless the caller explicitly opts into a reusable dynamic
        TensorAdaptor via from_dlpack(...).mark_layout_dynamic(...).
        """
        return TensorAdaptor._dynamic_cache_signature(tensor, None, False)

    def __cache_signature__(self):
        base = (self._orig_dtype, self.assumed_align, self.use_32bit_stride)
        if self._is_layout_dynamic:
            dyn_base = base + (
                "dynamic",
                len(self._orig_shape),
                self._dyn_leading_dim,
                self._dynamic_divisibility,
            )
            if self._cache_dynamic_shape:
                dyn_base += (
                    tuple(int(d) for d in self._orig_shape),
                    tuple(int(s) for s in self._orig_strides),
                )
            if not self._cached_dims:
                return dyn_base
            extras = tuple((d, int(self._orig_shape[d]), int(self._orig_strides[d])) for d in sorted(self._cached_dims))
            return dyn_base + (extras,)
        return base + (
            "static",
            tuple(int(d) for d in self._orig_shape),
            tuple(int(s) for s in self._orig_strides),
        )

    def mark_static(self, dims: Optional[List[int]] = None):
        """Include the listed dims' shape/stride values in the JIT cache key.

        ``dims=None`` marks every dim.  Call this when the compiled kernel
        bakes concrete shape values into the generated IR (for example
        ``num_records`` on a buffer resource derived from the static layout
        cosize), so that calls with different shapes do not reuse a stale
        compiled artifact.  Repeated calls accumulate.

        Returns ``self`` for chaining.
        """
        rank = len(self._orig_shape)
        target = range(rank) if dims is None else dims
        new_dims = set(self._cached_dims)
        for d in target:
            d = int(d)
            if not 0 <= d < rank:
                raise IndexError(f"dim {d} out of range for rank {rank}")
            new_dims.add(d)
        self._cached_dims = frozenset(new_dims)
        return self

    def _mark_layout_dynamic(self, leading_dim: int, divisibility: int):
        self.tensor_adaptor.mark_layout_dynamic(leading_dim, divisibility)
        resolved_leading_dim = self._auto_leading_dim(self._orig_strides) if leading_dim == -1 else int(leading_dim)
        if resolved_leading_dim is None:
            raise RuntimeError("Cannot determine leading dimension")
        self._dyn_leading_dim = resolved_leading_dim
        self._dynamic_divisibility = int(divisibility)
        self._is_layout_dynamic = True
        return self

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
        self.alignment = alignment

    def __get_ir_types__(self):
        ir_type = self.element_type
        if isinstance(ir_type, type) and issubclass(ir_type, Numeric):
            ir_type = self.element_type.ir_type
        return [PointerType.get(ir_type, self.address_space, self.alignment)]

    def __get_c_pointers__(self):
        return [ctypes.cast(ctypes.pointer(self.pointer), ctypes.c_void_p)]

    def __cache_signature__(self):
        return ("PointerAdaptor", self.element_type, str(self.address_space), self.alignment)


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
