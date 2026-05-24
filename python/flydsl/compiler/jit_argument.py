# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import inspect
from typing import Callable, Dict, List, Optional, Tuple, Type, get_origin

import torch

from .._mlir._mlir_libs._mlirDialectsFly import DLTensorAdaptor
from ..expr.typing import Boolean, Constexpr, Float32, Int32, Stream, Tensor
from .protocol import DslType, JitArgument

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
        # Dims listed here contribute their concrete (shape, stride) values
        # to the JIT cache key; other dims are excluded so a single compiled
        # kernel can serve calls with different runtime sizes.
        self._cached_dims: frozenset = frozenset()
        # Default = layout-dynamic memref so the IR is shape-independent and
        # a single compiled kernel serves calls with different sizes.  The
        # fast-path CallState packs an extra layout-buffer slot containing
        # the runtime shape / non-leading stride values; see
        # ``_reusable_slot_spec``.  Silently skipped if the tensor lacks an
        # unambiguous leading stride-1 dim (e.g. 0-rank, broadcast strides);
        # those tensors stay static and behave as before.
        try:
            self.tensor_adaptor.mark_layout_dynamic(-1, 1)
            # Cache resolved leading dim + stride/shape rank so the fast
            # path can pack the layout buffer without re-querying the C++
            # adaptor on every call.
            self._dyn_leading_dim = next(i for i, s in enumerate(self._orig_strides) if int(s) == 1)
            self._is_layout_dynamic = True
        except Exception:
            self._dyn_leading_dim = -1
            self._is_layout_dynamic = False

    @staticmethod
    def _extract_data_ptr(arg):
        return arg.data_ptr()

    @classmethod
    def _reusable_slot_spec(cls, arg):
        """Reusable slot for tensor arguments.

        Returns either:
        * a single ``(ctype, extract)`` tuple for static-memref tensors --
          only the data pointer changes between calls; OR
        * a list ``[(ctype, extract), (ctype_buf, pack_buf), ...]`` for
          dynamic-memref tensors -- the kernel ABI carries an extra layout
          buffer with the runtime shape / non-leading stride values; the
          fast path packs them into a fixed-size byte buffer per call
          (no DLPack export).

        ``extract`` for a scalar slot returns the value to assign to
        ``storage.value``; ``extract`` for a buffer slot writes into
        ``storage`` in place (memmove'd into the packed pointer's target).
        """
        if not hasattr(arg, "data_ptr"):
            return None

        # Build / reuse a TensorAdaptor to inspect the memref dynamism.
        adaptor = arg if isinstance(arg, cls) else cls(arg)
        if not getattr(adaptor, "_is_layout_dynamic", False):
            # Static memref: single slot, existing path.
            return ctypes.c_void_p, cls._extract_data_ptr

        # Dynamic memref: pre-compute the layout-buffer packing plan.
        rank = len(adaptor._orig_shape)
        leading = adaptor._dyn_leading_dim
        use_32bit_stride = bool(adaptor.use_32bit_stride)
        stride_dim_indices = tuple(d for d in range(rank) if d != leading)
        shape_size = rank * 4  # i32 per dim
        stride_elem = 4 if use_32bit_stride else 8
        stride_size = len(stride_dim_indices) * stride_elem
        buf_size = shape_size + stride_size
        buf_ctype = ctypes.c_byte * buf_size

        # Pre-built struct.Struct codecs for the shape and stride sections;
        # both are little-endian packed (matches C++ buildMemRefDesc).
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
            # ``t`` is the raw arg passed by the caller (torch.Tensor or
            # TensorAdaptor wrapping one).
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

        Matches ``__cache_signature__`` for a default-constructed TensorAdaptor
        (no ``mark_static`` calls).
        """
        return (tensor.dtype, None, False, tensor.dim())

    def __cache_signature__(self):
        base = (
            self._orig_dtype,
            self.assumed_align,
            self.use_32bit_stride,
            len(self._orig_shape),
        )
        if not self._cached_dims:
            return base
        # Pack only the marked dims as sorted (dim, shape, stride) triples.
        extras = tuple((d, int(self._orig_shape[d]), int(self._orig_strides[d])) for d in sorted(self._cached_dims))
        return base + (extras,)

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

    def mark_layout_dynamic(self, leading_dim: Optional[int] = None, divisibility: int = 1):
        if leading_dim is None:
            leading_dim = -1
        self.tensor_adaptor.mark_layout_dynamic(leading_dim, divisibility)
        return self


def from_dlpack(
    tensor: torch.Tensor, *, assumed_align: Optional[int] = None, use_32bit_stride: bool = False
) -> TensorAdaptor:
    return TensorAdaptor(tensor, assumed_align, use_32bit_stride)


JitArgumentRegistry.register(bool)(Boolean)
JitArgumentRegistry.register(int)(Int32)
JitArgumentRegistry.register(float)(Float32)
JitArgumentRegistry.register(torch.cuda.Stream)(Stream)
