# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

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


def _is_constexpr_annotation(annotation) -> bool:
    """Check if annotation is Constexpr or Constexpr[T]."""
    if annotation is Constexpr:
        return True
    return get_origin(annotation) is Constexpr


def _is_type_param_annotation(annotation) -> bool:
    """Check if annotation is Type, Type[T]."""
    origin = get_origin(annotation)
    return annotation is Type or origin is Type or origin is type


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

        if annotation is not inspect.Parameter.empty and _is_constexpr_annotation(annotation):
            constexpr_values[param_name] = value
            continue

        if annotation is not inspect.Parameter.empty and _is_type_param_annotation(annotation):
            constexpr_values[param_name] = value
            continue

        is_jit_arg = hasattr(value, "__fly_types__") and hasattr(value, "__fly_ptrs__")
        is_dsl_type = hasattr(value, "__fly_construct__") and hasattr(value, "__fly_values__")
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
        default_dynamic_layout: bool = True,
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
        self._dynamic_shape_dims: set[int] = set()
        self._dynamic_stride_dims: set[int] = set()
        self._shape_divisibility: dict[int, int] = {}
        self._stride_divisibility: dict[int, int] = {}
        if default_dynamic_layout:
            self.mark_layout_dynamic()

    @staticmethod
    def _tensor_address_space(tensor: torch.Tensor):
        device_type = getattr(getattr(tensor, "device", None), "type", None)
        if device_type == "cuda":
            return 1
        return 0

    @classmethod
    def _tensor_metadata(cls, arg):
        if isinstance(arg, cls):
            return (
                arg._orig_dtype,
                arg.tensor_adaptor.address_space,
                tuple(int(d) for d in arg._orig_shape),
                tuple(int(s) for s in arg._orig_strides),
                arg._dynamic_shape_dims,
                arg._dynamic_stride_dims,
                arg.use_32bit_stride,
                arg.tensor_adaptor.data_ptr,
            )
        return (
            arg.dtype,
            cls._tensor_address_space(arg),
            tuple(int(d) for d in arg.shape),
            tuple(int(s) for s in arg.stride()),
            set(range(arg.dim())),
            cls._default_dynamic_stride_dims(arg.shape, arg.stride()),
            False,
            arg.data_ptr(),
        )

    @staticmethod
    def _default_dynamic_stride_dims(shape, strides):
        rank = len(shape)
        dynamic_stride_dims = set(range(rank))
        leading_dim = TensorAdaptor._resolve_dynamic_leading_dim(shape, strides, None)
        if leading_dim is not None:
            dynamic_stride_dims.remove(leading_dim)
        return dynamic_stride_dims

    @classmethod
    def _extract_memref_values(cls, arg):
        _, _, shape, strides, _, _, _, data_ptr = cls._tensor_metadata(arg)
        return int(data_ptr), shape, strides

    @classmethod
    def _reusable_slot_spec(cls, arg):
        """Reusable slots for tensor arguments.

        Dynamic memrefs lower to data-pointer storage plus one layout buffer
        storage. Build those ctypes buffers directly in CallState so the hot
        path does not need an MLIR context.
        """
        if not isinstance(arg, cls) and not hasattr(arg, "__dlpack__"):
            return None
        dtype, address_space, shape, strides, dynamic_shape_dims, dynamic_stride_dims, use_32bit_stride, _ = (
            cls._tensor_metadata(arg)
        )
        stride_bytes = 4 if use_32bit_stride else 8
        layout_size = len(dynamic_shape_dims) * 4 + len(dynamic_stride_dims) * stride_bytes
        return (
            "tensor_memref",
            dtype,
            address_space,
            len(shape),
            tuple(sorted(dynamic_shape_dims)),
            tuple(sorted(dynamic_stride_dims)),
            use_32bit_stride,
            layout_size,
            cls._extract_memref_values,
        )

    def requires_memref_desc(func):
        def wrapper(self, *args, **kwargs):
            self.tensor_adaptor.build_memref_desc()
            return func(self, *args, **kwargs)

        return wrapper

    @requires_memref_desc
    def __fly_types__(self):
        return [self.tensor_adaptor.get_memref_type()]

    @requires_memref_desc
    def __fly_ptrs__(self):
        return self.tensor_adaptor.get_c_pointers()

    @staticmethod
    def _resolve_dynamic_leading_dim(shape, strides, leading_dim: Optional[int] = None):
        if leading_dim is not None:
            resolved = int(leading_dim)
            if resolved < 0 or resolved >= len(shape):
                raise RuntimeError("Cannot determine leading dimension")
            if int(strides[resolved]) != 1:
                raise RuntimeError("Leading dimension must have stride 1")
            return resolved

        stride_one_dims = [i for i, stride in enumerate(strides) if int(stride) == 1]
        if len(stride_one_dims) == 0:
            raise RuntimeError("Cannot determine leading dimension")
        if len(stride_one_dims) == 1:
            return stride_one_dims[0]

        non_unit_stride_one_dims = [i for i in stride_one_dims if int(shape[i]) > 1]
        if len(non_unit_stride_one_dims) == 1:
            return non_unit_stride_one_dims[0]
        raise RuntimeError("Cannot determine leading dimension")

    @staticmethod
    def _layout_cache_signature(
        dtype,
        shape,
        strides,
        leading_dim: Optional[int],
        divisibility: int,
    ):
        rank = len(shape)
        dynamic_shape_dims = set(range(rank))
        if leading_dim is None:
            dynamic_stride_dims = TensorAdaptor._default_dynamic_stride_dims(shape, strides)
        else:
            dynamic_stride_dims = set(range(rank))
            resolved_leading_dim = TensorAdaptor._resolve_dynamic_leading_dim(shape, strides, leading_dim)
            dynamic_stride_dims.remove(resolved_leading_dim)

        shape_divisibility = {i: 1 for i in dynamic_shape_dims}
        stride_divisibility = {i: int(divisibility) for i in dynamic_stride_dims}
        return (
            dtype,
            TensorAdaptor._dim_signature(shape, dynamic_shape_dims, shape_divisibility),
            TensorAdaptor._dim_signature(strides, dynamic_stride_dims, stride_divisibility),
        )

    @staticmethod
    def raw_cache_signature(tensor: torch.Tensor):
        """Lightweight cache sig from a raw tensor, no DLPack overhead.

        Raw tensor arguments use dynamic layout by default, matching the cache
        key behavior that excludes concrete shape values.
        """
        return (
            tensor.dtype,
            TensorAdaptor._tensor_address_space(tensor),
            None,
            False,
            *TensorAdaptor._layout_cache_signature(
                tensor.dtype,
                tuple(int(d) for d in tensor.shape),
                tuple(int(s) for s in tensor.stride()),
                leading_dim=None,
                divisibility=1,
            )[1:],
        )

    @staticmethod
    def _dim_signature(values, dynamic_dims: set[int], divisibility: dict[int, int]):
        sig = []
        for i, value in enumerate(values):
            if i in dynamic_dims:
                sig.append(("dyn", int(divisibility.get(i, 1))))
            else:
                sig.append(("static", int(value)))
        return tuple(sig)

    def __cache_signature__(self):
        return (
            self._orig_dtype,
            self.tensor_adaptor.address_space,
            self.assumed_align,
            self.use_32bit_stride,
            self._dim_signature(self._orig_shape, self._dynamic_shape_dims, self._shape_divisibility),
            self._dim_signature(self._orig_strides, self._dynamic_stride_dims, self._stride_divisibility),
        )

    def mark_layout_dynamic(self, leading_dim: Optional[int] = None, divisibility: int = 1):
        resolved_leading_dim = self._resolve_dynamic_leading_dim(self._orig_shape, self._orig_strides, leading_dim)
        self.tensor_adaptor.mark_layout_dynamic(resolved_leading_dim, divisibility)
        rank = len(self._orig_shape)
        self._dynamic_shape_dims = set(range(rank))
        self._shape_divisibility = {i: 1 for i in range(rank)}
        self._dynamic_stride_dims = set(range(rank))
        self._dynamic_stride_dims.remove(resolved_leading_dim)
        self._stride_divisibility = {i: int(divisibility) for i in self._dynamic_stride_dims}
        return self


def from_dlpack(
    tensor: torch.Tensor, *, assumed_align: Optional[int] = None, use_32bit_stride: bool = False
) -> TensorAdaptor:
    return TensorAdaptor(tensor, assumed_align, use_32bit_stride)


JitArgumentRegistry.register(bool)(Boolean)
JitArgumentRegistry.register(int)(Int32)
JitArgumentRegistry.register(float)(Float32)
JitArgumentRegistry.register(torch.cuda.Stream)(Stream)
