# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, get_origin

import torch

from .._mlir._mlir_libs._fly import DLTensorAdaptor
from ..expr.typing import Constexpr, Int32, Stream, Tensor
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
    return annotation is Type or get_origin(annotation) is Type


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

        is_jit_arg = hasattr(value, '__fly_types__') and hasattr(value, '__fly_ptrs__')
        is_dsl_type = hasattr(value, '__fly_construct__') and hasattr(value, '__fly_values__')
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
    ):
        self._tensor_keepalive = tensor
        dlpack_tensor = tensor
        if _FLOAT8_DTYPES and tensor.dtype in _FLOAT8_DTYPES:
            dlpack_tensor = tensor.view(torch.uint8)
            self._tensor_keepalive = dlpack_tensor

        self.tensor_adaptor = DLTensorAdaptor(dlpack_tensor.__dlpack__(stream=-1), assumed_align, use_32bit_stride)
        self.assumed_align = assumed_align
        self.use_32bit_stride = use_32bit_stride
        self._orig_dtype = tensor.dtype
        self._orig_shape = tensor.shape
        self._orig_strides = tensor.stride()

    @staticmethod
    def _extract_data_ptr(arg):
        return arg.data_ptr()

    @classmethod
    def _reusable_slot_spec(cls, arg):
        """Reusable slot for tensor arguments.

        For bare-pointer calling convention, only the data pointer changes
        between calls with the same shape/dtype/strides.
        """
        if not hasattr(arg, 'data_ptr'):
            return None
        return ctypes.c_void_p, cls._extract_data_ptr

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

    def __cache_signature__(self):
        return (
            self._orig_dtype,
            tuple(self._orig_shape),
            tuple(self._orig_strides),
            self.assumed_align,
            self.use_32bit_stride,
        )

    def mark_layout_dynamic(self, leading_dim: Optional[int] = None, divisibility: int = 1):
        if leading_dim is None:
            leading_dim = -1
        self.tensor_adaptor.mark_layout_dynamic(leading_dim, divisibility)
        return self


def from_dlpack(
    tensor: torch.Tensor, *, assumed_align: Optional[int] = None, use_32bit_stride: bool = False
) -> TensorAdaptor:
    return TensorAdaptor(tensor, assumed_align, use_32bit_stride)


JitArgumentRegistry.register(int)(Int32)
JitArgumentRegistry.register(torch.cuda.Stream)(Stream)
