import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, get_origin

import torch

from .._mlir._mlir_libs._fly import DLTensorAdaptor
from ..expr.typing import Constexpr, Int32, Stream, Tensor
from .protocol import DslType, JitArgument


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
        return cls.registry.get(py_type, (None, None))

    @classmethod
    def get_dsl_type(cls, jit_arg_type: type) -> Type[DslType]:
        return cls.jit_arg2dsl_type[jit_arg_type]


def _is_constexpr_annotation(annotation) -> bool:
    if annotation is Constexpr:
        return True
    return get_origin(annotation) is Constexpr


def _is_type_param_annotation(annotation) -> bool:
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

        if isinstance(value, JitArgument) and isinstance(value, DslType):
            jit_arg = value
            dsl_type = type(value)
        elif isinstance(value, JitArgument):
            jit_arg = value
            dsl_type = JitArgumentRegistry.get_dsl_type(type(value))
            if dsl_type is None:
                raise TypeError(
                    f"No DslType registered for JitArgument type {type(value).__name__} (parameter '{param_name}')"
                )
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


@JitArgumentRegistry.register(torch.Tensor, dsl_type=Tensor)
class TensorAdaptor:
    _default_use_standard_memref = False

    def __init__(
        self,
        tensor: torch.Tensor,
        assumed_align: Optional[int] = None,
        use_32bit_stride: bool = False,
    ):
        self.tensor_adaptor = DLTensorAdaptor(tensor.__dlpack__(), assumed_align, use_32bit_stride)
        self.assumed_align = assumed_align
        self.use_32bit_stride = use_32bit_stride
        self.use_standard_memref = self.__class__._default_use_standard_memref

    def requires_memref_desc(func):
        def wrapper(self, *args, **kwargs):
            self.tensor_adaptor.build_memref_desc()
            return func(self, *args, **kwargs)

        return wrapper

    @requires_memref_desc
    def __ir_types__(self):
        if self.use_standard_memref:
            from .._mlir import ir
            fly_str = str(self.tensor_adaptor.get_memref_type())
            dtype_map = {
                "f32": ir.F32Type.get, "f16": ir.F16Type.get, "bf16": ir.BF16Type.get,
                "f64": ir.F64Type.get, "si8": lambda: ir.IntegerType.get_signless(8),
                "si32": lambda: ir.IntegerType.get_signless(32),
                "si16": lambda: ir.IntegerType.get_signless(16),
            }
            for key, factory in dtype_map.items():
                if fly_str.startswith(f"!fly.memref<{key}"):
                    with ir.Location.unknown():
                        return [ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], factory())]
        return [self.tensor_adaptor.get_memref_type()]

    @requires_memref_desc
    def __c_pointers__(self):
        if self.use_standard_memref:
            return self._standard_memref_c_pointers()
        return self.tensor_adaptor.get_c_pointers()

    def _standard_memref_c_pointers(self):
        """Build standard MLIR memref descriptor fields for rank-1 memref<?xT>.

        Returns 5 pointers: one for each field of {alloc_ptr, align_ptr, offset, size, stride}.
        """
        import ctypes
        fly_ptrs = self.tensor_adaptor.get_c_pointers()
        pp = ctypes.cast(fly_ptrs[0], ctypes.POINTER(ctypes.c_void_p))
        gpu_ptr_val = pp[0]

        total_bytes = self.tensor_adaptor.size_in_bytes()
        fly_str = str(self.tensor_adaptor.get_memref_type())
        elem_bytes = 4
        for k, sz in [("f64", 8), ("si64", 8), ("f32", 4), ("si32", 4),
                       ("f16", 2), ("bf16", 2), ("si16", 2), ("si8", 1)]:
            if k in fly_str:
                elem_bytes = sz
                break
        numel = total_bytes // elem_bytes if elem_bytes > 0 else total_bytes

        class MemRefDescriptor(ctypes.Structure):
            _fields_ = [
                ("allocated", ctypes.c_void_p),
                ("aligned", ctypes.c_void_p),
                ("offset", ctypes.c_int64),
                ("size0", ctypes.c_int64),
                ("stride0", ctypes.c_int64),
            ]

        self._desc = MemRefDescriptor(
            allocated=gpu_ptr_val, aligned=gpu_ptr_val,
            offset=0, size0=numel, stride0=1,
        )
        base = ctypes.addressof(self._desc)
        return [ctypes.c_void_p(base + getattr(MemRefDescriptor, f).offset)
                for f in ("allocated", "aligned", "offset", "size0", "stride0")]

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
