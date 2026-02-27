import ctypes
from typing import Generic, TypeVar

from flydsl.runtime.device import get_rocm_arch

from .._mlir import ir
from .._mlir.dialects import gpu
from .numeric import (
    BFloat16,
    Boolean,
    Float,
    Float4E2M1FN,
    Float6E2M3FN,
    Float6E3M2FN,
    Float8E4M3,
    Float8E4M3B11FNUZ,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E8M0FNU,
    Float16,
    Float32,
    Float64,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    as_numeric,
)


def _vec(n: int, elem: ir.Type) -> ir.Type:
    return ir.VectorType.get([int(n)], elem)


def default_f8_type() -> ir.Type:
    """Select E4M3 f8 type compatible with the current GPU arch.

    - gfx95* (MI350): FP8 E4M3FN (OCP)
    - gfx94* (MI300): FP8 E4M3FNUZ
    """
    arch = ""
    try:
        arch = str(get_rocm_arch())
    except Exception:
        arch = ""
    if "gfx95" in arch:
        return Float8E4M3FN.ir_type
    return Float8E4M3FNUZ.ir_type


class Types:
    """Property-based MLIR type constructors backed by DSL numeric classes.

    Scalar properties delegate to ``<DslClass>.ir_type`` (single source of
    truth in ``numeric.py``).  Vector shortcuts and ``vec()`` use
    ``ir.VectorType`` directly.

    Usage::

        from flydsl.expr.typing import T
        T.f16            # ir.F16Type
        T.i32x4          # vector<4xi32>
        T.vec(8, T.f16)  # vector<8xf16>
    """

    # ---- Index ----
    @property
    def index(self) -> ir.Type:
        return ir.IndexType.get()

    # ---- Integer scalars & vectors ----
    @property
    def i8(self) -> ir.Type:
        return Int8.ir_type

    @property
    def i8x2(self) -> ir.Type:
        return _vec(2, Int8.ir_type)

    @property
    def i8x4(self) -> ir.Type:
        return _vec(4, Int8.ir_type)

    @property
    def i8x8(self) -> ir.Type:
        return _vec(8, Int8.ir_type)

    @property
    def i8x16(self) -> ir.Type:
        return _vec(16, Int8.ir_type)

    @property
    def i16(self) -> ir.Type:
        return Int16.ir_type

    @property
    def i16x2(self) -> ir.Type:
        return _vec(2, Int16.ir_type)

    @property
    def i16x4(self) -> ir.Type:
        return _vec(4, Int16.ir_type)

    @property
    def i16x8(self) -> ir.Type:
        return _vec(8, Int16.ir_type)

    @property
    def i32(self) -> ir.Type:
        return Int32.ir_type

    @property
    def i32x2(self) -> ir.Type:
        return _vec(2, Int32.ir_type)

    @property
    def i32x4(self) -> ir.Type:
        return _vec(4, Int32.ir_type)

    @property
    def i64(self) -> ir.Type:
        return Int64.ir_type

    @property
    def i64x2(self) -> ir.Type:
        return _vec(2, Int64.ir_type)

    # ---- Float scalars & vectors ----
    @property
    def f16(self) -> ir.Type:
        return Float16.ir_type

    @property
    def f16x2(self) -> ir.Type:
        return _vec(2, Float16.ir_type)

    @property
    def f16x4(self) -> ir.Type:
        return _vec(4, Float16.ir_type)

    @property
    def f16x8(self) -> ir.Type:
        return _vec(8, Float16.ir_type)

    @property
    def bf16(self) -> ir.Type:
        return BFloat16.ir_type

    @property
    def bf16x2(self) -> ir.Type:
        return _vec(2, BFloat16.ir_type)

    @property
    def bf16x4(self) -> ir.Type:
        return _vec(4, BFloat16.ir_type)

    @property
    def bf16x8(self) -> ir.Type:
        return _vec(8, BFloat16.ir_type)

    @property
    def f32(self) -> ir.Type:
        return Float32.ir_type

    @property
    def f32x2(self) -> ir.Type:
        return _vec(2, Float32.ir_type)

    @property
    def f32x4(self) -> ir.Type:
        return _vec(4, Float32.ir_type)

    @property
    def f64(self) -> ir.Type:
        return Float64.ir_type

    # ---- FP8 (arch-dependent shortcut) ----
    @property
    def f8(self) -> ir.Type:
        return default_f8_type()

    @property
    def f8x2(self) -> ir.Type:
        return _vec(2, default_f8_type())

    @property
    def f8x4(self) -> ir.Type:
        return _vec(4, default_f8_type())

    @property
    def f8x8(self) -> ir.Type:
        return _vec(8, default_f8_type())

    @property
    def f8x16(self) -> ir.Type:
        return _vec(16, default_f8_type())

    # ---- Dynamic vector constructor ----
    def vec(self, n: int, elem: ir.Type) -> ir.Type:
        return _vec(n, elem)


T = Types()


__all__ = [
    # MLIR type helpers
    "Types",
    "T",
    "default_f8_type",
    # DSL value types
    "Boolean",
    "Float",
    "BFloat16",
    "Float4E2M1FN",
    "Float6E2M3FN",
    "Float6E3M2FN",
    "Float8E4M3",
    "Float8E4M3B11FNUZ",
    "Float8E4M3FN",
    "Float8E4M3FNUZ",
    "Float8E5M2",
    "Float8E8M0FNU",
    "Float16",
    "Float32",
    "Float64",
    "Int4",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
    "Constexpr",
    "Tensor",
    "Stream",
    "Tuple3D",
    # Utility functions
    "as_numeric",
]


ValueT = TypeVar("ValueT")


class Constexpr(Generic[ValueT]):
    pass


class IntTuple:
    pass


class Basis:
    pass


class Layout:
    pass


class ComposedLayout:
    pass


class CoordTensor:
    pass


class Swizzle:
    pass


class Tensor:
    def __init__(self, value: ir.Value):
        self.value = value

    def __str__(self):
        return f"Tensor({self.value})"

    @classmethod
    def __new_from_ir_values__(cls, values):
        return Tensor(values[0])

    def __extract_ir_values__(self):
        return [self.value]


class Stream:
    def __init__(self, value=None):
        self.value = value

    def __ir_types__(self):
        return [gpu.AsyncTokenType.get()]

    def __c_pointers__(self):
        if self.value is None:
            # default nullptr stream
            return [ctypes.cast(ctypes.pointer(ctypes.c_void_p(0)), ctypes.c_void_p)]
        return [ctypes.cast(ctypes.pointer(ctypes.c_void_p(self.value.cuda_stream)), ctypes.c_void_p)]

    @classmethod
    def __new_from_ir_values__(cls, values):
        return Stream(values[0])

    def __extract_ir_values__(self):
        return [self.value]


class Tuple3D:
    def __init__(self, factory, dtype=Int32):
        self.factory = factory
        self.dtype = dtype

    def __getattr__(self, name):
        if name in ("x", "y", "z"):
            return self.dtype(self.factory(name))
        raise AttributeError(name)

    def __iter__(self):
        return iter((self.x, self.y, self.z))
