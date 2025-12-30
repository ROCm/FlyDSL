"""Shared MLIR IR type helpers for pyflir kernels."""

from __future__ import annotations

from _mlir import ir
from _mlir.extras.types import *



class Types:
    # ---- Scalars ----
    @property
    def index(self) -> ir.Type:
        return ir.IndexType.get()

    @property
    def i8(self) -> ir.Type:
        return ir.IntegerType.get_signless(8)
    @property
    def i8x2(self) -> ir.Type:
        return ir.VectorType.get([2], ir.IntegerType.get_signless(8))
    @property
    def i8x4(self) -> ir.Type:
        return ir.VectorType.get([4], ir.IntegerType.get_signless(8))
    @property
    def i8x8(self) -> ir.Type:
        return ir.VectorType.get([8], ir.IntegerType.get_signless(8))
    @property
    def i8x16(self) -> ir.Type:
        return ir.VectorType.get([16], ir.IntegerType.get_signless(8))

    @property
    def i16(self) -> ir.Type:
        return ir.IntegerType.get_signless(16)
    @property
    def i16x2(self) -> ir.Type:
        return ir.VectorType.get([2], ir.IntegerType.get_signless(16))
    @property
    def i16x4(self) -> ir.Type:
        return ir.VectorType.get([4], ir.IntegerType.get_signless(16))
    @property
    def i16x8(self) -> ir.Type:
        return ir.VectorType.get([8], ir.IntegerType.get_signless(16))

    @property
    def i32(self) -> ir.Type:
        return ir.IntegerType.get_signless(32)
    @property
    def i32x2(self) -> ir.Type:
        return ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    @property
    def i32x4(self) -> ir.Type:
        return ir.VectorType.get([4], ir.IntegerType.get_signless(32))

    @property
    def i64(self) -> ir.Type:
        return ir.IntegerType.get_signless(64)
    @property
    def i64x2(self) -> ir.Type:
        return ir.VectorType.get([2], ir.IntegerType.get_signless(64))

    @property
    def f16(self) -> ir.Type:
        return ir.F16Type.get()
    @property
    def f16x2(self) -> ir.Type:
        return ir.VectorType.get([2], ir.F16Type.get())
    @property
    def f16x4(self) -> ir.Type:
        return ir.VectorType.get([4], ir.F16Type.get())
    @property
    def f16x8(self) -> ir.Type:
        return ir.VectorType.get([8], ir.F16Type.get())

    @property
    def bf16(self) -> ir.Type:
        return ir.BF16Type.get()
    @property
    def bf16x2(self) -> ir.Type:
        return ir.VectorType.get([2], ir.BF16Type.get())
    @property
    def bf16x4(self) -> ir.Type:
        return ir.VectorType.get([4], ir.BF16Type.get())
    @property
    def bf16x8(self) -> ir.Type:
        return ir.VectorType.get([8], ir.BF16Type.get())
    @property
    def bf16x2(self) -> ir.Type:
        return ir.VectorType.get([2], ir.BF16Type.get())

    @property
    def f32(self) -> ir.Type:
        return ir.F32Type.get()

    @property
    def f32x2(self) -> ir.Type:
        return ir.VectorType.get([4], ir.F32Type.get())

    @property
    def f32x4(self) -> ir.Type:
        return ir.VectorType.get([4], ir.F32Type.get())

    @property
    def f64(self) -> ir.Type:
        return ir.F64Type.get()

    # ROCm kernels in this repo typically use E4M3FN.
    @property
    def f8(self) -> ir.Type:
        return ir.Float8E4M3FNType.get()
    @property
    def f8x2(self) -> ir.Type:
        return ir.VectorType.get([2], ir.Float8E4M3FNType.get())
    @property
    def f8x4(self) -> ir.Type:
        return ir.VectorType.get([4], ir.Float8E4M3FNType.get())
    @property
    def f8x8(self) -> ir.Type:
        return ir.VectorType.get([8], ir.Float8E4M3FNType.get())
    @property
    def f8x16(self) -> ir.Type:
        return ir.VectorType.get([16], ir.Float8E4M3FNType.get())

    # ---- Vectors ----
    def vec(self, n: int, elem: ir.Type) -> ir.Type:
        return ir.VectorType.get([int(n)], elem)


# Singleton instance for ergonomic use:
#   from pyflir.lang.ir.types import T
#   f16 = T.f16; i32 = T.i32; vec4f32 = T.vec(4, T.f32)
T = Types()


__all__ = ["Types", "T", "vec", "i8x2", "i8x4", "i8x8", "i8x16", "i16x2", "i16x4", 
            "i16x8", "i32x2", "i32x4", "f16", "bf16", "bf16x2", "bf16x4", "bf16x8", "f32", 
            "f32x2", "f32x4", "f64", "f8", "f8x2", "f8x4", "f8x8", "f8x16"]







