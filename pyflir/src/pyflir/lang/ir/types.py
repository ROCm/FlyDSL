"""Shared MLIR IR type helpers for pyflir kernels."""

from __future__ import annotations

from _mlir import ir


class Types:
    # ---- Scalars ----
    @property
    def index(self) -> ir.Type:
        return ir.IndexType.get()

    @property
    def i8(self) -> ir.Type:
        return ir.IntegerType.get_signless(8)

    @property
    def i16(self) -> ir.Type:
        return ir.IntegerType.get_signless(16)

    @property
    def i32(self) -> ir.Type:
        return ir.IntegerType.get_signless(32)

    @property
    def i64(self) -> ir.Type:
        return ir.IntegerType.get_signless(64)

    @property
    def f16(self) -> ir.Type:
        return ir.F16Type.get()

    @property
    def bf16(self) -> ir.Type:
        return ir.BF16Type.get()

    @property
    def f32(self) -> ir.Type:
        return ir.F32Type.get()

    @property
    def f64(self) -> ir.Type:
        return ir.F64Type.get()

    # ROCm kernels in this repo typically use E4M3FN.
    @property
    def f8(self) -> ir.Type:
        return ir.Float8E4M3FNType.get()

    # ---- Vectors ----
    def vec(self, n: int, elem: ir.Type) -> ir.Type:
        return ir.VectorType.get([int(n)], elem)


# Singleton instance for ergonomic use:
#   from pyflir.lang.ir.types import T
#   f16 = T.f16; i32 = T.i32; vec4f32 = T.vec(4, T.f32)
T = Types()


__all__ = ["Types", "T"]


