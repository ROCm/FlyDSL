# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""External function expression API.

This module provides a pure FFI callable for use inside ``@flyc.kernel`` bodies.
It declares external LLVM symbols and emits ``llvm.call`` operations, but does
not attach external bitcode or module-load initialization metadata. Use
``flydsl.compiler.extern_link`` for those compiler/runtime concerns.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .._mlir import ir
from .._mlir.dialects import llvm
from .._mlir.ir import (
    Attribute,
    DenseI32ArrayAttr,
    FlatSymbolRefAttr,
    InsertionPoint,
    IntegerType,
    TypeAttr,
)

_TYPE_MAP = {
    "int32": lambda: IntegerType.get_signless(32),
    "uint32": lambda: IntegerType.get_signless(32),
    "int64": lambda: IntegerType.get_signless(64),
    "uint64": lambda: IntegerType.get_signless(64),
    "float32": lambda: ir.F32Type.get(),
    "float64": lambda: ir.F64Type.get(),
}
_VOID_RET = "void"


def _resolve_type(name: str) -> Optional[ir.Type]:
    if name == _VOID_RET:
        return None
    factory = _TYPE_MAP.get(name)
    if factory is None:
        raise ValueError(f"ffi: unknown type {name!r}. Supported: {list(_TYPE_MAP)} or 'void'.")
    return factory()


def _get_no_bundle() -> DenseI32ArrayAttr:
    return DenseI32ArrayAttr.get([])


class ExternFunction:
    """Callable that emits an ``llvm.call`` op inside a ``@flyc.kernel`` body."""

    def __init__(
        self,
        symbol: str,
        arg_types: List[str],
        ret_type: str,
        is_pure: bool = False,
        bitcode_path: Optional[str] = None,
        module_init_fn: Optional[Any] = None,
    ):
        self.symbol = symbol
        self._arg_type_names = list(arg_types)
        self._ret_type_name = ret_type
        self.is_pure = is_pure

    def _resolve_types(self) -> tuple:
        arg_types = [_resolve_type(t) for t in self._arg_type_names]
        ret_type = _resolve_type(self._ret_type_name)
        return arg_types, ret_type

    def _already_declared(self, gpu_module_body) -> bool:
        for op in gpu_module_body.operations:
            if op.operation.name != "llvm.func":
                continue
            attrs = op.operation.attributes
            if "sym_name" not in attrs:
                continue
            name_attr = attrs["sym_name"]
            name = getattr(name_attr, "value", None)
            if name is None:
                name = str(name_attr).strip('"')
            if name == self.symbol:
                return True
        return False

    def _ensure_declared(self, gpu_module_body) -> None:
        if self._already_declared(gpu_module_body):
            return

        arg_types, ret_type = self._resolve_types()
        arg_strs = ", ".join(str(t) for t in arg_types)
        ret_str = "void" if ret_type is None else str(ret_type)
        fn_type = ir.Type.parse(f"!llvm.func<{ret_str} ({arg_strs})>")

        with InsertionPoint(gpu_module_body):
            llvm.LLVMFuncOp(
                self.symbol,
                TypeAttr.get(fn_type),
                sym_visibility="private",
            )

    def __call__(self, *args: Any) -> Any:
        from ..compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        if ctx is None or ctx.gpu_module_body is None:
            raise RuntimeError("ffi can only be called inside a @flyc.kernel body.")

        self._ensure_declared(ctx.gpu_module_body)
        arg_types, ret_type = self._resolve_types()

        if len(args) != len(arg_types):
            raise TypeError(f"ffi {self.symbol!r} expects {len(arg_types)} argument(s), got {len(args)}")

        from .._mlir.dialects import llvm as _llvm
        from .._mlir.ir import IntegerAttr

        raw_args: List[ir.Value] = []
        for arg_pos, arg in enumerate(args):
            expected_type = arg_types[arg_pos]

            if isinstance(arg, int):
                target_type = expected_type or IntegerType.get_signless(64)
                raw_args.append(_llvm.ConstantOp(target_type, IntegerAttr.get(target_type, arg)).result)
                continue

            if isinstance(arg, ir.Value):
                value = arg
            elif hasattr(arg, "__fly_values__"):
                values = arg.__fly_values__()
                if len(values) != 1:
                    raise ValueError(f"ffi argument must produce exactly 1 ir.Value, got {len(values)}")
                value = values[0]
            else:
                raise TypeError(f"ffi: cannot use argument of type {type(arg).__name__} as ir.Value")

            if expected_type is not None and value.type != expected_type:
                from .._mlir.dialects import arith as _arith

                value_is_int = isinstance(value.type, IntegerType)
                expected_is_int = isinstance(expected_type, IntegerType)
                if value_is_int and expected_is_int:
                    value_bits = IntegerType(value.type).width
                    expected_bits = IntegerType(expected_type).width
                    if value_bits > expected_bits:
                        value = _arith.TruncIOp(expected_type, value).result
                    elif value_bits < expected_bits:
                        value = _arith.ExtUIOp(expected_type, value).result

            raw_args.append(value)

        no_bundle = _get_no_bundle()
        callee_ref = FlatSymbolRefAttr.get(self.symbol)
        if ret_type is None:
            from .._mlir.ir import Operation

            Operation.create(
                "llvm.call",
                results=[],
                operands=raw_args,
                attributes={
                    "callee": callee_ref,
                    "operandSegmentSizes": DenseI32ArrayAttr.get([len(raw_args), 0]),
                    "op_bundle_sizes": no_bundle,
                    "CConv": Attribute.parse("#llvm.cconv<ccc>"),
                    "TailCallKind": Attribute.parse("#llvm.tailcallkind<none>"),
                    "fastmathFlags": Attribute.parse("#llvm.fastmath<none>"),
                },
            )
            return None

        call = llvm.CallOp(
            ret_type,
            raw_args,
            [],
            no_bundle,
            callee=callee_ref,
        )
        return call.result

    def __repr__(self) -> str:
        return f"ffi(symbol={self.symbol!r}, args={self._arg_type_names}, ret={self._ret_type_name!r})"


ffi = ExternFunction

__all__ = ["ffi", "ExternFunction"]
