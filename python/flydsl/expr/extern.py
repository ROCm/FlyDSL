# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""External function expression API.

This module provides a pure FFI callable for use inside ``@flyc.kernel`` bodies.
It declares external LLVM symbols and emits ``llvm.call`` operations, but does
not attach external bitcode or module-load initialization metadata. Use
``flydsl.compiler.extern_link`` for those compiler/runtime concerns.

``"ptr"`` and ``"ptr<N>"`` declare ``!llvm.ptr`` and ``!llvm.ptr<N>``. An ``i64``
SSA address is converted with ``llvm.inttoptr``; a Python int is not.
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
    IntegerAttr,
    IntegerType,
    TypeAttr,
)
from .meta import dsl_loc_tracing

_TYPE_MAP = {
    "int32": lambda: IntegerType.get_signless(32),
    "uint32": lambda: IntegerType.get_signless(32),
    "int64": lambda: IntegerType.get_signless(64),
    "uint64": lambda: IntegerType.get_signless(64),
    "float32": lambda: ir.F32Type.get(),
    "float64": lambda: ir.F64Type.get(),
}
_VOID_RET = "void"
_PTR_ADDR_BITS = 64


def _pointer_type(name: str) -> Optional[ir.Type]:
    """``ptr`` is ``!llvm.ptr``. ``ptr<N>`` is ``!llvm.ptr<N>``."""
    if name == "ptr":
        return llvm.PointerType.get()
    if not (name.startswith("ptr<") and name.endswith(">")):
        return None
    space = name[len("ptr<") : -1]
    if not space.isdigit():
        raise ValueError(f"ffi: pointer address space in {name!r} must be a non-negative integer, as in 'ptr<1>'.")
    return llvm.PointerType.get(int(space))


def _is_llvm_pointer(ty: Optional[ir.Type]) -> bool:
    return isinstance(ty, llvm.PointerType)


def _resolve_type(name: str) -> Optional[ir.Type]:
    if name == _VOID_RET:
        return None
    pointer = _pointer_type(name)
    if pointer is not None:
        return pointer
    factory = _TYPE_MAP.get(name)
    if factory is None:
        raise ValueError(
            f"ffi: unknown type {name!r}. Supported: {list(_TYPE_MAP)}, 'void', 'ptr', or 'ptr<address-space>'."
        )
    return factory()


def _address_to_llvm_ptr(value: ir.Value, ptr_type: ir.Type) -> ir.Value:
    """Interpret an i64 SSA address as ``ptr_type``. Other widths are rejected."""
    if not isinstance(value.type, IntegerType) or IntegerType(value.type).width != _PTR_ADDR_BITS:
        raise TypeError(f"ffi pointer argument expects an i{_PTR_ADDR_BITS} address or {ptr_type}, got {value.type}.")
    return llvm.inttoptr(ptr_type, value)


def _declared_func_type(op) -> ir.Type:
    return op.operation.attributes["function_type"].value


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
        if bitcode_path is not None or module_init_fn is not None:
            raise TypeError(
                "flydsl.expr.extern.ffi is link-agnostic and does not accept "
                "bitcode_path/module_init_fn. Wrap it with "
                "flydsl.compiler.extern_link.link_extern(..., bitcode_path=..., "
                "module_init_fn=...) instead."
            )
        self.symbol = symbol
        self._arg_type_names = list(arg_types)
        self._ret_type_name = ret_type
        self.is_pure = is_pure

    def _resolve_types(self) -> tuple:
        arg_types = [_resolve_type(t) for t in self._arg_type_names]
        ret_type = _resolve_type(self._ret_type_name)
        return arg_types, ret_type

    def _find_declaration(self, gpu_module_body):
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
                return op
        return None

    def _function_type(self) -> ir.Type:
        arg_types, ret_type = self._resolve_types()
        arg_strs = ", ".join(str(t) for t in arg_types)
        ret_str = "void" if ret_type is None else str(ret_type)
        return ir.Type.parse(f"!llvm.func<{ret_str} ({arg_strs})>")

    def _ensure_declared(self, gpu_module_body) -> None:
        fn_type = self._function_type()
        existing = self._find_declaration(gpu_module_body)
        if existing is not None:
            declared = _declared_func_type(existing)
            if declared != fn_type:
                raise TypeError(
                    f"ffi {self.symbol!r} is already declared as {declared}, cannot redeclare as {fn_type}."
                )
            return

        with InsertionPoint(gpu_module_body):
            llvm.LLVMFuncOp(
                self.symbol,
                TypeAttr.get(fn_type),
                sym_visibility="private",
            )

    @dsl_loc_tracing
    def __call__(self, *args: Any) -> Any:
        from ..compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        if ctx is None or ctx.gpu_module_body is None:
            raise RuntimeError("ffi can only be called inside a @flyc.kernel body.")

        self._ensure_declared(ctx.gpu_module_body)
        arg_types, ret_type = self._resolve_types()

        if len(args) != len(arg_types):
            raise TypeError(f"ffi {self.symbol!r} expects {len(arg_types)} argument(s), got {len(args)}")

        from .numeric import Numeric

        raw_args: List[ir.Value] = []
        for arg_pos, arg in enumerate(args):
            expected_type = arg_types[arg_pos]

            if isinstance(arg, Numeric) and isinstance(arg.value, (bool, int)):
                arg = int(arg.value)

            if isinstance(arg, int):
                if _is_llvm_pointer(expected_type):
                    raise TypeError(
                        f"ffi {self.symbol!r} argument {self._arg_type_names[arg_pos]!r} is a pointer; "
                        "pass an SSA address, not a Python int."
                    )
                target_type = expected_type or IntegerType.get_signless(64)
                raw_args.append(llvm.ConstantOp(target_type, IntegerAttr.get(target_type, arg)).result)
                continue

            if isinstance(arg, ir.Value):
                value = arg
            elif hasattr(arg, "__extract_to_ir_values__"):
                values = arg.__extract_to_ir_values__()
                if len(values) != 1:
                    raise ValueError(f"ffi argument must produce exactly 1 ir.Value, got {len(values)}")
                value = values[0]
            else:
                raise TypeError(f"ffi: cannot use argument of type {type(arg).__name__} as ir.Value")

            if expected_type is not None and value.type != expected_type:
                if _is_llvm_pointer(expected_type) or _is_llvm_pointer(value.type):
                    if _is_llvm_pointer(expected_type) and not _is_llvm_pointer(value.type):
                        value = _address_to_llvm_ptr(value, expected_type)
                    else:
                        raise TypeError(
                            f"ffi {self.symbol!r} argument {self._arg_type_names[arg_pos]!r} "
                            f"expects {expected_type}, got {value.type}."
                        )
                else:
                    from .._mlir.dialects import arith as _arith

                    value_is_int = isinstance(value.type, IntegerType)
                    expected_is_int = isinstance(expected_type, IntegerType)
                    if value_is_int and expected_is_int:
                        value_bits = IntegerType(value.type).width
                        expected_bits = IntegerType(expected_type).width
                        if value_bits > expected_bits:
                            value = _arith.TruncIOp(expected_type, value).result
                        elif value_bits < expected_bits:
                            # Use sign-extension for signed type names,
                            # zero-extension for unsigned.
                            type_name = self._arg_type_names[arg_pos]
                            if type_name.startswith("int"):
                                value = _arith.ExtSIOp(expected_type, value).result
                            else:
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
