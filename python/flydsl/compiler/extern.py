# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ExternFunction — emit ``llvm.call`` to external symbols.

Provides the low-level expression object that wraps external device functions
as callable objects inside ``@flyc.kernel`` bodies.

End users typically do **not** import this directly; instead they use the
pre-built wrappers from an integration layer::

    from my_framework.flydsl_ops import device_func

    @flyc.kernel
    def my_kernel(...):
        value = device_func()        # calls ExternFunction internally

This module is intentionally link-agnostic: it declares the symbol and emits
the call only.  Use ``flydsl.compiler.extern_link`` to attach external bitcode
and post-load module initialization metadata.
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

# ---------------------------------------------------------------------------
# Type mapping: mori ABI type strings → MLIR ir.Type factories
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "int32": lambda: IntegerType.get_signless(32),
    "uint32": lambda: IntegerType.get_signless(32),  # signless; signedness in semantics
    "int64": lambda: IntegerType.get_signless(64),
    "uint64": lambda: IntegerType.get_signless(64),
    "float32": lambda: ir.F32Type.get(),
    "float64": lambda: ir.F64Type.get(),
}

# Sentinel for void return
_VOID_RET = "void"


def _resolve_type(name: str) -> Optional[ir.Type]:
    """Return MLIR type for *name*, or None for void."""
    if name == _VOID_RET:
        return None
    factory = _TYPE_MAP.get(name)
    if factory is None:
        raise ValueError(f"ExternFunction: unknown type '{name}'. " f"Supported: {list(_TYPE_MAP)} or 'void'.")
    return factory()


# ---------------------------------------------------------------------------
# Shared no-bundle attribute (op_bundle_sizes = [])
# ---------------------------------------------------------------------------
def _get_no_bundle() -> DenseI32ArrayAttr:
    """Return a fresh ``op_bundle_sizes = []`` attribute for the current MLIR context."""
    return DenseI32ArrayAttr.get([])


# ---------------------------------------------------------------------------
# ExternFunction
# ---------------------------------------------------------------------------
class ExternFunction:
    """Callable that emits an ``llvm.call`` op inside a ``@flyc.kernel`` body.

    When called during kernel compilation:
    1. Ensures ``llvm.func private @<symbol>(...)`` is declared in the GPU module.
    2. Emits ``llvm.call @<symbol>(...)`` at the current insertion point.
    3. Returns the result as an ``ir.Value`` (auto-cast to ``ArithValue``).

    Parameters
    ----------
    symbol:
        Mangled C symbol name in the linked bitcode.
    arg_types:
        List of argument type strings (``"int32"``, ``"uint64"``, etc.).
    ret_type:
        Return type string, or ``"void"`` for functions returning nothing.
    is_pure:
        Hint for future optimisations (currently unused).
    bitcode_path, module_init_fn:
        Accepted for source compatibility but ignored by this FFI layer. Attach
        these through ``flydsl.compiler.extern_link.link_extern``.
    """

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

    # -- type resolution (lazy, per context) --------------------------------
    def _resolve_types(self) -> tuple:
        arg_types = [_resolve_type(t) for t in self._arg_type_names]
        ret_type = _resolve_type(self._ret_type_name)
        return arg_types, ret_type

    # -- declaration in GPU module ------------------------------------------
    def _already_declared(self, gpu_module_body) -> bool:
        """Return True iff an llvm.func with our symbol already lives in *body*.

        We scan the body's top-level ops instead of memoising on object
        ``id()`` — ExternFunction wrappers are module-level singletons
        whose lifetime spans every JIT compilation in the process, so an
        id-keyed cache would (a) grow without bound and (b) risk false
        hits once CPython recycles a gpu.module address across separate
        compilations.  The scan cost is O(#top-level-ops-in-gpu.module),
        typically <30 ops, i.e. microseconds per ensure_declared — lost
        in the noise of MLIR→LLVM compile time.
        """
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
        """Add ``llvm.func private`` declaration to *body* if absent, then
        register the bitcode / post-load side-effects on the current
        CompilationContext.  Idempotent across repeated calls within
        the same kernel compile."""
        if not self._already_declared(gpu_module_body):
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

    # -- callable interface -------------------------------------------------
    def __call__(self, *args: Any) -> Any:
        """Emit ``llvm.call`` at the current insertion point.

        Args are expected to be MLIR ``ir.Value`` objects (or objects with
        ``__fly_values__()``).  Returns the call result (or ``None`` for void).
        """
        from .kernel_function import CompilationContext  # local import

        ctx = CompilationContext.get_current()
        if ctx is None or ctx.gpu_module_body is None:
            raise RuntimeError(
                "ExternFunction can only be called inside a @flyc.kernel body. "
                "Make sure the kernel is compiled via @flyc.jit."
            )

        # Ensure the function is declared in the GPU module.
        self._ensure_declared(ctx.gpu_module_body)

        # Resolve types for this context
        arg_types, ret_type = self._resolve_types()

        if len(args) != len(arg_types):
            raise TypeError(f"ExternFunction '{self.symbol}' expects {len(arg_types)} " f"argument(s), got {len(args)}")

        # Unwrap arguments to raw ir.Value and coerce types as needed.
        from .._mlir.dialects import llvm as _llvm
        from .._mlir.ir import IntegerAttr

        raw_args: List[ir.Value] = []
        for arg_pos, a in enumerate(args):
            expected_type = arg_types[arg_pos]

            if isinstance(a, int):
                # Materialize Python int as constant of the expected arg type
                tgt = expected_type or IntegerType.get_signless(64)
                raw_args.append(_llvm.ConstantOp(tgt, IntegerAttr.get(tgt, a)).result)
                continue

            if isinstance(a, ir.Value):
                val = a
            elif hasattr(a, "__fly_values__"):
                vals = a.__fly_values__()
                if len(vals) != 1:
                    raise ValueError(f"ExternFunction arg must produce exactly 1 ir.Value, got {len(vals)}")
                val = vals[0]
            else:
                raise TypeError(f"ExternFunction: cannot use arg of type {type(a).__name__} as ir.Value")

            # Coerce type if needed (e.g. i64 → i32 via trunc, i32 → i64 via zext)
            if expected_type is not None and val.type != expected_type:
                from .._mlir.dialects import arith as _arith

                val_is_int = isinstance(val.type, IntegerType)
                exp_is_int = isinstance(expected_type, IntegerType)
                if val_is_int and exp_is_int:
                    v_bits = IntegerType(val.type).width
                    e_bits = IntegerType(expected_type).width
                    if v_bits > e_bits:
                        val = _arith.TruncIOp(expected_type, val).result
                    elif v_bits < e_bits:
                        val = _arith.ExtUIOp(expected_type, val).result

            raw_args.append(val)

        # Emit llvm.call.
        no_bundle = _get_no_bundle()
        callee_ref = FlatSymbolRefAttr.get(self.symbol)

        if ret_type is None:
            # void return: Operation.create is required because the
            # auto-generated llvm.CallOp Python binding always expects
            # at least one result type.  The explicit attribute dict
            # mirrors what CallOp would emit; if the MLIR version
            # changes attribute names this will need updating.
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
        else:
            call = llvm.CallOp(
                ret_type,
                raw_args,
                [],
                no_bundle,
                callee=callee_ref,
            )
            return call.result

    def __repr__(self) -> str:
        return f"ExternFunction(symbol={self.symbol!r}, " f"args={self._arg_type_names}, ret={self._ret_type_name!r})"
