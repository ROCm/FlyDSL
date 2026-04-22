# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ExternFunction — emit ``llvm.call`` to external LLVM bitcode symbols.

Provides the low-level ``ExternFunction`` class that framework integration
layers (e.g. ``mori.ir.flydsl``) use to wrap external device functions as
callable objects inside ``@flyc.kernel`` bodies.

End users typically do **not** import this directly; instead they use the
pre-built wrappers from the integration layer::

    import mori.ir.flydsl as mori_shmem

    @flyc.kernel
    def my_kernel(...):
        pe = mori_shmem.my_pe()        # calls ExternFunction internally
        mori_shmem.quiet_thread()

When ``bitcode_path`` is set on an ``ExternFunction``, the compilation
pipeline links the bitcode automatically via ``rocdl-attach-target``.
When omitted, the pipeline falls back to prefix-based auto-detection
(e.g. ``mori_shmem_*`` → ``mori.ir.bitcode.find_bitcode()``).
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
    "int32":   lambda: IntegerType.get_signless(32),
    "uint32":  lambda: IntegerType.get_signless(32),   # signless; signedness in semantics
    "int64":   lambda: IntegerType.get_signless(64),
    "uint64":  lambda: IntegerType.get_signless(64),
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
        raise ValueError(
            f"ExternFunction: unknown type '{name}'. "
            f"Supported: {list(_TYPE_MAP)} or 'void'."
        )
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
        Mangled C symbol name in the bitcode (e.g. ``"mori_shmem_ptr_p2p"``).
    arg_types:
        List of argument type strings (``"int32"``, ``"uint64"``, etc.).
    ret_type:
        Return type string, or ``"void"`` for functions returning nothing.
    is_pure:
        Hint for future optimisations (currently unused).
    bitcode_path:
        Optional path to the LLVM bitcode file (``.bc``) that provides this
        symbol.  When set, the compilation pipeline automatically links it
        via ``rocdl-attach-target l=<path>``.
    module_init_fn:
        Optional callable ``fn(hipModule_t) -> None`` invoked once on every
        GPU module that the JIT runtime loads for a kernel using this
        ExternFunction.  Use it to initialise device-side globals that the
        bitcode relies on (for example, writing runtime pointers into a
        ``__global__`` struct).  Must be deterministic and idempotent.
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
        self.symbol    = symbol
        self._arg_type_names = list(arg_types)
        self._ret_type_name  = ret_type
        self.is_pure   = is_pure
        self.bitcode_path = bitcode_path
        self.module_init_fn = module_init_fn
        # Cache resolved MLIR types per context (keyed by Context object id).
        self._types_cache: dict = {}
        # Track which (context, gpu.module body) pairs we have declared into.
        # We key by (id(Context.current), id(gpu_module_body)) to avoid false
        # cache hits caused by Python GC reusing object IDs across compilations.
        self._declared_in: set = set()

    # -- type resolution (lazy, per context) --------------------------------
    def _resolve_types(self) -> tuple:
        ctx = ir.Context.current
        key = id(ctx)
        if key not in self._types_cache:
            arg_types = [_resolve_type(t) for t in self._arg_type_names]
            ret_type  = _resolve_type(self._ret_type_name)
            self._types_cache[key] = (arg_types, ret_type)
        return self._types_cache[key]

    # -- declaration in GPU module ------------------------------------------
    def _ensure_declared(self, gpu_module_body) -> None:
        """Add ``llvm.func private`` declaration to the GPU module body if absent."""
        # Use (context_id, body_id) as the cache key to prevent false hits
        # when Python GC recycles object addresses across separate compilations.
        body_id = (id(ir.Context.current), id(gpu_module_body))
        if body_id in self._declared_in:
            return

        arg_types, ret_type = self._resolve_types()

        # Build LLVM function type string: !llvm.func<ret (args...)>
        arg_strs = ", ".join(str(t) for t in arg_types)
        if ret_type is None:
            ret_str = "void"
        else:
            ret_str = str(ret_type)
        fn_type_str = f"!llvm.func<{ret_str} ({arg_strs})>"
        fn_type = ir.Type.parse(fn_type_str)

        with InsertionPoint(gpu_module_body):
            op = llvm.LLVMFuncOp(
                self.symbol,
                TypeAttr.get(fn_type),
                sym_visibility="private",
            )

        self._declared_in.add(body_id)

        from .kernel_function import CompilationContext
        ctx = CompilationContext.get_current()
        if ctx is not None:
            ctx.extern_symbols.add(self.symbol)
            if self.bitcode_path is not None:
                ctx.link_libs.add(self.bitcode_path)
            if self.module_init_fn is not None and \
                    self.module_init_fn not in ctx.post_load_processors:
                ctx.post_load_processors.append(self.module_init_fn)

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
            raise TypeError(
                f"ExternFunction '{self.symbol}' expects {len(arg_types)} "
                f"argument(s), got {len(args)}"
            )

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
                    raise ValueError(
                        f"ExternFunction arg must produce exactly 1 ir.Value, got {len(vals)}"
                    )
                val = vals[0]
            else:
                raise TypeError(
                    f"ExternFunction: cannot use arg of type {type(a).__name__} as ir.Value"
                )

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
        bc = f", bitcode={self.bitcode_path!r}" if self.bitcode_path else ""
        return (
            f"ExternFunction(symbol={self.symbol!r}, "
            f"args={self._arg_type_names}, ret={self._ret_type_name!r}{bc})"
        )
