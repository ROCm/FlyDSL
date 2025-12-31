"""LLVM dialect re-export for FLIR tests."""

from _mlir.dialects.llvm import *  # noqa: F401,F403


def call_intrinsic(*args, **kwargs):
    """`llvm.call_intrinsic` wrapper that accepts ArithValue / wrappers in operands.

    We preserve the upstream signature by forwarding all args/kwargs, but we
    rewrite the `operands` positional argument (index 2) when present.
    """
    if len(args) >= 3:
        from _mlir.dialects import llvm as _llvm
        from . import arith as _arith_ext

        args = list(args)
        ops = args[2]
        if isinstance(ops, (list, tuple)):
            args[2] = [_arith_ext.unwrap(v) for v in ops]
        return _llvm.call_intrinsic(*args, **kwargs)

    # Fallback: if upstream signature changes, just delegate.
    from _mlir.dialects import llvm as _llvm

    return _llvm.call_intrinsic(*args, **kwargs)


