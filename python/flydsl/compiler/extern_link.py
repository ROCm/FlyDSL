# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Link metadata helpers for external device functions."""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..expr.extern import ExternFunction as FFIExternFunction

_EXTERN_RESOLVERS = []


def register_extern_resolver(resolver: Callable):
    """Register a resolver that can fill bitcode/init metadata for externs."""
    _EXTERN_RESOLVERS.append(resolver)
    return resolver


def _apply_resolvers(symbol: str, bitcode_path, module_init_fn):
    for resolver in _EXTERN_RESOLVERS:
        bitcode_path, module_init_fn = resolver(symbol, bitcode_path, module_init_fn)
    return bitcode_path, module_init_fn


def _register_link_metadata(bitcode_path: Optional[str], module_init_fn: Optional[Any]) -> None:
    from .kernel_function import CompilationContext

    ctx = CompilationContext.get_current()
    if ctx is None:
        return
    if bitcode_path is not None:
        ctx.add_link_lib(bitcode_path)
    if module_init_fn is not None and module_init_fn not in ctx.post_load_processors:
        ctx.post_load_processors.append(module_init_fn)


class LinkedExternFunction:
    """External callable plus bitcode/link/post-load metadata."""

    def __init__(
        self,
        extern: FFIExternFunction,
        *,
        bitcode_path: Optional[str] = None,
        module_init_fn: Optional[Any] = None,
    ):
        self.extern = extern
        self.symbol = extern.symbol
        self.bitcode_path = bitcode_path
        self.module_init_fn = module_init_fn

    def __call__(self, *args):
        _register_link_metadata(self.bitcode_path, self.module_init_fn)
        return self.extern(*args)

    def __repr__(self) -> str:
        bc = f", bitcode={self.bitcode_path!r}" if self.bitcode_path else ""
        return f"LinkedExternFunction({self.extern!r}{bc})"


def link_extern(
    extern: FFIExternFunction,
    *,
    bitcode_path: Optional[str] = None,
    module_init_fn: Optional[Any] = None,
) -> LinkedExternFunction:
    """Attach external bitcode and post-load initialization metadata to an FFI."""
    return LinkedExternFunction(
        extern,
        bitcode_path=bitcode_path,
        module_init_fn=module_init_fn,
    )


class BoundExternFunction(LinkedExternFunction):
    """Convenience wrapper: construct an FFI and attach link metadata in one step."""

    def __init__(
        self,
        symbol: str,
        arg_types,
        ret_type: str,
        is_pure: bool = False,
        bitcode_path: Optional[str] = None,
        module_init_fn: Optional[Any] = None,
    ):
        bitcode_path, module_init_fn = _apply_resolvers(symbol, bitcode_path, module_init_fn)
        super().__init__(
            FFIExternFunction(symbol, arg_types, ret_type, is_pure=is_pure),
            bitcode_path=bitcode_path,
            module_init_fn=module_init_fn,
        )


__all__ = [
    "BoundExternFunction",
    "LinkedExternFunction",
    "link_extern",
    "register_extern_resolver",
]
