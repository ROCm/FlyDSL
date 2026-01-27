# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Operation handlers for MLIR operations in the ASM backend.

This module contains handlers for various MLIR operations that are encountered
during the IR traversal for assembly code generation.
"""

from __future__ import annotations


def ssa(value) -> str:
    """Return a stable SSA name for an MLIR Value/OpResult/BlockArgument.

    Different Python bindings stringify SSA values differently (sometimes
    embedding the defining op). For indexing/caching we want the canonical SSA
    name like `%0` / `%arg0`.
    """
    try:
        if value is not None and hasattr(value, "get_name"):
            n = value.get_name()
            if n:
                return str(n)
    except Exception:
        pass
    return str(value)


__all__ = [n for n in globals().keys() if not n.startswith("__")]
