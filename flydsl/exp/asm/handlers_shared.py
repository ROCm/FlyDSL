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


__all__ = [n for n in globals().keys() if not n.startswith("__")]
