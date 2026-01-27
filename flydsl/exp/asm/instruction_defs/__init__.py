# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
AMDGCN instruction definitions in YAML format.

This package contains architecture-specific instruction definitions:
- common.yaml: Instructions shared across all architectures
- gfx942.yaml: GFX942 (CDNA3) specific instructions (optional)
- gfx950.yaml: GFX950 (CDNA3+) specific instructions (optional)

The YAML files are loaded by instruction_registry.py to provide
a unified instruction database for the ASM backend.
"""
