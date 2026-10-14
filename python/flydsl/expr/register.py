# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Symbolic LLVM register classes for pin_register."""

from dataclasses import dataclass
from functools import cached_property

__all__ = ["RegisterClass"]


@dataclass(frozen=True)
class RegisterClass:
    """LLVM architecture and TargetRegisterClass name, resolved during codegen.

    Placement numbers index the ordered members of this class. For AMDGPU's
    AGPR_32/VGPR_32/SGPR_32 classes these are the ISA aN/vN/sN numbers.
    Constructing a descriptor does not imply backend codegen support.
    """

    target: str
    name: str

    def _to_mlir(self):
        import json

        from .._mlir import ir

        return ir.Attribute.parse(f"#fly.register_class<{json.dumps(self.target)}, {json.dumps(self.name)}>")

    @cached_property
    def size_bits(self) -> int:
        """Bits per member for well-known classes; 0 otherwise."""
        _KNOWN = {"AGPR_32": 32, "VGPR_32": 32, "SGPR_32": 32}
        return _KNOWN.get(self.name, 0)

    @cached_property
    def member_count(self) -> int:
        """Member count for well-known classes; 0 otherwise."""
        _KNOWN = {"AGPR_32": 256, "VGPR_32": 256, "SGPR_32": 106}
        return _KNOWN.get(self.name, 0)
