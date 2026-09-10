# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Symbolic LLVM register classes and declarations on existing regmem storage."""

import json
from dataclasses import dataclass

from .meta import dsl_loc_tracing

__all__ = ["RegisterClass", "set_register"]


@dataclass(frozen=True)
class RegisterClass:
    """LLVM architecture and TargetRegisterClass name, resolved during codegen.

    Placement numbers index the ordered members of this class. For AMDGPU's
    AGPR_32/VGPR_32/SGPR_32 classes these are the ISA aN/vN/sN numbers.
    Constructing a descriptor does not imply backend codegen support.
    """

    target: str
    name: str

    def __post_init__(self):
        for value in (self.target, self.name):
            if (
                not isinstance(value, str)
                or not value
                or not value.isascii()
                or any(ord(c) < 32 or ord(c) >= 127 for c in value)
            ):
                raise ValueError("register class target and name must be nonempty printable ASCII strings")

    def _to_mlir(self):
        from .._mlir import ir

        return ir.Attribute.parse(f"#fly.register_class<{json.dumps(self.target)}, {json.dumps(self.name)}>")


@dsl_loc_tracing
def set_register(storage, *, register_class: RegisterClass, start: int):
    """Declare physical placement on existing regmem; return None.

    Call on the allocation base in the allocation's block. The declaration
    applies to the entire allocation. Target availability is checked at codegen.
    Instruction/class legality is the caller's responsibility. AMDGPU currently
    requires static, whole-dword slices and optimized SelectionDAG codegen.
    This does not force constant materialization. Native placement conflicts
    may produce a fatal LLVM diagnostic.
    """
    from .._mlir.dialects import fly
    from .typing import as_ir_value, is_generic_address_space

    if not isinstance(register_class, RegisterClass):
        raise TypeError("register_class must be a RegisterClass descriptor")
    if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < 2**63:
        raise ValueError("start must be a nonnegative signed 64-bit register class index")
    value = as_ir_value(storage)
    if not hasattr(value.type, "address_space") or not is_generic_address_space(
        value.type.address_space, fly.AddressSpace.Register
    ):
        raise ValueError("set_register requires a register-memory pointer or tensor")
    fly.set_register(value, reg_class=register_class._to_mlir(), start=start)
