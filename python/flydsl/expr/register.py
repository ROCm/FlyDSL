# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Symbolic LLVM register classes and declarations on existing regmem storage."""

import json
from dataclasses import dataclass
from functools import cached_property

from .meta import dsl_loc_tracing
from .struct import Arena, Storage

__all__ = ["RegisterClass", "RegisterAllocator", "set_register"]


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

    @cached_property
    def _layout(self):
        from .._mlir._mlir_libs._mlirDialectsFly import _get_register_class_layout

        return _get_register_class_layout(self.target, self.name)

    @property
    def size_bits(self):
        """Width of one class member, queried from LLVM's MC register model."""
        return self._layout[0]

    @property
    def member_count(self):
        """Number of class members; hardware availability is checked at codegen."""
        return self._layout[1]


class RegisterAllocator(Arena):
    """Allocate register-memory ``Storage[T]`` with optional placement constraints.

    With no ``register_class``, emit plain regmem and let LLVM choose both the
    class and numbers. This mode uses 32-bit storage slots, does not emit
    ``set_register``, and accepts only the default placement settings
    (``start_offset=0`` or None, ``register_alignment=1``). Storable byte
    alignment still applies. With a class, placement is checked by codegen.

    ``start_offset`` is a register class index, or None for LLVM-selected
    numbers. The default remains zero. ``register_alignment`` constrains the
    allocation origin in class members (a positive power of two); automatic
    slices preserve their word offset modulo this alignment without promising
    contiguous physical storage. Storable sizes and layout alignments
    remain in bytes. Alignment is applied to the absolute register byte offset;
    each allocation is rounded up to a whole class member. Separate allocators
    do not coordinate their ranges; overlapping live placements are diagnosed
    by the backend. Accesses must have static offsets and support promotion.

    AMDGPU requires instructions producing fixed values to ultimately support
    the chosen registers directly. LLVM's native class rewriting is used when
    available; remaining transfers inside fixed dataflow or interference at the
    requested numbers cause compilation failure. Consumers with unplaced results
    may read via a class transfer (for example, AGPR to VGPR before fptrunc).
    Such read temporaries do not inherit the allocation's register numbers.
    FlyDSL does not choose instruction encodings or force constants into
    registers. Storable layout support does not imply support for every class.
    """

    def __init__(
        self, register_class: RegisterClass | None = None, start_offset: int | None = 0, *, register_alignment: int = 1
    ):
        if register_class is not None and not isinstance(register_class, RegisterClass):
            raise TypeError("register_class must be a RegisterClass descriptor or None")
        if start_offset is not None and (
            isinstance(start_offset, bool) or not isinstance(start_offset, int) or not 0 <= start_offset < 2**63
        ):
            raise ValueError("start_offset must be a nonnegative signed 64-bit register class index")
        _check_register_alignment(register_alignment)
        if register_class is None and (start_offset not in (None, 0) or register_alignment != 1):
            raise ValueError("start_offset and register_alignment constraints require a register_class")
        self.register_alignment = register_alignment
        bits = register_class.size_bits if register_class is not None else 32
        if bits <= 0 or bits % 8:
            raise ValueError("RegisterAllocator requires byte-addressable register class members")
        self.register_class = register_class
        self.start_offset = start_offset
        self._register_bytes = bits // 8
        super().__init__(base_alignment=self._register_bytes)

    @property
    def base_ptr(self):
        raise RuntimeError("RegisterAllocator creates an independent pointer for each allocation")

    @property
    def allocated_registers(self):
        """Consumed class members (32-bit slots when unplaced), including padding."""
        return self.allocated_bytes // self._register_bytes

    @dsl_loc_tracing
    def allocate(self, storable_or_int, alignment=None):
        from math import lcm

        from .._mlir import ir
        from ..compiler.protocol import dsl_align_of, dsl_size_of
        from .numeric import Numeric, Uint8
        from .primitive import make_ptr
        from .typing import AddressSpace, Array, PointerType

        if isinstance(storable_or_int, Numeric) and not isinstance(storable_or_int.value, ir.Value):
            storable_or_int = int(storable_or_int.value)
        if isinstance(storable_or_int, bool):
            raise TypeError("allocation size must not be bool")
        raw = isinstance(storable_or_int, int)
        if raw and storable_or_int <= 0:
            raise ValueError("allocation size must be positive")
        storable = Array[Uint8, storable_or_int] if raw else storable_or_int
        nbytes = dsl_size_of(storable)
        natural = self._base_alignment if raw else dsl_align_of(storable)
        for value in (natural, alignment if alignment is not None else natural):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value & (value - 1):
                raise ValueError("alignment must be a positive power of two in bytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("Storable size must be a nonnegative integer in bytes")
        align = max(natural, alignment or natural)
        unit = self._register_bytes
        placement_align = lcm(align, unit * self.register_alignment)
        origin = (self.start_offset or 0) * unit
        absolute = origin + self._offset
        begin = (absolute + placement_align - 1) // placement_align * placement_align
        size = (nbytes + unit - 1) // unit * unit
        start = begin // unit
        if (
            self.register_class is not None
            and self.start_offset is not None
            and start + size // unit > self.register_class.member_count
        ):
            raise ValueError("allocation exceeds the LLVM register class member range")
        ptr_ty = PointerType.get(elem_ty=Uint8.ir_type, address_space=AddressSpace.Register, alignment=align)
        attrs = ir.DictAttr.get({"allocSize": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), size)})
        ptr = make_ptr(ptr_ty, [], dict_attrs=attrs)
        if size and self.register_class is not None:
            set_register(
                ptr,
                register_class=self.register_class,
                start=start if self.start_offset is not None else None,
                register_alignment=self.register_alignment,
            )
        self._offset = begin + size - origin
        return Storage[storable](ptr)


def _check_register_alignment(alignment):
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, int)
        or alignment <= 0
        or alignment >= 2**63
        or alignment & (alignment - 1)
    ):
        raise ValueError("register_alignment must be a positive power of two below 2**63")


@dsl_loc_tracing
def set_register(storage, *, register_class: RegisterClass, start: int | None = None, register_alignment: int = 1):
    """Declare class and optional physical placement on existing regmem; return None.

    Call on the allocation base in the allocation's block. The declaration
    applies to the entire allocation. With ``start=None``, LLVM chooses numbers
    in the requested class. ``register_alignment`` is measured in class members
    and constrains the origin; slices preserve their relative alignment residue.
    It does not guarantee contiguous storage in automatic mode. Target availability is checked at codegen.
    Instructions producing fixed values must accept their fixed inputs and
    outputs directly, possibly after LLVM's native class rewriting. Residual
    write-back copies are not supported. Consumers with unplaced outputs may
    use read copies. Transfers requiring implicit lane selection are rejected.
    It requires static, whole-dword slices and optimized SelectionDAG codegen.
    This does not force constant materialization. Native placement conflicts
    may produce a fatal LLVM diagnostic.
    """
    from .._mlir.dialects import fly
    from .typing import as_ir_value, is_generic_address_space

    if not isinstance(register_class, RegisterClass):
        raise TypeError("register_class must be a RegisterClass descriptor")
    if start is not None and (isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < 2**63):
        raise ValueError("start must be a nonnegative signed 64-bit register class index")
    _check_register_alignment(register_alignment)
    if start is not None and start % register_alignment:
        raise ValueError("start must satisfy register_alignment")
    value = as_ir_value(storage)
    if not hasattr(value.type, "address_space") or not is_generic_address_space(
        value.type.address_space, fly.AddressSpace.Register
    ):
        raise ValueError("set_register requires a register-memory pointer or tensor")
    fly.set_register(value, reg_class=register_class._to_mlir(), start=start, register_alignment=register_alignment)
