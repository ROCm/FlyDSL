# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Implementation of :mod:`flydsl.experimental.register_allocation`."""

from dataclasses import dataclass

from ...expr.meta import dsl_loc_tracing
from ...expr.struct import Arena, Storage

__all__ = [
    "RegisterClass",
    "RegisterAllocator",
    "register_place",
]


@dataclass(frozen=True)
class RegisterClass:
    """Symbolic reference to an LLVM register class, resolved during lowering.

    Only the target and TableGen class names and the member width are stored,
    never LLVM's numeric class IDs, which vary across LLVM versions. Placement
    numbers index the ordered members of the class. Creating a descriptor does
    not imply that the backend supports placement in it; lowering checks the
    class and its width.

    Attributes:
        target: LLVM target architecture name, such as ``"amdgcn"``.
        name: LLVM TableGen register class name, such as ``"VGPR_32"``.
        register_bits: Width in bits of one class member, such as ``32``.
    """

    target: str
    name: str
    register_bits: int

    def __post_init__(self):
        for value in (self.target, self.name):
            if (
                not isinstance(value, str)
                or not value
                or not value.isascii()
                or any(ord(c) < 32 or ord(c) >= 127 for c in value)
            ):
                raise ValueError("register class target and name must be nonempty printable ASCII strings")

    def ir_attr(self):
        from ..._mlir import ir

        return ir.Attribute.parse(f'#fly.register_class<"{self.target}", "{self.name}", {self.register_bits}>')


class RegisterAllocator(Arena):
    """Allocator of register-memory ``Storage[T]`` placed in one register class.

    The allocator has the ``allocate(T)`` / ``Storage[T]`` interface of
    ``SharedAllocator``, with sizes and alignments in bytes. Each ``allocate``
    call creates independent register memory, so there is no common base
    pointer and no ``free``.

    Attributes:
        register_class: The ``RegisterClass`` of every allocation.
        start_idx: The class member index where numbering starts.
    """

    def __init__(self, register_class: RegisterClass, start_idx: int):
        """Initializes the allocator.

        Args:
            register_class: Register class of every allocation.
            start_idx: Class member index of the first allocation.
        """
        if not isinstance(register_class, RegisterClass):
            raise TypeError("register_class must be a RegisterClass descriptor")
        if isinstance(start_idx, bool) or not isinstance(start_idx, int) or not start_idx >= 0:
            raise ValueError("start_idx must be a nonnegative signed register class index")

        bits = register_class.register_bits
        if bits % 8:
            raise ValueError("RegisterAllocator requires byte-addressable register class members")
        self.register_class = register_class
        self.start_idx = start_idx
        self._register_bytes = bits // 8
        super().__init__(base_alignment=self._register_bytes)

    @property
    def base_ptr(self):
        """Not available: each allocation has its own pointer."""
        raise RuntimeError("RegisterAllocator creates an independent pointer for each allocation")

    @dsl_loc_tracing
    def allocate(self, storable_or_int, alignment=None):
        """Allocates register memory for one Storable value.

        The allocation starts at the first class member index, at or after
        ``start_idx`` plus the space already allocated, that satisfies its
        alignment. An allocation with a nonzero size is placed through
        ``register_place``.

        Args:
            storable_or_int: A Storable type ``T``, or a positive byte count
                (an ``int`` or a compile-time ``Numeric``) that allocates
                ``Array[Uint8, n]``.
            alignment: Minimum alignment in bytes, a power of two. The
                allocation uses the larger of this and ``dsl_align_of(T)``,
                which is the class member size for a byte count.

        Returns:
            ``Storage[T]`` over the new register memory.
        """
        from math import lcm

        from ..._mlir import ir
        from ...compiler.protocol import dsl_align_of, dsl_size_of
        from ...expr.numeric import Numeric, Uint8
        from ...expr.primitive import make_ptr
        from ...expr.typing import AddressSpace, Array, PointerType

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
        placement_align = lcm(align, unit)
        origin = self.start_idx * unit
        absolute = origin + self._offset
        begin = (absolute + placement_align - 1) // placement_align * placement_align
        size = (nbytes + unit - 1) // unit * unit
        start = begin // unit

        ptr_ty = PointerType.get(elem_ty=Uint8.ir_type, address_space=AddressSpace.Register, alignment=align)
        attrs = ir.DictAttr.get({"allocSize": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), size)})
        ptr = make_ptr(ptr_ty, [], dict_attrs=attrs)
        if size:
            register_place(ptr, self.register_class, start_idx=start)
        self._offset = begin + size - origin
        return Storage[storable](ptr)


@dsl_loc_tracing
def register_place(
    storage,
    register_class: RegisterClass,
    *,
    start_idx: int,
):
    """Places existing register memory at fixed numbers of a register class.

    Use this for register memory that ``RegisterAllocator.allocate`` did not
    create, such as MMA fragments. Call it on the allocation base, in the
    block of the allocation; the declaration covers the entire allocation.
    Target support is checked during lowering.

    Args:
        storage: Register-memory pointer or tensor to place.
        register_class: Register class of the placement.
        start_idx: Class member index of the first register.
    """
    from ..._mlir.dialects import fly
    from ...expr.typing import as_ir_value, is_generic_address_space

    if not isinstance(register_class, RegisterClass):
        raise TypeError("register_class must be a RegisterClass descriptor")
    if isinstance(start_idx, bool) or not isinstance(start_idx, int) or not 0 <= start_idx < 2**31:
        raise ValueError("start_idx must be a nonnegative int32 register class index")

    value = as_ir_value(storage)
    if not hasattr(value.type, "address_space") or not is_generic_address_space(
        value.type.address_space, fly.AddressSpace.Register
    ):
        raise ValueError("register_place requires a register-memory pointer or tensor")
    fly.register_place(value, register_class=register_class.ir_attr(), start_idx=start_idx)
