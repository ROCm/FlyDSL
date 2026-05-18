# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU intrinsics and address space helpers.

Provides thread/block indexing (``thread_idx``, ``block_idx``),
synchronization (``barrier``), and shared memory address space
(``smem_space``/``lds_space``) for kernel authoring.

Usage::

    import flydsl.expr as fx

    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    fx.gpu.barrier()
"""

from .._mlir.dialects import gpu
from .._mlir.ir import Attribute
from .primitive import get_dyn_shared
from .struct import Arena
from .typing import Tuple3D
from ..compiler.protocol import dsl_align_of, dsl_size_of
from .numeric import Uint8
from .struct import Storage
from .typing import Array
from .._mlir import ir
from .._mlir.dialects._fly_enum_gen import AddressSpace
from .numeric import Uint8
from .primitive import make_ptr
from .typing import PointerType
from .struct import (
    CompositeKind,
    _effective_field_defs,
    _is_constexpr_type,
    is_composite_type,
    is_struct_type,
)

thread_id = gpu.thread_id
block_id = gpu.block_id

thread_idx = Tuple3D(gpu.thread_id)
block_idx = Tuple3D(gpu.block_id)
block_dim = Tuple3D(gpu.block_dim)
grid_dim = Tuple3D(gpu.grid_dim)

barrier = gpu.barrier

_int = int


def smem_space(int=False):
    """Return the GPU shared memory (LDS/workgroup) address space.

    Args:
        int: If True, return the integer value; otherwise return an
             MLIR ``#gpu.address_space<workgroup>`` attribute.
    """
    a = gpu.AddressSpace.Workgroup
    if int:
        return _int(a)
    return Attribute.parse(f"#gpu.address_space<{a}>")


lds_space = smem_space


class SharedAllocator(Arena):
    """LDS allocator with static / dynamic placement modes.

    ``static=True`` (default): each ``allocate(T)`` emits a separate
    ``fly.make_ptr`` per logical leaf / union. Like `__shared__` in C/C++.

    ``static=False``: ``allocate(T)`` hands out compile-time offsets into a
    single ``fly.get_dyn_shared`` base pointer, and all sub-pointers GEP off
    the same LDS global. Like `extern __shared__` in C/C++.
    """

    def __init__(
        self,
        base_alignment: int = Arena.DEFAULT_BASE_ALIGNMENT,
        *,
        static: bool = True,
    ):
        super().__init__(base_alignment=base_alignment)

        from ..compiler.kernel_function import KernelFunction

        kf = KernelFunction.get_current()
        if kf is None:
            raise RuntimeError("SharedAllocator can only be created inside a @kernel function")
        kf.register_shared_allocator(self)
        self._static = bool(static)
        self._base = None if self._static else get_dyn_shared()

    @property
    def is_static(self) -> bool:
        return self._static

    @property
    def base_ptr(self):
        if self._static:
            raise RuntimeError(
                "SharedAllocator(static=True) has no shared base pointer — each leaf "
                "sub-buffer is an independent `@__shared_alloc_<id>` symbol."
            )
        return self._base

    def allocate(self, storable_or_int, alignment=None):
        if not self._static:
            return super().allocate(storable_or_int, alignment)
        return self._allocate_static(storable_or_int, alignment)

    def _allocate_static(self, storable_or_int, alignment):
        if isinstance(storable_or_int, int):
            nbytes = storable_or_int
            if nbytes <= 0:
                raise ValueError(f"allocate size must be > 0, got {nbytes}")
            align = alignment if alignment is not None else self._base_alignment
            self._bump(nbytes, align)
            leaf_type = Array[Uint8, nbytes]
            ptr = self._allocate_static_shared(nbytes, align)
            return Storage[leaf_type](ptr)
        else:
            storable = storable_or_int
            nbytes = dsl_size_of(storable)
            align = dsl_align_of(storable) if alignment is None else max(dsl_align_of(storable), alignment)
            self._bump(nbytes, align)
            return self._build_static_tree(storable)

    def _build_static_tree(self, type_spec):
        """Recursively build a Storage tree over per-leaf `make_ptr` ops.

        - struct  → recurse into each field; each field emits its own `make_ptr`
                    and therefore lowers to its own LDS global.
        - union   → ONE `make_ptr` shared by all variants (size=max, align=max).
                    Each variant is wrapped as a re-typed view over the same
                    ptr.
        - leaf    → single `make_ptr`.
        """

        if is_composite_type(type_spec) and type_spec.__dsl_composite_kind__ == CompositeKind.Sum:
            nbytes = dsl_size_of(type_spec)
            align = dsl_align_of(type_spec)
            shared_ptr = self._allocate_static_shared(nbytes, align)
            prebuilt = {}
            for name, variant_ty in _effective_field_defs(type_spec):
                if _is_constexpr_type(variant_ty):
                    continue
                prebuilt[name] = Storage[variant_ty](shared_ptr)
            return Storage[type_spec](shared_ptr, prebuilt=prebuilt)
        elif is_struct_type(type_spec):
            prebuilt = {}
            for name, field_ty in _effective_field_defs(type_spec):
                if _is_constexpr_type(field_ty):
                    continue
                prebuilt[name] = self._build_static_tree(field_ty)
            return Storage[type_spec](None, prebuilt=prebuilt)
        else:
            nbytes = dsl_size_of(type_spec)
            align = dsl_align_of(type_spec)
            ptr = self._allocate_static_shared(nbytes, align)
            return Storage[type_spec](ptr)

    def _allocate_static_shared(self, nbytes: int, align: int):
        ptr_ty = PointerType.get(
            elem_ty=Uint8.ir_type,
            address_space=AddressSpace.Shared,
            alignment=align,
        )
        i32 = ir.IntegerType.get_signless(32)
        dict_attrs = ir.DictAttr.get(
            {
                "allocBytes": ir.IntegerAttr.get(i32, nbytes),
                "allocAlign": ir.IntegerAttr.get(i32, align),
            }
        )
        return make_ptr(ptr_ty, [], dict_attrs=dict_attrs)


__all__ = [
    "thread_id",
    "block_id",
    "thread_idx",
    "block_idx",
    "block_dim",
    "grid_dim",
    "barrier",
    "smem_space",
    "lds_space",
    "SharedAllocator",
]
