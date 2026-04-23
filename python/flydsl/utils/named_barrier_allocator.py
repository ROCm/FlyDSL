# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1250+ Named Barrier global allocator.

Each declared named barrier becomes an ``llvm.mlir.global`` of type
``!llvm.target<"amdgcn.named.barrier">`` in addrspace(3). The AMDGPU
backend automatically counts these globals and writes
``NamedBarCnt = ceil(N/4)`` into the kernel descriptor's
``COMPUTE_PGM_RSRC3_GFX125.NAMED_BAR_CNT[14:3]`` field
(see ``AMDHSAKernelDescriptor.h:229`` and ``AMDGPUAsmPrinter.cpp:1147``).

So the program only needs to declare the globals — no kernel-descriptor
or runtime tweaking required.

Usage::

    nbar_alloc = NamedBarrierAllocator(sym_prefix="myk_nbar")
    nbar_a = nbar_alloc.alloc(member_count=4, name_hint="A")
    nbar_b = nbar_alloc.alloc(member_count=4, name_hint="B")
    # ... inside gpu.module body:
    nbar_alloc.finalize()
    # ... inside kernel body:
    nbar_a.signal_var()      # producer
    nbar_a.wait()            # consumer
"""

from typing import List, Optional

from .._mlir import ir
from .._mlir.dialects import llvm as llvm_dialect


_NAMED_BARRIER_TYPE_STR = '!llvm.target<"amdgcn.named.barrier", 0>'


def _named_barrier_type():
    return ir.Type.parse(_NAMED_BARRIER_TYPE_STR)


def _ptr_addrspace3():
    return ir.Type.parse("!llvm.ptr<3>")


class NamedBarrier:
    """Handle to one allocated named barrier.

    Attributes:
        sym_name: Symbol name of the underlying ``llvm.mlir.global``.
        member_count: Compile-time member count used by ``signal_var``/``init``.
        imm_id: Reserved immediate barrier id (1..16). Currently UNUSED — the
            backend assigns the actual id based on the global. Use ``signal_var``
            (ptr-form) for signal; for ``wait`` we still need an immediate id,
            see :meth:`wait` for the lowering trick.
    """

    def __init__(self, sym_name: str, member_count: int, imm_id: int):
        self.sym_name = sym_name
        self.member_count = int(member_count)
        self.imm_id = int(imm_id)

    # ------------------------------------------------------------------
    # SSA helpers
    # ------------------------------------------------------------------
    def addrof(self):
        """Materialise an ``!llvm.ptr<3>`` SSA value pointing at the global."""
        ptr_ty = _ptr_addrspace3()
        return llvm_dialect.AddressOfOp(ptr_ty, self.sym_name).result

    # ------------------------------------------------------------------
    # ROCDL ops
    # ------------------------------------------------------------------
    def init(self):
        """Emit ``rocdl.s.barrier.init`` with the stored member count.

        ``member_count`` is encoded as an I32Attr (immediate).  Issue ONCE
        at kernel entry from any wave; the barrier auto-resets to
        *member_count* arrivals after each release, so steady-state per-iter
        code only needs ``signal(id)`` + ``wait(id)``.
        """
        from flydsl.expr import rocdl

        rocdl.s_barrier_init(self.addrof(), self.member_count)

    def signal_var(self):
        """Emit ``rocdl.s.barrier.signal.var`` (ptr + memberCnt) — CUDA bar.arrive.

        ``member_count`` is encoded as an I32Attr (immediate), not an SSA value.
        """
        from flydsl.expr import rocdl

        rocdl.s_barrier_signal_var(self.addrof(), self.member_count)

    def signal(self):
        """Emit ``rocdl.s.barrier.signal id=imm_id`` (id form, immediate)."""
        from flydsl.expr import rocdl

        rocdl.s_barrier_signal(self.imm_id)

    def join(self):
        """Emit ``rocdl.s.barrier.join`` (ptr) — the actual arrival.

        Per the canonical LLVM lowering test ``s-barrier-lowering.ll``,
        ``s.barrier.signal.var`` only programs member count; ``join`` is
        the per-wave arrival that increments the satisfaction counter.
        Wave order is: ``signal_var`` (program mc) → ``join`` (arrive) →
        ``wait id`` (block until count satisfied).
        """
        from flydsl.expr import rocdl

        rocdl.s_barrier_join(self.addrof())

    def wait(self):
        """Emit ``rocdl.s.barrier.wait id=imm_id`` (id form, immediate).

        Note: ``s.barrier.wait`` only accepts an immediate id (MI400 SPG L4391:
        "BAR# may only be a constant, not come from M0"). This relies on the
        backend allocating barriers in declaration order so that ``imm_id``
        matches the id LLVM picks for the corresponding global.
        """
        from flydsl.expr import rocdl

        rocdl.s_barrier_wait(self.imm_id)


class NamedBarrierAllocator:
    """Collect ``NamedBarrier`` declarations and emit globals at finalize."""

    MAX_NAMED_BARRIERS = 16  # gfx1250 hardware limit per WG

    def __init__(self, sym_prefix: str = "nbar"):
        self.sym_prefix = sym_prefix
        self.bars: List[NamedBarrier] = []
        self.finalized = False

    def alloc(self,
              member_count: int,
              name_hint: Optional[str] = None) -> NamedBarrier:
        """Reserve a new named barrier.

        Args:
            member_count: Number of waves that must signal before release
                (typically the WG wave count).
            name_hint: Optional human-readable suffix for the symbol name.

        Returns:
            A :class:`NamedBarrier` handle. Its ``imm_id`` is assigned in
            allocation order starting from 1 (matches the backend's per-kernel
            named-barrier allocation).
        """
        if self.finalized:
            raise RuntimeError("NamedBarrierAllocator: alloc() after finalize()")
        if len(self.bars) >= self.MAX_NAMED_BARRIERS:
            raise RuntimeError(
                f"NamedBarrierAllocator: exceeded {self.MAX_NAMED_BARRIERS} "
                f"named barriers per WG")

        idx = len(self.bars)
        suffix = f"_{name_hint}" if name_hint else ""
        sym_name = f"{self.sym_prefix}_{idx}{suffix}"
        # imm_id starts at 1 (id 0 is the Null barrier per Table 30).
        bar = NamedBarrier(sym_name=sym_name,
                           member_count=member_count,
                           imm_id=idx + 1)
        self.bars.append(bar)
        return bar

    def finalize(self):
        """Emit one ``llvm.mlir.global`` per declared barrier.

        Must be called inside the ``gpu.module`` body's ``InsertionPoint``,
        analogous to :meth:`SmemAllocator.finalize`.
        """
        if self.finalized:
            return
        if not self.bars:
            self.finalized = True
            return

        bar_ty = _named_barrier_type()
        i32 = ir.IntegerType.get_signless(32)
        link_attr = ir.Attribute.parse("#llvm.linkage<internal>")

        for bar in self.bars:
            # Initializer region exists but contains no blocks for target-ext
            # types ("amdgcn.named.barrier" lacks ZeroInit/byte representation).
            # Canonical text form: `llvm.mlir.global internal @sym() {...} : type`
            ir.Operation.create(
                "llvm.mlir.global",
                attributes={
                    "sym_name": ir.StringAttr.get(bar.sym_name),
                    "global_type": ir.TypeAttr.get(bar_ty),
                    "linkage": link_attr,
                    "addr_space": ir.IntegerAttr.get(i32, 3),
                },
                regions=1,
            )

        self.finalized = True


__all__ = ["NamedBarrier", "NamedBarrierAllocator"]
