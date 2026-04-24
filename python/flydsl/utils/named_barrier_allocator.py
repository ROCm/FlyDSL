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

Usage (canonical JOIN-SIGNAL-WAIT pattern)::

    nbar_alloc = NamedBarrierAllocator(sym_prefix="myk_nbar")
    nbar_a = nbar_alloc.alloc(member_count=4, name_hint="A")
    # ... inside gpu.module body:
    nbar_alloc.finalize()
    # ... inside kernel body:
    nbar_a.init()            # wave 0 only (gate this with an scf.if)
    gpu.barrier()            # WG barrier so every wave sees the INIT
    nbar_a.join()            # every wave: subscribe to the barrier
    # ... hot loop:
    nbar_a.signal()          # arrive
    nbar_a.wait()            # wait for memberCnt arrivals

A wave may be joined to at most one named barrier at a time.  Without a
prior ``join()``, ``wait()`` is a silent NOP.
"""

from typing import List, Optional

from .._mlir import ir
from .._mlir.dialects import llvm as llvm_dialect
from ..expr import rocdl


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
        imm_id: Immediate barrier id (1..16) used by the id-form ``signal``/
            ``wait`` ops. Assigned in allocation order; matches the AMDGPU
            backend's per-kernel named-barrier allocation.
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
        rocdl.s_barrier_init(self.addrof(), self.member_count)

    def signal_var(self):
        """Emit ``rocdl.s.barrier.signal.var`` (ptr + memberCnt) — CUDA bar.arrive.

        ``member_count`` is encoded as an I32Attr (immediate), not an SSA value.
        """
        rocdl.s_barrier_signal_var(self.addrof(), self.member_count)

    def signal(self):
        """Emit ``rocdl.s.barrier.signal id=imm_id`` (id form, immediate).

        S_BARRIER_SIGNAL increments the barrier's signalCnt; when signalCnt
        reaches memberCnt the barrier broadcasts "complete" to all waves 
        that are joined to it.
        """
        rocdl.s_barrier_signal(self.imm_id)

    def join(self):
        """Emit ``rocdl.s.barrier.join`` — subscribe this wave to the barrier.

        S_BARRIER_JOIN *does not* increment the barrier's signal/member count;
        it only sets ``wave.namedBarID`` and clears the wave's ``namedBarComplete``
        bit so the wave will be woken when the barrier next completes.

        A wave must JOIN a named barrier **before** calling :meth:`wait` —
        otherwise ``wave.namedBarID == 0`` and ``S_BARRIER_WAIT`` becomes a
        silent NOP ("if (NamedBarID == 0 && Bar# >= 0) return").

        A wave may be joined to at most one named barrier at a time; JOINing
        a different barrier switches the subscription. Since JOIN is sticky
        across loop iterations, the canonical pattern is JOIN-once-at-entry
        then SIGNAL+WAIT in the hot loop.

        Canonical sequence (see llvm/test/CodeGen/AMDGPU/s-barrier.ll): ::

            bar.join()            # kernel entry, all waves
            # per iteration:
            bar.signal()          # arrive
            bar.wait()            # block until all members have signalled
        """
        rocdl.s_barrier_join(self.addrof())

    def wait(self):
        """Emit ``rocdl.s.barrier.wait id=imm_id`` (id form, immediate).

        ``S_BARRIER_WAIT`` always blocks on the *most recently joined* 
        named barrier — the ``Bar#`` operand only selects between 
        named/workgroup/trap/cluster barriers, it does **not**
        pick an individual named barrier. The wave **must** have called
        :meth:`join` prior to :meth:`wait`, otherwise WAIT is a NOP.

        The immediate form is mandatory: "BAR# may only be a constant, not
        come from M0".
        """
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
