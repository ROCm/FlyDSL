# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``ScheduleRegion``: issue the ordered instruction wrappers called in a block in the order they are written, and keep
the explicit VALU wrappers between the ordered instructions they are written between.

Built on the AMDGPU backend's ``sched_barrier`` and ``sched_group_barrier`` (``AMDGPUIGroupLP.cpp``). The region is
opened and closed by ``sched_barrier(0)``, which ends the backend's scheduling region, so the region is scheduled on its
own.

An instruction of an exclusive class (``sched`` names it: MFMA, VMEM, DS, LDS DMA) is followed by
``sched_group_barrier(<its class>, 1, 0)``. A scheduling region forms one pipeline of these groups in barrier order; the
solver walks the instructions in their incoming order and gives each the earliest group with room that it matches,
which is its own barrier, so the instructions keep their written order through the pre-RA and post-RA schedulers.
SelectionDAG emits a block in source order on GCN, so the incoming order is the written one; the barriers, which have
side effects, also keep the memory optimizer from merging neighboring loads. A scheduling region that holds a group
barrier runs the IGroupLP mutation alone, without LLVM's load clustering.

An explicit VALU instruction (``sched`` is ``"valu"``) has no exclusive class: every VALU instruction of the region
would compete for its group. The explicit VALUs written between two ordered instructions form a span instead, walled in
by ``sched_barrier(salu)`` before the first and after the last. Only SALU crosses a wall, so the span's VALUs stay
between the ordered instructions around it, in any order among themselves. The closing wall is emitted after every
explicit VALU and the previous one erased, so it stands after the last. A wall has instructions on both sides, so it
keeps its place through both schedulers. The walls also stop the region's other VALU code, which the backend cannot
tell from explicit VALUs: it cannot enter or cross a span, and is free elsewhere. An instruction without a class
(``sched`` is None, scalar memory) gets nothing and is scheduled freely.

A scheduling region lies within one basic block, so run-time control flow splits the region into straight-line pieces,
each ordered on its own; the barriers of an instruction are emitted in the block the instruction is, and a span does not
reach into another block, so every piece has its own. The barriers count as memory accesses of unknown effect, so LLVM
does not move loads across them, out of a loop or into another piece. A group takes any instruction of its class, so
the region must not hold other instructions of the same classes (``fx.copy``, ``fx.gemm``, LDS loads); see
:class:`ScheduleRegion`.

``priority`` puts ``s_setprio`` at the region's edges. LLVM ends a scheduling region at ``s_setprio`` as at
``sched_barrier(0)`` (``SIInstrInfo::isSchedulingBoundary``), so the prioritized span is exactly the region. The
instruction takes only an immediate, so a run-time priority ``p`` is read from the first active lane and raises the
priority one scalar ``if`` at a time: ``s_setprio 0``, then ``if p >= k: s_setprio k`` for k = 1 .. max_priority. The
last ``s_setprio`` taken wins, so the chain also clamps ``p`` to 0 .. max_priority without a mask. The branches
have to be scalar, since ``s_setprio`` ignores EXEC: a lane-masked branch would issue it on both sides. ``>=`` tests in
one-sided ``if`` keep the control flow as written; ``==`` tests fold into a ``switch``, and the structurizer gives that
or an if-else flow blocks several times longer.
"""

from ....._mlir import ir
from ....._mlir.dialects import _rocdl_ops_gen as _rocdl
from ....._mlir.dialects import scf
from ....numeric import Int32
from ....typing import as_ir_value

# A region is its own scheduling region, so one group id serves every region.
_GROUP_ID = 0
_GROUP_BARRIER = "rocdl.sched.group.barrier"
_BARRIER = "rocdl.sched.barrier"

_open = None  # the open ScheduleRegion


def _mask(names):
    return ir.Attribute.parse(f"#rocdl<sched_group_mask {names}>")


class ScheduleRegion:
    """Issue the ordered instructions called in the ``with`` block in the order they are written.

    Every instruction wrapper of an exclusive scheduling class (``V_MFMA_*``, ``BUFFER_LOAD_*``, ``DS_READ_*``, ...)
    called in the block appears in the final ISA in call order. The explicit VALU wrappers (``V_CVT_*``, ``V_DOT*``,
    ``V_MED3_*``, ...) called between two of them stay between them, in any order among themselves. The rest of the
    block (address arithmetic and other VALU code, register moves, scalar loads, the ``s_waitcnt`` the backend inserts)
    is scheduled freely, except that VALU code cannot enter or cross a stretch of explicit VALUs; all of it stays
    inside the region: nothing moves into or out of it. ::

        with ScheduleRegion():
            x0 = cdna3.BUFFER_LOAD_DWORDX4(rsrc, off)
            x1 = cdna3.BUFFER_LOAD_DWORDX4(rsrc, off + 1024)
            c = cdna3.V_MFMA_F32_16X16X16_F16(a(x0), b(x0), c)  # issued before the third load
            h = cdna3.V_CVT_PKRTZ_F16_F32(p, q)  # between the MFMA and the third load
            x2 = cdna3.BUFFER_LOAD_DWORDX4(rsrc, off + 2048)

    Run-time control flow (``if`` / ``for`` / ``while``) may appear in the block. It splits the block into
    straight-line pieces, one per basic block: the code before it, each branch or loop body, the code after it. The
    order holds within each piece, and the pieces run as the control flow says; in the ISA text they appear in the
    block layout LLVM chooses. A loop is unrolled or not as anywhere else: by LLVM's own choice, or as ``unroll=N`` /
    ``unroll_full=True`` / ``unroll=1`` asks. Every copy of an unrolled body keeps the order, so the body repeats.

    ``priority`` (uniform ``i32``, 0 lowest to 3 highest) runs the block at that wave priority: the region opens with
    ``s_setprio priority`` and closes with ``s_setprio 0``, the priority kernel waves start at. A run-time value, such
    as one computed from the wave index, is taken from the wave's first active lane, clamped to 0 .. ``max_priority``,
    and selects the immediate by scalar branches before the region: one compare and branch per priority above 0, so a
    smaller ``max_priority`` tests fewer::

        wave = fx.Int32(fx.thread_idx.x) // 64
        with ScheduleRegion(priority=1 - wave % 2, max_priority=1):  # even waves at priority 1, odd waves at 0
            ...

    Requirements, checked where possible:

    - No ``sched_barrier`` / ``sched_group_barrier`` / ``iglp_opt`` of your own in the block; regions do not nest.

    Not checked: the block must not issue other instructions of the classes it orders (MFMA, VMEM, DS, LDS DMA), such as
    ``fx.copy`` / ``fx.gemm`` or a load from an LDS tensor; one of those can take an instruction's place in the order.
    Register spills (scratch VMEM) can do the same. Explicit VALUs are held in place only in a piece that also holds an
    ordered instruction: without one, LLVM's load clustering runs there too, and its edges can override the walls. An
    explicit VALU wrapper may compile to no instruction or to another one when its operands allow (a uniform
    ``V_READFIRSTLANE_B32``, a ``V_MED3_F32`` with constant bounds). Buffer resources and ``soffset`` must be
    wave-uniform: otherwise the backend wraps the instruction in a loop of its own. Branch conditions that are
    wave-uniform give scalar branches (``s_cbranch_scc*``); others give EXEC-mask code around the branch.
    """

    def __init__(self, priority=None, max_priority=3):
        self.priority = priority
        self.max_priority = max_priority

    def __enter__(self):
        global _open
        if _open is not None:
            raise RuntimeError("ScheduleRegion: regions do not nest")
        if self.priority is not None:
            _set_priority(self.priority, self.max_priority)
        self._block = ir.InsertionPoint.current.block
        self._entry = _rocdl.SchedBarrier(_mask("none"))
        self._emitted = {_GROUP_BARRIER: 0, _BARRIER: 0}  # barriers the region has in the block
        self._span = None  # (block, closing wall or None) of the open span of explicit VALUs
        _open = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _open
        _open = None
        if exc_type is not None:
            return
        if ir.InsertionPoint.current.block != self._block:
            raise RuntimeError("ScheduleRegion: the block ends in another block than it started")
        found = dict.fromkeys(self._emitted, 0)
        ops = [op.operation for op in self._block.operations]
        for op in ops[ops.index(self._entry.operation) + 1 :]:
            _count(op, found)
        for name, n in found.items():
            if n != self._emitted[name]:
                raise RuntimeError(f"ScheduleRegion: '{name}' in the region; the region already orders it")
        _rocdl.SchedBarrier(_mask("none"))
        if self.priority is not None:
            _rocdl.s_setprio(0)

    def _wall(self):
        """``sched_barrier(salu)``: only SALU crosses it."""
        self._emitted[_BARRIER] += 1
        return _rocdl.SchedBarrier(_mask("salu"))


def _set_priority(priority, max_priority):
    """``s_setprio priority``; a run-time ``priority`` is taken from its first active lane, clamped to 0 .. max."""
    p = as_ir_value(Int32(priority), keep_static=True)
    if isinstance(p, int):
        _rocdl.s_setprio(p)
        return
    p = Int32(_rocdl.readfirstlane(p.type, p))
    _rocdl.s_setprio(0)
    for k in range(1, max_priority + 1):
        if_op = scf.IfOp(as_ir_value(p >= k))
        with ir.InsertionPoint(if_op.then_block):
            _rocdl.s_setprio(k)
            scf.YieldOp([])


def _count(op, found):
    """Add the barriers in ``op`` and the ops nested in it to ``found``; raise at an ``iglp_opt``."""
    if op.name == "rocdl.iglp.opt":
        raise RuntimeError("ScheduleRegion: 'rocdl.iglp.opt' in the region; the region already orders it")
    if op.name in found:
        found[op.name] += 1
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                _count(inner.operation, found)


def check(instr):
    """Before ``instr`` is built in the open region (if any): an ordered instruction ends the span of explicit VALUs,
    and the first explicit VALU after it opens one with a wall."""
    if _open is None or instr.sched is None:
        return
    if instr.sched != "valu":
        _open._span = None
        return
    block = ir.InsertionPoint.current.block
    if _open._span is None or _open._span[0] != block:
        _open._wall()
        _open._span = (block, None)


def order(instr):
    """After ``instr`` is built in the open region (if any): a group barrier after an ordered instruction; after an
    explicit VALU, the wall that closes its span, in place of the previous one."""
    if _open is None or instr.sched is None:
        return
    if instr.sched != "valu":
        _rocdl.SchedGroupBarrier(_mask(instr.sched), 1, _GROUP_ID)
        _open._emitted[_GROUP_BARRIER] += 1
        return
    block, wall = _open._span
    if wall is not None:
        wall.operation.erase()
        _open._emitted[_BARRIER] -= 1
    _open._span = (block, _open._wall())
