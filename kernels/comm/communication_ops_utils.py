# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Low-level cross-card (P2P) communication primitives for communication kernels.

These use the FlyDSL memory APIs with explicit memory ordering and syncscope
so dispatch/combine can publish and observe data across cards.

Also hosts :class:`GeometryTuningTable`, the per-shape launch-geometry lookup
shared by the dispatch/combine ops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx

__all__ = [
    "store_i32_system",
    "store_i64_global_system",
    "fence_acquire",
    "fence_release",
    "fence_system_acquire",
    "fence_system_release",
    "fence_agent_acquire",
    "fence_agent_release",
    "load_i64_global",
    "atomic_add_global_at",
    "atomic_add_agent",
    "atomic_add_system",
    "atomic_xchg_global_at",
    "GeometryTuningTable",
]


def _to_ptr_global(v, dtype=fx.Int64, alignment=8):
    """Interpret an i64 byte address as a typed global pointer."""
    ptr_type = fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment)
    return fx.inttoptr(ptr_type, fx.Int64(v))


def _to_atomic_ptr_global(addr_i64, val):
    """Typed global pointer matching the atomic operand *val*.

    A bare Python literal keeps the default ``int -> Int32`` mapping the atomic
    helpers apply to untyped operands.
    """
    dtype = getattr(val, "dtype", None) or fx.as_numeric(val).dtype
    return _to_ptr_global(addr_i64, dtype, dtype.width // 8)


def store_i32_system(addr_i64, offset, val):
    """System-scope release i32 store at ``addr_i64 + offset*4``."""
    ptr = _to_ptr_global(addr_i64, fx.Int32, 4) + offset
    fx.generic_store(ptr, val, memory_order=fx.AtomicOrdering.Release, syncscope="one-as")


def store_i64_global_system(addr_i64, val):
    """System-scope release i64 store to ``addr_i64``."""
    gptr = _to_ptr_global(addr_i64)
    fx.generic_store(gptr, val, memory_order=fx.AtomicOrdering.Release, syncscope="one-as")


def fence_acquire(syncscope):
    """Emit an acquire fence for the selected AMDGPU memory scope."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope=syncscope)


def fence_release(syncscope):
    """Emit a release fence for the selected AMDGPU memory scope."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope=syncscope)


def fence_system_acquire():
    """System-scope acquire fence."""
    fence_acquire(fx.rocdl.SyncScope.OneAs)


def fence_system_release():
    """System-scope release fence."""
    fence_release(fx.rocdl.SyncScope.OneAs)


def fence_agent_acquire():
    """Agent-scope acquire fence."""
    fence_acquire(fx.rocdl.SyncScope.AgentOneAs)


def fence_agent_release():
    """Agent-scope release fence."""
    fence_release(fx.rocdl.SyncScope.AgentOneAs)


def load_i32_global_agent(addr_i64):
    """Volatile monotonic i32 load with agent-one-as visibility."""
    return fx.generic_load(
        _to_ptr_global(addr_i64, fx.Int32, 4),
        volatile=True,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def load_i64_global(addr_i64):
    """Relaxed global i64 load from ``addr_i64``.

    Returns a DSL scalar, so callers need no re-wrapping.
    """
    return fx.generic_load(_to_ptr_global(addr_i64))


def load_i32_acquire(addr_i64):
    """Volatile monotonic i32 load suitable for a spin-wait."""
    return fx.generic_load(
        _to_ptr_global(addr_i64, fx.Int32, 4),
        volatile=True,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope="one-as",
    )


def load_i64_acquire(addr_i64):
    """Volatile monotonic i64 load suitable for a spin-wait."""
    return fx.generic_load(
        _to_ptr_global(addr_i64, fx.Int64, 8),
        volatile=True,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope="one-as",
    )


@flyc.jit
def spin_until_eq_i32(addr_i64, val):
    """Spin until an i32 flag equals ``val``."""
    cur = fx.Int32(load_i32_acquire(addr_i64))
    while cur != fx.Int32(val):
        cur = fx.Int32(load_i32_acquire(addr_i64))
    return cur


@flyc.jit
def spin_until_gt_i32(addr_i64, val):
    """Spin until an i32 flag exceeds ``val`` and return the observed value."""
    cur = fx.Int32(load_i32_acquire(addr_i64))
    while cur <= fx.Int32(val):
        cur = fx.Int32(load_i32_acquire(addr_i64))
    return cur


def wait_i32_until_equals(addr_i64, expected):
    """Compatibility wrapper for the historical MegaMoE wait helper."""
    spin_until_eq_i32(addr_i64, expected)


def wait_i32_until_greater_than(addr_i64, expected):
    """Compatibility wrapper returning the first i32 value above ``expected``."""
    return spin_until_gt_i32(addr_i64, expected)


@flyc.jit
def wait_i64_until_equals(addr_i64, expected):
    """Spin until a system-visible i64 flag equals ``expected``."""
    cur = fx.Int64(load_i64_acquire(addr_i64))
    while cur != fx.Int64(expected):
        cur = fx.Int64(load_i64_acquire(addr_i64))


def _to_ptr_shared(v):
    """Cast an integer address to an aligned shared i32 pointer."""
    ptr_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
    return fx.inttoptr(ptr_type, fx.Int64(v))


def load_i32_system(addr_i64, index):
    """Compatibility system-scope i32 load at ``addr_i64 + index * 4``."""
    item_addr = fx.Int64(addr_i64) + fx.Int64(index) * fx.Int64(4)
    return fx.Int32(load_i32_acquire(item_addr))


def atomic_add_workgroup(addr_i64, val):
    """Workgroup-scope monotonic shared-memory fetch-and-add."""
    return fx.atomic_add(_to_ptr_shared(addr_i64), fx.Int32(val), syncscope="workgroup")


def atomic_add_global_at(addr_i64, val, syncscope="one-as"):
    """Monotonic global fetch-add with configurable agent/system visibility.

    Returns the pre-update value as a DSL scalar of ``val``'s type.
    """
    return fx.atomic_add(_to_atomic_ptr_global(addr_i64, val), val, syncscope=syncscope)


def atomic_add_agent(addr_i64, val):
    """Agent-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val, syncscope=fx.rocdl.SyncScope.Agent)


def atomic_add_system(addr_i64, val):
    """System-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val)


def atomic_xchg_global_at(addr_i64, val, syncscope="agent"):
    """Monotonic global exchange with configurable agent/system visibility.

    Returns the pre-update value as a DSL scalar of ``val``'s type.
    """
    return fx.atomic_xchg(_to_atomic_ptr_global(addr_i64, val), val, syncscope=syncscope)


@dataclass
class GeometryTuningTable:
    """Per-shape token-count -> (block_num, warp_num_per_block) lookup; rounds up
    to the smallest bucket >= count (largest on overflow, mori parity)."""

    dispatch: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    combine: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        for phase, tbl in (("dispatch", self.dispatch), ("combine", self.combine)):
            for n_tok, (bn, wpb) in tbl.items():
                if bn <= 0 or wpb <= 0:
                    raise ValueError(
                        f"GeometryTuningTable.{phase}[{n_tok}] must be positive, "
                        f"got block_num={bn}, warp_num_per_block={wpb}"
                    )

    @classmethod
    def from_tuning_file(
        cls, path, *, dtype, hidden_dim, zero_copy, topk=None, local_expert_num=None, combine_dtype="bf16"
    ):
        """Build a per-op table from a multi-shape tuning JSON, filtered to this
        op's shape; empty table => cfg defaults."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def _match(r, want_dtype, need_zc):
            if r.get("dtype") != want_dtype or int(r.get("hidden_dim", -1)) != hidden_dim:
                return False
            if topk is not None and "topk" in r and int(r["topk"]) != topk:
                return False
            if (
                local_expert_num is not None
                and "local_expert_num" in r
                and int(r["local_expert_num"]) != local_expert_num
            ):
                return False
            if need_zc and bool(r.get("zero_copy", False)) != bool(zero_copy):
                return False
            return True

        def _build(rules, want_dtype, need_zc):
            return {
                int(r["num_tokens"]): (int(r["block_num"]), int(r["warp_num_per_block"]))
                for r in rules
                if _match(r, want_dtype, need_zc)
            }

        return cls(
            dispatch=_build(raw.get("dispatch", []), dtype, need_zc=False),
            combine=_build(raw.get("combine", []), combine_dtype, need_zc=True),
        )

    def lookup(self, phase, num_tokens):
        """Smallest bucket >= num_tokens (largest on overflow); None if empty."""
        tbl = self.dispatch if phase == "dispatch" else self.combine
        if not tbl:
            return None
        if num_tokens in tbl:
            return tbl[num_tokens]
        candidates = [k for k in tbl if k >= num_tokens]
        return tbl[min(candidates)] if candidates else tbl[max(tbl)]
