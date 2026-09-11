# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Low-level cross-card (P2P) communication primitives for communication kernels.

The atomics and fences forward to the ``fx`` LLVM-dialect API, adding the
inttoptr into the global address space that it does not do itself. The ordered
loads/stores still build the dialect ops directly, which remains the only way
to attach memory ordering and syncscope to them -- neither buffer_ops nor
Pointer exposes it. Either way, dispatch/combine can publish and observe data
across cards.

Also hosts :class:`GeometryTuningTable`, the per-shape launch-geometry lookup
shared by the dispatch/combine ops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

import flydsl.expr as fx
from flydsl.expr import arith

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


def _to_fly_ptr_global(addr_i64, dtype, *, alignment):
    """Cast an i64 address to a global-address-space ``fx.Pointer`` of *dtype*."""
    ptr_ty = fx.PointerType.get(dtype.ir_type, address_space=fx.AddressSpace.Global, alignment=alignment)
    return fx.inttoptr(ptr_ty, fx.Int64(arith.unwrap(addr_i64)))


def _to_atomic_ptr_global(addr_i64, val):
    """Global ``fx.Pointer`` typed after the atomic operand *val*.

    A bare Python literal keeps the default ``int -> Int32`` mapping the atomic
    helpers apply to untyped operands.
    """
    dtype = getattr(val, "dtype", None) or fx.as_numeric(val).dtype
    return _to_fly_ptr_global(addr_i64, dtype, alignment=dtype.width // 8)


def store_i32_system(addr_i64, offset, val):
    """System-scope release i32 store at ``addr_i64 + offset*4``."""
    ptr = _to_fly_ptr_global(addr_i64, fx.Int32, alignment=4) + offset
    fx.generic_store(
        ptr,
        arith.unwrap(val),
        memory_order=fx.AtomicOrdering.Release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def store_i64_global_system(addr_i64, val):
    """System-scope release i64 store to ``addr_i64``."""
    fx.generic_store(
        _to_fly_ptr_global(addr_i64, fx.Int64, alignment=8),
        arith.unwrap(val),
        memory_order=fx.AtomicOrdering.Release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def fence_acquire(syncscope):
    """Emit an acquire fence for the selected AMDGPU memory scope."""
    fx.memory_fence(syncscope=syncscope, ordering=fx.AtomicOrdering.Acquire)


def fence_release(syncscope):
    """Emit a release fence for the selected AMDGPU memory scope."""
    fx.memory_fence(syncscope=syncscope, ordering=fx.AtomicOrdering.Release)


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


def load_i64_global(addr_i64):
    """Relaxed global i64 load from ``addr_i64``."""
    return fx.generic_load(_to_fly_ptr_global(addr_i64, fx.Int64, alignment=8))


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
