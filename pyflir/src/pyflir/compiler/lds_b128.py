"""Helpers to steer AMDGPU LDS instruction selection toward b128 ops.

This module provides a best-effort post-lowering rewrite on LLVM dialect IR:
- Detect 16-byte loads/stores from LDS (addrspace(3))
- Force `alignment=16` on those ops so LLVM AMDGPU can select ds_read_b128/ds_write_b128

This is intentionally opt-in via env vars in callers because it may be unsafe if
the program actually performs unaligned 16B LDS accesses.
"""

from __future__ import annotations

import re
from typing import Optional

from _mlir import ir
from _mlir.dialects import llvm


_RE_PTR_AS = re.compile(r"!llvm\.ptr<[^>]*,\s*(\d+)\s*>")
_RE_I_BITS = re.compile(r"!llvm\.i(\d+)$")
_RE_VEC = re.compile(r"!llvm\.vec<\s*(\d+)\s*x\s*([^>]+)\s*>$")
_RE_ARRAY = re.compile(r"!llvm\.array<\s*(\d+)\s*x\s*([^>]+)\s*>$")


def _llvm_ptr_addrspace(type_str: str) -> Optional[int]:
    """Parse LLVM dialect pointer addrspace from a type string."""
    m = _RE_PTR_AS.search(type_str)
    if not m:
        # Some builds print addrspace in a different form; best-effort fallback.
        if "addrspace(3)" in type_str:
            return 3
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _llvm_type_nbytes(type_str: str) -> Optional[int]:
    """Best-effort compute byte size for common LLVM dialect types."""
    type_str = type_str.strip()

    m = _RE_I_BITS.match(type_str)
    if m:
        bits = int(m.group(1))
        if bits % 8 != 0:
            return None
        return bits // 8

    m = _RE_VEC.match(type_str)
    if m:
        n = int(m.group(1))
        elem = m.group(2).strip()
        elem_bytes = _llvm_type_nbytes(elem)
        return None if elem_bytes is None else n * elem_bytes

    m = _RE_ARRAY.match(type_str)
    if m:
        n = int(m.group(1))
        elem = m.group(2).strip()
        elem_bytes = _llvm_type_nbytes(elem)
        return None if elem_bytes is None else n * elem_bytes

    # Unknown / structs / pointers, etc.
    return None


def force_lds_b128_alignment(module: ir.Module, *, nbytes: int = 16, alignment: int = 16) -> int:
    """Force alignment on 16B LLVM loads/stores from LDS addrspace(3).

    Returns the number of ops updated.
    """
    ctx = module.context
    i64 = ir.IntegerType.get_signless(64, context=ctx)
    align_attr = ir.IntegerAttr.get(i64, int(alignment))

    updated = 0

    def _maybe_update(op) -> ir.WalkResult:
        nonlocal updated

        if op.name not in {"llvm.load", "llvm.store"}:
            return ir.WalkResult.ADVANCE

        # Pointer operand is first for both llvm.load and llvm.store in MLIR LLVM dialect.
        # - load:  operand[0] = ptr
        # - store: operand[0] = value, operand[1] = ptr
        try:
            if op.name == "llvm.load":
                ptr = op.operands[0]
                val_type = op.result.type
            else:
                ptr = op.operands[1]
                val_type = op.operands[0].type
        except Exception:
            return ir.WalkResult.ADVANCE

        as_ = _llvm_ptr_addrspace(str(ptr.type))
        if as_ != 3:
            return ir.WalkResult.ADVANCE

        size = _llvm_type_nbytes(str(val_type))
        if size != nbytes:
            return ir.WalkResult.ADVANCE

        # Don't decrease an existing alignment.
        try:
            existing = op.attributes.get("alignment", None)
            if existing is not None:
                # IntegerAttr prints as an integer literal.
                if int(str(existing)) >= alignment:
                    return ir.WalkResult.ADVANCE
        except Exception:
            pass

        op.attributes["alignment"] = align_attr
        updated += 1
        return ir.WalkResult.ADVANCE

    module.operation.walk(_maybe_update)
    return updated


def _get_int_constant_value(v: ir.Value) -> Optional[int]:
    """Return integer value if `v` is defined by llvm.mlir.constant, else None."""
    try:
        op = v.get_defining_op()
    except Exception:
        op = None
    if op is None or op.name != "llvm.mlir.constant":
        return None
    try:
        value_attr = op.attributes.get("value", None)
        if value_attr is None:
            return None
        # IntegerAttr prints like "144 : i64" in MLIR, but str(attr) may vary.
        if isinstance(value_attr, ir.IntegerAttr):
            return int(value_attr.value)
        s = str(value_attr).strip()
        # Best-effort parse leading integer.
        return int(s.split()[0])
    except Exception:
        return None


def reassociate_lds_gep_adds(module: ir.Module, *, addrspace: int = 3) -> int:
    """Reassociate integer adds feeding LDS GEP indices to expose constant offsets.

    Pattern:
      %a = llvm.add %x, %c
      %b = llvm.add %base, %a
    Rewrites to:
      %t = llvm.add %base, %x
      %b2 = llvm.add %t, %c
    This helps AMDGPU fold `%c` into DS immediate offset instead of burning an extra vaddr VGPR.

    This is best-effort and intentionally conservative:
    - Only rewrites when the outer add directly feeds an `llvm.getelementptr` index
      whose base pointer is in `addrspace`.
    - Only handles integer-typed adds (i32/i64).

    Returns the number of `llvm.add` ops rewritten.
    """
    rewritten = 0

    def _walk(op) -> ir.WalkResult:
        nonlocal rewritten
        if op.name != "llvm.getelementptr":
            return ir.WalkResult.ADVANCE

        # Operand 0 is base pointer, operand 1.. are indices.
        try:
            base_ptr = op.operands[0]
        except Exception:
            return ir.WalkResult.ADVANCE

        if _llvm_ptr_addrspace(str(base_ptr.type)) != addrspace:
            return ir.WalkResult.ADVANCE

        # Only 1D GEPs are handled here (our LDS buffer is linearized).
        if len(op.operands) < 2:
            return ir.WalkResult.ADVANCE
        idx = op.operands[1]

        outer = idx.get_defining_op() if hasattr(idx, "get_defining_op") else None
        if outer is None or outer.name != "llvm.add":
            return ir.WalkResult.ADVANCE

        # outer: add(base, a)
        a0, a1 = outer.operands[0], outer.operands[1]
        # Identify inner add among operands.
        inner_val = None
        base_val = None
        for cand_inner, cand_base in [(a0, a1), (a1, a0)]:
            inner_op = cand_inner.get_defining_op() if hasattr(cand_inner, "get_defining_op") else None
            if inner_op is not None and inner_op.name == "llvm.add":
                inner_val = cand_inner
                base_val = cand_base
                break
        if inner_val is None or base_val is None:
            return ir.WalkResult.ADVANCE

        inner = inner_val.get_defining_op()
        b0, b1 = inner.operands[0], inner.operands[1]
        const_operand = b0
        const_val = _get_int_constant_value(const_operand)
        x_val = b1
        if const_val is None:
            const_operand = b1
            const_val = _get_int_constant_value(const_operand)
            x_val = b0
        if const_val is None:
            return ir.WalkResult.ADVANCE

        # Create new ops right before the outer add.
        with ir.InsertionPoint(outer):
            t = llvm.AddOp(base_val, x_val).result
            new_idx = llvm.AddOp(t, const_operand).result

        # Replace idx operand in GEP.
        try:
            op.operands[1] = new_idx
        except Exception:
            # Some bindings don't allow direct mutation; bail.
            return ir.WalkResult.ADVANCE

        rewritten += 1
        return ir.WalkResult.ADVANCE

    module.operation.walk(_walk)
    return rewritten


