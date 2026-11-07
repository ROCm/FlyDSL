# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register-placement hints that are not in upstream LLVM.

``rocdl.sched.bank`` / ``rocdl.sched.vgpr`` / ``rocdl.sched.sgpr`` /
``rocdl.sched.agpr`` come from the local ``schedule-bank-and-gpr`` patch.
Any path under this module is unstable (``docs/api_stability.md`` §2).
"""

from ..meta import dsl_loc_tracing

__all__ = [
    "sched_bank",
    "sched_vgpr",
    "sched_sgpr",
    "sched_agpr",
]


def _ods(name):
    from ..._mlir.dialects import rocdl as _rocdl

    op = getattr(_rocdl, name, None)
    if op is None:
        raise AttributeError(f"ROCDL op not found: {name} (needs the local schedule-bank-and-gpr LLVM patch)")
    return op


def _check_regnum(regnum):
    if regnum is not None and regnum < 0:
        raise ValueError(f"regnum must be a non-negative register number or None, got {regnum}")
    return regnum


def _to_ir(v):
    """Coerce DSL Numeric to ir.Value if needed."""
    from ..._mlir import ir as _ir
    from .. import arith as _arith_ext

    if isinstance(v, int):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=_ir.IntegerType.get_signless(32)))
    if isinstance(v, float):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=_ir.F32Type.get()))
    if not isinstance(v, _ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


@dsl_loc_tracing
def sched_bank(value, bank, soft=False, **kw):
    """VGPR bank scheduling hint (gfx1250).

    Returns *value* unchanged (identity at IR level).  Guides the register
    allocator to place *value* in the requested 256-register bank (0-3).
    Default is strict; pass ``soft=True`` for an advisory hint.
    """
    val_ir = _to_ir(value)
    return _ods("sched_bank")(res=val_ir.type, value=val_ir, bank=bank, soft=soft or None, **kw)


@dsl_loc_tracing
def sched_vgpr(value, regnum=None, soft=False, **kw):
    """VGPR exact-register scheduling hint.

    Returns *value* unchanged (identity at IR level).  Guides the register
    allocator to place *value* in the specified VGPR number (0-1023).
    For wide values (64/128/256-bit), *regnum* is the starting register.
    With ``regnum=None`` *value* only has to stay in the VGPR file, in any
    register.
    Default is strict; pass ``soft=True`` for an advisory hint.
    """
    val_ir = _to_ir(value)
    return _ods("sched_vgpr")(res=val_ir.type, value=val_ir, regnum=_check_regnum(regnum), soft=soft or None, **kw)


@dsl_loc_tracing
def sched_sgpr(value, regnum=None, soft=False, **kw):
    """SGPR exact-register scheduling hint.

    Returns *value* unchanged (identity at IR level).  Guides the register
    allocator to place *value* in the specified SGPR number (0-105).
    For wide values (64/128/256-bit), *regnum* is the starting register.
    With ``regnum=None`` *value* only has to stay in the SGPR file, in any
    register.
    Default is strict; pass ``soft=True`` for an advisory hint.
    """
    val_ir = _to_ir(value)
    return _ods("sched_sgpr")(res=val_ir.type, value=val_ir, regnum=_check_regnum(regnum), soft=soft or None, **kw)


@dsl_loc_tracing
def sched_agpr(value, regnum=None, soft=False, **kw):
    """AGPR exact-register scheduling hint (gfx90a/gfx940).

    Returns *value* unchanged (identity at IR level).  Guides the register
    allocator to place *value* in the specified AGPR number (0-255).
    For wide values (64/128/256-bit), *regnum* is the starting register.
    With ``regnum=None`` *value* only has to stay in the AGPR file, in any
    register.
    Default is strict; pass ``soft=True`` for an advisory hint.
    Only valid on targets with MAI instructions.
    """
    val_ir = _to_ir(value)
    return _ods("sched_agpr")(res=val_ir.type, value=val_ir, regnum=_check_regnum(regnum), soft=soft or None, **kw)
