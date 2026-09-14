# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# isort: skip_file
from .. import ir as _ir
from . import _fly_ops_gen
from ._fly_ops_gen import *
from ._fly_enum_gen import *

from .._mlir_libs._mlirDialectsFly import *


def _normalize_mma_operand_group(operand):
    """Keep the pre-variadic Python builder API source-compatible."""
    return [operand] if isinstance(operand, _ir.Value) else operand


def gemm(
    mma_atom,
    d,
    a,
    b,
    c,
    *,
    traversal_order=None,
    traversal_layout=None,
    loc=None,
    ip=None,
):
    return _fly_ops_gen.gemm(
        mma_atom,
        d,
        _normalize_mma_operand_group(a),
        _normalize_mma_operand_group(b),
        c,
        traversal_order=traversal_order,
        traversal_layout=traversal_layout,
        loc=loc,
        ip=ip,
    )


def mma_atom_call(mma_atom, d, a, b, c, *, loc=None, ip=None):
    return _fly_ops_gen.mma_atom_call(
        mma_atom,
        d,
        _normalize_mma_operand_group(a),
        _normalize_mma_operand_group(b),
        c,
        loc=loc,
        ip=ip,
    )


def mma_atom_call_ssa(results_, mma_atom, a, b, c, *, d=None, loc=None, ip=None):
    return _fly_ops_gen.mma_atom_call_ssa(
        results_,
        mma_atom,
        _normalize_mma_operand_group(a),
        _normalize_mma_operand_group(b),
        c,
        d=d,
        loc=loc,
        ip=ip,
    )
