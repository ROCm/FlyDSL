# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# isort: skip_file
from ._fly_ops_gen import *
from ._fly_ops_gen import mma_atom_call_ssa as _mma_atom_call_ssa
from ._fly_enum_gen import *

from .. import ir as _ir
from .._mlir_libs._mlirDialectsFly import *


def mma_atom_call_ssa(results_, mma_atom, a, b, c, *, d=None, loc=None, ip=None):
    """Accept legacy single A/B values as well as variadic operand groups."""
    a = [a] if isinstance(a, _ir.Value) else a
    b = [b] if isinstance(b, _ir.Value) else b
    return _mma_atom_call_ssa(results_, mma_atom, a, b, c, d=d, loc=loc, ip=ip)
