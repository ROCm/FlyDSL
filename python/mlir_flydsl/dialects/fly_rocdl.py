# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# isort: skip_file
# ruff: noqa: F401,F403
from ._fly_rocdl_ops_gen import *
from ._fly_rocdl_ops_gen import _Dialect
from ._fly_rocdl_enum_gen import *

from .._mlir_libs._mlirDialectsFlyROCDL import *


class _TargetAddressSpace:
    @property
    def BufferDesc(self):
        from .. import ir

        return ir.Attribute.parse("#fly_rocdl.buffer_desc")


TargetAddressSpace = _TargetAddressSpace()
