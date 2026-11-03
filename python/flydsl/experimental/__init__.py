# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Experimental APIs, one subpackage per feature.

Unlike ``fx.<extension>`` libraries, features are not flattened into a common
namespace: each one is imported explicitly by its subpackage, so every use is
visibly experimental. For example::

    from flydsl.experimental import register_allocation as fx_regalloc

    fx_regalloc.register_place(frag_A, fx_regalloc.rocdl.AGPR32, start_idx=0)

Names here are not part of the stable ``fx`` surface and may change or be
removed without a deprecation period.
"""

from . import register_allocation as register_allocation

__all__ = [
    "register_allocation",
]
