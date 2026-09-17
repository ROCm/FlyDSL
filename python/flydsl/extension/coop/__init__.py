# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cooperative algorithms over the threads of one kernel launch.

Layout — one subpackage per scope, one module per algorithm::

    coop/
    ├── _common.py     glue both scopes need
    ├── universal.py   the portable forms, dispatch turned off
    ├── warp/
    │   ├── reduce.py      warp_reduce
    │   ├── scan.py        warp_inclusive_scan, warp_exclusive_scan, warp_scan,
    │   │                      warp_scan_with_aggregate
    │   └── rocdl.py       ROCm overrides for the above
    └── block/
        ├── _spec.py       shared [...] specialization machinery
        ├── reduce.py      BlockReduce
        └── scan.py        BlockScan

That surface is flat — ``fx.coop.<name>`` — so callers never spell the scope
out twice (``fx.coop.warp_reduce``, not ``fx.coop.warp.warp_reduce``).

Those names are dispatched: on a target with an override, ``fx.coop.warp_reduce``
is the target's. ``fx.coop.universal`` is the same surface with dispatch turned
off — see :mod:`flydsl.extension.coop.universal`.
"""

from . import block as block
from . import universal as universal
from . import warp as warp
from .block import *
from .warp import *

__all__ = [
    # warp scope
    "warp_reduce",
    "warp_inclusive_scan",
    "warp_exclusive_scan",
    "warp_scan",
    "warp_scan_with_aggregate",
    # block scope
    "BlockReduceAlgorithm",
    "BlockReduce",
    "BlockScanAlgorithm",
    "BlockScan",
]
