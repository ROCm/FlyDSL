# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""NVVM dialect extension for NVIDIA GPU programming.

Target-specific tensor-core MMA atom and async/ldmatrix copy atom types for the
NVVM backend, plus the cp.async commit/wait-group barriers. Lazy-loaded from
``flydsl.expr`` so that ``import flydsl.expr`` does not require the FlyNVVM
Python bindings to be present.
"""

from ..._mlir.dialects.nvvm import cp_async_commit_group as cp_async_commit_group
from ..._mlir.dialects.nvvm import cp_async_wait_group as cp_async_wait_group
from ..meta import dsl_loc_tracing
from .universal import *


@dsl_loc_tracing
def commit_group():
    """Commit pending cp.async operations into a group (cp.async.commit_group)."""
    return cp_async_commit_group()


@dsl_loc_tracing
def wait_group(n):
    """Wait until at most n cp.async groups remain in flight (cp.async.wait_group)."""
    return cp_async_wait_group(n)
