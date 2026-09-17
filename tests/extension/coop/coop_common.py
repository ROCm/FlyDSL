#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Glue shared by the cooperative-algorithm tests."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None


def _powers_of_two(low, high):
    return tuple(1 << e for e in range(low.bit_length() - 1, high.bit_length()))


# The collectives size themselves off the target's wave, so the axes below have
# to follow it rather than a constant: 64 on CDNA, 32 on RDNA.
WARP_SIZE = current_target().warp_size

# Every logical warp width the collectives accept, narrowest first. Each is a
# power of two strictly inside the wave — a width equal to the wave is spelled
# ``None`` (the default), and it is last so a test that only wants the plain
# form can read it off the end.
WARP_WIDTHS = (*_powers_of_two(2, WARP_SIZE // 2), None)

# A single thread up to the 1024-thread launch limit, which is the whole range
# the block collectives are defined over. Below a wave they narrow their
# logical warp to the block rather than refusing it, so the widths under
# ``WARP_SIZE`` are the ones that exercise that narrowing.
BLOCK_THREADS = _powers_of_two(1, 1024)

# Just the block widths that fall inside a wave, for the tests that are about
# the narrowing itself rather than about width in general.
SUB_WARP_BLOCK_THREADS = _powers_of_two(1, WARP_SIZE // 2)

# The element types the collectives are exercised over, each paired with the
# name of the torch dtype it maps to. The pair travels together because a test
# needs the FlyDSL type to specialize the collective and the torch one to build
# its input; ``torch_dtype`` and ``dtype_id`` below take the name and the pair
# respectively.
#
# int16 stands in for uint16: the runtime has no memref element type for
# torch.uint16.
DTYPES = (
    (fx.Uint8, "torch.uint8"),
    (fx.Int16, "torch.int16"),
    (fx.Int32, "torch.int32"),
    (fx.Int64, "torch.int64"),
    (fx.Float32, "torch.float32"),
    (fx.Float64, "torch.float64"),
)

_TORCH_OF = {
    "torch.uint8": "uint8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.float32": "float32",
    "torch.float64": "float64",
}


def torch_dtype(name):
    """The ``torch`` dtype named by a :data:`DTYPES` entry."""
    return getattr(torch, _TORCH_OF[name])


def dtype_id(entry):
    """pytest id for a :data:`DTYPES` entry: just the FlyDSL type name."""
    return entry[0].__name__ if isinstance(entry, tuple) else str(entry)


def is_float(name):
    return torch_dtype(name).is_floating_point


def sample(name, n, *, seed=0):
    """Random input covering the dtype's range.

    Integers are drawn from the whole representable range so that summing them
    overflows — the wrap-around that :func:`wrap` reproduces on the host.
    """
    torch.manual_seed(seed)
    tdt = torch_dtype(name)
    if tdt.is_floating_point:
        return torch.randn(n, dtype=tdt, device="cuda")
    info = torch.iinfo(tdt)
    return torch.randint(info.min, info.max, (n,), dtype=tdt, device="cuda")


def wrap(total, name):
    """Fold a widened integer back into *name*'s range, the way the device does.

    ``torch.sum`` and ``torch.cumsum`` promote narrow integers to int64, so the
    reference has to be brought back down explicitly; the device arithmetic
    wraps at the type's own width.
    """
    tdt = torch_dtype(name)
    bits = 8 * tdt.itemsize
    if bits >= 64:
        # Already the width the device folded at; nothing to fold back.
        return total.to(tdt)
    wrapped = total % (1 << bits)
    if tdt.is_signed:
        wrapped = torch.where(wrapped >= (1 << (bits - 1)), wrapped - (1 << bits), wrapped)
    return wrapped.to(tdt)


def linear_tid(block_size):
    """Linear thread id inside *block_size*, ordered as ``gpu.thread_id`` is.

    Mirrors ``coop/_common.py:linear_thread_id``, which is what the collectives
    themselves index by — a test that ordered its threads differently would
    compare against the wrong permutation.
    """
    dim_x, dim_y, dim_z = block_size
    tid = fx.thread_idx.x
    if dim_y > 1:
        tid = tid + fx.thread_idx.y * fx.Int32(dim_x)
    if dim_z > 1:
        tid = tid + fx.thread_idx.z * fx.Int32(dim_x * dim_y)
    return tid
