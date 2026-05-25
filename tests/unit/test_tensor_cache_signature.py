#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for TensorAdaptor cache signature and the ``mark_static`` API.

Default cache key is lightweight (dtype + align + 32-bit-stride flag + rank +
layout ABI tag) so a single compiled kernel serves calls with different shapes.  When the
kernel's compiled IR depends on concrete shape values, the affected dims must
be marked with ``mark_static`` so they participate in the cache key.
"""

import ctypes
import struct

import pytest
import torch

import flydsl.compiler as flyc
from flydsl.compiler.jit_argument import TensorAdaptor


class _FakeLargeTensor:
    dtype = torch.float16
    shape = (2**31 + 1,)

    def data_ptr(self):
        return 0x1234

    def stride(self, dim=None):
        strides = (1,)
        return strides if dim is None else strides[dim]


# -----------------------------------------------------------------------------
# Default cache behavior: shape NOT in key, kernels reused across shapes
# -----------------------------------------------------------------------------


def test_default_cache_signature_shares_key_across_shapes():
    a = flyc.from_dlpack(torch.empty((4, 8), dtype=torch.float32))
    b = flyc.from_dlpack(torch.empty((100, 200), dtype=torch.float32))
    assert a.__cache_signature__() == b.__cache_signature__()


def test_default_cache_signature_differs_by_dtype():
    a = flyc.from_dlpack(torch.empty((4,), dtype=torch.float32))
    b = flyc.from_dlpack(torch.empty((4,), dtype=torch.float16))
    assert a.__cache_signature__() != b.__cache_signature__()


def test_default_cache_signature_differs_by_rank():
    a = flyc.from_dlpack(torch.empty((4,), dtype=torch.float32))
    b = flyc.from_dlpack(torch.empty((4, 1), dtype=torch.float32))
    assert a.__cache_signature__() != b.__cache_signature__()


def test_raw_cache_signature_matches_default_dlpack():
    """raw_cache_signature (lightweight path) and default __cache_signature__ agree."""
    t = torch.empty((4, 8), dtype=torch.float32)
    raw_sig = TensorAdaptor.raw_cache_signature(t)
    dlpack_sig = flyc.from_dlpack(t).__cache_signature__()
    assert raw_sig == dlpack_sig


def test_raw_cache_signature_shares_across_shapes():
    a = torch.empty((100,), dtype=torch.float32)
    b = torch.empty((999,), dtype=torch.float32)
    assert TensorAdaptor.raw_cache_signature(a) == TensorAdaptor.raw_cache_signature(b)


def test_raw_cache_signature_differs_by_rank():
    a = torch.empty((10,), dtype=torch.float32)
    b = torch.empty((2, 5), dtype=torch.float32)
    assert TensorAdaptor.raw_cache_signature(a) != TensorAdaptor.raw_cache_signature(b)


def test_dynamic_layout_packs_large_shape_dim_as_i64():
    tensor = _FakeLargeTensor()
    spec = TensorAdaptor._reusable_slot_spec(tensor)

    data_slot, layout_slot = spec
    data_ctype, data_extract = data_slot
    layout_ctype, layout_extract = layout_slot

    assert data_ctype is ctypes.c_void_p
    assert data_extract(tensor) == tensor.data_ptr()
    assert ctypes.sizeof(layout_ctype) == 8

    storage = layout_ctype()
    layout_extract(tensor, storage)

    assert struct.unpack_from("<q", memoryview(storage).cast("b"), 0)[0] == tensor.shape[0]


# -----------------------------------------------------------------------------
# mark_static: pin shape/stride values into the cache key
# -----------------------------------------------------------------------------


def test_mark_static_all_dims_includes_shape_in_key():
    a = flyc.from_dlpack(torch.empty((4, 8))).mark_static()
    b = flyc.from_dlpack(torch.empty((4, 8))).mark_static()
    c = flyc.from_dlpack(torch.empty((100, 200))).mark_static()
    assert a.__cache_signature__() == b.__cache_signature__()
    assert a.__cache_signature__() != c.__cache_signature__()


def test_mark_static_per_dim_only_marked_dim_in_key():
    """mark_static(dims=[1]) → dim 1 in key, dim 0 still 'dynamic'."""
    s0 = flyc.from_dlpack(torch.empty((4, 8))).mark_static(dims=[1])
    s1 = flyc.from_dlpack(torch.empty((100, 8))).mark_static(dims=[1])  # diff dim 0
    s2 = flyc.from_dlpack(torch.empty((4, 16))).mark_static(dims=[1])  # diff dim 1
    assert s0.__cache_signature__() == s1.__cache_signature__(), "dim 0 not in key"
    assert s0.__cache_signature__() != s2.__cache_signature__(), "dim 1 in key"


def test_mark_static_returns_self_for_chaining():
    t = flyc.from_dlpack(torch.empty((4, 8)))
    assert t.mark_static() is t
    assert t.mark_static(dims=[0]) is t


def test_mark_static_accumulates_dims():
    """Repeated calls union dim sets."""
    a = flyc.from_dlpack(torch.empty((4, 8, 16))).mark_static(dims=[0]).mark_static(dims=[2])
    b = flyc.from_dlpack(torch.empty((4, 8, 16))).mark_static(dims=[0, 2])
    assert a.__cache_signature__() == b.__cache_signature__()


def test_mark_static_out_of_range_raises():
    t = flyc.from_dlpack(torch.empty((4,)))
    with pytest.raises(IndexError):
        t.mark_static(dims=[5])
    with pytest.raises(IndexError):
        t.mark_static(dims=[-1])


def test_mark_static_default_vs_dynamic_differ_when_only_one_marked():
    """A marked tensor and a default tensor with the same shape produce different keys."""
    default = flyc.from_dlpack(torch.empty((4, 8)))
    marked = flyc.from_dlpack(torch.empty((4, 8))).mark_static()
    assert default.__cache_signature__() != marked.__cache_signature__()
