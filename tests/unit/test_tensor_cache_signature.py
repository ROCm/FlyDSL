#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for TensorAdaptor cache signature and the ``mark_static`` API.

``flyc.from_dlpack(t)`` returns a *static-layout* TensorAdaptor: every dim's
shape and stride participates in the JIT cache key, so distinct shapes get
distinct compiled kernels (safe default for shape-baked IR).  To share one
compiled kernel across shapes, chain ``.mark_layout_dynamic()`` -- that
switches the memref to layout-dynamic and elides shape/stride from the key
(only the leading stride-1 dim and divisibility stay).

Raw ``torch.Tensor`` arguments go through the auto-adapt path
(``TensorAdaptor(t)`` with ``dynamic_layout=True``).  They get a
layout-dynamic memref and the cache key elides shape/stride: one compile
serves all shapes, mirroring the long-standing behaviour.  Kernels that
bake concrete shape values at trace time must opt into shape-pinning via
``mark_static(dims=[...])``.
"""

import pytest
import torch

import flydsl.compiler as flyc
from flydsl.compiler.jit_argument import TensorAdaptor

# -----------------------------------------------------------------------------
# Default cache behavior: shape NOT in key, kernels reused across shapes
# -----------------------------------------------------------------------------


def test_dynamic_layout_cache_signature_shares_key_across_shapes():
    a = flyc.from_dlpack(torch.empty((4, 8), dtype=torch.float32)).mark_layout_dynamic()
    b = flyc.from_dlpack(torch.empty((100, 200), dtype=torch.float32)).mark_layout_dynamic()
    assert a.__cache_signature__() == b.__cache_signature__()


def test_default_static_cache_signature_differs_by_shape():
    """``from_dlpack`` defaults to static layout: shape participates in the key."""
    a = flyc.from_dlpack(torch.empty((4, 8), dtype=torch.float32))
    b = flyc.from_dlpack(torch.empty((100, 200), dtype=torch.float32))
    assert a.__cache_signature__() != b.__cache_signature__()


def test_default_cache_signature_differs_by_dtype():
    a = flyc.from_dlpack(torch.empty((4,), dtype=torch.float32))
    b = flyc.from_dlpack(torch.empty((4,), dtype=torch.float16))
    assert a.__cache_signature__() != b.__cache_signature__()


def test_default_cache_signature_differs_by_rank():
    a = flyc.from_dlpack(torch.empty((4,), dtype=torch.float32))
    b = flyc.from_dlpack(torch.empty((4, 1), dtype=torch.float32))
    assert a.__cache_signature__() != b.__cache_signature__()


def test_raw_cache_signature_matches_auto_adapted_tensor():
    """Lightweight raw_cache_signature matches the full __cache_signature__ of an auto-adapted (dynamic_layout=True) TensorAdaptor, so fast/slow paths share the same cache slot."""
    t = torch.empty((4, 8), dtype=torch.float32)
    raw_sig = TensorAdaptor.raw_cache_signature(t)
    auto_sig = TensorAdaptor(t).__cache_signature__()
    assert raw_sig == auto_sig


def test_raw_cache_signature_shares_across_shapes():
    """Raw tensors hit the layout-dynamic memref path; the cache key elides shape/stride so one compile serves all shapes."""
    a = torch.empty((100,), dtype=torch.float32)
    b = torch.empty((999,), dtype=torch.float32)
    assert TensorAdaptor.raw_cache_signature(a) == TensorAdaptor.raw_cache_signature(b)


def test_raw_cache_signature_differs_by_rank():
    a = torch.empty((10,), dtype=torch.float32)
    b = torch.empty((2, 5), dtype=torch.float32)
    assert TensorAdaptor.raw_cache_signature(a) != TensorAdaptor.raw_cache_signature(b)


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
    """On a layout-dynamic adaptor, mark_static(dims=[1]) → dim 1 in key, dim 0 stays elided."""
    s0 = flyc.from_dlpack(torch.empty((4, 8))).mark_layout_dynamic().mark_static(dims=[1])
    s1 = flyc.from_dlpack(torch.empty((100, 8))).mark_layout_dynamic().mark_static(dims=[1])  # diff dim 0
    s2 = flyc.from_dlpack(torch.empty((4, 16))).mark_layout_dynamic().mark_static(dims=[1])  # diff dim 1
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


def test_mark_layout_dynamic_rejects_ambiguous_leading_dim():
    """Shape (1, 1) has strides (1, 1) — two stride-1 dims. The C++ adaptor
    rejects this via ``markLayoutDynamic`` and the Python wrapper surfaces it
    as a RuntimeError instead of silently picking a leading dim.
    """
    t = flyc.from_dlpack(torch.empty((1, 1), dtype=torch.float32))
    with pytest.raises(RuntimeError):
        t.mark_layout_dynamic()


def test_mark_static_dynamic_vs_marked_differ():
    """A layout-dynamic tensor (shape elided) and the same tensor with mark_static() pinning all dims produce different keys."""
    dyn = flyc.from_dlpack(torch.empty((4, 8))).mark_layout_dynamic()
    marked = flyc.from_dlpack(torch.empty((4, 8))).mark_layout_dynamic().mark_static()
    assert dyn.__cache_signature__() != marked.__cache_signature__()
