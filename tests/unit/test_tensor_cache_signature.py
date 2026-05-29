#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for TensorAdaptor cache signatures.

Two adaptation paths produce two distinct cache-key shapes:

* ``flyc.from_dlpack(t)`` returns a *static-layout* TensorAdaptor: shape and
  stride are baked into the memref type, so every distinct shape ends up
  with its own compiled kernel. Chain ``.mark_layout_dynamic()`` to switch
  to a layout-dynamic memref whose key elides shape/stride (one compile
  serves all shapes).

* Raw ``torch.Tensor`` arguments go through the auto-adapt path
  (``TensorAdaptor(t)`` with ``dynamic_layout=True``) and behave like
  ``from_dlpack(t).mark_layout_dynamic()``: layout-dynamic memref, no
  shape/stride in the cache key.
"""

import pytest
import torch

import flydsl.compiler as flyc
from flydsl.compiler.jit_argument import TensorAdaptor


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


def test_auto_adapted_cache_signature_shares_across_shapes():
    """Raw tensors hit the layout-dynamic memref path; the cache key elides shape/stride so one compile serves all shapes."""
    a = torch.empty((100,), dtype=torch.float32)
    b = torch.empty((999,), dtype=torch.float32)
    assert TensorAdaptor(a).__cache_signature__() == TensorAdaptor(b).__cache_signature__()


def test_auto_adapted_cache_signature_differs_by_rank():
    a = torch.empty((10,), dtype=torch.float32)
    b = torch.empty((2, 5), dtype=torch.float32)
    assert TensorAdaptor(a).__cache_signature__() != TensorAdaptor(b).__cache_signature__()


def test_pick_unit_stride_axis_returns_first_match():
    """When several axes carry stride 1 (typical with degenerate axes), the
    helper returns the lowest qualifying index. Example: shape (4, 1, 8, 1)
    strides (8, 8, 1, 1) — axes 2 and 3 both qualify, axis 2 is returned.
    """
    t = torch.empty((4, 1, 8, 1), dtype=torch.float32)
    assert TensorAdaptor._pick_unit_stride_axis(t.stride()) == 2


def test_pick_unit_stride_axis_raises_without_unit_stride():
    """Strided slices have no axis with stride 1; raise instead of returning None."""
    sliced = torch.empty((4, 8))[:, ::2]  # strides (8, 2)
    with pytest.raises(RuntimeError, match="stride == 1"):
        TensorAdaptor._pick_unit_stride_axis(sliced.stride())


def test_auto_adapt_handles_size_one_degeneracies():
    """Tensors with several stride-1 axes (size-1 unsqueeze, size-0 axes
    whose stride PyTorch / DLPack happens to set to 1) must not silently
    drop into a static memref — they should stay layout-dynamic with the
    earliest unit-stride axis chosen.
    """
    # Fully degenerate (1, 1) tensor: every axis has stride 1; first wins.
    assert TensorAdaptor(torch.empty((1, 1)))._dyn_leading_dim == 0
    # (0, 8) is a real production case (size-0 outer axis). PyTorch's
    # stride view has only axis 1 at stride 1, so that's what we pick.
    assert TensorAdaptor(torch.empty((0, 8)))._dyn_leading_dim == 1


def test_auto_adapt_raises_when_no_unit_stride_axis():
    """If no axis has stride 1 at all (e.g. a strided slice) the tensor
    cannot be layout-dynamic; raise with an actionable hint instead of
    silently falling back to a static memref (which would pin shape into
    the cache key and trigger surprise per-shape recompiles).
    """
    base = torch.empty((4, 8), dtype=torch.float32)
    sliced = base[:, ::2]  # shape (4, 4) strides (8, 2) — no unit stride
    with pytest.raises(RuntimeError, match="auto-mark layout-dynamic"):
        TensorAdaptor(sliced)
    # Explicit escape hatch still works:
    flyc.from_dlpack(sliced)  # static memref, shape participates in key
