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

import ctypes

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


# --------------------------------------------------------------------------- #
# Per-dimension dynamic marking: mark_shape_dynamic / mark_stride_dynamic     #
# --------------------------------------------------------------------------- #


def test_mark_shape_dynamic_shares_key_across_dynamic_dim():
    """Marking only dim 0 (M) shape-dynamic shares one kernel across all M."""
    a = flyc.from_dlpack(torch.empty((4, 128), dtype=torch.float32)).mark_shape_dynamic(0)
    b = flyc.from_dlpack(torch.empty((999, 128), dtype=torch.float32)).mark_shape_dynamic(0)
    assert a.__cache_signature__() == b.__cache_signature__()


def test_mark_shape_dynamic_static_dims_still_specialize():
    a = flyc.from_dlpack(torch.empty((4, 128), dtype=torch.float32)).mark_shape_dynamic(0)
    b = flyc.from_dlpack(torch.empty((4, 256), dtype=torch.float32)).mark_shape_dynamic(0)
    assert a.__cache_signature__() != b.__cache_signature__()


def test_mark_shape_dynamic_only_touches_shape():
    """mark_shape_dynamic marks the shape leaf only; strides stay untouched."""
    t = flyc.from_dlpack(torch.empty((8, 128), dtype=torch.float32)).mark_shape_dynamic(0, divisibility=16)
    *_, shape_tuple, stride_tuple = t.__cache_signature__()
    assert shape_tuple[0] == -16  # dim0 shape dynamic, div=16
    assert shape_tuple[1] == 128  # dim1 shape static
    assert stride_tuple == (128, 1)  # all strides untouched/static
    assert t._shape_dyn_indices == (0,)
    assert t._stride_dyn_indices == ()


def test_mark_shape_and_stride_accumulate_without_reset():
    """Chaining the two marks accumulates; neither resets the other's dims."""
    t = (
        flyc.from_dlpack(torch.empty((8, 16, 32), dtype=torch.float32))
        .mark_shape_dynamic(0, divisibility=16)
        .mark_stride_dynamic([0, 1], divisibility=8)
    )
    *_, shape_tuple, stride_tuple = t.__cache_signature__()
    assert shape_tuple == (-16, 16, 32)  # only dim0 shape dynamic
    assert stride_tuple[0] == -8 and stride_tuple[1] == -8  # dims 0,1 stride dynamic
    assert stride_tuple[2] == 1  # dim2 stride still static
    assert t._shape_dyn_indices == (0,)
    assert t._stride_dyn_indices == (0, 1)


def test_mark_dynamic_list_with_per_dim_divisibility():
    t = flyc.from_dlpack(torch.empty((8, 16, 32), dtype=torch.float32)).mark_stride_dynamic([0, 2], [8, 4])
    *_, _, stride_tuple = t.__cache_signature__()
    assert stride_tuple[0] == -8
    assert stride_tuple[2] == -4
    assert t._stride_dyn_indices == (0, 2)


def test_mark_dynamic_broadcast_divisibility():
    t = flyc.from_dlpack(torch.empty((8, 16, 32), dtype=torch.float32)).mark_shape_dynamic([0, 1], 4)
    *_, shape_tuple, _ = t.__cache_signature__()
    assert shape_tuple[0] == -4 and shape_tuple[1] == -4


def test_mark_dynamic_negative_index():
    t = flyc.from_dlpack(torch.empty((8, 128), dtype=torch.float32)).mark_shape_dynamic(-1)
    assert t._shape_dyn_indices == (1,)


def test_mark_dynamic_rejects_int_dims_with_list_divisibility():
    t = flyc.from_dlpack(torch.empty((8, 128), dtype=torch.float32))
    with pytest.raises(ValueError, match="divisibility must be an int"):
        t.mark_shape_dynamic(0, [1, 2])


def test_mark_dynamic_rejects_length_mismatch():
    t = flyc.from_dlpack(torch.empty((8, 16, 32), dtype=torch.float32))
    with pytest.raises(ValueError, match="equal length"):
        t.mark_stride_dynamic([0, 1], [1, 2, 3])


def test_mark_dynamic_rejects_out_of_range():
    t = flyc.from_dlpack(torch.empty((8, 128), dtype=torch.float32))
    with pytest.raises(ValueError, match="out of range"):
        t.mark_shape_dynamic(5)


def test_mark_dynamic_allows_duplicates_last_wins():
    """Duplicate dims are allowed; the last divisibility for a repeated dim wins."""
    t = flyc.from_dlpack(torch.empty((8, 128), dtype=torch.float32)).mark_stride_dynamic([0, 0], [8, 16])
    *_, _, stride_tuple = t.__cache_signature__()
    assert stride_tuple[0] == -16  # second entry (div=16) overwrote the first
    assert t._stride_dyn_indices == (0,)


def test_mark_dynamic_rejects_non_power_of_two_divisibility():
    t = flyc.from_dlpack(torch.empty((8, 16), dtype=torch.float32))
    with pytest.raises(ValueError, match="power of two"):
        t.mark_shape_dynamic(0, divisibility=3)
    with pytest.raises(ValueError, match="power of two"):
        t.mark_stride_dynamic([0, 1], [8, 6])
    with pytest.raises(ValueError, match="power of two"):
        t.mark_shape_dynamic(0, divisibility=0)


def test_mark_dynamic_accepts_power_of_two_divisibility():
    # 1 (== 2**0), 2, 16 are all valid.
    t = flyc.from_dlpack(torch.empty((8, 16, 32), dtype=torch.float32)).mark_stride_dynamic([0, 1, 2], [1, 2, 16])
    *_, _, stride_tuple = t.__cache_signature__()
    assert stride_tuple == (-1, -2, -16)


def test_mark_dynamic_layout_buffer_plan():
    """The reusable-slot packing plan sizes the layout buffer to exactly the
    dynamic leaves: one i32 per dynamic shape + one i64 (default stride) per
    dynamic stride, independently controlled.
    """
    t = (
        flyc.from_dlpack(torch.empty((8, 16, 32), dtype=torch.float32))
        .mark_shape_dynamic(0)
        .mark_stride_dynamic([0, 1])
    )
    spec = TensorAdaptor._reusable_slot_spec(t)
    assert isinstance(spec, list) and len(spec) == 2
    buf_ctype, _ = spec[1]
    # 1 dynamic shape * 4 bytes + 2 dynamic strides * 8 bytes = 20.
    assert ctypes.sizeof(buf_ctype) == 1 * 4 + 2 * 8
