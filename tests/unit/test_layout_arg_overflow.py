#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""A layout-dynamic tensor argument passes each dynamic dimension as a signed
32-bit integer (#1176). A dimension that does not fit must fail with a message
naming it, not with struct's bare ``'i' format requires ...``.

Only the layout slot's host-side packing is exercised, so ``meta`` tensors
stand in for multi-GiB ones and no GPU is needed.
"""

import struct

import numpy as np
import pytest
import torch

import flydsl.compiler as flyc
from flydsl.compiler.jit_argument import TorchTensorJitArg

pytestmark = [pytest.mark.l0_backend_agnostic]

I32_MAX = 2**31 - 1


def _pack_layout(t, **kw):
    ctype, fill = TorchTensorJitArg(t, **kw).__c_abi_spec__()[1]
    buf = ctype()
    fill(t, buf)
    return bytes(buf)


def _meta(shape, stride):
    return torch.empty_strided(shape, stride, dtype=torch.float32, device="meta")


def test_largest_i32_dimension_packs():
    # Positive control: the boundary value itself must still go through, byte-exact.
    assert _pack_layout(_meta((I32_MAX,), (1,))) == struct.pack("<i", I32_MAX)


def test_multi_dim_with_large_product_packs():
    # Only each dimension is limited; a 2.5e9-element 2-D tensor is fine.
    assert _pack_layout(_meta((50000, 50000), (50000, 1))) == struct.pack("<iiq", 50000, 50000, 50000)


def test_oversized_dimension_names_the_dimension():
    with pytest.raises(OverflowError, match=r"tensor shape\[0\] = 2147483648 .*signed 32-bit") as exc:
        _pack_layout(_meta((I32_MAX + 1,), (1,)))
    # The struct.error it replaces must not be chained in front of it.
    assert exc.value.__suppress_context__


def test_layout_dynamic_without_dynamic_dims_packs_nothing():
    # Marked layout-dynamic, but no dimension is actually dynamic: an empty buffer.
    t = _meta((4, 8), (8, 1))
    ctype, fill = flyc.from_torch_tensor(t).mark_shape_dynamic([]).__c_abi_spec__()[1]
    buf = ctype()
    fill(t, buf)
    assert bytes(buf) == b""


def test_oversized_inner_dimension_is_located():
    with pytest.raises(OverflowError, match=r"tensor shape\[1\] = 2147483648 "):
        _pack_layout(_meta((3, I32_MAX + 1), (I32_MAX + 1, 1)))


def test_oversized_32bit_stride_is_reported():
    with pytest.raises(OverflowError, match=r"tensor stride\[0\] = 2147483648 .*use_32bit_stride"):
        _pack_layout(_meta((4, 2**30), (2**31, 1)), use_32bit_stride=True)


def test_dlpack_path_reports_oversized_dimension():
    # The DLPack-backed argument generates its own fill; it must use the same check.
    # A strided view of a single byte: nothing is allocated or touched.
    t = np.lib.stride_tricks.as_strided(np.zeros(1, dtype=np.int8), shape=(I32_MAX + 1,), strides=(1,))
    slots = flyc.from_dlpack(t).mark_layout_dynamic().__c_abi_spec__()
    (ptr_ctype, ptr_fill), (layout_ctype, layout_fill) = slots
    ptr_fill(t, ptr_ctype())  # opens the DLPack capsule the layout fill reads
    with pytest.raises(OverflowError, match=r"tensor shape\[0\] = 2147483648 "):
        layout_fill(t, layout_ctype())
