# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import ctypes
import struct

import torch

from flydsl.compiler.jit_argument import TensorAdaptor


class _FakeLargeTensor:
    dtype = torch.float16
    shape = (2**31 + 1,)

    def data_ptr(self):
        return 0x1234

    def stride(self, dim=None):
        strides = (1,)
        return strides if dim is None else strides[dim]


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
