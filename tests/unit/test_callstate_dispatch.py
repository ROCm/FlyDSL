# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Fast-dispatch packing/ABI guards for the precompiled-call path.

These pin the behaviour of the ``flyc.compile`` fast path so the exec-generated
straight-line ``CallState`` dispatch and the in-place dynamic-layout pack keep
writing exactly the same bytes/pointers across changing arguments, including the
implicit auto-stream slot.  They assert the *observable packing contract* (what
ends up in the packed pointer array and the layout buffer), not the dispatch
implementation, so they hold regardless of loop-vs-codegen internals.
"""

import ctypes
import struct

import pytest

torch = pytest.importorskip("torch")

from flydsl.compiler import jit_argument as ja  # noqa: E402
from flydsl.compiler.jit_function import CallState  # noqa: E402
from flydsl.expr.numeric import Int32  # noqa: E402


def _expected_layout_bytes(t):
    """Canonical dynamic-layout buffer: dynamic-shape i32's then dynamic-stride
    i32/i64's, little-endian -- matches C++ buildMemRefDesc."""
    ad = ja.TensorAdaptor(t)
    sd, std, u32 = ad._shape_dyn_indices, ad._stride_dyn_indices, ad.use_32bit_stride
    out = struct.pack("<" + "i" * len(sd), *[t.shape[d] for d in sd])
    out += struct.pack("<" + ("i" if u32 else "q") * len(std), *[t.stride(d) for d in std])
    return out


def test_dynamic_layout_buffer_pack_bytes():
    """``TensorAdaptor._reusable_slot_spec`` returns (data-ptr, layout-buffer)
    for a dynamic tensor; the pack fn writes exactly the canonical bytes."""
    t = torch.empty((4, 8), dtype=torch.float32)  # row-major: dynamic layout
    spec = ja.TensorAdaptor._reusable_slot_spec(t)
    assert isinstance(spec, list) and len(spec) == 2

    (_dp_ctype, dp_extract), (buf_ctype, pack) = spec
    storage = buf_ctype()
    pack(t, storage)

    assert bytes(storage) == _expected_layout_bytes(t)
    assert dp_extract(t) == t.data_ptr()


def test_callstate_dispatch_packs_changing_args_and_auto_stream():
    """CallState fills the packed array correctly when called with new args each
    time: data ptr, dynamic layout bytes, scalar value, and a NULL auto-stream."""
    proto = torch.empty((4, 8), dtype=torch.float32)
    spec_t = ja.TensorAdaptor._reusable_slot_spec(proto)
    spec_i = Int32._reusable_slot_spec(0)
    # arg layout: arg0 = tensor (2 slots), arg1 = int (1 slot); + auto-stream NULL.
    slot_specs = [(0, *spec_t[0]), (0, *spec_t[1]), (1, *spec_i), (-1, ctypes.c_void_p, None)]

    captured = []

    def func_exe(packed):
        # Dereference each packed cell via its slot ctype to read the value the
        # kernel ABI would see; do not touch CallState internals.
        row = []
        for i, (_arg_idx, ctype, _extract) in enumerate(slot_specs):
            obj = ctype.from_address(packed[i])
            row.append(obj.value if hasattr(obj, "value") else bytes(obj))
        captured.append(row)
        return None

    cs = CallState(slot_specs, func_exe)

    for k in range(3):
        t = torch.empty((4, 8), dtype=torch.float32)  # distinct data_ptr each call
        ival = 100 + k
        cs((t, ival))

        data_ptr, layout, scalar, auto_stream = captured[-1]
        assert data_ptr == t.data_ptr()
        assert layout == _expected_layout_bytes(t)
        assert scalar == ival
        assert auto_stream in (None, 0)  # auto-stream slot stays NULL (default stream)
