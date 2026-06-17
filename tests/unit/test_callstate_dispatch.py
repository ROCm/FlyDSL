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


def _expected_layout_bytes(t, use_32bit=False):
    """Canonical dynamic-layout buffer: dynamic-shape i32's then dynamic-stride
    i32/i64's, little-endian -- matches C++ buildMemRefDesc."""
    ad = ja.TorchTensorJitArg(t, use_32bit_stride=use_32bit)
    sd, std, u32 = ad.shape_dyn_indices, ad.stride_dyn_indices, ad.use_32bit_stride
    out = struct.pack("<" + "i" * len(sd), *[t.shape[d] for d in sd])
    out += struct.pack("<" + ("i" if u32 else "q") * len(std), *[t.stride(d) for d in std])
    return out


def _layouts():
    """Diverse dynamic-layout tensors: contiguous, non-contiguous (transpose /
    permute), and higher rank -- each keeps a unit-stride axis (required for a
    layout-dynamic memref)."""
    return [
        ("contig_2d", torch.empty((4, 8), dtype=torch.float32)),
        ("contig_2d_f16", torch.empty((7, 13), dtype=torch.float16)),
        ("contig_3d", torch.empty((2, 3, 5), dtype=torch.float32)),
        ("transposed_2d", torch.empty((8, 4), dtype=torch.float32).t()),
        ("permuted_3d", torch.empty((2, 3, 5), dtype=torch.float32).permute(1, 0, 2)),
    ]


@pytest.mark.parametrize("name,t", _layouts(), ids=[n for n, _ in _layouts()])
@pytest.mark.parametrize("use_32bit", [False, True], ids=["stride64", "stride32"])
def test_dynamic_layout_buffer_pack_bytes(name, t, use_32bit):
    """``TorchTensorJitArg.__c_abi_spec__`` returns (data-ptr, layout-buffer)
    for a dynamic tensor; the in-place fills write exactly the canonical bytes,
    across contiguous/non-contiguous layouts, ranks, and stride widths."""
    adaptor = ja.TorchTensorJitArg(t, use_32bit_stride=use_32bit)
    slots = adaptor.__c_abi_spec__()
    assert isinstance(slots, list) and len(slots) == 2

    (dp_ctype, dp_fill), (buf_ctype, pack) = slots
    storage = buf_ctype()
    pack(t, storage)  # raw tensor at dispatch time (no _tensor_keepalive -> reads t directly)
    assert bytes(storage) == _expected_layout_bytes(t, use_32bit)

    dp = dp_ctype(0)
    dp_fill(t, dp)
    assert dp.value == t.data_ptr()


def test_callstate_dispatch_packs_changing_args_and_auto_stream():
    """CallState fills the packed array correctly when called with new args each
    time: data ptr, dynamic layout bytes, scalar value, and a NULL auto-stream."""
    proto = torch.empty((4, 8), dtype=torch.float32)
    slots_t = ja.TorchTensorJitArg(proto).__c_abi_spec__()  # [(ctype, fill)] x2 (data ptr + layout)
    slots_i = Int32(0).__c_abi_spec__()  # [(ctype, fill)]
    # arg layout: arg0 = tensor (2 slots), arg1 = int (1 slot); + auto-stream NULL.
    slot_specs = [(0, *slots_t[0]), (0, *slots_t[1]), (1, *slots_i[0]), (-1, ctypes.c_void_p, None)]

    captured = []

    def func_exe(packed):
        # Dereference each packed cell via its slot ctype to read the value the
        # kernel ABI would see; do not touch CallState internals.
        row = []
        for i, (_arg_idx, ctype, _fill) in enumerate(slot_specs):
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
