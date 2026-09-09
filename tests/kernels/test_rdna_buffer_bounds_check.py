#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device tests for checked RDNA raw-buffer descriptors."""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not is_rdna_arch(_ARCH):
    pytest.skip(f"RDNA buffer descriptors require an RDNA target, got {_ARCH}", allow_module_level=True)


def _make_checked_load(*, records_source: str):
    def make_buffer(arg_a, arg_num_records):
        if records_source == "static":
            return fx.rocdl.make_buffer_tensor(arg_a, num_records_bytes=4)
        if records_source == "dynamic":
            return fx.rocdl.make_buffer_tensor(arg_a, num_records_bytes=arg_num_records)
        return fx.rocdl.make_buffer_tensor(arg_a, max_size=False)

    @flyc.kernel
    def checked_load_kernel(arg_a: fx.Tensor, arg_out: fx.Tensor, arg_num_records: fx.Int32):
        a_buf = make_buffer(arg_a, arg_num_records)
        tiled_a = fx.logical_divide(a_buf, fx.make_layout(1, 1))
        tid = fx.thread_idx.x

        fragment = fx.make_rmem_tensor(1, fx.Float32)
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        fx.copy_atom_call(copy_atom, fx.slice(tiled_a, (None, tid)), fragment)
        fx.ptr_store(Vec(fragment.load())[0], fx.get_iter(arg_out) + tid)

    @flyc.jit
    def launch(
        arg_a: fx.Tensor,
        arg_out: fx.Tensor,
        arg_num_records: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        checked_load_kernel(arg_a, arg_out, arg_num_records).launch(
            grid=(1, 1, 1),
            block=(2, 1, 1),
            stream=stream,
        )

    return launch


@pytest.mark.parametrize(
    "records_source",
    ["static", "dynamic", "derived"],
    ids=["explicit-static", "explicit-dynamic", "derived-layout"],
)
def test_num_records_enables_checked_buffer_load(records_source):
    """Finite descriptor sizes return zero for an out-of-bounds buffer load."""

    a = torch.tensor([7.0], device="cuda", dtype=torch.float32)
    out = torch.full((2,), -1.0, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()

    launch = _make_checked_load(records_source=records_source)
    launch(a, out, 4, stream=stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([7.0, 0.0]))
