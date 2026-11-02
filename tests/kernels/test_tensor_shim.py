# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Regressions for GTensor's typed pointer and byte-offset handling."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.common.tensor_shim import GTensor

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@flyc.kernel
def store_with_pointer_offset(dst: fx.Tensor):
    out = GTensor(fx.get_iter(dst), fx.BFloat16.ir_type, (8,), static_bytes_offset_i64=fx.Int64(8))
    out.store(fx.Int32(0), fx.Vector.filled(8, 1, fx.BFloat16), vec_size=8)


@flyc.jit
def launch_store(dst: fx.Tensor, stream: fx.Stream):
    store_with_pointer_offset(dst).launch(grid=(1,), block=(1,), stream=stream)


def test_gtensor_typed_pointer_byte_offset_preserves_canaries():
    output = torch.full((16,), -1, device="cuda", dtype=torch.bfloat16)
    launch_store(output, torch.cuda.current_stream())
    expected = torch.full_like(output, -1)
    expected[4:12] = 1
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
