# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Contracts for the shared softmax row-copy helpers."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.norm.softmax_common import _load_vec, _store_vec

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@pytest.mark.parametrize("dtype_str,dtype", [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)])
@pytest.mark.parametrize("scalar", [False, True])
def test_masked_copy_contract(dtype_str, dtype, scalar):
    """Check valid loads, safe inactive addresses, and suppressed stores."""
    width = 1 if scalar else (4 if dtype_str == "f32" else 8)
    source = torch.arange(3 * width, device="cuda", dtype=torch.float32).to(dtype).view(1, -1)
    output = torch.empty((1, 32 * width), device="cuda", dtype=dtype)
    guarded = torch.full_like(output, 123)

    @flyc.kernel
    def copy_kernel(source: fx.Tensor, output: fx.Tensor, guarded: fx.Tensor):
        tid = fx.thread_idx.x
        src = fx.slice(fx.rocdl.make_buffer_tensor(source), (0, None))
        dst = fx.slice(fx.rocdl.make_buffer_tensor(output), (0, None))
        guard = fx.slice(fx.rocdl.make_buffer_tensor(guarded), (0, None))
        value = _load_vec(src, tid, width, dtype_str, mask=tid < 3)
        _store_vec(dst, tid, value, width, dtype_str, mask=tid < 32)
        _store_vec(guard, tid, value, width, dtype_str, mask=tid < 3)

    @flyc.jit
    def launch(source: fx.Tensor, output: fx.Tensor, guarded: fx.Tensor):
        copy_kernel(source, output, guarded).launch(grid=(1, 1, 1), block=(32, 1, 1))

    launch(source, output, guarded)
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:, : 3 * width], source, rtol=0, atol=0)
    torch.testing.assert_close(guarded[:, : 3 * width], source, rtol=0, atol=0)
    assert bool((guarded[:, 3 * width :] == 123).all())
