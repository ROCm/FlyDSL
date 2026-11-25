# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.mark.parametrize("divisor", [2, 4, 128])
def test_signed_vector_floordiv_power_of_two(divisor, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    block = 64
    vec = 4
    size = block * vec

    @flyc.kernel
    def kernel(a: fx.Tensor, out: fx.Tensor):
        tid = fx.thread_idx.x
        a_buf = fx.rocdl.make_buffer_tensor(a)
        out_buf = fx.rocdl.make_buffer_tensor(out)
        a_div = fx.logical_divide(a_buf, fx.make_layout(vec, 1))
        out_div = fx.logical_divide(out_buf, fx.make_layout(vec, 1))
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        r_a = fx.make_rmem_tensor(vec, fx.Int32)
        r_out = fx.make_rmem_tensor(vec, fx.Int32)
        fx.copy_atom_call(copy_atom, fx.slice(a_div, (None, tid)), r_a)
        values = fx.memref_load_vec(r_a)
        fx.memref_store_vec(values // divisor, r_out)
        fx.copy_atom_call(copy_atom, r_out, fx.slice(out_div, (None, tid)))

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    boundaries = torch.tensor(
        [
            -(1 << 31),
            -(1 << 31) + 1,
            -257,
            -256,
            -255,
            -129,
            -128,
            -127,
            -9,
            -8,
            -7,
            -1,
            0,
            1,
            7,
            8,
            9,
            127,
            128,
            129,
            255,
            256,
            257,
            (1 << 31) - 1,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    repeats = (size + boundaries.numel() - 1) // boundaries.numel()
    a = boundaries.repeat(repeats)[:size].contiguous()
    out = torch.empty_like(a)
    launch(flyc.from_torch_tensor(a), flyc.from_torch_tensor(out))
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), torch.div(a.cpu(), divisor, rounding_mode="floor"))
