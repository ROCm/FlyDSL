# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("setter", ["field", "dict", "primitive"])
def test_tiled_copy_state_controls_buffer_offset(default_device, setter):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    @flyc.kernel
    def kernel(src: fx.Tensor, dst: fx.Tensor, offset: fx.Int32):
        tid = fx.thread_idx.x
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tiled = fx.make_tiled_copy(atom, fx.make_layout((64, 1), (1, 64)), (64, 1))
        if fx.const_expr(setter == "field"):
            updated = tiled.set_value("soffset", offset)
        elif fx.const_expr(setter == "dict"):
            updated = tiled.set_value({"soffset": offset})
        else:
            updated = fx.atom_set_value(tiled, "soffset", offset)
        src_tiles = fx.logical_divide(fx.rocdl.make_buffer_tensor(src), fx.make_layout(1, 1))
        src_tile = fx.slice(src_tiles, (None, tid))
        original_value = fx.make_rmem_tensor(1, fx.Float32)
        updated_value = fx.make_rmem_tensor(1, fx.Float32)
        fx.copy(tiled, src_tile, original_value)
        fx.copy(updated, src_tile, updated_value)
        dst[tid] = updated_value[0]
        dst[tid + 64] = original_value[0]

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, offset: fx.Int32, stream: fx.Stream):
        kernel(src, dst, offset).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    with torch.device(default_device):
        src = torch.arange(128, dtype=torch.float32, device="cuda")
        dst = torch.full((128,), -1.0, dtype=torch.float32, device="cuda")
        launch(src, dst, 64, torch.cuda.current_stream())
        expected = torch.cat(
            [
                torch.arange(64, 128, dtype=torch.float32, device="cpu"),
                torch.arange(64, dtype=torch.float32, device="cpu"),
            ]
        )
        torch.testing.assert_close(dst.cpu(), expected, atol=0, rtol=0)
