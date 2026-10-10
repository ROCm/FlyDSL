# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device bool→additive attention-mask conversion for gfx120x FA hosts.

Writes 0.0 for True and -inf for False into a contiguous fp32 buffer so
normalize_attn_mask does not call host ``masked_fill``.
"""

from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

_BLOCKS = (256,)


@lru_cache(maxsize=4)
def build_bool_mask_to_bias_module(block_threads: int = 256):
    block = int(block_threads)
    if block not in _BLOCKS:
        raise ValueError(f"bool mask block_threads={block} not in {_BLOCKS}")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def bool_mask_kernel(Mask: fx.Tensor, Out: fx.Tensor, n_elem: fx.Int32) -> None:
        idx = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        inb = idx < n_elem
        safe = inb.select(idx, fx.Int32(0))
        mask_t = ptr_buf_tensor(
            Mask,
            elem=fx.Uint8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_elem) * fx.Int64(1),
        )
        out_t = ptr_buf_tensor(
            Out,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_elem) * fx.Int64(4),
        )
        flag = fx.Uint8(buf_copy_load(mask_t, fx.Int64(safe), elem=fx.Uint8, unit_elems=1))
        is_true = flag != fx.Uint8(0)
        val = is_true.select(fx.Float32(0.0), fx.Float32(float("-inf")))
        if inb:
            buf_copy_store(out_t, fx.Int64(safe), val, elem=fx.Float32, unit_elems=1)

    @flyc.jit
    def launch(Mask: fx.Tensor, Out: fx.Tensor, n_elem: fx.Int32, stream: fx.Stream = fx.Stream(None)) -> None:
        grid = (fx.Int64(n_elem) + fx.Int64(block - 1)) // fx.Int64(block)
        bool_mask_kernel(Mask, Out, n_elem).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    launch.__name__ = f"bool_mask_to_bias_b{block}"
    return launch


def bool_mask_to_additive(
    mask: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Convert a bool mask to fp32 additive bias (True→0, False→-inf) on device."""
    require_gfx120x(mask.device, what="bool_mask_to_additive (gfx120x)")
    if mask.dtype != torch.bool:
        raise TypeError(f"bool_mask_to_additive expects bool, got {mask.dtype}")
    m = mask.detach().contiguous()
    out = torch.empty(m.shape, device=m.device, dtype=torch.float32)
    n = int(m.numel())
    if n == 0:
        return out
    launch = build_bool_mask_to_bias_module(256)
    flat_m = m.view(torch.uint8).reshape(-1)
    flat_o = out.reshape(-1)
    if stream is None:
        launch(flat_m, flat_o, n)
    else:
        launch(flat_m, flat_o, n, stream)
    return out


__all__ = ["bool_mask_to_additive", "build_bool_mask_to_bias_module"]
