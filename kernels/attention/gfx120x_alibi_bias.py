# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device ALiBi bias fill for gfx120x FlashAttention hosts.

Writes -slope * |i + Skv - Sq - j| into a contiguous fp32 buffer so the
attention kernel can consume it as additive bias (shared [Sq, Sk] or
per-head [H, Sq, Sk]). Replaces host torch.arange / abs folds.
"""

from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor
from kernels.common.gfx120x_pad import ensure_contiguous

_BLOCKS = (256,)


@lru_cache(maxsize=8)
def build_alibi_bias_module(block_threads: int = 256, per_head: bool = False):
    block = int(block_threads)
    if block not in _BLOCKS:
        raise ValueError(f"alibi bias block_threads={block} not in {_BLOCKS}")
    PER_HEAD = bool(per_head)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def alibi_kernel(
        Slopes: fx.Tensor,
        Out: fx.Tensor,
        seq_q: fx.Int32,
        seq_k: fx.Int32,
        num_heads: fx.Int32,
    ) -> None:
        idx = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        sk = seq_k
        sq = seq_q
        tile = sq * sk
        if const_expr(PER_HEAD):
            n_elem = num_heads * tile
        else:
            n_elem = tile
        inb = idx < n_elem
        safe = inb.select(idx, fx.Int32(0))
        if const_expr(PER_HEAD):
            h = safe // tile
            rem = safe - h * tile
        else:
            rem = safe
        i = rem // sk
        j = rem - i * sk
        dist_i = i + sk - sq - j
        dist = fx.absf(fx.Float32(dist_i))
        slopes = ptr_buf_tensor(
            Slopes,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(num_heads) * fx.Int64(4),
        )
        out_t = ptr_buf_tensor(
            Out,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_elem) * fx.Int64(4),
        )
        if const_expr(PER_HEAD):
            slope = fx.Float32(buf_copy_load(slopes, fx.Int64(h), elem=fx.Float32, unit_elems=1))
        else:
            slope = fx.Float32(buf_copy_load(slopes, fx.Int64(0), elem=fx.Float32, unit_elems=1))
        val = fx.Float32(0.0) - slope * dist
        if inb:
            buf_copy_store(out_t, fx.Int64(safe), val, elem=fx.Float32, unit_elems=1)

    @flyc.jit
    def launch(
        Slopes: fx.Tensor,
        Out: fx.Tensor,
        seq_q: fx.Int32,
        seq_k: fx.Int32,
        num_heads: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        if const_expr(PER_HEAD):
            n_elem = fx.Int64(num_heads) * fx.Int64(seq_q) * fx.Int64(seq_k)
        else:
            n_elem = fx.Int64(seq_q) * fx.Int64(seq_k)
        grid = (n_elem + fx.Int64(block - 1)) // fx.Int64(block)
        alibi_kernel(Slopes, Out, seq_q, seq_k, num_heads).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


def fill_alibi_bias(
    slopes: torch.Tensor,
    seq_len_q: int,
    seq_len_kv: int,
    *,
    per_head: bool,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Fill ALiBi bias on device. slopes is contiguous fp32 [H] (or [1])."""
    require_gfx120x(slopes.device, what="fill_alibi_bias (gfx120x)")
    s = ensure_contiguous(slopes.detach().to(dtype=torch.float32), stream=stream)
    if s.dim() != 1 or s.numel() < 1:
        raise ValueError(f"fill_alibi_bias: slopes must be 1D nonempty, got {tuple(s.shape)}")
    h = int(s.numel())
    sq, sk = int(seq_len_q), int(seq_len_kv)
    if sq < 0 or sk < 0:
        raise ValueError(f"fill_alibi_bias: seq lens must be >= 0, got q={sq} kv={sk}")
    if per_head:
        out = torch.empty((h, sq, sk), device=s.device, dtype=torch.float32)
    else:
        out = torch.empty((sq, sk), device=s.device, dtype=torch.float32)
        if h != 1:
            raise ValueError(f"fill_alibi_bias: uniform path needs 1 slope, got {h}")
    # Empty bias is a legal shape; skip the kernel when n_elem would be 0.
    if sq == 0 or sk == 0:
        return out
    launch = build_alibi_bias_module(256, per_head=per_head)
    if stream is None:
        launch(s, out, sq, sk, h)
    else:
        launch(s, out, sq, sk, h, stream)
    return out


__all__ = ["build_alibi_bias_module", "fill_alibi_bias"]
