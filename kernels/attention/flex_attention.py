# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL flex_attention: score_mod / mask_mod on the generic flash-attention kernel.

Ports the distinguishing features of PyTorch's ``torch.nn.attention.flex_attention``
onto the existing dense f16/bf16 forward kernel in ``flash_attn_generic``:

- ``score_mod(score, b, h, q_idx, kv_idx) -> score`` transforms each logit.
- ``mask_mod(b, h, q_idx, kv_idx) -> bool`` keeps (True) or drops (False) a position.

Both are compile-time Python callables over ``fx`` scalars (coords are ``fx.Int32``,
``score`` is ``fx.Float32``), captured into the build closure and specialized per
kernel. This mirrors PyTorch's codegen inlining: one compile per distinct mod.

Semantics match PyTorch: ``score_mod`` sees ``qk * sm_scale`` (scale is applied
before the mod), then ``mask_mod`` applies ``where(mask, score, -inf)``. Phase 1 is
dense (all KV blocks visited); block sparsity is a later phase.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import torch

import flydsl.expr as fx

__all__ = [
    "flydsl_flex_attention",
    "alibi_score_mod",
    "sliding_window_mask_mod",
]

_DTYPE_MAP = {torch.bfloat16: "bf16", torch.float16: "f16"}


# ── built-in mods ───────────────────────────────────────────────────────────


def alibi_score_mod(slope: float) -> Callable:
    """ALiBi positional bias: ``score + slope * (kv_idx - q_idx)``.

    ``slope`` is a per-kernel scalar; the returned callable is a score_mod.
    """

    def _mod(score, b, h, q_idx, kv_idx):
        bias = (kv_idx - q_idx).to(fx.Float32) * fx.Float32(slope)
        return fx.Float32(score) + bias

    _mod.__qualname__ = f"alibi_score_mod[slope={slope}]"
    return _mod


def sliding_window_mask_mod(window: int) -> Callable:
    """Sliding-window causal mask: keep ``kv_idx <= q_idx`` and ``q_idx - kv_idx <= window``."""

    def _mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        in_window = (q_idx - kv_idx) <= fx.Int32(window)
        return causal & in_window

    _mod.__qualname__ = f"sliding_window_mask_mod[window={window}]"
    return _mod


def causal_mask_mod(b, h, q_idx, kv_idx):
    """Plain causal mask expressed as a mask_mod: keep ``kv_idx <= q_idx``."""
    return kv_idx <= q_idx


# ── build cache ─────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def _build_flex(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_str: str,
    cross_seqlen: bool,
    block_m: int,
    flat_work_group_size: int,
    path_tag: str,
    waves_per_eu: int,
    daz: bool,
    return_lse: bool,
    sm_scale: Optional[float],
    score_mod: Optional[Callable],
    mask_mod: Optional[Callable],
):
    """Build (and cache) one flex dense launcher variant.

    Keyed by callable identity for the mods (module-level / factory-produced mods
    have stable identity), mirroring ``flash_attn_interface._build_dense``.
    """
    from kernels.attention.flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=False,  # mask_mod is the sole masking authority (avoids double-masking)
        dtype_str=dtype_str,
        sm_scale=sm_scale,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        block_m=block_m,
        flat_work_group_size=flat_work_group_size,
        path_tag=path_tag,
        waves_per_eu=waves_per_eu,
        daz=daz,
        return_lse=return_lse,
        score_mod=score_mod,
        mask_mod=mask_mod,
    )


def _dtype_str(t: torch.Tensor) -> str:
    s = _DTYPE_MAP.get(t.dtype)
    if s is None:
        raise ValueError(f"flydsl_flex_attention only supports bf16/f16, got {t.dtype!r}")
    return s


# ── public API ──────────────────────────────────────────────────────────────


def flydsl_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run FlyDSL flex_attention (dense f16/bf16 forward, gfx942/gfx950).

    Args:
        q: Query tensor ``[B, Sq, H, D]`` (BSHD).
        k: Key tensor ``[B, Skv, Hkv, D]``.
        v: Value tensor, same shape as k.
        score_mod: Optional ``(score, b, h, q_idx, kv_idx) -> score`` over ``fx``
            scalars. Receives ``qk * scale`` (PyTorch semantics).
        mask_mod: Optional ``(b, h, q_idx, kv_idx) -> bool`` over ``fx`` scalars;
            False positions are set to -inf.
        scale: Softmax scale; defaults to ``1/sqrt(D)``.
        num_kv_heads: KV head count for GQA/MQA; defaults to q num_heads (MHA).
        return_lse: Also return fp32 ``[B, H, Sq]`` log-sum-exp.
        out: Optional pre-allocated output; allocated if None.
        waves_per_eu: Occupancy hint.
        daz: Enable denormals-are-zero.
        stream: CUDA/HIP stream; defaults to the current stream for q.device.

    Returns:
        Output tensor with q's shape/dtype, or ``(out, lse)`` when ``return_lse``.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flex_attention: q/k/v must be CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"flydsl_flex_attention: q/k/v must share device; got {q.device}/{k.device}/{v.device}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"flydsl_flex_attention: q/k/v must share dtype; got {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dim() != 4:
        raise ValueError(f"flydsl_flex_attention: q must be 4D [B,Sq,H,D], got {q.dim()}D")

    dtype_str = _dtype_str(q)
    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    Hkv = k.shape[2]
    cross = Sq != Skv

    if num_kv_heads is None:
        num_kv_heads = Hkv
    if H % num_kv_heads != 0:
        raise ValueError(f"flydsl_flex_attention: num_heads ({H}) must be divisible by num_kv_heads ({num_kv_heads})")
    if D < 64 or D % 32 != 0:
        raise ValueError(f"flydsl_flex_attention: head_dim ({D}) must be >= 64 and a multiple of 32")

    from kernels.attention.flash_attn_interface import _dense_generic_tile

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream

        block_m, flat_work_group_size, path_tag = _dense_generic_tile(B, Sq, H, D, dtype_str, q.device)
        exe = _build_flex(
            num_heads=H,
            num_kv_heads=num_kv_heads,
            head_dim=D,
            dtype_str=dtype_str,
            cross_seqlen=cross,
            block_m=block_m,
            flat_work_group_size=flat_work_group_size,
            path_tag=path_tag,
            waves_per_eu=waves_per_eu,
            daz=daz,
            return_lse=return_lse,
            sm_scale=scale,
            score_mod=score_mod,
            mask_mod=mask_mod,
        )

        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        elif out.dtype != q.dtype:
            raise ValueError(f"flydsl_flex_attention: output dtype must match q dtype {q.dtype}, got {out.dtype}")

        q_flat = q.contiguous()
        k_flat = k.contiguous()
        v_flat = v.contiguous()
        o_flat = out.contiguous()

        lse = torch.empty((B, H, Sq), dtype=torch.float32, device=q.device) if return_lse else None

        kwargs: dict = dict(stream=launch_stream, lse=lse)
        if cross:
            kwargs["seq_len_kv"] = Skv
        exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)

    if return_lse:
        return out, lse
    return out
