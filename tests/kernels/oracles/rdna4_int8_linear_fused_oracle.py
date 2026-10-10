# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only torch oracles extracted from ``kernels/gemm/rdna4_int8_linear_fused.py``."""

from collections.abc import Sequence

import torch


def _lora_scale_as_float(scale: float | torch.Tensor) -> float:
    if isinstance(scale, torch.Tensor):
        return float(scale.detach().float().reshape(-1)[0].item())
    return float(scale)


def _pack_lora_adapters(
    lora_downs: torch.Tensor | Sequence[torch.Tensor] | None,
    lora_ups: torch.Tensor | Sequence[torch.Tensor] | None,
    lora_scales: float | torch.Tensor | Sequence[float | torch.Tensor] | None,
    *,
    k: int,
    n: int,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Normalize single/list LoRA args into ordered (down, up, scale) packs."""
    if lora_downs is None and lora_ups is None:
        if lora_scales is None:
            return []
        raise ValueError("lora_scales set without lora_downs/lora_ups")
    if lora_downs is None or lora_ups is None:
        raise ValueError("lora_downs and lora_ups must both be set or both None")

    if isinstance(lora_downs, torch.Tensor):
        downs: list[torch.Tensor] = [lora_downs]
    else:
        downs = list(lora_downs)
    if isinstance(lora_ups, torch.Tensor):
        ups: list[torch.Tensor] = [lora_ups]
    else:
        ups = list(lora_ups)
    if len(downs) != len(ups):
        raise ValueError(f"lora_downs/lora_ups length mismatch: {len(downs)} vs {len(ups)}")

    if lora_scales is None:
        scales_list: list[float | torch.Tensor] = [1.0] * len(downs)
    elif isinstance(lora_scales, (list, tuple)):
        scales_list = list(lora_scales)
    else:
        scales_list = [lora_scales] * len(downs)
    if len(scales_list) != len(downs):
        raise ValueError(f"lora_scales length {len(scales_list)} != adapters {len(downs)}")

    packs: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for i, (down, up, scale) in enumerate(zip(downs, ups, scales_list)):
        if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
            raise TypeError(f"adapter[{i}] down/up must be tensors")
        if down.dim() != 2 or up.dim() != 2:
            raise ValueError(f"adapter[{i}] down/up must be 2D")
        rank = int(down.shape[0])
        if tuple(down.shape) != (rank, k):
            raise ValueError(f"adapter[{i}] lora_down must be [{rank}, {k}]")
        if tuple(up.shape) != (n, rank):
            raise ValueError(f"adapter[{i}] lora_up must be [{n}, {rank}]")
        packs.append((down, up, _lora_scale_as_float(scale)))
    return packs


def reference_int8_linear_fused(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_down: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """Torch reference: separate int8 quant + int8_linear (+ optional LoRA)."""
    sa = scale_a.float().reshape(-1)
    sb = scale_b.float().reshape(-1)
    # per-row quant
    a_scaled = a_f.float() / sa.unsqueeze(1)
    a_q = a_scaled.round().clamp(-128, 127).to(torch.int8)
    acc = a_q.float() @ b_nk.float().T
    if sb.numel() == 1:
        out = acc.float() * sa.unsqueeze(1) * sb[0]
    else:
        out = acc.float() * sa.unsqueeze(1) * sb.unsqueeze(0)
    if lora_down is not None and lora_up is not None:
        # F.linear chain: hidden = a @ down.T; delta = hidden @ up.T
        hidden = a_f.float() @ lora_down.float().T
        out = out + float(lora_scale) * (hidden @ lora_up.float().T)
    return out.to(out_dtype)


def _reference_lora_residual(
    a_f: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    lora_scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Oracle LoRA residual (pure torch). Used only by ``reference_*`` helpers."""
    down = lora_down if lora_down.dtype == a_f.dtype else lora_down.to(dtype=a_f.dtype)
    up = lora_up if lora_up.dtype == a_f.dtype else lora_up.to(dtype=a_f.dtype)
    hidden = a_f @ down.t()
    delta = hidden @ up.t()
    if lora_scale != 1.0:
        delta = delta * float(lora_scale)
    return delta if delta.dtype == out_dtype else delta.to(dtype=out_dtype)


def reference_int8_linear_fused_multi(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_downs: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_ups: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_scales: float | torch.Tensor | Sequence[float | torch.Tensor] | None = 1.0,
) -> torch.Tensor:
    """Torch reference: base quant+mm + sequential LoRA residuals in load order."""
    m, k = a_f.shape
    n = b_nk.shape[0]
    packs = _pack_lora_adapters(lora_downs, lora_ups, lora_scales, k=k, n=n)
    out = reference_int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=out_dtype)
    for down, up, scale in packs:
        out.add_(_reference_lora_residual(a_f, down, up, scale, out_dtype))
    return out
