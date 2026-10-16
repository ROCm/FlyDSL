# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Pure-torch oracles for gfx120x kernel tests.

These functions are the numeric references. They are not product hosts.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from kernels.quant.rdna4_int4_codec import (
    dequant_int4_groupwise_signed,
    dequant_uint4_groupwise_awq,
    unpack_int4_row_major,
    unpack_uint4_row_major,
)

_F8_E4M3_MAX = 448.0
_F8_E5M2_MAX = 57344.0
_FP8_TORCH = (torch.float8_e4m3fn, torch.float8_e5m2)
_INT4_GROUP_SIZE = 64
_INT4_MAX = 7
_SCALE_FLOOR = 1e-10
_DEFAULT_GROUP = 64


def _fp8_max_for(dtype: torch.dtype) -> float:
    if dtype == torch.float8_e5m2:
        return _F8_E5M2_MAX
    return _F8_E4M3_MAX


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


def _reference_lora_residual(
    a_f: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    lora_scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Oracle LoRA residual. Used only by the fused reference helpers."""
    down = lora_down if lora_down.dtype == a_f.dtype else lora_down.to(dtype=a_f.dtype)
    up = lora_up if lora_up.dtype == a_f.dtype else lora_up.to(dtype=a_f.dtype)
    delta = (a_f @ down.T) @ up.T
    if lora_scale != 1.0:
        delta = delta * float(lora_scale)
    return delta if delta.dtype == out_dtype else delta.to(dtype=out_dtype)


def _build_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size**0.5)


def _pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    """Wire format: low nibble = even column, high = odd."""
    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def bottom_right_causal_bias(
    seq_len_q: int,
    seq_len_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Bottom-right causal additive bias ``[Sq, Skv]`` (0 / -inf).

    Query ``i`` attends to keys ``j <= i + Skv - Sq``.
    """
    q_idx = torch.arange(seq_len_q, device=device, dtype=torch.int32)[:, None]
    k_idx = torch.arange(seq_len_kv, device=device, dtype=torch.int32)[None, :]
    allow = k_idx <= (q_idx + (seq_len_kv - seq_len_q))
    bias = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.float32, device=device)
    return bias.masked_fill(~allow, float("-inf"))


def reference_swiglu_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Eager reference: two GEMMs + SiLU x mul + down, in f32."""
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).float()
    gate = x2d @ w_gate.float().T
    up = x2d @ w_up.float().T
    mid = F.silu(gate) * up
    y = mid @ w_down.float().T
    return y.reshape(*orig[:-1], w_down.shape[0])


def reference_dequant_awq_w4a16(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor | None,
    group_size: int = _DEFAULT_GROUP,
) -> torch.Tensor:
    """Torch reference. The arithmetic is ``dequant_uint4_groupwise_awq``."""
    return dequant_uint4_groupwise_awq(qweight, wscales, wzeros, group_size)


def reference_dequant_int4_grouped_to_int8(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    group_size: int = 16,
) -> torch.Tensor:
    """Torch reference for packed INT4 to an INT8 grid."""
    n, k_half = qdata.shape
    k = k_half * 2
    groups = k // group_size
    packed = qdata.to(torch.int32) & 0xFF
    quantized = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
    quantized[:, 0::2] = packed & 0xF
    quantized[:, 1::2] = (packed >> 4) & 0xF
    if codebook is not None:
        values = codebook.to(device=qdata.device, dtype=torch.float32)[quantized]
    else:
        values = quantized.float() - 8.0
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    return values.view(n, k).round().clamp(-127, 127).to(torch.int8)


def reference_quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for ConvRot W4A4 weight quant."""
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"quant_group_size must be {_INT4_GROUP_SIZE}")
    n, k = weight.shape
    w2d = weight.reshape(-1, k).to(torch.float32)
    h = _build_hadamard(convrot_groupsize, w2d.device, torch.float32)
    n_groups = k // convrot_groupsize
    wg = w2d.reshape(-1, n_groups, convrot_groupsize)
    rotated = torch.matmul(wg, h.T).reshape(-1, k)
    abs_max = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=_SCALE_FLOOR)
    scale = abs_max / float(_INT4_MAX)
    q = torch.round(rotated / scale).clamp(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return _pack_int4_row_major(q).reshape(n, k // 2), scale.reshape(n).to(torch.float32)


def reference_quantize_int8_convrot_weight(
    weight: torch.Tensor, group_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for ConvRot plus rowwise INT8."""
    orig_shape = tuple(weight.shape)
    k = int(orig_shape[-1])
    w2d = weight.reshape(-1, k).to(torch.float32)
    pad = (group_size - (k % group_size)) % group_size
    if pad:
        w2d = torch.nn.functional.pad(w2d, (0, pad))
    k_pad = int(w2d.shape[-1])
    h = _build_hadamard(group_size, w2d.device, torch.float32)
    n_groups = k_pad // group_size
    wg = w2d.reshape(-1, n_groups, group_size)
    rotated = torch.matmul(wg, h.T).reshape(-1, k_pad)
    abs_max = rotated.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 127.0).clamp(min=1e-30)
    q = torch.round(rotated / scale).clamp(-128, 127).to(torch.int8)
    q = q[..., :k].contiguous()
    return q.reshape(orig_shape), scale.reshape(*orig_shape[:-1], 1).to(torch.float32)


def reference_dequantize_int8_convrot_weight(
    q: torch.Tensor,
    scale: torch.Tensor | float | None,
    group_size: int = 256,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Torch oracle for the int8 ConvRot dequant kernel."""
    if out_dtype is None:
        out_dtype = torch.bfloat16
    orig = tuple(q.shape)
    k = int(orig[-1])
    q2 = q.reshape(-1, k).float()
    pad = (group_size - (k % group_size)) % group_size
    if pad:
        q2 = torch.nn.functional.pad(q2, (0, pad))
    k_pad = int(q2.shape[-1])
    if isinstance(scale, (int, float)):
        sc = torch.full((q2.shape[0], 1), float(scale), device=q.device, dtype=torch.float32)
    else:
        sc = scale.to(device=q.device, dtype=torch.float32).reshape(-1, 1)
    deq = q2 * sc
    h = _build_hadamard(group_size, q.device, torch.float32)
    n_groups = k_pad // group_size
    dg = deq.reshape(-1, n_groups, group_size)
    out = torch.matmul(dg, h.T).reshape(-1, k_pad)[..., :k].reshape(orig)
    return out.to(out_dtype)


def reference_dequant_svdquant_w4a4_weight(
    qweight: torch.Tensor, wscales: torch.Tensor, group_size: int = _INT4_GROUP_SIZE
) -> torch.Tensor:
    """Torch reference: signed INT4 times group scales."""
    return dequant_int4_groupwise_signed(qweight, wscales, group_size=group_size)


def reference_scaled_mm_svdquant_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
    group_size: int = _INT4_GROUP_SIZE,
) -> torch.Tensor:
    """Pure-torch SVDQuant W4A4 GEMM plus the LoRA-up term."""
    m, k_half = act.shape
    k = k_half * 2
    compute_dtype = wscales.dtype
    wgt_fp = dequant_int4_groupwise_signed(wgt, wscales, group_size=group_size)
    unpack_act = unpack_uint4_row_major if act_unsigned else unpack_int4_row_major
    act_int = unpack_act(act).to(compute_dtype)
    if k % group_size == 0:
        act_int = act_int.view(m, k // group_size, group_size)
        ascales_mng = ascales.t().unsqueeze(-1)
        act_fp = (act_int * ascales_mng).view(m, k)
    else:
        g_idx = torch.arange(k, device=act.device) // group_size
        act_fp = act_int * ascales.t()[:, g_idx]
    out = act_fp @ wgt_fp.t()
    lora_contribution = lora_act_in.float() @ lora_up.float().t()
    out = out + lora_contribution.to(out.dtype)
    if bias is not None:
        out = out + bias
    return out


def reference_scaled_mm_fp8_fused(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_down: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """Torch reference: separate quant plus matmul, and an optional LoRA."""
    sa = float(scale_a.reshape(-1)[0].item())
    sb = float(scale_b.reshape(-1)[0].item())
    fp8_dtype = b_nk.dtype if b_nk.dtype in _FP8_TORCH else torch.float8_e4m3fn
    fp8_max = _fp8_max_for(fp8_dtype)
    a_q = (a_f.float() / sa).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    out = (a_q.float() @ b_nk.float().T) * sa * sb
    if lora_down is not None and lora_up is not None:
        hidden = a_f.float() @ lora_down.float().T
        out = out + float(lora_scale) * (hidden @ lora_up.float().T)
    return out.to(out_dtype)


def reference_scaled_mm_fp8_fused_multi(
    a_f: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    lora_downs: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_ups: torch.Tensor | Sequence[torch.Tensor] | None = None,
    lora_scales: float | torch.Tensor | Sequence[float | torch.Tensor] | None = 1.0,
) -> torch.Tensor:
    """Torch reference: base quant plus matmul, then LoRA residuals in order."""
    _m, k = a_f.shape
    n = b_nk.shape[0]
    packs = _pack_lora_adapters(lora_downs, lora_ups, lora_scales, k=k, n=n)
    out = reference_scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=out_dtype)
    for down, up, scale in packs:
        out = out + _reference_lora_residual(a_f, down, up, scale, out_dtype)
    return out


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
    """Torch reference: int8 quant plus matmul, and an optional LoRA."""
    sa = scale_a.float().reshape(-1)
    sb = scale_b.float().reshape(-1)
    a_scaled = a_f.float() / sa.unsqueeze(1)
    a_q = a_scaled.round().clamp(-128, 127).to(torch.int8)
    acc = a_q.float() @ b_nk.float().T
    if sb.numel() == 1:
        out = acc.float() * sa.unsqueeze(1) * sb[0]
    else:
        out = acc.float() * sa.unsqueeze(1) * sb.unsqueeze(0)
    if lora_down is not None and lora_up is not None:
        hidden = a_f.float() @ lora_down.float().T
        out = out + float(lora_scale) * (hidden @ lora_up.float().T)
    return out.to(out_dtype)


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
    """Torch reference: base quant plus matmul, then LoRA residuals in order."""
    _m, k = a_f.shape
    n = b_nk.shape[0]
    packs = _pack_lora_adapters(lora_downs, lora_ups, lora_scales, k=k, n=n)
    out = reference_int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=out_dtype)
    for down, up, scale in packs:
        out.add_(_reference_lora_residual(a_f, down, up, scale, out_dtype))
    return out
