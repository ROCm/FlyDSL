# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch
import torch.nn.functional as F


# Scale-kind "enum" used by the ref helpers below.  Each kind determines
# how we dequantize (x, scale) to fp32 and how to recover the logical K
# dimension from x.shape[-1].
#   "mxfp4":  packed fp4 bytes (1 byte = 2 fp4 values);  logical_K == shape[-1] * 2
#             scale is uint8 E8M0 block scale, one byte per 32 logical elements.
#   "mxfp8":  raw fp8_e4m3fn bytes;                     logical_K == shape[-1]
#             scale is uint8 E8M0 block scale, one byte per 32 logical elements.
#   "scalar": plain fp/int tensor + per-token / per-row fp scale (may be None);
#             logical_K == shape[-1].  'scale is None' is the no-quant case.
_FP4_DTYPES = (torch.uint8, torch.float4_e2m1fn_x2)


def _detect_scale_kind(x: torch.Tensor, scale: torch.Tensor | None) -> str:
    """Classify the (x, scale) pair into one of {"mxfp4", "mxfp8", "scalar"}.

    The classifier is intentionally dtype-driven so callers can pass *either*
    packed FP4 bytes or real torch.float4 tensors for MX-FP4, while still
    distinguishing MX-FP8 (one fp8 byte per element) unambiguously.
    """
    if scale is None or scale.dtype != torch.uint8:
        return "scalar"
    # Both block-scale kinds use K_blocks == logical_K // 32.
    if x.dtype in _FP4_DTYPES and x.shape[-1] * 2 == scale.shape[-1] * 32:
        return "mxfp4"
    if x.dtype == torch.float8_e4m3fn and x.shape[-1] == scale.shape[-1] * 32:
        return "mxfp8"
    return "scalar"


def _logical_k(x: torch.Tensor, kind: str) -> int:
    """Return the logical K dim (unpacked) given the detected scale kind."""
    k = int(x.shape[-1])
    return k * 2 if kind == "mxfp4" else k


def _dequant_mxfp4_per_1x32(x_fp4: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize packed MXFP4 with per-1x32 e8m0 block scales to fp32."""
    from tests.kernels.utils import fp4_utils

    x_u8 = x_fp4.view(torch.uint8)
    logical_k = int(x_u8.shape[-1]) * 2
    if logical_k % 32 != 0:
        raise ValueError(f"FP4 logical K must be divisible by 32, got {logical_k}")

    x_f32 = fp4_utils.mxfp4_to_f32(x_u8.reshape(-1, logical_k // 2))
    scales_f32 = fp4_utils.e8m0_to_f32(scale_e8m0.view(torch.uint8).reshape(-1, logical_k // 32))
    x_f32 = x_f32 * scales_f32.repeat_interleave(32, dim=1)
    return x_f32.view(*x_u8.shape[:-1], logical_k)


def _dequant_mxfp8_per_1x32(x_fp8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize MX-FP8 (e4m3fn) activations with per-1x32 e8m0 block scales to fp32.

    Mirrors the MFMA-time semantics of the A8W4 stage1/stage2 kernels: the kernel
    reads the raw fp8 byte, casts to fp32, and multiplies by the 8-bit exponent
    scale (E8M0 = 2^(byte-127)) for every 32-element K block.
    """
    from tests.kernels.utils import fp4_utils

    logical_k = int(x_fp8.shape[-1])
    if logical_k % 32 != 0:
        raise ValueError(f"MX-FP8 logical K must be divisible by 32, got {logical_k}")

    x_f32 = x_fp8.reshape(-1, logical_k).to(torch.float32)
    scales_f32 = fp4_utils.e8m0_to_f32(
        scale_e8m0.view(torch.uint8).reshape(-1, logical_k // 32)
    )
    x_f32 = x_f32 * scales_f32.repeat_interleave(32, dim=1)
    return x_f32.view(*x_fp8.shape[:-1], logical_k)


def _dequant(x: torch.Tensor, scale: torch.Tensor | None, kind: str) -> torch.Tensor:
    """Unified fp32 dequantization driven by the detected ``kind``.

    - "mxfp4" / "mxfp8" delegate to the per-1x32 block-scale helpers above.
    - "scalar" handles both the no-quant (scale is None) and broadcast-scale
      paths (per-token / per-row fp scale).
    """
    if kind == "mxfp4":
        return _dequant_mxfp4_per_1x32(x, scale)
    if kind == "mxfp8":
        return _dequant_mxfp8_per_1x32(x, scale)
    x_f32 = x.to(torch.float32)
    return x_f32 if scale is None else x_f32 * scale


def torch_moe_gemm1(
    x_q: torch.Tensor,
    w1_q_flat: torch.Tensor,
    scale_x: torch.Tensor | None,
    scale_w1_flat: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inter_dim: int,
    doweight_stage1: bool,
    group_size: int = -1,
    scale_w1_groups: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [tokens, topk, inter_dim] fp32.

    Args:
        group_size: -1 for per-row scale (uses scale_w1_flat), >0 for group-wise scale.
        scale_w1_groups: Group-wise scale tensor of shape [E, K//group_size, 2*inter_dim] (Opt 0 layout).
                         Required when group_size > 0; ignored otherwise.
    """
    topk = topk_ids.shape[1]
    # Independent per-1x32 block-scale detection for x and w, so that mixed
    # precisions such as A8W4 (fp8 activation + mxfp4 weight) can use the correct
    # dequant per side.  See ``_detect_scale_kind`` for the classification rules.
    x_kind = _detect_scale_kind(x_q, scale_x)
    w_kind = _detect_scale_kind(w1_q_flat, scale_w1_flat)

    if x_q.dim() == 2:
        tokens = int(x_q.shape[0])
    elif x_q.dim() == 3:
        tokens, topk_x, _ = x_q.shape
        assert int(topk_x) == int(topk), \
            f"x_q topk mismatch: x_q.shape={tuple(x_q.shape)}, topk={topk}"
    else:
        raise ValueError(f"Unsupported x_q shape: {tuple(x_q.shape)}")
    model_dim = _logical_k(x_q, x_kind)
    # Derive experts from weight shapes (topk_ids may not cover all experts when tokens are tiny).
    if w1_q_flat.dim() == 2:
        experts = int(w1_q_flat.shape[0] // (2 * inter_dim))
    else:
        experts = int(w1_q_flat.shape[0])

    x = _dequant(x_q, scale_x, x_kind)

    if group_size > 0 and scale_w1_groups is not None:
        # Group-wise dequantization: w_dequant[e,n,k] = w_int[e,n,k] * scale[e, k//group_size, n]
        # Scale layout: [E, num_groups, N] (Opt 0: cache-friendly)
        w1 = w1_q_flat.to(torch.float32).view(experts, 2 * inter_dim, model_dim)
        num_groups = model_dim // group_size
        for g in range(num_groups):
            k_s, k_e = g * group_size, (g + 1) * group_size
            w1[:, :, k_s:k_e] *= scale_w1_groups[:, g, :].unsqueeze(-1)
    else:
        w1 = _dequant(w1_q_flat, scale_w1_flat, w_kind).view(experts, 2 * inter_dim, model_dim)

    out = torch.zeros((tokens, topk, inter_dim), device="cuda", dtype=torch.float32)
    for e in range(experts):
        # routes assigned to expert e
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)  # [num, 2] (t, slot)
        if idx.numel() == 0:
            continue
        t_idx = idx[:, 0]
        s_idx = idx[:, 1]
        x_in = x[t_idx, :] if x.dim() == 2 else x[t_idx, s_idx, :]
        y2 = F.linear(x_in, w1[e, :, :])  # [num, 2*inter_dim]
        gate = y2[:, :inter_dim]
        up = y2[:, inter_dim:]
        y = F.silu(gate) * up
        if doweight_stage1:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out[t_idx, s_idx, :] = y
    return out


def torch_moe_gemm2(
    a2_q: torch.Tensor,
    w2_q: torch.Tensor,
    scale_a2: torch.Tensor | None,
    scale_w2: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    model_dim: int,
    doweight_stage2: bool,
    group_size: int = -1,
    scale_w2_groups: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [tokens, model_dim] fp32.

    Semantics align with aiter `torch_moe_stage2`:
    - Dequantize `a2_q` and `w2_q` with per-token/row scales.
    - For each routed (token, slot) -> expert, compute y = a2 @ W2[expert]^T.
    - Optionally multiply routed weight in stage2 (when stage1 did *not*).
    - Reduce across topk by summing into [tokens, model_dim].

    Args:
        group_size: -1 for per-row scale (uses scale_w2), >0 for group-wise scale.
        scale_w2_groups: Group-wise scale tensor of shape [E, inter_dim//group_size, model_dim] (Opt 0 layout).
                         Required when group_size > 0; ignored otherwise.
    """
    assert a2_q.is_cuda and w2_q.is_cuda
    tokens, topk = topk_ids.shape

    # Independent per-1x32 block-scale detection for a2 and w2; see
    # ``_detect_scale_kind`` for the classification rules.
    a_kind = _detect_scale_kind(a2_q, scale_a2)
    w_kind = _detect_scale_kind(w2_q, scale_w2)

    inter_dim = _logical_k(a2_q, a_kind)
    if a_kind in ("mxfp4", "mxfp8"):
        if a2_q.dim() == 2:
            a2_q = a2_q.view(tokens, topk, -1)
            scale_a2 = scale_a2.view(tokens, topk, -1)
        elif a2_q.dim() != 3:
            raise ValueError(f"Unsupported {a_kind} a2 shape: {tuple(a2_q.shape)}")
    else:
        if a2_q.dim() != 3:
            raise ValueError(f"Unsupported a2_q shape: {tuple(a2_q.shape)}")

    # Derive experts from weight shapes (topk_ids may not cover all experts when tokens are tiny).
    if w2_q.dim() == 3:
        experts = int(w2_q.shape[0])
    else:
        experts = int(w2_q.shape[0] // model_dim)

    a2 = _dequant(a2_q, scale_a2, a_kind)

    if group_size > 0 and scale_w2_groups is not None:
        # Group-wise dequantization: w_dequant[e,n,k] = w_int[e,n,k] * scale[e, k//group_size, n]
        # Scale layout: [E, num_groups, N] (Opt 0: cache-friendly)
        w2 = w2_q.to(torch.float32).view(experts, model_dim, inter_dim)
        num_groups = inter_dim // group_size
        for g in range(num_groups):
            k_s, k_e = g * group_size, (g + 1) * group_size
            w2[:, :, k_s:k_e] *= scale_w2_groups[:, g, :].unsqueeze(-1)
    else:
        w2 = _dequant(w2_q, scale_w2, w_kind).view(experts, model_dim, inter_dim)

    out = torch.zeros((tokens, model_dim), device="cuda", dtype=torch.float32)
    for e in range(experts):
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)  # [num, 2] (t, slot)
        if idx.numel() == 0:
            continue
        t_idx = idx[:, 0]
        s_idx = idx[:, 1]
        y = F.linear(a2[t_idx, s_idx, :], w2[e, :, :])  # [num, model_dim]
        if doweight_stage2:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out.index_add_(0, t_idx, y)
    return out
