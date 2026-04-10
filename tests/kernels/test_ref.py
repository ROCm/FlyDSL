# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch
import torch.nn.functional as F


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
    is_fp4 = (
        scale_x is not None
        and scale_w1_flat is not None
        and x_q.shape[-1] * 2 == scale_x.shape[-1] * 32
        and w1_q_flat.shape[-1] * 2 == scale_w1_flat.shape[-1] * 32
    )

    if x_q.dim() == 2:
        tokens = int(x_q.shape[0])
        model_dim = int(x_q.shape[1]) * 2 if is_fp4 else int(x_q.shape[1])
    elif x_q.dim() == 3:
        tokens, topk_x, model_dim_raw = x_q.shape
        assert (
            int(topk_x) == int(topk)
        ), f"x_q topk mismatch: x_q.shape={tuple(x_q.shape)}, topk={topk}"
        model_dim = int(model_dim_raw) * 2 if is_fp4 else int(model_dim_raw)
    else:
        raise ValueError(f"Unsupported x_q shape: {tuple(x_q.shape)}")
    # Derive experts from weight shapes (topk_ids may not cover all experts when tokens are tiny).
    if w1_q_flat.dim() == 2:
        experts = int(w1_q_flat.shape[0] // (2 * inter_dim))
    else:
        experts = int(w1_q_flat.shape[0])

    if is_fp4:
        x = _dequant_mxfp4_per_1x32(x_q, scale_x)
    else:
        x = x_q.to(torch.float32) if scale_x is None else (x_q.to(torch.float32) * scale_x)

    if group_size > 0 and scale_w1_groups is not None:
        # Group-wise dequantization: w_dequant[e,n,k] = w_int[e,n,k] * scale[e, k//group_size, n]
        # Scale layout: [E, num_groups, N] (Opt 0: cache-friendly)
        w1 = w1_q_flat.to(torch.float32).view(experts, 2 * inter_dim, model_dim)
        num_groups = model_dim // group_size
        for g in range(num_groups):
            k_s, k_e = g * group_size, (g + 1) * group_size
            w1[:, :, k_s:k_e] *= scale_w1_groups[:, g, :].unsqueeze(-1)
    else:
        # Per-row dequantization.
        if is_fp4:
            w1 = _dequant_mxfp4_per_1x32(w1_q_flat, scale_w1_flat).view(experts, 2 * inter_dim, model_dim)
        else:
            w1 = (
                w1_q_flat.to(torch.float32)
                if scale_w1_flat is None
                else (w1_q_flat.to(torch.float32) * scale_w1_flat)
            )
            w1 = w1.view(experts, 2 * inter_dim, model_dim)

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
        if doweight_stage1:
            y2 = y2 * topk_weights[t_idx, s_idx].unsqueeze(-1)
        gate = y2[:, :inter_dim]
        up = y2[:, inter_dim:]
        y = F.silu(gate) * up
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

    is_fp4 = (
        scale_a2 is not None
        and scale_w2 is not None
        and a2_q.shape[-1] * 2 == scale_a2.shape[-1] * 32
        and w2_q.shape[-1] * 2 == scale_w2.shape[-1] * 32
    )

    if is_fp4:
        inter_dim = int(a2_q.shape[-1]) * 2
        if a2_q.dim() == 2:
            a2_q = a2_q.view(tokens, topk, -1)
            scale_a2 = scale_a2.view(tokens, topk, -1)
        elif a2_q.dim() != 3:
            raise ValueError(f"Unsupported FP4 a2 shape: {tuple(a2_q.shape)}")
    else:
        if a2_q.dim() != 3:
            raise ValueError(f"Unsupported a2_q shape: {tuple(a2_q.shape)}")
        _, _, inter_dim = a2_q.shape

    # Derive experts from weight shapes (topk_ids may not cover all experts when tokens are tiny).
    if w2_q.dim() == 3:
        experts = int(w2_q.shape[0])
    else:
        experts = int(w2_q.shape[0] // model_dim)

    # Dequantize inputs.
    if is_fp4:
        a2 = _dequant_mxfp4_per_1x32(a2_q, scale_a2)
    else:
        a2 = a2_q.to(torch.float32) if scale_a2 is None else (a2_q.to(torch.float32) * scale_a2)

    if group_size > 0 and scale_w2_groups is not None:
        # Group-wise dequantization: w_dequant[e,n,k] = w_int[e,n,k] * scale[e, k//group_size, n]
        # Scale layout: [E, num_groups, N] (Opt 0: cache-friendly)
        w2 = w2_q.to(torch.float32).view(experts, model_dim, inter_dim)
        num_groups = inter_dim // group_size
        for g in range(num_groups):
            k_s, k_e = g * group_size, (g + 1) * group_size
            w2[:, :, k_s:k_e] *= scale_w2_groups[:, g, :].unsqueeze(-1)
    else:
        # Per-row dequantization.
        if is_fp4:
            w2 = _dequant_mxfp4_per_1x32(w2_q, scale_w2).view(experts, model_dim, inter_dim)
        else:
            w2 = w2_q.to(torch.float32) if scale_w2 is None else (w2_q.to(torch.float32) * scale_w2)
            w2 = w2.view(experts, model_dim, inter_dim)

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
