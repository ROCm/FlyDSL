import torch
import torch.nn.functional as F

def torch_moe_gemm1(
    x_fp8: torch.Tensor,
    w1_fp8_flat: torch.Tensor,
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
        scale_w1_groups: Group-wise scale tensor of shape [E, 2*inter_dim, K//group_size].
                         Required when group_size > 0; ignored otherwise.
    """
    tokens, model_dim = x_fp8.shape
    topk = topk_ids.shape[1]
    # Derive experts from weight shapes (topk_ids may not cover all experts when tokens are tiny).
    if w1_fp8_flat.dim() == 2:
        experts = int(w1_fp8_flat.shape[0] // (2 * inter_dim))
    else:
        experts = int(w1_fp8_flat.shape[0])

    x = x_fp8.to(torch.float32) if scale_x is None else (x_fp8.to(torch.float32) * scale_x)

    if group_size > 0 and scale_w1_groups is not None:
        # Group-wise dequantization: w_dequant[e,n,k] = w_int[e,n,k] * scale[e,n,k//group_size]
        w1 = w1_fp8_flat.to(torch.float32).view(experts, 2 * inter_dim, model_dim)
        num_groups = model_dim // group_size
        for g in range(num_groups):
            k_s, k_e = g * group_size, (g + 1) * group_size
            w1[:, :, k_s:k_e] *= scale_w1_groups[:, :, g:g + 1]
    else:
        # Per-row dequantization (original path).
        w1 = (
            w1_fp8_flat.to(torch.float32)
            if scale_w1_flat is None
            else (w1_fp8_flat.to(torch.float32) * scale_w1_flat)
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
        y2 = F.linear(x[t_idx, :], w1[e, :, :])  # [num, 2*inter_dim]
        gate = y2[:, :inter_dim]
        up = y2[:, inter_dim:]
        y = F.silu(gate) * up
        if doweight_stage1:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out[t_idx, s_idx, :] = y
    return out


def torch_moe_gemm2(
    a2_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
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
    - Dequantize `a2_fp8` and `w2_fp8` with per-token/row scales.
    - For each routed (token, slot) -> expert, compute y = a2 @ W2[expert]^T.
    - Optionally multiply routed weight in stage2 (when stage1 did *not*).
    - Reduce across topk by summing into [tokens, model_dim].

    Args:
        group_size: -1 for per-row scale (uses scale_w2), >0 for group-wise scale.
        scale_w2_groups: Group-wise scale tensor of shape [E, model_dim, inter_dim//group_size].
                         Required when group_size > 0; ignored otherwise.
    """
    assert a2_fp8.is_cuda and w2_fp8.is_cuda
    tokens, topk, inter_dim = a2_fp8.shape
    # Derive experts from weight shapes (topk_ids may not cover all experts when tokens are tiny).
    if w2_fp8.dim() == 3:
        experts = int(w2_fp8.shape[0])
    else:
        experts = int(w2_fp8.shape[0] // model_dim)

    # Dequantize inputs.
    a2 = a2_fp8.to(torch.float32) if scale_a2 is None else (a2_fp8.to(torch.float32) * scale_a2)

    if group_size > 0 and scale_w2_groups is not None:
        # Group-wise dequantization: w_dequant[e,n,k] = w_int[e,n,k] * scale[e,n,k//group_size]
        w2 = w2_fp8.to(torch.float32).view(experts, model_dim, inter_dim)
        num_groups = inter_dim // group_size
        for g in range(num_groups):
            k_s, k_e = g * group_size, (g + 1) * group_size
            w2[:, :, k_s:k_e] *= scale_w2_groups[:, :, g:g + 1]
    else:
        # Per-row dequantization (original path).
        w2 = w2_fp8.to(torch.float32) if scale_w2 is None else (w2_fp8.to(torch.float32) * scale_w2)
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
