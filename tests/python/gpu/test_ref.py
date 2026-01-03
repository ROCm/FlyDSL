import torch
import torch.nn.functional as F

def torch_moe_gemm1(
    x_fp8: torch.Tensor,
    w1_fp8_flat: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w1_flat: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inter_dim: int,
    doweight_stage1: bool,
) -> torch.Tensor:
    """Return [tokens, topk, inter_dim] fp32."""
    tokens, model_dim = x_fp8.shape
    topk = topk_ids.shape[1]
    experts = int(topk_ids.max().item()) + 1

    x = x_fp8.to(torch.float32) * scale_x  # [tokens, model_dim] (scale_x [tokens,1])
    w1 = w1_fp8_flat.to(torch.float32) * scale_w1_flat  # [experts*2*inter_dim, model_dim] (scale [rows,1])
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
    scale_a2: torch.Tensor,
    scale_w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    model_dim: int,
    doweight_stage2: bool,
) -> torch.Tensor:
    """Return [tokens, model_dim] fp32.

    Semantics align with aiter `torch_moe_stage2`:
    - Dequantize `a2_fp8` and `w2_fp8` with per-token/row scales.
    - For each routed (token, slot) -> expert, compute y = a2 @ W2[expert]^T.
    - Optionally multiply routed weight in stage2 (when stage1 did *not*).
    - Reduce across topk by summing into [tokens, model_dim].
    """
    assert a2_fp8.is_cuda and w2_fp8.is_cuda
    tokens, topk, inter_dim = a2_fp8.shape
    experts = int(topk_ids.max().item()) + 1

    # Dequantize inputs.
    a2 = a2_fp8.to(torch.float32) * scale_a2  # scale_a2: [tokens, topk, 1]
    w2 = w2_fp8.to(torch.float32) * scale_w2  # scale_w2: [experts, model_dim, 1]
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
