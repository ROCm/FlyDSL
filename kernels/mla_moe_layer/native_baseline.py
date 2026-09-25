# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Optional TileRT baseline adapter for same-weight performance comparisons."""

from __future__ import annotations

import torch

from kernels.mla_moe_layer.config import MoeMode, as_moe_mode
from kernels.mla_moe_layer.reference import LayerWeights


def make_native_glm5_baseline(
    weights: LayerWeights,
    device,
    moe_mode: MoeMode | str = MoeMode.W8A8,
):
    """Populate TileRT's released GLM-5.2 wrapper from reference weights.

    TileRT is imported lazily and remains outside the FlyDSL execution path.
    Both its W8A8 and V4/W8A16 expert modes are supported.
    """

    mode = as_moe_mode(moe_mode)
    import tilert

    tilert.load_backend("glm5_2_rocm")
    from tilert.models.glm_5_2_rocm.ops import (
        proj_wkvb,
        rmsnorm_projq_wqb,
        rmsnorm_projx_wqkva,
        unprojo_allreduce,
    )
    from tilert.models.glm_5_2_rocm.ops.pure_mla_moe_layer import PureMlaMoeLayerGlm5
    from tilert.models.glm_5_2_rocm.ops.upgate_silu import swizzle_pair_interleaved

    baseline = PureMlaMoeLayerGlm5(device=device, num_heads=8, moe_w8a8=mode is MoeMode.W8A8)
    tensors, attention = weights.t, baseline.mla
    for module, name, packer in (
        (attention.m0, "qkv_a", rmsnorm_projx_wqkva.swizzle_weights_contig),
        (attention.m1, "q_b", rmsnorm_projq_wqb.swizzle_weights_contig),
        (attention.m4, "uk", proj_wkvb.swizzle_weights_contig),
        (attention.m6, "uv", proj_wkvb.swizzle_weights_contig),
        (attention.m7, "o", unprojo_allreduce.swizzle_v2),
    ):
        weight, scales = tensors[f"w_{name}"], tensors[f"s_{name}"]
        if name == "q_b":
            # FlyDSL groups [192 NoPE, 64 RoPE] within each head. The native
            # baseline stores every head's NoPE rows before all RoPE rows.
            blocks = torch.arange(8 * 4, device=device).reshape(8, 4)
            order64 = torch.cat((blocks[:, :3].flatten(), blocks[:, 3].flatten()))
            rows = (order64[:, None] * 64 + torch.arange(64, device=device)).flatten()
            weight = weight.view(torch.uint8)[rows].view(torch.float8_e4m3fn)
            scales = scales.repeat_interleave(2, dim=0)[order64]
        module.packed = packer(weight.cpu()).to(device)
        module.scales = scales
        if name in ("uk", "uv"):
            module.scales = module.scales.repeat_interleave(2, dim=0)

    attention.m0.gamma_arg = tensors["g_in"].float()
    attention.m1.gamma_arg = tensors["g_q"].float()
    attention.m3.gamma = tensors["g_kv"].float()

    experts = baseline.moe
    experts.front.router.w = tensors["w_r"].cpu()
    experts.front.router.gamma = tensors["g_post"].float()
    # The native bank stores the shared expert first; the reference bank stores it last.
    order = torch.tensor([256, *range(256)], device=device)
    experts.front.moe.w_fp8 = tensors["w_ug"].view(torch.uint8)[order].view(torch.float8_e4m3fn).cpu()
    experts.front.moe.scales = tensors["s_ug"][order]
    experts.down.w_fp8 = tensors["w_dn"].view(torch.uint8)[order].view(torch.float8_e4m3fn).cpu()
    experts.down.scales = tensors["s_dn"][order]
    if mode is MoeMode.W8A16:
        experts.front.moe.packed = torch.cat(
            [swizzle_pair_interleaved(weight, experts.front.moe.inter) for weight in experts.front.moe.w_fp8]
        ).to(device)
    baseline.moe_banks()
    experts.front.moe.w_fp8 = None
    experts.down.w_fp8 = None
    return baseline
