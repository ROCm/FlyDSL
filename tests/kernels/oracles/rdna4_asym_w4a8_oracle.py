# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only torch oracles extracted from ``kernels/quant/rdna4_asym_w4a8.py``."""

import torch


def reference_dequant_int4_grouped_to_int8(
    qdata: object, s_rel: object, codebook: object = None, group_size: int = 16
) -> torch.Tensor:
    """Torch reference for packed INT4 → INT8 grid (eager reference)."""
    import torch

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
