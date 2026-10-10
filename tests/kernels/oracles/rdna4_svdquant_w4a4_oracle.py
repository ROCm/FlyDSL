# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only torch oracles extracted from ``kernels/quant/rdna4_svdquant_w4a4.py``."""

import torch

from kernels.quant.rdna4_int4_codec import (
    dequant_int4_groupwise_signed,
    unpack_int4_row_major,
    unpack_uint4_row_major,
)

_INT4_GROUP_SIZE = 64


def reference_dequant_svdquant_w4a4_weight(
    qweight: torch.Tensor, wscales: torch.Tensor, group_size: int = _INT4_GROUP_SIZE
) -> torch.Tensor:
    """Torch reference: signed INT4 × group scales (eager reference weight path)."""
    return dequant_int4_groupwise_signed(qweight, wscales, group_size=group_size)


def reference_scaled_mm_svdquant_w4a4(
    act: object,
    wgt: object,
    ascales: object,
    wscales: torch.Tensor,
    lora_act_in: object,
    lora_up: object,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
    group_size: int = _INT4_GROUP_SIZE,
) -> torch.Tensor:
    """Pure-torch SVDQuant W4A4 GEMM + LoRA-up (eager reference semantic mirror)."""

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
