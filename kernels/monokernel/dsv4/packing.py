# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side weight packing into the DeepSeek-V4 kernel's 16-row MFMA tiles."""

from __future__ import annotations

import torch

from kernels.monokernel.dsv4.config import ExpertWeight, MoeMode, moe_format
from kernels.monokernel.packing import pack_bf16, pack_fp8

__all__ = ["pack_bf16", "pack_fp8", "pack_mxfp4", "pack_mxfp4_scales", "pack_layer_weights"]


def pack_mxfp4(q: torch.Tensor, gate_up: bool) -> torch.Tensor:
    """An MXFP4 bank [E, N, K / 2] in ATOM's gfx950 layout (aiter ``shuffle_weight``,
    ``is_guinterleave=True``), so the kernel can read ATOM's own bank."""
    q = q.view(torch.uint8)
    e, n, kp = q.shape
    if n % (32 if gate_up else 16) or kp % 64:
        raise ValueError(f"MXFP4 bank dimensions must be divisible by (16 per half, 128), got {(n, kp * 2)}")
    if gate_up:
        w = q.view(e, 2, n // 32, 16, kp // 64, 4, 16).permute(0, 2, 1, 4, 5, 3, 6)
    else:
        w = q.view(e, n // 16, 16, kp // 64, 4, 16).permute(0, 1, 3, 4, 2, 5)
    return w.contiguous().view(e, n, kp)


def pack_mxfp4_scales(s: torch.Tensor, gate_up: bool) -> torch.Tensor:
    """E8M0 scales [E, N, K / 32] in ATOM's gfx950 layout (aiter ``shuffle_scale``,
    ``is_guinterleave=True``); K / 32 is padded to a multiple of 8 with 1.0 (0x7F)."""
    s = s.view(torch.uint8)
    e, n, kb = s.shape
    kb8 = -(-kb // 8) * 8
    if kb8 != kb:
        s = torch.cat([s, torch.full((e, n, kb8 - kb), 0x7F, dtype=torch.uint8, device=s.device)], dim=-1)
    if gate_up:
        t = s.view(e, 2, n // 32, 16, kb8 // 8, 2, 4).permute(0, 2, 4, 6, 3, 5, 1)
    else:
        t = s.view(e, n // 32, 2, 16, kb8 // 8, 2, 4).permute(0, 1, 4, 6, 3, 5, 2)
    return t.contiguous().view(e, n, kb8)


ATTENTION_NAMES = ("w_qkv_a", "w_q_b", "w_o_a", "w_o_b")
# CSA layers only
OPTIONAL_ATTENTION_NAMES = ("w_i_q_b",)
EXPERT_NAMES = ("w_ug", "w_dn")
# the shared expert of an MXFP4 bank stays FP8
SHARED_EXPERT_NAMES = ("w_sug", "w_sdn")
HC_NAMES = ("hc_attn_fn", "hc_ffn_fn")


EXPERT_BANK_NAMES = ("w_ug", "s_ug", "w_dn", "s_dn")


def pack_layer_weights(
    tensors: dict[str, torch.Tensor],
    moe_mode: MoeMode | str = MoeMode.A8W4,
    experts: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Pack every matrix the V4 kernel reads; ``experts``, if given, is an already
    packed MXFP4 bank (e.g. ATOM's) used as is."""
    need = (*ATTENTION_NAMES, "w_r") if experts is not None else (*ATTENTION_NAMES, *EXPERT_NAMES, "w_r")
    missing = [name for name in need if name not in tensors]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")
    packed = {name: pack_fp8(tensors[name]) for name in ATTENTION_NAMES}
    packed.update({name: pack_fp8(tensors[name]) for name in OPTIONAL_ATTENTION_NAMES if name in tensors})
    if experts is not None:
        if moe_format(moe_mode).weight is not ExpertWeight.MXFP4_BLOCK32:
            raise ValueError("an external expert bank is MXFP4 (ATOM's layout)")
        packed.update({name: experts[name] for name in EXPERT_BANK_NAMES})
    elif moe_format(moe_mode).weight is ExpertWeight.MXFP4_BLOCK32:
        for name, scale, gate_up in (("w_ug", "s_ug", True), ("w_dn", "s_dn", False)):
            packed[name] = pack_mxfp4(tensors[name], gate_up)
            packed[scale] = pack_mxfp4_scales(tensors[scale], gate_up)
    else:
        packed.update({name: pack_fp8(tensors[name]) for name in EXPERT_NAMES})
    packed.update({name: pack_fp8(tensors[name]) for name in SHARED_EXPERT_NAMES if name in tensors})
    packed["w_r"] = pack_bf16(tensors["w_r"])
    if "w_qkv_c" in tensors:
        packed["w_qkv_c"] = pack_bf16(tensors["w_qkv_c"])
    for name in HC_NAMES:
        if name in tensors:
            packed[name] = pack_hc_fn(tensors[name])
    return packed


def pack_hc_fn(fn: torch.Tensor) -> torch.Tensor:
    """An fp32 mixer [rows, K] as bf16 [hi; lo] (hi + lo ~ 16 mantissa bits), as ATOM's mHC."""
    fn = fn.float()
    hi = fn.to(torch.bfloat16)
    lo = (fn - hi.float()).to(torch.bfloat16)
    return pack_bf16(torch.cat([hi, lo]))
