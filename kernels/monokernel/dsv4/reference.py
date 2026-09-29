# SPDX-License-Identifier: Apache-2.0
"""Independent Torch reference for the DSV4 A8W4 contract."""

import torch
import torch.nn.functional as F

from kernels.monokernel.formats import dequantize_mxfp4


def quant_dequant_a8(x, block_size=32):
    """ATOM MXFP8: per-32 E4M3, E8M0 scale rounded UP, including zero blocks.

    This deliberately differs from the nearest-scale helper used by K3's
    MXFP8 dense projections. Routed GEMM1 input and shared activations use
    this rule; routed GEMM1's fused output uses quant_dequant_routed_mid.
    """
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // block_size, block_size)
    amax = blocks.abs().amax(-1)
    bits = (amax / 448.0).contiguous().view(torch.int32)
    exponents = ((bits + 0x7FFFFF) >> 23).clamp(1, 254)
    scale = (exponents << 23).view(torch.float32)
    q = (blocks / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.float() * scale.unsqueeze(-1)).reshape(shape)


def quant_dequant_routed_mid(x):
    """Native AITER *_fp8 epilogue: FP32 SwiGLU directly to per-32 FP8."""
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    bits = blocks.abs().amax(-1).contiguous().view(torch.int32)
    exponent = ((((bits + 0x400000) & -0x800000) >> 23) - 8).clamp_min(0)
    inverse = ((254 - exponent) << 23).view(torch.float32)
    q = (blocks * inverse.unsqueeze(-1)).to(torch.float8_e4m3fn)
    # Reciprocal also handles E8M0 exponent zero (2**-127).
    return (q.float() / inverse.unsqueeze(-1)).reshape(shape)


def route(x, router, bias, config, hash_ids=None):
    # ATOM's BF16 ReplicatedLinear materializes BF16 logits before scoring.
    logits = F.linear(x.float(), router.float()).to(torch.bfloat16).float()
    scores = F.softplus(logits).sqrt()
    if hash_ids is None:
        # Stable sort gives lower expert id precedence on exact score ties.
        ids = torch.argsort(scores + bias.float(), dim=-1, descending=True, stable=True)[:, : config.top_k]
    else:
        ids = hash_ids.long()
    selected = scores.gather(1, ids)
    weights = selected / selected.sum(-1, keepdim=True) * config.route_scale
    return ids.to(torch.int32), weights


def moe_reference(
    x, router, bias, up, up_scale, down, down_scale, config, hash_ids=None, shared=None, *, return_parts=False
):
    ids, weights = route(x, router, bias, config, hash_ids)
    xq = quant_dequant_a8(x)
    out = torch.zeros_like(x, dtype=torch.float32)
    # Dequantize only selected experts, so full Pro TP1 fixtures fit comfortably.
    for sample in range(x.shape[0]):
        for slot in range(config.top_k + (shared is None)):
            expert = config.experts if slot == config.top_k else int(ids[sample, slot])
            wgu = dequantize_mxfp4(up[expert], up_scale[expert])
            wd = dequantize_mxfp4(down[expert], down_scale[expert])
            gate, linear = F.linear(xq[sample], wgu).chunk(2, -1)
            if config.swiglu_limit:
                gate = gate.clamp(max=config.swiglu_limit)
                linear = linear.clamp(-config.swiglu_limit, config.swiglu_limit)
            # Routed *_fp8 GEMM1 never materializes a BF16 activation.
            mid = F.silu(gate) * linear
            partial = F.linear(quant_dequant_routed_mid(mid), wd)
            factor = 1.0 if slot == config.top_k else weights[sample, slot]
            out[sample] += partial * factor
    out = out.to(torch.bfloat16)
    routed_out = out
    shared_out = None
    if shared is not None:
        su, sus, sd, sds = shared

        def dequant(w, scales):
            scales = torch.pow(2.0, scales.float() - 127)
            return w.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)

        gu = F.linear(quant_dequant_a8(x, 128), dequant(su, sus)).to(torch.bfloat16)
        g, u = gu.float().chunk(2, -1)
        g = g.clamp(max=config.swiglu_limit)
        u = u.clamp(-config.swiglu_limit, config.swiglu_limit)
        mid = (F.silu(g) * u).to(torch.bfloat16)
        shared_out = F.linear(quant_dequant_a8(mid, 128), dequant(sd, sds)).to(torch.bfloat16)
        out = out + shared_out
    if return_parts:
        return out, ids, weights, routed_out, shared_out
    return out, ids, weights
