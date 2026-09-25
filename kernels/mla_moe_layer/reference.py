# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Weights, layouts and Torch goldens for fused decode shards.

One rank's TP shard of one layer. Attention matrices use either row-major FP8
E4M3FN with FP32 block scales or BF16. Expert matrices use either block-scaled
FP8 or packed MXFP4 with per-1x32 E8M0 scales. The golden reduces through a
caller-supplied ``allreduce`` so a multi-rank run checks every rank against its
own shard.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.common.mx_formats import dequantize_mxfp4, quant_dequant_mxfp8, quantize_mxfp4
from kernels.mla_moe_layer.config import (
    EPS,
    FP8_MAX,
    GLM5_CONFIG,
    HIDDEN,
    INTER,
    N_EXPERTS,
    ROUTE_SCALE,
    SCALE_BM,
    SHARED_EXPERT,
    TOP_K,
    AttentionWeight,
    ExpertActivation,
    ExpertWeight,
    LayerConfig,
    MoeMode,
    as_layer_config,
    as_moe_mode,
    moe_format,
)


# (name, rows, K, BK) of every attention matrix, rows given per local head count H.
def attention_mats(heads: int, model_config: LayerConfig | str = GLM5_CONFIG):
    config = as_layer_config(model_config)
    return {
        "qkv_a": (config.qkv_a_rows, config.hidden, 128),
        "q_b": (heads * (config.nope_dim + config.pe_dim), config.q_lora, 128),
        "uk": (heads * config.kv_lora, config.nope_dim, 64),
        "uv": (heads * config.v_dim, config.kv_lora, 128),
        "o": (config.hidden, heads * config.v_dim, 128),
    }


def fp8_mats(heads: int):
    """Backward-compatible GLM-5 attention matrix description."""

    return attention_mats(heads, GLM5_CONFIG)


def scale_shape(rows: int, k: int, bk: int):
    return ((rows + SCALE_BM - 1) // SCALE_BM, k // bk)


def _rand_fp8(rows, k, bk, gen, device, lead=()):
    q = (torch.randn(*lead, rows, k, generator=gen, device=device) * 16).clamp(-FP8_MAX, FP8_MAX)
    q = q.to(torch.float8_e4m3fn)
    sr, sk = scale_shape(rows, k, bk)
    s = (torch.rand(*lead, sr, sk, generator=gen, device=device) * 0.4 + 0.8) / (16 * k**0.5)
    return q, s


def dequant(q: torch.Tensor, s: torch.Tensor, bk: int) -> torch.Tensor:
    rows, k = q.shape
    sf = s.repeat_interleave(SCALE_BM, 0)[:rows].repeat_interleave(bk, 1)
    return q.float() * sf


@dataclass
class LayerWeights:
    heads: int
    t: dict  # name -> tensor
    config: LayerConfig = GLM5_CONFIG


def make_weights(
    rank: int,
    heads: int = 8,
    device="cuda",
    seed: int = 1234,
    moe_mode: MoeMode | str = MoeMode.W8A8,
    model_config: LayerConfig | str = GLM5_CONFIG,
    attention_only: bool = False,
) -> LayerWeights:
    """Replicated tensors share ``seed``; TP shards add ``rank`` to it."""
    config = as_layer_config(model_config)
    if config != GLM5_CONFIG and not attention_only:
        raise ValueError(f"{config.name} currently supports the attention-only kernel path")
    expert_weight = moe_format(moe_mode).weight
    rep = torch.Generator(device=device).manual_seed(seed)
    shd = torch.Generator(device=device).manual_seed(seed + 1 + rank)
    t = {}
    bf = torch.bfloat16
    t["g_in"] = (1 + 0.1 * torch.randn(config.hidden, generator=rep, device=device)).to(bf)
    t["g_q"] = (1 + 0.1 * torch.randn(config.q_lora, generator=rep, device=device)).to(bf)
    t["g_kv"] = (1 + 0.1 * torch.randn(config.kv_lora, generator=rep, device=device)).to(bf)
    t["g_post"] = (1 + 0.1 * torch.randn(config.hidden, generator=rep, device=device)).to(bf)
    for name, (rows, k, bk) in attention_mats(heads, config).items():
        if name == "qkv_a" and config.attention_output_gate:
            core_rows = config.q_lora + config.kv_lora + config.pe_dim
            core = (torch.randn(core_rows, k, generator=rep, device=device) / k**0.5).to(bf)
            gate = (torch.randn(rows - core_rows, k, generator=shd, device=device) / k**0.5).to(bf)
            t[f"w_{name}"] = torch.cat((core, gate))
        elif config.attention_weight is AttentionWeight.BF16:
            gen = rep if name == "qkv_a" else shd
            t[f"w_{name}"] = (torch.randn(rows, k, generator=gen, device=device) / k**0.5).to(bf)
        else:
            gen = rep if name == "qkv_a" else shd
            t[f"w_{name}"], t[f"s_{name}"] = _rand_fp8(rows, k, bk, gen, device)
    if config.attention_weight is AttentionWeight.BF16:
        dummy_scale = torch.ones(1, dtype=torch.float32, device=device)
        for name in attention_mats(heads, config):
            t[f"s_{name}"] = dummy_scale
    if attention_only:
        return LayerWeights(heads, t, config)
    t["w_r"] = (
        torch.randn(config.n_experts, config.hidden, generator=rep, device=device) / config.hidden**0.5 * 4
    ).to(bf)
    t["bias"] = torch.randn(config.n_experts, generator=rep, device=device) * 0.1
    if expert_weight is ExpertWeight.FP8_BLOCK128:
        ug_q = torch.empty(
            config.n_experts + 1,
            2 * config.inter,
            config.hidden,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        ug_s = torch.empty(
            config.n_experts + 1,
            *scale_shape(2 * config.inter, config.hidden, 128),
            device=device,
        )
        dn_q = torch.empty(
            config.n_experts + 1,
            config.hidden,
            config.inter,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        dn_s = torch.empty(
            config.n_experts + 1,
            *scale_shape(config.hidden, config.inter, 128),
            device=device,
        )
        for e in range(config.n_experts + 1):
            ug_q[e], ug_s[e] = _rand_fp8(2 * config.inter, config.hidden, 128, shd, device)
            dn_q[e], dn_s[e] = _rand_fp8(config.hidden, config.inter, 128, shd, device)
    else:
        ug_q = torch.empty(
            config.n_experts + 1,
            2 * config.inter,
            config.hidden // 2,
            dtype=torch.uint8,
            device=device,
        )
        ug_s = torch.empty(
            config.n_experts + 1,
            2 * config.inter,
            config.hidden // 32,
            dtype=torch.uint8,
            device=device,
        )
        dn_q = torch.empty(
            config.n_experts + 1,
            config.hidden,
            config.inter // 2,
            dtype=torch.uint8,
            device=device,
        )
        dn_s = torch.empty(
            config.n_experts + 1,
            config.hidden,
            config.inter // 32,
            dtype=torch.uint8,
            device=device,
        )
        for e in range(config.n_experts + 1):
            ug = torch.randn(2 * config.inter, config.hidden, generator=shd, device=device) / config.hidden**0.5
            dn = torch.randn(config.hidden, config.inter, generator=shd, device=device) / config.inter**0.5
            ug_q[e], ug_s[e] = quantize_mxfp4(ug)
            dn_q[e], dn_s[e] = quantize_mxfp4(dn)
    t["w_ug"], t["s_ug"], t["w_dn"], t["s_dn"] = ug_q, ug_s, dn_q, dn_s
    return LayerWeights(heads, t, config)


def dequant_expert(q: torch.Tensor, scale: torch.Tensor, weight: ExpertWeight) -> torch.Tensor:
    """Decode one logical expert matrix for the torch reference."""

    if weight is ExpertWeight.MXFP4_BLOCK32:
        return dequantize_mxfp4(q, scale)
    return dequant(q, scale, 128)


def rope_table(
    max_seq: int,
    theta: float = 8.0e6,
    device="cuda",
    model_config: LayerConfig | str = GLM5_CONFIG,
):
    config = as_layer_config(model_config)
    inv = 1.0 / theta ** (torch.arange(0, config.pe_dim, 2, device=device, dtype=torch.float64) / config.pe_dim)
    ang = torch.arange(max_seq, device=device, dtype=torch.float64)[:, None] * inv[None]
    return torch.cos(ang).float().contiguous(), torch.sin(ang).float().contiguous()


def bf(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 and back (the precision of MFMA activation operands)."""
    return x.to(torch.bfloat16).float()


def rmsnorm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * g.float()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved pairs (2i, 2i+1); ``x`` [..., 64], ``cos``/``sin`` [32]."""
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def quant_dequant(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Per-``block`` dynamic FP8 E4M3FN quantization of the last dim, returned dequantized."""
    xb = x.float().reshape(*x.shape[:-1], -1, block)
    amax = xb.abs().amax(-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(x.shape)


def route(scores: torch.Tensor, bias: torch.Tensor):
    """sigmoid scores [E] -> (indices [8], probs [8]) in score order.

    Selection key (as in the kernel's packed-key argmax): the order-preserving bits
    of the f32 ``score + bias`` with the low byte replaced by ``255 - expert id``,
    so keys are unique and near-ties go to the lower expert id."""
    bits = (scores.float() + bias.float()).view(torch.int32).long()
    okey = torch.where(bits >= 0, bits ^ (1 << 31), ~bits & 0xFFFFFFFF) & 0xFFFFFFFF
    key = (okey & 0xFFFFFF00) | (255 - torch.arange(N_EXPERTS, device=scores.device))
    idx = torch.argsort(key, descending=True)[:TOP_K]
    p = scores[idx]
    return idx, p / p.sum() * ROUTE_SCALE


def golden_layer(
    W: LayerWeights,
    h,
    cur_pos: int,
    kv_cache,
    pe_cache,
    indices,
    cos,
    sin,
    allreduce,
    topk=2048,
    moe_mode: MoeMode | str = MoeMode.W8A8,
    attention_only: bool = False,
):
    """One rank's view of the layer. Mutates ``kv_cache``/``pe_cache`` like the kernel.

    Returns a dict of intermediates keyed like the kernel's debug scratch.
    """
    t, H, config = W.t, W.heads, W.config
    S = h.shape[0]
    if config.attention_weight is AttentionWeight.BF16:
        dq = {name: t[f"w_{name}"].float() for name in attention_mats(H, config)}
    else:
        dq = {
            name: dequant(t[f"w_{name}"], t[f"s_{name}"], bk) for name, (_, _, bk) in attention_mats(H, config).items()
        }
    # GEMV activations are bf16 (MFMA inputs); weights retain their configured format.
    x = bf(rmsnorm(h, t["g_in"])) if config.attention_input_norm else h.float()
    qkv = x @ dq["qkv_a"].T
    q_a = qkv[:, : config.q_lora]
    kv_end = config.q_lora + config.kv_lora + config.pe_dim
    kv_a = qkv[:, config.q_lora : kv_end]
    gate = qkv[:, kv_end:].view(S, H, config.v_dim) if config.attention_output_gate else None
    qb = (bf(rmsnorm(q_a, t["g_q"])) @ dq["q_b"].T).view(S, H, config.nope_dim + config.pe_dim)
    q_nope = qb[..., : config.nope_dim]
    pos = [cur_pos + s for s in range(S)]
    q_pe = torch.stack([rope(qb[s, :, config.nope_dim :], cos[pos[s]], sin[pos[s]]) for s in range(S)])
    q_lat = torch.einsum(
        "hkd,shd->shk",
        dq["uk"].view(H, config.kv_lora, config.nope_dim),
        bf(q_nope),
    )
    for s in range(S):
        kv_cache[pos[s]] = rmsnorm(kv_a[s, : config.kv_lora], t["g_kv"]).to(torch.bfloat16)
        pe_cache[pos[s]] = rope(kv_a[s, config.kv_lora :], cos[pos[s]], sin[pos[s]]).to(torch.bfloat16)
    kvf, pef = kv_cache.float(), pe_cache.float()
    o_lat = torch.empty(S, H, config.kv_lora, device=h.device)
    for s in range(S):
        kv_len = pos[s] + 1
        keys = indices[s].long() if kv_len > topk else torch.arange(kv_len, device=h.device)
        sc = (bf(q_lat[s]) @ kvf[keys].T + bf(q_pe[s]) @ pef[keys].T) * config.softmax_scale
        # split softmax over 64-key splits: bf16 unnormalized probs feed P V (MFMA)
        ms, ls, accs = [], [], []
        for k0 in range(0, len(keys), 64):
            scs = sc[:, k0 : k0 + 64]
            m = scs.amax(-1, keepdim=True)
            p = torch.exp(scs - m)
            ms.append(m)
            ls.append(p.sum(-1, keepdim=True))
            accs.append(bf(p) @ kvf[keys[k0 : k0 + 64]])
        mx = torch.stack(ms).amax(0)
        w = [torch.exp(m - mx) for m in ms]
        o_lat[s] = sum(a * wi for a, wi in zip(accs, w)) / sum(li * wi for li, wi in zip(ls, w))
    o = torch.einsum(
        "hvk,shk->shv",
        dq["uv"].view(H, config.v_dim, config.kv_lora),
        bf(o_lat),
    ).reshape(S, H * config.v_dim)
    if gate is not None:
        o = bf(o) * torch.sigmoid(gate.reshape(S, H * config.v_dim))
    a = allreduce(bf(o) @ dq["o"].T)
    if config.attention_residual:
        a = h.float() + a
    a = a.to(torch.bfloat16)
    if attention_only:
        return dict(q_a=q_a, kv_a=kv_a, q_nope=q_nope, q_pe=q_pe, q_lat=q_lat, o=o, a=a, gate=gate)
    moe = golden_moe(W, a, allreduce, moe_mode=moe_mode)
    res = dict(q_a=q_a, kv_a=kv_a, q_nope=q_nope, q_pe=q_pe, q_lat=q_lat, o=o, a=a)
    res.update(moe)
    return res


def golden_moe(
    W: LayerWeights,
    a,
    allreduce,
    mid=None,
    sel=None,
    prob=None,
    xq=None,
    moe_mode: MoeMode | str = MoeMode.W8A8,
):
    """MoE half of the layer from the post-attention hidden state ``a`` [S, HIDDEN] (bf16).

    ``xq`` [S, HIDDEN] overrides the quant-dequantized activation and
    ``mid``/``sel``/``prob`` ([S, 9, INTER] / [S, 9] / [S, 9]) the down-projection
    inputs, so each stage can be checked from the kernel's own inputs.
    """
    mode = as_moe_mode(moe_mode)
    fmt = moe_format(mode)
    t = W.t
    S = a.shape[0]
    out = {k: [] for k in ("sel", "prob", "mid")}
    x2 = rmsnorm(a, t["g_post"])
    scores = torch.sigmoid(bf(x2) @ t["w_r"].float().T)
    if fmt.activation is ExpertActivation.FP8_BLOCK128:
        xq_ref = quant_dequant(x2)
    elif fmt.activation is ExpertActivation.MXFP8_BLOCK32:
        xq_ref = quant_dequant_mxfp8(x2)
    else:
        xq_ref = bf(x2)
    xq = xq_ref if xq is None else xq.float()
    y = torch.zeros(S, HIDDEN, device=a.device)
    for s in range(S):
        idx, p = route(scores[s], t["bias"])
        experts = [SHARED_EXPERT] + idx.tolist()
        weights = [1.0] + p.tolist()
        mids = []
        for e, wgt in zip(experts, weights):
            ug = dequant_expert(t["w_ug"][e], t["s_ug"][e], fmt.weight) @ xq[s]
            value = torch.nn.functional.silu(ug[:INTER]) * ug[INTER:]
            mids.append(bf(value) if fmt.activation is ExpertActivation.BF16 else value)
        out["sel"].append(torch.tensor(experts, device=a.device, dtype=torch.int32))
        out["prob"].append(torch.tensor(weights, device=a.device))
        out["mid"].append(torch.stack(mids))
    for s in range(S):
        experts = out["sel"][s].tolist() if sel is None else sel[s].tolist()
        weights = out["prob"][s].tolist() if prob is None else prob[s].tolist()
        for j, (e, wgt) in enumerate(zip(experts, weights)):
            m = out["mid"][s][j] if mid is None else mid[s, j].float()
            if fmt.activation is ExpertActivation.FP8_BLOCK128:
                activation = quant_dequant(m)
            elif fmt.activation is ExpertActivation.MXFP8_BLOCK32:
                activation = quant_dequant_mxfp8(m)
            else:
                activation = bf(m)
            y[s] += wgt * (dequant_expert(t["w_dn"][e], t["s_dn"][e], fmt.weight) @ activation)
    x_out = (a.float() + allreduce(y)).to(torch.bfloat16)
    return dict(
        scores=scores,
        xq=xq_ref,
        x_out=x_out,
        sel=torch.stack(out["sel"]),
        prob=torch.stack(out["prob"]),
        mid=torch.stack(out["mid"]),
    )
