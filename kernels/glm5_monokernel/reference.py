# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Weights, layouts and torch goldens for the GLM-5 indexed decode MonoKernel.

The indexed variant covers selection refresh plus MLA and MoE for one rank's
TP shard. Attention matrices use row-major FP8 E4M3FN; expert matrices use
either block-scaled FP8 or packed MXFP4. The golden reduces across ranks
through a caller-supplied ``allreduce`` so a TP8 run checks each rank against
its own shard.
"""

from __future__ import annotations

import torch

from kernels.common.fused_layer_config import (
    FP8_MAX,
    HIDDEN,
    INTER,
    KV_LORA,
    N_EXPERTS,
    PE_DIM,
    Q_LORA,
    SHARED_EXPERT,
    ExpertWeight,
)
from kernels.common.fused_layer_reference import (
    LayerWeights,
    bf,
    dequant,
    fp8_mats,
    quant_dequant,
    rmsnorm,
    rope,
    route,
    scale_shape,
)
from kernels.common.fused_layer_reference import dequant_expert as _dequant_expert
from kernels.common.fused_layer_reference import golden_layer as _golden_attention
from kernels.glm5_monokernel.layout import INDEX_DIM, INDEX_HEADS, INDEX_Q_ROWS


def _rand_fp8(rows, k, bk, gen, device, lead=()):
    q = (torch.randn(*lead, rows, k, generator=gen, device=device) * 16).clamp(-FP8_MAX, FP8_MAX)
    q = q.to(torch.float8_e4m3fn)
    sr, sk = scale_shape(rows, k, bk)
    s = (torch.rand(*lead, sr, sk, generator=gen, device=device) * 0.4 + 0.8) / (16 * k**0.5)
    return q, s


def make_weights(
    rank: int,
    heads: int = 8,
    device="cuda",
    seed: int = 1234,
    with_indexer: bool = False,
    expert_mxfp4: bool = False,
) -> LayerWeights:
    """Replicated tensors share ``seed``; TP shards add ``rank`` to it."""
    rep = torch.Generator(device=device).manual_seed(seed)
    shd = torch.Generator(device=device).manual_seed(seed + 1 + rank)
    t = {}
    bf = torch.bfloat16
    t["g_in"] = (1 + 0.1 * torch.randn(HIDDEN, generator=rep, device=device)).to(bf)
    t["g_q"] = (1 + 0.1 * torch.randn(Q_LORA, generator=rep, device=device)).to(bf)
    t["g_kv"] = (1 + 0.1 * torch.randn(KV_LORA, generator=rep, device=device)).to(bf)
    t["g_post"] = (1 + 0.1 * torch.randn(HIDDEN, generator=rep, device=device)).to(bf)
    for name, (rows, k, bk) in fp8_mats(heads).items():
        gen = rep if name == "qkv_a" else shd
        t[f"w_{name}"], t[f"s_{name}"] = _rand_fp8(rows, k, bk, gen, device)
    if with_indexer:
        t["w_index_k"], t["s_index_k"] = _rand_fp8(INDEX_DIM, HIDDEN, 128, rep, device)
        t["w_index_q"], t["s_index_q"] = _rand_fp8(INDEX_Q_ROWS, Q_LORA, 128, rep, device)
        t["w_index_w"] = (torch.randn(INDEX_HEADS, HIDDEN, generator=rep, device=device) / HIDDEN**0.5).to(bf)
        t["g_index_k"] = (1 + 0.1 * torch.randn(INDEX_DIM, generator=rep, device=device)).float()
        t["b_index_k"] = (0.1 * torch.randn(INDEX_DIM, generator=rep, device=device)).float()
    t["w_r"] = (torch.randn(N_EXPERTS, HIDDEN, generator=rep, device=device) / HIDDEN**0.5 * 4).to(bf)
    t["bias"] = torch.randn(N_EXPERTS, generator=rep, device=device) * 0.1
    if expert_mxfp4:
        # Native MXFP4 storage: two E2M1 values per byte and one E8M0 scale
        # byte per 32 values.  Fixed representative scales keep test-weight
        # construction cheap while exercising the production storage contract.
        ug_q = torch.randint(
            0,
            256,
            (N_EXPERTS + 1, 2 * INTER, HIDDEN // 2),
            generator=shd,
            dtype=torch.uint8,
            device=device,
        )
        ug_s = torch.full(
            (N_EXPERTS + 1, 2 * INTER, HIDDEN // 32),
            118,
            dtype=torch.uint8,
            device=device,
        )
        dn_q = torch.randint(
            0,
            256,
            (N_EXPERTS + 1, HIDDEN, INTER // 2),
            generator=shd,
            dtype=torch.uint8,
            device=device,
        )
        dn_s = torch.full(
            (N_EXPERTS + 1, HIDDEN, INTER // 32),
            121,
            dtype=torch.uint8,
            device=device,
        )
    else:
        ug_q = torch.empty(N_EXPERTS + 1, 2 * INTER, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
        ug_s = torch.empty(N_EXPERTS + 1, *scale_shape(2 * INTER, HIDDEN, 128), device=device)
        dn_q = torch.empty(N_EXPERTS + 1, HIDDEN, INTER, dtype=torch.float8_e4m3fn, device=device)
        dn_s = torch.empty(N_EXPERTS + 1, *scale_shape(HIDDEN, INTER, 128), device=device)
        for e in range(N_EXPERTS + 1):
            ug_q[e], ug_s[e] = _rand_fp8(2 * INTER, HIDDEN, 128, shd, device)
            dn_q[e], dn_s[e] = _rand_fp8(HIDDEN, INTER, 128, shd, device)
    t["w_ug"], t["s_ug"], t["w_dn"], t["s_dn"] = ug_q, ug_s, dn_q, dn_s
    return LayerWeights(heads, t)


def indexer_golden(W: LayerWeights, h, q_a, cur_pos: int, index_cache, cos, sin, topk: int = 2048):
    """Reference for the fused BF16 indexer path; mutates ``index_cache``."""
    t = W.t
    S = h.shape[0]
    x = bf(rmsnorm(h, t["g_in"]))
    qn = bf(rmsnorm(q_a, t["g_q"]))
    wk = dequant(t["w_index_k"], t["s_index_k"], 128)
    wq = dequant(t["w_index_q"], t["s_index_q"], 128)
    raw_k = x @ wk.T
    mean = raw_k.mean(-1, keepdim=True)
    var = (raw_k - mean).square().mean(-1, keepdim=True)
    index_k = (raw_k - mean) * torch.rsqrt(var + 1e-6)
    index_k = index_k * t["g_index_k"] + t["b_index_k"]
    # The fused path stores the projection in a packed-BF16 mailbox before
    # applying RoPE in the score CTAs.
    index_q = bf(qn @ wq.T).view(S, INDEX_HEADS, INDEX_DIM)
    score_weights = x @ t["w_index_w"].float().T
    positions = torch.arange(cur_pos, cur_pos + S, device=h.device)
    for s in range(S):
        index_k[s, :PE_DIM] = rope(index_k[s, :PE_DIM], cos[positions[s]], sin[positions[s]])
        index_q[s, :, :PE_DIM] = rope(index_q[s, :, :PE_DIM], cos[positions[s]], sin[positions[s]])
        index_cache[positions[s]] = index_k[s].to(torch.bfloat16)
    index_q = index_q.to(torch.bfloat16).float()
    indices = []
    logits = []
    for s in range(S):
        bound = cur_pos + s + 1
        per_head = torch.relu(index_q[s] @ index_cache[:bound].float().T)
        score = (per_head * score_weights[s, :, None]).sum(0)
        # Stable descending order gives the same tie break as the kernel:
        # equal scores keep the smaller token index first.
        selected = torch.argsort(score, descending=True, stable=True)[: min(topk, bound)].to(torch.int32)
        if bound < topk:
            selected = torch.nn.functional.pad(selected, (0, topk - bound))
        indices.append(selected)
        logits.append(score)
    return torch.stack(indices), index_q, score_weights, index_k, logits


def golden_layer(W: LayerWeights, h, cur_pos: int, kv_cache, pe_cache, indices, cos, sin, allreduce, topk=2048):
    """Combine the shared GLM attention golden with the MonoKernel MoE format."""

    result = _golden_attention(
        W,
        h,
        cur_pos,
        kv_cache,
        pe_cache,
        indices,
        cos,
        sin,
        allreduce,
        topk=topk,
        attention_only=True,
    )
    result.pop("gate", None)
    result.update(golden_moe(W, result["a"], allreduce))
    return result


def golden_moe(W: LayerWeights, a, allreduce, mid=None, sel=None, prob=None, xq=None):
    """MoE half of the layer from the post-attention hidden state ``a`` [S, HIDDEN] (bf16).

    ``xq`` [S, HIDDEN] overrides the quant-dequantized activation and
    ``mid``/``sel``/``prob`` ([S, 9, INTER] / [S, 9] / [S, 9]) the down-projection
    inputs, so each stage can be checked from the kernel's own inputs.
    """
    t = W.t
    S = a.shape[0]
    expert_weight = ExpertWeight.MXFP4_BLOCK32 if t["w_ug"].dtype is torch.uint8 else ExpertWeight.FP8_BLOCK128
    out = {k: [] for k in ("sel", "prob", "mid")}
    x2 = rmsnorm(a, t["g_post"])
    scores = torch.sigmoid(bf(x2) @ t["w_r"].float().T)
    xq_ref = quant_dequant(x2)
    xq = xq_ref if xq is None else xq.float()
    y = torch.zeros(S, HIDDEN, device=a.device)
    for s in range(S):
        idx, p = route(scores[s], t["bias"], W.config)
        experts = [SHARED_EXPERT] + idx.tolist()
        weights = [1.0] + p.tolist()
        mids = []
        for e in experts:
            ug = _dequant_expert(t["w_ug"][e], t["s_ug"][e], expert_weight) @ xq[s]
            mids.append(torch.nn.functional.silu(ug[:INTER]) * ug[INTER:])
        out["sel"].append(torch.tensor(experts, device=a.device, dtype=torch.int32))
        out["prob"].append(torch.tensor(weights, device=a.device))
        out["mid"].append(torch.stack(mids))
    for s in range(S):
        experts = out["sel"][s].tolist() if sel is None else sel[s].tolist()
        weights = out["prob"][s].tolist() if prob is None else prob[s].tolist()
        for j, (e, wgt) in enumerate(zip(experts, weights)):
            m = out["mid"][s][j] if mid is None else mid[s, j].float()
            activation = m.to(torch.bfloat16).float() if t["w_dn"].dtype is torch.uint8 else quant_dequant(m)
            y[s] += wgt * (_dequant_expert(t["w_dn"][e], t["s_dn"][e], expert_weight) @ activation)
    x_out = (a.float() + allreduce(y)).to(torch.bfloat16)
    return dict(
        scores=scores,
        xq=xq_ref,
        x_out=x_out,
        sel=torch.stack(out["sel"]),
        prob=torch.stack(out["prob"]),
        mid=torch.stack(out["mid"]),
    )
