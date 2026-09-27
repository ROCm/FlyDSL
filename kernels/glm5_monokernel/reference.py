# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Weights, layouts and torch goldens for the GLM-5 indexed decode MonoKernel.

The indexed variant covers selection refresh plus MLA and MoE for one rank's
TP shard. All matrices are row-major ``[N, K]`` FP8 E4M3FN with fp32 block
scales ``[ceil(N / 128), K / BK]`` (``y = W @ x``). The golden reduces across
ranks through a caller-supplied ``allreduce`` so a TP8 run checks each rank
against its own shard.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
PE_DIM = 64
NOPE_DIM = 192
V_DIM = 256
QKV_A_ROWS = Q_LORA + KV_LORA + PE_DIM  # 2624
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
N_EXPERTS = 256
TOP_K = 8
MOE_SLOTS = 1 + TOP_K  # slot 0 = shared expert
SHARED_EXPERT = N_EXPERTS  # bank index of the shared expert
INTER = 256  # per-rank intermediate shard (2048 / TP8)
ROUTE_SCALE = 2.5
EPS = 1e-5
SCALE_BM = 128
FP8_MAX = 448.0
SOFTMAX_SCALE = (NOPE_DIM + PE_DIM) ** -0.5


# (name, rows, K, BK) of every FP8 matrix, rows given per local head count H.
def fp8_mats(heads: int):
    return {
        "qkv_a": (QKV_A_ROWS, HIDDEN, 128),
        "q_b": (heads * (NOPE_DIM + PE_DIM), Q_LORA, 128),
        "uk": (heads * KV_LORA, NOPE_DIM, 64),
        "uv": (heads * V_DIM, KV_LORA, 128),
        "o": (HIDDEN, heads * V_DIM, 128),
    }


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


def dequant_expert(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode either FP8/block-128 or native MXFP4/block-32 expert weights."""
    if q.dtype is not torch.uint8:
        return dequant(q, scale, 128)
    codes = q.view(torch.uint8).repeat_interleave(2, dim=-1)
    codes[..., 0::2] &= 0xF
    codes[..., 1::2] >>= 4
    values = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
        dtype=torch.float32,
        device=q.device,
    )
    scale_f32 = (scale.to(torch.int32) << 23).contiguous().view(torch.float32)
    return values[codes.long()] * scale_f32.repeat_interleave(32, dim=-1)


def rope_table(max_seq: int, theta: float = 8.0e6, device="cuda"):
    inv = 1.0 / theta ** (torch.arange(0, PE_DIM, 2, device=device, dtype=torch.float64) / PE_DIM)
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


def golden_layer(W: LayerWeights, h, cur_pos: int, kv_cache, pe_cache, indices, cos, sin, allreduce, topk=2048):
    """One rank's view of the layer. Mutates ``kv_cache``/``pe_cache`` like the kernel.

    Returns a dict of intermediates keyed like the kernel's debug scratch.
    """
    t, H = W.t, W.heads
    S = h.shape[0]
    dq = {n: dequant(t[f"w_{n}"], t[f"s_{n}"], bk) for n, (_, _, bk) in fp8_mats(H).items()}
    # GEMV activations are bf16 (MFMA inputs); weights are exact block-scaled FP8
    x = bf(rmsnorm(h, t["g_in"]))
    qkv = x @ dq["qkv_a"].T
    q_a, kv_a = qkv[:, :Q_LORA], qkv[:, Q_LORA:]
    qb = (bf(rmsnorm(q_a, t["g_q"])) @ dq["q_b"].T).view(S, H, NOPE_DIM + PE_DIM)
    q_nope = qb[..., :NOPE_DIM]
    pos = [cur_pos + s for s in range(S)]
    q_pe = torch.stack([rope(qb[s, :, NOPE_DIM:], cos[pos[s]], sin[pos[s]]) for s in range(S)])
    q_lat = torch.einsum("hkd,shd->shk", dq["uk"].view(H, KV_LORA, NOPE_DIM), bf(q_nope))
    for s in range(S):
        kv_cache[pos[s]] = rmsnorm(kv_a[s, :KV_LORA], t["g_kv"]).to(torch.bfloat16)
        pe_cache[pos[s]] = rope(kv_a[s, KV_LORA:], cos[pos[s]], sin[pos[s]]).to(torch.bfloat16)
    kvf, pef = kv_cache.float(), pe_cache.float()
    o_lat = torch.empty(S, H, KV_LORA, device=h.device)
    for s in range(S):
        kv_len = pos[s] + 1
        keys = indices[s].long() if kv_len > topk else torch.arange(kv_len, device=h.device)
        sc = (bf(q_lat[s]) @ kvf[keys].T + bf(q_pe[s]) @ pef[keys].T) * SOFTMAX_SCALE
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
    o = torch.einsum("hvk,shk->shv", dq["uv"].view(H, V_DIM, KV_LORA), bf(o_lat)).reshape(S, H * V_DIM)
    a = (h.float() + allreduce(bf(o) @ dq["o"].T)).to(torch.bfloat16)
    moe = golden_moe(W, a, allreduce)
    res = dict(q_a=q_a, kv_a=kv_a, q_nope=q_nope, q_pe=q_pe, q_lat=q_lat, o=o, a=a)
    res.update(moe)
    return res


def golden_moe(W: LayerWeights, a, allreduce, mid=None, sel=None, prob=None, xq=None):
    """MoE half of the layer from the post-attention hidden state ``a`` [S, HIDDEN] (bf16).

    ``xq`` [S, HIDDEN] overrides the quant-dequantized activation and
    ``mid``/``sel``/``prob`` ([S, 9, INTER] / [S, 9] / [S, 9]) the down-projection
    inputs, so each stage can be checked from the kernel's own inputs.
    """
    t = W.t
    S = a.shape[0]
    out = {k: [] for k in ("sel", "prob", "mid")}
    x2 = rmsnorm(a, t["g_post"])
    scores = torch.sigmoid(bf(x2) @ t["w_r"].float().T)
    xq_ref = quant_dequant(x2)
    xq = xq_ref if xq is None else xq.float()
    y = torch.zeros(S, HIDDEN, device=a.device)
    for s in range(S):
        idx, p = route(scores[s], t["bias"])
        experts = [SHARED_EXPERT] + idx.tolist()
        weights = [1.0] + p.tolist()
        mids = []
        for e, wgt in zip(experts, weights):
            ug = dequant_expert(t["w_ug"][e], t["s_ug"][e]) @ xq[s]
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
            y[s] += wgt * (dequant_expert(t["w_dn"][e], t["s_dn"][e]) @ activation)
    x_out = (a.float() + allreduce(y)).to(torch.bfloat16)
    return dict(
        scores=scores,
        xq=xq_ref,
        x_out=x_out,
        sel=torch.stack(out["sel"]),
        prob=torch.stack(out["prob"]),
        mid=torch.stack(out["mid"]),
    )
