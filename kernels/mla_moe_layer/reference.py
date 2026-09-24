# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Weights, layouts and the torch golden for the GLM-5 shared/reuse MLA+MoE layer.

One rank's TP shard of one layer. All matrices are row-major ``[N, K]`` FP8
E4M3FN with fp32 block scales ``[ceil(N / 128), K / BK]`` (``y = W @ x``).
The golden reduces across ranks through a caller-supplied ``allreduce`` so a
TP8 run checks each rank against its own shard.
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


def make_weights(rank: int, heads: int = 8, device="cuda", seed: int = 1234) -> LayerWeights:
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
    t["w_r"] = (torch.randn(N_EXPERTS, HIDDEN, generator=rep, device=device) / HIDDEN**0.5 * 4).to(bf)
    t["bias"] = torch.randn(N_EXPERTS, generator=rep, device=device) * 0.1
    ug_q = torch.empty(N_EXPERTS + 1, 2 * INTER, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    ug_s = torch.empty(N_EXPERTS + 1, *scale_shape(2 * INTER, HIDDEN, 128), device=device)
    dn_q = torch.empty(N_EXPERTS + 1, HIDDEN, INTER, dtype=torch.float8_e4m3fn, device=device)
    dn_s = torch.empty(N_EXPERTS + 1, *scale_shape(HIDDEN, INTER, 128), device=device)
    for e in range(N_EXPERTS + 1):
        ug_q[e], ug_s[e] = _rand_fp8(2 * INTER, HIDDEN, 128, shd, device)
        dn_q[e], dn_s[e] = _rand_fp8(HIDDEN, INTER, 128, shd, device)
    t["w_ug"], t["s_ug"], t["w_dn"], t["s_dn"] = ug_q, ug_s, dn_q, dn_s
    return LayerWeights(heads, t)


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


def quant_dequant(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Per-``block`` dynamic FP8 E4M3FN quantization of the last dim, returned dequantized."""
    xb = x.float().reshape(*x.shape[:-1], -1, block)
    amax = xb.abs().amax(-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(x.shape)


def route(scores: torch.Tensor, bias: torch.Tensor):
    """sigmoid scores [E] -> (indices [8], probs [8]); ties go to the lower expert id.

    The selected experts are returned in ascending id (their slot order)."""
    rank = scores + bias
    order = sorted(range(N_EXPERTS), key=lambda e: (-float(rank[e]), e))[:TOP_K]
    idx = torch.tensor(sorted(order), device=scores.device)
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
            ug = dequant(t["w_ug"][e], t["s_ug"][e], 128) @ xq[s]
            mids.append(torch.nn.functional.silu(ug[:INTER]) * ug[INTER:])
        out["sel"].append(torch.tensor(experts, device=a.device, dtype=torch.int32))
        out["prob"].append(torch.tensor(weights, device=a.device))
        out["mid"].append(torch.stack(mids))
    for s in range(S):
        experts = out["sel"][s].tolist() if sel is None else sel[s].tolist()
        weights = out["prob"][s].tolist() if prob is None else prob[s].tolist()
        for j, (e, wgt) in enumerate(zip(experts, weights)):
            m = out["mid"][s][j] if mid is None else mid[s, j].float()
            y[s] += wgt * (dequant(t["w_dn"][e], t["s_dn"][e], 128) @ quant_dequant(m))
    x_out = (a.float() + allreduce(y)).to(torch.bfloat16)
    return dict(
        scores=scores,
        xq=xq_ref,
        x_out=x_out,
        sel=torch.stack(out["sel"]),
        prob=torch.stack(out["prob"]),
        mid=torch.stack(out["mid"]),
    )
