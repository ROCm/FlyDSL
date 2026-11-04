# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Independent torch golden of the V4.1 mono stages (CPU or GPU, fp32 math).

Each function restates a stage's formula from the model semantics the MI350
mono kernels reproduce (aiter's ``mhc_fused_post_pre_delayed_rmsnorm`` seam,
ATOM's ``quantize_fp8``, the Triton dual norm and ``_mxfp8_quant_op``, GPT-J
RoPE, group-32 FP8 QAT, group32 FP8 linears). It shares no code with the
kernels. Quantizer outputs (codes, FP8 bytes) are exact; sums use torch's order,
so a kernel's fp32 reductions agree to rounding.

Tensor conventions (logical, unpacked): FP8 weights (N, K) ``float8_e4m3fn`` with
(N / 32, K / 32) E8M0 codes as uint8; activations bf16 / fp32 as noted.
"""

import math

import torch

from kernels.monokernel.dsv41.dims import HC, HEAD_DIM, HIDDEN, KV_DIM, Q_RANK

F32, BF16 = torch.float32, torch.bfloat16
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
EPS = 1e-20
HC_EPS = 1e-6
SINKHORN = 20
ROPE = 64
MIX = HC * (HC + 2)
K_MIX = HC * HIDDEN


def bf16(x):
    return x.to(BF16).to(F32)


def fma(a, b, c):
    """fp32 fma: the exact product and sum rounded once (via fp64; exact unless the
    fp64 sum itself rounds, which only differs from a true fma in ties)."""
    return (a.double() * b.double() + c.double()).to(F32)


def pow2(code):
    return torch.ldexp(torch.ones_like(code, dtype=F32), code.to(torch.int32) - 127)


def code_ceil(v):
    """Biased exponent of the smallest power of two >= v, clamped to [1, 254]."""
    bits = v.to(F32).view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return e.clamp(1, 254)


def code_round(amax):
    p2 = (amax.to(F32).view(torch.int32) + 0x200000) & -8388608
    return (((p2 >> 23) & 255) - 127 - 8).clamp(-127, 127) + 127


def fp8_rne(x):
    """f32 (in range) -> E4M3 -> f32, round to nearest even."""
    return x.to(FP8).to(F32)


def dequant_fp8_g32(w, s):
    """(N, K) E4M3 with (N / 32, K / 32) E8M0 codes -> fp32."""
    n, k = w.shape
    scale = pow2(s.to(torch.int32)).repeat_interleave(32, 0).repeat_interleave(32, 1)
    return w.to(F32) * scale[:n, :k]


def quantize_fp8_g32_ceil(y, floor):
    """ATOM ``quantize_fp8`` / rmsnorm_quant MXFP8: per 32, code_ceil(max(amax,
    floor) / 448), bytes fp8(clamp(y / 2^code)). -> (E4M3 f32 values, codes)."""
    g = y.reshape(*y.shape[:-1], -1, 32)
    amax = g.abs().amax(-1).clamp_min(floor)
    code = code_ceil(amax / FP8_MAX)
    step = pow2(code).unsqueeze(-1)
    q = fp8_rne((g / step).clamp(-FP8_MAX, FP8_MAX))
    return q.reshape(y.shape), code


def quantize_mxfp8_round(y):
    """``_mxfp8_quant_op``: per 32, code_round(amax), bytes fp8(y 2^(127 - code))."""
    g = y.reshape(*y.shape[:-1], -1, 32)
    code = code_round(g.abs().amax(-1))
    mul = pow2(254 - code).unsqueeze(-1)
    return fp8_rne(g * mul).reshape(y.shape), code


def mx_dequant(q, code):
    return (q.reshape(*q.shape[:-1], -1, 32) * pow2(code).unsqueeze(-1)).reshape(q.shape)


def rope_tail(v, cos, sin, pos, dim, rope=ROPE, round_each=False):
    """GPT-J RoPE of the last ``rope`` dims of each ``dim``-wide head of v (T, H*dim):
    pairs (2i, 2i + 1) with table entry pos * rope/2 + i."""
    t = v.shape[0]
    x = v.reshape(t, -1, dim).clone()
    tail = x[..., dim - rope :]
    a, b = tail[..., 0::2], tail[..., 1::2]
    cs = cos.reshape(-1)[(pos[:, None] * (rope // 2) + torch.arange(rope // 2)[None])].to(F32)[:, None]
    sn = sin.reshape(-1)[(pos[:, None] * (rope // 2) + torch.arange(rope // 2)[None])].to(F32)[:, None]
    e0, e1 = a * cs - b * sn, a * sn + b * cs
    if round_each:
        e0, e1 = bf16(e0), bf16(e1)
    tail[..., 0::2], tail[..., 1::2] = e0, e1
    return x.reshape(t, -1)


# ------------------------------------------------------------------ K1 stages
def seam_fold(res_in, comb_in, post_in, pend):
    """R_new (T, HC, H) bf16 values: ((post x + c0 r0) + c1 r1) + c2 r2, products
    rounded, then fma(c3, r3, .), rounded to bf16."""
    r = res_in.to(F32)
    cm = comb_in.to(F32)  # (T, HC from, HC to)
    x = pend.to(F32)[:, None, :]
    v = post_in.to(F32)[:, :, None] * x + r[:, 0:1, :] * cm[:, 0, :, None]
    for h in range(1, HC - 1):
        v = v + cm[:, h, :, None] * r[:, h : h + 1, :]
    return bf16(fma(cm[:, HC - 1, :, None].expand_as(v), r[:, HC - 1 : HC, :].expand_as(v), v))


def seam_collapse(rnew, pre_in):
    """The layer input: sum_h pre[h] R_new[h], products rounded, in order -> bf16."""
    acc = torch.zeros_like(rnew[:, 0])
    for h in range(HC):
        acc = acc + pre_in.to(F32)[:, h : h + 1] * rnew[:, h]
    return bf16(acc)


def seam_mix(rnew, hc_fn):
    """(mixes (T, 24), sums of squares (T,)): fn as bf16 hi + bf16 lo."""
    fn = hc_fn.to(F32)
    hi = bf16(fn)
    fn_eff = hi + bf16(fn - hi)
    flat = rnew.reshape(rnew.shape[0], -1)
    return flat @ fn_eff.T, (flat * flat).sum(-1)


def seam_gate(mix, sq, hc_scale, hc_base):
    """-> (pre (T, 4), post (T, 4), comb (T, 4, 4))."""
    sc = hc_scale.to(F32)
    base = hc_base.to(F32)
    v = mix * torch.rsqrt(sq / K_MIX + EPS)[:, None]
    pre = torch.sigmoid(v[:, :HC] * sc[0] + base[:HC]) + HC_EPS
    post = torch.sigmoid(v[:, HC : 2 * HC] * sc[1] + base[HC : 2 * HC]) * 2.0
    cv = (v[:, 2 * HC :] * sc[2] + base[2 * HC :]).reshape(-1, HC, HC)
    cv = torch.exp(cv - cv.amax(-1, keepdim=True))
    cv = cv / cv.sum(-1, keepdim=True) + HC_EPS
    cv = cv / (cv.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        cv = cv / (cv.sum(-1, keepdim=True) + HC_EPS)
        cv = cv / (cv.sum(-2, keepdim=True) + HC_EPS)
    return pre, post, cv


def attn_norm_quant(lin, attn_w):
    """RMSNorm -> bf16 -> quantize_fp8 (floor 1e-4). -> (bf16 row, E4M3 values, codes)."""
    x = lin.to(F32)
    r = torch.rsqrt((x * x).sum(-1, keepdim=True) / HIDDEN + EPS)
    y = bf16((x * r) * attn_w.to(F32))
    q, code = quantize_fp8_g32_ceil(y, 1e-4)
    return y, q, code


def fp8_linear(xq, xcode, w, ws):
    """MXFP8 activations (values, codes) x group32 FP8 weight -> fp32."""
    return mx_dequant(xq, xcode) @ dequant_fp8_g32(w, ws).T


def qkv_post(qkv, qn_w, kvn_w, cos, sin, pos):
    """q_norm -> MXFP8 (code_round) and kv_norm -> RoPE -> bf16 -> FP8 QAT ->
    the ring row. -> (q E4M3 values, q codes, ring row bf16 values)."""
    xq, xk = qkv[:, :Q_RANK], qkv[:, Q_RANK:]
    nq = torch.rsqrt((xq * xq).sum(-1, keepdim=True) / Q_RANK + EPS)
    nk = torch.rsqrt((xk * xk).sum(-1, keepdim=True) / KV_DIM + EPS)
    q8, qcode = quantize_mxfp8_round((xq * nq) * qn_w.to(F32))
    kn = bf16((xk * nk) * kvn_w.to(F32))
    e = rope_tail(kn, cos, sin, pos, KV_DIM, round_each=True)
    g = e.reshape(e.shape[0], -1, 32)
    step = pow2(code_ceil(g.abs().amax(-1).clamp_min(1e-4) / FP8_MAX)).unsqueeze(-1)
    ring = bf16(fp8_rne((g / step).clamp(-FP8_MAX, FP8_MAX)) * step).reshape(e.shape)
    return q8, qcode, ring


IDX_HEADS, IDX_DIM = 32, 128
IDX_WEIGHTS_SCALE = IDX_DIM**-0.5 * IDX_HEADS**-0.5
FP4_MIN_AMAX = 6.0 * 2.0**-126
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def indexer_query(w, normed, q8, qcode, cos, sin, pos):
    """K1's indexer projections: (iw (T, 32) bf16 values, iq (T, 4096) bf16 values)."""
    iw = bf16(normed @ w["iw_w"].to(F32).T)
    iq = bf16(fp8_linear(q8, qcode, w["iq_w"], w["iq_ws"]))
    iq = bf16(rope_tail(iq, cos, sin, pos, IDX_DIM))
    return iw, iq


def iquant_fp8(iq, iw):
    """``dynamic_per_token_scaled_quant`` a (token, head) row: -> (E4M3 values
    (T, 32, 128), scales (T, 32), head weights (T, 32))."""
    x = iq.reshape(iq.shape[0], IDX_HEADS, IDX_DIM)
    amax = x.abs().amax(-1).clamp_min(1e-10)
    scale = amax * torch.tensor(1.0 / FP8_MAX, dtype=F32)
    inv = 1.0 / scale
    q = fp8_rne((x * inv[..., None]).clamp(-FP8_MAX, FP8_MAX))
    return q, scale, (iw * scale) * IDX_WEIGHTS_SCALE


def e2m1_code(v):
    """``quantize_fp4``'s E2M1 nibble (value over its scale): nearest with its
    asymmetric midpoints (ties at 0.75, 1.75, 3.5 up; at 0.25, 1.25, 2.5, 5 down)."""
    m = v.abs()
    code = torch.zeros_like(m, dtype=torch.int32)
    for edge, value, inclusive in ((0.25, 1, False), (0.75, 2, True), (1.25, 3, False), (1.75, 4, True),
                                   (2.5, 5, False), (3.5, 6, True), (5.0, 7, False)):  # fmt: skip
        past = (m >= edge) if inclusive else (m > edge)
        code = torch.where(past, torch.full_like(code, value), code)
    return code | ((v.view(torch.int32) >> 31) & 1) << 3


def iquant_fp4(iq, iw):
    """``quantize_fp4`` a (token, head) row of 128, group 32: -> (nibbles (T, 32,
    128) int32, codes (T, 32, 4) int32, head weights (T, 32))."""
    x = iq.reshape(iq.shape[0], IDX_HEADS, 4, 32)
    amax = x.abs().amax(-1).clamp_min(FP4_MIN_AMAX)
    bits = (amax * torch.tensor(1.0 / 6.0, dtype=F32)).view(torch.int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    inv = torch.ldexp(torch.ones_like(amax), 127 - code)
    nib = e2m1_code((x * inv[..., None]).contiguous())
    return nib.reshape(iq.shape[0], IDX_HEADS, IDX_DIM), code, iw * IDX_WEIGHTS_SCALE


def split_attention(q, keys, sink, qk_scale, bk=16):
    """The decode attention of one token: q (H, 512) bf16 values, keys (n, 512)
    bf16 values (the token's rows in key order), sink (H,), qk_scale (log2 domain).
    Splits of ``bk`` keys: m_s, p = exp2(s - m_s), l_s = sum p, acc_s = bf16(p) @ V;
    combined by alpha_s = exp2(m_s - m_max); the sink folded in. -> (H, 512) bf16."""
    q, keys = q.to(F32), keys.to(F32)
    hn = q.shape[0]
    n = keys.shape[0]
    ms, ls, accs = [], [], []
    for s0 in range(0, max(n, 1), bk):
        k = keys[s0 : s0 + bk]
        if k.shape[0] == 0:
            ms.append(torch.full((hn,), -3.4028234663852886e38))
            ls.append(torch.zeros(hn))
            accs.append(torch.zeros(hn, keys.shape[1] if n else 512))
            continue
        sc = (q @ k.T) * qk_scale  # (H, keys)
        m = sc.amax(-1)
        p = torch.exp2(sc - m[:, None])
        ms.append(m)
        ls.append(p.sum(-1))
        accs.append(bf16(p) @ k)
    m = torch.stack(ms)  # (splits, H)
    m_max = m.amax(0)
    alpha = torch.exp2(m - m_max)
    l_comb = (torch.stack(ls) * alpha).sum(0)
    acc = (torch.stack(accs) * alpha[..., None]).sum(0)
    log2e = 1.4426950408889634
    m_final = torch.maximum(m_max, sink * log2e)
    alpha_kv = torch.exp2(m_max - m_final)
    alpha_sink = torch.exp2(sink * log2e - m_final)
    l_final = l_comb * alpha_kv + alpha_sink
    o = acc * alpha_kv[:, None] / l_final.clamp_min(1e-30)[:, None]
    return bf16(torch.where(l_final[:, None] > 0, o, torch.zeros_like(o)))


def attn_pre(w, x, tp):
    """The K1 outputs from the logical layer weights ``w`` and inputs ``x`` (dicts
    keyed as the kernel's arguments; ``pos`` (T,) positions)."""
    rnew = seam_fold(x["res_in"], x["comb_in"], x["post_in"], x["pend"]) if x.get("fold", True) else x["res_in"].to(F32)
    lin = seam_collapse(rnew, x["pre_in"])
    mix, sq = seam_mix(rnew, w["hc_fn"])
    pre, post, comb = seam_gate(mix, sq, w["hc_scale"], w["hc_base"])
    normed, x8, x8c = attn_norm_quant(lin, w["attn_w"])
    qkv = bf16(fp8_linear(x8, x8c, w["wqkv"], w["wqkv_s"]))
    q8, qcode, ring = qkv_post(qkv, w["qn_w"], w["kvn_w"], x["cos"], x["sin"], x["pos"])
    qb = bf16(fp8_linear(q8, qcode, w["wqb"], w["wqb_s"]))
    q = bf16(rope_tail(qb, x["cos"], x["sin"], x["pos"], HEAD_DIM))
    out = {
        "res_out": rnew, "lin": lin, "pre_out": pre, "post_out": post, "comb_out": comb, "normed": normed,
        "x8": x8, "x8s": x8c, "qkv": qkv, "qx8": q8, "qx8s": qcode, "ring": ring, "q_out": q,
    }  # fmt: skip
    if "iq_w" in w:
        iw, iq = indexer_query(w, normed, q8, qcode, x["cos"], x["sin"], x["pos"])
        out.update(iw=iw, iq=iq, iquant8=iquant_fp8(iq, iw), iquant4=iquant_fp4(iq, iw))
    return out


# ------------------------------------------------------------------ K2a stages
def inverse_rope_quant(o, cos, sin, pos):
    """aiter ``inverse_rope_group_quant``: the attention output (T, heads * 512)
    with each head's tail rotated back (fp32), group-32 FP8 (amax floor 1e-8, the
    ceil code). -> (E4M3 values, codes)."""
    return quantize_fp8_g32_ceil(rope_tail(o, cos, -sin.to(F32), pos, HEAD_DIM), 1e-8)


def wo_a(xq, xcode, w, ws, groups):
    """The grouped wo_a: group g's rows read group g's slice of the attention
    output. -> bf16 values (T, o_rows)."""
    xs = mx_dequant(xq, xcode).chunk(groups, -1)
    ws_ = dequant_fp8_g32(w, ws).chunk(groups, 0)
    return bf16(torch.cat([x @ wg.T for x, wg in zip(xs, ws_)], -1))


def ffn_seam(w, x, pend):
    """K2a's tail: the FFN seam over the attention output ``pend`` (T, HIDDEN): fold,
    collapse, the FFN gates and ffn_norm -> the MoE input (bf16 row)."""
    rnew = seam_fold(x["res_in"], x["comb_in"], x["post_in"], pend)
    lin = seam_collapse(rnew, x["pre_in"])
    mix, sq = seam_mix(rnew, w["ffn_hc_fn"])
    pre, post, comb = seam_gate(mix, sq, w["ffn_hc_scale"], w["ffn_hc_base"])
    normed, _, _ = attn_norm_quant(lin, w["ffn_w"])
    return {"res_out": rnew, "lin": lin, "pre_out": pre, "post_out": post, "comb_out": comb, "normed": normed}


def attn_post(w, x, attn, tp, rank, partials):
    """The K2a outputs of rank ``rank`` from its heads' attention output ``attn``
    (T, heads, 512) bf16 values and the other ranks' wo_b partials ``partials``
    (TP, T, HIDDEN; row ``rank`` ignored): the rank's own partial replaces it, the
    TP partials are summed in rank order (fp32) -> bf16."""
    from kernels.monokernel.dsv41.dims import Dims

    d = Dims(tp)
    t = attn.shape[0]
    xo, xos = inverse_rope_quant(attn.reshape(t, -1).to(F32), x["cos"], x["sin"], x["pos"])
    y = wo_a(xo, xos, w["woa"], w["woa_s"], d.groups)
    yq, ycode = quantize_fp8_g32_ceil(y, 1e-4)
    own = bf16(fp8_linear(yq, ycode, w["wob"], w["wob_s"]))
    parts = partials.to(F32).clone()
    parts[rank] = own
    acc = parts[0]
    for r in range(1, tp):
        acc = acc + parts[r]
    pend = bf16(acc)
    return {"xo": xo, "xos": xos, "y": y, "x8b": yq, "x8bs": ycode, "partial": own, "pend": pend} | ffn_seam(w, x, pend)


def random_post_layer(tp, seed=2, device="cpu"):
    """Seeded logical K2a weights of one rank."""
    from kernels.monokernel.dsv41.dims import Dims

    d = Dims(tp)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def fp8w(n, k, lo, hi):
        w = (torch.randn(n, k, generator=g) * 0.5).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        s = torch.randint(lo, hi, (n // 32, k // 32), generator=g, dtype=torch.uint8)
        return w.to(device), s.to(device)

    woa, woa_s = fp8w(d.o_rows, d.group_k, 116, 122)
    wob, wob_s = fp8w(HIDDEN, d.o_rows, 118, 124)
    return {
        "woa": woa, "woa_s": woa_s, "wob": wob, "wob_s": wob_s,
        "ffn_hc_fn": (torch.randn(MIX, K_MIX, generator=g) * 0.02).to(device),
        "ffn_hc_scale": torch.tensor([0.6, 0.8, 1.2]).to(device),
        "ffn_hc_base": (torch.randn(MIX, generator=g) * 0.1).to(device),
        "ffn_w": (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(BF16).to(device),
    }  # fmt: skip


# ------------------------------------------------------------------ K2b stages
MOE_SCALE = 1.5  # the routed weights' renormalized sum
SWIGLU_LIMIT = 10.0


def mxfp4_dequant(codes, scales):
    """(..., N, K) E2M1 nibbles (int, sign in bit 3) with (..., N, K / 32) E8M0 ->
    fp32."""
    mag = E2M1.to(codes.device)[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    sc = pow2(scales.to(torch.int32)).repeat_interleave(32, -1)
    return mag * sc[..., : codes.shape[-1]]


def moe_xq(x):
    """K2b's two MXFP8 copies of the FFN input (T, HIDDEN) bf16 values: the routed
    experts' (amax seeded 1e-10) and the shared expert's (floored 1e-4)."""
    return quantize_fp8_g32_ceil(x, 1e-10), quantize_fp8_g32_ceil(x, 1e-4)


def route_topk(logits, bias, topk):
    """aiter ``topk_gating_kernel_opt`` on bf16 logits (T, E): score =
    sqrt(softplus(logit)) (the logit itself past 20), chosen by score + bias,
    weighed by score (NaN -> 0), renormalized to MOE_SCALE. Ties go to the lowest
    expert % 64, then the lowest expert / 64 (the wave64 kernel's lane order).
    -> (ids (T, topk) int64, weights (T, topk) fp32)."""
    x = logits.to(F32)
    sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    score = torch.sqrt(sp)
    orig = torch.nan_to_num(score, nan=0.0)
    vals = torch.nan_to_num(score + bias.to(F32), nan=float("-inf")).cpu()
    orig = orig.cpu()
    e = torch.arange(x.shape[-1])
    # a strict order: value descending, then e % 64, then e // 64 ascending
    rank = (e % 64) * (x.shape[-1] // 64 + 1) + e // 64
    ids = []
    for t in range(x.shape[0]):
        order = sorted(range(x.shape[-1]), key=lambda i: (-float(vals[t, i]), int(rank[i])))
        ids.append(order[:topk])
    ids = torch.tensor(ids, dtype=torch.int64)
    w = torch.gather(orig, 1, ids)
    total = w.sum(-1, keepdim=True)
    return ids, w * (MOE_SCALE / total.clamp_min(1e-20))


def route_herd(logits, bias, topk):
    """aiter ``herd_topk`` (``MoeBuild.herd``): ``route_topk``'s top-(k + 1) by score +
    bias; each expert's count among all tokens' candidates; each token drops its
    least counted candidate (ties: the lower score, then the lower id), keeps the
    other k in candidate order, renormalized to MOE_SCALE.
    -> (ids (T, topk) int64, weights (T, topk) fp32)."""
    ids, _ = route_topk(logits, bias, topk + 1)
    x = logits.to(F32)
    score = torch.nan_to_num(torch.sqrt(torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))), nan=0.0).cpu()
    vals = torch.gather(score, 1, ids)
    pop = torch.bincount(ids.flatten(), minlength=x.shape[-1])
    keep_ids, keep_w = [], []
    for t in range(ids.shape[0]):
        cand = ids[t].tolist()
        drop = min(range(topk + 1), key=lambda j: (int(pop[cand[j]]), float(vals[t, j]), cand[j]))
        kept = [j for j in range(topk + 1) if j != drop]
        w = vals[t, kept]
        keep_ids.append([cand[j] for j in kept])
        keep_w.append(w * (MOE_SCALE / w.sum().clamp_min(1e-20)))
    return torch.tensor(keep_ids, dtype=torch.int64), torch.stack(keep_w)


def silu_mul_shared(g, u):
    """aiter silu_and_mul with a limit on bf16 gate / up: gate min limit (bf16),
    up clamped, x / (1 + exp(-x)) x up -> bf16."""
    g = bf16(torch.clamp(g, max=SWIGLU_LIMIT))
    u = torch.clamp(u, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    return bf16(g * torch.sigmoid(g) * u)


def shared_expert(x8, x8c, w):
    """The shared expert: FP8 gate / up -> bf16 -> silu x up -> quantize_fp8
    (floor 1e-4) -> FP8 down -> bf16. -> (intermediate bf16, output bf16)."""
    gu = fp8_linear(x8, x8c, w["sgu"], w["sgu_s"])
    half = gu.shape[-1] // 2
    mid = silu_mul_shared(bf16(gu[:, :half]), bf16(gu[:, half:]))
    mq, mc = quantize_fp8_g32_ceil(mid, 1e-4)
    return mid, bf16(fp8_linear(mq, mc, w["sw2"], w["sw2_s"]))


def routed_expert_mid(x8r, x8rc, w13, w13_s, inter):
    """An expert's ug for one token: MXFP8 x times the MXFP4 [gate; up] (logical
    rows: gate 0 .. inter, up inter .. 2 inter) -> clamp, silu x up -> bf16 ->
    MXFP8 (amax seeded 1e-10). -> (E4M3 values, codes) of the intermediate."""
    h = mx_dequant(x8r, x8rc) @ mxfp4_dequant(w13, w13_s).T
    g = torch.clamp(h[..., :inter], max=SWIGLU_LIMIT)
    u = torch.clamp(h[..., inter:], -SWIGLU_LIMIT, SWIGLU_LIMIT)
    v = bf16(g * torch.sigmoid(g) * u)
    return quantize_fp8_g32_ceil(v, 1e-10)


def moe_partial(w, x, ids, wts, tp):
    """One rank's K2b partial before the TP sum: the routed experts of each
    token's top-k (its rank's intermediate share), summed in bf16 in top-k
    order, plus the shared expert, -> bf16 (T, HIDDEN). ``ids`` / ``wts``: the
    route (teacher-forced from the kernel's when given)."""
    from kernels.monokernel.dsv41.dims import Dims

    d = Dims(tp)
    (xr, xrc), (xs, xsc) = moe_xq(x)
    _, shared = shared_expert(xs, xsc, w)
    out = torch.zeros_like(shared)
    for t in range(x.shape[0]):
        routed = torch.zeros(x.shape[-1], device=x.device)
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            mq, mc = routed_expert_mid(xr[t : t + 1], xrc[t : t + 1], w["w13"][e], w["w13_s"][e], d.inter)
            v = (mx_dequant(mq, mc) @ mxfp4_dequant(w["w2"][e], w["w2_s"][e]).T)[0]
            routed = bf16(routed + bf16(v * wts[t, k]))
        out[t] = bf16(routed + shared[t])
    return out


def random_router(experts=384, seed=3, device="cpu"):
    """The replicated router (gate weight bf16, bias): its own seed, so every rank
    of a TP group draws the same one without the rank-0 experts."""
    g = torch.Generator(device=device).manual_seed(seed)
    return {
        "gate_w": (torch.randn(experts, HIDDEN, generator=g, device=device) * 0.02).to(BF16),
        "bias": torch.randn(experts, generator=g, device=device) * 0.05,
    }


def _cache_dir():
    import os

    # the containers mount the host's /home/zihuang (root's ~ is not mounted)
    default = "/home/zihuang/dsv41_cache" if os.path.isdir("/home/zihuang") else os.path.expanduser("~/dsv41_cache")
    path = os.environ.get("DSV41_WEIGHT_CACHE", default)
    return path if path and path != "0" else None


def _pack_nibbles(codes):
    return (codes[..., 0::2] & 0xF) | ((codes[..., 1::2] & 0xF) << 4)


def _unpack_nibbles(packed):
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8, device=packed.device)
    out[..., 0::2], out[..., 1::2] = packed & 0xF, packed >> 4
    return out


def random_moe_layer(tp, experts=384, seed=3, device="cpu", router_seed=None):
    """Seeded logical K2b weights of one rank: the router (``random_router``,
    ``router_seed`` default ``seed``), the shared expert's 1/TP (FP8, group-32
    E8M0), every routed expert's 1/TP of its intermediate (MXFP4; the padding to
    ``Dims.inter`` zero).

    Cached on disk (``DSV41_WEIGHT_CACHE``, default ``/home/zihuang/dsv41_cache``;
    "0" turns it off): a rank's routed experts are ~2 GB, the FP4 codes kept two a byte; a
    file is written to a temporary name and renamed, so concurrent ranks never
    read half of one."""
    import os

    router_seed = seed if router_seed is None else router_seed
    cache = _cache_dir()
    name = f"moe_tp{tp}_e{experts}_s{seed}_r{router_seed}_v1.pt"
    if cache and os.path.exists(os.path.join(cache, name)):
        w = torch.load(os.path.join(cache, name), map_location=device)
        for k in ("w13", "w2"):
            w[k] = _unpack_nibbles(w.pop(k + "_p"))
        return w
    w = _generate_moe_layer(tp, experts, seed, device) | random_router(experts, router_seed, device)
    if cache:
        os.makedirs(cache, exist_ok=True)
        stored = {k: v for k, v in w.items() if k not in ("w13", "w2")}
        stored |= {k + "_p": _pack_nibbles(w[k]) for k in ("w13", "w2")}
        tmp = os.path.join(cache, f".{name}.{os.getpid()}")
        torch.save({k: v.cpu() for k, v in stored.items()}, tmp)
        os.replace(tmp, os.path.join(cache, name))
    return w


def _generate_moe_layer(tp, experts, seed, device):
    from kernels.monokernel.dsv41.dims import Dims

    d = Dims(tp)
    # generated on ``device``: the routed experts are ~2 GB a rank
    g = torch.Generator(device=device).manual_seed(seed)

    def fp8w(n, k, lo, hi):
        wq = (torch.randn(n, k, generator=g, device=device) * 0.5).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        return wq, torch.randint(lo, hi, (n // 32, k // 32), generator=g, dtype=torch.uint8, device=device)

    def fp4w(e, n, k, k_real, lo, hi):
        codes = torch.randint(0, 16, (e, n, k), generator=g, dtype=torch.uint8, device=device)
        sc = torch.randint(lo, hi, (e, n, k // 32), generator=g, dtype=torch.uint8, device=device)
        codes[..., k_real:] = 0
        return codes, sc

    sgu, sgu_s = fp8w(2 * d.sh_inter, HIDDEN, 116, 122)
    sw2, sw2_s = fp8w(HIDDEN, d.sh_inter, 118, 124)
    w13, w13_s = fp4w(experts, 2 * d.inter, HIDDEN, HIDDEN, 116, 120)
    # the gate / up rows past the real width: zero weights
    w13[:, d.inter_real : d.inter] = 0
    w13[:, d.inter + d.inter_real :] = 0
    w2, w2_s = fp4w(experts, HIDDEN, d.inter, d.inter_real, 118, 122)
    return {
        "sgu": sgu, "sgu_s": sgu_s, "sw2": sw2, "sw2_s": sw2_s,
        "w13": w13, "w13_s": w13_s, "w2": w2, "w2_s": w2_s,
    }  # fmt: skip


def random_layer(tp, seed=0, device="cpu"):
    """Seeded logical K1 weights of one rank (for tests and fixtures)."""
    from kernels.monokernel.dsv41.dims import Dims

    d = Dims(tp)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def fp8w(n, k):
        w = (torch.randn(n, k, generator=g) * 0.5).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        s = torch.randint(118, 124, (n // 32, k // 32), generator=g, dtype=torch.uint8)
        return w.to(device), s.to(device)

    wqkv, wqkv_s = fp8w(Q_RANK + KV_DIM, HIDDEN)
    wqb, wqb_s = fp8w(d.heads * HEAD_DIM, Q_RANK)
    iq_w, iq_ws = fp8w(IDX_HEADS * IDX_DIM, Q_RANK)
    index = {
        "iq_w": iq_w, "iq_ws": iq_ws,
        "iw_w": (torch.randn(IDX_HEADS, HIDDEN, generator=g) * 0.02).to(BF16).to(device),
    }  # fmt: skip
    return index | {
        "hc_fn": (torch.randn(MIX, K_MIX, generator=g) * 0.02).to(device),
        "hc_scale": torch.tensor([0.5, 0.7, 1.1]).to(device),
        "hc_base": (torch.randn(MIX, generator=g) * 0.1).to(device),
        "attn_w": (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(BF16).to(device),
        "qn_w": (1 + 0.1 * torch.randn(Q_RANK, generator=g)).to(BF16).to(device),
        "kvn_w": (1 + 0.1 * torch.randn(KV_DIM, generator=g)).to(BF16).to(device),
        "wqkv": wqkv, "wqkv_s": wqkv_s, "wqb": wqb, "wqb_s": wqb_s,
    }  # fmt: skip


def random_inputs(tokens, seed=1, max_pos=4096, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    comb = torch.rand(tokens, HC, HC, generator=g)
    inv = 1.0 / torch.arange(1, ROPE // 2 + 1, dtype=F32)
    ang = torch.arange(max_pos, dtype=F32)[:, None] * inv[None] * (2 * math.pi / max_pos)
    return {
        "res_in": torch.randn(tokens, HC, HIDDEN, generator=g).to(BF16).to(device),
        "pend": torch.randn(tokens, HIDDEN, generator=g).to(BF16).to(device),
        "comb_in": (comb / comb.sum(-2, keepdim=True)).to(device),
        "post_in": (torch.rand(tokens, HC, generator=g) * 2).to(device),
        "pre_in": (torch.rand(tokens, HC, generator=g) + 0.1).to(device),
        "cos": torch.cos(ang).to(BF16).to(device),
        "sin": torch.sin(ang).to(BF16).to(device),
        "pos": torch.randint(0, max_pos, (tokens,), generator=g).to(device),
    }  # fmt: skip


# ---------------------------------------------------------------- indexer
FLT_MAX = 3.4028234663852886e38


def index_logits(q8, w, keys8, kscale):
    """``deepgemm_fp8_paged_mqa_logits``' formula: q8 (T, 32, 128) and keys8 (N,
    128) the FP8 values (as fp32), w (T, 32) the head weights (the query scale
    folded in, ``iquant_fp8``), kscale (N,) the keys' scales -> (T, N) fp32
    logits, sum_h relu((q_h . k_n) kscale_n) w_h."""
    dots = torch.einsum("thd,nd->thn", q8.to(F32), keys8.to(F32))
    return (torch.relu(dots * kscale.to(F32)[None, None, :]) * w.to(F32)[:, :, None]).sum(1)


def order_key(x):
    """The u32 order key of fp32 values (int64): -0 below +0, NaN above +inf."""
    b = x.contiguous().to(F32).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(b >= 2**31, (~b) & 0xFFFFFFFF, b | 2**31)


def stable_topk(vals, k):
    """``top_k_per_row_decode(stable=True)`` of one row: the indices of the top k
    of ``vals`` by ``order_key``, ties to the lowest index, ascending, -1 padded
    to k."""
    n = vals.numel()
    if n <= k:
        ids = torch.arange(n)
    else:
        # (key, lowest index first) as one int64: key 2^31 + (2^31 - 1 - i)
        comp = order_key(vals) * 2**31 + (2**31 - 1 - torch.arange(n, dtype=torch.int64))
        ids = torch.topk(comp, k).indices.sort().values
    out = torch.full((k,), -1, dtype=torch.int64)
    out[: ids.numel()] = ids
    return out


def block_maxima(logits, bound):
    """``pick_candidate_blocks``' block scores of one row's first ``bound`` logits:
    an 8-row block's best (NaN ignored), capped at the largest finite, the newest
    block +inf."""
    nb = -(-bound // 8)
    v = torch.full((nb * 8,), float("-inf"))
    v[:bound] = torch.where(torch.isnan(logits[:bound]), torch.tensor(float("-inf")), logits[:bound].to(F32))
    best = v.view(nb, 8).amax(1).clamp(max=FLT_MAX)
    best[nb - 1] = float("inf")
    return best


def index_select(logits, bound, topk=512, topk_blocks=2048, producer=False):
    """One row's selection (``index_score``): its top ``topk`` columns of the first
    ``bound`` logits (every column, in order, when bound <= topk: the score stage
    does not run) and, on a candidate producer, its top ``topk_blocks`` blocks."""
    cols = stable_topk(logits[:bound], topk) if bound > topk else stable_topk(torch.zeros(max(bound, 0)), topk)
    if not producer:
        return cols, None
    if bound > topk:
        blocks = stable_topk(block_maxima(logits, bound), topk_blocks)
    else:
        blocks = stable_topk(torch.zeros(-(-max(bound, 0) // 8)), topk_blocks)
    return cols, blocks
