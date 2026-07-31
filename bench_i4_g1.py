# Micro-bench: a16wi4 gemm1 stage1 only, BALANCED routing, median-of-7 cold.
import os
import statistics
import sys

import torch

from tests.kernels.test_moe_gemm import (
    _a16wi4_pack_shuffle_w,
    _a16wi4_scale_ng_from_legacy,
    build_routing_buffers,
    flydsl_a16w4_gemm1,
)
from tests.kernels.test_ref import torch_moe_gemm1
from tests.test_common import run_perftest

A16WI4_GROUP = 32


def _cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-8).item()


def _balanced_topk(tokens, E, topk, dev):
    # Mirror aiter AITER_MOE_EXPERT_BALANCE: round-robin contiguous topk block per token.
    score = torch.zeros((tokens, E), dtype=torch.float32, device=dev)
    start = 0
    for t in range(tokens):
        cols = [(start + j) % E for j in range(topk)]
        score[t, cols] = 1.0
        start = (start + topk) % E
    tw, tid = torch.topk(score, topk, dim=1)
    tw = torch.ones_like(tw)  # doweight off in gemm1 by default
    return tid.to(torch.int32), tw.to(torch.float32)


def build_inputs(tokens, model_dim, inter_dim, E, topk, dev, tile_m=32):
    torch.manual_seed(0)
    N_OUT = 2 * inter_dim
    K = model_dim
    G = K // A16WI4_GROUP
    a_bf16 = (torch.randn(tokens, K, device=dev, dtype=torch.bfloat16) * 0.1)
    # int4 weights: values in [-8,7]; per (E, N, G) group scale.
    w_i8 = torch.randint(-8, 8, (E, N_OUT, K), device=dev, dtype=torch.int8)
    scale_ng = (torch.rand(E, N_OUT, G, device=dev, dtype=torch.float32) * 0.02 + 0.01)
    # pack + shuffle W per expert
    w_packed = torch.cat([_a16wi4_pack_shuffle_w(w_i8[e].to(torch.int8)) for e in range(E)]).contiguous()
    w_scale_kernel = _a16wi4_scale_ng_from_legacy(
        scale_ng.permute(0, 2, 1).contiguous(), None, E, N_OUT, K
    )
    tid, tw = _balanced_topk(tokens, E, topk, dev)
    routing = build_routing_buffers(
        topk_ids=tid, topk_weights=tw, experts=E, model_dim=model_dim, tile_m=tile_m
    )
    (sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, sorted_size, blocks) = routing
    inter_sorted = torch.zeros(sorted_size, inter_dim, device=dev, dtype=torch.bfloat16)
    return dict(
        a_bf16=a_bf16,
        w_packed=w_packed,
        w_scale_kernel=w_scale_kernel,
        w_i8=w_i8,
        scale_ng=scale_ng,
        tid=tid,
        tw=tw,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        m_indices=sorted_token_ids,
        inter_sorted=inter_sorted,
        sorted_size=sorted_size,
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        tile_m=tile_m,
    )


def run_g1(inp, tile_n=None, tile_m=None, **kw):
    return flydsl_a16w4_gemm1(
        a_bf16=inp["a_bf16"],
        w1_u8=inp["w_packed"],
        w1_scale_u8=inp["w_scale_kernel"],
        sorted_expert_ids=inp["sorted_expert_ids"],
        cumsum_tensor=inp["cumsum_tensor"],
        m_indices=inp["m_indices"],
        inter_sorted_bf16=inp["inter_sorted"],
        n_tokens=inp["tokens"],
        NE=inp["E"],
        D_HIDDEN=inp["model_dim"],
        D_INTER=inp["inter_dim"],
        topk=inp["topk"],
        tile_m=inp["tile_m"] if tile_m is None else tile_m,
        tile_n=tile_n,
        w_dtype="int4",
        **kw,
    )


def check_cos(inp):
    # Reference: torch gemm1 with groupwise scale [E, G, N]
    out = run_g1(inp)
    # gather sorted -> [tokens, topk, inter]
    sorted_token_ids = inp["m_indices"]
    tokens, topk = inp["tokens"], inp["topk"]
    inter = inp["inter_dim"]
    got = torch.zeros(tokens, topk, inter, device=out.device, dtype=torch.float32)
    st = sorted_token_ids
    tok = (st & 0x00FFFFFF)
    slot = (st >> 24)
    valid = tok < tokens
    idx = torch.nonzero(valid, as_tuple=False).flatten()
    got[tok[idx].long(), slot[idx].long(), :] = out[idx].float()
    # ref
    scale_groups = inp["scale_ng"].permute(0, 2, 1).contiguous()  # [E, G, N]
    ref = torch_moe_gemm1(
        inp["a_bf16"],
        inp["w_i8"].reshape(inp["E"] * 2 * inter, inp["model_dim"]),
        None,
        None,
        inp["tid"],
        inp["tw"],
        inter,
        doweight_stage1=False,
        group_size=A16WI4_GROUP,
        scale_w1_groups=scale_groups,
    )
    return _cos(got, ref)


def median_us(fn, iters=7, inner=20, warmup=5):
    vals = []
    for _ in range(iters):
        _, us = run_perftest(fn, num_iters=inner, num_warmup=warmup)
        vals.append(float(us))
    return statistics.median(vals)


def main():
    dev = "cuda"
    model_dim = int(os.environ.get("MD", 7168))
    inter_dim = int(os.environ.get("ID", 512))
    E = int(os.environ.get("E", 384))
    topk = int(os.environ.get("TOPK", 8))
    toks = [int(x) for x in os.environ.get("TOKS", "1024,2048,4096,8192").split(",")]
    tile_n_env = os.environ.get("TN", "").strip()
    tile_n = int(tile_n_env) if tile_n_env else None
    extra = {}
    if os.environ.get("KWAVE"):
        extra["k_wave"] = int(os.environ["KWAVE"])
    do_cos = os.environ.get("COS", "0") == "1"
    print(f"shape md={model_dim} id={inter_dim} E={E} topk={topk} tile_n={tile_n} extra={extra}")
    for tok in toks:
        inp = build_inputs(tok, model_dim, inter_dim, E, topk, dev)
        cos = check_cos(inp) if do_cos else -1.0
        out = run_g1(inp, tile_n=tile_n, **extra)
        h = float(out.float().abs().sum().item())  # output fingerprint for before/after parity
        us = median_us(lambda: run_g1(inp, tile_n=tile_n, **extra))
        print(f"tok={tok:6d}  us={us:9.3f}  cos={cos:.6f}  fp={h:.6e}")


if __name__ == "__main__":
    main()
