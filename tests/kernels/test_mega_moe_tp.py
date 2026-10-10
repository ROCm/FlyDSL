# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused TP MegaMoE layer (a4w4, MXFP4, gfx950) vs torch: accuracy + perf sweep, feature
cases and SpRsNorm, each ending in a markdown summary table::

    torchrun --nproc_per_node=4 tests/kernels/test_mega_moe_tp.py \\
        --models m3 glm5 -t 256 512 1024 2048

Run with plain ``python3`` it relaunches itself under torchrun on up to 8 GPUs, or
skips without >= 2 gfx950 GPUs. Exits 1 if any case failed. Under pytest
(``-m multi_gpu``) it runs that sweep on 4 GPUs.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.mega_moe_tp.mega_moe_tp_op import MegaMoeTP, MegaMoeTPConfig, mega_moe_tp_supported  # noqa: E402
from kernels.mega_moe_tp.sp_rs_norm import SpRsNorm  # noqa: E402
from tests.kernels.utils.gemm_common_utils import e8m0_shuffle, e8m0_to_f32, f32_to_mxfp4, mxfp4_to_f32  # noqa: E402
from tests.utils import shuffle_weight  # noqa: E402

SUPPORTED_GFX = ("gfx950",)
SEED = 123
# rel_l2 gate vs the torch reference: fp8 / bf16 comm, MXFP8 all-reduce second hop
RTOL = {"fp8": 0.045, "bf16": 0.01, "ag8": 0.05}
GRAPH_CALLS, PERF_ITERS, PERF_WARMUP = 10, 20, 3

MODES = ("ag_rs", "ar")
BF16 = torch.bfloat16
# SiTU-v2 (kimi3) gate / linear betas
SITUV2_BETA, SITUV2_LINEAR_BETA = 4.0, 25.0


@dataclass(frozen=True)
class ModelShape:
    name: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    act: str


MODELS = {
    "glm5": ModelShape("glm5", 6144, 2048, 257, 9, "silu"),
    "dsv3": ModelShape("dsv3", 7168, 2048, 256, 8, "silu"),
    "dsv4": ModelShape("dsv4", 7168, 3072, 384, 6, "silu"),
    "kimi3": ModelShape("kimi3", 3584, 3072, 896, 16, "situv2"),
    "m3": ModelShape("m3", 6144, 3072, 129, 5, "swiglu"),
}


def situ(shape):
    if shape.act == "situv2":
        return SITUV2_BETA, SITUV2_LINEAR_BETA
    return None


def mxfp4_quant(x):
    """MXFP4 per 1x32 along the last dim: E8M0 scale = amax / 6 rounded up to a power of
    two (as the kernel quantizes), elements RNE. Returns (packed uint8, scale uint8)."""
    shape = x.shape
    xb = x.float().reshape(-1, 32)
    e8 = ((((xb.abs().amax(1) / 6.0).view(torch.int32) + 0x7FFFFF) >> 23) & 0xFF).clamp(max=254)
    scale = (e8 << 23).view(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    q = f32_to_mxfp4(xb / scale.view(-1, 1)).view(torch.uint8)
    return q.reshape(*shape[:-1], shape[-1] // 2), e8.to(torch.uint8).reshape(*shape[:-1], shape[-1] // 32)


def mxfp4_dequant(q, scale):
    """f32 of mxfp4_quant's (q, scale), same leading shape."""
    shape = q.shape
    v = mxfp4_to_f32(q.reshape(-1, shape[-1])).reshape(-1, 32)
    v = v * e8m0_to_f32(scale.reshape(-1)).view(-1, 1)
    return v.reshape(*shape[:-1], shape[-1] * 2)


def torch_act(gate, up, act, limit=None):
    if act == "swiglu":
        gate, up = gate.clamp(max=limit or 7.0), up.clamp(-(limit or 7.0), limit or 7.0)
        return gate * torch.sigmoid(1.702 * gate) * (up + 1)
    if act == "situv2":
        b, lb = SITUV2_BETA, SITUV2_LINEAR_BETA
        return b * torch.tanh(gate / b) * torch.sigmoid(gate) * (lb * torch.tanh(up / lb))
    return F.silu(gate) * up


class Ctx:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", self.rank))
        torch.cuda.set_device(local)
        self.device = torch.device("cuda", local)
        dist.init_process_group("nccl", device_id=self.device)

    def all_ok(self, ok: bool) -> bool:
        t = torch.tensor([int(ok)], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def log(self, msg: str):
        if self.rank == 0:
            print(msg, flush=True)


def rel_l2(actual, expected) -> float:
    d = actual.float() - expected.float()
    t = torch.stack([(d * d).sum(), (expected.float() ** 2).sum()])
    dist.all_reduce(t)
    return float((t[0] / t[1]).sqrt()) if t[1] > 0 else float("nan")


def identical_across_ranks(t: torch.Tensor) -> bool:
    ref = t.clone()
    dist.broadcast(ref, src=0)
    ok = torch.tensor([int(torch.equal(ref, t))], device=t.device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


@dataclass
class Weights:
    inter: int
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w1_ref: torch.Tensor
    w1_scale_ref: torch.Tensor
    w2_ref: torch.Tensor
    w2_scale_ref: torch.Tensor


def _quant_experts(experts, rows, cols, magnitude, seed, device, chunk_bytes=512 << 20):
    chunk = max(1, min(experts, chunk_bytes // max(rows * cols * 2, 1)))
    qt = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scale = None
    gen = torch.Generator(device=device)
    for start in range(0, experts, chunk):
        n = min(chunk, experts - start)
        gen.manual_seed(seed + start)
        w = torch.randn((n, rows, cols), dtype=BF16, device=device, generator=gen)
        q, s = mxfp4_quant(w.mul_(magnitude))
        qt[start : start + n] = q
        if scale is None:
            scale = torch.empty((experts * rows, s.shape[-1]), dtype=s.dtype, device=device)
        scale[start * rows : (start + n) * rows] = s.view(n * rows, -1)
    return qt, scale


def build_weights(shape: ModelShape, ctx: Ctx, seed: int) -> Weights:
    inter = shape.inter_dim // ctx.world
    H, E = shape.model_dim, shape.experts
    base = seed + 1_000_000 * ctx.rank
    w1, w1s = _quant_experts(E, 2 * inter, H, H**-0.25, base, ctx.device)
    w2, w2s = _quant_experts(E, H, inter, inter**-0.25, base + 7, ctx.device)
    return Weights(
        inter,
        shuffle_weight(w1, layout=(16, 16)),
        e8m0_shuffle(w1s),
        shuffle_weight(w2, layout=(16, 16)),
        e8m0_shuffle(w2s),
        w1,
        w1s,
        w2,
        w2s,
    )


def route(rows, shape: ModelShape, kind: str, gen: torch.Generator, device):
    E, K = shape.experts, shape.topk
    if kind == "balanced":
        start = int(torch.randint(0, E, (1,), device=device, generator=gen))
        ids = (start + torch.arange(rows * K, device=device)) % E
        score = torch.full((rows, E), -1e4, device=device)
        score.scatter_(1, ids.view(rows, K), 0.0)
        score += 1e-3 * torch.randn((rows, E), device=device, generator=gen)
    else:
        score = torch.randn((rows, E), device=device, generator=gen)
        if kind == "hot":
            score[:, E - K :] += 100.0
        elif kind == "subset":
            # ~55% of the experts: the LB row-block count lands in its mixed split
            score[:, E * 35 // 64 :] = -1e4
    w, ids = torch.softmax(score, dim=-1).topk(K, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return w.float().contiguous(), ids.to(torch.int32).contiguous()


@torch.no_grad()
def torch_partial(shape, wt: Weights, x, w, ids):
    """This rank's partial sum (its inter slice) of the MoE layer: MXFP4 activations and
    weights dequantized to f32, GEMM1 -> activation -> bf16 -> MXFP4 -> GEMM2, weighted."""
    E, K, H, inter = shape.experts, shape.topk, shape.model_dim, wt.inter
    a1 = mxfp4_dequant(*mxfp4_quant(x))
    w1 = wt.w1_ref.view(E, 2 * inter, H // 2)
    s1 = wt.w1_scale_ref.view(E, 2 * inter, H // 32)
    out1 = torch.zeros((x.shape[0], K, 2 * inter), dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        mask = ids == e
        out1[mask] = a1[mask.nonzero()[:, 0]] @ mxfp4_dequant(w1[e], s1[e]).t()
    gate, up = out1.split([inter, inter], dim=-1)
    out1 = torch_act(gate, up, shape.act).to(BF16)
    a2 = mxfp4_dequant(*mxfp4_quant(out1))
    w2 = wt.w2_ref.view(E, H, inter // 2)
    s2 = wt.w2_scale_ref.view(E, H, inter // 32)
    out2 = torch.zeros((x.shape[0], K, H), dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        mask = ids == e
        out2[mask] = a2[mask] @ mxfp4_dequant(w2[e], s2[e]).t()
    return (out2 * w.view(x.shape[0], K, 1)).sum(1).to(BF16).float()


@dataclass
class Case:
    x: torch.Tensor
    w: torch.Tensor
    ids: torch.Tensor
    ref: torch.Tensor
    ids_all: torch.Tensor | None = None


def make_case(shape, wt, ctx, mode, tokens, kind, seed, mask=0.0) -> Case:
    m = tokens // ctx.world
    if mode == "ag_rs":
        gen = torch.Generator(device=ctx.device).manual_seed(seed + 7919 * ctx.rank)
        x = torch.randn((m, shape.model_dim), dtype=BF16, device=ctx.device, generator=gen)
        w, ids = route(m, shape, kind, gen, ctx.device)
    else:
        gen = torch.Generator(device=ctx.device).manual_seed(seed)
        x = torch.randn(
            (tokens, shape.model_dim),
            dtype=BF16,
            device=ctx.device,
            generator=gen,
        )
        w, ids = route(tokens, shape, kind, gen, ctx.device)
        for t in (x, w, ids):
            dist.broadcast(t, src=0)
    w_ref, ids_ref = w, ids
    if mask:
        sel = torch.rand(ids.shape, device=ctx.device, generator=gen)
        if mode != "ag_rs":
            dist.broadcast(sel, src=0)
        bad = sel < mask
        ids = torch.where(sel < mask / 2, -1, torch.where(bad, shape.experts + 3, ids))
        ids = ids.to(torch.int32).contiguous()
        w_ref = torch.where(bad, 0.0, w)
        ids_ref = torch.where(bad, 0, ids_ref).to(torch.int32)
    if mode == "ag_rs":
        parts = []
        for t in (x, w_ref, ids_ref):
            g = [torch.empty_like(t) for _ in range(ctx.world)]
            dist.all_gather(g, t)
            parts.append(torch.cat(g))
        full = torch_partial(shape, wt, *parts)
        dist.all_reduce(full)
        ref = full[ctx.rank * m : (ctx.rank + 1) * m]
        ids_all = parts[2]
    else:
        ref = torch_partial(shape, wt, x, w_ref, ids_ref)
        dist.all_reduce(ref)
        ids_all = ids_ref
    return Case(x.contiguous(), w, ids, ref, ids_all)


class CaseFailure(AssertionError):
    pass


def new_layer(shape, wt, ctx, mode, max_local_tokens, comm_dtype="fp8", ar_gather="bf16", **kw):
    beta = situ(shape)
    cfg = MegaMoeTPConfig(
        rank=ctx.rank,
        world_size=ctx.world,
        model_dim=shape.model_dim,
        inter_dim=wt.inter,
        experts=shape.experts,
        topk=shape.topk,
        max_local_tokens=max_local_tokens,
        activation=shape.act,
        beta=beta[0] if beta else None,
        linear_beta=beta[1] if beta else None,
        comm_mode=mode,
        comm_dtype=comm_dtype,
        ar_gather=ar_gather,
        **kw,
    )
    return MegaMoeTP(
        cfg,
        w1=wt.w1,
        w1_scale=wt.w1_scale,
        w2=wt.w2,
        w2_scale=wt.w2_scale,
        device=ctx.device,
    )


def call(layer, c: Case, out=None):
    layer.clear_errors()
    y = layer(c.x, c.w, c.ids, out=out)
    torch.cuda.synchronize()
    flags = torch.tensor([layer.poll_errors(), int(torch.isnan(y).any())], device=y.device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    if flags[0]:
        raise CaseFailure(f"watchdog error code (max over ranks) {int(flags[0])}")
    if flags[1]:
        raise CaseFailure("NaN in output")
    return y


def check(y, c: Case, rtol, what="fused vs torch"):
    e = rel_l2(y, c.ref)
    if not e < rtol:
        raise CaseFailure(f"{what}: rel_l2={e:.4f} >= {rtol}")
    return e


def case_varying_m(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    worst = 0.0
    for m in (8, 1, 32, 1, min(64, S.max_local), 8):
        c = make_case(shape, wt, ctx, mode, m * ctx.world, "random", SEED + m)
        worst = max(worst, check(call(layer, c).clone(), c, RTOL["fp8"], f"m={m}"))
    return worst


def case_layers(shape, wt, ctx, S, mode):
    layers = [new_layer(shape, wt, ctx, mode, S.max_local) for _ in range(3)]
    worst = 0.0
    for rnd in range(2):
        for i, layer in enumerate(layers):
            c = make_case(shape, wt, ctx, mode, 8 * ctx.world, "random", 100 * rnd + i)
            worst = max(worst, check(call(layer, c).clone(), c, RTOL["fp8"], f"layer {i}"))
        del layers[:2]
        gc.collect()
        layers += [new_layer(shape, wt, ctx, mode, S.max_local) for _ in range(2)]
    return worst


def case_out(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    tokens = 16 * ctx.world
    c1 = make_case(shape, wt, ctx, mode, tokens, "random", 1)
    c2 = make_case(shape, wt, ctx, mode, tokens, "random", 2)
    out = torch.empty_like(c1.ref, dtype=BF16)
    y = call(layer, c1, out=out)
    if y.data_ptr() != out.data_ptr():
        raise CaseFailure("out= was not returned")
    keep = out.clone()
    call(layer, c2)
    if not ctx.all_ok(torch.equal(out, keep)):
        raise CaseFailure("out= of call N changed by call N+1")
    return check(keep, c1, RTOL["fp8"])


def case_graph(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    cases = [make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", s) for s in (3, 4, 5)]
    eager = [call(layer, c).clone() for c in cases]
    c = Case(cases[0].x.clone(), cases[0].w.clone(), cases[0].ids.clone(), None)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            layer(c.x, c.w, c.ids)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = layer(c.x, c.w, c.ids)
    same = True
    for i in range(12):
        # every replay on new inputs: stale peer data would show
        k = i % len(cases)
        for dst, src in ((c.x, cases[k].x), (c.w, cases[k].w), (c.ids, cases[k].ids)):
            dst.copy_(src)
        g.replay()
        same = same and torch.equal(out, eager[k])
    torch.cuda.synchronize()
    flags = torch.tensor([layer.poll_errors(), int(not same)], device=out.device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    if flags[0]:
        raise CaseFailure(f"watchdog error during graph replay: {int(flags[0])}")
    if flags[1]:
        raise CaseFailure("graph replay != eager")


def case_dynamic(shape, wt, ctx, S, mode, lb):
    # the dynamic schedule up to 256 tokens (lb: the large-batch one from 1 token up, at
    # its smallest row tile, which takes the mixed GEMM2 split); odd local token
    # counts and a narrow routing
    kw = {"lb_min": 1, "lb_mt": 3, "lb_mt_small": 3, "lb_npp": 2, "lb_q": 3}
    layer = new_layer(shape, wt, ctx, mode, S.max_local, **(kw if lb else {"lb_min": 0}))
    worst = 0.0
    tokens = sorted({ctx.world, 3 * ctx.world, 33 * ctx.world, S.max_tokens})
    for t in (t for t in tokens if t <= S.max_local * ctx.world):
        for kind in ("random", "subset"):
            c = make_case(shape, wt, ctx, mode, t, kind, SEED + 3 * t)
            y = call(layer, c).clone()
            worst = max(worst, check(y, c, RTOL["fp8"], f"M={t} {kind}"))
    return worst


def ref_tail(y, res, nw, eps):
    # res_out = y + res; this rank's FP8 rows + per-row scales of GemmaRMSNorm(res_out)
    r = y.float() + res.float()
    xn = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * (nw.float() + 1.0)
    scale = xn.to(BF16).float().abs().amax(-1).clamp_min(1e-10) / 448.0
    return r.to(BF16), xn, scale


def case_tail(shape, wt, ctx, S):
    # fused tail, in place as ATOM calls it: tail=(res, res, w)
    layer = new_layer(shape, wt, ctx, "ag_rs", S.max_local, lb_min=1)
    eng = layer.engine
    t = S.max_tokens
    if not layer.tail_ok(t // ctx.world):
        return "skipped (no tail at this size)"
    c = make_case(shape, wt, ctx, "ag_rs", t, "random", 21)
    gen = torch.Generator(device=ctx.device).manual_seed(77 + ctx.rank)
    res = torch.randn(c.x.shape, generator=gen, device=ctx.device).to(BF16)
    nw = (0.1 * torch.randn((shape.model_dim,), device=ctx.device)).to(BF16)
    dist.broadcast(nw, src=0)
    y0 = call(layer, c).clone()
    r_ref, xn, s_ref = ref_tail(y0, res, nw, eng.tn_eps)
    gather = []
    for t_ in (xn, s_ref):
        g = [torch.empty_like(t_) for _ in range(ctx.world)]
        dist.all_gather(g, t_)
        gather.append(torch.cat(g))
    xn, s_ref = gather
    fp8 = torch.float8_e4m3fn
    q_ref = (xn.to(BF16).float() / s_ref[:, None]).clamp(-448, 448).to(fp8)
    e_ref = float((q_ref.float() * s_ref[:, None] - xn).norm() / xn.norm())
    for bf16 in (False, True):
        r = res.clone()
        outs = layer(c.x, c.w, c.ids, tail=(r, r, nw), bf16=bf16)
        torch.cuda.synchronize()
        y, q, s = outs[:3]
        e_q = float((q.view(fp8).float() * s[:, None] - xn).norm() / xn.norm())
        bad = [
            what
            for what, ok in (
                ("watchdog", not layer.poll_errors()),
                ("y", torch.equal(y, y0)),
                ("res_out", torch.equal(r, r_ref)),
                (f"q rel {e_q:.4f} (fp8 {e_ref:.4f})", e_q < 1.05 * e_ref),
                ("scale", float(((s - s_ref).abs() / s_ref).max()) < 1e-2),
                ("bf16 rows", not bf16 or rel_l2(outs[3], xn) < 0.01),
            )
            if not ok
        ]
        if not ctx.all_ok(not bad):
            raise CaseFailure(f"tail bf16={bf16}: {bad or 'other rank'}")
    return e_q


def case_sp_rs_norm(ctx, S):
    """SpRsNorm (+ the fused router) vs torch, twice back to back on new inputs."""
    H, eps = 6144, 1e-6
    E, K, scale, shared_w = 128, 4, 2.0, 0.5
    op = SpRsNorm(H, S.max_tokens, eps, device=ctx.device, router=(E, K, scale, shared_w))
    gen = torch.Generator(device=ctx.device).manual_seed(7)
    wg = (torch.randn((E, H), generator=gen, device=ctx.device) * H**-0.5 * 3).to(BF16)
    bias = (0.05 * torch.randn((E,), generator=gen, device=ctx.device)).float()
    w = (0.1 * torch.randn((H,), generator=gen, device=ctx.device)).to(BF16)
    worst = 0.0
    routed = S.max_tokens // (16 * ctx.world) * 16 * ctx.world
    for tokens, rt in ((routed, True), (S.max_tokens, False)) * 2:
        m = tokens // ctx.world
        g = torch.Generator(device=ctx.device).manual_seed(1000 + ctx.rank + tokens)
        part = torch.randn((tokens, H), generator=g, device=ctx.device).to(BF16)
        res = torch.randn((tokens, H), generator=g, device=ctx.device).to(BF16)
        ids = torch.empty((m, K + 1), dtype=torch.int32, device=ctx.device)
        tw = torch.empty((m, K + 1), dtype=torch.float32, device=ctx.device)
        rt = (wg, bias, ids, tw) if rt and op.routes(tokens) else None
        out, res_out = op(part, res, w, router=rt)
        torch.cuda.synchronize()
        tot = part.float()
        dist.all_reduce(tot)
        rows = slice(ctx.rank * m, (ctx.rank + 1) * m)
        h = tot[rows] + res[rows].float()
        ref = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (w.float() + 1)
        e = max(rel_l2(res_out[rows], h), rel_l2(out[rows], ref))
        worst = max(worst, e)
        bad, info = op.poll_errors() or not e < 0.01, f"rel_l2={e:.4f}"
        if rt is not None:
            logits = (ref.to(BF16).float() @ wg.float().t()).to(BF16)
            rid = torch.empty((m, K), dtype=torch.int32, device=ctx.device)
            rtw = torch.empty((m, K), dtype=torch.float32, device=ctx.device)
            topk_gating_ref(rtw, rid, logits, bias, scale)
            same = (ids[:, :K].sort(1).values == rid.sort(1).values).all(1)
            mine = torch.gather(tw[:, :K], 1, ids[:, :K].argsort(1))
            want = torch.gather(rtw, 1, rid.argsort(1))
            e_w = ((mine - want).abs() / want.abs().clamp_min(1e-6))[same]
            # bf16 logits: near-ties may pick another expert
            st = torch.tensor([float(same.sum()), m], device=ctx.device)
            dist.all_reduce(st)
            e_w = float(e_w.max()) if e_w.numel() else 0.0
            info += f" same experts {float(st[0] / st[1]):.3f} weight err {e_w:.1e}"
            bad = bad or float(st[0] / st[1]) < 0.9 or e_w > 0.02
            bad = bad or not bool((ids[:, K] == E).all() and (tw[:, K] == shared_w).all())
        if not ctx.all_ok(not bad):
            raise CaseFailure(f"M={tokens} router={rt is not None}: {info}")
    return worst


def topk_gating_ref(tw, ids, logits, bias, scale):
    """Sigmoid scores; top-k of score + bias; weights: the scores renormalized, x scale."""
    score = torch.sigmoid(logits.float())
    idx = (score + bias.float()).topk(ids.shape[1], dim=-1).indices
    w = score.gather(1, idx)
    tw.copy_(w / w.sum(-1, keepdim=True) * scale)
    ids.copy_(idx.to(torch.int32))


def case_masked(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    c = make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", 11, mask=0.2)
    e = check(call(layer, c).clone(), c, RTOL["fp8"], "masked ids")
    c2 = make_case(shape, wt, ctx, mode, 16 * ctx.world, "random", 12)
    check(call(layer, c2).clone(), c2, RTOL["fp8"], "next clean call")
    return e


def case_empty(shape, wt, ctx, S, mode):
    layer = new_layer(shape, wt, ctx, mode, S.max_local)
    c = make_case(shape, wt, ctx, mode, 4 * ctx.world, "random", 13)
    y = call(layer, Case(c.x[:0], c.w[:0], c.ids[:0], c.ref[:0]))
    if tuple(y.shape) != (0, shape.model_dim):
        raise CaseFailure(f"m=0 returned shape {tuple(y.shape)}")
    check(call(layer, c).clone(), c, RTOL["fp8"], "call after m=0")


def expect_raise(fn, what, errors=(ValueError, TypeError)):
    raised = 0
    try:
        fn()
        torch.cuda.synchronize()
    except errors:
        raised = 1
    t = torch.tensor([raised], device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if not t.item():
        raise CaseFailure(f"{what} was accepted (expected a host-side error)")


def case_validation(shape, wt, ctx, S):
    expect_raise(lambda: new_layer(shape, wt, ctx, "ag_rs", 4096), "max_local_tokens=4096")
    layer = new_layer(shape, wt, ctx, "ag_rs", S.max_local)
    c = make_case(shape, wt, ctx, "ag_rs", 8 * ctx.world, "random", 5)
    x, w, ids = c.x, c.w, c.ids
    expect_raise(lambda: layer(x.float(), w, ids), "fp32 hidden states")
    expect_raise(lambda: layer(x[:, : shape.model_dim // 2].contiguous(), w, ids), "hidden size")
    expect_raise(lambda: layer(x, w[:, :-1].contiguous(), ids[:, :-1].contiguous()), "topk")
    expect_raise(lambda: layer(x, w, ids.to(torch.int16)), "int16 topk_ids")
    expect_raise(lambda: layer(x, w, ids.cpu()), "topk_ids on the cpu")
    expect_raise(lambda: layer(x, w, ids, out=torch.empty_like(x[:1])), "out of the wrong shape")


def case_checked(shape, wt, ctx, S):
    os.environ["FLYDSL_MEGAMOE_TP_CHECK"] = "1"
    try:
        layer = new_layer(shape, wt, ctx, "ag_rs", S.max_local)
        m = 16 if ctx.rank == 0 else 8
        c = make_case(shape, wt, ctx, "ag_rs", 16 * ctx.world, "random", 14)
        expect_raise(lambda: layer(c.x[:m], c.w[:m], c.ids[:m]), "ag_rs with uneven m")
        layer = new_layer(shape, wt, ctx, "ar", S.max_local)
        c = make_case(shape, wt, ctx, "ar", 8 * ctx.world, "random", 15)
        ids = c.ids.clone()
        if ctx.rank == 1:
            ids[0, 0] = (ids[0, 0] + 1) % shape.experts
        expect_raise(lambda: layer(c.x, c.w, ids), "ar with routing that differs across ranks")
        check(call(layer, c).clone(), c, RTOL["fp8"], "checked call")
    finally:
        os.environ.pop("FLYDSL_MEGAMOE_TP_CHECK", None)


# Distributed state lives here, out of the bench_* signatures, so the summary tables
# hold only the sweep axes.
@dataclass
class Session:
    ctx: Ctx
    max_local: int
    max_tokens: int
    weights: dict
    layers: dict


S: Session | None = None


def _rank_max(v: float) -> float:
    t = torch.tensor([float(v)], device=S.ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _layer(model: str, mode: str, comm_dtype: str, ar_gather: str):
    key = (model, mode, comm_dtype, ar_gather)
    if key not in S.layers:
        S.layers[key] = new_layer(
            MODELS[model],
            S.weights[model],
            S.ctx,
            mode,
            S.max_local,
            comm_dtype,
            ar_gather,
        )
    return S.layers[key]


def _graph_us(layer, c: Case, out) -> float:
    """us per call of GRAPH_CALLS back-to-back calls replayed in one CUDA graph (the
    way ATOM runs it), the slowest rank's time."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            layer(c.x, c.w, c.ids, out=out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(GRAPH_CALLS):
            layer(c.x, c.w, c.ids, out=out)
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(PERF_WARMUP):
        g.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(PERF_ITERS):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1e3 / PERF_ITERS
    dist.barrier()
    return _rank_max(us / GRAPH_CALLS)


def _work(shape, wt, c: Case, tokens: int) -> tuple[int, int]:
    """Cluster-wide GEMM FLOPs and algorithmic bytes (input, the active experts' MXFP4
    weights, routing, output) of one layer call."""
    world, H, I, K = S.ctx.world, shape.model_dim, wt.inter, shape.topk  # noqa: E741
    flops = 6 * tokens * K * H * I * world
    ids = c.ids_all.flatten()
    active = int(torch.unique(ids[(ids >= 0) & (ids < shape.experts)]).numel())
    per_rank = (
        c.x.numel() * 2
        + active * (2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32))
        + c.ids.numel() * 8
        + c.ref.numel() * 2
    )
    return flops, per_rank * world


def _mismatch(ref, out, rtol, atol) -> float:
    """Fraction of elements outside atol + rtol * |ref|."""
    return float(((out - ref).abs() > atol + rtol * ref.abs()).float().mean())


def bench_mega_moe_tp(model: str, mode: str, tokens: int, route: str, seed: int):
    """MegaMoE TP vs the torch reference (not timed): rel_l2 gate, mismatch fraction err,
    CUDA-graph latency and roofline numbers per candidate."""
    ctx, shape, wt = S.ctx, MODELS[model], S.weights[model]
    c = make_case(shape, wt, ctx, mode, tokens, route, SEED + 1000 * seed + tokens)
    # name -> (comm_dtype, ar_gather, rel_l2 gate)
    candidates = {
        "megamoe_tp": ("fp8", "bf16", RTOL["fp8"]),
        "megamoe_tp bf16comm": ("bf16", "bf16", RTOL["bf16"]),
    }
    if mode == "ar":  # the MXFP8 all-gather second hop exists only for ar
        candidates["megamoe_tp ag8"] = ("fp8", "fp8", RTOL["ag8"])
    flops, nbytes = _work(shape, wt, c, tokens)
    ret = {"model": model, "mode": mode, "tokens": tokens, "route": route, "seed": seed}
    for name, (comm, agather, rtol) in candidates.items():
        layer = _layer(model, mode, comm, agather)
        out = torch.empty_like(c.ref, dtype=BF16)
        y = call(layer, c, out=out).clone()
        e = rel_l2(y, c.ref)
        ref = c.ref.to(torch.float32)
        err = _rank_max(_mismatch(ref, y.to(torch.float32), 0.1, 0.1 * float(ref.pow(2).mean().sqrt())))
        if not e < rtol:
            raise CaseFailure(f"{name}: rel_l2={e:.4f} >= {rtol}")
        if mode == "ar" and not identical_across_ranks(y):
            raise CaseFailure(f"{name}: all-reduce output differs across ranks")
        us = _graph_us(layer, c, out)
        cfg = layer.launch_config(tokens // ctx.world)
        ret[f"{name} cfg"] = f"mt{cfg.mt}{' lb' if cfg.lb else ''}{' ll' if cfg.ll else ''}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} rel_l2"] = e
    return ret


CASES = {
    "varying m": case_varying_m,
    "new / freed layers": case_layers,
    "out=": case_out,
    "cuda graph": case_graph,
    "dynamic": lambda *a: case_dynamic(*a, lb=False),
    "dynamic LB": lambda *a: case_dynamic(*a, lb=True),
    "masked ids": case_masked,
    "m=0": case_empty,
}
# model-level cases (fused tail: ag_rs only; the others build their own layers)
MODEL_CASES = {
    "fused tail": case_tail,
    "host validation": case_validation,
    "cross-rank checks": case_checked,
}


def bench_mega_moe_tp_case(model: str, mode: str, case: str):
    """One feature case (varying m, graph replay on new inputs, LB, masked ids, ...)."""
    shape, wt = MODELS[model], S.weights[model]
    if case in CASES:
        out = CASES[case](shape, wt, S.ctx, S, mode)
    else:
        out = MODEL_CASES[case](shape, wt, S.ctx, S)
    return {
        "model": model,
        "mode": mode,
        "case": case,
        "rel_l2": out if isinstance(out, float) else None,
        "note": out if isinstance(out, str) else "",
    }


def bench_sp_rs_norm(hidden: int, tokens: int):
    """SpRsNorm (+ the fused router) vs torch, twice back to back on new inputs."""
    return {"hidden": hidden, "tokens": tokens, "rel_l2": case_sp_rs_norm(S.ctx, S)}


def _run(fn, *args) -> dict:
    """fn (a bench_* case); a failure on any rank is recorded in the row's status."""
    ok, msg, row = True, "", None
    try:
        row = fn(*args)
    except CaseFailure as exc:
        ok, msg = False, str(exc)
    except Exception as exc:  # noqa: BLE001
        ok, msg = False, f"{type(exc).__name__}: {exc}"
        if S.ctx.rank == 0:
            traceback.print_exc()
    ok = S.ctx.all_ok(ok)
    row = row if row is not None else {"case": fn.__name__, "args": args}
    row["status"] = "PASS" if ok else f"FAIL {msg}".strip()
    return row


def _summary(name: str, rows: list) -> None:
    if rows and S.ctx.rank == 0:
        print(
            f"{name} summary (markdown):\n"
            + pd.DataFrame(rows)
            .pipe(lambda df: df[[c for c in df.columns if c != "status"] + ["status"]])
            .to_markdown(index=False),
            flush=True,
        )


def main() -> int:
    global S
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["m3", "glm5"],
        choices=list(MODELS),
        help="Models.\n    e.g.: --models m3",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=list(MODES),
        choices=list(MODES),
        help="comm_mode.\n    e.g.: --modes ag_rs",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[8, 64, 256],
        help="GLOBAL tokens per forward (multiples of tp).\n    e.g.: -t 256 2048",
    )
    parser.add_argument(
        "-r",
        "--routes",
        nargs="*",
        default=["balanced", "random", "hot"],
        choices=["balanced", "random", "hot", "subset"],
        help="Routing distributions.\n    e.g.: -r random",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1],
        help="Input / routing seeds.\n    e.g.: -s 0",
    )
    args = parser.parse_args()

    if get_rocm_arch() not in SUPPORTED_GFX:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"MegaMoE TP unsupported on {get_rocm_arch()}; skipping", flush=True)
        return 0
    ctx = Ctx()
    bad = [t for t in args.tokens if t <= 0 or t % ctx.world]
    if bad:
        raise ValueError(f"tokens {bad} must be positive multiples of tp={ctx.world}")
    max_tokens = max(args.tokens)
    S = Session(ctx, max(max_tokens // ctx.world, 128), max_tokens, {}, {})

    rows = {"accuracy + perf": [], "feature cases": [], "SpRsNorm": []}
    for model in args.models:
        S.weights = {model: build_weights(MODELS[model], ctx, SEED)}
        for mode, tokens, route, seed in itertools.product(args.modes, args.tokens, args.routes, args.seeds):
            rows["accuracy + perf"].append(_run(bench_mega_moe_tp, model, mode, tokens, route, seed))
        for mode, case in itertools.product(args.modes, CASES):
            rows["feature cases"].append(_run(bench_mega_moe_tp_case, model, mode, case))
        for case in MODEL_CASES:
            if case != "fused tail" or "ag_rs" in args.modes:
                rows["feature cases"].append(_run(bench_mega_moe_tp_case, model, "ag_rs", case))
        S.layers, S.weights = {}, {}
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    rows["SpRsNorm"].append(_run(bench_sp_rs_norm, 6144, max_tokens))

    for name, r in rows.items():
        _summary(f"MegaMoE TP tp{ctx.world} {name}", r)
    failed = [r for rs in rows.values() for r in rs if r["status"] != "PASS"]
    total = sum(len(rs) for rs in rows.values())
    if ctx.rank == 0:
        print(f"MegaMoE TP: {total - len(failed)}/{total} passed", flush=True)
        for r in failed:
            print("FAIL", {k: v for k, v in r.items() if " " not in k}, flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 1 if failed else 0


def _torchrun(nproc: int, args) -> int:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    cmd += [f"--nproc_per_node={nproc}", os.path.abspath(__file__), *args]
    return subprocess.call(cmd)


def _relaunch() -> int:
    """Plain ``python3`` run: torchrun on up to 8 GPUs, or skip."""
    n = torch.cuda.device_count()
    if n < 2 or not mega_moe_tp_supported():
        print("test_mega_moe_tp: skipped (needs >= 2 gfx950 GPUs)", flush=True)
        return 0
    return _torchrun(8 if n >= 8 else 4 if n >= 4 else 2, sys.argv[1:])


@pytest.mark.multi_gpu
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_mega_moe_tp_tp4():
    """MiniMax-M3 and GLM-5 at tp4, 256 .. 2048 global tokens (accuracy, features, perf)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("needs >= 4 GPUs")
    if not mega_moe_tp_supported():
        pytest.skip(f"needs gfx950, got {get_rocm_arch()}")
    assert _torchrun(4, ["--models", "m3", "glm5", "--tokens", "256", "512", "1024", "2048"]) == 0


if __name__ == "__main__":
    sys.exit(main() if "RANK" in os.environ else _relaunch())
