#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoEV2 end-to-end accuracy and performance tests."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

# Guard distributed-only imports so single-GPU CI can collect and skip this test without mori.
try:
    import mori.shmem as ms  # noqa: E402

    from kernels.comm.flydsl_dispatch_combine_intranode_op import (  # noqa: E402
        FlyDSLDispatchCombineConfig,
        FlyDSLDispatchCombineIntraNodeOp,
    )
    from tests.kernels.utils import gemm_common_utils  # noqa: E402  (weight e8m0 shuffle)
    from tests.utils import shuffle_weight  # noqa: E402

    _HARNESS_DEPS_ERROR = None
except Exception as _exc:  # noqa: BLE001
    ms = None
    FlyDSLDispatchCombineConfig = FlyDSLDispatchCombineIntraNodeOp = None
    gemm_common_utils = shuffle_weight = None
    _HARNESS_DEPS_ERROR = f"{type(_exc).__name__}: {_exc}"


NETWORKS = {
    "r1_v3": dict(model_dim=7168, inter_dim=2048, experts=256, topk=8),
    "v4_flash": dict(model_dim=4096, inter_dim=2048, experts=256, topk=6),
    "v4_pro": dict(model_dim=7168, inter_dim=3072, experts=384, topk=6, swiglu_limit=10.0),
}

# batch-size sweeps for --matrix / --full-bs.
CLASSIC_BS = [1, 8, 64, 512, 8192, 32768]
FULL_BS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def _per_1x32_fp4_quant(x):
    shape = x.shape
    rows = x.reshape(-1, 32).float()
    scale_e8m0 = gemm_common_utils.f32_to_e8m0(rows.abs().amax(dim=1) / 4.0)
    scale_f32 = gemm_common_utils.e8m0_to_f32(scale_e8m0)
    quantized = gemm_common_utils.f32_to_mxfp4(rows / scale_f32[:, None])
    return quantized.view(*shape[:-1], -1), scale_e8m0.view(*shape[:-1], shape[-1] // 32).view(torch.uint8)


def _per_1x32_mxfp8_quant(x):
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    shape = x.shape
    rows = x.contiguous().view(-1, 32).float()
    scale_e8m0 = gemm_common_utils.f32_to_e8m0(rows.abs().amax(dim=1).clamp_min(1e-30) / fp8_max)
    scale_f32 = gemm_common_utils.e8m0_to_f32(scale_e8m0).clamp_min(1e-30)
    quantized = (rows / scale_f32[:, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    scales = scale_e8m0.view(*shape[:-1], shape[-1] // 32).view(torch.uint8)
    return quantized.view(shape).contiguous(), scales.contiguous()


# Chained A8W4 accuracy gate for 61 residual layers.
_CHAIN_TOL = 0.10


def _info(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def _setup_dist(rank: int, world_size: int, master_port: int) -> int:
    if "LOCAL_RANK" not in os.environ:
        os.environ.update(
            {
                "LOCAL_RANK": str(rank),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
            }
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=dev)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return local_rank


def _cleanup() -> None:
    try:
        ms.shmem_finalize()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _all_max(dev, val: float) -> float:
    t = torch.tensor([float(val)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _all_mean(dev, val: float) -> float:
    # average across ranks (matches the official EP8 reporting methodology)
    t = torch.tensor([float(val)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / float(dist.get_world_size())


def _all_min_int(dev, val: int) -> int:
    t = torch.tensor([int(val)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def _make_profiler(active_iters: int, prof_warmup: int = 5):
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        schedule=torch.profiler.schedule(wait=1, warmup=prof_warmup, active=active_iters, repeat=1),
    )


def _kernel_table_from_trace(trace_path: str, op_tag: str, active_iters: int, skip_first: int):
    """Aggregate per-kernel GPU us/replay + E2E replay us from a chrome trace (valid window only)."""
    with open(trace_path) as f:
        tr = json.load(f)
    ev = tr["traceEvents"]
    kernel_events = [e for e in ev if e.get("cat") == "kernel"]
    cg_label = f"{op_tag}::cudagraph_replay"
    cg = sorted(
        [e for e in ev if e.get("cat") == "gpu_user_annotation" and cg_label in e.get("name", "")],
        key=lambda e: e["ts"],
    )[-active_iters:]
    cg = cg[skip_first:]
    valid = max(1, len(cg))
    if cg:
        t0 = cg[0]["ts"]
        t1 = cg[-1]["ts"] + cg[-1]["dur"]
        win = [e for e in kernel_events if t0 <= e["ts"] <= t1]
        e2e = sum(e["dur"] for e in cg) / valid
    else:
        win = kernel_events
        e2e = 0.0
    agg: dict = {}
    for e in win:
        n = e.get("name", "?")
        a = agg.setdefault(n, [0, 0.0])
        a[0] += 1
        a[1] += e["dur"]
    rows = sorted([(n, c / valid, tot / valid) for n, (c, tot) in agg.items()], key=lambda r: r[2], reverse=True)
    return rows, e2e, valid


def _profile_body(body, op_tag, args, rank, world, dev, out_dir, meta):
    """Profile CUDAGraph replays and report per-kernel and cross-rank timing."""
    ms.shmem_barrier_all()
    body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()  # eager warmup (jit)
    _cap = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=_cap):
        body()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()

    iters = max(1, int(args.iters))
    prof_warmup = 5
    skip_first = min(5, iters - 1) if iters > 1 else 0
    total_steps = 1 + prof_warmup + iters
    with _make_profiler(active_iters=iters, prof_warmup=prof_warmup) as prof:
        for _ in range(total_steps):
            with record_function(f"{op_tag}::cudagraph_replay"):
                g.replay()
            prof.step()

    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"{op_tag}_rank{rank}_trace.json")
    prof.export_chrome_trace(trace_path)
    rows, e2e, valid = _kernel_table_from_trace(trace_path, op_tag, iters, skip_first)

    # reduce E2E replay across ranks
    loc = torch.tensor([e2e], dtype=torch.float64, device=dev)
    s = loc.clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = loc.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = loc.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    if rank == 0:
        sep = "=" * 80
        print(f"\n{sep}")
        print(
            f"  PROFILE {op_tag}  EP={world}  bs={meta.get('tokens')}  "
            f"net={meta.get('network')}  quant={meta.get('quant')}  ({valid} valid iters)"
        )
        print(
            f"  E2E replay us/iter (avg/min/max across {world} ranks): "
            f"{s.item()/world:.1f} / {mn.item():.1f} / {mx.item():.1f}"
        )
        print(f"  {'kernel (rank0)':<52}{'calls/it':>9}{'gpu us/it':>11}")
        print(f"  {'-'*72}")
        for n, calls, us in rows[:12]:
            nm = n if len(n) <= 50 else n[:47] + "..."
            print(f"  {nm:<52}{calls:>9.2f}{us:>11.2f}")
        print(sep, flush=True)
    return {"e2e_us_avg": s.item() / world}


def _chunked_fp4_quant(x):
    """Row-chunked MX-FP4 quant (identical result; bounds the f32 temp)."""
    n = int(x.shape[1])
    chunk_rows = max(4096, ((2 << 30) // max(1, n * 4 * 8)) // 4096 * 4096)
    if x.ndim != 2 or x.shape[0] <= chunk_rows:
        return _per_1x32_fp4_quant(x)
    m = int(x.shape[0])
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    y = torch.empty((m, n // 2), device=x.device, dtype=fp4_dtype)
    s = torch.empty((m, n // 32), device=x.device, dtype=torch.uint8)
    for st in range(0, m, chunk_rows):
        en = min(st + chunk_rows, m)
        yc, sc = _per_1x32_fp4_quant(x[st:en])
        y[st:en].copy_(yc)
        s[st:en].copy_(sc)
        del yc, sc
        torch.cuda.empty_cache()
    return y, s


def _dequant_mx_to_f32(t_f32, quant_mode):
    """Round-trip a tensor through per-1x32 MX quantization for the accuracy oracle."""
    orig = tuple(t_f32.shape)
    t2d = t_f32.reshape(-1, orig[-1])
    if quant_mode == "fp8":
        q, s = _per_1x32_mxfp8_quant(t2d)  # q: fp8_e4m3fn [., K]; s: e8m0 u8 [., K//32]
        vf = q.float()
    else:
        q, s = _chunked_fp4_quant(t2d)  # q: fp4x2 [., K//2]; s: e8m0 u8 [., K//32]
        vf = gemm_common_utils.mxfp4_to_f32(q)  # [., K] f32 via the E2M1 LUT
    sf = gemm_common_utils.e8m0_to_f32(s).unsqueeze(-1).expand(-1, -1, 32).reshape(t2d.shape)
    return (vf * sf).reshape(orig).to(torch.float32)


def _rmsnorm(x, eps=1e-6):
    """Apply gain-free RMSNorm on the last dimension."""
    xf = x.float()
    n = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return n.to(x.dtype)


def _swiglu(gate, up, limit):
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up


def _calc_diff(x, y):
    """Return one minus FP64 cosine similarity."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def _relative_l2(a, b):
    """Return relative L2 error for NumPy-compatible inputs."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    numerator = float(((a - b) ** 2).sum())
    denominator = float((b**2).sum())
    return (numerator / denominator) ** 0.5 if denominator > 0 else -1.0


def _make_layer_routings(n_layers, tokens, experts, topk, dev, seed, rank):
    """Build deterministic per-layer routing shared by device and reference paths."""
    routings = []
    for lyr in range(n_layers):
        g = torch.Generator(device=dev).manual_seed(seed + 100 * rank + lyr)
        score = torch.rand(tokens, experts, generator=g, device=dev, dtype=torch.float32)
        _, ids = score.topk(topk, dim=-1)
        w = torch.rand(tokens, topk, generator=g, device=dev, dtype=torch.float32)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routings.append((ids.to(torch.int32).contiguous(), w.contiguous()))
    return routings


class RefModel:
    """Pure PyTorch FP32 reference for chained MoE residual layers."""

    def __init__(self, w1_f32, w2_f32, inter_dim, dev, swiglu_limit=0.0, sw1=None, sw2=None):
        self.w1_f32, self.w2_f32 = w1_f32, w2_f32  # full-precision [E, 2I, H], [E, H, I]
        self.inter_dim = inter_dim
        self.swiglu_limit = float(swiglu_limit)
        self.sw1, self.sw2 = sw1, sw2  # optional dense shared experts (None here)
        self._cache = {}

    def _expert(self, g):
        wd = self._cache.get(g)
        if wd is None:
            # quant->dequant to the kernel's exact lossy mxfp4 weights (same as _run_*'s oracle).
            wd = self._cache[g] = (
                _dequant_mx_to_f32(self.w1_f32[g], "fp4"),  # [2I, H]
                _dequant_mx_to_f32(self.w2_f32[g], "fp4"),  # [H, I]
            )
        return wd

    def _ffn(self, x, w1d, w2d):
        gate, up = (x @ w1d.t()).chunk(2, dim=-1)
        return _swiglu(gate, up, self.swiglu_limit) @ w2d.t()

    def _shared(self, x):
        if self.sw1 is None:
            return torch.zeros_like(x)
        acc = torch.zeros_like(x)
        for e in range(self.sw1.shape[0]):
            acc = acc + self._ffn(x, self.sw1[e].float(), self.sw2[e].float())
        return acc

    def layer(self, x, ids, wts):
        """Run one normalized routed FFN layer."""
        xn = _rmsnorm(x)
        out = torch.zeros_like(xn)
        ids_l = ids.long()
        wts_f = wts.float()
        for g in torch.unique(ids_l).tolist():
            sel = ids_l == g
            rows = sel.any(dim=1)
            w = (wts_f * sel).sum(dim=1)
            w1d, w2d = self._expert(int(g))
            out[rows] += w[rows, None] * self._ffn(xn[rows], w1d, w2d)
        return out + self._shared(xn)

    def run(self, x0, routings):
        """Chain N layers with residual: x = x + layer(x). Returns bf16 [ct,H]."""
        x = x0.float()
        for ids, wts in routings:
            x = x + self.layer(x, ids, wts)
        return x.to(torch.bfloat16)


def _prepare(
    dev,
    *,
    quant,
    tokens,
    model_dim,
    inter_dim,
    experts,
    topk,
    seed,
    rank=0,
    world=1,
    keep_ref=False,
    local_experts_only=False,
):
    """Generate inputs and packed weights for A8W4 or A4W4."""
    torch.manual_seed(seed)
    # Scale inputs and weights by model_dim**-0.25 to keep pre-activations O(1).
    init_scale = float(model_dim) ** -0.25
    x_fp32 = torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * init_scale
    weight_experts = experts // world if local_experts_only else experts
    w1_fp32 = torch.randn((weight_experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32)
    w1_fp32.mul_(init_scale)

    if quant not in ("a4w4", "a8w4"):
        raise SystemExit(f"unknown quant {quant!r} (use a8w4|a4w4)")
    a_dtype = "fp4" if quant == "a4w4" else "fp8"

    # weight: MX-FP4 + shuffle for the GEMM (shared across a8w4/a4w4)
    w1_flat = w1_fp32.view(weight_experts * (2 * inter_dim), model_dim)
    w1_fp4, w1_scale_raw = _chunked_fp4_quant(w1_flat)
    w_kernel = shuffle_weight(w1_fp4.view(torch.float4_e2m1fn_x2)).view(torch.uint8).contiguous()
    scale_w1_1d = gemm_common_utils.e8m0_shuffle(w1_scale_raw).view(torch.uint8).contiguous()
    # MegaMoEV2 uses interleaved gate/up while the ATOM baseline uses separated weights.
    w_kernel_gui = (
        gemm_common_utils.shuffle_weight_w4(
            w1_fp4.view(weight_experts, 2 * inter_dim, model_dim // 2), NLane=16, gate_up=True, moe_gemm=True
        )
        .view(torch.uint8)
        .contiguous()
    )
    scale_gui = (
        gemm_common_utils.shuffle_scale_w4(
            w1_scale_raw.view(weight_experts * 2 * inter_dim, model_dim // 32),
            experts_cnt=weight_experts,
            gate_up=True,
        )
        .view(torch.uint8)
        .contiguous()
    )
    w_ref_local = None
    if keep_ref:
        epr = experts // world
        w_ref_local = (
            w1_fp32.contiguous() if local_experts_only else w1_fp32[rank * epr : (rank + 1) * epr].contiguous()
        )
    del w1_fp32, w1_flat, w1_fp4, w1_scale_raw
    torch.cuda.empty_cache()

    # Vary distinct top-k routing by rank.
    torch.manual_seed(seed + 9973 + rank * 101)
    topk_ids = torch.stack([torch.randperm(experts, device=dev)[:topk] for _ in range(tokens)]).to(torch.int32)
    wts = torch.full((tokens, topk), 1.0 / topk, device=dev, dtype=torch.float32)

    return dict(
        w_kernel=w_kernel,
        scale_w1_1d=scale_w1_1d,
        w_kernel_gui=w_kernel_gui,
        scale_gui=scale_gui,
        w_ref_local=w_ref_local,
        local_experts_only=bool(local_experts_only),
        topk_ids=topk_ids,
        wts=wts,
        a_dtype=a_dtype,
        x_bf16=x_fp32.to(torch.bfloat16).contiguous(),
    )


def _run_full_e2e(
    args,
    rank,
    world,
    dev,
    *,
    model_dim,
    inter_dim,
    experts,
    epr,
    topk,
    swiglu_limit,
    run_tokens,
    mtpr,
    a_dtype,
    s1_out,
    w_kernel,
    scale_w1_1d,
    x_bf16,
    topk_ids,
    wts,
    w_kernel_gui=None,
    scale_gui=None,
):
    """Compare MegaMoEV2 with FP8- and BF16-dispatch ATOM pipelines."""

    from kernels.mega_moe import MegaMoEV2
    from kernels.mega_moe.quant import mxfp4_moe_scale_sort, per_1x32_mx_quant
    from kernels.moe.moe_sorting_kernel import moe_sorting_flydsl

    _is_fp4 = s1_out == "fp4"
    max_recv = world * mtpr
    tm, tn1, tk = 32, 128, 256  # ATOM gemm1 tile
    tm2, tn2, tk2 = 32, 128, 256  # ATOM GEMM2 tile
    _agv = (lambda t: t.view(torch.uint8)) if a_dtype == "fp4" else (lambda t: t)

    # Replicate MXFP4 W2 across ranks.
    torch.manual_seed(args.seed + 4242)
    w2_f32 = torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32) * (
        float(inter_dim) ** -0.25
    )
    w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
    _w2sl = slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
    w2_kernel = shuffle_weight(w2_fp4[_w2sl]).view(torch.uint8).contiguous().view(-1)
    w2_scale_1d = gemm_common_utils.e8m0_shuffle(w2_sr[_w2sl]).view(torch.uint8).contiguous().view(-1)

    # Free temporary W2 buffers before allocating the large FP32 oracle weights.
    del w2_fp4, w2_sr
    torch.cuda.empty_cache()
    _init = float(model_dim) ** -0.25
    torch.manual_seed(args.seed)
    _ = torch.randn((run_tokens, model_dim), device=dev, dtype=torch.float32)  # advance RNG like _prepare
    w1_all = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * _init
    w2_all = w2_f32.view(experts, model_dim, inter_dim)

    wc = wts[:run_tokens].contiguous()
    ic = topk_ids[:run_tokens].to(torch.int32).contiguous()
    # production stage-1 input: fp8/fp4 payload + e8m0 scale (FlyDSL MX quant, drop-in for aiter)
    x_q, x_sc = per_1x32_mx_quant(x_bf16[:run_tokens].contiguous(), quant_mode=("fp4" if _is_fp4 else "fp8"))
    x_sc = x_sc.view(torch.uint8)

    def _cg_time(body):
        ms.shmem_barrier_all()
        body()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()  # warmup (jit)
        _cap = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=_cap):
            body()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        _n = max(1, int(args.iters))
        _s = torch.cuda.Event(enable_timing=True)
        _e = torch.cuda.Event(enable_timing=True)
        _s.record()
        for _ in range(_n):
            g.replay()
        _e.record()
        torch.cuda.synchronize()
        return _all_mean(dev, _s.elapsed_time(_e) / _n)

    # Slice expert-major W1 because MegaMoEV2 indexes local experts.
    _wpe = w_kernel.numel() // experts  # per-expert uint8 elems (weight)
    _spe = scale_w1_1d.numel() // experts  # per-expert uint8 elems (scale)
    _w1_arg = w_kernel.reshape(-1)[rank * epr * _wpe : (rank + 1) * epr * _wpe].contiguous()
    _w1s_arg = scale_w1_1d.reshape(-1)[rank * epr * _spe : (rank + 1) * epr * _spe].contiguous()
    # MegaMoEV2 uses INTERLEAVE gate/up for A8W4 and A4W4; ATOM stays SEPARATED.
    assert w_kernel_gui is not None and scale_gui is not None
    _w1_arg_mega = w_kernel_gui.reshape(-1)[rank * epr * _wpe : (rank + 1) * epr * _wpe].contiguous()
    _w1s_arg_mega = scale_gui.reshape(-1)[rank * epr * _spe : (rank + 1) * epr * _spe].contiguous()
    moe = MegaMoEV2(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=args.quant,
        w1=_w1_arg_mega,
        w1_scale=_w1s_arg_mega,
        w2=w2_kernel,
        w2_scale=w2_scale_1d,
        max_tok_per_rank=mtpr,
        stage2_p2p_quant=args.stage2_p2p_quant,
        swiglu_limit=swiglu_limit,
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    skew_ms = float(getattr(args, "rank_skew_ms", 0.0))
    if skew_ms > 0:
        time.sleep(rank * skew_ms / 1000.0)

    _mega_out_holder = {}

    def _mega_body():
        _mega_out_holder["o"] = moe.forward_prequant(x_q, x_sc, wc, ic)

    _mega_body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    out_mega = _mega_out_holder["o"][:run_tokens].float().cpu().numpy().copy()

    cfg_a = FlyDSLDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=model_dim,
        max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        dispatch_dtype=torch.bfloat16,
        combine_dtype=torch.bfloat16,
        scale_dim=0,
        scale_type_size=0,
        enable_std_moe=False,
    )
    dc = FlyDSLDispatchCombineIntraNodeOp(cfg_a)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    # one setup dispatch to fix trc (constant for fixed routing) + populate routing tables.
    dc.total_recv.zero_()
    _bt0, _, _, _oidx0, _ = dc.dispatch(x_bf16[:run_tokens].contiguous(), wc, None, ic)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    trc = max(1, int(dc.total_recv.item()))
    if _all_min_int(dev, trc) <= 0:
        _info(rank, "[full-e2e] some rank got 0 recv; skipping")
        return None

    _max_pad = max_recv * topk + experts * tm
    _max_blocks = (_max_pad + tm - 1) // tm
    _scaleN_pad = ((model_dim // 32 + 7) // 8) * 8
    a_st = torch.empty(_max_pad, dtype=torch.int32, device=dev)
    a_sw = torch.empty(_max_pad, dtype=torch.float32, device=dev)
    a_se = torch.empty(_max_blocks, dtype=torch.int32, device=dev)
    a_se_local = torch.empty(_max_blocks, dtype=torch.int32, device=dev)  # gemm2 wants LOCAL expert ids
    a_nv = torch.zeros(2, dtype=torch.int32, device=dev)
    a_mbuf = torch.empty((max_recv, model_dim), dtype=torch.float16, device=dev)
    a1s = torch.empty(((_max_pad + 31) // 32 * 32, _scaleN_pad), dtype=torch.float8_e8m0fnu, device=dev)
    # Match GEMM2's routing weights in the baseline.
    recv_wts = torch.full((max_recv, topk), 1.0 / topk, device=dev, dtype=torch.float32)
    recv_topk = torch.empty((max_recv, topk), dtype=torch.int32, device=dev)
    _sentinel = torch.full((trc, topk), experts, dtype=torch.int32, device=dev)
    if _is_fp4:
        a2_e = torch.zeros((max_recv * topk, inter_dim // 2), dtype=torch.uint8, device=dev)
    else:
        a2_e = torch.zeros((max_recv * topk, inter_dim), dtype=torch.float8_e4m3fn, device=dev)
    _sbm = max(32, tm)
    _pr = ((_max_blocks * _sbm + 255) // 256) * 256
    _pc = (((inter_dim // 32) + 7) // 8) * 8
    a2s_e = torch.zeros(_pr * _pc + inter_dim, dtype=torch.uint8, device=dev)
    bias_d = torch.empty((0,), device=dev, dtype=torch.float32)
    from kernels.moe.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1, compile_mixed_moe_gemm2

    # ATOM GEMM1 and GEMM2 index local experts to avoid the 4 GiB buffer limit.
    gemm1 = compile_mixed_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=epr,
        topk=topk,
        tile_m=tm,
        tile_n=tn1,
        tile_k=tk,
        doweight_stage1=False,
        a_dtype=a_dtype,
        b_dtype="fp4",
        out_dtype=s1_out,
        act="silu",
        waves_per_eu=int(args.waves_per_eu),
        use_async_copy=bool(args.async_copy),
    )
    # Keep ATOM GEMM2 and combine as separate kernels.
    _g2exe = compile_mixed_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=epr,
        topk=topk,
        tile_m=tm2,
        tile_n=tn2,
        tile_k=tk2,
        doweight_stage2=True,
        a_dtype=s1_out,
        b_dtype="fp4",
        out_dtype="bf16",
        accumulate=True,
        persist_m=-1,
        sort_block_m=_sbm,
    )
    _g2out = torch.zeros(max_recv, model_dim, dtype=torch.bfloat16, device=dev)
    _g2c = {}

    def _run_gemm2_sep():
        _g2out.zero_()
        _ga = (
            _g2out,
            a2_e.view(-1),
            w2_kernel,
            a2s_e,
            w2_scale_1d,
            a_st,
            a_se_local,
            a_sw,
            a_nv,
            bias_d,
            max_recv,
            model_dim,
            inter_dim,
            int(_max_blocks),
            torch.cuda.current_stream(),
        )
        if _g2c.get("c") is None:
            _g2c["c"] = flyc.compile(_g2exe, *_ga)
        else:
            _g2c["c"](*_ga)

    _atom_out_holder = {}

    def _atom_body():
        # Recompute routing each replay to keep sorting and combine synchronized.
        a2_e.zero_()
        dc.total_recv.zero_()
        _bt, _, _, _oidx, _ = dc.dispatch(x_bf16[:run_tokens].contiguous(), wc, None, ic)
        _oi = _oidx[:trc].to(torch.int32)
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
        moe_sorting_flydsl(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc], int(experts), int(tm))
        _a1q, _a1sp = per_1x32_mx_quant(_bt[:trc].contiguous(), quant_mode=("fp4" if _is_fp4 else "fp8"))
        mxfp4_moe_scale_sort(a1s, _a1sp, a_st, a_nv, int(trc), int(model_dim))
        # Convert sorted global expert IDs to the local IDs used by both GEMMs.
        a_se_local.copy_(a_se - rank * epr)
        gemm1(
            a2_e.view(max_recv, topk, a2_e.shape[-1]),
            _agv(_a1q),
            _w1_arg,
            a1s.view(torch.uint8),
            _w1s_arg,
            a_st,
            a_se_local,
            a_sw,
            a_nv,
            bias_d,
            a2s_e,
            fx.Int32(trc),
            fx.Int32(inter_dim * 2),
            fx.Int32(model_dim),
            fx.Int32(int(_max_blocks)),
            stream=fx.Stream(torch.cuda.current_stream()),
        )
        _run_gemm2_sep()  # FlyDSL gemm2 (separate) -> _g2out [max_recv, model_dim]
        _r = dc.combine(_g2out, None, _oidx)  # FlyDSL combine (separate)
        _atom_out_holder["o"] = _r[0] if isinstance(_r, (tuple, list)) else _r

    _atom_body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    _ao = _atom_out_holder["o"]
    out_atom = _ao[:run_tokens].float().cpu().numpy().copy()

    # FP8 dispatch carries E8M0 scales while combine consumes BF16 GEMM2 output.
    _scale_mx_blocks = model_dim // 32
    cfg_fp8 = FlyDSLDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=model_dim,
        max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        # Dispatch is packed MX while combine consumes BF16.
        dispatch_dtype=(torch.float4_e2m1fn_x2 if _is_fp4 else torch.float8_e4m3fn),
        combine_dtype=torch.bfloat16,
        scale_dim=_scale_mx_blocks,
        scale_type_size=1,
        enable_std_moe=False,
    )
    assert cfg_fp8.is_fp4 == _is_fp4
    dcf = FlyDSLDispatchCombineIntraNodeOp(cfg_fp8)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    _atom8_holder = {}

    def _atom_fp8_body():
        # Quantize inside the timed FP8-dispatch pipeline.
        a2_e.zero_()
        dcf.total_recv.zero_()
        _xq, _xsc = per_1x32_mx_quant(x_bf16[:run_tokens].contiguous(), quant_mode=("fp4" if _is_fp4 else "fp8"))
        _xsc = _xsc.view(torch.uint8)
        _rx, _, _rs, _oidx, _ = dcf.dispatch(_xq, wc, _xsc, ic)  # fp8 dispatch (+ e8m0 scale)
        _oi = _oidx[:trc].to(torch.int32)
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
        moe_sorting_flydsl(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc], int(experts), int(tm))
        mxfp4_moe_scale_sort(a1s, _rs[:trc].contiguous(), a_st, a_nv, int(trc), int(model_dim))
        a_se_local.copy_(a_se - rank * epr)
        gemm1(
            a2_e.view(max_recv, topk, a2_e.shape[-1]),
            _agv(_rx[:trc]),
            _w1_arg,
            a1s.view(torch.uint8),
            _w1s_arg,
            a_st,
            a_se_local,
            a_sw,
            a_nv,
            bias_d,
            a2s_e,
            fx.Int32(trc),
            fx.Int32(inter_dim * 2),
            fx.Int32(model_dim),
            fx.Int32(int(_max_blocks)),
            stream=fx.Stream(torch.cuda.current_stream()),
        )
        _run_gemm2_sep()  # FlyDSL gemm2 (separate) -> _g2out
        _r = dcf.combine(_g2out, None, _oidx)  # FlyDSL combine (separate, on fp8 dispatch op)
        _atom8_holder["o"] = _r[0] if isinstance(_r, (tuple, list)) else _r

    _atom_fp8_body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    _a8 = _atom8_holder["o"]
    out_atom8 = _a8[:run_tokens].float().cpu().numpy().copy()

    # Dequantize one expert at a time for the routing-weighted FP32 oracle.
    x32 = x_bf16[:run_tokens].float()
    ids_l = ic[:run_tokens].long()
    wv = wc[:run_tokens].float()
    oracle_w = torch.zeros(run_tokens, model_dim, device=dev, dtype=torch.float32)
    for e in torch.unique(ids_l).tolist():
        sel = ids_l == int(e)  # [T, topk]: which (token, slot) route to expert e
        rows = sel.any(dim=1).nonzero().flatten()
        w_e = (wv * sel).sum(dim=1)[rows]  # per-token routing weight (summed over slots) for e
        w1e = _dequant_mx_to_f32(w1_all[e], "fp4")  # [2*inter_dim, model_dim]
        w2e = _dequant_mx_to_f32(w2_all[e], "fp4")  # [model_dim, inter_dim]
        xr = x32[rows]
        gate = xr @ w1e[:inter_dim].t()
        up = xr @ w1e[inter_dim : 2 * inter_dim].t()
        _a1 = _swiglu(gate, up, swiglu_limit)
        oracle_w[rows] += w_e[:, None] * (_a1 @ w2e.t())
        del w1e, w2e
    orw = oracle_w.cpu().numpy()

    # Compare all routing-weighted paths against the dequantized-weight oracle.
    _rm_w = _relative_l2(out_mega, orw)  # mega(prod)
    _ra_w = _relative_l2(out_atom, orw)  # atom-bf16 (reference)
    _ra8_w = _relative_l2(out_atom8, orw)  # atom-fp8 (primary baseline)
    _rma = _relative_l2(out_mega, out_atom8)  # mega vs primary baseline (should be ~0)
    _floor = 0.25 if _is_fp4 else 0.10
    _mega_ok = _rm_w < _floor
    _atom8_ok = _ra8_w < _floor
    # Fall back to cross-implementation agreement when the BF16 oracle check is unreliable.
    _oracle_broken = _ra_w > _floor
    # Allow the expected FP4 quantization divergence while requiring no material regression.
    _match_ok = (_rma < 5e-2) or (_rm_w <= _ra8_w + 2e-2)
    ok = _mega_ok if swiglu_limit > 0 else _match_ok and (_oracle_broken or (_mega_ok and _atom8_ok))

    # Gate on the worst expert shard across ranks.
    _rm_w_max = _all_max(dev, _rm_w)
    _ra8_w_max = _all_max(dev, _ra8_w)
    _ra_w_max = _all_max(dev, _ra_w)
    _rma_max = _all_max(dev, _rma)
    _all_ok = _all_max(dev, 0.0 if ok else 1.0) < 0.5  # any failing rank -> all_ok False

    # Use either profiler traces or lightweight event timing.
    if getattr(args, "profile", False):
        _pmeta = dict(tokens=run_tokens, network=args.network, quant=args.quant)
        _pdir = getattr(args, "profile_dir", "") or "/tmp/mega_prof"
        _tag = f"{args.network}_{args.quant}_bs{run_tokens}"
        _pm = _profile_body(_mega_body, f"mega_{_tag}", args, rank, world, dev, _pdir, _pmeta)
        _pa8 = _profile_body(_atom_fp8_body, f"atomfp8_{_tag}", args, rank, world, dev, _pdir, _pmeta)
        _pa = _profile_body(_atom_body, f"atombf16_{_tag}", args, rank, world, dev, _pdir, _pmeta)
        _t_mega, _t_atom8, _t_atom = (_pm["e2e_us_avg"] / 1e3, _pa8["e2e_us_avg"] / 1e3, _pa["e2e_us_avg"] / 1e3)
    else:
        _t_mega = _cg_time(_mega_body)  # MegaMoEV2 e2e (stage1+stage2)
        _t_atom8 = _cg_time(_atom_fp8_body)  # baseline e2e (fp8 dispatch)
        _t_atom = _cg_time(_atom_body)  # reference e2e (bf16 dispatch)
    # Populate Stage1 buffers once before isolated Stage2 timing.
    _t_mega_s2 = -1.0
    if not getattr(args, "profile", False):
        moe._run_fused_stage1(x_q, wc, x_sc, ic)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

        def _mega_s2_body():
            moe._run_stage2(run_tokens, None, True, moe._active_config)

        _t_mega_s2 = _cg_time(_mega_s2_body)

    if rank == 0:
        _e2e_warn = (
            "  [WARN torch-oracle unreliable for this shape: gated on mega-vs-baseline]" if _oracle_broken else ""
        )
        status = "PASS" if _all_ok else "FAIL"
        print(
            f"[FULL-E2E] {args.network} {args.quant} bs={run_tokens} seed={args.seed} -> "
            f"{status} (all {world} ranks){_e2e_warn}",
            flush=True,
        )
        print(
            f"  [precision vs WEIGHTED torch-oracle, MAX over {world} ranks]  mega(prod)={_rm_w_max:.3e}  "
            f"atom-fp8(baseline)={_ra8_w_max:.3e}  atom-bf16(ref)={_ra_w_max:.3e}  "
            f"mega-vs-baseline={_rma_max:.3e}  (floor~{_floor})",
            flush=True,
        )
        _timer = "profiler-e2e" if getattr(args, "profile", False) else "cuda-event"
        print(
            f"  [perf E2E (stage1+fused-stage2), ms | {_timer}]  baseline-fp8={_t_atom8:.4f}  "
            f"mega={_t_mega:.4f}  speedup={(_t_atom8 / _t_mega) if _t_mega > 0 else -1:.3f}  "
            f"| ref bf16-dispatch baseline={_t_atom:.4f}  (out=bf16)",
            flush=True,
        )
        print(f"  [perf STAGE2 (gemm2_fused_p2p + combine), ms | cuda-event]  stage2={_t_mega_s2:.4f}", flush=True)
    return dict(
        network=args.network,
        quant=args.quant,
        tokens=run_tokens,
        full_e2e_mega_relL2=_rm_w,
        full_e2e_atom_fp8_relL2=_ra8_w,
        full_e2e_atom_bf16_relL2=_ra_w,
        full_e2e_mega_vs_baseline=_rma,
        full_e2e_baseline_fp8_ms=_t_atom8,
        full_e2e_baseline_bf16_ms=_t_atom,
        full_e2e_mega_ms=_t_mega,
        full_e2e_pass=ok,
    )


_PERF_BASELINE_CACHE = {}
# Allow 5% for run-to-run, thermal, and clock variance.
_MEGA_PERF_TOL = 0.05
_MEGA_PERF_BASELINE = {
    "v4_flash:a8w4:8": 0.1135,
    "v4_flash:a8w4:16": 0.1232,
    "v4_flash:a8w4:32": 0.1274,
    "v4_flash:a8w4:64": 0.1352,
    "v4_flash:a8w4:128": 0.1488,
    "v4_flash:a8w4:512": 0.2631,
    "v4_flash:a8w4:2048": 0.7650,
    "v4_flash:a8w4:4096": 1.5590,
    "v4_flash:a8w4:8192": 2.9255,
    "v4_flash:a8w4:16384": 5.7008,
    "v4_pro:a8w4:8": 0.2574,
    "v4_pro:a8w4:16": 0.3157,
    "v4_pro:a8w4:32": 0.3267,
    "v4_pro:a8w4:64": 0.3380,
    "v4_pro:a8w4:128": 0.3508,
    "v4_pro:a8w4:512": 0.6221,
    "v4_pro:a8w4:2048": 1.6634,
    "v4_pro:a8w4:4096": 3.1158,
    "v4_pro:a8w4:8192": 6.0063,
    "v4_pro:a4w4:2": 0.1484,
    "v4_pro:a4w4:4": 0.2057,
    "v4_pro:a4w4:8": 0.2433,
    "v4_pro:a4w4:16": 0.2975,
    "v4_pro:a4w4:32": 0.3145,
    "v4_pro:a4w4:64": 0.3261,
    "v4_pro:a4w4:128": 0.3401,
    "v4_pro:a4w4:256": 0.4400,
    "v4_pro:a4w4:512": 0.5551,
    "v4_pro:a4w4:1024": 0.7363,
    "v4_pro:a4w4:4096": 1.9211,
    "v4_pro:a4w4:8192": 3.5759,
}


def _perf_key(network, quant, tokens, mtpr=0):
    key = f"{network}:{quant}:{tokens}"
    return f"{key}:mtpr{mtpr}" if mtpr else key


def _perf_baseline_lookup(path, network, quant, tokens, mtpr=0):
    """Look up a MegaMoEV2 latency baseline, optionally overriding the built-in table."""
    data = _MEGA_PERF_BASELINE
    if path:
        data = _PERF_BASELINE_CACHE.get(path)
        if data is None:
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:  # noqa: BLE001
                data = {}
            _PERF_BASELINE_CACHE[path] = data
    v = data.get(_perf_key(network, quant, tokens, mtpr))
    return float(v) if v is not None else None


def _run_mega_only(
    args,
    rank,
    world,
    dev,
    *,
    model_dim,
    inter_dim,
    experts,
    epr,
    topk,
    swiglu_limit,
    run_tokens,
    mtpr,
    quant,
    w_kernel,
    scale_w1_1d,
    x_bf16,
    topk_ids,
    wts,
    w_kernel_gui=None,
    scale_gui=None,
    w_ref_local=None,
    local_experts_only=False,
    check_acc=True,
    measure_perf=False,
    stage1_only=False,
):
    """Run the aiter-free MegaMoEV2 accuracy and performance CI path."""

    from kernels.mega_moe import MegaMoEV2

    _is_fp4_mega = quant == "a4w4"

    # The dequantized-weight oracle isolates kernel and activation-quantization error.
    _floor = 0.25 if quant == "a4w4" else 0.10

    if stage1_only:
        w2_f32 = None
        w2_kernel = torch.empty(1, dtype=torch.uint8, device=dev)
        w2_scale_1d = torch.empty(1, dtype=torch.uint8, device=dev)
    else:
        torch.manual_seed(args.seed + 4242 + (rank if local_experts_only else 0))
        w2_experts = epr if local_experts_only else experts
        w2_f32 = torch.randn((w2_experts * model_dim, inter_dim), device=dev, dtype=torch.float32)
        w2_f32.mul_(float(inter_dim) ** -0.25)
        w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
        _w2sl = (
            slice(0, epr * model_dim)
            if local_experts_only
            else slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
        )
        w2_kernel = shuffle_weight(w2_fp4[_w2sl]).view(torch.uint8).contiguous().view(-1)
        w2_scale_1d = gemm_common_utils.e8m0_shuffle(w2_sr[_w2sl]).view(torch.uint8).contiguous().view(-1)
        del w2_fp4, w2_sr
        torch.cuda.empty_cache()

    # MegaMoEV2 local weights: this rank's epr experts.
    _weight_experts = epr if local_experts_only else experts
    _wpe = w_kernel.numel() // _weight_experts
    _spe = scale_w1_1d.numel() // _weight_experts
    _w1sl = slice(0, epr * _wpe) if local_experts_only else slice(rank * epr * _wpe, (rank + 1) * epr * _wpe)
    _w1ssl = slice(0, epr * _spe) if local_experts_only else slice(rank * epr * _spe, (rank + 1) * epr * _spe)
    assert w_kernel_gui is not None and scale_gui is not None
    _w1 = w_kernel_gui.reshape(-1)[_w1sl].contiguous()
    _w1s = scale_gui.reshape(-1)[_w1ssl].contiguous()

    moe = MegaMoEV2(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=quant,
        w1=_w1,
        w1_scale=_w1s,
        w2=w2_kernel,
        w2_scale=w2_scale_1d,
        max_tok_per_rank=mtpr,
        stage2_p2p_quant=args.stage2_p2p_quant,
        swiglu_limit=swiglu_limit,
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    skew_ms = float(getattr(args, "rank_skew_ms", 0.0))
    if skew_ms > 0:
        time.sleep(rank * skew_ms / 1000.0)

    wc = wts[:run_tokens].contiguous()
    ic = topk_ids[:run_tokens].to(torch.int32).contiguous()
    x_in = x_bf16[:run_tokens].contiguous()
    if stage1_only:
        x_s1, scale_s1 = moe.quantize(x_in)

    _out = {}

    def _body():
        if stage1_only:
            moe._run_fused_stage1(x_s1, wc, scale_s1, ic)
        else:
            _out["o"] = moe.forward(x_in, wc, ic)

    _body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    out_mega = None if stage1_only else _out["o"][:run_tokens].float().cpu().numpy().copy()

    stage1_rel = -1.0
    if stage1_only and w_ref_local is not None:
        op, tile_m = moe._s1_op, int(moe._s1_active_tile_m)
        nvalid = int(op.num_valid.view(-1)[0].item())
        tiles = nvalid // tile_m
        trb = op.tile_row_base[:tiles].to(torch.int64)
        source_rows = (trb[:, None] + torch.arange(tile_m, device=dev)[None, :]).reshape(-1)
        src = op.srcmap_em[source_rows]
        valid = ((src & 0x00FFFFFF) < moe.max_recv) & ((src >> 24) < moe.topk)
        compact_rows = torch.arange(nvalid, device=dev, dtype=torch.int64)[valid]
        src_valid = src[valid]
        src_tok = (src_valid & 0x00FFFFFF).to(torch.int64) % moe.mtpr
        x_e8 = scale_s1.view(torch.uint8).view(-1, model_dim // 32)[src_tok].float()
        x_source = x_s1.view(torch.uint8) if _is_fp4_mega else x_s1
        x_payload = x_source.view(-1, model_dim // 2 if _is_fp4_mega else model_dim)[src_tok]
        if _is_fp4_mega:
            x_payload = x_payload.view(torch.float4_e2m1fn_x2)
        x_values = gemm_common_utils.mxfp4_to_f32(x_payload) if _is_fp4_mega else x_payload.float()
        inp_s1 = (
            x_values.view(-1, model_dim // 32, 32).mul(torch.pow(2.0, x_e8 - 127.0)[:, :, None]).reshape(-1, model_dim)
        )
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        cols = torch.arange(inter_dim // 32, device=dev, dtype=torch.int64)
        d0, d1, d2 = compact_rows >> 5, (compact_rows >> 4) & 1, compact_rows & 15
        d3, d4, d5 = cols >> 3, (cols >> 2) & 1, cols & 3
        scale_offsets = (
            d0[:, None] * (scale_cols * 32)
            + d3[None, :] * 256
            + d5[None, :] * 64
            + d2[:, None] * 4
            + d4[None, :] * 2
            + d1[:, None]
        )
        e8 = moe._s1_osd[scale_offsets]
        out_source = moe._s1_out.view(torch.uint8) if _is_fp4_mega else moe._s1_out
        out_payload = out_source.view(-1, inter_dim // 2 if _is_fp4_mega else inter_dim)[compact_rows]
        if _is_fp4_mega:
            out_payload = out_payload.view(torch.float4_e2m1fn_x2)
        out_values = gemm_common_utils.mxfp4_to_f32(out_payload) if _is_fp4_mega else out_payload.float()
        got_s1 = (
            out_values.view(-1, inter_dim // 32, 32)
            .mul(torch.pow(2.0, e8.float() - 127.0)[:, :, None])
            .reshape(-1, inter_dim)
        )
        eids_s1 = op.sorted_expert_ids[:tiles].repeat_interleave(tile_m)[valid]
        ref_s1 = torch.empty_like(got_s1)
        for expert in torch.unique(eids_s1).tolist():
            rows = torch.nonzero(eids_s1 == int(expert), as_tuple=False).flatten()
            w1e = _dequant_mx_to_f32(w_ref_local[expert - rank * epr], "fp4")
            xr = inp_s1[rows]
            gate = xr @ w1e[:inter_dim].t()
            up = xr @ w1e[inter_dim:].t()
            ref_s1[rows] = _swiglu(gate, up, swiglu_limit)
        if not torch.isfinite(got_s1).all() or not torch.isfinite(ref_s1).all():
            got_bad = int((~torch.isfinite(got_s1)).sum().item())
            ref_bad = int((~torch.isfinite(ref_s1)).sum().item())
            e8_min, e8_max = int(e8.min().item()), int(e8.max().item())
            print(
                f"[mega-diag rank={rank}] nonfinite: got={got_bad}/{got_s1.numel()} "
                f"ref={ref_bad}/{ref_s1.numel()} e8=[{e8_min},{e8_max}]",
                flush=True,
            )
        stage1_rel = (torch.norm(got_s1 - ref_s1) / torch.norm(ref_s1)).item()
        stage1_ratio = (torch.norm(got_s1) / torch.norm(ref_s1)).item()
        print(
            f"[mega-diag rank={rank}] stage1-vs-ref: relL2={stage1_rel:.4e} norm_ratio={stage1_ratio:.4f}",
            flush=True,
        )

    if stage1_only:
        stage1_rel_max = _all_max(dev, stage1_rel)
        stage1_ok = 0.0 <= stage1_rel_max < _floor
        if rank == 0:
            print(
                f"[STAGE1-ONLY] {args.network} {quant} bs={run_tokens} -> "
                f"{'PASS' if stage1_ok else 'FAIL'} (all {world} ranks) "
                f"[relL2={stage1_rel_max:.4e}, floor={_floor}]",
                flush=True,
            )
        return dict(
            network=args.network,
            quant=quant,
            tokens=run_tokens,
            mega_stage1_relL2=stage1_rel_max,
            full_e2e_pass=stage1_ok,
        )

    n_layers = int(getattr(args, "layers", 1))
    _acc_metric = -1.0  # relL2 (single layer) or 1-cosine (chain); the reported / gated value
    _acc_floor = _floor
    _acc_label = "relL2(vs oracle)"
    acc_ok = True
    if check_acc:
        if local_experts_only:
            assert w_ref_local is not None
            w1_all = w_ref_local
            w2_all = w2_f32.view(epr, model_dim, inter_dim)
        else:
            _init = float(model_dim) ** -0.25
            torch.manual_seed(args.seed)
            _ = torch.randn((run_tokens, model_dim), device=dev, dtype=torch.float32)
            w1_all = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * _init
            w2_all = w2_f32.view(experts, model_dim, inter_dim)
        if n_layers > 1:
            # Compare the residual device chain with the same-route PyTorch reference.
            routings = _make_layer_routings(n_layers, run_tokens, experts, topk, dev, args.seed + 4242, rank)
            xd = x_in
            for _ids_l, _wts_l in routings:
                xn = _rmsnorm(xd)
                out = moe.forward(xn, _wts_l, _ids_l)
                xd = xd + out[:run_tokens]
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            out_dev = xd[:run_tokens].float()
            out_ref = RefModel(w1_all, w2_all, inter_dim, dev, swiglu_limit).run(x_in, routings).float()
            _acc_metric = _calc_diff(out_ref, out_dev)  # 1 - cosine (fp64), end-to-end accumulated
            _acc_floor = _CHAIN_TOL
            _acc_label = f"cos_diff(chain x{n_layers})"
            acc_ok = _acc_metric < _acc_floor
        else:
            x32 = x_in.float()
            ids_l = ic.long()
            wv = wc.float()
            if local_experts_only:
                gathered_x = [torch.empty_like(x32) for _ in range(world)]
                gathered_ids = [torch.empty_like(ids_l) for _ in range(world)]
                gathered_wv = [torch.empty_like(wv) for _ in range(world)]
                dist.all_gather(gathered_x, x32)
                dist.all_gather(gathered_ids, ids_l)
                dist.all_gather(gathered_wv, wv)
                x32 = torch.cat(gathered_x)
                ids_l = torch.cat(gathered_ids)
                wv = torch.cat(gathered_wv)
            oracle = torch.zeros(ids_l.shape[0], model_dim, device=dev, dtype=torch.float32)
            expert_begin = rank * epr if local_experts_only else 0
            expert_end = expert_begin + epr if local_experts_only else experts
            for e in (e for e in torch.unique(ids_l).tolist() if expert_begin <= e < expert_end):
                sel = ids_l == int(e)  # [T, topk]: which (token, slot) route to expert e
                rows = sel.any(dim=1).nonzero().flatten()
                w_e = (wv * sel).sum(dim=1)[rows]  # per-token routing weight (summed over slots) for e
                local_e = e - expert_begin
                w1e = _dequant_mx_to_f32(w1_all[local_e], "fp4")
                w2e = _dequant_mx_to_f32(w2_all[local_e], "fp4")
                xr = x32[rows]
                gate = xr @ w1e[:inter_dim].t()
                up = xr @ w1e[inter_dim : 2 * inter_dim].t()
                _a1 = _swiglu(gate, up, swiglu_limit)
                oracle[rows] += w_e[:, None] * (_a1 @ w2e.t())
                del w1e, w2e
            if local_experts_only:
                dist.all_reduce(oracle, op=dist.ReduceOp.SUM)
                oracle = oracle.view(world, run_tokens, model_dim)[rank]
            _acc_metric = _relative_l2(out_mega, oracle.cpu().numpy())
            acc_ok = _acc_metric < _acc_floor
        del w1_all
        torch.cuda.empty_cache()
    relL2 = _acc_metric  # kept name for the return dict / downstream reporting

    # ---- perf: CUDAGraph device time (mean across ranks) ----
    mega_ms = -1.0
    mega_max_ms = -1.0
    prequant_ms = -1.0
    prequant_max_ms = -1.0
    stage1_ms = -1.0
    stage1_max_ms = -1.0
    stage2_ms = -1.0
    stage2_max_ms = -1.0
    perf_ok = True
    perf_note = ""
    if measure_perf:
        _n = max(1, int(args.iters))

        def _time_graph(fn):
            ms.shmem_barrier_all()
            fn()
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            capture_stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                fn()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_n):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            local_ms = start.elapsed_time(end) / _n
            return _all_mean(dev, local_ms), _all_max(dev, local_ms)

        x_s1, scale_s1 = moe.quantize(x_in)

        def _stage1_body():
            moe._run_fused_stage1(x_s1, wc, scale_s1, ic)

        def _prequant_body():
            _out["o"] = moe.forward_prequant(x_s1, scale_s1, wc, ic)

        stage1_ms, stage1_max_ms = _time_graph(_stage1_body)
        moe._run_fused_stage1(x_s1, wc, scale_s1, ic)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

        def _stage2_body():
            _out["o"] = moe._run_stage2(run_tokens, None, True, moe._active_config)

        stage2_ms, stage2_max_ms = _time_graph(_stage2_body)
        prequant_ms, prequant_max_ms = _time_graph(_prequant_body)
        mega_ms, mega_max_ms = _time_graph(_body)
        if args.profile:
            _profile_body(
                _body,
                f"mega_only_{args.network}_{quant}_bs{run_tokens}",
                args,
                rank,
                world,
                dev,
                args.profile_dir,
                dict(tokens=run_tokens, network=args.network, quant=quant),
            )
        _base = _perf_baseline_lookup(
            getattr(args, "perf_baseline", ""), args.network, quant, run_tokens, int(args.mtpr)
        )
        if _base is not None and _base > 0:
            _tol = float(args.perf_tol)
            perf_ok = mega_ms <= _base * (1.0 + _tol)
            perf_note = f"baseline={_base:.4f}ms +{_tol:.0%} -> {'OK' if perf_ok else 'REGRESSION'}"
        else:
            perf_note = "no committed baseline for this config (perf gate skipped)"

    ok = acc_ok and perf_ok
    _all_ok = _all_max(dev, 0.0 if ok else 1.0) < 0.5
    _relL2_max = _all_max(dev, relL2 if math.isfinite(relL2) and relL2 >= 0 else float("inf"))
    _ms_max = max(0.0, mega_max_ms)
    _prequant_ms_max = max(0.0, prequant_max_ms)
    _s1_ms_max = max(0.0, stage1_max_ms)
    _s2_ms_max = max(0.0, stage2_max_ms)
    _active_sbm = int(getattr(moe, "_s1_active_tile_m", -1))
    _active_g2_bm = int(moe._g2_active_block_m) if not stage1_only else -1
    _active_grid_mult = int(moe._active_config.stage1.grid_mult)
    if rank == 0:
        _acc_s = f"{_acc_label}={_relL2_max:.3e} (floor~{_acc_floor})" if check_acc else "acc:skip"
        _perf_s = (
            f"SBM={_active_sbm} G2_BM={_active_g2_bm} "
            f"GRID={_active_grid_mult} "
            f"stage1={stage1_ms:.4f}/{_s1_ms_max:.4f}ms "
            f"stage2={stage2_ms:.4f}/{_s2_ms_max:.4f}ms "
            f"prequant_e2e={prequant_ms:.4f}/{_prequant_ms_max:.4f}ms "
            f"bf16_e2e={mega_ms:.4f}/{_ms_max:.4f}ms(mean/max)  {perf_note}"
            if measure_perf
            else "perf:skip"
        )
        print(
            f"[MEGA-ONLY] {args.network} {quant} bs={run_tokens} seed={args.seed} -> "
            f"{'PASS' if _all_ok else 'FAIL'} (all {world} ranks)  [{_acc_s}]  [{_perf_s}]",
            flush=True,
        )
    return dict(
        network=args.network,
        quant=quant,
        tokens=run_tokens,
        world_size=world,
        experts_per_rank=epr,
        max_tok_per_rank=mtpr,
        perf_mtpr=int(args.mtpr),
        mega_only_relL2=relL2,
        mega_sorted_block_m=_active_sbm,
        mega_gemm2_block_m=_active_g2_bm,
        mega_stage1_grid_mult=_active_grid_mult,
        mega_stage1_ms=stage1_ms,
        mega_stage1_max_ms=stage1_max_ms,
        mega_stage2_ms=stage2_ms,
        mega_stage2_max_ms=stage2_max_ms,
        mega_prequant_ms=prequant_ms,
        mega_prequant_max_ms=prequant_max_ms,
        mega_only_ms=mega_ms,
        mega_only_max_ms=mega_max_ms,
        full_e2e_pass=bool(_all_ok),
    )


def run_one(args, rank, world, dev):
    net = NETWORKS[args.network]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    if int(args.experts_per_rank) > 0:
        experts = int(args.experts_per_rank) * world
    swiglu_limit = float(net.get("swiglu_limit", 0.0))
    # topk: --topk>0 overrides; else use the network's native topk (r1_v3=8, v4_*=6).
    topk = int(args.topk) if int(args.topk) > 0 else int(net["topk"])
    run_tokens = max(int(args.tokens), 1)  # allow bs=1 (1 token/rank); routing still reaches all ranks
    if experts % world != 0:
        raise SystemExit(f"experts={experts} must divide world={world}")
    epr = experts // world

    mtpr = int(args.mtpr) if int(args.mtpr) > 0 else max(16, 1 << (run_tokens - 1).bit_length())
    mega_only = bool(getattr(args, "mega_only", False))
    local_experts_only = mega_only and (args.skip_acc or args.stage1_only or int(args.layers) == 1)
    keep_ref = bool(args.stage1_only or (mega_only and not args.skip_acc and int(args.layers) == 1))
    T = _prepare(
        dev,
        quant=args.quant,
        tokens=run_tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        seed=args.seed,
        rank=rank,
        world=world,
        keep_ref=keep_ref,
        local_experts_only=local_experts_only,
    )
    w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
    topk_ids, wts = T["topk_ids"], T["wts"]
    a_dtype = T["a_dtype"]
    x_bf16 = T["x_bf16"]

    # CI path: aiter-free MegaMoEV2-only run (accuracy vs torch oracle + golden-baseline perf).
    if getattr(args, "mega_only", False):
        return _run_mega_only(
            args,
            rank,
            world,
            dev,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            epr=epr,
            topk=topk,
            swiglu_limit=swiglu_limit,
            run_tokens=run_tokens,
            mtpr=mtpr,
            quant=args.quant,
            w_kernel=w_kernel,
            scale_w1_1d=scale_w1_1d,
            x_bf16=x_bf16,
            topk_ids=topk_ids,
            wts=wts,
            w_kernel_gui=T.get("w_kernel_gui"),
            scale_gui=T.get("scale_gui"),
            w_ref_local=T.get("w_ref_local"),
            local_experts_only=bool(T.get("local_experts_only", False)),
            check_acc=not bool(args.skip_acc),
            measure_perf=bool(args.measure_perf),
            stage1_only=bool(args.stage1_only),
        )

    # The manual path compares MegaMoEV2 with the FlyDSL ATOM pipeline.
    return _run_full_e2e(
        args,
        rank,
        world,
        dev,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        epr=epr,
        topk=topk,
        swiglu_limit=swiglu_limit,
        run_tokens=run_tokens,
        mtpr=mtpr,
        a_dtype=a_dtype,
        s1_out=a_dtype,
        w_kernel=w_kernel,
        scale_w1_1d=scale_w1_1d,
        x_bf16=x_bf16,
        topk_ids=topk_ids,
        wts=wts,
        w_kernel_gui=T.get("w_kernel_gui"),
        scale_gui=T.get("scale_gui"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", type=str, default="v4_pro", choices=list(NETWORKS))
    p.add_argument("--quant", type=str, default="a8w4", choices=["a8w4", "a4w4"])
    p.add_argument(
        "--stage2-p2p-quant",
        type=str,
        default="auto",
        choices=["auto", "none", "fp8_blockwise_1x32"],
        help="override the Stage2 transport mode; auto keeps the production threshold",
    )
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--mtpr", type=int, default=0, help="0 selects the smallest power-of-two capacity for each BS")
    p.add_argument(
        "--topk",
        type=int,
        default=-1,
        help="-1 (default) = use the network's native topk (r1_v3=8, v4_*=6); >0 overrides",
    )
    p.add_argument(
        "--experts-per-rank",
        type=int,
        default=0,
        help="override total experts as world_size*N so local expert count stays fixed across world sizes",
    )
    p.add_argument("--waves-per-eu", type=int, default=4)
    p.add_argument("--async-copy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank-skew-ms", type=float, default=0.0, help="delay rank r by r*N ms before first forward")
    p.add_argument(
        "--layers",
        type=int,
        default=1,
        help="(--mega-only) N>1 runs the chained N-layer accumulation ACCURACY check (device "
        "moe.forward per layer + RMSNorm + residual vs the pure-torch dequant-weights RefModel over "
        "shared weights + per-layer routing; 1-cosine gate). Real DeepSeek-V4 depth is ~61.",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="run EACH bs with N distinct seeds (seed, seed+1, ...) for random-data "
        "coverage.  NOTE: per-run symmetric buffers are not freed, so large-bs x "
        "multi-seed needs a bigger heap, e.g. MORI_SHMEM_HEAP_SIZE=32G.",
    )
    p.add_argument("--master-port", type=int, default=29921)
    p.add_argument("--matrix", action="store_true", help="run all networks x classic bs")
    p.add_argument("--full-bs", action="store_true", help="use the full bs sweep (1..32768)")
    p.add_argument(
        "--bs-list",
        type=str,
        default="",
        help="comma list of batch sizes to sweep for the single --network/--quant "
        "(e.g. '256,2048,4096,8192'); overrides --tokens/--matrix.",
    )
    p.add_argument("--json-out", type=str, default="")
    p.add_argument(
        "--profile",
        action="store_true",
        help="measure with torch.profiler instead of cuda-event (mutually exclusive): "
        "dump chrome trace + per-kernel GPU table + E2E replay time. Default is the "
        "lightweight cuda-event timing.",
    )
    p.add_argument(
        "--profile-dir",
        type=str,
        default="/tmp/mega_prof",
        help="output dir for --profile chrome traces (default /tmp/mega_prof).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="CI gate: exit non-zero if ANY (network,bs,seed) case fails its correctness "
        "gate (or, with --min-speedup, regresses vs the atom-fp8 baseline) or errors, "
        "or if some rank ran zero cases. Default off keeps the manual-run behavior "
        "(always exit 0).",
    )
    p.add_argument(
        "--min-speedup",
        type=float,
        default=0.0,
        help="CI perf gate for the ATOM path (only meaningful with --strict): require the MegaMoEV2 E2E "
        "speedup vs the atom-fp8 production baseline to be >= this value for every case (0 = disabled).",
    )
    p.add_argument(
        "--mega-only",
        action="store_true",
        help="Aiter-free CI path: run only MegaMoEV2 (moe.forward), without ATOM or aiter. "
        "Accuracy is gated vs a torch f32 oracle; perf vs a committed golden baseline "
        "(--perf-baseline). This is what the multi-gpu CI runs.",
    )
    p.add_argument(
        "--stage1-only",
        action="store_true",
        help="(--mega-only) validate dispatch+GEMM1 only; do not execute GEMM2/combine or build W2.",
    )
    p.add_argument(
        "--measure-perf",
        action="store_true",
        help="(--mega-only) time moe.forward under CUDAGraph and gate against --perf-baseline "
        "(match-or-better within --perf-tol).",
    )
    p.add_argument(
        "--skip-acc",
        action="store_true",
        help="(--mega-only) skip the torch f32 accuracy oracle (avoids the full-weight alloc in "
        "perf-only benchmark runs).",
    )
    p.add_argument(
        "--perf-baseline",
        type=str,
        default="",
        help="(--mega-only) optional latency JSON overriding the built-in baseline "
        "({'network:quant:bs[:mtprN]': ms}); passes if measured <= golden * (1 + --perf-tol).",
    )
    p.add_argument(
        "--perf-tol",
        type=float,
        default=_MEGA_PERF_TOL,
        help="(--mega-only) fractional slack over the golden baseline that still counts as a match "
        "(default 0.05 = 5%%).",
    )
    p.add_argument(
        "--perf-out",
        type=str,
        default="",
        help="(--mega-only) write measured latencies as golden JSON "
        "({'network:quant:bs[:mtprN]': ms}) to this path (used on a reference machine).",
    )
    args = p.parse_args()
    if args.stage1_only and not args.mega_only:
        p.error("--stage1-only requires --mega-only")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, args.master_port)
    dev = torch.device("cuda", local_rank)

    bs_set = FULL_BS if args.full_bs else CLASSIC_BS
    if args.bs_list:
        combos = [(args.network, int(b)) for b in args.bs_list.split(",") if b.strip()]
    elif args.matrix:
        combos = [(net, bs) for net in NETWORKS for bs in bs_set]
    elif args.full_bs:
        combos = [(args.network, bs) for bs in bs_set]
    else:
        combos = [(args.network, int(args.tokens))]

    results = []
    _base_seed = int(args.seed)
    _n_seeds = max(1, int(args.n_seeds))
    _strict_fail = 0  # local count of failing cases (correctness or, with --min-speedup, perf/ERROR)
    for net, bs in combos:
        args.network = net
        args.tokens = bs
        for _si in range(_n_seeds):
            args.seed = _base_seed + _si
            try:
                r = run_one(args, rank, world, dev)
                if r is not None:
                    results.append(r)
                    if not r.get("full_e2e_pass", False):
                        _strict_fail += 1
                        _info(rank, f"[strict] {net} bs={bs} seed={args.seed} FAIL (correctness/perf gate)")
                    elif (
                        args.min_speedup > 0.0
                        and "full_e2e_baseline_fp8_ms" in r
                        and r.get("full_e2e_mega_ms", 0.0) > 0.0
                    ):
                        _sp = r["full_e2e_baseline_fp8_ms"] / r["full_e2e_mega_ms"]
                        if _sp < args.min_speedup:
                            _strict_fail += 1
                            _info(
                                rank,
                                f"[strict] {net} bs={bs} seed={args.seed} PERF regression: "
                                f"speedup={_sp:.3f} < min_speedup={args.min_speedup}",
                            )
            except Exception as e:  # noqa: BLE001
                import traceback

                if rank == 0:
                    traceback.print_exc()
                _strict_fail += 1
                _info(rank, f"[bench] {net} bs={bs} seed={args.seed} ERROR: {type(e).__name__}: {e}")
            # Release per-case allocations before the next large-network shape.
            import gc as _gc

            _gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            dist.barrier()
    args.seed = _base_seed

    if rank == 0 and args.json_out:
        with open(args.json_out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        _info(rank, f"[bench] wrote {len(results)} rows -> {args.json_out}")

    # Merge measured latencies into the optional golden baseline.
    if rank == 0 and getattr(args, "perf_out", ""):
        golden = {}
        if os.path.exists(args.perf_out):
            try:
                with open(args.perf_out) as f:
                    golden = json.load(f)
            except Exception:  # noqa: BLE001
                golden = {}
        for r in results:
            if r.get("mega_only_ms", -1.0) > 0.0:
                golden[_perf_key(r["network"], r["quant"], r["tokens"], r.get("perf_mtpr", 0))] = round(
                    float(r["mega_only_ms"]), 4
                )
        with open(args.perf_out, "w") as f:
            json.dump(golden, f, indent=2, sort_keys=True)
            f.write("\n")
        _info(rank, f"[perf] wrote {len(golden)} golden rows -> {args.perf_out}")

    # Reduce failures before teardown so every rank exits consistently.
    _strict_exit = False
    if args.strict:
        _global_fail = _all_max(dev, float(_strict_fail))
        _min_ran = _all_min_int(dev, len(results))
        if rank == 0:
            print(
                f"[strict] cases_run(min over ranks)={_min_ran}  " f"failing_cases(max over ranks)={int(_global_fail)}",
                flush=True,
            )
        _strict_exit = (_global_fail > 0.5) or (_min_ran == 0)

    torch.cuda.synchronize()
    dist.barrier()
    _cleanup()
    if _strict_exit:
        sys.exit(1)


import pytest  # noqa: E402


def _count_physical_gpus() -> int:
    """Physical GPU count via a fresh subprocess (bypasses HIP_VISIBLE_DEVICES + torch's cache)."""
    import subprocess as _sp

    env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    try:
        r = _sp.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        return int(r.stdout.strip()) if r.returncode == 0 else 0
    except Exception:  # noqa: BLE001
        return 0


def _gpu_arch() -> str:
    try:
        from flydsl.runtime.device import get_rocm_arch

        return str(get_rocm_arch() or "")
    except Exception:  # noqa: BLE001
        return ""


def _skip_unless_mega_8gpu() -> None:
    """Skip unless the host supports the eight-GPU A8W4 harness."""
    if _HARNESS_DEPS_ERROR is not None:
        pytest.skip(f"MegaMoEV2 deps unavailable (need mori + FlyDSL dispatch/combine): {_HARNESS_DEPS_ERROR}")
    arch = _gpu_arch()
    if not arch.startswith("gfx95"):
        pytest.skip(f"MegaMoEV2 A8W4/A4W4 requires CDNA4 (gfx95x); current arch: {arch or 'unknown'}")
    phys = _count_physical_gpus()
    if phys < 8:
        pytest.skip(f"requires >= 8 physical GPUs, found {phys}")


def _run_mega_8gpu(
    *, network, quant, bs_list, iters, measure_perf=False, skip_acc=False, layers=1, mtpr=0, timeout=2400
):
    """Run this file under eight-GPU torchrun and require a clean strict exit."""
    import subprocess as _sp

    env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    # Propagate the parent import path so child workers find flydsl._mlir.
    _extra_pp = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONPATH"] = _extra_pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
    # Allow the large FP32 oracle allocations to use non-contiguous free segments.
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        os.path.abspath(__file__),
        "--mega-only",
        "--network",
        network,
        "--quant",
        quant,
        "--bs-list",
        bs_list,
        "--iters",
        str(iters),
        "--strict",
    ]
    if mtpr > 0:
        cmd += ["--mtpr", str(mtpr)]
    if layers > 1:
        cmd += ["--layers", str(layers)]
    if measure_perf:
        cmd += ["--measure-perf", "--perf-tol", str(_MEGA_PERF_TOL)]
    if skip_acc:
        cmd += ["--skip-acc"]

    result = _sp.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
    # Surface the human-readable gate lines (accuracy / perf) regardless of outcome.
    for line in result.stdout.splitlines():
        if any(tag in line for tag in ("[MEGA-ONLY]", "[strict]")):
            print(line)
    assert result.returncode == 0, (
        f"MegaMoEV2 8-GPU {network}/{quant} bs={bs_list} mtpr={mtpr or 'auto'} "
        f"FAILED (exit {result.returncode}).\n"
        f"stdout (last 3000 chars):\n{result.stdout[-3000:]}\n"
        f"stderr (last 2000 chars):\n{result.stderr[-2000:]}"
    )
    return result


# (network, quant, bs_list) v4_pro accuracy shapes covered by committed tuning artifacts.
_MEGA_ACC_PARAMS = [
    ("v4_pro", "a8w4", "2048,8192"),
    ("v4_pro", "a4w4", "2,4,16,128,512,2048"),
]

# (network, quant, bs_list) perf-relevant batch sizes for the benchmark (golden) gate.
_MEGA_BENCH_PARAMS = [
    ("v4_pro", "a8w4", "4096,8192"),
    ("v4_pro", "a4w4", "2,4,8,16,32,64,128,256,512,1024,4096,8192"),
]


def _mega_id(network, quant, bs_list):
    return f"{network}-{quant}-bs{bs_list.replace(',', '_')}"


@pytest.mark.multi_gpu
@pytest.mark.parametrize("network,quant,bs_list", _MEGA_ACC_PARAMS, ids=[_mega_id(*p) for p in _MEGA_ACC_PARAMS])
def test_mega_moe_8gpu_accuracy(network, quant, bs_list):
    """Gate eight-GPU MegaMoEV2 accuracy against the routing-weighted FP32 oracle."""
    _skip_unless_mega_8gpu()
    _run_mega_8gpu(network=network, quant=quant, bs_list=bs_list, iters=5)


@pytest.mark.multi_gpu
@pytest.mark.benchmark
@pytest.mark.parametrize("network,quant,bs_list", _MEGA_BENCH_PARAMS, ids=[_mega_id(*p) for p in _MEGA_BENCH_PARAMS])
def test_mega_moe_8gpu_benchmark(network, quant, bs_list):
    """Gate eight-GPU MegaMoEV2 latency against the committed baseline."""
    _skip_unless_mega_8gpu()
    _run_mega_8gpu(network=network, quant=quant, bs_list=bs_list, iters=20, measure_perf=True, skip_acc=True)


_MEGA_A4_FIXED_MTPR = 8192
_MEGA_A4_FIXED_MTPR_BS = "8,32,64,128,256,1024,4096,8192"


@pytest.mark.multi_gpu
def test_mega_moe_8gpu_a4w4_fixed_mtpr_accuracy():
    """Cover production-style dynamic batches using one maximum-capacity configuration."""
    _skip_unless_mega_8gpu()
    _run_mega_8gpu(
        network="v4_pro",
        quant="a4w4",
        bs_list=_MEGA_A4_FIXED_MTPR_BS,
        mtpr=_MEGA_A4_FIXED_MTPR,
        iters=5,
    )


# A8W4 alone supports the chained-accumulation accuracy safeguard.
_MEGA_CHAIN_PARAMS = [
    ("v4_pro", "a8w4"),
]

# Number of chained MoE layers (real DeepSeek-V4 depth).
_MEGA_CHAIN_LAYERS = 61


@pytest.mark.multi_gpu
@pytest.mark.parametrize("network,quant", _MEGA_CHAIN_PARAMS, ids=[f"{n}-{q}" for n, q in _MEGA_CHAIN_PARAMS])
def test_mega_moe_8gpu_accuracy_chained(network, quant):
    """Gate the 61-layer residual chain against the dequantized-weight reference."""
    _skip_unless_mega_8gpu()
    _run_mega_8gpu(network=network, quant=quant, bs_list="128", iters=1, layers=_MEGA_CHAIN_LAYERS)


if __name__ == "__main__":
    main()
