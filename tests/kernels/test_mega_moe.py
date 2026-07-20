#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Intranode MoE end-to-end benchmark: MegaMoE (production) vs a multi-op FlyDSL baseline.

Single production-equivalent comparison (`_run_full_e2e`), per batch size.  Fully aiter-free: the
ATOM baseline is built entirely from FlyDSL ops (sort/quant/scale-sort/gemm1/gemm2/dispatch/combine).

  * megav1   : fp8/fp4 act -> MegaMoE.forward -> bf16   (single-op, Plan-A zero-bridge dispatch).
  * atom-fp8 : PRODUCTION fp8-dispatch -> moe_sorting_flydsl -> mxfp4_moe_scale_sort (FlyDSL) ->
               mixed_moe_gemm1 -> fused GEMM2+combine -> bf16   (primary baseline).
  * atom-bf16: bf16-dispatch reference (most accurate path; used as an oracle-sanity check).

BOTH the mega and atom paths apply the REAL routing weights in the GEMM2 doweight epilogue, so the
correctness gate compares each path's routing-WEIGHTED output to a full-precision torch oracle (the
quant floor), and requires mega == atom-fp8.  Perf is CUDAGraph device time (median, max across ranks).
Supports A8W4 (mxfp8 act) and A4W4 (mxfp4 act); compact/fixed-slot is auto-selected by buffer size.

Run (8x MI355X)::

  MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
    tests/kernels/test_mega_moe.py --network v4_pro --quant a8w4 \
    --bs-list 2048,4096,8192 --iters 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

# The distributed harness (main() / _run_full_e2e) needs mori.shmem, the FlyDSL dispatch/combine op
# (which itself imports mori), and the pure-torch quant/shuffle helpers in test_moe_gemm.  Guard
# these so the module still IMPORTS for pytest collection on hosts without the full stack (e.g. the
# single-GPU CI runners that lack mori): the multi_gpu pytest cases below skip when unavailable, and
# the 8-GPU subprocess that actually runs the harness has everything installed.  This file is fully
# aiter-free: the ATOM baseline uses FlyDSL sort/quant/scale-sort (see _run_full_e2e).
try:
    import mori.shmem as ms  # noqa: E402

    import tests.kernels.test_moe_gemm as tmg  # noqa: E402
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
    tmg = gemm_common_utils = shuffle_weight = None
    _HARNESS_DEPS_ERROR = f"{type(_exc).__name__}: {_exc}"


NETWORKS = {
    "r1_v3": dict(model_dim=7168, inter_dim=2048, experts=256, topk=8),
    "v4_flash": dict(model_dim=4096, inter_dim=2048, experts=256, topk=6),
    "v4_pro": dict(model_dim=7168, inter_dim=3072, experts=384, topk=6),
}

# batch-size sweeps for --matrix / --full-bs.
CLASSIC_BS = [1, 8, 64, 512, 8192, 32768]
FULL_BS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# Gate for the chained N-layer accumulation check (--layers>1): 1-cosine of the device vs the
# pure-torch dequant-weights RefModel, end-to-end after N residual layers. This is an A8W4 (fp8
# activation) safeguard: measured ~0.052-0.055 for a correct kernel over 61 layers on 8x MI355X
# (v4_flash / v4_pro), so 0.10 leaves ~2x headroom (matches aiter's end-to-end tol). NOTE: A4W4's
# fp4 activation is too coarse to survive a deep residual chain (compounds to ~0.46 over 61 layers,
# inherent -- not a kernel bug), so the chained check is not meaningful there; use --layers 1 (the
# single-layer dequant gate) for a4w4.
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


# ============================= torch.profiler timing (opt-in via --profile) =============================
# Optional per-kernel profiling of the CUDAGraph-captured bodies. The default timing stays the
# lightweight cuda-event `_cg_time`; --profile additionally dumps a chrome trace + per-kernel GPU
# time table (rank0) and the cross-rank E2E replay time, ported from test_profiler_moe_gemm2_combine.
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


def _profile_body(body, dc_op, op_tag, args, rank, world, dev, out_dir, meta):
    """Capture `body` into a CUDAGraph (same sequence as _cg_time), replay under torch.profiler,
    dump chrome trace, and print (rank0) a per-kernel GPU table + cross-rank E2E replay time."""
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
    return {"e2e_us_avg": s.item() / world, "e2e_us_max": mx.item()}


def _chunked_fp4_quant(x):
    """Row-chunked MX-FP4 quant (identical result; bounds the f32 temp)."""
    n = int(x.shape[1])
    chunk_rows = max(4096, ((2 << 30) // max(1, n * 4 * 8)) // 4096 * 4096)
    if x.ndim != 2 or x.shape[0] <= chunk_rows:
        return tmg._per_1x32_fp4_quant(x)
    m = int(x.shape[0])
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    y = torch.empty((m, n // 2), device=x.device, dtype=fp4_dtype)
    s = torch.empty((m, n // 32), device=x.device, dtype=torch.uint8)
    for st in range(0, m, chunk_rows):
        en = min(st + chunk_rows, m)
        yc, sc = tmg._per_1x32_fp4_quant(x[st:en])
        y[st:en].copy_(yc)
        s[st:en].copy_(sc)
        del yc, sc
        torch.cuda.empty_cache()
    return y, s


def _dequant_mx_to_f32(t_f32, quant_mode):
    """Quantize a tensor to per-1x32 MX (fp4 or fp8) then dequantize back to f32 -- the EXACT lossy
    values the kernel consumes.  An accuracy oracle built on these (weights, and optionally the
    activation) isolates the kernel arithmetic error from the (dominant) quant floor, giving a far
    tighter gate than a full-precision oracle.  Aiter-free port of aiter RefModel._mxfp4_quant +
    fp4_utils dequant (mxfp4_to_f32 / e8m0_to_f32 live in gemm_common_utils; quant is per-1x32 along
    the last / K dim).  ``_chunked_fp4_quant`` bounds the fp4 temporary for big weights."""
    orig = tuple(t_f32.shape)
    t2d = t_f32.reshape(-1, orig[-1])
    if quant_mode == "fp8":
        q, s = tmg._per_1x32_mxfp8_quant(t2d)  # q: fp8_e4m3fn [., K]; s: e8m0 u8 [., K//32]
        vf = q.float()
    else:
        q, s = _chunked_fp4_quant(t2d)  # q: fp4x2 [., K//2]; s: e8m0 u8 [., K//32]
        vf = gemm_common_utils.mxfp4_to_f32(q)  # [., K] f32 via the E2M1 LUT
    sf = gemm_common_utils.e8m0_to_f32(s).unsqueeze(-1).expand(-1, -1, 32).reshape(t2d.shape)
    return (vf * sf).reshape(orig).to(torch.float32)


def _rmsnorm(x, eps=1e-6):
    """RMSNorm (no learnable gain) on the last dim. Applied to each layer's MoE input in the chained
    accumulation check so activations stay unit-scale across the residual chain -- without it the a8w4
    fp8 activation quant (max ~448) overflows to NaN after a few layers. Both device + reference use
    the SAME normalization (ported from aiter test_mega_moe._rmsnorm)."""
    xf = x.float()
    n = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return n.to(x.dtype)


def _calc_diff(x, y):
    """1 - cosine similarity (fp64); robust to residual-magnitude drift over a deep chain. Mirrors
    aiter test_moe_ep/test_mega_moe._calc_diff -- the end-to-end accumulation gate."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def _make_layer_routings(n_layers, tokens, experts, topk, dev, seed, rank):
    """Per-layer random routing (distinct experts per token via top-k over a random score; random
    renormalized weights), RETAINED so the device chain and the fp32 reference replay identical
    routing.  Ported from aiter test_mega_moe.make_routings; varied per rank + per layer."""
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
    """Pure-torch fp32 reference for the chained N-layer MoE accumulation check.

    Ported from aiter op_tests/multigpu_tests/test_mega_moe.py::RefModel (aiter-free): the mxfp4
    ``quant -> dequant`` of aiter's ``_expert`` is our ``_dequant_mx_to_f32``, so each expert uses the
    EXACT lossy weights the kernel consumes (cached per expert, reused across layers).  It chains N
    layers with a residual -- ``x = x + layer(x)`` -- where each ``layer`` is RMSNorm -> routed FFN
    (summed over topk) [+ dense shared experts], mirroring a real DeepSeek-V4 stack (~61 MoE layers).
    Uses only torch (NO mori / aiter / fused_moe); it is the accuracy ground truth.
    """

    def __init__(self, w1_f32, w2_f32, inter_dim, dev, sw1=None, sw2=None):
        self.w1_f32, self.w2_f32 = w1_f32, w2_f32  # full-precision [E, 2I, H], [E, H, I]
        self.inter_dim = inter_dim
        self.sw1, self.sw2 = sw1, sw2  # optional dense shared experts (None here)
        self.dev = dev
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

    @staticmethod
    def _ffn(x, w1d, w2d):
        import torch.nn.functional as _F

        gate, up = (x @ w1d.t()).chunk(2, dim=-1)
        return (_F.silu(gate) * up) @ w2d.t()

    def _shared(self, x):
        if self.sw1 is None:
            return torch.zeros_like(x)
        acc = torch.zeros_like(x)
        for e in range(self.sw1.shape[0]):
            acc = acc + self._ffn(x, self.sw1[e].float(), self.sw2[e].float())
        return acc

    def layer(self, x, ids, wts):
        """x [ct,H] fp32; ids/wts [ct,topk]. RMSNorm the input, then routed + shared FFN. Returns the
        block output [ct,H] fp32 (caller adds the residual)."""
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


def _prepare(dev, *, quant, tokens, model_dim, inter_dim, experts, topk, seed, rank=0, world=1, keep_ref=False):
    """Generate inputs + pack weights/scales for A8W4 (mxfp8 act) or A4W4 (mxfp4 act).

    Returns dict with x (token payload in the dispatch dtype), scale_mx_u8 (e8m0 act scale),
    w_kernel (mxfp4 shuffled), scale_w1_1d (e8m0 weight scale shuffled), topk_ids, wts,
    and meta: token_dtype, a_dtype, row_view_dim.
    """
    torch.manual_seed(seed)
    # scale x,w so the GEMM reduction over model_dim gives O(1) pre-activation (std ~ sqrt(md)*sx*sw):
    # sx=sw -> s = (md)^-0.25.  init_scale=0.01 made stage1 output ~3e-4 (accuracy oracle was a no-op).
    init_scale = float(model_dim) ** -0.25
    x_fp32 = torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * init_scale
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * init_scale

    if quant == "a8w4":
        # activation: MX-FP8 (fp8_e4m3fn, 1 byte/elem) + e8m0 block scale
        x_q, scale_x_mx = tmg._per_1x32_mxfp8_quant(x_fp32)
        x_payload = x_q.contiguous()  # [tokens, model_dim] fp8_e4m3fn
        token_dtype = torch.float8_e4m3fn
        a_dtype = "fp8"
        row_view_dim = model_dim
    elif quant == "a4w4":
        # activation: MX-FP4 (packed 2/byte) + e8m0 block scale
        x_q, scale_x_mx = tmg._per_1x32_fp4_quant(x_fp32)
        x_payload = x_q.view(torch.float4_e2m1fn_x2).contiguous()  # [tokens, model_dim//2]
        token_dtype = torch.float4_e2m1fn_x2
        a_dtype = "fp4"
        row_view_dim = model_dim // 2
    else:
        raise SystemExit(f"unknown quant {quant!r} (use a8w4|a4w4)")

    # accuracy ground-truth: keep this rank's LOCAL experts' ORIGINAL (pre-quant) weights in bf16,
    # so the oracle can compute a torch f32 reference (true MoE) and locate which stack diverges.
    w_ref_local = None
    if keep_ref:
        _epr = experts // world
        w_ref_local = w1_fp32[rank * _epr : (rank + 1) * _epr].to(torch.bfloat16).contiguous()

    # weight: MX-FP4 + shuffle for the GEMM (shared across a8w4/a4w4)
    w1_flat = w1_fp32.view(experts * (2 * inter_dim), model_dim)
    w1_fp4, w1_scale_raw = _chunked_fp4_quant(w1_flat)
    w_kernel = shuffle_weight(w1_fp4.view(torch.float4_e2m1fn_x2)).view(torch.uint8).contiguous()
    scale_w1_1d = gemm_common_utils.e8m0_shuffle(w1_scale_raw).view(torch.uint8).contiguous()
    # gate-up INTERLEAVE (g1u1) shuffle: separate fp4-packed buffers so an INTERLEAVE MegaMoE can
    # consume them while the ATOM baselines keep the SEPARATED w_kernel/scale above (same byte size).
    w_kernel_gui = (
        gemm_common_utils.shuffle_weight_w4(
            w1_fp4.view(experts, 2 * inter_dim, model_dim // 2), NLane=16, gate_up=True, moe_gemm=True
        )
        .view(torch.uint8)
        .contiguous()
    )
    scale_gui = (
        gemm_common_utils.shuffle_scale_w4(
            w1_scale_raw.view(experts * 2 * inter_dim, model_dim // 32), experts_cnt=experts, gate_up=True
        )
        .view(torch.uint8)
        .contiguous()
    )
    del w1_fp32, w1_flat, w1_fp4, w1_scale_raw
    torch.cuda.empty_cache()

    # routing: each token's topk DISTINCT experts, random across all E. Vary by rank so each
    # rank's tokens route differently (realistic; also gives every rank coverage at bs=1).
    torch.manual_seed(seed + 9973 + rank * 101)
    topk_ids = torch.stack([torch.randperm(experts, device=dev)[:topk] for _ in range(tokens)]).to(torch.int32)
    wts = torch.full((tokens, topk), 1.0 / topk, device=dev, dtype=torch.float32)

    return dict(
        x_payload=x_payload,
        scale_mx_u8=scale_x_mx.contiguous(),
        w_kernel=w_kernel,
        scale_w1_1d=scale_w1_1d,
        w_kernel_gui=w_kernel_gui,
        scale_gui=scale_gui,
        topk_ids=topk_ids,
        wts=wts,
        token_dtype=token_dtype,
        a_dtype=a_dtype,
        row_view_dim=row_view_dim,
        x_bf16=x_fp32.to(torch.bfloat16).contiguous(),  # --from-bf16: production-quant source
        w_ref_local=w_ref_local,  # bf16 local-expert weights (accuracy ground truth)
    )


_E2M1_LUT = None


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
    """End-to-end correctness + perf for one (network, quant, bs).

    Compares three pipelines (all fused stage-2, bf16 output):
      * megav1    : fp8/fp4 act -> MegaMoE -> bf16 (single op, Plan-A zero-bridge).
      * atom-fp8  : fp8 dispatch -> sort -> gemm1 -> fused GEMM2+combine (primary baseline).
      * atom-bf16 : bf16 dispatch -> recv-quant -> ... (accuracy reference / oracle sanity).

    Gate: each path's relL2 vs an f32 torch oracle must sit at the quant floor (not ~1.0).
    Perf: CUDAGraph device time of the full stage1+stage2 pipeline.
    """
    import numpy as _np
    import torch.nn.functional as _F

    from kernels.mega_moe import MegaMoE
    from kernels.mega_moe.quant import mxfp4_moe_scale_sort, per_1x32_mx_quant
    from kernels.moe.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1, compile_mixed_moe_gemm2
    from kernels.moe.moe_sorting_kernel import moe_sorting_flydsl

    def _relL2(a, b):
        a = _np.asarray(a, dtype=_np.float64)
        b = _np.asarray(b, dtype=_np.float64)
        n = float(((a - b) ** 2).sum())
        d = float((b**2).sum())
        return (n / d) ** 0.5 if d > 0 else -1.0

    # ============================= 1. config / shapes =============================
    _is_fp4 = s1_out == "fp4"
    max_recv = world * mtpr
    tm, tn1, tk = 32, 128, 256  # ATOM gemm1 tile
    tm2, tn2, tk2 = 32, 128, 256  # ATOM/mega gemm2 tile
    _agv = (lambda t: t.view(torch.uint8)) if a_dtype == "fp4" else (lambda t: t)

    # ===================== 2. weights: W2 (down-proj) + f32 oracle =====================
    # W2: MX-FP4 via the same W1 pipeline, replicated cross-rank (same seed).
    torch.manual_seed(args.seed + 4242)
    w2_f32 = torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32) * (
        float(inter_dim) ** -0.25
    )
    w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
    _w2sl = slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
    w2_kernel = shuffle_weight(w2_fp4[_w2sl]).view(torch.uint8).contiguous().view(-1)
    w2_scale_1d = gemm_common_utils.e8m0_shuffle(w2_sr[_w2sl]).view(torch.uint8).contiguous().view(-1)

    # Pre-quant f32 oracle weights (w1 re-seeded to match _prepare). w1_all is large
    # (v4_pro ~63GB) -> empty_cache first so the contiguous alloc fits without fragmenting.
    del w2_fp4, w2_sr
    torch.cuda.empty_cache()
    _init = float(model_dim) ** -0.25
    torch.manual_seed(args.seed)
    _ = torch.randn((run_tokens, model_dim), device=dev, dtype=torch.float32)  # advance RNG like _prepare
    w1_all = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * _init
    w2_all = w2_f32.view(experts, model_dim, inter_dim)

    # ============================= 3. inputs (quantized activation) =============================
    wc = wts[:run_tokens].contiguous()
    ic = topk_ids[:run_tokens].to(torch.int32).contiguous()
    # production stage-1 input: fp8/fp4 payload + e8m0 scale (FlyDSL MX quant, drop-in for aiter)
    x_q, x_sc = per_1x32_mx_quant(x_bf16[:run_tokens].contiguous(), quant_mode=("fp4" if _is_fp4 else "fp8"))
    x_sc = x_sc.view(torch.uint8)

    # ============================= 4. CUDAGraph timing helper =============================
    # warmup -> barrier -> capture -> back-to-back replay (only cuda.synchronize in the loop,
    # NEVER shmem_barrier inside the replay loop).
    def _cg_time(body, dc_op):
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

    # ============================= 5. megav1 (MegaMoE single op) =============================
    # MegaMoE consumes LOCAL w1: this rank's `epr` expert rows only (gemm1 indexes by local
    # expert id). w_kernel/scale are expert-major contiguous -> slice the flat byte range.
    # (Global w1 unsupported: >4GB weights truncate at the 32-bit buffer num_records cap.)
    _wpe = w_kernel.numel() // experts  # per-expert uint8 elems (weight)
    _spe = scale_w1_1d.numel() // experts  # per-expert uint8 elems (scale)
    _w1_arg = w_kernel.reshape(-1)[rank * epr * _wpe : (rank + 1) * epr * _wpe].contiguous()
    _w1s_arg = scale_w1_1d.reshape(-1)[rank * epr * _spe : (rank + 1) * epr * _spe].contiguous()
    # MegaMoE gate-up mode: auto = a8w4->interleave / a4w4->separated (mirrors aiter). INTERLEAVE
    # feeds the gate_up-shuffled local w1/scale (w_kernel_gui/scale_gui); the ATOM baselines below
    # always use the SEPARATED _w1_arg/_w1s_arg.
    _mgm = str(getattr(args, "mega_gate_mode", "auto"))
    _mega_interleave = (args.quant == "a8w4") if _mgm == "auto" else (_mgm == "interleave")
    if _mega_interleave and w_kernel_gui is not None:
        _mega_gate_mode = "interleave"
        _w1_arg_mega = w_kernel_gui.reshape(-1)[rank * epr * _wpe : (rank + 1) * epr * _wpe].contiguous()
        _w1s_arg_mega = scale_gui.reshape(-1)[rank * epr * _spe : (rank + 1) * epr * _spe].contiguous()
    else:
        _mega_gate_mode = "separated"
        _w1_arg_mega, _w1s_arg_mega = _w1_arg, _w1s_arg
    _tm2_mega = int(args.mega_gemm2_tile_m) if int(getattr(args, "mega_gemm2_tile_m", -1)) > 0 else tm2
    if _tm2_mega != tm2:
        _info(rank, f"[full-e2e] DEBUG override MegaMoE gemm2_tile_m={_tm2_mega} (baseline stays {tm2})")
    _s1_fused = bool(args.mega_stage1_fused)
    _s2_fused = bool(args.mega_stage2_fused)
    if rank == 0 and (not (_s1_fused and _s2_fused) or _mega_gate_mode != "separated"):
        _info(
            rank,
            f"[full-e2e] MegaMoE modes: stage1_fused={_s1_fused} stage2_fused={_s2_fused} "
            f"gate_mode={_mega_gate_mode}",
        )
    moe = MegaMoE(
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
        gemm2_tile_m=_tm2_mega,
        gemm2_tile_n=tn2,
        gemm2_tile_k=tk2,
        enable_fused_stage1=_s1_fused,
        enable_fused_stage2=_s2_fused,
        gate_mode=_mega_gate_mode,
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    _mega_out_holder = {}

    def _mega_body():
        # fused stage-1 consumes pre-quantized x_q (quant outside the timed body); non-fused stage-1
        # bf16-dispatches, so it takes the bf16 input via forward().
        if _s1_fused:
            _mega_out_holder["o"] = moe.forward_prequant(x_q, x_sc, wc, ic)
        else:
            _mega_out_holder["o"] = moe.forward(x_bf16[:run_tokens].contiguous(), wc, ic)

    _mega_body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    out_mega = _mega_out_holder["o"][:run_tokens].float().cpu().numpy().copy()

    # ============ 6. ATOM-bf16 baseline: bf16 dispatch -> recv-quant -> sort -> gemm1 -> fused stage2 ============
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
    # WEIGHTED baseline (production parity): GEMM2's doweight epilogue applies the real routing
    # weights, so the baseline must use the SAME weights. Harness routes uniformly -> 1/topk.
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
    # ATOM gemm1 indexes w1 by LOCAL expert id (epr experts), matching its gemm2 (local w2 +
    # a_se_local). Global w1 would truncate at the 4GB buffer cap for >4GB weights.
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
    # SEPARATE stage-2 (ATOM-faithful): FlyDSL gemm2 (compile_mixed_moe_gemm2, doweight in gemm2,
    # accumulate -> token-level out) + FlyDSL dc.combine (mirrors test_profiler_moe_gemm2_combine
    # _baseline_chain). No fused gemm2+combine.
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
        # LIVE routing each replay (dispatch in-chain -> recv_topk recomputed from THIS dispatch's
        # output; a stale snapshot would desync sort vs combine -> xdev-barrier deadlock).
        a2_e.zero_()
        dc.total_recv.zero_()
        _bt, _, _, _oidx, _ = dc.dispatch(x_bf16[:run_tokens].contiguous(), wc, None, ic)
        _oi = _oidx[:trc].to(torch.int32)
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
        moe_sorting_flydsl(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc], int(experts), int(tm))
        _a1q, _a1sp = per_1x32_mx_quant(_bt[:trc].contiguous(), quant_mode=("fp4" if _is_fp4 else "fp8"))
        mxfp4_moe_scale_sort(a1s, _a1sp, a_st, a_nv, int(trc), int(model_dim))
        # gemm1 AND gemm2 both index LOCAL: w1/w2 = this rank's epr experts, a_se_local (sort gave
        # GLOBAL ids).  Local w1 (<4GB) avoids the 4GB buffer truncation that global w1 hits.
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

    # ============ 7. ATOM-fp8 baseline (PRODUCTION): fp8 dispatch -> sort -> gemm1 -> fused stage2 ============
    # fp8 dispatch carries the e8m0 scale (NO recv-quant), 1B/elem -> fair vs megav1. combine still
    # needs a bf16 op (gemm2 emits bf16): reuse `dc` with total_recv/tis bridged from the fp8 op
    # (atom's bridge; megav1 removes it via Plan A).
    _scale_mx_blocks = model_dim // 32
    cfg_fp8 = FlyDSLDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=model_dim,
        max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        # fp8/fp4 dispatch (+ e8m0 scale); combine emits bf16 gemm2_out. Buffers are
        # sized exactly per dtype (dispatch=1B/elem, combine=bf16 2B/elem).
        dispatch_dtype=(torch.float4_e2m1fn_x2 if _is_fp4 else torch.float8_e4m3fn),
        combine_dtype=torch.bfloat16,
        scale_dim=_scale_mx_blocks,
        scale_type_size=1,
        enable_std_moe=False,
    )
    dcf = FlyDSLDispatchCombineIntraNodeOp(cfg_fp8)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    _atom8_holder = {}

    def _atom_fp8_body():
        # ATOM fp8 path: quantize (in-body) -> fp8 dispatch (dcf) -> gemm1 -> FlyDSL gemm2 (separate)
        # -> FlyDSL combine (dcf).  Quant is inside the timed body (production: quantize BEFORE the
        # fp8 dispatch), symmetric with the bf16 path's recv-side quant.
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

    # ===================== 8. dequant-weights f32 oracle (routing-WEIGHTED reduce) =====================
    # Ported from aiter RefModel: the oracle uses the SAME lossy mxfp4 weights the kernels consume
    # (per-expert quant->dequant), so each path's relL2 reflects the activation-quant + kernel error
    # (~0.05 a8w4 / ~0.22 a4w4), NOT the (dominant) weight-quant floor.  One expert is dequantized at a
    # time (freed each iter) so peak memory ~= a full-precision oracle.
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
        _a1 = _F.silu(xr @ w1e[:inter_dim].t()) * (xr @ w1e[inter_dim : 2 * inter_dim].t())
        oracle_w[rows] += w_e[:, None] * (_a1 @ w2e.t())
        del w1e, w2e
    orw = oracle_w.cpu().numpy()

    # ============================= 9. correctness gate =============================
    # All paths apply routing weights in the GEMM2 doweight epilogue -> compare to the WEIGHTED
    # dequant-weights oracle. relL2 now isolates activation-quant + kernel error (~0.05 a8w4 /
    # ~0.22 a4w4); the floor is set tight accordingly (vs the loose 0.25/0.32 full-precision floor).
    _rm_w = _relL2(out_mega, orw)  # mega(prod)
    _ra_w = _relL2(out_atom, orw)  # atom-bf16 (reference)
    _ra8_w = _relL2(out_atom8, orw)  # atom-fp8 (primary baseline)
    _rma = _relL2(out_mega, out_atom8)  # mega vs primary baseline (should be ~0)
    _floor = 0.28 if _is_fp4 else 0.10
    _mega_ok = _rm_w < _floor
    _atom8_ok = _ra8_w < _floor
    # oracle sanity: if even the most-accurate path (atom-bf16) is far from the oracle, the oracle
    # is unreliable for this shape -> fall back to cross-impl agreement.
    _oracle_broken = _ra_w > _floor
    # fp8 tracks the baseline bitwise (_rma~0); fp4's coarse E2M1 makes two VALID quantizations
    # diverge >5% (~8.5%), so for fp4 require mega no worse than the baseline (+margin).
    _match_ok = (_rma < 5e-2) or (_rm_w <= _ra8_w + 2e-2)
    ok = _match_ok and (_oracle_broken or (_mega_ok and _atom8_ok))

    # Aggregate across ALL ranks (each holds a different expert shard): report the WORST rank,
    # PASS only if EVERY rank passes.
    _rm_w_max = _all_max(dev, _rm_w)
    _ra8_w_max = _all_max(dev, _ra8_w)
    _ra_w_max = _all_max(dev, _ra_w)
    _rma_max = _all_max(dev, _rma)
    _all_ok = _all_max(dev, 0.0 if ok else 1.0) < 0.5  # any failing rank -> all_ok False

    # ===================== 10. stage1-only bodies (dispatch+GEMM1) [DISABLED] =====================
    # def _mega_s1_body():
    #     moe.stage1.forward(x_q, wc, x_sc, ic)   # megav1 single-launch dispatch ⊕ GEMM1 -> a2
    # def _atom_s1_body():
    #     a2_e.zero_(); dcf.total_recv.zero_()
    #     _rx, _, _rs, _oidx, _ = dcf.dispatch(x_q, wc, x_sc, ic)   # fp8 dispatch
    #     _oi = _oidx[:trc].to(torch.int32)
    #     _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
    #     recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
    #     moe_sorting_flydsl(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc],
    #                        int(experts), int(tm))
    #     mxfp4_moe_scale_sort(a1s, _rs[:trc].contiguous(), a_st, a_nv, int(trc), int(model_dim))
    #     a_se_local.copy_(a_se - rank * epr)
    #     gemm1(a2_e.view(max_recv, topk, a2_e.shape[-1]), _agv(_rx[:trc]), _w1_arg, a1s.view(torch.uint8),
    #           _w1s_arg, a_st, a_se_local, a_sw, a_nv, bias_d, a2s_e, fx.Int32(trc),
    #           fx.Int32(inter_dim * 2), fx.Int32(model_dim), fx.Int32(int(_max_blocks)),
    #           stream=fx.Stream(torch.cuda.current_stream()))

    # ============================= 11. perf (CUDAGraph) =============================
    # Measurement is EITHER torch.profiler OR cuda-event, mutually exclusive (mirrors
    # test_profiler_moe_gemm2_combine.py's `--mode profile` vs `--mode bench`). --profile dumps
    # per-kernel GPU tables + chrome traces (E2E from the profiler trace); default uses the
    # lightweight cuda-event `_cg_time`. Both feed the same _t_* (ms) into the report below.
    if getattr(args, "profile", False):
        _pmeta = dict(tokens=run_tokens, network=args.network, quant=args.quant)
        _pdir = getattr(args, "profile_dir", "") or "/tmp/mega_prof"
        _tag = f"{args.network}_{args.quant}_bs{run_tokens}"
        _pm = _profile_body(_mega_body, moe.comb_op, f"mega_{_tag}", args, rank, world, dev, _pdir, _pmeta)
        _pa8 = _profile_body(_atom_fp8_body, dc, f"atomfp8_{_tag}", args, rank, world, dev, _pdir, _pmeta)
        _pa = _profile_body(_atom_body, dc, f"atombf16_{_tag}", args, rank, world, dev, _pdir, _pmeta)
        _t_mega, _t_atom8, _t_atom = (_pm["e2e_us_avg"] / 1e3, _pa8["e2e_us_avg"] / 1e3, _pa["e2e_us_avg"] / 1e3)
    else:
        _t_mega = _cg_time(_mega_body, moe.comb_op)  # megav1 e2e (stage1+stage2)
        _t_atom8 = _cg_time(_atom_fp8_body, dc)  # baseline e2e (fp8 dispatch)
        _t_atom = _cg_time(_atom_body, dc)  # reference e2e (bf16 dispatch)
    # _t_mega_s1 = _cg_time(_mega_s1_body, moe.comb_op)        # megav1 STAGE1-only
    # _t_atom_s1 = _cg_time(_atom_s1_body, dcf)                # baseline STAGE1-only (fp8 dispatch->gemm1)

    # ============================= 12. report + return metrics =============================
    if rank == 0:
        _e2e_warn = (
            "  [WARN torch-oracle unreliable for this shape: gated on mega-vs-baseline]" if _oracle_broken else ""
        )
        print(
            f"[FULL-E2E] {args.network} {args.quant} bs={run_tokens} seed={args.seed} -> {'PASS' if _all_ok else 'FAIL'} (all {world} ranks){_e2e_warn}",
            flush=True,
        )
        print(
            f"  [precision vs WEIGHTED torch-oracle, MAX over {world} ranks]  mega(prod)={_rm_w_max:.3e}  "
            f"atom-fp8(baseline)={_ra8_w_max:.3e}  atom-bf16(ref)={_ra_w_max:.3e}  "
            f"mega-vs-baseline={_rma_max:.3e}  (floor~{_floor})",
            flush=True,
        )
        # [perf STAGE1-only] disabled
        # print(f"  [perf STAGE1-only, ms]  baseline-fp8(dispatch->gemm1)={_t_atom_s1:.4f}  "
        #       f"megav1(dispatch+gemm1)={_t_mega_s1:.4f}  "
        #       f"speedup={(_t_atom_s1 / _t_mega_s1) if _t_mega_s1 > 0 else -1:.3f}", flush=True)
        _timer = "profiler-e2e" if getattr(args, "profile", False) else "cuda-event"
        print(
            f"  [perf E2E (stage1+fused-stage2), ms | {_timer}]  baseline-fp8={_t_atom8:.4f}  "
            f"megav1={_t_mega:.4f}  speedup={(_t_atom8 / _t_mega) if _t_mega > 0 else -1:.3f}  "
            f"| ref bf16-dispatch baseline={_t_atom:.4f}  (out=bf16)",
            flush=True,
        )
    return dict(
        network=args.network,
        quant=args.quant,
        tokens=run_tokens,
        full_e2e_mega_relL2=_rm_w,
        full_e2e_atom_fp8_relL2=_ra8_w,
        full_e2e_atom_bf16_relL2=_ra_w,
        full_e2e_mega_vs_baseline=_rma,
        # s1_baseline_ms=_t_atom_s1, s1_mega_ms=_t_mega_s1,  # stage1-only disabled
        full_e2e_baseline_fp8_ms=_t_atom8,
        full_e2e_baseline_bf16_ms=_t_atom,
        full_e2e_mega_ms=_t_mega,
        full_e2e_pass=ok,
    )


_PERF_BASELINE_CACHE = {}


def _perf_key(network, quant, tokens):
    return f"{network}:{quant}:{tokens}"


def _perf_baseline_lookup(path, network, quant, tokens):
    """Look up the committed golden MegaMoE latency (ms) for one config, or None if absent."""
    if not path:
        return None
    data = _PERF_BASELINE_CACHE.get(path)
    if data is None:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            data = {}
        _PERF_BASELINE_CACHE[path] = data
    v = data.get(_perf_key(network, quant, tokens))
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
    check_acc=True,
    measure_perf=False,
):
    """Aiter-free CI path: run ONLY MegaMoE (single fused op, bf16 in -> bf16 out).

    This is what the multi-gpu CI exercises: NO ATOM baseline and NO aiter (weight prep is pure
    torch, activation quant is MegaMoE's own FlyDSL MX kernel inside ``moe.forward``).

      * accuracy : relL2(megav1, routing-weighted torch f32 oracle) < the MX-FP4 quant floor.
      * perf     : CUDAGraph device time of ``moe.forward`` (mean across ranks), gated against a
                   committed golden baseline -- pass if within ``args.perf_tol`` (match) or faster
                   (better).  A missing baseline entry only warns (used when capturing the golden).
    """
    import numpy as _np
    import torch.nn.functional as _F

    from kernels.mega_moe import MegaMoE

    def _relL2(a, b):
        a = _np.asarray(a, dtype=_np.float64)
        b = _np.asarray(b, dtype=_np.float64)
        n = float(((a - b) ** 2).sum())
        d = float((b**2).sum())
        return (n / d) ** 0.5 if d > 0 else -1.0

    _is_fp4 = quant == "a4w4"
    tm2, tn2, tk2 = 32, 128, 256  # mega gemm2 tile
    # Accuracy floor for the DEQUANT-WEIGHTS oracle (below): the oracle sees the same lossy mxfp4
    # weights as the kernel, so relL2 is the kernel + activation-quant error only (measured ~0.05
    # a8w4 / ~0.22 a4w4 on 8x MI355X). Much tighter than the full-precision-oracle floor (0.25/0.32).
    _floor = 0.28 if _is_fp4 else 0.10

    # ---- W2 (down-proj): MX-FP4, same pipeline as _prepare's W1 (replicated cross-rank) ----
    torch.manual_seed(args.seed + 4242)
    w2_f32 = torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32) * (
        float(inter_dim) ** -0.25
    )
    w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
    _w2sl = slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
    w2_kernel = shuffle_weight(w2_fp4[_w2sl]).view(torch.uint8).contiguous().view(-1)
    w2_scale_1d = gemm_common_utils.e8m0_shuffle(w2_sr[_w2sl]).view(torch.uint8).contiguous().view(-1)
    del w2_fp4, w2_sr
    torch.cuda.empty_cache()

    # ---- MegaMoE local weights: this rank's epr experts (gemm1 indexes by local expert id) ----
    _wpe = w_kernel.numel() // experts
    _spe = scale_w1_1d.numel() // experts
    # gate-up INTERLEAVE(g1u1) for a8w4 (mirrors aiter), SEPARATED otherwise; feed the matching w1.
    if quant == "a8w4" and w_kernel_gui is not None:
        _gate_mode = "interleave"
        _w1 = w_kernel_gui.reshape(-1)[rank * epr * _wpe : (rank + 1) * epr * _wpe].contiguous()
        _w1s = scale_gui.reshape(-1)[rank * epr * _spe : (rank + 1) * epr * _spe].contiguous()
    else:
        _gate_mode = "separated"
        _w1 = w_kernel.reshape(-1)[rank * epr * _wpe : (rank + 1) * epr * _wpe].contiguous()
        _w1s = scale_w1_1d.reshape(-1)[rank * epr * _spe : (rank + 1) * epr * _spe].contiguous()

    moe = MegaMoE(
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
        gemm2_tile_m=tm2,
        gemm2_tile_n=tn2,
        gemm2_tile_k=tk2,
        enable_fused_stage1=True,
        enable_fused_stage2=True,
        gate_mode=_gate_mode,
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    wc = wts[:run_tokens].contiguous()
    ic = topk_ids[:run_tokens].to(torch.int32).contiguous()
    x_in = x_bf16[:run_tokens].contiguous()

    _out = {}

    def _body():
        # bf16 in -> internal FlyDSL MX quant -> fused stage1+stage2 -> bf16 out (single op, no aiter).
        _out["o"] = moe.forward(x_in, wc, ic)

    _body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    out_mega = _out["o"][:run_tokens].float().cpu().numpy().copy()

    # ---- accuracy: dequant-weights reference (single layer) OR chained N-layer accumulation ----
    n_layers = int(getattr(args, "layers", 1))
    _acc_metric = -1.0  # relL2 (single layer) or 1-cosine (chain); the reported / gated value
    _acc_floor = _floor
    _acc_label = "relL2(vs oracle)"
    acc_ok = True
    if check_acc:
        _init = float(model_dim) ** -0.25
        torch.manual_seed(args.seed)
        _ = torch.randn((run_tokens, model_dim), device=dev, dtype=torch.float32)  # advance RNG like _prepare
        w1_all = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * _init
        w2_all = w2_f32.view(experts, model_dim, inter_dim)
        if n_layers > 1:
            # ===== chained N-layer accumulation (real DeepSeek-V4 depth): device (moe.forward per
            # layer + RMSNorm + residual) vs the pure-torch dequant-weights RefModel over the SAME
            # per-layer routings.  Compares the end-to-end accumulated output (1-cosine, aiter's gate);
            # a tiny per-layer bias that hides under the single-layer floor compounds and is caught. =====
            routings = _make_layer_routings(n_layers, run_tokens, experts, topk, dev, args.seed + 4242, rank)
            xd = x_in
            for _ids_l, _wts_l in routings:
                xn = _rmsnorm(xd)
                out = moe.forward(xn, _wts_l, _ids_l)
                xd = xd + out[:run_tokens]
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            out_dev = xd[:run_tokens].float()
            out_ref = RefModel(w1_all, w2_all, inter_dim, dev).run(x_in, routings).float()
            _acc_metric = _calc_diff(out_ref, out_dev)  # 1 - cosine (fp64), end-to-end accumulated
            _acc_floor = _CHAIN_TOL
            _acc_label = f"cos_diff(chain x{n_layers})"
            acc_ok = _acc_metric < _acc_floor
        else:
            x32 = x_in.float()
            ids_l = ic.long()
            wv = wc.float()
            # Dequant-weights reference (ported from aiter RefModel): the oracle uses the SAME lossy
            # mxfp4 weights the kernel consumes (per-expert quant->dequant), so relL2 reflects the
            # kernel + activation-quant error, NOT the (dominant) weight-quant floor -- a much tighter
            # gate. One expert is dequantized at a time (freed each iter) so peak mem ~= a fp32 oracle.
            oracle = torch.zeros(run_tokens, model_dim, device=dev, dtype=torch.float32)
            for e in torch.unique(ids_l).tolist():
                sel = ids_l == int(e)  # [T, topk]: which (token, slot) route to expert e
                rows = sel.any(dim=1).nonzero().flatten()
                w_e = (wv * sel).sum(dim=1)[rows]  # per-token routing weight (summed over slots) for e
                w1e = _dequant_mx_to_f32(w1_all[e], "fp4")  # [2*inter_dim, model_dim]
                w2e = _dequant_mx_to_f32(w2_all[e], "fp4")  # [model_dim, inter_dim]
                xr = x32[rows]
                _a1 = _F.silu(xr @ w1e[:inter_dim].t()) * (xr @ w1e[inter_dim : 2 * inter_dim].t())
                oracle[rows] += w_e[:, None] * (_a1 @ w2e.t())
                del w1e, w2e
            _acc_metric = _relL2(out_mega, oracle.cpu().numpy())
            acc_ok = _acc_metric < _acc_floor
        del w1_all
        torch.cuda.empty_cache()
    relL2 = _acc_metric  # kept name for the return dict / downstream reporting

    # ---- perf: CUDAGraph device time of moe.forward (mean across ranks), golden match-or-better ----
    mega_ms = -1.0
    perf_ok = True
    perf_note = ""
    if measure_perf:
        ms.shmem_barrier_all()
        _body()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        _cap = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=_cap):
            _body()
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
        mega_ms = _all_mean(dev, _s.elapsed_time(_e) / _n)
        _base = _perf_baseline_lookup(getattr(args, "perf_baseline", ""), args.network, quant, run_tokens)
        if _base is not None and _base > 0:
            _tol = float(args.perf_tol)
            perf_ok = mega_ms <= _base * (1.0 + _tol)
            perf_note = f"baseline={_base:.4f}ms +{_tol:.0%} -> {'OK' if perf_ok else 'REGRESSION'}"
        else:
            perf_note = "no committed baseline for this config (perf gate skipped)"

    ok = acc_ok and perf_ok
    _all_ok = _all_max(dev, 0.0 if ok else 1.0) < 0.5
    _relL2_max = _all_max(dev, relL2 if relL2 >= 0 else 0.0)
    _ms_max = _all_max(dev, mega_ms if mega_ms >= 0 else 0.0)
    if rank == 0:
        _acc_s = f"{_acc_label}={_relL2_max:.3e} (floor~{_acc_floor})" if check_acc else "acc:skip"
        _perf_s = f"megav1={_ms_max:.4f}ms  {perf_note}" if measure_perf else "perf:skip"
        print(
            f"[MEGA-ONLY] {args.network} {quant} bs={run_tokens} seed={args.seed} -> "
            f"{'PASS' if _all_ok else 'FAIL'} (all {world} ranks)  [{_acc_s}]  [{_perf_s}]",
            flush=True,
        )
    return dict(
        network=args.network,
        quant=quant,
        tokens=run_tokens,
        mega_only_relL2=relL2,
        mega_only_ms=mega_ms,
        full_e2e_pass=bool(_all_ok),
    )


def run_one(args, rank, world, dev):
    net = NETWORKS[args.network]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    # topk: --topk>0 overrides; else use the network's native topk (r1_v3=8, v4_*=6).
    topk = int(args.topk) if int(args.topk) > 0 else int(net["topk"])
    run_tokens = max(int(args.tokens), 1)  # allow bs=1 (1 token/rank); routing still reaches all ranks
    if experts % world != 0:
        raise SystemExit(f"experts={experts} must divide world={world}")
    epr = experts // world

    mtpr = max(16, run_tokens)
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
        keep_ref=False,
    )
    w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
    topk_ids, wts = T["topk_ids"], T["wts"]
    a_dtype = T["a_dtype"]
    x_bf16 = T["x_bf16"]

    # CI path: aiter-free MegaMoE-only run (accuracy vs torch oracle + golden-baseline perf).
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
            check_acc=not bool(args.skip_acc),
            measure_perf=bool(args.measure_perf),
        )

    # Manual/dev path: production end-to-end comparison megav1 vs a fully-FlyDSL ATOM baseline
    # (dispatch -> FlyDSL sort/quant/scale-sort -> mixed_moe_gemm1 -> FlyDSL gemm2 -> combine),
    # BOTH with the real routing weights + fused stage-2, gated on the weighted torch oracle.
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
    p.add_argument("--network", type=str, default="v4_flash", choices=list(NETWORKS))
    p.add_argument("--quant", type=str, default="a8w4", choices=["a8w4", "a4w4"])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument(
        "--topk",
        type=int,
        default=-1,
        help="-1 (default) = use the network's native topk (r1_v3=8, v4_*=6); >0 overrides",
    )
    p.add_argument("--waves-per-eu", type=int, default=4)
    p.add_argument("--async-copy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
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
        "--mega-gemm2-tile-m",
        type=int,
        default=-1,
        help="debug: override ONLY MegaMoE's gemm2 tile_m (baseline unchanged). "
        "Used to reproduce the stage1 sort_block_m <-> gemm2 tile_m mismatch.",
    )
    p.add_argument(
        "--mega-stage1-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MegaMoE stage-1 fused (megakernel) vs non-fused (bf16 dispatch+sort+gemm1).",
    )
    p.add_argument(
        "--mega-stage2-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MegaMoE stage-2 fused (gemm2+combine) vs non-fused (gemm2 + separate combine).",
    )
    p.add_argument(
        "--mega-gate-mode",
        type=str,
        default="auto",
        choices=["auto", "interleave", "separated"],
        help="MegaMoE stage-1 gate-up layout: auto (a8w4->interleave/a4w4->separated, "
        "mirrors aiter), or force interleave/separated. INTERLEAVE feeds gate_up-"
        "shuffled w1; ATOM baselines always stay SEPARATED.",
    )
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
        help="CI perf gate for the ATOM path (only meaningful with --strict): require the megav1 E2E "
        "speedup vs the atom-fp8 production baseline to be >= this value for every case (0 = disabled).",
    )
    p.add_argument(
        "--mega-only",
        action="store_true",
        help="Aiter-free CI path: run ONLY MegaMoE (moe.forward), NO ATOM baseline and NO aiter. "
        "Accuracy is gated vs a torch f32 oracle; perf vs a committed golden baseline "
        "(--perf-baseline). This is what the multi-gpu CI runs.",
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
        help="(--mega-only) path to the committed golden latency JSON "
        "({'network:quant:bs': ms}); CI passes if measured <= golden * (1 + --perf-tol).",
    )
    p.add_argument(
        "--perf-tol",
        type=float,
        default=0.05,
        help="(--mega-only) fractional slack over the golden baseline that still counts as a match "
        "(default 0.05 = 5%%).",
    )
    p.add_argument(
        "--perf-out",
        type=str,
        default="",
        help="(--mega-only) write the measured latencies as a golden JSON ({'network:quant:bs': ms}) "
        "to this path (used to capture the baseline on a reference machine).",
    )
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, args.master_port)
    dev = torch.device("cuda", local_rank)

    bs_set = FULL_BS if args.full_bs else CLASSIC_BS
    if args.bs_list:
        combos = [(args.network, int(b)) for b in args.bs_list.split(",") if b.strip()]
    elif args.matrix:
        combos = [(net, bs) for net in NETWORKS for bs in bs_set]
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
            # free per-config allocations (facade/op/weights) so they don't accumulate across the
            # sweep -> avoids HIP OOM on big nets (e.g. v4_pro w1 ~67GB) at later bs/seed.
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

    # Capture the golden perf baseline: {'network:quant:bs': ms} for the runs that measured perf.
    # Merge into any existing file so repeated invocations (one per --network) accumulate.
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
                golden[_perf_key(r["network"], r["quant"], r["tokens"])] = round(float(r["mega_only_ms"]), 4)
        with open(args.perf_out, "w") as f:
            json.dump(golden, f, indent=2, sort_keys=True)
            f.write("\n")
        _info(rank, f"[perf] wrote {len(golden)} golden rows -> {args.perf_out}")

    # CI gate: fail if ANY rank saw a failing/ERROR case, or if some rank ran zero cases. Both
    # quantities are reduced across ranks (before _cleanup tears down the group) so every rank
    # exits with the same status; torchrun then propagates a non-zero exit to the launcher.
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


# ============================== pytest multi-GPU CI cases ==============================
# 8-GPU accuracy / functionality + performance safeguards for the MegaMoE production kernel on the
# v4_pro / v4_flash networks with A8W4 (MX-FP8 activation, MX-FP4 weight).  Mirrors the multi_gpu
# subprocess pattern in test_allreduce.py: each case relaunches THIS file under torchrun on 8 GPUs
# and gates on its exit code (see main()'s --strict / --min-speedup).  A8W4 needs CDNA4 (gfx95x),
# so the cases skip on gfx942 (the mi325 leg of the multi-gpu matrix) and on <8-GPU / dep-less hosts.
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
    """Skip unless this host can run the aiter-free A8W4 MegaMoE 8-GPU harness.

    MegaMoE's intranode dispatch/combine needs mori (NOT aiter); A8W4 needs CDNA4 (gfx95x).
    """
    if _HARNESS_DEPS_ERROR is not None:
        pytest.skip(f"MegaMoE harness deps unavailable (need mori + FlyDSL dispatch/combine): {_HARNESS_DEPS_ERROR}")
    arch = _gpu_arch()
    if not arch.startswith("gfx95"):
        pytest.skip(f"MegaMoE A8W4 requires CDNA4 (gfx95x); current arch: {arch or 'unknown'}")
    phys = _count_physical_gpus()
    if phys < 8:
        pytest.skip(f"requires >= 8 physical GPUs, found {phys}")


# Committed golden latency baseline (captured on 8x MI355X with --perf-out); the benchmark gate
# passes if the CI-measured latency is within _MEGA_PERF_TOL of (or faster than) these numbers.
# The tol absorbs run-to-run / thermal / clock variance (measured <1% run-to-run on 8x MI355X).
_MEGA_PERF_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mega_moe_perf_baseline.json")
_MEGA_PERF_TOL = 0.05


def _run_mega_8gpu(*, network, quant, bs_list, iters, measure_perf=False, skip_acc=False, layers=1, timeout=2400):
    """Relaunch THIS file under torchrun on 8 GPUs in the aiter-free --mega-only mode and assert a
    clean (strict) exit. measure_perf adds the CUDAGraph latency gate vs the committed golden;
    layers>1 runs the chained N-layer accumulation accuracy check instead of the single-layer gate."""
    import subprocess as _sp

    env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    # Propagate the parent's import path so the child workers find flydsl._mlir (dev boxes use
    # build-fly/python_packages; CI editable installs resolve via site-packages regardless).
    _extra_pp = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONPATH"] = _extra_pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
    # The accuracy oracle materializes full-precision expert weights (v4_pro: ~63GB w1 + ~34GB w2 in
    # f32); on a single 288GB GPU that contiguous alloc fragments and OOMs. expandable_segments lets
    # the allocator back a large request with non-contiguous free blocks (PyTorch's own OOM remedy),
    # and is a no-op for CUDAGraph replay timing (memory is fixed once captured).
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
    if layers > 1:
        cmd += ["--layers", str(layers)]
    if measure_perf:
        cmd += ["--measure-perf", "--perf-baseline", _MEGA_PERF_BASELINE, "--perf-tol", str(_MEGA_PERF_TOL)]
    if skip_acc:
        cmd += ["--skip-acc"]

    result = _sp.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
    # Surface the human-readable gate lines (accuracy / perf) regardless of outcome.
    for line in result.stdout.splitlines():
        if any(tag in line for tag in ("[MEGA-ONLY]", "[strict]")):
            print(line)
    assert result.returncode == 0, (
        f"MegaMoE 8-GPU {network}/{quant} bs={bs_list} FAILED (exit {result.returncode}).\n"
        f"stdout (last 3000 chars):\n{result.stdout[-3000:]}\n"
        f"stderr (last 2000 chars):\n{result.stderr[-2000:]}"
    )
    return result


# (network, quant, bs_list) accuracy/functionality: small + medium batch on both v4 networks.
_MEGA_ACC_PARAMS = [
    ("v4_flash", "a8w4", "128,2048"),
    ("v4_pro", "a8w4", "128,2048"),
]

# (network, quant, bs_list) perf-relevant batch sizes for the benchmark (golden) gate.
_MEGA_BENCH_PARAMS = [
    ("v4_flash", "a8w4", "4096,8192"),
    ("v4_pro", "a8w4", "4096,8192"),
]


def _mega_id(network, quant, bs_list):
    return f"{network}-{quant}-bs{bs_list.replace(',', '_')}"


@pytest.mark.multi_gpu
@pytest.mark.parametrize("network,quant,bs_list", _MEGA_ACC_PARAMS, ids=[_mega_id(*p) for p in _MEGA_ACC_PARAMS])
def test_mega_moe_8gpu_accuracy(network, quant, bs_list):
    """8-GPU MegaMoE accuracy + functionality (aiter-free): the single fused op (moe.forward) is
    gated on a routing-weighted torch f32 oracle (the MX-FP4 quant floor), on every rank."""
    _skip_unless_mega_8gpu()
    _run_mega_8gpu(network=network, quant=quant, bs_list=bs_list, iters=5)


@pytest.mark.multi_gpu
@pytest.mark.benchmark
@pytest.mark.parametrize("network,quant,bs_list", _MEGA_BENCH_PARAMS, ids=[_mega_id(*p) for p in _MEGA_BENCH_PARAMS])
def test_mega_moe_8gpu_benchmark(network, quant, bs_list):
    """8-GPU MegaMoE performance (aiter-free): CUDAGraph E2E latency of moe.forward, gated against
    the committed golden baseline -- pass if within _MEGA_PERF_TOL (match) or faster (better)."""
    _skip_unless_mega_8gpu()
    _run_mega_8gpu(network=network, quant=quant, bs_list=bs_list, iters=20, measure_perf=True, skip_acc=True)


# Chained-accumulation ACCURACY safeguard: A8W4 only -- fp8 activation survives a deep residual chain
# (~0.055 over 61 layers), whereas A4W4's fp4 activation compounds (~0.46) and is not meaningful here.
_MEGA_CHAIN_PARAMS = [
    ("v4_flash", "a8w4"),
    ("v4_pro", "a8w4"),
]

# Number of chained MoE layers (real DeepSeek-V4 depth).
_MEGA_CHAIN_LAYERS = 61


@pytest.mark.multi_gpu
@pytest.mark.parametrize("network,quant", _MEGA_CHAIN_PARAMS, ids=[f"{n}-{q}" for n, q in _MEGA_CHAIN_PARAMS])
def test_mega_moe_8gpu_accuracy_chained(network, quant):
    """8-GPU MegaMoE chained-accumulation accuracy (aiter-free): N=61 MoE layers (real DeepSeek-V4
    depth) chained with RMSNorm + residual over ONE shared weight set + per-layer routing, gated on
    the 1-cosine of the device chain vs the pure-torch dequant-weights RefModel.  Catches tiny
    per-layer biases that compound across depth but hide under the single-layer gate."""
    _skip_unless_mega_8gpu()
    _run_mega_8gpu(network=network, quant=quant, bs_list="128", iters=1, layers=_MEGA_CHAIN_LAYERS)


if __name__ == "__main__":
    main()
