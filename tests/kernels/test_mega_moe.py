#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Intranode MoE end-to-end benchmark: MegaMoE (production) vs the ATOM production stack.

Single production-equivalent comparison (`_run_full_e2e`), per batch size:

  * megav1   : fp8/fp4 act -> MegaMoE.forward -> bf16   (single-op, Plan-A zero-bridge dispatch).
  * atom-fp8 : PRODUCTION fp8-dispatch -> aiter.moe_sorting_fwd -> aiter.mxfp4_moe_sort_hip ->
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
import mori.shmem as ms  # noqa: E402

from kernels.dispatch_combine_intranode_op import (  # noqa: E402
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)

import tests.kernels.test_moe_gemm as tmg  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402  (weight e8m0 shuffle)
from tests.utils import shuffle_weight  # noqa: E402

try:
    import aiter  # noqa: E402
    HAS_AITER = bool(getattr(aiter, "moe_sorting_fwd", None)) and bool(getattr(aiter, "mxfp4_moe_sort_hip", None))
except Exception:  # noqa: BLE001
    HAS_AITER = False


NETWORKS = {
    "r1_v3":    dict(model_dim=7168, inter_dim=2048, experts=256, topk=8),
    "v4_flash": dict(model_dim=4096, inter_dim=2048, experts=256, topk=6),
    "v4_pro":   dict(model_dim=7168, inter_dim=3072, experts=384, topk=6),
}

# batch-size sweeps for --matrix / --full-bs.
CLASSIC_BS = [1, 8, 64, 512, 8192, 32768]
FULL_BS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def _info(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def _setup_dist(rank: int, world_size: int, master_port: int) -> int:
    if "LOCAL_RANK" not in os.environ:
        os.environ.update({
            "LOCAL_RANK": str(rank), "RANK": str(rank), "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(master_port),
        })
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank,
                                world_size=world_size, device_id=dev)
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
        record_shapes=False, with_stack=False,
        schedule=torch.profiler.schedule(wait=1, warmup=prof_warmup, active=active_iters, repeat=1),
    )


def _kernel_table_from_trace(trace_path: str, op_tag: str, active_iters: int, skip_first: int):
    """Aggregate per-kernel GPU us/replay + E2E replay us from a chrome trace (valid window only)."""
    with open(trace_path) as f:
        tr = json.load(f)
    ev = tr["traceEvents"]
    kernel_events = [e for e in ev if e.get("cat") == "kernel"]
    cg_label = f"{op_tag}::cudagraph_replay"
    cg = sorted([e for e in ev
                 if e.get("cat") == "gpu_user_annotation" and cg_label in e.get("name", "")],
                key=lambda e: e["ts"])[-active_iters:]
    cg = cg[skip_first:]
    valid = max(1, len(cg))
    if cg:
        t0 = cg[0]["ts"]; t1 = cg[-1]["ts"] + cg[-1]["dur"]
        win = [e for e in kernel_events if t0 <= e["ts"] <= t1]
        e2e = sum(e["dur"] for e in cg) / valid
    else:
        win = kernel_events; e2e = 0.0
    agg: dict = {}
    for e in win:
        n = e.get("name", "?"); a = agg.setdefault(n, [0, 0.0])
        a[0] += 1; a[1] += e["dur"]
    rows = sorted([(n, c / valid, tot / valid) for n, (c, tot) in agg.items()],
                  key=lambda r: r[2], reverse=True)
    return rows, e2e, valid


def _profile_body(body, dc_op, op_tag, args, rank, world, dev, out_dir, meta):
    """Capture `body` into a CUDAGraph (same sequence as _cg_time), replay under torch.profiler,
    dump chrome trace, and print (rank0) a per-kernel GPU table + cross-rank E2E replay time."""
    ms.shmem_barrier_all()
    body(); torch.cuda.synchronize(); ms.shmem_barrier_all()   # eager warmup (jit)
    _cap = torch.cuda.Stream(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=_cap):
        body()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()

    iters = max(1, int(args.iters)); prof_warmup = 5; skip_first = min(5, iters - 1) if iters > 1 else 0
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
    s = loc.clone(); dist.all_reduce(s, op=dist.ReduceOp.SUM)
    mx = loc.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mn = loc.clone(); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    if rank == 0:
        sep = "=" * 80
        print(f"\n{sep}")
        print(f"  PROFILE {op_tag}  EP={world}  bs={meta.get('tokens')}  "
              f"net={meta.get('network')}  quant={meta.get('quant')}  ({valid} valid iters)")
        print(f"  E2E replay us/iter (avg/min/max across {world} ranks): "
              f"{s.item()/world:.1f} / {mn.item():.1f} / {mx.item():.1f}")
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
        y[st:en].copy_(yc); s[st:en].copy_(sc)
        del yc, sc; torch.cuda.empty_cache()
    return y, s


def _prepare(dev, *, quant, tokens, model_dim, inter_dim, experts, topk, seed, rank=0,
             world=1, keep_ref=False):
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
        x_payload = x_q.contiguous()              # [tokens, model_dim] fp8_e4m3fn
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
        w_ref_local = w1_fp32[rank * _epr:(rank + 1) * _epr].to(torch.bfloat16).contiguous()

    # weight: MX-FP4 + shuffle for the GEMM (shared across a8w4/a4w4)
    w1_flat = w1_fp32.view(experts * (2 * inter_dim), model_dim)
    w1_fp4, w1_scale_raw = _chunked_fp4_quant(w1_flat)
    w_kernel = shuffle_weight(w1_fp4.view(torch.float4_e2m1fn_x2)).view(torch.uint8).contiguous()
    scale_w1_1d = fp4_utils.e8m0_shuffle(w1_scale_raw).view(torch.uint8).contiguous()
    del w1_fp32, w1_flat, w1_fp4, w1_scale_raw
    torch.cuda.empty_cache()

    # routing: each token's topk DISTINCT experts, random across all E. Vary by rank so each
    # rank's tokens route differently (realistic; also gives every rank coverage at bs=1).
    torch.manual_seed(seed + 9973 + rank * 101)
    topk_ids = torch.stack([
        torch.randperm(experts, device=dev)[:topk] for _ in range(tokens)
    ]).to(torch.int32)
    wts = torch.full((tokens, topk), 1.0 / topk, device=dev, dtype=torch.float32)

    return dict(
        x_payload=x_payload, scale_mx_u8=scale_x_mx.contiguous(), w_kernel=w_kernel,
        scale_w1_1d=scale_w1_1d, topk_ids=topk_ids, wts=wts,
        token_dtype=token_dtype, a_dtype=a_dtype, row_view_dim=row_view_dim,
        x_bf16=x_fp32.to(torch.bfloat16).contiguous(),  # --from-bf16: production-quant source
        w_ref_local=w_ref_local,                        # bf16 local-expert weights (accuracy ground truth)
    )


_E2M1_LUT = None


def _run_full_e2e(args, rank, world, dev, *, model_dim, inter_dim, experts, epr, topk,
                  run_tokens, mtpr, a_dtype, s1_out, w_kernel, scale_w1_1d, x_bf16, topk_ids, wts):
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
    from aiter import dtypes as _adt
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from kernels.mega_moe import MegaMoE
    from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1, compile_mixed_moe_gemm2

    def _relL2(a, b):
        a = _np.asarray(a, dtype=_np.float64); b = _np.asarray(b, dtype=_np.float64)
        n = float(((a - b) ** 2).sum()); d = float((b ** 2).sum())
        return (n / d) ** 0.5 if d > 0 else -1.0

    # ============================= 1. config / shapes =============================
    _is_fp4 = (s1_out == "fp4")
    max_recv = world * mtpr
    tm, tn1, tk = 32, 128, 256          # ATOM gemm1 tile
    tm2, tn2, tk2 = 32, 128, 256        # ATOM/mega gemm2 tile
    _agv = (lambda t: t.view(torch.uint8)) if a_dtype == "fp4" else (lambda t: t)

    # ===================== 2. weights: W2 (down-proj) + f32 oracle =====================
    # W2: MX-FP4 via the same W1 pipeline, replicated cross-rank (same seed).
    torch.manual_seed(args.seed + 4242)
    w2_f32 = (torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32)
              * (float(inter_dim) ** -0.25))
    w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
    _w2sl = slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
    w2_kernel = shuffle_weight(w2_fp4[_w2sl]).view(torch.uint8).contiguous().view(-1)
    w2_scale_1d = fp4_utils.e8m0_shuffle(w2_sr[_w2sl]).view(torch.uint8).contiguous().view(-1)

    # Pre-quant f32 oracle weights (w1 re-seeded to match _prepare). w1_all is large
    # (v4_pro ~63GB) -> empty_cache first so the contiguous alloc fits without fragmenting.
    del w2_fp4, w2_sr
    torch.cuda.empty_cache()
    _init = float(model_dim) ** -0.25
    torch.manual_seed(args.seed)
    _ = torch.randn((run_tokens, model_dim), device=dev, dtype=torch.float32)  # advance RNG like _prepare
    w1_all = (torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * _init)
    w2_all = w2_f32.view(experts, model_dim, inter_dim)

    # ============================= 3. inputs (quantized activation) =============================
    wc = wts[:run_tokens].contiguous()
    ic = topk_ids[:run_tokens].to(torch.int32).contiguous()
    # production stage-1 input: fp8/fp4 payload + e8m0 scale
    if _is_fp4:
        x_q, x_sc = per_1x32_mx_quant_hip(x_bf16[:run_tokens].contiguous(), quant_dtype=_adt.fp4x2)
    else:
        x_q, x_sc = per_1x32_mx_quant_hip(x_bf16[:run_tokens].contiguous(),
                                          quant_dtype=_adt.fp8, scale_type=_adt.fp8_e8m0)
    x_sc = x_sc.view(torch.uint8)

    # ============================= 4. CUDAGraph timing helper =============================
    # warmup -> barrier -> capture -> back-to-back replay (only cuda.synchronize in the loop,
    # NEVER shmem_barrier inside the replay loop).
    def _cg_time(body, dc_op):
        ms.shmem_barrier_all()
        body(); torch.cuda.synchronize(); ms.shmem_barrier_all()   # warmup (jit)
        _cap = torch.cuda.Stream(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=_cap):
            body()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        _n = max(1, int(args.iters))
        _s = torch.cuda.Event(enable_timing=True); _e = torch.cuda.Event(enable_timing=True)
        _s.record()
        for _ in range(_n):
            g.replay()
        _e.record(); torch.cuda.synchronize()
        return _all_mean(dev, _s.elapsed_time(_e) / _n)

    # ============================= 5. megav1 (MegaMoE single op) =============================
    # MegaMoE consumes LOCAL w1: this rank's `epr` expert rows only (gemm1 indexes by local
    # expert id). w_kernel/scale are expert-major contiguous -> slice the flat byte range.
    # (Global w1 unsupported: >4GB weights truncate at the 32-bit buffer num_records cap.)
    _wpe = w_kernel.numel() // experts          # per-expert uint8 elems (weight)
    _spe = scale_w1_1d.numel() // experts       # per-expert uint8 elems (scale)
    _w1_arg = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
    _w1s_arg = scale_w1_1d.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()
    _tm2_mega = int(args.mega_gemm2_tile_m) if int(getattr(args, "mega_gemm2_tile_m", -1)) > 0 else tm2
    if _tm2_mega != tm2:
        _info(rank, f"[full-e2e] DEBUG override MegaMoE gemm2_tile_m={_tm2_mega} (baseline stays {tm2})")
    _s1_fused = bool(args.mega_stage1_fused)
    _s2_fused = bool(args.mega_stage2_fused)
    if rank == 0 and not (_s1_fused and _s2_fused):
        _info(rank, f"[full-e2e] MegaMoE modes: stage1_fused={_s1_fused} stage2_fused={_s2_fused}")
    moe = MegaMoE(
        rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk, quant=args.quant, w1=_w1_arg, w1_scale=_w1s_arg,
        w2=w2_kernel, w2_scale=w2_scale_1d, max_tok_per_rank=mtpr, network=args.network,
        gemm2_tile_m=_tm2_mega, gemm2_tile_n=tn2, gemm2_tile_k=tk2,
        enable_fused_stage1=_s1_fused, enable_fused_stage2=_s2_fused)
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    _mega_out_holder = {}
    def _mega_body():
        # fused stage-1 consumes pre-quantized x_q (quant outside the timed body); non-fused stage-1
        # bf16-dispatches, so it takes the bf16 input via forward().
        if _s1_fused:
            _mega_out_holder["o"] = moe.forward_prequant(x_q, x_sc, wc, ic)
        else:
            _mega_out_holder["o"] = moe.forward(x_bf16[:run_tokens].contiguous(), wc, ic)
    _mega_body(); torch.cuda.synchronize(); ms.shmem_barrier_all()
    out_mega = _mega_out_holder["o"][:run_tokens].float().cpu().numpy().copy()

    # ============ 6. ATOM-bf16 baseline: bf16 dispatch -> recv-quant -> sort -> gemm1 -> fused stage2 ============
    cfg_a = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=world, hidden_dim=model_dim, max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr, num_experts_per_token=topk, data_type=torch.bfloat16,
        scale_dim=0, scale_type_size=0,
        enable_std_moe=False)
    dc = FlyDSLDispatchCombineIntraNodeOp(cfg_a)
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    # one setup dispatch to fix trc (constant for fixed routing) + populate routing tables.
    dc.total_recv.zero_()
    _bt0, _, _, _oidx0, _ = dc.dispatch(x_bf16[:run_tokens].contiguous(), wc, None, ic)
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    trc = max(1, int(dc.total_recv.item()))
    if _all_min_int(dev, trc) <= 0:
        _info(rank, "[full-e2e] some rank got 0 recv; skipping"); return None

    _max_pad = max_recv * topk + experts * tm
    _max_blocks = (_max_pad + tm - 1) // tm
    _scaleN_pad = ((model_dim // 32 + 7) // 8) * 8
    a_st = torch.empty(_max_pad, dtype=torch.int32, device=dev)
    a_sw = torch.empty(_max_pad, dtype=torch.float32, device=dev)
    a_se = torch.empty(_max_blocks, dtype=torch.int32, device=dev)
    a_se_local = torch.empty(_max_blocks, dtype=torch.int32, device=dev)  # gemm2 wants LOCAL expert ids
    a_nv = torch.zeros(2, dtype=torch.int32, device=dev)
    a_mbuf = torch.empty((max_recv, model_dim), dtype=torch.float16, device=dev)
    a1s = torch.empty(((_max_pad + 31) // 32 * 32, _scaleN_pad), dtype=_adt.fp8_e8m0, device=dev)
    # WEIGHTED baseline (production parity): GEMM2's doweight epilogue applies the real routing
    # weights, so the baseline must use the SAME weights. Harness routes uniformly -> 1/topk.
    recv_wts = torch.full((max_recv, topk), 1.0 / topk, device=dev, dtype=torch.float32)
    recv_topk = torch.empty((max_recv, topk), dtype=torch.int32, device=dev)
    _sentinel = torch.full((trc, topk), experts, dtype=torch.int32, device=dev)
    if _is_fp4:
        a2_e = torch.zeros((max_recv * topk, inter_dim // 2), dtype=torch.uint8, device=dev)
    else:
        a2_e = torch.zeros((max_recv * topk, inter_dim), dtype=torch.float8_e4m3fn, device=dev)
    _sbm = max(32, tm); _pr = ((_max_blocks * _sbm + 255) // 256) * 256
    _pc = (((inter_dim // 32) + 7) // 8) * 8
    a2s_e = torch.zeros(_pr * _pc + inter_dim, dtype=torch.uint8, device=dev)
    bias_d = torch.empty((0,), device=dev, dtype=torch.float32)
    _qd = _adt.fp4x2 if _is_fp4 else _adt.fp8
    _stp = None if _is_fp4 else _adt.fp8_e8m0
    # ATOM gemm1 indexes w1 by LOCAL expert id (epr experts), matching its gemm2 (local w2 +
    # a_se_local). Global w1 would truncate at the 4GB buffer cap for >4GB weights.
    gemm1 = compile_mixed_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=epr, topk=topk,
        tile_m=tm, tile_n=tn1, tile_k=tk, doweight_stage1=False, a_dtype=a_dtype, b_dtype="fp4",
        out_dtype=s1_out, act="silu", waves_per_eu=int(args.waves_per_eu),
        use_async_copy=bool(args.async_copy))
    # SEPARATE stage-2 (ATOM-faithful): FlyDSL gemm2 (compile_mixed_moe_gemm2, doweight in gemm2,
    # accumulate -> token-level out) + FlyDSL dc.combine (mirrors test_profiler_moe_gemm2_combine
    # _baseline_chain). No fused gemm2+combine.
    _g2exe = compile_mixed_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=epr, topk=topk,
        tile_m=tm2, tile_n=tn2, tile_k=tk2, doweight_stage2=True,
        a_dtype=s1_out, b_dtype="fp4", out_dtype="bf16",
        accumulate=True, persist_m=-1, sort_block_m=_sbm)
    _g2out = torch.zeros(max_recv, model_dim, dtype=torch.bfloat16, device=dev)
    _g2c = {}

    def _run_gemm2_sep():
        _g2out.zero_()
        _ga = (_g2out, a2_e.view(-1), w2_kernel, a2s_e, w2_scale_1d,
               a_st, a_se_local, a_sw, a_nv, bias_d,
               max_recv, model_dim, inter_dim, int(_max_blocks),
               torch.cuda.current_stream())
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
        aiter.moe_sorting_fwd(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc],
                              int(experts), int(tm), None, None, 0)
        if _is_fp4:
            _a1q, _a1sp = per_1x32_mx_quant_hip(_bt[:trc].contiguous(), quant_dtype=_qd)
        else:
            _a1q, _a1sp = per_1x32_mx_quant_hip(_bt[:trc].contiguous(), quant_dtype=_qd, scale_type=_stp)
        aiter.mxfp4_moe_sort_hip(a1s, _a1sp, a_st, a_nv, int(trc), int(model_dim))
        # gemm1 AND gemm2 both index LOCAL: w1/w2 = this rank's epr experts, a_se_local (aiter gave
        # GLOBAL ids).  Local w1 (<4GB) avoids the 4GB buffer truncation that global w1 hits.
        a_se_local.copy_(a_se - rank * epr)
        gemm1(a2_e.view(max_recv, topk, a2_e.shape[-1]), _agv(_a1q), _w1_arg, a1s.view(torch.uint8),
              _w1s_arg, a_st, a_se_local, a_sw, a_nv, bias_d, a2s_e, fx.Int32(trc),
              fx.Int32(inter_dim * 2), fx.Int32(model_dim), fx.Int32(int(_max_blocks)),
              stream=fx.Stream(torch.cuda.current_stream()))
        _run_gemm2_sep()                              # FlyDSL gemm2 (separate) -> _g2out [max_recv, model_dim]
        _r = dc.combine(_g2out, None, _oidx)          # FlyDSL combine (separate)
        _atom_out_holder["o"] = _r[0] if isinstance(_r, (tuple, list)) else _r
    _atom_body(); torch.cuda.synchronize(); ms.shmem_barrier_all()
    _ao = _atom_out_holder["o"]
    out_atom = _ao[:run_tokens].float().cpu().numpy().copy()

    # ============ 7. ATOM-fp8 baseline (PRODUCTION): fp8 dispatch -> sort -> gemm1 -> fused stage2 ============
    # fp8 dispatch carries the e8m0 scale (NO recv-quant), 1B/elem -> fair vs megav1. combine still
    # needs a bf16 op (gemm2 emits bf16): reuse `dc` with total_recv/tis bridged from the fp8 op
    # (atom's bridge; megav1 removes it via Plan A).
    _scale_mx_blocks = model_dim // 32
    cfg_fp8 = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=world, hidden_dim=model_dim, max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr, num_experts_per_token=topk,
        data_type=(torch.float4_e2m1fn_x2 if _is_fp4 else torch.float8_e4m3fn),
        scale_dim=_scale_mx_blocks,
        scale_type_size=1, enable_std_moe=False,
        # combine emits bf16 gemm2_out -> size shmem buffers for the largest dtype (bf16=2B/elem).
        max_token_type_size=2)
    dcf = FlyDSLDispatchCombineIntraNodeOp(cfg_fp8)
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    _atom8_holder = {}
    def _atom_fp8_body():
        # ATOM fp8 path: quantize (in-body) -> fp8 dispatch (dcf) -> gemm1 -> FlyDSL gemm2 (separate)
        # -> FlyDSL combine (dcf).  Quant is inside the timed body (production: quantize BEFORE the
        # fp8 dispatch), symmetric with the bf16 path's recv-side quant.
        a2_e.zero_()
        dcf.total_recv.zero_()
        if _is_fp4:
            _xq, _xsc = per_1x32_mx_quant_hip(x_bf16[:run_tokens].contiguous(), quant_dtype=_qd)
        else:
            _xq, _xsc = per_1x32_mx_quant_hip(x_bf16[:run_tokens].contiguous(),
                                              quant_dtype=_qd, scale_type=_stp)
        _xsc = _xsc.view(torch.uint8)
        _rx, _, _rs, _oidx, _ = dcf.dispatch(_xq, wc, _xsc, ic)   # fp8 dispatch (+ e8m0 scale)
        _oi = _oidx[:trc].to(torch.int32)
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
        aiter.moe_sorting_fwd(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc],
                              int(experts), int(tm), None, None, 0)
        aiter.mxfp4_moe_sort_hip(a1s, _rs[:trc].contiguous(), a_st, a_nv, int(trc), int(model_dim))
        a_se_local.copy_(a_se - rank * epr)
        gemm1(a2_e.view(max_recv, topk, a2_e.shape[-1]), _agv(_rx[:trc]), _w1_arg, a1s.view(torch.uint8),
              _w1s_arg, a_st, a_se_local, a_sw, a_nv, bias_d, a2s_e, fx.Int32(trc),
              fx.Int32(inter_dim * 2), fx.Int32(model_dim), fx.Int32(int(_max_blocks)),
              stream=fx.Stream(torch.cuda.current_stream()))
        _run_gemm2_sep()                              # FlyDSL gemm2 (separate) -> _g2out
        _r = dcf.combine(_g2out, None, _oidx)         # FlyDSL combine (separate, on fp8 dispatch op)
        _atom8_holder["o"] = _r[0] if isinstance(_r, (tuple, list)) else _r
    _atom_fp8_body(); torch.cuda.synchronize(); ms.shmem_barrier_all()
    _a8 = _atom8_holder["o"]
    out_atom8 = _a8[:run_tokens].float().cpu().numpy().copy()

    # ===================== 8. f32 torch oracle (routing-WEIGHTED reduce) =====================
    x32 = x_bf16[:run_tokens].float(); ti = ic[:run_tokens].long()
    oracle_w = torch.zeros(run_tokens, model_dim, device=dev, dtype=torch.float32)
    wv = wc[:run_tokens].float()
    for k in range(topk):
        ek = ti[:, k]
        for e in torch.unique(ek).tolist():
            rows = (ek == e).nonzero().flatten()
            xr = x32[rows]
            _Wg = w1_all[e, :inter_dim]; _Wu = w1_all[e, inter_dim:2 * inter_dim]
            _a1 = _F.silu(xr @ _Wg.t()) * (xr @ _Wu.t())
            _o2 = _a1 @ w2_all[e].t()
            oracle_w[rows] += wv[rows, k:k + 1] * _o2
    orw = oracle_w.cpu().numpy()

    # ============================= 9. correctness gate =============================
    # All paths apply routing weights in the GEMM2 doweight epilogue -> compare to the WEIGHTED
    # oracle. The e2e floor is dominated by the fp4 *weights* (~0.20 a8w4 / ~0.30 a4w4).
    _rm_w = _relL2(out_mega, orw)         # mega(prod)
    _ra_w = _relL2(out_atom, orw)         # atom-bf16 (reference)
    _ra8_w = _relL2(out_atom8, orw)       # atom-fp8 (primary baseline)
    _rma = _relL2(out_mega, out_atom8)    # mega vs primary baseline (should be ~0)
    _floor = 0.32 if _is_fp4 else 0.25
    _mega_ok = _rm_w < _floor
    _atom8_ok = _ra8_w < _floor
    # oracle sanity: if even the most-accurate path (atom-bf16) is far from the oracle, the oracle
    # is unreliable for this shape -> fall back to cross-impl agreement.
    _oracle_broken = (_ra_w > _floor)
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
    _all_ok = _all_max(dev, 0.0 if ok else 1.0) < 0.5   # any failing rank -> all_ok False

    # ===================== 10. stage1-only bodies (dispatch+GEMM1) [DISABLED] =====================
    # def _mega_s1_body():
    #     moe.stage1.forward(x_q, wc, x_sc, ic)   # megav1 single-launch dispatch ⊕ GEMM1 -> a2
    # def _atom_s1_body():
    #     a2_e.zero_(); dcf.total_recv.zero_()
    #     _rx, _, _rs, _oidx, _ = dcf.dispatch(x_q, wc, x_sc, ic)   # fp8 dispatch
    #     _oi = _oidx[:trc].to(torch.int32)
    #     _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
    #     recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
    #     aiter.moe_sorting_fwd(recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc],
    #                           int(experts), int(tm), None, None, 0)
    #     aiter.mxfp4_moe_sort_hip(a1s, _rs[:trc].contiguous(), a_st, a_nv, int(trc), int(model_dim))
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
        _t_mega, _t_atom8, _t_atom = (_pm["e2e_us_avg"] / 1e3,
                                      _pa8["e2e_us_avg"] / 1e3, _pa["e2e_us_avg"] / 1e3)
    else:
        _t_mega = _cg_time(_mega_body, moe.comb_op)          # megav1 e2e (stage1+stage2)
        _t_atom8 = _cg_time(_atom_fp8_body, dc)              # baseline e2e (fp8 dispatch)
        _t_atom = _cg_time(_atom_body, dc)                   # reference e2e (bf16 dispatch)
    # _t_mega_s1 = _cg_time(_mega_s1_body, moe.comb_op)        # megav1 STAGE1-only
    # _t_atom_s1 = _cg_time(_atom_s1_body, dcf)                # baseline STAGE1-only (fp8 dispatch->gemm1)

    # ============================= 12. report + return metrics =============================
    if rank == 0:
        _e2e_warn = "  [WARN torch-oracle unreliable for this shape: gated on mega-vs-baseline]" if _oracle_broken else ""
        print(f"[FULL-E2E] {args.network} {args.quant} bs={run_tokens} seed={args.seed} -> {'PASS' if _all_ok else 'FAIL'} (all {world} ranks){_e2e_warn}", flush=True)
        print(f"  [precision vs WEIGHTED torch-oracle, MAX over {world} ranks]  mega(prod)={_rm_w_max:.3e}  "
              f"atom-fp8(baseline)={_ra8_w_max:.3e}  atom-bf16(ref)={_ra_w_max:.3e}  "
              f"mega-vs-baseline={_rma_max:.3e}  (floor~{_floor})", flush=True)
        # [perf STAGE1-only] disabled
        # print(f"  [perf STAGE1-only, ms]  baseline-fp8(dispatch->gemm1)={_t_atom_s1:.4f}  "
        #       f"megav1(dispatch+gemm1)={_t_mega_s1:.4f}  "
        #       f"speedup={(_t_atom_s1 / _t_mega_s1) if _t_mega_s1 > 0 else -1:.3f}", flush=True)
        _timer = "profiler-e2e" if getattr(args, "profile", False) else "cuda-event"
        print(f"  [perf E2E (stage1+fused-stage2), ms | {_timer}]  baseline-fp8={_t_atom8:.4f}  "
              f"megav1={_t_mega:.4f}  speedup={(_t_atom8 / _t_mega) if _t_mega > 0 else -1:.3f}  "
              f"| ref bf16-dispatch baseline={_t_atom:.4f}  (out=bf16)", flush=True)
    return dict(network=args.network, quant=args.quant, tokens=run_tokens,
                full_e2e_mega_relL2=_rm_w, full_e2e_atom_fp8_relL2=_ra8_w, full_e2e_atom_bf16_relL2=_ra_w,
                full_e2e_mega_vs_baseline=_rma,
                # s1_baseline_ms=_t_atom_s1, s1_mega_ms=_t_mega_s1,  # stage1-only disabled
                full_e2e_baseline_fp8_ms=_t_atom8, full_e2e_baseline_bf16_ms=_t_atom,
                full_e2e_mega_ms=_t_mega, full_e2e_pass=ok)


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
    T = _prepare(dev, quant=args.quant, tokens=run_tokens, model_dim=model_dim,
                 inter_dim=inter_dim, experts=experts, topk=topk, seed=args.seed, rank=rank,
                 world=world, keep_ref=False)
    w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
    topk_ids, wts = T["topk_ids"], T["wts"]
    a_dtype = T["a_dtype"]
    x_bf16 = T["x_bf16"]

    # The production end-to-end comparison: megav1 (single-op MegaMoE) vs the ATOM production stack,
    # BOTH with the real routing weights + fused stage-2, gated on the weighted torch oracle.
    if not HAS_AITER:
        _info(rank, "[full-e2e] needs aiter; skipping"); return None
    return _run_full_e2e(
        args, rank, world, dev, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, epr=epr, topk=topk, run_tokens=run_tokens, mtpr=mtpr,
        a_dtype=a_dtype, s1_out=a_dtype, w_kernel=w_kernel,
        scale_w1_1d=scale_w1_1d, x_bf16=x_bf16, topk_ids=topk_ids, wts=wts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", type=str, default="v4_flash", choices=list(NETWORKS))
    p.add_argument("--quant", type=str, default="a8w4", choices=["a8w4", "a4w4"])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=-1,
                   help="-1 (default) = use the network's native topk (r1_v3=8, v4_*=6); >0 overrides")
    p.add_argument("--waves-per-eu", type=int, default=4)
    p.add_argument("--async-copy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="run EACH bs with N distinct seeds (seed, seed+1, ...) for random-data "
                        "coverage.  NOTE: per-run symmetric buffers are not freed, so large-bs x "
                        "multi-seed needs a bigger heap, e.g. MORI_SHMEM_HEAP_SIZE=32G.")
    p.add_argument("--master-port", type=int, default=29921)
    p.add_argument("--matrix", action="store_true", help="run all networks x classic bs")
    p.add_argument("--full-bs", action="store_true", help="use the full bs sweep (1..32768)")
    p.add_argument("--bs-list", type=str, default="",
                   help="comma list of batch sizes to sweep for the single --network/--quant "
                        "(e.g. '256,2048,4096,8192'); overrides --tokens/--matrix.")
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--mega-gemm2-tile-m", type=int, default=-1,
                   help="debug: override ONLY MegaMoE's gemm2 tile_m (baseline unchanged). "
                        "Used to reproduce the stage1 sort_block_m <-> gemm2 tile_m mismatch.")
    p.add_argument("--mega-stage1-fused", action=argparse.BooleanOptionalAction, default=True,
                   help="MegaMoE stage-1 fused (megakernel) vs non-fused (bf16 dispatch+sort+gemm1).")
    p.add_argument("--mega-stage2-fused", action=argparse.BooleanOptionalAction, default=True,
                   help="MegaMoE stage-2 fused (gemm2+combine) vs non-fused (gemm2 + separate combine).")
    p.add_argument("--profile", action="store_true",
                   help="measure with torch.profiler instead of cuda-event (mutually exclusive): "
                        "dump chrome trace + per-kernel GPU table + E2E replay time. Default is the "
                        "lightweight cuda-event timing.")
    p.add_argument("--profile-dir", type=str, default="/tmp/mega_prof",
                   help="output dir for --profile chrome traces (default /tmp/mega_prof).")
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
    for net, bs in combos:
        args.network = net
        args.tokens = bs
        for _si in range(_n_seeds):
            args.seed = _base_seed + _si
            try:
                r = run_one(args, rank, world, dev)
                if r is not None:
                    results.append(r)
            except Exception as e:  # noqa: BLE001
                import traceback
                if rank == 0:
                    traceback.print_exc()
                _info(rank, f"[bench] {net} bs={bs} seed={args.seed} ERROR: {type(e).__name__}: {e}")
            # free per-config allocations (facade/op/weights) so they don't accumulate across the
            # sweep -> avoids HIP OOM on big nets (e.g. v4_pro w1 ~67GB) at later bs/seed.
            import gc as _gc
            _gc.collect(); torch.cuda.empty_cache()
            torch.cuda.synchronize(); dist.barrier()
    args.seed = _base_seed

    if rank == 0 and args.json_out:
        with open(args.json_out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        _info(rank, f"[bench] wrote {len(results)} rows -> {args.json_out}")

    torch.cuda.synchronize(); dist.barrier(); _cleanup()


if __name__ == "__main__":
    main()
