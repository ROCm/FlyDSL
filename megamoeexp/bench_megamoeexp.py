#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Intranode MoE **STAGE-1-ONLY** benchmark: megamoeexp (dev) vs MegaMoE stage-1 (production)
vs ATOM stage-1 baseline.

Per the design ``docs/moe_stage1_overlap_design.md`` §7 (G1 progressive-frontier overlap) + §8,
this bench exercises **ONLY megastage1** (single-launch dispatch ⊕ GEMM1 -> a2).  NO stage-2
(gemm2/combine) is built or timed.  Three stage-1 stacks, all 8-rank:

  * megaexp  : fp8/fp4 act -> ``megamoeexp`` FusedMoEMegaStage1.forward -> a2   (path under dev;
               ``--frontier`` builds it with FUSED_MEGA_FRONTIER=1, ``--prefetch-ktiles N`` sets
               FUSED_MEGA_PREFETCH_KTILES=N -> weight-prefetch C).
  * megav1   : fp8/fp4 act -> production ``kernels`` FusedMoEMegaStage1.forward -> a2   (reference).
  * atom     : fp8 dispatch -> aiter sort -> mixed_moe_gemm1 -> a2   (baseline).

Accuracy (8-rank MAX-reduce, any failing rank -> FAIL), design §8 "megaexp≈prod vs oracle":
  For the accuracy pass the mega stage-1 is built with ``out_dtype="f16"`` (dequantized a2, no
  packed MX scale to unpack).  a2 rows live in the ATOM logical layout ``t*topk+s`` (t=src_global).
  (1) ``relL2(megaexp_a2, megav1_a2)`` -- HARD gate ~0: G1 frontier / prefetch are perf-only and
      MUST NOT change numerics.
  (2) both vs a torch **a2 oracle** ``silu(x_t @ Wg_e) * (x_t @ Wu_e)`` (bf16 ref weights), at the
      fp8/fp4 quant floor -- the independent correctness gate.  The oracle needs peers' x + topk_ids
      (all_gather); gated to ``bs <= --oracle-max`` (above it, only the exp-vs-prod gate runs).

Perf (CUDAGraph device time, us) -- built with the PRODUCTION native a2 dtype (fp8/fp4):
  stage-1-only megaexp(dev) vs megav1(prod) vs atom baseline, with exp-vs-prod / exp-vs-baseline
  speedups.  This is the whole metric; there is no e2e number.

Run (8x MI355X / gfx950)::

  MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
    megamoeexp/bench_megamoeexp.py --network v4_pro --quant a8w4 \
    --bs-list 64,128,256 --iters 30 --warmup 8 [--frontier] [--prefetch-ktiles 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # megamoeexp/ -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mori.shmem as ms  # noqa: E402

import flydsl.expr as fx  # noqa: E402
import tests.kernels.test_moe_gemm as tmg  # noqa: E402
from kernels.dispatch_combine_intranode_op import (  # noqa: E402
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from tests.kernels.utils import fp4_utils  # noqa: E402  (weight e8m0 shuffle)
from tests.utils import shuffle_weight  # noqa: E402

try:
    import aiter  # noqa: E402

    HAS_AITER = bool(getattr(aiter, "moe_sorting_fwd", None)) and bool(getattr(aiter, "mxfp4_moe_sort_hip", None))
except Exception:  # noqa: BLE001
    HAS_AITER = False

# This env's aiter lacks ``per_1x32_mx_quant_hip``.  The mega stage-1 __init__ (via
# fused_moe_megakernel) never calls it on the pre-quantized forward path, but guards on its import.
# Inject a tmg-based equivalent BEFORE the stage-1 modules are imported (lazily in _run_stage1).
try:
    import aiter.ops.quant as _aq  # noqa: E402

    if not hasattr(_aq, "per_1x32_mx_quant_hip"):

        def _per_1x32_mx_quant_hip_shim(x, quant_dtype=None, scale_type=None):
            _tag = str(quant_dtype).lower()
            if "fp4" in _tag or "f4" in _tag:
                return tmg._per_1x32_fp4_quant(x.contiguous())
            return tmg._per_1x32_mxfp8_quant(x.contiguous())

        _aq.per_1x32_mx_quant_hip = _per_1x32_mx_quant_hip_shim
except Exception:  # noqa: BLE001
    pass


NETWORKS = {
    "r1_v3": dict(model_dim=7168, inter_dim=2048, experts=256, topk=8),
    "v4_flash": dict(model_dim=4096, inter_dim=2048, experts=256, topk=6),
    "v4_pro": dict(model_dim=7168, inter_dim=3072, experts=384, topk=6),
}

# batch-size sweeps for --matrix / --full-bs.
CLASSIC_BS = [1, 8, 64, 512, 8192, 32768]
FULL_BS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


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


def _qact(x, is_fp4):
    """bf16 activation -> (fp8/fp4 payload, e8m0 scale uint8) via the SAME tmg helpers as _prepare."""
    if is_fp4:
        q, s = tmg._per_1x32_fp4_quant(x.contiguous())
        return q.view(torch.float4_e2m1fn_x2).contiguous(), s.view(torch.uint8)
    q, s = tmg._per_1x32_mxfp8_quant(x.contiguous())
    return q.contiguous(), s.view(torch.uint8)


def _prepare(dev, *, quant, tokens, model_dim, inter_dim, experts, topk, seed, rank=0, world=1, keep_ref=False):
    """Generate inputs + pack weights/scales for A8W4 (mxfp8 act) or A4W4 (mxfp4 act).

    Returns dict with x (token payload in the dispatch dtype), scale_mx_u8 (e8m0 act scale),
    w_kernel (mxfp4 shuffled), scale_w1_1d (e8m0 weight scale shuffled), topk_ids, wts,
    and (keep_ref) w_ref_local: THIS rank's local experts' pre-quant bf16 weights (a2 oracle).
    """
    torch.manual_seed(seed)
    # scale x,w so the GEMM reduction over model_dim gives O(1) pre-activation (std ~ sqrt(md)*sx*sw):
    # sx=sw -> s = (md)^-0.25.
    init_scale = float(model_dim) ** -0.25
    x_fp32 = torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * init_scale
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * init_scale

    if quant == "a8w4":
        x_q, scale_x_mx = tmg._per_1x32_mxfp8_quant(x_fp32)
        x_payload = x_q.contiguous()  # [tokens, model_dim] fp8_e4m3fn
        token_dtype = torch.float8_e4m3fn
        a_dtype = "fp8"
        row_view_dim = model_dim
    elif quant == "a4w4":
        x_q, scale_x_mx = tmg._per_1x32_fp4_quant(x_fp32)
        x_payload = x_q.view(torch.float4_e2m1fn_x2).contiguous()  # [tokens, model_dim//2]
        token_dtype = torch.float4_e2m1fn_x2
        a_dtype = "fp4"
        row_view_dim = model_dim // 2
    else:
        raise SystemExit(f"unknown quant {quant!r} (use a8w4|a4w4)")

    # a2 oracle ground-truth: keep this rank's LOCAL experts' ORIGINAL (pre-quant) weights in bf16.
    w_ref_local = None
    if keep_ref:
        _epr = experts // world
        w_ref_local = w1_fp32[rank * _epr : (rank + 1) * _epr].to(torch.bfloat16).contiguous()

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
    topk_ids = torch.stack([torch.randperm(experts, device=dev)[:topk] for _ in range(tokens)]).to(torch.int32)
    wts = torch.full((tokens, topk), 1.0 / topk, device=dev, dtype=torch.float32)

    return dict(
        x_payload=x_payload,
        scale_mx_u8=scale_x_mx.contiguous(),
        w_kernel=w_kernel,
        scale_w1_1d=scale_w1_1d,
        topk_ids=topk_ids,
        wts=wts,
        token_dtype=token_dtype,
        a_dtype=a_dtype,
        row_view_dim=row_view_dim,
        x_bf16=x_fp32.to(torch.bfloat16).contiguous(),
        w_ref_local=w_ref_local,
    )


def _relL2(a, b):
    import numpy as _np

    a = _np.asarray(a, dtype=_np.float64)
    b = _np.asarray(b, dtype=_np.float64)
    n = float(((a - b) ** 2).sum())
    d = float((b**2).sum())
    return (n / d) ** 0.5 if d > 0 else -1.0


def _cg_time(args, dev, body):
    """CUDAGraph device time (test_profiler pattern): warmup -> capture -> back-to-back replay,
    event-timed, mean across ranks.  Returns ms."""
    ms.shmem_barrier_all()
    body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()  # warmup (jit)
    _cap = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=_cap):
        body()
    for _ in range(max(1, int(args.warmup))):
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


def _build_mega_s1(
    cls,
    *,
    rank,
    world,
    model_dim,
    inter_dim,
    experts,
    topk,
    quant,
    w1,
    w1s,
    mtpr,
    network,
    out_dtype,
    frontier_env,
    prefetch_env,
):
    """Construct a mega FusedMoEMegaStage1 (prod or exp) under the given frontier/prefetch env.

    The G1 frontier + weight-prefetch depth are baked into the kernel at COMPILE (construction)
    time; set them around the ctor and restore, so a caller's env is not mutated for the rest of
    the sweep (the ctor snapshots them internally)."""
    _saved = {k: os.environ.get(k) for k in ("FUSED_MEGA_FRONTIER", "FUSED_MEGA_PREFETCH_KTILES")}
    os.environ["FUSED_MEGA_FRONTIER"] = frontier_env
    os.environ["FUSED_MEGA_PREFETCH_KTILES"] = prefetch_env
    try:
        op = cls(
            rank=rank,
            world_size=world,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            w1=w1,
            w1_scale=w1s,
            max_tok_per_rank=mtpr,
            network=network,
            out_dtype=out_dtype,
        )
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return op


def _run_stage1(
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
    w_kernel,
    scale_w1_1d,
    w_ref_local,
    x_bf16,
    topk_ids,
    wts,
):
    """STAGE-1-ONLY 3-way: megaexp(dev) vs megav1(prod) vs atom baseline.  Accuracy on the a2
    (silu(gate)*up) intermediate via f16-output stage-1 + torch oracle; perf on the native dtype."""
    import gc as _gc

    import torch.nn.functional as _F

    from kernels.fused_moe_megakernel import FusedMoEMegaStage1 as ProdS1
    from megamoeexp.fused_moe_megakernel import FusedMoEMegaStage1 as ExpS1

    is_fp4 = a_dtype == "fp4"
    _frt_env = "1" if bool(getattr(args, "frontier", False)) else "0"
    _pf_env = str(max(0, int(getattr(args, "prefetch_ktiles", 0))))

    # LOCAL w1 slice (this rank's epr experts; the gemm1 kernel indexes by the local expert id).
    _wpe = w_kernel.numel() // experts
    _spe = scale_w1_1d.numel() // experts
    w1 = w_kernel.reshape(-1)[rank * epr * _wpe : (rank + 1) * epr * _wpe].contiguous()
    w1s = scale_w1_1d.reshape(-1)[rank * epr * _spe : (rank + 1) * epr * _spe].contiguous()

    # quantized activation (production stage-1 input)
    x_q, x_sc = _qact(x_bf16[:run_tokens], is_fp4)
    wc = wts[:run_tokens].contiguous()
    ic = topk_ids[:run_tokens].to(torch.int32).contiguous()

    a2rows = world * mtpr * topk

    # ============ ACCURACY: f16-output stage-1 (dequantized a2, no packed MX scale) ============
    exp_f16_op = _build_mega_s1(
        ExpS1,
        rank=rank,
        world=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=args.quant,
        w1=w1,
        w1s=w1s,
        mtpr=mtpr,
        network=args.network,
        out_dtype="f16",
        frontier_env=_frt_env,
        prefetch_env=_pf_env,
    )
    prod_f16_op = _build_mega_s1(
        ProdS1,
        rank=rank,
        world=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=args.quant,
        w1=w1,
        w1s=w1s,
        mtpr=mtpr,
        network=args.network,
        out_dtype="f16",
        frontier_env="0",
        prefetch_env="0",
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    _eo = exp_f16_op.forward(x_q, wc, x_sc, ic)
    _po = prod_f16_op.forward(x_q, wc, x_sc, ic)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    exp_a2 = _eo["out"].reshape(a2rows, inter_dim).float()
    prod_a2 = _po["out"].reshape(a2rows, inter_dim).float()

    # (1) exp-vs-prod: G1/prefetch are perf-only -> must be ~0.
    _rme_vs_prod = _relL2(exp_a2.cpu().numpy(), prod_a2.cpu().numpy())

    # (2) torch a2 oracle (gated by bs): each rank owns local experts [rank*epr, (rank+1)*epr) and
    #     wrote logical rows (p*mtpr+j)*topk+s for every (p,j,s) routing to one of them.  Needs peers'
    #     x + topk_ids -> all_gather (bounded to bs <= --oracle-max).
    _rme_or = _rm_or = -1.0
    if run_tokens <= int(args.oracle_max):
        x_all = [torch.empty_like(x_bf16[:run_tokens]) for _ in range(world)]
        id_all = [torch.empty_like(ic) for _ in range(world)]
        dist.all_gather(x_all, x_bf16[:run_tokens].contiguous())
        dist.all_gather(id_all, ic.contiguous())
        x_all = torch.stack(x_all).float()  # [world, run_tokens, model_dim]
        id_all = torch.stack(id_all)  # [world, run_tokens, topk] i32
        oracle = torch.zeros(a2rows, inter_dim, device=dev, dtype=torch.float32)
        written = torch.zeros(a2rows, dtype=torch.bool, device=dev)
        for _el in range(epr):
            _eg = rank * epr + _el
            _hit = (id_all == _eg).nonzero(as_tuple=False)  # [K, 3] -> (p, j, s)
            if _hit.numel() == 0:
                continue
            _p, _j, _s = _hit[:, 0], _hit[:, 1], _hit[:, 2]
            _xr = x_all[_p, _j]  # [K, model_dim]
            _Wg = w_ref_local[_el, :inter_dim].float()
            _Wu = w_ref_local[_el, inter_dim : 2 * inter_dim].float()
            _a2 = _F.silu(_xr @ _Wg.t()) * (_xr @ _Wu.t())  # [K, inter_dim]
            _rows = (_p * mtpr + _j) * topk + _s
            oracle[_rows] = _a2
            written[_rows] = True
        _wr = written.nonzero(as_tuple=False).flatten()
        if _wr.numel() > 0:
            _o = oracle[_wr].cpu().numpy()
            _rme_or = _relL2(exp_a2[_wr].cpu().numpy(), _o)
            _rm_or = _relL2(prod_a2[_wr].cpu().numpy(), _o)

    # a2 quant floor: activation fp8/fp4 + weight fp4 -> fp4-weight-dominated.
    _floor = 0.32 if is_fp4 else 0.22
    _exp_vs_prod_ok = _rme_vs_prod < 5e-2
    _oracle_ok = (_rme_or < 0) or (_rme_or < _floor) or (_rme_or <= _rm_or + 2e-2)
    _ok = _exp_vs_prod_ok and _oracle_ok
    _all_ok = _all_max(dev, 0.0 if _ok else 1.0) < 0.5
    _rme_vs_prod_max = _all_max(dev, _rme_vs_prod)
    _rme_or_max = _all_max(dev, _rme_or)
    _rm_or_max = _all_max(dev, _rm_or)

    # free the f16 accuracy ops before building the native perf ops (bound peak symmetric memory).
    del exp_f16_op, prod_f16_op, exp_a2, prod_a2
    if run_tokens <= int(args.oracle_max):
        del oracle, written
    _gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    # ============ PERF: native-dtype stage-1 (fp8/fp4 a2), stage-1-only CUDAGraph time ============
    exp_op = _build_mega_s1(
        ExpS1,
        rank=rank,
        world=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=args.quant,
        w1=w1,
        w1s=w1s,
        mtpr=mtpr,
        network=args.network,
        out_dtype="auto",
        frontier_env=_frt_env,
        prefetch_env=_pf_env,
    )
    prod_op = _build_mega_s1(
        ProdS1,
        rank=rank,
        world=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=args.quant,
        w1=w1,
        w1s=w1s,
        mtpr=mtpr,
        network=args.network,
        out_dtype="auto",
        frontier_env="0",
        prefetch_env="0",
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    def _exp_body():
        exp_op.forward(x_q, wc, x_sc, ic)

    def _prod_body():
        prod_op.forward(x_q, wc, x_sc, ic)

    # ---- ATOM baseline stage-1: fp8 dispatch -> aiter sort -> mixed_moe_gemm1 -> a2 ----
    from aiter import dtypes as _adt

    from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1

    max_recv = world * mtpr
    tm, tn1, tk = 32, 128, 256
    _agv = (lambda t: t.view(torch.uint8)) if is_fp4 else (lambda t: t)
    _scale_mx_blocks = model_dim // 32
    cfg_fp8 = FlyDSLDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=model_dim,
        max_num_inp_token_per_rank=mtpr,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        data_type=(torch.float4_e2m1fn_x2 if is_fp4 else torch.float8_e4m3fn),
        scale_dim=_scale_mx_blocks,
        scale_type_size=1,
        enable_std_moe=False,
    )
    dcf = FlyDSLDispatchCombineIntraNodeOp(cfg_fp8)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    dcf.total_recv.zero_()
    _rx0, _, _rs0, _oidx0, _ = dcf.dispatch(x_q, wc, x_sc, ic)  # one setup dispatch to fix trc
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    trc = max(1, int(dcf.total_recv.item()))
    if _all_min_int(dev, trc) <= 0:
        _info(rank, "[stage1] some rank got 0 recv; skipping")
        return None

    _max_pad = max_recv * topk + experts * tm
    _max_blocks = (_max_pad + tm - 1) // tm
    _scaleN_pad = ((model_dim // 32 + 7) // 8) * 8
    a_st = torch.empty(_max_pad, dtype=torch.int32, device=dev)
    a_sw = torch.empty(_max_pad, dtype=torch.float32, device=dev)
    a_se = torch.empty(_max_blocks, dtype=torch.int32, device=dev)
    a_se_local = torch.empty(_max_blocks, dtype=torch.int32, device=dev)
    a_nv = torch.zeros(2, dtype=torch.int32, device=dev)
    a_mbuf = torch.empty((max_recv, model_dim), dtype=torch.float16, device=dev)
    a1s = torch.empty(((_max_pad + 31) // 32 * 32, _scaleN_pad), dtype=_adt.fp8_e8m0, device=dev)
    recv_wts = torch.full((max_recv, topk), 1.0 / topk, device=dev, dtype=torch.float32)
    recv_topk = torch.empty((max_recv, topk), dtype=torch.int32, device=dev)
    _sentinel = torch.full((trc, topk), experts, dtype=torch.int32, device=dev)
    if is_fp4:
        a2_e = torch.zeros((max_recv * topk, inter_dim // 2), dtype=torch.uint8, device=dev)
    else:
        a2_e = torch.zeros((max_recv * topk, inter_dim), dtype=torch.float8_e4m3fn, device=dev)
    _sbm = max(32, tm)
    _pr = ((_max_blocks * _sbm + 255) // 256) * 256
    _pc = (((inter_dim // 32) + 7) // 8) * 8
    a2s_e = torch.zeros(_pr * _pc + inter_dim, dtype=torch.uint8, device=dev)
    bias_d = torch.empty((0,), device=dev, dtype=torch.float32)
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
        out_dtype=a_dtype,
        act="silu",
        waves_per_eu=int(args.waves_per_eu),
        use_async_copy=bool(args.async_copy),
    )

    def _atom_body():
        a2_e.zero_()
        dcf.total_recv.zero_()
        _rx, _, _rs, _oidx, _ = dcf.dispatch(x_q, wc, x_sc, ic)  # fp8 dispatch (+ e8m0 scale)
        _oi = _oidx[:trc].to(torch.int32)
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        recv_topk[:trc].copy_(torch.where(_loc, _oi, _sentinel))
        aiter.moe_sorting_fwd(
            recv_topk[:trc], recv_wts[:trc], a_st, a_sw, a_se, a_nv, a_mbuf[:trc], int(experts), int(tm), None, None, 0
        )
        aiter.mxfp4_moe_sort_hip(a1s, _rs[:trc].contiguous(), a_st, a_nv, int(trc), int(model_dim))
        a_se_local.copy_(a_se - rank * epr)
        gemm1(
            a2_e.view(max_recv, topk, a2_e.shape[-1]),
            _agv(_rx[:trc]),
            w1,
            a1s.view(torch.uint8),
            w1s,
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

    _atom_body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    _t_exp = _cg_time(args, dev, _exp_body)
    _t_prod = _cg_time(args, dev, _prod_body)
    _t_atom = _cg_time(args, dev, _atom_body)

    def _us(ms_val):
        return ms_val * 1e3

    def _spd(num, den):
        return (num / den) if den > 0 else -1.0

    if rank == 0:
        _frt = "ON" if bool(getattr(args, "frontier", False)) else "off"
        _pf = int(getattr(args, "prefetch_ktiles", 0))
        _orc = f"{_rme_or_max:.3e}" if _rme_or_max >= 0 else "skipped(bs>oracle-max)"
        print(
            f"[S1] {args.network} {args.quant} bs={run_tokens} seed={args.seed} "
            f"frontier={_frt} prefetch_ktiles={_pf} -> {'PASS' if _all_ok else 'FAIL'} "
            f"(all {world} ranks)",
            flush=True,
        )
        print(
            f"  [a2 accuracy, relL2 MAX over {world} ranks]  megaexp-vs-prod={_rme_vs_prod_max:.3e}  "
            f"| vs torch-oracle: megaexp={_orc}  megav1(prod)={_rm_or_max:.3e}  (floor~{_floor})",
            flush=True,
        )
        print(
            f"  [perf STAGE1-only, us]  megaexp(dev)={_us(_t_exp):8.2f}  megav1(prod)={_us(_t_prod):8.2f}  "
            f"baseline-fp8={_us(_t_atom):8.2f}  | exp-vs-prod={_spd(_t_prod, _t_exp):.3f}x  "
            f"exp-vs-baseline={_spd(_t_atom, _t_exp):.3f}x",
            flush=True,
        )

    # exp_op/prod_op/dcf/gemm1 are captured by the timing closures above; they are freed when this
    # function returns (scope exit) -> a `del` here would be a delete-of-closure-cell error.
    _gc.collect()
    torch.cuda.empty_cache()

    return dict(
        network=args.network,
        quant=args.quant,
        tokens=run_tokens,
        frontier=bool(getattr(args, "frontier", False)),
        prefetch_ktiles=int(getattr(args, "prefetch_ktiles", 0)),
        s1_megaexp_vs_prod_relL2=_rme_vs_prod,
        s1_megaexp_oracle_relL2=_rme_or,
        s1_megav1_oracle_relL2=_rm_or,
        s1_megaexp_us=_us(_t_exp),
        s1_megav1_us=_us(_t_prod),
        s1_baseline_us=_us(_t_atom),
        s1_pass=bool(_all_ok),
    )


def run_one(args, rank, world, dev):
    net = NETWORKS[args.network]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    topk = int(args.topk) if int(args.topk) > 0 else int(net["topk"])
    run_tokens = max(int(args.tokens), 1)
    if experts % world != 0:
        raise SystemExit(f"experts={experts} must divide world={world}")
    epr = experts // world

    # mtpr must be a power of two (mega dispatch decodes dest_enc with shift/mask).
    mtpr = 16
    while mtpr < run_tokens:
        mtpr <<= 1
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
        keep_ref=True,
    )
    if not HAS_AITER:
        _info(rank, "[stage1] needs aiter; skipping")
        return None
    return _run_stage1(
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
        a_dtype=T["a_dtype"],
        w_kernel=T["w_kernel"],
        scale_w1_1d=T["scale_w1_1d"],
        w_ref_local=T["w_ref_local"],
        x_bf16=T["x_bf16"],
        topk_ids=T["topk_ids"],
        wts=T["wts"],
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
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="run EACH bs with N distinct seeds (seed, seed+1, ...) for random-data "
        "coverage.  Per-run symmetric buffers are not freed, so large-bs x multi-seed "
        "needs a bigger heap, e.g. MORI_SHMEM_HEAP_SIZE=40G.",
    )
    p.add_argument("--master-port", type=int, default=29921)
    p.add_argument("--matrix", action="store_true", help="run all networks x classic bs")
    p.add_argument("--full-bs", action="store_true", help="use the full bs sweep (1..32768)")
    p.add_argument(
        "--bs-list",
        type=str,
        default="",
        help="comma list of batch sizes to sweep for the single --network/--quant "
        "(e.g. '64,128,256,512'); overrides --tokens/--matrix.",
    )
    p.add_argument(
        "--oracle-max",
        type=int,
        default=1024,
        help="max bs for the torch a2 oracle (all_gathers peers' x/topk_ids); above it "
        "only the exp-vs-prod accuracy gate runs.",
    )
    p.add_argument("--json-out", type=str, default="")
    p.add_argument(
        "--frontier",
        action="store_true",
        help="build megaexp stage-1 with FUSED_MEGA_FRONTIER=1 (design §7 G1 overlap). "
        "default off -> megaexp == production megav1 (sanity floor).",
    )
    p.add_argument(
        "--prefetch-ktiles",
        type=int,
        default=0,
        help="FUSED_MEGA_PREFETCH_KTILES for megaexp (weight-prefetch C depth, §4.3/§7.2②). "
        "0 = off; requires --frontier.",
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

    torch.cuda.synchronize()
    dist.barrier()
    _cleanup()


if __name__ == "__main__":
    main()
