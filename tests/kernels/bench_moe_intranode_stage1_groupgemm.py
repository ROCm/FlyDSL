#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Intranode MoE stage1 benchmark: ATOM (production) baseline vs the fused megakernel.

Compares stage1 operators under CUDAGraph replay (the only reported metric):

  (1) atom        : PRODUCTION bf16-dispatch (quant-POST), a4w4 (--from-bf16): bf16 dispatch ->
                    aiter.moe_sorting_fwd -> fused_dynamic_mx{fp4,fp8}_quant_moe_sort
                    (quant + scale-sort, on bf16 recv) -> mixed_moe_gemm1.
  (2) atom_fp8    : PRODUCTION fp8-dispatch (quant-PRE), a8w4: quantize bf16->fp8 BEFORE dispatch
                    (Mori use_fp8_dispatch equivalent) -> aiter.moe_sorting_fwd ->
                    aiter.mxfp4_moe_sort_hip (scale-sort only, no re-quant) -> mixed_moe_gemm1.
  (3) fused       : the operator under test -- expert-major dispatch + group-GEMM, single-launch
                    megakernel (--mega) or the 2-launch facade        [see docs/moe_stage1_mega.md]

ATOM's sort / scale-sort / quant are all aiter (production) kernels; fused is compared against them.
Supports A8W4 (mxfp8 act) and A4W4 (mxfp4 act).

Timing: CUDAGraph replay, CUDA events, median, max across ranks.

Run (8x MI355X)::

  MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=8 \
    tests/kernels/bench_moe_intranode_stage1_groupgemm.py --network v4_flash --tokens 64 \
    --quant a8w4 --iters 30 --warmup 8 --check-correctness
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flydsl.expr as fx  # noqa: E402
import mori.shmem as ms  # noqa: E402

from kernels.dispatch_combine_intranode_op import (  # noqa: E402
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1  # noqa: E402  (ATOM baseline GEMM)

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

# ATOM baseline GEMM tile -> aiter model_configs tune csv (the `flydsl_moe1_*` winner = ATOM's choice).
# v4_flash has no tune entry -> baseline keeps the default tile.
_ATOM_TUNE_FILE = {
    ("r1_v3", "a4w4"):  "dsv3_fp4_tuned_fmoe.csv",
    ("v4_pro", "a8w4"): "dsv4_fp8fp4_tuned_fmoe.csv",
}


def _atom_tuned_tile(network, quant, bs, cu_num, model_dim, inter_dim, epr, topk):
    """ATOM/aiter TUNED gemm1 tile for this exact shape.

    Reads aiter's per-model tune csv, keeps rows matching the EP shape
    (cu_num, token=bs, model_dim, inter_dim, expert=epr, topk, q_dtype_a/_w),
    then picks the lowest-us1 ``flydsl_moe1_*`` row (the family the bench's
    baseline GEMM runs) and parses ``tile_m x tile_n x tile_k`` from its kernel
    name.  Returns the tuple or None (untuned shape -> caller uses the default).

    Note: v4_pro's tuned kernels carry ``gui`` (gate-up interleave); only the
    tile is transferred to the baseline here, not the interleave layout.
    """
    fname = _ATOM_TUNE_FILE.get((network, quant))
    if fname is None:
        return None
    try:
        import csv as _csv, re as _re, aiter as _aiter
    except Exception:
        return None
    path = os.path.join(os.path.dirname(_aiter.__file__), "configs", "model_configs", fname)
    if not os.path.exists(path):
        return None
    qa = "torch.float4_e2m1fn_x2" if quant == "a4w4" else "torch.float8_e4m3fn"
    qw = "torch.float4_e2m1fn_x2"
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))

    def _scan(match_cu):
        best = None  # (us1, tile_m, tile_n, tile_k)
        for r in rows:
            try:
                if match_cu and int(r["cu_num"]) != cu_num:
                    continue
                if (int(r["token"]) != bs
                        or int(r["model_dim"]) != model_dim or int(r["inter_dim"]) != inter_dim
                        or int(r["expert"]) != epr or int(r["topk"]) != topk
                        or r["q_dtype_a"] != qa or r["q_dtype_w"] != qw):
                    continue
                m = _re.search(r"flydsl_moe1_\w*?_t(\d+)x(\d+)x(\d+)", r["kernelName1"])
                if not m:
                    continue
                us1 = float(r["us1"])
                if best is None or us1 < best[0]:
                    best = (us1, int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except (KeyError, ValueError, TypeError):
                continue
        return best

    best = _scan(match_cu=True) or _scan(match_cu=False)  # exact CU first, else any CU in the tune file
    return best[1:] if best else None
CLASSIC_BS = [1, 8, 64, 512, 8192, 32768]
FULL_BS = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


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


def _cuda_event_ms(replay, iters: int, warmup: int) -> float:
    for _ in range(max(1, warmup)):
        replay()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record(); replay(); e[i].record()
    torch.cuda.synchronize()
    samples = sorted(s[i].elapsed_time(e[i]) for i in range(iters))
    return samples[len(samples) // 2]  # median ms


def _profiler_ms(replay, iters: int, warmup: int, label: str = "op") -> float:
    """SKILL-compliant CUDAGraph timing: true GPU time from the chrome-trace
    ``gpu_user_annotation`` dur of each replay (no launch overhead), skipping the
    first cold-start iters.  no-reset between replays (avoids P2P shmem inflation).
    Ref: tests/kernels/test_profiler_dispatch_combine.py + testscript-SKILL.md."""
    from torch.profiler import profile, ProfilerActivity, schedule, record_function
    import json as _json, tempfile as _tf, os as _os
    cg_name = f"{label}::cudagraph_replay"
    pw = 10
    for _ in range(max(1, warmup)):
        replay()
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=pw, active=iters)) as prof:
        for _ in range(1 + pw + iters):
            with record_function(cg_name):
                replay()
            prof.step()
    torch.cuda.synchronize()
    tp = _tf.mktemp(suffix=".json")
    prof.export_chrome_trace(tp)
    try:
        with open(tp) as f:
            tr = _json.load(f)
    finally:
        try:
            _os.unlink(tp)
        except OSError:
            pass
    cg = sorted([e for e in tr.get("traceEvents", [])
                 if e.get("cat") == "gpu_user_annotation" and cg_name in e.get("name", "")],
                key=lambda e: e["ts"])
    durs = [e["dur"] for e in cg[-iters:]]
    durs = durs[5:] if len(durs) > 5 else durs
    if not durs:
        return -1.0
    return (sum(durs) / len(durs)) / 1000.0  # ms (dur in us)


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


def _prepare(dev, *, quant, tokens, model_dim, inter_dim, experts, topk, seed, rank=0):
    """Generate inputs + pack weights/scales for A8W4 (mxfp8 act) or A4W4 (mxfp4 act).

    Returns dict with x (token payload in the dispatch dtype), scale_mx_u8 (e8m0 act scale),
    w_kernel (mxfp4 shuffled), scale_w1_1d (e8m0 weight scale shuffled), topk_ids, wts,
    and meta: token_dtype, a_dtype, row_view_dim.
    """
    torch.manual_seed(seed)
    init_scale = 0.01
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
    )


def run_one(args, rank, world, dev):
    net = NETWORKS[args.network]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    # topk: --topk>0 overrides; else use the network's native topk (r1_v3=8, v4_*=6).
    topk = int(args.topk) if int(args.topk) > 0 else int(net["topk"])
    tile_k = int(args.tile_k)
    run_tokens = max(int(args.tokens), 1)  # allow bs=1 (1 token/rank); routing still reaches all ranks
    if experts % world != 0:
        raise SystemExit(f"experts={experts} must divide world={world}")
    epr = experts // world
    mtpr = max(16, run_tokens)
    try:
        _cu = int(torch.cuda.get_device_properties(dev).multi_processor_count)
    except Exception:
        _cu = 256
    # ATOM-baseline GEMM tile: use aiter's TUNED config where available (r1_v3/v4_pro -> the
    # flydsl_moe1 winner for this exact shape); else the default.  The mega facade auto-picks its
    # OWN deployed tile internally (decode tile_n=256 / prefill 128, tile_m 64/128) -- not overridden.
    _tt = (_atom_tuned_tile(args.network, args.quant, run_tokens, _cu, model_dim, inter_dim, epr, topk)
           if HAS_AITER else None)
    if _tt is not None:
        tile_m, tile_n, tile_k = _tt
        _info(rank, f"[atom-tune] {args.network} {args.quant} bs={run_tokens} cu={_cu} -> "
                    f"baseline tile m{tile_m} n{tile_n} k{tile_k}")
    else:
        tile_m = 128 if run_tokens >= 8192 else 32
        tile_n = 128 if run_tokens >= 8192 else 64
    scale_mx_blocks = model_dim // 32

    T = _prepare(dev, quant=args.quant, tokens=run_tokens, model_dim=model_dim,
                 inter_dim=inter_dim, experts=experts, topk=topk, seed=args.seed, rank=rank)
    x_payload, scale_mx_u8 = T["x_payload"], T["scale_mx_u8"]
    w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
    topk_ids, wts = T["topk_ids"], T["wts"]
    token_dtype, a_dtype, row_view_dim = T["token_dtype"], T["a_dtype"], T["row_view_dim"]
    x_bf16 = T["x_bf16"]

    # --from-bf16: BOTH paths quantize the bf16 source with the production HIP kernel
    # (per_1x32_mx_quant_hip) INSIDE the timed cudagraph body, so the activation-quant cost
    # is included end-to-end.  Default (off) feeds the pre-quantized fp4 payload (unchanged).
    def _x_in():
        if not getattr(args, "from_bf16", False):
            return x_payload[:run_tokens].contiguous(), scale_mx_u8[:run_tokens].contiguous()
        from aiter.ops.quant import per_1x32_mx_quant_hip
        from aiter import dtypes as _adt
        if args.quant == "a4w4":   # MXFP4: packed fp4x2 + e8m0 byte scale
            _xq, _sxq = per_1x32_mx_quant_hip(x_bf16[:run_tokens].contiguous(), quant_dtype=_adt.fp4x2)
        else:                       # MXFP8 (a8w4 act): fp8 output + e8m0 BYTE scale (not fp32)
            _xq, _sxq = per_1x32_mx_quant_hip(x_bf16[:run_tokens].contiguous(),
                                              quant_dtype=_adt.fp8, scale_type=_adt.fp8_e8m0)
        return _xq.contiguous(), _sxq.view(torch.uint8).contiguous()

    _info(rank, f"[bench] net={args.network} quant={args.quant} bs={run_tokens} md={model_dim} "
                f"id={inter_dim} E={experts}(epr={epr}) topk={topk} tile=({tile_m},{tile_n},{tile_k})")

    # ---- compile GEMM (serialize across ranks; compile-only) ----
    gemm = None
    for pe in range(world):
        if rank == pe:
            gemm = compile_mixed_moe_gemm1(
                model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, doweight_stage1=False,
                a_dtype=a_dtype, b_dtype="fp4", out_dtype="f16", act="silu",
                waves_per_eu=int(args.waves_per_eu), use_async_copy=bool(args.async_copy),
                use_cshuffle_epilog=(None if int(args.cshuffle) < 0 else bool(int(args.cshuffle))),
                xcd_swizzle=int(os.environ.get("BASELINE_XCD", "0")),  # FAIRNESS: let atom baseline GEMM also use xcd
            )
        dist.barrier()

    # ---- ep_dispatch op ----
    bn = min(_cu, 64) if mtpr <= 32 else (min(_cu, 128) if mtpr <= 128 else _cu)
    cfg = FlyDSLDispatchCombineConfig(
        rank=rank, world_size=world, hidden_dim=model_dim,
        max_num_inp_token_per_rank=mtpr, num_experts_per_rank=epr,
        num_experts_per_token=topk, data_type=token_dtype,
        warp_num_per_block=int(args.warps), block_num=bn,
        scale_dim=scale_mx_blocks, scale_type_size=1, enable_std_moe=False,
    )
    dc = FlyDSLDispatchCombineIntraNodeOp(cfg)
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    # ---- warm dispatch (cross-rank handshake) + read total_recv ----
    dc.total_recv.zero_()
    out_tok, out_wts, out_scales, out_idx_dev, _trv = dc.dispatch(
        x_payload[:run_tokens].contiguous(), wts.contiguous(),
        scale_mx_u8[:run_tokens].contiguous(), topk_ids.contiguous())
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    tr = int(dc.total_recv.item())
    _tr = max(1, tr)
    print(f"[bench] rank{rank}: total_recv={tr}", flush=True)
    min_tr = _all_min_int(dev, tr)
    if min_tr <= 0:
        _info(rank, f"[bench] min_tr={min_tr}; skipping (need tokens on every rank).")
        return None

    rx = out_tok                                  # [mr, row_view_dim] token payload
    rs = out_scales                               # [mr, scale_mx_blocks] uint8 (e8m0)
    # fp4 tensors aren't DLPack-able by the JIT (dtype code 17); pass A as uint8 view
    # (same bytes; the GEMM's a_dtype='fp4' defines the element layout).
    def _agv(t):
        return t.view(torch.uint8) if a_dtype == "fp4" else t
    rx_g = _agv(rx)

    # received-slot routing: local experts kept, sentinel(=experts) for non-local
    _oi = out_idx_dev[:_tr].to(torch.long)
    _local = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
    recv_topk = torch.where(_local, _oi.to(torch.int32),
                            torch.full_like(_oi, experts, dtype=torch.int32)).to(torch.int32).contiguous()
    recv_wts = torch.full((_tr, topk), 1.0 / topk, device=dev, dtype=torch.float32)

    # ---- sort buffers (over-allocate: M*topk + E*tile_m) ----
    max_pad = _tr * topk + experts * tile_m
    max_blocks = (max_pad + tile_m - 1) // tile_m
    scale_cols_pad = ((scale_mx_blocks + 7) // 8) * 8
    scale_rows_pad = ((max_pad + 31) // 32) * 32

    def _new_sort_bufs():
        return dict(
            st=torch.empty(max_pad, dtype=torch.int32, device=dev),
            sw=torch.empty(max_pad, dtype=torch.float32, device=dev),
            se=torch.empty(max_blocks, dtype=torch.int32, device=dev),
            nv=torch.zeros(2, dtype=torch.int32, device=dev),
            mbuf=torch.empty((_tr, model_dim), dtype=torch.float16, device=dev),
        )

    bias_d = torch.empty((0,), device=dev, dtype=torch.float32)
    osd = torch.empty((0,), device=dev, dtype=torch.uint8)

    # ---------- ATOM baseline (aiter sort stack) ----------
    # PRODUCTION-ALIGNED (--from-bf16): mirror how ATOM actually runs stage1, i.e.
    #   MoriPrepareAndFinalize.prepare (scale=None -> dispatch the BF16 token; Mori has no fp4
    #   dispatch, fp8 only under use_fp8_dispatch) followed by aiter.fused_moe stage1 which quantises
    #   AFTER dispatch with the quant-fused sort interface:
    #     bf16 dispatch -> moe_sorting_fwd -> fused_dynamic_mx{fp4,fp8}_quant_moe_sort(bf16_recv,
    #     sorted_ids, ...)  [quant + scale-sort in ONE]  -> mixed_moe_gemm1.
    # The activation-quant therefore runs on the bf16 RECEIVED tokens (not pre-quantised before
    # dispatch).  Without --from-bf16 we keep the legacy pre-quant path for back-compat.
    atom_body = None
    at_out = None
    # BASELINE SLIM: a4w4 has NO fp4 dispatch in production (Mori dispatches bf16 then quant-AFTER via
    # the auto/fused/split sort branches) -> atom_body IS the a4w4 baseline.  a8w4 uses the fp8-dispatch
    # quant-BEFORE path (atom_fp8 below) as its ONLY baseline, so atom_body is not built for a8w4.
    if HAS_AITER and not args.no_atom and args.quant == "a4w4":
        at = _new_sort_bufs()
        at_out = torch.zeros((_tr, topk, inter_dim), device=dev, dtype=torch.float16)
        if bool(getattr(args, "from_bf16", False)):
            from aiter import dtypes as _adt
            from aiter.ops.quant import (  # production quant-fused MoE sort + its forced sub-paths
                fused_dynamic_mxfp4_quant_moe_sort, fused_dynamic_mxfp8_quant_moe_sort,
                fused_dynamic_mx_quant_moe_sort_hip, per_1x32_mx_quant_hip)
            _qpath = getattr(args, "atom_quant_path", "auto")   # auto | fused | split (ATOM use_fused branch)
            _qd = _adt.fp4x2 if args.quant == "a4w4" else _adt.fp8
            _N = model_dim
            _out_cols = _N // 2 if args.quant == "a4w4" else _N
            _scaleN_pad = ((_N + 31) // 32 + 7) // 8 * 8
            _auto_qs = (fused_dynamic_mxfp4_quant_moe_sort if args.quant == "a4w4"
                        else fused_dynamic_mxfp8_quant_moe_sort)

            def _quant_sort(bt2, M):
                # ATOM stage1 quant + scale-sort on the bf16 received tokens.  Selectable per ATOM's
                # use_fused dispatch (aiter/ops/quant.py:953-957):
                #   auto  -> fused_dynamic_mx_quant_moe_sort: fused single kernel at M<=8*256/topk, else split
                #   fused -> force the single fused HIP kernel (quant+sort+swizzle, 1 launch; re-quant topk x)
                #   split -> force per_1x32_mx_quant_hip (quant each row once) + mxfp4_moe_sort_hip (byte sort)
                if _qpath == "auto":
                    return _auto_qs(bt2, sorted_ids=at["st"], num_valid_ids=at["nv"],
                                    token_num=int(M), topk=int(topk), block_size=int(tile_m))
                _scale = torch.empty(((at["st"].shape[0] + 31) // 32 * 32, _scaleN_pad),
                                     dtype=_adt.fp8_e8m0, device=dev)
                if _qpath == "fused":
                    _o = torch.empty(int(M), _out_cols, dtype=_qd, device=dev)
                    fused_dynamic_mx_quant_moe_sort_hip(_o, _scale, bt2, at["st"], at["nv"],
                                                        int(M), int(tile_m), 32)
                    return _o, _scale
                _stp = _adt.fp8_e8m0 if args.quant == "a8w4" else None
                _o, _spt = per_1x32_mx_quant_hip(bt2, scale=None, quant_dtype=_qd, scale_type=_stp,
                                                 shuffle=False, num_rows=None, num_rows_factor=1)
                aiter.mxfp4_moe_sort_hip(_scale, _spt, at["st"], at["nv"], int(M), _N)
                return _o, _scale
            # Separate BF16 dispatch op (no activation scale): production dispatches the bf16 token and
            # quantises later.  Same op/config/topk as `dc` => identical recv-slot routing, so the
            # recv_topk derived from `dc` aligns with this op's received bf16 rows.
            cfg_b = FlyDSLDispatchCombineConfig(
                rank=rank, world_size=world, hidden_dim=model_dim,
                max_num_inp_token_per_rank=mtpr, num_experts_per_rank=epr,
                num_experts_per_token=topk, data_type=torch.bfloat16,
                warp_num_per_block=int(args.warps), block_num=bn,
                scale_dim=0, scale_type_size=0, enable_std_moe=False,
            )
            dc_b = FlyDSLDispatchCombineIntraNodeOp(cfg_b)
            torch.cuda.synchronize(); ms.shmem_barrier_all()
            dc_b.total_recv.zero_()                 # warm: establish P2P handshake state pre-capture
            dc_b.dispatch(x_bf16[:run_tokens].contiguous(), wts.contiguous(), None, topk_ids.contiguous())
            torch.cuda.synchronize(); ms.shmem_barrier_all()

            def _atom_body(stream):
                dc_b.total_recv.zero_()
                bt, _, _, _, _ = dc_b.dispatch(x_bf16[:run_tokens].contiguous(), wts.contiguous(),
                                               None, topk_ids.contiguous())
                aiter.moe_sorting_fwd(recv_topk, recv_wts, at["st"], at["sw"], at["se"], at["nv"],
                                      at["mbuf"], int(experts), int(tile_m), None, None, 0)
                # quant (bf16 recv -> fp4/fp8) + scale-sort via the selected ATOM path (auto/fused/split)
                a1q, a1s = _quant_sort(bt[:_tr].contiguous(), _tr)
                gemm(at_out, _agv(a1q), w_kernel, a1s.view(torch.uint8), scale_w1_1d,
                     at["st"], at["se"], at["sw"], at["nv"], bias_d, osd,
                     fx.Int32(_tr), fx.Int32(inter_dim * 2), fx.Int32(model_dim),
                     fx.Int32(int(max_blocks)), stream=stream)
            atom_body = _atom_body
        else:
            at_sx = torch.empty((scale_rows_pad, scale_cols_pad), dtype=torch.uint8, device=dev)

            def _atom_body(stream):
                dc.total_recv.zero_()
                _xi, _sxi = _x_in()
                dc.dispatch(_xi, wts.contiguous(), _sxi, topk_ids.contiguous())
                aiter.moe_sorting_fwd(recv_topk, recv_wts, at["st"], at["sw"], at["se"], at["nv"],
                                      at["mbuf"], int(experts), int(tile_m), None, None, 0)
                aiter.mxfp4_moe_sort_hip(at_sx, rs[:_tr].contiguous(), at["st"], at["nv"],
                                         int(_tr), int(model_dim))
                gemm(at_out, rx_g, w_kernel, at_sx, scale_w1_1d, at["st"], at["se"], at["sw"], at["nv"],
                     bias_d, osd, fx.Int32(_tr), fx.Int32(inter_dim * 2), fx.Int32(model_dim),
                     fx.Int32(int(max_blocks)), stream=stream)
            atom_body = _atom_body

    # ---------- ATOM fp8-dispatch baseline (MoriPrepareAndFinalize use_fp8_dispatch) ----------
    # Quantize the bf16 source to fp8 BEFORE dispatch (FlyDSL ep_dispatch `dc` carries the fp8 token +
    # e8m0 scale -- Mori-equivalent), then aiter.moe_sorting_fwd + mxfp4_moe_sort_hip (scale-sort only,
    # input already quantized -> no re-quant) + mixed_moe_gemm1.  Mirrors how ATOM runs when fp8
    # dispatch is enabled (dispatch moves 1 byte/elem instead of bf16's 2).  Reported as atom_fp8.
    atom_fp8_body = None
    at_fp8_out = None
    if (HAS_AITER and not args.no_atom and getattr(args, "atom_fp8_dispatch", False)
            and bool(getattr(args, "from_bf16", False)) and args.quant == "a8w4"):  # fp8-dispatch = a8w4-only baseline
        at8 = _new_sort_bufs()
        at8_sx = torch.empty((scale_rows_pad, scale_cols_pad), dtype=torch.uint8, device=dev)
        at_fp8_out = torch.zeros((_tr, topk, inter_dim), device=dev, dtype=torch.float16)

        def _atom_fp8_body(stream):
            dc.total_recv.zero_()
            _xi, _sxi = _x_in()                         # bf16 -> fp8(+e8m0) BEFORE dispatch
            dc.dispatch(_xi, wts.contiguous(), _sxi, topk_ids.contiguous())  # dispatch fp8 + scale
            aiter.moe_sorting_fwd(recv_topk, recv_wts, at8["st"], at8["sw"], at8["se"], at8["nv"],
                                  at8["mbuf"], int(experts), int(tile_m), None, None, 0)
            aiter.mxfp4_moe_sort_hip(at8_sx, rs[:_tr].contiguous(), at8["st"], at8["nv"],
                                     int(_tr), int(model_dim))   # scale-sort only (no re-quant)
            gemm(at_fp8_out, rx_g, w_kernel, at8_sx, scale_w1_1d, at8["st"], at8["se"], at8["sw"], at8["nv"],
                 bias_d, osd, fx.Int32(_tr), fx.Int32(inter_dim * 2), fx.Int32(model_dim),
                 fx.Int32(int(max_blocks)), stream=stream)
        atom_fp8_body = _atom_fp8_body

    # ---------- mega (single-launch fused dispatch⊕GEMM megakernel) ----------
    fused_body = None
    facade = None
    if args.mega:
        from kernels.fused_moe_megakernel import FusedMoEMegaStage1
        facade = FusedMoEMegaStage1(
            rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
            experts=experts, topk=topk, quant=args.quant, w1=w_kernel, w1_scale=scale_w1_1d,
            network=args.network, scheme=args.mega_scheme, unit_size=-1, tile_n=-1,
            tile_k=tile_k, warp_num_per_block=int(args.warps), max_tok_per_rank=mtpr,
            waves_per_eu=int(args.waves_per_eu), use_async_copy=bool(args.async_copy))
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        def _facade_body(stream):
            _xi, _sxi = _x_in()
            facade.forward(_xi, wts.contiguous(), _sxi, topk_ids.contiguous(), stream=stream)
        fused_body = _facade_body

    # ============ accuracy: mega vs ATOM, mapped back to (k_slot, src_token), element-wise ======
    # Both sides decode each output row to the SAME 32-bit key  (k_slot<<24)|(src_pe*mtpr+src_tok):
    #   * mega : srcmap_em[row] already IS that key; value = facade._out[row].
    #   * atom : recv slot s, k-th expert (LOCAL only) -> key=(k<<24)|tok_id_to_src[s], value=at_out[s,k].
    # The ATOM ref = single-`dc` dispatch -> aiter.moe_sorting_fwd -> aiter.mxfp4_moe_sort_hip
    # (scale-sort) -> mixed_moe_gemm1 (the production aiter sort/scale-sort stack).  mega is CORRECT
    # iff same key set (no dropped/extra rows) AND every key matches element-wise within tol.
    if args.check_correctness and not (HAS_AITER and getattr(args, "mega", False)):
        _info(rank, "[bench] --check-correctness needs aiter + --mega; skipping")
    elif args.check_correctness:
        _strm = fx.Stream(torch.cuda.current_stream())
        # ---- ATOM reference (single dc; tok_id_to_src/out_idx/at_out all from this dc) ----
        dc.total_recv.zero_()
        _cxi, _csxi = _x_in()
        _rxt, _, _rst, _oidx, _ = dc.dispatch(_cxi, wts.contiguous(), _csxi, topk_ids.contiguous())
        torch.cuda.synchronize(); ms.shmem_barrier_all()
        _trc = max(1, int(dc.total_recv.item()))
        _oi = _oidx[:_trc].to(torch.long)                                  # (trc, topk) global experts
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        _rtopk = torch.where(_loc, _oi.to(torch.int32),
                             torch.full_like(_oi, experts, dtype=torch.int32)).to(torch.int32).contiguous()
        _rwts = torch.full((_trc, topk), 1.0 / topk, device=dev, dtype=torch.float32)
        _mp = _trc * topk + experts * tile_m
        _mb = (_mp + tile_m - 1) // tile_m
        _c_st = torch.empty(_mp, dtype=torch.int32, device=dev)
        _c_sw = torch.empty(_mp, dtype=torch.float32, device=dev)
        _c_se = torch.empty(_mb, dtype=torch.int32, device=dev)
        _c_nv = torch.zeros(2, dtype=torch.int32, device=dev)
        _c_mbuf = torch.empty((_trc, model_dim), dtype=torch.float16, device=dev)
        aiter.moe_sorting_fwd(_rtopk, _rwts, _c_st, _c_sw, _c_se, _c_nv, _c_mbuf,
                              int(experts), int(tile_m), None, None, 0)
        _scp = ((scale_mx_blocks + 7) // 8) * 8
        _srp = ((_mp + 31) // 32) * 32
        _c_sx = torch.empty((_srp, _scp), dtype=torch.uint8, device=dev)
        aiter.mxfp4_moe_sort_hip(_c_sx, _rst[:_trc].contiguous(), _c_st, _c_nv, int(_trc), int(model_dim))
        _at_out = torch.zeros((_trc, topk, inter_dim), device=dev, dtype=torch.float16)
        gemm(_at_out, _agv(_rxt), w_kernel, _c_sx, scale_w1_1d, _c_st, _c_se, _c_sw, _c_nv,
             bias_d, osd, fx.Int32(_trc), fx.Int32(inter_dim * 2), fx.Int32(model_dim),
             fx.Int32(int(_mb)), stream=_strm)
        torch.cuda.synchronize(); ms.shmem_barrier_all()
        _tis = dc.shmem_tok_id_to_src[:_trc].cpu().numpy()                 # recv slot -> src_pe*mtpr+src_tok
        _oicpu = _oi.cpu().numpy()
        _atc = _at_out.float().cpu().numpy()
        d_atom = {}
        for s in range(_trc):
            for k in range(topk):
                e = int(_oicpu[s, k])
                if rank * epr <= e < (rank + 1) * epr:
                    d_atom[(k << 24) | int(_tis[s])] = _atc[s, k]
        # ---- MEGA (real rows mapped via srcmap_em; fixedslot le*cap | handshake dense ebase) ----
        _mxi, _msxi = _x_in()
        facade.forward(_mxi, wts.contiguous(), _msxi, topk_ids.contiguous(), stream=_strm)
        torch.cuda.synchronize(); ms.shmem_barrier_all()
        _sm = facade.op.srcmap_em.cpu().numpy()
        _cnt = facade.op.ll_count.cpu().numpy()
        _mout = facade._out[:, 0, :].float().cpu().numpy()
        _cap = facade.cap; _tm = facade.unit_size
        d_meg = {}
        _eb = 0
        for le in range(epr):
            c = int(_cnt[le])
            base = le * _cap if facade.scheme == "fixedslot" else _eb
            for r in range(base, base + c):
                d_meg[int(_sm[r])] = _mout[r]
            _eb += ((c + _tm - 1) // _tm) * _tm
        # ---- compare (element-wise, tolerance; atom vs mega are different stacks -> not bit-exact) ----
        import numpy as _np
        _kr, _km = set(d_atom), set(d_meg)
        _common = _kr & _km
        _maxe = 0.0
        _nbad = 0
        for kk in _common:
            e = float(_np.abs(d_atom[kk] - d_meg[kk]).max())
            _maxe = max(_maxe, e)
            if e > 1e-2:
                _nbad += 1
        _t = torch.tensor([len(_kr), len(_km), len(_kr - _km), len(_km - _kr), _nbad, int(_maxe * 1e6)],
                          device=dev, dtype=torch.long)
        dist.all_reduce(_t, op=dist.ReduceOp.SUM)
        _na, _nm, _oa, _om, _nb, _me6 = [int(v) for v in _t.cpu()]
        _ok = (_oa == 0 and _om == 0 and _nb == 0)
        if rank == 0:
            print(f"[CORRECTNESS] mega({args.mega_scheme}) vs atom {args.network} {args.quant} "
                  f"bs={run_tokens}: atom_keys={_na} mega_keys={_nm} atom_only={_oa} mega_only={_om} "
                  f"mismatch_rows={_nb} maxerr={_me6 / 1e6:.2e} -> {'PASS' if _ok else 'FAIL'}", flush=True)
    # ============ cudagraph timing ============
    def _capture(body):
        torch.cuda.synchronize(); ms.shmem_barrier_all()
        body(fx.Stream(torch.cuda.current_stream()))
        torch.cuda.synchronize(); ms.shmem_barrier_all()
        cap = torch.cuda.Stream(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cap):
            body(fx.Stream(torch.cuda.current_stream()))
        return g

    _prof_only = set(s for s in (getattr(args, "prof_only", "") or "").split(",") if s)

    def _time(body, label):
        if body is None:
            return -1.0
        if _prof_only and label not in _prof_only:
            return -1.0  # rocprofv3 isolation: only capture+replay the selected path(s)
        try:
            g = _capture(body)
            for _ in range(max(5, int(args.warmup))):
                g.replay()
            torch.cuda.synchronize(); ms.shmem_barrier_all()
            if bool(getattr(args, "profiler_time", False)):
                return _all_mean(dev, _profiler_ms(g.replay, int(args.iters), int(args.warmup), label))
            return _all_mean(dev, _cuda_event_ms(g.replay, int(args.iters), int(args.warmup)))
        except Exception as e:  # noqa: BLE001
            _info(rank, f"[bench] {label} cudagraph FAILED: {type(e).__name__}: {e}")
            return -1.0

    atom_ms = _time(atom_body, "atom")
    atom_fp8_ms = _time(atom_fp8_body, "atom_fp8")
    fused_ms = _time(fused_body, "fused")

    def _sp(a, b):
        return (a / b) if (a > 0 and b > 0) else -1.0

    _fp8_str = (f" atom_fp8={atom_fp8_ms:.4f} fused/atom_fp8={_sp(atom_fp8_ms, fused_ms):.3f}"
                if atom_fp8_ms > 0 else "")
    _fused_tag = f"mega({args.mega_scheme})" if getattr(args, "mega", False) else "fused"
    _info(rank, f"[bench] kind={_fused_tag} {args.network} bs={run_tokens} quant={args.quant} | "
                f"atom={atom_ms:.4f} fused={fused_ms:.4f} | "
                f"fused/atom={_sp(atom_ms, fused_ms):.3f}{_fp8_str} (tr={tr})")

    return dict(
        network=args.network, quant=args.quant, world_size=world, tokens=run_tokens,
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        fused_kind=(f"mega_{args.mega_scheme}" if getattr(args, "mega", False) else "facade_2launch"),
        timing_mode="cudagraph_mean_across_ranks",
        atom_ms=atom_ms, atom_fp8_ms=atom_fp8_ms, fused_ms=fused_ms,
        speedup_fused_over_atom=_sp(atom_ms, fused_ms),
        speedup_fused_over_atom_fp8=_sp(atom_fp8_ms, fused_ms),
        total_recv=tr,
    )


def _print_result(r):
    _afp8 = r.get("atom_fp8_ms", -1.0)
    _afp8_str = (f"atom_fp8={_afp8:.4f} fused/atom_fp8={r.get('speedup_fused_over_atom_fp8', -1.0):.3f} | "
                 if _afp8 and _afp8 > 0 else "")
    print(f"[RESULT] net={r['network']} quant={r['quant']} bs={r['tokens']} | "
          f"atom={r['atom_ms']:.4f} fused={r['fused_ms']:.4f} | "
          f"{_afp8_str}"
          f"fused/atom={r['speedup_fused_over_atom']:.3f} tr={r['total_recv']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", type=str, default="v4_flash", choices=list(NETWORKS))
    p.add_argument("--quant", type=str, default="a8w4", choices=["a8w4", "a4w4"])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=-1,
                   help="-1 (default) = use the network's native topk (r1_v3=8, v4_*=6); >0 overrides")
    p.add_argument("--tile-k", type=int, default=256)
    p.add_argument("--waves-per-eu", type=int, default=4)
    p.add_argument("--async-copy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cshuffle", type=int, default=-1)
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--master-port", type=int, default=29921)
    p.add_argument("--check-correctness", action="store_true",
                   help="accuracy: mega vs ATOM, each output row mapped back to its source token "
                        "(k_slot, src) and compared element-wise (needs aiter + --mega).")
    p.add_argument("--no-atom", action="store_true", help="skip the aiter ATOM baseline")
    p.add_argument("--mega", action="store_true",
                   help="time the SINGLE-LAUNCH fused dispatch+GEMM megakernel (FusedMoEMegaStage1); "
                        "the 'fused' column reports the megakernel vs atom / atom_fp8.")
    p.add_argument("--mega-scheme", choices=["fixedslot", "handshake"], default="fixedslot",
                   help="megakernel dispatch scheme: 'fixedslot' (decode, persistent GEMM) or "
                        "'handshake' (prefill producer/consumer overlap).")
    p.add_argument("--profiler-time", action="store_true", help="time via torch.profiler device_time (cross-check vs CUDA event)")
    p.add_argument("--from-bf16", action="store_true",
                   help="production E2E: quantize the bf16 source with per_1x32_mx_quant_hip INSIDE "
                        "every timed body (baseline + fused), so activation-quant cost is included.")
    p.add_argument("--atom-quant-path", choices=["auto", "fused", "split"], default="auto",
                   help="ATOM baseline (--from-bf16) stage1 quant+scale-sort path: 'auto' = ATOM's "
                        "fused_dynamic_mx_quant_moe_sort use_fused dispatch (fused at small M, split at "
                        "large M); 'fused' = force the single fused HIP kernel everywhere; 'split' = force "
                        "per_1x32_mx_quant_hip + mxfp4_moe_sort_hip everywhere.")
    p.add_argument("--atom-fp8-dispatch", action="store_true",
                   help="add a 2nd ATOM baseline mirroring MoriPrepareAndFinalize use_fp8_dispatch: "
                        "quantize the bf16 source to fp8 BEFORE dispatch (FlyDSL ep_dispatch carries "
                        "the fp8 payload + e8m0 scale, Mori-equivalent), then aiter.moe_sorting_fwd + "
                        "mxfp4_moe_sort_hip (scale-sort only, no re-quant) + mixed_moe_gemm1.  Requires "
                        "--from-bf16; reported as atom_fp8.")
    p.add_argument("--prof-only", type=str, default="",
                   help="rocprofv3 isolation: comma list of paths to capture+replay "
                        "(atom,atom_fp8,fused); others are skipped so the kernel trace "
                        "contains only the selected path(s). Empty = time all.")
    p.add_argument("--matrix", action="store_true", help="run all networks x classic bs")
    p.add_argument("--full-bs", action="store_true", help="use the full bs sweep (1..32768)")
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, args.master_port)
    dev = torch.device("cuda", local_rank)

    bs_set = FULL_BS if args.full_bs else CLASSIC_BS
    if args.matrix:
        combos = [(net, bs) for net in NETWORKS for bs in bs_set]
    else:
        combos = [(args.network, int(args.tokens))]

    results = []
    for net, bs in combos:
        args.network = net
        args.tokens = bs
        try:
            r = run_one(args, rank, world, dev)
            if r is not None:
                results.append(r)
                if rank == 0:
                    _print_result(r)
        except Exception as e:  # noqa: BLE001
            import traceback
            if rank == 0:
                traceback.print_exc()
            _info(rank, f"[bench] {net} bs={bs} ERROR: {type(e).__name__}: {e}")
        torch.cuda.synchronize(); dist.barrier()

    if rank == 0 and args.json_out:
        with open(args.json_out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        _info(rank, f"[bench] wrote {len(results)} rows -> {args.json_out}")
    if rank == 0 and not args.matrix:
        for r in results:
            _print_result(r)

    torch.cuda.synchronize(); dist.barrier(); _cleanup()


if __name__ == "__main__":
    main()
