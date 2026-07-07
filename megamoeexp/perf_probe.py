#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LOW-MEMORY stage-1 perf probe (fits in ~20 GiB/GPU, runs even when VRAM is partly occupied).

Unlike bench_megamoeexp.py's _prepare (which builds the FULL 384-expert fp32 weight tensor = 63 GiB
per rank for the oracle), this generates ONLY each rank's LOCAL epr experts' weights (~8 GiB) since
perf timing does not need the oracle.  exp(frontier) and prod(production) get the SAME local weights.

CUDAGraph device time (us), mean across ranks.  Env attribution knobs (const_expr-gated kernels):
  FUSED_MEGA_SKIP_GEMM=1  -> consumer loop zeroed (dispatch-only time)
  FUSED_MEGA_SKIP_BODY=1  -> consumer loop kept, GEMM body skipped (dispatch + per-tile overhead)

Run: MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=8 megamoeexp/perf_probe.py \
        --network v4_pro --quant a8w4 --tokens 64 [--frontier] [--prefetch-ktiles N]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tests.kernels.test_moe_gemm as tmg  # noqa: E402

try:
    import aiter.ops.quant as _aq  # noqa: E402

    if not hasattr(_aq, "per_1x32_mx_quant_hip"):

        def _shim(x, quant_dtype=None, scale_type=None):
            t = str(quant_dtype).lower()
            return (tmg._per_1x32_fp4_quant if ("fp4" in t or "f4" in t) else tmg._per_1x32_mxfp8_quant)(x.contiguous())

        _aq.per_1x32_mx_quant_hip = _shim
except Exception:  # noqa: BLE001
    pass

import mori.shmem as ms  # noqa: E402

from megamoeexp.bench_megamoeexp import NETWORKS, _cg_time, _cleanup, _qact, _setup_dist  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.utils import shuffle_weight  # noqa: E402


def _build(cls, frontier, prefetch, **kw):
    _s = {k: os.environ.get(k) for k in ("FUSED_MEGA_FRONTIER", "FUSED_MEGA_PREFETCH_KTILES")}
    os.environ["FUSED_MEGA_FRONTIER"] = "1" if frontier else "0"
    os.environ["FUSED_MEGA_PREFETCH_KTILES"] = str(prefetch)
    try:
        return cls(**kw)
    finally:
        for k, v in _s.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="v4_pro", choices=list(NETWORKS))
    p.add_argument("--quant", default="a8w4", choices=["a8w4", "a4w4"])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frontier", action="store_true")
    p.add_argument("--prefetch-ktiles", type=int, default=0)
    p.add_argument("--master-port", type=int, default=29971)
    p.add_argument(
        "--skew",
        type=float,
        default=0.0,
        help="STRAGGLER harness: fraction [0,1] of each token's topk routed to a small HOT expert set "
        "(first experts//16).  0=uniform (symmetric).  >0 concentrates load on few experts -> those "
        "experts' dispatch finishes later (per-expert readiness variance) = the straggler scenario the "
        "G1 frontier is designed to overlap.  Only affects routing, not weights.",
    )
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, args.master_port)
    dev = torch.device("cuda", local_rank)

    net = NETWORKS[args.network]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    topk = int(net["topk"])
    epr = experts // world
    tokens = int(args.tokens)
    mtpr = 16
    while mtpr < tokens:
        mtpr <<= 1
    is_fp4 = args.quant == "a4w4"

    # ---- LOCAL-ONLY weights, generated DIRECTLY as random fp4 bytes (~1 GiB, no 8 GiB fp32) ----
    # Perf timing does not depend on weight VALUES, only their shape/dtype/layout, so skip the
    # fp32 gen + quant entirely and hand the kernel random fp4-packed bytes + random e8m0 scale.
    init_scale = float(model_dim) ** -0.25
    _wfp4 = torch.randint(0, 256, (epr * (2 * inter_dim), model_dim // 2), dtype=torch.uint8, device=dev)
    w1 = shuffle_weight(_wfp4.view(torch.float4_e2m1fn_x2)).view(torch.uint8).contiguous()
    _wsc = torch.randint(0, 256, (epr * (2 * inter_dim), model_dim // 32), dtype=torch.uint8, device=dev)
    w1s = fp4_utils.e8m0_shuffle(_wsc).view(torch.uint8).contiguous()
    del _wfp4, _wsc
    torch.cuda.empty_cache()

    # ---- activation (small) ----
    torch.manual_seed(args.seed + 1)
    x_bf16 = (torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * init_scale).to(torch.bfloat16)
    torch.manual_seed(args.seed + 9973 + rank * 101)
    _skew = max(0.0, min(1.0, float(args.skew)))
    if _skew <= 0.0:
        topk_ids = torch.stack([torch.randperm(experts, device=dev)[:topk] for _ in range(tokens)]).to(torch.int32)
    else:
        # STRAGGLER: weighted sampling of topk DISTINCT experts, HOT set (first experts//16) up-weighted
        # so it absorbs most tokens -> hot experts finish dispatch late (per-expert readiness variance).
        _nhot = max(1, experts // 16)
        _w = torch.ones(experts, device=dev)
        _w[:_nhot] = 1.0 + _skew * 200.0  # higher skew -> hotter
        topk_ids = torch.multinomial(_w.expand(tokens, experts), topk, replacement=False).to(torch.int32)
    wts = torch.full((tokens, topk), 1.0 / topk, device=dev, dtype=torch.float32)
    x_q, x_sc = _qact(x_bf16, is_fp4)

    from kernels.fused_moe_megakernel import FusedMoEMegaStage1 as ProdS1
    from megamoeexp.fused_moe_megakernel import FusedMoEMegaStage1 as ExpS1

    kw = dict(rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim, experts=experts,
              topk=topk, quant=args.quant, w1=w1, w1_scale=w1s, max_tok_per_rank=mtpr,
              network=args.network, out_dtype="auto")
    exp = _build(ExpS1, args.frontier, args.prefetch_ktiles, **kw)
    prod = _build(ProdS1, False, 0, **kw)
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    def _eb():
        exp.forward(x_q, wts, x_sc, topk_ids)

    def _pb():
        prod.forward(x_q, wts, x_sc, topk_ids)

    t_exp = _cg_time(args, dev, _eb) * 1e3
    t_prod = _cg_time(args, dev, _pb) * 1e3
    if rank == 0:
        _frt = "ON" if args.frontier else "off"
        _mode = "full"
        if os.environ.get("FUSED_MEGA_SKIP_GEMM", "0") not in ("0", "", "false"):
            _mode = "skip_gemm(dispatch-only)"
        elif os.environ.get("FUSED_MEGA_SKIP_BODY", "0") not in ("0", "", "false"):
            _mode = "skip_body(dispatch+per-tile)"
        print(f"[PERF] {args.network} {args.quant} bs={tokens} frontier={_frt} pf={args.prefetch_ktiles} "
              f"mode={_mode} -> exp(dev)={t_exp:8.2f}us  prod={t_prod:8.2f}us  exp-vs-prod={t_prod/t_exp:.3f}x",
              flush=True)
    torch.cuda.synchronize(); dist.barrier()
    _cleanup()
    os._exit(0)


if __name__ == "__main__":
    main()
