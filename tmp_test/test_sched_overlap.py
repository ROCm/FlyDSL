#!/usr/bin/env python3
"""A/B correctness (+ device-time) for the dynamic claim scheduler in the experimental MegaMoEExp.

Compares the SAME MegaMoEExp end-to-end output with the dynamic claim scheduler OFF vs ON:
  OFF: FLYDSL_TMP_SCHED unset  -> static round-robin GEMM consumer (current behavior)
  ON : FLYDSL_TMP_SCHED=1 (+ FLYDSL_TMP_FORCE_COMPACT=1) -> single-launch barrier-free claim loop

The claim scheduler only REORDERS which block computes which tile (per-tile readiness via l1_ready),
so the math is identical -> ON-vs-OFF relL2 must be ~0.  This is the M2 (GEMM1 claim) gate; once M3
fuses GEMM2 it stays the same gate end-to-end.

Run on the 8-GPU node (production /home/ghu/FlyDSL; here uses the docker-overlay flydsl via PYTHONPATH):
    python tmp_test/test_sched_overlap.py --network v4_pro --bs-list 64,256 --iters 30
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mori.shmem as ms  # noqa: E402
from tests.kernels.bench_moe_intranode_stage1_groupgemm import (  # noqa: E402
    NETWORKS, _prepare, _setup_dist, _cleanup, _all_mean, _all_max, _info,
    _chunked_fp4_quant,
)

# MegaMoEExp only needs aiter's activation quant (per_1x32_mx_quant_hip), NOT the baseline
# moe_sorting ops that bench's HAS_AITER also requires -> check the quant op specifically.
try:
    from aiter.ops.quant import per_1x32_mx_quant_hip as _PQ  # noqa: F401
    HAS_AITER = True
except Exception:  # noqa: BLE001
    HAS_AITER = False
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.utils import shuffle_weight  # noqa: E402


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _relL2(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = float(((a - b) ** 2).sum()); d = float((b ** 2).sum())
    return (n / d) ** 0.5 if d > 0 else (0.0 if n == 0 else -1.0)


def _build_w2_local(args, rank, world, dev, *, model_dim, inter_dim, experts, epr):
    torch.manual_seed(args.seed + 4242)
    w2_f32 = (torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32)
              * (float(inter_dim) ** -0.25))
    w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
    _sl = slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
    w2k = shuffle_weight(w2_fp4[_sl]).view(torch.uint8).contiguous().view(-1)
    w2s = fp4_utils.e8m0_shuffle(w2_sr[_sl]).view(torch.uint8).contiguous().view(-1)
    del w2_f32, w2_fp4, w2_sr; torch.cuda.empty_cache()
    return w2k, w2s


def _run_mode(args, rank, world, dev, *, net, bs, T, w2k, w2s, sched_on):
    from kernels.megamoe_exp import MegaMoEExp
    model_dim, inter_dim, experts, topk = net["model_dim"], net["inter_dim"], net["experts"], net["topk"]
    epr = experts // world
    mtpr = _next_pow2(max(16, bs))
    # Force compact in BOTH modes so the A/B isolates ONLY the claim scheduler (compact static
    # round-robin vs compact claim loop); otherwise OFF=fixedslot vs ON=compact would conflate two
    # changes.  Only FLYDSL_TMP_SCHED differs between the two runs.
    os.environ["FLYDSL_TMP_FORCE_COMPACT"] = "1"
    if sched_on:
        os.environ["FLYDSL_TMP_SCHED"] = "1"
    else:
        os.environ.pop("FLYDSL_TMP_SCHED", None)
    w_kernel, scale_w1 = T["w_kernel"], T["scale_w1_1d"]
    _wpe = w_kernel.numel() // experts; _spe = scale_w1.numel() // experts
    w1 = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
    w1s = scale_w1.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()
    moe = MegaMoEExp(
        rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk, quant="a8w4", w1=w1, w1_scale=w1s,
        w2=w2k, w2_scale=w2s, max_tok_per_rank=mtpr, network=net_name(args), stage2_mode="fused")
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    sched_active = bool(getattr(moe.stage1, "_tmp_sched", False))
    wc = T["wts"][:bs].contiguous()
    ic = T["topk_ids"][:bs].to(torch.int32).contiguous()
    xb = T["x_bf16"][:bs].contiguous()

    def _body():
        return moe.forward_bf16(xb, wc, ic)

    # correctness snapshot
    out = _body()
    torch.cuda.synchronize()
    snap = out[:bs].float().cpu().numpy().copy()

    # device-time
    ms.shmem_barrier_all(); _body(); torch.cuda.synchronize(); ms.shmem_barrier_all()
    g = torch.cuda.CUDAGraph(); _cap = torch.cuda.Stream()
    with torch.cuda.graph(g, stream=_cap):
        _body()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    n = max(1, int(args.iters)); s.record()
    for _ in range(n):
        g.replay()
    e.record(); torch.cuda.synchronize()
    ms_t = _all_mean(dev, s.elapsed_time(e) / n)

    del moe; torch.cuda.empty_cache(); ms.shmem_barrier_all()
    return ms_t, sched_active, snap


def net_name(args):
    return args.network


def _bench_bs(args, rank, world, dev, net, bs):
    from aiter import dtypes as _adt  # noqa: F401  (ensure aiter present)
    model_dim, inter_dim, experts, topk = net["model_dim"], net["inter_dim"], net["experts"], net["topk"]
    epr = experts // world
    T = _prepare(dev, quant="a8w4", tokens=bs, model_dim=model_dim, inter_dim=inter_dim,
                 experts=experts, topk=topk, seed=args.seed, rank=rank, world=world)
    w2k, w2s = _build_w2_local(args, rank, world, dev, model_dim=model_dim,
                               inter_dim=inter_dim, experts=experts, epr=epr)
    off_ms, off_act, out_off = _run_mode(args, rank, world, dev, net=net, bs=bs, T=T, w2k=w2k, w2s=w2s, sched_on=False)
    on_ms, on_act, out_on = _run_mode(args, rank, world, dev, net=net, bs=bs, T=T, w2k=w2k, w2s=w2s, sched_on=True)
    rl = _all_max(dev, _relL2(out_on, out_off))
    del T, w2k, w2s, out_off, out_on; torch.cuda.empty_cache()
    return dict(bs=bs, off_ms=off_ms, on_ms=on_ms, relL2=rl, off_act=off_act, on_act=on_act,
                speedup=(off_ms / on_ms) if on_ms > 0 else -1.0)


def _worker(rank, world, args):
    try:
        _setup_dist(rank, world, args.master_port)
        dev = torch.device("cuda", rank)
        if not HAS_AITER:
            _info(rank, "[sched-test] needs aiter; abort"); return
        net = NETWORKS[args.network]
        rows = []
        for bs in [int(b) for b in args.bs_list.split(",") if b.strip()]:
            try:
                r = _bench_bs(args, rank, world, dev, net, bs)
                rows.append(r)
                if rank == 0:
                    ok = (0.0 <= r["relL2"] <= float(args.tol)) and r["on_act"] and (not r["off_act"])
                    print(f"[{args.network} sched-AB] bs={bs:>5}  OFF={r['off_ms']*1000:8.2f}us  "
                          f"ON={r['on_ms']*1000:8.2f}us  speedup={r['speedup']:.3f}x  "
                          f"relL2(on,off)={r['relL2']:.2e}  sched[off={r['off_act']},on={r['on_act']}]  "
                          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
            except Exception as ex:  # noqa: BLE001
                if rank == 0:
                    print(f"[{args.network} sched-AB] bs={bs} ERROR: {type(ex).__name__}: {ex}", flush=True)
                torch.cuda.empty_cache()
                try:
                    ms.shmem_barrier_all()
                except Exception:
                    pass
        if rank == 0 and rows:
            print(f"\n==== SUMMARY ({args.network} sched-AB end-to-end, a8w4) ====", flush=True)
            for r in rows:
                ok = (0.0 <= r["relL2"] <= float(args.tol)) and r["on_act"] and (not r["off_act"])
                print(f"  bs={r['bs']:>5}  speedup={r['speedup']:.3f}x  relL2={r['relL2']:.2e}  "
                      f"{'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        _cleanup()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", type=str, default="v4_pro", choices=list(NETWORKS))
    p.add_argument("--bs-list", type=str, default="64,256")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--tol", type=float, default=2e-2, help="max relL2(on,off) for PASS (quant-level)")
    p.add_argument("--master-port", type=int, default=29591)
    args = p.parse_args()
    world = int(args.world)
    if world > torch.cuda.device_count():
        raise SystemExit(f"need {world} GPUs, have {torch.cuda.device_count()}")
    mp.spawn(_worker, args=(world, args), nprocs=world, join=True)


if __name__ == "__main__":
    main()
