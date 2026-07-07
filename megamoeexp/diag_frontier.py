#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Async-launch + side-stream buffer poll to localize the G1 frontier deadlock.

Launches ONE frontier=1 stage1.forward (async, no sync), then polls the dispatch
sync-buffers (running / expert_ready / cnt_ready / ll_count / meta) from the host on
a SEPARATE stream while the (hung) kernel occupies the CUs.  SDMA D2H copies proceed
even while compute is deadlocked, so the buffer snapshots reveal which phase stalled:

  producer  -> running (peer atomics) + expert_ready (per-(src,expert) epoch publish)
  block0    -> cnt_ready (per-expert "count ready") + ll_count + meta (final epoch)
  consumer  -> spins cnt_ready[e]

Interpretation:
  expert_ready has 0s              -> a producer never published (coverage/epoch bug)
  expert_ready all=epoch, cnt_ready all 0 -> block0 stuck in expert_ready wait
  cnt_ready partially 1            -> block0 stuck on a specific expert
  cnt_ready all 1, meta=0          -> block0 stuck at gb1 arrival wait
  cnt_ready all 1, meta=epoch      -> block0 DONE -> hang is in the consumer

torchrun --standalone --nproc_per_node=8 megamoeexp/diag_frontier.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("FUSED_MEGA_FRONTIER", "1")
os.environ.setdefault("FUSED_MEGA_PREFETCH_KTILES", "0")

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

from megamoeexp.bench_megamoeexp import NETWORKS, _cleanup, _prepare, _qact, _setup_dist  # noqa: E402


def _snap(side, t):
    """Blocking D2H copy on `side` stream (independent of the hung kernel's stream)."""
    h = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
    with torch.cuda.stream(side):
        h.copy_(t, non_blocking=True)
    side.synchronize()
    return h


def main():
    net_name = os.environ.get("PROBE_NET", "v4_pro")
    quant = os.environ.get("PROBE_QUANT", "a8w4")
    tokens = int(os.environ.get("PROBE_TOKENS", "64"))
    poll_s = float(os.environ.get("DIAG_POLL_S", "20"))

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, 29951)
    dev = torch.device("cuda", local_rank)

    net = NETWORKS[net_name]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    topk = int(net["topk"])
    epr = experts // world
    mtpr = 16
    while mtpr < tokens:
        mtpr <<= 1

    from megamoeexp.fused_moe_megakernel import FusedMoEMegaStage1 as ExpS1

    T = _prepare(dev, quant=quant, tokens=tokens, model_dim=model_dim, inter_dim=inter_dim,
                 experts=experts, topk=topk, seed=0, rank=rank, world=world, keep_ref=False)
    w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
    is_fp4 = T["a_dtype"] == "fp4"
    _wpe = w_kernel.numel() // experts
    _spe = scale_w1_1d.numel() // experts
    w1 = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
    w1s = scale_w1_1d.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()

    op = ExpS1(rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
               experts=experts, topk=topk, quant=quant, w1=w1, w1_scale=w1s,
               max_tok_per_rank=mtpr, network=net_name, out_dtype="f16")
    assert op._frontier, "frontier not enabled -- check FUSED_MEGA_FRONTIER"

    x_q, x_sc = _qact(T["x_bf16"][:tokens], is_fp4)
    wc = T["wts"][:tokens].contiguous()
    ic = T["topk_ids"][:tokens].to(torch.int32).contiguous()

    if rank == 0:
        print(f"[diag] net={net_name} quant={quant} bs={tokens} world={world} epr={epr} "
              f"experts={experts} frontier={op._frontier}", flush=True)

    side = torch.cuda.Stream()
    ms.shmem_barrier_all()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    # ---- async launch: enqueue the frontier kernel, DO NOT sync ----
    _ = op.forward(x_q, wc, x_sc, ic)
    done_ev = torch.cuda.Event()
    done_ev.record()  # completes only when the kernel (on the same stream) finishes

    dop = op.op
    t0 = time.time()
    last = None
    while time.time() - t0 < poll_s:
        time.sleep(1.0)
        if done_ev.query():
            if rank == 0:
                print(f"[diag] kernel COMPLETED at t={time.time()-t0:.1f}s -> NO deadlock", flush=True)
            _cleanup()
            os._exit(0)
        run = _snap(side, dop.running).numpy()
        er = _snap(side, dop.expert_ready).numpy().reshape(world, epr)
        cr = _snap(side, dop.cnt_ready).numpy()
        ll = _snap(side, dop.ll_count[:epr]).numpy()
        meta = int(_snap(side, op._meta).numpy()[0])
        _rem = None
        # per-peer count of published expert_ready entries (>=1 means epoch published this launch)
        er_pub = [int((er[p] >= 1).sum()) for p in range(world)]
        _remstr = f" remaining(nz={_rmnz},sum={_rmsum})" if _rem is not None else ""
        line = (f"[r{rank} t={time.time()-t0:4.1f}s] running(nz={int((run>0).sum())},sum={int(run.sum())}) "
                f"expert_ready_pub/peer={er_pub} cnt_ready(set={int((cr>0).sum())}/{epr}) "
                f"ll_count(nz={int((ll>0).sum())},sum={int(ll.sum())}) meta={meta}{_remstr}")
        if line != last:
            print(line, flush=True)
            last = line
    if rank == 0:
        print("[diag] poll window elapsed; kernel still resident => DEADLOCK confirmed", flush=True)
    # abandon the hung kernel; process teardown frees the context.
    _cleanup()
    os._exit(0)


if __name__ == "__main__":
    main()
