#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolate why frontier ExpS1(ON) is bit-exact vs ExpS1(OFF) in isolation, yet wrong in the bench.

Bench runs exp(ON).forward() then prod.forward() BACK-TO-BACK on one stream with NO barrier
between; this script reproduces that and compares against the SAME pair with a barrier between,
to test for a cross-launch interference in the frontier path.

torchrun --standalone --nproc_per_node=8 megamoeexp/diag_accuracy.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

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

from megamoeexp.bench_megamoeexp import NETWORKS, _cleanup, _prepare, _qact, _setup_dist  # noqa: E402


def _build(cls, frontier, **kw):
    _saved = os.environ.get("FUSED_MEGA_FRONTIER")
    os.environ["FUSED_MEGA_FRONTIER"] = "1" if frontier else "0"
    os.environ["FUSED_MEGA_PREFETCH_KTILES"] = "0"
    try:
        return cls(**kw)
    finally:
        if _saved is None:
            os.environ.pop("FUSED_MEGA_FRONTIER", None)
        else:
            os.environ["FUSED_MEGA_FRONTIER"] = _saved


def main():
    net_name = os.environ.get("PROBE_NET", "v4_pro")
    quant = os.environ.get("PROBE_QUANT", "a8w4")
    tokens = int(os.environ.get("PROBE_TOKENS", "64"))

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, 29962)
    dev = torch.device("cuda", local_rank)

    net = NETWORKS[net_name]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    topk = int(net["topk"])
    epr = experts // world
    mtpr = 16
    while mtpr < tokens:
        mtpr <<= 1

    from kernels.fused_moe_megakernel import FusedMoEMegaStage1 as ProdS1
    from megamoeexp.fused_moe_megakernel import FusedMoEMegaStage1 as ExpS1

    T = _prepare(dev, quant=quant, tokens=tokens, model_dim=model_dim, inter_dim=inter_dim,
                 experts=experts, topk=topk, seed=0, rank=rank, world=world, keep_ref=False)
    w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
    is_fp4 = T["a_dtype"] == "fp4"
    _wpe = w_kernel.numel() // experts
    _spe = scale_w1_1d.numel() // experts
    w1 = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
    w1s = scale_w1_1d.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()

    kw = dict(rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim, experts=experts,
              topk=topk, quant=quant, w1=w1, w1_scale=w1s, max_tok_per_rank=mtpr, network=net_name,
              out_dtype="f16")
    exp = _build(ExpS1, True, **kw)       # frontier ON
    prod = _build(ProdS1, False, **kw)    # production OFF
    a2rows = world * mtpr * topk

    x_q, x_sc = _qact(T["x_bf16"][:tokens], is_fp4)
    wc = T["wts"][:tokens].contiguous()
    ic = T["topk_ids"][:tokens].to(torch.int32).contiguous()

    def _rl(a, b):
        n = float(((a - b) ** 2).sum()); d = float((b ** 2).sum())
        return (n / d) ** 0.5 if d > 0 else -1.0

    # ---- Flow A: bench-style, NO barrier between exp and prod ----
    ms.shmem_barrier_all()
    eoA = exp.forward(x_q, wc, x_sc, ic)["out"].reshape(a2rows, inter_dim).float().clone()
    poA = prod.forward(x_q, wc, x_sc, ic)["out"].reshape(a2rows, inter_dim).float().clone()
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    eoA = eoA.cpu().numpy(); poA = poA.cpu().numpy()

    # ---- Flow B: barrier BETWEEN exp and prod ----
    ms.shmem_barrier_all()
    eoB = exp.forward(x_q, wc, x_sc, ic)["out"].reshape(a2rows, inter_dim).float().clone()
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    poB = prod.forward(x_q, wc, x_sc, ic)["out"].reshape(a2rows, inter_dim).float().clone()
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    eoB = eoB.cpu().numpy(); poB = poB.cpu().numpy()

    if rank == 0:
        print(f"[acc2] net={net_name} bs={tokens} epr={epr} a2rows={a2rows}", flush=True)
        print(f"[acc2] FLOW A (no barrier between): relL2(exp,prod)={_rl(eoA,poA):.3e}", flush=True)
        print(f"[acc2] FLOW B (barrier between):    relL2(exp,prod)={_rl(eoB,poB):.3e}", flush=True)
        print(f"[acc2] exp output changed A->B: relL2(expA,expB)={_rl(eoA,eoB):.3e}", flush=True)
        print(f"[acc2] prod output changed A->B: relL2(prodA,prodB)={_rl(poA,poB):.3e}", flush=True)

    _cleanup()
    os._exit(0)


if __name__ == "__main__":
    main()
