#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal STAGE-1-only probe for megamoeexp on gfx942 (MI308).

Skips stage-2 (gemm2->combine) entirely.  Builds MegaMoEExp with use_async_copy=False (gfx942 lacks
the gfx950 128-bit buffer_load_to_lds), runs ONLY stage1.forward (dispatch+GEMM1 -> a2), and checks
it codegens + runs.  torchrun --standalone --nproc_per_node=4 megamoeexp/probe_stage1.py
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tests.kernels.test_moe_gemm as tmg  # noqa: E402

# gfx942 env aiter lacks per_1x32_mx_quant_hip; MegaMoEExp __init__ guards on its import (never used
# on the pre-quantized stage1 path).  Inject a tmg-based equivalent BEFORE the stage module imports.
try:
    import aiter.ops.quant as _aq  # noqa: E402
    if not hasattr(_aq, "per_1x32_mx_quant_hip"):
        def _shim(x, quant_dtype=None, scale_type=None):
            t = str(quant_dtype).lower()
            return (tmg._per_1x32_fp4_quant if ("fp4" in t or "f4" in t) else tmg._per_1x32_mxfp8_quant)(x.contiguous())
        _aq.per_1x32_mx_quant_hip = _shim
except Exception:  # noqa: BLE001
    pass

from tests.kernels.bench_moe_intranode_stage1_groupgemm import (  # noqa: E402
    NETWORKS,
    _cleanup,
    _prepare,
    _setup_dist,
)


def _qact(x, is_fp4):
    if is_fp4:
        q, s = tmg._per_1x32_fp4_quant(x.contiguous())
        return q.view(torch.float4_e2m1fn_x2).contiguous(), s.view(torch.uint8)
    q, s = tmg._per_1x32_mxfp8_quant(x.contiguous())
    return q.contiguous(), s.view(torch.uint8)


def main():
    net_name = os.environ.get("PROBE_NET", "v4_flash")
    quant = os.environ.get("PROBE_QUANT", "a8w4")
    tokens = int(os.environ.get("PROBE_TOKENS", "64"))
    frontier = os.environ.get("FUSED_MEGA_FRONTIER", "0") not in ("0", "", "false")
    async_copy = os.environ.get("PROBE_ASYNC", "0") not in ("0", "", "false")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, 29901)
    dev = torch.device("cuda", local_rank)

    net = NETWORKS[net_name]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    topk = int(net["topk"])
    epr = experts // world
    mtpr = max(16, tokens)

    from megamoeexp.fused_moe_stage1_stage2 import MegaMoEExp

    T = _prepare(dev, quant=quant, tokens=tokens, model_dim=model_dim, inter_dim=inter_dim,
                 experts=experts, topk=topk, seed=0, rank=rank, world=world, keep_ref=False)
    w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
    topk_ids, wts = T["topk_ids"], T["wts"]
    is_fp4 = (T["a_dtype"] == "fp4")

    # LOCAL expert slice of w1 (this rank's epr experts), like the bench.
    _wpe = w_kernel.numel() // experts
    _spe = scale_w1_1d.numel() // experts
    w1 = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
    w1s = scale_w1_1d.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()
    # w2 unused by stage1 but MegaMoEExp ctor needs shapes; give a dummy fp4 tensor of right size.
    w2 = torch.zeros(epr * model_dim * (inter_dim // 2), dtype=torch.uint8, device=dev)
    w2s = torch.zeros(epr * model_dim * (inter_dim // 32), dtype=torch.uint8, device=dev)

    if rank == 0:
        print(f"[probe] net={net_name} quant={quant} bs={tokens} world={world} "
              f"frontier={frontier} async={async_copy} epr={epr}", flush=True)

    moe = MegaMoEExp(
        rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk, quant=quant, w1=w1, w1_scale=w1s, w2=w2, w2_scale=w2s,
        max_tok_per_rank=mtpr, network=net_name, use_async_copy=async_copy, stage2_mode="fused")

    x_q, x_sc = _qact(T["x_bf16"][:tokens], is_fp4)
    wc = wts[:tokens].contiguous()
    ic = topk_ids[:tokens].to(torch.int32).contiguous()

    import mori.shmem as ms
    ms.shmem_barrier_all()
    out = moe.stage1.forward(x_q, wc, x_sc, ic)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    a2 = out["out"]
    nv = out["num_valid"]
    if rank == 0:
        print(f"[probe] STAGE1 OK: a2 shape={tuple(a2.shape)} dtype={a2.dtype} "
              f"num_valid={nv.flatten()[:2].tolist()} a2.abs().sum()={a2.float().abs().sum().item():.3e}",
              flush=True)
    _cleanup()


if __name__ == "__main__":
    main()
