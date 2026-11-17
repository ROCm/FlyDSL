#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Reference harness: search RMSNorm (``kernels/norm/rmsnorm_kernel.py``) and
export the winner through the ``@autotune`` offline-artifact path.

    FLYDSL_AUTOTUNE_CONFIG_DIR=/tmp/art python3 \
        .claude/skills/search-optimize-flydsl/scripts/tune_rmsnorm.py --m 8192 --n 8192 --cold

A memory-bound kernel: use ``--cold`` so the roofline fraction is HBM-bound.
"""

import argparse
import json
import sys

import torch

from flydsl.tuning import Budget, Fidelity, Param, SearchSpace, tune
from flydsl.tuning.artifacts import export_winner
from flydsl.tuning.flybackend import FlyBackend
from flydsl.tuning.roofline import Roofline, measure_peaks
from kernels.norm.rmsnorm_autotune import _rmsnorm_tuner
from kernels.norm.rmsnorm_kernel import default_block_threads, rmsnorm_direct


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--strategy", default="auto")
    ap.add_argument("--evals", type=int, default=100)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--export", action="store_true", help="write the winner as an offline artifact")
    args = ap.parse_args()
    M, N = args.m, args.n

    dev = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    gamma = torch.randn(N, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ref = x.float() * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-5) * gamma.float()
    call = dict(N=N, dtype_str="bf16", weight_dtype_str="bf16", stream=torch.cuda.current_stream())

    backend = FlyBackend(
        rmsnorm_direct,
        (x, gamma, out, M),
        kwargs=call,
        oracle=lambda: (torch.allclose(out.float(), ref, rtol=2e-2, atol=2e-2), "vs fp32 reference"),
        prepare=lambda: out.fill_(float("nan")),
    )
    space = SearchSpace(
        [
            Param.ordinal("BLOCK_THREADS", (64, 128, 256, 512, 1024)),
            Param.categorical("waves_per_eu", (None, 1, 2, 3, 4)),
        ],
        name="rmsnorm",
    )
    shipped = {"BLOCK_THREADS": default_block_threads(N), "waves_per_eu": None}
    res = tune(
        space,
        backend,
        strategy=args.strategy,
        budget=Budget(evaluations=args.evals),
        baseline=space.encode(shipped),
        confirm=3,
        confirm_rounds=5,
        ladder=(Fidelity(reps=10, warmup=2, cold=args.cold),),
    )
    rl = Roofline(5.0 * M * N, 2.0 * M * N * 2 + N * 2, measure_peaks(), "bf16")
    report = {
        "shape": [M, N],
        "best_config": res.best_config,
        "confirmed_speedup": res.confirmed.get("speedup"),
        "p_value": res.confirmed.get("p_value"),
        "roofline": rl.report(res.best_value * 1e3),
    }
    if args.export:
        report["artifact"] = str(export_winner(_rmsnorm_tuner, res.best_config, x, gamma, out, M, **call))
    json.dump(report, sys.stdout, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
