#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Reference harness: search the gfx950 bf16 GEMM (``kernels/gemm/gemm_a16w16_gfx950.py``).

    python3 .claude/skills/search-optimize-flydsl/scripts/tune_gemm_a16w16.py \
        --m 32 --n 8192 --k 8192 --strategy hill --evals 120 --cold

Copy this file as the starting point for another kernel: the pieces to replace
are the inputs, ``configure`` (config -> launcher call), ``valid`` (the builder's
own legality check), ``oracle`` and the roofline bytes. The baseline is the
configuration ``gemm_a16w16`` ships when no ``user_kwargs`` are given.
"""

import argparse
import json
import sys
import time

import torch

from flydsl.tuning import Budget, Fidelity, Param, SearchSpace, tune
from flydsl.tuning.flybackend import FlyBackend
from flydsl.tuning.roofline import gemm_roofline, measure_peaks
from kernels.gemm.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    _dynamic_tensor_arg,
    gemm_a16w16_gfx950,
    get_split_k_buffers,
    make_gemm_a16w16_param_and_validate,
)

SHIPPED = dict(
    block_m=256,
    block_n=256,
    block_k=64,
    stages=2,
    split_k=1,
    m_waves=2,
    n_waves=4,
    k_waves=1,
    group_m=0,
    use_half_tile_interleaved=True,
    waves_per_eu=None,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--strategy", default="auto")
    ap.add_argument("--evals", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold", action="store_true", help="flush the cache before each timed launch (memory-bound)")
    ap.add_argument("--rel-tol", type=float, default=1e-2, help="relative L2 error vs the fp32 reference")
    ap.add_argument("--trace", default=None, help="write the search trace (JSON lines) here")
    args = ap.parse_args()
    M, N, K = args.m, args.n, args.k

    dev = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16).t()  # (K, N) column-major: layout "nt"
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = a.float() @ b.float()
    stream = torch.cuda.current_stream()
    sem, sig = get_split_k_buffers(stream, dev)
    a_arg, b_arg, out_arg = _dynamic_tensor_arg(a, 1), _dynamic_tensor_arg(b, 0), _dynamic_tensor_arg(out, 1)

    def param_of(c):
        f = {k: c.get(k, v) for k, v in SHIPPED.items() if k != "waves_per_eu"}
        f.update(
            a_is_transposed=False,
            b_is_transposed=True,
            in_dtype_id=GEMM_A16W16_DTYPE_BF16,
            out_dtype_id=GEMM_A16W16_DTYPE_BF16,
            has_bias=False,
        )
        return make_gemm_a16w16_param_and_validate(M, N, K, f)  # None when the shape/config is unsupported

    def configure(cfg):  # launcher-keyword part of the config -> the full launcher call
        return (out_arg, a_arg, b_arg, a_arg, sem, sig, cfg.get("split_k", 1), param_of(cfg), stream), {}

    def oracle():
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        return rel < args.rel_tol, f"rel_l2={rel:.3g}"

    backend = FlyBackend(
        gemm_a16w16_gfx950,
        (),
        oracle=oracle,
        configure=configure,
        prepare=lambda: out.fill_(float("nan")),  # a skipped store must fail the oracle
    )
    space = SearchSpace(
        [
            Param.ordinal("block_m", (64, 128, 256)),
            Param.ordinal("block_n", (64, 128, 256)),
            Param.ordinal("block_k", (32, 64, 128)),
            Param.ordinal("stages", (2, 3, 4)),
            Param.ordinal("m_waves", (1, 2, 4)),
            Param.ordinal("n_waves", (1, 2, 4)),
            Param.ordinal("k_waves", (1, 2)),
            Param.ordinal("group_m", (0, 4, 8)),
            Param.boolean("use_half_tile_interleaved"),
            Param.ordinal("split_k", (1, 2, 4)),
            Param.categorical("waves_per_eu", (None, 1, 2)),
        ],
        is_valid=lambda c: c.get("block_k", 64) <= K and param_of(c) is not None,
        name="a16w16",
    )
    t0 = time.time()
    res = tune(
        space,
        backend,
        strategy=args.strategy,
        budget=Budget(evaluations=args.evals),
        seed=args.seed,
        baseline=space.encode(SHIPPED),
        confirm=3,
        confirm_rounds=5,
        trace=args.trace,
        ladder=(Fidelity(reps=8, warmup=2, cold=args.cold),),
    )
    peaks = measure_peaks(dtypes=("bf16",))
    rl = gemm_roofline(M, N, K, peaks, "bf16", a_bytes=2, b_bytes=2, c_bytes=2)
    statuses: dict = {}
    for o in res.observations:
        statuses[str(o.status)] = statuses.get(str(o.status), 0) + 1
    report = {
        "shape": [M, N, K],
        "strategy": args.strategy,
        "seed": args.seed,
        "cold": args.cold,
        "wall_s": round(time.time() - t0, 1),
        "compile_s": round(res.meter.compile_s, 1),
        "evaluations": res.evaluations,
        "statuses": statuses,
        "best_config": res.best_config,
        "confirmed_speedup": res.confirmed.get("speedup"),
        "p_value": res.confirmed.get("p_value"),
        "roofline": rl.report(res.best_value * 1e3),
    }
    json.dump(report, sys.stdout, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
