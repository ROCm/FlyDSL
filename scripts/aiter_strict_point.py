#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run ONE aiter MoE point through the strict, AOT-checked, model-correct path.

This replaces the aiter *legacy CLI* path (which sets ``strict_accuracy=False``,
``check_aot_cache=False``, hardcodes ``ActivationType.Swiglu`` for the fp8/fp4
case, and times with warmup=2/iters=5) with a direct call to aiter's
``test_fmoe`` using:

* the model's TRUE activation and gate mode (passed by the caller),
* ``strict_accuracy=True`` and ``check_aot_cache=True`` (the AOT-cache-wrapped
  variant ``test_fmoe_with_aot_cache_check`` — so an AOT-cache miss raises),
* the locked e2e measurement protocol (warmup/iters injected by monkeypatching
  the module's ``run_perftest`` reference).

It prints one machine-readable ``STRICT_RESULT {json}`` line with e2e us,
logits_diff, correctness pass/fail, and the strict/AOT/protocol flags actually
used, which ``moe_tuning_harness.parse_strict_aiter_output`` consumes.

Usage:
  python3 scripts/aiter_strict_point.py \
    --model-dim 7168 --inter-dim 256 -e 257 -k 9 -t 16 \
    --aq fp4 --wq fp4 --act silu --gate separated \
    [--warmup 10 --iters 100] [--no-aot] [--aiter-repo /sgl-workspace/aiter]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys


def _load_aiter_module(aiter_repo: str):
    """Import test_moe_2stage.py without running its default CLI sweep.

    The module has no ``__main__`` guard, so executing it runs the bottom sweep;
    we set argv to ``--no-legacy --no-flydsl-csv`` first to make that sweep empty.
    """
    sys.argv = ["test_moe_2stage.py", "--no-legacy", "--no-flydsl-csv"]
    path = f"{aiter_repo}/op_tests/test_moe_2stage.py"
    spec = importlib.util.spec_from_file_location("aiter_test_moe_2stage", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DTYPES = {}


def _resolve_dtypes():
    from aiter import dtypes

    return {
        "fp4": dtypes.fp4x2,
        "fp8": dtypes.fp8,
        "bf16": dtypes.bf16,
        "fp16": dtypes.fp16,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="strict single-case aiter MoE guardrail")
    ap.add_argument("--model-dim", type=int, required=True)
    ap.add_argument("--inter-dim", type=int, required=True)
    ap.add_argument("-e", "--experts", type=int, required=True)
    ap.add_argument("-k", "--topk", type=int, required=True)
    ap.add_argument("-t", "--token", type=int, required=True)
    ap.add_argument("--aq", required=True, help="activation quant dtype: fp4|fp8|bf16")
    ap.add_argument("--wq", default="fp4", help="weight quant dtype (fp4)")
    ap.add_argument("--act", required=True, help="silu|swiglu")
    ap.add_argument("--gate", default="separated", help="separated|interleave")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-aot", action="store_true", help="disable AOT-cache check (records it)")
    ap.add_argument("--aiter-repo", default="/sgl-workspace/aiter")
    args = ap.parse_args(argv)

    mod = _load_aiter_module(args.aiter_repo)
    import aiter

    dts = _resolve_dtypes()
    aq, wq = dts[args.aq], dts[args.wq]
    act = getattr(aiter.ActivationType, args.act.capitalize())
    check_aot = not args.no_aot

    # Inject the locked e2e protocol by wrapping the module's run_perftest so the
    # internal warmup=2/iters=5 are overridden with the locked values.
    _orig_run_perftest = mod.run_perftest

    # True timed-loop e2e distribution: after a warmup, time the fused_moe call per
    # iteration (median + p95 over `iters`) IN ADDITION TO aiter's own rotated
    # average.  We keep aiter's rotated average as the median e2e_us (it defeats L2
    # via arg rotation, matching the L2-flush intent and staying comparable across
    # runs) and use the per-iteration loop only for the e2e p95 dispersion.
    e2e_dist = {"median": None, "p95": None}
    # run_perftest's own control kwargs are NOT forwarded to the timed callable.
    _PERF_CTRL_KW = ("num_iters", "num_warmup", "testGraph", "num_rotate_args", "needTrace")

    def _locked_run_perftest(func, *a, **kw):
        # aiter's rotated average (locked warmup/iters) -> the comparable median.
        kw_avg = dict(kw)
        kw_avg["num_iters"] = args.iters
        kw_avg["num_warmup"] = args.warmup
        data, avg = _orig_run_perftest(func, *a, **kw_avg)
        e2e_dist["median"] = avg
        # Per-iteration p95 dispersion (best-effort; does not change the median).
        try:
            import torch

            call_kw = {k: v for k, v in kw.items() if k not in _PERF_CTRL_KW}
            lat = []
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            for _ in range(max(1, args.iters)):
                ev0.record()
                func(*a, **call_kw)
                ev1.record()
                ev1.synchronize()
                lat.append(ev0.elapsed_time(ev1) * 1000.0)  # ms -> us
            ordered = sorted(lat)
            idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
            e2e_dist["p95"] = ordered[idx]
        except Exception:
            e2e_dist["p95"] = None
        return data, avg

    mod.run_perftest = _locked_run_perftest

    test_fn = mod.test_fmoe_with_aot_cache_check if check_aot else mod.test_fmoe

    result = {
        "strict_accuracy": True,
        "check_aot_cache": check_aot,
        "warmup": args.warmup,
        "iters": args.iters,
        "act": args.act,
        "gate": args.gate,
        "aq": args.aq,
        "wq": args.wq,
    }
    try:
        ret = test_fn(
            aiter.dtypes.bf16,
            args.token,
            args.model_dim,
            args.inter_dim,
            args.experts,
            args.topk,
            act,
            args.gate,
            aiter.QuantType.per_1x32,
            aq,
            wq,
            use_g1u1=True,
            doweight_stage1=False,
            strict_accuracy=True,
            check_aot_cache=check_aot,
        )
        if ret is None:
            result.update({"error": "skipped_or_none", "error_category": "skipped", "correctness_pass": False})
        else:
            ld = float(ret["logits_diff"])
            result.update(
                {
                    "e2e_us": e2e_dist["median"] if e2e_dist["median"] is not None else float(ret["us"]),
                    "e2e_us_p95": e2e_dist["p95"],
                    "logits_diff": ld,
                    "correctness_pass": ld <= 0.01,
                    "error_category": "" if ld <= 0.01 else "correctness",
                }
            )
    except Exception as e:  # AOT miss, strict assertion, or runtime error.
        name = type(e).__name__
        msg = str(e)
        if "AOT cache miss" in msg:
            cat = "aot_miss"
        elif name == "AssertionError" or "accuracy check failed" in msg:
            cat = "correctness"
        elif "out of memory" in msg.lower() or "OOM" in msg:
            cat = "oom"
        else:
            cat = "runtime"
        result.update({"error": f"{name}: {msg[:200]}", "error_category": cat, "correctness_pass": False})
    finally:
        mod.run_perftest = _orig_run_perftest

    print("STRICT_RESULT " + json.dumps(result), flush=True)
    return 0 if result.get("correctness_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
