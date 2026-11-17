#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Reference harness: search the dense flash-attention prefill kernel
(``kernels/attention/flash_attn_generic.py``) for one MHA/GQA problem.

    python3 .claude/skills/search-optimize-flydsl/scripts/tune_flash_attn_prefill.py \
        --seq 2048 --heads 64 --head-dim 256 --strategy hill --evals 120

What this harness shows that the GEMM and RMSNorm ones do not:

- The kernel's knobs are arguments of a *builder* (``build_flash_attn_func_module``
  returns a different ``@flyc.jit`` closure per config), so the backend is made
  with ``launcher_for=`` instead of one fixed launcher.
- Several levers are ``FLYDSL_FLASH_ATTN_FUNC_*`` environment variables the
  kernel reads while it builds or traces; they are ``env:<NAME>`` config keys.
- The baseline is not hard-coded: the harness calls the real entry point
  ``flydsl_flash_attn_func`` once and records which builder it reached and with
  which arguments (``route`` in the report).

- Before a search, ``--screen check`` (the default) tries every knob level alone
  against the baseline and drops the ones that do not compile or fail the oracle
  (``screened_out`` in the report), so a broken code path does not eat evaluations.
  ``--probe`` prints that one-factor table (status, resources, paired time) and stops.
- ``--levers shipped`` searches only the knobs the kernel had before this branch;
  ``--levers all`` adds the default-off builder arguments in ``LEVER_DEFAULTS``
  (``causal_reverse_q``, ``pipe_k``, ``v_pad``, ...), levers this harness's diagnosis led to.
- ``--run-config baseline|file.json`` compiles, checks and times one config, for ISA
  dumps (``FLYDSL_DUMP_IR=1``) and profiler runs.

The kernel is compute bound, so timing is hot (the default); ``--cold`` flushes
the cache before each launch for comparison.
"""

import argparse
import dataclasses
import inspect
import json
import math
import statistics
import sys
import time

import torch

import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch
from flydsl.tuning import Budget, Fidelity, Param, SearchSpace, tune
from flydsl.tuning.flybackend import FlyBackend, env_overrides, split_env
from flydsl.tuning.roofline import Roofline, measure_peaks
from flydsl.tuning.stats import paired_t_greater
from kernels.attention import flash_attn_interface as fai
from kernels.attention.flash_attn_generic import build_flash_attn_func_module
from kernels.attention.flash_attn_utils import _make_flash_attn_generic_traits

LDS_LIMIT_BYTES = 160 * 1024  # gfx950 LDS per workgroup
_ENV = "env:FLYDSL_FLASH_ATTN_FUNC_"
# Defaults of every lever, as the kernel reads them when the variable is unset.
ENV_DEFAULTS = {
    _ENV + "ENABLE_DMA": False,
    _ENV + "ENABLE_LDS_VEC16": True,
    _ENV + "REDUCE_MODE": "xor",
    _ENV + "K_PAD": 0,
    _ENV + "NATIVE_EXP2": True,
    _ENV + "TREE_REDUCE": False,
    _ENV + "PERM_PACK": True,
}
# Levers added on this branch: default-off builder arguments (part of the kernel's JIT cache key), not
# environment variables. ``--levers shipped`` leaves them out of the search. Keys are the builder's keyword names.
LEVER_DEFAULTS = {
    "causal_reverse_q": False,  # grid order: longest causal q tiles first
    "pipe_k": False,  # carry the next K tile in registers across the loop
    "v_pad": None,  # V LDS row padding in elements (None: the kernel's own 4); 32 makes the transposed reads conflict-free
    "qk_depth": None,  # K packs (ks steps) read ahead of the QK MFMAs (None: the kernel's own 2)
    "pv_depth": 1,  # V pack steps read ahead of the PV MFMAs
    "sched_lds": 0,  # pin the K/V LDS read pipeline: 1 = sched_group_barrier pattern, 2 = hard sched_barrier fences
    "kv_buffer_loads": False,  # stage K/V with bounds-checked buffer loads instead of clamped global loads
    "k_pad": None,  # K LDS row padding in elements (None: the FLYDSL_FLASH_ATTN_FUNC_K_PAD default)
    "k_swizzle": True,  # K LDS XOR swizzle; off + k_pad=8 is conflict-free with affine (immediate-offset) reads
    "addr_fold": False,  # one per-lane LDS base pointer + constant offsets for the K (unswizzled) and V reads
}
NEW_LEVERS = tuple(LEVER_DEFAULTS)
LLVM_DEFAULTS = {
    "llvm:amdgpu-sched-strategy": None,
    "llvm:amdgpu-schedule-metric-bias": None,
    "llvm:amdgpu-mfma-vgpr-form": None,
}


def route_of_shipped_entry(q, k, v):
    """Call the shipped entry point once and record which dense builder it
    reached and with which arguments. Returns ``(builder name, kwargs)``."""
    seen = []
    saved = {n: getattr(fai, n) for n in ("_build_dense", "_build_dense_dualwave")}

    def spy(name):
        def inner(**kw):
            seen.append((name, dict(kw)))
            return saved[name](**kw)

        return inner

    try:
        for n in saved:
            setattr(fai, n, spy(n))
        fai.flydsl_flash_attn_func(q, k, v, causal=True)
    finally:
        for n, f in saved.items():
            setattr(fai, n, f)
    if len(seen) != 1:
        raise RuntimeError(f"expected one dense builder call, saw {[s[0] for s in seen]}")
    return seen[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--strategy", default="auto")
    ap.add_argument("--evals", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold", action="store_true", help="flush the cache before each timed launch")
    ap.add_argument("--rel-tol", type=float, default=1e-2, help="relative L2 error vs the fp32 reference")
    ap.add_argument("--atol", type=float, default=1e-2, help="elementwise |err| <= atol + rtol * |ref|")
    ap.add_argument("--rtol", type=float, default=3e-2, help="elementwise |err| <= atol + rtol * |ref|")
    ap.add_argument("--probe", action="store_true", help="one-factor-at-a-time table around the baseline; no search")
    ap.add_argument(
        "--run-config",
        default=None,
        help="'baseline' or a JSON file of knob overrides: compile, check and time that one config "
        "(for ISA dumps and profiler runs); no search",
    )
    ap.add_argument("--run-reps", type=int, default=9, help="timed samples for --run-config")
    ap.add_argument(
        "--compare",
        default=None,
        help="JSON file {name: knob overrides, ...} (ordered): time them interleaved with the baseline for "
        "--run-reps rounds and print an ablation table with paired speedups; no search",
    )
    ap.add_argument(
        "--screen",
        choices=("check", "compile", "none"),
        default="check",
        help="before the search, drop knob levels that fail alone: 'compile' = do not compile, 'check' = also "
        "fail the numeric oracle (one launch each)",
    )
    ap.add_argument(
        "--levers",
        choices=("all", "shipped"),
        default="all",
        help="'shipped': search only the knobs the kernel had before this branch (none of the builder-argument levers in LEVER_DEFAULTS)",
    )
    ap.add_argument("--confirm-rounds", type=int, default=7)
    ap.add_argument("--no-references", action="store_true", help="skip the torch / aiter reference timings")
    ap.add_argument("--trace", default=None, help="write the search trace (JSON lines) here")
    args = ap.parse_args()
    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim

    dev = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)

    # Independent oracle: fp32 causal attention in torch, one head group at a time.
    ref = torch.empty(B, S, H, D, device=dev, dtype=torch.float32)
    causal_mask = torch.ones(S, S, device=dev, dtype=torch.bool).triu(1)
    for h0 in range(0, H, 8):
        qh = q[:, :, h0 : h0 + 8].float().transpose(1, 2)
        kh = k[:, :, h0 : h0 + 8].float().transpose(1, 2)
        vh = v[:, :, h0 : h0 + 8].float().transpose(1, 2)
        p = (qh @ kh.transpose(-1, -2) * scale).masked_fill(causal_mask, float("-inf")).softmax(-1)
        ref[:, :, h0 : h0 + 8] = (p @ vh).transpose(1, 2)
    ref_norm = ref.norm()

    def oracle():
        diff = (out.float() - ref).abs()
        rel = (diff.square().sum().sqrt() / ref_norm).item()
        within = (diff <= args.atol + args.rtol * ref.abs()).all().item()  # NaN (a skipped store) is False
        return bool(rel < args.rel_tol and within), f"rel_l2={rel:.3g} max_abs={diff.max().item():.3g}"

    # ---------------------------------------------------------------- baseline
    route, route_kw = route_of_shipped_entry(q, k, v)
    arch = get_rocm_arch()
    shipped_traits = _make_flash_attn_generic_traits(
        H,
        H,
        D,
        arch,
        causal=True,
        dtype_str="bf16",
        flat_work_group_size=route_kw["flat_work_group_size"],
        block_m=route_kw["block_m"],
        path_tag=route_kw["path_tag"],
        waves_per_eu=route_kw["waves_per_eu"],
        daz=route_kw["daz"],
    )
    shipped = {
        "block_m": route_kw["block_m"],
        "path_tag": shipped_traits.PATH_TAG,  # "auto" resolved: N32 unless causal D=128
        "waves_per_eu": route_kw["waves_per_eu"],
        "daz": route_kw["daz"],
        **ENV_DEFAULTS,
        **LEVER_DEFAULTS,
        **LLVM_DEFAULTS,
    }
    shipped_fwg = route_kw["flat_work_group_size"]
    assert shipped_fwg == 2 * shipped["block_m"], "the builder ties flat_work_group_size to 32 rows per wave"

    def lever_kwargs(cfg):
        return {k: cfg[k] for k in LEVER_DEFAULTS}

    def build(cfg):
        """The shipped route's builder call, with this config's knobs."""
        return build_flash_attn_func_module(
            num_heads=H,
            head_dim=D,
            causal=True,
            dtype_str="bf16",
            num_kv_heads=H,
            cross_seqlen=False,
            block_m=cfg["block_m"],
            flat_work_group_size=2 * cfg["block_m"],
            path_tag=cfg["path_tag"],
            waves_per_eu=route_kw["waves_per_eu"],
            daz=cfg["daz"],
            **lever_kwargs(cfg),
        )

    def launcher_for(cfg):
        # The builder returns a python wrapper; the @flyc.jit launcher and the
        # compile hints the wrapper applies are in its closure.
        nl = inspect.getclosurevars(build(cfg)).nonlocals
        return nl["launch_flash_attn_generic"]

    base_hints = inspect.getclosurevars(build(shipped)).nonlocals["_fmha_compile_hints"]
    stream = fx.Stream(torch.cuda.current_stream())
    call = (q, k, v, out, out, out, out, out, 0, B, S, S, stream)  # the dense call of the shipped route

    def legality(cfg):
        """The builder's own checks (asserts and traits) plus the LDS limit; ``None`` if legal."""
        try:
            with env_overrides(split_env(cfg)):
                build(cfg)
                t = _make_flash_attn_generic_traits(
                    H,
                    H,
                    D,
                    arch,
                    causal=True,
                    dtype_str="bf16",
                    flat_work_group_size=2 * cfg["block_m"],
                    block_m=cfg["block_m"],
                    path_tag=cfg["path_tag"],
                    waves_per_eu=route_kw["waves_per_eu"],
                    daz=cfg["daz"],
                    **lever_kwargs(cfg),
                )
        except (AssertionError, ValueError, NotImplementedError) as e:
            return f"{type(e).__name__}: {e}"
        if 2 * t.LDS_KV_TOTAL_SIZE > LDS_LIMIT_BYTES:
            return f"LDS {2 * t.LDS_KV_TOTAL_SIZE} B > {LDS_LIMIT_BYTES} B"
        return None

    def canonical(cfg):
        """Key of the effective config: N128 always uses DMA, so its env DMA value is moot."""
        if legality(cfg) is not None:
            return None
        eff = {k: v for k, v in cfg.items() if v is not None}
        if eff["path_tag"] == "N128":
            eff[_ENV + "ENABLE_DMA"] = True
        return json.dumps(eff, sort_keys=True)

    backend = FlyBackend(
        None,
        call,
        oracle,
        launcher_for=launcher_for,
        compile_hints=base_hints,
        prepare=lambda: out.fill_(float("nan")),  # a skipped store must fail the oracle
    )
    params = [
        # tile geometry: 32 rows per wave, so the work-group size follows block_m
        Param.ordinal("block_m", (64, 128, 256)),
        Param.categorical("path_tag", ("N32", "N128")),  # N128: 128 KV columns per iteration, K/V through DMA
        Param.boolean(_ENV + "ENABLE_DMA"),  # N32 with K/V DMA into LDS
        Param.boolean(_ENV + "ENABLE_LDS_VEC16"),
        Param.ordinal(_ENV + "K_PAD", (0, 8, 16)),
        # softmax / epilogue code shape
        Param.categorical(_ENV + "REDUCE_MODE", ("xor", "ds_bpermute")),
        Param.boolean(_ENV + "NATIVE_EXP2"),
        Param.boolean(_ENV + "TREE_REDUCE"),
        Param.boolean(_ENV + "PERM_PACK"),
        Param.boolean("daz"),
        Param.boolean("causal_reverse_q"),  # grid order: causal q tiles longest-first
        Param.boolean("pipe_k"),  # next K tile's global load overlaps this tile's compute
        Param.categorical("v_pad", (None, 16, 32)),  # V LDS row padding: conflict-free transposed reads at 32
        Param.categorical("qk_depth", (None, 3, 4, 6)),
        Param.ordinal("pv_depth", (1, 2, 3, 4)),
        Param.categorical("sched_lds", (0, 1, 2)),
        Param.boolean("kv_buffer_loads"),
        Param.categorical("k_pad", (None, 8)),
        Param.boolean("k_swizzle"),
        Param.boolean("addr_fold"),
        # occupancy and scheduling
        Param.ordinal("waves_per_eu", (1, 2, 4)),
        Param.boolean("llvm:enable-post-misched"),
        Param.boolean("llvm:lsr-drop-solution"),
        Param.categorical("llvm:amdgpu-sched-strategy", (None, "max-ilp", "max-memory-clause")),
        Param.categorical("llvm:amdgpu-schedule-metric-bias", (None, 10, 50, 100)),
        Param.categorical("llvm:amdgpu-mfma-vgpr-form", (None, False)),
    ]
    space = SearchSpace(params, canonical=canonical, name="flash_attn_prefill")
    baseline = dict(shipped)
    baseline.update({"llvm:enable-post-misched": base_hints["llvm_options"]["enable-post-misched"]})
    baseline.update({"llvm:lsr-drop-solution": base_hints["llvm_options"]["lsr-drop-solution"]})
    if legality(baseline) is not None:
        raise RuntimeError(f"baseline is not legal: {legality(baseline)}")
    base_genome = space.encode(baseline)

    # The baseline has to pass the oracle before anything else is believed.
    comp = backend.compile(baseline, "baseline")
    if not comp.ok:
        raise RuntimeError(f"baseline does not compile: {comp.detail}")
    ok, why = backend.check(baseline, "baseline", comp.artifact)
    if not ok:
        raise RuntimeError(f"baseline fails the oracle ({why}); fix the tolerance or the harness first")
    baseline_info = {"oracle": why, "resources": comp.resources, "compile_s": round(comp.seconds, 2)}

    if args.run_config:
        cfg = dict(baseline)
        if args.run_config != "baseline":
            with open(args.run_config) as f:
                cfg.update(json.load(f))
        key = space.key(space.encode(cfg))
        if key is None:
            raise RuntimeError(f"config is not legal: {legality(cfg)}")
        c = backend.compile(cfg, key)
        if not c.ok:
            raise RuntimeError(f"does not compile: {c.detail}")
        ok, why = backend.check(cfg, key, c.artifact)
        ms = statistics.median(
            backend.measure_batch([(cfg, key, c.artifact)], Fidelity(reps=args.run_reps, warmup=2))[0]
        )
        json.dump(
            {"config": cfg, "resources": c.resources, "oracle_ok": ok, "oracle": why, "ms": ms}, sys.stdout, indent=1
        )
        print()
        return 0

    if args.compare:
        with open(args.compare) as f:
            named = json.load(f)
        json.dump(compare(backend, space, baseline, comp, named, args, out), sys.stdout, indent=1)
        print()
        return 0

    if args.probe:
        json.dump(probe(backend, space, baseline, base_genome, comp, args), sys.stdout, indent=1, default=str)
        print()
        return 0

    # Compile-only screen: a level that does not even compile on its own (an LLVM option this toolchain lacks,
    # a broken code path) is dropped before the search, and reported, instead of eating evaluations.
    params, fixed, pruned = screen(backend, legality, params, baseline, args.screen)
    if args.levers == "shipped":  # the kernel's own knobs only: leave this branch's new levers at their defaults
        fixed.update({k: baseline[k] for k in NEW_LEVERS})
        params = [p for p in params if p.name not in NEW_LEVERS]
    space = SearchSpace(params, canonical=canonical, fixed=fixed, name="flash_attn_prefill")
    base_genome = space.encode(baseline)

    t0 = time.time()
    res = tune(
        space,
        backend,
        strategy=args.strategy,
        budget=Budget(evaluations=args.evals),
        seed=args.seed,
        baseline=base_genome,
        confirm=3,
        confirm_rounds=args.confirm_rounds,
        trace=args.trace,
        ladder=(Fidelity(reps=8, warmup=2, cold=args.cold),),
    )
    wall = time.time() - t0

    # ---------------------------------------------------------------- report
    statuses: dict = {}
    problems: dict = {}
    seen_by_status: dict = {}
    for o in res.observations:
        statuses[str(o.status)] = statuses.get(str(o.status), 0) + 1
        if o.status != "ok" and not o.cached and o.key is not None:
            problems.setdefault(str(o.status), {}).setdefault(o.detail[:160], 0)
            problems[str(o.status)][o.detail[:160]] += 1
            # Which knob values the failing configs share (count of failures per value, and of all
            # evaluated configs with that value, so a value that always fails stands out).
            cfg = space.decode(o.trial.genome)
            by = seen_by_status.setdefault(str(o.status), {"n": 0, "values": {}})
            by["n"] += 1
            for kname, kval in cfg.items():
                by["values"][f"{kname}={kval!r}"] = by["values"].get(f"{kname}={kval!r}", 0) + 1
    evaluated: dict = {}
    for o in res.observations:
        if o.key is not None and not o.cached:
            for kname, kval in space.decode(o.trial.genome).items():
                evaluated[f"{kname}={kval!r}"] = evaluated.get(f"{kname}={kval!r}", 0) + 1
    attribution = {
        st: {
            "failures": d["n"],
            "always_fail_values": {
                v: f"{c}/{evaluated[v]}" for v, c in sorted(d["values"].items()) if c == evaluated.get(v) and c >= 3
            },
        }
        for st, d in seen_by_status.items()
    }
    compiled = [o.compile_s for o in res.observations if o.compile_s > 0 and not o.cached]
    peaks = measure_peaks(dtypes=("bf16",))
    flops = 4.0 * B * H * S * S * D / 2  # causal: half of the score matrix
    nbytes = 4.0 * B * S * H * D * 2  # Q, K, V and O once, bf16
    rl = Roofline(flops, nbytes, peaks, "bf16")
    best_ms = res.best_value
    win = res.confirmed.get("winner")
    report = {
        "shape": {"batch": B, "seq": S, "heads": H, "head_dim": D, "causal": True, "dtype": "bf16", "arch": arch},
        "route": {"builder": route, "kwargs": route_kw},
        "oracle": {
            "reference": "fp32 torch causal attention",
            "rel_l2_lt": args.rel_tol,
            "elementwise": f"|err| <= {args.atol} + {args.rtol} * |ref|",
        },
        "baseline": {"config": baseline, **baseline_info},
        "strategy": args.strategy,
        "levers": args.levers,
        "seed": args.seed,
        "timing": "cold" if args.cold else "hot",
        "wall_s": round(wall, 1),
        "compile_s_total": round(res.meter.compile_s, 1),
        "compile_s_mean": round(statistics.mean(compiled), 2) if compiled else None,
        "compile_s_max": round(max(compiled), 2) if compiled else None,
        "evaluations": res.evaluations,
        "space_size": space.size(),
        "screened_out": pruned,
        "statuses": statuses,
        "problems": problems,
        "failure_attribution": attribution,
        "best_config": res.best_config,
        "best_resources": next((dict(o.resources) for o in res.observations if o.key == win and o.resources), None),
        "confirmed_speedup": res.confirmed.get("speedup"),
        "p_value": res.confirmed.get("p_value"),
        "confirmed_values_ms": res.confirmed.get("values"),
        "roofline": {
            "bound": rl.bound,
            "floor_us": rl.floor_us,
            "peaks": {"hbm_TBps": peaks.hbm_tbs, "bf16_TFLOPS": peaks.flops_peak("bf16")},
            "winner": rl.report(best_ms * 1e3),
            "baseline": (
                rl.report(res.confirmed["values"][res.confirmed["baseline"]] * 1e3)
                if res.confirmed.get("baseline") in (res.confirmed.get("values") or {})
                else None
            ),
        },
        "top": [
            {"config": space.decode(o.trial.genome), "ms": o.value, "resources": dict(o.resources)}
            for o in sorted((o for o in res.observations if o.ok and o.full), key=lambda o: o.value)[:5]
        ],
    }
    if not args.no_references:
        report["references_ms"] = reference_timings(q, k, v, ref, args)
    json.dump(report, sys.stdout, indent=1, default=str)
    print()
    return 0


def compare(backend, space, baseline, base_comp, named, args, out_tensor):
    """Ablation table: the baseline and each named config timed interleaved for ``--run-reps`` rounds; paired
    speedup (geometric mean of per-round ratios) and one-sided p-value against the baseline and the previous row."""
    rows = [("baseline", baseline, "baseline", base_comp.artifact)]
    out = {"rows": []}
    backend.check(baseline, "baseline", base_comp.artifact)  # leaves the baseline's output in out_tensor
    base_out = out_tensor.clone()
    for name, over in named.items():
        cfg = {**baseline, **over}
        key = space.key(space.encode(cfg))
        if key is None:
            out["rows"].append({"name": name, "status": "invalid"})
            continue
        c = backend.compile(cfg, key)
        if not c.ok:
            out["rows"].append({"name": name, "status": "compile_error", "detail": c.detail})
            continue
        ok, why = backend.check(cfg, key, c.artifact)
        if not ok:
            out["rows"].append({"name": name, "status": "numeric_fail", "detail": why})
            continue
        rows.append((name, cfg, key, c.artifact))
        out.setdefault("resources", {})[name] = c.resources
        out.setdefault("bitwise_equal_to_baseline", {})[name] = bool(torch.equal(out_tensor, base_out))
    samples = backend.measure_batch([(c, k, a) for _n, c, k, a in rows], Fidelity(reps=args.run_reps, warmup=3))
    ms = {n: s for (n, *_r), s in zip(rows, samples)}
    names = list(ms)

    def paired(slow, fast):
        logs = [math.log(a) - math.log(b) for a, b in zip(ms[slow], ms[fast])]
        return {"speedup": math.exp(statistics.mean(logs)), "p": paired_t_greater(logs, 0.0)}

    for i, n in enumerate(names):
        row = {"name": n, "median_ms": statistics.median(ms[n]), "rounds": len(ms[n])}
        if i:
            row["vs_baseline"] = paired("baseline", n)
            row["vs_previous"] = paired(names[i - 1], n)
        if i > 1:  # the first named config is the reference (the previous round's winner, say)
            row["vs_first"] = paired(names[1], n)
        out["rows"].append(row)
    out["baseline_resources"] = base_comp.resources
    out["samples_ms"] = ms
    return out


def screen(backend, legality, params, baseline, mode):
    """Drop every knob level whose single flip from the baseline does not compile (``mode='compile'``) or also
    fails the numeric oracle (``'check'``). Returns ``(params, fixed, pruned)``: the reduced params, the baseline
    values of params left with one level, and a ``{knob, status, detail}`` list of what was dropped."""
    if mode == "none":
        return params, {}, []
    kept, fixed, pruned = [], {}, []
    for p in params:
        levels = []
        for lvl in p.levels:
            cfg = {**baseline, **dict(lvl)}
            if cfg == baseline or legality(cfg) is not None:  # the space refuses illegal levels itself
                levels.append(lvl)
                continue
            label = f"{p.name}={dict(lvl)[p.name]!r}"
            c = backend.compile(cfg, "screen")
            if not c.ok:
                pruned.append({"knob": label, "status": "compile_error", "detail": c.detail})
                continue
            if mode == "check":
                ok, why = backend.check(cfg, "screen", c.artifact)
                if not ok:
                    pruned.append({"knob": label, "status": "numeric_fail", "detail": why})
                    continue
            levels.append(lvl)
        if len(levels) < 2:
            fixed.update(dict(levels[0]))
            continue
        kept.append(dataclasses.replace(p, levels=tuple(levels)))
    return kept, fixed, pruned


def probe(backend, space, baseline, base_genome, base_comp, args):
    """Every single-knob neighbor of the baseline: status, resources, numeric
    verdict and (when it passes) a paired time against the baseline. No search."""
    rows, timed = [], [("baseline", baseline, "baseline", base_comp.artifact)]
    for p in space.params:
        for lvl in p.levels:
            cfg = {**baseline, **dict(lvl)}
            if cfg == baseline:
                continue
            label = f"{p.name}={dict(lvl)[p.name]!r}"
            key = space.key(space.encode(cfg))
            if key is None:
                rows.append({"knob": label, "status": "invalid", "detail": "refused by the legality predicate"})
                continue
            comp = backend.compile(cfg, key)
            if not comp.ok:
                rows.append({"knob": label, "status": "compile_error", "detail": comp.detail})
                continue
            ok, why = backend.check(cfg, key, comp.artifact)
            row = {"knob": label, "status": "ok" if ok else "numeric_fail", "detail": why, "resources": comp.resources}
            row["compile_s"] = round(comp.seconds, 2)
            rows.append(row)
            if ok:
                timed.append((label, cfg, key, comp.artifact))
    fid = Fidelity(reps=9, warmup=2, cold=args.cold)
    samples = backend.measure_batch([(c, k, a) for _l, c, k, a in timed], fid)
    med = {label: statistics.median(s) for (label, *_r), s in zip(timed, samples)}
    for r in rows:
        if r["knob"] in med:
            r["ms"] = med[r["knob"]]
            r["speedup_vs_baseline"] = med["baseline"] / med[r["knob"]]
    return {"baseline_ms": med["baseline"], "baseline_resources": base_comp.resources, "rows": rows}


def time_fn(fn, inner=10, reps=9, warmup=3):
    """Median ms per call of ``fn`` over ``reps`` event-timed batches of ``inner`` calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(20_000_000)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e) / inner)
    return statistics.median(samples)


def reference_timings(q, k, v, ref, args):
    """Hot median times of torch SDPA and (when importable) aiter on the same problem."""
    out = {}
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D] views
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def run_sdpa():
        return sdpa(qh, kh, vh, is_causal=True)

    try:
        o = run_sdpa().transpose(1, 2)
        rel = ((o.float() - ref).norm() / ref.norm()).item()
        out["torch_sdpa"] = {"ms": time_fn(run_sdpa), "rel_l2_vs_fp32": rel}
    except Exception as e:  # noqa: BLE001 - a reference that cannot run is reported, not fatal
        out["torch_sdpa"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        import aiter

        def run_aiter():
            return aiter.flash_attn_func(q, k, v, causal=True)

        o = run_aiter()
        o = o[0] if isinstance(o, tuple) else o
        rel = ((o.float() - ref).norm() / ref.norm()).item()
        out["aiter_flash_attn_func"] = {"ms": time_fn(run_aiter), "rel_l2_vs_fp32": rel}
    except Exception as e:  # noqa: BLE001
        out["aiter_flash_attn_func"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
