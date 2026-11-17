# Search-driven kernel tuning

`flydsl.tuning` searches a kernel's config space instead of sweeping it. It is
for the case the offline autotuner ([`autotune_guide.md`](autotune_guide.md))
does not cover: a space too large to enumerate, levers that only pay in
combination, and a need to know when to stop.

FlyDSL compiles a config in well under a second (measured numbers below), so
the loop is: compile every candidate without a GPU, throw away the ones that
spill or exceed LDS, then spend GPU time only on survivors, and stop when the
kernel reaches its roofline.

The engine is ported from the rocKE tuning search engine
(`rocke.dispatch.tuning.search` in ROCm/rocm-libraries) and detached from the
rocKE dispatcher. FlyDSL owns this copy.

## What each piece does

| Module | Role |
|---|---|
| `params` | `Param` (ordinal, categorical, bool, gated) and `SearchSpace`: validity, repair, canonical keys |
| `strategy`, `de`, `halving`, `surrogate`, `local`, `portfolio` | random, exhaustive, differential evolution, successive halving / Hyperband, TPE, GP with expected improvement, hill climbing with lookback, a bandit portfolio |
| `measure` | The `Backend` protocol and the `Evaluator` that enforces the measurement rules |
| `flybackend` | `FlyBackend`: the `Backend` for a `@flyc.jit` launcher |
| `resources` | VGPR, SGPR, spill, scratch and LDS counts read from the compiled code object |
| `roofline` | Datasheet and measured peaks, arithmetic intensity, `% of roofline` |
| `stats`, `driver`, `db`, `replay` | Paired statistics and noise model, the ask/tell loop with checkpoint and resume, tuned-record DB and traces, offline strategy replay |
| `artifacts` | Write a winner as an `@autotune` offline artifact |

## Measurement rules

The `Evaluator` enforces these for every backend:

1. **Numeric check before the first timing.** A config that compiles and
   launches but computes the wrong answer is `numeric_fail`, never "fast". The
   oracle is yours: an independent reference at a declared tolerance.
2. **Spills are rejected before timing.** A config whose binary has scratch or
   VGPR/SGPR spills is `spill`. `FlyBackend.compile` reads this from the code
   object, so it costs no GPU time.
3. **Interleaved timing.** `FlyBackend.measure_batch` times one sample per
   config per round in shuffled order, between GPU-side backlogs (the
   `do_bench` approach), so host launch gaps are not timed and slow drift hits
   every config alike.
4. **Paired confirmation.** The finalists and the baseline are re-measured
   together for several rounds; the result carries a paired speedup and a
   p-value. Only a significant speedup is worth shipping.

## Config keys

`FlyBackend` takes a config dict. Keys mean:

- `waves_per_eu`, `num_warps`: the `flydsl.autotune.Config` fields of the same name.
- `llvm:<option>`: an LLVM option for this compile, e.g.
  `"llvm:amdgpu-kernarg-preload-count": 8` (`compile_hints["llvm_options"]`).
- anything else: a `Constexpr` keyword argument of the launcher.

## Minimal use

```python
import torch
from flydsl.tuning import Budget, Fidelity, Param, SearchSpace, tune
from flydsl.tuning.flybackend import FlyBackend
from flydsl.tuning.roofline import Roofline, measure_peaks

backend = FlyBackend(
    rmsnorm_direct, (x, gamma, out, M),
    kwargs=dict(N=N, dtype_str="bf16", weight_dtype_str="bf16", stream=torch.cuda.current_stream()),
    oracle=lambda: (torch.allclose(out.float(), ref, rtol=2e-2, atol=2e-2), "vs fp32 reference"),
    prepare=lambda: out.fill_(float("nan")),      # a skipped store must fail the oracle
)
space = SearchSpace([
    Param.ordinal("BLOCK_THREADS", (64, 128, 256, 512, 1024)),
    Param.categorical("waves_per_eu", (None, 1, 2, 3, 4)),
])
baseline = space.encode({"BLOCK_THREADS": 512, "waves_per_eu": None})
result = tune(space, backend, strategy="auto", budget=Budget(evaluations=100),
              baseline=baseline, confirm=3, confirm_rounds=5,
              ladder=(Fidelity(reps=10, warmup=2),))
print(result.best_config, result.confirmed["speedup"], result.confirmed["p_value"])
```

`strategy="auto"` enumerates a space no larger than the budget and otherwise
runs the default portfolio (hill climbing, GP and DE).

## Strategy by space size

- Space no larger than the evaluation budget: `exhaustive` (what `auto` picks).
- A few hundred configs: `gp`, or the default `portfolio`.
- Sparse or rugged grids (tile geometry with many invalid corners): `hill`.
- Larger spaces and larger budgets: `de`.
- `sh` / `hyperband` pay only when a rung is a much cheaper proxy problem, not
  merely fewer reps.

Per-family defaults (`FAMILY_DEFAULTS`) hold only `default`: the rocKE version
chose per-family entries by replaying recorded sweeps, and no FlyDSL sweeps are
recorded yet. `flydsl.tuning.replay` is the tool to record them.

## Roofline

A search needs a stopping rule. `roofline.Roofline(flops, bytes_moved, peaks,
dtype)` gives the arithmetic intensity, the bound (memory or compute), the
fastest time the peaks allow and the fraction a measured time reaches.

- `nominal_peaks(device_name)` returns datasheet peaks for known products
  (MI300X, MI325X, MI350X, MI355X) and `None` otherwise. These are upper
  bounds no kernel reaches.
- `measure_peaks()` measures achievable copy bandwidth and matmul throughput on
  the device in this process. Use it as the stopping ceiling.

Peaks for other architectures are never guessed: pass `Peaks(...)` built from
measured numbers.

## Serving a winner

`artifacts.export_winner(tuner, config, *args, **kwargs)` writes the winner as
the offline artifact that an `@autotune(..., artifact_name=...)` adopter already
serves from `FLYDSL_AUTOTUNE_CONFIG_DIR`. Nothing in the serving path changes.
Artifacts carry launcher kwargs, `num_warps` and `waves_per_eu` only, so a
winner that sets an `llvm:` key is refused: apply the option in the launcher.

A searched winner is a measured result for one device, toolchain and problem.
Retune after a compiler, kernel or compile-hint change, and read the
paired speedup before shipping.

## Command line

```bash
python -m flydsl.tuning strategies
python -m flydsl.tuning demo --strategy portfolio --budget 64   # synthetic backend, no GPU
python -m flydsl.tuning show --trace trace.jsonl
python -m flydsl.tuning tune --factory mypkg.tuning:make --budget 64 --db tuned.json
```

`tune --factory module:function` calls `function(args)` and expects a dict with
`space` and `backend` (optionally `prior`, `warm`, `ladder`, `baseline`,
`record`).

## Pilot results (gfx950, MI355X)

Three problems, run with `FlyBackend` on FlyDSL 0.5.0 (disk cache off, so
compile times are real). Speedups are paired and confirmed against the kernel's
shipped default; every config passed the numeric gate (fp32 reference).
Roofline fractions use `measure_peaks()` on the same device (about 5.0 TB/s
read bandwidth, 1290 TFLOPS bf16 matmul). Absolute latencies stay out of this
page.

| Problem | Timing | Result | Roofline |
|---|---|---|---|
| RMSNorm 8192x8192 bf16 | hot | No config beat the default beyond noise: the ten fastest are within 0.2% and all use the default `BLOCK_THREADS=512` | working set fits the last-level cache, so the hot time reads above the HBM figure |
| RMSNorm 8192x8192 bf16 | cold | +2.4% (p = 0.002) with `BLOCK_THREADS=1024`, `waves_per_eu=4`; the winner is exported as an `@autotune` artifact and served back | about 0.77 of the bandwidth floor |
| bf16 GEMM 4096x4096x4096 | hot | Default is already at the top; `hill` found +0.7% (p = 5e-5), `portfolio` and `random` found nothing better | at or above the measured matmul peak |
| bf16 GEMM 32x8192x8192 | hot | 5.3x over the default (`256x256` tile) with `split_k` and a smaller tile, all three strategies | memory-bound; hot time is cache-assisted |
| bf16 GEMM 32x8192x8192 | cold | 2.34-2.47x over the default with the knob values the harness first listed; 2.86-2.88x after widening the values (`block_m/n=32`, `block_k=256`, more stages and waves); 3.0x with the `k_rotate` lever below | 0.55 of the steady-state bandwidth floor at the best; torch matmul, same timing, takes 13% longer than the winner |

Cost of a search (cold GEMM, 120 evaluations, 12 runs): 35-90 s wall for most
runs, one at 270 s. Compiling a config takes 0.16-0.7 s on average and up to
about 2 s for the larger tiles. Every strategy reached within 5% of its own best
in 23 evaluations or fewer; the final winners differ by less than the noise, so
this space does not rank the strategies. `random` spent 19-20% of its
evaluations on configs that failed the numeric gate, against 1-9% for `hill` and
`gp`.

### Closing the gap on the cold small-M GEMM

The first search left the cold problem at 0.44 of the bandwidth floor. Widening
the knob values moved it to 0.52. Then the top configs were the signal: six
structurally different configs (one to four waves, three to five stages, with
and without slice-K) were all within 0.3% of each other. A plateau across
configs that differ in how much they keep in flight points at a limit outside
those knobs. The hypothesis: all workgroups stream the same K tile at the same
time, so their loads hit the same memory channels.

The lever is `k_rotate` in `kernels/gemm/gemm_a16w16_gfx950.py`: workgroup `p`
starts its K loop at tile `(p * k_rotate) % k_tiles` and wraps. It reorders only
the fp32 accumulation. Added as a launcher `Constexpr` and put in the search
space, the four wide searches (two strategies, two seeds) all landed on
`k_rotate` 3 or 7 and 48.5-49.3 us against 51.5 without it. A paired A/B of one
config at `k_rotate` 0, 1, 3 and 7, 20 interleaved rounds, cold: `k_rotate=3` is
1.065x faster than 0 (p = 2e-27); 7 is tied with 3; 1 is slower than both. Applied
to a differently shaped config it did not help (one search-phase comparison was 7% slower), so keep
it as a searched knob, not a default.

Where it stands: 3.0x over the shipped default, and the torch matmul timed the same
way takes 13% longer,, at 0.55 of the steady-state bandwidth floor. That
floor is from a 2 GB copy. A cold kernel reading 134 MB does not reach
steady-state bandwidth: on the same device a cold torch read-reduction of 134 MB
takes 66 us (2.1 TB/s) and a cold 134 MB copy moves 3.7 TB/s, against 4.8 TB/s
for 2 GB. The achievable floor at this size is therefore above the 2 GB figure
and unmeasured here. The remaining distance is not closed.

Two findings about the kernel, not the engine:

- `make_gemm_a16w16_param_and_validate` admits `block_k=32`, and every one of
  the seven numeric failures printed across these runs had `block_k=32` with a
  relative L2 error of 1.23: wrong results, not noise. The gate kept them out
  of the timings.
- The cold small-M result leaves more than half of the bandwidth floor
  unreached. Closing it needs a new lever (the skill's step 5), not more
  search over the existing space.

## Tests

`tests/unit/tuning/` runs without a GPU: strategies, statistics, search spaces,
failure handling, replay and the DB use a synthetic backend; the code-object
reader and the roofline arithmetic have their own tests. GPU behavior of
`FlyBackend` is exercised by the pilot runs recorded in the agent skill
([`search-optimize-flydsl`](../.claude/skills/search-optimize-flydsl/SKILL.md)).

## See also

- [Offline autotune configs](autotune_guide.md)
- [Kernel tuning guide](kernel_tuning_guide.md)
- `/search-optimize-flydsl`, `/isa-resource-diff`, `/kernel-trace-analysis`
