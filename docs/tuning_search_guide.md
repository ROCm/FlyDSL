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
   object, so it costs no GPU time. The one exception is the confirmation's
   `baseline`: the shipped default is the reference whether or not it spills
   (the D=256 attention default does).
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
  `None` leaves the option at LLVM's default.
- `env:<NAME>`: an environment variable set while this config is built, traced and
  compiled, and restored afterwards (`None` unsets it), for kernels that read a
  `FLYDSL_*` variable at build or trace time. Booleans become `"1"`/`"0"`.
- anything else: a `Constexpr` keyword argument of the launcher (with
  `launcher_for`, an argument of the builder).

### Kernels whose knobs are builder arguments

Many kernels are built by a `build_*_module(...)` function that returns a different
`@flyc.jit` closure per configuration instead of one launcher with `Constexpr`
arguments. Pass `launcher_for(config_kwargs) -> launcher` in place of `launcher`;
the backend builds one launcher per config. `compile_hints` are hints applied to
every compile (a builder's wrapper often sets `fast_fp_math` or `llvm_options` that
the bare closure does not); the config's `waves_per_eu` and `llvm:` keys override
them, key by key for `llvm_options`.

```python
backend = FlyBackend(None, call_args, oracle,
                     launcher_for=lambda cfg: closure_of(build(cfg)),   # a @flyc.jit function
                     compile_hints=hints_of(build(shipped)), prepare=poison_output)
```

`.claude/skills/search-optimize-flydsl/scripts/tune_flash_attn_prefill.py` is the
worked example. It takes the closure and the hints out of the builder's wrapper with
`inspect.getclosurevars`, and finds the baseline by calling the shipped entry point
once and recording which builder it reached.

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
winner that sets an `llvm:` or `env:` key is refused: apply the option in the launcher.

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

Four problems, run with `FlyBackend` on FlyDSL 0.5.0 (disk cache off, so
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
| bf16 GEMM 32x8192x8192 | cold | 2.34-2.47x over the default across four strategies and three seeds | about 0.44 of the bandwidth floor: the knobs this kernel exposes do not reach the roofline |
| flash attention prefill, bf16 causal, S=2048, H=64, D=256 | hot | existing knobs only: 1.51-1.56x over the default in 8 of 9 runs (`hill`, `portfolio`, `gp`, three seeds; p < 1e-13 each), 1.34x in the ninth; the default spills 88 VGPRs and `waves_per_eu=1` (or 4) removes the spill | 0.21 of the compute floor for the default, 0.32 for the winner |
| flash attention prefill, same | hot | plus two new levers (below): 2.11-2.14x in 8 of 9 runs, 1.89x in the ninth | 0.45 |
| flash attention prefill, same | cold | 2.12x (one `hill` run) | 0.42 |

Cost of a search (cold GEMM, 120 evaluations, 12 runs): 35-90 s wall for most
runs, one at 270 s. Compiling a config takes 0.16-0.7 s on average and up to
about 2 s for the larger tiles. Every strategy reached within 5% of its own best
in 23 evaluations or fewer; the final winners differ by less than the noise, so
this space does not rank the strategies. `random` spent 19-20% of its
evaluations on configs that failed the numeric gate, against 1-9% for `hill` and
`gp`.

Two findings about the kernel, not the engine:

- `make_gemm_a16w16_param_and_validate` admits `block_k=32`, and every one of
  the seven numeric failures printed across these runs had `block_k=32` with a
  relative L2 error of 1.23: wrong results, not noise. The gate kept them out
  of the timings.
- The cold small-M result leaves more than half of the bandwidth floor
  unreached. Closing it needs a new lever (the skill's step 5), not more
  search over the existing space.

### Flash attention prefill (bf16, causal, S=2048, H=64, D=256)

`flydsl_flash_attn_func` sends this shape to the generic kernel
(`flash_attn_generic.py`, tile 128 rows, 256 threads, `N32` path, `waves_per_eu=2`),
not to the dual-wave gfx950 kernel, which only takes D=64/128. The harness is
`.claude/skills/search-optimize-flydsl/scripts/tune_flash_attn_prefill.py`; it reads
the baseline off the real entry point. The gate is an fp32 torch reference at
rel. L2 < 1e-2 and `|err| <= 1e-2 + 3e-2 |ref|` elementwise (the baseline reaches 3e-3);
the output is NaN-poisoned before each check. The problem is compute-bound
(512 flop/byte against a crossover of about 235), so timing is hot. Roofline: `4*B*H*S*S*D/2`
flops, Q, K, V and O once, against `measure_peaks()` (about 1270 TFLOPS bf16 and 5.4 TB/s).
torch SDPA takes 1.10x the default's time on the same inputs (0.19 of the floor); aiter is not
installed on the node.

- **Search cost** (150 evaluations including the confirmation): 62-71 s per run, 0.29-0.34 s
  mean compile per config (0.43-0.57 s at most). The winners of a lever set agree within
  about 3% except one `gp` seed each, so this space does not rank the strategies; `gp` seed 1 on the existing knobs stopped at 1.34x (it kept
  `ENABLE_LDS_VEC16=0`, which loses without the K prefetch below).
- **Statuses.** With the compile-only/numeric screen on (default), 18 runs produced
  1733 `ok` and 463 `spill`: every one of the 256 evaluations at tile 256 spills, and 246 of the
  318 at `waves_per_eu=2` (both cap the wave at 256 registers; the winner uses 352-390). The
  screen found three kernel limits from single flips of the default, which the numeric
  gate and the compile step reported with their reasons: `path_tag=N128` and
  `ENABLE_DMA=1` return wrong results (rel. L2 0.67-1.0) at D=256, and
  `REDUCE_MODE=ds_bpermute` fails to compile (`TypeError: dtype must be a Numeric subclass`).
  Without the screen, 55 of `hill`'s 122 search evaluations were not `ok` (31 `numeric_fail`,
  9 `compile_error`, 15 `spill`) and it still reached 2.20x; 114 of `random`'s 122 were not
  (60 `compile_error`, 37 `numeric_fail`, 17 `spill`) and it reached 1.68x.
- **What mattered** (paired, 25 interleaved rounds, all outputs bit-identical to the default):

  | Config on top of the default | Speedup |
  |---|---|
  | `waves_per_eu=1` | 1.58x |
  | `CAUSAL_REVERSE_Q` alone | 1.19x |
  | both | 1.84x |
  | `waves_per_eu=1` + `PIPE_K` | 1.70x |
  | all three | 1.92x |
  | all three + `ENABLE_LDS_VEC16=0` | 2.00x (it loses, 0.89x, without `PIPE_K`) |
  | the same + `llvm:amdgpu-sched-strategy=max-memory-clause` | 2.17x (2.22x in a rerun on the final files) |
  | the nine search winners | 1.93x (one `gp`), 2.15-2.19x (eight) |

  `daz`, `TREE_REDUCE`, `lsr-drop-solution`, `NATIVE_EXP2`, `PERM_PACK`, `K_PAD`,
  `schedule-metric-bias`, `mfma-vgpr-form` and `enable-post-misched` are within noise or
  a fraction of a percent once the rows above are set, although individual winners carry them.
- **The two new levers** came from reading the loop and the counters, then went into the space
  as default-off knobs (`env:` keys): `CAUSAL_REVERSE_Q` dispatches the longest causal Q tiles
  first (the grid runs in block order and a tile costs `q_tile + 1` KV iterations); `PIPE_K` carries
  the next K tile in registers so its global load overlaps compute. They are experimental, dense
  only, and measured on this shape only.
- **Still 0.45 of the floor.** Counters (`rocprofv3 --pmc`) put MFMA busy at about 28% of SIMD
  cycles for the winner (11% for the default, 19% with the existing knobs only; `SQ_VALU_MFMA_BUSY_CYCLES`
  over 1024 SIMDs times `GRBM_GUI_ACTIVE` per XCD); the wave is waiting
  on something 40% of its cycles, and 54% of LDS-active cycles are bank conflicts (67% for the
  default). One wave per SIMD at 386 registers cannot overlap the softmax VALU
  (about 6 VALU instructions per MFMA) with the MFMAs. Closing that needs a new structure
  (a dual-wave/ping-pong schedule for D=256, or a register diet that allows two waves
  per SIMD), not more search over these knobs.

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
