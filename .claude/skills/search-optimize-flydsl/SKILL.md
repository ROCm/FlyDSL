---
name: search-optimize-flydsl
description: >
  Optimize a FlyDSL kernel by searching its config space: compile every
  candidate without a GPU, reject spills and LDS overflow from the code object,
  time only the survivors with a numeric gate and paired confirmation, and stop
  at the roofline. Use when single-lever edits stall, when the space is too large
  to enumerate (tile geometry, waves, stages, waves_per_eu, LLVM options), or
  when asked to tune a kernel, find the best config, or report how close a
  kernel is to roofline.
  Usage: /search-optimize-flydsl
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Search-driven optimization (FlyDSL)

FlyDSL compiles a config in a fraction of a second, so the cheap move is to
search instead of reasoning about one lever at a time. This skill is the loop;
the API and the measurement rules are in
[`docs/tuning_search_guide.md`](../../../docs/tuning_search_guide.md). Read that
first for `FlyBackend`, `SearchSpace`, `tune` and `roofline`.

## Principles

1. **Know the ceiling before searching.** Compute the roofline (flops, minimum
   bytes, measured peaks). It decides what to tune and when to stop.
2. **Levers are knobs.** Every tile size, wave count, stage count, `waves_per_eu`
   and `llvm:<option>` goes in the space, including ones that lose alone. A lever
   that loses alone can win in combination or on a new base.
3. **Compile screen first.** `FlyBackend.compile` reads VGPR, SGPR, spill,
   scratch and LDS from the code object. Spills and overflow cost no GPU time.
4. **A number has to earn its place.** Numeric check before any timing,
   interleaved timing, paired confirmation against the baseline. A kernel that
   launches is not a kernel that is correct.
5. **The LLM proposes levers; the engine searches them.** Use the ISA and the
   trace to invent knobs; use the engine to find where they pay.
6. **A searched winner is a measured result** for one device, toolchain and
   problem. See [Shipping](#shipping).

## Procedure

### 0. Orient
- Read the kernel's builder and launcher (`kernels/<family>/`) and its existing
  test. Note every `Constexpr` that changes the binary, and which values the
  builder rejects (a validity predicate often already exists).
- Compile the shipped default and read its resources (`FlyBackend.compile`). A default
  that already spills or sits at a register cap points at occupancy
  (`waves_per_eu`, wave count) before anything else; the flash-attention D=256 default
  spills 88 VGPRs and `waves_per_eu=1` alone is worth 1.58x.
- Compute the roofline for the problem:

```python
from flydsl.tuning.roofline import Roofline, gemm_roofline, measure_peaks
peaks = measure_peaks()                      # achievable HBM TB/s and matmul TFLOPS here
rl = gemm_roofline(M, N, K, peaks, "bf16", a_bytes=2, b_bytes=2, c_bytes=2)
print(rl.bound, rl.floor_us)
```

  `memory`: tune loads, vector width, occupancy, waves. `compute`: tune tile
  geometry, stages, LDS layout, scheduling. Datasheet peaks
  (`nominal_peaks(name)`) are an upper bound, not a target.
- Time memory-bound kernels with `Fidelity(cold=True)` (flushes the cache).
  A hot loop over a working set that fits the last-level cache reads above
  the HBM roofline.

### 1. Knob inventory
Keep one table per kernel and shape class: knob, values, default, legality, status
(untested, lost, won, combination) and the base it was measured on. A "lost" row
carries the base it lost on. Typical FlyDSL knobs:

| Kind | Examples | Config key |
|---|---|---|
| Tile geometry | `block_m/n/k`, `tile_m/n/k` | launcher `Constexpr` |
| Parallelism | `m_waves`, `n_waves`, `k_waves`, `BLOCK_THREADS`, `num_warps`, `split_k` | launcher `Constexpr` / `num_warps` |
| Pipelining | `stages`, `lds_stage`, `use_async_copy`, half-tile interleave | launcher `Constexpr` |
| Scheduling | `group_m`, `waves_per_eu` | launcher / `waves_per_eu` |
| Compiler | `amdgpu-kernarg-preload-count`, scheduling flags (`/llvm`) | `llvm:<option>` |
| Environment | a `FLYDSL_*` variable the kernel reads while it builds or traces | `env:<NAME>` |
| Builder arguments | `build_*_module(...)` tile, waves, path (the builder returns a new `@flyc.jit` closure per config) | plain key, with `launcher_for=` |

### 2. Harness
Build a `FlyBackend` around the real launcher. The oracle is independent (fp32
reference at a declared dtype and tolerance); `prepare` poisons the outputs so a
skipped store cannot pass.

```python
from flydsl.tuning import Budget, Fidelity, Param, SearchSpace, tune
from flydsl.tuning.flybackend import FlyBackend

backend = FlyBackend(launcher, args, kwargs=fixed_kwargs, oracle=oracle,
                     prepare=lambda: out.fill_(float("nan")))
# A launcher that takes a structured argument (a param dataclass) passes
# configure=lambda cfg: (args_for(cfg), kwargs_for(cfg)) instead of kwargs.
space = SearchSpace([...], is_valid=builder_accepts)   # builder's own predicate
result = tune(space, backend, strategy="auto", budget=Budget(evaluations=150),
              baseline=space.encode(shipped_default), confirm=3, confirm_rounds=5,
              ladder=(Fidelity(reps=8, warmup=2),))
```

- Build the baseline from what the kernel ships today for this arch and shape.
  A baseline that fails to build inflates every win. Do not read the router and
  guess: call the real entry point once with a spy on the builder and record the
  arguments it used (`route` in `tune_flash_attn_prefill.py`).
- A kernel built by `build_*_module(...)` has no single launcher: pass
  `launcher_for=cfg -> @flyc.jit closure` (and `compile_hints=` for the hints its
  wrapper adds) instead of `launcher`. See the guide's "Kernels whose knobs are
  builder arguments".
- Before searching, flip each knob level alone against the baseline (`--probe` /
  `--screen` in the attention harness): compile, numeric check, resources, paired time.
  It costs a minute, names the levels that are broken (not slow) and lets the search
  skip them. Do not infer which knob broke a config from a 12-evaluation smoke run:
  the first guess here (`NATIVE_EXP2`, `TREE_REDUCE`) was wrong; the culprits were
  `ENABLE_DMA` and `REDUCE_MODE`.
- Read `Observation.detail` for every non-ok status: an opaque `compile_error`
  hides an unsupported arch or a validity gap.
- Set `FLYDSL_RUNTIME_ENABLE_CACHE=0` when you want honest compile times; leave
  the disk cache on for repeat searches.

### 3. Choose the strategy by space size
- Space no larger than the budget: `exhaustive` (`strategy="auto"` picks it).
- A few hundred configs: `gp` or the default `portfolio`.
- Sparse or rugged tile grids: `hill`.
- Larger spaces and budgets: `de`.
- Always pass `baseline=` so the confirmation reports a paired speedup.

### 4. Lookback
The hill climber keeps a stack: single moves from the base, then every subset of
the near-winners, then two-step moves on ordinal knobs. Accept the first
improvement beyond noise and push the old base. When a base is exhausted, revisit
the stack, then a foreign winner, then a random unseen config. After a
structural lever lands, re-sweep the levers that lost.

### 5. Propose new levers
Look at the ISA of the current best with `/isa-resource-diff` (spills, VGPR,
LDS, occupancy) and `/kernel-trace-analysis` (stalls, `s_waitcnt`). Pick the
bottleneck the roofline says matters. Turn each hypothesis into a default-off
launcher `Constexpr` (validated in the builder), add it to the space, and go
back to step 2. `/llvm` lists the compile hints and `-mllvm` flags worth trying
as `llvm:` keys. A lever applied outside the launcher signature is a prototype:
the cache key and the artifact never see it.

### 6. Per shape class
Search per class (decode vs prefill, small-M vs large-M, aligned vs ragged), not
per shape. Keep the best per class and run the full cohort before claiming
anything; a smoke set under-samples variance.

### 7. Stop
Stop when the confirmed winner reaches the measured roofline within noise, or
when a round of new levers and a re-sweep finds nothing beyond noise. Report the
fraction of roofline, not just the speedup over the baseline.

### 8. Record
- Report relative, paired speedups with the numeric-gate statement (tolerance
  and reference). Keep absolute latencies in the internal record.
- Keep losses: a lost lever with its base is information.
- Write `result.summary()` and the trace (`trace=` path) next to the kernel's
  notes so the search can be resumed or replayed.

## Shipping

`artifacts.export_winner(tuner, config, *args, **kwargs)` writes the winner as
the offline artifact an `@autotune(..., artifact_name=...)` adopter already
serves from `FLYDSL_AUTOTUNE_CONFIG_DIR`. It carries launcher kwargs,
`num_warps` and `waves_per_eu`; a winner that sets an `llvm:` key is refused
(apply the option in the launcher). Prefer a winner that becomes the default
over one that needs its own artifact per shape. Retune after a compiler, kernel
or compile-hint change.

## Reference harnesses
Copy one and replace the inputs, the config-to-launch mapping, the oracle and the
roofline bytes:

- `.claude/skills/search-optimize-flydsl/scripts/tune_gemm_a16w16.py`: structured `param` launcher through `configure=`,
  the builder's own legality check as `is_valid`, split-K, cold fidelity.
- `.claude/skills/search-optimize-flydsl/scripts/tune_rmsnorm.py`: plain `Constexpr` launcher, and the export to an
  `@autotune` offline artifact (`--export`).
- `.claude/skills/search-optimize-flydsl/scripts/tune_flash_attn_prefill.py`: builder-style kernel (`launcher_for=`),
  `env:` and `llvm:` knobs, baseline read off the shipped entry point, the single-flip screen (`--screen`, `--probe`),
  paired ablation tables (`--compare`) and one-config runs for ISA dumps and profiler passes (`--run-config`).

The evidence behind this skill is in
[`docs/tuning_search_guide.md`](../../../docs/tuning_search_guide.md#pilot-results-gfx950-mi355x):
a search with 120 evaluations takes about a minute, and on the small-M GEMM it
found 2.4x (cold) over the shipped default but stayed at about 0.44 of the
bandwidth floor. Searching the existing knobs gets you to the best config of
the existing kernel; the roofline gap that remains is a new-lever problem
(step 5).

## Failure modes
Seen in the pilots:
- The shipped default spilled (flash attention D=256, 88 VGPRs). The spill gate keeps
  spilled configs out of the search, but the confirmation's baseline is exempt;
  before that fix a spilling default was dropped and no speedup was reported.
- Time through the compiled callable `FlyBackend` builds, never through the Python
  entry point: a direct `@flyc.jit` call re-traces every time (65 ms per call with the
  cache off), which hides the kernel.
- A lever that loses alone can win in combination, and the reverse: `ENABLE_LDS_VEC16=0`
  was 0.89x on `waves_per_eu=1` + reverse-order grid and +4% once the K prefetch was on.
  Read ablations from the paired table (`--compare`), not from one run's winner config:
  winners carry several knobs that are within noise.
- The levers that mattered most after the register fix (causal tile order +17%, K
  prefetch +4%) were not in the kernel's knob list. They came from the loop, the
  ISA mix and the counters, and were then put into the space as default-off knobs.
- A tolerance copied from another dtype rejected every config, including the
  baseline. Check that the baseline passes the oracle first.
- A hot timing of a cache-resident working set read above the HBM roofline.
  Use cold fidelity for memory-bound kernels.
- The builder's validity check admitted `block_k=32` for the GEMM and the kernel
  returned wrong results (rel L2 1.23). Only the numeric gate caught it.
- `random` spent about a fifth of its evaluations on numeric failures.
- Flash attention D=256, second round (2.2x to 2.8x over the default): read the final ISA of the winner before
  proposing levers. Three things were invisible in the knob list: the scheduler had sunk every LDS read next to its
  MFMA (so a deeper Python prefetch changed nothing until `sched_barrier(0)` fences pinned it), 32 loop-invariant
  LDS address registers capped the pipeline at the 256 arch VGPRs (one base pointer plus constant offsets fixed it),
  and a burst of 16 `buffer_load` plus 16 `ds_write` stalls a one-wave-per-SIMD kernel's issue (spreading them one per
  MFMA pair was +13%; making them L2-resident was +3%).
- Model the hardware before probing it: a 20-line bank-conflict simulator for the 64-bank gfx950 LDS predicted the
  4x conflict on the V reads and reproduced the 54% conflict counter; the counter then confirmed 0 after the fix.
- Ablate by removing work, not inputs: deleting the GEMM inputs lets dead-code elimination delete the kernel. Compare
  ablations that keep the dataflow (no exp, no rescale, no barriers, no staging) in one paired run.
- A "broken" knob can be a bug: `ENABLE_DMA=1` returned wrong results at D=256 because the DMA fetch hard-coded the XOR
  swizzle mask of D=128. Read why a level fails before dropping it from the space.
- Passing a lever is not the same as the compiler honoring it: compare resources (`vgpr`, `lds_bytes`) between the
  rows of an ablation table; identical resources for different knob values mean the knob did nothing
  (`waves_per_eu=2` on a 141 KB-LDS kernel, `sched_lds` without fences).
- A structural change can need its second half to pay: `kv_dbuf` alone was 0.97x, `spread_stage` on top 1.13x.
  Stop a structural branch when two successive attempts lose to the current winner, and record both.

From the rocKE case study this engine came from:
- A fixed arm order with best-of-N selection produces a phantom win; the backend
  shuffles every round, and the confirmation re-times finalists.
- A one-knob "rejected" verdict on an old base is not a verdict on the new base.
- A config can pass the builder's validity check and still fail the numeric gate;
  the gate is what keeps it out of the timings.
- Scratch cache winners do not fingerprint the kernel source. After a kernel edit,
  bump the adopter's schema key or clear `~/.flydsl/autotune`.
