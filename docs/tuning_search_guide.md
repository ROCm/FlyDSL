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
| bf16 GEMM 32x8192x8192 | cold | 2.34-2.47x over the default with the knob values the harness first listed; 2.86-2.88x after widening the values (`block_m/n=32`, `block_k=256`, more stages and waves); 3.0x with the `k_rotate` lever below | 0.53-0.55 of the steady-state bandwidth floor at the best; torch matmul, same timing, takes 13% longer than the winner |
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
way takes 13% longer, at 0.53-0.55 of the steady-state bandwidth floor. That
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
- The cold small-M result stalled at 0.44-0.52 of the bandwidth floor until a
  new lever (the skill's step 5) was added; more search over the existing
  values did not move it.

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
- **Still 0.45 of the floor** after the search above (about 580 TFLOPS). Counters put MFMA busy at about 28%
  of SIMD cycles (`SQ_VALU_MFMA_BUSY_CYCLES` over 1024 SIMDs times `GRBM_GUI_ACTIVE` per XCD), 40% of wave
  cycles waiting, and 54% of LDS-active cycles bank conflicts. The second round below went after those.

#### Second round: structure (all levers are default-off builder arguments, part of the JIT cache key)

The environment levers `CAUSAL_REVERSE_Q` and `PIPE_K` became builder arguments (`causal_reverse_q`, `pipe_k`).
Every row below is paired and interleaved on one node, every output is bit-identical to the shipped default, and the
numbers are times converted to TFLOPS with 137.4 GFLOP per call against the measured 1270 TFLOPS matmul peak
(nodes differ by 3%: m10-09 gave 184.9 us for the final row, m07-13 gave 190.5 us).

| Config (cumulative) | Time | TFLOPS | of peak | Paired speedup |
|---|---|---|---|---|
| shipped default | 534-538 us | 256 | 0.20 | 1.00x |
| previous winner (`waves_per_eu=1`, reverse Q, `PIPE_K`, vec8 LDS, K_PAD 8, tree reduce, `max-memory-clause`) | 237 us | 580 | 0.46 | 2.24x |
| + `v_pad=32` | 227.6 us | 604 | 0.48 | 1.04x over the row above |
| + `kv_buffer_loads` | 214.5 us | 641 | 0.50 | 1.11x over the previous winner |
| + `sched_lds=2`, `qk_depth=4`, `pv_depth=3`, `k_swizzle=False`, `addr_fold` | 210-215 us | 640-655 | 0.50-0.51 | 2.50x |
| + `kv_dbuf`, `spread_stage` | 184.9-190.5 us | 721-743 | 0.57-0.585 | 2.83x (p = 6e-30) |

What each lever did, with the evidence:

- **LDS bank conflicts** (`v_pad`, `k_pad`/`k_swizzle`/`k_swz_shift`, `v_swz_shift`). gfx950 has 64 banks and
  services `ds_read_b128` in four 16-lane phases and `ds_read_b64_tr_b16` in two 32-lane phases. A 20-line phase
  simulator (lane address in, worst bank multiplicity per phase out) predicted 4.0x for the V transposed reads at the
  shipped 260-element row stride and 1.0x at stride 288; the K reads are 2.0x with the shipped XOR and 1.0x with
  `K_PAD=8`. `SQ_LDS_BANK_CONFLICT` went from 31.2M to 0 and `SQ_LDS_IDX_ACTIVE` from 57.9M to 26.7M. The model also
  reproduced the measured 54% conflict share for the previous winner (384 of 704 LDS cycles per wave). Worth 4% alone.
- **K/V staging loads** (`kv_buffer_loads`): the clamped 64-bit global loads cost about 9 VALU instructions each, 150
  per tile (a third of all VALU). Bounds-checked buffer loads read rows past the end as zero. 2.4% alone, 10.6% with
  `v_pad`.
- **LDS read pipeline** (`qk_depth`, `pv_depth`, `sched_lds`, `addr_fold`). The ISA showed every K and V read issued
  right before its MFMA and waited with `s_waitcnt lgkmcnt`, whatever depth the Python asked for: the scheduler sinks
  loads, and 32 loop-invariant address registers (one per K step, one per V step) used up the 256 arch VGPRs. One
  per-lane LDS pointer plus constant offsets frees them (VGPR 376 to 332); `sched_barrier(0)` fences between steps
  (`sched_lds=2`) then make the depth real (8 K reads in flight, `lgkmcnt(6)`). `sched_group_barrier` (mode 1) did the
  same within 1%. Together +12% on top of `kv_buffer_loads`; none of it works without the fences.
- **Spreading the staging** (`spread_stage`, needs `kv_dbuf`): ablating the K/V global loads and LDS stores took 26%
  off the loop, but making the K/V loads L2-resident took only 3%, so the cost is instruction issue, not memory. A
  wave cannot issue an MFMA while it sits in a burst of 16 `buffer_load` and 16 `ds_write`. `kv_dbuf` double-buffers K
  and V in LDS (141 KB) and stages the next tile in registers, one barrier per tile; alone it is 0.97x. Emitting one
  load or store per MFMA pair (`GenericKvStageHooks`, pinned by the fences) makes it 1.13x.
- **Ablations on the 214 us config** (diagnostic build, outputs invalid): no exp 6%, no O rescale 5%, no barriers
  12%, no staging loads/stores 24-26%, K/V L2-resident 3%. Ablations that remove the GEMM inputs collapse the kernel
  by dead-code elimination and say nothing.
- **A bug fix**: the DMA K fetch hard-coded the XOR mask 0x7 while the K reader uses `head_dim/16 - 1` (15 at D=256).
  `ENABLE_DMA=1` (and `path_tag=N128`, which implies DMA) returned wrong results at D=256 for that reason; with the
  mask taken from the traits the DMA path is bit-identical to the default. It is slower than the register-staged path
  (234-242 us against 215 us), so it is not used; the `N128` path was not re-measured.

Attempts that did not pay, and the base they lost on:

| Attempt | Result | Base |
|---|---|---|
| `xcd_remap` (heads dealt to XCDs in contiguous groups) | 0.80x | 210 us row |
| DMA K/V into LDS, conflict-free swizzles (`k_swz_shift=3`, `v_swz_shift=5`) | 0.89-0.92x | 215 us row |
| `kv_dbuf` without `spread_stage` | 0.97x | 215 us row |
| `swp` (S(j+1) overlapped with P(j)), fences, no interleave | 0.93x | 210 us row |
| `swp` + `spread_stage` | 1.075x, below `kv_dbuf`+`spread_stage` | 210 us row |
| `swp` + `swp_valu` (exp, rescale, row max emitted between the MFMA steps) | 1.05x, below `kv_dbuf`+`spread_stage` | 215 us row |
| `waves_per_eu=2` on the final config | same binary (337 VGPR), no change | final row |
| deeper read pipelines (`qk_depth` 6, `pv_depth` 4) | within noise of 4/3 | final row |

- **Where it stops.** Two successive structural attempts after the last win (`swp`, `swp_valu`) each failed to beat the
  final configuration, at 0.57-0.585 of the measured matmul peak (0.75 was the target). The final loop is MFMA busy about
  31% of cycles. What is left: the softmax vector work (exp, row max, sum, pack, about 20% of the loop on the ablations)
  is still serialized between the two GEMMs, and the `swp` structure that would overlap it ran slower than it saves
  with this scheduler. A true two-waves-per-SIMD kernel needs the O (128) and Q (64) registers split across waves, with
  P and the row statistics exchanged through LDS: not attempted, only reasoned (192 of 256 registers go to O and Q).
- **Not verified:** the levers are measured on bf16 D=256 only; `kv_dbuf`/`swp` require the dense N32 path with
  `kv_buffer_loads`; `swp_valu` is laid out for D=256.

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
