# AMDGPU `-mllvm` flags and `llvm_options`

Load this when sweeping or auditing LLVM command-line options. For function
attributes (`amdgpu-waves-per-eu` and friends) see `amdgpu-attributes.md` —
those are **not** flags.

## Check before you use

The pin registers ~130 `-amdgpu*` options, and co-located LLVM builds on this box
**disagree on which exist**. A flag appearing in a shipped kernel is not evidence.

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/llvm_knob_check.py <flag>...
```

## How FlyDSL passes them

```python
launch_fn.compile_hints["llvm_options"] = {"enable-post-misched": False}
```

Names go in **without** leading dashes. Two different mechanisms apply them,
selected by `FLYDSL_COMPILE_LLVM_DIR`:

| Mode | Mechanism |
|---|---|
| embedded (default) | in-process `cl::opt` mutation via `addOccurrence()` |
| external (`FLYDSL_COMPILE_LLVM_DIR` set) | the same in-process pass, **then** `--name=value` argv on a subprocess `mlir-opt` |

⚠ **External mode does not bypass the in-process check.** `jit_function.py`
enters `_llvm_options(llvm_opts)` before launching the external tool in *both*
modes, so an option the embedded registry rejects fails before the subprocess
ever starts. There is no "skip validation" path.

⚠ **Value type selects the setter**: `bool` → bool, `int` → int, `str` → str.
A `cl::boolOrDefault` option (like `lsr-drop-solution`) accepts only 0/1 — passing
`4` is invalid in either mode.

⚠ **Embedded mode mutates global process state.** The context manager restores
prior values in a `finally`, but a concurrent compile on another thread sees the
mutated option.

⚠ **`llvm_options` merges shallowly.** An inner dict wholly replaces the outer
one — to add a flag, merge by hand (see `SKILL.md` §5).

## Scheduler selection

Two paths, not equivalent.

**Modern:** `-amdgpu-sched-strategy=<name>` with
`max-ilp | max-memory-clause | iterative-ilp | iterative-minreg | iterative-maxocc | coexec`.

⚠ **A bad value is silently ignored** — it is a `cl::opt<std::string>` with no
validation, and the dispatch falls through to the default max-occupancy
scheduler. [measured] `-amdgpu-sched-strategy=bogus` exits 0 with no diagnostic,
so a typo in a sweep script produces baseline numbers that look like a real data
point. **Assert the strategy took effect.**

⚠ `coexec` is gfx1250-intended but degrades to a **warning, not an error**, on
other targets — and still runs, measurably changing codegen. Never put it in a
CDNA sweep.

⚠ The **function attribute form wins over the flag** (`AMDGPUTargetMachine.cpp:602-611`).

**Legacy:** `-misched=gcn-max-occupancy | gcn-max-ilp | …` (MachineSchedRegistry
names). Still present; prefer the modern selector.

## Flags worth sweeping, cheapest first

| Flag | Default | Effect |
|---|---|---|
| `-amdgpu-sched-strategy=<name>` | max-occupancy | Whole scheduling policy. Start here. |
| `-amdgpu-schedule-metric-bias=<N>` | — | Bias the scheduler's occupancy-vs-ILP tradeoff. Try 10/25/50/100. |
| `-amdgpu-vgpr-threshold-percent=<N>` | — | Try 0/70/80/90 when spilling. |
| `-amdgpu-unroll-threshold-local` / `-private` | — | Unroll aggressiveness for LDS/private-heavy loops. |
| `-enable-post-misched` | **true** | Post-RA scheduling. Turned **off** by several FMHA kernels here to protect a hand-tuned schedule. |
| `-amdgpu-max-memory-clause=<N>` | 15 | Pre-RA clause length — holds registers live. ⚠ Not the same as `-amdgpu-hard-clause-length-limit` (the gfx10+ `s_clause` encoding limit). |
| `-amdgpu-spill-vgpr-to-agpr` | true | AGPR-bearing targets only; spill into idle AGPRs instead of scratch. |
| `-amdgpu-mfma-padding-ratio=<pct>` | 0 | Pads inter-MFMA latency with `s_nop`s — a probe for MFMA-latency-bound kernels. |
| `-lsr-drop-solution` | — | ⚠ `cl::boolOrDefault`: **0/1 only**. |
| `-disable-machine-sink` | false | Used by the gfx950 dualwave FMHA kernels. |

## Bisect instruments

Use these to answer "which stage changed my schedule?", then tune.

- `-amdgpu-disable-unclustered-high-rp-reschedule`
- `-amdgpu-disable-clustered-low-occupancy-reschedule`
- `-amdgpu-disable-rewrite-mfma-form-sched-stage` (⚠ already **true** by default)
- `-opt-bisect-limit=<N>` — generic LLVM pass bisection.

## Analysis flags

| Flag | Notes |
|---|---|
| `-pass-remarks-analysis=kernel-resource-usage` | **The best single instrument.** VGPRs, AGPRs, SGPRs, ScratchSize, Occupancy, spills, LDS — no assembly parsing. Add `-pass-remarks-output=<f.yaml>` for structured output. |
| `-print-after-all` / `-print-before=<pass>` | Large; redirect to a file. |
| `-amdgpu-next-use-analysis-dump-distance-as-json=<f>` | Machine-readable live-range data. Its sibling `-amdgpu-next-use-analysis-config` defaults to `graphics`; `compute` is arguably the right preset for GEMM/attention workloads. |
| `-time-passes` | Works on Release builds. |
| ⚠ `-stats` | Needs a stats-enabled build. On Release it **parses cleanly and prints nothing** — grepping it for "spill" yields a false "no spills". |
| ⚠ `-debug-only=<x>` | Needs an assertions build; **hard-errors** otherwise. |

## ⚠ Flags that do NOT exist (but appear in this repo)

| Name | Reality |
|---|---|
| `amdgpu-schedule-regions` | Not in the pin. `llc` rejects it. Was set by `flash_attn_generic.py`; removed. |
| `--amdgpu-waves-per-eu` | A **function attribute**, not a flag. |
| `--amdgpu-num-vgpr` | A **function attribute**, not a flag. `llc` even suggests `--amdgpu-stress-vgpr` instead. |
| `-amdgpu-limit-vgpr-pressure` | Does not exist. |
| `-amdgpu-coerce-illegal-types` | Does not exist. |
| bare `-amdgpu-igrouplp` | Does not exist. Use `sched_group_barrier` / `iglp_opt` intrinsics. |

## Non-obvious spellings

A naive `grep amdgpu` misses these:

`-enable-amdgpu-aa` (prefix reversed) · `-amdgcn-skip-cache-invalidations`
(`amdgcn-`, not `amdgpu-`) · `-disable-promote-alloca-to-vector` / `-to-lds`
(no prefix) · `-sgpr-regalloc` / `-vgpr-regalloc` / `-wwm-regalloc` (no prefix) ·
`-amdgpu-function-calls` (declared in `R600TargetMachine.cpp`, not
`AMDGPUTargetMachine.cpp`, though it governs GCN).

## Defaults that surprise

- `-amdgpu-atomic-optimizer-strategy` defaults to **Iterative**, not DPP.
- `-amdgpu-disable-rewrite-mfma-form-sched-stage` defaults **true** (stage off).
- `-amdgpu-codegenprepare-widen-constant-loads` defaults **false**, while
  `-amdgpu-late-codegenprepare-widen-constant-loads` defaults **true** — to
  actually disable widening you must target the **late** one.

## Unsafe by design

Instruments, not shipping configuration: `-amdgcn-skip-cache-invalidations`
(incorrect code), `-amdgpu-waitcnt-forcezero` (pathologically slow),
`-amdgpu-xnack` / `-amdgpu-sramecc` (change the target-ID and ABI; can yield an
unloadable binary).

## Software pipelining is not available

`MachinePipeliner` has **zero** hits in `llvm/lib/Target/AMDGPU`. The generic
`-enable-pipeliner` (default true) and every `-pipeliner-*` flag parse but do
nothing on an AMDGPU compile. Achieve pipelining through unrolling plus
`llvm.amdgcn.iglp.opt` / `sched_group_barrier` instead — see
`sched-primitives.md`.
