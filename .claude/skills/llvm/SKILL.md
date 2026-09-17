---
name: llvm
description: >
  Tune and analyse a FlyDSL kernel at the LLVM level: pick a compile hint, function
  attribute, or AMDGPU backend flag, then PROVE it reached codegen. Covers the
  `compile_hints` keys, the `llvm_options` cl::opt escape hatch, `rocdl.*` / raw LLVM
  function attributes, per-arch options for gfx950 and gfx1250, dumping and reading
  `NN_llvm_ir.ll`, replaying the LLVM half with `llc`, and the knobs in this repo that
  silently do nothing. Use when asked to reduce spills, raise occupancy, change the
  instruction schedule, set waves_per_eu or flat_work_group_size, pass an
  `-mllvm` or `-amdgpu-*` flag, sweep scheduler strategies, read LLVM IR or ISA the
  compiler produced, or explain why a tuning knob had no measurable effect. For
  before/after register and spill deltas use `/isa-resource-diff`; for why-is-it-slow
  with a profiler use `/kernel-trace-analysis`; for building LLVM use `/build-flydsl`.
  Usage: /llvm <question, kernel, or knob>
allowed-tools: Read Write Edit Bash Grep Glob
---

# LLVM-Level Kernel Tuning and Analysis

FlyDSL reaches LLVM through four lanes, and **only some of them are alive**. This
skill routes you to the one that works for your knob, then makes you prove it
landed. The repo is knob-rich and instrument-poor: four knobs were found doing
nothing at all, each looking like a real recompile because an unknown hint
still changes the JIT cache key.

**Hard rule: never report "the knob had no effect" without a positive control.**
Establish that your instrument reads *something* before you record a negative.
This project has been burned by false negatives repeatedly; §7 is not optional.

## Pick the right skill first

| Question | Skill |
|---|---|
| Which LLVM knob do I turn, and did it actually apply? | **this skill** (compile-only, no GPU) |
| Did registers / spills / LDS move between two builds? | `/isa-resource-diff` (owns `*_final_isa.s` parsing) |
| *Why* is this kernel slow — which instructions stall? | `/kernel-trace-analysis` (needs a GPU run + ATT trace) |
| How do I collect a trace at all? | `/capture-kernel-trace` |
| Which commit made it slow? | `/bisect-perf-regression` |
| How do I build or rebuild LLVM? | `/build-flydsl` |
| How do I add a backend op in C++/TableGen? | `/add-target-atom-op` |
| GEMM tiling / pipelining strategy | `/gemm-optimization` |

This skill is **compile-only**. It answers "what did the compiler emit, and
why" — never "where did the cycles go". Register and spill questions get
measured by `isa_resource_table.py`; do not re-parse ISA by hand.

## Step 1 — Establish your toolchain

Do this before quoting any flag. Co-located LLVM builds on this box **disagree
on which flags exist**, so "the flag is not recognised" is meaningless until you
know which binary you asked.

```bash
cat thirdparty/llvm-build-info.json        # the pin that compiles kernels
echo "${FLYDSL_COMPILE_LLVM_DIR:-<empty = embedded>}"   # which LLVM does final codegen
```

⚠ Answer flag-existence questions against the **checkout matching the pin** in
`thirdparty/llvm-build-info.json`. A newer local LLVM will register flags the pin
does not have, so "llc accepts it" proves nothing unless the versions agree.

- Some local trees are empty or stale — confirm `llvm/lib/Target/AMDGPU` is
  populated before trusting a grep over one.
- A runnable `llc` from a build of that pin covers gfx942 / gfx950 / gfx1250. If
  yours came from a different tree, cross-check flag existence against the pin.
- ⚠ `build-fly/bin/fly-opt` **cannot run on the host** (missing `GLIBC_2.33/2.34`,
  `GLIBCXX_3.4.29/3.4.30`). Run it through the dev container instead.
- A second pin exists in `thirdparty/custom-llvm-tools.json` (ROCm fork,
  `c8cf6da4`) used **only** for the external `mlir-opt` tarball.

⚠ **The host Python is 3.6; this repo targets 3.10+.** Both helper scripts below
refuse to run on the host with a clear message and exit 2 — run them in the dev
container, which has 3.10 and sees the same bind-mounted paths:

```bash
docker exec -w $PWD <dev-container> python3 <script> <args>
```

Before using any `llvm_options` flag, check it exists in *your* LLVM:

```bash
docker exec -w $PWD <dev-container> python3 \
    ${CLAUDE_SKILL_DIR}/scripts/llvm_knob_check.py enable-post-misched
```

A flag appearing in a shipped kernel is **not** evidence that it exists — see §8.

## Step 2 — Dump hygiene

```bash
export FLYDSL_DUMP_DIR=$(mktemp -d)
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python your_kernel.py
ls $FLYDSL_DUMP_DIR/*/            # expect ~22 files, 00_origin.mlir .. 21_final_isa.s
```

⚠ Traps that manufacture false negatives, all confirmed in this repo:

- **`FLYDSL_RUNTIME_ENABLE_CACHE=0` is mandatory.** A cache hit skips compilation
  and you get **no dumps at all**.
- ⚠ **`scripts/dumpir.sh` lies about it.** It prints `(cache disabled)` but never
  sets the variable, so a cache hit yields no dumps under a banner claiming
  otherwise. Set it yourself.
- **Stage numbers are positional and unstable.** `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`
  inserts a stage at 19 and shifts `llvm_ir`/`final_isa` to 21/22. **Always match
  by suffix** (`*_llvm_ir.ll`, `*_final_isa.s`), never by index.
- **Dump dirs are keyed by kernel name only.** Different JIT specialisations
  overwrite each other. Use a fresh dir per run and compare whole trees.
- **`scripts/run_tests.sh` skips the MLIR stage** when FileCheck is missing. It
  says so explicitly and `FLYDSL_REQUIRE_FILECHECK=1` makes it an error, but
  still assert you saw `PASS` lines, never merely the absence of `FAIL`.
- The ISA dump is **skipped entirely** in external-LLVM mode.

## Step 3 — The four lanes into LLVM

| Lane | Reaches LLVM as | Status |
|---|---|---|
| `compile_hints` → attribute path | `rocdl.waves_per_eu` → `"amdgpu-waves-per-eu"` | **live** |
| `compile_hints` → `opts=` CLI path | `gpu-module-to-binary{opts="..."}` | ⚠ **DEAD on AMD** |
| `llvm_options` | LLVM `cl::opt` set in-process (or argv in external mode) | **live** |
| `kernel_attrs` / `value_attrs` | any MLIR attr on `gpu.func`, incl. raw `passthrough` | **live** |
| ROCDL intrinsics | `llvm.amdgcn.sched.*`, `s_setprio`, `s_waitcnt` | **live** |

⚠ **The `opts=` lane is discarded on AMD.** `TargetOptions::tokenizeCmdOptions()`
is called only by the XeVM and NVVM targets; ROCDL never calls it, and
`ROCDL::assembleIsa()` takes no flags parameter, so anything placed there is
dropped without a diagnostic. FlyDSL used to route `--amdgpu-waves-per-eu`,
`--amdgpu-num-vgpr` and `-g` through it; `RocmBackend._pipeline_parts` now leaves
`bin_cli_opts` empty, with a comment recording why. The first two were never
command-line flags in any case: they are IR function attributes, which `llc`
rejects outright when passed as flags.

`waves_per_eu` works *only* because `RocmBackend.lower_compile_hints` also sets
the `rocdl.waves_per_eu` MLIR attribute. `maxnreg` had no such second path — see §8.

## Step 4 — The compile_hints keys

| Key | Reaches LLVM as | Validated? | Verify with |
|---|---|---|---|
| `waves_per_eu` | `rocdl.waves_per_eu` → `"amdgpu-waves-per-eu"="N"` | bool/int/range | grep `*_llvm_ir.ll` |
| ~~`maxnreg`~~ | **removed** — now raises; see §8 | raises | n/a |
| `fast_fp_math` | `rocdl-attach-target{fast=true}` **+** ambient tracing fastmath | truthiness | `fadd fast float` in IR |
| `unsafe_fp_math` | `rocdl-attach-target{unsafe-math=true}` only | truthiness | ⚠ pipeline string only — **no** op flags |
| `fastmath` | explicit fastmath flags on traced ops | normalised | `fadd fast float` in IR |
| `llvm_options` | arbitrary LLVM `cl::opt`s | type dispatch | ISA diff |

⚠ **The dict is un-whitelisted.** There is no key whitelist and no unknown-key
error. A typo is ignored by every consumer **but still rides the JIT cache key**,
so you get a genuine recompile that behaves exactly like no hint. This is the
single most likely way to conclude a knob "does not work".

`fast_fp_math` and `unsafe_fp_math` are **not** a symmetric pair: the first also
seeds ambient tracing fastmath, the second does not.

## Step 5 — Hint layering rules

- `merge_compile_hints` **drops `None`** — you cannot clear an inherited hint.
  `waves_per_eu=0` is the only sentinel ("no override"), and `0` and absent
  produce **different cache keys**.
- The merge is **shallow**: an inner `llvm_options` dict wholly **replaces** the
  outer one. To add one flag, merge by hand:
  ```python
  opts = {**fn.compile_hints.get("llvm_options", {}), "new-flag": True}
  ```
- `flyc.compile[{...}](fn)` **mutates the JitFunction permanently** — it is not a
  scoped copy. For scoped behaviour use `CompilationContext.compile_hints({...})`.
- The autotuner reaches only `waves_per_eu`; every other key must be set around
  the tuned call, e.g. in a `CompilationContext.compile_hints({...})` block.

## Step 6 — Arch applicability

| Knob | gfx942 | gfx950 | gfx1250 | Gate |
|---|---|---|---|---|
| `amdgpu-waves-per-eu` | works (max 8) | works (max 8) | works (**max 16**) | — |
| `amdgpu-num-vgpr` | ⚠ **doubled** | ⚠ **doubled** | exact | `hasGFX90AInsts()` |
| `amdgpu-agpr-alloc` | works | works | **inert** | `hasMAIInsts()` |
| `amdgpu-flat-work-group-size` | works | works | works | — |
| `-amdgpu-sched-strategy=coexec` | ⚠ warns, **runs anyway** | ⚠ warns, **runs anyway** | intended | `hasGFX1250Insts()` |
| `amdgpu-expert-scheduling-mode` | ⚠ gate false (GFX12+) | ⚠ gate false | works | `>= GFX12` |
| `-mattr=+wavefrontsize64` | works | works | ⚠⚠ **emits an empty object** | `supportsWave64()` |
| legacy `s_waitcnt` intrinsic | works | works | ⚠⚠ **aborts llc** | — |

⚠ **The same number means different things across families.**
`amdgpu-num-vgpr=64` yields VGPR 64 + AGPR 64 = **128** on gfx942/gfx950 (silently
doubled, `GCNSubtarget.cpp:625-626`) but exactly **64** on gfx1250. Max occupancy
is 8 waves/SIMD on CDNA and **16** on gfx1250, so `waves_per_eu=2` requests 1/4 of
peak on CDNA and 1/8 on gfx1250. **Do not port these values unexamined.**

> Load `references/arch-gfx950.md` when tuning for gfx950 / MI355X.
> Load `references/arch-gfx1250.md` when tuning for gfx1250.

## Step 7 — Verify it took effect

**Every claim is an artifact grep, never a timing measurement.** Timings on
gfx950 carry ±14% clock noise; a knob that changed nothing in the IR cannot have
changed the runtime.

| Knob | Assert on | Expect |
|---|---|---|
| `waves_per_eu` | `*_llvm_ir.ll` | `"amdgpu-waves-per-eu"="N"` |
| `flat_work_group_size` | `*_final_isa.s` | `.max_flat_workgroup_size: N` |
| `fast_fp_math`, `fastmath` | `*_llvm_ir.ll` | `fast` / `nnan` etc. on the op, e.g. `fadd fast float` |
| `unsafe_fp_math` | pipeline string | `unsafe-math=true` on `rocdl-attach-target`; it adds **no** op flags |
| denormal mode | `*_final_isa.s` | `.amdhsa_float_denorm_mode_32` — ⚠ never the IR attribute |
| an `llvm_options` cl::opt | two `*_final_isa.s` | instruction order differs |
| any ROCDL intrinsic | `*_final_isa.s` | the **mnemonic** is present |
| registers / spills / LDS | `isa_resource_table.py` | never hand-parse ISA |

```bash
docker exec -w $PWD <dev-container> python3 \
    ${CLAUDE_SKILL_DIR}/scripts/llvm_ir_attrs.py $FLYDSL_DUMP_DIR
# compare two runs directly:
docker exec -w $PWD <dev-container> python3 \
    ${CLAUDE_SKILL_DIR}/scripts/llvm_ir_attrs.py <before> <after> --diff
```

### The mandatory negative control

After an assertion passes, **repeat it with a deliberately misspelled key**:

```python
CompilationContext.compile_hints({"waves_per_eU": 4})   # note the typo
```

The assertion **must now fail**. If it still passes, your instrument is dead —
stop and fix that before recording any result. An un-whitelisted hint dict makes
this failure silent by default (§4).

### Replaying just the LLVM half

```bash
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
    -pass-remarks-analysis=kernel-resource-usage NN_llvm_ir.ll -o /dev/null
```

`kernel-resource-usage` prints VGPRs, AGPRs, SGPRs, ScratchSize, Occupancy,
spills and LDS in one shot — prefer it to scraping assembly.

⚠ Two blockers on this route:
- The dumped `.ll` has **no `target triple`** and no `target-cpu` attribute. Always
  pass `-mtriple` **and** `-mcpu`, or you silently target a generic subtarget.
- MLIR emits `f0x`-prefixed hex float literals. The pinned LLVM parses these
  fine ([measured] on gfx950 and gfx1250); if an older `llc` rejects one, check
  its version before rewriting the constant.

⚠ `-stats` and `-debug-only` need an assertions build. On a Release `llc`,
`-stats` **parses cleanly and prints nothing** — grepping it for "spill" yields a
false "no spills".

## Step 8 — Knobs in this repo that do not work

Each was verified by running `llc` against the pinned LLVM. Do not reach for these.

| Knob | Reality | Status |
|---|---|---|
| `maxnreg` | **Was silently inert.** Its only consumer emitted `--amdgpu-num-vgpr` into the dead `opts=` lane, and that is not a flag. No attribute fallback, yet it still changed the cache key — so it looked like a real recompile, and `autotune.Config` was searching an identity transform. | removed; the hint now raises `ValueError` and `Config(maxnreg=…)` raises `TypeError`. Stale cache files drop the key rather than failing to load |
| `amdgpu-schedule-regions` | **Does not exist** in the pin. `llc --help-hidden` has zero hits and rejects it, so in external mode it would fail the `mlir-opt` subprocess. | removed from `flash_attn_generic.py` |
| `lsr-drop-solution=<int>` | **Invalid value.** The option is `cl::opt<boolOrDefault>`; `llc` reports "invalid value for boolean argument". Kernels correctly pass `True`. | test sample switched to `amdgpu-schedule-metric-bias` |
| `amdgpu-expert-scheduling-mode` on gfx942 | **Gate is false.** `hasExpertSchedulingMode()` is `>= GFX12`. [measured] toggling it changes nothing on gfx942/gfx950 while visibly changing gfx1250 codegen. On gfx12 it is also *additive* — it raises waitcnt count and enters hardware SCHED_MODE 2, not a free win. | removed from the gfx942 path |
| `FLYDSL_LLVM_ENABLE_POST_MISChed` | **Casing typo**, one occurrence tree-wide; the natural `..._MISCHED` spelling did nothing. Benign by accident: the default matched intent. | spelling fixed |

Also note `FLYDSL_COMPILE_OPT_LEVEL` is validated and folded into the cache key
but has **zero consumers** — `rocdl-attach-target` hard-codes `O=2`.

## Step 9 — Failure-mode taxonomy

When a knob "does nothing", identify which of these you are in **before**
concluding anything about the hardware:

1. **Silent ignore** — typo'd hint key, or `waves_per_eu=0`. Signature: a
   recompile happened, IR is unchanged. Check: grep the attribute in `*_llvm_ir.ll`.
2. **Flag does not exist** — differs between LLVM trees, so check against the
   one that will compile. Check: `llvm_knob_check.py`.
3. **Dead lane** — anything routed through `gpu-module-to-binary opts=` on AMD.
4. **Type confusion** — `llvm_options` dispatches on the Python type;
   `boolOrDefault` options mis-route. External mode does not skip this — both
   modes run the in-process setter first.
5. **Verify-then-vanish** — the op selects on one arch and is *silently dropped*
   on another (e.g. `rocdl.global_prefetch` without the gfx1250 ISel pattern; <!-- api-check: ignore -->
   this is why `BufferCopyLDS64b` was deprecated). Check: the mnemonic in
   `*_final_isa.s`.
6. **Dead instrument** — cache hit with no dumps, `-stats` on a Release build,
   FileCheck skipped, or an empty-object compile. Check: the positive control.

## Step 10 — Tuning workflow

Cheapest and highest-signal first. Re-verify after **every** step.

1. **Measure the baseline** — `isa_resource_table.py` plus
   `-pass-remarks-analysis=kernel-resource-usage`. Identify the binding limiter
   (VGPR / LDS / SGPR) before turning anything.
2. **`-amdgpu-sched-strategy`** ∈ {default, `max-ilp`, `max-memory-clause`}.
   ⚠ A bad value is **silently ignored** (no validation) and looks like a
   legitimate baseline data point — assert the strategy took effect.
   ⚠ Never include `coexec` in a CDNA sweep: it warns and runs anyway.
3. **`-amdgpu-schedule-metric-bias`** ∈ {10, 25, 50, 100}.
4. **`waves_per_eu`** — set the occupancy target. On the GEMMs in this tree it is
   emitted as a **matched pair** with `rocdl.flat_work_group_size`
   (1↔"256,256", 2↔"512,512"); do not set one without the other there.
5. **Unroll thresholds** — `-amdgpu-unroll-threshold-local` / `-private`.

⚠ Tune `flat-work-group-size` **before** `waves-per-eu`: the workgroup-implied
bound **wins** over an explicit conflicting waves-per-eu
(`AMDGPUSubtarget.cpp:193-201`). Out-of-range values on either are **discarded
wholesale**, not clamped, with no diagnostic.

**Empirical priors from this tree** (source: shipped kernels, not re-measured):
`waves_per_eu=2` in 12 of 15 uses. **Arch is not the selecting axis** — wave count
and kernel family are (4-wave/256-thread → 1, 8-wave/512-thread → 2). Arch selects
the *scheduling* flags instead. Three sibling gfx950 FMHA kernels **disagree** on
`enable-post-misched`, so do not generalise a family setting. The autotuners treat
"do not set the knob" as preferred, penalising a set `waves_per_eu` within a 2% tie
band.

⚠ **Benchmark noise on gfx950 reaches ±14%** from clock variation. Run isolated
(60–80 iters), take a median-of-7, and treat any win smaller than that as noise.
Numbers quoted in `docs/kernel_tuning_guide.md` (the `maxnreg` ~4.5× regression,
scheduler +8–10%) are that document's measurements, **not** re-verified here.

## Analyzer scripts

Both need Python 3.10+, so run them via `docker exec -w $PWD <dev-container> python3 …`
(they exit 2 with instructions if launched on the host's 3.6).

- `${CLAUDE_SKILL_DIR}/scripts/llvm_knob_check.py <flag>...` — does this flag exist
  in the LLVM that will actually compile? Distinguishes cl::opt from function
  attribute from missing. Exit 0 = all are cl::opts, 1 = something is missing or is
  an attribute, 2 = usage error.
- `${CLAUDE_SKILL_DIR}/scripts/llvm_ir_attrs.py <dump-dir> [<dir2> --diff]` —
  per-kernel table of the AMDGPU function attributes present in the dumped LLVM IR,
  or an attribute diff between two runs. Resolves artifacts by glob suffix, never
  by stage index.

Do not reimplement either inline.

## References

Load on demand — each is a deep per-topic file, not needed for every question.

| File | Load when |
|---|---|
| `references/arch-gfx950.md` | tuning for gfx950 / MI355X |
| `references/arch-gfx1250.md` | tuning for gfx1250 |
| `references/amdgpu-attributes.md` | you need the full AMDGPU function-attribute list |
| `references/mllvm-flags.md` | you are sweeping or auditing `-mllvm` / `llvm_options` flags |
| `references/sched-primitives.md` | you are placing `sched_*` barriers, `s_setprio`, or `s_waitcnt` |

## Verified environment

| Thing | Value |
|---|---|
| LLVM pin (kernels) | `e2a39f504`, upstream, LLVM 24 |
| LLVM pin (external mlir-opt) | `c8cf6da4`, ROCm fork |
| Authoritative source tree | a checkout of the pin in `thirdparty/llvm-build-info.json` |
| Runnable driver | `llc` built from that pin (LLVM 24, assertions on) |
| Targets verified | gfx942, gfx950, gfx1250 |
| Dev container | any image with Python 3.10+ |
| Host Python | 3.6.8 — **too old for these scripts**; use the container |
| Scratch | a private dir — **not** `/tmp` on a shared machine |

Facts in this skill were established on 2026-09-14/15 by running `llc` against
the pinned LLVM and by reading `llvm/lib/Target/AMDGPU` in the pin. Claims taken
from repo documentation without re-measurement are marked as such at the point of
use.
