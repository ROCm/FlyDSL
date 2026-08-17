---
name: isa-resource-diff
description: >
  Detect per-kernel GPU resource regressions (VGPR, SGPR, register spills, scratch,
  static LDS) by diffing the final ISA before and after a change, using
  scripts/isa_resource_table.py. Compile-only: needs no GPU and no profiler run, so
  it works on any target the compiler supports and runs in seconds. Use when asked
  whether a change increased register pressure, caused spilling, or hurt occupancy,
  when reviewing a kernel change for resource impact, or as a fast pre-check before
  spending a profiling run.
  Usage: /isa-resource-diff [<test-or-command>] [--arch <gfx>]
allowed-tools: Read Write Bash Grep Glob
---

# ISA Resource Diff

Compare per-kernel register, spill, scratch, and LDS usage between two builds to
catch resource regressions that functional tests do not surface.

## Pick the right skill first

| Question | Skill |
|---|---|
| Did my change increase registers / cause spills / grow LDS? | **this skill** (compile-only, seconds, no GPU) |
| *Why* is this kernel slow — which instructions stall, and on what? | `/kernel-trace-analysis` (needs a GPU run + rocprofv3 ATT trace) |
| Which commit made it slow? | `/bisect-perf-regression` (needs a runnable benchmark) |
| How do I collect a trace at all? | `/capture-kernel-trace` |

This skill measures **resources, not time**. A clean result here does not mean
performance is unchanged — it means register/LDS/spill pressure is unchanged.
A regression here is a strong, cheap signal that is usually worth acting on
before profiling, because spilling and occupancy cliffs dominate most kernel
slowdowns. See §7 of `docs/kernel_tuning_guide.md` for what to do about one.

## Arguments

| Argument | Required | Description |
|---|---|---|
| `<TEST-OR-COMMAND>` | No | What to run to produce dumps. Defaults to asking the user. Example: `pytest tests/kernels/test_softmax.py -q` |
| `--arch <gfx>` | No | Target for compile-only runs, e.g. `gfx950`. Omit to use local hardware |

If the user already has two dump directories or two JSON snapshots, skip to Step 3.

## Workflow

### Step 1 — Capture the "before" side

Check out or stash to the baseline state first, then:

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa-before FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 -m pytest tests/kernels/test_softmax.py -q
```

`FLYDSL_RUNTIME_ENABLE_CACHE=0` is **required, not optional** — see Pitfalls.

For a target without local hardware, add `ARCH=<gfx> COMPILE_ONLY=1`:

```bash
ARCH=gfx950 COMPILE_ONLY=1 \
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa-before FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 -m pytest tests/kernels/test_softmax.py -q
```

### Step 2 — Capture the "after" side

Apply the change, then rerun **the identical command** into a *fresh* directory
(`/tmp/isa-after`). Same test, same parameters, same arch, same cache setting.

### Step 3 — Diff

```bash
python3 scripts/isa_resource_table.py diff /tmp/isa-before /tmp/isa-after
```

The tool requires **Python 3.10+**. If `python3` is older it exits 2 with a clear
message; use `python3.10 scripts/isa_resource_table.py …` instead.

Both sides may independently be a dump directory or a `.json` snapshot, so a
baseline can be captured once and reused:

```bash
python3 scripts/isa_resource_table.py summarize /tmp/isa-before --json baseline.json
python3 scripts/isa_resource_table.py diff baseline.json /tmp/isa-after
```

## Reading the output

```
* = regression trigger; other columns are informational.
  vgpr = total (arch+acc, LLVM's occupancy number); arch_vgpr/agpr are its split -- do not add them.
kernel                               *vgpr     arch_vgpr  agpr ... *lds_static_bytes lds_read ...
-------------------------------------------------------------------------------------------------
gemm::d128_fmha_fwd_kernel_0 942->960(+18) 942->960(+18)     0 ... 212992->229376(+16384)      12

compared 1 of 1 kernels; 0 unchanged; 1 changed; worsened: 2; improved: 0
RESULT: REGRESSION
```

Only changed and problematic kernels are printed. The last two stdout lines are
always a count line and a `RESULT:` verdict that matches the exit code exactly.

**Columns marked `*` are regression triggers**; the rest are context. The full
column reference is in `docs/testing_benchmarking_guide.md` §"Compare per-kernel
ISA resources". Three things are easy to misread:

- **Do not add `arch_vgpr` and `agpr` to `vgpr`.** `vgpr` is already the total
  (arch + accumulator) and is the only VGPR-family trigger. The other two are its
  split, shown so you can tell *which half* moved. Moving accumulators into AGPRs
  — which the tuning guide recommends — deliberately does not count as a regression.
- **`n/a` is not `0`.** It means the quantity does not exist on this target, e.g.
  `scratch_store`/`scratch_load` on a target that spills through `buffer_*`. Use
  `scratch_bytes`, which is exact everywhere.
- **`?` means unparsed** — the tool could not read something it reports. Any `?`
  forces exit 2. Never treat it as unchanged.

### Exit codes

| Code | stdout | Meaning | What an agent should do |
|---|---|---|---|
| `0` | `RESULT: OK` | Everything comparable, no trigger increased | Proceed |
| `1` | `RESULT: REGRESSION` | Everything comparable, a trigger increased | Investigate — this is a real finding |
| `2` | `RESULT: NOT TRUSTWORTHY` | The tool cannot answer | **Fix the inputs and rerun. Do not report "no regression"** |

Exit `1` is a claim about the code under test; exit `2` is a claim about the
tool's own confidence. A crash, unreadable input, an empty dump directory, a
kernel present on only one side, or a mismatched architecture all report `2`,
never `1`.

For scripting, `-q/--quiet` prints only the verdict line, and `--json PATH`
writes the full comparison as machine-readable JSON:

```bash
python3 scripts/isa_resource_table.py diff /tmp/isa-before /tmp/isa-after -q --json report.json
case $? in
  0) echo "no resource regression" ;;
  1) echo "REGRESSION — see report.json" ;;
  *) echo "inconclusive — inputs are bad, do not claim a clean result" ;;
esac
```

## Acting on a regression

Map the column that moved to a cause, then follow `docs/kernel_tuning_guide.md`:

| Column increased | Usual cause |
|---|---|
| `vgpr` | More live values; larger tiles or deeper pipelining/unrolling |
| `vgpr_spill` / `sgpr_spill` / `scratch_bytes` | Register pressure crossed the budget — normally the most damaging of these signals |
| `lds_static_bytes` | Bigger shared tiles or added double buffering; may cross an occupancy step |
| `sgpr` alone, by 2, with `numbered_sgpr` flat | Usually just VCC becoming live — rarely meaningful |

If a resource regression is confirmed but the kernel is not actually slower,
say so rather than "fixing" it: these are proxies for occupancy, not timings.
Confirm with `/kernel-trace-analysis` before reworking a kernel.

## Pitfalls

These silently produce a *confident wrong answer* if ignored:

- **Dump directories are keyed by kernel name only, with no specialization key.**
  Every JIT specialization of one kernel writes to the same directory and
  overwrites the previous one, so a parametrized test leaves only its last
  variant. Diff one shape at a time when the answer must be exact.
- **A cache hit produces no dump at all.** Hence `FLYDSL_RUNTIME_ENABLE_CACHE=0`
  on both sides. Without it a kernel can silently vanish from one side, which
  the tool reports as `ONLY IN BEFORE` and exit 2.
- **`lds_static_bytes` is static LDS only.** A kernel using
  `SharedAllocator(static=False)` reports `0` no matter how much LDS it takes at
  dispatch; the tool prints a `dynamic LDS in use` note for it. LDS regressions
  in those kernels are invisible here.
- **The stage number in `NN_final_isa.s` varies between runs.** Nothing cleans the
  dump directory, so reusing one can leave two files side by side. The tool uses
  the highest-numbered one and warns; prefer a fresh directory per run.
- **Cross-architecture diffs are refused** (exit 2), because register files and
  LDS banking differ. Compare like with like.
- **Warnings on stderr never change the exit code.** They describe the input tree
  and are worth reading before trusting a clean result.

## Verifying the tool itself

`tests/unit/test_isa_resource_table.py` is backend-agnostic and needs no build:

```bash
python3 -m pytest tests/unit/test_isa_resource_table.py -q
```

It generates its ISA input in `make_isa()` rather than checking dumps in, so the
test file states exactly which parts of LLVM's output the parser relies on, for a
CDNA shape and an RDNA shape. If the tool ever reports `?` on a dump that looks
healthy, LLVM's assembly format has probably drifted — update `make_isa()` to
match the new shape rather than loosening the parser, since the failure is
deliberately loud.
