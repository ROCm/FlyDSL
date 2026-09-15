---
name: isa-resource-diff
description: >
  Detect per-kernel GPU resource regressions (VGPR, SGPR, register spills, scratch,
  static LDS) by diffing the final ISA before and after a change, using
  its `isa_resource_table.py` helper. Compile-only: needs no GPU and no profiler run, so
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
python3 ${CLAUDE_SKILL_DIR}/scripts/isa_resource_table.py diff /tmp/isa-before /tmp/isa-after
```

The tool requires **Python 3.10+**. If `python3` is older it exits 2 with a clear
message; use `python3.10 ${CLAUDE_SKILL_DIR}/scripts/isa_resource_table.py …` instead.

Both sides may independently be a dump directory or a `.json` snapshot, so a
baseline can be captured once and reused:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/isa_resource_table.py summarize /tmp/isa-before --json baseline.json
python3 ${CLAUDE_SKILL_DIR}/scripts/isa_resource_table.py diff baseline.json /tmp/isa-after
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

Only changed and problematic kernels are printed. The **last** stdout line is
always the `RESULT:` verdict and matches the exit code exactly — script against
that line, not against a fixed number of them: when there are problems, a count of
untrustworthy items is printed above it.

**Columns marked `*` are regression triggers**; the rest are context.

| Column | Trigger | Read from | What it is |
|---|---|---|---|
| `vgpr` | yes | `.vgpr_count` metadata | The count that decides occupancy: `arch + accumulator` where the register file is unified (gfx90a and later MFMA-capable parts), `max(arch, accumulator)` where it is split (gfx908) or where there are no AGPRs at all |
| `arch_vgpr` | no | `.set` symbol `num_vgpr` | The arch VGPR count on its own |
| `agpr` | no | `.set` symbol `num_agpr` | The accumulator VGPR count on its own |
| `sgpr` | yes | `.sgpr_count` metadata | SGPRs including the fixed extras (VCC, XNACK, FLAT_SCRATCH) |
| `numbered_sgpr` | no | `.set` symbol `numbered_sgpr` | SGPRs without those extras; tells a real increase from VCC becoming live |
| `vgpr_spill` | yes | `.vgpr_spill_count` metadata | VGPRs the allocator spilled |
| `sgpr_spill` | yes | `.sgpr_spill_count` metadata | SGPRs the allocator spilled |
| `scratch_bytes` | yes | `.private_segment_fixed_size` metadata | Private segment per work-item; exact on every target |
| `lds_static_bytes` | yes | `.group_segment_fixed_size` metadata | Statically allocated LDS per work-group |
| `lds_read` / `lds_write` | no | instruction count | `ds_read`/`ds_load` and `ds_write`/`ds_store` sites |
| `scratch_store` / `scratch_load` | no | instruction count | `scratch_*` sites; `n/a` where spilling goes through `buffer_*` |
| `matrix_ops` | no | instruction count | MFMA / WMMA / sparse-MFMA (`v_smfmac_*`) sites |

Three things are easy to misread:

- **Do not add `arch_vgpr` and `agpr` to `vgpr`.** `vgpr` is LLVM's own
  occupancy number and is the only VGPR-family trigger; the other two are shown so
  you can tell *which half* moved, not so you can total them. On a unified register
  file `vgpr` already *is* their sum; on gfx908, where the files are split, it is
  `max(arch, agpr)` — which is still the occupancy number there, since a wave that
  needs 64 arch VGPRs and 60 AGPRs occupies 64 slots in each of two 256-entry
  files. Either way, moving accumulators into AGPRs — which the tuning guide
  recommends — deliberately does not count as a regression.
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
tool's own confidence. Everything that would leave the answer partial reports `2`
and never `1`: a crash, an empty dump directory, a dump file that does not parse
or does not decode, a metadata entry with no kernel identity or a duplicated one,
a negative resource count, a kernel present on only one side, and a target that
differs — or that the tool cannot name — on either side.

For scripting, `-q/--quiet` drops the table (the untrustworthy count still
prints above the verdict), and `--json PATH`
writes the full comparison as machine-readable JSON:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/isa_resource_table.py diff /tmp/isa-before /tmp/isa-after -q --json report.json
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
- **Diffs across targets are refused** (exit 2). Register files and LDS banking
  differ between architectures, and `xnack`/`sramecc` change code generation
  within one, so both sides must report the same processor *and* the same target
  features. A target the tool cannot name is refused for the same reason. The
  triple's environment field is normalized, so `amdgcn-amd-amdhsa--gfx942` and
  `amdgcn-amd-amdhsa-unknown-gfx942` are the same target.
- **Warnings on stderr never change the exit code.** They describe the input tree
  and are worth reading before trusting a clean result.
