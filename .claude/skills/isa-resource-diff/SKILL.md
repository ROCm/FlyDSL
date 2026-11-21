---
name: isa-resource-diff
description: >
  Detect per-kernel GPU resource regressions (VGPR, SGPR, register spills, scratch,
  static LDS) by diffing the final ISA before and after a change, using
  its `isa_resource_table.py` helper. Compile-only: needs no GPU and no profiler run, so
  it works on any target the compiler supports and runs in seconds. Use when asked
  whether a change increased register pressure, caused spilling, or hurt occupancy,
  when reviewing a kernel change for resource impact, or as a fast pre-check before
  spending a profiling run. Its `loops` subcommand also shows what a change did
  to a hot loop (instruction mix, wait order, VGPR liveness estimate).
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
| Which LLVM knob moves these numbers, and did it apply? | `/llvm` (compile hints, function attributes, `-mllvm` flags) |

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

For a target without local hardware, add `FLYDSL_GPU_ARCH=<gfx> COMPILE_ONLY=1`:

```bash
FLYDSL_GPU_ARCH=gfx950 COMPILE_ONLY=1 \
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

### Step 4 — Diff the hot loops before timing

The table cannot see what a change did to a loop's schedule. `loops` prints, per
loop (a backward branch in the final ISA), its instruction mix, its lowest
`s_waitcnt vmcnt`, and the order of its memory, MFMA, wait and barrier
instructions, in a form that a plain text diff compares:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/isa_resource_table.py loops /tmp/isa-before --kernel attn > before.txt
python3 ${CLAUDE_SKILL_DIR}/scripts/isa_resource_table.py loops /tmp/isa-after --kernel attn > after.txt
diff -u before.txt after.txt
```

```
attn_tile_kernel: vgpr 252
  loop 1 .LBB0_13 - .LBB0_22+: instr 904 valu 543 mfma 128 s_nop 73 v_cmp 65 v_cndmask 65 scratch 0 vmcnt_min 8
      K0 BAR DRx2 K1 M DR M ... BR LD K1 LDx8 W16 K0 M DR M W8 K0 Mx23 ...
```

Tokens: `LD`/`ST` global or buffer load/store, `SL`/`SS` scratch load/store,
`DR`/`DW` LDS read/write, `M` MFMA/WMMA, `W<n>` `vmcnt(n)`, `K<n>`
`lgkmcnt(n)`, `BAR` barrier, `BR` conditional branch; `x<n>` repeats. Look
for a wait that drops (`W8` -> `W0` after the first loads of a phase), a new
`BR` around loads, and `SL`/`SS` inside the loop. Observed on gfx942: a change
that removed ~50-150 instructions from a decode kernel's KV loop at equal VGPRs
ran 2-4% slower. Its loop waited `vmcnt(0)` right after the first K load of
the QK phase, where the old loop waited `vmcnt(8)`; the table said `OK`.

- `--blocks` prints every basic block, not just loops. Use it when a change
  adds a runtime branch that duplicates code (two versions of a stage that
  rejoin): at the join the waitcnt pass merges both versions' outstanding
  loads and may replace per-chunk `vmcnt(N..0)` waits with one `vmcnt(0)`.
  Compare the blocks of each version and the blocks after the join. Observed
  on gfx942: ~1 us more per call on short launches at unchanged VGPR and spill
  counts.
- `--live` adds a VGPR liveness estimate (arch + accumulator; backward dataflow
  over the basic blocks): the maximum live count and the instruction where it
  peaks, per kernel and per loop or block. Ask it before moving a load
  earlier: which region holds the peak, and how much room is left where the
  load would be in flight. It is syntactic (EXEC-masked writes count as full
  writes, allocation padding is ignored), so it reads below `vgpr`: 1-18
  below on 155 gfx942 kernels checked, never above.

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

Only changed and problematic kernels are printed. A clean table does not clear
the hot loop: before timing a change that keeps the resources, run Step 4. The **last** stdout line is
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
- **`COMPILE_ONLY` retargets the compiler, not your host code.**
  `FLYDSL_GPU_ARCH` changes what `get_rocm_arch()` returns. Launcher code that
  queries the device while tracing still sees the local GPU: for example
  `torch.cuda.get_device_properties(...).gcnArchName` used for an arch dispatch,
  or `multi_processor_count` used to size splits or grids (as
  `kernels/attention/pa_decode_fp8.py` does). If such a value picks a code path
  or becomes a compile-time constant, the dump holds the local GPU's variant
  compiled to the target ISA. Then both sides of the diff can describe a kernel
  the target never runs, and before/after differences can appear or vanish
  for the wrong reason. Observed: a compile-only check on a gfx942 host
  reported 12 changed planner kernels per target in an aiter kernel that
  dispatched on `gcnArchName`. With the query returning the target arch, all
  of them were identical. Dispatch on `get_rocm_arch()` in launcher code;
  otherwise confirm on the target hardware.
- **Warnings on stderr never change the exit code.** They describe the input tree
  and are worth reading before trusting a clean result.
