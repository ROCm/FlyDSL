---
name: review-pr
description: >
  Advisory AI code review for FlyDSL pull requests. Catches silent-correctness bugs,
  JIT cache-key mistakes, architecture-capability holes, index-width overflow, layout
  algebra errors, dead flags, and tests or benchmarks that cannot fail. Never acts as a
  merge gate. Use when asked to review a FlyDSL PR, a branch, or a set of local changes,
  optionally with an explicit validation report path.
allowed-tools: Read Bash Grep Glob
---

# FlyDSL PR Review — advisory tier

This skill supplies hints to a human reviewer. Its judgement is stochastic and never blocks a
merge. Only a reproducible blocker from an explicitly supplied, head-matched
`validation_report.json` may be treated as a deterministic gate.

**What makes this repo different from a kernel library.** FlyDSL is a compiler *and* a kernel
collection, so a defect can live at four depths: the Python DSL front end, the JIT cache, the
MLIR passes, or one leaf kernel. The blast radius differs by orders of magnitude between them,
and Step 4 exists to place the diff before any rule fires.

**The dominant failure mode here is silence.** Of roughly thirty distinct defects reconstructed
from merged bugfix PRs, eleven returned silently wrong numbers, five were silent performance
regressions, two were tests or benchmarks that lied, and only six crashed. Weight the review
toward "what would fail without saying anything," not "what would fail."

---

## Step 1 — Fetch

```bash
# PR number, optional owner/repo, optional validation report path.
# Also accepts owner/repo#N as the first argument.
.claude/skills/review-pr/fetch.sh "$PR" "${REPO:-ROCm/FlyDSL}" "${VALIDATION_REPORT:-}"
```

The script prints the PR identity, the CI rollup with its caveats, human review comments with
Copilot filtered out, and the deterministic diff scan. It ends with `work_dir=...`. Read
`$WORK/pr.diff` and `$WORK/pr_meta.json` before proceeding.

**Getting a validation report.** Without one this review is static-only and may not assert any
runtime behaviour — including that performance did not regress. To produce one, run the
companion executor and pass its output back into `fetch.sh`:

```bash
git worktree add --detach /tmp/fly-base "$BASE"
python3 .claude/skills/validate-kernel-pr/validate_pr.py \
    --repo /tmp/fly-base --patch "$WORK/pr.diff" --head-sha "$HEAD" \
    --tests <target> --bench-cmd "bash scripts/run_benchmark.sh" \
    --out /tmp/validation_report.json
```

That executor is the only thing in this repository that gates on performance; see its skill for
why (`scripts/compare_benchmark.py` returns 0 unconditionally). Its `perf` stage compares base
and head A/B/A on one claimed GPU against a noise floor measured during the run. A `BLOCK` from
it is reproducible evidence and may gate; this review's own verdict stays advisory.

**Copilot is filtered deliberately.** It authors about 59% of inline comments on this repo
(1134 of 1936). Its findings are mostly style and are not evidence of a real defect; the human
comments are the signal. Do not repeat a Copilot finding as if a maintainer had raised it.

**What CI green actually means here — state this accurately or not at all.**

| Check | What a green tick proves | What it does not prove |
|---|---|---|
| `test` matrix (mi325 / mi355 / mi35x / navi) | Tests passed on four runners | Nothing about shapes the suite does not contain |
| benchmark step | The benchmark *ran* | **Nothing about performance.** `scripts/compare_benchmark.py` prints ratios and `return 0` unconditionally — a regression cannot turn CI red. Only a `validate-kernel-pr` report covers this |
| `multi-gpu` | — | Skipped unless a maintainer adds the `multi-gpu` label |
| ATOM / vLLM / SGLang | — | These are nightly cron workflows, never PR checks. A change aiter consumes has **zero** downstream coverage at PR time |
| docs-only path | — | `detect-changes` can substitute a green placeholder for the whole GPU matrix |

**Cross-file verification — do this before reporting any kernel or compiler finding.** The diff
shows changed lines, not the whole story.

- The other half of a mask, a barrier, or an address derivation often lives in a sibling module
  (`kernels/common/`, `flash_attn_utils.py`) rather than the file in the diff. Read the function
  in the head checkout, not just the hunk.
- Layout and lowering behavior is split across Python (`expr/primitive.py` emits `fly.*` ops) and
  C++ (`lib/Dialect/Fly/`). Grep both before claiming a lowering is missing.
- Before claiming a value is not widened, check whether the widening happens on the line *above*
  (`phys = fx.Int64(phys_row[sub])` then `phys * n_kv * ...` is correct, and PR #1064 is exactly
  that shape).

**Classify every CI failure before blaming the PR.** Compare against main in the same window; a
shard that fails identically on main is baseline noise. Note the standing cross-repo hazard: CI
clones aiter at `main` HEAD, so an aiter-side API change can redden PRs that touch nothing
related (PR #710).

---

## Step 2 — Semantic Understanding (answer all five before rules)

**Q1 — What changed computationally?** Not "improves perf" — which algorithm, address
computation, layout, or lowering changed?

**Q2 — At what depth?** Leaf kernel / DSL front end (`python/flydsl/expr`) / JIT and cache
(`python/flydsl/compiler`) / MLIR pass (`lib/`)? This decides Step 4.

**Q3 — Hardware and dtype scope.** gfx942 (CDNA3, wave64, 64KB LDS) / gfx950 (CDNA4, wave64,
160KB LDS) / gfx1100, gfx1151, gfx1201 (RDNA, wave32, 64KB) / gfx1250 (wave32, 320KB LDS, WMMA +
TDM)? bf16 / f16 / fp8 / mxfp4? Prefill or decode?

**Q4 — Performance claim: what is the mechanism?** Not "faster" — fewer HBM round trips, better
XCD locality, deeper prefetch, a fastmath flag that now actually reaches the op?

**Q5 — Does the description explain WHY, or only WHAT?** Surface-only descriptions correlate with
generated code; treat as elevated risk and lean harder on Step 6 and Step 7.

---

## Step 3 — PR Type Classification

Determines which Step 5 categories are mandatory.

- [ ] **New or modified kernel** → E (movement math), D (index width), B (gate holes), A3 (cache
      tag), T (test can fail), Step 6 (sibling diff)
- [ ] **New trait / constexpr / builder kwarg** → A3 (in the cache tag?), A4 (read anywhere?),
      B2 (threaded through every builder route?)
- [ ] **Touches `compiler/jit_function.py`, `jit_argument.py`, `protocol.py`, `_flydsl_key`** →
      all of A, plus Step 4 Tier 1
- [ ] **New arch, or an arch predicate change** → C1–C5, and the four-place arch checklist in C5
- [ ] **`expr/numeric.py`, `typing.py`, `primitive.py`** → G4, G5, Tier 1
- [ ] **MLIR pass / dialect / TableGen (`lib/`, `include/`)** → G1–G3, G6, FileCheck coverage
- [ ] **`expr/arith.py` or any `fx.*` math wrapper** → F1, F2
- [ ] **Attention / online-softmax numerics** → F3 (wave-uniform branch), F4 (fp8 normalizer)
- [ ] **Host launcher / wrapper** → H1–H4, D2
- [ ] **Perf PR** → P1–P4, and the cold-cache rule in P3
- [ ] **Test or benchmark only** → T1–T7
- [ ] **Touches anything aiter imports** (`kernels/**` reachable from `aiter/ops/flydsl/`) →
      I1–I3, and the downstream-coverage note in Step 1
- [ ] **Autotune / tuned configs** → A5 (tuning schema), M1 (dead knob)

---

## Step 4 — Blast Radius

Apply these questions to every file in the diff, including new files.

```
Q1 — If this file were wrong, would EVERY kernel miscompile, mislower, or fail to build?
     -> YES -> Tier 1 (compiler backbone)

Q2 — Would it silently change which compiled artifact is served, or make a served
     artifact stale?
     -> YES -> Tier 1c (cache backbone) — the failure is invisible by construction

Q3 — Is it the arch capability table or an arch-specific atom directory, so a bug
     breaks one whole architecture family?
     -> YES -> Tier 2

Otherwise -> Tier 3 (one kernel / one op).
```

| Tier | Files | Failure mode |
|---|---|---|
| **1** | `compiler/jit_function.py`, `kernel_function.py`, `ast_rewriter.py`, `expr/typing.py`, `expr/numeric.py`, `expr/primitive.py`, `compiler/backends/rocm.py` | Every kernel: wrong codegen or mass compile failure |
| **1** | `lib/Dialect/Fly/Transforms/LayoutLowering.cpp`, `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`, `lib/Dialect/Fly/IR/FlyOps.cpp`, `lib/Dialect/Fly/Utils/IntTupleUtils.cpp`, `lib/Dialect/Fly/Utils/NormalForm.cpp` | Every kernel: wrong addresses, usually with no crash |
| **1c** | `_flydsl_key()`, `_CACHE_INVALIDATING_ENV_VARS`, `jit_argument.py` `__cache_signature__`, per-kernel `cache_tag` tuples | Stale or missed artifact — **silent by construction** |
| **2** | `runtime/device.py` (`get_warp_size`, `is_rdna_arch`), `utils/smem_allocator.py` (`SMEM_CAPACITY_MAP`), the per-family atom directories under `lib/Dialect/FlyROCDL/` (CDNA3, CDNA4, GFX11, GFX120X, GFX1250), `expr/rocdl/` | One architecture family |
| **3** | `kernels/**`, most `tests/kernels/test_*.py` | One kernel or op |

**Mandatory Tier 1 / 1c checks — answer before writing the verdict:**

- [ ] List every changed public symbol and grep its callers: `rg -n '<symbol>' python/ kernels/ tests/`.
      A caller not covered by the PR's tests is a finding.
- [ ] For a Tier 1c change, state the **direction** (see A) and which existing on-disk caches it
      invalidates or wrongly preserves.
- [ ] For `lib/` changes, is there a FileCheck test under `tests/mlir/` pinning the emitted
      arithmetic, not merely the result type? PR #1052 is the model: it asserts `arith.constant 48`,
      `muli`, `addi`.
- [ ] State plainly: if this change is wrong, what breaks and how would anyone notice?

**De-facto ownership** (no CODEOWNERS file; derived from `git log --format='%an' -- <path>`,
re-derive per path rather than trusting this snapshot):

| Path | Top committer |
|---|---|
| `python/flydsl/expr/`, `lib/Dialect/`, `lib/Conversion/`, `python/flydsl/compiler/` | Feng Shijie (@sjfeng1999) |
| `kernels/**`, `tests/kernels/` | Felix Li (@coderfeli) |
| `compiler/ast_rewriter.py` | Xudong Yuan (@xudoyuan) |
| `kernels/norm/`, attention correctness | @jhinpan |
| `lib/CAPI/`, `lib/Bindings/`, `lib/Runtime/`, `tools/` | @jli-melchior |

---

## Step 5 — Rule Checklist

Severity is advisory: 🔴 high risk / ⚠️ should fix / 📝 note.

**🔴 evidence threshold.** Before firing any 🔴, write down the concrete input that triggers it —
the shape, dtype, arch, sequence length, or flag combination. "This multiply could overflow" is
not a finding; "at `head_dim=128, H=64`, `S >= 131072` puts the flattened element count past
2^31" is. If you cannot name the triggering case, **downgrade to ⚠️ or drop it.** A 🔴 that reads
as a definite defect but names no demonstrable input is how a false positive lands on a
maintainer's PR.

---

### A — Cache and Recompilation Correctness

*The signature FlyDSL failure. It is silent by construction: the wrong artifact runs and nothing
reports an error.* Cache keys have two levels — `manager_key` (function source, transitive helper
sources, closure scalars, `_flydsl_key()`) and a per-call tuple (`_env_`, `_target_`, `_hints_`,
argument signatures). **Every A finding must state its direction**, because the fixes are
opposite.

**A1 — Over-keying breaks AOT reuse** ⚠️/🔴
Adding to the key something whose effect is already captured by a resolved object. The canonical
case is an arch env var: AOT precompilation runs on a GPU-less build host with `ARCH=gfx950`
injected, while runtime leaves it unset and detects from the device. The resolved `GPUTarget` is
identical but the raw `_env_` segment diverges, so every shipped cache misses.
Real example (PR #624): removed `ARCH`, `FLYDSL_GPU_ARCH`, `HSA_OVERRIDE_GFX_VERSION` from
`_CACHE_INVALIDATING_ENV_VARS`, which downstream surfaced in aiter as `RuntimeError: AOT cache miss`.
The current tuple is five genuine codegen knobs; `FLYDSL_COMPILE_LLVM_DIR` and
`FLYDSL_EXTRA_SOURCE_DIRS` are path-like and carry the same latent hazard.
FP self-check: is the added value's effect genuinely *not* already represented by `_target_` or
an argument signature? If it is represented, adding it is the bug.
→ `⚠️ A1: [var] added to the cache key but its effect is already captured by [X] — build-host and runtime values diverge, so AOT caches miss`

**A2 — A runtime value leaking into the key** ⚠️
`inspect.signature()` under `from __future__ import annotations` (PEP 563) stringifies `fx.Int32`,
so the argument is no longer recognized as a DSL type and its **value** joins the key — a fresh
compile per distinct value.
Real example (PR #556): fixed by `resolve_signature(..., eval_str=True)`; the regression test pins
both directions, `("n", int) in key` and `("n", (int, 1)) not in key`.
→ `⚠️ A2: [arg] enters the key by value not by type — every distinct value recompiles`

**A3 — A new trait or kwarg not added to the hand-maintained `cache_tag`** 🔴
`cache_tag` tuples are hand-written (three in `flash_attn_utils.py` alone, 44 / 20 / 29 entries)
and nothing enforces completeness. A new trait absent from the tag means two configurations share
one compiled artifact.
Trigger: the diff adds a trait/constexpr/builder kwarg that reaches codegen, and no `cache_tag`
line changes in the same diff. The Step 1 scan reports this pairing under `cache_key`.
→ `🔴 A3: [trait] affects codegen but is absent from [file]'s cache_tag — [config X] and [config Y] will share one artifact`

**A4 — A trait in the cache key that nothing reads** ⚠️
The dual of A3, and trivially checkable: count read sites for every name in `cache_tag`; zero
reads means the user-facing flag does nothing.
Real example (PR #1020): `DUALWAVE_SWP_LAZY_RESCALE` was in the fp8 tag while both call sites used
`lazy_correct_o` unconditionally — "so the flag did nothing," fixed with a real `const_expr` branch.
→ `⚠️ A4: [trait] is in cache_tag but has no read site — the flag is inert; wire it or remove it`

**A5 — Tuned results outliving the kernel that produced them** ⚠️
When a kernel's numerics or codegen change, autotune winners recorded under the old behavior are
still served unless the tuning schema version moves.
Real example (PR #1022): the LDS-barrier fix also bumped `TUNING_SCHEMA` for exactly this reason.
→ `⚠️ A5: kernel behavior changed but the tuning schema did not — previously tuned configs stay selected`

**A6 — Device identity confused with architecture identity** ⚠️
HSACO is arch-specific, not device-specific. Folding `device_id` into a *disk* key stores a
duplicate artifact per device; resolving arch from "the first GPU `rocminfo` reports" compiles
against the wrong device on a mixed-GPU box.
Real examples: PR #597 (duplicate per-device artifacts, and two differing instances producing the
same disk key); PR #403 (`_get_lds_size_per_cu()` cached process-wide with no device parameter);
PR #1057 (fixed by resolving arch from `A.device` and adding it to the key).
→ `⚠️ A6: [quantity] resolved from the first enumerated GPU / keyed by device — resolve from the tensor's device and key by arch`

---

### B — Gate and Flag Threading

*The code looks complete; certain inputs silently take a path that drops a feature.*

**B1 — A new branch inserted above an existing validation** 🔴
A fast path, early return, or recursive split placed before the block that raises on unsupported
combinations. The `raise` still exists but is now unreachable on the new path, and the rebuilt
`kw` for the sub-call often omits arguments.
Real example (PR #1020): a split ran before fp8 modifier validation while `kw` omitted `bias`,
`alibi_slopes` and `sink`; the reviewer forced the path and got output bit-identical to the call
with no modifier, where the unsplit path raised `NotImplementedError`.
Real example (PR #844): `return_lse=True` with the caller omitting `lse` fell back to `Out`, so
fp32 LSE overwrote the output buffer.
→ `🔴 B1: [param] is dropped on [new path] which now precedes its validation — [what silently computes wrong]; assert or forward it`

**B2 — A flag not threaded through every route** 🔴/⚠️
A new kwarg on a public entry point where the file has several builder or launcher routes
(`_build_splitk`, `_build_varlen`, paged vs dense, fp8 vs bf16). The unthreaded routes fall back to
the library default, so the override is a no-op — and tests stay green because they pass the flag
explicitly.
Real example (PR #1056): `causal_lpt=False` not honored by `_build_splitk` or long `_build_varlen`.
Check: enumerate every builder that reaches the gated code and confirm each accepts *and forwards*
the flag.
→ `🔴/⚠️ B2: [flag] is not forwarded by [routes] — the caller's override is silently ignored on those paths`

**B3 — A default-off trait placed in front of previously unconditional behavior** 🔴
The diff shape is `if const_expr(not A and not B)` becoming `if const_expr(NEW_TRAIT and not A ...)`
where `NEW_TRAIT` defaults `False` and only the builder in the PR's scope passes it. Pure silent
perf regression; nothing fails because nothing covered the mapping.
Real example (PR #1009): `XCD_SWIZZLE` defaulted `False` and only the fp8 builder passed it, so bf16
lost the XCD remap for two weeks — restoring it recovered **+17.4% at S=180,180 and +19.4% at
S=239,580**, bit-identical output. Note the sibling builder's comment "states the opposite
rationale and is wrong," so comments are not evidence here.
→ `🔴 B3: [trait] defaults False and only [builder] passes it — [other builders] silently lose [behavior]`

**B4 — A surviving always-false gate** 🔴
`False and ...`, `if False:`, `and False` left in a condition dead-codes the path entirely.
Real example (PR #650): `if const_expr(False and N >= tile_cols ...)` had hidden two compile
failures and kept the vectorized softmax fast path dead.
→ `🔴 B4: [line] can never be true — the path is dead; remove the literal or fix the underlying failure`

**B5 — Fail-open where the contract promises fail-closed** ⚠️
A `catch` or `if (failed(...))` returning to a default path with no diagnostic; `continue` on an
unparseable input; a warning where an error is contracted.
Real examples: PR #470 (`emitWithFlyPasses` failure silently skipped an optimization the user
explicitly opted into via `compile_hints`); PR #1015 (a corrupted byte changed a reported count
while the snapshot still read "trustworthy, no problems," and a `vgpr_count` of -1 was classified
as an improvement, exit 0).
→ `⚠️ B5: [path] falls back silently — the caller opted in and gets no signal; emit a diagnostic or fail`

---

### C — Architecture Capability

*Correct for gfx942/wave64; silently wrong on gfx1250.*

**C1 — A capability derived from a family predicate** 🔴
`is_rdna_arch()` answers "is this RDNA," which is **not** "what is the wave size." **gfx1250 is
wave32 and is not matched by RDNA prefixes** — this is the repo's standing trap.
Real example (PR #1024): `32 if is_rdna_arch(arch) else 64` produced the wrong `wave64` codegen flag
for gfx1250. The workaround had metastasized into the docs, which read that `get_warp_size` returns
64 for gfx1250 and "the gfx1250 kernels hardcode `WAVE_SIZE = 32` themselves." Fixed by one source
of truth, `runtime/device.py::get_warp_size`.
Tell: a kernel hard-coding `WAVE_SIZE = 32` while the shared helper reports 64 means the predicate
is wrong, not that the kernel is clever.
→ `🔴 C1: [capability] derived from [family predicate] — gfx1250 lands on the wrong side; route through get_warp_size / the capability accessor`

**C2 — An arch prefix predicate too wide or too narrow** ⚠️
`startswith("gfx12")` catches gfx1250, which is not a GFX120X WMMA part; `gfx10*` routed to a GFX11
WMMA atom that older RDNA targets do not have.
Real examples: PR #544 (*"can we restrict this to gfx11 so older RDNA targets do not select an
unsupported WMMA atom?"*); PR #943 (renamed to `GFX120X_WMMA` and narrowed the guard from `gfx12` to
`gfx120` so it no longer overlaps gfx1250).
→ `⚠️ C2: prefix [X] also matches [arch] which lacks [feature] — narrow the predicate`

**C3 — Arch resolved from the first enumerated GPU** ⚠️ — see A6; report under whichever fits.

**C4 — A tolerance or heuristic derived on one arch applied to all** ⚠️
Real example (PR #1022): a gfx950-derived tolerance applied everywhere, while the exact production
row `M=32768, N=8192, bf16` failed four independent Navi runs at 0.523–0.533 scaled relative error
against a 0.02 limit.
→ `⚠️ C4: tolerance/heuristic [value] was derived on [arch] but applies to all — [arch B] behaves differently`

**C5 — A new arch added in fewer than four places** ⚠️
Adding an architecture requires all of: `SMEM_CAPACITY_MAP` (`utils/smem_allocator.py`),
`get_warp_size` (`runtime/device.py`), the `lib/Dialect/FlyROCDL/<FAMILY>/` atom directory, and the
test skip predicates (`tests/arch_compat.py`, per-test `skipif`). Missing the LDS entry means
`check_smem_capacity` silently skips the check for the unknown arch.
FP self-check: an arch string inside an already arch-specific file (`*_gfx1250.py`) is normal; fire
only when a shared dispatch or capability path learned a new value.
→ `⚠️ C5: [arch] added to [places] but not [missing] — [consequence]`

---

### D — Index Width and the ABI Boundary

**D1 — Index × stride with no 64-bit widening** 🔴
Structural trigger, not a name list: a position-shaped value multiplied by an extent-shaped value
producing an address, with no `fx.Int64(...)` on the line or the line above. The repo idiom widens
the *position* operand first.
Real example (PR #1064): `(((phys_row[sub] * n_kv + kv_h) * STEPS_PER_PAGE + step) * head_dim + head_element) * 16`
became `phys = fx.Int64(phys_row[sub])` then the same chain — "physical page ids fit in i32, but
their element offsets do not once an individual KV cache grows beyond 2 GiB."
The Step 1 scan lists candidates. Work the list: clear each one, or name the production size at
which the product passes 2^31. If the list is empty, say so rather than skipping the category.
→ `🔴 D1: [expr] multiplies in int32 — widen [operand] to fx.Int64; overflows at [concrete size]`

**D2 — A tensor flattened before a launch** 🔴
`_LayoutPlan` builds `struct_fmt` with shapes packed unconditionally as `i` (int32); only *strides*
honor `use_32bit_stride`. A flattened tensor therefore makes the packed dim the whole element count.
Real example (PR #1020): the fp8 attention path flattened Q/K/V/O to 1-D and overflowed at
`B*S*H*D >= 2^31`, i.e. `S >= 131072` at `D=128, H=64`. The bf16 path passes the natural 4-D shape
and is exempt — which is also the FP self-check.
→ `🔴 D2: [tensor] is flattened before launch — the packed int32 extent overflows at [shape]; pass the natural rank`

**D3 — Element extent and descriptor byte span are two separate bounds** ⚠️/🔴
The signed int32 element extent (2^31) and the unsigned buffer-descriptor byte span (2^32) are
different limits; a guard on one is not a guard on the other, and a validation that checks only
`.shape` checks neither.
Real examples: PR #1056 (*"bound the runtime pitch before launch (signed-int32 element extent and
unsigned-32-bit descriptor byte span, both exclusive)"*); PR #960 (`row * stride + column` in signed
i32 while public validation checked only shape — "the kernel may silently read the wrong data");
PR #1020 (a `q.numel()`-only guard that also ran only when `B > 1`, so cross-attention with short Q
and large K/V never fired it).
→ `⚠️/🔴 D3: the guard covers [one bound] but not [the other] — [input] passes validation and still wraps`

**D4 — A defaulted width on a width-carrying C++ factory** ⚠️
`IntAttr::getDynamic(MLIRContext*, int32_t width = 32, ...)` silently records width 32 for an i64
value.
Real example (PR #729): fixed by reading the actual `IntegerType` width and rejecting anything but
32 or 64.
→ `⚠️ D4: getDynamic called without an explicit width — an i64 value is recorded as i32`

---

### E — Data-Movement Shape Math

**E1 — Truncating division in a copy decomposition** 🔴
Any `//` (or a `x if x < K else K` clamp) deriving a chunk count or width where the numerator is
not provably a multiple of the denominator. Chained truncating divisions are the worst case.
Real example (PR #1064): `head_dim=192` gives `QCHUNK = 12`, so `QLOAD_UNIT = 8` and
`N_QLOADS = 12 // 8 = 1` — "rounding it down to one 8-element load leaves one third of every query
row unstaged in LDS." Fixed to `8 if QCHUNK % 8 == 0 else 4` plus `assert QCHUNK % 4 == 0`.
Real example (PR #1007): `bytes_per_thread_a = (tile_m * tile_k * elem_bytes) // total_threads` then
`// a_load_bytes` dropped the remainder, giving relative error ~1.0 against `torch.mm` at
`tile_m=112, tile_n=256, tile_k=64, bf16` — and `_TILE_PRELOAD_TABLE` advertised those tiles as
tuned, so autotuning produced garbage. The fix validates the *tile size*, not either truncated
intermediate: `if a_tile_bytes % (total_threads * a_load_bytes) != 0: raise`.
→ `🔴 E1: [expr] truncates for [shape] — [what is never fetched]; assert the exact divisibility or ceil-div with a masked tail`

**E2 — Copy-atom width ≠ register vector width × dtype width** 🔴
`BufferCopy128b` with a `make_rmem_tensor(K, dtype)` where `K * dtype.width != 128`. Hard LLVM
abort, not a graceful error: `CastInst::Create: Assertion 'castIsValid(...)' failed`.
Real examples: PR #650 (`VEC_WIDTH = 8` correct for 16-bit, invalid for f32; fixed to
`vec_width = 128 // elem_bits`); PR #564 (same defect via a `--vec-width 8` CLI flag; fixed by
capping the atom at 128b and chunking as `chunk * block_dim + tid` to stay coalesced).
Check the identity for **every dtype the factory accepts**. A hard-coded vector width next to a
bit-width-named atom is a defect until proven otherwise.
→ `🔴 E2: [atom] pairs with [K]x[dtype] = [bits] bits — invalid cast at lowering for [dtype]`

**E3 — A store mask guarding fewer dimensions than the address fuses** 🔴
When the address fuses dimensions (`off = col * dhw + row`), "out of range in dim k" does not imply
"out of bounds in memory," so hardware OOB suppression does not fire — padded rows alias real
elements of the next column.
Real example (PR #969): fp8 conv3d epilogue masked only `col_valid`; 12/200 runs failed at exactly
`rel_err 3.075e-01` (the identical value ruled out a plain race). The bf16 kernel already guarded
rows via `_row_chk` and the split-K path already checked `row < npq`; only the non-atomic fp8 store
was missing it.
Check: count dimensions in the address expression versus dimensions in the mask, then compare
against every sibling writing the same layout.
→ `🔴 E3: mask covers [dims] but the address fuses [dims] — padded rows alias live elements and race real stores`

**E4 — A launch path over 256 threads without `known_block_size`** ⚠️
Without it the AMDGPU backend keeps its default max flat workgroup size of 256 and the launch
aborts.
Real example (PR #639): rmsnorm crashed on real DeepSeek-R1 (`N=512`, `N=1536`) and Qwen3 q/k-norm
(`N=128`) shapes; the shape list had no `M > 8192, N <= 2048` entry, so the path was never exercised.
→ `⚠️ E4: [path] can launch [n] threads without known_block_size — launch aborts at [shape]`

**E5 — A raw block-id read after a remap** 🔴
Once a kernel derives remapped block indices, any later raw `gpu.block_id(...)` bypasses the remap.
Real example (PR #782): `preshuffle_gemm`'s epilogue re-read `gpu.block_id("x")` and scrambled the
output; the same PR fixed `/` used instead of `//` on index values in `xcd_remap_bx_by` and its
assumption that `num_wgs % num_xcds == 0`.
Expect exactly one derivation site per kernel.
→ `🔴 E5: [line] re-reads the raw block id after the remap at [line] — output is written to the unremapped tile`

---

### F — Numerics and Fastmath

**F1 — An `fx.*` math wrapper without `@dsl_math_wrap_result`** 🔴
The decorator is what fills `fastmath` from `current_fastmath()`. A wrapper that takes `**kwargs`,
hand-rolls its own `Vector`/`Numeric` wrapping, and calls the raw `arith.*` builder emits
`fastmath<none>` whatever the enclosing kernel requested.
Real example (PR #1035): `maxnumf` lacked it while its siblings `maximumf`/`minimumf` had it — a
mechanical migration had moved 29 call sites across attention, paged decode, MLA and MoE onto the
undecorated function. Cost **11.1–11.9% throughput at long sequence lengths, and nothing failed**;
the fix recovered +8.4%/+9.0% bit-identically.
Check: diff any new or edited `fx.*` wrapper against its nearest sibling. A missing decorator or a
missing named `fastmath` parameter is the whole bug.
→ `🔴 F1: [wrapper] lacks @dsl_math_wrap_result / a named fastmath param — emits fastmath<none> at every call site`

**F2 — A Python literal where a fastmath enum is required** ⚠️
`fastmath=True` stringifies into an invalid `#arith.fastmath<True>` attribute (PR #650). Set
fastmath once via the compile hint rather than per op (the house rule, PR #894, #848).
→ `⚠️ F2: fastmath=[literal] is not an enum value — use the FastMathFlags enum or the ambient compile hint`

**F3 — A wave-uniform branch whose body assumes the predicate held per lane** 🔴
`ballot` + `read_exec` collapses a per-lane predicate into one decision for the whole wave, so lanes
that did *not* trigger are dragged into the body.
Real example (PR #1033): a dragged lane had `m_tile_max < m_row`, so `exp2(m_row - m_tile_max)`
scaled the accumulator *up* to `inf`, then `inf/inf`. At `B=2, S=8192, H=32, D=128` with K scaled
32×, **9.2% of the output was NaN; at 64×, all of it** — default configuration, not a regression.
The eager path never had the bug because it took `m_new = maxnumf(m_row, m_tile_max)` first.
Ask: is this body safe for a lane whose own predicate was false? Require idempotence or monotonicity.
→ `🔴 F3: the [ballot] body consumes [per-lane value] unguarded — non-triggering lanes produce [inf/NaN] at [input]`

**F4 — A low-precision intermediate underflowing while its normalizer sums wider** 🔴
A softmax-like reduction where `P` is cast to fp8 (e4m3 min subnormal 2^-9) before the second
matmul, while the normalizer accumulates before the cast, with no scale-up into the format's range.
Real example (PR #1020): with `V` all ones the exact answer is 1.0; bf16 gave 0.99996 and **fp8 gave
0.637**, 58–83% of rows short. Fixed by adding `_P_HEADROOM_LOG2 = log2(448)` as a free FMA addend.
A random-init model does not show it — the PR needed trained q/k projections.
→ `🔴 F4: [value] is cast to [fp8] while [normalizer] sums pre-cast — rows underflow at [distribution]`

**F5 — Rounding-mode divergence between a fast path and its generic sibling** ⚠️
A hand-written conversion (`+ 0x8000`, `>> 16`, `cvt_pk_bf16_f32`) on a vectorized path where the
scalar/tail path converts differently.
Real example (PR #848): `+ 0x8000` rounds exact ties away from zero; an exact-tie probe for float
bits `0x3f808000` gave bf16 `0x3f81` versus `0x3f80` from both `_to_elem_vec` and PyTorch. The
reviewer's point generalizes: the divergence "stays within the bf16 `atol=2e-2`, so tests won't
catch it."
→ `⚠️ F5: [fast path] converts with [mode] while [generic path] uses [mode] — systematic bias inside tolerance`

**F6 — Compile-time folding that does not wrap like the hardware** ⚠️
Static folding on Python's arbitrary-precision ints disagrees with the `arith.*` op it replaces.
Real example (PR #973): `Uint32(0xFFFFFFFF) + Uint32(2)` folded to `0x100000001` rather than `1`;
fixed with a `_wrap_int` reduction at every construction and fold site (with a load-bearing
`Boolean` carve-out, since width-1 signed would map True to -1).
→ `⚠️ F6: [fold] does not match what arith.[op] computes at [width] — reduce to the type's width`

---

### G — Layout Algebra and Compiler Internals

**G1 — A `{value, attr}` adaptor pair describing different shapes** 🔴
In `IntTupleValueAdaptor`, pairing a scalar value with a nested-tuple attr yields silently wrong
coordinates. Per issue #1049 this is severe: "it can make edge predicates incorrectly true and
therefore permit out-of-bounds memory accesses."
Real example (PR #1052): a `(48,48)` tile whose layout correctly carried `48E0, 48E1` returned grid
coordinate `(0,1)` as logical `(0,1)` instead of `(0,48)` — the SSA value was the raw dynamic tile
id rather than `tile_id * 48`.
→ `🔴 G1: [adaptor] pairs a [scalar] value with a [nested] attr — coordinates resolve to [wrong value]`

**G2 — Type inference that updates part of a result type** 🔴
A slice that recomputes the layout but leaves the base unmoved.
Real example (PR #707): `SliceOp` returned `CoordTensorType::get(srcTy.getBase(), newLayout)`; the
fix adds `layoutCrd2Idx` + `intTupleAdd` to move the base.
Check: when a type-inference method slices, is *every* component updated?
→ `🔴 G2: [op]'s inferred type updates [component] but not [component] — addresses stay at the source base`

**G3 — Reading a value off an attribute without its static predicate** 🔴
A dynamic `IntAttr` carries `value == 0`, so `getValue()` silently returns 0.
Real example (PR #926): `add_offset(reg_ptr, dynamic)` lowered to `ub.poison` + `vector.extract %0[0]`
— "the dynamic index is dropped entirely and the access is hardcoded to slot 0 — a deterministic
read of the wrong element, not merely an undefined value." Note the second-order lesson: guarding
the offset alone was insufficient; the fix returns the root unconditionally and makes only the
offset optional.
→ `🔴 G3: getValue() without isStatic() — a dynamic attr reads back as 0 and [access] silently targets slot 0`

**G4 — Reconstructed DSL values losing wrapper-only metadata** ⚠️
`__construct_from_ir_values__` is a classmethod and cannot see the exemplar instance, so attributes
living only on the Python wrapper (multi-dim shape, signedness) are lost at every `scf` region
boundary.
Real example (PR #759): a `Vector` with logical shape `(4,1)` collapsed to `(4,)`; the next
`vec_sum += vec` broadcast to `(4,4)` and the `vector<16xf32>` failed to match the loop's
`vector<4xf32>` iter_arg. Fixed by threading an optional `exemplar` through the protocol.
→ `⚠️ G4: [attribute] is wrapper-only and is dropped when the value crosses [region] — thread the exemplar`

**G5 — Promotion implemented twice, or signedness re-derived from MLIR** 🔴
**MLIR integers are signless**, so re-deriving a DSL dtype from an element type cannot recover
unsignedness: a `Uint32` result becomes `Int32`, and downstream shifts, divides and comparisons
switch to signed semantics on unsigned data with no diagnostic.
Real example (PR #816): `Vector._apply_op` did no dtype coercion and reverse-engineered the result
dtype from the produced IR, while `Numeric` auto-promoted. Unified into one lattice; it also fixed
`Float16 + Int32` yielding `Float16` instead of widening to `Float32`.
Any coercion added to one operand kind must be shown to be shared with the other.
→ `🔴 G5: [logic] derives a DSL dtype from a signless MLIR type / duplicates the [other] lattice — unsigned data gets signed semantics`

**G6 — Metadata read before a transform that invalidates it** 🔴
The visible tell is a transform call sitting between the type read and the op construction.
Real example (PR #1043): alignment was read from the pointer *type* and attached to an op built on
the *swizzled* value, over-promising alignment; and a non-power-of-two byte count crashed
`llvm::Align` (`!fly.memref<f32, global, 32:1, align<12>>` crashed `mlir-translate`). Nothing on the
path rejected it — the Python binding checks only a multiple of the element size, `AlignAttr` has no
verifier, and `llvm.load`'s verifier accepts it. Fixed by `getLLVMAlignment` doing both the swizzle
gcd and a round-down to a power of two.
→ `🔴 G6: [metadata] is read before [transform] which weakens it — the op over-promises [property]`

---

### H — Host, Stream and Tensor Hygiene

**H1 — An internal copy racing an explicit stream** 🔴
`q.contiguous()`, `torch.cat(...)`, `.to(...)` execute on the **ambient current stream** while the
kernel consumes the result on the caller-supplied `stream=`.
Real examples: PR #1066 (*"the new side-stream test only uses already-contiguous inputs, so every
copy is a no-op and cannot catch this"* — note the test gap is part of the finding); PR #1020 (a
`torch.cat` with no dependency; a delayed side-stream reproduction "returned permanently corrupted
data even after synchronization").
FP self-check: does the launcher actually accept a `stream=` argument, or does everything run on
the current stream? No explicit stream, no race.
→ `🔴 H1: [copy] runs on the ambient stream while the kernel consumes it on [stream] — add a stream dependency`

**H2 — A reshape that silently materializes the output** 🔴
`reshape()` on an *output* tensor can produce a contiguous temporary; the kernel writes the
temporary and the caller's tensor is never updated.
Real example (PR #403): `final_output` / `split_output` / `split_lse` never got updated.
→ `🔴 H2: [out] is reshaped before the launch — the kernel may write a temporary and the caller sees stale data`

**H3 — `assert` as the only guard on a kernel precondition** ⚠️
Under `python -O` the assertion disappears while the kernel still reconstructs addresses from the
assumed layout.
Real example (PR #801): "it can silently read the wrong row."
→ `⚠️ H3: [precondition] is enforced only by assert — it vanishes under -O; raise, or make the tensor contiguous`

**H4 — Cross-device tensor and stream mixing** ⚠️
Real example (PR #1022): the wrapper accepted tensors on `cuda:0` with a stream from `cuda:1`, so
validation and launch could run against device-0 pointers from device 1.
→ `⚠️ H4: [tensor] device and [stream] device are not checked — validate they match`

---

### I — Cross-Repo Coupling with aiter

FlyDSL production code does not import aiter; the coupling is (a) tests and benchmarks that compare
against aiter, and (b) aiter hosting FlyDSL kernels under `aiter/ops/flydsl/`.

**I1 — A positional call into aiter** 🔴
Real example (PR #710): aiter reordered `pa_reduce_v1`, so `0 → final_output`, `output → final_lse`,
`None → num_kv_splits`. **CI clones aiter at `main` HEAD, so this failed every PR in the repo** the
moment aiter changed, including PRs nowhere near attention. Parameter names are the stable contract.
→ `🔴 I1: [call] passes [n] arguments positionally into aiter — an upstream reorder reddens every PR; pass by keyword`

**I2 — `importorskip` that does not cover symbol imports** ⚠️
It covers a missing *module*, not a drifted API; a top-level `from aiter.ops.attention import X`
raises `ImportError` during collection and takes down the whole session (exit 2).
Real example (PR #840). Use `try/except ImportError` with a module-level skip.
→ `⚠️ I2: [import] is not guarded — an aiter API change aborts collection for the entire file`

**I3 — A stale intra-repo module path after a refactor** ⚠️
Module-level imports abort collection for the whole file.
Real example (PR #870): `kernels/gemm/mxfp4_preshuffle.py` still imported from `kernels.mma.*` after
the helpers moved to `kernels/common/mma/`. When a PR moves a module, grep the old path tree-wide.
→ `⚠️ I3: [module] moved but [importer] still uses the old path — collection fails for that file`

**I4 — A change aiter consumes, with no downstream signal** ⚠️
ATOM / vLLM / SGLang integration are **cron-only**. If the diff changes the behavior or signature of
a kernel reachable from `aiter/ops/flydsl/`, no PR check covers it.
→ `⚠️ I4: [kernel] is consumed by aiter but downstream integration is nightly-only — ask for a manual downstream run before merge`

---

### T — Test and Benchmark Methodology

*The largest category by reviewer volume, and the one most worth being blunt about: a suite that
cannot fail is worse than no suite, because it reports green.*

**T1 — A test that cannot fail** 🔴
Concrete shapes seen here: a decorator inserted mid-function truncating the previous test body; a
timing stub returning a constant; random inputs used to "reproduce" a deterministic failure; a
parametrize matrix whose values never satisfy the fast-path predicate; a reference computed from the
same precomputed objects as the kernel output.
Real examples: PR #1056 (a new test inserted mid-function ended `test_xcd_swizzle_is_bit_identical`
after its local `run` definition — "all four parametrized XCD cases pass without launching a
kernel"); PR #899 (a stub returning constant `1.0` so `BLOCK=64` "wins because it is listed first,
not because it is fastest"); PR #960 (a `--bias` flag that never generated a bias, "producing a
false PASS and misleading performance data").
→ `🔴 T1: [test] cannot fail because [reason] — the regression is unguarded; replace with [independent oracle]`

**T2 — A regression test that passes on the pre-fix commit** 🔴
The local standard of evidence, set by the reviewers themselves: run the submitted test on the
defective head and show it fails there.
Real example (PR #1033): *"I ran this submitted test logic unchanged on the previous defective head
465226df: both cases pass … So reintroducing the -16 per-step cap would leave this regression green."*
Ask for the input that makes the bug visible — near-uniform or random-init inputs hide this repo's
whole numerics class (#1020 needed trained q/k projections, #1033 needed K scaled 32×, #1064 needed
`head_dim=192`, #969 needed `dhw == npq`).
→ `🔴 T2: [test] would also pass before the fix — it pins nothing; **Author must** show it failing on the parent commit`

**T3 — A benchmark measuring the wrong thing** 🔴
Real examples: PR #654 (`for m in re.finditer(...): pass` keeps the **last** match, so layernorm
reported the fully-scalar `fused_add_smoothquant` variant — **1.69 TB/s for months against a real
5.6**, and the current-vs-main gate could not catch it because main was mislabeled the same way);
PR #848 (speedup columns inverted against the formula, printing 3.04 µs vs 1.88 µs as 0.62x);
PR #675 (a dashboard comparing main against a rebuild of the same commit, so `vs_main` was
current-vs-itself).
When a PR changes a benchmark *parser*, ask whether the stored baseline was produced by the same
parser.
→ `🔴 T3: [benchmark] reports [wrong quantity] — the published number is [X] not [Y]`

**T4 — A reference the test's own dtype cannot represent** ⚠️
Real example (PR #643): the harness built block-scale activations from raw fp8 codes (~448) instead
of dequantized values (~0.2), a ~2000× inflation; stage-1 intermediates reached `absmax ≈ 2.4e7`,
overflowing f16 in both the kernel **and** the torch reference; and the test used `return` instead of
`assert`. It reported a correct kernel as 100% wrong. **Diagnostic tell: aiter's own CK kernel failed
the same comparison identically — when every independent implementation "fails," the harness is wrong.**
→ `⚠️ T4: reference [expr] overflows [dtype] at the magnitudes this test generates — the comparison is invalid`

**T5 — Tolerance widened, or shape rows disabled** ⚠️
Compare head against base rather than judging the absolute value: repos legitimately differ per
kernel. A test-only widening with no numerical justification is the concern; a widening alongside a
kernel change needs the justification stated.
→ `⚠️ T5: [tolerance] widened from [x] to [y] with no stated justification — the test no longer guards [what]`

**T6 — A test that only runs in one harness** ⚠️
`run_benchmark.sh` executes test files as scripts, so pytest-only variants never run there, while
`__main__`-only entry points are invisible to pytest.
Real example (PR #481). Also honor the placement rule: `tests/unit/` is for compiler unit tests,
kernel and autotune tests go in `tests/kernels/` (PR #980).
→ `⚠️ T6: [test] runs only under [harness] — it never executes in [the other]`

**T7 — A measurement taken with a warm JIT cache** 🔴
`CLAUDE.md` states it directly: the disk cache "normally invalidates on kernel source and closure
changes. Disable it when debugging stale artifacts, **changing C++ passes, or changing helper code
that is not part of the traced closure**." PR #1009 says the same in situ: "this change touches
helper code outside the traced closure, so the JIT cache key does not move with it and a warm cache
serves the previous kernel."
Therefore: **any measurement in a PR that edits `lib/` or non-closure helpers, without
`FLYDSL_RUNTIME_ENABLE_CACHE=0`, is unverified.** PRs #1009, #1033 and #1035 all state it explicitly;
that is the local standard.
→ `🔴 T7: the PR edits [C++ pass / non-closure helper] but the numbers were taken with a warm cache — **Author must** re-run with FLYDSL_RUNTIME_ENABLE_CACHE=0`

---

### P — Performance Evidence

**P1 — A perf claim with no numbers** ⚠️
`CONTRIBUTING.md` requires hardware, baseline, optimized and improvement. Screenshots are not
numbers. Exception: PRs adding benchmarks without claiming an improvement.
→ `⚠️ P1: perf claimed with no [hardware / baseline / units]`

**P2 — Toy shapes only** ⚠️
This repo's real numbers come from long context (S=180,180 and S=239,580 appear in #1009/#1035) and
production GEMM/MoE rows. `M=1` / `M=16` only is the generated-test signature.
→ `⚠️ P2: benchmark omits [production shape class]`

**P3 — Not reproducible** ⚠️
Missing script, ROCm version, GPU model, or arch. Combine with T7: also missing the cache state.
→ `⚠️ P3: perf claim missing [reproduction detail]`

**P4 — A dead tuning knob** ⚠️
An autotune axis that produces byte-identical ISA across all its values is not a knob.
Real example (PR #785): `Config(waves_per_eu=…)` reached codegen only as `gpu-module-to-binary opts=`
and was "silently dropped by the AMDGPU backend"; the audit found `maxnreg` had the same bug. It was
found because someone asked the one-line question *"waves_per_eu does any effect in FLYDSL?"*
→ `⚠️ P4: [axis] may not reach codegen — **Reviewer should ask** for an ISA diff across two values`

---

### Housekeeping (quick scan — mostly 📝)

These are the conventions maintainers restate most often. Report at most one or two, and only when
the diff makes them concrete.

| Check | Trigger | Flag |
|---|---|---|
| Raw MLIR instead of `fx.*` | `ir.IntegerType`, `arith.*`, `vector.*`, `scf.*`, `memref_alloca`, `ArithValue`, `_to_raw` on a `+` line | `📝 use the fx.* surface (fx.Int32/fx.Int64, fx.copy, fx.gemm) rather than raw dialects` |
| `scf.IfOp` / `_if_then` | explicit scf control flow in a kernel | `📝 use native Python if/for inside @flyc.jit; a device helper containing control flow needs its own @flyc.jit` |
| Inline asm | `inline_asm` | `⚠️ use the s_waitcnt / sched_barrier wrappers; has_side_effects=True can cause a data hazard (#1073)` |
| Deprecated allocator | `SmemAllocator` | `📝 SmemAllocator is deprecated — use SharedAllocator` |
| Duplicated helper | a byte-for-byte copy of an existing helper | `📝 shared helpers belong in kernels/common/; a partial dedup that leaves the copies raises the count (#795)` |
| Tuned configs in a kernel file | config tables inside `kernels/*.py` | `📝 tuning configs belong behind a dispatcher, not in the kernel file (#780)` |
| Launch not via `_run_compiled` | direct `flyc.compile` in a launch path | `📝 route through _run_compiled to cache the CompiledFunction (#776)` |
| Comment volume | comment lines outnumbering code in a new file | `📝 comment density is out of line with the repo (median ~1 line per block)` |
| Missing license header | new source file without the SPDX header | `📝 add the SPDX/copyright header required by CONTRIBUTING.md` |
| New dependency | new import or requirement entry | `📝 CONTRIBUTING.md requires justification for any new third-party dependency` |
| **Counter-rule** | validation added on both sides of the Python/C++ boundary | Do **not** flag missing validation here — maintainers actively reject "over check" (#644, #506, #297) |

---

## Step 6 — The Sibling Diff

**This is the highest-yield move in this repo; do it explicitly, not as an afterthought.** In the
majority of reconstructed defects, the correct code already existed in a sibling path and the bug
was the asymmetry:

| PR | Sibling that was already right |
|---|---|
| #1033 | the eager path took `m_new = maxnumf(...)` first; only the lazy path did not |
| #969 | the bf16 kernel guarded rows via `_row_chk`, and the fp8 split-K path checked `row < npq`; only the non-atomic fp8 store did not |
| #1035 | `maximumf` / `minimumf` carried `@dsl_math_wrap_result`; only `maxnumf` did not |
| #650 | the generic path already used `FastMathFlags.fast` |
| #1009 | the bf16 builder had the XCD remap until a default-off trait was placed in front of it |

Procedure: for every function the diff touches, list its siblings — `_opt` / `_v2` / prefill vs
decode / fwd vs bwd / bf16 vs fp8 / gfx942 vs gfx950 vs gfx1250 / vectorized vs scalar tail — and
compare field by field. Any asymmetry in masking, index width, rounding mode, decorator, or clamp
order is a candidate defect. Report which side you believe is correct and why.

---

## Step 7 — Generated-Code Diagnostic

A clean description does not make the code correct; these checks are mandatory whenever the diff
changes code. Report each as `[verified]` or `[inferred]`.

1. **Hallucinated symbol sweep.** List every symbol new to this diff — function, kwarg, trait,
   attribute, import — and grep each against its definition. Any symbol you cannot locate is a defect
   until proven real.
2. **Twin divergence.** See Step 6. This is the signature generated-code bug: mirrored code where one
   side was left unadapted.
3. **Claim ↔ code, and number provenance.** Does the code enforce what the description asserts? Take
   the most impressive number in the PR and trace it to a script output or log line. A number you
   cannot trace is `[unverified]` — never repeat it as fact.
4. **Safety theater.** For each new guard: is it reachable, will it ever fire, does it swallow a real
   error? (See B5.)
5. **Test calibrated to pass.** See T1/T2. Is the reference a structural twin of the kernel, so the
   same bug lives in both?
6. **Magic constant without derivation.** A new tile size, threshold or epsilon with no stated tuning
   basis.

Additional warning signs: perf numbers that are suspiciously round; a "Test Plan" left as template
text; an AI-attribution footer. Three or more, or any structural check firing, warrants: "elevated
generated-code risk — recommend manual verification of the dispatch logic and test coverage."

---

## Step 8 — Free-Form Review

Read the diff as a domain expert.

- **LDS budget.** gfx942 64KB, gfx950 160KB, gfx1100/1151/1201 64KB, gfx1250 320KB
  (`SMEM_CAPACITY_MAP`). An arch missing from that map makes `check_smem_capacity` skip the check
  entirely. On gfx1250, note that `ds_read`/`ds_write` immediate offsets are 16-bit, so an allocation
  past 64KB can push addressing into VGPRs.
- **Wave size.** wave64 on gfx942/gfx950; wave32 on gfx10xx/11xx/12xx **including gfx1250**. Any
  reduction, ballot, or lane-indexed layout written against wave64 needs an explicit check.
- **XCD mapping.** `NUM_XCD_GFX950 = 8`. A remap is a bijection, so a correct XCD change is
  bit-identical — if output changes, the remap is wrong, not merely retuned.
- Does the tiling make sense for the target's MFMA (CDNA) versus WMMA (RDNA/gfx1250) atom shapes?
- Are hardware tile constants named rather than raw literals scattered through the kernel?
- For mixed fp8 flavors, is the fn/fnuz distinction handled by dtype rather than by arch name?

---

## Step 8.5 — Blind-Spot Check

Answer in full before the verdict: **"Is there any correctness risk, resource hazard, or behavioral
edge case in this diff that none of Steps 1–8 caught?"** If yes, add it to the findings.

---

## Step 9 — Verdict

**Output rules (strictly enforced):**

- Run Steps 1–8 internally. Do **not** narrate steps, show checklists, or name which rules fired.
- Output **only** the card below — nothing before it, nothing after it.
- If there are no findings, omit the findings section entirely.
- "What it does" is one sentence for a reviewer who has not read the diff.
- **At most 5 findings, most severe first.** Rank by severity then blast radius and drop the rest;
  this is a readability limit, not a recall claim.
- **State the validation evidence** on the line under the verdict. Without an exact-head
  `validation_report.json`, write `NOT RUN` and make no runtime claim — findings about perf,
  accuracy or launch failure are then `[inferred]` and phrased as questions.
- Do **not** use rule codes (A3, D1, T7…) in the output; they are internal labels.

```
## FlyDSL PR #NNN — [title]

**[One sentence: what this PR does, in plain terms.]**

Review (advisory): [✅ NO FINDINGS | ⚠️ NEEDS WORK | 🔴 HIGH RISK]
Validation (deterministic): [PASS/NEEDS_WORK/BLOCK/INCONCLUSIVE — target, arch, skipped stages | NOT RUN — no exact-head validation_report.json]
Performance: [per-row result vs the measured noise floor, and whether the cold-cache rule applied | NOT MEASURED — CI's benchmark step cannot fail, so nothing has checked this]
CI coverage: [which GPU runners ran; whether multi-gpu and downstream were skipped]

🔴 [finding]
⚠️ [finding]
📝 [note]
```

Each finding has **three parts**, and is tagged `[verified]` or `[inferred]`:

1. **Problem** — what is wrong, with file:line.
2. **Impact** — what happens at runtime if it is not fixed (silent wrong values / NaN / crash / stale
   artifact / perf regression).
3. **Action** — ends with a verb phrase: "**Author must** …" or "**Reviewer should ask** …". No verb
   means the finding is incomplete; drop it.

`[verified]` means traced through the actual code. `[inferred]` means plausible but unconfirmed —
say so and frame it as a question rather than asserting a root cause.

Good findings:

- `🔴 [verified] kernels/attention/pa_decode_tile.py:549 multiplies phys_row[sub] by n_kv, STEPS_PER_PAGE and head_dim entirely in int32. At head_dim=128 with a KV cache above 2 GiB the offset wraps to a lower page, so decode silently attends to the wrong tokens with no error. **Author must** wrap the page id as fx.Int64 before the multiply chain.`
- `🔴 [verified] The new causal_lpt flag is accepted by the public entry point but neither _build_splitk nor the long _build_varlen route forwards it, so on those two paths the builder default silently re-enables LPT. Every test passes the flag explicitly, so the suite stays green. **Author must** thread the flag through both builders and add a case that exercises the default.`
- `⚠️ [inferred] The 1.31x number is not traceable to any output in the PR, and the diff edits a helper outside the traced closure, so a warm JIT cache would have served the previous kernel. **Author must** re-run with FLYDSL_RUNTIME_ENABLE_CACHE=0 and attach the log.`

Bad findings — do not produce these:

- `⚠️ Missing perf numbers` — no impact, no action.
- `🔴 A3 violation` — a rule code means nothing to a reviewer.
- `⚠️ This multiply might overflow` — no concrete triggering input, so it fails the 🔴 threshold.

---

## Adding New Rules

When a human reviewer catches something this skill missed:

1. Add it to Step 5 under the right category **with a real PR number and a quote as evidence**.
2. If it is mechanically detectable, add a category to `scan_flydsl_diff.py` and seed a defect to
   confirm the scanner fires on it. A check never observed firing is decoration, not a check.
3. Commit as `review-pr: add [rule] from PR#[NNN] — [one line]`.

The skill grows from real review history, not hypothetical patterns.
