# MegaMoE review and cleanup

## Review the exact revision and full business call chain

Resolve PR head, merge base and current main, then inspect both changed code
and its consumers: public wrapper, constructor, config, dispatch/plan,
G1/activation/quant, G2/scatter/combine, framework adapter, AOT and CI driver.
Old findings are leads, not findings on a new head. Re-check each claimed fix.

Start by identifying EP versus TP and gfx950 versus gfx1250. Their rank-token,
wave, memory-copy and communication contracts differ. Never transfer a
requirement just because two classes have the same short name.

For each finding provide:

```text
severity and exact reviewed SHA
small changed-line range
actual code variable/field names
legal triggering input or concrete interleaving
observable failure and affected callers
new in this diff or pre-existing on its base
reproduction status, evidence and limits
smallest coherent repair and tests
```

Use these distinct labels: confirmed correctness bug, conditional/source-proven
bug, measured performance regression, coverage gap, cleanup suggestion and
unverified hypothesis. Environment-variable count or a large diff alone is
not a correctness finding. "No new findings" is not blanket release approval.
Do not post GitHub review comments unless the user requests external posting.

## High-value audits

- **Storage aliases:** two Python tensor/view objects can share the same
  storage and byte range. Trace base pointer, offset, shape, stride and
  lifetime. Different variable names do not create separate scratch.
- **LDS work slots:** a scheduler must not overwrite a descriptor while any
  consumer wave still uses it. Single-thread writes plus an old barrier are
  not automatically a lifetime proof for subsequent loop iterations.
- **Ready/acquire:** move a wait into a scheduler only if every consuming
  thread's acquire follows observation of that same work item's readiness.
  Queue-ready and tile-metadata-ready are different publications.
- **Packed tags:** derive masks, shifts and ranges using the actual encoder
  and decoder fields. Enumerate every legal tag/index and test round trips;
  do not explain a "tag1/tag3" issue with invented variable names.
- **Cross-rank selection:** unequal TPR, empty ranks, masks, cache state and
  framework P/D metadata must not split protocol or collective participation.
- **Compiled geometry:** exercise the real factory/bundle, not only a CPU
  dataclass. Bundle construction can compile variants not reached by one
  runtime bucket. Hardware-specific tune predicates must honor target scope.
- **AOT reachability:** runtime eligibility, parent profiles and child banks
  must agree. A full build can accidentally hide a missing cropped-profile
  artifact. Test both default and trimmed deployments with run-only.
- **API drift:** JIT and AOT may call a shared selector differently. A missing
  required ep_size in an AOT call is not necessarily a changed public
  MegaMoEV2 interface; inspect both call sites and the synthetic merge state.
- **CI reachability:** a test file in a skip list is not coverage. Record the
  actual workflow/job/command and final binary under test.

For source-only work label unexecuted GPU cases clearly. A dry-run cache hit
does not validate P2P memory ordering, reference error or graph replay.

## Cleanup against the actual FlyDSL version

Read the active checkout's
[kernel-code-cleanup skill](../../kernel-code-cleanup/SKILL.md) completely
before a kernel cleanup. If updating AITER, also verify its supported FlyDSL
dependency and the imported build. A recipe valid on current FlyDSL main may
not exist in the serving wheel.

Prefer the supported high-level tensor/layout, pointer-buffer, copy/MMA and
cooperative scan/reduce APIs. Search existing shared helpers first. Current
cleanup guidance evolves: do not permanently encode ptr_buf_tensor, Gtensor
or make_buffer_tensor as universally interchangeable names.

For each replacement preserve:

```text
byte versus element offsets; descriptor range and OOB behavior
address space, integer width, alignment and alias metadata
volatile/cache policy and vector width
atomic operation, scope, ordering and return-value semantics
quant rounding, scale layout, activation/clamp and contraction behavior
loop carried state, SSA dominance and compile-time/runtime branching
LDS layout, barrier participation, MFMA operand form and wait placement
```

Do not hide raw MLIR or inline assembly behind a new local facade and call
that migration complete. Find an equivalent public API, compile paired
variants and compare actual ISA. If no equivalent exists, document the exact
instruction/semantic gap and localize the unavoidable boundary. Do not widen
this task into a compiler/API redesign without authorization.

Compare code-object metadata and ISA with isolated caches/dump directories:
VGPR/SGPR/LDS, scratch, residency, memory widths, wait/atomic/fence scope,
MFMA schedule and hot-loop instruction count. Identical ISA is useful but
does not prove unchanged host launches, allocations or collective behavior.
Finish with eight-rank numerical/Graph/stress and paired performance gates.

Remove branches, helpers and knobs only after proving no supported public
entry, AOT profile, framework mode or test needs them. Preserve source history
through git; do not keep an unused compatibility subsystem indefinitely.
Separate mechanical API cleanup from protocol/resource/layout redesign.

## CSV and asynchronous GPU faults

A tune CSV change can select a new implementation with a different output
layout/reduce contract. Record the exact dispatch tuple, CSV path, matched row,
G1/G2 and terminal reduction kernel. A fault reported by the next HIP call is
not proof that the next kernel caused it; use queue/error or dispatch evidence
to locate the first failing launch.

Use same-image, same-framework, same-model A/B with only CSV selection changed.
A rollback that boots successfully is a mitigation/control, not by itself a
root-cause fix. Synthetic NaNs, duplicate routes or initialized scratch may
prove a conditional bounds bug without proving that real CI generated those
inputs. Keep those conclusions separate. Save the authentic triggering
metadata when reproducing and convert it into a minimal regression.
