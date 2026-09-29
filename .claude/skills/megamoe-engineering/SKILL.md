---
name: megamoe-engineering
description: >
  Develop, review, debug, validate, and tune AITER/FlyDSL MegaMoE on AMD
  multi-GPU EP systems. Use for MegaMoE or MegaMoEV2 architecture, MTPR,
  fixed-slot/compact dispatch, fused prepare/quant, payload dedup/fanout,
  Stage1/Stage2, A8W4/A4W4/SmoothQuant paths, AOT/JIT, 8-GPU correctness,
  CUDA Graph hangs, ATT traces, or ATOM/SGLang end-to-end performance. Do not
  use for an ordinary non-Mega fused_moe/GEMM task.
allowed-tools: Read Edit Bash Grep Glob Agent WebFetch WebSearch
---

# MegaMoE Engineering

Use this skill as a decision system, not as permission to replay old commands.
MegaMoE spans AITER, FlyDSL, MORI and serving-framework code; stale source,
runtime or routing assumptions can produce plausible but invalid results.

## Start every task here

1. Read the active repository's `CLAUDE.md` and locate any `AGENTS.md` that
   governs files in scope.
2. Classify the task and read only the matching references below.
3. Resolve the current source, branch, commit, dirty state, imported Python
   files, FlyDSL build, PyTorch/HIP, MORI and GPU model. Never infer these from
   a directory name or an earlier report.
4. Use a clean, isolated worktree based on the requested branch. Preserve
   unrelated dirty work; never copy a historical source tree over current
   code.
5. Read the current implementation and tests before treating any historical
   document as truth. A dated result is evidence for its recorded stack only.
6. Before using an eight-GPU node, identify every `/dev/kfd` holder and its
   container/job owner. Do not stop another workload without fresh authority
   in the current conversation.

The read-only helper [scripts/preflight.sh](scripts/preflight.sh) records most
of the provenance needed in steps 3 and 6.

## Reference router

| Task | Read |
|---|---|
| Understand or change the algorithm, buffers, dispatch or quant path | [architecture-and-protocol.md](references/architecture-and-protocol.md) |
| Find the authoritative source, tests, configs, framework hooks or local evidence | [repository-map.md](references/repository-map.md) |
| Build a node/container, select a runtime, inspect GPU ownership or diagnose apparent compilation hangs | [environment-and-safety.md](references/environment-and-safety.md) |
| Prove correctness, CUDA Graph safety, AOT coverage, CI reachability or whole-model health | [validation.md](references/validation.md) |
| Establish baselines, interpret Stage1/Stage2/E2E time, capture ATT/PMC or tune a shape | [performance-and-tracing.md](references/performance-and-tracing.md) |
| Diagnose a hang, race, silent error, buffer overflow or cross-rank protocol mismatch | [correctness-and-debugging.md](references/correctness-and-debugging.md) |
| Add a model/network or reason about V4-Pro, M13, GLM-5.2 or Kimi-K3 geometry | [models-and-shapes.md](references/models-and-shapes.md) |
| Locate prior runbooks, performance tables, traces, reviews and incident reports | [evidence-index.md](references/evidence-index.md) |

For a PR review, read architecture, validation, correctness and the evidence
index. For performance tuning, read architecture, validation, performance and
the relevant model section. For a serving regression, also read environment
and repository map.

## Source-of-truth order

When sources disagree, use this order:

1. current checked-out production source and its tests;
2. the exact serving image/runtime and import-path evidence;
3. artifacts from a run with recorded commits, command and raw logs;
4. a dated design/review document;
5. remembered behavior or an unversioned performance number.

Stop and resolve a disagreement at levels 1-3. Do not average incompatible
results or describe a workaround as a root-cause fix.

## Non-negotiable engineering rules

- Correctness precedes performance. A candidate is not tunable until eager
  output, eight-rank reference comparison and CUDA Graph replay pass.
- Preserve the exact mathematical order of quantization, routing, GEMM,
  routing-weight application and combine. A faster non-equivalent pipeline is
  a correctness bug.
- Cross-rank wire fields must be derived from rank-invariant inputs. Local
  token buckets may select compute geometry, but may not silently select a
  different payload layout, ready protocol or P2P dtype.
- A zero-token rank must still enter the collective protocol when peers have
  work. Returning early can strand the EP group.
- A ready flag does not make prior P2P writes visible. Audit release, publish,
  wait and acquire as one protocol.
- CUDA Graph and long queued chains are separate gates from one synchronized
  eager call. Test back-to-back launches and the framework's real layer count.
- Treat `max_tok_per_rank`/MTPR as an ABI and capacity choice, not only a tune
  hint. Validate runtime tokens, source encoding, buffer sizes and the
  fixed/compact boundary.
- Do not claim AOT from a warm JIT cache. Build the declared profile and prove
  a fresh process succeeds with run-only/cache-miss enforcement.
- Do not add Stage1 and Stage2 event times and call the result E2E. Kernels,
  streams and ranks can overlap; measure the graph span and report rank mean
  and rank max separately.
- Compare candidates on the same machine, container, runtime, weights, route,
  MTPR and timing path. Re-run the baseline adjacent to the candidate when the
  expected gain is near run-to-run noise.
- A normal non-fatbin JIT specialization should finish in seconds on the known
  setup. Minutes of silence require process/GPU/port/cache inspection; do not
  label it "still compiling" without compiler activity or a new artifact.
- Tune from trace evidence. First name the critical stage and instruction or
  resource bottleneck, then change the smallest relevant knob or code path.
- Keep production defaults free of experimental environment switches and dead
  configurations. Use temporary overrides only to establish causality, then
  either encode a validated selector rule or remove the experiment.

## Standard work loops

### Functional or protocol change

1. Draw the before/after dataflow and list every producer, consumer, buffer,
   epoch/ready flag and compile-key field affected.
2. Add a CPU/config invariant test where possible.
3. Compile representative fixed and compact variants with cache isolation.
4. Run the focused eight-rank accuracy/Graph/stress matrix.
5. Re-run the performance anchors and serving cases affected by the change.
6. Review the final diff against the entire MegaMoE call chain, not only the
   edited function.

### Performance tuning

1. Freeze source and environment; record a fresh paired baseline.
2. Collect synchronized per-rank kernel timing and identify the critical rank.
3. Capture ATT for the dominant kernel; use PMC when bandwidth, occupancy or
   cache behavior is the unresolved question.
4. Form one measurable hypothesis. Change one code path or a bounded config
   family.
5. Run the cheap correctness gate, then paired perf. Reject noisy/non-general
   wins.
6. Repeat until the requested target or an evidenced hardware/protocol limit.
7. Full-scan all supported buckets and both MTPR modes before cleanup.

### Hang investigation

1. Confirm whether the active GPU kernel is prepare, Stage1, Stage2, combine or
   an unrelated framework kernel. Capture dispatch traces; do not infer it from
   the last host log line.
2. Reproduce with a watchdog and an owned-process cleanup trap.
3. Preserve asymmetric rank tokens, routing and queued launch depth while
   bisecting one protocol phase at a time.
4. Record progress counters from a non-blocking/pinned path where possible;
   ordinary D2H copies can wait behind a wedged persistent kernel.
5. Prove the exact unsatisfied wait or missing visibility edge. A timeout,
   extra synchronization or relaxed comparison alone is not a fix.

## Definition of done

A completed change reports:

- repo/branch/full commit and diff scope;
- image/runtime/FlyDSL/AITER/MORI/PyTorch/HIP/GPU provenance;
- exact command and raw-log location;
- functional matrix with eager, reference error, Graph replay and stress;
- Stage1, Stage2/combine and E2E rank-mean/rank-max data;
- base-versus-candidate delta for every requested shape;
- AOT/JIT and serving results when those paths are affected;
- cleanup status, known limitations and processes/artifacts left behind.

Use a Markdown table for final comparisons. Keep raw logs and traces outside
the repository; commit only durable instructions or small curated baselines.

## Knowledge freshness

This skill intentionally records durable invariants and routes to dated
evidence. It cannot make old performance numbers current. When a task reveals
a new validated invariant, incident root cause, model geometry or canonical
runner, update the relevant reference in a dedicated documentation change.
Record the validating commit and date. Never promote an unverified experiment
or an active worktree path into a universal rule.
