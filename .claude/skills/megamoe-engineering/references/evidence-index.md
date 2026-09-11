# Evidence index and historical lessons

This file routes agents to prior evidence available in the established
workspace. These artifacts are not shipped with FlyDSL and may not exist on a
different machine. Their dates, branches and environments bound every claim.

## How to consume historical evidence

1. Read the artifact's provenance and command before its conclusion.
2. Resolve the referenced commit; distinguish PR head, merge result, reverted
   history and later corrected PR.
3. Verify that imported code/build paths matched the named checkout.
4. Treat a table without raw logs and environment as a lead, not a baseline.
5. Re-run the baseline adjacent to the candidate for any current decision.
6. Extract durable invariants into this skill only after current-source review
   and a reproducer confirm them.

## Architecture and implementation records

| Artifact | Use | Caveat |
|---|---|---|
| `/data/ghu/MEGA_MOE_V2_IMPLEMENTATION.md` | Historical end-to-end algorithm, data structures and kernel roles. | Predates later #5346 production changes; never use as current launch-boundary truth. |
| `/data/ghu/megamoe_aot_design.md` | AOT bundle/preload goals and cache reasoning. | Verify current AOT registry, job signatures and deployment profiles. |
| `/data/ghu/MEGA_V2_DEADLOCK_REPORT.md` | Deadlock investigation patterns and counter protocol evidence. | Confirm the exact commit before claiming the same root cause. |
| `/data/ghu/MEGA_V2_CUDAGRAPH_DEADLOCK_EXPERIMENTS.md` | CUDA Graph/queued-launch experiments. | Experiments include rejected mitigations; a passing workaround is not necessarily the fix. |

## Runbooks and environment records

| Artifact | Use | Caveat |
|---|---|---|
| `/data/ghu/run_book_hgl.md` | ATOM whole-model command history and operational checks. | Pin image, ATOM and AITER anew; do not paste an old command blindly. |
| `/data/ghu/MEGAMOE_RUNBOOK.md` | SGLang fixed-length/AgentX workflow history. | Confirm the current AMD MegaMoE backend flags in SGLang source. |
| `/data/ghu/host47_megamoe_env_setup_20260826.md` | Known M13/FlyDSL-universe Python and library setup. | Host/container builds can be replaced; verify imports and ABI. |
| `/data/ghu/hgl_0822_final_patch.md` | Prior ATOM concurrency commands and standard performance reference. | Historical stack only; useful for reproducer construction, not a current claim. |

## Performance records

| Artifact | Use | Caveat |
|---|---|---|
| `/data/ghu/mega_moe_v2_perf_compare_20260824.md` | Curated MTPR=MAX and MTPR=tokens comparison. | Confirm which prepare/quant architecture and commits produced it. |
| `/data/ghu/mega_moe_v2_perf_compare_20260824.full_backup.md` | Longer optimization diary and early fused-prepare evidence. | Contains intermediate/rejected states; do not quote the last-looking number without provenance. |
| `/data/ghu/mega_moe_v2_v4_pro_performance_host46.md` | V4-Pro microbenchmark reference. | Host46 environment and contention changed repeatedly. |
| `/data/ghu/m13_smooth_fullscan_compare_20260827.md` | Byte M13 SmoothQuant full-scan comparison. | Check labels and commits: experimental fused variants sometimes regressed most buckets. |
| `/data/ghu/m13_mx_a8w4_prefill_tuning_20260909.md` | Active M13 MX A8W4 1024-32768 tuning log. | Work in progress until final full scan, accuracy and Graph gates are recorded. |
| `/data/ghu/tmp.md` | PR performance table and historical SGLang fixed/AgentX results. | Convenience summary; use linked/raw runbook evidence for PR claims. |

## Reviews and compatibility records

| Artifact | Use | Caveat |
|---|---|---|
| `/data/ghu/flydsl_universe_megamoe_review_20260826.md` | Review of customer MegaMoE fork and protocol risks. | Re-check current branch; some findings were fixed or bypassed later. |
| `/data/ghu/flydsl_pr972_review_20260831.md` | A4W4 PR review, including Stage1 queue race and A8W4 interaction analysis. | Tie every finding to its reviewed revision. |
| `/data/ghu/aiter_pr4980_review_20260909.md` | Independent AITER PR review and MegaMoE compatibility notes. | Draft evidence until review commit and reproducer are recorded. |

## Trace helpers and raw traces

Useful local helpers:

```text
/data/ghu/parse_megamoe_trace.py
/data/ghu/analyze_megamoe_kernel_trace.py
/data/ghu/run_universe_a8w4m13_8_att_flexible_20260828.sh
```

Example raw trace directory:

```text
/data/ghu/0825_c4096_trace
```

The directory name was later identified as a concurrency-512 capture despite
the `c4096` label. This is a standing lesson: derive graph batch, shapes and
kernel calls from trace metadata, not filenames.

## Relevant upstream history

| Link | Interpretation |
|---|---|
| AITER PR `#5001` | Large MegaMoE optimization effort; merged then reverted. Use for design/perf archaeology, not as production head. |
| AITER PR `#5346` | Corrected/reopened production line containing the MegaMoE work. Verify current main ancestry. |
| AITER issue `#5322` | GLM-5.2 support request and expected geometry/integration context. |
| AITER PR `#4980` | Separately reviewed AITER change; assess interaction against its exact revision. |
| FlyDSL PR `#972` | A4W4-focused change reviewed for shared MegaMoE/A8W4 regressions. |
| FlyDSL PR `#1098` | Cooperative scan/reduce extension API migration reference. |

Use GitHub or the local git object database to resolve exact SHAs. PR numbers
are identifiers, not immutable source snapshots.

## Durable lessons extracted from the evidence

- A clean microbenchmark can coexist with a large whole-model regression when
  launch order, AOT, MTPR padding or graph overlap changes.
- The historical best architecture for one path is not automatically correct
  for SmoothQuant: quantization order and Stage1 register pressure matter.
- A `produced no kernel` AOT summary can be the downstream symptom of a host
  selector/API exception; preserve the earliest exception.
- Equal local token counts hide rank-dependent wire decisions and zero-token
  protocol failures.
- Stage1/Stage2 times overlap; tables that appear not to add up can still be
  correct if E2E is measured over the overlapped graph.
- A raw trace filename, branch name or directory suffix is not provenance.
- Killing all GPU processes may make a run pass while destroying root-cause
  evidence and another user's job; ownership-aware cleanup is mandatory.

## Evidence record template

Create a task report with this minimum header:

```markdown
# <task and date>

## Provenance
- Host/GPU:
- Image digest:
- AITER/FlyDSL/MORI/framework commits:
- Dirty state and imported module paths:
- Python/Torch/HIP:
- Cache/AOT mode:

## Workload
- Model geometry and quant path:
- Rank tokens / MTPR / route / seed:
- Exact command:
- Raw logs/traces:

## Correctness
- Eager/reference error:
- CUDA Graph replay:
- Stress/variable-rank/AOT:

## Performance
| Shape | Base rank max | Candidate rank max | Delta | Config |
|---|---:|---:|---:|---|

## Root cause or conclusion
<evidence, limitations and remaining work>
```

If a new artifact supersedes an old one, keep the old row but mark the scope
and link the successor. Silent replacement erases the history needed to debug
regressions.
