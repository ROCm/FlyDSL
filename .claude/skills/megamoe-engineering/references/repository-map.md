# Repository and source map

MegaMoE is not implemented in one repository. Resolve the active stack before
editing or benchmarking.

## Repository roles

| Repository | Responsibility |
|---|---|
| AITER | Production `MegaMoEV2`, config selection, AOT jobs, MORI integration, operator tests and benchmark. |
| FlyDSL | DSL/compiler/runtime, extension collectives, buffer tensor API, JIT/AOT/preload behavior and kernel-authoring standards. |
| FlyDSL_universe | Customer/experimental kernel tree, including ByteDance native and SmoothQuant variants. It can intentionally diverge from AITER. |
| MORI | Symmetric-memory allocation, P2P pointers, barriers and the comparison EP dispatch/combine implementation. |
| ATOM | Production serving integration, CUDA Graph, EPLB, continuous batching and end-to-end benchmark. |
| SGLang | Independent serving integration and fixed-length/AgentX validation. |

Do not infer that a similarly named MegaMoE in a framework uses AITER/FlyDSL.
Confirm the imported module and environment switch in the exact framework
commit.

## AITER production map

Locate the current tree with:

```bash
rg --files aiter/ops/flydsl/kernels/mega_moe aiter/aot/flydsl \
  op_tests/multigpu_tests op_tests/flydsl_tests | sort
```

As of the #5346 snapshot, the principal files are:

```text
aiter/ops/flydsl/kernels/mega_moe/
  __init__.py
  mega_moe_v2.py
  mega_moe_config.py
  mega_moe_prepare.py
  dispatch.py
  mega_moe_stage1.py
  gemm1.py
  gemm_util.py
  mega_moe_stage2.py
  mega_moe_stage2_aligned_pair.py
  gemm2.py
  quant.py

aiter/aot/flydsl/mega_moe.py
op_tests/multigpu_tests/test_mega_moe_v2.py
op_tests/multigpu_tests/bench_mega_moe_v2.py
op_tests/flydsl_tests/test_mega_moe_config.py
```

Related code can live outside the directory:

- dispatch/combine host and kernel implementation;
- `communication_ops_utils.py` for system/agent atomics and fences;
- `tensor_shim.py` for `ptr_buf_tensor`, compiled launch and preload helpers;
- AOT registry/common driver;
- fused-MoE CSV selectors used by the MORI comparison;
- serving hooks in ATOM/SGLang.

Search by public type and protocol names, not a memorized path:

```bash
rg -n "MegaMoEV2|DispatchSlot|build_mega_moe_bundle_plan|preload_aot_bundles"
rg -n "MORI_EP_LAUNCH_CONFIG_MODE|AITER_MEGA_MOE|FLYDSL_RUNTIME_RUN_ONLY"
```

## FlyDSL map

MegaMoE consumes current FlyDSL APIs. Before carrying code between commits,
inspect:

```text
python/flydsl/compiler/
python/flydsl/expr/
python/flydsl/extension/
kernels/common/
.claude/skills/kernel-code-cleanup/SKILL.md
.claude/skills/kernel-trace-analysis/SKILL.md
.claude/skills/capture-kernel-trace/SKILL.md
```

The #1098 FlyDSL API migration replaced private warp scan/reduce helpers with
the public cooperative extension. Current forms include:

```python
fx.coop.warp_reduce(value, fx.ReductionOp.MAX, width=64)
fx.coop.warp_scan_with_aggregate(value, fx.ReductionOp.ADD, width=64)
fx.coop.warp_exclusive_scan(value, fx.ReductionOp.ADD, width=64)
```

Preserve invalid-lane zeroing, chunk carry and aggregate semantics when moving
MegaMoE. A textual helper rename is not proof of equivalence.

Use the current layout/buffer surface. `buffer_ops` and raw MLIR compatibility
can be stale even when an old kernel still compiles against a private build.

## ByteDance FlyDSL-universe map

The customer branch has historically lived at:

```text
repository: /home/ghu/FlyDSL_universe
branch:     ghu/mega_moe_v1_copy
kernel:     kernels/mega_moe/
tests:      tests/kernels/test_mega_moe_v2.py
            tests/kernels/test_mega_moe_int8.py
unit:       tests/unit/test_mega_moe_config.py
```

These are discovery hints, not permission to modify that checkout. It has often
contained valuable uncommitted work. Fetch the requested remote head and create
an independent worktree. Record both the remote commit and the original dirty
status without changing it.

Important runner candidates on the known machines include:

```text
/home/ghu/opus_verify_46/run8.sh
/home/ghu/opus_verify_47/run_universe_a8w4m13_8.sh
/data/ghu/run_universe_a8w4m13_8.sh
/data/ghu/run_m13_mx_a8w4_prefill_matrix_20260909.sh
```

Inspect the runner before each use. Test flags and import paths evolve.

## Framework map

For ATOM, locate the current MegaMoE backend rather than copying an old server
command:

```bash
rg -n "moe-backend.*mega|MegaMoE|AITER_MEGA_MOE|MTPR|enable-tbo" .
```

For SGLang, distinguish AMD FlyDSL MegaMoE from the unrelated CUDA/DeepGEMM
backend with the same name. Search the exact integration switch and public
wrapper. Historical switches included `SGLANG_AMD_USE_FLYDSL_MEGA_MOE` and
`SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR`; verify current spelling in source.

## Configuration sources

MegaMoE production geometry is primarily rule-selected in
`mega_moe_config.py`, not an offline JSON tuner. The baseline MORI/fused-MoE
comparison can use AITER model CSVs. Always print the resolved config and file
path from the benchmark; otherwise a comparison can silently use a generic or
wrong-model entry.

When adding a model, inspect all of:

1. network/test shape registry;
2. Stage1 selector;
3. Stage2 selector;
4. terminal combine geometry table/CSV;
5. AOT deployment profiles and cache identity;
6. framework-side shared-expert and activation semantics.

## Branch and worktree discipline

Use a clean worktree instead of stashing someone else's work:

```bash
git fetch origin main
git worktree add -b <task-branch> <new-path> origin/main
git -C <new-path> status --short --branch
```

Before claiming a PR is "main plus this patch":

```bash
git merge-base --is-ancestor origin/main HEAD
git diff --check origin/main..HEAD
git diff --name-only origin/main..HEAD
```

If unrelated attention, TopK, GEMM or framework files appear, do not use the
branch for a controlled MegaMoE comparison.

For two large MegaMoE PRs, run a real merge simulation and review every
conflicting protocol field. Never resolve Stage1/Stage2/config/AOT conflicts by
blindly taking `ours` or `theirs`.

## Provenance that must accompany a result

Record:

```text
date/time and timezone
host GPU model and count
image name and immutable digest
repository full commits and dirty status
actual imported module paths
Python, PyTorch, HIP/ROCm, FlyDSL and MORI versions
FLYDSL build/library path
JIT cache mode and directory
MTPR, tokens/rank, rank-token vector, route distribution and random seed
exact command, raw log directory and trace directory
```

Directory names such as `main`, `final`, `optimized` or `delivered` are not
provenance.
