# FlyDSL tests

Pytest configuration lives in [`pytest.ini`](pytest.ini) in this directory. Run pytest from the **repository root** (or pass `-c tests/pytest.ini`) so this file is picked up.

## Test tiering

The project uses a **layered model** so CI and contributors can select tests by dependency (CPU-only vs MLIR with ROCDL vs real GPU). The full specification is [**RFC : Test tiering and multi-backend CI matrix**](https://github.com/ROCm/FlyDSL/issues/275).

| Tier | Meaning |
|------|---------|
| **L0** | Backend-agnostic: no `FLYDSL_COMPILE_BACKEND` / device-runtime assumption; no vendor target dialect (`rocdl`, …). |
| **L1a** | Compile-tier, **no** vendor target dialect; portable Fly + upstream dialects only. |
| **L1b** | Compile-tier with **target-specific** lowering (e.g. Fly→ROCDL); still **no** GPU execution for correctness. |
| **L2** | Device-tier: needs GPU, driver, and runtime (often PyTorch) for launch and checks. |

**Pytest markers** (registered in `pytest.ini`) mirror these tiers:

| Marker | Typical tier |
|--------|----------------|
| `l0_backend_agnostic` | L0 |
| `l1a_compile_no_target_dialect` | L1a |
| `l1b_target_dialect` | L1b |
| `l2_device` | L2 |
| `rocm_lower` | Use **with** `l1b_target_dialect` or `l2_device` when the test assumes the ROCDL stack. |

**Legacy:** `large_shape` — used for slow/large kernel shapes; `scripts/run_tests.sh` skips it unless `RUN_TESTS_FULL=1`.

`scripts/run_tests.sh` always excludes `multi_gpu` and `benchmark` tests. Those
tests have dedicated CI jobs/steps and running them in the broad correctness
suite duplicates work (and can accidentally consume every GPU on a multi-GPU
host). CI can still opt into the full suite, including `large_shape` tests, with
`RUN_TESTS_FULL=1`; the exclusions apply in both modes.

### Rollout status

First-pass annotations now cover `tests/unit` and `tests/kernels` for clearly classified files (L0/L1a/L1b/L2).

Current high-traffic mapping:

- `tests/kernels/*.py`: `l2_device` + `rocm_lower`
- `tests/unit/*`: mixed by file (`l0_backend_agnostic`, `l1a_compile_no_target_dialect`, `l1b_target_dialect + rocm_lower`, `l2_device + rocm_lower`)
- `tests/mlir/Conversion/*.mlir`: treated as L1b + ROCm-lowering coverage (selected by FileCheck runner, not pytest markers)
- `tests/mlir/LayoutAlgebra/*.mlir`: treated as L1a compile-tier coverage where applicable (FileCheck, not pytest markers)

## Environment variables (source of truth: `env.py`)

Use the same names as [`python/flydsl/utils/env.py`](../python/flydsl/utils/env.py). Do not introduce alternate spellings in scripts or docs.

| Purpose | Variable |
|---------|----------|
| Compile backend id | `FLYDSL_COMPILE_BACKEND` (default `rocm`) |
| Override GPU arch for compile | `ARCH` |
| Compile without execution | `COMPILE_ONLY` |
| JIT cache directory | `FLYDSL_RUNTIME_CACHE_DIR` |
| Enable/disable JIT disk cache | `FLYDSL_RUNTIME_ENABLE_CACHE` (`0` / `false` to disable; in-memory cache remains active) |
| Force autotuning search | `FLYDSL_AUTOTUNE` (`1` / `true` to ignore the heuristic/cached best) |
| Autotune result cache directory | `FLYDSL_AUTOTUNE_CACHE_DIR` |
| IR dump | `FLYDSL_DUMP_IR`, `FLYDSL_DUMP_DIR` |
| Device runtime kind | `FLYDSL_RUNTIME_KIND` |
| ROCm arch hints (detection helpers) | `FLYDSL_GPU_ARCH`, `HSA_OVERRIDE_GFX_VERSION` |

Session-level pytest options are supported in `tests/conftest.py`:

- `--flydsl-compile-backend` -> sets `FLYDSL_COMPILE_BACKEND`
- `--flydsl-compile-arch` -> sets `ARCH`

When these options are unset, default environment behavior remains unchanged.

## Running pytest

From the repo root after a successful build / `pip install -e .`:

```bash
export PYTHONPATH="${PWD}/build-fly/python_packages:${PWD}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PWD}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH}"
```

Examples:

```bash
# Default PR correctness suite (matches scripts/run_tests.sh)
python3 -m pytest tests/kernels/ tests/language/ tests/unit/ tests/system/ tests/extension/ tests/python/examples/ \
  -m "not large_shape and not multi_gpu and not benchmark" -v

# Exhaustive single-GPU correctness suite (RUN_TESTS_FULL=1)
python3 -m pytest tests/kernels/ tests/language/ tests/unit/ tests/system/ tests/extension/ tests/python/examples/ \
  -m "not multi_gpu and not benchmark" -v

# When tests are annotated — examples (forward-looking)
# python3 -m pytest tests/ -m "l0_backend_agnostic or l1a_compile_no_target_dialect" -v
# python3 -m pytest tests/ -m "l2_device" -v
```

The JIT disk cache auto-invalidates when kernel source or closure values change. Only disable it when modifying C++ passes or non-closure helper functions:

```bash
export FLYDSL_RUNTIME_ENABLE_CACHE=0  # or: rm -rf ~/.flydsl/cache
```

Cooperative tests use a representative parameter matrix by default so each
axis value is covered without compiling its full Cartesian product. Set
`FLYDSL_COOP_TESTS_FULL=1` to run every cooperative-test combination; this is
also when the shared default-device fixture covers both CPU and CUDA defaults.
This switch is independent of `RUN_TESTS_FULL`, which continues to control
`large_shape`.
See [coop test coverage](extension/coop/README.md) for the matrix selection,
cross-target compilation and stress-input policies.

## MLIR FileCheck tests

`tests/mlir/**/*.mlir` checks are driven by **`scripts/run_tests.sh`** (FileCheck + `fly-opt`), not by pytest. Tiering for those may be documented in parallel in this README as the RFC rollout continues; see RFC open questions.

## Downstream consumer coverage

vLLM and ATOM author their own FlyDSL kernels. Those kernels are not in this
repository, so a change here can break them while this repo's CI stays green.
Three different mechanisms cover that risk, and which one applies depends on
where the kernel's source lives.

| Where the kernel lives | Covered by | Example |
|---|---|---|
| This repo (`kernels/`) | ordinary tests under `tests/kernels/` | `moe_2stage_a16wmix` -> `tests/kernels/test_moe_gemm.py` |
| aiter (`aiter.ops.flydsl.*`) | aiter's own `op_tests/`, run from `.github/workflows/flydsl.yaml` | GDN prefill -> `op_tests/test_gated_delta_rule.py` |
| vLLM itself | `tests/unit/test_downstream_api_contract.py` | TurboQuant / UltraQuant decode, DeepSeek-V4.1 mono |

### `tests/unit/test_downstream_api_contract.py`

vLLM's own kernels reach past the documented DSL into compiler, runtime and raw
MLIR-binding internals (`flydsl.utils.smem_allocator`, `flydsl.expr.typing.
as_ir_value`, `flydsl.compiler.ast_rewriter.ASTRewriter`, `flydsl._mlir.ir` and
`_mlir.dialects.{scf,llvm}`, ...). That file freezes exactly those symbols and
signatures.

**If you rename, move, or re-sign anything in `python/flydsl/` and that test goes
red, the change is about to break vLLM.** It is not a veto: update the downstream
repository and that file in the same change. The file lists the downstream call
sites it mirrors, so the blast radius is visible without leaving this repo.

`flydsl.utils.smem_allocator` is the case to watch. It is marked legacy and warns
on use, and nothing under `kernels/` or `tests/` imports it any more -- but three
vLLM paged-decode kernels do. Removing it per the deprecation plan would fail no
other test here.

### Coverage this repo does not have

- **The vLLM-side kernels themselves** (`tq_decode`, `tq_decode_gqa6`,
  `ultraquant_decode_hd256`, the DeepSeek-V4.1 `mono` layer). Their source is in
  vLLM; only the API contract above protects them from this side.
- **`flydsl_fp8_mqa_logits`**, and the aiter MoE `op_tests`. See the
  `[CI] Run aiter's FlyDSL GDN op_tests in the PR gate` commit message for why
  each one is not wired yet.
- **Multi-GPU paths by default.** `MegaMoEV2`, allreduce, monokernel GLM-5/Kimi-K3
  and dispatch/combine at 8 ranks are all `multi_gpu`, which `scripts/run_tests.sh`
  deselects and which CI runs only on a label-triggered job. The single-GPU and
  2-GPU gates under them (`tests/unit/test_monokernel_ipc_host.py`,
  `tests/kernels/test_dispatch_combine_2gpu.py`) exist precisely because of that.

### Gotcha 1: which tree you are actually testing

`import flydsl` resolves to `build-fly/python_packages/flydsl/`, the staged build
output -- not `python/flydsl/`. Editing the source tree and re-running pytest
without rebuilding tests the **old** code, silently. `kernels/` is imported from
the repo root and does not have this problem.

To check that a test really fails when the thing it guards is broken, build an
overlay rather than editing the staged copy:

```bash
RT=/tmp/fly-overlay; rm -rf $RT; mkdir -p $RT/flydsl
cp -r python/flydsl/. $RT/flydsl/
ln -s "${PWD}/build-fly/python_packages/flydsl/_mlir" $RT/flydsl/_mlir
export PYTHONPATH="$RT:${PWD}/build-fly/python_packages:${PWD}"
# now break something under $RT/flydsl and re-run the test
```

### Gotcha 2: a warm JIT cache makes IR-inspecting tests fail

Tests that assert on generated IR -- `tests/unit/test_kernel_known_block_size.py`,
`tests/unit/test_fastmath_context.py`, `tests/kernels/test_softmax_autotune.py`,
`tests/extension/random/test_random.py` -- read IR that is only produced on a cache
miss. Run the suite twice in a row locally and the second run fails ~19 of them
against a cache the first run filled, with no source change in between:

```
cold cache:  2 failed, 6020 passed in 1584s
warm cache: 22 failed, 6000 passed in  217s   # same commit
```

CI never sees this, since each job starts in a fresh container. Locally, clear the
cache (or disable it) before trusting a full-suite result:

```bash
rm -rf ~/.flydsl/cache                # or: export FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d)
export FLYDSL_RUNTIME_ENABLE_CACHE=0  # alternative: skip the disk cache entirely
```

### Gotcha 3: a failed HIP call can take down the rest of the session

Some HIP failures leave the context unusable for the whole process, not just the
call that failed. `hipIpcOpenMemHandle` on a handle the same process exported is
one: it returns `hipErrorInvalidContext`, and every later
`torch.*(device="cuda")` in that pytest session then raises "HIP error: invalid
device context". A test doing this passes alone and in its own file, and only
shows up as an unrelated test failing later in the full suite.

If a test you did not touch starts failing on a plain CUDA allocation, suspect an
earlier test in the same session rather than your own change, and bisect by running
the suspect file first. `tests/kernels/test_moe_gemm.py` documents the same hazard
for the memory-unsafe mxfp4 kernels, which is why they are marked `run=False`
rather than merely `xfail`.
