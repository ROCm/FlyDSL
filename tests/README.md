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
| AOT worker-process limit | `FLYDSL_AOT_WORKERS` (positive: explicit limit; unset, empty, zero, or negative: automatic CPU/memory limit) |
| AOT automatic memory cap | `FLYDSL_AOT_MEM_PER_WORKER_GB` (default `2.0`; non-positive disables) |
| AOT per-job timeout | `FLYDSL_AOT_TIMEOUT` (default `1200` seconds; non-positive disables) |
| AOT abnormal-exit retries | `FLYDSL_AOT_MAX_RETRIES` (default `2`) |

`flydsl.utils.parallel.run_parallel_jobs` uses the Linux `fork` multiprocessing
context and is intended for compile-only AOT jobs. It is not a device-runtime
executor. Its automatic memory cap requires the optional `psutil` package and
logs a warning before falling back to the CPU-based worker limit when the
optional package is unavailable or memory cannot be queried. A worker killed by
`SIGKILL` (`exitcode=-9`; possible OOM) halves the worker limit at most once per
launch wave before retrying. Every healthy completion after a backoff helps
restore the limit additively, including work launched before that backoff.
Timeouts do not change the global worker limit. No backoff is applied when
retries are disabled, and retries are appended behind jobs that have not
started yet.
Progress and final summary lines report terminal failure counts separately from
the number of jobs that have finished. Scheduler messages use the FlyDSL logger;
set `FLYDSL_DEBUG_LOG_TO_CONSOLE=1` and `FLYDSL_DEBUG_LOG_LEVEL=INFO` to emit
progress and summaries to the console.

Failed results retain `compile_time=None` and include a machine-readable
`failure` mapping. Its `kind` distinguishes `compile_error`,
`worker_exception`, `worker_crash`, `possible_oom`, `timeout`,
and `invalid_result`; `reason` and `attempts` are always included when known,
while process failures also carry `exitcode`. Uncaught Python exceptions are
not retried and carry bounded `reason` and `traceback` strings. A valid atomic
result remains authoritative if the process is killed during teardown.

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
# Default: full pytest areas (same idea as scripts/run_tests.sh pytest step)
python3 -m pytest tests/kernels/ tests/language/ tests/unit/ tests/python/examples/ -v

# Exclude large shapes (matches run_tests.sh when RUN_TESTS_FULL is unset)
python3 -m pytest tests/kernels/ tests/language/ tests/unit/ tests/python/examples/ -m "not large_shape" -v

# When tests are annotated — examples (forward-looking)
# python3 -m pytest tests/ -m "l0_backend_agnostic or l1a_compile_no_target_dialect" -v
# python3 -m pytest tests/ -m "l2_device" -v
```

The JIT disk cache auto-invalidates when kernel source or closure values change. Only disable it when modifying C++ passes or non-closure helper functions:

```bash
export FLYDSL_RUNTIME_ENABLE_CACHE=0  # or: rm -rf ~/.flydsl/cache
```

## MLIR FileCheck tests

`tests/mlir/**/*.mlir` checks are driven by **`scripts/run_tests.sh`** (FileCheck + `fly-opt`), not by pytest. Tiering for those may be documented in parallel in this README as the RFC rollout continues; see RFC open questions.
