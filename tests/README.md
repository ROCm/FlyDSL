# FlyDSL tests

Pytest configuration lives in [`pytest.ini`](pytest.ini) in this directory. Run pytest from the **repository root** (or pass `-c tests/pytest.ini`) so this file is picked up.

## Test tiering (RFC 0003)

The project uses a **layered model** so CI and contributors can select tests by dependency (CPU-only vs MLIR with ROCDL vs real GPU). The full specification is [**RFC 0003: Test tiering and multi-backend CI matrix**](../rfc/0003-test-tiering-and-backend-matrix.md).

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
| Enable/disable JIT cache | `FLYDSL_RUNTIME_ENABLE_CACHE` (`0` / `false` to disable) |
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
# Default: full pytest areas (same idea as scripts/run_tests.sh pytest step)
python3 -m pytest tests/kernels/ tests/unit/ tests/python/examples/ -v

# Exclude large shapes (matches run_tests.sh when RUN_TESTS_FULL is unset)
python3 -m pytest tests/kernels/ tests/unit/ tests/python/examples/ -m "not large_shape" -v

# When tests are annotated — examples (forward-looking)
# python3 -m pytest tests/ -m "l0_backend_agnostic or l1a_compile_no_target_dialect" -v
# python3 -m pytest tests/ -m "l2_device" -v
```

Disable JIT cache while iterating:

```bash
export FLYDSL_RUNTIME_ENABLE_CACHE=0
```

## MLIR FileCheck tests

`tests/mlir/**/*.mlir` checks are driven by **`scripts/run_tests.sh`** (FileCheck + `fly-opt`), not by pytest. Tiering for those may be documented in parallel in this README as the RFC rollout continues; see RFC open questions.

