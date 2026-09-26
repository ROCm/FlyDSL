# Testing and benchmarking guide

FlyDSL has separate checks for Python semantics, MLIR lowering, GPU
correctness, multi-GPU behavior, and performance. Choose the narrowest test
that proves a change, then run the repository wrapper before submitting a
cross-cutting change.

## 1. Test layout

| Area | Location | What it covers |
|---|---|---|
| Language | `tests/language/` | Numeric promotion/folding, composites, storage, and DSL language behavior |
| Extensions | `tests/extension/` | Cooperative algorithms, random generation, dispatch, and validation |
| Unit | `tests/unit/` | Compiler/runtime helpers that can often run without a GPU |
| System | `tests/system/` | Cross-component compiler and runtime behavior |
| GPU kernels | `tests/kernels/` | End-to-end compilation, launch, correctness, and benchmark entry points |
| Python examples | `tests/python/examples/` | AOT/JIT workflows that are also executable examples |
| MLIR | `tests/mlir/` | `fly-opt` + FileCheck coverage for layout algebra, conversion, and transforms |
| Standalone examples | `examples/` | Runnable teaching programs, checked separately by the test wrapper |

Pytest configuration and the detailed marker policy live in
`tests/pytest.ini` and `tests/README.md`.

## 2. Build and environment

Build FlyDSL before running tests:

```bash
bash scripts/build.sh -j64
python -m pip install -e .
```

The repository runners set these paths automatically. For a manual session:

```bash
export PYTHONPATH="${PWD}/build-fly/python_packages:${PWD}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PWD}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH}"
```

Set `FLY_BUILD_DIR` if the build is not under `build-fly/`. PyTorch-based GPU
tests require a ROCm build of PyTorch compatible with the machine's ROCm
installation.

## 3. Repository test runner

The supported broad correctness command is:

```bash
bash scripts/run_tests.sh
```

It performs three stages in order:

1. pytest over `tests/kernels`, `tests/language`, `tests/unit`,
   `tests/system`, `tests/extension`, and `tests/python/examples`;
2. standalone scripts under `examples/` and `examples/extension/*/`, filtered
   by `tests/arch_compat.py`;
3. every `// RUN:` line in `tests/mlir/**/*.mlir`, using `fly-opt` and
   FileCheck.

The default pytest selection excludes `large_shape`, `multi_gpu`, and
`benchmark`. Set `RUN_TESTS_FULL=1` to include `large_shape`; multi-GPU and
benchmark tests remain separate. The wrapper also disables broad-suite
autotuning for determinism and enables the opt-in quantization correctness
coverage.

FileCheck is allowed to be absent in a local run, but the script reports the
entire MLIR stage as skipped. Set `FLYDSL_REQUIRE_FILECHECK=1` when a missing
FileCheck must fail the run (as CI should).

## 4. Running focused tests

Run pytest from the repository root so `tests/pytest.ini` is discovered:

```bash
python -m pytest tests/language/test_numeric.py -q
python -m pytest tests/extension/coop -q
python -m pytest tests/kernels/test_vec_add.py -q
python -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape" -q
```

Common markers:

| Marker | Meaning |
|---|---|
| `l0_backend_agnostic` | No backend/runtime assumption |
| `l1a_compile_no_target_dialect` | Compile-only, portable Fly/upstream dialects |
| `l1b_target_dialect` | Compile-only but target-specific lowering |
| `l2_device` | Requires a real GPU/runtime |
| `rocm_lower` | Assumes ROCDL/ROCm lowering; combine with L1b or L2 |
| `large_shape` | Slow or memory-heavy parameter cases |
| `multi_gpu` | Requires multiple visible GPUs/process coordination |
| `benchmark` | Performance-only case, excluded from correctness runs |

The rollout status and current directory-to-tier mapping are maintained in
`tests/README.md`.

## 5. Architecture filtering

`tests/arch_compat.py` is the central pre-execution map for whole test files and
standalone examples with fixed architecture requirements. Individual tests and
kernel builders can impose stricter constraints for an instruction, dtype, or
shape.

Useful controls:

```bash
# Select a visible GPU explicitly.
HIP_VISIBLE_DEVICES=1 python -m pytest tests/kernels/test_softmax.py -q

# Cross-compile or compile-test for a named architecture.
ARCH=gfx950 COMPILE_ONLY=1 python my_compile_test.py

# Equivalent pytest options used by the test configuration.
python -m pytest tests/unit \
  --flydsl-compile-backend rocm \
  --flydsl-compile-arch gfx950
```

`scripts/run_tests.sh` chooses the visible GPU with the most free memory when
multiple devices exist and `HIP_VISIBLE_DEVICES` is unset.

## 6. MLIR/FileCheck tests

MLIR tests are ordinary `.mlir` files under:

- `tests/mlir/LayoutAlgebra/`
- `tests/mlir/Conversion/`
- `tests/mlir/Transforms/`

Each file can contain multiple `// RUN:` lines. The repository wrapper executes
all of them, expanding `%fly-opt`, `%FileCheck`, and `%s` substitutions. To run
one case manually, copy its `RUN` command and substitute the built tools, for
example:

```bash
build-fly/bin/fly-opt --fly-canonicalize tests/mlir/LayoutAlgebra/my_test.mlir \
  | FileCheck tests/mlir/LayoutAlgebra/my_test.mlir
```

Add a focused MLIR test for a new operation, verifier rule, rewrite, or lowering
pattern even when a Python test also exercises it.

## 7. Writing a GPU correctness test

A useful kernel test contains:

1. a small deterministic shape that compiles quickly;
2. boundary shapes for tails, alignment, empty/partial work, or split paths;
3. a reference implementation independent of the kernel;
4. tolerances justified by dtype and accumulation order;
5. architecture/optional-dependency skips with a clear reason;
6. a benchmark path separated from correctness assertions.

Minimal structure:

```python
import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def my_kernel(src: fx.Tensor, dst: fx.Tensor, n: fx.Int32):
    i = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if i < n:
        dst[i] = src[i]


@flyc.jit
def launch(src: fx.Tensor, dst: fx.Tensor, n: fx.Int32):
    my_kernel(src, dst, n).launch(
        grid=((n + 255) // 256, 1, 1),
        block=(256, 1, 1),
    )


@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_my_kernel():
    src = torch.randn(1000, device="cuda")
    dst = torch.empty_like(src)
    launch(src, dst, src.numel())
    torch.testing.assert_close(dst, src)
```

For project-specific comparison/timing helpers, inspect `tests/test_common.py`,
`tests/utils.py`, and `tests/kernels/benchmark_common.py`. Prefer standard
`torch.testing.assert_close` when the special mismatch-ratio behavior is not
needed.

## 8. Benchmark runner

The repository benchmark entry point is:

```bash
bash scripts/run_benchmark.sh
```

With no operation arguments it runs the configured softmax, softmax backward,
RMSNorm, FlashAttention, MLA decode, GEMM, and MoE groups. Select a subset with
positional names or `--only`:

```bash
bash scripts/run_benchmark.sh softmax rmsnorm
bash scripts/run_benchmark.sh --only flash_attn,gemm
bash scripts/run_benchmark.sh --list
bash scripts/run_benchmark.sh --output-csv /tmp/flydsl-bench.csv
```

Supported group names are `softmax`, `softmax_bwd`, `rmsnorm`, `flash_attn`,
`mla`, `gemm`, and `moe`. The GEMM group includes the enabled preshuffle,
A16W16, FP8 row-scale, and FP4 variants for the detected architecture.

The runner:

- configures build/runtime paths;
- detects CDNA/RDNA capabilities and skips unsupported groups;
- selects a GPU by free memory when appropriate;
- writes per-command logs under `${BENCH_LOG_DIR:-/tmp/flydsl_bench}`;
- prints normalized bandwidth/throughput rows;
- optionally writes CSV through `BENCH_OUTPUT_CSV` or `--output-csv`;
- distinguishes an explicit pytest skip from a real benchmark failure.

Shape lists near the top of `scripts/run_benchmark.sh` are the source of truth
for dashboard/default coverage. Do not copy them into documentation, where they
would become stale.

## 9. Reliable performance measurement

Before comparing kernels:

- run correctness with the exact dtype, layout, and edge/tail conditions;
- warm compilation/caches separately from device timing;
- use the same stream and synchronization policy for both implementations;
- report shape, dtype, strides/layout, GPU, ROCm version, FlyDSL revision, and
  relevant environment overrides;
- collect enough samples to separate signal from clock/system noise;
- preserve a baseline log or CSV with the change.

`flydsl.autotune.do_bench` and repository benchmark helpers use GPU events and
batched launch windows. The measured callable must enqueue asynchronous work
and must not synchronize internally.

For instruction-level diagnosis, use the rocprofv3 ATT/PMC workflow in
[Kernel tuning](kernel_tuning_guide.md). For regressions with a known good
revision, bisect correctness first and performance second.

## 10. Debugging compilation and cache state

Dump the compiler pipeline:

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python my_test.py
```

Useful core controls include:

| Variable | Purpose |
|---|---|
| `FLYDSL_DUMP_IR` / `FLYDSL_DUMP_DIR` | Numbered IR and ISA dump output |
| `FLYDSL_DEBUG_PRINT_AFTER_ALL` | Print IR after every pass |
| `FLYDSL_DEBUG_SHOW_STACKTRACE` | Show internal frames for compile failures |
| `FLYDSL_RUNTIME_CACHE_DIR` | Select persistent cache location |
| `FLYDSL_RUNTIME_ENABLE_CACHE=0` | Disable persistent cache reads/writes |
| `FLYDSL_RUNTIME_RUN_ONLY=1` | Require an existing cached artifact |
| `FLYDSL_AUTOTUNE=1` | Force a new autotune search |

The cache normally invalidates when traced source, referenced dependencies,
compile hints, target, or toolchain change. Disable or clear it only when
investigating state outside that fingerprint, such as newly built C++ passes.

## 11. Documentation and repository checks

Run the non-style repository checks with:

```bash
python3 scripts/check_repo.py
python3 scripts/check_repo.py --list
```

Build documentation with warnings treated as errors:

```bash
python -m pip install -r docs/requirements.txt
make -C docs clean html SPHINXOPTS="-W --keep-going"
```

The documentation/API drift check validates published docs as well as the
agent-facing instructions, including referenced repository paths and `fx.*` /
`rocdl.*` spellings.
