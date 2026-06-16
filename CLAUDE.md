# FlyDSL Project Guide

FlyDSL (Flexible Layout Python DSL) is a Python DSL and MLIR compiler stack for
authoring high-performance AMD GPU kernels with explicit layout algebra, tiling,
copy atoms, and MMA atoms. The stack targets ROCm/HIP through the Fly and
FlyROCDL dialects, lowering to ROCDL/HSACO.

## Agent Operating Guidelines

These guidelines reduce common LLM coding mistakes. They bias toward caution
over speed; use judgment for trivial tasks.

### Think Before Coding

- Do not assume silently. State assumptions when they affect the implementation.
- If multiple interpretations exist, present them instead of picking one without explanation.
- If a simpler approach exists, say so and push back when the requested path seems overbuilt.
- If something is unclear, stop, name the confusion, and ask.

### Simplicity First

- Write the minimum code that solves the requested problem.
- Do not add features, abstractions, configurability, or fallback behavior that was not requested.
- Avoid abstractions for single-use code. If a solution grows far larger than necessary, simplify it.
- Prefer invariants and direct control flow over speculative defensive code.

### Surgical Changes

- Touch only what the task requires. Do not refactor or reformat adjacent code opportunistically.
- Match existing style even when a different style might be preferable.
- If unrelated dead code or cleanup is noticed, mention it rather than deleting it.
- Remove imports, variables, or helpers made unused by your own changes, but do not remove pre-existing dead code unless asked.
- Every changed line should trace directly to the user's request.

### Goal-Driven Execution

- Turn tasks into verifiable goals before implementation.
- For bug fixes, reproduce or identify the failing behavior, then verify the fix.
- For refactors, preserve behavior and run focused before/after checks when practical.
- For multi-step tasks, state a brief plan with the verification for each step.
- Keep looping until the stated success criteria are met or a real blocker is surfaced.

## Repository Layout

```text
FlyDSL/
├── .claude/skills/                 # Project-local Claude Code skills (kernel authoring, profiling, build); git-tracked
├── python/
│   ├── flydsl/                    # Python DSL core
│   │   ├── expr/                  # DSL expression API; direct children are TARGET-NEUTRAL (typing, primitive, gpu, derived, struct, numeric, math, vector, arith, meta, extern; + utils/)
│   │   │   └── rocdl/             # Target-specific ROCDL package (cdna4, cluster, inline_asm, tdm_ops, universal); lazy-loaded via __init__'s _LAZY_MODULES
│   │   ├── compiler/              # @flyc.kernel / @flyc.jit, AST rewriting, JIT cache, backends
│   │   ├── runtime/               # Device runtime and GPU arch detection
│   │   ├── utils/                 # EnvManager, SmemAllocator (legacy), logger
│   │   │                          #   newer kernels use SharedAllocator in expr/gpu.py
│   │   └── autotune.py            # Autotuner (@autotune, Config)
│   └── mlir_flydsl/               # MLIR Python binding package source
├── include/flydsl/                # C++ TableGen headers for Fly / FlyROCDL dialects and passes
├── lib/                           # C++ dialect implementation, conversions, runtime wrappers, Python bindings
│   └── Dialect/FlyROCDL/{CDNA3,CDNA4,GFX11,GFX1250}/  # Per-subtarget atom lowering: MmaAtom (MFMA on CDNA3/4, WMMA on GFX11/1250) + CopyAtom (Buffer/LDS, CDNA3/4 only)
├── tools/                         # fly-opt
├── kernels/                       # Production kernels, importable as kernels.*
├── tests/
│   ├── kernels/                   # GPU correctness / benchmark harnesses
│   ├── unit/                      # Python compiler, runtime, and layout tests
│   ├── system/                    # Cross-cutting compile/system tests
│   ├── mlir/                      # FileCheck tests driven by scripts/run_tests.sh
│   └── python/examples/           # AOT compile/cache pytest tests (aot_example.py)
├── examples/                      # 01-vectorAdd, 02-tiledCopy, 03-tiledMma, 04-preshuffle_gemm
├── scripts/                       # build, test, benchmark, wheel, debug helper scripts
├── docs/                          # Sphinx documentation source
├── thirdparty/                    # Vendored dlpack and tvm-ffi
└── build-fly/                     # Generated build output; do not edit
```

## Documentation Map

| Topic | Doc | Notes |
|---|---|---|
| Architecture & compiler pipeline | [`docs/architecture_guide.md`](docs/architecture_guide.md) | Project structure, AST tracing, MLIR pass pipeline, JIT/runtime |
| Layout algebra | [`docs/layout_system_guide.md`](docs/layout_system_guide.md) | Shape/Stride/Layout/Coord APIs, products, divides, coordinate mapping |
| CuTe layout reference | [`docs/cute_layout_algebra_guide.md`](docs/cute_layout_algebra_guide.md) | Mathematical background and FlyDSL mapping of CuTe concepts |
| Kernel authoring | [`docs/kernel_authoring_guide.md`](docs/kernel_authoring_guide.md) | `@flyc.kernel`, `@flyc.jit`, launch config, LDS, tiled copy/MMA |
| Pre-built kernels | [`docs/prebuilt_kernels_guide.md`](docs/prebuilt_kernels_guide.md) | Norm, Softmax, GEMM, MoE, attention, dtype/config notes |
| External bitcode integration | [`docs/extern_integration_guide.md`](docs/extern_integration_guide.md) | `ffi` + `link_extern`: plug pre-compiled LLVM bitcode into the JIT pipeline (`python/flydsl/expr/extern.py`, `compiler/extern_link.py`) |
| Testing & benchmarking | [`docs/testing_benchmarking_guide.md`](docs/testing_benchmarking_guide.md) | Test categories, benchmark harness, performance comparisons |
| Test tiering and env vars | [`tests/README.md`](tests/README.md) | L0/L1a/L1b/L2 markers, FileCheck flow, canonical env variable names |

Public docs are deployed from `.github/workflows/docs.yml` to
<https://rocm.github.io/FlyDSL>.

## Build & Test

```bash
bash scripts/build_llvm.sh -j64       # Build LLVM/MLIR once
bash scripts/build.sh -j64            # Build FlyDSL C++ + Python bindings
pip install -e .                      # Editable Python install

# If not relying on editable install paths:
export PYTHONPATH="${PWD}/build-fly/python_packages:${PWD}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PWD}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH}"

bash scripts/run_tests.sh             # Pytest + examples + MLIR FileCheck
RUN_TESTS_FULL=1 bash scripts/run_tests.sh  # Include large_shape tests; this is the CI invocation (flydsl.yaml, test-whl.yaml)
bash scripts/run_benchmark.sh         # Performance benchmarks (all ops)
bash scripts/run_benchmark.sh --only softmax,moe   # subset; --list to enumerate ops
bash scripts/run_benchmark.sh --output_csv /tmp/bench.csv   # emit CSV for diffing
python3 scripts/compare_benchmark.py base.csv cur.csv  # ratio report vs a baseline
```

Useful direct commands:

```bash
python3 -m pytest tests/kernels/ tests/unit/ tests/system/ tests/python/examples/ -m "not large_shape" -v
python3 -m pytest tests/kernels/test_pa.py -v
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 -m pytest tests/kernels/test_pa.py -v
```

`scripts/run_tests.sh` auto-selects the GPU with the most free VRAM when
`HIP_VISIBLE_DEVICES` is unset and sets `FLYDSL_RUN_QUANT=1`.

Reproduce the CI Python style gate locally before pushing:

```bash
bash scripts/check_python_style.sh            # black + ruff over committed diff vs origin/main
bash scripts/check_python_style.sh --fix      # auto-format the changed files
bash scripts/check_python_style.sh --fix --include-local  # also staged/untracked
```

This wraps `.github/scripts/check_python_style.sh`, the same checker run by the
`Check Python Code Style` job in `.github/workflows/pre-checks.yaml`.

## Environment Variables

Use names from `python/flydsl/utils/env.py`; do not introduce alternate spellings.

| Purpose | Variable |
|---|---|
| Compile backend | `FLYDSL_COMPILE_BACKEND` (default `rocm`) |
| Override compile arch | `ARCH` |
| Compile without execution | `COMPILE_ONLY` |
| JIT cache directory | `FLYDSL_RUNTIME_CACHE_DIR` |
| Enable/disable JIT disk cache | `FLYDSL_RUNTIME_ENABLE_CACHE` (`0` / `false` disables disk cache; in-memory cache remains) |
| AOT-cache-only execution | `FLYDSL_RUNTIME_RUN_ONLY` (`1` skips JIT; loads disk cache only, raises on cache miss; incompatible with `FLYDSL_DUMP_IR=1`) |
| External LLVM/MLIR codegen | `FLYDSL_COMPILE_LLVM_DIR` (install prefix; enables external-binary final codegen, part of the JIT cache key) |
| IR dumps | `FLYDSL_DUMP_IR`, `FLYDSL_DUMP_DIR` |
| Runtime kind | `FLYDSL_RUNTIME_KIND` |
| GPU arch hints | `FLYDSL_GPU_ARCH`, `HSA_OVERRIDE_GFX_VERSION` |
| Debug info / pass diagnostics | `FLYDSL_DEBUG_ENABLE_DEBUG_INFO`, `FLYDSL_DEBUG_PRINT_AFTER_ALL`, `FLYDSL_DEBUG_AST_DIFF` |

The JIT disk cache normally invalidates on kernel source and closure changes.
Disable it when debugging stale artifacts, changing C++ passes, or changing
helper code that is not part of the traced closure.

## Code Style

- **Python**: black line length 120; ruff checks `E`, `W`, `F`, and `I`. Config lives in `pyproject.toml`.
- **Imports**: isort treats `flydsl` as first-party.
- **C++**: LLVM style, `ColumnLimit: 100` in `.clang-format`; C++17 via top-level `CMakeLists.txt`.
- **CI style gate**: `.github/workflows/pre-checks.yaml` runs a Python check (black + ruff, `.github/scripts/check_python_style.sh`) and a C++ check (`clang-format-18`, `.github/scripts/check_cpp_style.sh`) on every PR. Reproduce the Python gate locally with `scripts/check_python_style.sh`.
- **Generated output**: never edit `build-fly/python_packages/`, generated `_mlir` bindings, or other build outputs directly.
- **Third-party code**: avoid touching `thirdparty/` unless the task explicitly requires it.

## GPU Architecture Support

| Arch | Chips | Wave size | MMA path | Notes |
|---|---|---|---|---|
| `gfx942` | MI300X / MI308X | 64 | MFMA | CDNA3 baseline; preshuffle GEMM, PA decode, CDNA BufferCopy |
| `gfx950` / `gfx95*` | MI350 / MI355X | 64 | MFMA | CDNA4 path; FP4, MFMA scale, wider LDS copy paths, 160KB LDS |
| `gfx11*` | RDNA3 / RDNA3.5 (Strix Halo, e.g. gfx1151) | 32 | WMMA | No MFMA; f16/bf16 (and i8/i4) WMMA GEMM; legacy v16-operand WMMA ABI; **no native FP8** (kernels fail-fast); `kernels/rdna3_f16_gemm.py`. `is_rdna_arch()` returns True. |
| `gfx120*` | RDNA4 (gfx1201 = Radeon AI PRO R9700) | 32 | WMMA | RDNA path, wave32; new v8-operand WMMA ABI; native FP8. `is_rdna_arch()` returns True. |
| `gfx1250` | MI450 | 32 | WMMA / TDM | FP8/FP4 GEMM, MoE, async/TDM copy helpers, 320KB LDS. NOTE: `is_rdna_arch('gfx1250')` returns **False** and `get_warp_size` returns 64 — the gfx1250 kernels hardcode `WAVE_SIZE = 32` themselves. |

Use `from flydsl.runtime.device import get_rocm_arch, is_rdna_arch` rather than
hard-coding behavior when possible. `is_rdna_arch()`
(`python/flydsl/runtime/device.py`) is the single source of truth for CDNA vs
RDNA and is wave32-true only for `gfx10*`/`gfx11*`/`gfx120*` prefixes; it does
**not** match `gfx1250`. Shared wave-size logic is `get_warp_size(arch)` in
`kernels/kernels_common.py` (`32 if is_rdna_arch(arch) else 64`), so it returns
64 for gfx1250 — gfx1250 kernels set wave32 explicitly.
`tests/kernels/test_rdna_gemm.py` shows the gfx11* (v16 ABI) vs gfx120* (v8 ABI)
kernel-selection pattern.

## Kernel Entry Points

This is routing guidance, not a complete kernel inventory. Search the current `kernels/` tree before edits; keep user-facing catalogs in `docs/prebuilt_kernels_guide.md`.

- **Attention**: start paged-decode changes in `kernels/pa_decode_fp8.py` and `tests/kernels/test_pa.py`; sliding-window is a backend mode imported by `pa_decode_fp8.py`, not a separate public entry point.
- **GEMM / MoE**: choose by architecture and dtype first (CDNA MFMA, RDNA wave32 WMMA, gfx1250 TDM/WMMA/MX-scale). `tests/kernels/test_rdna_gemm.py` shows gfx11* vs gfx120* dispatch; reuse topical `*_common.py` / `*_utils.py` helpers.
- **Other kernels**: keep regression tests close to touched `kernels/` modules; preserve host-shim vs device-kernel boundaries for multi-GPU communication (`custom_all_reduce.py` wraps the lower-level implementation).
- **New families**: add focused `tests/kernels/test_*.py` coverage, update `docs/prebuilt_kernels_guide.md` for public APIs, and only add durable routing rules here when they prevent likely agent mistakes.

## Kernel Authoring Conventions

- Prefer the layout API for new kernels: `fx.rocdl.make_buffer_tensor()` plus logical layout operations and `fx.copy_atom_call`. Raw `buffer_ops.create_buffer_resource()` / manual byte offsets are legacy.
- Use `@flyc.kernel` for device kernels and `@flyc.jit` for launch wrappers; kernel modules are normally imported from `kernels.*`.
- Use `range_constexpr` for compile-time unrolled Python loops. Use `range(start, stop, step, init=[...])` for `scf.for` loops with loop-carried values.
- Keep `scf.for` state explicit and compact. Clear `SmemPtr._view_cache = None` after exiting `scf.for` when shared-memory views are recreated, to avoid MLIR dominance issues.
- Allocate shared memory with `SharedAllocator` (`flydsl.expr.gpu`, reached as `fx.SharedAllocator`) over a `@fx.struct` storage layout for new kernels. The typical `static=True` (default) mode emits a per-leaf static LDS global and the compiler sizes it, so `launch(smem=...)` is left unset. Only the `static=False` (dynamic) mode makes the launch wrapper auto-infer smem from `SharedAllocator.allocated_bytes` when `smem=None`; an explicit `smem` must be >= that size. The legacy `utils.smem_allocator.SmemAllocator` / `SmemPtr` path remains for un-migrated kernels (PR #506 added SharedAllocator; PR #541 migrated norm/softmax/fp8-gemm).
- Do not define a value only inside an `if`/`else` branch and use it after the branch. Hoist the value or return a single explicit merged value.
- Nested helpers inside `@flyc.kernel` / `@flyc.jit` may read captured values, but should not mutate captured outer variables. Pass values explicitly and return updated state.
- Avoid early `return` and branch-local `return` / `yield` in traced functions. Keep a single explicit exit path so MLIR result types stay well-defined.
- Prefer arch-specific helper modules and constants over inline scattered `gfx*` conditionals.
- **Helper placement.** Do not scatter small helpers across unrelated modules and do not duplicate an existing one; search for and reuse an existing helper first. Shared kernel helpers belong in `kernels/kernels_common.py` (wave size via `get_warp_size`, `dtype_to_elem_type`, `validate_moe_dtypes`, the `_if_then` SCF context manager, LLVM-ptr/stream helpers); domain-specific shared helpers go in the existing topical modules (`kernels/moe_common.py`, `layout_utils.py`, `pipeline_utils.py`, `fp8_gemm_utils.py`, `dpp_utils.py`, `mfma_epilogues.py`, `mfma_preshuffle_pipeline.py`). DSL-level numeric/arith and type helpers belong in `python/flydsl/expr/utils/arith.py` / `python/flydsl/expr/numeric.py`; compiler/runtime-wide utilities (env, logger, smem allocator) in `python/flydsl/utils/`. (PR #388 extracted shared `_if_then`/`validate_moe_dtypes` into `kernels_common.py`; PR #448 removed redundant numeric wrappers in favor of existing `fx.*` type methods.)
- **`expr/` is target-neutral.** The direct child modules of `python/flydsl/expr/` (`typing`, `primitive`, `gpu`, `derived`, `struct`, `arith`, `math`, `vector`, `numeric`, `meta`, `extern`, `utils/`) must stay backend-agnostic: they may not import ROCDL/HIP bindings (`flydsl._mlir.dialects.rocdl`, `_mlirDialectsFlyROCDL`, `fly_rocdl`). `import flydsl.expr` must succeed without the FlyROCDL bindings; `tests/unit/test_expr_optional_rocdl.py` enforces this in CI. New target-specific (ROCDL/HIP, MFMA/WMMA, buffer/TDM/cluster) expr code goes in the `expr/rocdl/` package (`cdna4`, `cluster`, `inline_asm`, `tdm_ops`, `universal`), never in a new top-level `expr/*.py`. The target-specific modules `buffer_ops`, `rocdl`, and `tdm_ops` are lazy-loaded from `expr/__init__.py` via `__getattr__` (`_LAZY_MODULES`); add new backend modules to that lazy map rather than eager-importing them (PR #521).
- **`expr/rocdl` is a package.** `expr/rocdl/` (`__init__.py` + `cluster.py`, `tdm_ops.py`, `cdna4.py`, `universal.py`, `inline_asm.py`) holds all target-specific ROCDL/MFMA/WMMA/buffer/TDM/cluster code. `from flydsl.expr import rocdl` and `flydsl.expr.rocdl` bind to `expr/rocdl/__init__.py`. Import submodules explicitly, e.g. `from flydsl.expr.rocdl import cluster`; `flydsl.expr.tdm_ops` is a lazy alias for `flydsl.expr.rocdl.tdm_ops` (see `expr/__init__.py` `_LAZY_MODULES`).

## Testing Notes

- `tests/kernels/*.py` are generally `l2_device` + `rocm_lower` and require GPU execution.
- `tests/unit/*` mixes backend-agnostic, compile-tier, and device-tier tests; check markers before broad edits.
- `tests/mlir/**/*.mlir` are FileCheck tests run by `scripts/run_tests.sh`, not pytest.
- `tests/arch_compat.py` is the source of truth for examples/tests that are RDNA-compatible versus CDNA-only.
- Pytest markers are registered in `tests/pytest.ini`: `large_shape`, the tier markers (`l0_backend_agnostic`, `l1a_compile_no_target_dialect`, `l1b_target_dialect`, `l2_device`, `rocm_lower`), plus `multi_gpu` (multi-GPU tests; auto-skipped when GPU count is insufficient) and `benchmark` (long-running perf tests).
- Multi-GPU coverage (`tests/kernels/test_flydsl_shmem.py`, `tests/kernels/test_allreduce.py`) runs under `pytest -m multi_gpu`. The shmem regression skips below 2 GPUs; allreduce has a 4-GPU accuracy case that skips below 4 GPUs and 8-GPU accuracy/benchmark cases that skip below 8 GPUs. It is gated in CI to a label-triggered job (PR label `multi-gpu` or manual dispatch) on the 8-GPU runner matrix (`linux-flydsl-mi325-8`, `linux-flydsl-mi355-8`); it does not run in the default `scripts/run_tests.sh` flow.
- For paged-attention changes, start with `tests/kernels/test_pa.py`; reference semantics live in `reference_masked_attention()` and `torch_mha_extend()`.
