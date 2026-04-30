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
├── python/
│   ├── flydsl/                    # Python DSL core
│   │   ├── expr/                  # DSL expression API: primitive, typing, arith, vector, gpu, math, rocdl, buffer_ops
│   │   ├── compiler/              # @flyc.kernel / @flyc.jit, AST rewriting, JIT cache, backends
│   │   ├── runtime/               # Device runtime and GPU arch detection
│   │   ├── utils/                 # EnvManager, SmemAllocator, logger
│   │   └── autotune.py            # Autotuner (@autotune, Config)
│   └── mlir_flydsl/               # MLIR Python binding package source
├── include/flydsl/                # C++ TableGen headers for Fly / FlyROCDL dialects and passes
├── lib/                           # C++ dialect implementation, conversions, runtime wrappers, Python bindings
├── tools/                         # fly-opt
├── kernels/                       # Production kernels, importable as kernels.*
├── tests/
│   ├── kernels/                   # GPU correctness / benchmark harnesses
│   ├── unit/                      # Python compiler, runtime, and layout tests
│   ├── system/                    # Cross-cutting compile/system tests
│   ├── mlir/                      # FileCheck tests driven by scripts/run_tests.sh
│   └── python/examples/           # Pytest coverage for examples
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
RUN_TESTS_FULL=1 bash scripts/run_tests.sh  # Include large_shape tests
bash scripts/run_benchmark.sh         # Performance benchmarks
```

Useful direct commands:

```bash
python3 -m pytest tests/kernels/ tests/unit/ tests/system/ tests/python/examples/ -m "not large_shape" -v
python3 -m pytest tests/kernels/test_pa.py -v
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 -m pytest tests/kernels/test_pa.py -v
```

`scripts/run_tests.sh` auto-selects the GPU with the most free VRAM when
`HIP_VISIBLE_DEVICES` is unset and sets `FLYDSL_RUN_QUANT=1`.

## Environment Variables

Use names from `python/flydsl/utils/env.py`; do not introduce alternate spellings.

| Purpose | Variable |
|---|---|
| Compile backend | `FLYDSL_COMPILE_BACKEND` (default `rocm`) |
| Override compile arch | `ARCH` |
| Compile without execution | `COMPILE_ONLY` |
| JIT cache directory | `FLYDSL_RUNTIME_CACHE_DIR` |
| Enable/disable JIT disk cache | `FLYDSL_RUNTIME_ENABLE_CACHE` (`0` / `false` disables disk cache; in-memory cache remains) |
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
- **Generated output**: never edit `build-fly/python_packages/`, generated `_mlir` bindings, or other build outputs directly.
- **Third-party code**: avoid touching `thirdparty/` unless the task explicitly requires it.

## GPU Architecture Support

| Arch | Chips | Wave size | MMA path | Notes |
|---|---|---|---|---|
| `gfx942` | MI300X / MI308X | 64 | MFMA | CDNA3 baseline; preshuffle GEMM, PA decode, CDNA BufferCopy |
| `gfx950` / `gfx95*` | MI350 / MI355X | 64 | MFMA | CDNA4 path; FP4, MFMA scale, wider LDS copy paths, 160KB LDS |
| `gfx120*` | Radeon AI PRO R9700 class | 32 | WMMA | RDNA path; use `is_rdna_arch()` / wave32-aware helpers |
| `gfx1250` | MI450 | 32 | WMMA / TDM | FP8/FP4 GEMM, MoE, async/TDM copy helpers |

Use `from flydsl.runtime.device import get_rocm_arch, is_rdna_arch` rather than
hard-coding behavior when possible. Shared wave-size logic lives in
`kernels/kernels_common.py`.

## Kernel Entry Points

- **Paged attention decode**: `kernels/pa_decode_fp8.py`; primary regression harness is `tests/kernels/test_pa.py`.
- **Flash/MLA attention**: `kernels/flash_attn_func.py`, `kernels/mla_fwd_decode.py`, `kernels/mla_fwd_decode_m16x8_fp8_fp8.py`.
- **GEMM**: `kernels/preshuffle_gemm.py`, `kernels/preshuffle_gemm_v2.py`, `kernels/blockscale_preshuffle_gemm.py`, `kernels/hgemm_splitk.py`.
- **gfx1250 GEMM/MoE**: `kernels/gemm_common_gfx1250.py`, `kernels/gemm_fp8fp4_gfx1250.py`, `kernels/wmma_gemm_gfx1250.py`, `kernels/moe_gemm_2stage_*_gfx1250.py`.
- **MoE**: `kernels/moe_gemm_2stage.py`, `kernels/moe_blockscale_2stage.py`, `kernels/mixed_moe_gemm_2stage.py`.
- **Elementwise / reductions / communication**: `kernels/layernorm_kernel.py`, `kernels/rmsnorm_kernel.py`, `kernels/softmax_kernel.py`, `kernels/fused_rope_cache_kernel.py`, `kernels/custom_all_reduce.py`.

## Kernel Authoring Conventions

- Prefer the layout API for new kernels: `fx.rocdl.make_buffer_tensor()` plus logical layout operations and `fx.copy_atom_call`. Raw `buffer_ops.create_buffer_resource()` / manual byte offsets are legacy.
- Use `@flyc.kernel` for device kernels and `@flyc.jit` for launch wrappers; kernel modules are normally imported from `kernels.*`.
- Use `range_constexpr` for compile-time unrolled Python loops. Use `range(start, stop, step, init=[...])` for `scf.for` loops with loop-carried values.
- Keep `scf.for` state explicit and compact. Clear `SmemPtr._view_cache = None` after exiting `scf.for` when shared-memory views are recreated, to avoid MLIR dominance issues.
- Do not define a value only inside an `if`/`else` branch and use it after the branch. Hoist the value or return a single explicit merged value.
- Nested helpers inside `@flyc.kernel` / `@flyc.jit` may read captured values, but should not mutate captured outer variables. Pass values explicitly and return updated state.
- Avoid early `return` and branch-local `return` / `yield` in traced functions. Keep a single explicit exit path so MLIR result types stay well-defined.
- Prefer arch-specific helper modules and constants over inline scattered `gfx*` conditionals.

## Testing Notes

- `tests/kernels/*.py` are generally `l2_device` + `rocm_lower` and require GPU execution.
- `tests/unit/*` mixes backend-agnostic, compile-tier, and device-tier tests; check markers before broad edits.
- `tests/mlir/**/*.mlir` are FileCheck tests run by `scripts/run_tests.sh`, not pytest.
- `tests/arch_compat.py` is the source of truth for examples/tests that are RDNA-compatible versus CDNA-only.
- For paged-attention changes, start with `tests/kernels/test_pa.py`; reference semantics live in `reference_masked_attention()`, `torch_mha_extend()`, and `torch_mha_extend2()`.
