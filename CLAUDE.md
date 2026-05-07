# FlyDSL Project Guide

FlyDSL (Flexible Layout Python DSL) — a Python DSL and MLIR-based compiler stack for authoring high-performance GPU kernels with explicit layouts and tiling on AMD GPUs (MI300X/MI350/MI355X/MI450).

## Repository Layout

```
FlyDSL/
├── python/
|   |── flydsl/             # Python DSL core
│   |   |── expr/           # DSL expression API (arith, vector, gpu, rocdl, buffer_ops, math, mem_ops, derived)
│   |   |── compiler/       # JIT compilation (@kernel, @jit, backends, AST rewriter)
│   |   |── runtime/        # Device runtime, GPU arch detection (get_rocm_arch, is_rdna_arch)
│   |   |── utils/          # SmemAllocator, env, logger
│   |   └── autotune.py     # Autotuner (@autotune decorator, Config)
│   └── mlir_flydsl/        # Embedded MLIR Python bindings (built, never edit)
├── kernels/                # Production GPU kernels (importable as kernels.*)
├── include/flydsl/         # C++ Fly dialect headers
├── lib/                    # C++ dialect implementation + Python bindings
├── tests/                  # kernels/, mlir/, pyir/, unit/
├── examples/               # 01-vectorAdd, 02-tiledCopy, 03-tiledMma, 04-preshuffle_gemm
├── scripts/                # build_llvm.sh, build.sh, run_tests.sh, run_benchmark.sh
├── thirdparty/             # Third-party deps (dlpack, tvm-ffi)
├── tools/                  # fly-opt tool (C++)
└── docs/                   # Detailed guides (see below)
```

## Detailed Documentation

For in-depth information, refer to these docs:

| Topic | Doc | Description |
|---|---|---|
| Architecture & compilation pipeline | [`docs/architecture_guide.md`](docs/architecture_guide.md) | Project structure, MLIR pass pipeline, compilation stages |
| Layout algebra | [`docs/layout_system_guide.md`](docs/layout_system_guide.md) | Core types, layout operations, coordinate mapping, products & divides |
| Kernel authoring | [`docs/kernel_authoring_guide.md`](docs/kernel_authoring_guide.md) | @flyc.kernel/@flyc.jit, expression API, launch config, shared memory |
| Pre-built kernels | [`docs/prebuilt_kernels_guide.md`](docs/prebuilt_kernels_guide.md) | LayerNorm, RMSNorm, Softmax, GEMM — config, dtypes, pipelines |
| Testing & benchmarking | [`docs/testing_benchmarking_guide.md`](docs/testing_benchmarking_guide.md) | Test categories, running tests, benchmark harness |

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

### Import Troubleshooting

- If importing FlyDSL through AITER fails with
  `AttributeError: module 'flydsl._mlir.dialects.fly' has no attribute 'is_profile_congruent'`,
  the Python sources and compiled MLIR Python extension are out of sync. The
  source `python/flydsl/expr/primitive.py` expects
  `flydsl._mlir.dialects.fly.is_profile_congruent`, which is exported by
  `lib/Bindings/Python/FlyExtension.cpp`; an older `_mlirDialectsFly` shared
  object will not have it.
- Fix by rebuilding/reinstalling FlyDSL from the same checkout and ensuring
  `PYTHONPATH`/`LD_LIBRARY_PATH` point at that build before running AITER tests:
  `bash scripts/build.sh -j64 && pip install -e .`, then export
  `${PWD}/build-fly/python_packages` and `${PWD}` as shown above. Verify with
  `python -c "from flydsl._mlir.dialects import fly; print(hasattr(fly, 'is_profile_congruent'))"`.

## Code Style

- **Python**: black (line-length=120), ruff for linting. Config in `pyproject.toml`.
- **C++**: LLVM style (ColumnLimit=100). Config in `.clang-format`.
- **Imports**: isort with `flydsl` as known first-party.

## GPU Architecture Support

| Arch | Chips | Wave size | MMA | Key features |
|---|---|---|---|---|
| **CDNA3** | gfx942 (MI300X) | 64 | MFMA | BufferCopy, preshuffle GEMM |
| **CDNA4** | gfx950 (MI350/MI355X) | 64 | MFMA | MFMA_SCALE, FP4, 160KB LDS |
| **RDNA4** | gfx1201 (Radeon AI PRO R9700) | 32 | WMMA | RDNA-specific GEMM |
| **gfx1250** | MI450 | 32 | WMMA | TDM ops, FP8/FP4 GEMM, multi-stage pipeline |

## Key Conventions & Pitfalls

- **JIT cache**: Always set `FLYDSL_RUNTIME_ENABLE_CACHE=0` when iterating on kernel code (in-memory cache still active)
- **PYTHONPATH**: `PYTHONPATH=./` is required when running from repo root
- **Kernel imports**: `from kernels.pa_decode_fp8 import ...` (kernels/ is a package)
- **`_mlir` package**: Auto-generated during build — never edit directly
- **SmemPtr cache**: Clear `SmemPtr._view_cache = None` after exiting `scf.for` to avoid MLIR dominance errors
- **Layout API vs buffer_ops**: New kernels should use `fx.rocdl.make_buffer_tensor()` + `copy_atom_call` (layout API). Raw `buffer_ops.create_buffer_resource()` is legacy
- **Arch detection**: Use `from flydsl.runtime.device import get_rocm_arch`
- **`range` vs `range_constexpr`**: Use `range_constexpr` for compile-time unrolled loops; `range(start, stop, step, init=[...])` for `scf.for` with loop-carried values
- **Branch-local defs**: Do not define a value inside `if/else` and then use it after the branch. Hoist the variable or rewrite the logic so later uses see a single explicit definition path.
- **Nested helper captures**: Inside `@flyc.kernel` / `@flyc.jit`, nested helper functions must not mutate captured outer variables. Read-only capture is acceptable, but writes should go through explicit parameters / returns.
- **Single-exit control flow**: Avoid early `return`. Do not place `return` or `yield` inside `if/else` branches; keep a single explicit exit path so MLIR result types stay well-defined.
