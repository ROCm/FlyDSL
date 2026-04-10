# FlyDSL Project Guide

FlyDSL (Flexible Layout Python DSL) — a Python DSL and MLIR-based compiler stack for authoring high-performance GPU kernels with explicit layouts and tiling on AMD GPUs (MI300X/MI350/MI450).

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
bash scripts/build_llvm.sh   # Build LLVM/MLIR (one-time)
bash scripts/build.sh        # Build FlyDSL (uses Ninja)
pip install -e .              # Install in dev mode

PYTHONPATH=./ pytest tests/   # All tests
PYTHONPATH=./ python tests/kernels/test_pa.py --num_iters 50  # Specific test
FLYDSL_DUMP_IR=1 PYTHONPATH=./ python tests/kernels/test_pa.py # Dump MLIR IR at each pipeline stage (also disables disk cache)
```

## Code Style

- **Python**: black (line-length=120), ruff for linting. Config in `pyproject.toml`.
- **C++**: LLVM style (ColumnLimit=100). Config in `.clang-format`.
- **Imports**: isort with `flydsl` as known first-party.

## GPU Architecture Support

| Arch | Chips | Wave size | MMA | Key features |
|---|---|---|---|---|
| **CDNA3** | gfx942/gfx950 (MI300X) | 64 | MFMA | BufferCopy, preshuffle GEMM |
| **RDNA** | gfx10xx/gfx11xx/gfx12xx | 32 | WMMA | RDNA-specific GEMM |
| **gfx1250** | MI400 | 32 | WMMA | TDM ops, FP8/FP4 GEMM, multi-stage pipeline |

## Key Conventions & Pitfalls

- **JIT cache**: Always set `FLYDSL_RUNTIME_ENABLE_CACHE=0` when iterating on kernel code (in-memory cache still active)
- **PYTHONPATH**: `PYTHONPATH=./` is required when running from repo root
- **Kernel imports**: `from kernels.pa_decode_fp8 import ...` (kernels/ is a package)
- **`_mlir` package**: Auto-generated during build — never edit directly
- **SmemPtr cache**: Clear `SmemPtr._view_cache = None` after exiting `scf.for` to avoid MLIR dominance errors
- **Layout API vs buffer_ops**: New kernels should use `fx.rocdl.make_buffer_tensor()` + `copy_atom_call` (layout API). Raw `buffer_ops.create_buffer_resource()` is legacy
- **Arch detection**: Use `from flydsl.runtime.device import get_rocm_arch`
- **`range` vs `range_constexpr`**: Use `range_constexpr` for compile-time unrolled loops; `range(start, stop, step, init=[...])` for `scf.for` with loop-carried values
