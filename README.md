# FlyDSL (<span style="color:#2f81f7"><strong>F</strong></span>lexible <span style="color:#2f81f7"><strong>l</strong></span>ayout p<span style="color:#2f81f7"><strong>y</strong></span>thon DSL)
> A Python DSL and a MLIR stack for authoring high‑performance GPU kernels with explicit layouts and tiling. 

FlyDSL is the **Python front‑end** of the project: a *Flexible Layout Python DSL* for expressing
tiling, partitioning, data movement, and kernel structure at a high level.

**FlyDSL**: FlyDSL is powered by the Fly dialect:
an end‑to‑end, MLIR‑native compiler stack for GPU kernels. Its core is the `fly` dialect—a first‑class
layout IR with explicit algebra and coordinate mapping, plus a composable lowering pipeline to GPU/ROCDL.

## Overview

- **FlyDSL (Python DSL)**: author kernels in Python and compile them through the Fly dialect
  - Primary package: `python/flydsl/`
  - Kernel examples: `kernels/` (importable as `kernels.*`)
- **Fly dialect**: the layout IR and compiler foundation
  - Core abstractions: `!fly.int_tuple`, `!fly.layout`, `!fly.coord_tensor`, `!fly.memref`
  - Algebra ops: composition/product/divide/partition + coordinate mapping ops
- **Embedded MLIR Python runtime** (`_mlir`)
  - No external `mlir` python wheel is required: MLIR python bindings are included with the FlyDSL package/build artifacts

### Repository layout

```
FlyDSL/
├── scripts/                   # build & test scripts
│   ├── build_llvm.sh          # build LLVM/MLIR from source
│   ├── build.sh               # build FlyDSL (C++ + Python bindings)
│   ├── run_tests.sh           # run tests
│   └── run_benchmark.sh       # run performance benchmarks
├── include/flydsl/            # C++ Fly/FlyROCDL dialect headers
├── lib/                       # C++ dialect implementation + Python bindings
├── python/
│   ├── flydsl/                # Python DSL sources
│   │   ├── expr/              # DSL expression API (primitive, arith, vector, gpu, rocdl, buffer_ops, math, mem_ops)
│   │   ├── compiler/          # JIT compilation pipeline (ast_rewriter, kernel_function, jit_function, backends/)
│   │   ├── runtime/           # Device runtime (device.py, device_runtime/)
│   │   ├── utils/             # Utilities (smem_allocator, env, logger)
│   │   └── autotune.py        # Triton-style autotune module
│   └── mlir_flydsl/           # MLIR Python bindings (built, not edited)
├── examples/                  # Runnable examples
│   ├── 01-vectorAdd.py        # Vector addition with layout algebra
│   ├── 02-tiledCopy.py        # Tiled copy with partitioned tensors
│   ├── 03-tiledMma.py         # Tiled MMA (GEMM) with MFMA atoms
│   └── 04-preshuffle_gemm.py  # Preshuffle GEMM end-to-end example
├── kernels/                   # Production GPU kernels (importable as `kernels.*`)
├── tests/                     # All tests (kernels/, mlir/, unit/)
├── CMakeLists.txt             # top-level CMake
└── setup.py                   # Python packaging
```

## Getting started

### Prerequisites

- **Python**: Python 3.10+ with `pip`
- **ROCm**: required for GPU execution, tests, and benchmarks (tested on ROCm 6.x, 7.x)

### Install FlyDSL

For most users, install the published package directly:

```bash
pip install flydsl
```

### Verify the install

```bash
python -c "import flydsl; print('FlyDSL installed')"
```

### Build from source

Build from source only if you are developing FlyDSL itself or need a custom MLIR/LLVM build.

Prerequisites for source builds:

- **Build tools**: `cmake` (>=3.20), C++17 compiler, optionally `ninja`
- **Python deps**: `nanobind`, `numpy`, `pybind11` (installed by `scripts/build_llvm.sh`; install them manually if you skip that step)

```bash
# Clone ROCm LLVM and build MLIR (takes ~30min with -j64)
bash scripts/build_llvm.sh -j64

# Build FlyDSL C++ dialects, compiler passes, and Python bindings
bash scripts/build.sh -j64

# Install in development mode
pip install -e .
```

If you already have an MLIR build with Python bindings enabled, point to it instead:

```bash
pip install nanobind numpy pybind11  # build.sh does not install these
export MLIR_PATH=/path/to/llvm-project/build-flydsl/mlir_install
MLIR_PATH=$MLIR_PATH bash scripts/build.sh -j64
pip install -e .
```

> **Note**: If `MLIR_PATH` is set in your environment pointing to a wrong LLVM build, `unset MLIR_PATH` first.

### Run tests

Tests and examples require `pytest`, `pandas`, and a ROCm build of `torch` (not installed by `pip install -e .`):

```bash
pip install pytest pandas
# torch must be a ROCm build matching your ROCm version (rocm7.2 shown):
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# Run GEMM correctness tests (fast, ~15s)
python -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape"

# Run performance benchmarks
bash scripts/run_benchmark.sh
```

**Test layout, pytest markers, and environment variables** used by the suite are documented in [**`tests/README.md`**](tests/README.md) .

### Quick reference

```bash
# Install from PyPI:
pip install flydsl

# Full source build from scratch:
bash scripts/build_llvm.sh -j64   # one-time: build LLVM/MLIR
bash scripts/build.sh -j64        # build FlyDSL
pip install -e .                  # install in dev mode
bash scripts/run_tests.sh         # verify

# Rebuild after code changes (C++ only):
bash scripts/build.sh -j64

# Rebuild after Python-only changes:
# No rebuild needed — editable install picks up changes automatically.
```

### Troubleshooting

- **Wrong LLVM picked up** (`std::gcd not found`, `redeclaration` errors)
  - `unset MLIR_PATH` and let `build.sh` auto-detect, or set it to the correct path.

- **`No module named flydsl`**
  - Run `pip install flydsl`. For source checkouts, run `pip install -e .` after building.

- **MLIR `.so` load errors**
  - Add MLIR build lib dir to the loader path:
    -  export LD_LIBRARY_PATH=$(pwd)/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH

## Documentation

**Full documentation: [rocm.github.io/FlyDSL](https://rocm.github.io/FlyDSL)**

| **Topic** | **Description** | **Guide** |
|---|---|---|
| Architecture | Compilation pipeline, project structure, environment config | [Architecture Guide](docs/architecture_guide.md) |
| Layout System | FlyDSL layout algebra — Shape, Stride, Layout, Coord, all operations | [Layout Guide](docs/layout_system_guide.md) |
| Kernel Authoring | Writing GPU kernels — MlirModule, tiled copies, MFMA, shared memory | [Kernel Guide](docs/kernel_authoring_guide.md) |
| Pre-built Kernels | Available kernels — GEMM, MoE, Softmax, Norm — config and usage | [Kernels Reference](docs/prebuilt_kernels_guide.md) |
| Testing & Benchmarks | Test infrastructure, benchmarking, performance comparison | [Testing Guide](docs/testing_benchmarking_guide.md) |

- **Kernel cache issues** (stale results after code changes)
  - The JIT disk cache auto-invalidates on source/closure changes; only needed for C++ pass or non-closure helper changes
  - Clear manually: `rm -rf ~/.flydsl/cache` or `export FLYDSL_RUNTIME_ENABLE_CACHE=0`

## 📐 Layout System

FlyDSL introduces a layout system to express complex data mapping patterns on GPUs (tiling, swizzling, vectorization).

### Core Abstractions

1.  **Shape**: The extent of dimensions (e.g., `(M, N)`).
2.  **Stride**: The distance between elements in memory (e.g., `(1, M)` for column-major).
3.  **Layout**: A pair of `(Shape, Stride)` that maps a logical **Coordinate** to a physical linear **Index**.

Formula: `Index = dot(Coord, Stride) = sum(c_i * s_i)`

### Operations

*   **Construction**: `make_shape`, `make_stride`, `make_layout`, `make_coord`
*   **Mapping**:
    *   `crd2idx(coord, layout) -> index`: Convert logical coordinate to physical index.
    *   `idx2crd(index, layout) -> coord`: Convert physical index to logical coordinate.
*   **Inspection**: `size`, `cosize`, `rank`
*   **Algebra**:
    *   `composition(A, B)`: Compose layouts (A ∘ B).
    *   `product(A, B)`: Combine layouts (Logical, Tiled, Blocked, etc.).
    *   `divide(A, B)`: Partition layout A by B (Logical, Tiled, etc.).

## Documentation

| **Topic** | **Description** | **Guide** |
|---|---|---|
| Architecture | Compilation pipeline, project structure, environment config | [Architecture Guide](docs/architecture_guide.md) |
| Layout System | Fly layout algebra — Shape, Stride, Layout, Coord, all operations | [Layout Guide](docs/layout_system_guide.md) |
| Kernel Authoring | Writing GPU kernels — `@flyc.kernel`, `@flyc.jit`, expression API | [Kernel Guide](docs/kernel_authoring_guide.md) |
| Pre-built Kernels | Available kernels — GEMM, Softmax, Norm — config and usage | [Kernels Reference](docs/prebuilt_kernels_guide.md) |
| Testing & Benchmarks | Test infrastructure, benchmarking, performance comparison | [Testing Guide](docs/testing_benchmarking_guide.md) |

## 🐍 Python API (`flydsl`)

### `@flyc.kernel` / `@flyc.jit` API

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu

@flyc.kernel
def my_kernel(arg_a: fx.Tensor, arg_b: fx.Tensor, n: fx.Constexpr[int]):
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    # ... kernel body using layout ops ...

@flyc.jit
def launch(arg_a: fx.Tensor, arg_b: fx.Tensor, n: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(arg_a, arg_b, n).launch(
        grid=(grid_x, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )
```

### Compilation Pipeline

On first call, `@flyc.jit` traces the Python function into an MLIR module, then compiles it through `MlirCompiler`. The pass list is built by `RocmBackend._pipeline_parts()` in three stages — see [`docs/architecture_guide.md`](docs/architecture_guide.md#3-compilation-pipeline) for the per-pass table.

```
Python Function (@flyc.kernel / @flyc.jit)
        │
        ▼  AST Rewriting + Tracing
   MLIR Module (fly, gpu, arith, scf, memref, vector dialects)
        │
        ▼  MlirCompiler.compile()
   ┌──────────────────────────────────────────────────────────┐
   │ A. pre_binary_fragments  (Fly → ROCDL)                   │
   │    fly-rewrite-func-signature → fly-canonicalize →       │
   │    fly-layout-lowering → fly-int-swizzle-simplify →      │
   │    canonicalize → fly-convert-atom-call-to-ssa-form →    │
   │    fly-promote-regmem-to-vectorssa →                     │
   │    convert-fly-to-rocdl → canonicalize →                 │
   │    gpu.module(convert-scf-to-cf, cse,                    │
   │       convert-gpu-to-rocdl{...}, fly-rocdl-cluster-attr) │
   ├──────────────────────────────────────────────────────────┤
   │ B. binary_prep_fragments  (→ LLVM)                       │
   │    rocdl-attach-target{chip=gfxNNN} →                    │
   │    convert-scf-to-cf → convert-cf-to-llvm →              │
   │    gpu-to-llvm → convert-vector/arith/func-to-llvm →     │
   │    reconcile-unrealized-casts                            │
   ├──────────────────────────────────────────────────────────┤
   │ C. binary_fragment                                       │
   │    gpu-module-to-binary{format=fatbin}                   │
   └──────────────────────────────────────────────────────────┘
        │
        ▼
   Cached Compiled Artifact (ExecutionEngine)
```

Compiled kernels are cached to disk (`~/.flydsl/cache/`) and reused on subsequent calls with the same type signature.

## ⚙️ Hierarchical Kernel Control

FlyDSL keeps the tiling hierarchy explicit across block, warp, thread, and instruction scopes using layout algebra:

```python
import flydsl.expr as fx

# Define thread and value layouts for tiled copy
thr_layout = fx.make_layout((THR_M, THR_N), (1, THR_M))
val_layout = fx.make_layout((VAL_M, VAL_N), (1, VAL_M))

# Create tiled copy with vectorized atoms
copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
layout_thr_val = fx.raked_product(thr_layout, val_layout)
tile_mn = fx.make_tile(fx.make_layout(THR_M, 1), fx.make_layout(VAL_M, 1))
tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)

# Partition tensor across blocks and threads
thr_copy = tiled_copy.get_slice(tid)
partition_src = thr_copy.partition_S(block_tile_A)
partition_dst = thr_copy.partition_D(register_fragment)

# Execute copy
fx.copy(copy_atom, partition_src, partition_dst)
```

With per-level partitions, you can allocate register fragments, emit predicate masks, and schedule MFMA/vector instructions while retaining full knowledge of the execution hierarchy.

## 🧮 Minimal VecAdd Example

This condensed snippet mirrors `examples/01-vectorAdd.py`, showing how to define GPU kernels with layout algebra and tiled copies:

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    # Partition tensors by block
    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))

    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))

    # Load to registers, compute, store via copy atoms
    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA = fx.make_rmem_tensor(1, fx.Float32)
    rB = fx.make_rmem_tensor(1, fx.Float32)
    rC = fx.make_rmem_tensor(1, fx.Float32)

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

@flyc.jit
def vectorAdd(
    A: fx.Tensor, B: fx.Tensor, C,
    n: fx.Int32,  # dynamic int32
    const_n: fx.Constexpr[int],  # static int32, affects JIT cache-key
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    vectorAddKernel(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream,
    )

# Usage
n = 128
A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
C = torch.zeros(n, dtype=torch.float32).cuda()
vectorAdd(A, B, C, n, n + 1, stream=torch.cuda.Stream())

torch.cuda.synchronize()
print("Result correct:", torch.allclose(C, A + B))
```

See `examples/` for more examples including tiled copy (`02-tiledCopy.py`), tiled MMA (`03-tiledMma.py`), and preshuffle GEMM (`04-preshuffle_gemm.py`).

## ✅ Testing Status

| Category | Test File | Description |
|----------|-----------|-------------|
| **Preshuffle GEMM** | `test_preshuffle_gemm.py` | FP8, INT8, INT4, BF16, FP4 |
| **Blockscale GEMM** | `test_blockscale_preshuffle_gemm.py` | Blockscale preshuffle GEMM |
| **HGEMM Split-K** | `test_hgemm_splitk.py` | FP16 GEMM split-K |
| **MoE GEMM** | `test_moe_gemm.py` | MoE 2-stage (gate/up + reduce) |
| **MoE Blockscale** | `test_moe_blockscale.py` | MoE blockscale 2-stage |
| **MoE Reduce** | `test_moe_reduce.py` | MoE reduce kernel |
| **PagedAttention** | `test_pa.py` | Paged attention decode (FP8) — *WIP perf tuning* |
| **FlashAttention** | `test_flash_attn_fwd.py` | Flash attention — *WIP perf tuning* |
| **LayerNorm** | `test_layernorm.py` | LayerNorm (layout API) |
| **RMSNorm** | `test_rmsnorm.py` | RMSNorm (layout API) |
| **Softmax** | `test_softmax.py` | Softmax (layout API) |
| **Fused RoPE** | `test_fused_rope_cache.py` | Fused RoPE + KV cache |
| **AllReduce** | `test_allreduce.py` | Multi-GPU all-reduce |
| **RDNA GEMM** | `test_rdna_gemm.py` | RDNA FP16/FP8 GEMM |
| **GFX1250 GEMM** | `test_gemm_fp8fp4_gfx1250.py` | GFX1250 FP8/FP4 GEMM |
| **WMMA GEMM** | `test_wmma_gemm_gfx1250.py` | GFX1250 WMMA GEMM |
| **VecAdd** | `test_vec_add.py` | Basic vector addition |
| **Quantization** | `test_quant.py` | Quantization utilities |

**Verified Platforms**:
*   AMD MI300X/MI308X (gfx942), AMD MI350/MI355X (gfx950), AMD MI450 (gfx1250), Radeon AI PRO R9700 (gfx1201)
*   Linux / ROCm 6.x, 7.x

## 🙏 Acknowledgements

FlyDSL's design is inspired by ideas from several projects:

- [Categorical Foundations for CuTe Layouts](https://arxiv.org/abs/2601.05972) — mathematical framework for layout algebra ([companion code](https://github.com/ColfaxResearch/layout-categories))
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — CuTe layout algebra concepts (BSD-3-Clause parts only; no EULA-licensed code was referenced)
- [ROCm Composable Kernel](https://github.com/ROCm/composable_kernel) — tile-based kernel design patterns for AMD GPUs
- [ROCm AIter](https://github.com/ROCm/aiter) — test infrastructure and performance comparison baselines (MIT)
- [Triton](https://github.com/triton-lang/triton) — Python DSL for GPU kernel authoring

## 📄 License

Apache License 2.0

## Disclaimer

This is an experimental feature/tool and is not part of the official ROCm distribution. It is provided for evaluation and testing purposes only.
For further usage or inquiries, please initiate a discussion thread with the original authors.
