# Architecture & Compilation Pipeline Guide

> FlyDSL project structure, compilation stages, key abstractions, and configuration.

## Quick Reference

| Component | Description | Key File |
|---|---|---|
| **FlyDSL** | Python DSL front-end for authoring GPU kernels | `flydsl/src/flydsl/` |
| **FLIR** | Flexible Layout IR -- MLIR dialect with layout algebra | `flir/` |
| **Compiler** | `flir.compile()` -- end-to-end DSL-to-binary pipeline | `flydsl/src/flydsl/compiler/compiler.py` |
| **Pipeline** | Fluent pass-pipeline builder | `flydsl/src/flydsl/compiler/pipeline.py` |
| **Executor** | MLIR ExecutionEngine wrapper for JIT execution | `flydsl/src/flydsl/compiler/executor.py` |
| **MlirModule** | Base class for kernel module authoring | `flydsl/src/flydsl/lang/ir/module.py` |

---

## 1. Project Structure

```
FlyDSL/
├── flir/                          # C++ MLIR dialect + compiler infrastructure
│   ├── include/flir/
│   │   ├── FlirOps.td             # FLIR layout ops (make_shape, crd2idx, composition, ...)
│   │   ├── FlirTypeDefs.td        # Custom types (!flir.shape, !flir.layout, ...)
│   │   ├── FlirAttrDefs.td        # Attributes (#flir.underscore, #flir.dync_i32)
│   │   ├── FlirPasses.td          # Pass declarations (flir-to-standard, trivial-dce)
│   │   ├── FlirRocmOps.td         # ROCm ops (MFMA, LDS, copy, barriers)
│   │   └── FlirRocmDialect.td     # ROCm dialect declaration
│   ├── lib/Dialect/
│   │   ├── Flir/
│   │   │   ├── FlirOps.cpp        # Op verifiers and builders
│   │   │   ├── FlirLayoutAlgebra.cpp  # Type inference for composition/product/divide
│   │   │   └── FlirDialect.cpp    # Dialect registration
│   │   └── FlirRocm/              # ROCm dialect implementation
│   ├── lib/Transforms/
│   │   ├── FlirToStandard.cpp     # flir-to-standard lowering pass
│   │   └── FlirDCE.cpp            # trivial-dce pass
│   ├── python_bindings/           # Python ↔ C++ bridge
│   │   ├── dialects/flir.py       # Low-level Python bindings for FLIR ops
│   │   └── FlirRegisterPasses.cpp # Register passes and dialects from Python
│   ├── tools/flir-opt/            # CLI tool for running passes on .mlir files
│   └── build.sh                   # Build script (CMake + ninja)
│
├── flydsl/                        # Python package (src layout)
│   └── src/flydsl/
│       ├── compiler/
│       │   ├── compiler.py        # flir.compile() -- top-level compilation entry
│       │   ├── pipeline.py        # Pipeline fluent API
│       │   ├── executor.py        # ExecutionEngineExecutor (JIT runner)
│       │   ├── context.py         # RAIIMLIRContext, RAIIMLIRContextModule
│       │   ├── cache.py           # On-disk compilation cache
│       │   └── flir_opt_helper.py # Helper for invoking flir-opt
│       ├── dialects/ext/
│       │   ├── flir.py            # High-level Python API for layout algebra
│       │   ├── gpu.py             # GPU dialect extensions (launch, barrier, ...)
│       │   ├── rocm.py            # ROCm dialect Python helpers
│       │   ├── arith.py           # Arithmetic extensions
│       │   └── ...                # scf, memref, vector, func, rocdl, ...
│       ├── lang/ir/
│       │   ├── module.py          # MlirModule base class, @kernel / @jit decorators
│       │   └── types.py           # Type helpers
│       ├── runtime/
│       │   └── device.py          # get_rocm_arch() -- GPU architecture detection
│       └── utils/
│           └── smem_allocator.py  # SmemAllocator for LDS management
│
├── kernels/                       # Pre-built GPU kernels
│   ├── preshuffle_gemm.py         # GEMM with B-preshuffle
│   ├── mixed_preshuffle_gemm.py   # Mixed-precision GEMM (FP4/FP8)
│   ├── moe_gemm_2stage.py         # MoE GEMM (2-stage)
│   ├── mixed_moe_gemm_2stage.py   # Mixed MoE GEMM
│   ├── layernorm_kernel.py        # LayerNorm
│   ├── rmsnorm_kernel.py          # RMSNorm
│   ├── softmax_kernel.py          # Softmax
│   ├── reduce.py                  # Warp/block reduction primitives
│   ├── mfma_epilogues.py          # MFMA result writeback patterns
│   └── mfma_preshuffle_pipeline.py # Preshuffle helpers for MFMA kernels
│
├── tests/
│   ├── pyir/                      # Python IR tests (no GPU required)
│   ├── kernels/                   # GPU kernel tests + benchmarks
│   └── test_common.py             # Shared test utilities
│
└── scripts/                       # Build and test helpers
    ├── build_llvm.sh              # Build MLIR from ROCm llvm-project
    ├── run_tests.sh               # Run all tests
    └── run_benchmark.sh           # Run benchmarks
```

---

## 2. Compilation Pipeline

### 2.1 High-Level Flow

```
Python DSL (@kernel / @jit)
        │
        ▼
   MLIR Module (FLIR dialect ops)
        │
        ▼  flir.compile()
   ┌────────────────────────────────────────────┐
   │  flir-to-standard                          │  FLIR → scf + arith + memref
   │  trivial-dce                               │  Dead code elimination
   │  canonicalize                              │  Standard MLIR canonicalization
   │  cse                                       │  Common subexpression elimination
   │  gpu-kernel-outlining                      │  Outline GPU kernels
   │  gpu.module(convert-scf-to-cf)             │  SCF → ControlFlow (inside gpu.module)
   │  gpu.module(convert-gpu-to-rocdl)          │  GPU → ROCDL (inside gpu.module)
   │  gpu.module(reconcile-unrealized-casts)    │  Clean up casts
   │  rocdl-attach-target{chip=gfxNNN}          │  Attach ROCm target
   │  gpu-to-llvm                               │  GPU types → LLVM types
   │  reconcile-unrealized-casts                │  Clean up remaining casts
   │  gpu-module-to-binary{format=fatbin}       │  Emit HSACO binary
   └────────────────────────────────────────────┘
        │
        ▼
   ExecutionEngineExecutor (JIT runner)
```

### 2.2 Pipeline Stages in Detail

The pipeline is defined in `compiler/compiler.py:_pipeline_fragments()`:

| Stage | Pass | Description |
|---|---|---|
| 1 | `flir-to-standard` | Lowers all `flir.*` ops to standard MLIR (scf, arith, memref). Coordinate mapping (`crd2idx`, `idx2crd`) becomes arithmetic. Layout algebra ops are folded/lowered. |
| 2 | `trivial-dce` | Removes trivially dead ops (no side effects, unused results). |
| 3 | `canonicalize` | Standard MLIR canonicalization (constant folding, identity removal, etc.). |
| 4 | `cse` | Common subexpression elimination. |
| 5 | `gpu-kernel-outlining` | Moves GPU kernel bodies into `gpu.func` inside `gpu.module`. |
| 6 | `convert-scf-to-cf` | Lowers `scf.for`/`scf.if` to `cf.br`/`cf.cond_br` (inside `gpu.module`). |
| 7 | `convert-gpu-to-rocdl` | Converts `gpu.thread_id`, `gpu.block_id`, etc. to ROCDL intrinsics (inside `gpu.module`). |
| 8 | `reconcile-unrealized-casts` | Cleans up unrealized conversion casts inside `gpu.module`. |
| 9 | `rocdl-attach-target` | Attaches `#rocdl.target<chip=gfxNNN>` for the target GPU. |
| 10 | `gpu-to-llvm` | Converts GPU-related types/ops to LLVM dialect (host-side launch infrastructure). |
| 11 | `reconcile-unrealized-casts` | Final cast cleanup. |
| 12 | `gpu-module-to-binary` | Compiles the GPU module to HSACO binary (embedded in the module as a blob). |

### 2.3 Using the Pipeline API

The `Pipeline` class provides a fluent builder:

```python
from flydsl.compiler.pipeline import Pipeline

pipeline = (
    Pipeline()
    .flir_to_standard()
    .canonicalize()
    .cse()
    .rocdl_attach_target(chip="gfx942")
    .Gpu(Pipeline().convert_gpu_to_rocdl(runtime="HIP"))
    .gpu_to_llvm()
    .lower_to_llvm()
    .gpu_module_to_binary(format="bin")
)

# Run pipeline on a module
result = pipeline.run(module)

# Or use the string representation
print(pipeline)  # builtin.module(flir-to-standard,canonicalize,...)
```

Key `Pipeline` methods:
- **FLIR passes**: `.flir_to_standard()`, `.flir_coord_lowering()` (alias)
- **Optimization**: `.canonicalize()`, `.cse()`, `.inline()`, `.symbol_dce()`, `.sccp()`
- **Conversion**: `.convert_scf_to_cf()`, `.convert_gpu_to_rocdl()`, `.gpu_to_llvm()`, `.lower_to_llvm()`
- **Target**: `.rocdl_attach_target(chip=...)`, `.gpu_module_to_binary(format=...)`
- **Nesting**: `.Gpu(nested_pipeline)`, `.Func(nested_pipeline)`, `.Module(nested_pipeline)`
- **Composition**: `pipeline_a + pipeline_b`, `pipeline_a += pipeline_b`

### 2.4 Using `flir.compile()`

For most use cases, the high-level `flir.compile()` handles the full pipeline:

```python
from flydsl.compiler.compiler import compile

# Compile and get an executor
executor = compile(my_module, opt_level=3)

# Call a kernel function
executor.my_kernel(A, B, C)
```

`flir.compile()` automatically:
1. Detects the target GPU architecture (or reads `ARCH` env var)
2. Overrides `gpu.module` targets consistently
3. Checks the on-disk cache (can skip recompilation)
4. Runs the full pipeline
5. Returns an `ExecutionEngineExecutor` (or `None` if `COMPILE_ONLY=1`)

---

## 3. Key Abstractions

### 3.1 `RAIIMLIRContextModule`

Sets up an MLIR context with FLIR dialects registered, a default location, a module, and an insertion point:

```python
from flydsl.compiler.context import RAIIMLIRContextModule

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
# ctx.context  -- MLIR Context
# ctx.module   -- MLIR Module
# ctx.location -- Default Location
```

### 3.2 `MlirModule`

Base class for structured kernel authoring. Subclass it, define `@kernel` and `@jit` methods:

```python
from flydsl.lang.ir.module import MlirModule, kernel, jit
from flydsl.dialects.ext import flir

class MyKernels(MlirModule):
    GPU_MODULE_NAME = "my_kernels"

    @kernel
    def my_kernel(self, A: T.memref(1024, T.f32())):
        tid = flir.thread_idx("x")
        # ... kernel body ...

# Instantiate to emit MLIR
mod = MyKernels()
print(mod.module)  # prints MLIR
```

Key class attributes:
- `GPU_MODULE_NAME` -- name for the `gpu.module` container
- `GPU_MODULE_TARGETS` -- optional target list (overridden by `flir.compile()`)
- `ALLOW_UNREGISTERED_DIALECTS` -- default `True`

Key decorators:
- `@kernel` -- emits a `gpu.func` with `gpu.kernel` attribute, enables range-loop lowering
- `@jit` -- emits a host-side `func.func` with `llvm.emit_c_interface`

### 3.3 `ExecutionEngineExecutor`

Wraps MLIR's `ExecutionEngine` for JIT execution:

```python
executor = compile(mod)
# Dynamic attribute lookup → calls compiled function
executor.my_kernel(tensor_a, tensor_b)
```

Features:
- Automatically resolves `_mlir_ciface_*` symbols
- Supports PyTorch tensors as arguments (auto-expands to memref descriptor)
- Handles both flattened and packed calling conventions

### 3.4 `Pipeline`

Fluent pass-pipeline builder (see Section 2.3).

### 3.5 `FileCache`

On-disk compilation cache (inspired by Triton):

- Cache key: SHA-256 of `(chip, pipeline, input_sha256, flydsl version, git commit, python version, soabi, platform)`
- Storage: `$FLIR_CACHE_DIR` or `$XDG_CACHE_HOME/flydsl/<key>/` (default: `~/.cache/flydsl/<key>/`)
- Atomic writes with file locking
- Controlled by environment variables (see Section 4)

---

## 4. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COMPILE_ONLY` | `0` | If `1`, compile without creating an executor. Returns `None`. |
| `FLIR_DUMP_IR` | `0` | If `1`, dump intermediate IR at each pipeline stage. |
| `FLIR_DUMP_DIR` | `my_ir_dumps` | Directory for IR dumps when `FLIR_DUMP_IR=1`. |
| `ARCH` | auto-detect | Override target GPU architecture (e.g., `gfx942`, `gfx950`). |
| `FLIR_CHIP` | -- | Alternative to `ARCH` for target chip (checked by `get_rocm_arch()`). |
| `FLIR_GPU_ARCH` | -- | Alternative to `ARCH` for target chip (checked by `get_rocm_arch()`). |
| `FLIR_CACHE_DIR` | `~/.cache/flydsl` | Override compilation cache directory. |
| `FLIR_NO_CACHE` | `0` | If `1`, disable compilation caching. |
| `FLIR_CACHE_DISABLE` | `0` | Alternative to `FLIR_NO_CACHE`. |
| `FLIR_REBUILD` | `0` | If `1`, force recompilation (ignore cache). |
| `FLIR_CACHE_REBUILD` | `0` | Alternative to `FLIR_REBUILD`. |

### Architecture Detection Priority

`get_rocm_arch()` in `runtime/device.py` checks in order:
1. `FLIR_CHIP` env var
2. `FLIR_GPU_ARCH` env var
3. `HSA_OVERRIDE_GFX_VERSION` env var (supports `9.4.2` → `gfx942` format)
4. PyTorch `torch.cuda.get_device_properties().gcnArchName`
5. Default: `gfx942`

---

## 5. Target Hardware

| Architecture | GPU | LDS per CU | Notes |
|---|---|---|---|
| `gfx942` | MI300A / MI300X | 64 KB | CDNA 3, primary development target |
| `gfx950` | MI350 | 160 KB | CDNA 4, larger LDS |
| `gfx90a` | MI250X | 64 KB | CDNA 2 (verified platform) |

---

## 6. IR Dump Workflow

Enable with `FLIR_DUMP_IR=1`:

```bash
FLIR_DUMP_IR=1 FLIR_DUMP_DIR=./dumps python test_my_kernel.py
```

Produces numbered `.mlir` files:
```
dumps/my_kernel/
├── 00_target_overridden.mlir
├── 03_flir_to_standard.mlir
├── 04_trivial_dce.mlir
├── 05_canonicalize.mlir
├── 06_cse.mlir
├── 07_gpu_kernel_outlining.mlir
├── 08_convert_scf_to_cf.mlir
├── 09_convert_gpu_to_rocdl.mlir
├── 10_reconcile_unrealized_casts.mlir
├── 11_rocdl_attach_target.mlir
├── 12_gpu_to_llvm.mlir
├── 13_reconcile_unrealized_casts.mlir
├── 14_gpu_module_to_binary.mlir
└── 15_final_isa.s                  # AMD ISA assembly (best-effort)
```

---

## 7. Source Files

| File | Description |
|---|---|
| `flydsl/src/flydsl/compiler/compiler.py` | `flir.compile()` entry point, pipeline construction, IR dump logic |
| `flydsl/src/flydsl/compiler/pipeline.py` | `Pipeline` fluent API, `run_pipeline()`, `lower_flir_to_standard()` |
| `flydsl/src/flydsl/compiler/executor.py` | `ExecutionEngineExecutor`, shared library resolution |
| `flydsl/src/flydsl/compiler/context.py` | `RAIIMLIRContext`, `RAIIMLIRContextModule`, dialect registration |
| `flydsl/src/flydsl/compiler/cache.py` | `FileCache`, `make_cache_key()`, cache env var handling |
| `flydsl/src/flydsl/runtime/device.py` | `get_rocm_arch()` GPU detection |
| `flydsl/src/flydsl/lang/ir/module.py` | `MlirModule`, `@kernel`, `@jit` decorators |
| `flir/include/flir/FlirPasses.td` | Pass declarations (flir-to-standard, trivial-dce) |
| `flir/lib/Transforms/FlirToStandard.cpp` | FLIR → standard lowering implementation |
| `flir/lib/Transforms/FlirDCE.cpp` | Dead code elimination implementation |
