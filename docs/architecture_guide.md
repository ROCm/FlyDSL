# Architecture & Compilation Pipeline Guide

> FlyDSL project structure, compilation stages, key abstractions, and configuration.

## Quick Reference

| Component | Description | Key File |
|---|---|---|
| **FlyDSL** | Python DSL front-end for authoring GPU kernels | `python/flydsl/` |
| **FlyDSL Compiler** | `@flyc.jit` / `@flyc.kernel` — trace-based JIT compiler | `python/flydsl/compiler/` |
| **FlyDSL Expr** | DSL expression ops (arith, vector, gpu, buffer, rocdl) | `python/flydsl/expr/` |
| **FLIR** | Flexible Layout IR — MLIR dialect with layout algebra | `flir/` |
| **FLIR Backend** | `flydsl_` — MLIR dialect extensions and `MlirModule` (legacy API) | `flydsl_/src/flydsl_/` |
| **MlirCompiler** | End-to-end MLIR pass pipeline (DSL → binary) | `python/flydsl/compiler/jit_function.py` |
| **JitCompiledFunction** | MLIR ExecutionEngine wrapper for JIT execution | `python/flydsl/compiler/jit_executor.py` |

---

## 1. Project Structure

```
FlyDSL/
├── flir/                             # C++ MLIR dialect + compiler infrastructure
│   ├── include/flir/
│   │   ├── FlirOps.td                # FLIR layout ops (make_shape, crd2idx, composition, ...)
│   │   ├── FlirTypeDefs.td           # Custom types (!flir.shape, !flir.layout, ...)
│   │   ├── FlirAttrDefs.td           # Attributes (#flir.underscore, #flir.dync_i32)
│   │   ├── FlirPasses.td             # Pass declarations (flir-to-standard, trivial-dce)
│   │   ├── FlirRocmOps.td            # ROCm ops (MFMA, LDS, copy, barriers)
│   │   └── FlirRocmDialect.td        # ROCm dialect declaration
│   ├── lib/Dialect/
│   │   ├── Flir/
│   │   │   ├── FlirOps.cpp           # Op verifiers and builders
│   │   │   ├── FlirLayoutAlgebra.cpp  # Type inference for composition/product/divide
│   │   │   └── FlirDialect.cpp       # Dialect registration
│   │   └── FlirRocm/                 # ROCm dialect implementation
│   ├── lib/Transforms/
│   │   ├── FlirToStandard.cpp        # flir-to-standard lowering pass
│   │   └── FlirDCE.cpp               # trivial-dce pass
│   ├── python_bindings/              # Python ↔ C++ bridge
│   │   ├── dialects/flir.py          # Low-level Python bindings for FLIR ops
│   │   └── FlirRegisterPasses.cpp    # Register passes and dialects from Python
│   ├── tools/flir-opt/               # CLI tool for running passes on .mlir files
│   └── build.sh                      # Build script (CMake + ninja)
│
├── python/flydsl/                    # High-level Python DSL package (NEW)
│   ├── __init__.py                   # Package version
│   ├── compiler/
│   │   ├── __init__.py               # Public API: jit, kernel, from_dlpack
│   │   ├── jit_function.py           # @jit decorator, MlirCompiler, JitCacheManager
│   │   ├── kernel_function.py        # @kernel decorator, KernelFunction, KernelLauncher
│   │   ├── jit_executor.py           # JitCompiledFunction (ExecutionEngine wrapper)
│   │   ├── jit_argument.py           # Argument conversion (Tensor, Stream, Int32)
│   │   ├── ast_rewriter.py           # AST rewriting for Python control flow → MLIR
│   │   └── protocol.py              # DslType / JitArgument protocols
│   ├── expr/
│   │   ├── __init__.py               # Public expr API
│   │   ├── typing.py                 # Types (T.f32, Tensor, Stream, Constexpr)
│   │   ├── numeric.py                # DSL numeric types (Float32, Int32, ...)
│   │   ├── primitive.py              # Primitive operations
│   │   ├── derived.py                # Derived / composite operations
│   │   ├── arith.py                  # Arithmetic dialect ops
│   │   ├── vector.py                 # Vector dialect ops
│   │   ├── gpu.py                    # GPU dialect ops (thread_id, block_id, barrier)
│   │   ├── buffer_ops.py             # Buffer / memory operations
│   │   └── rocdl.py                  # ROCm-specific intrinsics
│   ├── lang/ir/
│   │   └── types.py                  # T / Types helper (backward-compat shim)
│   ├── runtime/
│   │   └── device.py                 # get_rocm_arch() — GPU architecture detection
│   └── utils/
│       ├── env.py                    # EnvManager — typed environment config
│       ├── logger.py                 # Logging utilities
│       └── smem_allocator.py         # SmemAllocator for LDS management
│
├── flydsl_/                          # MLIR backend package (legacy / lower-level)
│   └── src/flydsl_/
│       ├── compiler/
│       │   ├── compiler.py           # compile() for FLIR-based modules
│       │   ├── pipeline.py           # Pipeline fluent API
│       │   ├── executor.py           # ExecutionEngineExecutor
│       │   ├── context.py            # RAIIMLIRContext, RAIIMLIRContextModule
│       │   ├── cache.py              # On-disk compilation cache
│       │   └── flir_opt_helper.py    # Helper for invoking flir-opt
│       ├── dialects/ext/
│       │   ├── flir.py               # Layout algebra Python API
│       │   ├── gpu.py                # GPU dialect extensions
│       │   ├── rocm.py               # ROCm dialect helpers
│       │   └── ...                   # arith, scf, memref, vector, func, rocdl, ...
│       ├── lang/ir/
│       │   └── module.py             # MlirModule base class (legacy @kernel/@jit)
│       └── utils/
│           └── smem_allocator.py     # SmemAllocator
│
├── kernels/                          # Pre-built GPU kernels
│   ├── preshuffle_gemm_flyc.py       # GEMM with B-preshuffle (NEW flyc API)
│   ├── layernorm_kernel.py           # LayerNorm
│   ├── rmsnorm_kernel.py             # RMSNorm
│   ├── softmax_kernel.py             # Softmax
│   ├── reduce.py                     # Warp/block reduction primitives
│   ├── mfma_epilogues.py             # MFMA result writeback patterns
│   ├── mfma_preshuffle_pipeline.py   # Preshuffle helpers for MFMA kernels
│   └── layout_utils.py              # Layout computation utilities
│
├── tests/
│   ├── mlir/                         # MLIR IR tests (no GPU required)
│   ├── pyir/                         # Python IR tests (no GPU required)
│   ├── kernels/                      # GPU kernel tests + benchmarks
│   ├── python/                       # New Python DSL tests
│   │   ├── examples/                 # Example-based tests
│   │   ├── gpu/                      # GPU tests with new API
│   │   └── ir/                       # IR generation tests
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_common.py                # Shared test utilities
│   └── utils.py                      # Compilation helpers
│
└── scripts/                          # Build and test helpers
    ├── build.sh                      # Build FLIR (CMake + ninja)
    ├── build_llvm.sh                 # Build MLIR from ROCm llvm-project
    ├── run_tests.sh                  # Run GEMM test suite
    ├── run_benchmark.sh              # Run benchmarks
    └── dumpir.sh                     # Dump intermediate IR
```

---

## 2. Two-Layer Architecture

FlyDSL uses a two-layer architecture:

### Layer 1: High-Level Python DSL (`python/flydsl/`)

The user-facing API. Kernel authors use `@flyc.jit` and `@flyc.kernel` decorators with expression operations from `flydsl.expr`. This layer:

- **Traces** Python functions via AST rewriting and execution
- **Generates** standard MLIR dialects (gpu, arith, scf, memref, vector, rocdl)
- **Compiles** through the `MlirCompiler` pass pipeline
- **Caches** compiled kernels to disk for fast re-use
- **Executes** via MLIR ExecutionEngine

### Layer 2: MLIR Backend (`flydsl_/`)

The lower-level FLIR dialect API. Provides:

- `MlirModule` base class for structured IR construction
- FLIR layout algebra operations (composition, product, divide)
- `Pipeline` fluent API for custom pass pipelines
- Direct MLIR dialect extensions

Most new kernel development should use the Layer 1 API. Layer 2 is used internally and for kernels that need direct FLIR dialect access.

---

## 3. Compilation Pipeline

### 3.1 High-Level Flow

```
Python Function (@flyc.kernel / @flyc.jit)
        │
        ▼  AST Rewriting
   Transformed Python Function
        │
        ▼  Tracing (execution inside MLIR Context)
   MLIR Module (gpu, arith, scf, memref dialects)
        │
        ▼  MlirCompiler.compile()
   ┌────────────────────────────────────────────────┐
   │  gpu-kernel-outlining                          │  Outline GPU kernels
   │  fly-canonicalize                              │  FlyDSL-specific canonicalization
   │  fly-layout-lowering                           │  Layout algebra lowering
   │  convert-fly-to-rocdl                          │  Fly ops → ROCDL intrinsics
   │  canonicalize                                  │  Standard MLIR canonicalization
   │  gpu.module(convert-scf-to-cf,                 │  SCF → ControlFlow
   │             convert-gpu-to-rocdl{...})         │  GPU → ROCDL (inside gpu.module)
   │  rocdl-attach-target{chip=gfxNNN}              │  Attach ROCm target
   │  convert-scf-to-cf                             │  Host-side SCF → CF
   │  convert-cf-to-llvm                            │  CF → LLVM dialect
   │  gpu-to-llvm                                   │  GPU types → LLVM types
   │  convert-arith-to-llvm                         │  Arith → LLVM
   │  convert-func-to-llvm                          │  Func → LLVM
   │  reconcile-unrealized-casts                    │  Clean up casts
   │  gpu-module-to-binary{format=fatbin}           │  Emit HSACO binary
   └────────────────────────────────────────────────┘
        │
        ▼
   JitCompiledFunction (ExecutionEngine)
```

### 3.2 Pipeline Stages in Detail

The pipeline is defined in `MlirCompiler._pipeline_fragments()`:

| Stage | Pass | Description |
|---|---|---|
| 1 | `gpu-kernel-outlining` | Moves GPU kernel bodies into `gpu.func` inside `gpu.module`. |
| 2 | `fly-canonicalize` | FlyDSL-specific canonicalization (custom pass). |
| 3 | `fly-layout-lowering` | Lowers layout algebra operations to standard arithmetic. |
| 4 | `convert-fly-to-rocdl` | Converts FlyDSL ops to ROCDL intrinsics. |
| 5 | `canonicalize` | Standard MLIR canonicalization (constant folding, etc.). |
| 6 | `convert-scf-to-cf` + `convert-gpu-to-rocdl` | Lowers SCF and GPU ops to ROCDL (inside `gpu.module`). |
| 7 | `rocdl-attach-target` | Attaches `#rocdl.target<chip=gfxNNN>` for the target GPU. |
| 8 | `convert-scf-to-cf` | Host-side SCF lowering. |
| 9 | `convert-cf-to-llvm` | ControlFlow → LLVM dialect. |
| 10 | `gpu-to-llvm` | GPU types/ops → LLVM dialect (host-side launch). |
| 11 | `convert-arith-to-llvm` | Arithmetic → LLVM. |
| 12 | `convert-func-to-llvm` | Function → LLVM. |
| 13 | `reconcile-unrealized-casts` | Final cast cleanup. |
| 14 | `gpu-module-to-binary` | Compiles GPU module to HSACO binary (fatbin). |

### 3.3 JIT Compilation Flow

When a `@flyc.jit` function is called:

1. **Cache check** — look up by argument type signature (in-memory → disk)
2. **AST rewriting** — `ASTRewriter.transform` converts Python `for`/`if` to MLIR `scf.for`/`scf.if`
3. **MLIR module creation** — sets up `gpu.container_module` with target
4. **Argument conversion** — `convert_to_jit_arguments` maps Python args to IR types
5. **Function tracing** — execute transformed function body to generate MLIR ops
6. **GPU kernel emission** — `@kernel` calls emit `gpu.func` into `gpu.module`
7. **Pipeline compilation** — `MlirCompiler.compile()` runs the full pass pipeline
8. **Execution** — `JitCompiledFunction` wraps MLIR ExecutionEngine for invoking the compiled code
9. **Cache store** — compiled function is serialized to disk for future runs

---

## 4. Key Abstractions

### 4.1 `@flyc.jit` — Host Launcher

Decorates a Python function as a JIT-compiled host launcher:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.jit
def launch(a: fx.Tensor, b: fx.Tensor, n: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(a, b, n).launch(grid=(n // 256,), block=(256,), stream=stream)
```

Key behaviors:
- First call triggers compilation; subsequent calls with the same type signature use cached binary
- `Constexpr[T]` parameters become compile-time constants (affect cache key)
- `Tensor` parameters map to memref descriptors via DLPack
- `Stream` parameters pass CUDA/HIP stream to the GPU runtime
- When called inside an existing MLIR context, acts as a normal function (composable)

### 4.2 `@flyc.kernel` — GPU Kernel

Decorates a Python function as a GPU kernel:

```python
@flyc.kernel
def my_kernel(a: fx.Tensor, b: fx.Tensor, n: fx.Constexpr[int]):
    tid = fx.gpu.thread_id("x")
    bid = fx.gpu.block_id("x")
    # ... kernel body ...
```

Key behaviors:
- Can only be called inside a `@flyc.jit` function
- Calling returns a `KernelLauncher` — you must call `.launch()` to emit the launch op
- Supports `Constexpr[T]` for compile-time specialization
- Emits a `gpu.func` with `gpu.kernel` attribute into the `gpu.module`

### 4.3 `KernelLauncher`

Returned by calling a `@kernel` function. Use `.launch()` to configure and emit the GPU launch:

```python
launcher = my_kernel(a, b, 1024)
launcher.launch(
    grid=(num_blocks, 1, 1),
    block=(256, 1, 1),
    smem=shared_mem_bytes,
    stream=stream_value,
)
```

### 4.4 `JitCompiledFunction`

Wraps MLIR's `ExecutionEngine` for JIT execution:

- Thread-safe with lazy engine initialization
- Serializable (pickle) for disk caching
- Supports packed calling convention via `ctypes`
- Provides `.print_ir()` for debugging compiled/original IR

### 4.5 `DslType` / `JitArgument` Protocols

Extensible type system for mapping Python values to MLIR:

```python
# DslType protocol — for values used inside kernel/jit functions
class DslType(Protocol):
    @classmethod
    def __new_from_ir_values__(cls, values: List[ir.Value]) -> "DslType": ...
    def __extract_ir_values__(self) -> List[ir.Value]: ...

# JitArgument protocol — for values passed at the host boundary
class JitArgument(Protocol):
    def __ir_types__(self) -> List[ir.Type]: ...
    def __c_pointers__(self) -> List[ctypes.c_void_p]: ...
```

Built-in types: `Tensor`, `Stream`, `Int32`, `Constexpr[T]`

Register custom types:
```python
from flydsl.compiler import JitArgumentRegistry

@JitArgumentRegistry.register(MyPythonType, dsl_type=MyDslType)
class MyJitArg:
    def __ir_types__(self): ...
    def __c_pointers__(self): ...
```

### 4.6 `ASTRewriter`

Transforms Python control flow to MLIR ops at the AST level:

- `for i in range(n)` → `scf.for`
- `for i in range_constexpr(n)` → compile-time unrolled loop
- `if condition` → `scf.if`
- `const_expr(value)` → compile-time constant

### 4.7 `MlirModule` (Legacy API, in `flydsl_`)

The older structured kernel authoring API. Still used by some kernels:

```python
from flydsl_.lang.ir.module import MlirModule, kernel, jit
from flydsl_.dialects.ext import flir

class MyKernels(MlirModule):
    GPU_MODULE_NAME = "my_kernels"

    @kernel
    def my_kernel(self, A):
        tid = flir.thread_idx("x")
        # ...
```

---

## 5. Environment Variables

### 5.1 Compilation Options (`FLYDSL_COMPILE_*`)

| Variable | Default | Description |
|---|---|---|
| `FLYDSL_COMPILE_OPT_LEVEL` | `2` | Optimization level (0–3) |
| `COMPILE_ONLY` | `0` | If `1`, compile without creating an executor. Returns `None`. |
| `ARCH` | auto-detect | Override target GPU architecture (e.g., `gfx942`, `gfx950`). |

### 5.2 Debug Options (`FLYDSL_DEBUG_*`)

| Variable | Default | Description |
|---|---|---|
| `FLYDSL_DUMP_IR` | `false` | Dump intermediate IR at each pipeline stage. |
| `FLYDSL_DUMP_DIR` | `~/.flydsl/debug` | Directory for IR dumps. |
| `FLYDSL_DEBUG_DUMP_ASM` | `false` | Dump final AMD ISA assembly. |
| `FLYDSL_DEBUG_AST_DIFF` | `false` | Print AST diff during rewrite. |
| `FLYDSL_DEBUG_PRINT_ORIGIN_IR` | `false` | Print origin IR before compilation. |
| `FLYDSL_DEBUG_PRINT_AFTER_ALL` | `false` | Print IR after each MLIR pass. |
| `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` | `true` | Generate debug info in compiled code. |
| `FLYDSL_DEBUG_ENABLE_VERIFIER` | `true` | Verify IR module. |
| `FLYDSL_DEBUG_LOG_LEVEL` | `WARNING` | Logging level (DEBUG, INFO, WARNING, ERROR). |

### 5.3 Runtime Options (`FLYDSL_RUNTIME_*`)

| Variable | Default | Description |
|---|---|---|
| `FLYDSL_RUNTIME_CACHE_DIR` | `~/.flydsl/cache` | Directory for caching compiled kernels. |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | `true` | Enable kernel caching. |

### 5.4 Architecture Detection Priority

`get_rocm_arch()` in `runtime/device.py` checks in order:
1. `FLYDSL_GPU_ARCH` env var
2. `HSA_OVERRIDE_GFX_VERSION` env var (supports `9.4.2` → `gfx942` format)
3. `rocm_agent_enumerator` system tool
4. Default: `gfx942`

---

## 6. Target Hardware

| Architecture | GPU | LDS per CU | Notes |
|---|---|---|---|
| `gfx942` | MI300A / MI300X | 64 KB | CDNA 3, primary development target |
| `gfx950` | MI350 | 160 KB | CDNA 4, larger LDS |
| `gfx90a` | MI250X | 64 KB | CDNA 2 (verified platform) |

---

## 7. IR Dump Workflow

Enable with `FLYDSL_DUMP_IR=1`:

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python test_my_kernel.py
```

Produces numbered `.mlir` files:
```
dumps/my_func_name/
├── 00_original.mlir
├── 01_gpu-kernel-outlining.mlir
├── 02_fly-canonicalize.mlir
├── 03_fly-layout-lowering.mlir
├── 04_convert-fly-to-rocdl.mlir
├── 05_canonicalize.mlir
├── 06_convert-scf-to-cf.mlir
├── 07_rocdl-attach-target.mlir
├── 08_convert-scf-to-cf.mlir
├── 09_convert-cf-to-llvm.mlir
├── 10_gpu-to-llvm.mlir
├── 11_convert-arith-to-llvm.mlir
├── 12_convert-func-to-llvm.mlir
├── 13_reconcile-unrealized-casts.mlir
├── 14_gpu-module-to-binary.mlir
└── final_isa.s                      # AMD ISA assembly (best-effort)
```

---

## 8. Source Files

| File | Description |
|---|---|
| `python/flydsl/compiler/jit_function.py` | `@jit` decorator, `MlirCompiler`, `JitCacheManager` |
| `python/flydsl/compiler/kernel_function.py` | `@kernel` decorator, `KernelFunction`, `KernelLauncher`, `CompilationContext` |
| `python/flydsl/compiler/jit_executor.py` | `JitCompiledFunction` — ExecutionEngine wrapper |
| `python/flydsl/compiler/jit_argument.py` | `JitArgumentRegistry`, `TensorAdaptor`, `from_dlpack` |
| `python/flydsl/compiler/ast_rewriter.py` | `ASTRewriter` — Python AST → MLIR control flow |
| `python/flydsl/compiler/protocol.py` | `DslType`, `JitArgument` protocols |
| `python/flydsl/expr/typing.py` | `Types` (`T`), `Tensor`, `Stream`, `Constexpr` |
| `python/flydsl/expr/numeric.py` | DSL numeric types (Float32, Int32, ...) |
| `python/flydsl/utils/env.py` | `EnvManager` — typed environment variable configuration |
| `python/flydsl/runtime/device.py` | `get_rocm_arch()` GPU detection |
| `flydsl_/src/flydsl_/compiler/compiler.py` | `compile()` for FLIR-based modules (legacy) |
| `flydsl_/src/flydsl_/compiler/pipeline.py` | `Pipeline` fluent API (legacy) |
| `flydsl_/src/flydsl_/lang/ir/module.py` | `MlirModule`, legacy `@kernel` / `@jit` decorators |
| `flir/include/flir/FlirPasses.td` | Pass declarations (flir-to-standard, trivial-dce) |
| `flir/lib/Transforms/FlirToStandard.cpp` | FLIR → standard lowering implementation |
