# Architecture & compilation pipeline guide

This guide covers the FlyDSL project structure, compilation stages, key abstractions, and environment configuration.

## Quick Reference

| Component | Description | Key File |
|---|---|---|
| **FlyDSL** | Python DSL front-end for authoring GPU kernels | `python/flydsl/` |
| **FlyDSL Compiler** | `@flyc.jit` / `@flyc.kernel` — trace-based JIT compiler | `python/flydsl/compiler/` |
| **FlyDSL Expr** | DSL expression ops (arith, vector, gpu, buffer, rocdl) | `python/flydsl/expr/` |
| **Fly Dialect** | Flexible Layout IR — MLIR dialect with layout algebra | `include/flydsl/Dialect/Fly/` |
| **MlirCompiler** | End-to-end MLIR pass pipeline (DSL → binary) | `python/flydsl/compiler/jit_function.py` |
| **JITCFunction** | MLIR ExecutionEngine wrapper for JIT execution | `python/flydsl/compiler/jit_executor.py` |

---

## 1. Project structure

```
FlyDSL/
├── include/flydsl/              # C++/TableGen declarations for Fly and FlyROCDL
├── lib/                         # dialect, lowering, C API, and runtime implementation
├── python/
│   ├── flydsl/
│   │   ├── compiler/            # JIT/AOT tracing, backends, argument adapters
│   │   ├── expr/                # language values, layout algebra, GPU/ROCDL ops
│   │   ├── extension/           # cooperative and random algorithm libraries
│   │   ├── runtime/             # device runtime and AOT library discovery
│   │   └── autotune.py          # search, validation, cache, and artifacts
│   └── mlir_flydsl/             # generated/built Python bindings
├── tools/                       # fly-opt and flydsl-lsp-server
├── examples/                    # runnable examples and onboarding notebooks
├── kernels/
│   ├── gemm/                    # CDNA/RDNA/gfx1250 GEMM implementations
│   ├── norm/                    # LayerNorm, RMSNorm, and softmax
│   ├── attention/               # FlashAttention, paged attention, MLA, RoPE
│   ├── moe/ and mega_moe/       # routing, expert GEMMs, fused MoE
│   ├── conv/                    # BF16/FP8 implicit-GEMM convolution
│   ├── comm/                    # all-reduce and intranode dispatch/combine
│   └── common/                  # shared memory, layout, DPP, and MMA helpers
├── tests/                       # language, extension, kernel, system, and MLIR tests
├── docs/                        # Sphinx documentation source
└── scripts/                     # build, test, benchmark, release, and checks
```

---

## 2. Architecture

The user-facing API lives in `python/flydsl/`. Kernel authors use `@flyc.jit` and `@flyc.kernel` decorators with expression operations from `flydsl.expr`:

- **Traces** Python functions via AST rewriting and execution
- **Generates** Fly dialect ops and standard MLIR dialects (gpu, arith, scf, memref, vector, rocdl)
- **Compiles** through the `MlirCompiler` pass pipeline (Fly → ROCDL → LLVM → HSACO)
- **Caches** compiled kernels to disk for fast reuse
- **Executes** via MLIR ExecutionEngine

The Fly dialect (`include/flydsl/Dialect/Fly/`) provides the MLIR-level layout algebra (composition, product, divide, coordinate mapping). Python DSL operations in `flydsl.expr` lower to Fly dialect ops during tracing, and are then compiled through the `MlirCompiler` pipeline.

---

## 3. Compilation pipeline

### 3.1 High-level flow

```
Python Function (@flyc.kernel / @flyc.jit)
        │
        ▼  AST Rewriting
   Transformed Python Function
        │
        ▼  Tracing (execution inside MLIR Context)
   MLIR Module (fly, gpu, arith, scf, memref, vector dialects)
        │
        ▼  MlirCompiler.compile()
   ┌────────────────────────────────────────────────────────┐
   │ Stage A — pre_binary_fragments  (Fly → ROCDL)          │
   │   fly-rewrite-func-signature                           │
   │   fly-canonicalize                                     │
   │   fly-layout-lowering                                  │
   │   fly-int-swizzle-simplify                             │
   │   canonicalize                                         │
   │   fly-convert-atom-call-to-ssa-form                    │
   │   fly-promote-regmem-to-vectorssa                      │
   │   convert-fly-to-rocdl                                 │
   │   canonicalize                                         │
   │   gpu.module(convert-scf-to-cf, cse,                   │
   │              convert-rocdl-fastmath-ops,              │
   │              convert-gpu-to-rocdl{chipset=gfxNNN ...}, │
   │              fly-rocdl-cluster-attr)                   │
   ├────────────────────────────────────────────────────────┤
   │ Stage B — binary_prep_fragments  (→ LLVM)              │
   │   rocdl-attach-target{chip=gfxNNN ...}                 │
   │   convert-scf-to-cf                                    │
   │   convert-cf-to-llvm                                   │
   │   gpu-to-llvm{use-bare-pointers-...=true}              │
   │   convert-vector-to-llvm                               │
   │   convert-arith-to-llvm                                │
   │   convert-func-to-llvm                                 │
   │   reconcile-unrealized-casts                           │
   │   ensure-debug-info-scope-on-llvm-func  (optional)     │
   ├────────────────────────────────────────────────────────┤
   │ Stage C — binary_fragment                              │
   │   gpu-module-to-binary{format=fatbin opts="..."}       │
   └────────────────────────────────────────────────────────┘
        │
        ▼
   JITCFunction (ExecutionEngine)
```

### 3.2 Pipeline stages in detail

The pipeline is built by `RocmBackend._pipeline_parts()` in
`python/flydsl/compiler/backends/rocm.py`. The orchestrator
`_pipeline_fragments_for_mode()` in `jit_function.py` decides whether to run
the pipeline as a single combined pass list (`pipeline_fragments()`) or split
it for external LLVM codegen (`external_binary_pipeline_fragments()`). External
mode runs Stages A and B with the bundled MLIR runtime, then invokes the
external LLVM toolchain only for Stage C (`gpu-module-to-binary`).

**Stage A — `pre_binary_fragments`** (Fly dialect → ROCDL lowering)

The tables explain the stages conceptually. The following excerpt is included
directly from the backend at documentation build time, so it is the exact pass
definition for this checkout:

```{literalinclude} ../python/flydsl/compiler/backends/rocm.py
:language: python
:start-at:         pre_binary_fragments = [
:end-before:         return [*pre_binary_fragments
:dedent: 8
```

| # | Pass | Description |
|---|---|---|
| 1 | `fly-rewrite-func-signature` | Rewrite DSL types at function and SCF control-flow boundaries; lowers `IntTuple` / `Layout` / `ComposedLayout` / `CoordTensor` / `MemRef` to packed LLVM struct types and reconstructs them in the body via constructor ops. |
| 2 | `fly-canonicalize` | FlyDSL-specific canonicalization (folds `!fly.layout` algebra when shapes are static). |
| 3 | `fly-layout-lowering` | Lowers layout algebra (`fly.crd2idx`, partitions, divides) to concrete `arith` + `vector` ops. |
| 4 | `fly-int-swizzle-simplify` | Algebraically simplifies the swizzle-shaped arith sequences emitted by `applySwizzle`. |
| 5 | `canonicalize` | Standard MLIR canonicalization (constant folding, etc.). |
| 6 | `fly-convert-atom-call-to-ssa-form` | Converts `copy_atom_call` / `mma_atom_call` to their SSA counterparts; promotes register tensors to vector SSA values. |
| 7 | `fly-promote-regmem-to-vectorssa` | Promotes `fly.make_ptr(register)` memory semantics to vector SSA values (requires #6). |
| 8 | `convert-fly-to-rocdl` | Lowers remaining Fly ops to upstream MLIR and ROCDL operations: buffer/TDM copies plus MFMA or WMMA instructions selected for the target. |
| 9 | `canonicalize` | Second canonicalization round after ROCDL lowering. |
| 10 | `gpu.module(convert-scf-to-cf, cse, convert-rocdl-fastmath-ops, convert-gpu-to-rocdl{chipset=gfxNNN ...}, fly-rocdl-cluster-attr)` | Inside the GPU module: SCF→CF, CSE, ROCDL fast-math ops lowering, GPU intrinsics→ROCDL, then `fly-rocdl-cluster-attr` injects `amdgpu-cluster-dims` into the `llvm.func` `passthrough`. |

**Stage B — `binary_prep_fragments`** (LLVM lowering, host + kernel)

| # | Pass | Description |
|---|---|---|
| 11 | `rocdl-attach-target{chip=gfxNNN ...}` | Attaches `#rocdl.target<chip=gfxNNN>` (plus `fast`/`unsafe-math`/`wave64` options) to the GPU module for codegen. |
| 12 | `convert-scf-to-cf` | Host-side SCF → ControlFlow. |
| 13 | `convert-cf-to-llvm` | ControlFlow → LLVM dialect. |
| 14 | `gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}` | GPU types and host launcher → LLVM. |
| 15 | `convert-vector-to-llvm` | Vector → LLVM. |
| 16 | `convert-arith-to-llvm` | Arith → LLVM. |
| 17 | `convert-func-to-llvm` | Func → LLVM. |
| 18 | `reconcile-unrealized-casts` | Final cast cleanup. |

When `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`, Stage B appends
`ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}` after
`reconcile-unrealized-casts` and before Stage C.

**Stage C — `binary_fragment`**

| # | Pass | Description |
|---|---|---|
| 19 | `gpu-module-to-binary{format=fatbin opts="..."}` | Invokes the LLVM AMDGPU backend and emits an HSA fatbin. |

`gpu-kernel-outlining` is no longer a pass in the runtime pipeline. Kernel
outlining happens during Python tracing, when `@flyc.kernel` emits
`gpu.func` ops directly into a `gpu.container_module`.

### 3.3 JIT compilation flow

When a `@flyc.jit` function is called:

1. **Cache check**: Look up by argument type signature (in-memory → disk).
2. **AST rewriting**: `ASTRewriter.transform` converts Python `for`/`if` to MLIR `scf.for`/`scf.if`.
3. **MLIR module creation**: Sets up `gpu.container_module` with target.
4. **Argument conversion**: `convert_to_jit_arguments` maps Python args to IR types.
5. **Function tracing**: Execute the transformed function body to generate MLIR ops.
6. **GPU kernel emission**: `@kernel` calls emit `gpu.func` into `gpu.module`.
7. **Pipeline compilation**: `MlirCompiler.compile()` runs the full pass pipeline.
8. **Execution**: `JITCFunction` wraps MLIR ExecutionEngine to invoke the compiled code.
9. **Cache store**: Serialize the compiled function to disk for future runs.

### 3.4 AOT export flow

`flyc.compile_aot(launcher, *args)` runs steps 2-7 through the same
`JitFunction._trace_and_compile` helper as the JIT and stops there: no cache,
no ExecutionEngine, no launch. It records the ABI of the lowered entry
(`AbiSlot` per packed argument). The callable returned by `flyc.compile` builds
the same export view from its already-lowered artifact, without a second
compile. `export_to_c(file_path, file_name, function_prefix)` then works on a
fresh copy of the lowered module:

1. **Namespace**: every defined symbol becomes `<name>__<symbol>` with internal
   linkage (references follow through the symbol table; external declarations
   keep their names).
2. **Offloading handler**: the `gpu.binary` switches to `#fly.aot_module<"name">`,
   whose LLVM translation embeds the binary and emits `<name>__module_init/load/unload`
   backed by the per-device module table in `libfly_jit_runtime.so`.
3. **Entry wrapper**: `int32_t <name>(void **args)` unpacks the slots, calls the
   launcher and returns the first runtime error recorded during the call.
4. **Metadata**: the JSON result is embedded as `<name>__metadata`.
5. **Emission**: `emit_host_object` (C API `flydslEmitHostObject`,
   `lib/CAPI/HostObject/`) translates the module to LLVM IR and emits a PIC
   object for the generic CPU of the host triple, in process. The exported
   global symbols must be exactly the entry, lifecycle and metadata symbols
   before the object is published atomically.

See [`aot_export_guide.md`](aot_export_guide.md) for the user-facing contract.

---

## 4. Key abstractions

### 4.1 `@flyc.jit` — Host launcher

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
- The first call triggers compilation; subsequent calls with the same type signature use the cached binary.
- `Constexpr[T]` parameters become compile-time constants and affect the cache key.
- `Tensor` parameters map to memref descriptors via DLPack.
- `Stream` parameters pass a CUDA/HIP stream to the GPU runtime.
- When called inside an existing MLIR context, the function acts as a normal composable function.

### 4.2 `@flyc.kernel` — GPU kernel

Decorates a Python function as a GPU kernel:

```python
@flyc.kernel
def my_kernel(a: fx.Tensor, b: fx.Tensor, n: fx.Constexpr[int]):
    tid = fx.gpu.thread_id("x")
    bid = fx.gpu.block_id("x")
    # ... kernel body ...
```

Key behaviors:
- You can only call this inside a `@flyc.jit` function.
- Calling the kernel returns a `KernelLauncher`; you must call `.launch()` to emit the launch op.
- Supports `Constexpr[T]` for compile-time specialization.
- Emits a `gpu.func` with `gpu.kernel` attribute into the `gpu.module`.

### 4.3 `KernelLauncher`

Calling a `@kernel` function returns a `KernelLauncher`. Use `.launch()` to configure and emit the GPU launch:

```python
launcher = my_kernel(a, b, 1024)
launcher.launch(
    grid=(num_blocks, 1, 1),
    block=(256, 1, 1),
    smem=shared_mem_bytes,
    stream=stream_value,
)
```

### 4.4 `JITCFunction`

Wraps MLIR's `ExecutionEngine` for JIT execution:

- Thread-safe with lazy engine initialization.
- Serializable (pickle) for disk caching.
- Supports packed calling convention via `ctypes`.
- Provides `.print_ir()` for debugging compiled or original IR.

### 4.5 `DslType` / `JitArgument` protocols

Extensible type system for mapping Python values to MLIR. The language-level
contracts, including `Storable`, are in [DSL protocols](language/dsl_protocols.md).

```python
# DslType protocol — for values used inside kernel/jit functions
class DslType(Protocol):
    @classmethod
    def __construct_from_ir_values__(
        cls, values: List[ir.Value], exemplar: "DslType | None" = None
    ) -> "DslType": ...
    def __extract_to_ir_values__(self) -> List[ir.Value]: ...

# JitArgument protocol — for values passed at the host boundary
class JitArgument(Protocol):
    def __get_ir_types__(self) -> List[ir.Type]: ...
    def __cache_signature__(self) -> object: ...
    def __c_abi_spec__(self) -> List[Tuple[type, Callable]]: ...
```

Built-in types: `Tensor`, `Stream`, `Int32`, and `Constexpr[T]`

To register custom types:
```python
from flydsl.compiler import JitArgumentRegistry

@JitArgumentRegistry.register(MyPythonType, dsl_type=MyDslType)
class MyJitArg:
    def __get_ir_types__(self): ...
    def __cache_signature__(self): ...
    def __c_abi_spec__(self): ...
```

`__c_abi_spec__()` returns ordered `(ctype, fill)` slots for the packed C
interface; they need not be one-to-one with `__get_ir_types__()`. Do not
implement `__get_c_pointers__` — that hook has been removed.

### 4.6 `ASTRewriter`

Transforms Python control flow to MLIR ops at the AST level:

- `for i in range(n)` → `scf.for`
- `for i in range_constexpr(n)` → compile-time unrolled loop
- `if condition` → `scf.if`
- `const_expr(value)` → compile-time constant

---

## 5. Environment variables

### 5.1 Compilation options (`FLYDSL_COMPILE_*`)

| Variable | Default | Description |
|---|---|---|
| `FLYDSL_COMPILE_OPT_LEVEL` | `2` | Optimization level (0–3) |
| `COMPILE_ONLY` | `0` | If `1`, compile without creating an executor. Returns `None`. |
| `ARCH` | auto-detect | Override target GPU architecture (e.g., `gfx942`, `gfx950`). |

### 5.2 Debug options (`FLYDSL_DEBUG_*`)

| Variable | Default | Description |
|---|---|---|
| `FLYDSL_DUMP_IR` | `false` | Dump intermediate IR at each pipeline stage. |
| `FLYDSL_DUMP_DIR` | `~/.flydsl/debug` | Directory for IR dumps. |
| `FLYDSL_DEBUG_DUMP_ASM` | `false` | Dump final AMD ISA assembly. |
| `FLYDSL_DEBUG_AST_DIFF` | `false` | Print AST diff during rewrite. |
| `FLYDSL_DEBUG_PRINT_ORIGIN_IR` | `false` | Print origin IR before compilation. |
| `FLYDSL_DEBUG_PRINT_AFTER_ALL` | `false` | Print IR after each MLIR pass. |
| `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` | `false` | Generate debug info in compiled code. |
| `FLYDSL_DEBUG_ENABLE_VERIFIER` | `true` | Verify IR module. |
| `FLYDSL_DEBUG_LOG_LEVEL` | `WARNING` | Logging level (DEBUG, INFO, WARNING, ERROR). |

### 5.3 Runtime options (`FLYDSL_RUNTIME_*`)

| Variable | Default | Description |
|---|---|---|
| `FLYDSL_RUNTIME_CACHE_DIR` | `~/.flydsl/cache` | Directory for caching compiled kernels. |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | `true` | Enable kernel disk caching (in-memory cache is always active). |

### 5.4 Architecture detection priority

`get_rocm_arch()` in `runtime/device.py` checks in the following order:
1. `ARCH` compile-target override
2. `FLYDSL_GPU_ARCH` env var
3. `HSA_OVERRIDE_GFX_VERSION` env var (supports `9.4.2` → `gfx942` format)
4. `rocm_agent_enumerator` system tool
5. Default: `gfx942`

Setting `ARCH` makes compilation independent of GPU discovery. In particular,
`flyc.compile_aot(...).export_to_c(...)` can complete device code generation,
host object emission and CPU linking when the build host has no visible GPU.
The normal `flyc.compile(...)` path still performs its initial launch and is
therefore not the CPU-only AOT entry point.

---

## 6. Target hardware

| Architecture | GPU | LDS per CU | Notes |
|---|---|---|---|
| `gfx942` | MI300A / MI300X | 64 KB | CDNA 3, primary development target |
| `gfx950` | MI350 / MI355X | 160 KB | CDNA 4, larger LDS |
| `gfx1201` | Radeon AI PRO R9700 | 64 KB | RDNA 4 |
| `gfx1250` | — | 320 KB | GFX12, wave32, WMMA, TDM ops |
| `gfx90a` | MI250X | 64 KB | CDNA 2 (verified platform) |

---

## 7. IR dump workflow

Enable with `FLYDSL_DUMP_IR=1`:

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python test_my_kernel.py
```

This produces numbered dump files (exact pass count tracks `RocmBackend._pipeline_parts()`):
```
dumps/my_func_name/
├── 00_origin.mlir
├── 01_fly_rewrite_func_signature.mlir
├── 02_fly_canonicalize.mlir
├── 03_fly_layout_lowering.mlir
├── 04_fly_int_swizzle_simplify.mlir
├── 05_canonicalize.mlir
├── 06_fly_convert_atom_call_to_ssa_form.mlir
├── 07_fly_promote_regmem_to_vectorssa.mlir
├── 08_convert_fly_to_rocdl.mlir
├── 09_canonicalize.mlir
├── 10_convert_scf_to_cf_cse_convert_gpu_to_rocdl.mlir
│                                      # also runs fly-rocdl-cluster-attr
├── 11_rocdl_attach_target.mlir
├── 12_convert_scf_to_cf.mlir
├── 13_convert_cf_to_llvm.mlir
├── 14_gpu_to_llvm.mlir
├── 15_convert_vector_to_llvm.mlir
├── 16_convert_arith_to_llvm.mlir
├── 17_convert_func_to_llvm.mlir
├── 18_reconcile_unrealized_casts.mlir
├── 19_gpu_module_to_binary.mlir
├── 20_llvm_ir.ll
└── 21_final_isa.s                    # AMD ISA assembly (best-effort)
```

If `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`, the debug-info pass adds an extra numbered dump before `gpu_module_to_binary`.

---

## 8. Source files

| File | Description |
|---|---|
| `python/flydsl/compiler/jit_function.py` | `@jit` decorator, `MlirCompiler`, `JitCacheManager` |
| `python/flydsl/compiler/kernel_function.py` | `@kernel` decorator, `KernelFunction`, `KernelLauncher`, `CompilationContext` |
| `python/flydsl/compiler/jit_executor.py` | `JITCFunction` — ExecutionEngine wrapper |
| `python/flydsl/compiler/aot.py` | `compile_aot`, `AOTCompiledFunction.export_to_c`, ABI model |
| `python/flydsl/compiler/aot_config.py` | Linker-flags CLI for exported objects |
| `python/flydsl/runtime/libraries.py` | `find_runtime_libraries()` |
| `python/flydsl/compiler/jit_argument.py` | `JitArgumentRegistry`, `TensorAdaptor`, `from_dlpack` |
| `python/flydsl/compiler/ast_rewriter.py` | `ASTRewriter` — Python AST → MLIR control flow |
| `python/flydsl/compiler/protocol.py` | `get_ir_types`, `extract_to_ir_values`, `construct_from_ir_values` protocols |
| `python/flydsl/expr/typing.py` | `Types` (`T`), `Tensor`, `Stream`, `Constexpr` |
| `python/flydsl/expr/primitive.py` | Layout algebra primitives (make_shape, crd2idx, copy, gemm) |
| `python/flydsl/expr/derived.py` | Derived types (`CopyAtom`, `MmaAtom`, `TiledCopy`) |
| `python/flydsl/expr/numeric.py` | DSL numeric types (Float32, Int32, ...) |
| `python/flydsl/utils/env.py` | `EnvManager` — typed environment variable configuration |
| `python/flydsl/runtime/device.py` | `get_rocm_arch()` GPU detection |
| `include/flydsl/Dialect/Fly/IR/FlyOps.td` | Fly dialect op definitions |
| `include/flydsl/Dialect/Fly/Transforms/Passes.td` | Pass declarations (fly-layout-lowering, etc.) |
