# The Compilation Pipeline

This chapter follows a kernel from Python source to the HSACO code object the GPU
runs, naming every stage. If you know how `hipcc` turns a `.cpp` into a fat binary,
this is the FlyDSL equivalent — with the twist that most of it happens **at
runtime**, the first time you call the kernel, and is then cached.

## What MLIR is

Every stage below produces or consumes **MLIR**, so it is worth one paragraph
before we start. MLIR (*Multi-Level Intermediate Representation*) is a subproject
of LLVM — the same LLVM whose IR `hipcc` already lowers your kernels to. The
difference is in the name: LLVM IR is *one* fixed, low-level instruction set,
whereas MLIR is **extensible**. Its ops, types, and attributes are grouped into
namespaces called **dialects**, and multiple dialects coexist in a single module
at different levels of abstraction. FlyDSL leans on exactly that: a high-level
`fly` dialect that speaks in layouts and tiles, standard mid-level dialects
(`arith`, `scf`, `memref`, `vector`, `gpu`) for arithmetic and control flow, and
low-level `rocdl`/`llvm` for the hardware.

Compilation is then **progressive lowering**: passes rewrite ops from higher
dialects into lower ones, one small typed step at a time, until only `llvm`/`rocdl`
remains. This is the key contrast with the `hipcc` world you know — Clang lowers
C++ to LLVM IR in a single large jump, discarding your tiling structure on the
way, while MLIR keeps that structure as *real ops* through many verifiable steps.
And because every dialect has a textual form that MLIR can print and re-parse,
you can dump and read the IR after any stage (§2.8) — which is what makes the rest
of this chapter, and the layout algebra in Chapters 3–6, inspectable rather than
opaque.

> **HIP/CK-Tile → FlyDSL.** Think of LLVM IR as a single language and MLIR as a
> family of interoperating languages sharing one file. Your `fly.*` tiling ops are
> a dialect that simply does not exist in the `hipcc` pipeline; they are lowered
> away by passes instead of by the C++ template expander.

## The big picture

```
  Python function  (@flyc.jit + @flyc.kernel)
        │  (1) AST rewriting            python-level, before tracing
        ▼
  Rewritten Python
        │  (2) Tracing                  execute body once → emit MLIR ops
        ▼
  MLIR module  (fly + gpu + arith + scf + memref + vector dialects)
        │  (3) Pass pipeline            MlirCompiler.compile()
        │      Stage A: Fly → ROCDL
        │      Stage B: ROCDL/GPU/SCF → LLVM dialect
        │      Stage C: LLVM → HSACO fatbin  (gpu-module-to-binary)
        ▼
  MLIR module containing a gpu.binary blob (HSACO)
        │  (4) CompiledArtifact         IR text cached (in-memory + on disk)
        ▼
  ExecutionEngine  → hipModuleLoadData → hipModuleLaunchKernel
```

Compare with the two C++ worlds you know:

| Stage | FlyDSL | HIP (`hipcc`) | CK-Tile |
|-------|--------|---------------|---------|
| Specialization | Python trace + `Constexpr` | preprocessor + templates | heavy C++ templates |
| Front-end | Python AST → MLIR (fly dialect) | Clang → LLVM IR | Clang → LLVM IR |
| Mid-level tiling IR | **Fly dialect layout algebra** (explicit ops) | none (done in C++ by hand) | CK C++ tile abstractions (compiled away) |
| Target lowering | MLIR passes → ROCDL → LLVM | Clang codegen → LLVM AMDGPU | same as HIP |
| Back-end | LLVM AMDGPU → HSACO | LLVM AMDGPU → HSACO | LLVM AMDGPU → HSACO |
| When | **at first call (JIT), cached** | ahead of time | ahead of time |

The key structural difference: FlyDSL has a **first-class tiling/layout IR** (the
Fly dialect). In HIP and CK-Tile, your tiling decisions live in C++ and are gone by
the time LLVM sees the code. In FlyDSL they are explicit MLIR ops that dedicated
passes optimize and lower. That is what makes the layout algebra in Chapters 3–6
*real objects* you can print and inspect, not just a coding convention.

## Stage 1 — AST rewriting (host, before tracing)

Before your function is ever traced, FlyDSL rewrites its **Python syntax tree** so
that ordinary Python control flow becomes DSL dispatch calls that can emit
structured-control-flow ops. This happens once, when the decorator is applied, via
`ASTRewriter.transform()` (`python/flydsl/compiler/ast_rewriter.py`). The
registered transformers, in order:

| Transformer | Rewrites |
|-------------|----------|
| `RewriteBoolOps` | `and`/`or`/`not` → `dsl_and_`/`dsl_or_`/`dsl_not_`; chained comparisons → pairwise conjunction |
| `ReplaceIfWithDispatch` | `if cond:` → dispatch that emits `scf.if` for dynamic conditions (or folds statically); ternaries too |
| `InsertEmptyYieldForSCFFor` | `for` → `scf.for` dispatch; `range_constexpr` → plain unrolled Python `range`; `range(..., init=[...])` → loop-carried form |
| `ReplaceYieldWithSCFYield` | `yield` → `scf.yield` emission |
| `CanonicalizeWhile` | `while cond:` → `scf.while` dispatch |
| `FallbackLocations` | (only when `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`) wraps statements so MLIR ops inherit the correct source line |

The rewriter replaces the function's `__code__` in place. The original,
pre-rewrite function is kept so the cache-key machinery can still see the closure
it captured (§2.7).

> **HIP/CK-Tile → FlyDSL.** There is no C++ analogue — this is the price of using
> Python as the metaprogramming language. It is invisible in normal use; you only
> notice it via the control-flow rules in §1.3.

## Stage 2 — Tracing (host, once)

Calling a `@flyc.jit` function from ordinary Python (no MLIR context active) drives
tracing in `JitFunction.__call__` (`python/flydsl/compiler/jit_function.py:1367`):

1. Create an MLIR `Context` and a top-level `Module` marked
   `gpu.container_module`.
2. Create a `gpu.module("kernels")` carrying the `#rocdl.target<chip=gfxNNN>`
   attribute for your device.
3. Create a host `func.FuncOp` (the launcher) with `llvm.emit_c_interface`.
4. Convert the Python/torch arguments into typed DSL arguments (dtype, rank,
   strides read from the tensors).
5. **Execute the (rewritten) function body** inside the MLIR insertion point. Every
   `fx.*` call and every arithmetic/indexing operation emits ops. A call to a
   `@flyc.kernel` emits a `gpu.func` (via `KernelFunction._emit_kernel`) and its
   `.launch(...)` emits a `gpu.launch_func`.

The result is an MLIR module mixing the `fly`, `gpu`, `arith`, `scf`, `memref`, and
`vector` dialects. This is dump **#1** if you set `FLYDSL_DUMP_IR=1` (§2.8).

## Stage 3 — The pass pipeline

`MlirCompiler.compile()` (`python/flydsl/compiler/jit_function.py:788`) runs a pass
pipeline assembled by `RocmBackend._pipeline_parts()`
(`python/flydsl/compiler/backends/rocm.py:36`). It has three stages.

### Stage A — Fly dialect → ROCDL

This is where the layout algebra becomes concrete arithmetic and where copy/MMA
atoms become AMD hardware intrinsics.

| # | Pass | What it does |
|---|------|--------------|
| 1 | `fly-rewrite-func-signature` | Lower DSL types (IntTuple, Layout, MemRef, …) to packed LLVM structs at function / `scf` boundaries; drop fully-static args and rebuild them via `fly.static` |
| 2 | `fly-canonicalize` | Fly-specific constant folding of layout algebra when shapes are static |
| 3 | `fly-layout-lowering` | Lower layout ops (`crd2idx`, divides, partitions) to `arith` + `vector` |
| 4 | `fly-int-swizzle-simplify` | Algebraically simplify XOR-swizzle bit math (exposes CSE/const-fold) |
| 5 | `canonicalize` | upstream MLIR canonicalization |
| 6 | `fly-convert-atom-call-to-ssa-form` | Turn `copy_atom_call`/`mma_atom_call` into SSA form; promote register tensors toward `vector` SSA |
| 7 | `fly-promote-regmem-to-vectorssa` | Promote register-memory pointers to `vector` SSA values |
| 8 | `convert-fly-to-rocdl` | **The main conversion.** Copy atoms → `rocdl.buffer_load/store` (or TDM on gfx1250); MMA atoms → `rocdl.mfma.*` (CDNA) / `rocdl.wmma.*` (RDNA/gfx1250) |
| 9 | `canonicalize` | upstream canonicalization |
| 10 | `gpu.module(convert-scf-to-cf, cse, convert-gpu-to-rocdl{chipset=gfxNNN}, fly-rocdl-cluster-attr)` | Lower SCF, CSE, GPU→ROCDL, attach cluster-dims workaround |

The Fly passes live in `lib/Dialect/Fly/Transforms/` and the conversion in
`lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`. The **per-subtarget atom lowering**
(what an MFMA/BufferCopy actually becomes on gfx942 vs gfx950 vs gfx1250) lives in
`lib/Dialect/FlyROCDL/{CDNA3,CDNA4,GFX11,GFX1250}/`.

> **HIP/CK-Tile → FlyDSL.** Stage A is the part with *no* C++ analogue: it is the
> compiler doing, as IR-to-IR rewrites, the tiling/partitioning/fragment mapping
> that you would write by hand in CK-Tile. `convert-fly-to-rocdl` is the moment a
> `fx.gemm` over an MFMA atom turns into the same `rocdl.mfma.*` intrinsic your
> CK-Tile kernel would emit via `__builtin_amdgcn_mfma_*`.

### Stage B — ROCDL/GPU/SCF → LLVM dialect

All upstream MLIR passes; nothing Fly-specific:

| # | Pass |
|---|------|
| 11 | `rocdl-attach-target{chip=gfxNNN O=2 abi=600 …}` (adds `fast`/`unsafe-math`/`wave64` per compile hints) |
| 12–13 | `convert-scf-to-cf`, `convert-cf-to-llvm` (host control flow) |
| 14 | `gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}` |
| 15–17 | `convert-vector-to-llvm`, `convert-arith-to-llvm`, `convert-func-to-llvm` |
| 18 | `reconcile-unrealized-casts` |
| 18a | `ensure-debug-info-scope-on-llvm-func` (only with `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`) |

### Stage C — LLVM dialect → HSACO

| # | Pass | What it does |
|---|------|--------------|
| 19 | `gpu-module-to-binary{format=fatbin opts="…"}` | Invoke the LLVM AMDGPU back-end (the MC layer) to produce AMDGCN machine code and embed the HSACO/ELF code object as a `gpu.binary` blob |

This single pass is the boundary between MLIR and the LLVM AMDGPU code generator —
the analogue of the LLVM back-end that `hipcc` invokes. CLI options can include
`-g` (debug), `--amdgpu-waves-per-eu=N`, `--amdgpu-num-vgpr=N`.

### Where the arch/tuning knobs enter

Compile hints (`waves_per_eu`, `maxnreg`, `fast_fp_math`, `unsafe_fp_math`,
`llvm_options`) flow from the traced context into `_pipeline_parts()` and become
pass options and `passthrough` attributes on each `gpu.func`
(`python/flydsl/compiler/backends/rocm.py`). `llvm_options` are applied via a
scoped override of LLVM `cl::opt` values (`compiler/llvm_options.py`).

## Stage 4 — Loading and launching

The compiled module's textual IR is stored in a `CompiledArtifact`. On first use,
`jit_executor.py` parses that IR into a fresh context and creates an MLIR
`ExecutionEngine` (opt level 3) linked against the FlyDSL runtime shared libs. The
engine JIT-compiles the *host* side and loads the embedded GPU binary via
`mgpuModuleLoad` → `hipModuleLoadData`. Launches go through `mgpuLaunchKernel` →
`hipModuleLaunchKernel`. These `mgpu*` C wrappers live in
`lib/Runtime/ROCm/FlyRocmRuntimeWrappers.cpp`.

For zero-overhead repeat launches, FlyDSL builds a `CallState` that pre-packs the
ctypes argument buffer, so subsequent calls skip re-inspection entirely.

> **HIP/CK-Tile → FlyDSL.** `mgpuModuleLoad`/`mgpuLaunchKernel` are thin wrappers
> over the HIP module/driver API — the same `hipModuleLoadData` /
> `hipModuleLaunchKernel` you would call by hand when launching a precompiled code
> object.

## The JIT cache (why the second call is instant)

Everything above happens on a **cache miss**. The cache has two tiers, keyed by a
composite key (`JitFunction._build_full_cache_key`):

- a stable snapshot of the module globals the function (transitively) references;
- cache-invalidating env vars (`FLYDSL_COMPILE_OPT_LEVEL`, `FLYDSL_COMPILE_BACKEND`,
  `FLYDSL_COMPILE_LLVM_DIR`, `FLYDSL_DEBUG_ENABLE_DEBUG_INFO`, …);
- the `GPUTarget` (backend, arch, warp size);
- the compile hints;
- per-argument signatures — `Constexpr` values baked in; tensors contribute dtype,
  rank, strides;
- a **toolchain fingerprint** (`_flydsl_key`): a hash over all `flydsl.compiler/
  expr/runtime/utils` sources, the native `.so`s, and the version. Any change to
  compiler code or the pass pipeline invalidates every disk entry.

Tiers:

- **In-memory** (`_mem_cache`): always on; holds `CompiledArtifact`s for the
  process.
- **On-disk** (`JitCacheManager`): pickled artifacts under
  `${FLYDSL_RUNTIME_CACHE_DIR}/…`, guarded by file locks for multi-process safety.
  Disable with `FLYDSL_RUNTIME_ENABLE_CACHE=0` (in-memory cache remains).
  `FLYDSL_RUNTIME_RUN_ONLY=1` loads only from disk and errors on a miss.

> **Gotcha.** The disk cache invalidates on *kernel source and closure* changes,
> but if you edit C++ passes or helper code that is not part of the traced
> closure, disable the cache (`FLYDSL_RUNTIME_ENABLE_CACHE=0`) or you may run a
> stale artifact. The toolchain fingerprint covers the shipped `.so`s, but a local
> rebuild you have not reinstalled can still surprise you.

## Seeing it for yourself

The single most useful habit for a C++ programmer learning FlyDSL is to **dump the
IR at each stage** and read it. Set:

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/ir \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python your_kernel.py
```

You will get numbered files: the initial traced module (fly + gpu dialects), then
the module after Stage A (ROCDL), after Stage B (LLVM dialect), and the final
module with the `gpu.binary` blob. Diffing "before/after Stage A" is the fastest
way to build intuition for what the layout algebra in the next chapters actually
compiles to.

Other useful switches: `FLYDSL_DEBUG_PRINT_AFTER_ALL` (print IR after every pass),
`FLYDSL_DEBUG_AST_DIFF` (see the AST rewrite from §2.3), and
`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` (line info for the profiler; incompatible with
`FLYDSL_RUNTIME_RUN_ONLY`). See Chapter 9 for the full table. Chapter 8 puts these
switches to work on real failures.

With the pipeline in hand, we can now look at the objects that flow through it: the
Fly dialect's layouts, tiles, and atoms.
