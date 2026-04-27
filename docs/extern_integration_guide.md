# External bitcode integration (`ExternFunction`)

This document describes how a framework (e.g. mori's shmem device API) plugs
its pre-compiled LLVM bitcode into FlyDSL's JIT pipeline and participates in
post-load device-side initialisation, **without** FlyDSL's compiler ever
importing the framework.

For the mori-side view — cold-start cost, ABI metadata, the three-piece
contract, and user-level `@flyc.kernel` examples — see
[mori/python/mori/ir/flydsl/README.md](https://github.com/ROCm/mori/blob/main/python/mori/ir/flydsl/README.md).

## 1. The `ExternFunction` surface

`flydsl.compiler.extern.ExternFunction` emits an `llvm.call` to an external
C symbol inside a `@flyc.kernel` body.  Its constructor takes:

| Parameter | Purpose |
|---|---|
| `symbol` | Mangled C symbol in the bitcode (e.g. `"mori_shmem_my_pe"`) |
| `arg_types`, `ret_type` | MLIR-friendly type names (`"int32"`, `"uint64"`, `"void"`, …) |
| `is_pure` | Metadata for future lowering to `llvm.func readnone / willreturn` attributes |
| `bitcode_path` | Optional `.bc` path fed to `rocdl-attach-target l=<path>` |
| `module_init_fn` | Optional `(hipModule_t) -> None` callable invoked once per loaded GPU module |

Frameworks typically pre-construct `ExternFunction` wrappers for their entire
device ABI (see mori's `mori.ir.flydsl.ops._build_all`) and expose them as
module-level callables.

## 2. How the JIT pipeline picks things up

Each call to an `ExternFunction` inside a `@flyc.kernel` body triggers
`_ensure_declared`, which populates the active `CompilationContext`:

* `ctx.link_libs.add(bitcode_path)` — fed to `rocdl-attach-target` so
  the external symbols are resolvable at GPU-binary generation time.
* `ctx.post_load_processors.append(module_init_fn)` — queued to run
  after `ExecutionEngine.initialize()` has loaded each GPU module.

`JitFunction.__call__` snapshots these lists from the context and hands them
to `MlirCompiler.compile(..., link_libs=...)` and
`CompiledArtifact(post_load_processors=...)` respectively.

The compiler path **never imports the framework** — everything flows through
`CompilationContext`.  Adding a new framework (Triton-on-FlyDSL, a custom
in-house DSL, …) only requires building matching `ExternFunction` wrappers.

## 3. The post-load callback C++ contract

`module_init_fn` typically writes runtime pointers into device-side globals
(e.g. mori's `globalGpuStates`) that the framework's bitcode relies on.
Triggering it at exactly the right moment requires cooperation with the
runtime; FlyDSL uses a per-thread callback registered through
`mgpuSetModuleLoadCallback` in `FlyRocmRuntimeWrappers.cpp`.

Read the comment block atop
[`FlyRocmRuntimeWrappers.cpp`](../lib/Runtime/FlyRocmRuntimeWrappers.cpp) for
the authoritative concurrency contract.  The short version:

* The callback slot is `thread_local` by design, so multiple Python threads
  can JIT concurrently without interfering with each other.
* **Do not** replace it with a `mutex`-protected global — that silently
  re-routes one artifact's module-load events into another artifact's
  post-load processors.
* The contract assumes MLIR `ExecutionEngine.initialize()` calls
  `hipModuleLoadData` synchronously on the same thread.  This holds for every
  MLIR release we currently build against.

On the Python side,
[`jit_executor.py::CompiledArtifact._ensure_engine`](../python/flydsl/compiler/jit_executor.py)
enforces a **post-condition**: if any `post_load_processors` were registered
but `ExecutionEngine.initialize()` produced zero observed module loads in the
calling thread, it raises `RuntimeError` immediately.  This turns a silent
contract violation (e.g. a future MLIR release that loads modules on a
worker thread) into a loud, top-of-stack failure — the correct fix in that
case is upstream MLIR (a per-ExecutionEngine hook), not a local mutex.

## 4. Pickling / on-disk cache contract

`CompiledArtifact` is pickleable for on-disk JIT caching.  The
serialisation rules are:

* `ExternFunction` instances are **never** pickled — they are module-level
  singletons reachable via normal `import`/attribute access.
* `post_load_processors` callables are serialised as
  `"module:qualname"` strings and re-imported on cache hit.  Lambdas,
  `functools.partial`, and bound methods cannot be represented and will
  cause `__getstate__` to raise `pickle.PicklingError` **at cache-write
  time**.

Silent drops are intentionally *not* allowed: a cached kernel that round-tripped
without its initialiser would later GPU-fault on uninitialised device globals,
with a stack that gives no hint about the missing processor.  Failing loudly at
pickle time shifts that diagnostic from production into the development cycle.

If a callable cannot legitimately be hoisted to top-level (e.g. an instance
method closing over runtime state), the caller should either:

1. wrap it in a thin top-level function that re-acquires the state on each
   call, or
2. suppress the disk-cache write path for that specific artifact and rely on
   the in-memory cache only.

## 5. Related files

| File | Role |
|---|---|
| [`python/flydsl/compiler/extern.py`](../python/flydsl/compiler/extern.py) | `ExternFunction` class + `llvm.call` emitter |
| [`python/flydsl/compiler/kernel_function.py`](../python/flydsl/compiler/kernel_function.py) | `CompilationContext` (carries `link_libs`, `post_load_processors`) |
| [`python/flydsl/compiler/jit_function.py`](../python/flydsl/compiler/jit_function.py) | Passes `link_libs` into `MlirCompiler.compile` and propagates `post_load_processors` to `CompiledArtifact` |
| [`python/flydsl/compiler/jit_executor.py`](../python/flydsl/compiler/jit_executor.py) | Runs `post_load_processors` after `ExecutionEngine.initialize()`; enforces the post-condition check |
| [`lib/Runtime/FlyRocmRuntimeWrappers.cpp`](../lib/Runtime/FlyRocmRuntimeWrappers.cpp) | C++ runtime: `mgpuSetModuleLoadCallback` (thread_local) + concurrency contract |
