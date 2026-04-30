# External bitcode integration (`ffi` + `link_extern`)

This document describes how a framework (e.g. mori's shmem device API) plugs
its pre-compiled LLVM bitcode into FlyDSL's JIT pipeline and participates in
post-load device-side initialisation, **without** FlyDSL's compiler ever
importing the framework.

For the mori-side view — cold-start cost, ABI metadata, the three-piece
contract, and user-level `@flyc.kernel` examples — see
[mori/python/mori/ir/flydsl/README.md](https://github.com/ROCm/mori/blob/main/python/mori/ir/flydsl/README.md).

## 1. The expression-level `ffi` surface

`flydsl.expr.extern.ffi` emits an `llvm.call` to an external C symbol inside a
`@flyc.kernel` body.  It is intentionally link-agnostic and mirrors a normal
expression builder: declare the external prototype if needed, then emit the
call at the current insertion point.

| Parameter | Purpose |
|---|---|
| `symbol` | Mangled C symbol in the external library |
| `arg_types`, `ret_type` | MLIR-friendly type names (`"int32"`, `"uint64"`, `"void"`, …) |
| `is_pure` | Metadata for future lowering to `llvm.func readnone / willreturn` attributes |

Frameworks can pre-construct `ffi` wrappers for their device ABI and expose
them as module-level callables.

## 2. Linking external bitcode

`flydsl.compiler.extern_link.link_extern` attaches compilation/runtime metadata
to a pure `ffi` callable:

```python
from flydsl.expr.extern import ffi
from flydsl.compiler.extern_link import link_extern

my_pe = link_extern(
    ffi("mori_shmem_my_pe", [], "int32"),
    bitcode_path=get_bitcode_path(),
    module_init_fn=shmem_module_init,
)
```

The wrapper registers:

* `bitcode_path` in `CompilationContext.link_libs`, which is fed to
  `rocdl-attach-target l=<path>` so external symbols are resolvable during GPU
  binary generation.
* `module_init_fn` in `CompilationContext.post_load_processors`, which is
  invoked once per loaded `hipModule_t`.

`flydsl.expr.extern.ExternFunction` remains as a compatibility wrapper that
constructs `ffi(...)` and applies `link_extern(...)` in one step, but new
integrations should prefer the explicit two-layer form.

## 3. How the JIT pipeline picks things up

Each call to a linked extern inside a `@flyc.kernel` body first registers its
link metadata on the active `CompilationContext`, then delegates to the
underlying `ffi` callable to emit the `llvm.call`.

`JitFunction.__call__` snapshots `link_libs` and `post_load_processors` and hands them
to `MlirCompiler.compile(..., link_libs=...)` and
`CompiledArtifact(post_load_processors=...)` respectively.

The compiler path **never imports the framework** — everything flows through
`CompilationContext`.  Adding a new framework (Triton-on-FlyDSL, a custom
in-house DSL, …) only requires building matching `ffi + link_extern` wrappers.

## 4. The post-load module capture contract

`module_init_fn` typically writes runtime pointers into device-side globals
(e.g. mori's `globalGpuStates`) that the framework's bitcode relies on.
Triggering it at exactly the right moment requires cooperation with the
runtime.  FlyDSL installs a custom GPU offloading handler,
`#fly.explicit_module`, on JIT GPU modules.  During LLVM translation this
handler emits lookup-able `flydsl_gpu_module_init` and
`flydsl_gpu_module_load_to_device` functions instead of relying on a global
constructor.  The Python executor calls those functions explicitly and owns the
returned `hipModule_t` handles.

The short version:

* The loaded-module list is owned by one `GpuJitModule` instance.
* There is no global or thread-local module-load callback in the C++ runtime.
* Multiple Python threads can JIT concurrently because each compiled artifact
  exposes and calls its own FlyDSL ROCm module loader functions.

On the Python side,
[`jit_executor.py::CompiledArtifact._ensure_engine`](../python/flydsl/compiler/jit_executor.py)
enforces a **post-condition**: if any `post_load_processors` were registered
but explicit module loading produced zero observed module loads, it raises
`RuntimeError` immediately.  This turns a silent contract violation into a
loud, top-of-stack failure instead of letting the first kernel launch fault on
uninitialised device globals.

## 5. Pickling / on-disk cache contract

`CompiledArtifact` is pickleable for on-disk JIT caching.  The
serialisation rules are:

* `ffi` / linked extern instances are **never** pickled — they are module-level
  callables reachable via normal `import`/attribute access.
* `post_load_processors` callables are serialised as
  `"module:qualname"` strings and re-imported on cache hit.  Lambdas,
  `functools.partial`, and bound methods cannot be represented and will
  cause `__getstate__` to raise `pickle.PicklingError` **at cache-write
  time**.
* Extern-linked artifacts are not written to the on-disk cache.  Their external
  bitcode is a compilation input, so the in-memory cache is used for
  same-process reuse while avoiding stale fatbins across processes.

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

## 6. Related files

| File | Role |
|---|---|
| [`python/flydsl/compiler/extern.py`](../python/flydsl/compiler/extern.py) | Pure FFI `ExternFunction` class + `llvm.call` emitter |
| [`python/flydsl/compiler/extern_link.py`](../python/flydsl/compiler/extern_link.py) | `link_extern`, linked extern wrapper, resolver registration |
| [`python/flydsl/expr/extern.py`](../python/flydsl/expr/extern.py) | Public expression API: `ffi`, compatibility `ExternFunction`, `link_extern` |
| [`python/flydsl/compiler/kernel_function.py`](../python/flydsl/compiler/kernel_function.py) | `CompilationContext` (carries `link_libs`, `post_load_processors`) |
| [`python/flydsl/compiler/jit_function.py`](../python/flydsl/compiler/jit_function.py) | Passes `link_libs` into `MlirCompiler.compile` and propagates `post_load_processors` to `CompiledArtifact` |
| [`python/flydsl/compiler/jit_executor.py`](../python/flydsl/compiler/jit_executor.py) | Looks up FlyDSL ROCm module loader symbols, owns `GpuJitModule`, and runs `post_load_processors` |
| [`lib/Runtime/FlyRocmRuntimeWrappers.cpp`](../lib/Runtime/FlyRocmRuntimeWrappers.cpp) | C++ runtime: stateless `mgpuModuleLoad` wrapper |
