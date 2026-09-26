# AOT export to C objects

`flyc.compile(...).export_to_c(...)` compiles a `@flyc.jit` launcher ahead
of time into a position-independent host object (`.o`) and matching C header.
The object embeds the GPU binary, so a C/C++ program or a framework package
(for example a PyTorch AOTInductor shared library) can launch the kernels
without Python and without the FlyDSL JIT.

This is different from pre-warming the JIT disk cache
(`tests/python/examples/aot_example.py`, `FLYDSL_RUNTIME_CACHE_DIR`,
`FLYDSL_RUNTIME_RUN_ONLY=1`): a pre-warmed cache still needs Python and FlyDSL
at run time, while an exported object needs only the FlyDSL runtime library.

## 1. Quick start

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def axpy_kernel(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32, alpha: fx.Float32):
    i = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if i < n:
        Y[i] = alpha * X[i] + Y[i]


@flyc.jit
def axpy(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32, alpha: fx.Float32, stream: fx.Stream = fx.Stream(None)):
    axpy_kernel(X, Y, n, alpha).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


ptr = flyc.from_c_void_p(fx.Float32, 0)
compiled = flyc.compile(axpy, ptr, ptr, 1024, 1.0)
result = compiled.export_to_c(
    file_path="build",
    file_name="axpy",
    function_prefix="my_axpy",
)
print(result.to_json(indent=2))
```

Link the object into a shared library or executable:

```bash
cc -shared -o build/libmy_kernels.so build/axpy.o \
   $(python -m flydsl.compiler.aot_config --ldflags --libs)
```

and call it from C:

```c
#include "axpy.h"

my_axpy__module_init();
my_axpy__module_load(-1);                 /* load on the current HIP device */
int32_t status = my_axpy_call(x, y, n, 2.0f, stream);
/* ... */
my_axpy__module_unload();
```

## 2. `compile`

```python
flyc.compile(launcher, *args, **kwargs) -> CompiledFunction
```

- `launcher` is a `@flyc.jit` function, a bound method (`obj.launch`) or an
  unbound method with `self` passed first. Anything else raises `TypeError`.
- Arguments are bound, specialized and cache-keyed exactly as for a normal call:
  defaults, keyword arguments, `Constexpr` / `Type` parameters, global captures
  and compile hints (`flyc.compile[{...}]`, `jit.compile_hints`) behave the same.
  Tensors only need correct metadata; `device="meta"` tensors work.
- Null pointer wrappers and non-device tensors are compile placeholders. With
  those representative arguments, the launcher is traced and lowered without
  creating an execution engine, loading a GPU module or launching a kernel. The
  execution engine is created lazily only if the returned specialization is
  called. Live device arguments keep the established `flyc.compile` behavior and
  submit the representative call once.
- `Constexpr` and `Type` arguments are baked in and do not appear in the ABI.
- The result keeps only the lowered IR and plain metadata, never the example
  tensors, pointers or streams.

Rejected with a clear error: argument types other than tensors, pointers,
scalars, streams and `@fx.struct`s of those (`NotImplementedError` naming the
argument), more than one
stream parameter, launchers that register post-load processors (Python
callbacks run after GPU module load), and extern-linked launchers under
`FLYDSL_COMPILE_LLVM_DIR` external codegen.

The returned callable exports the same lowered specialization and does not
compile it again:

```python
compiled = flyc.compile(axpy, ptr, ptr, 1024, 1.0)
compiled.export_to_c(file_path="build", file_name="axpy", function_prefix="my_axpy")
```

### CPU-only compilation

Use null pointer wrappers or non-device tensors as the representative
arguments. In that form, `flyc.compile` does not query the device runtime,
create an execution engine or launch a kernel. No visibility environment
variable is required. Set `ARCH` to the architecture of the deployment GPU so
both the compiler backend and architecture-sensitive kernel construction use
the same target:

```bash
ARCH=gfx950 python build_kernel.py
```

The ROCm compilation toolchain and FlyDSL runtime libraries must still be
installed. The generated object can be compiled and linked on the CPU-only
host, then loaded and launched later on a compatible GPU host. If `ARCH` and
the other architecture overrides are all absent and hardware detection is not
available, the existing `gfx942` fallback is used.

## 3. `export_to_c`

```python
compiled.export_to_c(
    file_path,
    file_name,
    function_prefix="",
    bundle_runtime_libraries=True,
) -> CExportResult
```

- `file_path` is an existing output directory. The method writes
  `<file_path>/<file_name>.o` and `<file_path>/<file_name>.h`.
- `function_prefix` defaults to `file_name`. It must be a non-reserved C/C++
  identifier that does not collide with a symbol exported by the runtime
  libraries (`ValueError` otherwise). It names the entry point and prefixes
  every other symbol the object defines.
- The output directory must exist (`FileNotFoundError` otherwise). The object
  and header are published atomically. Output paths must not alias each other
  or an installed or bundled runtime library.
- The same artifact can be exported repeatedly under different names; the
  artifact is not modified and exporting under the same name is deterministic.
- With `bundle_runtime_libraries=True` the FlyDSL runtime libraries are copied
  next to the object, named by SONAME. An existing identical file is reused; a
  different file with the same name is an error. With `False`, the result lists
  the installed libraries instead.

The original object-path form remains supported for compatibility and can
optionally omit the header:

```python
compiled.export_to_c(
    object_file_path="build/axpy.o",
    function_name="my_axpy",
    header_file_path="build/axpy.h",
)
```

### Symbols

For `function_name = "k"` the object exports exactly:

| Symbol | Signature |
|---|---|
| `k` | `int32_t k(void **args)` |
| `k__module_init` | `int32_t k__module_init(void)` |
| `k__module_load` | `int32_t k__module_load(int32_t device)` |
| `k__module_unload` | `int32_t k__module_unload(void)` |
| `k__metadata` | `const char[]`, NUL-terminated JSON |

and nothing else. All other definitions -- the traced launcher, helpers, the
embedded GPU binary and module state -- are renamed to `k__<name>` and have
internal linkage; external declarations keep their names. Objects exported under
different names therefore link into one library without conflicts. Export
fails if the object would expose any other global symbol.

### `CExportResult`

A frozen result with `to_dict()` / `to_json()`:

| Field | Meaning |
|---|---|
| `schema_version` | `1`. Bumped when a field is removed, renamed or changes meaning; new optional fields do not bump it. |
| `object_file_path`, `header_file_path` | Written files (`header_file_path` is `None` when not requested). |
| `symbol` | Entry point (`function_name`). |
| `module_init_symbol`, `module_load_symbol`, `module_unload_symbol` | Lifecycle functions. |
| `metadata_symbol` | Embedded JSON copy of this result (without output paths). |
| `runtime_libraries` | `RuntimeLibrary(path, soname)` entries to link and ship. |
| `abi` | Tuple of `AbiSlot`, one per `args[i]`. |
| `return_abi` | `ReturnAbi(kind="status", ctype="int32", success=0)`. |
| `backend`, `arch`, `host_triple` | Compile backend, GPU arch, host target triple of the object. |
| `flydsl_version` | FlyDSL version that produced the object. |

Consumers should check `schema_version` before reading the rest.

## 4. Calling convention

`args[i]` points to host storage for ABI slot `i`; every slot needs storage
(pass a pointer to a `NULL` `void *` for the default stream). Slots are ordered
by launcher parameter, each parameter contributing:

| Argument | Slots (`kind` / `ctype`) |
|---|---|
| Tensor, static layout (`flyc.from_torch_tensor`, `flyc.from_dlpack`) | `tensor_data` / `pointer` |
| Tensor, dynamic layout (a plain `torch.Tensor`, `mark_*_dynamic`) | `tensor_data` / `pointer`, then `tensor_layout` / `bytes` |
| `flyc.from_c_void_p` pointer | `pointer` / `pointer` |
| Integer / `Boolean` scalar | `scalar` / `int8`...`int64`, `uint8`...`uint64`, `bool` |
| `Float32` / `Float64` | `scalar` / `float` / `double` |
| `Float16` / `BFloat16` | `scalar` / `uint16` with `encoding` `float16_bits` / `bfloat16_bits` |
| `fx.Stream` parameter | `stream` / `pointer` (keeps its `arg_index` / `arg_name`) |
| `@fx.struct` | The slots of each runtime field in declaration order, with `field` set to the dotted field path; `Constexpr` fields contribute nothing |

When the launcher has no stream parameter, a final implicit `stream` slot with
`arg_index = arg_name = None` is appended; `NULL` selects the default stream.

A `tensor_layout` slot holds the values of the dynamic shape dimensions
(`shape_dims`, each `shape_bits` wide) followed by the dynamic stride
dimensions (`stride_dims`, each `stride_bits` wide), packed without padding.
Widths come from the lowered entry signature, e.g. `(2, 3)` / `(5, 1)` with a
dynamic layout gives `shape_dims=[0, 1]`, `stride_dims=[0]`, `shape_bits=32`,
`stride_bits=64`, `size=16`, `alignment=1`. `size` and `alignment` of every slot
describe the storage `args[i]` must point to.

## 5. C header

The optional header declares the entry and lifecycle functions, the status
codes, one packed `typedef` per dynamic layout (with a static size assertion)
and a typed inline wrapper:

```c
typedef struct __attribute__((packed)) { int32_t shape[2]; int64_t stride[1]; } k_A_layout_t;
static inline int32_t k_call(void *A_data, const k_A_layout_t *A_layout, ..., void *stream);
```

Header types are fixed width (`intN_t`, `uintN_t`, `float`, `double`, `bool`,
`void *`).

## 6. Status codes and module lifecycle

Every exported function returns `int32_t`: `0` on success, a positive
`hipError_t` value for HIP failures, or a negative FlyDSL code:

| Code | Meaning |
|---|---|
| `-1` (`FLYDSL_AOT_ERR_NOT_INITIALIZED`) | `__module_init` not called, or called after `__module_unload` |
| `-2` (`FLYDSL_AOT_ERR_NOT_LOADED`) | Module not loaded on the calling thread's current device |
| `-3` (`FLYDSL_AOT_ERR_INVALID_DEVICE`) | Device index out of range |

A `0` status from the entry means every kernel launch was *submitted*.
Asynchronous execution errors surface only when the caller synchronizes the
stream or device. When a launch fails, later launches in the same call still
run and the first error is returned.

Lifecycle:

1. `__module_init()` registers the embedded GPU binary. Idempotent.
2. `__module_load(device)` loads it on `device` (`device < 0`: the current
   device). Idempotent per device; call it once per device that will launch.
3. The entry resolves its kernels on the calling thread's **current** HIP
   device, so set the device (as framework device guards do) before calling.
4. `__module_unload()` unloads the module from every loaded device and releases
   the state. It waits for previously submitted work on those devices before
   releasing their code objects. Idempotent; later entry calls return `-1`
   until init/load again.

All lifecycle functions and the entry are thread safe; a launch that races with
`__module_unload` returns `-1` rather than touching released state. Kernel
handles are cached per device.

## 7. Runtime libraries and linking

```python
from flydsl.runtime import find_runtime_libraries

find_runtime_libraries()  # (RuntimeLibrary(path=".../libfly_jit_runtime.so", soname="libfly_jit_runtime.so"),)
```

returns the FlyDSL-distributed libraries an exported object needs, including
their FlyDSL-distributed dependencies; system libraries (ROCm, libc) are not
included. The same resolver backs the command-line helper:

```bash
python -m flydsl.compiler.aot_config --libdir    # library directory
python -m flydsl.compiler.aot_config --ldflags   # -L<dir> -Wl,-rpath,<dir>
python -m flydsl.compiler.aot_config --libs      # -l:libfly_jit_runtime.so
```

Output is shell-quoted. When shipping a package, copy the libraries listed in
`CExportResult.runtime_libraries` (or use the bundled copies) and give the
final library an `$ORIGIN`-relative rpath.

## 8. Limitations

- Linux ELF, native host triple only (`host_triple` must match at export time).
  The host code targets the generic CPU of that architecture, so it runs on any
  machine of the same architecture.
- Only the ROCm backend implements the AOT hooks.
- Launcher return values are not supported; the entry returns only the status.
- Reloading an exported object into Python is not provided.
