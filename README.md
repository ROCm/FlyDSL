# FlyDSL

[![CI](https://github.com/ROCm/FlyDSL/actions/workflows/ci.yaml/badge.svg)](https://github.com/ROCm/FlyDSL/actions/workflows/ci.yaml)
[![Benchmark](https://github.com/ROCm/FlyDSL/actions/workflows/flydsl.yaml/badge.svg)](https://github.com/ROCm/FlyDSL/actions/workflows/flydsl.yaml)
[![Dashboard](https://img.shields.io/badge/Performance-Dashboard-blue)](https://rocm.github.io/FlyDSL/ci-dashboard/)
[![Docs](https://img.shields.io/badge/Docs-rocm.github.io%2FFlyDSL-blue)](https://rocm.github.io/FlyDSL)

FlyDSL is a Python DSL and MLIR compiler stack for writing high-performance GPU
kernels with explicit layout algebra. It targets AMD GPUs through ROCm/HIP and
keeps data layout, tiling, memory movement, synchronization, and instruction
selection visible to the kernel author.

The project has two layers:

- `flydsl`, the installable Python package, provides the expression language,
  JIT/AOT compiler, extension libraries, and runtime integration.
- The Fly and FlyROCDL MLIR dialects provide layout-aware IR and the lowering
  pipeline to ROCDL, LLVM, and GPU binaries.

The repository also contains `kernels/`, a source-tree library of production
kernel implementations and integration examples. It is intentionally not part
of the `flydsl` wheel and does not have the package's API-stability guarantee.

## Install

Install the published package:

```bash
python -m pip install flydsl
python -c "import flydsl; print(flydsl.__version__)"
```

GPU execution requires a compatible ROCm installation. PyTorch-based examples
and tests additionally require a ROCm build of PyTorch that matches the local
ROCm version. See the [installation guide](docs/installation.rst) for supported
workflows, release/version guidance, source builds, and troubleshooting.

## First kernel

The normal authoring model is a GPU kernel declared with `@flyc.kernel` and a
host launcher declared with `@flyc.jit`:

```python
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def copy_kernel(src: fx.Tensor, dst: fx.Tensor, n: fx.Int32):
    i = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if i < n:
        dst[i] = src[i]


@flyc.jit
def copy(src: fx.Tensor, dst: fx.Tensor, n: fx.Int32):
    threads = 256
    copy_kernel(src, dst, n).launch(
        grid=((n + threads - 1) // threads, 1, 1),
        block=(threads, 1, 1),
    )


x = torch.arange(1024, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
copy(x, y, x.numel())
torch.cuda.synchronize()
assert torch.equal(x, y)
```

For a layout-driven vector-add walkthrough, use the
[quick start](docs/quickstart.rst). The complete runnable examples live in
[`examples/`](examples), including tiled copy, tiled MMA, gather/scatter, and
cooperative collectives.

## API map

| Namespace | Purpose | Reference |
|---|---|---|
| `flydsl.expr` (`fx`) | DSL values, layout algebra, memory, tiled copy/MMA, arithmetic, GPU and ROCm operations | [Expression API](docs/api/dsl.rst) |
| `flydsl.compiler` (`flyc`) | `kernel`, `jit`, specialization, argument adapters, backends, and AOT export | [Compiler API](docs/api/compiler.rst) |
| `flydsl.extension.coop` / `fx.coop` | Warp- and block-scope reduce, scan, exchange, load/store, and sorting | [Extension API](docs/api/extensions.rst) |
| `flydsl.extension.random` / `fx.random` | Philox-based integer, uniform, and normal random generation | [Extension API](docs/api/extensions.rst) |
| `flydsl.runtime` | Runtime selection and libraries used by exported objects | [Runtime and configuration](docs/api/runtime.rst) |
| `flydsl.autotune` | Configuration search, validation, caching, and offline artifacts | [Runtime and configuration](docs/api/runtime.rst) |
| `kernels.*` | Repository-only kernel implementations and integration references | [Kernel catalog](docs/api/kernels.rst) |

The [API stability policy](docs/api_stability.md) defines which import paths are
stable. To print the complete stable API catalog for the checked-out revision:

```bash
python3 scripts/list_stable_apis.py
```

## Documentation

The rendered documentation is at
[rocm.github.io/FlyDSL](https://rocm.github.io/FlyDSL). Useful entry points:

- [Installation](docs/installation.rst) and [quick start](docs/quickstart.rst)
- [Kernel authoring](docs/kernel_authoring_guide.md)
- [Layout algebra](docs/layout_system_guide.md)
- [Language types and protocols](docs/language/arithmetic_types.md)
- [Architecture and compilation pipeline](docs/architecture_guide.md)
- [Kernel tuning](docs/kernel_tuning_guide.md)
- [Testing and benchmarking](docs/testing_benchmarking_guide.md)
- [AOT export to C](docs/aot_export_guide.md)

## Build from source

Source development requires Python 3.10+, CMake 3.20+, a C++17 compiler, and
an LLVM/MLIR build with Python bindings. From the repository root:

```bash
# Build the project's ROCm LLVM/MLIR dependency (one-time).
bash scripts/build_llvm.sh -j64

# Build the dialects, compiler tools, runtime, and embedded Python bindings.
bash scripts/build.sh -j64

# Install the Python package in editable mode.
python -m pip install -e .

# Run the default single-GPU correctness suite, examples, and MLIR checks.
bash scripts/run_tests.sh
```

To reuse an existing compatible MLIR installation:

```bash
export MLIR_PATH=/path/to/mlir_install
bash scripts/build.sh -j64
python -m pip install -e .
```

The build scripts accept `FLY_BUILD_DIR` to select a non-default build
directory. See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and
[tests/README.md](tests/README.md) for test tiers and markers.

## Repository layout

```text
FlyDSL/
├── python/flydsl/       # installable Python DSL, compiler, extensions, runtime
├── include/flydsl/      # Fly/FlyROCDL public C++ and TableGen declarations
├── lib/                 # dialect, conversion, transform, C API, and runtime code
├── tools/               # fly-opt and flydsl-lsp-server
├── examples/            # runnable teaching examples and notebooks
├── kernels/             # repository-only optimized kernel implementations
├── tests/               # language, compiler, kernel, system, and MLIR tests
├── docs/                # Sphinx documentation source
└── scripts/             # build, test, benchmark, release, and validation helpers
```

## License

FlyDSL is licensed under the [Apache License 2.0](LICENSE).
