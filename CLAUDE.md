# FlyDSL Project Guide

FlyDSL (Flexible Layout Python DSL) — a Python DSL and MLIR-based compiler stack for authoring high-performance GPU kernels with explicit layouts and tiling on AMD GPUs (MI300X/MI350).

## Repository Layout

```
FlyDSL/
├── python/flydsl/          # Python DSL core
│   ├── expr/               # DSL expression API (arith, vector, gpu, rocdl, buffer_ops)
│   ├── compiler/           # JIT compilation (ast_rewriter, kernel_function, jit_function)
│   ├── utils/              # Utilities (smem_allocator, env, logger)
│   └── _mlir/              # Embedded MLIR Python bindings (built, not edited)
├── kernels/                # Production GPU kernels (importable as kernels.*)
│   ├── pa_decode_fp8.py    # Paged attention decode (FP8)
│   ├── preshuffle_gemm.py  # GEMM kernels
│   ├── layernorm_kernel.py # LayerNorm
│   ├── softmax_kernel.py   # Softmax
│   └── layout_utils.py     # Layout coordinate helpers
├── include/flydsl/         # C++ Fly dialect headers
├── lib/                    # C++ dialect implementation
├── tests/                  # All tests
│   ├── kernels/            # Kernel correctness tests (test_pa, test_preshuffle_gemm, etc.)
│   ├── pyir/               # IR-level tests
│   └── unit/               # Unit tests (streams, async, etc.)
├── examples/               # Runnable examples (vectorAdd, tiledCopy, tiledMma)
├── scripts/                # Build & test scripts
│   ├── build_llvm.sh       # Build LLVM/MLIR from source (~30min)
│   ├── build.sh            # Build FlyDSL C++ + Python bindings (~5min)
│   └── run_tests.sh        # Run all tests
└── docs/                   # Architecture, layout system, kernel authoring guides
```

## Build & Install

```bash
# Build LLVM/MLIR (one-time, ~30min)
bash scripts/build_llvm.sh

# Build FlyDSL
bash scripts/build.sh

# Install in dev mode
pip install -e .
```

## Running Tests

```bash
# All tests
PYTHONPATH=./ pytest tests/

# Specific kernel test
PYTHONPATH=./ python tests/kernels/test_pa.py --num_iters 50

# Disable JIT cache during development
FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=./ python tests/kernels/test_pa.py
```

## Code Style

- **Python**: black (line-length=120), ruff for linting. Config in `pyproject.toml`.
- **C++**: LLVM style (ColumnLimit=100). Config in `.clang-format`.
- **Imports**: isort with `flydsl` as known first-party.

## Key Concepts

### DSL Expression API (`python/flydsl/expr/`)

Kernels are written in Python using the FlyDSL expression API:
- `arith` — arithmetic ops (constant, select, index_cast, trunci, extsi, etc.)
- `vector` — vector ops (extract, insert, load_op, store, broadcast, from_elements, bitcast)
- `gpu` — GPU indexing (thread_idx, block_idx, barrier)
- `rocdl` — AMD-specific intrinsics (mfma, cvt_pk_fp8_f32, ds_bpermute)
- `buffer_ops` — buffer resource ops (create_buffer_resource, buffer_load, buffer_store)

### Kernel Authoring Pattern

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops
from flydsl.expr.typing import T, Int32

@flyc.kernel
def my_kernel(input_ptr: fx.Tensor, output_ptr: fx.Tensor, N: Int32):
    tid = gpu.thread_idx.x + gpu.block_idx.x * arith.constant(256, type=T.i32)
    rsrc_in = buffer_ops.create_buffer_resource(input_ptr, max_size=True)
    val = buffer_ops.buffer_load(rsrc_in, tid, vec_width=1, dtype=T.f32)
    # ... compute ...
    rsrc_out = buffer_ops.create_buffer_resource(output_ptr, max_size=True)
    buffer_ops.buffer_store(result, rsrc_out, tid)
```

### SmemAllocator & SmemPtr

Shared memory (LDS) is managed via `SmemAllocator` and `SmemPtr`:
```python
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

allocator = SmemAllocator(None, arch=arch, global_sym_name="my_smem")
allocator.ptr = size_in_bytes
base = allocator.get_base()
lds_view = SmemPtr(base, offset, T.f32, shape=(N,)).get()  # returns memref for loads/stores
```

### scf.for Loops with Loop-Carried Values

FlyDSL supports `scf.for` loops via Python `range()` with `init=` keyword:
```python
loop_start = arith.index(0)
loop_stop = arith.index(N)
loop_step = arith.index(1)
for iv, state in range(loop_start, loop_stop, loop_step, init=[init_val1, init_val2]):
    # Use state[0], state[1] ...
    # Yield updated values:
    results = yield [new_val1, new_val2]
# After loop: results contains final values
```

Important: clear `SmemPtr._view_cache = None` after exiting scf.for to avoid MLIR dominance errors in epilogue code.

## Development Notes

- Always set `FLYDSL_RUNTIME_ENABLE_CACHE=0` when iterating on kernel code to bypass JIT cache
- `PYTHONPATH=./` is required when running from the repo root
- Kernel files in `kernels/` are importable as `from kernels.pa_decode_fp8 import ...`
- The `_mlir` package is auto-generated during build — never edit it directly
