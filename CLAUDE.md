# FlyDSL Project Guide

FlyDSL (Flexible Layout Python DSL) — a Python DSL and MLIR-based compiler stack for authoring high-performance GPU kernels with explicit layouts and tiling on AMD GPUs (MI300X/MI350).

## Repository Layout

```
FlyDSL/
├── python/
│   ├── flydsl/             # Python DSL core
│   │   ├── expr/           # DSL expression API (arith, vector, gpu, rocdl, buffer_ops, math, mem_ops, derived)
│   │   ├── compiler/       # JIT compilation (ast_rewriter, kernel_function, jit_function, backends/)
│   │   ├── runtime/        # Device runtime (device.py, device_runtime/)
│   │   ├── utils/          # Utilities (smem_allocator, env, logger)
│   │   └── autotune.py     # Triton-style autotune module
│   └── mlir_flydsl/        # MLIR Python bindings (built, not edited)
├── kernels/                # Production GPU kernels (importable as kernels.*)
│   ├── pa_decode_fp8.py    # Paged attention decode (FP8)
│   ├── preshuffle_gemm.py  # GEMM (preshuffle layout)
│   ├── blockscale_preshuffle_gemm.py  # Blockscale GEMM
│   ├── hgemm_splitk.py     # FP16 GEMM split-K
│   ├── moe_gemm_2stage.py  # MoE GEMM (2-stage gate/up + reduce)
│   ├── moe_blockscale_2stage.py  # MoE Blockscale GEMM
│   ├── mixed_moe_gemm_2stage.py  # Mixed-precision MoE GEMM
│   ├── layernorm_kernel.py # LayerNorm (layout API)
│   ├── rmsnorm_kernel.py   # RMSNorm (layout API)
│   ├── softmax_kernel.py   # Softmax (layout API)
│   ├── flash_attn_func.py  # FlashAttention
│   ├── fused_rope_cache_kernel.py  # Fused RoPE + KV cache
│   ├── custom_all_reduce.py       # Multi-GPU all-reduce
│   ├── rdna_f16_gemm.py           # RDNA FP16 GEMM
│   ├── rdna_fp8_preshuffle_gemm.py # RDNA FP8 GEMM
│   ├── gemm_common_gfx1250.py     # GFX1250 GEMM common
│   ├── gemm_fp8fp4_gfx1250.py     # GFX1250 FP8/FP4 GEMM
│   ├── wmma_gemm_gfx1250.py       # GFX1250 WMMA GEMM
│   ├── mfma_preshuffle_pipeline.py # Shared MFMA/preshuffle helpers
│   ├── mfma_epilogues.py          # MFMA epilogue helpers
│   ├── pipeline_utils.py          # Pipeline utility helpers
│   ├── kernels_common.py          # Common kernel utilities
│   └── tensor_shim.py             # GTensor/STensor abstraction
├── include/flydsl/         # C++ Fly/FlyROCDL dialect headers
├── lib/                    # C++ dialect implementation + Python bindings
├── tests/                  # All tests
│   ├── kernels/            # Kernel correctness tests (test_pa, test_preshuffle_gemm, etc.)
│   ├── mlir/               # MLIR-level tests (Conversion, LayoutAlgebra, Transforms)
│   └── unit/               # Unit tests (streams, async, etc.)
├── examples/               # Runnable examples (vectorAdd, tiledCopy, tiledMma, preshuffle_gemm)
├── scripts/                # Build & test scripts
│   ├── build_llvm.sh       # Build LLVM/MLIR from source (~30min)
│   ├── build.sh            # Build FlyDSL C++ + Python bindings (~5min)
│   └── run_tests.sh        # Run all tests
├── thirdparty/             # Third-party deps (dlpack, tvm-ffi)
├── tools/                  # fly-opt tool (C++)
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

# Dump MLIR IR at each pipeline stage (also disables disk cache)
FLYDSL_DUMP_IR=1 PYTHONPATH=./ python tests/kernels/test_pa.py
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
- `rocdl` — AMD-specific intrinsics (mfma, exp2, rcp, cvt_pk_fp8_f32, ds_bpermute, BufferCopy atoms, make_buffer_tensor)
- `buffer_ops` — buffer resource ops (create_buffer_resource, buffer_load, buffer_store)
- `typing` — Type system (`T` for MLIR types, `Int32`/`Float32`/`Index` numeric classes)

### Kernel Authoring Patterns

Two APIs for global memory access coexist — use whichever fits the kernel:

**Raw buffer_ops** (used in GEMM kernels, PagedAttention, FlashAttention):
```python
@flyc.kernel
def my_kernel(input_ptr: fx.Tensor, output_ptr: fx.Tensor, N: Int32):
    tid = gpu.thread_idx.x + gpu.block_idx.x * fx.Int32(256)
    rsrc_in = buffer_ops.create_buffer_resource(input_ptr, max_size=True)
    val = buffer_ops.buffer_load(rsrc_in, tid, vec_width=1, dtype=T.f32)
    # ... compute ...
    rsrc_out = buffer_ops.create_buffer_resource(output_ptr, max_size=True)
    buffer_ops.buffer_store(result, rsrc_out, tid)
```

**Layout API** (used in layernorm, rmsnorm, softmax, fused RoPE):
```python
@flyc.kernel
def my_kernel(Input: fx.Tensor, Output: fx.Tensor, N: Int32):
    Input_buf = fx.rocdl.make_buffer_tensor(Input)
    Output_buf = fx.rocdl.make_buffer_tensor(Output)
    row_in = fx.slice(Input_buf, (bid, None))
    in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
    vec_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register)
    r = fx.memref_alloca(vec_reg_ty, fx.make_layout(VEC_WIDTH, 1))
    fx.copy_atom_call(copy_atom, fx.slice(in_div, (None, idx)), r)  # load
    # ... compute ...
    fx.copy_atom_call(copy_atom, r, fx.slice(out_div, (None, idx)))  # store
```

**Layout algebra** (used everywhere for index decomposition):
```python
    layout = fx.make_layout((4, 64), (64, 1))
    coord = fx.idx2crd(tid, layout)
    wave_id = fx.get(coord, 0)
    lane_id = fx.get(coord, 1)
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
loop_start = fx.Index(0)
loop_stop = fx.Index(N)
loop_step = fx.Index(1)
for iv, state in range(loop_start, loop_stop, loop_step, init=[init_val1, init_val2]):
    # Use state[0], state[1] ...
    # Yield updated values:
    results = yield [new_val1, new_val2]
# After loop: results contains final values
```

Important: clear `SmemPtr._view_cache = None` after exiting scf.for to avoid MLIR dominance errors in epilogue code.

## Development Notes

- The JIT disk cache auto-invalidates when kernel source code or closure values change, so `FLYDSL_RUNTIME_ENABLE_CACHE=0` is usually unnecessary. Set it only when:
  - You modified C++ dialect code (MLIR passes) and rebuilt without bumping the version
  - You changed a helper function that isn't captured as a closure of the decorated kernel/jit function
  - Alternatively, clear the disk cache directly: `rm -rf ~/.flydsl/cache`
- `FLYDSL_DUMP_IR=1` implicitly disables disk cache reads/writes, so no need to also set `ENABLE_CACHE=0`
- `PYTHONPATH=./` is required when running from the repo root
- Kernel files in `kernels/` are importable as `from kernels.pa_decode_fp8 import ...`
- The `mlir_flydsl` package is auto-generated during build — never edit it directly
