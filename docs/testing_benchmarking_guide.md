# Testing & Benchmarking Guide

> Test infrastructure, running tests, benchmark harness, writing new tests, and performance measurement.

## Quick Reference

| Category | Location | Requires GPU | Description |
|---|---|---|---|
| **MLIR IR tests** | `tests/mlir/*.mlir` | No | Verify FLIR → standard lowering |
| **Python IR tests** | `tests/pyir/test_*.py` | No | Python-based MLIR generation + lowering |
| **GPU kernel tests** | `tests/kernels/test_*.py` | Yes | Full compilation → GPU execution |
| **Python DSL tests** | `tests/python/` | Varies | Tests for new `@flyc.kernel` API |

**Run GEMM tests:**
```bash
bash scripts/run_tests.sh
```

**Run benchmarks:**
```bash
bash scripts/run_benchmark.sh
```

---

## 1. Test Categories

### 1.1 MLIR IR Tests (`tests/mlir/`)

Direct MLIR lowering verification using the `flir-opt` tool. Validates that FLIR operations lower correctly to standard MLIR dialects without needing a GPU.

**Files:**
| Test File | Description |
|---|---|
| `test_basic.mlir` | Basic FLIR operation lowering |
| `test_crd2idx.mlir` | Coordinate-to-index mapping |
| `test_idx2crd.mlir` | Index-to-coordinate mapping |
| `test_size.mlir` | Size query operation |
| `test_composition.mlir` | Layout composition |
| `test_product_divide.mlir` | Product and divide operations |
| `test_local_ops.mlir` | local_partition and local_tile |
| `test_coord_lowering.mlir` | Static coordinate lowering |
| `test_coord_lowering_dynamic.mlir` | Dynamic coordinate lowering |
| `comprehensive_test.mlir` | Full integration test |

**Running individually:**
```bash
# Build flir-opt first if needed
cmake --build build-fly --target flir-opt -j$(nproc)

# Run a single test
build-fly/bin/flir-opt --flir-to-standard tests/mlir/test_basic.mlir
```

### 1.2 Python IR Tests (`tests/pyir/`)

Python-based tests that generate MLIR IR using the FlyDSL Python API and verify the IR structure and lowering. No GPU execution required.

**Files:**
| Test File | Description |
|---|---|
| `test_layout_algebra.py` | Layout algebra: coalesce, composition, divide, product, complement |
| `test_product_divide.py` | Pythonic product/divide operator tests |
| `test_local_ops.py` | Thread-level partitioning and tiling |
| `test_nested_layouts.py` | Nested/hierarchical layout construction |
| `test_basic_ops.py` | Basic FLIR operation generation |
| `test_arith_operators.py` | Arithmetic operator overloading |
| `test_passes.py` | Pipeline pass execution |
| `test_static_vs_dynamic.py` | Static vs dynamic value handling |
| `test_lang_module_descriptors.py` | MlirModule @kernel/@jit descriptors |
| `test_rocdl_ops.py` | ROCm dialect operations |
| `test_rocir_basic.py` | ROCm IR basic ops |
| `test_rocir_coord_ops.py` | ROCm coordinate operations |
| `test_rocir_product.py` | ROCm product operations |
| `test_rocir_divide.py` | ROCm divide operations |
| `test_rocir_local.py` | ROCm local operations |
| `test_rocir_print.py` | ROCm IR printing |

**Running individually:**
```bash
python tests/pyir/test_layout_algebra.py
```

### 1.3 GPU Kernel Tests (`tests/kernels/`)

Full end-to-end tests: compile FlyDSL kernels, execute on GPU, validate against PyTorch reference.

**Files:**
| Test File | Kernel | Description |
|---|---|---|
| `test_vec_add.py` | VecAdd | Vector addition (C = A + B) |
| `test_softmax.py` | Softmax | Row-wise softmax |
| `test_layernorm.py` | LayerNorm | Layer normalization |
| `test_rmsnorm.py` | RMSNorm | RMS normalization |
| `test_preshuffle_gemm.py` | GEMM | Preshuffle MFMA GEMM (fp8/int8/int4/bf16/fp4) |
| `test_eltwise_add.py` | EltAdd | Element-wise addition |
| `test_matrix_trans.py` | Transpose | Matrix transpose |
| `test_quant.py` | Quantization | Quantization ops |
| `test_gpu_simple.py` | Simple GPU | Minimal GPU kernel test |
| `test_gpu_layout.py` | Layout GPU | GPU layout operations |
| `test_gpu_rocdsl.py` | ROCm DSL | ROCm DSL integration |
| `test_gpu_with_rocir_coords.py` | Coords GPU | Coordinate operations on GPU |
| `test_shared_working.py` | Shared Mem | Shared memory operations |
| `test_ref.py` | Reference | Reference implementations |

**Running individually:**
```bash
python tests/kernels/test_softmax.py
python tests/kernels/test_preshuffle_gemm.py --in_dtype fp8 -M 16 -N 5120 -K 8192
```

### 1.4 Python DSL Tests (`tests/python/`)

Tests for the new `@flyc.kernel` / `@flyc.jit` API:

```
tests/python/
├── examples/     # Example-based tests
├── gpu/          # GPU execution tests with new API
└── ir/           # IR generation tests
```

---

## 2. Test Runner Scripts

### 2.1 `scripts/run_tests.sh`

Runs the preshuffle GEMM test suite via pytest:

```bash
bash scripts/run_tests.sh
```

**Features:**
- Auto-discovers build directory (`build-fly/`)
- Sets up `PYTHONPATH` and `LD_LIBRARY_PATH`
- Runs `pytest tests/kernels/test_preshuffle_gemm.py`
- By default skips `large_shape`-marked tests (set `RUN_TESTS_FULL=1` for all)
- Outputs pass/fail summary

**Environment setup:**
```bash
PYTHONPATH="${BUILD_DIR}/python_packages:${REPO_ROOT}:${PYTHONPATH}"
LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH}"
```

### 2.2 `scripts/run_benchmark.sh`

Specialized benchmarking harness for performance characterization.

**Default configurations:**
```bash
# Softmax/LayerNorm: "M,N,dtype"
SOFTMAX_SHAPES='32768,8192,bf16'
LAYERNORM_SHAPES='32768,8192,bf16'

# Preshuffle GEMM: "dtype,M,N,K,tile_m,tile_n,tile_k"
GEMM_SHAPES='
fp8,16,40960,5120,16,128,256
fp8,16,77824,5120,16,128,256
fp8,5120,5120,8320,64,256,128
fp8,9728,8192,8320,64,256,128
int8,9728,8192,8320,64,256,128
int4,9728,8192,8320,64,256,128
bf16,5120,5120,8320,64,256,128
'

# FP4 GEMM (gfx950 only): "M,N,K,tile_m,tile_n,tile_k"
GEMM_FP4_SHAPES='8192,8192,8192,64,128,256'
```

**Selective execution:**
```bash
bash scripts/run_benchmark.sh                    # default: GEMM only
bash scripts/run_benchmark.sh softmax             # only softmax
bash scripts/run_benchmark.sh gemm moe            # GEMM and MoE
bash scripts/run_benchmark.sh --only softmax,layernorm
bash scripts/run_benchmark.sh --list              # list available ops
```

**Output format:** Tabular with TB/s and TFLOPS columns:
```
op             shape                              dtype       TB/s    TFLOPS
-------------- ---------------------------------- ---------- ---------- ----------
gemm           16x40960x5120                      fp8         1.234     56.789
```

**Logs:** Written to `${BENCH_LOG_DIR:-/tmp/flir_bench}/`

---

## 3. Pytest Configuration

### 3.1 `tests/conftest.py`

Pytest configuration with MLIR context fixtures. Supports both the new Fly dialect and legacy FLIR dialect.

**Fixtures:**

```python
@pytest.fixture
def ctx():
    """Fresh MLIR context per test with dialects registered."""
    # Creates Context, yields object with: ctx.context, ctx.module, ctx.location

@pytest.fixture
def module(ctx):
    """Provides ctx.module."""

@pytest.fixture
def insert_point(ctx):
    """Sets insertion point to module body."""
```

**Build discovery:** Supports multiple build layouts:
- `build-fly/python_packages` (preferred, new Fly dialect)
- `.flir/build/python_packages/flydsl` (legacy FLIR)
- `build/python_packages/flydsl` (fallback)

**Session hook:** Prevents pytest exit code 5 (no tests collected) from being treated as failure.

---

## 4. Performance Measurement

### 4.1 `tests/test_common.py`

Core performance testing utilities (adapted from AIter).

**`perftest()` decorator:**
```python
@perftest(num_iters=20, num_warmup=3, testGraph=False, num_rotate_args=0)
def my_kernel_test(Input, Output):
    # Kernel invocation
    ...
```

Features:
- Device memory profiling to determine rotation count
- Torch CUDA event timing
- HIPGraph capture mode (`testGraph=True`)
- Cache-aware iteration calculation

**`checkAllclose()` function:**
```python
checkAllclose(output, reference, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05)
```
Returns a mismatch ratio in [0, 1] (0 = pass).

**`verify_output()` function:**
```python
verify_output(c_out, c_ref, atol=1e-2, rtol=1e-2, msg='')
```
High-level validation wrapper around `checkAllclose`.

### 4.2 `tests/kernels/benchmark_common.py`

Shared benchmark harness for performance comparison.

**Key functions:**
```python
# Measure device time (torch CUDA events)
gpu_us = bench_gpu_us_torch(fn, warmup=20, iters=200)
```

---

## 5. Compilation Utilities (`tests/utils.py`)

### `compile_to_hsaco()`

Standalone compilation path for legacy (FLIR-based) tests:

```python
from tests.utils import compile_to_hsaco

hsaco = compile_to_hsaco(mlir_module, kernel_name="my_kernel")
```

**Pipeline stages:**
1. FLIR coordinate lowering
2. `flir-to-standard` lowering
3. `canonicalize` + `cse`
4. Attach ROCDL target (auto-detect GPU arch)
5. `convert-gpu-to-rocdl` (SCF→CF, bare pointer memref)
6. `gpu-to-llvm` + `lower-to-llvm`
7. `gpu-module-to-binary`

### Weight Utilities

```python
from tests.utils import pertoken_quant, shuffle_weight

# Per-token quantization (handles NaN/Inf)
quantized, scales = pertoken_quant(tensor, dtype=torch.float8_e4m3fnuz)

# Weight preshuffle for MFMA (layout 16x16)
shuffled = shuffle_weight(weight, layout=(16, 16))
```

---

## 6. Writing New Tests

### 6.1 PyIR Test Pattern (No GPU)

```python
# tests/pyir/test_my_feature.py
import flydsl.expr as fx
from flydsl.expr.typing import T

def test_my_layout_op(ctx, insert_point):
    shape = fx.make_shape(4, 8)
    stride = fx.make_stride(8, 1)
    layout = fx.make_layout(shape, stride)
    result = fx.size(layout)
    ir_str = str(ctx.module)
    assert "fly.make_layout" in ir_str
```

### 6.2 GPU Kernel Test Pattern (New API)

```python
# tests/kernels/test_my_kernel.py
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu
from tests.test_common import checkAllclose

@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    # ... kernel body ...

@flyc.jit
def launch(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, N).launch(grid=(N // 256,), block=(256,), stream=stream)

def test_my_kernel():
    N = 1024
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.empty(N, device="cuda", dtype=torch.float32)

    launch(A, B, N)

    # Reference
    ref = A  # or some computation

    # Validate
    err = checkAllclose(B, ref, rtol=1e-2, atol=1e-2)
    assert err == 0, f"Mismatch: {err * 100:.2f}%"
```

### 6.3 GPU Kernel Test Pattern (Legacy API)

```python
# tests/kernels/test_my_kernel.py
import torch
from flydsl.dialects.ext import flir, arith
from flydsl.compiler.compiler import compile
from tests.test_common import checkAllclose

class MyKernel(flir.MlirModule):
    GPU_MODULE_NAME = "my_kernel"

    @flir.kernel
    def kernel_func(self, ...):
        ...

def test_my_kernel():
    mod = MyKernel()
    executor = compile(mod.module)
    # ... setup, execute, validate ...
```

### 6.4 Benchmark Test Pattern

```python
from tests.kernels.benchmark_common import bench_gpu_us_torch

def benchmark_my_kernel():
    # Setup
    launch_fn = compile_my_kernel(...)

    def run():
        launch_fn(input_tensor, output_tensor)

    # Measure
    gpu_us = bench_gpu_us_torch(run, warmup=20, iters=200)

    # Compute metrics
    total_bytes = 2 * M * N * elem_size
    bandwidth_tbs = total_bytes / (gpu_us * 1e-6) / 1e12
    print(f"Time: {gpu_us:.1f} us, Bandwidth: {bandwidth_tbs:.2f} TB/s")
```

---

## 7. GEMM Test CLI Arguments

The `test_preshuffle_gemm.py` test supports extensive CLI configuration:

```bash
python tests/kernels/test_preshuffle_gemm.py \
    --in_dtype fp8 \
    -M 16 -N 5120 -K 8192 \
    --tile_m 16 --tile_n 128 --tile_k 256 \
    --lds_stage 2 \
    --num_iters 20 \
    --num_warmup 3 \
    --no_aiter_bench \
    --test_graph        # or -tg for HIPGraph mode
    --wfp4              # FP4 weight path (gfx950 only)
```

---

## 8. Test Configuration via Environment Variables

| Variable | Used By | Description |
|---|---|---|
| `ROCDSL_SOFTMAX_SHAPES` | `test_softmax.py` | Override softmax test shapes (`"M,N,dtype;..."`) |
| `ROCDSL_LAYERNORM_SHAPES` | `test_layernorm.py` | Override layernorm test shapes |
| `FLYDSL_DUMP_IR` | Compiler | Dump intermediate IR at each pipeline stage |
| `FLYDSL_DUMP_DIR` | Compiler | IR dump directory (default: `~/.flydsl/debug`) |
| `FLIR_DUMP_IR` | `tests/utils.py` | Dump IR in legacy compilation path |
| `FLIR_DUMP_DIR` | `tests/utils.py` | Legacy IR dump directory |
| `FLIR_ENABLE_IR_PRINTING` | `tests/utils.py` | Print IR to console |
| `FLIR_TIME_COMPILE` | `tests/utils.py` | Print per-stage compilation timing |
| `RUN_TESTS_FULL` | `run_tests.sh` | Set to `1` to run all parametrized cases |
| `BENCH_LOG_DIR` | `run_benchmark.sh` | Benchmark log directory (default: `/tmp/flir_bench`) |

---

## 9. IR Dump Workflow

### New API (via `MlirCompiler`)

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python my_test.py
```

Produces numbered `.mlir` files per pipeline stage plus `final_isa.s`.

### Legacy API (via `compile_to_hsaco`)

```bash
FLIR_DUMP_IR=1 FLIR_DUMP_DIR=./dumps python tests/kernels/test_softmax.py
```

### Dedicated IR Dump Script

```bash
bash scripts/dumpir.sh
```

---

## 10. Source Files

| File | Description |
|---|---|
| `scripts/run_tests.sh` | GEMM test runner (pytest) |
| `scripts/run_benchmark.sh` | Benchmark harness with configurable shapes |
| `scripts/dumpir.sh` | IR dump helper script |
| `tests/conftest.py` | Pytest fixtures (MLIR context, module, insert point) |
| `tests/test_common.py` | `perftest()`, `checkAllclose()`, `verify_output()` |
| `tests/utils.py` | `compile_to_hsaco()`, `pertoken_quant()`, `shuffle_weight()` |
| `tests/kernels/benchmark_common.py` | `bench_gpu_us_torch()`, benchmark harness |
| `tests/mlir/*.mlir` | MLIR IR lowering tests |
| `tests/pyir/test_*.py` | Python IR generation tests (16 files) |
| `tests/kernels/test_*.py` | GPU kernel tests (14 files) |
| `tests/python/` | New Python DSL tests |
