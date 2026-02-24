# Testing & Benchmarking Guide

> Test infrastructure, running tests, benchmark harness, writing new tests, and performance comparison with AIter.

## Quick Reference

| Category | Location | Requires GPU | Description |
|---|---|---|---|
| **MLIR IR tests** | `tests/mlir/*.mlir` | No | Verify FLIR → standard lowering |
| **Python IR tests** | `tests/pyir/test_*.py` | No | Python-based MLIR generation + lowering |
| **GPU kernel tests** | `tests/kernels/test_*.py` | Yes | Full compilation → GPU execution |
| **Benchmarks** | `scripts/run_benchmark.sh` | Yes | Performance characterization |

**Run all tests:**
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

Direct MLIR lowering verification using the `flir-opt` tool. These tests validate that FLIR operations lower correctly to standard MLIR dialects without needing a GPU.

**Files:**
| Test File | Description |
|---|---|
| `test_basic.mlir` | Basic FLIR operation lowering |
| `test_simple.mlir` | Simple layout operations |
| `test_ops.mlir` | General FLIR ops |
| `test_all_ops.mlir` | Comprehensive op coverage |
| `test_crd2idx.mlir` | Coordinate-to-index mapping |
| `test_idx2crd.mlir` | Index-to-coordinate mapping |
| `test_size.mlir` | Size query operation |
| `test_cosize.mlir` | Cosize query operation |
| `test_rank.mlir` | Rank query operation |
| `test_get.mlir` | Get (element access) operation |
| `test_composition.mlir` | Layout composition |
| `test_product_divide.mlir` | Product and divide operations |
| `test_all_product_divide.mlir` | All product/divide variants |
| `test_local_ops.mlir` | local_partition and local_tile |
| `test_layout.mlir` | Layout construction |
| `test_layout_amd.mlir` | AMD-specific layout patterns |
| `test_chained.mlir` | Chained operations |
| `test_chained_operations.mlir` | Complex chained sequences |
| `test_coord_lowering.mlir` | Static coordinate lowering |
| `test_coord_lowering_dynamic.mlir` | Dynamic coordinate lowering |
| `test_pass.mlir` | Pass pipeline |
| `comprehensive_test.mlir` | Full integration test |

**Running individually:**
```bash
# Build flir-opt first if needed
cmake --build .flir/build --target flir-opt -j$(nproc)

# Run a single test
.flir/build/bin/flir-opt --flir-to-standard tests/mlir/test_basic.mlir
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

Full end-to-end tests: compile FlyDSL kernels to HSACO binary, execute on GPU, validate against PyTorch reference.

**Files:**
| Test File | Kernel | Description |
|---|---|---|
| `test_vec_add.py` | VecAdd | Vector addition (C = A + B) |
| `test_softmax.py` | Softmax | Row-wise softmax |
| `test_layernorm.py` | LayerNorm | Layer normalization |
| `test_rmsnorm.py` | RMSNorm | RMS normalization |
| `test_preshuffle_gemm.py` | GEMM | Preshuffle MFMA GEMM (fp8/int8/int4/bf16) |
| `test_moe_gemm.py` | MoE GEMM | MoE 2-stage GEMM |
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

---

## 2. Test Runner Scripts

### 2.1 `scripts/run_tests.sh`

Main test orchestrator that runs all 4 test categories sequentially.

**Environment setup:**
```bash
# Auto-detected PYTHONPATH
PYTHONPATH="${REPO_ROOT}/flydsl/src:${BUILD_DIR}/python_packages/flydsl:${REPO_ROOT}:${PYTHONPATH}"
```

**Categories executed:**
1. **MLIR IR tests**: Iterates `tests/mlir/*.mlir`, runs `flir-opt --flir-to-standard`
2. **Python IR tests**: Runs `tests/pyir/test_*.py`
3. **GPU execution tests**: Detects ROCm via `rocm-smi`, runs `tests/kernels/test_*.py`
4. **GPU HIPGraph tests**: Runs GEMM/MoE tests with `-tg` flag for graph capture mode

**Output:**
- Logs to `/tmp/{test_name}.log`
- Extracts performance metrics (TFLOPS, bandwidth) from output
- Final summary with pass/fail counts

### 2.2 `scripts/run_benchmark.sh`

Specialized benchmarking harness for performance characterization.

**Default configurations:**
```bash
# Softmax/LayerNorm
SOFTMAX_SHAPES='32768,8192,bf16'
LAYERNORM_SHAPES='32768,8192,bf16'

# Preshuffle GEMM: "dtype,M,N,K,tile_m,tile_n,tile_k"
GEMM_SHAPES='
fp8,16,40960,5120,16,128,256
fp8,16,77824,5120,16,128,256
fp8,5120,5120,8320,64,256,128
fp8,9728,8192,8320,64,256,128
int8,9728,8192,8320,64,256,128
'

# MoE: "tokens,model_dim,inter_dim,experts,topk,tile_m,tile_n,tile_k,tile_n2,tile_k2"
MOE_SHAPES='
32768,8192,8192,16,4,64,128,128,256,128
64,6144,1024,128,8,16,64,256,64,256
'
```

**Selective execution:**
```bash
# Run only specific benchmarks
bash scripts/run_benchmark.sh softmax
bash scripts/run_benchmark.sh gemm moe
bash scripts/run_benchmark.sh --only softmax,layernorm
```

**Logs:** Written to `${BENCH_LOG_DIR:-/tmp/flir_bench}/`

---

## 3. Pytest Configuration

### 3.1 `tests/conftest.py`

Pytest configuration with MLIR context fixtures.

**Fixtures:**

```python
@pytest.fixture
def ctx():
    """Fresh MLIR context per test with FLIR dialects registered."""
    # Creates Context, registers FLIR extensions, yields object with:
    #   ctx.context, ctx.module, ctx.location

@pytest.fixture
def module(ctx):
    """Provides ctx.module."""

@pytest.fixture
def insert_point(ctx):
    """Sets insertion point to module body."""
```

**Build discovery:** Supports two build layouts:
- Preferred: `.flir/build/python_packages/flydsl`
- Fallback: `build/python_packages/flydsl`

**Session hook:** Prevents pytest exit code 5 (no tests collected) from being treated as failure.

---

## 4. Performance Measurement

### 4.1 `tests/test_common.py`

Core performance testing utilities, adapted from AIter's test infrastructure.

**`perftest()` decorator:**
```python
@perftest(num_iters=20, num_warmup=3, testGraph=False, num_rotate_args=0, needTrace=False)
def my_kernel_test(Input, Output):
    # Kernel invocation
    ...
```

Features:
- Device memory profiling to determine iteration count
- Torch CUDA event timing
- HIPGraph capture mode (`testGraph=True`)
- Optional tensorboard trace profiling
- Cache-aware iteration calculation

**`checkAllclose()` function:**
```python
checkAllclose(output, reference, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05)
```
Returns a mismatch ratio in [0, 1] (0 = pass). Uses colored ANSI output for pass/fail indication.

**`verify_output()` function:**
```python
verify_output(c_out, c_ref, atol=1e-2, rtol=1e-2, msg='')
```
High-level validation wrapper around `checkAllclose`.

### 4.2 `tests/kernels/benchmark_common.py`

Shared benchmark harness for FLIR vs AIter comparison.

**Key types:**
```python
@dataclass(frozen=True)
class PerfRow:
    op: str           # e.g., "softmax"
    shape: str        # e.g., "32768x8192"
    dtype: str        # e.g., "bf16"
    flir_gpu_us: Optional[float]
    aiter_gpu_us: Optional[float]

    @property
    def speedup_aiter_vs_flir(self) -> Optional[float]:
        ...  # flir_us / aiter_us
```

**Key functions:**
```python
# Measure device time (torch CUDA events)
gpu_us = bench_gpu_us_torch(fn, warmup=20, iters=200)

# Enable AIter comparison (best-effort)
aiter_available = maybe_enable_aiter()  # respects AITER_REPO env var

# Run comparative sweep
rows = run_compare_sweep(configs=[(M, N, dtype), ...])
print_perf_table(rows)
```

**Environment variables:**
| Variable | Description | Default |
|---|---|---|
| `BENCH_CONFIGS` | Override benchmark configs | Default shapes |
| `BENCH_WARMUP` | Warmup iterations | 10 |
| `BENCH_ITERS` | Benchmark iterations | 50 |
| `AITER_REPO` | Path to AIter repo for comparison | Auto-detect |

---

## 5. Writing New Tests

### 5.1 PyIR Test Pattern

```python
# tests/pyir/test_my_feature.py
from flydsl.dialects.ext import flir, arith
from flydsl.lang.ir.module import MlirModule, kernel, jit
import _mlir.extras.types as T

class _MyTest(MlirModule):
    GPU_MODULE_NAME = "my_test"

    @jit
    def my_operation(self):
        shape = flir.make_shape(4, 8)
        stride = flir.make_stride(8, 1)
        layout = flir.make_layout(shape, stride)
        result = flir.size(layout)
        return result

def test_my_feature():
    mod = _MyTest()
    ir_str = str(mod.module)
    # Verify IR contains expected operations
    assert "flir.make_layout" in ir_str
    assert "flir.size" in ir_str
```

### 5.2 GPU Kernel Test Pattern

```python
# tests/kernels/test_my_kernel.py
import torch
from tests.utils import compile_to_hsaco
from tests.test_common import checkAllclose

def test_my_kernel():
    M, N = 1024, 1024
    dtype = torch.float16

    # 1. Build kernel
    from kernels.my_kernel import build_my_module
    executor = build_my_module(M, N, "f16")

    # 2. Prepare data
    input_tensor = torch.randn(M, N, dtype=dtype, device="cuda")
    output_tensor = torch.empty_like(input_tensor)

    # 3. Execute kernel
    executor(input_tensor, output_tensor, M)

    # 4. Reference
    reference = torch.nn.functional.layer_norm(input_tensor, [N])

    # 5. Validate
    err = checkAllclose(output_tensor, reference, rtol=1e-2, atol=1e-2)
    assert err == 0, f"Mismatch: {err * 100:.2f}%"
```

### 5.3 Benchmark Test Pattern

```python
# Add to tests/kernels/test_my_kernel.py
from tests.kernels.benchmark_common import bench_gpu_us_torch

def benchmark_my_kernel():
    executor = build_my_module(M, N, dtype)

    # Wrap kernel call
    def run():
        executor(input_tensor, output_tensor, M)

    # Measure
    gpu_us = bench_gpu_us_torch(run, warmup=20, iters=200)

    # Compute metrics
    total_bytes = 2 * M * N * elem_size  # read input + write output
    bandwidth_tbs = total_bytes / (gpu_us * 1e-6) / 1e12
    print(f"Time: {gpu_us:.1f} us, Bandwidth: {bandwidth_tbs:.2f} TB/s")
```

---

## 6. Test Configuration via Environment Variables

| Variable | Used By | Description |
|---|---|---|
| `ROCDSL_SOFTMAX_SHAPES` | `test_softmax.py` | Override softmax test shapes (format: `"M,N,dtype;..."`) |
| `ROCDSL_LAYERNORM_SHAPES` | `test_layernorm.py` | Override layernorm test shapes |
| `ROCDSL_COMPARE_AITER` | kernel tests | Set to `1` to enable AIter comparison |
| `FLIR_DUMP_IR` | `tests/utils.py` | Set to `1` to dump intermediate IR |
| `FLIR_DUMP_DIR` | `tests/utils.py` | IR dump directory (default: `/tmp/flir_dump`) |
| `FLIR_ENABLE_IR_PRINTING` | `tests/utils.py` | Print IR to console |
| `FLIR_TIME_COMPILE` | `tests/utils.py` | Print per-stage compilation timing |
| `FLIR_BUILD_DIR` | `conftest.py` | Override build directory path |

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
    --wfp4              # W4 (INT4) weight path
```

---

## 8. Compilation Utilities (`tests/utils.py`)

The `compile_to_hsaco()` function provides a standalone compilation path for tests:

**Pipeline stages:**
1. `flir-to-standard` lowering
2. `trivial-dce` (dead code elimination)
3. `canonicalize`
4. `cse` (common subexpression elimination)
5. Attach ROCDL target (auto-detect `gpu_arch`)
6. `convert-gpu-to-rocdl` (SCF→CF, bare pointer memref)
7. `gpu-to-llvm`
8. `lower-to-llvm`
9. `gpu-module-to-binary`

**Weight utilities:**
```python
from tests.utils import pertoken_quant, shuffle_weight

# Per-token quantization (handles NaN/Inf)
quantized, scales = pertoken_quant(tensor, dtype=torch.float8_e4m3fnuz)

# Weight preshuffle for MFMA (layout 16x16)
shuffled = shuffle_weight(weight, layout=(16, 16))
```

---

## 9. Source Files

| File | Description |
|---|---|
| `scripts/run_tests.sh` | Main test orchestrator (4 categories) |
| `scripts/run_benchmark.sh` | Benchmark harness with configurable shapes |
| `tests/conftest.py` | Pytest fixtures (MLIR context, module, insert point) |
| `tests/test_common.py` | `perftest()`, `checkAllclose()`, `verify_output()` |
| `tests/utils.py` | `compile_to_hsaco()`, `pertoken_quant()`, `shuffle_weight()` |
| `tests/kernels/benchmark_common.py` | `PerfRow`, `bench_gpu_us_torch()`, `print_perf_table()` |
| `tests/mlir/*.mlir` | MLIR IR lowering tests (22 files) |
| `tests/pyir/test_*.py` | Python IR generation tests (16 files) |
| `tests/kernels/test_*.py` | GPU kernel tests (15 files) |
| `tests/kernels/utils/fp4_utils.py` | FP4 packing/shuffling utilities |
