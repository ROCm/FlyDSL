# GPU Implementation for RocDSL

## Overview
GPU kernel execution support using HIP runtime compilation.

## Components

### Runtime Module
- `python/rocdsl/runtime/hip_util.py` - HIP runtime wrappers
- `python/rocdsl/runtime/__init__.py` - Module exports

### Tests
- `tests/python/test_gpu_simple.py` - Basic vector addition
- `tests/python/test_gpu_layout.py` - Layout-based indexing tests

### Examples  
- `examples/gpu_matrix_ops.py` - Matrix transpose and GELU
- `examples/README_GPU.md` - Usage documentation

## Test Results
All tests passing on AMD gfx942:
- Basic GPU test: ✅ PASS
- Matrix transpose: ✅ PASS (error: 0.0)
- Strided layout: ✅ PASS (0 errors/512)
- GELU activation: ✅ PASS (throughput: ~89 GB/s)

## Usage
```python
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
# Compile and run GPU kernels
```

See `examples/README_GPU.md` for details.
