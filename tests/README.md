# RocDSL Tests

## Directory Structure

```
tests/
├── mlir/              # MLIR IR test files (.mlir)
│   ├── test_basic.mlir
│   ├── test_crd2idx.mlir
│   ├── test_size.mlir
│   └── ...
└── python/            # Python binding tests
    ├── test_passes.py          # Pass management (16 tests) ✅
    ├── test_arith_operators.py # Operator overloading
    ├── test_basic_ops.py       # CuTe basic operations
    ├── conftest.py             # Pytest fixtures
    ├── pytest.ini              # Pytest configuration
    └── examples/
        └── test_gemm.py        # GEMM example
```

## Running Tests

### MLIR Tests
```bash
# Run with cute-opt
./build/tools/cute-opt/cute-opt tests/mlir/test_crd2idx.mlir --cute-to-standard
./build/tools/cute-opt/cute-opt tests/mlir/test_size.mlir --cute-to-standard
```

### Python Tests

**In Docker with PYTHONPATH:**
```bash
docker exec -it felixatt bash -c "export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core && cd /mnt/raid0/felix/rocDSL && python3 -m pytest tests/python/test_passes.py -v"
```

**Run all Python tests:**
```bash
docker exec -it felixatt bash -c "export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core && cd /mnt/raid0/felix/rocDSL && python3 -m pytest tests/python/ -v"
```

## Test Status

| Test Suite | Location | Status | Count |
|------------|----------|--------|-------|
| Pass Management | `python/test_passes.py` | ✅ Passing | 16/16 |
| Operator Overload | `python/test_arith_operators.py` | ⚠️ Partial | 2/5 |
| CuTe Basic Ops | `python/test_basic_ops.py` | ❌ Needs dialect | 0/7 |
| MLIR IR Tests | `mlir/*.mlir` | ✅ Passing | 20+ |

See `python/TESTING.md` for detailed testing instructions.
