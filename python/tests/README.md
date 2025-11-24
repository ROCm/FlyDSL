# RocDSL Python Tests

Python unit tests for RocDSL (CuTe dialect Python bindings), following the structure of mlir-python-extras.

## Structure

```
tests/
├── __init__.py                  # Package marker
├── conftest.py                  # Pytest fixtures and configuration
├── test_basic_ops.py           # Basic operations (make_shape, make_layout, etc.)
├── test_product_divide.py      # Product and divide operations
└── test_local_ops.py           # Local operations (partition, tile)
```

## Running Tests

### Prerequisites

Ensure RocDSL is built and Python bindings are installed:

```bash
# From project root
cd build
ninja

# Install Python package in development mode
cd ../python
pip install -e .
```

### Run All Tests

```bash
cd python
pytest
```

### Run Specific Test Files

```bash
pytest tests/test_basic_ops.py
pytest tests/test_product_divide.py  
pytest tests/test_local_ops.py
```

### Run Specific Tests

```bash
pytest tests/test_basic_ops.py::test_make_shape
pytest tests/test_product_divide.py::test_logical_product
pytest tests/test_local_ops.py::test_local_partition
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

### With Coverage

```bash
pytest --cov=rocdsl --cov-report=html
```

## Test Organization

### test_basic_ops.py

Tests for fundamental CuTe operations:

- `test_make_shape` - Shape creation
- `test_make_layout` - Layout creation from shape/stride
- `test_size_operation` - Size computation
- `test_get_shape_stride` - Extract shape and stride
- `test_rank_operation` - Rank computation
- `test_cosize_operation` - Cosize (stride extent)
- `test_composition` - Layout composition

### test_product_divide.py

Tests for tiling and partitioning operations:

**Product (Tiling):**
- `test_logical_product` - Basic tiling
- `test_zipped_product` - Interleaved tiling
- `test_tiled_product` - Shaped tiling
- `test_flat_product` - Flattened product
- `test_raked_product` - Raked/strided pattern
- `test_blocked_product` - Blocked pattern

**Divide (Partitioning):**
- `test_logical_divide` - Basic partitioning
- `test_zipped_divide` - Interleaved partitioning
- `test_tiled_divide` - Shaped partitioning
- `test_flat_divide` - Flattened divide

### test_local_ops.py

Tests for thread/block level operations:

- `test_local_partition` - Thread-level partitioning
- `test_local_tile` - Block-level tiling
- `test_local_partition_small` - Small tile sizes
- `test_local_tile_coords` - Different coordinates
- `test_combined_partition_tile` - Combined operations

## Test Fixtures

### `ctx` Fixture

Provides a fresh MLIR context for each test with:
- `ctx.context` - MLIR context
- `ctx.module` - MLIR module
- `ctx.location` - Unknown location

Usage:
```python
def test_something(ctx):
    with InsertionPoint(ctx.module.body):
        # Your test code here
        pass
    
    ctx.module.operation.verify()
    
    ir = str(ctx.module)
    assert "cute.operation" in ir
```

## Writing New Tests

Follow this template:

```python
"""Test description."""

import pytest
from mlir.ir import InsertionPoint
from mlir.dialects import func, arith

try:
    import rocdsl.dialects.ext.cute as cute
except ImportError:
    pytest.skip("RocDSL dialect not available", allow_module_level=True)


def test_my_operation(ctx):
    """Test my operation."""
    with InsertionPoint(ctx.module.body):
        @func.FuncOp.from_py_func(name="test_my_op")
        def test_func():
            c8 = arith.constant(8, index=True)
            
            # Use cute operations
            result = cute.my_operation(c8)
            
            return result
    
    # Verify the module
    ctx.module.operation.verify()
    
    # Check generated IR
    ir = str(ctx.module)
    assert "cute.my_operation" in ir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## CI Integration

Add to your CI pipeline:

```yaml
- name: Run Python Tests
  run: |
    cd python
    pip install -e .[test]
    pytest --junitxml=test-results.xml
```

## Troubleshooting

### Import Error

```
ModuleNotFoundError: No module named rocdsl
```

**Solution:** Install the package in development mode:
```bash
cd python && pip install -e .
```

### MLIR Dialect Not Available

```
Skipped: RocDSL dialect not available
```

**Solution:** Ensure cute-opt is built and the CuTe dialect library is available:
```bash
cd build && ninja cute-opt
```

### Verification Errors

```
MLIR verification failed
```

**Solution:** Check that:
1. All operations use correct types (`index` not `!cute.int`)
2. Operation signatures match TableGen definitions
3. Module structure is valid

## Coverage

Current test coverage:

- ✅ Basic Operations: 9/9 (100%)
- ✅ Product Operations: 6/6 (100%)
- ✅ Divide Operations: 4/4 (100%)
- ✅ Local Operations: 2/2 (100%)

**Total: 21/21 operations (100% coverage)**

## References

- Similar structure to [mlir-python-extras](https://github.com/makslevental/mlir-python-extras)
- Follows pytest best practices
- Compatible with MLIR Python bindings API
