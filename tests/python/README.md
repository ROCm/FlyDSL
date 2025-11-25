# Rocir Python Tests

测试文件按功能分类组织:

## 目录结构

```
tests/python/
├── ir/           # IR测试 (MLIR生成和转换,不涉及GPU执行)
│   ├── test_rocir_basic.py
│   ├── test_rocir_product.py
│   ├── test_rocir_divide.py
│   ├── test_rocir_local.py
│   ├── test_rocir_coord_ops.py
│   ├── test_arith_operators.py
│   ├── test_basic_ops.py
│   ├── test_local_ops.py
│   ├── test_product_divide.py
│   └── test_passes.py
│
├── gpu/          # GPU测试 (GPU kernel编译和执行)
│   ├── test_gpu_rocdsl.py          # Rocir layouts + GPU kernels
│   ├── test_gpu_layout.py          # GPU layout tests
│   ├── test_gpu_simple.py          # Simple GPU kernel tests
│   ├── test_gpu_with_rocir_coords.py  # Coordinate ops on GPU
│   └── test_shared_working.py      # Shared memory optimization
│
└── examples/     # 示例和演示
    └── test_gpu_gemm.py
```

## 测试分类

### IR测试 (tests/python/ir/)
- **目的**: 测试MLIR IR生成、操作和转换
- **环境**: 不需要GPU,纯Python + MLIR
- **内容**:
  - Rocir dialect操作 (make_shape, make_layout, crd2idx等)
  - Layout algebra (product, divide, partition等)
  - Pass管道 (lowering, optimization等)

### GPU测试 (tests/python/gpu/)
- **目的**: 测试GPU kernel编译和实际执行
- **环境**: 需要ROCm GPU (AMD Instinct系列)
- **内容**:
  - MLIR → HSACO编译
  - HIP kernel执行
  - GPU性能测试
  - Shared memory优化
  - Layout在GPU kernel中的应用

## 运行测试

### 运行所有测试
```bash
./run_tests.sh
```

### 只运行IR测试
```bash
python3 tests/python/ir/test_*.py
```

### 只运行GPU测试
```bash
python3 tests/python/gpu/test_*.py
```

## 权限规范

- Python文件 (*.py): `644` (rw-r--r--)
- Shell脚本 (*.sh): `755` (rwxr-xr-x)
- 目录: `755` (rwxr-xr-x)

如需修复权限,运行项目根目录下的权限修复脚本。
