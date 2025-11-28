#!/bin/bash
set -ex

# Set up environment
export MLIR_PATH=${MLIR_PATH:-/home/yanronli/llvm-project/buildmlir}
export PYTHONPATH=$MLIR_PATH/tools/mlir/python_packages/mlir_core:$PYTHONPATH

# Build C++ components
mkdir -p build && cd build
cmake .. -DMLIR_DIR=$MLIR_PATH/lib/cmake/mlir
make -j$(nproc)
make rocir-opt -j$(nproc)
make RocirPythonModules -j$(nproc)
make RocirPythonOpsIncGen -j$(nproc)
cd -

# Install Python package
cd python
python setup.py develop
cd -

echo "✓ Build complete!"
echo "✓ rocir-opt: ./build/tools/rocir-opt/rocir-opt"
echo "✓ Python bindings installed in development mode"
echo ""
echo "To use Python bindings, ensure these are set:"
echo "  export MLIR_PATH=$MLIR_PATH"
echo "  export PYTHONPATH=$MLIR_PATH/tools/mlir/python_packages/mlir_core:\$PYTHONPATH"
