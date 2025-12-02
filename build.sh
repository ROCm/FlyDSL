#!/bin/bash
set -ex

# Set up environment
if [ -z "$MLIR_PATH" ]; then
    # Default path based on build_llvm.sh
    DEFAULT_MLIR_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/llvm-project/buildmlir"
    if [ -d "$DEFAULT_MLIR_PATH" ]; then
        echo "MLIR_PATH not set. Using default: $DEFAULT_MLIR_PATH"
        export MLIR_PATH="$DEFAULT_MLIR_PATH"
    else
        echo "Error: MLIR_PATH not set and default location ($DEFAULT_MLIR_PATH) not found."
        echo "Please run ./build_llvm.sh first or set MLIR_PATH."
        exit 1
    fi
fi

export PYTHONPATH=$MLIR_PATH/tools/mlir/python_packages/mlir_core:$PYTHONPATH

# Install Python requirements
pip install -r python/requirements.txt

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
