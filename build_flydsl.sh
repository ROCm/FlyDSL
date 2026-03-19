export MLIR_PATH=/home/jli10004/flydsl/mi450-cmodel-env/llvm-project/mlir_install
mkdir -p build-fly && cd build-fly
cmake .. -DMLIR_DIR=/home/jli10004/flydsl/mi450-cmodel-env/llvm-project/mlir_install/lib/cmake/mlir -GNinja
NPROC=$(nproc 2>/dev/null || echo 4)
ninja -j${NPROC}
