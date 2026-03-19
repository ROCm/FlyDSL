#export MLIR_PATH=/home/jli10004/flydsl/llvm-project/buildmlir
#export PYTHONPATH=/data/jli/flydsl-ws/flydsl-prev/build/python_packages/:/data/jli/flydsl-ws/flydsl-prev/flydsl_/src:/data/jli/flydsl-ws/flydsl-prev:$PYTHONPATH
#export PATH=/data/jli/flydsl-ws/flydsl-prev/build/bin:$PATH

MLIR_INSTALL=/home/jli10004/flydsl/mi450-cmodel-env/llvm-project/mlir_install
MLIR_PATH=/home/jli10004/flydsl/mi450-cmodel-env/llvm-project/build-flydsl

export PYTHONPATH=/home/jli10004/flydsl/mi450-cmodel-env/flydsl-prev/build-fly/python_packages:/home/jli10004/flydsl/mi450-cmodel-env/llvm-project/build-flydsl/tools/mlir/python_packages/mlir_core:$PYTHONPATH
export PYTHONPATH=/home/jli10004/flydsl/mi450-cmodel-env/flydsl-prev/python:$PYTHONPATH
export LD_LIBRARY_PATH=$MLIR_INSTALL/lib:$LD_LIBRARY_PATH
