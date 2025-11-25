#!/bin/bash
mkdir -p build && cd build
cmake .. -DMLIR_DIR=/mnt/raid0/felix/llvm-project/buildmlir/lib/cmake/mlir
make -j; make rocir-opt -j; make _rocirDialect -j;  make RocirPythonOpsIncGen -j;
cd -;
cd python;
python setup.py develop;
cd -;
sh run_tests.sh;
