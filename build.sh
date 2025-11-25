#!/bin/bash
mkdir -p build && cd build
cmake .. -DMLIR_DIR=/mnt/raid0/felix/llvm-project/buildmlir/lib/cmake/mlir
make -j; make cute-opt -j; make _cuteDialect -j;
cd -;
cd python;
python setup.py develop;
cd -;
sh run_tests.sh;
