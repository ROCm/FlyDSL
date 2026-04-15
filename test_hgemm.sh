#!/bin/bash
#########################################################################
# File Name:    test.sh
# Created Time: Mon Apr 13 09:14:34 2026
#########################################################################
source $GFX1250_MODEL_PATH/ffmlite_env.sh
for SK in 1; do
  echo "===== split-k=${SK} ====="
  EMU_MODE=1 FLYDSL_RUNTIME_ENABLE_CACHE=1 FLYDSL_DUMP_IR=1 PYTHONPATH=/workspace/ffm/FlyDSL/build-fly/python_packages:/workspace/ffm/FlyDSL python tests/kernels/test_hgemm_splitk_gfx1250.py -M 256 -N 8192 -K 1024 --split-k ${SK} --tile-m 128 --tile-n 128 --tile-k 64 --num-buffers 2 --m-warp 2 --n-warp 2 --num-iters 16 --num-warmup 3
done
