#########################################################################
# File Name:    test.sh
# Created Time: Mon Apr 13 09:14:34 2026
#########################################################################
#!/bin/bash
source $GFX1250_MODEL_PATH/am_env.sh
source $GFX1250_MODEL_PATH/ffmlite_env.sh
FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 PYTHONPATH=/workspace/ffm/FlyDSL/build-fly/python_packages:/workspace/ffm/FlyDSL python tests/kernels/test_hgemm_splitk_gfx1250.py -M 256 -N 256 -K 2048 --split-k 4 --tile-m 64 --tile-n 128 --tile-k 128 --num-buffers 3
