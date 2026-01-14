#!/bin/bash
# dumpir.sh <script> <arguments>
# Example: bash scripts/dumpir.sh python tests/kernels/benchmark/matrixTranspose.py
export FLIR_DUMP_IR=1

exec "$@"
