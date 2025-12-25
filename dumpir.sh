#!/bin/bash
# dumpir.sh <script> <arguments>
# Example: ./dumpir.sh python tests/benchmark/matrixTranspose.py

export ROCDSL_DUMP_IR=1
export ROCDSL_DUMP_DIR=flir_dumpxx

exec "$@"
