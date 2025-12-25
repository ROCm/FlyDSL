#!/bin/bash
# dumpir.sh <script> <arguments>
# Example: ./dumpir.sh python tests/benchmark/matrixTranspose.py
set -euo pipefail

# Allow override from the environment; default to a local dump directory.
: "${FLIR_DUMP_DIR:=./my_ir_dumps}"

export FLIR_DUMP_IR=1
export FLIR_DUMP_DIR

exec "$@"
