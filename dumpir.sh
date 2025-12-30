#!/bin/bash
# dumpir.sh <script> <arguments>
# Example: ./dumpir.sh python tests/benchmark/matrixTranspose.py

export FLIR_DUMP_IR=1
export FLIR_DUMP_DIR

export PYTHONPATH="${SCRIPT_DIR}/pyflir/src:${PYTHON_PACKAGE_ROOT}:${SCRIPT_DIR}:${PYTHONPATH}"

exec "$@"
