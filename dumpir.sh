#!/bin/bash
# dumpir.sh <script> <arguments>
# Example: ./dumpir.sh python tests/benchmark/matrixTranspose.py
FLIR_DUMP_IR=1 FLIR_DUMP_DIR=./my_ir_dumps ROCDSL_DUMP_IR=1 ROCDSL_DUMP_DIR=./my_ir_dumps $@
