#!/bin/bash
# Dump IR at each compilation stage + final ISA assembly.
# Usage: bash scripts/dumpir.sh python my_kernel.py [args...]
set -e
cd "$(dirname "$0")/.."

export FLYDSL_DUMP_IR=1
export FLYDSL_DEBUG_DUMP_ASM=1
export FLYDSL_DUMP_DIR="${FLYDSL_DUMP_DIR:-./my_flydsl_ir}"

echo "[dumpir] IR + ASM dumps -> ${FLYDSL_DUMP_DIR}"
"$@"
rc=$?

echo "[dumpir] Done. Files in ${FLYDSL_DUMP_DIR}:"
find "${FLYDSL_DUMP_DIR}" -type f | sort
exit $rc
