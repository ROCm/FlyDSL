#!/bin/bash
# Dump IR at each compilation stage.
# Usage: bash scripts/dumpir.sh python my_kernel.py [args...]
set -e
cd "$(dirname "$0")/.."

export FLYDSL_DUMP_IR=1
export FLYDSL_DUMP_DIR="${FLYDSL_DUMP_DIR:-/tmp/flydsl_dump_ir}"

echo "[dumpir] IR dumps -> ${FLYDSL_DUMP_DIR}"
"$@"
rc=$?

exit $rc
