#!/bin/bash
# Dump IR at each compilation stage.
# Usage: bash scripts/dumpir.sh python my_kernel.py [args...]
set -e
cd "$(dirname "$0")/.."

export FLYDSL_DEBUG_DUMP_IR=1
export FLYDSL_DEBUG_DUMP_DIR="${FLYDSL_DEBUG_DUMP_DIR:-/tmp/flydsl_dump_ir}"

echo "[dumpir] IR dumps -> ${FLYDSL_DEBUG_DUMP_DIR}"
"$@"
rc=$?

if [ -d "${FLYDSL_DEBUG_DUMP_DIR}" ]; then
    echo ""
    echo "[dumpir] Dumped files:"
    find "${FLYDSL_DEBUG_DUMP_DIR}" -type f | sort | while read -r f; do
        echo "  $f"
    done
fi

exit $rc
