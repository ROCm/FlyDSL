#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Dump IR at each compilation stage.
# Usage: bash scripts/dumpir.sh python my_kernel.py [args...]
#
# Environment (set automatically):
#   FLYDSL_DUMP_DIR          output directory (default: /tmp/flydsl_dump_ir)
set -e
cd "$(dirname "$0")/.."

export FLYDSL_DUMP_IR=1
export FLYDSL_DEBUG_DUMP_ASM=1
export FLYDSL_DUMP_DIR="${FLYDSL_DUMP_DIR:-./my_flydsl_ir}"

echo "[dumpir] IR dumps -> ${FLYDSL_DUMP_DIR} (cache disabled)"
"$@"
rc=$?

echo "[dumpir] Done. Files in ${FLYDSL_DUMP_DIR}:"
find "${FLYDSL_DUMP_DIR}" -type f | sort
exit $rc
