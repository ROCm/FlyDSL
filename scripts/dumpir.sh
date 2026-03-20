#!/bin/bash
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Dump IR at each compilation stage.
# Usage: bash scripts/dumpir.sh python my_kernel.py [args...]
#
# Environment (set automatically):
#   FLYDSL_DUMP_DIR          output directory (default: /tmp/flydsl_dump_ir)
set -e
cd "$(dirname "$0")/.."

export FLYDSL_DUMP_IR=1
export FLYDSL_DUMP_DIR="${FLYDSL_DUMP_DIR:-/tmp/flydsl_dump_ir}"

echo "[dumpir] IR dumps -> ${FLYDSL_DUMP_DIR} (cache disabled)"
"$@"
rc=$?

exit $rc
