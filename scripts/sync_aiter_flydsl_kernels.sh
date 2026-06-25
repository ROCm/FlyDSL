#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Sync aiter's vendored FlyDSL MoE kernels with this FlyDSL checkout so the aiter
# fused-MoE e2e + strict-correctness guardrail (op_tests/test_moe_2stage.py) runs
# against the SAME kernel sources we tune here.
#
# Why this is needed: aiter pins `flydsl==0.1.8` and ships its own (older) vendored
# copies under aiter/ops/flydsl/kernels/.  Against the installed FlyDSL compiler
# (0.2.x) those stale copies crash during MLIR emission BEFORE producing any number
# (`'Int32' object has no attribute 'type'`, then `arith.extsi i64->i32 cast
# incompatible`).  Overlaying the current FlyDSL kernel sources resolves the skew;
# the e2e path then produces real us + logits_diff and the strict correctness gate
# (`logits_diff <= 0.01`) can be applied.  This is an aiter-environment integration
# step, not a change to the FlyDSL kernels themselves.
#
# Idempotent.  Backs up the originals once to <aiter>/ops/flydsl/kernels/.orig_bak/.
# Usage:  bash scripts/sync_aiter_flydsl_kernels.sh [AITER_REPO]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AITER_REPO="${1:-/sgl-workspace/aiter}"
SRC="${REPO_ROOT}/kernels"
DST="${AITER_REPO}/aiter/ops/flydsl/kernels"
BAK="${DST}/.orig_bak"

if [[ ! -d "${DST}" ]]; then
  echo "ERROR: aiter vendored kernel dir not found: ${DST}" >&2
  exit 1
fi

# The MoE 2-stage kernel and its sibling deps imported via `from .<name>`.
FILES=(
  mixed_moe_gemm_2stage.py
  moe_gemm_2stage.py
  moe_common.py
  mfma_epilogues.py
  mfma_preshuffle_pipeline.py
  layout_utils.py
)

mkdir -p "${BAK}"
for f in "${FILES[@]}"; do
  if [[ ! -f "${SRC}/${f}" ]]; then
    echo "ERROR: missing FlyDSL source: ${SRC}/${f}" >&2
    exit 1
  fi
  # Back up the original aiter copy once.
  if [[ -f "${DST}/${f}" && ! -f "${BAK}/${f}" ]]; then
    cp "${DST}/${f}" "${BAK}/${f}"
  fi
  cp "${SRC}/${f}" "${DST}/${f}"
  echo "synced ${f}"
done

# Clear the aiter FlyDSL JIT cache so stale compiled artifacts are not reused.
CACHE="${AITER_REPO}/aiter/jit/flydsl_cache"
if [[ -d "${CACHE}" ]]; then
  rm -rf "${CACHE:?}/"* 2>/dev/null || true
  echo "cleared aiter flydsl JIT cache: ${CACHE}"
fi

echo "done: aiter vendored FlyDSL MoE kernels synced from ${SRC}"
