#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Print the actions/cache key for the MLIR install that the given tree builds.
#
# Derived from file *contents*, not paths, so it is computable for any commit
# whose LLVM inputs have been extracted anywhere on disk - that is what lets a PR
# look up the install belonging to its base commit's pin. Both workflow call
# sites use this script; inlining the formula elsewhere would let the two keys
# drift and return an install built from the wrong pin.
#
# Total by construction: a missing input contributes a marker, not an error. It
# replaced a hashFiles() expression, which tolerated missing files too, and a PR
# that deletes one must still get a usable key rather than a failed job.
set -euo pipefail

TREE_ROOT="${1:?usage: ci_mlir_cache_key.sh <tree-root>}"

INPUTS=(
  thirdparty/llvm-build-info.json
  thirdparty/llvm-rocdl-lld-argv0.patch
  scripts/build_llvm.sh
)

# Per-file digests, so moving bytes across a file boundary changes the key.
digests=""
for input in "${INPUTS[@]}"; do
  if [[ -f "${TREE_ROOT}/${input}" ]]; then
    digests+="${input}:$(sha256sum <"${TREE_ROOT}/${input}" | cut -d' ' -f1)"$'\n'
  else
    digests+="${input}:absent"$'\n'
  fi
done
digest="$(printf '%s' "${digests}" | sha256sum | cut -c1-40)"

printf 'mlir-install-%s-%s-%s-%s-%s\n' \
  "${RUNNER_OS:?RUNNER_OS must be set}" \
  "${RUNNER_ARCH:?RUNNER_ARCH must be set}" \
  "${MLIR_CACHE_VERSION:?MLIR_CACHE_VERSION must be set}" \
  "${LLVM_BUILD_PROFILE:?LLVM_BUILD_PROFILE must be set}" \
  "${digest}"
