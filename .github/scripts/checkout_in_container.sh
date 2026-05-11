#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: checkout_in_container.sh <container> <checkout-dir> [--submodules]

Required env:
  CHECKOUT_REPOSITORY  GitHub repository, e.g. ROCm/FlyDSL
  CHECKOUT_SHA         Commit SHA to fetch and check out
EOF
}

if [ "$#" -lt 2 ]; then
  usage >&2
  exit 2
fi

CONTAINER="$1"
CHECKOUT_DIR="$2"
shift 2

INIT_SUBMODULES=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --submodules)
      INIT_SUBMODULES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

: "${CHECKOUT_REPOSITORY:?CHECKOUT_REPOSITORY is required}"
: "${CHECKOUT_SHA:?CHECKOUT_SHA is required}"

docker exec \
  -e CHECKOUT_REPOSITORY="${CHECKOUT_REPOSITORY}" \
  -e CHECKOUT_SHA="${CHECKOUT_SHA}" \
  -e CHECKOUT_DIR="${CHECKOUT_DIR}" \
  -e INIT_SUBMODULES="${INIT_SUBMODULES}" \
  "${CONTAINER}" bash -c '
    set -euo pipefail
    echo "Checking out ${CHECKOUT_REPOSITORY}@${CHECKOUT_SHA} in ${CHECKOUT_DIR}."
    mkdir -p "${CHECKOUT_DIR}"
    cd "${CHECKOUT_DIR}"
    find . -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    git init
    git remote add origin "https://github.com/${CHECKOUT_REPOSITORY}.git"
    git fetch --no-tags --depth=1 origin "${CHECKOUT_SHA}"
    git checkout --force FETCH_HEAD
    git config --global --add safe.directory "${CHECKOUT_DIR}"

    if [ "${INIT_SUBMODULES}" = "1" ]; then
      git submodule sync --recursive
      git submodule update --init --recursive
    fi
  '
