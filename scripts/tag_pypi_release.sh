#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

release=0
message=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) release=1 ;;
    -m|--message)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      message="$2"
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/tag_pypi_release.sh [--release] [-m message]"
      echo "Default creates v<base>.dev<git_rev_count>; --release creates v<base>."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

base="$(
  git show HEAD:python/flydsl/__init__.py |
    sed -n 's/^__version__ = "\(.*\)"/\1/p'
)"

if [[ -z "${base}" ]]; then
  echo "Failed to read __version__ from HEAD:python/flydsl/__init__.py" >&2
  exit 1
fi

tag="v${base}"
if [[ "${release}" != "1" ]]; then
  tag="${tag}.dev$(git rev-list --count HEAD)"
fi

echo "${tag}"
if [[ -n "${message}" ]]; then
  git tag -a "${tag}" -m "${message}"
else
  git tag "${tag}"
fi
