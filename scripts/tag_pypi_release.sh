#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

release=0
message=""

die() {
  echo "$*" >&2
  exit 1
}

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
      echo "Default creates the next monotonic v<base>.devN tag; --release creates v<base>."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$(git rev-parse --is-shallow-repository)" == "true" ]]; then
  die "Refusing to create a release tag from a shallow repository. Run: git fetch --unshallow --tags origin"
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  die "Refusing to create a release tag with tracked changes in the working tree."
fi

origin_main="refs/remotes/origin/main"
if ! git show-ref --verify --quiet "${origin_main}"; then
  die "Missing origin/main. Run: git fetch origin main --tags"
fi

if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "${origin_main}")" ]]; then
  die "HEAD must match origin/main before creating a release tag. Fetch or push main first."
fi

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
  commit_count="$(git rev-list --count HEAD)"
  max_dev_number=0
  while IFS= read -r existing_tag; do
    suffix="${existing_tag#${tag}.dev}"
    if [[ "${suffix}" =~ ^[0-9]+$ ]] && ((10#${suffix} > max_dev_number)); then
      max_dev_number=$((10#${suffix}))
    fi
  done < <(git tag --list "${tag}.dev*")

  dev_number=$((10#${commit_count}))
  if ((max_dev_number >= dev_number)); then
    dev_number=$((max_dev_number + 1))
  fi

  tag="${tag}.dev${dev_number}"
fi

if git show-ref --verify --quiet "refs/tags/${tag}"; then
  die "Tag ${tag} already exists."
fi

echo "${tag}"
if [[ -n "${message}" ]]; then
  git tag -a "${tag}" -m "${message}"
else
  git tag "${tag}"
fi
