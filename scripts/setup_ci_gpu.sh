#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

# Shared environment for the single-GPU correctness and benchmark jobs.
# The workflow supplies the checked-out source directory, a unique container
# name, the Docker image, and the architecture-specific aiter/GPU settings.
set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set}"
: "${GITHUB_ENV:?GITHUB_ENV must be set}"
: "${CI_SOURCE_DIR:?CI_SOURCE_DIR must be set}"
: "${CI_CONTAINER:?CI_CONTAINER must be set}"
: "${DOCKER_IMAGE:?DOCKER_IMAGE must be set}"
ci_wheel_dir="${GITHUB_WORKSPACE}/${CI_SOURCE_DIR}-wheels"

shopt -s nullglob
pr_wheels=("${ci_wheel_dir}"/pr/*.whl)
if [ "${#pr_wheels[@]}" -ne 1 ]; then
  printf 'Expected exactly one PR wheel, found %s:\n' "${#pr_wheels[@]}" >&2
  printf '  %s\n' "${pr_wheels[@]}" >&2
  exit 1
fi
(cd "${ci_wheel_dir}/pr" && sha256sum -c SHA256SUMS)
if [ -d "${ci_wheel_dir}/base" ]; then
  base_wheels=("${ci_wheel_dir}"/base/*.whl)
  if [ "${#base_wheels[@]}" -ne 1 ]; then
    printf 'Expected exactly one base wheel, found %s:\n' "${#base_wheels[@]}" >&2
    printf '  %s\n' "${base_wheels[@]}" >&2
    exit 1
  fi
  (cd "${ci_wheel_dir}/base" && sha256sum -c SHA256SUMS)
else
  echo "::warning title=Baseline wheel unavailable::Exact base benchmark will be skipped."
fi
(cd "${ci_wheel_dir}/tools" && sha256sum -c SHA256SUMS)

# Remove only this job's container. Other runs and runner slots may share Docker.
docker rm -f "${CI_CONTAINER}" >/dev/null 2>&1 || true
if [ -f /etc/podinfo/gha-render-devices ]; then
  ci_device_flags="$(</etc/podinfo/gha-render-devices)"
  read -r -a ci_device_args <<<"${ci_device_flags//$'\n'/ }"
else
  ci_device_args=(--device /dev/dri)
fi
ci_gpu_env_args=()
if [ -n "${CI_HIP_VISIBLE_DEVICES:-}" ]; then
  ci_gpu_env_args=(--env "HIP_VISIBLE_DEVICES=${CI_HIP_VISIBLE_DEVICES}")
fi
docker run -dt --network=host --user root --device=/dev/kfd \
  "${ci_device_args[@]}" "${ci_gpu_env_args[@]}" \
  -v "${GITHUB_WORKSPACE}/${CI_SOURCE_DIR}:/flydsl-test" \
  -v "${ci_wheel_dir}:/flydsl-ci-wheels:ro" \
  --ipc=host --group-add video --shm-size 16g --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined -w /flydsl-test \
  --name "${CI_CONTAINER}" "${DOCKER_IMAGE}"

docker exec "${CI_CONTAINER}" bash -c 'apt-get update && apt-get install -y cmake build-essential patchelf'
docker exec "${CI_CONTAINER}" bash -c 'python3 -m pip install -U pip setuptools wheel'
docker exec "${CI_CONTAINER}" bash -c 'python3 -m pip install "ninja>=1.11.1"'
docker exec "${CI_CONTAINER}" bash -c 'if command -v rocm-sdk >/dev/null 2>&1; then rocm-sdk init; fi'
docker exec "${CI_CONTAINER}" bash -c 'python3 -m pip install -U "hypothesis>=6.82.0"'
docker exec "${CI_CONTAINER}" bash -c 'git config --global --add safe.directory /flydsl-test'
ROCM_PATH="$(docker exec "${CI_CONTAINER}" bash -c 'if command -v rocm-sdk >/dev/null 2>&1; then rocm-sdk path --root; elif [ -d /opt/rocm ]; then printf "/opt/rocm\n"; fi')"
test -n "${ROCM_PATH}"
docker exec "${CI_CONTAINER}" test -d "${ROCM_PATH}"
export ROCM_PATH
printf 'ROCM_PATH=%s\n' "${ROCM_PATH}" >>"${GITHUB_ENV}"

docker exec "${CI_CONTAINER}" bash -c 'python3 -m pip install --force-reinstall --no-deps /flydsl-ci-wheels/pr/*.whl'
docker exec "${CI_CONTAINER}" bash -c 'mkdir -p /flydsl-test/build-fly/bin && cp /flydsl-ci-wheels/pr/fly-opt /flydsl-test/build-fly/bin/fly-opt && chmod +x /flydsl-test/build-fly/bin/fly-opt'
docker exec "${CI_CONTAINER}" bash -c 'cp /flydsl-ci-wheels/tools/FileCheck /usr/local/bin/FileCheck && chmod +x /usr/local/bin/FileCheck'
docker exec -e ROCM_PATH "${CI_CONTAINER}" /flydsl-test/build-fly/bin/fly-opt --version
docker exec -e ROCM_PATH "${CI_CONTAINER}" python3 -c 'import flydsl; from flydsl._mlir.ir import Context; print("Installed FlyDSL CI wheel")'

if [ "${CI_PREPARE_AITER:-true}" = "true" ]; then
  docker exec "${CI_CONTAINER}" bash -c 'rm -rf /tmp/aiter && git clone --depth 1 --recursive --shallow-submodules https://github.com/ROCm/aiter.git /tmp/aiter'
  docker exec -i "${CI_CONTAINER}" python3 <<'PYTHON'
from pathlib import Path

src = Path('/tmp/aiter/requirements.txt')
lines = [line for line in src.read_text().splitlines() if line.strip() and not line.strip().startswith('flydsl==')]
Path('/tmp/aiter/requirements-flydsl-ci.txt').write_text('\n'.join(lines) + '\n')
PYTHON
  docker exec "${CI_CONTAINER}" python3 -m pip install -r /tmp/aiter/requirements-flydsl-ci.txt
  docker exec "${CI_CONTAINER}" bash -c 'cd /tmp/aiter && bash .github/scripts/install_triton.sh'
  docker exec "${CI_CONTAINER}" python3 -c 'import torch; import triton; assert tuple(map(int, triton.__version__.split("+", 1)[0].split(".")[:3])) >= (3, 6, 0), triton.__version__; print("Using Triton", triton.__version__, "with PyTorch", torch.__version__)'
fi
