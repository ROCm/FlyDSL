#!/bin/bash
set -euo pipefail

TRITON_INDEX_URL=${AITER_TRITON_INDEX_URL:-"https://pypi.amd.com/triton/rocm-7.2.0/simple/"}
TRITON_SPEC=${AITER_TRITON_SPEC:-"triton==3.7.0+amd.rocm7.2.0.gitd1660454"}

python3 -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true

echo "Installing ${TRITON_SPEC} from ${TRITON_INDEX_URL}"
python3 -m pip install --extra-index-url "${TRITON_INDEX_URL}" "${TRITON_SPEC}"

python3 - <<'PY'
import triton


def parse_version(version: str) -> tuple[int, ...]:
    version = version.split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


if parse_version(triton.__version__) < (3, 6, 0):
    raise SystemExit(f"triton>=3.6.0 is required by AITER Gluon kernels, found {triton.__version__}")

print(f"Installed triton {triton.__version__}")
PY
