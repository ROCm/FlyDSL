# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GPU gate for the Windows CI job: fail, never skip, without the target GPU.

The shared Windows runners differ: some enumerate an iGPU as HIP device 0, and a
runner slot can see no GPU at all, which would turn GPU tests into skips and the
job green. This lists the HIP devices torch sees, picks the first one with the
target arch, and re-checks in a fresh process, with only that device visible,
that torch and FlyDSL both see exactly one device of that arch. The chosen
HIP_VISIBLE_DEVICES is printed and, with --github-env, appended to $GITHUB_ENV.

Exit codes: 0 selected, 3 no device with the target arch, 4 the re-check failed.
"""

import argparse
import json
import os
import subprocess
import sys

_LIST = """
import json, torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(json.dumps([torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] for i in range(n)]))
"""

_RECHECK = """
import json, torch
from flydsl.runtime.device import get_rocm_arch, get_rocm_device_count
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if n else ""
print(json.dumps({"torch_count": n, "torch_arch": arch,
                  "flydsl_count": get_rocm_device_count(), "flydsl_arch": get_rocm_arch()}))
"""


def _query(code, env):
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        return None
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", required=True, help="target arch, e.g. gfx1201")
    parser.add_argument("--github-env", action="store_true", help="append HIP_VISIBLE_DEVICES to $GITHUB_ENV")
    args = parser.parse_args()

    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION"):
        print(f"{var}={os.environ.get(var, '<unset>')}")
    env = dict(os.environ)
    # An arch override would make FlyDSL's answer below meaningless.
    env.pop("FLYDSL_GPU_ARCH", None)
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)

    archs = _query(_LIST, env) or []
    print(f"HIP devices: {archs}")
    if args.arch not in archs:
        print(f"FAIL: no {args.arch} device is visible", file=sys.stderr)
        return 3

    # Indices are relative to whatever the runner already made visible.
    index = archs.index(args.arch)
    preset = env.get("HIP_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES")
    visible = preset.split(",")[index].strip() if preset else str(index)
    env["HIP_VISIBLE_DEVICES"] = visible
    env.pop("CUDA_VISIBLE_DEVICES", None)
    seen = _query(_RECHECK, env)
    print(f"With HIP_VISIBLE_DEVICES={visible}: {seen}")
    want = {"torch_count": 1, "torch_arch": args.arch, "flydsl_count": 1, "flydsl_arch": args.arch}
    if seen != want:
        print(f"FAIL: expected {want}", file=sys.stderr)
        return 4

    if args.github_env:
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
            f.write(f"HIP_VISIBLE_DEVICES={visible}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
