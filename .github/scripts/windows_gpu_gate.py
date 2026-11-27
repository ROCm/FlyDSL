# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GPU gate for the Windows CI job: fail, never skip, without the target GPU.

The shared Windows runners differ: some enumerate an iGPU as HIP device 0, and a
runner slot can see no GPU at all, which would turn GPU tests into skips and the
job green. This lists the HIP devices torch sees, picks the first one with the
target arch, and re-checks in a fresh process, with only that device visible,
that torch and FlyDSL both see exactly one device of that arch. The chosen
HIP_VISIBLE_DEVICES is printed and, with --github-env, appended to $GITHUB_ENV.

With --driver-runtime it runs before anything is installed: it asks the display
driver's own HIP runtime (System32) which devices the runner leaves visible, and
passes if one of them has the target arch, can allocate memory, and its adapter
reports OK. That lets a bad runner slot fail in seconds rather than after the
build. If that runtime does not load or crashes, the check cannot tell and the
full gate after the install decides.

Exit codes: 0 selected, 3 no device with the target arch, 4 the re-check failed,
5 the display driver's HIP did not load or crashed (--driver-runtime only).
"""

import argparse
import ctypes
import json
import os
import re
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


def _driver_runtime_gate(arch):
    adapters = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.Status }",
        ],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    adapters = [line.rsplit("|", 1) for line in adapters if "|" in line]
    for name, status in adapters:
        print(f"adapter: {name} | {status}")
    healthy = {name for name, status in adapters if status == "OK"}

    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    # Never amdhip64_6.dll beside amdhip64_7.dll: a leftover one can crash on the newer driver.
    dll = next((d for d in ("amdhip64_7.dll", "amdhip64_6.dll") if os.path.exists(os.path.join(system32, d))), None)
    hip = None
    # Python's default DLL search skips PATH; winmode=0 is the standard search, PATH included.
    for winmode in (None, 0) if dll else ():
        try:
            hip = ctypes.WinDLL(os.path.join(system32, dll), winmode=winmode)
            break
        except OSError as e:
            print(f"{dll} (winmode={winmode}): {type(e).__name__}: {e}")
    # Inconclusive, not a failure: the SDK runtime installed later does not need
    # the driver's HIP, and the full gate decides.
    if hip is None:
        what = dll or "amdhip64_7.dll or amdhip64_6.dll"
        print(f"INCONCLUSIVE: the display driver's HIP ({what} in {system32}) did not load")
        return 5
    try:
        count = ctypes.c_int(0)
        err = hip.hipGetDeviceCount(ctypes.byref(count))
        print(f"{dll}: hipGetDeviceCount -> {count.value} (error {err})")
        found = False
        for i in range(count.value if err == 0 else 0):
            # name[256] leads every hipDeviceProp_t layout; gcnArchName moves between
            # versions, so look for it rather than hard-coding an offset.
            prop = ctypes.create_string_buffer(8192)
            get_properties = getattr(hip, "hipGetDevicePropertiesR0600", None) or hip.hipGetDeviceProperties
            get_properties(prop, i)
            name = prop.raw[:256].split(b"\0")[0].decode(errors="replace")
            match = re.search(rb"(gfx[0-9a-f]+)[:\0]", prop.raw[256:])
            dev_arch = match.group(1).decode() if match else "?"
            mem = ctypes.c_void_p()
            usable = hip.hipSetDevice(i) == 0 and hip.hipMalloc(ctypes.byref(mem), ctypes.c_size_t(1 << 20)) == 0
            if usable:
                hip.hipFree(mem)
            adapter_ok = name in healthy
            print(
                f"HIP device {i}: {name} | {dev_arch} | {'allocates' if usable else 'cannot allocate'}"
                f" | adapter {'OK' if adapter_ok else 'not OK'}"
            )
            found = found or (dev_arch == arch and usable and adapter_ok)
    except OSError as e:  # e.g. an access violation inside the driver's HIP
        print(f"INCONCLUSIVE: the display driver's HIP ({dll}) raised {type(e).__name__}: {e}")
        return 5
    if not found:
        print(f"FAIL: no visible, working {arch} device", file=sys.stderr)
        return 3
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", required=True, help="target arch, e.g. gfx1201")
    parser.add_argument("--github-env", action="store_true", help="append HIP_VISIBLE_DEVICES to $GITHUB_ENV")
    parser.add_argument("--driver-runtime", action="store_true", help="check with the display driver's HIP only")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep the report ahead of a FAIL line on stderr

    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION"):
        print(f"{var}={os.environ.get(var, '<unset>')}")
    if args.driver_runtime:
        return _driver_runtime_gate(args.arch)
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
