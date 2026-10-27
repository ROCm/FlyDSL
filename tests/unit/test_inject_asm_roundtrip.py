#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End to end through tools/inject_asm/inject_asm.py prepare and run, on the vecAdd kernel.

Every step runs in its own process, as it would in a real dump/edit/re-run loop.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

# Dumps for gfx11 and later need prepare --clang with the clang of the LLVM FlyDSL
# was built with; the ROCm clang this test assembles with cannot read them.
_arch = str(get_rocm_arch())
if not _arch.startswith("gfx9"):
    pytest.skip(f"the ROCm clang cannot reassemble a {_arch} dump", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs vecAdd once and saves the result, so phases can be compared across processes.
DRIVER = """
import sys
import torch
import flydsl.compiler as flyc
from tests.kernels.test_vec_add import vecAdd

SIZE, THREADS, VEC_WIDTH = 4096, 256, 4
torch.manual_seed(0)
a = torch.randn(SIZE, device="cuda", dtype=torch.float32)
b = torch.randn(SIZE, device="cuda", dtype=torch.float32)
c = torch.zeros(SIZE, device="cuda", dtype=torch.float32)

stream = torch.cuda.Stream()
tA = flyc.from_torch_tensor(a).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)
vecAdd(tA, b, c, SIZE, SIZE, THREADS, VEC_WIDTH, stream=stream)
torch.cuda.synchronize()
torch.save(c.cpu(), sys.argv[1])
"""


def _tool(*args, **extra_env) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FLYDSL_RUNTIME_", "FLYDSL_DUMP_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")])
    env.update(extra_env)
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "inject_asm" / "inject_asm.py"), *map(str, args)]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)


def _ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_prepare_edit_run(tmp_path):
    ws, out = tmp_path / "ws", tmp_path / "out.pt"
    _ok(_tool("prepare", ws, "--", sys.executable, "-c", DRIVER, out))
    # The caching run is the last one prepare makes, so out.pt holds the compiler's result.
    reference = torch.load(out)

    # The unedited dump must give the compiler's result bit for bit. An exported
    # COMPILE_ONLY must not reach the command: it would skip the launch and leave
    # out.pt at its zero init while the run still exits 0.
    _ok(_tool("run", ws, "--baseline", COMPILE_ONLY="1"))
    torch.testing.assert_close(torch.load(out), reference, rtol=0, atol=0)

    # Return from the kernel entry immediately: the output must stay at its zero
    # init, or the edited code never reached the GPU.
    edited = ws / "edited.s"
    text = edited.read_text(encoding="utf-8")
    (kernel,) = re.findall(r"^\s*\.amdhsa_kernel\s+(\S+)", text, re.M)
    lines = text.splitlines(keepends=True)
    entry = next(i for i, line in enumerate(lines) if line.startswith(f"{kernel}:"))
    lines.insert(entry + 1, "\ts_endpgm\n")
    edited.write_text("".join(lines), encoding="utf-8")
    _ok(_tool("run", ws))
    torch.testing.assert_close(torch.load(out), torch.zeros_like(reference), rtol=0, atol=0)

    # run uses RUN_ONLY: a changed cache key is an error, not a silent recompile.
    proc = _tool("run", ws, FLYDSL_COMPILE_OPT_LEVEL="1")  # part of the key; prepare ran at the default
    assert proc.returncode != 0
    assert "no usable AOT cache" in proc.stderr
