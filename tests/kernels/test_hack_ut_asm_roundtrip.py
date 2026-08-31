#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end coverage for FLYDSL_HACK_UT_ASM (run a kernel from a dumped .s).

Each phase runs in its own process: the override is read from the environment and
the in-process caches are deliberately left live, so mixing phases in one process
would not exercise what a real dump/edit/re-run loop does.
"""

import os
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


def _run_phase(out_path: Path, extra_env: dict) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")])
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER, str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"driver failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stdout


def _dump_isa(tmp_path: Path) -> Path:
    """Phase A: run with FLYDSL_DUMP_IR and return the dumped .s."""
    dump_dir = tmp_path / "dump"
    _run_phase(
        tmp_path / "ref.pt",
        {"FLYDSL_DUMP_IR": "1", "FLYDSL_DUMP_DIR": str(dump_dir), "FLYDSL_RUNTIME_ENABLE_CACHE": "0"},
    )
    dumped = sorted(dump_dir.rglob("*_final_isa.s"))
    assert len(dumped) == 1, f"expected exactly one dumped ISA, got {dumped}"
    return dumped[0]


def test_dumped_isa_roundtrips(tmp_path):
    """Re-running the un-edited dump must reproduce the compiler's own results."""
    isa_path = _dump_isa(tmp_path)
    reference = torch.load(tmp_path / "ref.pt")

    stdout = _run_phase(tmp_path / "hacked.pt", {"FLYDSL_HACK_UT_ASM": str(isa_path)})
    assert "FLYDSL_HACK_UT_ASM: substituted" in stdout

    torch.testing.assert_close(torch.load(tmp_path / "hacked.pt"), reference, rtol=0, atol=0)


def test_edited_isa_actually_runs(tmp_path):
    """An edit to the .s must change the result, or the override is a no-op."""
    isa_path = _dump_isa(tmp_path)
    reference = torch.load(tmp_path / "ref.pt")

    # Return from the kernel entry immediately, leaving the output at its zero init.
    lines = isa_path.read_text(encoding="utf-8").splitlines(keepends=True)
    entry = next(i for i, line in enumerate(lines) if line.startswith(f"{isa_path.parent.name}:"))
    lines.insert(entry + 1, "\ts_endpgm\n")
    edited = tmp_path / "edited.s"
    edited.write_text("".join(lines), encoding="utf-8")

    _run_phase(tmp_path / "edited.pt", {"FLYDSL_HACK_UT_ASM": str(edited)})
    result = torch.load(tmp_path / "edited.pt")

    assert not torch.equal(result, reference), "edited ISA produced the original result"
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)


def test_mismatched_kernel_is_rejected(tmp_path):
    """Pointing at a .s for a different kernel must fail loudly, not silently pass through."""
    isa_path = _dump_isa(tmp_path)
    wrong = tmp_path / "wrong.s"
    wrong.write_text(
        isa_path.read_text(encoding="utf-8").replace(isa_path.parent.name, "someOtherKernel"),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")])
    env["FLYDSL_HACK_UT_ASM"] = str(wrong)
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER, str(tmp_path / "unused.pt")],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode != 0
    assert "refusing to substitute" in proc.stderr


def test_missing_file_is_rejected(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")])
    env["FLYDSL_HACK_UT_ASM"] = str(tmp_path / "nonexistent.s")
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER, str(tmp_path / "unused.pt")],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode != 0
    assert "points at a missing file" in proc.stderr
