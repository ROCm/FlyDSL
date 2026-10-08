#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End to end: dump the ISA, cache the kernel, patch the entry, run it under RUN_ONLY.

Each phase runs in its own process, as it would in a real dump/edit/re-run loop.
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


def _env(extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FLYDSL_RUNTIME_", "FLYDSL_DUMP_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")])
    env.update(extra)
    return env


def _run(out: Path, extra: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", DRIVER, str(out)], env=_env(extra), capture_output=True, text=True, timeout=600
    )


def _ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _prepare(tmp_path: Path) -> tuple:
    """Dump the ISA and cache the compiler's artifact; return (isa_path, cache_dir, reference)."""
    dump = tmp_path / "dump"
    _ok(_run(tmp_path / "dumped.pt", {"FLYDSL_DUMP_IR": "1", "FLYDSL_DUMP_DIR": str(dump)}))
    (isa,) = sorted(dump.rglob("*_final_isa.s"))

    cache = tmp_path / "cache"
    _ok(_run(tmp_path / "ref.pt", {"FLYDSL_RUNTIME_CACHE_DIR": str(cache)}))
    return isa, cache, torch.load(tmp_path / "ref.pt")


def _patch(cache: Path, isa: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "patch_cache_asm.py"), str(cache), str(isa)],
        env=_env({}),
        capture_output=True,
        text=True,
        timeout=600,
    )
    _ok(proc)
    assert "patched" in proc.stdout


def _run_patched(cache: Path, out: Path) -> torch.Tensor:
    _ok(_run(out, {"FLYDSL_RUNTIME_CACHE_DIR": str(cache), "FLYDSL_RUNTIME_RUN_ONLY": "1"}))
    return torch.load(out)


def test_unedited_dump_reproduces_the_compiler(tmp_path):
    """Patching in the dump unchanged must give the compiler's own result, bit for bit."""
    isa, cache, reference = _prepare(tmp_path)
    _patch(cache, isa)
    torch.testing.assert_close(_run_patched(cache, tmp_path / "out.pt"), reference, rtol=0, atol=0)


def test_edited_assembly_is_what_runs(tmp_path):
    """An edit must change the result, or the patch never reached the GPU."""
    isa, cache, reference = _prepare(tmp_path)

    # Return from the kernel entry immediately, leaving the output at its zero init.
    lines = isa.read_text(encoding="utf-8").splitlines(keepends=True)
    entry = next(i for i, line in enumerate(lines) if line.startswith(f"{isa.parent.name}:"))
    lines.insert(entry + 1, "\ts_endpgm\n")
    isa.write_text("".join(lines), encoding="utf-8")
    _patch(cache, isa)

    result = _run_patched(cache, tmp_path / "out.pt")
    assert not torch.equal(result, reference), "the edited assembly produced the compiler's result"
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)


def test_run_only_miss_is_an_error(tmp_path):
    """The workflow relies on RUN_ONLY failing on a miss rather than recompiling."""
    proc = _run(
        tmp_path / "out.pt", {"FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "empty"), "FLYDSL_RUNTIME_RUN_ONLY": "1"}
    )
    assert proc.returncode != 0
    assert "no usable AOT cache" in proc.stderr
