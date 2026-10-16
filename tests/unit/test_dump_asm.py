# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FLYDSL_DEBUG_DUMP_ASM writes the final ISA and keeps the JIT cache, compiled without a GPU."""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

UNIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = UNIT_DIR.parents[1]
# Each run is a fresh process importing the kernel below from this file.
DRIVER = "from test_dump_asm import single\nsingle()\n"


@flyc.kernel
def tiny_kernel():
    pass


@flyc.jit
def single():
    tiny_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))


def _run(tmp_path: Path, **extra) -> subprocess.CompletedProcess:
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("FLYDSL_RUNTIME_", "FLYDSL_DUMP_", "FLYDSL_DEBUG_"))
    }
    env.update(
        PYTHONPATH=os.pathsep.join([str(UNIT_DIR), str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")]),
        COMPILE_ONLY="1",
        ARCH="gfx950",
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    env.update({k: str(v) for k, v in extra.items()})
    return subprocess.run([sys.executable, "-c", DRIVER], env=env, capture_output=True, text=True, timeout=600)


def _object_bytes(entry: Path) -> bytes:
    from flydsl._mlir import ir
    from flydsl._mlir._mlir_libs._mlirDialectsGPU import ObjectAttr
    from flydsl.compiler.jit_function import _create_mlir_context

    with open(entry, "rb") as f:
        artifact = pickle.load(f)
    with _create_mlir_context():
        module = ir.Module.parse(artifact._ir_text)
        binary = next(op.operation for op in module.body.operations if op.operation.name == "gpu.binary")
        return bytes(ObjectAttr(ir.ArrayAttr(binary.attributes["objects"])[0]).object)


def test_dump_asm_writes_the_isa_and_the_cache_entry(tmp_path):
    """One run gives the final .s and the cache entry compiled with it, and dumps nothing else."""
    proc = _run(tmp_path, FLYDSL_DEBUG_DUMP_ASM=1, FLYDSL_DUMP_DIR=tmp_path / "dump")
    assert proc.returncode == 0, proc.stderr

    dumped = sorted(p.relative_to(tmp_path / "dump") for p in (tmp_path / "dump").rglob("*") if p.is_file())
    assert dumped == [Path("tiny_kernel_0/final_isa.s")], dumped
    assert ".amdhsa_kernel tiny_kernel_0" in (tmp_path / "dump" / dumped[0]).read_text()
    assert len(list((tmp_path / "cache").rglob("*.pkl"))) == 1


def test_dump_asm_compiles_on_a_cache_hit(tmp_path):
    """A disk-cache hit would skip compilation and leave nothing to dump, so the run recompiles."""
    assert _run(tmp_path).returncode == 0
    proc = _run(tmp_path, FLYDSL_DEBUG_DUMP_ASM=1, FLYDSL_DUMP_DIR=tmp_path / "dump")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "dump" / "tiny_kernel_0" / "final_isa.s").is_file()


def test_dump_asm_does_not_change_the_code_object(tmp_path):
    """Splitting the pipeline to dump the ISA must not change what gets compiled."""
    assert _run(tmp_path / "plain").returncode == 0
    assert _run(tmp_path / "asm", FLYDSL_DEBUG_DUMP_ASM=1, FLYDSL_DUMP_DIR=tmp_path / "dump").returncode == 0
    (plain,) = (tmp_path / "plain" / "cache").rglob("*.pkl")
    (asm,) = (tmp_path / "asm" / "cache").rglob("*.pkl")
    assert _object_bytes(plain) == _object_bytes(asm)


def test_dump_asm_is_refused_with_run_only(tmp_path):
    proc = _run(tmp_path, FLYDSL_DEBUG_DUMP_ASM=1, FLYDSL_RUNTIME_RUN_ONLY=1)
    assert proc.returncode != 0
    assert "incompatible with FLYDSL_DEBUG_DUMP_ASM" in proc.stderr


WARM_DRIVER = """
import os, sys
from test_dump_asm import single
single()                                   # warm both in-process caches without a dump
os.environ["FLYDSL_DEBUG_DUMP_ASM"] = "1"
for d in sys.argv[1:]:
    os.environ["FLYDSL_DUMP_DIR"] = d
    single()
    single()                               # same directory: served from cache, not dumped again
"""


def test_dump_asm_after_warm_in_process_caches(tmp_path):
    """Enabling the dump, or pointing it elsewhere, mid-process still writes the ISA once per directory."""
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("FLYDSL_RUNTIME_", "FLYDSL_DUMP_", "FLYDSL_DEBUG_"))
    }
    env.update(
        PYTHONPATH=os.pathsep.join([str(UNIT_DIR), str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")]),
        COMPILE_ONLY="1",
        ARCH="gfx950",
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    dirs = [tmp_path / "first", tmp_path / "second"]
    proc = subprocess.run(
        [sys.executable, "-c", WARM_DRIVER, *map(str, dirs)], env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr
    for d in dirs:
        assert (d / "tiny_kernel_0" / "final_isa.s").is_file(), d
    assert proc.stdout.count("dump final_isa ->") == len(dirs), proc.stdout
