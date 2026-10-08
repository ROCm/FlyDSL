# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""scripts/patch_cache_asm.py against real cache entries, compiled without a GPU."""

import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("patch_cache_asm", REPO_ROOT / "scripts" / "patch_cache_asm.py")
pca = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pca)

_CLANG = Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "llvm" / "bin" / "clang"
requires_clang = pytest.mark.skipif(not _CLANG.is_file(), reason=f"ROCm clang not found at {_CLANG}")

ISA_TEMPLATE = """\t.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
\t.amdhsa_code_object_version 6
\t.text
{kernels}"""

KERNEL_TEMPLATE = """\t.globl\t{name}
\t.p2align\t8
\t.type\t{name},@function
{name}:
\ts_endpgm
.Lfunc_end_{name}:
\t.size\t{name}, .Lfunc_end_{name}-{name}

\t.amdhsa_kernel {name}
\t\t.amdhsa_group_segment_fixed_size 0
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size 0
\t\t.amdhsa_next_free_vgpr 4
\t\t.amdhsa_next_free_sgpr 1
\t\t.amdhsa_accum_offset 4
\t.end_amdhsa_kernel
"""


def _isa(tmp_path: Path, *names: str) -> Path:
    path = tmp_path / "edited.s"
    path.write_text(ISA_TEMPLATE.format(kernels="".join(KERNEL_TEMPLATE.format(name=n) for n in names)))
    return path


def _populate(cache: Path, *calls: str) -> list:
    """Compile the given calls into ``cache`` without a GPU; return the entries written."""
    driver = "import sys\nfrom tests.kernels import _patch_cache_asm_kernels as k\nfor c in sys.argv[1:]:\n    eval(c, vars(k))\n"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")]),
        COMPILE_ONLY="1",
        ARCH="gfx950",
        FLYDSL_RUNTIME_CACHE_DIR=str(cache),
    )
    env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    proc = subprocess.run([sys.executable, "-c", driver, *calls], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    entries = sorted(cache.rglob("*.pkl"))
    assert len(entries) == len(calls), entries
    return entries


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


def _patch(*args) -> str:
    with pytest.raises(SystemExit) as excinfo:
        pca.main([str(a) for a in args])
    return str(excinfo.value.code)


@pytest.mark.parametrize("relative", [".flydsl/cache", ".flydsl", ".flydsl/cache/single_0123"])
def test_refuses_anything_overlapping_the_default_cache(tmp_path, monkeypatch, relative):
    """A patched entry in the default cache would be served to every later ordinary run.

    Entries are found recursively, so a parent of the default cache reaches it as
    surely as the cache itself, and a subdirectory is part of it.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    msg = _patch(tmp_path / "home" / relative, _isa(tmp_path, "tiny_kernel_0"))
    assert "overlaps the default cache" in msg


@requires_clang
def test_patch_replaces_the_code_object(tmp_path):
    (entry,) = _populate(tmp_path / "cache", "single()")
    original = _object_bytes(entry)
    asm = _isa(tmp_path, "tiny_kernel_0")

    pca.main([str(tmp_path / "cache"), str(asm)])
    patched = _object_bytes(entry)
    assert patched != original
    assert patched == pca.assemble(asm, "gfx950")
    assert not asm.with_suffix(".hsaco").exists(), "saving the code object is opt-in"

    pca.main([str(tmp_path / "cache"), str(asm), "--save-hsaco"])
    assert asm.with_suffix(".hsaco").read_bytes() == patched


@requires_clang
def test_specializations_are_refused(tmp_path):
    """Two constexpr values launch the same symbol and their entries cannot be told apart.

    Patching one of them by guess would, half the time, leave the run executing the
    compiler's code under RUN_ONLY with no error -- so neither is touched.
    """
    first, second = _populate(tmp_path / "cache", "sized(64)", "sized(128)")
    before = {first: _object_bytes(first), second: _object_bytes(second)}

    msg = _patch(tmp_path / "cache", _isa(tmp_path, "tiny_kernel_0"))
    assert "matches 2 entries" in msg and "running only the case you dumped" in msg
    assert {first: _object_bytes(first), second: _object_bytes(second)} == before


@requires_clang
def test_partial_s_is_refused(tmp_path):
    """Replacing the binary with only some of its kernels would drop the rest."""
    (entry,) = _populate(tmp_path / "cache", "both()")
    before = entry.read_bytes()

    msg = _patch(tmp_path / "cache", _isa(tmp_path, "tiny_kernel_0"))
    assert "other_kernel_1" in msg and "would drop them" in msg
    assert entry.read_bytes() == before

    pca.main([str(tmp_path / "cache"), str(_isa(tmp_path, "tiny_kernel_0", "other_kernel_1"))])
    assert entry.read_bytes() != before
