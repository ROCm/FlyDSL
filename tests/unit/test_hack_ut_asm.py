# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os
from pathlib import Path

import pytest

from flydsl.compiler.backends.rocm import RocmBackend
from flydsl.compiler.diagnostics import DSLCompileError
from flydsl.compiler.hack_asm import _assemble_isa_to_hsaco, _infer_kernel_names_from_s

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

ELF_MAGIC = b"\x7fELF"

MINIMAL_ISA = """\t.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
\t.amdhsa_code_object_version 6
\t.text
\t.globl\ttiny_kernel
\t.p2align\t8
\t.type\ttiny_kernel,@function
tiny_kernel:
\ts_endpgm
.Lfunc_end0:
\t.size\ttiny_kernel, .Lfunc_end0-tiny_kernel

\t.amdhsa_kernel tiny_kernel
\t\t.amdhsa_group_segment_fixed_size 0
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size 0
\t\t.amdhsa_next_free_vgpr 4
\t\t.amdhsa_next_free_sgpr 1
\t\t.amdhsa_accum_offset 4
\t.end_amdhsa_kernel
"""


_CLANG = Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "llvm" / "bin" / "clang"
requires_clang = pytest.mark.skipif(not _CLANG.is_file(), reason=f"ROCm clang not found at {_CLANG}")


def test_infer_kernel_names_from_s():
    names = _infer_kernel_names_from_s(MINIMAL_ISA)
    assert names == {"tiny_kernel"}


def test_infer_kernel_names_from_s_ignores_unrelated_directives():
    assert _infer_kernel_names_from_s("\t.text\n\t.p2align 8\n") == set()


def test_isa_assemble_arch_matches_pipeline():
    backend = RocmBackend(RocmBackend.make_target("gfx950"))
    assert backend.isa_assemble_arch() == "gfx950"


def test_binary_cli_options_tracks_compile_hints():
    backend = RocmBackend(RocmBackend.make_target("gfx950"))
    assert backend.binary_cli_options(compile_hints={}) == ""
    opts = backend.binary_cli_options(compile_hints={"waves_per_eu": 2, "maxnreg": 128})
    assert "--amdgpu-waves-per-eu=2" in opts
    assert "--amdgpu-num-vgpr=128" in opts


@requires_clang
def test_assemble_isa_to_hsaco_roundtrip(tmp_path):
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    obj = _assemble_isa_to_hsaco(src, "gfx950")
    assert obj.startswith(ELF_MAGIC)
    assert b"tiny_kernel" in obj


@requires_clang
def test_assemble_isa_to_hsaco_reports_assembler_diagnostics(tmp_path):
    src = tmp_path / "bad.s"
    src.write_text("this is not valid asm\n", encoding="utf-8")
    with pytest.raises(DSLCompileError) as excinfo:
        _assemble_isa_to_hsaco(src, "gfx950")
    # The clang stderr must reach the user, not be swallowed into a generic message.
    assert "invalid instruction" in str(excinfo.value)


@requires_clang
def test_assemble_isa_to_hsaco_rejects_arch_mismatch(tmp_path):
    """A .s dumped for another arch must be refused, not silently assembled."""
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    with pytest.raises(DSLCompileError) as excinfo:
        _assemble_isa_to_hsaco(src, "gfx942")
    assert "target id" in str(excinfo.value)
