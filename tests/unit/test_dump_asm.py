# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FLYDSL_DEBUG_DUMP_ASM writes only the final ISA that FLYDSL_DUMP_IR would write."""

import pytest

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def test_dump_asm_writes_only_the_final_isa(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path / "dump"))
    monkeypatch.delenv("FLYDSL_DUMP_IR", raising=False)

    @flyc.kernel
    def kernel():
        pass

    @flyc.jit
    def launch():
        kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))

    launch()
    dumped = [p for p in (tmp_path / "dump").rglob("*") if p.is_file()]
    assert len(dumped) == 1 and dumped[0].name.endswith("_final_isa.s"), dumped
    assert ".amdhsa_kernel" in dumped[0].read_text()
