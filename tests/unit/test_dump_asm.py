# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FLYDSL_DEBUG_DUMP_ASM writes only the final ISA that FLYDSL_DUMP_IR would write."""

import pytest

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@pytest.mark.parametrize("warm_disk_cache", [False, True])
def test_dump_asm_writes_only_the_final_isa(monkeypatch, tmp_path, warm_disk_cache):
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path / "dump"))
    monkeypatch.delenv("FLYDSL_DUMP_IR", raising=False)
    monkeypatch.delenv("FLYDSL_DEBUG_DUMP_ASM", raising=False)
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")

    @flyc.kernel
    def kernel():
        pass

    def make_launcher():
        @flyc.jit
        def launch():
            kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))

        return launch

    if warm_disk_cache:
        make_launcher()()
        assert list((tmp_path / "cache").rglob("*.pkl"))

    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "1")
    make_launcher()()
    dumped = [p for p in (tmp_path / "dump").rglob("*") if p.is_file()]
    assert len(dumped) == 1 and dumped[0].name.endswith("_final_isa.s"), dumped
    assert ".amdhsa_kernel" in dumped[0].read_text()


def test_run_only_rejects_isa_dumping(monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "1")

    @flyc.jit
    def launch():
        pass

    with pytest.raises(ValueError, match="RUN_ONLY=1.*FLYDSL_DEBUG_DUMP_ASM=1"):
        launch()
