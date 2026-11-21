# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A gfx950 kernel that LLVM places in VGPRs plus AGPRs is lowered again so that it fits the VGPRs."""

import re

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function
from flydsl.compiler.backends import rocm
from flydsl.compiler.backends.rocm import RocmBackend
from kernels.gemm.preshuffle_gemm import compile_preshuffle_gemm

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

# 252 VGPRs and one 8-register MFMA operand in AGPRs; reversing the local allocation order fits it in 252 VGPRs.
FITS_WITHOUT_AGPRS = dict(N=5120, K=1536, tile_m=64, tile_n=256, tile_k=256, use_async_copy=True, lds_stage=2)
# 256 VGPRs and 24 AGPRs; every allocation order within 256 registers spills.
NEEDS_A_SPILL = dict(N=5120, K=1536, tile_m=128, tile_n=256, tile_k=128, lds_stage=2)


@pytest.fixture(autouse=True)
def _compile_only(monkeypatch):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


def _compiled_ir(**config):
    m, n, k = 256, config["N"], config["K"]
    launch = compile_preshuffle_gemm(in_dtype="fp8", **config)
    flyc.compile(
        launch,
        torch.empty(m * n, dtype=torch.bfloat16),
        torch.empty(m * k, dtype=torch.float8_e4m3fn),
        torch.empty(n * k, dtype=torch.float8_e4m3fn),
        torch.empty(m, dtype=torch.float32),
        torch.empty(n, dtype=torch.float32),
        torch.empty(0, dtype=torch.bfloat16),
        m,
        n,
        fx.Stream(None),
    )
    return launch._last_compiled[1].ir


def _registers(ir):
    return {
        name: int(re.search(rf"{name} = (\d+)", ir).group(1))
        for name in ("vgpr_count", "agpr_count", "vgpr_spill_count")
    }


def _without_retries(monkeypatch):
    monkeypatch.setattr(RocmBackend, "codegen_retries", lambda self, compiled, **_: ())


def test_a_kernel_that_overflows_into_agprs_is_placed_in_the_vgprs(monkeypatch):
    assert _registers(_compiled_ir(**FITS_WITHOUT_AGPRS)) == {"vgpr_count": 252, "agpr_count": 0, "vgpr_spill_count": 0}
    # The probe: the first lowering has to overflow, or the check above proves nothing.
    _without_retries(monkeypatch)
    assert _registers(_compiled_ir(**FITS_WITHOUT_AGPRS)) == {"vgpr_count": 260, "agpr_count": 8, "vgpr_spill_count": 0}


def test_a_retry_that_needs_a_spill_is_not_taken(monkeypatch):
    lowerings = []
    lower = jit_function.MlirCompiler._lower.__func__
    monkeypatch.setattr(
        jit_function.MlirCompiler,
        "_lower",
        classmethod(lambda cls, *a, **k: lowerings.append(1) or lower(cls, *a, **k)),
    )
    retried = _compiled_ir(**NEEDS_A_SPILL)
    assert len(lowerings) > 1
    _without_retries(monkeypatch)
    assert retried == _compiled_ir(**NEEDS_A_SPILL)
    assert _registers(retried)["agpr_count"] > 0


def test_a_kernel_that_lds_limits_to_one_wave_is_not_lowered_again(monkeypatch):
    # The kernel uses 32768 bytes of LDS per 256-thread work group: with 40000 bytes per CU only one group fits.
    monkeypatch.setitem(rocm.SMEM_CAPACITY_MAP, "gfx950", 40000)
    assert _registers(_compiled_ir(**FITS_WITHOUT_AGPRS)) == {"vgpr_count": 260, "agpr_count": 8, "vgpr_spill_count": 0}


def test_the_kernels_own_agpr_and_allocator_choices_are_kept(monkeypatch):
    proposed = {}
    codegen_retries = RocmBackend.codegen_retries

    def retries(self, compiled, *, source_asm, llvm_options):
        proposed["default"] = codegen_retries(self, compiled, source_asm=source_asm, llvm_options=llvm_options)
        proposed["inline asm"] = codegen_retries(
            self, compiled, source_asm=source_asm + "llvm.inline_asm", llvm_options=llvm_options
        )
        proposed["own allocation order"] = codegen_retries(
            self, compiled, source_asm=source_asm, llvm_options={"greedy-reverse-local-assignment": False}
        )
        return ()

    monkeypatch.setattr(RocmBackend, "codegen_retries", retries)
    _compiled_ir(**FITS_WITHOUT_AGPRS)
    assert len(proposed["default"]) == 4
    assert len(proposed["inline asm"]) == 0
    assert [sorted(options) for options, _prepare in proposed["own allocation order"]] == [
        ["greedy-regclass-priority-trumps-globalness"]
    ] * 2
