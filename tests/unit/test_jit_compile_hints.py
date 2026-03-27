# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from pathlib import Path

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


@flyc.kernel
def _sched_probe_kernel():
    pass


@flyc.jit
def _launch_default(stream: fx.Stream = fx.Stream(None)):
    _sched_probe_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@flyc.jit(compile_hints={"expert_scheduling_mode": True})
def _launch_expert(stream: fx.Stream = fx.Stream(None)):
    _sched_probe_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def _compile_and_read_final_isa(jit_fn, dump_dir: Path, monkeypatch) -> str:
    dump_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump_dir))

    # Use public compile API; compiler will emit final_isa.s into dump_dir.
    flyc.compile(jit_fn)

    isa_files = sorted(dump_dir.rglob("*final_isa.s"))
    assert isa_files, f"Expected final_isa.s under {dump_dir}, got none"
    return isa_files[-1].read_text(encoding="utf-8")


def test_expert_scheduling_mode_emits_sched_mode2(tmp_path, monkeypatch):
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"Expert scheduling mode ISA check requires gfx1250, got {arch}")

    isa_default = _compile_and_read_final_isa(_launch_default, tmp_path / "default", monkeypatch)
    isa_expert = _compile_and_read_final_isa(_launch_expert, tmp_path / "expert", monkeypatch)

    sched_mode2 = "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2"
    assert sched_mode2 not in isa_default
    assert sched_mode2 in isa_expert
