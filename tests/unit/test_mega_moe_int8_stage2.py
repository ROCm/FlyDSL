# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect

import pytest

import kernels.mega_moe.mega_moe_stage2 as stage2_module
from kernels.mega_moe.mega_moe_config import select_mega_moe_config
from kernels.mega_moe.mega_moe_stage2 import compile_mega_moe_stage2


def _compile_kwargs(**overrides):
    kwargs = dict(
        model_dim=256,
        inter_dim=256,
        experts=8,
        topk=2,
        rank=0,
        npes=1,
        max_tok=128,
        recv_cap=128,
        comb_inp_nbytes=128 * 2 * 256 * 2,
        BM=32,
        BN=128,
        BK=256,
        HIDDEN_MAX=256,
        INTER_MAX=256,
        a_dtype="fp8",
        SBM=32,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("mode", ["a8w4smooth", "w8a8smooth"])
def test_int8_stage2_launcher_is_quant_specialized(monkeypatch, mode):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    launch = compile_mega_moe_stage2(**_compile_kwargs(quant_mode=mode))
    assert callable(launch)


def test_int8_stage2_rejects_unvalidated_tiles(monkeypatch):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    with pytest.raises(AssertionError, match="32x128x256"):
        compile_mega_moe_stage2(
            **_compile_kwargs(quant_mode="a8w4smooth", BN=256)
        )


def test_stage2_launcher_threads_lqq_qparams():
    parameters = inspect.signature(stage2_module.run_mega_moe_stage2).parameters
    assert parameters["quant_mode"].default == "a8w4"
    assert parameters["qscale_w"].default is None
    assert parameters["qzero_w"].default is None


def test_m13_token_128_uses_32x128x256_gemm2():
    config = select_mega_moe_config(
        128,
        128,
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
        quant_mode="a8w4smooth",
    )
    stage2 = config.stage2
    assert (stage2.block_m, stage2.block_n, stage2.block_k) == (32, 128, 256)


def test_m13_a8w4_config_remains_unchanged():
    a8w4 = select_mega_moe_config(
        128,
        128,
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
    )
    int8 = select_mega_moe_config(
        128,
        128,
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
        quant_mode="a8w4smooth",
    )
    assert (a8w4.stage2.block_m, a8w4.stage2.block_n) == (32, 128)
    assert (int8.stage2.block_m, int8.stage2.block_n, int8.stage2.block_k) == (
        a8w4.stage2.block_m,
        a8w4.stage2.block_n,
        a8w4.stage2.block_k,
    )
    # The native decode and SmoothQuant selectors intentionally retain their
    # independently validated persistent-CU settings.
    assert a8w4.stage2.persist_cu == 240
    assert int8.stage2.persist_cu == 96
