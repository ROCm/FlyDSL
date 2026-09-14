# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect

import pytest
import torch

import kernels.mega_moe.mega_moe_stage2 as stage2_module
from kernels.comm.flydsl_dispatch_combine_intranode_kernel import (
    make_combine_kernel,
)
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


def _kernel_name(launcher):
    closure = dict(
        zip(
            launcher._original_func.__code__.co_freevars,
            launcher._original_func.__closure__,
        )
    )
    return closure["kernel_epilog_v2"].cell_contents._name


@pytest.mark.parametrize("mode", ["a8w4smooth", "w8a8smooth"])
def test_int8_stage2_launcher_is_quant_specialized(monkeypatch, mode):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    launch = compile_mega_moe_stage2(**_compile_kwargs(quant_mode=mode))
    assert callable(launch)


def test_int8_stage2_rejects_unvalidated_tiles(monkeypatch):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    with pytest.raises(AssertionError, match="32x128x256"):
        compile_mega_moe_stage2(**_compile_kwargs(quant_mode="a8w4smooth", BN=256))


def test_blockwise_stage2_rejects_undersized_p2p_buffer(monkeypatch):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    required = 128 * 2 * 288  # align32(256 fp8 bytes + 8 E8M0 bytes)
    with pytest.raises(ValueError, match="smaller than the required"):
        compile_mega_moe_stage2(
            **_compile_kwargs(
                quant_mode="a8w4",
                p2p_quant_type="fp8_blockwise_1x32",
                comb_inp_nbytes=required - 1,
            )
        )


def test_blockwise_stage2_accepts_exact_minimum_p2p_buffer(monkeypatch):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    required = 128 * 2 * 288  # align32(256 fp8 bytes + 8 E8M0 bytes)
    launch = compile_mega_moe_stage2(
        **_compile_kwargs(
            quant_mode="a8w4",
            p2p_quant_type="fp8_blockwise_1x32",
            comb_inp_nbytes=required,
        )
    )
    assert callable(launch)


def test_blockwise_stage2_rejects_required_buffer_overflow(monkeypatch):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    with pytest.raises(ValueError, match="exceeds the 32-bit"):
        compile_mega_moe_stage2(
            **_compile_kwargs(
                model_dim=8192,
                HIDDEN_MAX=8192,
                topk=8,
                max_tok=32768,
                recv_cap=8 * 32768,
                comb_inp_nbytes=None,
                quant_mode="a8w4",
                p2p_quant_type="fp8_blockwise_1x32",
            )
        )


def test_blockwise_combine_rejects_zero_copy():
    with pytest.raises(ValueError, match="incompatible with zero_copy"):
        make_combine_kernel(
            rank=0,
            npes=8,
            experts_per_token=8,
            hidden_dim=256,
            hidden_elem_size=2,
            max_tok_per_rank=128,
            block_num=128,
            warp_num_per_block=8,
            data_type=torch.bfloat16,
            zero_copy=True,
            skip_stage1=True,
            blockwise_fp8_transport=True,
        )


def test_stage2_launcher_threads_lqq_qparams():
    parameters = inspect.signature(stage2_module.run_mega_moe_stage2).parameters
    assert parameters["quant_mode"].default == "a8w4"
    assert parameters["qscale_w"].default is None
    assert parameters["qzero_w"].default is None
    assert parameters["persist_n_major"].default is False


def test_n_major_has_distinct_cache_and_kernel_identity(monkeypatch):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    kwargs = _compile_kwargs(persist=True, cu_num=8)
    default = compile_mega_moe_stage2(**kwargs)
    n_major = compile_mega_moe_stage2(**kwargs, persist_n_major=True)

    assert "_nm1" not in _kernel_name(default)
    assert "_nm1" in _kernel_name(n_major)

    calls = []

    def fake_compile(**compile_kwargs):
        calls.append(compile_kwargs)
        return object()

    monkeypatch.setattr(stage2_module, "compile_mega_moe_stage2", fake_compile)
    stage2_module._G2_LAUNCH_CACHE.clear()
    try:
        cached_default = stage2_module._get_g2_launch(**kwargs, persist_n_major=False)
        cached_default_again = stage2_module._get_g2_launch(**kwargs, persist_n_major=False)
        cached_n_major = stage2_module._get_g2_launch(**kwargs, persist_n_major=True)
    finally:
        stage2_module._G2_LAUNCH_CACHE.clear()

    assert cached_default is cached_default_again
    assert cached_default is not cached_n_major
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["a8w4smooth", "w8a8smooth"])
def test_int8_stage2_rejects_n_major_override(monkeypatch, mode):
    monkeypatch.setattr(stage2_module, "get_rocm_arch", lambda: "gfx950")
    with pytest.raises(ValueError, match="only supported by native A8W4 Stage2"):
        compile_mega_moe_stage2(
            **_compile_kwargs(
                quant_mode=mode,
                persist=True,
                cu_num=8,
                persist_n_major=True,
            )
        )


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
    assert int8.stage2.persist_cu == 36
