# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import pytest

from kernels.mega_moe.mega_moe_config import (
    apply_mega_moe_quant_config,
    nearest_token_bucket,
    select_mega_moe_config,
)

_STANDARD_PROFILES = {
    1: (32, 256, 4, 1, 64, 0, 1, 2, 32, 256, 0, 0, 0, "none"),
    4: (32, 256, 4, 1, 128, 0, 1, 2, 32, 256, 0, 0, 0, "none"),
    8: (32, 256, 4, 2, 128, 0, 1, 2, 32, 128, 0, 0, 0, "none"),
    16: (32, 128, 4, 4, 96, 0, 1, 1, 32, 128, 0, 0, 0, "fp8_blockwise_1x32"),
    32: (32, 128, 4, 3, 128, 0, 0, 2, 32, 128, 0, 0, 0, "fp8_blockwise_1x32"),
    64: (32, 128, 4, 3, 208, 0, 0, 2, 32, 256, 0, 0, 0, "fp8_blockwise_1x32"),
    128: (32, 128, 4, 3, 224, 0, 0, 2, 32, 256, 1, 64, 0, "fp8_blockwise_1x32"),
    256: (64, 512, 8, 1, 160, 1, 1, 2, 64, 256, 1, 128, 0, "fp8_blockwise_1x32"),
    512: (64, 512, 8, 2, 128, 1, 0, 2, 64, 256, 1, 240, 1, "fp8_blockwise_1x32"),
    1024: (64, 512, 8, 2, 128, 1, 0, 2, 32, 256, 1, 240, 1, "fp8_blockwise_1x32"),
    2048: (64, 512, 8, 1, 32, 1, 1, 2, 32, 256, 1, 240, 1, "fp8_blockwise_1x32"),
    4096: (128, 512, 8, 1, 32, 1, 0, 2, 64, 256, 1, 256, 0, "fp8_blockwise_1x32"),
    8192: (128, 512, 8, 1, 32, 1, 0, 2, 64, 256, 1, 240, 0, "fp8_blockwise_1x32"),
    16384: (128, 512, 8, 1, 32, 1, 1, 2, 64, 256, 1, 256, 0, "fp8_blockwise_1x32"),
    32768: (128, 512, 8, 1, 32, 1, 1, 2, 64, 256, 1, 240, 0, "fp8_blockwise_1x32"),
}


def _profile(config):
    stage1 = config.stage1
    stage2 = config.stage2
    return (
        stage1.sort_block_m,
        stage1.tile_n,
        stage1.num_waves,
        stage1.grid_mult,
        stage1.num_dispatch_cu,
        int(stage1.mfma_amajor),
        int(stage1.use_tile_resource),
        stage1.waves_per_eu_hint,
        stage2.block_m,
        stage2.block_n,
        int(stage2.persist),
        stage2.persist_cu,
        int(stage2.persist_strided),
        config.p2p_quant,
    )


@pytest.mark.parametrize("tokens,expected", _STANDARD_PROFILES.items())
def test_standard_profiles_match_tuned_artifacts(tokens, expected):
    config = select_mega_moe_config(tokens, max(16, tokens))
    stage1 = config.stage1
    stage2 = config.stage2

    assert _profile(config) == expected
    assert stage1.async_a_copy == (tokens >= 256 and tokens != 2048)
    assert stage1.b_nt == (0 if tokens == 1 or tokens >= 1024 else 3)
    assert stage1.work_shards == (4 if tokens >= 8192 else 8)
    assert stage1.external_grouping == (tokens >= 2048)
    assert stage1.external_counting == (tokens >= 8192)
    assert stage1.pipe_weights and stage1.swizzle_a
    assert not stage1.active_expert_producer and not stage1.cooperative_payload_copy
    assert stage2.use_nt == (tokens <= 128)
    assert stage2.b_hoist == (tokens != 64)
    assert stage2.ascale_prefetch
    assert stage2.spatial_partition == 402
    expected_deep_a = stage2.block_n == 256 and (tokens in (256, 512) or tokens >= 1024)
    assert stage2.deep_a_pipeline == expected_deep_a
    assert not stage2.deep_a_pipeline or stage2.b2stage


@pytest.mark.parametrize(
    "tokens,bucket",
    [(2, 1), (3, 4), (6, 8), (16300, 16384), (16400, 16384), (24576, 32768), (65536, 32768)],
)
def test_nearest_token_bucket_prefers_larger_on_ties(tokens, bucket):
    assert nearest_token_bucket(tokens) == bucket


def test_mtpr_selects_fixed_or_compact_configs():
    fixed = select_mega_moe_config(128, 128)
    compact = select_mega_moe_config(128, 8192)

    assert (fixed.stage1.tile_n, fixed.stage1.num_waves, fixed.stage1.num_dispatch_cu) == (128, 4, 224)
    assert (compact.stage1.tile_n, compact.stage1.num_waves, compact.stage1.num_dispatch_cu) == (512, 8, 128)
    for tokens in (8, 16, 32):
        assert select_mega_moe_config(tokens, 128).stage2.block_n == 128
        assert select_mega_moe_config(tokens, 8192).stage2.block_n == 256


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (1, "none"),
        (15, "none"),
        (16, "fp8_blockwise_1x32"),
        (17, "fp8_blockwise_1x32"),
    ],
)
def test_auto_fp8_threshold_is_inclusive(tokens, expected):
    mtpr = max(16, 1 << (tokens - 1).bit_length())
    assert select_mega_moe_config(tokens, mtpr).p2p_quant == expected


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (64, (32, 256, False, 0, True, False, False)),
        (128, (32, 256, True, 64, True, False, True)),
        (256, (64, 256, True, 128, False, False, True)),
        (512, (64, 256, True, 240, False, True, True)),
    ],
)
def test_forced_fp8_uses_quant_aware_small_batch_profiles(tokens, expected):
    config = select_mega_moe_config(tokens, tokens, "fp8_blockwise_1x32")
    stage2 = config.stage2

    assert (
        stage2.block_m,
        stage2.block_n,
        stage2.persist,
        stage2.persist_cu,
        stage2.use_nt,
        stage2.persist_strided,
        stage2.b_hoist,
    ) == expected
    assert config.p2p_quant == "fp8_blockwise_1x32"
    assert stage2.deep_a_pipeline == (tokens in (256, 512))
    if tokens == 256:
        assert config.stage1.sort_block_m == 64


def test_nearby_tokens_share_the_bucket_config():
    assert select_mega_moe_config(500, 512) is select_mega_moe_config(512, 512)


def test_a4_config_overrides_are_quant_specific():
    base_512 = select_mega_moe_config(500, 512)
    a4 = apply_mega_moe_quant_config(base_512, 500, "fp4").stage1
    assert (a4.b_nt, a4.num_dispatch_cu, a4.waves_per_eu_hint) == (0, 160, 1)

    base_1024 = select_mega_moe_config(1024, 1024)
    a4_1024 = apply_mega_moe_quant_config(base_1024, 1024, "fp4")
    assert (a4_1024.stage1.sort_block_m, a4_1024.stage1.num_dispatch_cu) == (128, 88)
    assert a4_1024.stage2.block_m == 64

    a8 = apply_mega_moe_quant_config(base_512, 500, "fp8").stage1
    assert (a8.b_nt, a8.num_dispatch_cu, a8.waves_per_eu_hint) == (3, 128, 2)

    a4_256 = apply_mega_moe_quant_config(select_mega_moe_config(256, 256), 256, "fp4")
    assert a4_256.stage1.waves_per_eu_hint == 1
    assert not select_mega_moe_config(256, 256, "none").stage2.deep_a_pipeline
    with pytest.raises(ValueError, match="unsupported activation dtype"):
        apply_mega_moe_quant_config(base_512, 500, "bf16")


@pytest.mark.parametrize(
    "tokens,expected_sort_block_m",
    [(8, 64), (32, 64), (64, 64), (128, 64), (256, 64), (1024, 128), (4096, 128), (8192, 128)],
)
def test_a4_fixed_mtpr_profiles_support_async_copy(tokens, expected_sort_block_m):
    config = apply_mega_moe_quant_config(select_mega_moe_config(tokens, 8192), tokens, "fp4")
    stage1 = config.stage1
    a_k_step_bytes = stage1.tile_k // 2
    total_threads = stage1.num_waves * 64

    assert stage1.sort_block_m == expected_sort_block_m
    assert stage1.async_a_copy
    assert (stage1.sort_block_m * (a_k_step_bytes // 16)) % total_threads == 0


@pytest.mark.parametrize("tokens,mtpr", [(0, 16), (17, 16), (1, 0), (1, 24)])
def test_invalid_shape_is_rejected(tokens, mtpr):
    with pytest.raises(ValueError):
        select_mega_moe_config(tokens, mtpr)


def test_invalid_p2p_quant_is_rejected():
    with pytest.raises(ValueError, match="unsupported p2p_quant"):
        select_mega_moe_config(64, 64, "fp16")
