# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import pytest

from kernels.mega_moe.mega_moe_config import (
    MAX_MTPR_CLASS,
    TOKEN_BUCKETS,
    apply_mega_moe_quant_config,
    expert_config_class,
    mtpr_config_class,
    nearest_token_bucket,
    select_mega_moe_config,
)


@pytest.mark.parametrize(
    "tokens,bucket",
    [(2, 1), (3, 4), (6, 8), (16300, 16384), (16400, 16384), (24576, 32768), (65536, 32768)],
)
def test_nearest_token_bucket_prefers_larger_on_ties(tokens, bucket):
    assert nearest_token_bucket(tokens) == bucket


@pytest.mark.parametrize("mtpr", [2048, 4096, 8192, 16384, 32768, 65536])
def test_large_mtpr_uses_one_config_class(mtpr):
    assert mtpr_config_class(mtpr) == MAX_MTPR_CLASS


@pytest.mark.parametrize("tokens", [1, 8, 32, 128, 256, 512, 1024, 2048])
def test_large_mtpr_configs_are_capacity_invariant(tokens):
    reference = select_mega_moe_config(tokens, 2048)
    for mtpr in (4096, 8192, 16384, 32768):
        if tokens <= mtpr:
            assert select_mega_moe_config(tokens, mtpr) is reference


@pytest.mark.parametrize(
    "tokens,sbm,dispatch_cu,work_shards,persist_cu,stage2_bm,skew_cu,payload_rows,payload_ready",
    [
        (1, 32, 224, 1, 240, 32, 0, 384, True),
        (32, 32, 64, 1, 240, 32, 0, 384, True),
        (128, 32, 192, 4, 240, 32, 0, 384, True),
        (256, 64, 160, 4, 240, 32, 0, 384, True),
        (512, 64, 64, 4, 240, 32, 96, 384, True),
        (1024, 64, 64, 4, 224, 32, 96, 384, True),
        (2048, 64, 64, 8, 240, 64, 0, 0, False),
        (4096, 128, 64, 4, 240, 64, 96, 384, True),
        (8192, 128, 96, 4, 240, 64, 96, 384, True),
        (16384, 128, 32, 4, 192, 64, 96, 1536, True),
        (32768, 128, 32, 4, 256, 64, 0, 768, True),
    ],
)
def test_large_mtpr_profiles_follow_geometry_rules(
    tokens,
    sbm,
    dispatch_cu,
    work_shards,
    persist_cu,
    stage2_bm,
    skew_cu,
    payload_rows,
    payload_ready,
):
    config = select_mega_moe_config(tokens, max(2048, tokens))
    stage1, stage2 = config.stage1, config.stage2

    assert stage1.sort_block_m == sbm
    assert stage1.num_dispatch_cu == dispatch_cu
    assert stage1.work_shards == work_shards
    assert stage1.grid_mult == 1
    assert stage1.use_tile_resource
    assert stage1.payload_chunk_rows == payload_rows
    assert stage1.payload_tile_ready == payload_ready
    assert stage2.block_m == stage2_bm
    assert stage2.persist_cu == persist_cu
    assert stage2.skew_cu == skew_cu
    assert config.p2p_quant == "fp8_blockwise_1x32"


def test_fixed_and_bounded_compact_profiles_remain_specialized():
    fixed = select_mega_moe_config(128, 128)
    bounded = select_mega_moe_config(512, 512)

    assert (fixed.stage1.tile_n, fixed.stage1.num_waves, fixed.stage1.num_dispatch_cu) == (128, 4, 224)
    assert not fixed.stage1.payload_tile_ready and fixed.p2p_quant == "none"
    assert (bounded.stage1.sort_block_m, bounded.stage1.grid_mult) == (64, 2)
    assert bounded.stage1.num_dispatch_cu == 128
    assert not bounded.stage1.payload_tile_ready and bounded.p2p_quant == "none"


def test_large_mtpr_protocol_is_rank_invariant_across_token_buckets():
    configs = {tokens: select_mega_moe_config(tokens, 32768) for tokens in TOKEN_BUCKETS}

    assert {config.p2p_quant for config in configs.values()} == {"fp8_blockwise_1x32"}
    assert (configs[2048].stage1.payload_chunk_rows, configs[2048].stage1.payload_tile_ready) == (0, False)
    assert configs[16384].stage1.payload_chunk_rows == 1536
    assert configs[32768].stage1.payload_chunk_rows == 768
    for tokens, config in configs.items():
        if tokens not in (2048, 16384, 32768):
            assert (config.stage1.payload_chunk_rows, config.stage1.payload_tile_ready) == (384, True)


def test_non_v4_pro_keeps_generic_large_rules():
    config = select_mega_moe_config(2048, 2048, model_dim=3584, inter_dim=1536)

    assert (config.stage1.payload_chunk_rows, config.stage1.payload_tile_ready) == (0, False)
    assert (config.stage2.block_m, config.stage2.block_n, config.stage2.persist_cu, config.stage2.skew_cu) == (
        64,
        128,
        240,
        0,
    )


@pytest.mark.parametrize(
    "a_dtype,p2p_quant",
    [
        ("fp4", "none"),
        ("fp4", "fp8_blockwise_1x32"),
        ("fp8", "none"),
        ("fp8", "fp8_blockwise_1x32"),
    ],
)
def test_v4_pro_bs2048_rule_is_transport_and_dtype_invariant(a_dtype, p2p_quant):
    config = apply_mega_moe_quant_config(
        select_mega_moe_config(2048, 2048, p2p_quant, a_dtype=a_dtype),
        2048,
        a_dtype,
    )

    assert (config.stage1.payload_chunk_rows, config.stage1.payload_tile_ready) == (0, False)
    assert (
        config.stage2.block_m,
        config.stage2.block_n,
        config.stage2.persist_cu,
        config.stage2.persist_strided,
        config.stage2.skew_cu,
    ) == (64, 256, 240, True, 0)


@pytest.mark.parametrize("a_dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("p2p_quant", ["none", "fp8_blockwise_1x32"])
def test_bs2_uses_two_grid_epochs_for_both_activation_dtypes(a_dtype, p2p_quant):
    config = apply_mega_moe_quant_config(
        select_mega_moe_config(2, 16, p2p_quant, a_dtype=a_dtype),
        2,
        a_dtype,
    )

    assert config.stage1.grid_mult == 2


@pytest.mark.parametrize("tokens,grid_mult", [(4, 4), (16, 3)])
@pytest.mark.parametrize("p2p_quant", ["none", "fp8_blockwise_1x32"])
def test_a4_small_decode_grid_overrides_are_transport_invariant(tokens, grid_mult, p2p_quant):
    config = apply_mega_moe_quant_config(
        select_mega_moe_config(tokens, 16, p2p_quant, a_dtype="fp4"),
        tokens,
        "fp4",
    )

    assert config.stage1.grid_mult == grid_mult


@pytest.mark.parametrize("p2p_quant", ["none", "fp8_blockwise_1x32"])
@pytest.mark.parametrize(
    "tokens,expected",
    [
        (4096, (1, 72, True, 0, False)),
        (8192, (2, 32, False, 0, False)),
    ],
)
def test_a4_large_stage1_overrides_are_transport_invariant(tokens, expected, p2p_quant):
    config = apply_mega_moe_quant_config(
        select_mega_moe_config(tokens, tokens, p2p_quant, a_dtype="fp4"),
        tokens,
        "fp4",
    )
    stage1 = config.stage1

    assert (
        stage1.grid_mult,
        stage1.num_dispatch_cu,
        stage1.swizzle_a,
        stage1.payload_chunk_rows,
        stage1.payload_tile_ready,
    ) == expected


@pytest.mark.parametrize("experts_per_rank", [48, 52, 56, 64])
def test_redundant_experts_share_one_wave_geometry(experts_per_rank):
    base = select_mega_moe_config(8192, 32768, experts_per_rank=48)
    redundant = select_mega_moe_config(8192, 32768, experts_per_rank=experts_per_rank)

    assert expert_config_class(experts_per_rank) == 64
    assert redundant is base


def test_multiple_expert_waves_scale_payload_producers():
    base = select_mega_moe_config(4096, 32768, experts_per_rank=48)
    wide = select_mega_moe_config(4096, 32768, experts_per_rank=80)

    assert wide.stage1.num_dispatch_cu == 2 * base.stage1.num_dispatch_cu
    assert wide.stage2 == base.stage2


def test_model_geometry_selects_tile_widths():
    config = select_mega_moe_config(8192, 32768, model_dim=3584, inter_dim=1536)

    assert config.stage1.tile_n == 256
    assert config.stage2.block_n == 128


def test_a4_auto_p2p_restores_fp8_at_mtpr_1024_without_losing_rank_invariance():
    configs = [select_mega_moe_config(tokens, 1024, a_dtype="fp4") for tokens in TOKEN_BUCKETS if tokens <= 1024]

    assert {config.p2p_quant for config in configs} == {"fp8_blockwise_1x32"}
    assert select_mega_moe_config(1024, 1024, "none", a_dtype="fp4").p2p_quant == "none"


def test_explicit_p2p_quant_override_is_preserved():
    assert select_mega_moe_config(64, 64, "fp8_blockwise_1x32").p2p_quant == "fp8_blockwise_1x32"
    assert select_mega_moe_config(2048, 2048, "none").p2p_quant == "none"


@pytest.mark.parametrize(
    "a_dtype,tokens,persist_cu,persist_strided",
    [
        ("fp8", 256, 128, False),
        ("fp8", 512, 240, True),
        ("fp4", 512, 240, True),
    ],
)
def test_fp8_transport_restores_tuned_medium_stage2_profiles(a_dtype, tokens, persist_cu, persist_strided):
    config = apply_mega_moe_quant_config(
        select_mega_moe_config(tokens, tokens, "fp8_blockwise_1x32", a_dtype=a_dtype),
        tokens,
        a_dtype,
    )
    stage2 = config.stage2

    assert (stage2.block_m, stage2.block_n) == (64, 256)
    assert (stage2.persist, stage2.persist_cu, stage2.use_nt) == (True, persist_cu, False)
    assert (stage2.persist_strided, stage2.deep_a_pipeline) == (persist_strided, True)

    none = apply_mega_moe_quant_config(
        select_mega_moe_config(tokens, tokens, "none", a_dtype=a_dtype),
        tokens,
        a_dtype,
    ).stage2
    assert (none.block_m, none.block_n) == (32, 128)


def test_nearby_tokens_share_the_bucket_config():
    assert select_mega_moe_config(500, 8192) is select_mega_moe_config(512, 32768)


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

    a4_fp8_1024 = apply_mega_moe_quant_config(
        select_mega_moe_config(1024, 1024, "fp8_blockwise_1x32", a_dtype="fp4"),
        1024,
        "fp4",
    )
    assert a4_fp8_1024.stage1.grid_mult == 1
    assert a4_fp8_1024.stage2.persist_cu == 224

    a4_fixed_256 = apply_mega_moe_quant_config(
        select_mega_moe_config(256, 8192, a_dtype="fp4"),
        256,
        "fp4",
    )
    assert (
        a4_fixed_256.stage2.block_m,
        a4_fixed_256.stage2.block_n,
        a4_fixed_256.stage2.persist_cu,
        a4_fixed_256.stage2.deep_a_pipeline,
    ) == (64, 256, 128, True)

    with pytest.raises(ValueError, match="unsupported activation dtype"):
        apply_mega_moe_quant_config(base_512, 500, "bf16")
    with pytest.raises(ValueError, match="unsupported activation dtype"):
        select_mega_moe_config(512, 512, a_dtype="bf16")


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
    assert not stage1.payload_chunk_rows or stage1.payload_chunk_rows % stage1.sort_block_m == 0


@pytest.mark.parametrize(
    "tokens,mtpr,kwargs",
    [
        (0, 16, {}),
        (17, 16, {}),
        (1, 0, {}),
        (1, 24, {}),
        (1, 16, {"experts_per_rank": 0}),
        (1, 16, {"model_dim": 0}),
    ],
)
def test_invalid_shape_is_rejected(tokens, mtpr, kwargs):
    with pytest.raises(ValueError):
        select_mega_moe_config(tokens, mtpr, **kwargs)


def test_invalid_p2p_quant_is_rejected():
    with pytest.raises(ValueError, match="unsupported p2p_quant"):
        select_mega_moe_config(64, 64, "fp16")
