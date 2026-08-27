# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import pytest

from kernels.mega_moe.mega_moe_config import (
    A8W4_DECODE_MTPRS,
    A8W4SMOOTH_DECODE_MTPRS,
    MAX_MTPR_CLASS,
    TOKEN_BUCKETS,
    expert_config_class,
    fixed_slot_max_mtpr,
    mtpr_config_class,
    nearest_token_bucket,
    select_mega_moe_config,
)

M13_A8W4 = dict(
    experts_per_rank=48,
    model_dim=3584,
    inter_dim=1280,
    quant_mode="a8w4",
    world_size=8,
    topk=8,
    num_cu=256,
)

M13_A8W4SMOOTH = dict(M13_A8W4, quant_mode="a8w4smooth")


def test_w8a8smooth_always_uses_compact_dispatch():
    assert fixed_slot_max_mtpr(
        "w8a8smooth",
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
        world_size=8,
        topk=8,
    ) == 0


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
    reference = select_mega_moe_config(tokens, 2048, quant_mode="w8a8smooth")
    for mtpr in (4096, 8192, 16384, 32768):
        if tokens <= mtpr:
            assert select_mega_moe_config(tokens, mtpr, quant_mode="w8a8smooth") is reference


@pytest.mark.parametrize(
    "tokens,sbm,dispatch_cu,work_shards,persist_cu",
    [
        (1, 32, 224, 1, 240),
        (32, 32, 64, 1, 240),
        (128, 32, 192, 4, 240),
        (256, 64, 160, 4, 240),
        (512, 64, 64, 4, 240),
        (1024, 64, 64, 4, 224),
        (2048, 64, 64, 8, 256),
        (4096, 128, 64, 4, 240),
        (8192, 128, 96, 4, 240),
        (16384, 128, 32, 4, 192),
        (32768, 128, 32, 4, 240),
    ],
)
def test_large_mtpr_profiles_follow_geometry_rules(tokens, sbm, dispatch_cu, work_shards, persist_cu):
    config = select_mega_moe_config(
        tokens, max(2048, tokens), quant_mode="w8a8smooth"
    )
    stage1, stage2 = config.stage1, config.stage2

    assert stage1.sort_block_m == sbm
    assert stage1.num_dispatch_cu == dispatch_cu
    assert stage1.work_shards == work_shards
    assert stage1.grid_mult == 1
    assert stage1.use_tile_resource
    assert stage1.payload_chunk_rows == 384
    assert stage1.payload_tile_ready
    assert stage2.block_m == (64 if sbm == 128 else 32)
    assert stage2.persist_cu == persist_cu
    assert stage2.skew_cu == (96 if tokens >= 512 else 0)
    assert config.p2p_quant == "fp8_blockwise_1x32"


def test_fixed_and_bounded_compact_profiles_remain_specialized():
    fixed = select_mega_moe_config(128, 128, **M13_A8W4)
    bounded = select_mega_moe_config(512, 512, **M13_A8W4)

    assert (fixed.stage1.tile_n, fixed.stage1.num_waves, fixed.stage1.num_dispatch_cu) == (128, 4, 224)
    assert not fixed.stage1.payload_tile_ready and fixed.p2p_quant == "none"
    assert (bounded.stage1.sort_block_m, bounded.stage1.grid_mult) == (64, 2)
    assert bounded.stage1.num_dispatch_cu == 64
    assert bounded.stage2.persist_cu == 240
    assert not bounded.stage1.payload_tile_ready and bounded.p2p_quant == "none"


@pytest.mark.parametrize("tokens", A8W4_DECODE_MTPRS)
def test_native_a8w4_decode_contract(tokens):
    config = select_mega_moe_config(tokens, tokens, **M13_A8W4)
    stage1 = config.stage1

    assert not stage1.external_grouping
    assert not stage1.external_counting
    assert stage1.payload_chunk_rows == 0
    assert not stage1.payload_tile_ready


@pytest.mark.parametrize("tokens", A8W4SMOOTH_DECODE_MTPRS)
def test_a8w4smooth_decode_contract(tokens):
    config = select_mega_moe_config(tokens, tokens, **M13_A8W4SMOOTH)
    stage1 = config.stage1

    assert fixed_slot_max_mtpr(
        "a8w4smooth", 48, 3584, 1280, world_size=8, topk=8
    ) == 512
    assert not stage1.external_grouping
    assert not stage1.external_counting
    assert stage1.payload_chunk_rows == 0
    assert not stage1.payload_tile_ready


def test_small_fixed_slot_profiles_use_low_overhead_geometry():
    generic = [
        select_mega_moe_config(tokens, tokens, **M13_A8W4)
        for tokens in (1, 4, 8)
    ]
    tuned = [
        select_mega_moe_config(tokens, tokens, **M13_A8W4SMOOTH)
        for tokens in (1, 4, 8)
    ]

    assert [config.stage1.num_dispatch_cu for config in generic] == [160, 128, 32]
    assert [config.stage1.grid_mult for config in generic] == [1, 1, 2]
    assert [config.stage1.tile_n for config in generic] == [256, 256, 256]
    assert all(config.stage2.use_nt for config in generic)
    assert [config.stage1.num_dispatch_cu for config in tuned] == [64, 64, 192]
    assert [config.stage1.grid_mult for config in tuned] == [1, 1, 1]
    assert [config.stage1.tile_n for config in tuned] == [128, 128, 256]
    assert [config.stage1.b_nt for config in tuned] == [0, 0, 0]
    assert not any(config.stage2.use_nt for config in tuned)


def test_non_smooth_a8w4_bounded_profiles_match_production_tuning():
    configs = [select_mega_moe_config(512, 512, **M13_A8W4)]

    assert [config.stage1.num_dispatch_cu for config in configs] == [64]
    assert [config.stage1.grid_mult for config in configs] == [2]


def test_a8w4smooth_is_decode_only():
    generic512 = select_mega_moe_config(512, 512, **M13_A8W4)
    tuned512 = select_mega_moe_config(512, 512, **M13_A8W4SMOOTH)
    assert generic512.stage2.persist_cu == 240
    assert tuned512.stage2.persist_cu == 240

    for mtpr in (1024, 2048, 4096, 32768):
        with pytest.raises(ValueError, match="tokens=MTPR"):
            select_mega_moe_config(min(512, mtpr), mtpr, **M13_A8W4SMOOTH)

    with pytest.raises(ValueError, match="tokens=MTPR"):
        select_mega_moe_config(2, 2, **M13_A8W4SMOOTH)


def test_large_mtpr_protocol_is_rank_invariant_across_token_buckets():
    configs = [
        select_mega_moe_config(tokens, 32768, quant_mode="w8a8smooth")
        for tokens in TOKEN_BUCKETS
    ]

    assert {config.p2p_quant for config in configs} == {"fp8_blockwise_1x32"}
    assert {config.stage1.payload_chunk_rows for config in configs} == {384}
    assert {config.stage1.payload_tile_ready for config in configs} == {True}


@pytest.mark.parametrize("experts_per_rank", [48, 52, 56, 64])
def test_redundant_experts_share_one_wave_geometry(experts_per_rank):
    base = select_mega_moe_config(
        8192, 32768, experts_per_rank=48, quant_mode="w8a8smooth"
    )
    redundant = select_mega_moe_config(
        8192, 32768, experts_per_rank=experts_per_rank,
        quant_mode="w8a8smooth",
    )

    assert expert_config_class(experts_per_rank) == 64
    assert redundant is base


def test_multiple_expert_waves_scale_payload_producers():
    base = select_mega_moe_config(
        4096, 32768, experts_per_rank=48, quant_mode="w8a8smooth"
    )
    wide = select_mega_moe_config(
        4096, 32768, experts_per_rank=80, quant_mode="w8a8smooth"
    )

    assert wide.stage1.num_dispatch_cu == 2 * base.stage1.num_dispatch_cu
    assert wide.stage2 == base.stage2


def test_model_geometry_selects_tile_widths():
    config = select_mega_moe_config(
        8192, 32768, model_dim=3584, inter_dim=1536,
        quant_mode="w8a8smooth",
    )

    assert config.stage1.tile_n == 256
    assert config.stage2.block_n == 128


@pytest.mark.parametrize("quant_mode", ["a8w4smooth", "w8a8smooth"])
def test_m13_token128_int8_uses_validated_tiles_without_p2p_quant(quant_mode):
    config = select_mega_moe_config(
        128,
        128,
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
        quant_mode=quant_mode,
    )

    assert (
        config.stage2.block_m,
        config.stage2.block_n,
        config.stage2.block_k,
    ) == (32, 128, 256)
    assert config.p2p_quant == "none"


@pytest.mark.parametrize(
    "tokens,stage1_expected,persist_cu",
    [
        (8, (32, 256, 8, 1, 192, 0, 2, False), 0),
        (16, (32, 128, 4, 2, 96, 0, 1, True), 0),
        (32, (32, 128, 4, 2, 192, 0, 2, False), 0),
        (64, (32, 128, 4, 2, 128, 0, 1, True), 0),
        (128, (32, 128, 4, 2, 192, 0, 2, False), 96),
        (256, (64, 256, 8, 3, 224, 0, 2, True), 128),
        (512, (64, 256, 8, 1, 208, 0, 1, True), 240),
    ],
)
def test_m13_a8w4smooth_fixed_slot_uses_cudagraph_tuning(tokens, stage1_expected, persist_cu):
    config = select_mega_moe_config(
        tokens,
        tokens,
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
        quant_mode="a8w4smooth",
    )
    stage1 = config.stage1

    assert (
        stage1.sort_block_m,
        stage1.tile_n,
        stage1.num_waves,
        stage1.grid_mult,
        stage1.num_dispatch_cu,
        stage1.b_nt,
        stage1.waves_per_eu_hint,
        stage1.swizzle_a,
    ) == stage1_expected
    assert config.stage2.persist_cu == persist_cu


def test_mxfp4_transport_config_is_w8a8smooth_prefill_only():
    common = dict(
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
        quant_mode="w8a8smooth",
        world_size=8,
        topk=8,
        num_cu=256,
    )
    established = select_mega_moe_config(4096, 32768, **common)
    transport = select_mega_moe_config(
        4096, 32768, dispatch_quant="mxfp4", **common
    )

    assert transport is established


@pytest.mark.parametrize("mtpr", [1024, 2048, 32768])
def test_native_a8w4_rejects_non_decode_capacity(mtpr):
    with pytest.raises(ValueError, match="tokens=MTPR"):
        select_mega_moe_config(512, mtpr, **M13_A8W4)


@pytest.mark.parametrize("quant_mode", ["a8w4", "a8w4smooth"])
def test_mxfp4_transport_rejects_decode_modes(quant_mode):
    with pytest.raises(ValueError, match="only supported by w8a8smooth"):
        select_mega_moe_config(
            32,
            32,
            dispatch_quant="mxfp4",
            **dict(M13_A8W4, quant_mode=quant_mode),
        )


def test_narrow_int_stage2_residency_tracks_route_pressure():
    common = dict(
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
        quant_mode="a8w4smooth",
    )
    token128 = select_mega_moe_config(128, 128, **common)
    token256 = select_mega_moe_config(256, 256, **common)
    token512 = select_mega_moe_config(512, 512, **common)

    assert [
        token128.stage2.persist_cu,
        token256.stage2.persist_cu,
        token512.stage2.persist_cu,
    ] == [96, 128, 240]
    assert token128.stage2.use_nt and token256.stage2.use_nt
    assert not token512.stage2.use_nt


@pytest.mark.parametrize(
    "override",
    [
        {"experts_per_rank": 64},
        {"model_dim": 4096},
        {"inter_dim": 1536},
        {"world_size": 4},
        {"topk": 4},
    ],
)
def test_a8w4smooth_rejects_non_m13_shapes(override):
    kwargs = dict(M13_A8W4SMOOTH)
    kwargs.update(override)
    with pytest.raises(ValueError, match="specialized for M13 EP8"):
        select_mega_moe_config(8, 8, **kwargs)


@pytest.mark.parametrize("tokens", A8W4SMOOTH_DECODE_MTPRS)
def test_m13_a8w4smooth_uses_low_pressure_int_pipeline(tokens):
    config = select_mega_moe_config(tokens, tokens, **M13_A8W4SMOOTH)

    assert config.stage1.pipe_weights is False
    assert config.stage1.async_a_copy is False


def test_m13_a8w4smooth_token512_uses_cudagraph_tuning():
    config = select_mega_moe_config(
        512,
        512,
        experts_per_rank=48,
        model_dim=3584,
        inter_dim=1280,
        quant_mode="a8w4smooth",
    )

    assert (
        config.stage1.sort_block_m,
        config.stage1.tile_n,
        config.stage1.num_waves,
        config.stage1.grid_mult,
        config.stage1.num_dispatch_cu,
        config.stage1.b_nt,
        config.stage1.waves_per_eu_hint,
    ) == (64, 256, 8, 1, 208, 0, 1)


def test_nearby_tokens_share_the_bucket_config():
    assert select_mega_moe_config(
        500, 8192, quant_mode="w8a8smooth"
    ) is select_mega_moe_config(512, 32768, quant_mode="w8a8smooth")


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
