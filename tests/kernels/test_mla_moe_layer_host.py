# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CPU checks for indexed MLA + MoE configuration and weight packing."""

from dataclasses import replace

import pytest
import torch

from kernels.common.mx_formats import dequantize_mxfp4, quant_dequant_mxfp8, quantize_mxfp4
from kernels.mla_moe_layer.config import (
    GLM5_CONFIG,
    KIMI_K3_CONFIG,
    ExpertActivation,
    ExpertWeight,
    MoeMode,
    as_moe_mode,
    moe_format,
    validate_shard,
)
from kernels.mla_moe_layer.kernel_layout import layout, stage_tasks
from kernels.mla_moe_layer.packing import (
    pack_a16w4_scale,
    pack_a16w4_weight,
    pack_bf16,
    pack_fp8,
    pack_mxfp4,
)
from kernels.mla_moe_layer.reference import golden_layer, kimi_attn_res, make_weights, rope_table, situ


def test_pack_fp8_uses_mfma_lane_order():
    raw = torch.arange(16 * 64, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(16, 64)
    packed = pack_fp8(raw.view(torch.float8_e4m3fn)).view(torch.uint8)
    expected = torch.tensor(
        [*range(0, 8), *range(32, 40), *range(64, 72), *range(96, 104)],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def test_pack_bf16_uses_mfma_lane_order():
    raw = torch.arange(16 * 64, dtype=torch.int16).reshape(16, 64)
    packed = pack_bf16(raw.view(torch.bfloat16)).view(torch.int16)
    expected = torch.tensor([*range(8), *range(64, 72)], dtype=torch.int16)
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def test_pack_mxfp4_matches_bf16_mfma_k32_steps():
    raw = torch.arange(16 * 64, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(16, 64)
    packed = pack_mxfp4(raw)
    expected = torch.tensor(
        [*range(0, 4), *range(16, 20), *range(32, 36), *range(48, 52)],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def test_pack_a16w4_uses_standard_16x64_weight_tile_order():
    raw = torch.arange(16 * 32, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(16, 32)
    packed = pack_a16w4_weight(raw)
    expected = torch.tensor(
        [*range(0, 16), *range(32, 48), *range(64, 80), *range(96, 112)],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def test_pack_a16w4_scale_interleaves_32_row_pairs():
    raw = torch.arange(256 * 8, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(256, 8)
    packed = pack_a16w4_scale(raw)
    expected = torch.tensor([0, 128, 4, 132, 8, 136, 12, 140], dtype=torch.uint8)
    torch.testing.assert_close(packed[: expected.numel()], expected, atol=0, rtol=0)


def test_mxfp4_quantization_round_trips_known_values():
    values = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.0] * 4], dtype=torch.float32)
    packed, scale = quantize_mxfp4(values)
    torch.testing.assert_close(dequantize_mxfp4(packed, scale), values, atol=0, rtol=0)


def test_mxfp8_quantization_uses_independent_32_value_groups():
    values = torch.cat([torch.full((32,), 0.5), torch.full((32,), 64.0)]).reshape(1, 64)
    got = quant_dequant_mxfp8(values)
    expected = torch.cat([torch.full((32,), 0.4375), torch.full((32,), 56.0)]).reshape(1, 64)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.parametrize("value", tuple(mode.value for mode in MoeMode))
def test_moe_mode_accepts_public_values(value):
    assert as_moe_mode(value).value == value


def test_moe_modes_map_to_independent_activation_and_weight_formats():
    assert moe_format(MoeMode.W8A8).activation is ExpertActivation.FP8_BLOCK128
    assert moe_format(MoeMode.W8A16).activation is ExpertActivation.BF16
    assert moe_format(MoeMode.A16W4).weight is ExpertWeight.MXFP4_BLOCK32
    assert moe_format(MoeMode.A8W4).activation is ExpertActivation.MXFP8_BLOCK32


@pytest.mark.parametrize(
    "args",
    [
        (3, 8, 0, 1, 2048),
        (16, 8, 0, 1, 2048),
        (1, 10, 0, 1, 2048),
        (1, 8, 1, 1, 2048),
        (1, 8, 0, 3, 2048),
        (1, 8, 0, 1, 2000),
    ],
)
def test_validate_shard_rejects_unsupported_contract(args):
    with pytest.raises(ValueError):
        validate_shard(*args)


def test_validate_shard_accepts_eight_samples():
    validate_shard(8, 8, 0, 1, 2048)


def test_validate_kimi_k3_attention_shard_accepts_tp8_geometry():
    validate_shard(8, 12, 7, 8, 2048, KIMI_K3_CONFIG)


def test_kimi_k3_mla_golden_uses_pre_normalized_input_and_defers_residual():
    config = replace(
        KIMI_K3_CONFIG,
        name="tiny_kimi_k3",
        hidden=64,
        q_lora=64,
        kv_lora=64,
        pe_dim=64,
        nope_dim=64,
        v_dim=64,
        n_experts=8,
        top_k=2,
        inter=16,
        local_heads=1,
        attention_output_gate=False,
    )
    weights = make_weights(0, heads=1, device="cpu", model_config=config, attention_only=True)
    weights.t["g_in"].fill_(float("nan"))
    weights.t["w_o"].zero_()
    hidden = torch.arange(64, dtype=torch.bfloat16).reshape(1, 64)
    kv_cache = torch.zeros(64, 64, dtype=torch.bfloat16)
    pe_cache = torch.zeros(64, 64, dtype=torch.bfloat16)
    cos, sin = rope_table(64, device="cpu", model_config=config)

    got = golden_layer(
        weights,
        hidden,
        0,
        kv_cache,
        pe_cache,
        torch.arange(64, dtype=torch.int32).reshape(1, 64),
        cos,
        sin,
        lambda value: value,
        sparse_attention_topk=64,
        attention_only=True,
    )

    torch.testing.assert_close(got["a"], torch.zeros_like(hidden), atol=0, rtol=0)


def test_validate_kimi_k3_attention_shard_rejects_glm_head_count():
    with pytest.raises(ValueError, match="kimi_k3 requires 12 local heads"):
        validate_shard(1, GLM5_CONFIG.local_heads, 0, 8, 2048, KIMI_K3_CONFIG)


def test_common_layout_reserves_two_epoch_slots_for_attention_only():
    config = replace(KIMI_K3_CONFIG, name="tiny_layout", hidden=64, local_heads=12)
    scratch, symmetric = layout(4, 12, 2, 128, model_config=config, attention_only=True)
    part = 2 * 4 * 64 * 8

    assert scratch["gate"] > 0
    assert "scores" not in scratch
    assert symmetric == {
        "attn": 0,
        "ffn": 2 * part,
        "_part_stride": part,
        "_bytes": 2 * part,
    }


def test_common_schedule_splits_twelve_heads_across_two_ctas_at_s1():
    s1 = dict(stage_tasks(1, 12, 128, KIMI_K3_CONFIG, attention_only=True))
    s4 = dict(stage_tasks(4, 12, 128, KIMI_K3_CONFIG, attention_only=True))

    assert s1["split"] == 4
    assert s4["split"] == 8


def test_kimi_k3_situ_applies_bounded_gate_and_up_branches():
    values = torch.tensor([[0.0, 4.0, 25.0, -25.0]])
    got = situ(values, beta=4.0, linear_beta=25.0)
    expected = torch.tensor([[0.0, -56.9593]])
    torch.testing.assert_close(got, expected, atol=0.25, rtol=0)


def test_kimi_attn_res_updates_prefix_and_block_bank_before_mixing():
    prefix = torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)
    delta = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
    blocks = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
    one = torch.ones(2, dtype=torch.bfloat16)
    output, updated = kimi_attn_res(prefix, delta, blocks, one, one, None, 0, 0)
    expected = torch.tensor([[2.0, 2.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(updated, expected, atol=0, rtol=0)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(blocks[:, 0], expected, atol=0, rtol=0)
