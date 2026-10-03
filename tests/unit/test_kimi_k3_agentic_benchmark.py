# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from __future__ import annotations

import pytest
import torch

from kernels.monokernel.k3.tools.agentic_kda import (
    INVOCATIONS_PER_GRAPH,
    MIN_REPEATS,
    WARMUPS,
    AgenticKdaShape,
    critical_rank_kernel_profile,
    critical_rank_medians,
    graph_epoch_plan,
    make_argument_parser,
    parity_metrics,
    q8_conv_reference,
    q8_recurrence_reference,
)
from kernels.monokernel.k3.kernel import (
    agentic_conv_writeback_requires_all,
    agentic_recurrence_tokens_per_cta,
    mtp_conv_waits_for_previous,
)


def test_agentic_benchmark_protocol_defaults_are_reproducible() -> None:
    args = make_argument_parser().parse_args([])

    assert INVOCATIONS_PER_GRAPH == args.layers == 32
    assert WARMUPS == args.warmups == 8
    assert MIN_REPEATS == args.repeats == 50
    assert args.batches == (1, 2, 4)


def test_agentic_recurrence_uses_four_token_resident_chunks() -> None:
    assert agentic_recurrence_tokens_per_cta(agentic_batch_size=0) == 2
    assert agentic_recurrence_tokens_per_cta(agentic_batch_size=1) == 4
    assert agentic_recurrence_tokens_per_cta(agentic_batch_size=4) == 4


def test_critical_rank_medians_take_each_replay_slowest_rank() -> None:
    rank_samples = (
        (100.0, 300.0, 500.0),
        (200.0, 250.0, 600.0),
    )

    assert critical_rank_medians(rank_samples, layers=10) == pytest.approx(
        (20.0, 30.0, 60.0)
    )


def test_kernel_profile_selects_median_replay_critical_rank() -> None:
    profiles = (
        (
            {"rank": 0, "repeat": 0, "graph_us": 2000.0},
            {"rank": 0, "repeat": 1, "graph_us": 3000.0},
            {"rank": 0, "repeat": 2, "graph_us": 2500.0},
        ),
        (
            {"rank": 1, "repeat": 0, "graph_us": 2400.0},
            {"rank": 1, "repeat": 1, "graph_us": 2800.0},
            {
                "rank": 1,
                "repeat": 2,
                "graph_us": 2600.0,
                "kernels": {
                    "front": {"calls": 32, "total_us": 960.0},
                    "tail": {"calls": 128, "total_us": 1280.0},
                },
                "stages": {"attention": 15.0},
            },
        ),
    )

    result = critical_rank_kernel_profile(profiles, layers=32)

    assert result["critical_rank"] == 1
    assert result["median_repeat"] == 2
    assert result["profiled_graph_us_per_layer"] == pytest.approx(81.25)
    assert result["stages"] == {"attention": 15.0}
    assert result["kernels"][0] == {
        "name": "tail",
        "calls_per_layer": 4.0,
        "total_us_per_layer": 40.0,
        "mean_us": 10.0,
    }


def test_agentic_shape_selects_mixed_recurrence_snapshots() -> None:
    shape = AgenticKdaShape(batch=4)
    snapshots = torch.tensor(
        (
            (10, 11, 12, 13, 14, 15, 16, 17),
            (20, 21, 22, 23, 24, 25, 26, 27),
            (30, 31, 32, 33, 34, 35, 36, 37),
            (40, 41, 42, 43, 44, 45, 46, 47),
        ),
        dtype=torch.int32,
    )
    accepted = torch.tensor((1, 4, 8, 2), dtype=torch.int32)

    inputs, outputs = shape.transition_slots(snapshots, accepted)

    assert inputs.tolist() == [
        [10, 10, 11, 12, 13, 14, 15, 16],
        [23, 20, 21, 22, 23, 24, 25, 26],
        [37, 30, 31, 32, 33, 34, 35, 36],
        [41, 40, 41, 42, 43, 44, 45, 46],
    ]
    assert outputs.data_ptr() == snapshots.data_ptr()


def test_agentic_shape_builds_mixed_acceptance_conv_rollback_plan() -> None:
    shape = AgenticKdaShape(batch=4)
    accepted = torch.tensor((1, 3, 6, 8), dtype=torch.int32)

    reads, final = shape.conv_window_plan(accepted)

    for request, count in enumerate(accepted.tolist()):
        old = torch.arange(10, dtype=torch.int64) + request * 100
        draft = torch.arange(8, dtype=torch.int64) + request * 1000
        source = torch.cat((old, draft))
        independent_reads = []
        for token in range(8):
            history = old[count - 1 : count + 2].tolist() + draft[:token].tolist()
            independent_reads.append(history[-3:])
        expected_final = torch.cat((old[count : count + 2], draft))

        assert source[reads[request]].tolist() == independent_reads
        assert torch.equal(source[final[request]], expected_final)


def test_graph_epoch_plan_uses_all_layers_and_one_advance() -> None:
    layers, advances = graph_epoch_plan(32)

    assert layers == tuple(range(32))
    assert advances == 1


def test_agentic_convolution_tokens_have_no_false_dependency_chain() -> None:
    assert [
        mtp_conv_waits_for_previous(agentic_batch_size=4, token=token)
        for token in range(8)
    ] == [False] * 8
    assert [
        agentic_conv_writeback_requires_all(token) for token in range(8)
    ] == [False] * 7 + [True]
    assert mtp_conv_waits_for_previous(agentic_batch_size=0, token=1)


def test_parity_metrics_compute_booleans_from_untouched_values() -> None:
    expected = torch.tensor((1.0, 2.0))
    actual = expected.clone()
    metrics = parity_metrics(
        expected_output=expected,
        actual_output=actual,
        expected_state=expected,
        actual_state=actual,
        expected_conv=expected,
        actual_conv=actual,
        rank_equal=False,
    )

    assert metrics["conv_exact"] is True
    assert metrics["state_exact"] is True
    assert metrics["rank_equal"] is False
    assert metrics["output_nonzero"] is True


def test_q8_reference_uses_mixed_input_slots_and_fp32_carry() -> None:
    batch, query_len, slots, heads, width = 8, 8, 64, 1, 2
    snapshots = torch.arange(slots, dtype=torch.int32).view(batch, query_len)
    accepted = torch.arange(1, query_len + 1, dtype=torch.int32)
    initial = torch.empty(slots, heads, width, width, dtype=torch.float16)
    for slot in range(slots):
        initial[slot].fill_(slot + 1)
    query = torch.zeros(batch, query_len, heads, width)
    key = torch.zeros_like(query)
    query[..., 0] = 1
    key[..., 0] = 1
    value = torch.zeros_like(query)
    gate = torch.zeros_like(query)
    beta = torch.full((batch, query_len, heads), -100.0)

    _, persisted, carry = q8_recurrence_reference(
        initial,
        snapshots,
        accepted,
        query,
        key,
        value,
        gate,
        beta,
        torch.zeros(heads),
        torch.zeros(heads, width),
    )

    decay = torch.exp(torch.tensor(-2.5))
    for request, count in enumerate(accepted.tolist()):
        input_slot = int(snapshots[request, count - 1])
        for token in range(query_len):
            expected = initial[input_slot].float() * decay ** (token + 1)
            torch.testing.assert_close(
                persisted[int(snapshots[request, token])].float(),
                expected.half().float(),
            )
        torch.testing.assert_close(carry[request], expected)


def test_q8_conv_reference_rolls_back_two_committed_rows_and_eight_drafts() -> None:
    snapshots = torch.arange(64, dtype=torch.int32).view(8, 8)
    accepted = torch.arange(1, 9, dtype=torch.int32)
    conv = torch.arange(64 * 10, dtype=torch.bfloat16).view(64, 10, 1)
    drafts = torch.arange(800, 864, dtype=torch.bfloat16).view(8, 8, 1)
    weight = torch.ones(1, 4, dtype=torch.bfloat16)

    output, persisted = q8_conv_reference(
        conv, snapshots, accepted, drafts, weight
    )

    for request, count in enumerate(accepted.tolist()):
        slot = int(snapshots[request, 0])
        expected_first = (
            conv[slot, count - 1 : count + 2, 0].float().sum()
            + drafts[request, 0, 0].float()
        )
        torch.testing.assert_close(
            output[request, 0, 0],
            expected_first * torch.sigmoid(expected_first),
        )
        assert torch.equal(
            persisted[slot, :2, 0],
            conv[slot, count : count + 2, 0],
        )
        assert torch.equal(persisted[slot, 2:, 0], drafts[request, :, 0])
