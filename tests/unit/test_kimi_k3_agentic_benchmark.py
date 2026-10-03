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
    critical_rank_medians,
    make_argument_parser,
)


def test_agentic_benchmark_protocol_defaults_are_reproducible() -> None:
    args = make_argument_parser().parse_args([])

    assert INVOCATIONS_PER_GRAPH == args.layers == 32
    assert WARMUPS == args.warmups == 8
    assert MIN_REPEATS == args.repeats == 50
    assert args.batches == (1, 2, 4)


def test_critical_rank_medians_take_each_replay_slowest_rank() -> None:
    rank_samples = (
        (100.0, 300.0, 500.0),
        (200.0, 250.0, 600.0),
    )

    assert critical_rank_medians(rank_samples, layers=10) == pytest.approx(
        (20.0, 30.0, 60.0)
    )


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
