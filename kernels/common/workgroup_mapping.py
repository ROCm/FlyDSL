# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reusable workgroup-to-tile mappings for cache locality."""

import flydsl.expr as fx
from flydsl.expr import arith, const_expr


def remap_linear_pid_for_xcd(pid, num_workgroups, *, num_xcds: int = 8):
    """Map round-robin physical XCD assignment to contiguous logical ranges.

    CDNA dispatches consecutive workgroups round-robin across XCDs.  This
    mapping gives every XCD a contiguous range of logical workgroups, which
    keeps data reused by nearby tiles in the same XCD-local L2 slice.  Uneven
    grids assign one additional workgroup to the first ``num_workgroups %
    num_xcds`` XCDs.

    Set ``num_xcds=1`` to disable the remapping.
    """
    assert num_xcds > 0
    pid = fx.Int32(pid)

    if const_expr(num_xcds != 1):
        workgroups_per_xcd = num_workgroups // num_xcds
        remainder = num_workgroups % num_xcds
        xcd = pid % num_xcds
        local_pid = pid // num_xcds
        preceding_extra = arith.select(xcd < remainder, xcd, remainder)
        pid = xcd * workgroups_per_xcd + preceding_extra + local_pid

    return pid


def grouped_mn_from_linear(pid, num_pid_m, num_pid_n, *, group_size_m: int = 4):
    """Convert a linear PID to an M/N tile using grouped-M traversal.

    Visiting several M tiles before advancing N improves B-tile reuse.  The
    final group is shortened when ``num_pid_m`` is not divisible by
    ``group_size_m``.  Set ``group_size_m=1`` for ordinary N-fastest order.
    """
    assert group_size_m > 0
    pid = fx.Int32(pid)

    if const_expr(group_size_m != 1):
        pids_per_group = group_size_m * num_pid_n
        group_id = pid // pids_per_group
        first_pid_m = group_id * group_size_m
        remaining_m = num_pid_m - first_pid_m
        actual_group_size_m = arith.select(remaining_m < group_size_m, remaining_m, group_size_m)
        pid_in_group = pid % pids_per_group
        pid_m = first_pid_m + pid_in_group % actual_group_size_m
        pid_n = pid_in_group // actual_group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    return pid_m, pid_n


def remap_xcd_grouped_pid(
    pid,
    num_pid_m,
    num_pid_n,
    *,
    num_xcds: int = 8,
    group_size_m: int = 4,
):
    """Apply XCD-aware linear remapping followed by grouped-M traversal."""
    pid = remap_linear_pid_for_xcd(pid, num_pid_m * num_pid_n, num_xcds=num_xcds)
    return grouped_mn_from_linear(pid, num_pid_m, num_pid_n, group_size_m=group_size_m)


__all__ = [
    "grouped_mn_from_linear",
    "remap_linear_pid_for_xcd",
    "remap_xcd_grouped_pid",
]
