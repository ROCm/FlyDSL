# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Ticket-based tracking for VMEM and LGKM operations to enable optimal wait placement.

The ticketing system tracks outstanding memory operations and enables coalescing
of s_waitcnt instructions to minimize unnecessary waits while ensuring correctness.
"""

from typing import Optional


class Ticketing:
    """
    Tracks outstanding VMEM and LGKM operations via tickets for optimal s_waitcnt placement.

    This class manages ticket issuance and wait coalescing for two types of memory operations:
    - VMEM (Vector Memory): buffer_load/buffer_store operations
    - LGKM (LDS/GDS/Konstant/Message): ds_read/ds_write, s_load, sendmsg operations

    Ticket System:
    - Each operation is assigned a monotonically increasing ticket number
    - Tickets track the order of issued operations
    - Wait coalescing ensures we emit minimal s_waitcnt instructions

    Coalescing Rule:
    - vmcnt(N) or lgkmcnt(N) waits for all operations with ticket <= (last_ticket - N)
    - We only emit a wait if it's stricter than the previous wait
    - This avoids redundant waits and maximizes latency hiding
    """

    def __init__(self):
        # VMEM ticket tracking
        self._vmem_last_ticket = -1  # Last issued VMEM ticket
        self._vmem_last_wait_threshold = None  # Last vmcnt(N) emitted, for coalescing

        # LGKM ticket tracking
        self._lgkm_last_ticket = -1  # Last issued LGKM ticket
        self._lgkm_last_wait_threshold = None  # Last lgkmcnt(N) emitted, for coalescing

    # ========= VMEM Ticketing =========

    def next_vmem_ticket(self) -> int:
        """
        Issue next VMEM ticket for a buffer load/store operation.

        Returns:
            New ticket number for this VMEM operation
        """
        self._vmem_last_ticket += 1
        # Reset wait threshold since a new operation invalidates previous waits
        self._vmem_last_wait_threshold = None
        return self._vmem_last_ticket

    def compute_vmem_wait(self, min_required_ticket: int) -> Optional[int]:
        """
        Compute minimal vmcnt(N) threshold to ensure ticket is ready.

        The threshold allows (last_ticket - min_required_ticket) newer operations
        to remain in-flight while guaranteeing min_required_ticket is complete.

        Coalescing: Only returns a threshold if it's stricter than the last wait,
        otherwise returns None (indicating no wait needed).

        Args:
            min_required_ticket: Ticket that must be ready before proceeding

        Returns:
            vmcnt threshold N to emit in s_waitcnt vmcnt(N), or None if no wait needed

        Example:
            If last_ticket=5 and min_required_ticket=3:
            - threshold = max(0, 5-3) = 2
            - This means: wait for ticket 3, allow tickets 4 and 5 to remain in-flight
            - Emits: s_waitcnt vmcnt(2)
        """
        # Compute threshold: allow (last_ticket - min_required_ticket) newer operations
        threshold = max(0, self._vmem_last_ticket - min_required_ticket)

        # Coalesce: only return threshold if stricter than last wait (or first wait)
        if (
            self._vmem_last_wait_threshold is None
            or threshold < self._vmem_last_wait_threshold
        ):
            self._vmem_last_wait_threshold = threshold
            return threshold

        return None

    # ========= LGKM Ticketing =========

    def next_lgkm_ticket(self) -> int:
        """
        Issue next LGKM ticket for an LDS/scalar memory operation.

        Returns:
            New ticket number for this LGKM operation
        """
        self._lgkm_last_ticket += 1
        # Reset wait threshold since a new operation invalidates previous waits
        self._lgkm_last_wait_threshold = None
        return self._lgkm_last_ticket

    def compute_lgkm_wait(self, min_required_ticket: int) -> Optional[int]:
        """
        Compute minimal lgkmcnt(N) threshold to ensure ticket is ready.

        The threshold allows (last_ticket - min_required_ticket) newer operations
        to remain in-flight while guaranteeing min_required_ticket is complete.

        Coalescing: Only returns a threshold if it's stricter than the last wait,
        otherwise returns None (indicating no wait needed).

        Args:
            min_required_ticket: Ticket that must be ready before proceeding

        Returns:
            lgkmcnt threshold N to emit in s_waitcnt lgkmcnt(N), or None if no wait needed

        Example:
            If last_ticket=3 and min_required_ticket=1:
            - threshold = max(0, 3-1) = 2
            - This means: wait for ticket 1, allow tickets 2 and 3 to remain in-flight
            - Emits: s_waitcnt lgkmcnt(2)
        """
        # Compute threshold: allow (last_ticket - min_required_ticket) newer operations
        threshold = max(0, self._lgkm_last_ticket - min_required_ticket)

        # Coalesce: only return threshold if stricter than last wait (or first wait)
        if (
            self._lgkm_last_wait_threshold is None
            or threshold < self._lgkm_last_wait_threshold
        ):
            self._lgkm_last_wait_threshold = threshold
            return threshold

        return None

    def observe_lgkm_wait(self, threshold: int) -> None:
        """
        Record an externally emitted or implied LGKM wait.

        This method updates the internal coalescing state to reflect that an
        lgkmcnt(threshold) wait has been issued, either by external code
        (e.g., user-specified barriers) or by the scheduler.

        After observing a wait, subsequent compute_lgkm_wait() calls will
        return None unless a new, stricter wait is required.

        Args:
            threshold: The lgkmcnt threshold that was waited on (0 for full drain)

        Example:
            For a shared memory barrier that drains all LGKM:
            - emit s_waitcnt lgkmcnt(0)
            - call observe_lgkm_wait(0)
            - emit s_barrier
            This prevents redundant lgkmcnt waits after the barrier until
            new LGKM operations occur.
        """
        if (
            self._lgkm_last_wait_threshold is None
            or threshold < self._lgkm_last_wait_threshold
        ):
            self._lgkm_last_wait_threshold = threshold

    def observe_vmem_wait(self, threshold: int) -> None:
        """
        Record an externally emitted VMEM wait.

        This method updates the internal coalescing state to reflect that a
        vmcnt(threshold) wait has been issued by external code (e.g., MLIR's
        rocdl.s.waitcnt).

        After observing a wait, subsequent VMEM operations won't need to wait
        again unless new VMEM operations are issued first.

        Args:
            threshold: The vmcnt threshold that was waited on (0 for full drain)
        """
        if (
            self._vmem_last_wait_threshold is None
            or threshold < self._vmem_last_wait_threshold
        ):
            self._vmem_last_wait_threshold = threshold
