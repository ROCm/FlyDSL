# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from typing import NamedTuple

RECORD_BYTES = 32
PAYLOAD_TYPES = ("uint64", "int64", "float32", "float64", "bool")

EVENTS_PER_WAVE = 256

UNPAIRED_SLOT = 0xFFFFFFFF

LAUNCH_COUNT_BYTE = 4
CURSOR_STRIDE_SLOTS = 8
MAX_PARTITIONS = 256
MIN_PARTITION_WAVES = 96


class Layout(NamedTuple):
    """Where waves find their slots. Slot 0 holds the launch count and partition p's wave counter is at
    slot CURSOR_STRIDE_SLOTS * (p + 1). The ``wave_slots`` slots after the counters are cut into
    EVENTS_PER_WAVE-slot wave regions, the last possibly short, and wave k of partition p takes region
    k * partitions + p."""

    partitions: int
    header_slots: int
    wave_slots: int

    @property
    def regions(self) -> int:
        return -(-self.wave_slots // EVENTS_PER_WAVE)

    def capacity(self, partition: int) -> int:
        """Waves partition ``partition`` can number before they find no region."""
        return max(0, -(-(self.regions - partition) // self.partitions))

    def cursor_byte(self, partition: int) -> int:
        return (partition + 1) * CURSOR_STRIDE_SLOTS * RECORD_BYTES


def _layout(total_slots: int, partitions: int) -> Layout:
    header_slots = CURSOR_STRIDE_SLOTS * (partitions + 1)
    return Layout(partitions, header_slots, total_slots - header_slots)


def layout(capacity_slots: int) -> Layout:
    """The layout of a buffer of ``capacity_slots + 1`` slots: the most partitions, up to MAX_PARTITIONS,
    that each still hold MIN_PARTITION_WAVES waves. A larger buffer never has fewer partitions."""
    partitions = MAX_PARTITIONS
    while partitions > 1:
        candidate = _layout(capacity_slots + 1, partitions)
        if candidate.capacity(partitions - 1) >= MIN_PARTITION_WAVES:
            return candidate
        partitions //= 2
    return _layout(capacity_slots + 1, 1)
