# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Kernel-level linear scan register allocator.

This module provides a linear scan register allocator that handles the entire
kernel, including:
- Virtual registers from expression emission
- MFMA accumulator blocks
- Buffer descriptors
- LDS addresses

Key features:
- Constraint-aware: respects alignment, size, and class requirements
- Precoloring support: ABI-mandated registers (v0, s[0:1], etc.)
- Range allocation: contiguous blocks for pairs, quads, MFMA accumulators
- No spilling: fails compilation with diagnostic if allocation fails
- Separate VGPR and SGPR allocation tracks

The allocator uses the classic linear scan algorithm:
1. Sort live ranges by start point
2. Process ranges in order
3. At each range start, expire any finished ranges
4. Allocate from free pool, respecting constraints
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import os

from .kernel_ir import (
    KernelProgram,
    KVReg,
    KSReg,
    RegClass,
)
from .kernel_liveness import LiveRange, LivenessInfo, compute_liveness
from .kernel_generator import PhysicalMapping


# Debug flag
DEBUG_ALLOC = os.environ.get("WAVE_KERNEL_ALLOC_DEBUG", "0") == "1"


class AllocationError(Exception):
    """Raised when register allocation fails (no spilling)."""

    pass


@dataclass
class AllocationStats:
    """Statistics from register allocation."""

    peak_vgprs: int = 0
    peak_sgprs: int = 0
    total_vregs: int = 0
    total_sregs: int = 0
    ranges_allocated: int = 0
    ranges_expired: int = 0


@dataclass
class RegPool:
    """
    Pool of available physical registers.

    Manages allocation and freeing of physical registers with support for:
    - Single register allocation
    - Contiguous range allocation with alignment
    - Reserved/precolored registers
    """

    reg_class: RegClass
    max_regs: int
    reserved: Set[int] = field(default_factory=set)

    # Free list (sorted for range allocation)
    _free: List[int] = field(default_factory=list, init=False)
    # Currently allocated
    _allocated: Set[int] = field(default_factory=set, init=False)
    # Track peak usage
    _peak: int = field(default=0, init=False)

    def __post_init__(self):
        # Initialize free list with all non-reserved registers
        self._free = sorted([i for i in range(self.max_regs) if i not in self.reserved])
        self._allocated = set()

    def alloc_single(self) -> int:
        """Allocate a single register."""
        if not self._free:
            raise AllocationError(
                f"Out of {self.reg_class.name}s: all {self.max_regs} registers in use"
            )
        reg = self._free.pop(0)  # Take lowest available
        self._allocated.add(reg)
        self._peak = max(self._peak, len(self._allocated))
        if DEBUG_ALLOC:
            prefix = "v" if self.reg_class == RegClass.VGPR else "s"
            print(f"[ALLOC] {prefix}{reg} (single)")
        return reg

    def alloc_range(self, size: int, alignment: int = 1) -> int:
        """
        Allocate a contiguous range of registers.

        Args:
            size: Number of consecutive registers needed
            alignment: Required alignment (base must be divisible by this)

        Returns:
            Base register index of the allocated range

        Raises:
            AllocationError if no suitable range is available
        """
        if size <= 0:
            raise ValueError(f"Invalid allocation size: {size}")

        # Find first suitable range in free list
        # Need to find 'size' consecutive registers starting at aligned index
        free_set = set(self._free)

        for candidate in self._free:
            # Check alignment
            if candidate % alignment != 0:
                continue

            # Check if all registers in range are available
            range_regs = list(range(candidate, candidate + size))
            if all(r in free_set and r < self.max_regs for r in range_regs):
                # Found suitable range - allocate it
                for r in range_regs:
                    self._free.remove(r)
                    self._allocated.add(r)
                self._peak = max(self._peak, len(self._allocated))
                if DEBUG_ALLOC:
                    prefix = "v" if self.reg_class == RegClass.VGPR else "s"
                    print(
                        f"[ALLOC] {prefix}[{candidate}:{candidate+size-1}] (range, align={alignment})"
                    )
                return candidate

        # No suitable range found
        raise AllocationError(
            f"Cannot allocate {size} contiguous {self.reg_class.name}s with alignment {alignment}. "
            f"Free: {len(self._free)}, allocated: {len(self._allocated)}, max: {self.max_regs}"
        )

    def free_single(self, reg: int) -> None:
        """Free a single register."""
        if reg in self.reserved:
            return  # Don't free reserved registers
        if reg not in self._allocated:
            return  # Already free
        self._allocated.discard(reg)
        # Insert in sorted position
        import bisect

        bisect.insort(self._free, reg)
        if DEBUG_ALLOC:
            prefix = "v" if self.reg_class == RegClass.VGPR else "s"
            print(f"[FREE] {prefix}{reg}")

    def free_range(self, base: int, size: int) -> None:
        """Free a contiguous range of registers."""
        for r in range(base, base + size):
            self.free_single(r)

    @property
    def peak_usage(self) -> int:
        """Get peak number of registers allocated."""
        return self._peak

    @property
    def current_usage(self) -> int:
        """Get current number of registers allocated."""
        return len(self._allocated)


@dataclass
class KernelRegAlloc:
    """
    Kernel-level linear scan register allocator.

    Allocates physical registers to virtual registers in a KernelProgram
    using live range information.
    """

    # Configuration
    max_vgprs: int = 256
    max_sgprs: int = 104

    # Reserved registers (from ABI)
    reserved_vgprs: Set[int] = field(default_factory=set)
    reserved_sgprs: Set[int] = field(default_factory=set)

    # Precolored registers (virtual -> physical mapping that must be honored)
    precolored_vregs: Dict[KVReg, int] = field(default_factory=dict)
    precolored_sregs: Dict[KSReg, int] = field(default_factory=dict)

    # Internal state
    _vgpr_pool: RegPool = field(init=False)
    _sgpr_pool: RegPool = field(init=False)
    _stats: AllocationStats = field(default_factory=AllocationStats, init=False)

    def __post_init__(self):
        # Include precolored physical registers in the reserved sets
        # This prevents the allocator from using them for other variables
        vgpr_reserved = set(self.reserved_vgprs)
        for vreg, phys in self.precolored_vregs.items():
            vgpr_reserved.add(phys)

        sgpr_reserved = set(self.reserved_sgprs)
        for sreg, phys in self.precolored_sregs.items():
            sgpr_reserved.add(phys)

        self._vgpr_pool = RegPool(RegClass.VGPR, self.max_vgprs, vgpr_reserved)
        self._sgpr_pool = RegPool(RegClass.SGPR, self.max_sgprs, sgpr_reserved)

    def reserve_vgpr(self, reg: int) -> None:
        """Reserve a physical VGPR (prevents allocation)."""
        self.reserved_vgprs.add(reg)
        if reg in self._vgpr_pool._free:
            self._vgpr_pool._free.remove(reg)

    def reserve_sgpr(self, reg: int) -> None:
        """Reserve a physical SGPR (prevents allocation)."""
        self.reserved_sgprs.add(reg)
        if reg in self._sgpr_pool._free:
            self._sgpr_pool._free.remove(reg)

    def precolor_vreg(self, vreg: KVReg, phys: int) -> None:
        """Force a virtual VGPR to a specific physical register."""
        self.precolored_vregs[vreg] = phys
        self.reserve_vgpr(phys)

    def precolor_sreg(self, sreg: KSReg, phys: int) -> None:
        """Force a virtual SGPR to a specific physical register."""
        self.precolored_sregs[sreg] = phys
        self.reserve_sgpr(phys)

    def allocate(self, program: KernelProgram) -> PhysicalMapping:
        """
        Allocate physical registers for all virtual registers in the program.

        Uses linear scan algorithm:
        1. Compute live ranges
        2. Sort by start point
        3. Process each range: expire finished, allocate new

        Args:
            program: The kernel program to allocate registers for

        Returns:
            PhysicalMapping from virtual to physical registers

        Raises:
            AllocationError if allocation fails (no spilling)
        """
        # Compute liveness
        liveness = compute_liveness(program)

        # Validate SSA
        from .kernel_liveness import validate_ssa

        errors = validate_ssa(program)
        if errors:
            raise AllocationError(f"SSA validation failed:\n" + "\n".join(errors))

        # Initialize mapping with precolored registers
        vreg_map: Dict[int, int] = {}
        sreg_map: Dict[int, int] = {}
        vreg_ranges: Dict[int, Tuple[int, int]] = {}  # component_id -> (base_id, size)
        sreg_ranges: Dict[int, Tuple[int, int]] = {}

        for vreg, phys in self.precolored_vregs.items():
            vreg_map[vreg.id] = phys
        for sreg, phys in self.precolored_sregs.items():
            sreg_map[sreg.id] = phys

        # Active ranges (currently allocated, sorted by end point)
        active_vranges: List[Tuple[int, LiveRange, int]] = []  # (end, range, phys_base)
        active_sranges: List[Tuple[int, LiveRange, int]] = []

        # Process VGPR ranges
        for lr in liveness.vreg_ranges:
            # Skip precolored
            if isinstance(lr.reg, KVReg) and lr.reg.id in vreg_map:
                continue

            # Expire finished ranges
            active_vranges = self._expire_ranges(
                active_vranges, lr.start, self._vgpr_pool
            )

            # Allocate
            try:
                if lr.size == 1:
                    phys = self._vgpr_pool.alloc_single()
                else:
                    phys = self._vgpr_pool.alloc_range(lr.size, lr.alignment)
            except AllocationError as e:
                self._report_allocation_failure(lr, liveness, "VGPR")
                raise

            vreg_map[lr.reg.id] = phys

            # For ranges, track all component registers
            if lr.size > 1:
                for i in range(lr.size):
                    vreg_ranges[lr.reg.id + i] = (lr.reg.id, lr.size)

            active_vranges.append((lr.end, lr, phys))
            active_vranges.sort(key=lambda x: x[0])
            self._stats.ranges_allocated += 1

        # Process SGPR ranges
        for lr in liveness.sreg_ranges:
            # Skip precolored
            if isinstance(lr.reg, KSReg) and lr.reg.id in sreg_map:
                continue

            # Expire finished ranges
            active_sranges = self._expire_ranges(
                active_sranges, lr.start, self._sgpr_pool
            )

            # Allocate
            try:
                if lr.size == 1:
                    phys = self._sgpr_pool.alloc_single()
                else:
                    phys = self._sgpr_pool.alloc_range(lr.size, lr.alignment)
            except AllocationError as e:
                self._report_allocation_failure(lr, liveness, "SGPR")
                raise

            sreg_map[lr.reg.id] = phys

            # For ranges, track all component registers
            if lr.size > 1:
                for i in range(lr.size):
                    sreg_ranges[lr.reg.id + i] = (lr.reg.id, lr.size)

            active_sranges.append((lr.end, lr, phys))
            active_sranges.sort(key=lambda x: x[0])
            self._stats.ranges_allocated += 1

        # Record stats - include BOTH reserved and dynamically allocated registers
        # We need the highest register index used + 1 to get the count

        # Calculate highest VGPR index used (including reserved and ranges)
        max_vgpr = -1
        for phys_base in vreg_map.values():
            max_vgpr = max(max_vgpr, phys_base)
        # Account for ranges (vreg_ranges maps individual component to (base, size))
        # Note: vreg_map keys are integers (virtual reg ids), not KVReg objects
        for vreg_id, (base, size) in vreg_ranges.items():
            phys_base = vreg_map.get(base)
            if phys_base is not None:
                max_vgpr = max(max_vgpr, phys_base + size - 1)
        # Include reserved
        if self.reserved_vgprs:
            max_vgpr = max(max_vgpr, max(self.reserved_vgprs))
        self._stats.peak_vgprs = max_vgpr + 1 if max_vgpr >= 0 else 0

        # Calculate highest SGPR index used (including reserved and ranges)
        max_sgpr = -1
        for phys_base in sreg_map.values():
            max_sgpr = max(max_sgpr, phys_base)
        # Account for ranges (sreg_ranges maps individual component to (base, size))
        # Note: sreg_map keys are integers (virtual reg ids), not KSReg objects
        for sreg_id, (base, size) in sreg_ranges.items():
            phys_base = sreg_map.get(base)
            if phys_base is not None:
                max_sgpr = max(max_sgpr, phys_base + size - 1)
        # Include reserved
        if self.reserved_sgprs:
            max_sgpr = max(max_sgpr, max(self.reserved_sgprs))
        self._stats.peak_sgprs = max_sgpr + 1 if max_sgpr >= 0 else 0

        self._stats.total_vregs = len(vreg_map)
        self._stats.total_sregs = len(sreg_map)

        return PhysicalMapping(vreg_map, sreg_map, vreg_ranges, sreg_ranges)

    def _expire_ranges(
        self,
        active: List[Tuple[int, LiveRange, int]],
        current_point: int,
        pool: RegPool,
    ) -> List[Tuple[int, LiveRange, int]]:
        """
        Expire ranges that have ended before current_point.

        Returns the updated active list (ranges still alive).
        """
        still_active = []
        for end, lr, phys in active:
            if end < current_point:
                # This range has ended - free its registers
                pool.free_range(phys, lr.size)
                self._stats.ranges_expired += 1
            else:
                still_active.append((end, lr, phys))
        return still_active

    def _report_allocation_failure(
        self, failed_range: LiveRange, liveness: LivenessInfo, reg_class: str
    ) -> None:
        """Print diagnostic information when allocation fails."""
        print(f"\n{'='*60}")
        print(f"REGISTER ALLOCATION FAILED")
        print(f"{'='*60}")
        print(f"Failed to allocate: {failed_range}")
        print(f"Register class: {reg_class}")
        print(f"Size: {failed_range.size}, Alignment: {failed_range.alignment}")
        print(f"\nLive ranges at point {failed_range.start}:")

        # Find overlapping ranges
        if reg_class == "VGPR":
            ranges = liveness.vreg_ranges
        else:
            ranges = liveness.sreg_ranges

        overlapping = [r for r in ranges if r.overlaps(failed_range)]
        overlapping.sort(key=lambda r: -r.length)  # Sort by length descending

        print(f"Total overlapping: {len(overlapping)}")
        print(f"Top contributors (by range length):")
        for i, r in enumerate(overlapping[:10]):
            print(f"  {i+1}. {r}")

        print(f"{'='*60}\n")

    @property
    def stats(self) -> AllocationStats:
        """Get allocation statistics."""
        return self._stats


def allocate_kernel(
    program: KernelProgram,
    reserved_vgprs: Optional[Set[int]] = None,
    reserved_sgprs: Optional[Set[int]] = None,
    precolored_vregs: Optional[Dict[KVReg, int]] = None,
    precolored_sregs: Optional[Dict[KSReg, int]] = None,
) -> Tuple[PhysicalMapping, AllocationStats]:
    """
    Convenience function to allocate registers for a kernel program.

    Args:
        program: The kernel program
        reserved_vgprs: Physical VGPRs that cannot be allocated (ABI)
        reserved_sgprs: Physical SGPRs that cannot be allocated (ABI)
        precolored_vregs: Virtual VGPRs that must map to specific physical regs
        precolored_sregs: Virtual SGPRs that must map to specific physical regs

    Returns:
        Tuple of (PhysicalMapping, AllocationStats)
    """
    allocator = KernelRegAlloc(
        max_vgprs=program.max_vgprs,
        max_sgprs=program.max_sgprs,
        reserved_vgprs=reserved_vgprs or set(),
        reserved_sgprs=reserved_sgprs or set(),
        precolored_vregs=precolored_vregs or {},
        precolored_sregs=precolored_sregs or {},
    )

    # Apply ABI reservations from program
    for vgpr in program.abi.get_reserved_vgprs():
        allocator.reserve_vgpr(vgpr)
    for sgpr in program.abi.get_reserved_sgprs():
        allocator.reserve_sgpr(sgpr)

    mapping = allocator.allocate(program)
    return mapping, allocator.stats
