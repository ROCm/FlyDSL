# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host side of in-kernel wave tracing: allocate the buffer, bind it, read it back.

The device writes records into a buffer whose address is passed to instrumented
kernels as a trailing implicit argument (added by ``compiler/kernel_function.py``;
the buffer itself is owned by :mod:`flydsl.expr.rocdl.ktrace_emit`, and the stores
are emitted by ``convert-fly-ktrace-to-rocdl``).  Nothing here resolves a device
symbol, which is what keeps traced kernels disk-cacheable: registering a
``post_load_processors`` callback would set ``extern_linked`` and disable the cache.

Allocation and readback go through ``libfly_jit_runtime.so`` -- the same runtime the
JIT already loads to launch kernels -- rather than opening ``libamdhip64.so`` again
here. Its ``mgpuTraceBuffer*`` entry points wrap the HIP calls this buffer needs with
the synchronous semantics it depends on; see ``lib/Runtime/ROCm``.

It sits beside :mod:`~flydsl.runtime.device_runtime.rocm` because this is still HIP
glue, and the package is where the tree keeps it. It is deliberately *not*
imported by the package ``__init__`` -- doing so would pull ``flydsl.expr`` into
every ``import flydsl.runtime``, which today needs none of it.
"""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass

from ...expr.experimental import ktrace as _ktrace

_HIP_SUCCESS = 0
_RT = None


def _runtime():
    """Handle on libfly_jit_runtime, which already backs every kernel launch.

    Loaded lazily, so importing flydsl never needs a GPU runtime.

    Unguarded, unlike the buffer itself: CDLL on an already-loaded library returns the
    same one and the argtypes assignments are idempotent, so a race converges rather
    than leaking a second allocation.
    """
    global _RT
    if _RT is None:
        from ...compiler.jit_executor import _resolve_runtime_libs

        _RT = ctypes.CDLL(_resolve_runtime_libs()[0])
        _RT.mgpuMemAlloc.restype = ctypes.c_void_p
        _RT.mgpuMemAlloc.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_bool]
        _RT.mgpuTraceBufferClear.restype = ctypes.c_int
        _RT.mgpuTraceBufferClear.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        _RT.mgpuTraceBufferRead.restype = ctypes.c_int
        _RT.mgpuTraceBufferRead.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    return _RT


def _check(code: int, what: str) -> None:
    if code != _HIP_SUCCESS:
        raise RuntimeError(f"{what} failed with HIP error {code}")


def _current_device() -> int:
    """The HIP device the calling thread is bound to.

    Delegates rather than querying HIP here: runtime/device_runtime already owns that
    query, including the soname fallback for older ROCm installs.
    """
    from . import get_device_runtime

    return get_device_runtime().current_device_id()


@dataclass
class TraceBuffer:
    """A device trace buffer bound to one loaded module."""

    device_ptr: int
    capacity_slots: int
    # The device this address belongs to. A pointer from one device is not valid on
    # another without peer access, and the readback below only synchronises the
    # current one, so a trace taken after switching devices would be read off the
    # wrong queue -- silently, since the addresses are plain integers.
    device_id: int = -1

    def reset(self) -> None:
        """Zero the whole buffer so the next launch starts from an empty trace.

        The cursor alone is not enough.  Records are dropped by :func:`decode_records`
        on a zero timestamp, so slots left behind by a longer previous launch would be
        re-read as records of the next one once the cursor rewinds past them.  Each
        wave's counter word has to go back to zero for the same reason.

        Called by :func:`flydsl.expr.experimental.ktrace.collect` and
        :func:`flydsl.expr.experimental.ktrace.summary` once the records have been read back, so a
        process that traces repeatedly -- a benchmark loop, an autotune sweep -- does
        not accumulate claims until the buffer overflows.
        """
        # Runs from a finally, so it can be reached even when the read above raised:
        # check the device here too rather than memsetting an address this device does
        # not own.
        self._require_own_device()
        _check(
            _runtime().mgpuTraceBufferClear(
                ctypes.c_void_p(self.device_ptr), (self.capacity_slots + 1) * _ktrace.RECORD_BYTES
            ),
            "clearing the trace buffer",
        )

    def _require_own_device(self) -> None:
        """Refuse to read a buffer that belongs to another device.

        One buffer is allocated per process, so a second launcher on a different
        device would be handed this address: invalid there without peer access, and
        the readback below would drain the wrong device's queue. Tracing across
        devices needs a buffer per device; until then this is an error rather than a
        trace that looks fine and describes the wrong work.
        """
        if self.device_id < 0:
            return
        current = _current_device()
        if current != self.device_id:
            raise RuntimeError(
                f"ktrace: the trace buffer was allocated on device {self.device_id} but the "
                f"current device is {current}. Tracing spans one device per process; run the "
                "traced kernels on the device the buffer belongs to, or trace them in "
                "separate processes."
            )

    def claimed_slots(self) -> int:
        """Slots waves *reserved*, not events written.

        Every wave that reaches the prologue claims ``ktrace.EVENTS_PER_WAVE`` slots
        up front, whether or not it goes on to record, so this is a high-water mark: 64
        waves at the default 256 reads 16384 even if only two events were emitted. It is
        the right quantity for the overflow check (it measures what the run *wanted*), and
        :func:`decode_records` drops the untouched slots.
        """
        self._require_own_device()
        out = ctypes.c_uint32(0)
        _check(
            _runtime().mgpuTraceBufferRead(ctypes.byref(out), ctypes.c_void_p(self.device_ptr), 4),
            "reading the trace cursor",
        )
        return int(out.value)

    def read_records(self) -> list[dict]:
        """Copy back and decode the records written since the last read.

        Raises when waves claimed more slots than the buffer holds.  The trace is
        rejected rather than trimmed: a partial trace has unbalanced push/pop and
        renders as a corrupt timeline, which is worse than no trace at all.
        """
        claimed = self.claimed_slots()
        if claimed > self.capacity_slots:
            # +1 for the cursor: allocate() keeps slot 0 for it and reports
            # capacity_bytes // RECORD_BYTES - 1, so suggesting claimed * RECORD_BYTES
            # sends the caller to a buffer that is still one slot short and raises
            # again with the identical message.
            needed = (claimed + 1) * _ktrace.RECORD_BYTES
            raise RuntimeError(
                f"trace overflowed: waves claimed {claimed} slots, buffer holds "
                f"{self.capacity_slots}. Re-run with FLYDSL_KTRACE_BUFFER_BYTES>={needed}, "
                f"a block filter (FLYDSL_KTRACE_BLOCKS), or fewer event sites."
            )

        nbytes = claimed * _ktrace.RECORD_BYTES
        if nbytes == 0:
            return []
        raw = (ctypes.c_char * nbytes)()
        # Skip the global cursor slot. Each wave's first reserved word is its own event
        # counter, not a record; decode_records drops it because its timestamp is zero.
        _check(
            _runtime().mgpuTraceBufferRead(raw, ctypes.c_void_p(self.device_ptr + _ktrace.RECORD_BYTES), nbytes),
            "reading the trace records",
        )
        records, overflowed = decode_records(bytes(raw))
        if overflowed:
            worst = max(count for _, count in overflowed)
            raise RuntimeError(
                f"trace truncated: {len(overflowed)} wave(s) emitted more than "
                f"{_ktrace.EVENTS_PER_WAVE - 1} events (worst: {worst}). The records past "
                "that point were dropped, so ranges are unbalanced and phases would be "
                "mispaired -- a plausible-looking timeline for the wrong phase. Annotate "
                "at a coarser granularity, or restrict recording with FLYDSL_KTRACE_BLOCKS."
            )
        return records


def decode_records(raw: bytes) -> list[dict]:
    """Decode packed 32-byte records.

    Slots never written keep the buffer's zero fill; those decode to kind 0 with a zero
    timestamp and are dropped, which is how a wave that used fewer events than it
    reserved leaves no trace of the unused tail.

    Each wave's first reserved slot holds its event counter, not a record.  That word is
    a small integer landing in the ``ts`` field, so it would otherwise decode as a record
    with an absurd timestamp -- skip it by position, which is exact: wave *k* claims
    ``EVENTS_PER_WAVE`` slots starting at ``k * EVENTS_PER_WAVE + 1`` (slot 0 is the
    global cursor), so the counters are exactly those base slots.

    Those counters are also what detects a wave that ran past its budget.  The device
    bumps the counter on every event and only then suppresses the store, so a counter
    above the per-wave capacity is the exact count of what the wave attempted.  Reading
    it is the only way to see the overflow: the global cursor counts *claims*
    (EVENTS_PER_WAVE per wave) and is identical whether a wave emitted one event or a
    thousand.  Returned alongside the records for :meth:`TraceBuffer.read_records` to
    reject on.
    """
    per_wave = _ktrace.EVENTS_PER_WAVE
    out = []
    overflowed = []
    for off in range(0, len(raw) - _ktrace.RECORD_BYTES + 1, _ktrace.RECORD_BYTES):
        # `raw` starts at slot 1, so the absolute slot index is the offset index plus one.
        abs_slot = off // _ktrace.RECORD_BYTES + 1
        if per_wave > 0 and (abs_slot - 1) % per_wave == 0:
            # This wave's counter word, not a record. Its value is the number of events
            # the wave attempted, which is what reveals a truncated wave.
            attempted = struct.unpack_from("<I", raw, off)[0]
            if attempted > per_wave - 1:
                overflowed.append((abs_slot // per_wave, attempted))
            continue
        ts, payload, hw_id, xcc_id, packed, start_slot = struct.unpack_from("<QQIIII", raw, off)
        if ts == 0:
            continue
        out.append(
            {
                "ts": ts,
                "payload": payload,
                "hw_id": hw_id,
                "xcc_id": xcc_id,
                "event_id": packed & 0xFFFFFF,
                "kind": (packed >> 24) & 0xFF,
                "start_slot": start_slot,
                # Absolute buffer slot, matching the index the kernel stores in
                # start_slot: the copy starts one record past the buffer, so the
                # offset within `raw` is one slot short of the absolute index.
                "slot": abs_slot,
                # HW_ID fields, per AMD's shipped HIP header. SE_ID is 2 bits on
                # gfx942/gfx950, unlike the 3 bits used on gfx908/gfx90a.
                "wave": hw_id & 0xF,
                "simd": (hw_id >> 4) & 0x3,
                "cu": (hw_id >> 8) & 0xF,
                "sh": (hw_id >> 12) & 0x1,
                "se": (hw_id >> 13) & 0x3,
            }
        )
    return out, overflowed


def buffer_bytes() -> int:
    from ...utils import env

    return int(env.ktrace.buffer_bytes)


def allocate(capacity_bytes: int | None = None) -> TraceBuffer:
    """Allocate a trace buffer for the current process.

    The buffer address is passed to instrumented kernels as a trailing argument, so
    nothing here needs a loaded module or a device-symbol lookup -- which is what keeps
    traced kernels disk-cacheable.
    """
    capacity_bytes = capacity_bytes if capacity_bytes is not None else buffer_bytes()
    capacity_slots = capacity_bytes // _ktrace.RECORD_BYTES
    if capacity_slots == 0:
        raise ValueError(f"FLYDSL_KTRACE_BUFFER_BYTES={capacity_bytes} is smaller than one record")

    # mgpuMemAlloc reports failures to stderr and returns null, so the null is what has
    # to be caught: it would otherwise reach the kernel and fault on the first store.
    rt = _runtime()
    address = rt.mgpuMemAlloc(capacity_bytes, None, False)
    if not address:
        raise RuntimeError(f"ktrace: could not allocate a {capacity_bytes}-byte trace buffer")
    device_ptr = ctypes.c_void_p(address)
    _check(rt.mgpuTraceBufferClear(device_ptr, capacity_bytes), "clearing the trace buffer")

    # The kernel's slot counter lives in the buffer's first record slot, so the host reads
    # exactly the memory the kernel incremented.
    return TraceBuffer(
        device_ptr=device_ptr.value or 0,
        capacity_slots=capacity_slots - 1,  # slot 0 is the cursor
        device_id=_current_device(),
    )


def dump_dir() -> str:
    from ...utils import env

    return env.ktrace.dump_dir or os.getcwd()
