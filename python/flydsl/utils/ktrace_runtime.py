# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host side of in-kernel wave tracing: allocate the buffer, bind it, read it back.

The device writes records into a buffer whose address is passed to instrumented
kernels as a trailing implicit argument (added by ``compiler/kernel_function.py``;
the buffer itself is owned by :mod:`flydsl.expr.rocdl.ktrace_emit`, and the stores
are emitted by ``convert-fly-ktrace-to-rocdl``).  Nothing here resolves a device
symbol, which is what keeps traced kernels disk-cacheable: registering a
``post_load_processors`` callback would set ``extern_linked`` and disable the cache.

Allocation goes through ``ctypes`` on ``libamdhip64.so`` -- already a ``DT_NEEDED``
of ``libfly_jit_runtime.so``, so it is loaded in-process before any ktrace code runs.
The dependency is confined to this module.
"""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass

from ..expr import ktrace as _ktrace

_HIP_SUCCESS = 0
_hip = None


def _hip_lib():
    """Handle on libamdhip64, loaded lazily so importing flydsl never needs HIP."""
    global _hip
    if _hip is None:
        try:
            _hip = ctypes.CDLL("libamdhip64.so")
        except OSError as exc:  # pragma: no cover - depends on the ROCm install
            raise RuntimeError(
                "ktrace needs libamdhip64.so to allocate its device trace buffer; "
                "it is normally already loaded as a dependency of libfly_jit_runtime.so"
            ) from exc
        for name, argtypes in (
            ("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
            ("hipMemset", [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]),
            ("hipDeviceSynchronize", []),
        ):
            fn = getattr(_hip, name)
            fn.restype = ctypes.c_int
            fn.argtypes = argtypes
        _hip.hipMemcpy.restype = ctypes.c_int
        _hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return _hip


def _check(code: int, what: str) -> None:
    if code != _HIP_SUCCESS:
        raise RuntimeError(f"{what} failed with HIP error {code}")


_MEMCPY_D2H = 2


@dataclass
class TraceBuffer:
    """A device trace buffer bound to one loaded module."""

    device_ptr: int
    capacity_slots: int

    def reset(self) -> None:
        """Zero the whole buffer so the next launch starts from an empty trace.

        The cursor alone is not enough.  Records are dropped by :func:`decode_records`
        on a zero timestamp, so slots left behind by a longer previous launch would be
        re-read as records of the next one once the cursor rewinds past them.  Each
        wave's counter word has to go back to zero for the same reason.

        Called by :func:`flydsl.expr.ktrace.collect` and
        :func:`flydsl.expr.ktrace.summary` once the records have been read back, so a
        process that traces repeatedly -- a benchmark loop, an autotune sweep -- does
        not accumulate claims until the buffer overflows.
        """
        _check(
            _hip_lib().hipMemset(ctypes.c_void_p(self.device_ptr), 0, (self.capacity_slots + 1) * _ktrace.RECORD_BYTES),
            "hipMemset(trace buffer)",
        )

    def synchronize(self) -> None:
        """Wait for every in-flight kernel before touching the buffer.

        A blocking ``hipMemcpy`` orders against the NULL stream only.  FlyDSL kernels
        routinely launch on a user-supplied ``fx.Stream``, and that work is *not* ordered
        against a null-stream copy: the readback would see a mid-flight cursor and copy
        slots the kernel had not written yet, producing a short trace with unbalanced
        push/pop.  Worse, :func:`reset` would memset the buffer while the kernel was still
        storing into it.
        """
        _check(_hip_lib().hipDeviceSynchronize(), "hipDeviceSynchronize(before trace readback)")

    def claimed_slots(self) -> int:
        """Slots waves *reserved*, not events written.

        Every wave that reaches the prologue claims ``ktrace.EVENTS_PER_WAVE`` slots
        up front, whether or not it goes on to record, so this is a high-water mark: 64
        waves at the default 256 reads 16384 even if only two events were emitted. It is
        the right quantity for the overflow check (it measures what the run *wanted*), and
        :func:`decode_records` drops the untouched slots.
        """
        self.synchronize()
        out = ctypes.c_uint32(0)
        _check(
            _hip_lib().hipMemcpy(ctypes.byref(out), ctypes.c_void_p(self.device_ptr), 4, _MEMCPY_D2H),
            "hipMemcpy(cursor)",
        )
        return int(out.value)

    def read_records(self) -> list[dict]:
        """Copy back and decode the records written by the last launch.

        Raises when waves claimed more slots than the buffer holds.  The trace is
        rejected rather than trimmed: a partial trace has unbalanced push/pop and
        renders as a corrupt timeline, which is worse than no trace at all.
        """
        claimed = self.claimed_slots()
        if claimed > self.capacity_slots:
            needed = claimed * _ktrace.RECORD_BYTES
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
            _hip_lib().hipMemcpy(raw, ctypes.c_void_p(self.device_ptr + _ktrace.RECORD_BYTES), nbytes, _MEMCPY_D2H),
            "hipMemcpy(records)",
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
    from . import env

    return int(env.ktrace.buffer_bytes)


def allocate(capacity_bytes: int | None = None) -> TraceBuffer:
    """Allocate a trace buffer for the current process.

    The buffer address is passed to instrumented kernels as a trailing argument, so
    nothing here needs a loaded module or a device-symbol lookup -- which is what keeps
    traced kernels disk-cacheable.
    """
    hip = _hip_lib()
    capacity_bytes = capacity_bytes if capacity_bytes is not None else buffer_bytes()
    capacity_slots = capacity_bytes // _ktrace.RECORD_BYTES
    if capacity_slots == 0:
        raise ValueError(f"FLYDSL_KTRACE_BUFFER_BYTES={capacity_bytes} is smaller than one record")

    device_ptr = ctypes.c_void_p()
    _check(hip.hipMalloc(ctypes.byref(device_ptr), capacity_bytes), "hipMalloc(trace buffer)")
    _check(hip.hipMemset(device_ptr, 0, capacity_bytes), "hipMemset(trace buffer)")

    # The kernel's slot counter lives in the buffer's first record slot, so the host reads
    # exactly the memory the kernel incremented.
    return TraceBuffer(
        device_ptr=device_ptr.value or 0,
        capacity_slots=capacity_slots - 1,  # slot 0 is the cursor
    )


def dump_dir() -> str:
    from . import env

    return env.ktrace.dump_dir or os.getcwd()
