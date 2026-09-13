# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host side of in-kernel event tracing: allocate the buffer, bind it, read it back.

The device writes records into a buffer whose address is passed to instrumented
kernels as a trailing implicit argument (see :mod:`flydsl.expr.iket_emit` and
``compiler/kernel_function.py``).  Nothing here resolves a device symbol, which is
what keeps traced kernels disk-cacheable: registering a ``post_load_processors``
callback would set ``extern_linked`` and disable the disk cache.

Allocation goes through ``ctypes`` on ``libamdhip64.so`` -- already a ``DT_NEEDED``
of ``libfly_jit_runtime.so``, so it is loaded in-process before any iket code runs.
The dependency is confined to this module.
"""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass

from ..expr import iket as _iket

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
                "iket needs libamdhip64.so to allocate its device trace buffer; "
                "it is normally already loaded as a dependency of libfly_jit_runtime.so"
            ) from exc
        for name, argtypes in (
            ("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
            ("hipFree", [ctypes.c_void_p]),
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
_MEMCPY_H2D = 1


@dataclass
class TraceBuffer:
    """A device trace buffer bound to one loaded module."""

    device_ptr: int
    capacity_slots: int
    cursor_ptr: int
    bufptr_ptr: int

    def reset(self) -> None:
        """Zero the cursor so the next launch starts at slot 0."""
        zero = ctypes.c_uint32(0)
        _check(
            _hip_lib().hipMemcpy(ctypes.c_void_p(self.cursor_ptr), ctypes.byref(zero), 4, _MEMCPY_H2D),
            "hipMemcpy(cursor=0)",
        )

    def claimed_slots(self) -> int:
        """Slots waves *claimed*, which may exceed capacity -- that is the overflow signal."""
        out = ctypes.c_uint32(0)
        _check(
            _hip_lib().hipMemcpy(ctypes.byref(out), ctypes.c_void_p(self.cursor_ptr), 4, _MEMCPY_D2H),
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
            needed = claimed * _iket.RECORD_BYTES
            raise RuntimeError(
                f"iket trace overflowed: waves claimed {claimed} slots, buffer holds "
                f"{self.capacity_slots}. Re-run with FLYDSL_IKET_BUFFER_BYTES>={needed}, "
                f"a block filter (FLYDSL_IKET_BLOCKS), or fewer event sites."
            )

        nbytes = claimed * _iket.RECORD_BYTES
        if nbytes == 0:
            return []
        raw = (ctypes.c_char * nbytes)()
        _check(
            _hip_lib().hipMemcpy(raw, ctypes.c_void_p(self.device_ptr), nbytes, _MEMCPY_D2H),
            "hipMemcpy(records)",
        )
        return decode_records(bytes(raw))

    def free(self) -> None:
        if self.device_ptr:
            _hip_lib().hipFree(ctypes.c_void_p(self.device_ptr))
            self.device_ptr = 0


def decode_records(raw: bytes) -> list[dict]:
    """Decode packed 32-byte records.

    Slots never written keep the buffer's zero fill; those decode to kind 0 with a zero
    timestamp and are dropped, which is how a wave that used fewer events than it
    reserved leaves no trace of the unused tail.
    """
    out = []
    for off in range(0, len(raw) - _iket.RECORD_BYTES + 1, _iket.RECORD_BYTES):
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
                "slot": off // _iket.RECORD_BYTES,
                # HW_ID fields, per AMD's shipped HIP header. SE_ID is 2 bits on
                # gfx942/gfx950, unlike the 3 bits used on gfx908/gfx90a.
                "wave": hw_id & 0xF,
                "simd": (hw_id >> 4) & 0x3,
                "cu": (hw_id >> 8) & 0xF,
                "sh": (hw_id >> 12) & 0x1,
                "se": (hw_id >> 13) & 0x3,
            }
        )
    return out


def buffer_bytes() -> int:
    from . import env

    return int(env.iket.buffer_bytes)


def allocate(capacity_bytes: int | None = None) -> TraceBuffer:
    """Allocate a trace buffer for the current process.

    The buffer address is passed to instrumented kernels as a trailing argument, so
    nothing here needs a loaded module or a device-symbol lookup -- which is what keeps
    traced kernels disk-cacheable.
    """
    hip = _hip_lib()
    capacity_bytes = capacity_bytes if capacity_bytes is not None else buffer_bytes()
    capacity_slots = capacity_bytes // _iket.RECORD_BYTES
    if capacity_slots == 0:
        raise ValueError(f"FLYDSL_IKET_BUFFER_BYTES={capacity_bytes} is smaller than one record")

    device_ptr = ctypes.c_void_p()
    _check(hip.hipMalloc(ctypes.byref(device_ptr), capacity_bytes), "hipMalloc(trace buffer)")
    _check(hip.hipMemset(device_ptr, 0, capacity_bytes), "hipMemset(trace buffer)")

    cursor = ctypes.c_void_p()
    _check(hip.hipMalloc(ctypes.byref(cursor), 4), "hipMalloc(trace cursor)")
    _check(hip.hipMemset(cursor, 0, 4), "hipMemset(trace cursor)")

    buf = TraceBuffer(
        device_ptr=device_ptr.value or 0,
        capacity_slots=capacity_slots,
        cursor_ptr=cursor.value or 0,
        bufptr_ptr=0,
    )
    return buf


def dump_dir() -> str:
    from . import env

    return env.iket.dump_dir or os.getcwd()
