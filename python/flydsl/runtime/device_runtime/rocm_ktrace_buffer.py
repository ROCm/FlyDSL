# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass

from ...expr.rocdl import ktrace_format as _format

_HIP_SUCCESS = 0
_RT = None


def _runtime():
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
    from . import get_device_runtime

    return get_device_runtime().current_device_id()


@dataclass
class TraceBuffer:

    device_ptr: int
    capacity_slots: int
    device_id: int = -1

    @property
    def layout(self) -> _format.Layout:
        return _format.layout(self.capacity_slots)

    def synchronize(self) -> None:
        self.launch_count()

    def reset(self) -> None:
        self._require_own_device()
        _check(
            _runtime().mgpuTraceBufferClear(
                ctypes.c_void_p(self.device_ptr), (self.capacity_slots + 1) * _format.RECORD_BYTES
            ),
            "clearing the trace buffer",
        )

    def _require_own_device(self) -> None:
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

    def _read(self, offset: int, nbytes: int, what: str) -> bytes:
        self._require_own_device()
        raw = (ctypes.c_char * nbytes)()
        _check(_runtime().mgpuTraceBufferRead(raw, ctypes.c_void_p(self.device_ptr + offset), nbytes), what)
        return bytes(raw)

    def claimed_waves(self) -> list[int]:
        """Waves each partition has numbered since the last reset; past its regions on overflow."""
        layout = self.layout
        raw = self._read(0, layout.header_slots * _format.RECORD_BYTES, "reading the trace counters")
        return [struct.unpack_from("<I", raw, layout.cursor_byte(p))[0] for p in range(layout.partitions)]

    def launch_count(self) -> int:
        raw = self._read(_format.LAUNCH_COUNT_BYTE, 4, "reading the trace launch count")
        return struct.unpack("<I", raw)[0]

    def read_records(self) -> list[dict]:
        layout = self.layout
        claimed = self.claimed_waves()
        excess, partition = max((waves - layout.capacity(p), p) for p, waves in enumerate(claimed))
        busiest = max(claimed)
        if excess > 0:
            raise RuntimeError(
                f"trace overflowed: one of the buffer's {layout.partitions} partitions numbered "
                f"{claimed[partition]} waves but holds {layout.capacity(partition)}. Re-run with "
                f"FLYDSL_KTRACE_BUFFER_BYTES>={_bytes_holding(busiest, layout)}, a block filter "
                "(capture(blocks=...) or FLYDSL_KTRACE_BLOCKS), or fewer launches between reads."
            )

        # The regions in use are a prefix: partition p's last wave sits in region (waves - 1) * partitions + p.
        regions = max(((waves - 1) * layout.partitions + p + 1 for p, waves in enumerate(claimed) if waves), default=0)
        if not regions:
            return []
        nslots = min(regions * _format.EVENTS_PER_WAVE, layout.wave_slots)
        raw = self._read(
            layout.header_slots * _format.RECORD_BYTES, nslots * _format.RECORD_BYTES, "reading the trace records"
        )
        records, overflowed = decode_records(raw, first_slot=layout.header_slots)
        if overflowed:
            worst = max(attempted for _, attempted, _ in overflowed)
            if all(holds == _format.EVENTS_PER_WAVE - 1 for _, _, holds in overflowed):
                cause = (
                    f"emitted more than {_format.EVENTS_PER_WAVE - 1} events (worst: {worst}). The records past "
                    "that point were dropped, so ranges are unbalanced and phases would be "
                    "mispaired -- a plausible-looking timeline for the wrong phase. Annotate "
                    "at a coarser granularity or sample fewer loop iterations. Increasing "
                    "FLYDSL_KTRACE_BUFFER_BYTES or filtering other blocks does not increase "
                    "the per-wave event limit."
                )
            else:
                cause = (
                    f"ran out of room at the end of a trace partition (worst: {worst} events). The records past "
                    "that point were dropped, so ranges are unbalanced. Re-run with "
                    f"FLYDSL_KTRACE_BUFFER_BYTES>={_bytes_holding(busiest, layout)} or a block filter."
                )
            raise RuntimeError(f"trace truncated: {len(overflowed)} wave(s) {cause}")
        return records


def _decode_payload(bits: int, packed: int):
    if not packed & (1 << 31):
        return None
    encoding = (packed >> 27) & 0x7
    if encoding == 0:
        return bits
    if encoding == 1:
        return bits - (1 << 64) if bits & (1 << 63) else bits
    if encoding == 2:
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
    if encoding == 3:
        return struct.unpack("<d", struct.pack("<Q", bits))[0]
    if encoding == 4:
        return bool(bits)
    raise ValueError(f"ktrace: unknown payload type {encoding}")


def _bytes_holding(waves: int, layout: _format.Layout) -> int:
    """A buffer size whose every partition holds ``waves`` waves. Partitions only split as the buffer
    grows, so no partition of the larger buffer is busier than the busiest one now."""
    nbytes = (layout.header_slots + waves * layout.partitions * _format.EVENTS_PER_WAVE) * _format.RECORD_BYTES
    while True:
        grown = _format.layout(nbytes // _format.RECORD_BYTES - 1)
        if grown.capacity(grown.partitions - 1) >= waves:
            return nbytes
        nbytes *= 2


def decode_records(raw: bytes, first_slot: int = 1) -> tuple[list[dict], list[tuple[int, int, int]]]:
    """Decode consecutive wave regions whose first header is at slot ``first_slot``. Also returns
    (header slot, events issued, events the region holds) for every wave that issued more events
    than its region holds."""
    per_wave = _format.EVENTS_PER_WAVE
    total = len(raw) // _format.RECORD_BYTES
    out = []
    overflowed = []
    dispatch_id = 0
    queue_id = 0
    block = (0, 0, 0)
    wave_id = 0
    for index in range(total):
        off = index * _format.RECORD_BYTES
        abs_slot = first_slot + index
        if index % per_wave == 0:
            attempted, wave_word, dispatch_id, queue_id, block_x, block_y = struct.unpack_from("<IIQQII", raw, off)
            block = (block_x, block_y, wave_word >> 8)
            wave_id = wave_word & 0xFF
            holds = min(per_wave, total - index) - 1
            if attempted > holds:
                overflowed.append((abs_slot, attempted, holds))
            continue
        ts, payload, hw_id, xcc_id, packed, start_slot = struct.unpack_from("<QQIIII", raw, off)
        if ts == 0:
            continue
        out.append(
            {
                "ts": ts,
                "payload": _decode_payload(payload, packed),
                "payload_type": _format.PAYLOAD_TYPES[(packed >> 27) & 0x7] if packed & (1 << 31) else None,
                "hw_id": hw_id,
                "xcc_id": xcc_id,
                "dispatch_id": dispatch_id,
                "queue_id": queue_id,
                "block": block,
                "wave_id": wave_id,
                "event_id": packed & 0xFFFFFF,
                "kind": (packed >> 24) & 0x7,
                "start_slot": start_slot,
                "slot": abs_slot,
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
    capacity_bytes = capacity_bytes if capacity_bytes is not None else buffer_bytes()
    capacity_slots = capacity_bytes // _format.RECORD_BYTES
    if capacity_slots == 0:
        raise ValueError(f"FLYDSL_KTRACE_BUFFER_BYTES={capacity_bytes} is smaller than one record")

    rt = _runtime()
    address = rt.mgpuMemAlloc(capacity_bytes, None, False)
    if not address:
        raise RuntimeError(f"ktrace: could not allocate a {capacity_bytes}-byte trace buffer")
    device_ptr = ctypes.c_void_p(address)
    _check(rt.mgpuTraceBufferClear(device_ptr, capacity_bytes), "clearing the trace buffer")

    return TraceBuffer(
        device_ptr=device_ptr.value or 0,
        capacity_slots=capacity_slots - 1,
        device_id=_current_device(),
    )


def dump_dir() -> str:
    from ...utils import env

    return env.ktrace.dump_dir or os.getcwd()
