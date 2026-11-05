# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


from __future__ import annotations

import threading

from . import ktrace_format as _format


def _buffer_capacity_slots() -> int:
    from ...utils import env

    return max(int(env.ktrace.buffer_bytes) // _format.RECORD_BYTES - 1, 1)


_BUFFER = None

_BUFFER_LOCK = threading.Lock()


def ensure_buffer():
    global _BUFFER
    if _BUFFER is not None:
        _BUFFER._require_own_device()
        return _BUFFER

    with _BUFFER_LOCK:
        if _BUFFER is None:
            from ...runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

            _BUFFER = ktrace_buffer.allocate()
        else:
            _BUFFER._require_own_device()
    return _BUFFER


def require_dispatchable_buffer():
    if _BUFFER is not None:
        _BUFFER._require_own_device()


def current_buffer():
    return _BUFFER
