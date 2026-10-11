# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Process-wide ROCm trace-buffer state.

The device record stores are emitted by ``convert-fly-ktrace-to-rocdl``. Shared
event names and payload coercion live in ``ktrace_common``; the old helper names
below forward there for callers using this internal module.
"""

from __future__ import annotations

import threading

from ..experimental.ktrace_common import (
    merge_names as merge_names,
)
from ..experimental.ktrace_common import (
    name_table as name_table,
)
from ..experimental.ktrace_common import (
    payload_value,
)
from . import ktrace_format as _format


def _payload_value(payload):
    """Compatibility alias for callers using the former emitter helper."""
    return payload_value(payload)


def _buffer_capacity_slots() -> int:
    """Record slots the buffer holds, excluding the reserved cursor slot."""
    from ...utils import env

    return max(int(env.ktrace.buffer_bytes) // _format.RECORD_BYTES - 1, 1)


_BUFFER = None

# Guards the allocation below. Two threads reaching a first traced launch together would
# otherwise both see _BUFFER as None and both hipMalloc: the loser's buffer leaks for the
# life of the process, and the records its kernels wrote land in memory nothing reads.
# Same shape, and the same remedy, as the lazy engine init in compiler/jit_executor.py.
_BUFFER_LOCK = threading.Lock()


def ensure_buffer():
    """Allocate the process-wide trace buffer on first use.

    Checks the device on every call, not just on allocation, because a caller that
    reaches here after switching devices would otherwise be handed an address the
    current device does not own.

    This catches the paths that build a CallState. It does NOT cover dispatch from an
    already-built one -- the pointer is frozen into its ABI storage and never read
    from here again -- which is what :func:`require_dispatchable_buffer` is for.
    """
    global _BUFFER
    # Fast path: once allocated, the check needs no lock -- _BUFFER is only ever assigned
    # once, and the device check reads the buffer rather than mutating it.
    if _BUFFER is not None:
        _BUFFER._require_own_device()
        return _BUFFER

    with _BUFFER_LOCK:
        # Re-check under the lock: another thread may have allocated while this one waited.
        if _BUFFER is None:
            from ...runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

            _BUFFER = ktrace_buffer.allocate()
        else:
            _BUFFER._require_own_device()
    return _BUFFER


def require_dispatchable_buffer():
    """Check the buffer still belongs to this device, before a traced launch.

    Runs on every dispatch, not just where the pointer is first read: the address is
    frozen into the CallState's ABI storage, and a cached CallState or a
    CompiledFunction reuses it without ever calling ensure_buffer() again. Switching
    devices in between would hand the kernel an address this device does not own.
    """
    if _BUFFER is not None:
        _BUFFER._require_own_device()


def current_buffer():
    """The trace buffer, or None if nothing has been traced yet."""
    return _BUFFER
