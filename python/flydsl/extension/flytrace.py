# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Target-neutral in-kernel tracing with automatic static/dynamic recording.

Annotate kernel code with mark(), range_start()/range_end(),
range_push()/range_pop(), or boundary()/end(). Wrap the unchanged host launch
in capture() to allocate and collect a trace. In @jit,
with configure(block=(x, y, z)) scopes block selection to enclosed launches.
Without capture, annotations disappear and kernel signatures remain unchanged.

``capture(mode="auto")`` keeps a compact one-word-per-event fast path for
statically reconstructible control flow, and otherwise selects a bounded
dynamic recorder. The dynamic path supports runtime loops, branches, payloads,
runtime launch dimensions, selected block lists, and all-grid capture bounded by
``max_blocks``. The public layers are backend-independent; the current ROCm
recorder supports gfx942/gfx950 and user-provided streams.

Export with ``cap.export("trace.json")`` for Perfetto. Dynamic recorder overflow
is non-fatal: export reports dropped records and adds a warning event. Tune the
per-wave capacity with ``max_events``.

For simultaneous rocprofv3 ATT, capture(hardware=True).save(path) preserves
absolute clocks and hardware identities; merge_att() produces a combined
Perfetto trace after rocprofv3 has decoded the dispatch. ATT identity merging is
currently limited to gfx942.
"""

from ._flytrace import (
    RangeToken,
    boundary,
    capture,
    configure,
    end,
    mark,
    pop,
    push,
    range_end,
    range_pop,
    range_push,
    range_start,
)
from ._flytrace_backend import TraceBackend
from ._flytrace_backend import register_trace_backend as register_backend


def merge_att(flytrace_path, att_directory, output_path, *, max_blocks=0):
    """Merge a compatible hardware-instruction trace with raw flytrace data."""

    from ._flytrace_att import merge_att as implementation

    return implementation(flytrace_path, att_directory, output_path, max_blocks=max_blocks)

__all__ = [
    "RangeToken",
    "TraceBackend",
    "boundary",
    "capture",
    "configure",
    "end",
    "mark",
    "merge_att",
    "pop",
    "push",
    "range_end",
    "range_pop",
    "range_push",
    "range_start",
    "register_backend",
]
