# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Wave-level GPU phase tracing with automatic static/dynamic recording.

Annotate uniform kernel code with mark(), boundary()/end(), or push()/pop().
Wrap the unchanged host launch in capture() to allocate and collect a trace.
In @jit, with configure(block=(x, y, z)) scopes block selection to enclosed launches.
Without capture, annotations disappear and kernel signatures remain unchanged.

``capture(mode="auto")`` keeps a compact one-word-per-event fast path for
statically reconstructible control flow, and otherwise selects a bounded
dynamic recorder. The dynamic path supports runtime loops, branches, payloads,
runtime launch dimensions, selected block lists, and all-grid capture bounded by
``max_blocks``. Both paths support gfx942/gfx950 and user-provided streams.

Export with ``cap.export("trace.json")`` for Perfetto. Dynamic recorder overflow
is non-fatal: export reports dropped records and adds a warning event. Tune the
per-wave capacity with ``max_events``.

For simultaneous rocprofv3 ATT, capture(hardware=True).save(path) preserves
absolute clocks and hardware identities; merge_att() produces a combined
Perfetto trace after rocprofv3 has decoded the dispatch. ATT identity merging is
currently limited to gfx942.
"""

from ._flytrace import boundary, capture, configure, end, mark, pop, push
from ._flytrace_att import merge_att

__all__ = ["capture", "configure", "mark", "boundary", "end", "push", "pop", "merge_att"]
