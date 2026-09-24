# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Wave timestamps with compile-time layout and automatic capture.

Annotate uniform kernel code with mark(), boundary()/end(), or push()/pop().
Wrap the unchanged host launch in capture() to allocate and collect a trace.
In @jit, with configure(block=(x, y, z)) scopes CTA selection to enclosed launches.
Without capture, annotations disappear and kernel signatures remain unchanged.

For simultaneous rocprofv3 ATT, capture(hardware=True).save(path) preserves
absolute clocks and hardware identities; merge_att() produces a combined
Perfetto trace after rocprofv3 has decoded the dispatch.

The experimental gfx942 backend requires static launch dimensions and
constant-bound, positive-step loops; see capture for runtime restrictions.
"""

from ._flytrace import boundary, capture, configure, end, mark, pop, push
from ._flytrace_att import merge_att

__all__ = ["capture", "configure", "mark", "boundary", "end", "push", "pop", "merge_att"]
