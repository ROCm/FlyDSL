# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x FlashAttention host call.

Wave64 / CDNA attention stays on ``kernels/attention/flash_attn_interface.py``
once ``is_gfx120x`` is false. This file calls the gfx120x host directly so a
port can see the argument order without the router.

Allowlist: ``tests/arch_compat.py`` ``("gfx120*",)``.
Call graph: ``docs/gfx120x_call_graph.md``.
"""

import sys

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func

_arch = (get_rocm_arch() or "").lower().split(":")[0]
if not _arch.startswith("gfx120"):
    print(f"SKIP examples/08-flash_attn_gfx120x.py: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

# BSHD. D must be in [64, 384] and a multiple of 32. One head, no GQA.
q = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16)
out = flydsl_flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
print(f"arch={_arch} flash_attn out", tuple(out.shape), out.dtype)
