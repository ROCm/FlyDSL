# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x in-register SwiGLU MLP.

``K`` and the feed-forward width are both 16, so this calls
``fused_swiglu_mlp_inreg`` and the mid stays in registers.
``fused_swiglu_mlp_nmajor`` is the other host. It always writes the mid
to memory, including at 16×16.

Allowlist: ``("gfx120*",)``.
"""

import sys

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.rdna4_fused_mlp_nmajor import fused_swiglu_mlp_inreg

_arch = (get_rocm_arch() or "").lower().split(":")[0]
if not _arch.startswith("gfx120"):
    print(f"SKIP examples/15-fused_mlp_gfx120x.py: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, k, ffn = 16, 16, 16
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
w_gate = torch.randn(ffn, k, device="cuda", dtype=torch.bfloat16)
w_up = torch.randn(ffn, k, device="cuda", dtype=torch.bfloat16)
w_down = torch.randn(k, ffn, device="cuda", dtype=torch.bfloat16)
y = fused_swiglu_mlp_inreg(x, w_gate, w_up, w_down)
torch.cuda.synchronize()
print(f"arch={_arch} fused_swiglu_mlp_inreg", tuple(y.shape), y.dtype)
