# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x ConvRot W4A4 linear.

Default ``linear_dtype`` is ``"int4"`` (native iu4). Pass ``"int8"`` to unpack
to iu8. This is not the AWQ W4A16 path.

Allowlist: ``("gfx120*",)``.
"""

import sys

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.quant.rdna4_convrot_w4a4 import convrot_w4a4_linear, quantize_convrot_w4a4_weight

_arch = (get_rocm_arch() or "").lower().split(":")[0]
if not _arch.startswith("gfx120"):
    print(f"SKIP examples/11-convrot_w4a4_gfx120x.py: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k = 16, 64, 64
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
qweight, wscales = quantize_convrot_w4a4_weight(w, convrot_groupsize=64)
y = convrot_w4a4_linear(x, qweight, wscales, convrot_groupsize=64)
torch.cuda.synchronize()
print(f"arch={_arch} convrot_w4a4_linear", tuple(y.shape), y.dtype)
