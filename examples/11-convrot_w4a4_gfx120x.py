# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x ConvRot W4A4 linear.

Default ``linear_dtype`` is ``"int4"``: native ``iu4_gemm``. A K that is not a
Hadamard multiple is zero-filled in the kernel. The packed width is that
multiple. Pass ``linear_dtype="int8"`` to unpack to iu8. This demo uses K=64.
This is not the AWQ W4A16 path.

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.quant.rdna4_convrot_w4a4 import convrot_w4a4_linear, quantize_convrot_w4a4_weight

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k = 16, 64, 64
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
qweight, wscales = quantize_convrot_w4a4_weight(w, convrot_groupsize=64)
y = convrot_w4a4_linear(x, qweight, wscales, convrot_groupsize=64)
torch.cuda.synchronize()
# Shape/dtype contract + finite check (full ConvRot oracle is in tests).
ok = tuple(y.shape) == (m, n) and y.dtype == x.dtype and bool(torch.isfinite(y).all())
print(f"arch={_arch} convrot_w4a4_linear", tuple(y.shape), y.dtype, "correct:", ok)
if not ok:
    sys.exit(1)
