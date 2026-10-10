# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x SVDQuant W4A4 linear host.

Signed int4 pack + LoRA residual. Separate family from AWQ and native iu4.

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.quant.rdna4_int4_codec import pack_int4_row_major
from kernels.quant.rdna4_svdquant_w4a4 import svdquant_w4a4_linear

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k, r, g = 4, 64, 256, 8, 64
dtype = torch.bfloat16
torch.manual_seed(17)
codes = torch.randint(-7, 8, (n, k), device="cuda", dtype=torch.int32)
qweight = pack_int4_row_major(codes)
wscales = (torch.randn((k // g, n), device="cuda", dtype=dtype) * 0.05).abs() + 1e-3
x = torch.randn((m, k), device="cuda", dtype=dtype)
smooth = (torch.randn(k, device="cuda", dtype=dtype).abs() * 0.3) + 0.2
proj_down = torch.randn((k, r), device="cuda", dtype=dtype) * 0.01
proj_up = torch.randn((n, r), device="cuda", dtype=dtype) * 0.01
y = svdquant_w4a4_linear(x, qweight, wscales, proj_down, proj_up, smooth, bias=None, pad_size=16, group_size=g)
torch.cuda.synchronize()
ok = tuple(y.shape) == (m, n) and y.dtype == dtype and bool(torch.isfinite(y).all())
print(f"arch={_arch} svdquant_w4a4_linear", tuple(y.shape), y.dtype, "correct:", ok)
if not ok:
    sys.exit(1)
