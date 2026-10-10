# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x asymmetric W4A8 int8 linear.

Quantize weights to grouped int4 + channel scales, then run ``w4a8_int8_linear``.

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.quant.rdna4_asym_w4a8 import quantize_w4a8_int8_weight, w4a8_int8_linear

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k = 32, 64, 256
group_size, convrot_g = 16, 64
torch.manual_seed(18)
x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
packed, s_rel, s_channel, correction, cb = quantize_w4a8_int8_weight(
    w, group_size=group_size, convrot_groupsize=convrot_g, codebook=True, scale_dtype=torch.float32
)
assert correction is None
y = w4a8_int8_linear(
    x,
    packed,
    s_rel,
    s_channel,
    codebook=cb,
    group_size=group_size,
    convrot_groupsize=convrot_g,
    out_dtype=torch.bfloat16,
)
torch.cuda.synchronize()
ok = tuple(y.shape) == (m, n) and y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all())
print(f"arch={_arch} w4a8_int8_linear", tuple(y.shape), y.dtype, "correct:", ok)
if not ok:
    sys.exit(1)
