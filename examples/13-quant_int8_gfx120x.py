# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x rowwise int8 quant.

``quantize_int8_rowwise`` calls ``build_quantize_int8_rowwise_module``.
Tensorwise quant is ``quantize_int8_tensorwise`` in the sibling module.
K must be a multiple of 8 for bf16 (128-bit vector).

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.quant.rdna4_quantize_int8_rowwise import quantize_int8_rowwise

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

torch.manual_seed(7)
x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
q, scale = quantize_int8_rowwise(x)
torch.cuda.synchronize()
amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
ref_scale = (amax / 127.0).reshape_as(scale)
# Kernel uses rocdl.rcp + roundeven; IEEE / + torch.round can differ on ties.
ref_q = torch.round(x.float() / ref_scale.float()).clamp(-128, 127).to(torch.int8)
scale_ok = torch.allclose(scale.float(), ref_scale.float(), atol=1e-5, rtol=1e-4)
mism = int((q != ref_q).sum().item())
frac = mism / q.numel()
ok = scale_ok and frac < 0.02
print(
    f"arch={_arch} quantize_int8_rowwise",
    tuple(q.shape),
    q.dtype,
    tuple(scale.shape),
    f"q_mismatch_frac={frac:.4f}",
    "correct:",
    ok,
)
if not ok:
    sys.exit(1)
