# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x int8-weight linear: W8A16 and iu8 (distinct dtype contracts).

``w8a16_linear`` keeps activations bf16/fp16 and casts int8 weights in-register
(bf16×int8). ``int8_linear`` runs int8×int8 iu8 WMMA after
``quantize_int8_rowwise`` (highest-TOPS path on large-K production shapes). Call
each explicitly — there is no size/K auto router between them.

A K that is not a multiple of 16 is zero-filled in the kernel. This demo uses
K=64, which is already a tile.

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

from kernels.gemm.rdna4_int8_linear import int8_linear  # noqa: E402
from kernels.gemm.rdna4_w8a16_linear import w8a16_linear  # noqa: E402
from kernels.quant.rdna4_quantize_int8_rowwise import quantize_int8_rowwise  # noqa: E402

m, n, k = 32, 64, 64
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
weight_scale = torch.rand(n, device="cuda", dtype=torch.float32).abs() + 0.01

# W8A16: bf16 acts × int8 weights (float WMMA).
y_w8 = w8a16_linear(x, weight, weight_scale)
torch.cuda.synchronize()
ref_w8 = (x.float() @ (weight.float() * weight_scale.reshape(-1, 1)).T).to(torch.bfloat16)
ok_w8 = torch.allclose(y_w8.float(), ref_w8.float(), atol=2e-2, rtol=2e-2)
print(f"arch={_arch} w8a16_linear", tuple(y_w8.shape), y_w8.dtype, "correct:", ok_w8)

# iu8: quantize acts then int8×int8 WMMA. A short K stays in the kernel.
a8, x_scale = quantize_int8_rowwise(x)
y_iu8 = int8_linear(a8, weight, x_scale.reshape(-1), weight_scale, out_dtype=torch.bfloat16)
torch.cuda.synchronize()
ref_iu8 = ((a8.float() * x_scale.reshape(-1, 1).float()) @ (weight.float() * weight_scale.reshape(-1, 1)).T).to(
    torch.bfloat16
)
ok_iu8 = torch.allclose(y_iu8.float(), ref_iu8.float(), atol=5e-2, rtol=5e-2)
print(f"arch={_arch} int8_linear", tuple(y_iu8.shape), y_iu8.dtype, "correct:", ok_iu8)

if not (ok_w8 and ok_iu8):
    if not ok_w8:
        print("W8A16 max diff:", (y_w8.float() - ref_w8.float()).abs().max().item())
    if not ok_iu8:
        print("iu8 max diff:", (y_iu8.float() - ref_iu8.float()).abs().max().item())
    sys.exit(1)
