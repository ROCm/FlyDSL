# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x AWQ W4A16 fused GEMV.

Unsigned nibble pack + group scales/zeros. This is not native iu4 and not
ConvRot (see ``examples/11`` / ``examples/14``).

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.quant.rdna4_awq_w4a16 import gemv_awq_w4a16

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)


def _pack_uint4_row_major(values: torch.Tensor) -> torch.Tensor:
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


m, n, k, g = 4, 32, 128, 64
dtype = torch.bfloat16
torch.manual_seed(16)
codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
qweight = _pack_uint4_row_major(codes)
wscales = (torch.randn((k // g, n), device="cuda", dtype=dtype) * 0.05).abs() + 1e-3
wzeros = torch.randn((k // g, n), device="cuda", dtype=dtype) * 0.01
x = torch.randn((m, k), device="cuda", dtype=dtype)
y = gemv_awq_w4a16(x, qweight, wscales, wzeros, bias=None, group_size=g)
torch.cuda.synchronize()
# AWQ dequant: (code - 8) * scale + zero; oracle lives in tests.
codes_f = codes.to(dtype) - 8
w = (codes_f.view(n, k // g, g) * wscales.T.unsqueeze(-1) + wzeros.T.unsqueeze(-1)).reshape(n, k)
ref = x @ w.T
ok = torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=2e-2)
print(f"arch={_arch} gemv_awq_w4a16", tuple(y.shape), y.dtype, "correct:", ok)
if not ok:
    print("Max diff:", (y.float() - ref.float()).abs().max().item())
    sys.exit(1)
