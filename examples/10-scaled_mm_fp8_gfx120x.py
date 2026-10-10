# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x FP8 scaled matmul (fixed product entry).

``scaled_mm_fp8`` launches ``build_scaled_mm_fp8_module``. A K that is not a
multiple of 16 is zero-filled in the kernel. This demo uses K=64. Pass
``tile=`` to pin a launch config; otherwise the host uses ``pick_tile_config``.

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.gemm.rdna4_scaled_mm_fp8 import scaled_mm_fp8

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k = 32, 64, 64
a_f = torch.randn(m, k, device="cuda", dtype=torch.float32)
b_f = torch.randn(n, k, device="cuda", dtype=torch.float32)
a = a_f.to(torch.float8_e4m3fn)
b = b_f.to(torch.float8_e4m3fn)
scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
y = scaled_mm_fp8(a, b, scale_a, scale_b)
torch.cuda.synchronize()
ref = (a.float() * float(scale_a) @ (b.float() * float(scale_b)).T).to(y.dtype)
ok = torch.allclose(y.float(), ref.float(), atol=5e-1, rtol=5e-2)
print(f"arch={_arch} scaled_mm_fp8", tuple(y.shape), y.dtype, "correct:", ok)
if not ok:
    print("Max diff:", (y.float() - ref.float()).abs().max().item())
    sys.exit(1)
