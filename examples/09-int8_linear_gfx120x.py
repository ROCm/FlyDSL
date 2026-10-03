# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x default int8 linear.

``int8_linear_auto`` calls ``select_int8_kernel``. K < 256 uses
``w8a16_gemm``. K >= 256 uses the iu8 WMMA path. If ``a_int8`` or
``x_scale`` is omitted, that path calls ``quantize_int8_rowwise``. This
shape stays on W8A16 so the example is one call.

Allowlist: ``("gfx120*",)``.
"""

import sys

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.rdna4_int8_linear_dispatch import int8_linear_auto

_arch = (get_rocm_arch() or "").lower().split(":")[0]
if not _arch.startswith("gfx120"):
    print(f"SKIP examples/09-int8_linear_gfx120x.py: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k = 32, 64, 64
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
weight_scale = torch.rand(n, device="cuda", dtype=torch.float32).abs() + 0.01
y = int8_linear_auto(x, weight, weight_scale)
torch.cuda.synchronize()
print(f"arch={_arch} int8_linear_auto", tuple(y.shape), y.dtype)
