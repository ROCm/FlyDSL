# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x FP8 scaled matmul.

``scaled_mm_fp8_auto`` picks a tile and calls ``build_scaled_mm_fp8_module``.
There is no ``scaled_mm_fp8()`` function. The fixed-tile path is the builder.

Allowlist: ``("gfx120*",)``.
"""

import sys

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.rdna4_scaled_mm_fp8 import build_scaled_mm_fp8_module
from kernels.gemm.rdna4_scaled_mm_fp8_auto import scaled_mm_fp8_auto

_arch = (get_rocm_arch() or "").lower().split(":")[0]
if not _arch.startswith("gfx120"):
    print(f"SKIP examples/10-scaled_mm_fp8_gfx120x.py: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k = 32, 64, 64
a = torch.randn(m, k, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
b = torch.randn(n, k, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
y = scaled_mm_fp8_auto(a, b, scale_a, scale_b)
# Fixed-tile path: build_scaled_mm_fp8_module(...) then _run_compiled.
# tests/kernels/test_rdna4_scaled_mm_fp8.py is that launch. Do not call a
# function named scaled_mm_fp8; it does not exist.
del build_scaled_mm_fp8_module
torch.cuda.synchronize()
print(f"arch={_arch} scaled_mm_fp8_auto", tuple(y.shape), y.dtype)
