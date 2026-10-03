# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x native iu4 GEMM.

Pack logical int4 rows, then call ``iu4_gemm``. This is not AWQ and not the
one-tile WMMA demo in ``examples/07``.

Allowlist: ``("gfx120*",)``.
"""

import sys

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.rdna4_iu4_gemm import iu4_gemm
from kernels.quant.rdna4_int4_codec import pack_int4_row_major

_arch = (get_rocm_arch() or "").lower().split(":")[0]
if not _arch.startswith("gfx120"):
    print(f"SKIP examples/14-iu4_gemm_gfx120x.py: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k = 16, 64, 64
a = torch.randint(-8, 7, (m, k), device="cuda", dtype=torch.int8)
b = torch.randint(-8, 7, (n, k), device="cuda", dtype=torch.int8)
scale_a = torch.ones(m, device="cuda", dtype=torch.float32)
scale_b = torch.ones(n, device="cuda", dtype=torch.float32)
y = iu4_gemm(
    pack_int4_row_major(a),
    pack_int4_row_major(b),
    scale_a,
    scale_b,
    out_dtype=torch.bfloat16,
)
torch.cuda.synchronize()
print(f"arch={_arch} iu4_gemm", tuple(y.shape), y.dtype)
