# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x native iu4 GEMM.

Pack logical int4 rows, then call ``iu4_gemm``. A short K is zero-filled.
``prefer_native=False`` unpacks to iu8. This is not AWQ and not ``examples/07``.

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.gemm.rdna4_iu4_gemm import iu4_gemm
from kernels.quant.rdna4_int4_codec import pack_int4_row_major

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
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
ref = (a.float() @ b.float().T).to(torch.bfloat16)
ok = torch.allclose(y.float(), ref.float(), atol=1e-2, rtol=1e-2)
print(f"arch={_arch} iu4_gemm", tuple(y.shape), y.dtype, "correct:", ok)
if not ok:
    print("Max diff:", (y.float() - ref.float()).abs().max().item())
    sys.exit(1)
