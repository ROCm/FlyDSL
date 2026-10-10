# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x MXFP8 block GEMM (software E8M0 + dense fp8 WMMA).

Quantize with ``quantize_mxfp8_device``, then ``mxfp8_block_gemm``. A K that
is not a multiple of 16 zero-fills the tail lanes. Scales stay one per
started group of 32.

gfx120x only.
"""

import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.gemm.rdna4_mxfp8_block_gemm import mxfp8_block_gemm
from kernels.quant.rdna4_mxfp8_e8m0 import dequantize_mxfp8_device, quantize_mxfp8_device

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, n, k = 64, 64, 64
torch.manual_seed(19)
a_f = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
b_f = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
a_q, a_s = quantize_mxfp8_device(a_f)
b_q, b_s = quantize_mxfp8_device(b_f)
y = mxfp8_block_gemm(a_q, b_q, a_s, b_s, out_dtype=torch.bfloat16)
torch.cuda.synchronize()
a_dq = dequantize_mxfp8_device(a_q, a_s)
b_dq = dequantize_mxfp8_device(b_q, b_s)
ref = (a_dq @ b_dq.T).to(torch.bfloat16)
ok = torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=2e-2)
print(f"arch={_arch} mxfp8_block_gemm", tuple(y.shape), y.dtype, "correct:", ok)
if not ok:
    print("Max diff:", (y.float() - ref.float()).abs().max().item())
    sys.exit(1)
