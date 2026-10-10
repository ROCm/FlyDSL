# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x in-register SwiGLU MLP.

``K`` and the feed-forward width are both 16, so this calls
``fused_swiglu_mlp_inreg`` and the mid stays in registers.
``fused_swiglu_mlp_nmajor`` may pick inreg at 16×16 or LDS / staged paths
for larger shapes.

gfx120x only.
"""

import sys

import torch
import torch.nn.functional as F

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.gemm.rdna4_fused_mlp_nmajor import fused_swiglu_mlp_inreg

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

m, k, ffn = 16, 16, 16
# Mild magnitudes — bf16 SiLU×mul + three GEMMs accumulate error (see unit tests).
torch.manual_seed(7)
x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
w_gate = (torch.randn(ffn, k, device="cuda") * 0.5).to(torch.bfloat16)
w_up = (torch.randn(ffn, k, device="cuda") * 0.5).to(torch.bfloat16)
w_down = (torch.randn(k, ffn, device="cuda") * 0.5).to(torch.bfloat16)
y = fused_swiglu_mlp_inreg(x, w_gate, w_up, w_down)
torch.cuda.synchronize()
gate = x.float() @ w_gate.float().T
up = x.float() @ w_up.float().T
ref = (F.silu(gate) * up) @ w_down.float().T
ok = torch.allclose(y.float(), ref, atol=5e-2, rtol=5e-2)
print(f"arch={_arch} fused_swiglu_mlp_inreg", tuple(y.shape), y.dtype, "correct:", ok)
if not ok:
    print("Max diff:", (y.float() - ref).abs().max().item())
    sys.exit(1)
