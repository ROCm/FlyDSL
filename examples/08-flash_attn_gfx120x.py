# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x FlashAttention host call.

Wave64 / CDNA attention stays on ``kernels/attention/flash_attn_interface.py``
once ``is_gfx120x`` is false. This file calls the gfx120x host directly so a
port can see the argument order without the router.

gfx120x only.
"""

import math
import sys

import torch

from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func  # noqa: E402

# BSHD. A head dim that is not a tile is zero-filled in the kernel. This demo
# uses D=64, which is already a tile. One head, no GQA.
torch.manual_seed(7)
q = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16)
out = flydsl_flash_attn_func(q, k, v, causal=True)
window = flydsl_flash_attn_func(q, k, v, causal=False, sliding_window=(16, 0))
torch.cuda.synchronize()
if window.dtype != q.dtype:
    print(f"sliding_window dtype {window.dtype} != {q.dtype}")
    sys.exit(1)

scale = 1.0 / math.sqrt(q.shape[-1])
qf = q.float().transpose(1, 2)  # [B,H,Sq,D]
kf = k.float().transpose(1, 2)
vf = v.float().transpose(1, 2)
scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
Sq = q.shape[1]
qi = torch.arange(Sq, device=q.device)[None, None, :, None]
ki = torch.arange(Sq, device=q.device)[None, None, None, :]
scores = scores.masked_fill(ki > qi, float("-inf"))
ref = torch.matmul(torch.softmax(scores, dim=-1), vf).transpose(1, 2)
ok = torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)
print(f"arch={_arch} flash_attn out", tuple(out.shape), out.dtype, "correct:", ok)
if not ok:
    print("Max diff:", (out.float() - ref).abs().max().item())
    sys.exit(1)
