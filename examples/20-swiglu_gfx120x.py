# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x elementwise SiLU×mul (bare SwiGLU act).

``examples/15`` is the fused MLP. This file launches ``build_silu_mul_module``
only. Chunk-2 SwiGLU is ``build_swiglu_chunk_module`` (see tests).

gfx120x only.
"""

import sys

import torch
import torch.nn.functional as F

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import PointerJitArg
from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x
from kernels.common.gfx120x_swiglu import build_silu_mul_module

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)


def _ptr(tensor: object) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


m, n = 32, 256
gate = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
up = torch.randn_like(gate)
out = torch.empty_like(gate)
args = (_ptr(gate), _ptr(up), _ptr(out), m * n, torch.cuda.current_stream())
compiled = flyc.compile(build_silu_mul_module(), *args)
compiled(*args)
torch.cuda.synchronize()
ref = F.silu(gate.float()) * up.float()
ok = torch.allclose(out.float(), ref, atol=3e-2, rtol=3e-2)
print(f"arch={_arch} silu_mul", tuple(out.shape), out.dtype, "correct:", ok)
if not ok:
    print("Max diff:", (out.float() - ref).abs().max().item())
    sys.exit(1)
