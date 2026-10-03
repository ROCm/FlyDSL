# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x rowwise int8 quant.

``quantize_int8_rowwise`` calls ``build_quantize_int8_rowwise_module``.
Tensorwise quant is ``quantize_int8_tensorwise`` in the sibling module.
K must be a multiple of 8 for bf16 (128-bit vector).

Allowlist: ``("gfx120*",)``.
"""

import sys

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.quant.rdna4_quantize_int8_rowwise import quantize_int8_rowwise

_arch = (get_rocm_arch() or "").lower().split(":")[0]
if not _arch.startswith("gfx120"):
    print(f"SKIP examples/13-quant_int8_gfx120x.py: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
q, scale = quantize_int8_rowwise(x)
torch.cuda.synchronize()
print(f"arch={_arch} quantize_int8_rowwise", tuple(q.shape), q.dtype, tuple(scale.shape))
