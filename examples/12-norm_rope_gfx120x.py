# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x RoPE builder launch.

``build_rope_module`` returns a kernel. This script passes pointers
(``flyc.from_c_void_p``) plus the pair counts, matching
``tests/kernels/test_gfx120x_norm_rope.py``. AdaLN and RMS+RoPE are different:
their launchers take ``fx.Tensor`` plus the row, group, and eps arguments.

gfx120x only.
"""

import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import PointerJitArg
from kernels.common.gfx120x_arch import get_gcn_arch, is_gfx120x

_arch = get_gcn_arch()
if not is_gfx120x():
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

from kernels.norm.rope_gfx120x import build_rope_module  # noqa: E402


def _ptr(tensor: object) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


rows, hd = 2, 64
pairs = hd // 2
x = torch.randn(rows, hd, device="cuda", dtype=torch.bfloat16)
freqs = torch.randn(rows, pairs, 2, 2, device="cuda", dtype=torch.float32)
out = torch.empty_like(x)
n_pairs_total = rows * pairs
args = (
    _ptr(x),
    _ptr(freqs),
    _ptr(out),
    n_pairs_total,
    pairs,
    1,
    rows,
    1,
    rows,
    1,
    rows * pairs,
    torch.cuda.current_stream(),
)
compiled = flyc.compile(build_rope_module("bfloat16", n_pairs_total=n_pairs_total), *args)
compiled(*args)
torch.cuda.synchronize()
ok = tuple(out.shape) == (rows, hd) and out.dtype == x.dtype and bool(torch.isfinite(out).all())
print(f"arch={_arch} rope", tuple(out.shape), out.dtype, "correct:", ok)
if not ok:
    sys.exit(1)
