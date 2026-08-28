#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness tests for gfx11 integer WMMA atoms.

The lowering tests cover the ROCDL operation shape, but they cannot catch a
fragment-layout or sign-control error on real hardware.  These tests execute
one wave and one 16x16x16 IU8/IU4 WMMA instruction, then require the INT32
result to match Torch exactly.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl  # noqa: E402,F401 -- preload comgr before torch/HIP loads LLVM
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir.dialects import fly  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx11"):
    pytest.skip(f"RDNA3 integer WMMA requires gfx11*, got {_ARCH}", allow_module_level=True)

WAVE_SIZE = 32
M = N = K = 16


def _compile_single_integer_wmma(elem_cls, *, sign_a, sign_b):
    """Build C[16,16] = A[16,16] @ B[16,16].T with one gfx11 atom."""

    @flyc.kernel
    def wmma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        lane = fx.thread_idx.x
        lane16 = lane % 16
        lane_half = lane // 16

        c2d = fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1)))

        # gfx11 duplicates each A/B fragment between lanes 0-15 and 16-31.
        if fx.const_expr(elem_cls is fx.Int4):
            # Each i32 contains 8 logical nibbles.  Keeping global storage
            # packed also avoids introducing illegal scalar i4 memory ops.
            a2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, 2), (2, 1)))
            b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, 2), (2, 1)))
            a_vec = fx.Vector.from_elements([a2d[lane16, word] for word in fx.range_constexpr(2)], fx.Int32).bitcast(
                fx.Int4
            )
            b_vec = fx.Vector.from_elements([b2d[lane16, word] for word in fx.range_constexpr(2)], fx.Int32).bitcast(
                fx.Int4
            )
        else:
            a2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1)))
            b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1)))
            a_vec = fx.Vector.from_elements([a2d[lane16, k].to(elem_cls) for k in fx.range_constexpr(K)], elem_cls)
            b_vec = fx.Vector.from_elements([b2d[lane16, k].to(elem_cls) for k in fx.range_constexpr(K)], elem_cls)
        acc = fx.Vector.filled(8, 0, fx.Int32)

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(
                M,
                N,
                K,
                elem_cls,
                fx.Int32,
                sign_a=sign_a,
                sign_b=sign_b,
            )
        )
        result = fx.Vector(
            fly.mma_atom_call_ssa(
                [fx.Vector.make_type(8, fx.Int32)],
                mma_atom,
                a_vec.ir_value(),
                b_vec.ir_value(),
                acc.ir_value(),
            )
        )

        # gfx11 interleaves C rows between the two lane halves.
        for value_idx in fx.range_constexpr(8):
            row = 2 * value_idx + lane_half
            c2d[row, lane16] = result[value_idx]

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        wmma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize(
    "elem_cls,torch_dtype,low,high,sign_a,sign_b",
    [
        pytest.param(fx.Int8, torch.int8, -128, 128, True, True, id="i8-signed"),
        pytest.param(fx.Uint8, torch.uint8, 0, 256, False, False, id="i8-unsigned"),
        pytest.param(fx.Int4, torch.int8, -8, 8, True, True, id="i4-signed"),
        pytest.param(fx.Int4, torch.uint8, 0, 16, False, False, id="i4-unsigned"),
    ],
)
def test_single_integer_wmma_atom(elem_cls, torch_dtype, low, high, sign_a, sign_b):
    torch.manual_seed(2026)
    logical_a = torch.randint(low, high, (M, K), dtype=torch_dtype)
    logical_b = torch.randint(low, high, (N, K), dtype=torch_dtype)
    logical_a[0].fill_(low)
    logical_a[1].fill_(high - 1)
    logical_a[2].zero_()
    logical_b[0].fill_(high - 1)
    logical_b[1].fill_(low)
    logical_b[2].zero_()
    if elem_cls is fx.Int4:
        shifts = (torch.arange(8, dtype=torch.int64) * 4).reshape(1, 1, 8)
        a = (((logical_a.to(torch.int64) & 0xF).reshape(M, 2, 8) << shifts).sum(dim=-1)).to(torch.int32)
        b = (((logical_b.to(torch.int64) & 0xF).reshape(N, 2, 8) << shifts).sum(dim=-1)).to(torch.int32)
    else:
        a = logical_a
        b = logical_b
    a = a.contiguous().to("cuda")
    b = b.contiguous().to("cuda")
    c = torch.full((M, N), -1, dtype=torch.int32, device="cuda")

    launch = _compile_single_integer_wmma(elem_cls, sign_a=sign_a, sign_b=sign_b)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = logical_a.to(torch.int64) @ logical_b.to(torch.int64).T
    torch.testing.assert_close(c.cpu(), ref.to(torch.int32), atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
