#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness for gfx120x (RDNA4) integer WMMA atoms.

RDNA4 uses the v8 register ABI: each lane holds 8 A/B elements (K/2), not the
gfx11 duplicated 16-element fragments.
  iu8 -> vector<2xi32> packed
  iu4 -> scalar i32 packed (gfx12 ABI; not gfx11 vector<2xi32>)
"""

import os
import sys

import pytest  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collections.abc import Callable  # noqa: E402

import flydsl  # noqa: E402,F401
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 integer WMMA requires gfx120*, got {_ARCH}", allow_module_level=True)

WAVE_SIZE = 32
M = N = K = 16


def _compile_single_integer_wmma(elem_cls: object, *, sign_a: bool, sign_b: bool) -> Callable[..., None]:
    """Build C[16,16] = A[16,16] @ B[16,16].T with one gfx120x iu8/iu4 atom."""

    @flyc.kernel
    def wmma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor) -> None:
        lane = fx.thread_idx.x
        lane16 = lane % 16
        lane_half = lane // 16

        c2d = fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1)))

        # RDNA4 v8 ABI: K = (lane/16)*8 + val → 8 elements per lane.
        if fx.const_expr(elem_cls is fx.Int4):
            # Each i32 holds 8 nibbles for one K-half. Global storage stays
            # packed to avoid illegal scalar i4 memory ops.
            a2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, 2), (2, 1)))
            b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, 2), (2, 1)))
            a_vec = fx.Vector.from_elements([a2d[lane16, lane_half]], fx.Int32).bitcast(fx.Int4)
            b_vec = fx.Vector.from_elements([b2d[lane16, lane_half]], fx.Int32).bitcast(fx.Int4)
        else:
            a2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1)))
            b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1)))
            a_vec = fx.Vector.from_elements(
                [a2d[lane16, lane_half * 8 + k].to(elem_cls) for k in fx.range_constexpr(8)],
                elem_cls,
            )
            b_vec = fx.Vector.from_elements(
                [b2d[lane16, lane_half * 8 + k].to(elem_cls) for k in fx.range_constexpr(8)],
                elem_cls,
            )
        acc = fx.Vector.filled(8, 0, fx.Int32)

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, elem_cls, fx.Int32, sign_a=sign_a, sign_b=sign_b))
        # Single atom, rank-1 fragments. ``fx.gemm`` is the form new kernels use.
        a_frag = fx.make_rmem_tensor(8, elem_cls)
        b_frag = fx.make_rmem_tensor(8, elem_cls)
        c_frag = fx.make_rmem_tensor(8, fx.Int32)
        a_frag.store(a_vec)
        b_frag.store(b_vec)
        c_frag.store(acc)
        fx.gemm(mma_atom, c_frag, [a_frag], [b_frag], c_frag)
        result = fx.Vector(c_frag.load())

        # C layout (gfx1250 helper): M = (lane/16)*8 + v, N = lane%16
        for value_idx in fx.range_constexpr(8):
            row = lane_half * 8 + value_idx
            c2d[row, lane16] = result[value_idx]

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)) -> None:
        wmma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize(
    "sign_a, sign_b",
    [(True, True), (False, False), (True, False)],
    ids=["signed", "unsigned", "mixed_sign"],
)
def test_single_iu8_wmma_atom(sign_a: bool, sign_b: bool) -> None:
    torch.manual_seed(0)
    # Keep values in a range that is exact for both signed and unsigned interpret.
    lo, hi = (-8, 8) if (sign_a or sign_b) else (0, 15)
    a = torch.randint(lo, hi, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(lo, hi, (N, K), device="cuda", dtype=torch.int8)
    c = torch.zeros(M, N, dtype=torch.int32, device="cuda")

    launch = _compile_single_integer_wmma(fx.Int8, sign_a=sign_a, sign_b=sign_b)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    a_ref = a.to(torch.int32)
    b_ref = b.to(torch.int32)
    if not sign_a:
        a_ref = a.to(torch.uint8).to(torch.int32)
    if not sign_b:
        b_ref = b.to(torch.uint8).to(torch.int32)
    # ROCm PyTorch lacks int32 addmm; compute reference in float32 then cast.
    ref = (a_ref.to(torch.float32) @ b_ref.to(torch.float32).T).to(torch.int32)
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "sign_a, sign_b, lo, hi",
    [
        pytest.param(True, True, -8, 8, id="i4-signed"),
        pytest.param(False, False, 0, 16, id="i4-unsigned"),
    ],
)
def test_single_iu4_wmma_atom(sign_a: bool, sign_b: bool, lo: int, hi: int) -> None:
    """16x16x16 iu4 atom vs torch int reference (packed host storage)."""
    torch.manual_seed(2026)
    logical_a = torch.randint(lo, hi, (M, K), dtype=torch.int8)
    logical_b = torch.randint(lo, hi, (N, K), dtype=torch.int8)
    # Pack 8 nibbles per i32 word; two words cover K=16 (lane_half selects word).
    shifts = (torch.arange(8, dtype=torch.int64) * 4).reshape(1, 1, 8)
    a = (((logical_a.to(torch.int64) & 0xF).reshape(M, 2, 8) << shifts).sum(dim=-1)).to(torch.int32)
    b = (((logical_b.to(torch.int64) & 0xF).reshape(N, 2, 8) << shifts).sum(dim=-1)).to(torch.int32)
    a = a.contiguous().to("cuda")
    b = b.contiguous().to("cuda")
    c = torch.zeros(M, N, dtype=torch.int32, device="cuda")

    launch = _compile_single_integer_wmma(fx.Int4, sign_a=sign_a, sign_b=sign_b)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    a_ref = logical_a.to(torch.int32)
    b_ref = logical_b.to(torch.int32)
    if not sign_a:
        a_ref = logical_a.to(torch.uint8).to(torch.int32)
    if not sign_b:
        b_ref = logical_b.to(torch.uint8).to(torch.int32)
    ref = (a_ref.to(torch.float32) @ b_ref.to(torch.float32).T).to(torch.int32)
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
