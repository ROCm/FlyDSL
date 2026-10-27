#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness tests for gfx120 integer WMMA atoms.

One wave, C[16,N] = A[16,K] @ B[16,K].T. IU4 inputs are packed 8 nibbles per
int32. C scatter is RDNA4 M = 8*g + v, not gfx11 M = 2*v + g.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl  # noqa: E402,F401
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir.dialects import fly  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from tests.kernels.int4_pack import pack_int4  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 integer WMMA requires gfx120x, got {_ARCH}", allow_module_level=True)

WAVE_SIZE = 32
M = N = 16


def _compile_integer_wmma(elem_cls, k, *, sign_a, sign_b):
    packed_k_words = k // 8

    @flyc.kernel
    def wmma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        lane = fx.thread_idx.x
        lane16 = lane % 16
        g = lane // 16
        c2d = fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1)))

        if fx.const_expr(elem_cls is fx.Int4):
            a2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, packed_k_words), (packed_k_words, 1)))
            b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, packed_k_words), (packed_k_words, 1)))
            if fx.const_expr(k == 16):
                a_vec = fx.Vector.from_elements([a2d[lane16, g]], fx.Int32).bitcast(fx.Int4)
                b_vec = fx.Vector.from_elements([b2d[lane16, g]], fx.Int32).bitcast(fx.Int4)
            else:
                a_vec = fx.Vector.from_elements(
                    [a2d[lane16, 2 * g + w] for w in fx.range_constexpr(2)], fx.Int32
                ).bitcast(fx.Int4)
                b_vec = fx.Vector.from_elements(
                    [b2d[lane16, 2 * g + w] for w in fx.range_constexpr(2)], fx.Int32
                ).bitcast(fx.Int4)
        else:
            a2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, k), (k, 1)))
            b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, k), (k, 1)))
            # gfx120 IU8 is 8 elems/lane (vector<2xi32>), not gfx11's 16-elem v16 ABI.
            a_vec = fx.Vector.from_elements(
                [a2d[lane16, 8 * g + kk].to(elem_cls) for kk in fx.range_constexpr(8)], elem_cls
            )
            b_vec = fx.Vector.from_elements(
                [b2d[lane16, 8 * g + kk].to(elem_cls) for kk in fx.range_constexpr(8)], elem_cls
            )

        acc = fx.Vector.filled(8, 0, fx.Int32)
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, k, elem_cls, fx.Int32, sign_a=sign_a, sign_b=sign_b))
        result = fx.Vector(
            fly.mma_atom_call_ssa(
                [fx.Vector.make_type(8, fx.Int32)],
                mma_atom,
                a_vec.ir_value(),
                b_vec.ir_value(),
                acc.ir_value(),
            )
        )
        for value_idx in fx.range_constexpr(8):
            row = 8 * g + value_idx
            c2d[row, lane16] = result[value_idx]

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        wmma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


def _corner_rows(logical, low, high):
    logical[0].fill_(low)
    logical[1].fill_(high - 1)
    logical[2].zero_()
    return logical


@pytest.mark.parametrize(
    "elem_cls,torch_dtype,low,high,sign_a,sign_b,k",
    [
        pytest.param(fx.Int4, torch.int8, -8, 8, True, True, 16),
        pytest.param(fx.Int4, torch.uint8, 0, 16, False, False, 16),
        pytest.param(fx.Int8, torch.int8, -128, 128, True, True, 16),
        pytest.param(fx.Uint8, torch.uint8, 0, 256, False, False, 16),
        pytest.param(fx.Int4, torch.int8, -8, 8, True, True, 32),
    ],
    ids=["i4-signed-k16", "i4-unsigned-k16", "i8-signed-k16", "i8-unsigned-k16", "i4-signed-k32"],
)
def test_rdna4_integer_wmma_atom(elem_cls, torch_dtype, low, high, sign_a, sign_b, k):
    torch.manual_seed(2026)
    logical_a = _corner_rows(torch.randint(low, high, (M, k), dtype=torch_dtype), low, high)
    logical_b = torch.randint(low, high, (N, k), dtype=torch_dtype)
    logical_b[0].fill_(high - 1)
    logical_b[1].fill_(low)
    logical_b[2].zero_()
    if elem_cls is fx.Int4:
        a = torch.from_numpy(pack_int4(logical_a.cpu().numpy())).contiguous()
        b = torch.from_numpy(pack_int4(logical_b.cpu().numpy())).contiguous()
    else:
        a = logical_a
        b = logical_b
    a = a.to("cuda")
    b = b.to("cuda")
    c = torch.full((M, N), -1, dtype=torch.int32, device="cuda")

    launch = _compile_integer_wmma(elem_cls, k, sign_a=sign_a, sign_b=sign_b)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = logical_a.to(torch.int64) @ logical_b.to(torch.int64).T
    torch.testing.assert_close(c.cpu(), ref.to(torch.int32), atol=0, rtol=0)


def _compile_iu8_gemm():
    i8 = fx.Int8
    i32 = fx.Int32

    @flyc.kernel
    def wmma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((M, 16), (16, 1)))
        bB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((N, 16), (16, 1)))
        bC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((M, N), (N, 1)))
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, 16, i8, i32, sign_a=True, sign_b=True))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(i8.width), i8)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(i8.width), i8)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(i32.width), i32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
        fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
        fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        wmma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


def test_iu8_signed_k16_via_gemm():
    torch.manual_seed(2026)
    logical_a = torch.randint(-128, 128, (M, 16), dtype=torch.int8)
    logical_b = torch.randint(-128, 128, (N, 16), dtype=torch.int8)
    logical_a[0].fill_(-128)
    logical_a[1].fill_(127)
    logical_a[2].zero_()
    a = logical_a.contiguous().to("cuda")
    b = logical_b.contiguous().to("cuda")
    c = torch.full((M, N), -1, dtype=torch.int32, device="cuda")
    launch = _compile_iu8_gemm()
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = logical_a.to(torch.int64) @ logical_b.to(torch.int64).T
    torch.testing.assert_close(c.cpu(), ref.to(torch.int32), atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
